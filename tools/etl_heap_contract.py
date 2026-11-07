"""Native Windows host-heap generations, separate from RSS/device/pinned bytes."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import groupby

from etl_process_contract import integer
from etl_cpu_contract import SAMPLE_FILTER
from etl_point_phases import select_native_points


def heap_catalog(metadata, process, cpu, *, snapshot=None, native_ranges=None):
    pid = process["selected_pid"]
    if (
        cpu.get("schema") != "rns8_etl_cpu_observations_v1"
        or type(pid) is not int
        or cpu.get("selected_pid") != pid
        or type(metadata.get("selected_pid")) is not int
        or metadata["selected_pid"] != pid
    ):
        raise ValueError("same-process native CPU/thread custody required for heap events")
    total = integer(
        metadata.get("observed_events"), "native event count", minimum=1, maximum=1_000_000
    )
    observed = integer(metadata.get("observed_heap_events"), "heap events", maximum=total)
    undecoded = integer(
        metadata.get("undecoded_heap_events"), "undecoded heap events", maximum=observed
    )
    excluded = integer(
        metadata.get("excluded_heap_events"), "excluded heap events", maximum=observed
    )
    unsupported = integer(
        metadata.get("unsupported_heap_events"), "unsupported heap events", maximum=observed
    )
    rows = metadata.get("heap_events")
    if (
        not isinstance(rows, list)
        or len(rows) + undecoded + excluded + unsupported != observed
        or len(rows) + len(metadata["sample_events"]) + len(metadata["schedule_events"]) > 65536
        or metadata.get("heap_filter") != SAMPLE_FILTER
        or metadata.get("heap_record_namespace") != "second_native_event_walk"
        or integer(
            metadata.get("sample_walk_events"), "heap event walk", minimum=1, maximum=1_000_000
        )
        != total
    ):
        raise ValueError("complete bounded native heap projection required")
    seen = {r["record_index"] for r in metadata["sample_events"] + metadata["schedule_events"]}
    lifetimes = defaultdict(lambda: {1: [], 2: []})
    for row in metadata["thread_events"]:
        if row["opcode"] in {1, 2}:
            lifetimes[row["tid"]][row["opcode"]].append(row["time_filetime_100ns"])
    groups, outside, points = defaultdict(list), [], []
    first, last = process["actual_start_filetime_100ns"], process["actual_end_filetime_100ns"]
    thread_generations = defaultdict(list)
    for generation in cpu["threads"]["generations"]:
        thread_generations[generation["tid"]].append(generation)
    thread_starts = {}
    for tid, generations in thread_generations.items():
        generations.sort(key=lambda r: r["start_filetime_100ns"])
        thread_starts[tid] = [r["start_filetime_100ns"] for r in generations]
    for row in rows:
        if not isinstance(row, dict) or row.get("wire_contract") != "windows_kernel_HeapTrace_v0":
            raise ValueError("supported native heap wire contract required")
        ordinal = integer(row.get("record_index"), "heap record", minimum=1, maximum=total)
        if ordinal in seen:
            raise ValueError("heap/sample/scheduler record identity duplicated")
        seen.add(ordinal)
        stamp = integer(row.get("time_filetime_100ns"), "heap timestamp", maximum=2**63 - 1)
        native_pid = integer(row.get("native_header_pid"), "heap PID", maximum=2**32 - 1)
        tid = integer(row.get("native_header_tid"), "heap TID", maximum=2**32 - 1)
        if native_pid not in {pid, 2**32 - 1}:
            raise ValueError("native heap belongs to a known foreign process")
        if native_pid == 2**32 - 1:
            life = lifetimes.get(tid)
            if (
                life is None
                or len(life[1]) != 1
                or len(life[2]) != 1
                or not life[1][0] <= stamp < life[2][0]
            ):
                raise ValueError("unknown-PID heap record contradicts native unique-thread filter")
        opcode = integer(row.get("opcode"), "heap opcode", minimum=32, maximum=36)
        width = row.get("pointer_bytes")
        words = row.get("payload_words")
        count = {32: 1, 33: 3, 34: 5, 35: 1, 36: 2}[opcode]
        if (
            type(width) is not int
            or width not in {4, 8}
            or not isinstance(words, list)
            or len(words) != count
        ):
            raise ValueError("bounded native heap payload width/word count required")
        for word in words:
            integer(word, "heap pointer-sized word", maximum=2 ** (8 * width) - 1)
        if opcode == 35:
            if row.get("tail") is not None:
                raise ValueError("heap destroy has no trailing field")
        else:
            integer(row.get("tail"), "heap raw trailing field", maximum=2**32 - 1)
        if not words[0]:
            raise ValueError("native heap handle cannot be null")
        if (
            process.get("process_identity_complete") is not True
            or first is None
            or last is None
            or not first <= stamp < last
        ):
            outside.append(
                dict(record_index=ordinal, reason="outside_complete_observed_process_custody")
            )
            continue
        groups[words[0]].append(row)
        position = bisect_right(thread_starts.get(tid, []), stamp) - 1
        if (
            not cpu["threads"]["undecoded_thread_events"]
            and position >= 0
            and stamp < thread_generations[tid][position]["end_filetime_100ns"]
        ):
            points.append(dict(record_index=ordinal, tid=tid, time_filetime_100ns=stamp))
    selection = (
        select_native_points(points, pid, snapshot, native_ranges)
        if native_ranges is not None
        and native_ranges.get("state") == "native_measured_windows_bound"
        else None
    )
    phases = {r["native_record"]: r for r in selection["points"]} if selection else {}
    allocations, unresolved, buckets = [], outside, Counter()
    counts = Counter(row["opcode"] for values in groups.values() for row in values)
    current = peak = requested = matched = destroyed = 0
    timeline = defaultdict(int)
    for heap, events in sorted(groups.items()):
        generation = None
        live = {}

        def invalidate(stamp, ordinal, reason, *, live=live):
            nonlocal current
            for record in live.values():
                record.update(
                    invalidated_by_record=ordinal,
                    invalidated_filetime_100ns=stamp,
                    unresolved_end_reason=reason,
                )
                current -= record["bytes"]
            live.clear()

        def release(pointer, row, reason, *, live=live, heap=heap):
            nonlocal current, matched, destroyed
            record = live.pop(pointer, None)
            if record is None:
                unresolved.append(
                    dict(
                        record_index=row["record_index"],
                        heap_handle=heap,
                        pointer=pointer,
                        reason="free_or_realloc_without_observed_allocation",
                    )
                )
                return None
            record.update(
                end_record=row["record_index"],
                end_filetime_100ns=row["time_filetime_100ns"],
                end_reason=reason,
                free_phase=phases.get(row["record_index"]),
            )
            record["observed_event_lifetime_ns"] = (
                row["time_filetime_100ns"] - record["start_filetime_100ns"]
            ) * 100
            current -= record["bytes"]
            if reason == "heap_destroy":
                destroyed += 1
            else:
                matched += 1
            return record

        def allocate(pointer, size, row, heap_generation, *, live=live, heap=heap):
            nonlocal current, requested
            if not pointer:
                unresolved.append(
                    dict(record_index=row["record_index"], reason="null_native_allocation_address")
                )
                return
            if pointer in live:
                previous = live.pop(pointer)
                previous.update(
                    invalidated_by_record=row["record_index"],
                    invalidated_filetime_100ns=row["time_filetime_100ns"],
                    unresolved_end_reason="pointer_replaced_without_observed_free",
                )
                current -= previous["bytes"]
                unresolved.append(
                    dict(
                        record_index=row["record_index"],
                        reason="pointer_replaced_without_observed_free",
                    )
                )
            record = dict(
                generation_id=row["record_index"],
                heap_handle=heap,
                heap_generation_id=heap_generation,
                pointer=pointer,
                bytes=size,
                start_record=row["record_index"],
                start_filetime_100ns=row["time_filetime_100ns"],
                end_record=None,
                end_filetime_100ns=None,
                end_reason=None,
                allocation_phase=phases.get(row["record_index"]),
                free_phase=None,
            )
            live[pointer] = record
            allocations.append(record)
            current += size
            requested += size
            buckets[size.bit_length()] += 1

        for stamp, same_time in groupby(
            sorted(events, key=lambda r: r["time_filetime_100ns"]),
            lambda r: r["time_filetime_100ns"],
        ):
            batch = list(same_time)
            referenced = Counter()
            for row in batch:
                words, opcode = row["payload_words"], row["opcode"]
                addresses = (
                    {words[2]}
                    if opcode == 33
                    else set(words[1:3])
                    if opcode == 34
                    else {words[1]}
                    if opcode == 36
                    else set()
                )
                referenced.update(addresses - {0})
            if len(batch) > 1 and (
                any(v > 1 for v in referenced.values())
                or any(r["opcode"] in {32, 35} for r in batch)
            ):
                before = current
                invalidate(stamp, batch[0]["record_index"], "same_time_heap_order_ambiguous")
                generation = None
                unresolved.extend(
                    dict(record_index=r["record_index"], reason="same_time_heap_order_ambiguous")
                    for r in batch
                )
                timeline[stamp] += current - before
                continue
            before = current
            for row in batch:
                words, opcode = row["payload_words"], row["opcode"]
                if opcode == 32:
                    if generation is not None or live:
                        invalidate(
                            stamp, row["record_index"], "heap_recreated_without_observed_destroy"
                        )
                        unresolved.append(
                            dict(
                                record_index=row["record_index"],
                                reason="heap_recreated_without_observed_destroy",
                            )
                        )
                    generation = row["record_index"]
                elif opcode == 35:
                    for pointer in list(live):
                        release(pointer, row, "heap_destroy")
                    generation = None
                elif opcode == 33:
                    allocate(words[2], words[1], row, generation)
                elif opcode == 36:
                    release(words[1], row, "free")
                elif not words[1] or not words[2]:
                    unresolved.append(
                        dict(
                            record_index=row["record_index"],
                            reason="null_reallocation_outcome_unresolved_old_generation_preserved",
                        )
                    )
                else:
                    previous = release(words[2], row, "reallocation")
                    if previous is not None and previous["bytes"] != words[4]:
                        unresolved.append(
                            dict(
                                record_index=row["record_index"],
                                reason="reallocation_old_native_size_differs",
                                observed_allocation_bytes=previous["bytes"],
                                reported_old_bytes=words[4],
                            )
                        )
                    allocate(words[1], words[3], row, generation)
            # Same-time independent pointer records commute for boundary live
            # bytes; do not invent an internal transient high-water ordering.
            timeline[stamp] += current - before
    # Heaps are processed independently; reconstruct their combined boundary
    # totals in time order instead of adding independent per-heap maxima.
    at_boundary = 0
    for _, delta in sorted(timeline.items()):
        at_boundary += delta
        peak = max(peak, at_boundary)
    return dict(
        schema="rns8_etl_host_heap_observations_v1",
        selected_pid=pid,
        timestamp_domain="native_ProcessTrace_converted_FILETIME_100ns",
        native_record_namespace="second_native_event_walk",
        header_loss=process["header_loss"],
        state="observed_allocation_generations_partial_history"
        if allocations
        else "no_resolved_native_allocation_generations",
        memory_domain="native_Windows_heap_reported_payload_not_RSS_device_or_pinned_storage",
        allocations=allocations,
        admitted_native_record_indices=sorted(
            r["record_index"] for values in groups.values() for r in values
        ),
        unresolved=unresolved,
        summary=dict(
            allocation_events=counts[33],
            reallocation_events=counts[34],
            free_events=counts[36],
            heap_create_events=counts[32],
            heap_destroy_events=counts[35],
            observed_generations=len(allocations),
            matched_generation_ends=matched,
            heap_destroy_generation_ends=destroyed,
            native_reported_allocated_bytes=requested,
            conditional_tracked_live_boundary_high_water_bytes=peak,
            conditional_tracked_retained_bytes=current,
            size_buckets_bit_length=dict(sorted(buckets.items())),
            within_selected_operation_generation_count=sum(
                r["allocation_phase"] is not None
                and r.get("free_phase") is not None
                and r["allocation_phase"]["measured_root_id"] == r["free_phase"]["measured_root_id"]
                for r in allocations
            ),
        ),
        selected_operation_points=selection,
        retained_heap_events=len(rows),
        undecoded_global_heap_events=undecoded,
        unsupported_selected_heap_events=unsupported,
        excluded_global_heap_events=excluded,
        observed_global_heap_events=observed,
        candidate_event_counts_by_opcode=dict(sorted(Counter(r["opcode"] for r in rows).items())),
        complete_heap_history_proven=False,
        zero_allocation_behavior_proven=False,
        tracking_condition="no_unobserved_intervening_allocation_free_heap_generation_or_lost_event",
        heap_generation_missing_means="preexisting_heap_not_an_invented_create",
        exact_process_peak_bytes=None,
        sampled_RSS_bytes=None,
        device_or_pinned_bytes=None,
        source_callstacks="pending_native_stack_address_association",
        production_latency=False,
        promotion_allowed=False,
    )
