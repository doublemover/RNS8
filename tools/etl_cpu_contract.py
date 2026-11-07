"""Typed native ETL thread generations and CPU points in native marker windows."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from itertools import groupby

from etl_process_contract import integer
from etl_point_phases import select_native_points

PRODUCER_SOURCE = "https://github.com/microsoft/perfview/blob/4f3118c0fe7f624ae93a3249ae2ff7b6e1800e98/src/TraceEvent/Parsers/KernelTraceEventParser.cs"
SAMPLE_FILTER = (
    "known_selected_header_PID_or_unique_selected_thread_Start_End_for_unknown_header_PID"
)


def thread_catalog(metadata, process):
    if (
        not isinstance(process, dict)
        or process.get("schema") != "rns8_etl_selected_process_custody_v1"
    ):
        raise ValueError("native selected-process custody required for CPU observations")
    pid = integer(process.get("selected_pid"), "thread PID", minimum=1, maximum=2**32 - 1)
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("selected_pid")) is not int
        or metadata["selected_pid"] != pid
    ):
        raise ValueError("native ETL CPU metadata belongs to another process")
    total = integer(
        metadata.get("observed_events"), "native event walk", minimum=1, maximum=1_000_000
    )
    observed = integer(metadata.get("observed_thread_events"), "thread event count", maximum=total)
    undecoded = integer(
        metadata.get("undecoded_thread_events"), "undecoded threads", maximum=observed
    )
    rows = metadata.get("thread_events")
    if (
        not isinstance(rows, list)
        or len(rows) > min(16384, observed)
        or len(rows) + undecoded > observed
    ):
        raise ValueError("bounded selected native thread inventory required")
    start = integer(
        metadata["header"].get("start_filetime_100ns"), "trace start", maximum=2**63 - 1
    )
    end = integer(
        metadata["header"].get("end_filetime_100ns"), "trace end", minimum=start, maximum=2**63 - 1
    )
    grouped, seen = defaultdict(list), set()
    for row in rows:
        if not isinstance(row, dict) or type(row.get("pid")) is not int or row["pid"] != pid:
            raise ValueError("native thread belongs to another process")
        tid = integer(row.get("tid"), "native TID", minimum=1, maximum=2**32 - 1)
        ordinal = integer(row.get("record_index"), "thread record", minimum=1, maximum=total)
        if ordinal in seen:
            raise ValueError("repeated native thread record identity")
        seen.add(ordinal)
        integer(row.get("opcode"), "thread opcode", minimum=1, maximum=4)
        integer(row.get("event_version"), "thread version", maximum=255)
        integer(row.get("time_filetime_100ns"), "thread timestamp", minimum=start, maximum=end)
        if type(row.get("pointer_bytes")) is not int or row["pointer_bytes"] not in {4, 8}:
            raise ValueError("native thread recorded width unsupported")
        grouped[tid].append(row)
    generations, unresolved, rundown = [], [], []
    for tid, events in sorted(grouped.items()):
        pending = None
        for stamp, values in groupby(
            sorted(events, key=lambda row: row["time_filetime_100ns"]),
            lambda row: row["time_filetime_100ns"],
        ):
            values = list(values)
            lifetime = [row for row in values if row["opcode"] in {1, 2}]
            rundown.extend(
                dict(
                    tid=tid,
                    record_index=row["record_index"],
                    time_filetime_100ns=stamp,
                    opcode=row["opcode"],
                )
                for row in values
                if row["opcode"] in {3, 4}
            )
            if len(lifetime) > 1:
                unresolved.append(
                    dict(
                        tid=tid,
                        reason="same_time_lifetime_order_ambiguous",
                        record_indices=[row["record_index"] for row in lifetime],
                    )
                )
                if pending is not None:
                    unresolved.append(
                        dict(
                            tid=tid,
                            reason="start_before_ambiguous_boundary",
                            record_index=pending["record_index"],
                        )
                    )
                pending = None
                continue
            if not lifetime:
                continue
            row = lifetime[0]
            if row["opcode"] == 1:
                if pending is not None:
                    unresolved.append(
                        dict(
                            tid=tid,
                            reason="replacement_without_observed_end",
                            record_index=pending["record_index"],
                        )
                    )
                pending = row
            elif pending is None:
                unresolved.append(
                    dict(
                        tid=tid,
                        reason="end_without_observed_start",
                        record_index=row["record_index"],
                    )
                )
            elif pending["pointer_bytes"] != row["pointer_bytes"]:
                unresolved.append(
                    dict(
                        tid=tid,
                        reason="recorded_width_changes_within_thread",
                        record_index=row["record_index"],
                    )
                )
                pending = None
            else:
                generations.append(
                    dict(
                        tid=tid,
                        generation_id=pending["record_index"],
                        start_filetime_100ns=pending["time_filetime_100ns"],
                        end_filetime_100ns=stamp,
                        start_record=pending["record_index"],
                        end_record=row["record_index"],
                    )
                )
                pending = None
        if pending is not None:
            unresolved.append(
                dict(
                    tid=tid,
                    reason="start_without_observed_end",
                    record_index=pending["record_index"],
                )
            )
    return dict(
        schema="rns8_etl_thread_generations_v1",
        selected_pid=pid,
        generation_namespace="first_native_event_walk_Start_record_index",
        generations=generations,
        unresolved=unresolved,
        rundown_snapshots=rundown,
        undecoded_thread_events=undecoded,
        complete_thread_coverage_proven=False,
    )


def cpu_catalog(metadata, process, *, snapshot=None, native_ranges=None):
    threads = thread_catalog(metadata, process)
    pid, total = threads["selected_pid"], metadata["observed_events"]
    if (
        integer(
            metadata.get("sample_walk_events"), "sample event walk", minimum=1, maximum=1_000_000
        )
        != total
    ):
        raise ValueError("native sample and metadata walks disagree")
    observed = integer(metadata.get("observed_sample_events"), "sample count", maximum=total)
    undecoded = integer(
        metadata.get("undecoded_sample_events"), "undecoded samples", maximum=observed
    )
    excluded = integer(metadata.get("excluded_sample_events"), "excluded samples", maximum=observed)
    known = observed + sum(
        integer(metadata.get(key), "native provider count", maximum=total)
        for key in (
            "observed_process_events",
            "observed_image_events",
            "observed_marker_events",
            "observed_thread_events",
            "observed_schedule_events",
            "observed_heap_events",
        )
    )
    if known > total:
        raise ValueError("disjoint native provider counts exceed the actual event walk")
    rows = metadata.get("sample_events")
    if (
        not isinstance(rows, list)
        or len(rows) > 100_000
        or len(rows) + undecoded + excluded != observed
        or metadata.get("sample_filter") != SAMPLE_FILTER
        or metadata.get("sample_record_namespace") != "second_native_event_walk"
        or metadata.get("complete_selected_sample_coverage_proven") is not False
    ):
        raise ValueError("complete bounded native candidate-sample projection required")
    by_thread = defaultdict(list)
    for row in threads["generations"]:
        by_thread[row["tid"]].append(row)
    starts = {}
    for tid, generations in by_thread.items():
        generations.sort(key=lambda row: row["start_filetime_100ns"])
        if any(
            left["end_filetime_100ns"] > right["start_filetime_100ns"]
            for left, right in zip(generations, generations[1:])
        ):
            raise ValueError("native thread generations overlap")
        starts[tid] = [row["start_filetime_100ns"] for row in generations]
    first, last = process["actual_start_filetime_100ns"], process["actual_end_filetime_100ns"]
    raw_lifetimes = defaultdict(lambda: {1: [], 2: []})
    for row in metadata["thread_events"]:
        if row["opcode"] in {1, 2}:
            raw_lifetimes[row["tid"]][row["opcode"]].append(row["time_filetime_100ns"])
    points, unresolved, seen = [], [], set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("wire_contract") != "windows_kernel_SampledProfile_v2"
        ):
            raise ValueError("unknown native CPU sample wire contract")
        ordinal = integer(row.get("record_index"), "sample record", minimum=1, maximum=total)
        if ordinal in seen:
            raise ValueError("duplicate native sample record")
        seen.add(ordinal)
        tid = integer(row.get("tid"), "sample TID", minimum=1, maximum=2**32 - 1)
        native_pid = integer(row.get("native_header_pid"), "sample header PID", maximum=2**32 - 1)
        native_tid = integer(row.get("native_header_tid"), "sample header TID", maximum=2**32 - 1)
        if native_pid not in {pid, 2**32 - 1}:
            raise ValueError("native candidate filter retained a known foreign process")
        stamp = integer(row.get("time_filetime_100ns"), "sample time", maximum=2**63 - 1)
        if native_pid == 2**32 - 1:
            lifetime = raw_lifetimes.get(tid)
            if (
                lifetime is None
                or len(lifetime[1]) != 1
                or len(lifetime[2]) != 1
                or not lifetime[1][0] <= stamp < lifetime[2][0]
            ):
                raise ValueError("unknown-PID sample contradicts native unique-thread filter")
        width = row.get("pointer_bytes")
        if type(width) is not int or width not in {4, 8}:
            raise ValueError("sample pointer domain unsupported")
        ip = integer(
            row.get("instruction_pointer"),
            "sample instruction pointer",
            maximum=2 ** (width * 8) - 1,
        )
        packed = integer(
            row.get("packed_count_flags_rank"), "packed native sample word", maximum=2**32 - 1
        )
        for field, value in (
            ("count", packed & 0xFFFF),
            ("sample_flags", (packed >> 16) & 255),
            ("rank", packed >> 24),
        ):
            if type(row.get(field)) is not int or row[field] != value:
                raise ValueError("native sample packed fields contradict original word")
        if type(row.get("processor_index_mode")) is not bool:
            raise ValueError("native processor-index mode required")
        integer(
            row.get("processor"),
            "sample processor",
            maximum=65535 if row["processor_index_mode"] else 255,
        )
        position = bisect_right(starts.get(tid, []), stamp) - 1
        generation = by_thread[tid][position] if position >= 0 else None
        if generation is not None and not stamp < generation["end_filetime_100ns"]:
            generation = None
        reason = None
        if process.get("process_identity_complete") is not True or first is None or last is None:
            reason = "native_process_identity_incomplete"
        elif not first <= stamp < last:
            reason = "outside_observed_process_window"
        elif native_tid not in {tid, 2**32 - 1}:
            reason = "native_header_thread_contradicts_sample_payload"
        elif threads["undecoded_thread_events"]:
            reason = "undecoded_thread_events_can_hide_generation_boundaries"
        elif generation is None:
            reason = "no_observed_closed_thread_generation"
        elif row["sample_flags"] & 3:
            reason = "DPC_or_ISR_not_selected_process_useful_CPU"
        elif row["sample_flags"] & 4:
            reason = "unknown_reserved_sample_flag"
        if reason:
            unresolved.append(dict(record_index=ordinal, tid=tid, reason=reason))
        else:
            points.append(
                dict(
                    record_index=ordinal,
                    tid=tid,
                    thread_generation_id=generation["generation_id"],
                    time_filetime_100ns=stamp,
                    instruction_pointer=ip,
                )
            )
    selection = (
        _select(points, pid, snapshot, native_ranges)
        if native_ranges is not None
        and native_ranges.get("state") == "native_measured_windows_bound"
        else None
    )
    return dict(
        schema="rns8_etl_cpu_observations_v1",
        selected_pid=pid,
        producer_source=PRODUCER_SOURCE,
        state="native_points_bound_to_observed_thread_generations"
        if points
        else "no_resolved_native_CPU_points",
        threads=threads,
        samples=points,
        unresolved=unresolved,
        observed_sample_events=observed,
        retained_candidate_samples=len(rows),
        undecoded_global_sample_events=undecoded,
        excluded_global_sample_events=excluded,
        selected_operation=selection,
        sample_filter=metadata["sample_filter"],
        sample_record_namespace="second_native_event_walk",
        timestamp_domain="native_ProcessTrace_converted_FILETIME_100ns",
        CPU_duration_ns=None,
        weighted_sampling_exposure_ns=None,
        loaded_symbols_verified=False,
        source_attribution="pending_native_address_module_and_PDB_association",
        complete_selected_sample_coverage_proven=False,
        production_latency=False,
        promotion_allowed=False,
        thread_join_condition="no_unobserved_intervening_thread_lifetime_boundary",
    )


def _select(points, pid, snapshot, ranges):
    value = select_native_points(points, pid, snapshot, ranges)
    value["samples"] = [
        dict(
            native_sample_record=row["native_record"],
            **{key: item for key, item in row.items() if key != "native_record"},
        )
        for row in value.pop("points")
    ]
    value["outside_or_unbound_thread_samples"] = value.pop("outside_or_unbound_thread_points")
    value["sample_count"] = value.pop("point_count")
    return value
