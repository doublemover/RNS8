"""Native original-event stack joins; fragments are never silently concatenated."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

from etl_process_contract import integer

BOUND = "raw_QPC_bound_by_native_ordinal_header_context_payload_replay"


def stack_catalog(metadata, process, cpu, *, heap=None, schedule=None):
    pid = process["selected_pid"]
    if (
        type(pid) is not int
        or cpu.get("schema") != "rns8_etl_cpu_observations_v1"
        or cpu.get("selected_pid") != pid
        or type(metadata.get("selected_pid")) is not int
        or metadata["selected_pid"] != pid
    ):
        raise ValueError("same-process native custody required for stack attribution")
    total = integer(
        metadata.get("observed_events"), "native event walk", minimum=1, maximum=1_000_000
    )
    source_rows = [("cpu_sample", r) for r in metadata["sample_events"]]
    source_rows += [("heap_event", r) for r in metadata["heap_events"]]
    source_rows += [("schedule_event", r) for r in metadata["schedule_events"]]
    if len(source_rows) > 65536:
        raise ValueError("native stack source point budget exhausted")
    bound = metadata.get("raw_binding_state") == BOUND
    raw_walk = integer(metadata.get("raw_walk_events"), "raw native event walk", maximum=total)
    source_count = integer(
        metadata.get("raw_source_point_count"), "raw source count", maximum=65536
    )
    if metadata.get("raw_record_namespace") != "third_native_event_walk":
        raise ValueError("native raw walk identity namespace differs")
    clock = integer(metadata["header"].get("clock_type"), "native clock type", maximum=3)
    frequency = integer(
        metadata["header"].get("perf_frequency"), "native clock frequency", maximum=2**63 - 1
    )
    if bound:
        if (
            type(clock) is not int
            or clock != 1
            or type(frequency) is not int
            or frequency <= 0
            or raw_walk != total
            or source_count != len(source_rows)
            or not source_count
        ):
            raise ValueError("native raw QPC point replay differs from its source inventory")
    elif (
        raw_walk
        or source_count
        or metadata.get("raw_binding_state")
        not in {"no_selected_native_points", "unsupported_non_QPC_clock"}
        or metadata["raw_binding_state"] == "no_selected_native_points"
        and (source_rows or clock != 1 or frequency <= 0)
        or metadata["raw_binding_state"] == "unsupported_non_QPC_clock"
        and clock == 1
        and frequency > 0
    ):
        raise ValueError("native raw unsupported/empty state contradicts clock/source inventory")
    observed = integer(metadata.get("observed_stack_events"), "stack events", maximum=total)
    undecoded = integer(
        metadata.get("undecoded_stack_events"), "undecoded stack events", maximum=observed
    )
    excluded = integer(
        metadata.get("excluded_stack_events"), "excluded stack events", maximum=observed
    )
    unsupported = integer(
        metadata.get("unsupported_stack_events"), "unsupported stack events", maximum=observed
    )
    frames_total = integer(
        metadata.get("stack_frames"), "retained native stack frames", maximum=262144
    )
    fragments = metadata.get("stack_events")
    if (
        not isinstance(fragments, list)
        or len(fragments) > 8192
        or len(fragments) + undecoded + excluded + unsupported != observed
        or not bound
        and (observed or fragments or frames_total)
    ):
        raise ValueError("complete bounded native stack projection required")
    by_thread = defaultdict(list)
    for row in cpu["threads"]["generations"]:
        by_thread[row["tid"]].append(row)
    starts = {}
    for tid, generations in by_thread.items():
        generations.sort(key=lambda r: r["start_filetime_100ns"])
        starts[tid] = [r["start_filetime_100ns"] for r in generations]
    admitted_heap = set(heap["admitted_native_record_indices"]) if heap else set()
    admitted_cpu = {r["record_index"] for r in cpu["samples"]}
    targets, point_keys = {}, defaultdict(list)
    seen_source = set()
    for kind, row in source_rows:
        ordinal = integer(row.get("record_index"), "source point record", minimum=1, maximum=total)
        if ordinal in seen_source:
            raise ValueError("duplicate native stack source event identity")
        seen_source.add(ordinal)
        if not bound:
            if "raw_time_qpc" in row:
                raise ValueError("unbound native point cannot claim a raw QPC time")
            continue
        qpc = integer(row.get("raw_time_qpc"), "native raw QPC time", maximum=2**63 - 1)
        stamp = integer(
            row.get("time_filetime_100ns"), "native converted point time", maximum=2**63 - 1
        )
        if kind == "cpu_sample":
            tids = [(row["tid"], kind)] if ordinal in admitted_cpu else []
        elif kind == "heap_event":
            tids = [(row["native_header_tid"], kind)] if ordinal in admitted_heap else []
        elif schedule is None:
            tids = []
        elif row["wire_contract"] == "windows_kernel_CSwitch_v2":
            tids = [(row["old_tid"], "schedule_out"), (row["new_tid"], "schedule_in")]
        else:
            tids = [(row["tid"], "ready_event")]
        for tid, role in tids:
            position = bisect_right(starts.get(tid, []), stamp) - 1
            if (
                process.get("process_identity_complete") is not True
                or cpu["threads"]["undecoded_thread_events"]
                or position < 0
                or not process["actual_start_filetime_100ns"]
                <= stamp
                < process["actual_end_filetime_100ns"]
                or stamp >= by_thread[tid][position]["end_filetime_100ns"]
            ):
                continue
            identity = (ordinal, tid, role)
            if identity in targets:
                raise ValueError("duplicate derived native stack target")
            targets[identity] = dict(
                native_point_record=ordinal,
                tid=tid,
                source_kind=role,
                thread_generation_id=by_thread[tid][position]["generation_id"],
                time_filetime_100ns=stamp,
                raw_time_qpc=qpc,
                stack_records=[],
            )
            point_keys[pid, tid, qpc].append(identity)
    seen, parsed_frames, unresolved = set(), 0, []
    # Ordinals from the raw walk are checked against original selected source
    # ordinals too, but never treated as the stack's original event timestamp.
    source_incomplete = any(
        metadata.get(key, 0)
        for key in ("undecoded_sample_events", "undecoded_heap_events", "undecoded_schedule_events")
    )
    for row in fragments:
        if (
            not isinstance(row, dict)
            or row.get("wire_contract") != "windows_kernel_StackWalk_prefix_versions_0_to_2"
        ):
            raise ValueError("supported native StackWalk prefix required")
        ordinal = integer(row.get("record_index"), "stack record", minimum=1, maximum=raw_walk)
        if ordinal in seen or ordinal in seen_source:
            raise ValueError("native stack/source record identities alias")
        seen.add(ordinal)
        if type(row.get("pid")) is not int or row["pid"] != pid:
            raise ValueError("native stack belongs to another process")
        tid = integer(row.get("tid"), "stack original TID", maximum=2**32 - 1)
        qpc = integer(row.get("event_time_qpc"), "stack original QPC time", maximum=2**63 - 1)
        integer(row.get("event_version"), "stack prefix version", maximum=2)
        width = row.get("pointer_bytes")
        frames = row.get("frames")
        if (
            type(width) is not int
            or width not in {4, 8}
            or not isinstance(frames, list)
            or len(frames) > 192
        ):
            raise ValueError("bounded native stack pointer domain required")
        for frame in frames:
            integer(frame, "native stack frame IP", maximum=2 ** (8 * width) - 1)
        parsed_frames += len(frames)
        if parsed_frames > 262144:
            raise ValueError("native stack frame budget exhausted")
        matches = point_keys.get((pid, tid, qpc), [])
        reason = (
            "original_event_candidates_have_undecoded_records"
            if source_incomplete
            else "original_event_PID_TID_raw_QPC_not_unique_or_missing"
            if len(matches) != 1
            else "empty_native_stack_fragment"
            if not frames
            else None
        )
        if reason:
            unresolved.append(dict(stack_record=ordinal, reason=reason))
        else:
            targets[matches[0]]["stack_records"].append(ordinal)
    if parsed_frames != frames_total:
        raise ValueError("native stack frame accounting differs")
    if (
        observed
        + sum(
            integer(metadata.get(key), "disjoint native provider count", maximum=total)
            for key in (
                "observed_process_events",
                "observed_image_events",
                "observed_marker_events",
                "observed_thread_events",
                "observed_sample_events",
                "observed_schedule_events",
                "observed_heap_events",
            )
        )
        > total
    ):
        raise ValueError("disjoint native providers exceed the complete native event walk")
    return dict(
        schema="rns8_etl_native_stack_observations_v1",
        selected_pid=pid,
        state="native_original_event_stack_fragments_associated"
        if any(r["stack_records"] for r in targets.values())
        else "no_resolved_native_stack_association",
        source_point_namespace="second_native_event_walk",
        stack_record_namespace="third_native_event_walk",
        raw_binding_state=metadata["raw_binding_state"],
        targets=list(targets.values()),
        fragments=fragments,
        unresolved=unresolved,
        retained_stack_frames=frames_total,
        undecoded_global_stack_events=undecoded,
        unsupported_global_stack_events=unsupported,
        excluded_global_stack_events=excluded,
        observed_global_stack_events=observed,
        targets_without_observed_stack=sum(not r["stack_records"] for r in targets.values()),
        association_condition="unique_observed_original_event_PID_TID_raw_QPC_tuple_with_no_undecoded_selected_source_family",
        fragment_order="native_leaf_to_caller_within_each_fragment_no_cross_fragment_concatenation",
        complete_unwound_or_inlined_callchain_proven=False,
        complete_stack_coverage_proven=False,
        loaded_symbol_identity_verified=False,
        header_loss=process["header_loss"],
        production_latency=False,
        promotion_allowed=False,
    )
