"""Match native marker slots and SDK launch identities to measured operations."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
import re

from host_range_contract import PHASES, range_inventory, measured_roots, uint
from interval_observations import interval_summary
from native_phase_attribution import attribute


def join(timeline: dict, snapshot: dict) -> dict:
    if (
        not isinstance(timeline, dict)
        or timeline.get("schema") != "rns8_native_rocpd_timeline_v1"
        or timeline.get("clock") != "native_ROCprofiler_SDK_timestamp_ns_in_one_process"
        or not isinstance(snapshot, dict)
        or snapshot.get("marker_label_contract") != "rns8_phase_slot_v1"
    ):
        raise ValueError(
            "actual native ROCpd timeline and exact phase-slot label contract required"
        )
    pid = uint(timeline.get("process_id"), "native timeline PID", positive=True)
    index = range_inventory(snapshot, pid)
    roots = measured_roots(index)
    rows, event_rows = timeline.get("records"), timeline.get("events")
    if (
        not isinstance(rows, list)
        or len(rows) > 100_000
        or not isinstance(event_rows, list)
        or len(event_rows) > 1_000_000
    ):
        raise ValueError("bounded native interval/event projection required")
    events = {}
    for row in event_rows:
        if not isinstance(row, dict):
            raise ValueError("native timeline event must be an object")
        identity = uint(row.get("id"), "native event ID", positive=True)
        if identity in events:
            raise ValueError("native marker event identity is duplicated")
        events[identity] = row
    slots, api_by_internal, dispatches, related = {}, defaultdict(list), [], []
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or type(row.get("kind")) is not str:
            raise ValueError("native interval has no typed table identity")
        identity = (row["kind"], uint(row.get("id"), "native interval ID", positive=True))
        if identity in identities:
            raise ValueError("native interval table identity is duplicated")
        identities.add(identity)
        first, last = (
            uint(row.get("start_ns"), "native start"),
            uint(row.get("end_ns"), "native end"),
        )
        if last < first:
            raise ValueError("native interval is reversed")
        event = events.get(row.get("native_event_id"))
        if row.get("native_event_id") is not None and event is None:
            raise ValueError("native interval refers to a missing event")
        if row["kind"] == "kernel_dispatch":
            dispatches.append(row)
            continue
        if row["kind"] in {"memory_copy", "memory_lifecycle", "hip_event", "graph_launch"}:
            related.append(row)
        if row["kind"] != "host_region" or event is None:
            continue
        if event.get("category") == "HIP_RUNTIME_API":
            internal = event.get("stack_id")
            if internal is not None and uint(internal, "HIP internal correlation") != 0:
                api_by_internal[uint(internal, "HIP internal correlation")].append(row)
        name = row.get("name")
        if not isinstance(name, str) or not name.startswith("rns8."):
            continue
        if event.get("category") != "MARKER_CORE_RANGE_API":
            continue  # A similarly named API is not a captured marker.
        match = re.fullmatch(r"rns8\.([a-z_]+)#slot=(0|[1-9][0-9]{0,19})", name)
        if not match:
            raise ValueError("native RNS8 marker does not implement the declared slot contract")
        slot = int(match[2])
        if slot >= len(snapshot["records"]) or slot in slots:
            raise ValueError("native RNS8 marker slot is duplicated or outside its snapshot")
        scope = snapshot["records"][slot]
        metadata = event.get("extdata")
        if (
            match[1] != PHASES[scope["phase"]]
            or not isinstance(metadata, dict)
            or metadata.get("message") != name
            or row.get("thread_id") != scope["start_thread_id"]
        ):
            raise ValueError("native marker message/phase/thread identity differs")
        slots[slot] = row
    if set(slots) != set(range(len(snapshot["records"]))):
        raise ValueError("one native marker for every published snapshot slot is required")
    markers = {scope["range_id"]: slots[slot] for slot, scope in enumerate(snapshot["records"])}
    windows = defaultdict(list)
    for identity, scope in roots.items():
        if scope["stop_state"] != 1 or scope["start_thread_id"] != scope["stop_thread_id"]:
            raise ValueError("measured native operation lacks synchronous host scope closure")
        native = markers[identity]
        windows[scope["start_thread_id"]].append((native["start_ns"], native["end_ns"], identity))
    starts = {}
    for thread, spans in windows.items():
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("native same-thread measured operations overlap")
        starts[thread] = [row[0] for row in spans]
    for identity, scope in index.records.items():
        if scope["stop_state"] != 1:
            continue  # Async completion is represented, not treated as nested CPU work.
        parent = next(
            (
                value
                for value in index.parents[identity]
                if index.records[value]["stop_state"] == 1
                and index.records[value]["start_thread_id"] == scope["start_thread_id"]
            ),
            None,
        )
        if parent is not None:
            child, enclosing = markers[identity], markers[parent]
            if (
                not enclosing["start_ns"]
                <= child["start_ns"]
                <= child["end_ns"]
                <= enclosing["end_ns"]
            ):
                raise ValueError("native host marker escapes its explicit same-thread parent")
    selected, unresolved, outside = [], [], []
    related_selected, related_unresolved, related_outside = [], [], []
    for dispatch in [*dispatches, *related]:
        is_kernel = dispatch["kind"] == "kernel_dispatch"
        accepted = selected if is_kernel else related_selected
        rejected = unresolved if is_kernel else related_unresolved
        excluded = outside if is_kernel else related_outside
        event = events.get(dispatch.get("native_event_id"))
        internal = event.get("stack_id") if event else None
        if internal is not None:
            uint(internal, "native dispatch internal correlation")
        candidates = api_by_internal.get(internal, []) if internal not in (None, 0) else []
        identity = (
            dict(
                dispatch_id=dispatch["dispatch_id"],
                agent_id=dispatch["agent_id"],
                queue_id=dispatch["queue_id"],
                native_table_id=dispatch["id"],
            )
            if is_kernel
            else dict(kind=dispatch["kind"], native_table_id=dispatch["id"], native_record=dispatch)
        )
        if len(candidates) != 1:
            rejected.append(dict(identity, reason="missing_or_ambiguous_exact_HIP_API_correlation"))
            continue
        api = candidates[0]
        thread = api.get("thread_id")
        if thread is None or (
            dispatch.get("thread_id") is not None and dispatch["thread_id"] != thread
        ):
            rejected.append(dict(identity, reason="native_launch_thread_missing_or_conflicting"))
            continue
        spans = windows.get(thread, [])
        position = bisect_right(starts.get(thread, []), api["start_ns"]) - 1
        if position < 0 or not spans[position][0] <= api["start_ns"] < spans[position][1]:
            excluded.append(dict(identity, reason="launch_API_outside_selected_measured_markers"))
            continue
        first, last, owner = spans[position]
        if (
            not api["end_ns"] <= last
            or not api["start_ns"] <= dispatch["start_ns"] <= dispatch["end_ns"] <= last
        ):
            rejected.append(
                dict(identity, reason="launch_or_device_work_escapes_completed_operation")
            )
            continue
        scope = roots[owner]
        accepted.append(
            dict(
                identity,
                operation_id=scope["operation_id"],
                iteration_id=scope["iteration_id"],
                host_range_id=owner,
                native_marker_id=markers[owner]["id"],
                native_api_id=api["id"],
                launch_thread_id=thread,
                sdk_internal_correlation=internal,
                api_start_ns=api["start_ns"],
                api_end_ns=api["end_ns"],
                start_ns=dispatch["start_ns"],
                end_ns=dispatch["end_ns"],
            )
        )
    phases = attribute(index, markers, roots, [*selected, *related_selected])
    return dict(
        schema="rns8_native_marker_operation_join_v1",
        process_id=pid,
        marker_slot_count=len(slots),
        selected_operation_count=len(roots),
        selected_dispatches=selected,
        unresolved_dispatches=unresolved,
        outside_measured_dispatches=outside,
        selected_related_intervals=related_selected,
        unresolved_related_intervals=related_unresolved,
        outside_measured_related_intervals=related_outside,
        phase_attribution=phases,
        measured_native_windows=interval_summary(
            [(first, last) for spans in windows.values() for first, last, _ in spans]
        ),
        selected_device_intervals=interval_summary(
            [(row["start_ns"], row["end_ns"]) for row in selected]
        ),
        native_marker_identity_complete=True,
        host_monotonic_to_GPU_clock_join="unbound_not_used_for_identity_matching",
        association="exact_native_marker_slot_and_HIP_API_internal_correlation_in_one_process",
        API_completion_to_kernel_start_dependency_inferred=False,
        loss_status=timeline.get("loss_status"),
        complete_run_coverage=False,
        production_latency=False,
        promotion_allowed=False,
    )
