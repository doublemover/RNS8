"""Shared exclusive same-thread phase intersections for observed host intervals."""

from __future__ import annotations

from collections import defaultdict

from host_range_contract import PHASES, range_inventory, measured_roots
from native_phase_attribution import exclusive_segments


def select_intervals(intervals, snapshot, pid, *, native_ranges=None):
    native = native_ranges is not None
    if native and (
        native_ranges.get("state") != "native_measured_windows_bound"
        or type(native_ranges.get("selected_pid")) is not int
        or native_ranges["selected_pid"] != pid
    ):
        raise ValueError("bound same-process native marker windows required")
    index = range_inventory(
        snapshot, pid, clock_kind="windows_qpc" if native else "linux_monotonic"
    )
    native_markers = {row["slot"]: row for row in native_ranges["markers"]} if native else {}

    def boundaries(identity, row):
        if native:
            marker = native_markers.get(identity)
            if marker is None:
                return [None] * 4
            return [marker["start_ns"]] * 2 + [marker["end_ns"]] * 2
        return [
            row[key]
            for key in ("start_before_ns", "start_after_ns", "stop_before_ns", "stop_after_ns")
        ]

    roots = measured_roots(index)
    for identity, row in roots.items():
        stamps = boundaries(identity, row)
        if (
            row["stop_state"] != 1
            or row["start_thread_id"] != row["stop_thread_id"]
            or any(value is None for value in stamps)
            or stamps != sorted(stamps)
        ):
            raise ValueError("completed synchronous measured scheduling window required")
    groups, markers, allowed, owners = defaultdict(list), {}, set(), {}
    unresolved_scopes, blocked_threads = [], set()
    for identity, row in index.records.items():
        stamps = boundaries(identity, row)
        if (
            row["stop_state"] != 1
            or row["start_thread_id"] != row["stop_thread_id"]
            or any(value is None for value in stamps)
            or stamps != sorted(stamps)
        ):
            unresolved_scopes.append(identity)
            if (
                row["stop_state"] == 0
                or row["stop_state"] == 1
                and row["start_thread_id"] == row["stop_thread_id"]
            ):
                # No bounded child interval exists to mask. Withhold phase
                # credit for this thread rather than silently credit its parent.
                blocked_threads.add(row["start_thread_id"])
            continue
        markers[identity] = dict(start_ns=stamps[1], end_ns=stamps[2])
        groups[row["start_thread_id"]].append(identity)
        owner = next((p for p in (identity, *index.parents[identity]) if p in roots), None)
        if owner is not None and all(
            row[key] == roots[owner][key]
            for key in ("operation_id", "iteration_id", "operation_namespace", "role")
        ):
            root = roots[owner]
            root_bounds = boundaries(owner, root)
            if root_bounds[1] <= stamps[1] <= stamps[2] <= root_bounds[2]:
                allowed.add(identity)
                owners[identity] = owner
    by_thread = defaultdict(list)
    for row in intervals:
        by_thread[row["tid"]].append(row)
    phases, selected = defaultdict(int), []
    for tid, identities in sorted(groups.items()):
        if tid in blocked_threads:
            continue
        spans = exclusive_segments(identities, index, markers, allowed)
        cursor = 0
        for row in sorted(by_thread.get(tid, []), key=lambda r: r["start_ns"]):
            while cursor < len(spans) and spans[cursor][1] <= row["start_ns"]:
                cursor += 1
            position = cursor
            while position < len(spans) and spans[position][0] < row["end_ns"]:
                start, end, leaf = spans[position]
                if leaf is not None:
                    first, last = max(start, row["start_ns"]), min(end, row["end_ns"])
                    if first < last:
                        phase = PHASES[index.records[leaf]["phase"]]
                        phases[phase] += last - first
                        selected.append(
                            dict(
                                tid=tid,
                                host_range_id=leaf,
                                measured_root_id=owners[leaf],
                                phase=phase,
                                start_ns=first,
                                end_ns=last,
                                out_record=row["out_record"],
                                in_record=row["in_record"],
                                preempted_at_out=row.get("preempted_at_out"),
                            )
                        )
                        if len(selected) > 100_000:
                            raise ValueError("selected scheduling intersection budget exhausted")
                position += 1
    return dict(
        timestamp_domain="native_ETL_FILETIME_relative_to_process_start_ns"
        if native
        else "linux_monotonic_ns",
        intervals=selected,
        exclusive_phase_off_cpu_thread_ns=dict(sorted(phases.items())),
        conditional_off_cpu_thread_ns=sum(phases.values()),
        unresolved_scope_ids=sorted(unresolved_scopes),
        phase_unbound_threads=sorted(blocked_threads),
        scope="same_thread_half_open_measured_host_intersections_not_additive_process_wall",
        cpu_running_time_inferred=False,
        blocked_wait_cause_verified=False,
    )
