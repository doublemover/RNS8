"""Observed per-thread off-CPU intervals; no inferred sleep/wakeup or GPU edges."""

from __future__ import annotations

from collections import defaultdict
from bisect import bisect_right
from itertools import chain, groupby

from host_range_contract import PHASES, uint, range_inventory, measured_roots
from native_phase_attribution import exclusive_segments
from perf_recording_contract import SCHEMA


def summarize(recording: dict, *, pid: int, snapshot=None) -> dict:
    uint(pid, "schedule PID", positive=True)
    if (
        not isinstance(recording, dict)
        or recording.get("schema") != SCHEMA
        or recording.get("complete_record_walk") is not True
        or recording.get("clock", {}).get("attribute_clock_id") != 1
        or type(recording.get("clock", {}).get("attribute_clock_id")) is not int
    ):
        raise ValueError("audited native monotonic per-thread recording required")
    threads, seen = defaultdict(list), set()
    rows, boundaries = recording.get("context_switches"), recording.get("task_lifetime_boundaries")
    if any(not isinstance(value, list) or len(value) > 100_000 for value in (rows, boundaries)):
        raise ValueError("bounded native switch/task observations required")
    exec_times = []
    for row, boundary in chain(((r, False) for r in rows), ((r, True) for r in boundaries)):
        if not isinstance(row, dict):
            raise ValueError("typed native scheduling observation required")
        for field in ("pid", "tid", "record_index", "time_ns"):
            uint(row.get(field), "schedule " + field, positive=field != "time_ns")
        if row["record_index"] in seen:
            raise ValueError("repeated native scheduling record identity")
        seen.add(row["record_index"])
        if type(row.get("clock_id")) is not int or row["clock_id"] != 1:
            raise ValueError("native scheduling clocks differ")
        if boundary:
            if row.get("kind") not in {"fork", "exit", "exec"}:
                raise ValueError("unknown native task-lifetime observation")
        elif (
            row.get("direction") not in {"in", "out"}
            or type(row.get("preempted")) is not bool
            or row["direction"] == "in"
            and row["preempted"]
        ):
            raise ValueError("invalid native switch direction/preemption")
        if row["pid"] == pid:
            threads[row["tid"]].append((row["time_ns"], boundary, row))
            if boundary and row["kind"] == "exec":
                exec_times.append(row["time_ns"])
    exec_times.sort()
    intervals, unresolved = [], []
    for tid, events in sorted(threads.items()):
        pending = None
        for timestamp, group in groupby(
            sorted(events, key=lambda item: item[0]), lambda item: item[0]
        ):
            group = list(group)
            if len(group) != 1:
                unresolved.append(
                    dict(
                        tid=tid,
                        time_ns=timestamp,
                        reason="same_time_order_ambiguous",
                        native_records=[item[2]["record_index"] for item in group],
                    )
                )
                if pending is not None:
                    unresolved.append(
                        dict(
                            tid=tid,
                            native_record=pending["record_index"],
                            reason="out_before_ambiguous_boundary",
                        )
                    )
                pending = None
                continue
            _, boundary, row = group[0]
            if boundary:
                if pending is not None:
                    unresolved.append(
                        dict(
                            tid=tid,
                            native_record=pending["record_index"],
                            reason="task_lifetime_boundary_before_in",
                        )
                    )
                pending = None
                continue
            if row["direction"] == "out":
                if pending is not None:
                    unresolved.append(
                        dict(
                            tid=tid,
                            native_record=pending["record_index"],
                            reason="out_without_matching_in",
                        )
                    )
                pending = row
            elif pending is None:
                unresolved.append(
                    dict(
                        tid=tid, native_record=row["record_index"], reason="in_without_observed_out"
                    )
                )
            else:
                if bisect_right(exec_times, pending["time_ns"]) != bisect_right(
                    exec_times, timestamp
                ):
                    unresolved.append(
                        dict(
                            tid=tid,
                            native_record=pending["record_index"],
                            reason="process_exec_before_in",
                        )
                    )
                    pending = None
                    continue
                intervals.append(
                    dict(
                        pid=pid,
                        tid=tid,
                        start_ns=pending["time_ns"],
                        end_ns=timestamp,
                        out_record=pending["record_index"],
                        in_record=row["record_index"],
                        preempted_at_out=pending["preempted"],
                        state="preempted_TASK_RUNNING_at_out"
                        if pending["preempted"]
                        else "nonpreempt_out_state_unknown",
                    )
                )
                pending = None
        if pending is not None:
            unresolved.append(
                dict(
                    tid=tid, native_record=pending["record_index"], reason="out_without_observed_in"
                )
            )
    selected = _select(intervals, snapshot, pid) if snapshot is not None else None
    return dict(
        schema="rns8_host_schedule_attribution_v1",
        state="observed_pairs_partial_coverage" if intervals else "no_resolved_switch_pairs",
        pid=pid,
        clock_id=1,
        scope="selected_process_observed_per_thread_switch_pairs",
        intervals=intervals,
        unresolved=unresolved,
        conditional_off_cpu_thread_ns=sum(row["end_ns"] - row["start_ns"] for row in intervals),
        duration_condition="no_unobserved_intervening_switch_or_task_generation_boundary",
        selected_operation=selected,
        recording_blockers=recording.get("blockers", []),
        observed_switch_records=sum(
            not boundary for events in threads.values() for _, boundary, _ in events
        ),
        complete_scheduler_coverage_proven=False,
        exact_blocked_time_ns=None,
        wakeup_to_run_delay_ns=None,
        active_spin_time_ns=None,
        gpu_dependency_edges_verified=False,
        production_latency=False,
        promotion_allowed=False,
    )


def _select(intervals, snapshot, pid):
    index = range_inventory(snapshot, pid)
    roots = measured_roots(index)
    for row in roots.values():
        stamps = [
            row[key]
            for key in ("start_before_ns", "start_after_ns", "stop_before_ns", "stop_after_ns")
        ]
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
        stamps = [
            row[key]
            for key in ("start_before_ns", "start_after_ns", "stop_before_ns", "stop_after_ns")
        ]
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
            if root["start_after_ns"] <= stamps[1] <= stamps[2] <= root["stop_before_ns"]:
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
        for row in by_thread.get(tid, []):
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
                                preempted_at_out=row["preempted_at_out"],
                            )
                        )
                        if len(selected) > 100_000:
                            raise ValueError("selected scheduling intersection budget exhausted")
                position += 1
    return dict(
        intervals=selected,
        exclusive_phase_off_cpu_thread_ns=dict(sorted(phases.items())),
        conditional_off_cpu_thread_ns=sum(phases.values()),
        unresolved_scope_ids=sorted(unresolved_scopes),
        phase_unbound_threads=sorted(blocked_threads),
        scope="same_thread_half_open_measured_host_intersections_not_additive_process_wall",
        cpu_running_time_inferred=False,
        blocked_wait_cause_verified=False,
    )
