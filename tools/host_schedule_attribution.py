"""Observed per-thread off-CPU intervals; no inferred sleep/wakeup or GPU edges."""

from __future__ import annotations

from collections import defaultdict
from bisect import bisect_right
from itertools import chain, groupby

from host_range_contract import uint
from host_interval_phases import select_intervals
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
    selected = select_intervals(intervals, snapshot, pid) if snapshot is not None else None
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
