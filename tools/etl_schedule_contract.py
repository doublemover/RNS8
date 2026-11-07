"""Native Windows CSwitch/ReadyThread pairs with explicit coverage conditions."""

from __future__ import annotations

from collections import defaultdict
from itertools import groupby

from etl_process_contract import integer
from host_interval_phases import select_intervals


def schedule_catalog(metadata, process, cpu, *, snapshot=None, native_ranges=None):
    if (
        cpu.get("schema") != "rns8_etl_cpu_observations_v1"
        or type(process.get("selected_pid")) is not int
        or cpu.get("selected_pid") != process["selected_pid"]
        or type(metadata.get("selected_pid")) is not int
        or metadata.get("selected_pid") != process["selected_pid"]
    ):
        raise ValueError("same-process native thread custody required for scheduling")
    total = integer(
        metadata.get("observed_events"), "native event walk", minimum=1, maximum=1_000_000
    )
    observed = integer(metadata.get("observed_schedule_events"), "scheduler events", maximum=total)
    undecoded = integer(
        metadata.get("undecoded_schedule_events"), "undecoded scheduler events", maximum=observed
    )
    excluded = integer(
        metadata.get("excluded_schedule_events"), "excluded scheduler events", maximum=observed
    )
    rows = metadata.get("schedule_events")
    if (
        not isinstance(rows, list)
        or len(rows) + undecoded + excluded != observed
        or len(rows) + len(metadata["sample_events"]) > 65536
        or metadata.get("schedule_filter") != "unique_selected_thread_Start_End_window"
        or metadata.get("schedule_record_namespace") != "second_native_event_walk"
        or integer(
            metadata.get("sample_walk_events"), "scheduler walk", minimum=1, maximum=1_000_000
        )
        != total
    ):
        raise ValueError("complete bounded native scheduler projection required")
    raw_lifetimes = defaultdict(lambda: {1: [], 2: []})
    for row in metadata["thread_events"]:
        if row["opcode"] in {1, 2}:
            raw_lifetimes[row["tid"]][row["opcode"]].append(row)
    windows = {}
    for tid, lifetime in raw_lifetimes.items():
        if (
            len(lifetime[1]) == len(lifetime[2]) == 1
            and lifetime[1][0]["time_filetime_100ns"] < lifetime[2][0]["time_filetime_100ns"]
        ):
            windows[tid] = (
                lifetime[1][0]["time_filetime_100ns"],
                lifetime[2][0]["time_filetime_100ns"],
                lifetime[1][0]["record_index"],
            )

    def owned(tid, stamp):
        return tid in windows and windows[tid][0] <= stamp < windows[tid][1]

    events, seen = defaultdict(list), {row["record_index"] for row in metadata["sample_events"]}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("typed native scheduler row required")
        ordinal = integer(row.get("record_index"), "scheduler record", minimum=1, maximum=total)
        if ordinal in seen:
            raise ValueError("scheduler/sample records have duplicate native identities")
        seen.add(ordinal)
        stamp = integer(row.get("time_filetime_100ns"), "scheduler timestamp", maximum=2**63 - 1)
        for key in ("native_header_pid", "native_header_tid"):
            integer(row.get(key), "native scheduler header", maximum=2**32 - 1)
        if type(row.get("processor_index_mode")) is not bool:
            raise ValueError("native scheduler processor-index mode required")
        integer(
            row.get("processor"),
            "native scheduler processor",
            maximum=65535 if row["processor_index_mode"] else 255,
        )
        switched = row.get("wire_contract") == "windows_kernel_CSwitch_v2"
        if not switched and row.get("wire_contract") != "windows_kernel_ReadyThread_v2":
            raise ValueError("unsupported native scheduler wire contract")
        controls = row.get("control_bytes")
        if (
            not isinstance(controls, list)
            or len(controls) != (8 if switched else 4)
            or any(type(value) is not int or not 0 <= value <= 255 for value in controls)
        ):
            raise ValueError("original bounded native scheduler control bytes required")
        if switched:
            old = integer(row.get("old_tid"), "old thread", maximum=2**32 - 1)
            new = integer(row.get("new_tid"), "new thread", maximum=2**32 - 1)
            integer(row.get("new_wait_time"), "raw native wait field", maximum=2**32 - 1)
            integer(row.get("reserved"), "native reserved field", maximum=2**32 - 1)
            selected = [(old, "out"), (new, "in")]
        else:
            selected = [
                (integer(row.get("tid"), "readied thread", minimum=1, maximum=2**32 - 1), "ready")
            ]
        selected = [(tid, kind) for tid, kind in selected if owned(tid, stamp)]
        if not selected:
            raise ValueError("native scheduler projection contradicts unique owned thread filter")
        for tid, kind in selected:
            events[tid].append((stamp, kind, row))
    intervals, unresolved = [], []
    origin = process.get("actual_start_filetime_100ns")
    trusted = (
        process.get("process_identity_complete") is True
        and cpu["threads"]["undecoded_thread_events"] == 0
        and undecoded == 0
    )

    def reject(tid, reason, records):
        unresolved.append(dict(tid=tid, reason=reason, native_records=records))

    for tid, stream in sorted(events.items()):
        pending, ready = None, []
        for stamp, group in groupby(
            sorted(stream, key=lambda entry: entry[0]), lambda entry: entry[0]
        ):
            group = list(group)
            if len(group) != 1:
                reject(
                    tid,
                    "same_time_scheduler_order_ambiguous",
                    [e[2]["record_index"] for e in group],
                )
                if pending:
                    reject(
                        tid, "out_before_ambiguous_scheduler_boundary", [pending["record_index"]]
                    )
                pending, ready = None, []
                continue
            _, kind, row = group[0]
            if kind == "ready":
                if pending is None:
                    reject(tid, "ready_without_observed_out", [row["record_index"]])
                else:
                    ready.append(row)
                continue
            if row["old_tid"] == row["new_tid"]:
                reject(tid, "same_old_new_thread_not_a_switch", [row["record_index"]])
                pending, ready = None, []
                continue
            if kind == "out":
                if pending:
                    reject(tid, "out_without_matching_in", [pending["record_index"]])
                pending, ready = row, []
                continue
            if pending is None:
                reject(tid, "in_without_observed_out", [row["record_index"]])
                continue
            if not trusted:
                reject(
                    tid,
                    "incomplete_process_or_undecoded_scheduler_thread_boundaries",
                    [pending["record_index"], row["record_index"]],
                )
                pending, ready = None, []
                continue
            start, end = pending["time_filetime_100ns"], stamp
            if not origin <= start < end <= process["actual_end_filetime_100ns"]:
                reject(
                    tid,
                    "switch_pair_outside_observed_process_window",
                    [pending["record_index"], row["record_index"]],
                )
                pending, ready = None, []
                continue
            state = pending["control_bytes"][6]
            if state == 4:
                reject(
                    tid,
                    "terminated_thread_rescheduled_without_new_generation",
                    [pending["record_index"], row["record_index"]],
                )
                pending, ready = None, []
                continue
            interval = dict(
                tid=tid,
                thread_generation_id=windows[tid][2],
                start_ns=(start - origin) * 100,
                end_ns=(end - origin) * 100,
                out_record=pending["record_index"],
                in_record=row["record_index"],
                out_processor=pending["processor"],
                in_processor=row["processor"],
                old_thread_state=state,
                old_wait_reason=pending["control_bytes"][4],
                old_wait_mode=pending["control_bytes"][5],
                ready_records=[r["record_index"] for r in ready],
                waiting_segment=None,
                runnable_segment=None,
            )
            known_controls = pending["reserved"] == 0 and pending["control_bytes"][3] == 0
            if known_controls and state in {1, 3} and not ready:
                interval["runnable_segment"] = [interval["start_ns"], interval["end_ns"]]
                interval["classification"] = "Ready_or_Standby_at_out_no_observed_intervening_ready"
            elif (
                known_controls
                and state == 5
                and pending["control_bytes"][5] <= 1
                and len(ready) == 1
                and ready[0]["control_bytes"][0] <= 2
                and ready[0]["control_bytes"][2] & ~1 == 0
                and ready[0]["control_bytes"][3] == 0
            ):
                wake = (ready[0]["time_filetime_100ns"] - origin) * 100
                interval["waiting_segment"] = [interval["start_ns"], wake]
                interval["runnable_segment"] = [wake, interval["end_ns"]]
                interval["classification"] = (
                    "Waiting_at_out_one_observed_unswapped_ReadyThread_then_in"
                )
            else:
                interval["classification"] = "off_CPU_state_transition_unresolved"
            intervals.append(interval)
            pending, ready = None, []
        if pending:
            reject(tid, "out_without_observed_in", [pending["record_index"]])
    selected = None
    if native_ranges is not None and native_ranges.get("state") == "native_measured_windows_bound":
        selected = {}
        for kind, key in (
            ("off_cpu", None),
            ("waiting", "waiting_segment"),
            ("runnable", "runnable_segment"),
        ):
            segments = [
                dict(row, start_ns=row[key][0], end_ns=row[key][1]) if key else row
                for row in intervals
                if key is None or row[key] is not None
            ]
            result = select_intervals(
                segments, snapshot, process["selected_pid"], native_ranges=native_ranges
            )
            result["conditional_thread_ns"] = result.pop("conditional_off_cpu_thread_ns")
            result["exclusive_phase_thread_ns"] = result.pop("exclusive_phase_off_cpu_thread_ns")
            selected[kind] = result
    return dict(
        schema="rns8_etl_schedule_observations_v1",
        selected_pid=process["selected_pid"],
        intervals=intervals,
        unresolved=unresolved,
        observed_global_schedule_events=observed,
        retained_schedule_events=len(rows),
        undecoded_global_schedule_events=undecoded,
        excluded_global_schedule_events=excluded,
        conditional_off_cpu_thread_ns=sum(r["end_ns"] - r["start_ns"] for r in intervals),
        conditional_waiting_thread_ns=sum(
            r["waiting_segment"][1] - r["waiting_segment"][0]
            for r in intervals
            if r["waiting_segment"]
        ),
        conditional_runnable_thread_ns=sum(
            r["runnable_segment"][1] - r["runnable_segment"][0]
            for r in intervals
            if r["runnable_segment"]
        ),
        duration_condition="no_unobserved_intervening_switch_ready_or_thread_generation_boundary",
        selected_operation_by_state=selected,
        timestamp_domain="native_ETL_FILETIME_relative_to_process_start_ns",
        native_record_namespace="second_native_event_walk",
        active_spin_time_ns=None,
        wait_object_or_lock_owner_verified=False,
        waker_or_GPU_dependency_verified=False,
        complete_scheduler_coverage_proven=False,
        production_latency=False,
        promotion_allowed=False,
    )
