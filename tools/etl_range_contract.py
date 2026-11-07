"""Bind private native ETW marker slots to completed benchmark operation scopes."""

from __future__ import annotations

from collections import defaultdict

from etl_process_contract import integer
from host_range_contract import range_inventory, measured_roots
from native_phase_attribution import attribute


def operation_ranges(metadata, process, snapshot):
    if (
        not isinstance(process, dict)
        or process.get("schema") != "rns8_etl_selected_process_custody_v1"
        or process.get("process_identity_complete") is not True
    ):
        raise ValueError("complete selected native ETL process custody required for markers")
    pid = integer(process.get("selected_pid"), "marker PID", minimum=1, maximum=2**32 - 1)
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != "rns8_native_etl_process_metadata_v1"
        or type(metadata.get("selected_pid")) is not int
        or metadata["selected_pid"] != pid
    ):
        raise ValueError("native ETL marker metadata belongs to another process")
    total = integer(
        metadata.get("observed_events"), "native event count", minimum=1, maximum=1_000_000
    )
    first = integer(process.get("actual_start_filetime_100ns"), "process start", maximum=2**63 - 1)
    last = integer(
        process.get("actual_end_filetime_100ns"), "process end", minimum=first, maximum=2**63 - 1
    )
    index = range_inventory(snapshot, pid, clock_kind="windows_qpc")
    if (
        process["clock"]["native_type"] == 1
        and process["clock"]["perf_frequency"] != snapshot["host_clock"]["frequency_hz"]
    ):
        raise ValueError("native ETL and observed QPC frequencies contradict each other")
    if (
        snapshot.get("marker_label_contract") != "rns8_etw_slot_v1"
        or type(snapshot.get("native_write_failures")) is not int
        or snapshot["native_write_failures"] != 0
    ):
        raise ValueError("complete native ETW slot writer contract required")
    observed = integer(metadata.get("observed_marker_events"), "marker count", maximum=total)
    undecoded = integer(
        metadata.get("undecoded_marker_events"), "undecoded markers", maximum=observed
    )
    rows = metadata.get("marker_events")
    if (
        not isinstance(rows, list)
        or len(rows) > min(8192, observed)
        or len(rows) + undecoded > observed
    ):
        raise ValueError("bounded native ETW marker inventory required")
    by_slot, seen = defaultdict(list), set()
    for row in rows:
        if not isinstance(row, dict) or row.get("wire_contract") != "rns8_etw_slot_v1":
            raise ValueError("unknown native ETW marker wire contract")
        if integer(row.get("pid"), "native marker PID", minimum=1, maximum=2**32 - 1) != pid:
            raise ValueError("native ETW marker belongs to another process")
        integer(row.get("tid"), "native marker TID", minimum=1, maximum=2**32 - 1)
        ordinal = integer(row.get("record_index"), "marker record index", minimum=1, maximum=total)
        if ordinal in seen:
            raise ValueError("duplicate native ETW record identity")
        seen.add(ordinal)
        slot = integer(row.get("slot"), "marker slot", maximum=4095)
        event = integer(row.get("event_id"), "marker event", minimum=1, maximum=2)
        integer(row.get("time_filetime_100ns"), "marker timestamp", minimum=first, maximum=last)
        by_slot[slot].append((event, row))
    markers, unresolved = {}, []
    for slot, scope in enumerate(snapshot["records"]):
        if scope["range_id"] != slot:
            raise ValueError("ETW range ID differs from its actual native slot")
        events = by_slot.pop(slot, [])
        starts = [row for event, row in events if event == 1]
        stops = [row for event, row in events if event == 2]
        if len(starts) != 1 or len(stops) != 1 or not scope["stop_state"]:
            unresolved.append(
                dict(slot=slot, reason="native_start_stop_or_host_completion_unresolved")
            )
            continue
        start, stop = starts[0], stops[0]
        if (
            start["tid"] != scope["start_thread_id"]
            or stop["tid"] != scope["stop_thread_id"]
            or start["time_filetime_100ns"] > stop["time_filetime_100ns"]
        ):
            raise ValueError("native ETW marker threads/order contradict host snapshot")
        markers[slot] = dict(
            id=slot,
            start_ns=(start["time_filetime_100ns"] - first) * 100,
            end_ns=(stop["time_filetime_100ns"] - first) * 100,
            start_record=start["record_index"],
            stop_record=stop["record_index"],
        )
    if by_slot:
        raise ValueError("native ETW markers have no matching snapshot slot")
    roots = measured_roots(index)
    for identity, row in roots.items():
        if row["stop_state"] != 1 or row["start_thread_id"] != row["stop_thread_id"]:
            unresolved.append(
                dict(slot=identity, reason="measured_root_not_completed_synchronous_host_scope")
            )
    missing_roots = sorted(set(roots) - markers.keys())
    if missing_roots:
        unresolved.append(
            dict(reason="measured_native_window_unresolved", host_range_ids=missing_roots)
        )
    phases = None
    if not unresolved and not undecoded:
        for identity, row in index.records.items():
            parent = row["parent_range_id"]
            if (
                parent is not None
                and row["stop_state"] == 1
                and row["start_thread_id"] == row["stop_thread_id"]
            ):
                if (
                    not markers[parent]["start_ns"]
                    <= markers[identity]["start_ns"]
                    <= markers[identity]["end_ns"]
                    <= markers[parent]["end_ns"]
                ):
                    raise ValueError("native ETW child escapes its parent window")
        phases = attribute(
            index,
            markers,
            roots,
            [],
            clock_domain="native_ETL_FILETIME_relative_to_process_start_ns",
        )
    return dict(
        schema="rns8_etl_operation_ranges_v1",
        selected_pid=pid,
        process_key=process["process_key"],
        state="native_measured_windows_bound"
        if phases is not None
        else "partial_native_marker_association",
        timestamp_domain="native_ProcessTrace_converted_FILETIME_100ns",
        phase_origin_filetime_100ns=first,
        phase_unit="nanoseconds_relative_to_observed_process_start",
        markers=[dict(slot=slot, **row) for slot, row in sorted(markers.items())],
        measured_root_ids=sorted(roots),
        unresolved=unresolved,
        undecoded_marker_events=undecoded,
        phase_attribution=phases,
        header_loss=process["header_loss"],
        wpa_relative_clock_join="unbound",
        host_QPC_to_ETL_clock_join="unbound",
        GPU_clock_join="unbound",
        complete_loss_free_acquisition_proven=False,
        production_latency=False,
        promotion_allowed=False,
    )
