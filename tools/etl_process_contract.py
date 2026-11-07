"""Typed custody for native ETL loss counters and a selected process lifetime."""

from __future__ import annotations

from pathlib import PureWindowsPath

SCHEMA = "rns8_native_etl_process_metadata_v1"


def integer(value, label, *, maximum=2**64 - 1, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("native ETL integer outside domain: " + label)
    return value


def process_contract(
    metadata, *, pid, parent_pid, executable, decoder_pid, source_sha256, file_bytes
):
    if (
        type(metadata) is not dict
        or metadata.get("schema") != SCHEMA
        or metadata.get("reader_source_sha256") != source_sha256
        or type(metadata.get("decoder_pid")) is not int
        or metadata["decoder_pid"] != decoder_pid
        or type(metadata.get("selected_pid")) is not int
        or metadata["selected_pid"] != pid
        or metadata.get("complete_event_walk") is not True
        or type(metadata.get("native_process_trace_status")) is not int
        or metadata["native_process_trace_status"] != 0
        or type(metadata.get("file_bytes")) is not int
        or metadata["file_bytes"] != file_bytes
        or metadata.get("event_timestamp_domain") != "native_ProcessTrace_converted_FILETIME_100ns"
        or any(
            metadata.get(key) is not False
            for key in (
                "complete_loss_free_acquisition_proven",
                "image_path_or_loaded_PDB_verified",
                "production_latency",
                "promotion_allowed",
            )
        )
    ):
        raise ValueError("native ETL decoder identity/PID/file/completion contract differs")
    integer(pid, "selected PID", minimum=1, maximum=2**32 - 1)
    integer(parent_pid, "parent PID", minimum=1, maximum=2**32 - 1)
    integer(decoder_pid, "decoder PID", minimum=1, maximum=2**32 - 1)
    events = integer(metadata.get("observed_events"), "event count", minimum=1, maximum=1000000)
    processes = integer(
        metadata.get("observed_process_events"), "process event count", maximum=events
    )
    undecoded = integer(
        metadata.get("undecoded_process_events"), "undecoded process events", maximum=processes
    )
    header = metadata.get("header")
    if (
        type(header) is not dict
        or type(header.get("pointer_bytes")) is not int
        or header["pointer_bytes"] != 8
    ):
        raise ValueError("native-width Windows ETL header required")
    start = integer(header.get("start_filetime_100ns"), "trace start", maximum=2**63 - 1)
    end = integer(header.get("end_filetime_100ns"), "trace end", minimum=start, maximum=2**63 - 1)
    if not end:
        raise ValueError("finalized ETL header required")
    clock = integer(header.get("clock_type"), "recorded clock", maximum=2**32 - 1)
    frequency = integer(header.get("perf_frequency"), "performance frequency", maximum=2**63 - 1)
    lost_events = integer(header.get("events_lost"), "header events lost", maximum=2**32 - 1)
    lost_buffers = integer(header.get("buffers_lost"), "header buffers lost", maximum=2**32 - 1)
    integer(header.get("buffers_written"), "buffers written", maximum=2**32 - 1)
    rows = metadata.get("process_events")
    if type(rows) is not list or len(rows) > min(16384, processes):
        raise ValueError("bounded native selected process records required")
    keys, starts, stops, blockers = set(), [], [], []
    expected = PureWindowsPath(str(executable)).name.casefold()
    for row in rows:
        if type(row) is not dict or type(row.get("pid")) is not int or row["pid"] != pid:
            raise ValueError("native ETL process record belongs to another PID")
        key = integer(row.get("process_key"), "process key", minimum=1)
        parent = integer(row.get("parent_pid"), "native parent PID", maximum=2**32 - 1)
        timestamp = integer(row.get("time_filetime_100ns"), "process timestamp", maximum=2**63 - 1)
        opcode = integer(row.get("opcode"), "process opcode", maximum=255)
        integer(row.get("event_version"), "process event version", maximum=255)
        if (
            opcode not in {1, 2, 3, 4, 39}
            or type(row.get("pointer_bytes")) is not int
            or row["pointer_bytes"] not in (4, 8)
        ):
            raise ValueError("unsupported native process event or pointer width")
        if not start <= timestamp <= end:
            raise ValueError("native process event is outside the finalized trace window")
        keys.add(key)
        image = row.get("image_basename_ascii")
        if image is None:
            blockers.append("native_image_basename_unresolved")
        elif (
            type(image) is not str
            or not image.isascii()
            or len(image) > 4096
            or image.casefold() != expected
        ):
            raise ValueError("native ETL selected process image differs from benchmark")
        if parent != parent_pid:
            raise ValueError("native ETL process was not launched by this recording owner")
        if opcode == 1:
            starts.append(timestamp)
        elif opcode == 2:
            stops.append(timestamp)
    if len(keys) != 1:
        blockers.append("selected_PID_missing_or_reused_in_native_recording")
    if len(starts) != 1 or len(stops) != 1:
        blockers.append("one_native_Start_and_End_pair_not_observed")
    elif starts[0] > stops[0]:
        raise ValueError("native process end precedes its start")
    if undecoded:
        blockers.append("undecoded_native_process_records_can_hide_PID_reuse")
    if clock not in (1, 2, 3) or clock == 1 and not frequency:
        blockers.append("native_clock_type_or_frequency_unresolved")
    if lost_events or lost_buffers:
        blockers.append("native_header_reports_event_or_buffer_loss")
    return dict(
        schema="rns8_etl_selected_process_custody_v1",
        selected_pid=pid,
        parent_pid=parent_pid,
        process_key=next(iter(keys)) if len(keys) == 1 else None,
        actual_start_filetime_100ns=starts[0] if len(starts) == 1 else None,
        actual_end_filetime_100ns=stops[0] if len(stops) == 1 else None,
        header_loss=dict(
            events_lost=lost_events, buffers_lost=lost_buffers, counts_are_additive=False
        ),
        clock=dict(
            native_type=clock,
            perf_frequency=frequency,
            event_domain=metadata["event_timestamp_domain"],
        ),
        blockers=sorted(set(blockers)),
        process_identity_complete=not blockers,
        scope="one_observed_native_process_key_Start_End_parent_and_ASCII_image_not_loaded_PDB_or_CSV_clock_join",
        complete_loss_free_acquisition_proven=False,
        promotion_allowed=False,
    )
