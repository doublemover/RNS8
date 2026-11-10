"""Observed Windows image generations; paths/checksums are not loaded PDB proof."""

from __future__ import annotations

from collections import defaultdict

from etl_process_contract import integer, SCHEMA


def image_catalog(metadata, process):
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != SCHEMA
        or not isinstance(process, dict)
        or process.get("schema") != "rns8_etl_selected_process_custody_v1"
        or type(process.get("selected_pid")) is not int
        or type(process.get("process_identity_complete")) is not bool
    ):
        raise ValueError("typed native process custody required for image attribution")
    pid = integer(metadata.get("selected_pid"), "selected image PID", minimum=1, maximum=2**32 - 1)
    if (
        process.get("selected_pid") != pid
        or metadata.get("event_timestamp_domain") != "native_ProcessTrace_converted_FILETIME_100ns"
    ):
        raise ValueError("native image/process identity or clock differs")
    observed = integer(metadata.get("observed_image_events"), "image events", maximum=1000000)
    undecoded = integer(
        metadata.get("undecoded_image_events"), "undecoded image events", maximum=observed
    )
    if observed > integer(metadata.get("observed_events"), "all native events", maximum=1000000):
        raise ValueError("native image count exceeds the event walk")
    rows = metadata.get("image_events")
    if (
        not isinstance(rows, list)
        or len(rows) > min(4096, observed)
        or len(rows) + undecoded > observed
    ):
        raise ValueError("bounded selected native image events required")
    first = integer(metadata["header"]["start_filetime_100ns"], "trace start", maximum=2**63 - 1)
    last = integer(
        metadata["header"]["end_filetime_100ns"], "trace end", minimum=first, maximum=2**63 - 1
    )
    groups, text_bytes = defaultdict(list), 0
    for position, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or integer(row.get("pid"), "image PID", minimum=1, maximum=2**32 - 1) != pid
        ):
            raise ValueError("native image belongs to another process")
        opcode = integer(row.get("opcode"), "image opcode", maximum=255)
        stamp = integer(row.get("time_filetime_100ns"), "image timestamp", maximum=2**63 - 1)
        base = integer(row.get("image_base"), "image base", minimum=1)
        size = integer(row.get("image_size"), "image size", minimum=1)
        integer(row.get("image_checksum"), "image checksum", maximum=2**32 - 1)
        integer(row.get("image_timestamp"), "native image TimeDateStamp", maximum=2**32 - 1)
        integer(row.get("event_version"), "image event version", maximum=255)
        width = integer(row.get("pointer_bytes"), "image pointer width", maximum=8)
        path = row.get("image_path_utf8")
        if (
            opcode not in {2, 3, 4, 10}
            or width not in {4, 8}
            or base + size > 1 << (8 * width)
            or width == 8
            and base + size == 1 << 64
            or not first <= stamp <= last
        ):
            raise ValueError("native image event span/opcode/width/window invalid")
        if (
            type(path) is not str
            or not path
            or len(path.encode("utf-8")) > 16384
            or len(path.encode("utf-16-le")) + 2 > 8192
            or any(ord(c) < 32 for c in path)
        ):
            raise ValueError("bounded nonempty native image path required")
        text_bytes += len(path.encode("utf-8"))
        if text_bytes > 1024**2:
            raise ValueError("native image path budget exhausted")
        groups[base].append(dict(row, record_index=position))
    generations, unresolved, rundown = [], [], []
    fields = ("image_size", "image_checksum", "image_timestamp", "image_path_utf8")
    for base, records in sorted(groups.items()):
        same_time = defaultdict(list)
        for row in records:
            same_time[row["time_filetime_100ns"]].append(row)
        live = None
        for stamp, events in sorted(same_time.items()):
            if len(events) != 1:
                if live is not None:
                    live["state"] = "ambiguous_following_image_events"
                live = None
                unresolved.extend(
                    dict(record=row, reason="same_base_same_timestamp_order_unresolved")
                    for row in events
                )
                continue
            row = events[0]
            if row["opcode"] in {3, 4}:
                rundown.append(row)
                if live is not None and any(live["load"][field] != row[field] for field in fields):
                    live["state"] = "conflicting_rundown_snapshot"
                    live = None
                    unresolved.append(
                        dict(record=row, reason="rundown_differs_from_observed_live_image")
                    )
                continue
            if row["opcode"] == 10:
                if live is not None:
                    live["state"] = "replacement_without_observed_unload"
                    unresolved.append(dict(record=row, reason="prior_image_unload_unobserved"))
                live = dict(
                    generation_id=row["record_index"],
                    image_base=base,
                    load=row,
                    unload=None,
                    state="load_observed_unload_unobserved",
                )
                generations.append(live)
            elif live is None:
                unresolved.append(dict(record=row, reason="unmatched_image_unload"))
            elif any(live["load"][field] != row[field] for field in fields):
                live["state"] = "conflicting_unload_metadata"
                unresolved.append(
                    dict(record=row, reason="unload_differs_from_observed_live_image")
                )
                live = None
            else:
                live.update(
                    unload=row,
                    state="observed_load_unload_pair",
                    observed_lifetime_100ns=stamp - live["load"]["time_filetime_100ns"],
                )
                live = None
    return dict(
        schema="rns8_etl_image_catalog_v1",
        selected_pid=pid,
        event_timestamp_domain=metadata["event_timestamp_domain"],
        observed_image_events=observed,
        undecoded_image_events=undecoded,
        selected_image_events=len(rows),
        generations=generations,
        rundown_snapshots=rundown,
        unresolved=unresolved,
        selected_process_identity_complete=process["process_identity_complete"],
        raw_module_presence_observed=bool(rows),
        scope="native_image_paths_spans_checksums_timestamps_and_observed_generations_not_loaded_file_bytes_or_PDB_identity",
        identity_namespace="selected_native_ETL_image_record_index_within_capture",
        complete_image_history_proven=False,
        loaded_PDB_identity_verified=False,
        CSV_module_or_IP_join="unbound",
        production_latency=False,
        promotion_allowed=False,
    )
