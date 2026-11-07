"""Read immutable ROCm profiling databases through the native 3.0.4 schema.

Schema source: ROCm/rocm-systems 7da43026e4f2ef7600433c38c820732f7a205d22,
projects/rocprofiler-sdk/source/share/rocprofiler-sdk-rocpd/rocpd_tables.sql.
No display-name, timestamp-nearest or ordinal-to-agent inference is performed.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sqlite3
import time

from json_evidence import parse_evidence_json

VERSION = "3.0.4"


@dataclass(frozen=True)
class NativeMetric:
    process_id: int
    agent_id: int
    queue_id: int
    dispatch_id: int
    event_id: int
    metric_id: int
    raw_name: str
    symbol: str
    description: str | None
    unit: str | None
    block: str | None
    vendor_expression: str | None
    value_type: str | None
    is_derived: int | None
    value: float | None
    xcc: int | None
    shader_engine: int | None
    instance: int | None
    definition_instance: int | None
    start_ns: int
    end_ns: int
    missingness: str | None
    aggregation: str = "individual_native_observation_no_cross_instance_or_replay_aggregation"
    definition_version: str = "AMD_rocpd_3.0.4"


def _uint(value, name: str, *, positive=False) -> int:
    if type(value) is not int or not (1 if positive else 0) <= value < 2**64:
        raise ValueError("native rocpd " + name + " must be an unsigned integer")
    return value


def _optional_uint(value, name: str):
    return None if value is None else _uint(value, name)


def _read_timeline(rows, indexed, process, *, agents, queues, streams):
    """Read only real3.0.4 tables; database IDs never become HIP ordinals."""
    guid, begin, end = process["guid"], process["init"], process["fini"]
    owners = rows("rocpd_info_process", "id", " WHERE guid=?", (guid,))
    if len(owners) != 1 or owners[0]["id"] != process["id"]:
        raise ValueError("native timeline GUID does not have one unambiguous process owner")
    strings = {}
    for row in rows("rocpd_string", "id,string", " WHERE guid=?", (guid,)):
        key = _uint(row["id"], "string ID", positive=True)
        value = row["string"]
        if key in strings or type(value) is not str or "\0" in value or len(value.encode()) > 4096:
            raise ValueError("native rocpd string identity/value invalid or over budget")
        strings[key] = value
    threads = indexed("rocpd_info_thread", "id,tid")
    for thread in threads.values():
        _uint(thread["tid"], "native thread ID", positive=True)
    events = {}
    for row in rows(
        "rocpd_event",
        "id,stack_id,parent_stack_id,correlation_id,category_id,json(extdata) AS extdata",
        " WHERE guid=?",
        (guid,),
    ):
        key = _uint(row["id"], "timeline event ID", positive=True)
        if key in events:
            raise ValueError("duplicate native timeline event identity")
        category = _optional_uint(row.pop("category_id"), "category string ID")
        if category is not None and category not in strings:
            raise ValueError("native event category has no same-GUID string")
        row["category"] = strings.get(category)
        for field in ("stack_id", "parent_stack_id", "correlation_id"):
            row[field] = _optional_uint(row[field], field)
        row["extdata"] = parse_evidence_json(row["extdata"])
        events[key] = row

    def interval(row):
        first, last = _uint(row["start"], "timeline start"), _uint(row["end"], "timeline end")
        if not begin <= first <= last <= end:
            raise ValueError("native timeline interval is outside its completed process")
        return first, last

    def name(identifier, field):
        if identifier is None:
            return None
        _uint(identifier, field, positive=True)
        if identifier not in strings:
            raise ValueError("native timeline " + field + " has no same-GUID string")
        return strings[identifier]

    def reference(identifier, inventory, field):
        if identifier is not None:
            _uint(identifier, field, positive=True)
        if identifier is not None and identifier not in inventory:
            raise ValueError("native timeline " + field + " is outside its process/node/GUID")
        return identifier

    specifications = (
        ("host_region", "rocpd_region", "id,tid,start,end,name_id,event_id", ()),
        (
            "memory_copy",
            "rocpd_memory_copy",
            "id,tid,start,end,name_id,dst_agent_id,dst_address,src_agent_id,src_address,size,queue_id,stream_id,graph_exec_id,graph_node_id,region_name_id,event_id",
            ("size", "graph_exec_id", "graph_node_id"),
        ),
        (
            "memory_lifecycle",
            "rocpd_memory_allocate",
            "id,tid,agent_id,type,level,start,end,address,size,queue_id,stream_id,event_id,json(extdata) AS extdata",
            ("size",),
        ),
        (
            "hip_event",
            "rocpd_hip_event",
            "id,tid,start,end,name_id,agent_id,queue_id,stream_id,hip_event_handle,source_queue_id,event_id",
            ("hip_event_handle",),
        ),
        (
            "graph_launch",
            "rocpd_graph_launch",
            "id,tid,agent_id,queue_id,start,end,graph_exec_id,kernel_dispatch_count,event_id",
            ("graph_exec_id", "kernel_dispatch_count"),
        ),
        (
            "kernel_dispatch",
            "rocpd_kernel_dispatch",
            "id,tid,agent_id,queue_id,stream_id,start,end,dispatch_id,kernel_id,region_name_id,event_id,graph_exec_id,graph_node_id",
            ("dispatch_id", "kernel_id", "graph_exec_id", "graph_node_id"),
        ),
    )
    records, counts = [], {}
    for kind, table, columns, numeric in specifications:
        entries = indexed(table, columns)
        counts[kind] = len(entries)
        for row in entries.values():
            row = dict(row)
            first, last = interval(row)
            thread = reference(row.pop("tid"), threads, "thread")
            event = reference(row.pop("event_id"), events, "event")
            if kind == "host_region" and (thread is None or event is None):
                raise ValueError("native host region has no actual thread/event association")
            for field in numeric:
                row[field] = _uint(row[field], field)
            for field in ("address", "dst_address", "src_address"):
                if field in row:
                    row[field] = _optional_uint(row[field], field)
            for field, inventory in (
                ("agent_id", agents),
                ("dst_agent_id", agents),
                ("src_agent_id", agents),
                ("queue_id", queues),
                ("source_queue_id", queues),
                ("stream_id", streams),
            ):
                if field in row:
                    row[field] = reference(row[field], inventory, field)
            for field in ("name_id", "region_name_id"):
                if field in row:
                    row[field.removesuffix("_id")] = name(row.pop(field), field)
            if kind == "memory_lifecycle":
                if row["type"] not in {"ALLOC", "FREE", "REALLOC", "RECLAIM"} or row[
                    "level"
                ] not in {"REAL", "VIRTUAL", "SCRATCH"}:
                    raise ValueError("unknown native memory lifecycle type/level")
                row["extdata"] = parse_evidence_json(row["extdata"])
            row.update(
                kind=kind,
                start_ns=first,
                end_ns=last,
                native_thread_table_id=thread,
                thread_id=threads[thread]["tid"] if thread is not None else None,
                native_event_id=event,
            )
            records.append(row)
            if len(records) > 100_000:
                raise ValueError("native interval timeline exceeds its record budget")
    return dict(
        schema="rns8_native_rocpd_timeline_v1",
        process_id=process["pid"],
        native_process_table_id=process["id"],
        native_node_id=process["nid"],
        process_guid=guid,
        clock="native_ROCprofiler_SDK_timestamp_ns_in_one_process",
        process_interval_ns=[begin, end],
        records=records,
        events=list(events.values()),
        agents=list(agents.values()),
        record_identity="kind_and_native_table_id_within_process_node_GUID",
        record_counts=counts,
        empty_table_means="no_observation_not_zero_work_or_full_coverage",
        instant_samples="not_in_this_interval_projection",
        loss_status="unknown_no_loss_counter_in_this_schema_projection",
        complete_run_coverage=False,
        production_latency=False,
    )


def read_process(
    path: Path,
    process_id: int,
    *,
    maximum_rows=1_000_000,
    timeout_seconds=20.0,
    include_timeline=False,
) -> dict:
    path = path.resolve(strict=True)
    _uint(process_id, "process ID", positive=True)
    if (
        type(maximum_rows) is not int
        or not 1 <= maximum_rows <= 1_000_000
        or type(timeout_seconds) not in (int, float)
        or not 0 < timeout_seconds <= 300
        or type(include_timeline) is not bool
    ):
        raise ValueError("positive bounded native database limits required")
    # An immutable receipt must include a finalized standalone database, never
    # ignore a live WAL or let SQLite create/update sidecars while reading it.
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("native database is not a finalized standalone snapshot")
    deadline = time.monotonic() + timeout_seconds
    try:
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=1)
        ) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.enable_load_extension(False)
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1024**2)
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            connection.row_factory = sqlite3.Row
            text_bytes = 0

            def bounded_text(raw):
                nonlocal text_bytes
                text_bytes += len(raw)
                if text_bytes > 64 * 1024**2:
                    raise ValueError("native rocpd cumulative decoded-text budget exhausted")
                return raw.decode("utf-8")

            connection.text_factory = bounded_text

            remaining_rows = maximum_rows

            def rows(table: str, columns: str, where="", values=()):
                nonlocal remaining_rows
                # Identifiers/conditions are source constants; all input values
                # are parameters. Bound every materialized table projection.
                order = " ORDER BY id" if columns == "id" or columns.startswith("id,") else ""
                cursor = connection.execute(
                    "SELECT " + columns + " FROM " + table + where + order, values
                )
                result = cursor.fetchmany(remaining_rows + 1)
                if len(result) > remaining_rows:
                    raise ValueError("native database exceeds the cumulative adapter row limit")
                remaining_rows -= len(result)
                return [dict(row) for row in result]

            versions = rows("rocpd_metadata", "value", " WHERE tag='schema_version'")
            if not versions or {row["value"] for row in versions} != {VERSION}:
                raise ValueError("native rocpd schema version is unsupported or mixed")
            processes = rows(
                "rocpd_info_process",
                "id,guid,nid,pid,init,fini",
                " WHERE pid=?",
                (process_id,),
            )
            if len(processes) != 1:
                raise ValueError("native database does not uniquely identify this process/node")
            process = processes[0]
            pid, nid, guid = process["id"], process["nid"], process["guid"]
            _uint(pid, "process table ID", positive=True)
            _uint(nid, "node table ID", positive=True)
            if not isinstance(guid, str) or not guid:
                raise ValueError("native database process has no capture GUID")
            begin, end = (
                _uint(process["init"], "process init"),
                _uint(process["fini"], "process fini"),
            )
            if end <= begin:
                raise ValueError("native process did not finalize")
            select = " WHERE pid=? AND nid=? AND guid=?"
            parameters = (pid, nid, guid)

            def indexed(table, columns):
                result = {}
                for row in rows(table, columns, select, parameters):
                    identifier = _uint(row["id"], table + " ID", positive=True)
                    if identifier in result:
                        raise ValueError("duplicate native database identity")
                    result[identifier] = row
                return result

            agents = indexed(
                "rocpd_info_agent",
                "id,name,type,uuid,absolute_index,logical_index,type_index",
            )
            objects = indexed("rocpd_info_code_object", "id,agent_id,uri,storage_type")
            symbols = indexed("rocpd_info_kernel_symbol", "id,code_object_id,kernel_name")
            queues = indexed("rocpd_info_queue", "id")
            streams = indexed("rocpd_info_stream", "id")
            dispatches = indexed(
                "rocpd_kernel_dispatch",
                "id,agent_id,kernel_id,dispatch_id,queue_id,stream_id,start,end,event_id,"
                "grid_size_x,grid_size_y,grid_size_z,workgroup_size_x,workgroup_size_y,"
                "workgroup_size_z,graph_exec_id,graph_node_id,group_segment_size,private_segment_size",
            )
            if not dispatches:
                raise ValueError("native database has no executed dispatch")
            events = rows(
                "rocpd_event",
                "id,stack_id,parent_stack_id,correlation_id",
                " WHERE guid=?",
                (guid,),
            )
            event_ids = {}
            for event in events:
                identifier = _uint(event["id"], "event ID", positive=True)
                if identifier in event_ids:
                    raise ValueError("ambiguous native event identity")
                event_ids[identifier] = event
            normalized, dispatch_events, seen = [], {}, set()
            for dispatch in dispatches.values():
                aid, kid, qid = (dispatch[key] for key in ("agent_id", "kernel_id", "queue_id"))
                key = (
                    _uint(aid, "agent ID"),
                    _uint(qid, "queue ID"),
                    _uint(dispatch["dispatch_id"], "dispatch ID"),
                )
                if key in seen:
                    raise ValueError("duplicate process/agent/queue/dispatch identity")
                seen.add(key)
                symbol = symbols.get(kid)
                if (
                    aid not in agents
                    or agents[aid]["type"] != "GPU"
                    or symbol is None
                    or qid not in queues
                    or dispatch["stream_id"] not in streams
                ):
                    raise ValueError("dispatch refers outside its process/node/GUID inventories")
                code = objects.get(symbol["code_object_id"])
                if code is None or code["agent_id"] != aid or code["storage_type"] != "FILE":
                    raise ValueError("dispatch has no matching file-backed loaded code object")
                event = event_ids.get(dispatch["event_id"])
                if event is None or dispatch["event_id"] in dispatch_events:
                    raise ValueError("dispatch has missing or ambiguous native event association")
                start, stop = (
                    _uint(dispatch["start"], "dispatch start"),
                    _uint(dispatch["end"], "dispatch end"),
                )
                if not begin <= start < stop <= end:
                    raise ValueError("dispatch lies outside its completed process interval")
                info = dict(
                    agent_id=dict(handle=aid),
                    kernel_id=kid,
                    queue_id=dict(handle=qid),
                    dispatch_id=dispatch["dispatch_id"],
                    grid_size={
                        axis: _uint(dispatch["grid_size_" + axis], "grid size", positive=True)
                        for axis in "xyz"
                    },
                    workgroup_size={
                        axis: _uint(
                            dispatch["workgroup_size_" + axis],
                            "workgroup size",
                            positive=True,
                        )
                        for axis in "xyz"
                    },
                )
                normalized.append(
                    dict(
                        dispatch_info=info,
                        start_timestamp=start,
                        end_timestamp=stop,
                        correlation_id=dict(
                            # generateRocpd stores SDK internal identity in stack_id;
                            # correlation_id is the distinct caller/external value.
                            internal=_uint(event["stack_id"], "internal stack correlation ID"),
                            external=_uint(event["correlation_id"], "external correlation ID"),
                        ),
                        native_rocpd=dict(
                            guid=guid,
                            node_id=nid,
                            stream_id=dispatch["stream_id"],
                            event_id=dispatch["event_id"],
                            parent_stack_id=_optional_uint(
                                event["parent_stack_id"], "parent stack ID"
                            ),
                            graph_exec_id=dispatch["graph_exec_id"],
                            graph_node_id=dispatch["graph_node_id"],
                        ),
                    )
                )
                dispatch_events[dispatch["event_id"]] = dispatch
            definitions = indexed(
                "rocpd_info_pmc",
                "id,agent_id,name,symbol,description,units,block,expression,value_type,is_derived,instance_id",
            )
            samples = rows(
                "rocpd_pmc_event",
                "id,event_id,sample_id,pmc_id,value,xcc,shader_engine,instance",
                " WHERE guid=?",
                (guid,),
            )
            metrics, unresolved = [], []
            for sample in samples:
                dispatch = dispatch_events.get(sample["event_id"])
                definition = definitions.get(sample["pmc_id"])
                if sample["sample_id"] is not None or dispatch is None or definition is None:
                    unresolved.append(
                        dict(
                            sample_id=sample["id"],
                            reason="streaming_or_unmatched_native_counter",
                        )
                    )
                    continue
                if definition["agent_id"] != dispatch["agent_id"]:
                    raise ValueError("native metric belongs to another dispatch agent")
                value = sample["value"]
                numeric = (
                    float(value) if type(value) in (int, float) and math.isfinite(value) else None
                )
                unit = definition["units"] or None
                missingness = (
                    "native_value_unavailable"
                    if numeric is None
                    else "vendor_unit_unavailable"
                    if unit is None
                    else None
                )
                metrics.append(
                    asdict(
                        NativeMetric(
                            process_id,
                            dispatch["agent_id"],
                            dispatch["queue_id"],
                            dispatch["dispatch_id"],
                            sample["event_id"],
                            definition["id"],
                            definition["name"],
                            definition["symbol"],
                            definition["description"],
                            unit,
                            definition["block"],
                            definition["expression"],
                            definition["value_type"],
                            definition["is_derived"],
                            numeric,
                            _optional_uint(sample["xcc"], "XCC"),
                            _optional_uint(sample["shader_engine"], "shader engine"),
                            _optional_uint(sample["instance"], "instance"),
                            _optional_uint(definition["instance_id"], "definition instance"),
                            dispatch["start"],
                            dispatch["end"],
                            missingness,
                        )
                    )
                )
            result = dict(
                schema_version=VERSION,
                process_guid=guid,
                identity_namespace="rocpd_3.0.4_process_node_GUID_scoped_database_ids",
                native=dict(
                    metadata=dict(pid=process_id, init_time=begin, fini_time=end),
                    agents=[
                        dict(id=dict(handle=key), name=row["name"]) for key, row in agents.items()
                    ],
                    code_objects=[
                        dict(
                            code_object_id=key,
                            agent_id=dict(handle=row["agent_id"]),
                            uri=row["uri"],
                        )
                        for key, row in objects.items()
                    ],
                    kernel_symbols=[
                        dict(
                            kernel_id=key,
                            code_object_id=row["code_object_id"],
                            kernel_name=row["kernel_name"],
                        )
                        for key, row in symbols.items()
                    ],
                    buffer_records=dict(kernel_dispatch=normalized),
                ),
                metrics=metrics,
                unresolved_metrics=unresolved,
                loss_status="unknown_no_loss_counter_in_this_schema_projection",
                complete_run_coverage=False,
            )
            if include_timeline:
                result["timeline"] = _read_timeline(
                    rows, indexed, process, agents=agents, queues=queues, streams=streams
                )
            if time.monotonic() > deadline:
                raise ValueError("native rocpd materialization exceeded its time budget")
            return result
    except sqlite3.Error as error:
        raise ValueError(
            "unsupported, malformed, nonfinalized or over-budget rocpd database: " + str(error)
        ) from error
