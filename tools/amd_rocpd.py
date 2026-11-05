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
    aggregation: str = (
        "individual_native_observation_no_cross_instance_or_replay_aggregation"
    )
    definition_version: str = "AMD_rocpd_3.0.4"


def _uint(value, name: str, *, positive=False) -> int:
    if type(value) is not int or not (1 if positive else 0) <= value < 2**64:
        raise ValueError("native rocpd " + name + " must be an unsigned integer")
    return value


def _optional_uint(value, name: str):
    return None if value is None else _uint(value, name)


def read_process(
    path: Path, process_id: int, *, maximum_rows=1_000_000, timeout_seconds=20.0
) -> dict:
    path = path.resolve(strict=True)
    _uint(process_id, "process ID", positive=True)
    if (
        type(maximum_rows) is not int
        or maximum_rows < 1
        or not 0 < timeout_seconds <= 300
    ):
        raise ValueError("positive bounded native database limits required")
    # An immutable receipt must include a finalized standalone database, never
    # ignore a live WAL or let SQLite create/update sidecars while reading it.
    if any(
        Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")
    ):
        raise ValueError("native database is not a finalized standalone snapshot")
    deadline = time.monotonic() + timeout_seconds
    try:
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=1)
        ) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.enable_load_extension(False)
            connection.set_progress_handler(
                lambda: int(time.monotonic() > deadline), 1000
            )
            connection.row_factory = sqlite3.Row

            def rows(table: str, columns: str, where="", values=()):
                # Identifiers/conditions are source constants; all input values
                # are parameters. Bound every materialized table projection.
                cursor = connection.execute(
                    "SELECT " + columns + " FROM " + table + where, values
                )
                result = cursor.fetchmany(maximum_rows + 1)
                if len(result) > maximum_rows:
                    raise ValueError(
                        "native table exceeds the bounded adapter row limit"
                    )
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
                raise ValueError(
                    "native database does not uniquely identify this process/node"
                )
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
            symbols = indexed(
                "rocpd_info_kernel_symbol", "id,code_object_id,kernel_name"
            )
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
            events = rows("rocpd_event", "id,correlation_id", " WHERE guid=?", (guid,))
            event_ids = {}
            for event in events:
                identifier = _uint(event["id"], "event ID", positive=True)
                if identifier in event_ids:
                    raise ValueError("ambiguous native event identity")
                event_ids[identifier] = event
            normalized, dispatch_events, seen = [], {}, set()
            for dispatch in dispatches.values():
                aid, kid, qid = (
                    dispatch[key] for key in ("agent_id", "kernel_id", "queue_id")
                )
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
                    raise ValueError(
                        "dispatch refers outside its process/node/GUID inventories"
                    )
                code = objects.get(symbol["code_object_id"])
                if (
                    code is None
                    or code["agent_id"] != aid
                    or code["storage_type"] != "FILE"
                ):
                    raise ValueError(
                        "dispatch has no matching file-backed loaded code object"
                    )
                event = event_ids.get(dispatch["event_id"])
                if event is None or dispatch["event_id"] in dispatch_events:
                    raise ValueError(
                        "dispatch has missing or ambiguous native event association"
                    )
                start, stop = (
                    _uint(dispatch["start"], "dispatch start"),
                    _uint(dispatch["end"], "dispatch end"),
                )
                if not begin <= start < stop <= end:
                    raise ValueError(
                        "dispatch lies outside its completed process interval"
                    )
                info = dict(
                    agent_id=dict(handle=aid),
                    kernel_id=kid,
                    queue_id=dict(handle=qid),
                    dispatch_id=dispatch["dispatch_id"],
                    grid_size={
                        axis: _uint(
                            dispatch["grid_size_" + axis], "grid size", positive=True
                        )
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
                            internal=_uint(event["correlation_id"], "correlation ID"),
                            external=0,
                        ),
                        native_rocpd=dict(
                            guid=guid,
                            node_id=nid,
                            stream_id=dispatch["stream_id"],
                            event_id=dispatch["event_id"],
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
                if (
                    sample["sample_id"] is not None
                    or dispatch is None
                    or definition is None
                ):
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
                    float(value)
                    if type(value) in (int, float) and math.isfinite(value)
                    else None
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
                            _optional_uint(
                                definition["instance_id"], "definition instance"
                            ),
                            dispatch["start"],
                            dispatch["end"],
                            missingness,
                        )
                    )
                )
            return dict(
                schema_version=VERSION,
                process_guid=guid,
                identity_namespace="rocpd_3.0.4_process_node_GUID_scoped_database_ids",
                native=dict(
                    metadata=dict(pid=process_id, init_time=begin, fini_time=end),
                    agents=[
                        dict(id=dict(handle=key), name=row["name"])
                        for key, row in agents.items()
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
    except sqlite3.Error as error:
        raise ValueError(
            "unsupported, malformed, nonfinalized or over-budget rocpd database: "
            + str(error)
        ) from error
