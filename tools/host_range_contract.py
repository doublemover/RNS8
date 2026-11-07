"""Shared bounded native host-range identities; no sample or GPU-clock inference."""

from __future__ import annotations

from dataclasses import dataclass


PHASES = (
    "preparation",
    "source_pack",
    "proof",
    "dispatch_enqueue",
    "reduction",
    "export",
    "completion",
    "publication",
    "oracle",
    "cleanup",
    "public_call",
)
FIELDS = {
    "range_id",
    "parent_range_id",
    "operation_id",
    "iteration_id",
    "operation_namespace",
    "role",
    "phase",
    "stop_state",
    "process_id",
    "start_thread_id",
    "stop_thread_id",
    "start_before_ns",
    "start_after_ns",
    "stop_before_ns",
    "stop_after_ns",
}


def uint(value, name, *, positive=False):
    if type(value) is not int or not int(positive) <= value < 1 << 64:
        raise ValueError("invalid host range " + name)
    return value


@dataclass(frozen=True)
class RangeInventory:
    records: dict[int, dict]
    depth: dict[int, int]
    parents: dict[int, tuple[int, ...]]


def range_inventory(snapshot: dict, pid: int, *, clock_kind="linux_monotonic") -> RangeInventory:
    uint(pid, "selected PID", positive=True)
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema") != "rns8_profiler_ranges_v1"
        or snapshot.get("provider_status") != "active"
        or type(snapshot.get("dropped")) is not int
        or snapshot["dropped"] != 0
        or type(snapshot.get("unpublished")) is not int
        or snapshot["unpublished"] != 0
        or type(snapshot.get("claimed")) is not int
        or type(snapshot.get("capacity")) is not int
        or not 1 <= snapshot["claimed"] <= snapshot["capacity"] <= 4096
    ):
        raise ValueError("complete bounded active host range snapshot required")
    clock = snapshot.get("host_clock")
    linux_clock = (
        not isinstance(clock, dict)
        or clock.get("schema") != "rns8_linux_host_range_clock_v1"
        or type(clock.get("clock_id")) is not int
        or clock["clock_id"] != 1
        or clock.get("name") != "CLOCK_MONOTONIC"
        or clock.get("unit") != "nanoseconds"
        or clock.get("timestamp_scope") != "brackets_around_native_ROCTx_calls"
    )
    windows_clock = (
        not isinstance(clock, dict)
        or clock.get("schema") != "rns8_windows_host_range_clock_v1"
        or clock.get("name") != "QueryPerformanceCounter"
        or clock.get("unit") != "nanoseconds"
        or clock.get("timestamp_scope") != "brackets_around_native_ETW_calls"
        or type(clock.get("frequency_hz")) is not int
        or not 0 < clock["frequency_hz"] < 1 << 63
    )
    if (
        clock_kind == "linux_monotonic"
        and linux_clock
        or clock_kind == "windows_qpc"
        and windows_clock
        or clock_kind not in {"linux_monotonic", "windows_qpc"}
    ):
        raise ValueError("native host range clock metadata is unsupported")
    rows = snapshot.get("records")
    if not isinstance(rows, list) or len(rows) != snapshot["claimed"]:
        raise ValueError("range snapshot count differs from actual records")
    records, depth = {}, {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("unknown or incomplete host range record contract")
        for key in (
            "range_id",
            "operation_id",
            "iteration_id",
            "operation_namespace",
            "role",
            "phase",
            "stop_state",
        ):
            uint(row[key], key)
        if (
            row["range_id"] in records
            or row["operation_namespace"] > 2
            or row["role"] > 5
            or row["phase"] > 10
            or row["stop_state"] > 4
        ):
            raise ValueError("duplicate range identity or unknown host range tag")
        if uint(row["process_id"], "PID", positive=True) != pid:
            raise ValueError("host range belongs to a different process")
        uint(row["start_thread_id"], "start TID", positive=True)
        if row["parent_range_id"] is not None:
            uint(row["parent_range_id"], "parent ID")
        for key in ("start_before_ns", "start_after_ns", "stop_before_ns", "stop_after_ns"):
            if row[key] is not None:
                uint(row[key], key)
        if row["stop_state"]:
            uint(row["stop_thread_id"], "stop TID", positive=True)
        elif any(
            row[key] is not None for key in ("stop_thread_id", "stop_before_ns", "stop_after_ns")
        ):
            raise ValueError("pending range exposes unobserved stop metadata")
        records[row["range_id"]] = row
    for identity in records:
        path, seen, current = [], set(), identity
        while current is not None and current not in depth:
            if current not in records or current in seen or len(path) >= 128:
                raise ValueError("missing/cyclic/overbudget host range ancestry")
            seen.add(current)
            path.append(current)
            current = records[current]["parent_range_id"]
        level = 0 if current is None else depth[current]
        for item in reversed(path):
            level += 1
            depth[item] = level

    parents = {}
    for identity in records:
        chain = []
        current = records[identity]["parent_range_id"]
        while current is not None:
            chain.append(current)
            current = records[current]["parent_range_id"]
        parents[identity] = tuple(chain)
    return RangeInventory(records, depth, parents)


def measured_roots(index: RangeInventory) -> dict[int, dict]:
    records = index.records
    candidate_roots = {
        identity
        for identity, row in records.items()
        if row["role"] == 3 and row["operation_namespace"] == 1 and row["phase"] == 10
    }
    roots = {}
    for identity in sorted(candidate_roots):
        row = records[identity]
        parents = [parent for parent in index.parents[identity] if parent in candidate_roots]
        if any(records[parent]["operation_id"] != row["operation_id"] for parent in parents):
            raise ValueError(
                "nested distinct measured operations have ambiguous measured ownership"
            )
        if not parents:
            if not row["operation_id"] or row["operation_id"] in {
                item["operation_id"] for item in roots.values()
            }:
                raise ValueError("measured operation identity is zero or repeated")
            roots[identity] = row
    if not roots:
        raise ValueError("no measured benchmark-operation host range selected")
    return roots
