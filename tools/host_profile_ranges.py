"""Join native CPU samples to completed measured host ranges in the same clock."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
import heapq

from host_profile_native import SCHEMA as NATIVE_SCHEMA
from interval_observations import interval_summary
from perf_recording_contract import decoded_sample_witness

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


def join(native: dict, snapshot: dict) -> dict:
    if (
        not isinstance(native, dict)
        or native.get("schema") != NATIVE_SCHEMA
        or native.get("scope") != "selected_process_total_not_selected_operation"
        or not isinstance(native.get("clock"), dict)
        or type(native["clock"].get("attribute_clock_id")) is not int
        or native["clock"]["attribute_clock_id"] != 1
    ):
        raise ValueError("raw-audited native CLOCK_MONOTONIC CPU samples required")
    samples, pid = native.get("samples"), uint(native.get("pid"), "sample PID", positive=True)
    if not isinstance(samples, list) or not 1 <= len(samples) <= 100000:
        raise ValueError("nonempty bounded native CPU samples required")
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
    if (
        not isinstance(clock, dict)
        or clock.get("schema") != "rns8_linux_host_range_clock_v1"
        or type(clock.get("clock_id")) is not int
        or clock["clock_id"] != 1
        or clock.get("name") != "CLOCK_MONOTONIC"
        or clock.get("unit") != "nanoseconds"
        or clock.get("timestamp_scope") != "brackets_around_native_ROCTx_calls"
    ):
        raise ValueError("actual native host range clock differs from the sample clock")
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

    def ancestors(identity):
        result = []
        current = records[identity]["parent_range_id"]
        while current is not None:
            result.append(current)
            current = records[current]["parent_range_id"]
        return result

    candidate_roots = {
        identity
        for identity, row in records.items()
        if row["role"] == 3 and row["operation_namespace"] == 1 and row["phase"] == 10
    }
    roots = {}
    for identity in sorted(candidate_roots):
        row = records[identity]
        parents = [parent for parent in ancestors(identity) if parent in candidate_roots]
        if any(records[parent]["operation_id"] != row["operation_id"] for parent in parents):
            raise ValueError("nested distinct measured operations have ambiguous sample ownership")
        if not parents:
            if not row["operation_id"] or row["operation_id"] in {
                item["operation_id"] for item in roots.values()
            }:
                raise ValueError("measured operation identity is zero or repeated")
            roots[identity] = row
    if not roots:
        raise ValueError("no measured benchmark-operation host range selected")
    by_thread, root_by_thread = defaultdict(list), defaultdict(list)
    for identity, row in records.items():
        owners = [item for item in (identity, *ancestors(identity)) if item in roots]
        if not owners:
            continue
        if len(owners) != 1:
            raise ValueError("host range has ambiguous measured roots")
        root = owners[0]
        if row["stop_state"] != 1:
            if identity == root:
                raise ValueError("measured host operation did not close synchronously")
            continue  # Async completion ranges are not inferred CPU work intervals.
        if row["start_thread_id"] != row["stop_thread_id"]:
            raise ValueError("closed host scope changed its executing thread")
        stamps = [
            row[key]
            for key in ("start_before_ns", "start_after_ns", "stop_before_ns", "stop_after_ns")
        ]
        if any(value is None for value in stamps) or stamps != sorted(stamps):
            raise ValueError("selected host range lacks ordered observed clock brackets")
        if row["role"] in (1, 2, 4, 5):
            raise ValueError("setup/warmup/oracle/teardown is nested inside selected measured work")
        if row["start_thread_id"] != roots[root]["start_thread_id"]:
            continue  # A parent span alone does not prove a worker's operation binding.
        by_thread[row["start_thread_id"]].append((stamps[0], stamps[3], identity, root))
        if identity == root:
            root_by_thread[row["start_thread_id"]].append((stamps[0], stamps[3], identity))
    for thread, intervals in by_thread.items():
        included = {item[2] for item in intervals}
        for start, end, identity, _ in intervals:
            parent = next((item for item in ancestors(identity) if item in included), None)
            if parent is not None and not (
                records[parent]["start_after_ns"]
                <= start
                <= end
                <= records[parent]["stop_before_ns"]
            ):
                raise ValueError("host child interval escapes its observed parent body")
        stack = []
        for start, end, identity, _ in sorted(
            intervals, key=lambda item: (item[0], -item[1], depth[item[2]])
        ):
            while stack and stack[-1][1] <= start:
                stack.pop()
            if stack and (end > stack[-1][1] or stack[-1][2] not in ancestors(identity)):
                raise ValueError("crossing or unrelated overlapping host scopes")
            stack.append((start, end, identity))
        ordered = sorted(root_by_thread[thread])
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("same-thread measured roots overlap")
        root_by_thread[thread] = ordered
    sample_threads = defaultdict(list)
    for row in samples:
        if not isinstance(row, dict) or uint(row.get("pid"), "sample PID", positive=True) != pid:
            raise ValueError("sample PID differs from the selected range process")
        for key in ("tid", "time_ns", "period"):
            uint(row.get(key), "sample " + key, positive=key != "time_ns")
        sample_threads[row["tid"]].append(row)
    selected, phases, ambiguous, outside = [], {}, 0, 0
    for thread, thread_samples in sample_threads.items():
        spans = root_by_thread.get(thread, [])
        starts = [span[0] for span in spans]
        intervals = sorted(by_thread.get(thread, []))
        cursor, active = 0, []
        for sample in sorted(thread_samples, key=lambda item: item["time_ns"]):
            timestamp = sample["time_ns"]
            while cursor < len(intervals) and intervals[cursor][0] <= timestamp:
                start, end, identity, root = intervals[cursor]
                heapq.heappush(active, (-depth[identity], identity, end, root))
                cursor += 1
            while active and active[0][2] <= timestamp:
                heapq.heappop(active)
            position = bisect_right(starts, timestamp) - 1
            if position < 0 or timestamp >= spans[position][1]:
                outside += 1
                continue
            root = spans[position][2]
            if not records[root]["start_after_ns"] <= timestamp < records[root]["stop_before_ns"]:
                ambiguous += 1
                continue
            if not active or active[0][3] != root:
                raise ValueError("selected sample lacks its measured root interval")
            identity = active[0][1]
            if (
                not records[identity]["start_after_ns"]
                <= timestamp
                < records[identity]["stop_before_ns"]
            ):
                ambiguous += 1
                continue
            phase = PHASES[records[identity]["phase"]]
            bucket = phases.setdefault(phase, dict(samples=0, sampled_cpu_exposure_ns=0))
            bucket["samples"] += 1
            bucket["sampled_cpu_exposure_ns"] += sample["period"]
            selected.append(sample)
    if not selected:
        raise ValueError(
            "no CPU samples resolved inside measured host ranges at this sampling exposure"
        )
    windows = [(row["start_after_ns"], row["stop_before_ns"]) for row in roots.values()]
    return dict(
        schema="rns8_host_sample_operation_join_v1",
        clock_id=1,
        pid=pid,
        selected_operation_count=len(roots),
        selected_samples=len(selected),
        observer_boundary_ambiguous_samples=ambiguous,
        outside_or_unbound_thread_samples=outside,
        exclusive_sampled_phases=dict(sorted(phases.items())),
        selected_sample_witness=decoded_sample_witness(selected),
        measured_host_windows=interval_summary(windows),
        scope="observed_same_thread_host_ranges_not_GPU_service_or_exact_CPU_duration",
        source_content_verified=False,
        worker_scope_inferred=False,
        production_latency=False,
        promotion_allowed=False,
    )
