"""Same-file native ETL point selection in exclusive measured operation phases."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

from host_range_contract import range_inventory, measured_roots
from native_phase_attribution import attribute


def select_native_points(
    points, pid, snapshot, ranges, *, identity_field="record_index", result_field="native_record"
):
    if (identity_field, result_field) not in {
        ("record_index", "native_record"),
        ("source_target_id", "source_target_id"),
    }:
        raise ValueError("explicit native/derived point identity namespace required")
    if type(ranges.get("selected_pid")) is not int or ranges["selected_pid"] != pid:
        raise ValueError("native CPU ranges belong to another process")
    index = range_inventory(snapshot, pid, clock_kind="windows_qpc")
    roots = measured_roots(index)
    markers = {row["slot"]: row for row in ranges["markers"]}
    groups = defaultdict(list)
    for identity, root in roots.items():
        row = markers[identity]
        groups[root["start_thread_id"]].append((row["start_ns"], row["end_ns"], identity))
    starts = {}
    for tid, windows in groups.items():
        windows.sort()
        if any(left[1] > right[0] for left, right in zip(windows, windows[1:])):
            raise ValueError("same-thread measured native windows overlap")
        starts[tid] = [row[0] for row in windows]
    selected, outside = [], 0
    origin = ranges["phase_origin_filetime_100ns"]
    for point in points:
        time = (point["time_filetime_100ns"] - origin) * 100
        position = bisect_right(starts.get(point["tid"], []), time) - 1
        if position < 0 or time >= groups[point["tid"]][position][1]:
            outside += 1
            continue
        owner = groups[point["tid"]][position][2]
        # Reuse ancestry containment for a native point, not a HIP API event.
        selected.append(
            dict(
                host_range_id=owner,
                launch_thread_id=point["tid"],
                api_start_ns=time,
                api_end_ns=time,
                native_api_id=point[identity_field],
            )
        )
    report = attribute(
        index,
        markers,
        roots,
        selected,
        clock_domain="native_ETL_FILETIME_relative_to_process_start_ns",
    )
    return dict(
        points=[
            {
                result_field: row["native_api_id"],
                "measured_root_id": row["host_range_id"],
                "phase": row["launch_phase"],
                "phase_range_id": row["launch_phase_range_id"],
                "tid": row["launch_thread_id"],
            }
            for row in report["records"]
        ],
        outside_or_unbound_thread_points=outside,
        point_count=len(selected),
        boundary="native_ETW_marker_points_include_unseparated_provider_call_overhead",
        CPU_time_inferred=False,
        GPU_dependencies_inferred=False,
    )
