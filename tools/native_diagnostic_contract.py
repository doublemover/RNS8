"""Shared observed native-event/allocation/copy contract for separate batches."""

import math

COPY_COUNTERS = {
    "device_allocation_calls",
    "device_free_calls",
    "h2d_calls",
    "h2d_bytes",
    "d2h_calls",
    "d2h_bytes",
}


def validate_native_observation(info, *, scope, calls_key, calls):
    fixed = dict(
        scope=scope,
        inside_timed_samples=False,
        inside_sustained_interval=False,
        exact_reference_passed=True,
        native_event_receipts_complete=True,
        copy_scope="tracked_completed_synchronous_HIP_copy_wrappers",
    )
    fixed[calls_key] = calls
    counters = COPY_COUNTERS
    if (
        not isinstance(info, dict)
        or set(info)
        != {*fixed, *counters, "allocation_epoch", "event_labels", "event_microseconds"}
        or any(type(info[k]) is not type(v) or info[k] != v for k, v in fixed.items())
        or any(type(info[k]) is not int or not 0 <= info[k] < (1 << 64) for k in counters)
        or type(info["allocation_epoch"]) is not int
        or not 0 < info["allocation_epoch"] < (1 << 64)
    ):
        raise ValueError("native observation record")
    for direction in ("h2d", "d2h"):
        calls, size = info[direction + "_calls"], info[direction + "_bytes"]
        if not ((calls == 0 and size == 0) or 0 < calls <= size):
            raise ValueError("native copy counter denominator")
    labels, durations = info["event_labels"], info["event_microseconds"]
    if (
        not isinstance(labels, dict)
        or not 1 <= len(labels) <= 64
        or any(
            not isinstance(k, str)
            or not 1 <= len(k) <= 256
            or type(v) is not int
            or not 1 <= v <= 65536
            for k, v in labels.items()
        )
        or sum(labels.values()) > 65536
        or not isinstance(durations, dict)
        or set(labels) != set(durations)
        or any(
            type(v) not in (int, float) or not 0 <= v <= 1e15 or not math.isfinite(v)
            for v in durations.values()
        )
    ):
        raise ValueError("native event receipts")
