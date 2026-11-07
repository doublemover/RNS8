"""Observed marker clock brackets; no epoch/rate fit or GPU clock qualification."""

from __future__ import annotations

from bisect import bisect_left, bisect_right


def observe(index, markers, selected):
    policy = dict(
        schema="rns8_native_marker_clock_observation_v1",
        host_clock="CLOCK_MONOTONIC_nanoseconds",
        native_clock="native_ROCprofiler_SDK_timestamp_ns_in_one_process",
        model="conditional_monotonic_order_only_no_rate_offset_or_epoch_fit",
        GPU_clock_normalization_verified=False,
        usable_for_GPU_duration_or_dependency_claims=False,
        production_latency=False,
    )
    anchors, missing = [], []
    for identity, scope in index.records.items():
        marker = markers[identity]
        for endpoint in ("start", "stop"):
            first, last = scope[endpoint + "_before_ns"], scope[endpoint + "_after_ns"]
            if first is None or last is None or endpoint == "stop" and not scope["stop_state"]:
                missing.append(
                    dict(
                        host_range_id=identity,
                        endpoint=endpoint,
                        reason="host_clock_bracket_unavailable",
                    )
                )
                continue
            if first > last:
                return dict(
                    policy,
                    state="inconsistent",
                    reason="reversed_host_marker_bracket",
                    host_range_id=identity,
                )
            anchors.append(
                dict(
                    host_range_id=identity,
                    native_marker_id=marker["id"],
                    endpoint=endpoint,
                    native_ns=marker["start_ns" if endpoint == "start" else "end_ns"],
                    host_lower_ns=first,
                    host_upper_ns=last,
                )
            )
    if not anchors:
        return dict(policy, state="unavailable", missing=missing)
    # Equal native timestamps must have one feasible host bracket. Prefix and
    # suffix envelopes then constrain order without a constant-offset or rate fit.
    grouped = {}
    for row in anchors:
        first, last = grouped.get(row["native_ns"], (0, (1 << 64) - 1))
        grouped[row["native_ns"]] = (
            max(first, row["host_lower_ns"]),
            min(last, row["host_upper_ns"]),
        )
    timestamps = sorted(grouped)
    lowers, uppers, lower = [], [], 0
    for timestamp in timestamps:
        lower = max(lower, grouped[timestamp][0])
        lowers.append(lower)
    upper = (1 << 64) - 1
    for timestamp in reversed(timestamps):
        upper = min(upper, grouped[timestamp][1])
        uppers.append(upper)
    uppers.reverse()
    if any(first > last for first, last in zip(lowers, uppers)):
        return dict(
            policy,
            state="inconsistent",
            reason="native_marker_order_conflicts_with_host_brackets",
            anchors=anchors,
            missing=missing,
        )

    def bound(timestamp):
        before, after = bisect_right(timestamps, timestamp) - 1, bisect_left(timestamps, timestamp)
        first = lowers[before] if before >= 0 else None
        last = uppers[after] if after < len(timestamps) else None
        return dict(
            native_ns=timestamp,
            host_lower_ns=first,
            host_upper_ns=last,
            state="bracketed" if first is not None and last is not None else "unbracketed",
        )

    return dict(
        policy,
        state="observed_order_envelopes",
        anchors=anchors,
        missing=missing,
        interval_endpoints=[
            dict(
                host_range_id=row["host_range_id"],
                native_table_id=row["native_table_id"],
                kind=row.get("kind", "kernel_dispatch"),
                start=bound(row["start_ns"]),
                end=bound(row["end_ns"]),
            )
            for row in selected
        ],
    )
