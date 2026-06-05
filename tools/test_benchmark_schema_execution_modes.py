#!/usr/bin/env python3
"""Focused schema self-tests for execution-mode metadata validators."""

from __future__ import annotations

import copy

from benchmark_schema import validate_capture
from metadata_registry_constants import (
    GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS,
    GROUPED_STRATEGY_DEVICE_DESCRIPTOR_POLICIES,
)
from test_benchmark_schema import (
    as_grouped_dispatch_capture,
    as_hip_graph_replay_capture,
    as_host_api_batch_capture,
    expect_invalid,
    expect_valid,
)


def main() -> int:
    base = expect_valid("v4_bounded_i64_ck.json")
    direct_hip_base = expect_valid("v4_bounded_i64_adaptive_hip.json")

    host_batch = as_host_api_batch_capture(base)
    validate_capture(host_batch)

    stale_host_batch_note = copy.deepcopy(host_batch)
    stale_host_batch_note["timing_metadata"]["phase_notes"]["end_to_end"] = "single call"
    expect_invalid(stale_host_batch_note, "benchmark_host_api_batch phase note end_to_end")

    grouped = as_grouped_dispatch_capture(base)
    validate_capture(grouped)

    grouped_device_pack_gemm = copy.deepcopy(grouped)
    grouped_device_pack_gemm["grouped_dispatch"][
        "execution_strategy"
    ] = GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS
    grouped_device_pack_gemm["timing_metadata"][
        "grouped_dispatch_execution_strategy"
    ] = GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS
    device_pack_gemm_policy = GROUPED_STRATEGY_DEVICE_DESCRIPTOR_POLICIES[
        GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS
    ]
    grouped_device_pack_gemm["grouped_dispatch"]["task_descriptor_contract"][
        "device_descriptor_policy"
    ] = device_pack_gemm_policy
    validate_capture(grouped_device_pack_gemm)

    grouped_bad_descriptor_policy = copy.deepcopy(grouped)
    grouped_bad_descriptor_policy["grouped_dispatch"]["task_descriptor_contract"][
        "device_descriptor_policy"
    ] = device_pack_gemm_policy
    expect_invalid(
        grouped_bad_descriptor_policy,
        "grouped task descriptor device_descriptor_policy must match execution strategy",
    )

    graph = as_hip_graph_replay_capture(direct_hip_base)
    validate_capture(graph)

    bad_graph_launches = copy.deepcopy(graph)
    bad_graph_launches["hip_graph_replay"]["total_graph_launches"] += 1
    expect_invalid(bad_graph_launches, "hip_graph_replay.total_graph_launches must equal warmups + repeats")

    print("benchmark schema execution-mode self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
