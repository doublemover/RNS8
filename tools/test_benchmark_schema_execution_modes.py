#!/usr/bin/env python3
"""Focused schema self-tests for execution-mode metadata validators."""

from __future__ import annotations

import copy

from benchmark_schema import validate_capture
from metadata_registry_constants import (
    GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS,
    GROUPED_STRATEGY_DEVICE_DESCRIPTOR_POLICIES,
    GROUPED_TASK_BUCKET_POLICIES,
    GROUPED_TASK_DESCRIPTOR_LAYOUTS,
)
from test_benchmark_schema import (
    as_grouped_dispatch_capture,
    as_hip_graph_full_bounded_capture,
    as_hip_graph_full_finite_capture,
    as_hip_graph_full_wrap64_capture,
    as_hip_graph_replay_capture,
    as_host_api_batch_capture,
    expect_invalid,
    expect_valid,
)


def main() -> int:
    base = expect_valid("v4_bounded_i64_ck.json")
    direct_hip_base = expect_valid("v4_bounded_i64_adaptive_hip.json")
    wrap64_base = expect_valid("v4_wrap64_hip.json")

    host_batch = as_host_api_batch_capture(base)
    validate_capture(host_batch)

    stale_host_batch_note = copy.deepcopy(host_batch)
    stale_host_batch_note["timing_metadata"]["phase_notes"]["end_to_end"] = "single call"
    expect_invalid(stale_host_batch_note, "benchmark_host_api_batch phase note end_to_end")

    grouped = as_grouped_dispatch_capture(base)
    validate_capture(grouped)

    bucketed_grouped = copy.deepcopy(grouped)
    bucketed_grouped["grouped_dispatch"]["task_count"] = 8
    bucketed_grouped["avg_end_to_end_per_task_us"] = bucketed_grouped["avg_end_to_end_us"] / 8.0
    bucketed_grouped["avg_pack_per_task_us"] = bucketed_grouped["avg_pack_us"] / 8.0
    bucketed_grouped["avg_rns_gemm_per_task_us"] = bucketed_grouped["avg_rns_gemm_us"] / 8.0
    bucketed_grouped["avg_crt_export_per_task_us"] = bucketed_grouped["avg_crt_export_us"] / 8.0
    bucketed_grouped["timing_metadata"]["grouped_dispatch_task_count"] = 8
    bucketed_contract = bucketed_grouped["grouped_dispatch"]["task_descriptor_contract"]
    bucketed_contract.update(
        {
            "descriptor_layout": "same_contract_bucketed_resident_task_triplets_v1",
            "bucket_policy": "same_contract_shape_buckets",
            "bucket_count": 2,
            "task_count": 8,
            "same_shape_required": False,
            "shape_key": "multiple_shape_buckets",
            "buckets": [
                {
                    "bucket_index": 0,
                    "task_offset": 0,
                    "task_count": 4,
                    "shape_key": "m=64;n=64;k=64;tile_m=128;tile_n=128;prefix=9",
                    "semantics": bucketed_grouped["semantics"],
                    "output_domain": "native_i64_u64_host",
                },
                {
                    "bucket_index": 1,
                    "task_offset": 4,
                    "task_count": 4,
                    "shape_key": "m=128;n=128;k=128;tile_m=128;tile_n=128;prefix=9",
                    "semantics": bucketed_grouped["semantics"],
                    "output_domain": "native_i64_u64_host",
                },
            ],
        }
    )
    validate_capture(bucketed_grouped)
    assert "same_contract_shape_buckets" in GROUPED_TASK_BUCKET_POLICIES
    assert "same_contract_bucketed_resident_task_triplets_v1" in GROUPED_TASK_DESCRIPTOR_LAYOUTS

    bucketed_bad_offset = copy.deepcopy(bucketed_grouped)
    bucketed_bad_offset["grouped_dispatch"]["task_descriptor_contract"]["buckets"][1]["task_offset"] = 5
    expect_invalid(
        bucketed_bad_offset,
        "bucketed grouped task descriptor task_offset must be contiguous",
    )

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

    full_graph = as_hip_graph_full_bounded_capture(direct_hip_base)
    validate_capture(full_graph)

    bad_full_graph_reuse = copy.deepcopy(full_graph)
    bad_full_graph_reuse["reuse_packed_inputs"] = True
    expect_invalid(
        bad_full_graph_reuse,
        "bounded pack/GEMM/export hip_graph_replay captures must use reuse_packed_inputs=false",
    )

    bad_full_graph_scope = copy.deepcopy(full_graph)
    bad_full_graph_scope["hip_graph_replay"]["scope"] = "direct_hip_reused_inputs_residue_current_rns_chain"
    expect_invalid(
        bad_full_graph_scope,
        "bounded pack/GEMM/export hip_graph_replay captures must set hip_graph_replay.scope",
    )

    finite_graph = as_hip_graph_full_finite_capture(direct_hip_base)
    validate_capture(finite_graph)

    bad_finite_graph_benchmark = copy.deepcopy(finite_graph)
    bad_finite_graph_benchmark["benchmark"] = "rns8_hip_graph_replay_bounded_pack_gemm_export"
    expect_invalid(
        bad_finite_graph_benchmark,
        "finite-u8 pack/GEMM/export hip_graph_replay captures must use benchmark",
    )

    bad_finite_graph_policy = copy.deepcopy(finite_graph)
    bad_finite_graph_policy["hip_graph_replay"][
        "final_export_policy"
    ] = "logical_export_and_output_d2h_captured_inside_graph_each_repeat"
    expect_invalid(
        bad_finite_graph_policy,
        "finite-u8 pack/GEMM/export hip_graph_replay.final_export_policy is stale or unsupported",
    )

    wrap64_graph = as_hip_graph_full_wrap64_capture(wrap64_base)
    validate_capture(wrap64_graph)

    bad_wrap64_graph_policy = copy.deepcopy(wrap64_graph)
    bad_wrap64_graph_policy["hip_graph_replay"][
        "final_export_policy"
    ] = "logical_export_and_output_d2h_captured_inside_graph_each_repeat"
    expect_invalid(
        bad_wrap64_graph_policy,
        "wrap64 pack/GEMM/export hip_graph_replay.final_export_policy is stale or unsupported",
    )

    print("benchmark schema execution-mode self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
