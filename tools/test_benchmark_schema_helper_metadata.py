#!/usr/bin/env python3
"""Focused schema self-tests for helper-lane and output-policy metadata."""

from __future__ import annotations

import copy

from benchmark_schema import validate_capture
from test_benchmark_schema import (
    add_helper_lane_fields,
    add_timing_helper_fields,
    as_residue_chain_final_export_capture,
    as_residue_current_chain_capture,
    expect_invalid,
    expect_valid,
)


def main() -> int:
    base = expect_valid("v4_bounded_i64_ck.json")
    adaptive = expect_valid("v4_bounded_i64_adaptive_hip.json")

    helper_lane = add_helper_lane_fields(copy.deepcopy(base))
    validate_capture(helper_lane)

    bad_output_policy = copy.deepcopy(helper_lane)
    bad_output_policy["output_policy"]["destination_layout"] = "padded_row_major"
    expect_invalid(bad_output_policy, "output_policy.destination_layout must match output_ld_padding")

    missing_target = copy.deepcopy(helper_lane)
    del missing_target["target_variant"]
    expect_invalid(missing_target, "HIP helper-lane captures must include target_variant")

    bad_selector_reason = copy.deepcopy(helper_lane)
    bad_selector_reason["auto_selector"]["rejected_candidates"][0]["reason"] = "unsupported"
    expect_invalid(bad_selector_reason, "fixed rejection reason")

    bad_allocation_counter = copy.deepcopy(helper_lane)
    bad_allocation_counter["device_allocation"]["measured_repeat_delta"]["allocated_bytes"] = -1
    expect_invalid(bad_allocation_counter, "device_allocation.measured_repeat_delta.allocated_bytes")

    direct_reducer = add_helper_lane_fields(copy.deepcopy(adaptive))
    add_timing_helper_fields(direct_reducer, reducer="direct_hip_fixed_prefix_9_generated_reducer_v1")
    validate_capture(direct_reducer)

    exact_wide_prefix18_reducer = copy.deepcopy(direct_reducer)
    exact_wide_prefix18_reducer["timing_metadata"][
        "generated_reducer_identity"
    ] = "direct_hip_fixed_prefix_18_generated_reducer_v1"
    validate_capture(exact_wide_prefix18_reducer)

    stale_reducer = copy.deepcopy(direct_reducer)
    stale_reducer["timing_metadata"]["generated_reducer_identity"] = "generic"
    expect_invalid(stale_reducer, "declared reducer identity")

    undeclared_prefix_reducer = copy.deepcopy(direct_reducer)
    undeclared_prefix_reducer["timing_metadata"]["generated_reducer_identity"] = (
        "direct_hip_fixed_prefix_10_generated_reducer_v1"
    )
    expect_invalid(undeclared_prefix_reducer, "declared reducer identity")

    residue_current = as_residue_current_chain_capture(base)
    validate_capture(residue_current)

    bad_residue_current_export = copy.deepcopy(residue_current)
    bad_residue_current_export["output_policy"]["final_checksum_export_after_repeats"] = False
    expect_invalid(
        bad_residue_current_export,
        "residue-current chain captures must declare output_policy.final_checksum_export_after_repeats=true",
    )

    final_export = as_residue_chain_final_export_capture(base)
    validate_capture(final_export)

    bad_final_export_next_op = copy.deepcopy(final_export)
    bad_final_export_next_op["requested_next_op"]["resolved"] = "rns-gemm"
    expect_invalid(
        bad_final_export_next_op,
        "residue-chain final-export captures must declare requested_next_op.resolved=final-export",
    )

    print("benchmark schema helper metadata self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
