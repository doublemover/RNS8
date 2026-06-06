#!/usr/bin/env python3
"""Focused schema self-tests for contract and variant metadata validators."""

from __future__ import annotations

import copy

from benchmark_schema import validate_capture
from test_benchmark_schema import (
    add_helper_lane_fields,
    as_exact_wide_capture,
    as_grouped_dispatch_capture,
    as_hip_graph_replay_capture,
    as_reused_pack_capture,
    expect_invalid,
    expect_valid,
)


def add_optional_contracts(capture: dict) -> dict:
    contracted = copy.deepcopy(capture)
    contracted["streaming_overlap"] = {
        "requested": True,
        "pipeline": "pack_gemm_export_double_buffered",
        "buffering": "two_stage_host_device_ring",
        "dependency_contract": "same_stream_ordered_pack_gemm_export",
        "transfer_policy": "async_h2d_d2h_with_host_sync_at_repeat_boundary",
        "capture_status": "metadata_only_unsupported_for_execution_path",
        "promotion_eligible": False,
    }
    contracted["workload_proxy"] = {
        "enabled": True,
        "family": "fhe_lattice_proxy",
        "label": "synthetic_rlwe_gemm_proxy",
        "tower_role": "coefficient_tower",
        "reuse_profile": "same_B_across_batch",
        "transform_role": "ntt_domain_bridge_candidate",
        "output_domain_requirement": "rns_residue_current",
        "compatibility_claim": False,
    }
    contracted["release_gate"] = {
        "name": "local_gfx1100_cleanup_schema_gate",
        "requested": True,
        "classification_tier": "windows_gfx1100_local_only",
        "cpu_reference_policy": "same_contract_cpu_reference_required",
        "memory_cap_policy": "bounded_fixture_only",
        "resume_policy": "rerun_full_fixture_on_failure",
        "review_status": "pending_reviewed_summary",
        "cache_eligible": False,
        "blockers": ["evidence_not_reviewed_for_promotion"],
    }
    contracted["verification_amortization"] = {
        "enabled": True,
        "policy": "reuse_exact_reference_checksum_across_repeats",
        "reused_reference_structure": "same_shape_same_seed_fixture",
        "final_exact_comparison_required": True,
        "final_exact_comparison_status": "passed",
        "promotion_eligible": False,
    }
    return contracted


def main() -> int:
    base = expect_valid("v4_bounded_i64_ck.json")
    direct_hip_base = expect_valid("v4_bounded_i64_adaptive_hip.json")

    reused = add_helper_lane_fields(as_reused_pack_capture(base))
    reused["reuse_contract"] = {
        "enabled": True,
        "operand_role": "A+B",
        "source_version_inputs": "A_and_B_source_versions_match_prepack_setup",
        "setup_scope": "one_time_before_warmups",
        "output_domain": "native_i64_u64_host",
        "next_op": "final-export",
        "target_fingerprint": "gfx1100",
        "backend_fingerprint": "ck_fixture",
        "workspace_fingerprint": "persistent_matrix_residency_fixture",
        "setup_cost_us": 123.0,
        "measured_repeat_count": reused["repeats"],
        "break_even_repeat_count": None,
        "promotion_eligible": False,
        "invalidation_reasons": ["fixture_not_reviewed_for_promotion"],
    }
    validate_capture(reused)

    stale_reuse_role = copy.deepcopy(reused)
    stale_reuse_role["reuse_contract"]["operand_role"] = "A+C"
    expect_invalid(stale_reuse_role, "reuse_contract.operand_role must be one of")

    exact = add_helper_lane_fields(as_exact_wide_capture(base))
    exact["exact_output_contract"] = {
        "requested_final_output": "exact_wide_limb_host",
        "limb_count": 4,
        "status_policy": "structurally_elided",
        "output_domain_after_measured_repeats": "exact_wide_limb_host",
        "final_checksum_export_after_repeats": False,
    }
    exact["export_variant"] = {
        "name": "default",
        "source": "current_backend_export_path",
        "selector_source": "rns8_internal_export_plan",
        "selector_key": (
            "semantics=exact_wide_signed;backend=hip-direct;target_id=gfx1100;prefix=9;"
            "limb_count=4;signedness=signed;output_layout=fixed_u64_limbs;"
            "status_policy=none;d2h_policy=host_ld_padded;"
            "final_output_mode=final_host_output;"
            "selected_kernel=hip_direct_export_exact_wide_signed_limbs_device"
        ),
        "selector_policy": "semantic_prefix_limb_layout_status_d2h_backend_target",
        "semantic_contract": "exact_wide_signed",
        "backend": "hip-direct",
        "target_id": "gfx1100",
        "prefix_contract": "prefix=9;min_selected=9;max_selected=9;groups=1",
        "signedness": "signed",
        "output_layout": "fixed_u64_limbs",
        "limb_count": 4,
        "status_policy": "structurally_elided",
        "selector_status_policy": "none",
        "d2h_policy": "host_ld_padded",
        "final_output_mode": "final_host_output",
        "cache_visibility": "exact_shape_selector_metadata_only",
        "stale_entry_reason": "selector_key_mismatch_rejects_semantic_prefix_limb_layout_status_d2h_backend_target",
        "status_elision_reason": "exact_wide_requested_limb_count_covers_range_status",
        "requires_tile_metadata": False,
        "all_zero_tiled_output": False,
        "selected_kernel": "hip_direct_export_exact_wide_signed_limbs_device",
        "constants_placement": "backend_default",
        "promotion_eligible": True,
        "promotion_blocker": None,
    }
    validate_capture(exact)

    reviewable_fixed_limb = copy.deepcopy(exact)
    reviewable_fixed_limb["export_variant"]["name"] = "exact-wide-fixed-limb-export"
    reviewable_fixed_limb["export_variant"]["source"] = "reviewable_exact_wide_fixed_limb_selector"
    reviewable_fixed_limb["export_variant"]["promotion_eligible"] = True
    reviewable_fixed_limb["export_variant"]["promotion_blocker"] = None
    validate_capture(reviewable_fixed_limb)

    stale_reviewable_status_reason = copy.deepcopy(reviewable_fixed_limb)
    stale_reviewable_status_reason["export_variant"]["status_elision_reason"] = None
    expect_invalid(
        stale_reviewable_status_reason,
        "export_variant.promotion_eligible=true is allowed only for default or exact-wide fixed-limb selector captures",
    )

    stale_reviewable_layout = copy.deepcopy(reviewable_fixed_limb)
    stale_reviewable_layout["export_variant"]["output_layout"] = "scalar_i64"
    expect_invalid(
        stale_reviewable_layout,
        "export_variant.promotion_eligible=true is allowed only for default or exact-wide fixed-limb selector captures",
    )

    stale_compact_promotable = copy.deepcopy(reviewable_fixed_limb)
    stale_compact_promotable["export_variant"]["name"] = "compact-d2h-export-candidate"
    expect_invalid(
        stale_compact_promotable,
        "export_variant.promotion_eligible=true is allowed only for default or exact-wide fixed-limb selector captures",
    )

    stale_exact_domain = copy.deepcopy(exact)
    stale_exact_domain["exact_output_contract"]["requested_final_output"] = "linux_instinct_claim"
    expect_invalid(stale_exact_domain, "exact_output_contract.requested_final_output must be one of")

    stale_export_layout = copy.deepcopy(exact)
    stale_export_layout["export_variant"]["output_layout"] = "device_magic"
    expect_invalid(stale_export_layout, "export_variant.output_layout must be one of")

    stale_export_status = copy.deepcopy(exact)
    stale_export_status["export_variant"]["selector_status_policy"] = "host_assumed_ok"
    expect_invalid(stale_export_status, "export_variant.selector_status_policy must be one of")

    stale_export_d2h = copy.deepcopy(exact)
    stale_export_d2h["export_variant"]["d2h_policy"] = "compact_unproven"
    expect_invalid(stale_export_d2h, "export_variant.d2h_policy must be one of")

    stale_export_mode = copy.deepcopy(exact)
    stale_export_mode["export_variant"]["final_output_mode"] = "secret_route"
    expect_invalid(stale_export_mode, "export_variant.final_output_mode must be one of")

    stale_export_key = copy.deepcopy(exact)
    stale_export_key["export_variant"]["selector_key"] = "semantics=exact_wide_signed;selected_kernel=other"
    expect_invalid(stale_export_key, "export_variant.selector_key must include selected_kernel")

    stale_export_tile_flag = copy.deepcopy(exact)
    stale_export_tile_flag["export_variant"]["requires_tile_metadata"] = "false"
    expect_invalid(stale_export_tile_flag, "export_variant.requires_tile_metadata must be a boolean")

    grouped = as_grouped_dispatch_capture(base)
    validate_capture(grouped)

    stale_grouped_status = copy.deepcopy(grouped)
    stale_grouped_status["grouped_dispatch"]["capture_status"] = "stale_status"
    expect_invalid(stale_grouped_status, "grouped_dispatch.capture_status must be one of")

    graph = as_hip_graph_replay_capture(direct_hip_base)
    validate_capture(graph)

    stale_graph_status = copy.deepcopy(graph)
    stale_graph_status["hip_graph_replay"]["capture_status"] = "stale_graph_status"
    expect_invalid(stale_graph_status, "hip_graph_replay.capture_status must be one of")

    optional = add_optional_contracts(add_helper_lane_fields(copy.deepcopy(base)))
    validate_capture(optional)

    stale_overlap = copy.deepcopy(optional)
    stale_overlap["streaming_overlap"]["capture_status"] = "stale_overlap_status"
    expect_invalid(stale_overlap, "streaming_overlap.capture_status must be one of")

    stale_proxy = copy.deepcopy(optional)
    stale_proxy["workload_proxy"]["family"] = "linux_instinct_readiness"
    expect_invalid(stale_proxy, "workload_proxy.family must be one of")

    stale_release_gate = copy.deepcopy(optional)
    stale_release_gate["release_gate"]["review_status"] = "claimed_ready"
    expect_invalid(stale_release_gate, "release_gate.review_status must be one of")

    print("benchmark schema contract metadata self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
