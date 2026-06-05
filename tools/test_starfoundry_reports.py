#!/usr/bin/env python3
"""Self-test Starfoundry schema/reporting helpers."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture
from test_benchmark_schema import add_helper_lane_fields

import fhe_workload_report
import many_small_grouped_report
import modulus_set_search
import promotion_ledger
import release_gate_report
import resident_workspace_report
import reuse_contract_report
import scheduler_overlap_report
import target_validation_report
import tile_shape_report


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "benchmark_schema"


def starfoundry_capture() -> dict:
    capture = add_helper_lane_fields(copy.deepcopy(load_capture(FIXTURE_DIR / "v4_bounded_i64_hipblaslt.json")))
    capture.setdefault("selected_prefix", capture["prefix"])
    capture.setdefault("requested_max_prefix", capture["prefix"])
    capture.setdefault("contract_prefix_policy", "fixed_requested")
    capture.setdefault("residue_planes_requested", capture["prefix"])
    capture.setdefault("residue_planes_selected", capture["selected_prefix"])
    capture.setdefault("residue_planes_skipped", 0)
    capture.setdefault("residue_plane_skip_fraction", 0.0)
    selected_kernel = capture["selected_kernel"]
    capture["reuse_contract"] = {
        "enabled": True,
        "operand_role": "B",
        "source_version_inputs": "monotonic_source_version_per_repeat_when_packing_runs",
        "setup_scope": "persistent_plan_workspace_prepacked_reuse",
        "setup_cost_us": 12.0,
        "measured_repeat_count": capture["repeats"],
        "break_even_repeat_count": 3,
        "output_domain": "native_i64_u64_host",
        "next_op": "reuse-b",
        "target_fingerprint": "gfx1100",
        "backend_fingerprint": capture["backend_selected"],
        "kernel_fingerprint": selected_kernel,
        "workspace_fingerprint": "0B:resident_device_buffers",
        "promotion_eligible": False,
        "invalidation_reasons": ["source_version_changed"],
    }
    capture["exact_output_contract"] = {
        "requested_final_output": "native_i64_u64_host",
        "limb_count": None,
        "output_logical_ld": capture["output_logical_ld"],
        "status_policy": "required",
        "kernel_identity": selected_kernel,
        "output_domain_after_measured_repeats": "native_i64_u64_host",
        "final_checksum_export_after_repeats": False,
    }
    capture["export_variant"] = {
        "name": "default",
        "source": "current_backend_export_path",
        "limb_count": None,
        "status_policy": "required",
        "selected_kernel": selected_kernel,
        "constants_placement": "backend_default",
        "promotion_eligible": True,
        "promotion_blocker": None,
    }
    capture["reconstruction_variant"] = {
        "name": "default_garner",
        "family": "garner_fixed_prefix",
        "prefix_count": capture["selected_prefix"],
        "kernel_identity": selected_kernel,
        "controller": "benchmark_metadata_only",
        "promotion_eligible": True,
        "promotion_blocker": None,
    }
    capture["modulus_set"] = {
        "name": "default",
        "source": "rns8_default_modulus_ladder",
        "execution_ladder": "rns8_default_8bit_coprime_ladder",
        "experimental": False,
        "product_bits": 72,
        "prefix_count": capture["selected_prefix"],
        "pairwise_coprime_proof": "schema_declared_current_ladder_or_offline_search_report",
        "reducer_cost_hint": "backend_default",
        "cache_promotion_blocker": None,
    }
    capture["residue_count_policy"] = {
        "policy": capture["contract_prefix_policy"],
        "requested_prefix": capture["requested_max_prefix"],
        "selected_prefix": capture["selected_prefix"],
        "minimum_range_prefix": capture["schedule_metadata"]["min_required_prefix"],
        "redundant_residue_count": 0,
        "autotune_scope": "current_exact_cache",
        "cache_promotion_blocker": None,
    }
    capture["tile_shape_variant"] = {
        "name": "hipblaslt-default-128x128",
        "tile_m": capture["tile_m"],
        "tile_n": capture["tile_n"],
        "tile_k": capture["k_block_size"],
        "selected_kernel_identity": selected_kernel,
        "resource_report_key": f"tile_m={capture['tile_m']};tile_n={capture['tile_n']};tile_k={capture['k_block_size']};kernel={selected_kernel}",
        "shape_family_bucket": "medium",
        "stale_kernel_rejection": "selected_kernel_identity_must_match_capture",
    }
    capture["grouped_dispatch"] = {
        "requested": True,
        "task_count": 8,
        "descriptor_identity": "same_shape_m=512;n=512;k=512;semantics=bounded_i64",
        "source_hash": "1",
        "output_hash": "final_checksum_u64",
        "setup_scope": "persistent_plan_workspace_resident_matrices",
        "capture_status": "metadata_only_unsupported_for_execution_path",
        "unsupported_reason": "grouped_dispatch_not_executed_by_current_benchmark_path",
        "promotion_eligible": False,
    }
    capture["hip_graph_replay"] = {
        "requested": False,
        "available": False,
        "used": False,
        "status": "not_requested",
        "scope": "not_applicable",
        "descriptor_identity": "fixed_plan_workspace_descriptor:m=512;n=512;k=512",
        "plan_identity": capture["backend_metadata"]["autotune_key"],
        "setup_scope": "persistent_plan_workspace_resident_matrices",
        "capture_status": "not_requested",
        "unsupported_reason": None,
        "promotion_eligible": False,
        "capture_us": 0,
        "instantiate_us": 0,
        "graph_launches_per_measured_repeat": 0,
        "total_graph_launches": 0,
        "captured_chain_length": 0,
        "timing_policy": "not_applicable",
        "setup_policy": "not_applicable",
        "final_export_policy": "not_applicable",
        "caveat": None,
    }
    capture["adaptive_grouped_scheduler"] = {
        "requested": True,
        "strategy": "prefix_tile_zero_mask_grouped_descriptors",
        "descriptor_identity": "prefix=9;tile_m=128;tile_n=128;zero_tiles=0",
        "group_count": 1,
        "active_tile_count": capture["schedule_metadata"]["tile_count"],
        "zero_tile_count": 0,
        "selected_prefix_histogram": "min=9;max=9",
        "capture_status": "metadata_only_unsupported_for_execution_path",
        "unsupported_reason": "adaptive_grouped_scheduler_not_executed_by_current_path",
        "promotion_eligible": False,
    }
    capture["resident_lifetime"] = {
        "enabled": True,
        "matrix_roles": "A/B/C explicit benchmark resident roles",
        "source_version_policy": "monotonic_per_import_or_pack",
        "current_storage_state": "native_i64_u64_host",
        "output_domain": "native_i64_u64_host",
        "workspace_identity": "0B:resident_device_buffers",
        "stale_source_rejection": "source_version_descriptor_semantic_prefix_target_workspace_mismatch",
        "promotion_eligible": False,
    }
    capture["workspace_arena"] = {
        "enabled": True,
        "arena_identity": "test-key|resident_device_buffers",
        "size_bytes": 0,
        "high_water_mark_bytes": 0,
        "suballocation_count": 5,
        "measured_repeat_allocation_free": True,
        "source_version_policy": "plan_target_backend_semantic_shape_prefix_output_policy",
        "stream_safety": "single_stream_owner",
        "promotion_eligible": False,
    }
    capture["streaming_overlap"] = {
        "requested": True,
        "pipeline": "pack_next_gemm_current_export_previous",
        "buffering": "double_buffered_benchmark_only",
        "dependency_contract": "pack_before_gemm;gemm_before_export;status_before_host_read;final_sync_before_checksum",
        "transfer_policy": "compact_or_padded_output_policy_declared_by_output_policy",
        "capture_status": "metadata_only_unsupported_for_execution_path",
        "unsupported_reason": "streaming_overlap_not_executed_by_current_path",
        "promotion_eligible": False,
    }
    capture["workload_proxy"] = {
        "enabled": True,
        "label": "fhe:key_switch_digit_aggregation",
        "family": "fhe_lattice_proxy",
        "tower_role": "dense_gemm_adjacent_proxy",
        "reuse_profile": "B",
        "transform_role": "not_a_public_fhe_backend",
        "output_domain_requirement": "native_i64_u64_host",
        "compatibility_claim": False,
    }
    capture["release_gate"] = {
        "name": "large-release-validation-4096-budgeted",
        "requested": True,
        "classification_tier": "cpu_backed_release_candidate_pending_review",
        "cpu_reference_policy": "chunked_when_large_fixed_seed_checksum_recorded",
        "memory_cap_policy": "declared_by_sweep_runner_or_not_applicable",
        "resume_policy": "scenario_id_and_capture_path_stable_under_temp",
        "review_status": "pending_reviewed_summary",
        "cache_eligible": False,
        "blockers": ["reviewed_summary_missing"],
    }
    capture["verification_amortization"] = {
        "enabled": True,
        "policy": "reuse_shape_seed_reference_inputs",
        "reused_reference_structure": "shape_seed_semantic_reference_inputs",
        "final_exact_comparison_required": True,
        "final_exact_comparison_status": "checksum_recorded_reference_required",
        "promotion_eligible": False,
    }
    return capture


def expect_invalid(capture: dict, needle: str) -> None:
    try:
        validate_capture(capture)
    except BenchmarkSchemaError as exc:
        if needle not in str(exc):
            raise AssertionError(f"expected {needle!r}, got {exc}") from exc
        return
    raise AssertionError(f"expected invalid capture containing {needle!r}")


def main() -> int:
    capture = starfoundry_capture()
    validate_capture(capture)

    bad_tile = copy.deepcopy(capture)
    bad_tile["tile_shape_variant"]["selected_kernel_identity"] = "stale_kernel"
    expect_invalid(bad_tile, "tile_shape_variant.selected_kernel_identity")

    bad_modulus = copy.deepcopy(capture)
    bad_modulus["modulus_set"]["name"] = "stale-generic"
    expect_invalid(bad_modulus, "modulus_set.name")

    bad_grouped = copy.deepcopy(capture)
    bad_grouped["grouped_dispatch"]["promotion_eligible"] = True
    expect_invalid(bad_grouped, "grouped_dispatch task_count > 1")

    bad_proxy = copy.deepcopy(capture)
    bad_proxy["workload_proxy"]["compatibility_claim"] = True
    expect_invalid(bad_proxy, "workload_proxy.compatibility_claim")

    bad_arena = copy.deepcopy(capture)
    bad_arena["workspace_arena"]["measured_repeat_allocation_free"] = "yes"
    expect_invalid(bad_arena, "workspace_arena.measured_repeat_allocation_free")

    bad_overlap = copy.deepcopy(capture)
    bad_overlap["streaming_overlap"]["promotion_eligible"] = True
    expect_invalid(bad_overlap, "streaming_overlap requested")

    bad_gate = copy.deepcopy(capture)
    bad_gate["release_gate"]["blockers"] = "reviewed_summary_missing"
    expect_invalid(bad_gate, "release_gate.blockers")

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        capture_path = tmp / "starfoundry.json"
        capture_path.write_text(json.dumps(capture), encoding="utf-8")
        cache_path = tmp / "cache.json"
        cache_path.write_text(
            json.dumps({"schema_version": 1, "entries": [{"key": capture["backend_metadata"]["autotune_key"]}]}),
            encoding="utf-8",
        )
        assert reuse_contract_report.build_report([capture_path])["capture_count"] == 1
        assert promotion_ledger.build_ledger([capture_path], cache_path)["entries"][0]["installed_cache_entry"] is True
        assert target_validation_report.build_report([capture_path])["capture_count"] == 1
        assert tile_shape_report.build_report([capture_path])["groups"][0]["rows"][0]["variant_name"]
        assert many_small_grouped_report.build_report([capture_path])["groups"][0]["rows"][0]["mode"] == "grouped_dispatch"
        assert fhe_workload_report.build_report([capture_path])["groups"][0]["rows"][0]["family"] == "fhe_lattice_proxy"
        assert resident_workspace_report.build_report([capture_path])["rows"][0]["arena_enabled"] is True
        assert scheduler_overlap_report.build_report([capture_path])["rows"][0]["overlap_requested"] is True
        release_report = release_gate_report.build_report([capture_path])
        assert release_report["schema"] == "rns8_release_gate_report_v2"
        assert release_report["blocker_counts"]["reviewed_summary_missing"] == 1
        assert release_report["blocker_counts"]["missing_required_baselines"] == 1
        assert release_report["groups"][0]["missing_required_baselines"] == [
            "cpu-reference",
            "hip-direct",
            "hip-vector-alu-int64",
        ]

    search = modulus_set_search.build_report([("test", [251, 253, 255, 256])], 32)
    assert search["candidates"][0]["pairwise_coprime"] is True
    assert search["candidates"][0]["satisfies_min_bits"] is True

    print("starfoundry report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
