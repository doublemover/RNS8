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
    capture.setdefault("target_id", "gfx1100")
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
        "selector_source": "rns8_internal_export_plan",
        "selector_key": (
            f"semantics={capture['semantics']};backend={capture['backend_selected']};"
            f"target_id={capture['target_id']};prefix={capture['selected_prefix']};limb_count=0;"
            f"signedness=signed;output_layout=scalar_i64;status_policy=range_checked_status_buffer;"
            f"d2h_policy=host_ld_padded;final_output_mode=final_host_output;selected_kernel={selected_kernel}"
        ),
        "selector_policy": "semantic_prefix_limb_layout_status_d2h_backend_target",
        "semantic_contract": capture["semantics"],
        "backend": capture["backend_selected"],
        "target_id": capture["target_id"],
        "prefix_contract": "prefix=9;min_selected=9;max_selected=9;groups=1",
        "signedness": "signed",
        "output_layout": "scalar_i64",
        "limb_count": None,
        "status_policy": "required",
        "selector_status_policy": "range_checked_status_buffer",
        "d2h_policy": "host_ld_padded",
        "final_output_mode": "final_host_output",
        "cache_visibility": "exact_shape_selector_metadata_only",
        "stale_entry_reason": "selector_key_mismatch_rejects_semantic_prefix_limb_layout_status_d2h_backend_target",
        "status_elision_reason": None,
        "requires_tile_metadata": False,
        "all_zero_tiled_output": False,
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
        "redundant_residue_count": max(
            0,
            capture["selected_prefix"] - capture["schedule_metadata"]["min_required_prefix"],
        ),
        "autotune_scope": "current_exact_cache",
        "cache_promotion_blocker": None,
    }
    capture["tile_shape_variant"] = {
        "name": "hipblaslt-default-128x128",
        "tile_m": capture["tile_m"],
        "tile_n": capture["tile_n"],
        "tile_k": capture["k_block_size"],
        "k_block_policy": "auto",
        "split_k_mode": "single_gpu_no_split_k",
        "accumulator_safety_key": f"k_block_size={capture['k_block_size']};k_block_cap=65536;safe_for_k_block=true",
        "selected_kernel_identity": selected_kernel,
        "resource_report_key": f"tile_m={capture['tile_m']};tile_n={capture['tile_n']};tile_k={capture['k_block_size']};kernel={selected_kernel}",
        "shape_family_bucket": "medium",
        "resource_report_required": "isa_or_counter_for_non_default_k_block_policy",
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
        "source_version_policy": "monotonic_per_import_pack_or_gemm_output_hash",
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
        "setup_allocation_delta": {"allocate_calls": 1, "free_calls": 0, "allocated_bytes": 4096},
        "measured_repeat_allocation_delta": {"allocate_calls": 0, "free_calls": 0, "allocated_bytes": 0},
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
        ledger_entry = promotion_ledger.build_ledger([capture_path], cache_path)["entries"][0]
        assert ledger_entry["installed_cache_entry"] is True
        assert "hip_graph_replay_non_promoting" not in ledger_entry["promotion_blockers"]
        shape_shadow_path = tmp / "shape-family-shadow-report.json"
        shape_shadow_path.write_text(
            json.dumps(
                {
                    "schema": "rns8_shape_family_shadow_report_v2",
                    "policy": "non_routing_shape_family_recommendations_require_exact_review_before_AUTO",
                    "boundary_fields": [
                        "target_id",
                        "target_family",
                        "semantic_contract",
                        "signedness",
                        "finite_modulus",
                        "layout",
                        "output_contract",
                        "export_selector",
                        "limb_count",
                    ],
                    "recommendations": [
                        {
                            "basis_cache_key": capture["backend_metadata"]["autotune_key"],
                            "would_recommend": True,
                            "recommendation_is_exact_cache_hit": False,
                            "runtime_routing_allowed": False,
                            "promotion_eligible": False,
                            "promotion_blockers": [
                                "shape_family_shadow_only_no_routing_change",
                                "exact_query_not_reviewed",
                                "representative_matrix_requires_same_target_layout_contract_review",
                            ],
                            "rejected_boundary_candidates": [
                                {
                                    "basis_cache_key": capture["backend_metadata"]["autotune_key"],
                                    "boundary_blockers": ["boundary_output_contract_mismatch"],
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        shadow_ledger = promotion_ledger.build_ledger(
            [capture_path],
            cache_path,
            shape_family_shadow_reports=[shape_shadow_path],
        )
        shadow_entry = shadow_ledger["entries"][0]
        assert shadow_ledger["shape_family_shadow_report_count"] == 1
        assert shadow_ledger["shape_family_shadow_summary"]["recommendation_count"] == 1
        assert shadow_ledger["shape_family_shadow_summary"]["non_exact_recommendation_count"] == 1
        assert shadow_ledger["shape_family_shadow_summary"]["blocked_recommendation_count"] == 1
        assert shadow_ledger["shape_family_shadow_summary"]["boundary_rejected_recommendation_count"] == 1
        assert shadow_entry["shape_family_recommendation_status"] == "exact_cache_entry_shadow_basis_non_routing"
        assert shadow_entry["shape_family_shadow_query_count"] == 1
        assert "exact_query_not_reviewed" in shadow_entry["shape_family_shadow_blockers"]
        assert target_validation_report.build_report([capture_path])["capture_count"] == 1
        tile_row = tile_shape_report.build_report([capture_path])["groups"][0]["rows"][0]
        assert tile_row["variant_name"]
        assert tile_row["k_block_policy"] == "auto"
        assert tile_row["accumulator_safety_key"]
        assert many_small_grouped_report.build_report([capture_path])["groups"][0]["rows"][0]["mode"] == "grouped_dispatch"
        assert fhe_workload_report.build_report([capture_path])["groups"][0]["rows"][0]["family"] == "fhe_lattice_proxy"
        resident_report = resident_workspace_report.build_report([capture_path])
        assert resident_report["rows"][0]["arena_enabled"] is True
        assert resident_report["arena_ready_count"] == 1
        assert "measured_repeat_allocation_delta_nonzero" not in resident_report["rows"][0]["promotion_blockers"]
        assert scheduler_overlap_report.build_report([capture_path])["rows"][0]["overlap_requested"] is True
        release_report = release_gate_report.build_report([capture_path])
        assert release_report["schema"] == "rns8_release_gate_report_v2"
        assert release_report["blocker_counts"]["reviewed_summary_missing"] == 1
        assert release_report["blocker_counts"]["missing_required_baselines"] == 1
        assert release_report["blocker_counts"]["required_baseline_release_review_missing"] == 1
        assert release_report["groups"][0]["missing_required_baselines"] == [
            "cpu-reference",
            "hip-direct",
            "hip-vector-alu-int64",
        ]
        assert release_report["groups"][0]["missing_release_review_required_baselines"] == [
            "cpu-reference",
            "hip-direct",
            "hip-vector-alu-int64",
        ]

        reviewed_capture = copy.deepcopy(capture)
        reviewed_capture["backend_metadata"]["performance_validated"] = False
        reviewed_capture["comparison_baseline"] = {
            "status": "required_not_recorded",
            "speedup_claimed": False,
            "selected_reference": None,
            "required_before_speedup_claim": [
                "same_contract_cpu_reference",
                "same_contract_direct_hip_vector_alu_int64",
                "same_contract_direct_hip_correctness",
            ],
            "reason": "performance_validated=false; raw capture has not been promoted against same-contract CPU and GPU baseline evidence",
        }
        reviewed_capture["grouped_dispatch"].update(
            {
                "requested": False,
                "task_count": 1,
                "capture_status": "not_requested",
                "unsupported_reason": None,
                "promotion_eligible": False,
            }
        )
        reviewed_path = tmp / "reviewed-promotable.json"
        reviewed_path.write_text(json.dumps(reviewed_capture), encoding="utf-8")
        review_report_path = tmp / "review-report.json"
        review_report_path.write_text(
            json.dumps(
                {
                    "groups": [
                        {
                            "candidates": [
                                {
                                    "capture": str(reviewed_path),
                                    "promotable": True,
                                    "promotion_blockers": [],
                                    "speedup_vs_direct_hip": 1.5,
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        reviewed_ledger = promotion_ledger.build_ledger([reviewed_path], cache_path, [review_report_path])
        reviewed_entry = reviewed_ledger["entries"][0]
        assert reviewed_entry["review_report_promotable"] is True
        assert reviewed_entry["performance_validated"] is True
        assert reviewed_entry["speedup_margin"] == 1.5
        assert reviewed_entry["promotion_blockers"] == []
        selector_review_capture = copy.deepcopy(reviewed_capture)
        selector_review_capture["export_variant"]["name"] = "compact-d2h-export-candidate"
        selector_review_capture["export_variant"]["promotion_eligible"] = False
        selector_review_capture["export_variant"]["promotion_blocker"] = "experimental_export_variant"
        selector_review_capture["export_variant"]["selector_key"] += ";export_variant=compact-d2h-export-candidate"
        selector_review_path = tmp / "selector-review-only.json"
        selector_review_path.write_text(json.dumps(selector_review_capture), encoding="utf-8")
        selector_review_ledger = promotion_ledger.build_ledger([selector_review_path], cache_path)
        selector_review_entry = selector_review_ledger["entries"][0]
        assert selector_review_entry["cache_scope"] == "selector_review_only_non_default"
        assert "selector_review_only_not_runtime_cache_route" in selector_review_entry["promotion_blockers"]
        assert "export_selector_contract_mismatch" in selector_review_entry["stale_invalidation_reasons"]
        failure_path = tmp / "starfoundry-cpu.failed.json"
        failure_path.write_text(
            json.dumps(
                {
                    "command": [
                        "rns8-bench.exe",
                        "--backend",
                        "cpu",
                        "--semantics",
                        "bounded-i64",
                        "--m",
                        str(capture["m"]),
                        "--n",
                        str(capture["n"]),
                        "--k",
                        str(capture["k"]),
                        "--warmups",
                        "3",
                        "--repeats",
                        "9",
                        "--seed",
                        "20260605",
                        "--release-gate",
                        "large-release-validation-4096-budgeted",
                    ],
                    "returncode": None,
                    "timed_out": True,
                    "timeout_seconds": 60.0,
                    "stdout": "",
                    "stderr": "",
                }
            ),
            encoding="utf-8",
        )
        failed_release_report = release_gate_report.build_report([capture_path, failure_path])
        assert failed_release_report["failed_capture_count"] == 1
        assert failed_release_report["input_count"] == 2
        assert failed_release_report["failed_rows"][0]["backend"] == "cpu-reference"
        assert failed_release_report["failed_rows"][0]["failure_kind"] == "timeout"
        assert failed_release_report["blocker_counts"]["failed_required_baselines"] == 1
        assert failed_release_report["blocker_counts"]["required_baseline_release_review_missing"] == 1
        assert failed_release_report["blocker_counts"]["required_baseline_timeout"] == 1
        failed_group = failed_release_report["groups"][0]
        assert failed_group["failed_capture_count"] == 1
        assert failed_group["required_baselines_attempted"] == ["cpu-reference"]
        assert failed_group["missing_required_baselines"] == [
            "cpu-reference",
            "hip-direct",
            "hip-vector-alu-int64",
        ]
        assert failed_group["missing_release_review_required_baselines"] == [
            "cpu-reference",
            "hip-direct",
            "hip-vector-alu-int64",
        ]
        assert failed_group["failed_required_baselines"] == ["cpu-reference"]
        assert failed_group["timed_out_required_baselines"] == ["cpu-reference"]
        assert failed_group["historical_failed_required_baselines"] == ["cpu-reference"]
        assert failed_group["historical_timed_out_required_baselines"] == ["cpu-reference"]
        assert failed_group["superseded_required_baseline_failures"] == []
        assert failed_group["unattempted_required_baselines"] == ["hip-direct", "hip-vector-alu-int64"]
        assert failed_group["required_baselines_complete"] is False
        assert failed_group["required_baselines_release_review_complete"] is False
        assert failed_group["required_baseline_attempts_complete"] is False

        original_load = release_gate_report._load
        try:
            completed_required_path = tmp / "completed-cpu.json"
            completed_required_path.write_text("{}", encoding="utf-8")

            def fake_load(path: Path) -> dict:
                loaded = original_load(capture_path)
                loaded["_path"] = str(path)
                loaded["backend_selected"] = "cpu-reference"
                loaded["backend_requested"] = "cpu-reference"
                return loaded

            release_gate_report._load = fake_load
            superseded_report = release_gate_report.build_report([completed_required_path, failure_path])
        finally:
            release_gate_report._load = original_load

        superseded_group = superseded_report["groups"][0]
        assert superseded_group["required_baselines_attempted"] == ["cpu-reference"]
        assert superseded_group["failed_required_baselines"] == []
        assert superseded_group["timed_out_required_baselines"] == []
        assert superseded_group["historical_failed_required_baselines"] == ["cpu-reference"]
        assert superseded_group["historical_timed_out_required_baselines"] == ["cpu-reference"]
        assert superseded_group["superseded_required_baseline_failures"][0]["backend"] == "cpu-reference"
        assert "failed_required_baselines" not in superseded_group["blockers"]
        assert "required_baseline_timeout" not in superseded_group["blockers"]

    search = modulus_set_search.build_report([("test", [251, 253, 255, 256])], 32)
    assert search["candidates"][0]["pairwise_coprime"] is True
    assert search["candidates"][0]["satisfies_min_bits"] is True

    print("starfoundry report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
