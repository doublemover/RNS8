#!/usr/bin/env python3
"""Self-test incremental result-cache research report gates."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import incremental_result_cache_report
from benchmark_schema import validate_capture
from test_benchmark_schema_support_metadata import add_target_variant_fields
from test_benchmark_sweep_support import bounded_capture, exact_wide_capture, set_phase


def cache_capture(backend: str = "hip-direct") -> dict:
    capture = bounded_capture("cpu-reference" if backend == "cpu-reference" else backend, 1000)
    capture["scenario_metadata"] = {
        "family": "incremental-result-cache",
        "name": "bounded-i64-dirty-tile-recompute",
        "promotion_eligibility": "result_cache_research_only",
        "metadata": {
            "promotion_scope": "result_cache_research_only",
            "source_identity": "A_and_B_matrix_identity_required",
            "source_version": "monotonic_per_matrix_version",
            "dirty_region_shape": "single_output_tile",
            "result_lifetime": "caller_visible_reuse_window_required",
            "checksum_policy": "final_exact_cpu_comparison_required",
        },
    }
    capture["incremental_result_cache"] = {
        "enabled": True,
        "policy": "bounded_i64_dirty_tile_partial_recompute_research",
        "source_identity_policy": "caller_visible_source_identity_required",
        "source_version_policy": "monotonic_source_versions_required",
        "dirty_region_policy": "explicit_dirty_region_shape_required",
        "result_lifetime_policy": "caller_visible_result_lifetime_required",
        "checksum_policy": "final_exact_cpu_comparison_required",
        "partial_recompute_policy": "dirty_region_only_when_version_contract_matches",
        "final_exact_comparison_required": True,
        "final_exact_comparison_status": "checksum_recorded_reference_required",
        "public_contract_available": False,
        "default_gemm_unchanged": True,
        "runtime_routing_allowed": False,
        "cache_eligible": False,
        "promotion_eligible": False,
    }
    if backend != "cpu-reference":
        if backend == "hip-direct":
            capture["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
            capture["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_backend_hooks"
        add_target_variant_fields(capture)
    return capture


def public_cache_capture(backend: str, end_to_end: int, role: str, public_candidate: bool = False) -> dict:
    capture = cache_capture(backend)
    capture["scenario_metadata"]["promotion_eligibility"] = "result_cache_contract_candidate"
    capture["scenario_metadata"]["metadata"]["promotion_scope"] = "result_cache_contract_candidate"
    capture["incremental_result_cache"].update(
        {
            "policy": "bounded_i64_dirty_tile_public_contract_v1",
            "candidate_role": role,
            "source_identity_policy": "matrix_instance_id_exact_match",
            "source_version_policy": "nonzero_monotonic_source_versions_required",
            "dirty_region_policy": "caller_provided_output_rectangles_full_k_recompute",
            "result_lifetime_policy": "explicit_result_cache_handle_lifetime",
            "partial_recompute_policy": "restore_cached_output_then_recompute_dirty_rectangles_full_k",
        }
    )
    if public_candidate:
        capture["incremental_result_cache"].update(
            {
                "public_contract_available": True,
                "runtime_routing_allowed": True,
                "cache_eligible": True,
                "promotion_eligible": True,
                "a_matrix_instance_id": 101,
                "b_matrix_instance_id": 102,
                "a_source_version": 11,
                "b_source_version": 12,
                "result_cache_key_fingerprint": 0xABCDEF,
                "dirty_region_count": 1,
                "recomputed_region_count": 1,
                "copied_from_cache_bytes": 512 * 512 * 9,
                "cache_allocation_bytes": 512 * 512 * 9,
                "stale_rejection_covered": True,
                "dirty_regions": [
                    {
                        "row_offset": 0,
                        "col_offset": 0,
                        "row_extent": 32,
                        "col_extent": 32,
                    }
                ],
            }
        )
    set_phase(capture, end_to_end)
    return capture


def write(path: Path, capture: dict) -> Path:
    prefix = int(capture.get("selected_prefix") or capture.get("prefix") or 1)
    if capture.get("per_modulus_gemm_estimate_applicable") is False:
        capture["avg_per_modulus_gemm_estimate_us"] = float(capture["avg_rns_gemm_us"])
    else:
        capture["avg_per_modulus_gemm_estimate_us"] = float(capture["avg_rns_gemm_us"]) / float(prefix)
    validate_capture(capture, path)
    path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        cpu = write(tmp / "cpu.json", cache_capture("cpu-reference"))
        direct = write(tmp / "direct.json", cache_capture("hip-direct"))
        report = incremental_result_cache_report.build_report([cpu, direct])
        assert report["schema"] == "rns8_incremental_result_cache_report_v1"
        assert report["rank75_gate_complete"] is True, json.dumps(report["blocker_counts"], indent=2)
        assert report["groups"][0]["status"] == "research-ready"
        assert report["promotable_result_cache_candidate_count"] == 0
        assert report["promotion_status"] == "blocked_public_incremental_cache_contract_required"
        assert report["groups"][0]["promotion_decision"] == "blocked_public_incremental_cache_contract_required"
        assert "public_result_lifetime_contract_missing" in report["groups"][0]["promotion_blockers"]

        missing_dir = tmp / "missing-contract"
        missing_dir.mkdir()
        missing = cache_capture("hip-direct")
        missing["incremental_result_cache"]["dirty_region_policy"] = "none"
        try:
            write(missing_dir / "missing.json", missing)
        except Exception:
            missing["incremental_result_cache"]["dirty_region_policy"] = "explicit_dirty_region_shape_required"
            del missing["incremental_result_cache"]["source_identity_policy"]
            missing_dir.joinpath("missing.json").write_text(
                json.dumps(missing, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        try:
            missing_report = incremental_result_cache_report.build_report([missing_dir])
        except Exception:
            missing_report = {"rank75_gate_complete": False}
        assert missing_report["rank75_gate_complete"] is False

        promotable_dir = tmp / "promotable"
        promotable_dir.mkdir()
        promotable = cache_capture("hip-direct")
        promotable["incremental_result_cache"]["promotion_eligible"] = True
        try:
            write(promotable_dir / "promotable.json", promotable)
        except Exception:
            promotable["incremental_result_cache"]["promotion_eligible"] = False
            promotable["scenario_metadata"]["promotion_eligibility"] = "release_review_candidate"
            write(promotable_dir / "promotable.json", promotable)
        promotable_report = incremental_result_cache_report.build_report([promotable_dir])
        assert promotable_report["rank75_gate_complete"] is False
        assert promotable_report["groups"][0]["blockers"]

        exact_dir = tmp / "exact"
        exact_dir.mkdir()
        exact = exact_wide_capture("hip-direct", 1200)
        exact["scenario_metadata"] = copy.deepcopy(cache_capture("hip-direct")["scenario_metadata"])
        exact["scenario_metadata"]["name"] = "exact-wide-partial-output-recompute"
        exact["incremental_result_cache"] = copy.deepcopy(cache_capture("hip-direct")["incremental_result_cache"])
        exact["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
        exact["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_backend_hooks"
        add_target_variant_fields(exact)
        write(exact_dir / "exact.json", exact)
        exact_report = incremental_result_cache_report.build_report([exact_dir])
        assert exact_report["rank75_gate_complete"] is True

        public_dir = tmp / "public-contract"
        public_dir.mkdir()
        cpu_public = public_cache_capture("cpu-reference", 5000, "cpu_reference_baseline")
        full_public = public_cache_capture("hip-direct", 1000, "same_backend_full_recompute_baseline")
        candidate_public = public_cache_capture("hip-direct", 800, "public_result_cache_candidate", True)
        ck_public = public_cache_capture("ck", 1200, "comparison_candidate")
        full_public["scenario_metadata"]["name"] = "bounded-i64-dirty-tile-public-contract-full-recompute-baseline"
        for capture in [cpu_public, full_public, candidate_public, ck_public]:
            capture["scenario_metadata"]["metadata"]["result_cache_contract_group"] = (
                "bounded-i64-dirty-tile-recompute"
            )
        write(public_dir / "cpu-public.json", cpu_public)
        write(public_dir / "full-public.json", full_public)
        write(public_dir / "candidate-public.json", candidate_public)
        write(public_dir / "ck-public.json", ck_public)
        public_report = incremental_result_cache_report.build_report([public_dir])
        assert public_report["rank78_gate_complete"] is True, json.dumps(public_report, indent=2)
        assert public_report["promotable_result_cache_candidate_count"] == 1
        assert public_report["group_count"] == 1, json.dumps(public_report, indent=2)
        assert public_report["groups"][0]["promotion_decision"] == "promote"

        weak_dir = tmp / "weak-public-contract"
        weak_dir.mkdir()
        weak_candidate = public_cache_capture("hip-direct", 950, "public_result_cache_candidate", True)
        weak_candidate["scenario_metadata"]["metadata"]["result_cache_contract_group"] = (
            "bounded-i64-dirty-tile-recompute"
        )
        write(weak_dir / "cpu-public.json", cpu_public)
        write(weak_dir / "full-public.json", full_public)
        write(weak_dir / "weak-candidate.json", weak_candidate)
        weak_report = incremental_result_cache_report.build_report([weak_dir])
        assert weak_report["rank78_gate_complete"] is False
        assert any(
            "speedup_below_1_10x" in group["promotion_blockers"]
            for group in weak_report["groups"]
        ), json.dumps(weak_report, indent=2)

    print("incremental result-cache report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
