#!/usr/bin/env python3
"""Self-test Direct-HIP prefix fusion reporting."""

from __future__ import annotations

import copy

import direct_hip_prefix_fusion_report as report


def capture(
    *,
    kernel: str,
    median_us: float,
    semantics: str = "bounded_i64",
    mode: str = "public_oneshot_transient_native_inputs",
    path: str = "capture.json",
    checksum: int = 1234,
    schema_status: str = "valid",
) -> dict:
    return {
        "_path": path,
        "_schema_status": schema_status,
        "_schema_errors": ["legacy schema"] if schema_status != "valid" else [],
        "schema_version": 4,
        "benchmark": "rns8_bounded_gemm_public_oneshot"
        if mode == "public_oneshot_transient_native_inputs"
        else "rns8_bounded_gemm_persistent_rns",
        "benchmark_execution_mode": mode,
        "backend_selected": "hip-direct",
        "backend_requested": "hip-direct",
        "selected_kernel": kernel,
        "backend_metadata": {"selected_kernel": kernel},
        "semantics": semantics,
        "m": 512,
        "n": 512,
        "k": 512,
        "prefix": 9,
        "selected_prefix": 9,
        "requested_max_prefix": 9,
        "contract_prefix_policy": "fixed_requested",
        "input_distribution": "signed_uniform_-16_16" if semantics == "bounded_i64" else "unsigned_uniform_0_16",
        "seed": 20260606,
        "warmups": 3,
        "repeats": 9,
        "checksum_u64": checksum,
        "timing_summary_us": {
            "end_to_end": {"median": median_us},
            "rns_gemm": {"median": median_us * 0.5},
        },
        "timing_metadata": {
            "gpu_event_timing": True,
            "gpu_event_timing_status": "available",
            "gpu_event_phase_order": ["oneshot_native_input_h2d", "rns_gemm_kernel_group", "crt_export"],
            "gpu_event_timing_source": "hipEventElapsedTime",
        },
    }


def export_selector_report() -> dict:
    return {
        "capture_count": 2,
        "groups": [
            {
                "key": {"backend": "hip-direct", "semantics": "exact_wide_signed"},
                "rows": [
                    {
                        "backend": "hip-direct",
                        "target_id": "gfx1100",
                        "semantics": "exact_wide_signed",
                        "shape": {"m": 1024, "n": 1024, "k": 1024},
                        "capture_path": "prefix20.json",
                        "selected_kernel": "hip_direct_export_exact_wide_signed_limbs_device",
                        "export_variant": "prefix20-fixed-export-candidate",
                        "reconstruction_variant": "default_garner",
                        "median_end_to_end_us": 1500.0,
                        "median_export_us": 300.0,
                        "selector_promotion_eligible": False,
                        "promotion_blockers": ["experimental_export_variant"],
                    }
                ],
            }
        ],
    }


def main() -> int:
    baseline = capture(
        kernel=report.ONESHOT_V1_KERNEL,
        median_us=3000.0,
        path="before-v1.json",
        schema_status="legacy_schema",
    )
    candidate = capture(
        kernel=report.ONESHOT_V2_COLPAIR_KERNEL,
        median_us=1000.0,
        path="current-v2.json",
    )
    resident = capture(
        kernel="direct_hip_prefix9_grouped_rns_gemm_v1",
        median_us=800.0,
        mode="persistent_resident_matrices",
        path="resident.json",
    )
    resident_default = capture(
        kernel=report.RESIDENT_DEFAULT_KERNEL,
        median_us=900.0,
        mode="persistent_resident_matrices",
        path="before-rerun-resident-default.json",
    )
    resident_colpair = capture(
        kernel=report.RESIDENT_COLPAIR_KERNEL,
        median_us=1200.0,
        mode="persistent_resident_matrices",
        path="resident-colpair.json",
    )

    result = report.build_direct_hip_prefix_fusion_report(
        [baseline, candidate, resident, resident_default, resident_colpair],
        [export_selector_report()],
    )
    summary = result["summary"]
    assert summary["one_shot_colpair_comparisons"] == 1
    assert summary["resident_colpair_comparisons"] == 1
    assert summary["prefix20_export_selector_rows"] == 1
    assert summary["candidate_one_shot_wins"] == 1
    assert summary["candidate_resident_wins"] == 0
    assert summary["deprioritized"] == 1
    assert summary["experimental"] == 1
    assert summary["legacy_before_captures"] == 1

    one_shot = next(item for item in result["comparisons"] if item["kind"] == "prefix9_public_one_shot_colpair")
    assert one_shot["decision"] == "candidate_one_shot_win"
    assert one_shot["speedup_vs_legacy_v1"] == 3.0
    assert one_shot["resident_reference_faster"] is True
    assert "baseline_legacy_schema" in one_shot["blockers"]

    resident_row = next(
        item for item in result["comparisons"] if item["kind"] == "prefix9_resident_selected_prefix_colpair"
    )
    assert resident_row["decision"] == "deprioritize"
    assert resident_row["speedup_vs_resident_default"] == 0.75

    missing = report.build_direct_hip_prefix_fusion_report([candidate], [])
    assert missing["summary"]["experimental"] == 1
    assert missing["comparisons"][0]["decision"] == "keep_experimental"
    assert "missing_legacy_v1_one_shot_baseline" in missing["comparisons"][0]["blockers"]

    mismatch = copy.deepcopy(candidate)
    mismatch["checksum_u64"] = 999
    mismatch_report = report.build_direct_hip_prefix_fusion_report([baseline, mismatch], [])
    assert mismatch_report["comparisons"][0]["decision"] == "keep_experimental"
    assert "checksum_mismatch_with_v1_baseline" in mismatch_report["comparisons"][0]["blockers"]

    print("direct-HIP prefix fusion report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
