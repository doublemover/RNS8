#!/usr/bin/env python3
"""Self-test Direct-HIP prefix fusion reporting."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

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
        "input_distribution": "signed_uniform_-16_16"
        if semantics == "bounded_i64"
        else "unsigned_uniform_0_16",
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
            "gpu_event_phase_order": [
                "oneshot_native_input_h2d",
                "rns_gemm_kernel_group",
                "crt_export",
            ],
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


def production_controls() -> None:
    from test_direct_hip_report_support import collect_resident_processes
    from benchmark_schema import validate_capture
    from test_review_reference_support import add_synthetic_full_reference
    from diagnostic_companion import build_binding
    import json

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        captures, paths, bindings = collect_resident_processes(
            root,
            "matched",
            dict(
                default=(report.RESIDENT_DEFAULT_KERNEL, 100),
                candidate=(report.RESIDENT_COLPAIR_KERNEL, 80),
            ),
        )
        for fixture in captures:
            validate_capture(fixture)

        def build(values, receipts=bindings):
            return report.build_direct_hip_prefix_fusion_report(
                values, diagnostic_bindings=receipts
            )

        row = build(captures)["comparisons"][0]
        assert row["decision"] == "candidate_resident_win", row["blockers"]
        assert row["speedup_vs_resident_default"] == 1.25
        assert row["promotion_eligible"] is False
        assert len(row["comparison"]["candidate_production_observations"]) == 2
        for order in (list(reversed(captures)), captures[2:] + captures[:2]):
            assert build(order)["comparisons"][0]["speedup_vs_resident_default"] == 1.25
        copied = build(captures[:5] + [copy.deepcopy(captures[4])])["comparisons"][0]
        assert copied["speedup_vs_resident_default"] is None
        incomplete = dict(bindings)
        incomplete.pop(str(paths[-1].resolve()))
        assert build(captures, incomplete)["comparisons"][0]["speedup_vs_resident_default"] is None
        changed = copy.deepcopy(captures)
        for fixture in changed[-2:]:
            fixture["seed"] += 1
            add_synthetic_full_reference(fixture)
        assert build(changed)["comparisons"][0]["speedup_vs_resident_default"] is None
        changed = copy.deepcopy(captures)
        for fixture in changed[-2:]:
            fixture["build_manifest"]["source_digest"] = "9" * 64
        assert build(changed)["comparisons"][0]["speedup_vs_resident_default"] is None

        # Rename a baseline to "after" and a candidate to "before"; file labels
        # cannot supply experiment arms or change strongest baseline selection.
        renamed, receipts = [], {}
        for i, source in enumerate(paths):
            target = source.parent / ("before-candidate.json" if i >= 4 else "after-anchor.json")
            target.write_bytes(source.read_bytes())
            raw = json.loads(target.read_text())
            raw["_path"] = str(target)
            renamed.append(raw)
            if i >= 2:
                receipts[str(target.resolve())] = build_binding(
                    target, source.parent / "instrumented.json"
                )
        renamed_report = build(renamed, receipts)
        assert renamed_report["comparisons"][0]["speedup_vs_resident_default"] == 1.25
        loaded = report.build_report(
            [Path(raw["_path"]) for raw in renamed], diagnostic_bindings=receipts
        )
        assert loaded["comparisons"][0]["decision"] == "candidate_resident_win"


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
    # Historical minimal records and private "valid" labels cannot grant credit.
    assert summary["legacy_capture_count"] == 5
    assert summary["candidate_one_shot_wins"] == 0
    assert summary["candidate_resident_wins"] == 0
    assert result["comparisons"] == []
    assert len(result["legacy_inventory"]) == 5
    assert result["prefix20_export_selector_rows"][0]["selector_promotion_eligible"] is False
    production_controls()

    print("direct-HIP prefix fusion report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
