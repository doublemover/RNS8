#!/usr/bin/env python3
"""Self-test zero-skip expansion readiness reporting."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from benchmark_schema import load_capture
import zero_skip_expansion_report


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def with_path(capture: dict, path: str) -> dict:
    capture = copy.deepcopy(capture)
    capture["_path"] = path
    return capture


def set_proof_source(capture: dict, source: str) -> None:
    scenario = capture.setdefault("scenario_metadata", {})
    metadata = scenario.setdefault("metadata", {})
    metadata["proof_source"] = source


def set_row_col_proofs(capture: dict, *, backend: str, kernel: str, proof_source: str) -> dict:
    capture = with_path(capture, f"{backend}-row-col.json")
    capture["backend_selected"] = backend
    capture["backend_requested"] = backend
    capture["selected_kernel"] = kernel
    capture.setdefault("backend_metadata", {})["selected_kernel"] = kernel
    schedule = capture["schedule_metadata"]
    schedule.update(
        {
            "flags": 2,
            "zero_a_row_proof_count": 1,
            "zero_b_col_proof_count": 1,
            "zero_row_col_product_count": 129,
            "planner_zero_a_row_count": 1,
            "planner_zero_b_col_count": 1,
            "planner_zero_row_col_product_count": 129,
        }
    )
    set_proof_source(capture, proof_source)
    return capture


def main() -> int:
    direct_fixture = load_capture(FIXTURE_DIR / "v4_bounded_u64_adaptive_hip.json")
    ck_fixture = load_capture(FIXTURE_DIR / "v4_bounded_u64_adaptive_ck.json")
    rocwmma_fixture = load_capture(FIXTURE_DIR / "v4_bounded_u64_adaptive_rocwmma.json")
    hipblaslt_fixture = load_capture(FIXTURE_DIR / "v4_bounded_i64_hipblaslt.json")

    direct_kernel = "direct_hip_tiled_active_prefix_zero_row_col_skip_rns_gemm_v1"
    direct_caller = set_row_col_proofs(
        direct_fixture,
        backend="hip-direct",
        kernel=direct_kernel,
        proof_source="caller_provided_zero_proofs",
    )
    direct_scan = set_row_col_proofs(
        direct_fixture,
        backend="hip-direct",
        kernel=direct_kernel,
        proof_source="scan_derived_exact_seeded_input_prepass",
    )
    direct_scan["_path"] = "hip-direct-scan-derived-row-col.json"
    ck_fallback = set_row_col_proofs(
        ck_fixture,
        backend="ck",
        kernel="ck_wmma_cshuffle_tiled_i8_i32_default_moduli_static_centered_epilogue_v3",
        proof_source="caller_provided_zero_proofs",
    )
    rocwmma_fallback = set_row_col_proofs(
        rocwmma_fixture,
        backend="rocwmma",
        kernel="rocwmma_i8_i32_signed_tiled_mod251_255_256_hot_residue_v2",
        proof_source="caller_provided_zero_proofs",
    )
    hipblaslt_unsupported = set_row_col_proofs(
        hipblaslt_fixture,
        backend="hipblaslt",
        kernel="hipblaslt_int8_i32_matmul",
        proof_source="caller_provided_zero_proofs",
    )

    report = zero_skip_expansion_report.compare_zero_skip_expansion(
        [direct_caller, direct_scan, ck_fallback, rocwmma_fallback, hipblaslt_unsupported]
    )
    rows = {row["capture"]: row for row in report["captures"]}
    assert report["summary"]["captures"] == 5
    assert report["summary"]["direct_hip_row_col_skip_candidates"] == 1
    assert report["summary"]["experimental_direct_hip_row_col_skips"] == 1
    assert report["summary"]["accelerator_row_col_skip_candidates"] == 0
    assert report["summary"]["accelerator_correct_full_tile_fallbacks"] == 2
    assert report["summary"]["unsupported_row_col_captures"] == 1
    assert report["summary"]["rank22_expansion_ready"] is False

    assert rows["hip-direct-row-col.json"]["row_col_decision"] == "candidate_row_col_skip"
    assert rows["hip-direct-scan-derived-row-col.json"]["row_col_decision"] == "experimental_row_col_skip"
    assert "proof_source_is_scan_derived_or_unknown" in rows["hip-direct-scan-derived-row-col.json"]["row_col_blockers"]
    assert rows["ck-row-col.json"]["row_col_decision"] == "correct_full_tile_fallback"
    assert "backend_computes_full_tile_for_row_col_products" in rows["ck-row-col.json"]["row_col_blockers"]
    assert rows["rocwmma-row-col.json"]["row_col_decision"] == "correct_full_tile_fallback"
    assert rows["hipblaslt-row-col.json"]["row_col_decision"] == "unsupported"

    with tempfile.TemporaryDirectory() as tmp:
        md_path = Path(tmp) / "zero-skip-report.md"
        zero_skip_expansion_report.write_markdown(report, md_path)
        text = md_path.read_text(encoding="utf-8")
        assert "Zero-Skip Expansion Report" in text
        assert "correct_full_tile_fallback" in text

    print("zero skip expansion report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
