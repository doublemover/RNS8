#!/usr/bin/env python3
"""Self-test bound-discovery comparison reporting."""

from __future__ import annotations

import copy
from pathlib import Path

import bound_discovery_report
from benchmark_schema import load_capture, validate_capture


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def with_path(capture: dict, path: str) -> dict:
    capture = copy.deepcopy(capture)
    capture["_path"] = path
    return capture


def mark_release_counts(capture: dict) -> None:
    capture["warmups"] = bound_discovery_report.RELEASE_MIN_WARMUPS
    capture["repeats"] = bound_discovery_report.RELEASE_MIN_REPEATS


def set_phase(capture: dict, end_to_end: int) -> None:
    repeats = capture.get("repeats", bound_discovery_report.RELEASE_MIN_REPEATS)
    for phase in bound_discovery_report.PHASES:
        value = end_to_end if phase == "end_to_end" else int(capture["timing_summary_us"][phase]["median"])
        capture["raw_timings_us"][phase] = [value] * repeats
        capture["timing_summary_us"][phase] = {
            "avg": float(value),
            "median": float(value),
            "p95": float(value),
        }
        avg_field = {
            "pack": "avg_pack_us",
            "rns_gemm": "avg_rns_gemm_us",
            "crt_export": "avg_crt_export_us",
            "end_to_end": "avg_end_to_end_us",
        }[phase]
        capture[avg_field] = float(value)
    timings = capture.get("gpu_event_timings_us")
    summaries = capture.get("gpu_event_timing_summary_us")
    if isinstance(timings, dict) and isinstance(summaries, dict):
        for phase, values in list(timings.items()):
            current = summaries.get(phase, {})
            value = float(current.get("median", values[-1] if values else 0.0))
            timings[phase] = [value] * repeats
            summaries[phase] = {
                "avg": value,
                "median": value,
                "p95": value,
            }


def static_global_capture(end_to_end: int = 1000) -> dict:
    capture = load_capture(FIXTURE_DIR / "v4_bounded_i64_ck.json")
    validate_capture(capture)
    capture = with_path(capture, "static-global-ck.json")
    mark_release_counts(capture)
    capture["bound_source"] = "static_profile"
    capture["bound_mode"] = "global"
    capture["bound_discovery"] = {
        "source": "static_profile_contract",
        "static_bound": capture["bound"],
        "selected_bound": capture["bound"],
        "discovered_global_bound": None,
        "candidate_row_sum_col_max": None,
        "candidate_row_max_col_sum": None,
        "row_abs_sum_max": None,
        "row_abs_max": None,
        "col_abs_sum_max": None,
        "col_abs_max": None,
        "zero_row_count": None,
        "zero_col_count": None,
    }
    set_phase(capture, end_to_end)
    return capture


def input_scan_global_capture(static_capture: dict, end_to_end: int, setup_us: int) -> dict:
    capture = with_path(static_capture, "input-scan-global-ck.json")
    capture["bound_source"] = "input_scan"
    capture["bound"] = min(int(capture["bound"]), 4096)
    capture["bound_discovery"] = {
        "source": "input_row_column_abs_summary",
        "static_bound": static_capture["bound"],
        "selected_bound": capture["bound"],
        "discovered_global_bound": capture["bound"],
        "candidate_row_sum_col_max": capture["bound"],
        "candidate_row_max_col_sum": capture["bound"] * 2,
        "row_abs_sum_max": 64,
        "row_abs_max": 16,
        "col_abs_sum_max": 64,
        "col_abs_max": 16,
        "zero_row_count": 1,
        "zero_col_count": 2,
    }
    capture["avg_global_bound_scan_us"] = float(setup_us)
    capture["global_bound_scan_us"] = setup_us
    capture["raw_timings_us"]["global_bound_scan"] = [setup_us]
    capture["timing_summary_us"]["global_bound_scan"] = {
        "avg": float(setup_us),
        "median": float(setup_us),
        "p95": float(setup_us),
    }
    set_phase(capture, end_to_end)
    return capture


def proof_mask_capture(static_capture: dict, end_to_end: int, setup_us: int, *, with_proofs: bool = True) -> dict:
    capture = with_path(static_capture, "proof-mask-per-tile-ck.json")
    capture["bound_source"] = "input_scan"
    capture["bound_mode"] = "per_tile"
    capture["bound_kind"] = "per_tile_max_abs"
    capture["bound"] = 0
    capture["tile_m"] = 64
    capture["tile_n"] = 64
    capture["tile_bounds_u64"] = {
        "source": "exact_seeded_input_prepass",
        "pattern": "exact_output_tile_max_abs_v1",
        "order": "row_major_output_tiles",
        "count": 2,
        "min": 0,
        "max": 4096,
        "hash_u64": 123,
    }
    zero_a = 1 if with_proofs else 0
    zero_b = 2 if with_proofs else 0
    capture["schedule_metadata"].update(
        {
            "bound_kind": "per_tile_max_abs",
            "effective_bound": 0,
            "tile_m": 64,
            "tile_n": 64,
            "tile_rows": 1,
            "tile_cols": 2,
            "tile_count": 2,
            "adaptive_prefix_active": True,
            "adaptive_skip_active": True,
            "adaptive_execution_applied": True,
            "zero_a_row_proof_count": zero_a,
            "zero_b_col_proof_count": zero_b,
            "zero_row_col_product_count": zero_a * capture["n"] + (capture["m"] - zero_a) * zero_b,
            "planner_zero_a_row_count": zero_a,
            "planner_zero_b_col_count": zero_b,
            "planner_zero_row_col_product_count": zero_a * capture["n"] + (capture["m"] - zero_a) * zero_b,
        }
    )
    capture["avg_tile_bound_scan_us"] = float(setup_us)
    capture["tile_bound_scan_us"] = setup_us
    capture["raw_timings_us"]["tile_bound_scan"] = [setup_us]
    capture["timing_summary_us"]["tile_bound_scan"] = {
        "avg": float(setup_us),
        "median": float(setup_us),
        "p95": float(setup_us),
    }
    set_phase(capture, end_to_end)
    return capture


def main() -> int:
    static = static_global_capture()
    input_scan_win = input_scan_global_capture(static, 700, 100)
    proof_mask_win = proof_mask_capture(static, 650, 100)
    report = bound_discovery_report.compare_bound_discovery([static, input_scan_win, proof_mask_win])
    assert report["summary"]["comparisons"] == 2
    assert report["summary"]["candidate_wins"] == 2
    assert {item["variant"] for item in report["comparisons"]} == {"input_scan_global", "proof_mask_per_tile"}
    assert all(item["candidate_setup_inclusive_median_us"] < 1000 for item in report["comparisons"])

    input_scan_loss = input_scan_global_capture(static, 950, 100)
    input_scan_loss["_path"] = "input-scan-global-ck-loss.json"
    loss_report = bound_discovery_report.compare_bound_discovery([static, input_scan_loss])
    assert loss_report["summary"]["deprioritized"] == 1
    assert loss_report["comparisons"][0]["decision"] == "deprioritize"
    assert (
        "candidate_not_faster_than_same_backend_static_setup_inclusive"
        in loss_report["comparisons"][0]["blockers"]
    )

    missing_baseline = input_scan_global_capture(static, 700, 100)
    missing_baseline["backend_selected"] = "hipblaslt"
    missing_baseline["backend_requested"] = "hipblaslt"
    missing_baseline["_path"] = "input-scan-global-hipblaslt-missing.json"
    missing_report = bound_discovery_report.compare_bound_discovery([static, missing_baseline])
    assert missing_report["summary"]["missing_baselines"] == 1
    assert missing_report["comparisons"][0]["decision"] == "missing_baseline"

    no_proof = proof_mask_capture(static, 650, 100, with_proofs=False)
    no_proof["_path"] = "tile-bound-no-proof-ck.json"
    no_proof_report = bound_discovery_report.compare_bound_discovery([static, no_proof])
    assert no_proof_report["summary"]["experimental"] == 1
    assert no_proof_report["comparisons"][0]["variant"] == "tile_bound_per_tile"
    assert "missing_zero_row_col_proof_coverage" in no_proof_report["comparisons"][0]["blockers"]

    print("bound discovery report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
