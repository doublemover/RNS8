#!/usr/bin/env python3
"""Self-test layout-search report classification."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from benchmark_schema import load_capture
import layout_search_report


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def with_common_layout(capture: dict, *, name: str, role: str, baseline: str | None = None) -> dict:
    capture = copy.deepcopy(capture)
    capture["_path"] = f"{name}.json"
    capture["warmups"] = 3
    capture["repeats"] = 9
    capture["scenario_metadata"] = {
        "family": "layout-search",
        "name": name,
        "semantics": "bounded-i64",
        "backend": "hip-direct",
        "modulus": None,
        "exact_wide_limb_count": None,
        "shape": {"m": capture["m"], "n": capture["n"], "k": capture["k"]},
        "output_domain": "host_export",
        "residue_chain_length": 1,
        "output_ld_padding": 0,
        "residue_channel_fusion": False,
        "native_to_rns_bridge": False,
        "vector_to_rns_chain": False,
        "metadata": {
            "workflow_name": "end_to_end_layout_search",
            "layout_role": "fixture_layout",
            "layout_variant_role": role,
            "layout_variant_name": name,
            "promotion_scope": "layout_comparison_only",
        },
    }
    if baseline:
        capture["scenario_metadata"]["metadata"]["layout_baseline_name"] = baseline
    capture.setdefault("timing_metadata", {})["gpu_event_timing"] = True
    capture["timing_metadata"]["gpu_event_timing_status"] = "available"
    capture["timing_metadata"]["gpu_event_phase_order"] = ["pack", "rns_gemm", "crt_export"]
    capture["timing_metadata"]["pack_layout"] = "resident_rns_residue_planes"
    capture["timing_metadata"]["residue_group_layout"] = "one_modulus_per_residue_plane"
    capture["output_policy"] = {
        "destination_layout": "contiguous_row_major",
        "logical_ld": capture["n"],
        "ld_padding": 0,
    }
    capture["timing_summary_us"] = {
        "pack": {"median": 20.0},
        "rns_gemm": {"median": 40.0},
        "crt_export": {"median": 30.0},
        "end_to_end": {"median": 100.0},
    }
    capture["gpu_event_timing_summary_us"] = {
        "crt_export_d2h": {"median": 5.0},
    }
    return capture


def make_candidate(base: dict, *, name: str, baseline: str, median: float, actual_delta: bool = True) -> dict:
    capture = with_common_layout(base, name=name, role="candidate", baseline=baseline)
    capture["timing_summary_us"]["end_to_end"]["median"] = median
    if actual_delta:
        capture["scenario_metadata"]["residue_channel_fusion"] = True
        capture["timing_metadata"]["pack_layout"] = "native_i8_row_major_residue_channel_width3"
        capture["timing_metadata"]["residue_group_layout"] = "first_prefix9_moduli_contiguous_width3_groups"
    return capture


def main() -> int:
    fixture = load_capture(FIXTURE_DIR / "v4_bounded_i64_hipblaslt.json")
    fixture["backend_requested"] = "hip-direct"
    fixture["backend_selected"] = "hip-direct"
    fixture["selected_kernel"] = "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3"
    baseline = with_common_layout(fixture, name="default-layout", role="baseline")
    winner = make_candidate(fixture, name="packed-layout", baseline="default-layout", median=80.0)
    metadata_only = make_candidate(
        fixture,
        name="metadata-only-layout",
        baseline="default-layout",
        median=70.0,
        actual_delta=False,
    )
    loser = make_candidate(fixture, name="slow-layout", baseline="default-layout", median=120.0)
    missing = make_candidate(fixture, name="missing-baseline-layout", baseline="absent-layout", median=75.0)
    report = layout_search_report.build_report([baseline, winner, metadata_only, loser, missing])
    rows = {row["layout_variant_name"]: row for row in report["rows"]}
    assert report["schema"] == "rns8_layout_search_report_v1"
    assert report["baseline_count"] == 1
    assert report["candidate_count"] == 4
    assert rows["packed-layout"]["decision"] == "promote locally"
    assert rows["packed-layout"]["speedup_vs_default_layout"] == 1.25
    assert "residue_channel_fusion_enabled" in rows["packed-layout"]["layout_deltas"]
    assert rows["metadata-only-layout"]["decision"] == "keep experimental"
    assert "metadata_only_layout_candidate" in rows["metadata-only-layout"]["promotion_blockers"]
    assert rows["slow-layout"]["decision"] == "drop/deprioritize"
    assert rows["missing-baseline-layout"]["decision"] == "keep experimental"
    assert "missing_default_layout_baseline" in rows["missing-baseline-layout"]["promotion_blockers"]

    with tempfile.TemporaryDirectory() as tmp:
        md_path = Path(tmp) / "layout-search-report.md"
        layout_search_report.write_markdown(report, md_path)
        text = md_path.read_text(encoding="utf-8")
        assert "Layout Search Report" in text
        assert "packed-layout" in text

    print("layout search report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
