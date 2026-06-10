#!/usr/bin/env python3
"""Self-test exact-wide residue-current chain pairing."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import exact_wide_chain_report


def phase_summary(value: float) -> dict:
    return {"avg": value, "median": value, "p95": value}


def capture(*, backend: str, mode: str, median_us: float) -> dict:
    residue_current = mode == "residue_current"
    return {
        "_path": f"{backend}-{mode}.json",
        "schema_version": 4,
        "benchmark_execution_mode": "residue_current_rns_chain" if residue_current else "residue_chain_final_host_export",
        "backend_requested": backend,
        "backend_selected": backend,
        "semantics": "exact_wide_signed",
        "m": 512,
        "n": 512,
        "k": 512,
        "prefix": 20,
        "selected_prefix": 20,
        "requested_max_prefix": 20,
        "contract_prefix_policy": "fixed_requested_residue_chain",
        "seed": 20260606,
        "input_distribution": "signed_uniform_-16_16",
        "exact_wide_limb_count": 4,
        "residue_chain_length": 3,
        "residue_chain_final_export": not residue_current,
        "residue_output_mode": "residue_current_rns" if residue_current else "host_export",
        "warmups": 3,
        "repeats": 9,
        "plan_packing": {
            "output_device_current": residue_current,
            "output_host_current": not residue_current,
        },
        "plan_lowering": {
            "rns_continuation_available": True,
        },
        "requested_next_op": {
            "resolved": "rns-gemm" if residue_current else "final-export",
        },
        "output_policy": {
            "per_repeat_logical_export": not residue_current,
            "final_checksum_export_after_repeats": residue_current,
            "status_handling": "structurally_elided",
        },
        "timing_metadata": {
            "gpu_event_timing": backend != "cpu-reference",
            "gpu_event_timing_status": "available" if backend != "cpu-reference" else "not_applicable",
            "gpu_event_timing_source": "hipEventElapsedTime" if backend != "cpu-reference" else None,
            "gpu_event_phase_order": ["pack", "rns_gemm"] if residue_current else ["pack", "rns_gemm", "crt_export"],
        },
        "timing_summary_us": {
            "end_to_end": phase_summary(median_us),
        },
    }


def first_row(report: dict) -> dict:
    return report["groups"][0]["rows"][0]


def main() -> int:
    cpu_final = capture(backend="cpu-reference", mode="final_output", median_us=100000.0)
    gpu_final = capture(backend="hip-direct", mode="final_output", median_us=1000.0)
    gpu_residue = capture(backend="hip-direct", mode="residue_current", median_us=600.0)
    report = exact_wide_chain_report.build_report_from_captures([cpu_final, gpu_final, gpu_residue])
    row = first_row(report)
    assert report["summary"]["ready_pairs"] == 1
    assert report["summary"]["experimental"] == 0
    assert row["decision"] == "paired_residue_current_chain_ready"
    assert round(row["final_output_vs_residue_current_ratio"], 4) == 1.6667
    assert round(row["cpu_vs_final_output_speedup"], 4) == 100.0
    assert row["residue_lifetime"]["available"] is True

    missing_final = exact_wide_chain_report.build_report_from_captures([cpu_final, gpu_residue])
    missing_final_row = first_row(missing_final)
    assert missing_final_row["decision"] == "keep_experimental"
    assert "missing_same_backend_final_output_capture" in missing_final_row["blockers"]

    broken_lifetime = copy.deepcopy(gpu_residue)
    broken_lifetime["plan_packing"]["output_device_current"] = False
    broken = exact_wide_chain_report.build_report_from_captures([cpu_final, gpu_final, broken_lifetime])
    broken_row = first_row(broken)
    assert broken_row["decision"] == "keep_experimental"
    assert "incomplete_residue_current_lifetime_metadata" in broken_row["blockers"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        capture_path = root / "capture.json"
        report_path = root / "exact-wide-chain-report.json"
        rns_report_path = root / "rns-chain-report.json"
        review_path = root / "review_report.json"
        manifest_path = root / "scenario_manifest.json"
        for path in [capture_path, report_path, rns_report_path, review_path, manifest_path]:
            path.write_text("{}", encoding="utf-8")
        expanded = exact_wide_chain_report.expand_inputs([root])
        assert capture_path in expanded
        assert report_path not in expanded
        assert rns_report_path not in expanded
        assert review_path not in expanded
        assert manifest_path not in expanded

    print("exact wide chain report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
