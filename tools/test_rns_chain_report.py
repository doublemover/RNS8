#!/usr/bin/env python3
"""Self-test final-output RNS-chain reporting."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import rns_chain_report


def phase_summary(value: float) -> dict:
    return {"avg": value, "median": value, "p95": value}


def capture(
    *,
    backend: str,
    median_us: float,
    pack_mode: str = "per_repeat_repack",
    setup_us: float | None = None,
    repeats: int = 9,
    warmups: int = 3,
    with_source_identity: bool = True,
) -> dict:
    reuse = pack_mode != "per_repeat_repack"
    operands = {
        "prepacked_reuse": ["A", "B"],
        "prepacked_reuse_a": ["A"],
        "prepacked_reuse_b": ["B"],
    }.get(pack_mode, [])
    result = {
        "_path": f"{backend}-{pack_mode}-{median_us}.json",
        "schema_version": 4,
        "benchmark_execution_mode": "residue_chain_final_host_export",
        "semantics": "exact_wide_signed",
        "bound_kind": "none",
        "bound_mode": "none",
        "bound": 0,
        "bound_source": "static_profile",
        "m": 128,
        "n": 128,
        "k": 128,
        "prefix": 20,
        "selected_prefix": 20,
        "requested_max_prefix": 20,
        "contract_prefix_policy": "fixed_requested_residue_chain",
        "residue_planes_requested": 20,
        "residue_planes_selected": 20,
        "residue_planes_skipped": 0,
        "residue_output_mode": "host_export",
        "residue_chain_length": 3,
        "residue_chain_final_export": True,
        "tile_m": 128,
        "tile_n": 128,
        "k_block_size": 65536,
        "seed": 20260605,
        "input_distribution": "signed_uniform_-16_16",
        "backend_requested": backend,
        "backend_selected": backend,
        "pack_mode": pack_mode,
        "reuse_packed_inputs": reuse,
        "prepack_reuse_operands": operands,
        "prepack_reuse_strategy": "persistent_matrix_residency" if reuse else "none",
        "warmups": warmups,
        "repeats": repeats,
        "avg_prepack_setup_us": setup_us,
        "requested_next_op": {"resolved": "final-export"},
        "output_policy": {
            "destination_layout": "contiguous_row_major",
            "status_handling": "structurally_elided",
            "per_repeat_logical_export": True,
            "final_checksum_export_after_repeats": False,
        },
        "exact_output_contract": {
            "requested_final_output": "exact_wide_limb_host",
            "limb_count": 4,
            "status_policy": "structurally_elided",
        },
        "timing_metadata": {
            "benchmark_execution_mode": "residue_chain_final_host_export",
            "residue_chain_final_export": True,
            "pack_mode": pack_mode,
            "prepack_reuse_operands": operands,
            "prepack_reuse_strategy": "persistent_matrix_residency" if reuse else "none",
            "gpu_event_timing": backend != "cpu-reference",
            "gpu_event_timing_status": "available" if backend != "cpu-reference" else "not_applicable",
            "gpu_event_timing_source": "hip_events" if backend != "cpu-reference" else None,
            "gpu_event_phase_order": ["pack", "rns_gemm", "crt_export"] if backend != "cpu-reference" else [],
        },
        "timing_summary_us": {
            "pack": phase_summary(100.0),
            "rns_gemm": phase_summary(200.0),
            "crt_export": phase_summary(100.0),
            "end_to_end": phase_summary(median_us),
        },
    }
    if reuse and with_source_identity:
        result["device_allocation"] = {
            "tracking_available": True,
            "source": "hip_direct_allocation_counters_snapshot",
            "setup_scope": "persistent_plan_workspace_prepacked_reuse",
            "source_version_inputs": "monotonic_source_version_per_repeat_when_packing_runs",
        }
    return result


def first_gpu_row(report: dict, *, backend: str = "hip-direct", pack_mode: str = "per_repeat_repack") -> dict:
    return next(
        row
        for group in report["groups"]
        for row in group["rows"]
        if row["backend"] == backend and row["pack_mode"] == pack_mode
    )


def main() -> int:
    cpu = capture(backend="cpu-reference", median_us=5000.0)
    direct = capture(backend="hip-direct", median_us=1000.0)
    report = rns_chain_report.build_report_from_captures([cpu, direct])
    row = first_gpu_row(report)
    assert report["summary"]["candidate_wins"] == 1
    assert row["decision"] == "candidate_final_output_chain_win"
    assert round(row["speedup_vs_cpu"], 4) == 5.0

    reuse = capture(backend="hip-direct", median_us=700.0, pack_mode="prepacked_reuse_b", setup_us=900.0)
    reuse_report = rns_chain_report.build_report_from_captures([cpu, direct, reuse])
    reuse_row = first_gpu_row(reuse_report, pack_mode="prepacked_reuse_b")
    assert reuse_row["decision"] == "candidate_reuse_chain_win"
    assert round(reuse_row["setup_inclusive_median_per_repeat_us"], 4) == 800.0
    assert round(reuse_row["speedup_vs_same_backend_nonreuse"], 4) == 1.25
    assert reuse_row["break_even_repeats_same_backend"] == 4

    slow_reuse = copy.deepcopy(reuse)
    slow_reuse["_path"] = "slow-reuse.json"
    slow_reuse["avg_prepack_setup_us"] = 3600.0
    slow_report = rns_chain_report.build_report_from_captures([cpu, direct, slow_reuse])
    slow_row = first_gpu_row(slow_report, pack_mode="prepacked_reuse_b")
    assert slow_row["decision"] == "deprioritize"
    assert "reuse_not_faster_than_same_backend_nonreuse_setup_inclusive" in slow_row["blockers"]

    missing_cpu_report = rns_chain_report.build_report_from_captures([direct])
    missing_row = first_gpu_row(missing_cpu_report)
    assert missing_row["decision"] == "missing_baseline"
    assert "missing_cpu_final_output_baseline" in missing_row["blockers"]

    missing_identity = copy.deepcopy(reuse)
    missing_identity["_path"] = "missing-identity.json"
    del missing_identity["device_allocation"]
    identity_report = rns_chain_report.build_report_from_captures([cpu, direct, missing_identity])
    identity_row = first_gpu_row(identity_report, pack_mode="prepacked_reuse_b")
    assert identity_row["decision"] == "keep_experimental"
    assert "missing_device_allocation_metadata" in identity_row["blockers"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report_path = tmp_path / "rns-chain-report.json"
        capture_path = tmp_path / "capture.json"
        report_path.write_text("{}", encoding="utf-8")
        capture_path.write_text("{}", encoding="utf-8")
        expanded = rns_chain_report.expand_inputs([tmp_path])
        assert capture_path in expanded
        assert report_path not in expanded

    print("rns chain report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
