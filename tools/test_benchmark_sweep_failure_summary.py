#!/usr/bin/env python3
"""Self-test compact benchmark sweep failure summaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import benchmark_sweep_failure_summary as summary


def _capture(backend: str, checksum: int, scenario_name: str) -> dict:
    return {
        "backend_selected": backend,
        "backend_requested": backend,
        "selected_kernel": f"{backend}_kernel",
        "semantics": "bounded_i64",
        "finite_modulus": None,
        "bound_kind": "global_max_abs",
        "bound_mode": "global",
        "bound": 16384,
        "bound_source": "static_profile",
        "m": 64,
        "n": 64,
        "k": 64,
        "prefix": 9,
        "layout": "row_major",
        "tile_m": 128,
        "tile_n": 128,
        "k_block_size": 64,
        "exact_wide_limb_count": None,
        "residue_chain_length": 1,
        "residue_output_mode": "host_export",
        "seed": 20260606,
        "input_distribution": "signed_uniform_-16_16",
        "reuse_packed_inputs": False,
        "pack_mode": "per_repeat_repack",
        "requested_next_op": {"resolved": "host_export"},
        "output_policy": {
            "destination_layout": "contiguous_row_major",
            "status_handling": "required",
        },
        "timing_metadata": {
            "benchmark_execution_mode": "persistent_resident_matrices",
            "fusion_mode": "none",
            "residue_group_width": 1,
        },
        "checksum_u64": checksum,
        "scenario_metadata": {
            "family": "summary-test",
            "name": scenario_name,
            "promotion_eligibility": "release_review_candidate",
            "output_domain": "host_export",
            "metadata": {"workflow_name": scenario_name},
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        out = Path(tmp_name) / "cdna-all-promotable-mi300x-test"
        scenarios = out / "rank-scenarios" / "all" / "scenarios" / "summary-test"

        a_cpu = _capture("cpu-reference", 111, "workflow-a")
        a_direct = _capture("hip-direct", 111, "workflow-a")
        b_cpu = _capture("cpu-reference", 222, "workflow-b")
        b_direct = _capture("hip-direct", 222, "workflow-b")
        c_cpu = _capture("cpu-reference", 333, "workflow-c")
        c_ck = _capture("ck", 444, "workflow-c")
        for name, payload in [
            ("a-cpu.json", a_cpu),
            ("a-direct.json", a_direct),
            ("b-cpu.json", b_cpu),
            ("b-direct.json", b_direct),
            ("c-cpu.json", c_cpu),
            ("c-ck.json", c_ck),
        ]:
            _write(scenarios / name, payload)

        failed = copy.deepcopy(c_ck)
        failed["stderr"] = "rns8-bench start\nrns8_create_plan: range error"
        _write(scenarios / "bad.failed.json", failed)
        _write(
            out / "rank-scenarios" / "all" / "review_report.json",
            {
                "groups": [
                    {
                        "semantics": "bounded_i64",
                        "shape": {"m": 64, "n": 64, "k": 64},
                        "contract_key": "release-candidate-contract",
                        "candidates": [
                            {
                                "backend": "ck",
                                "accelerator_backend": True,
                                "scenario_promotion_scope": "release_review_candidate",
                                "selected_kernel": "ck_kernel",
                                "median_end_to_end_us": 123,
                                "speedup_vs_direct_hip": 0.8,
                                "speedup_vs_vector_alu": 1.2,
                                "capture": str(scenarios / "c-ck.json"),
                                "promotion_blockers": ["not_faster_than_direct_hip"],
                            },
                            {
                                "backend": "hip-direct",
                                "accelerator_backend": False,
                                "scenario_promotion_scope": "release_review_candidate",
                                "selected_kernel": "direct_kernel",
                                "median_end_to_end_us": 100,
                                "speedup_vs_direct_hip": 1.0,
                                "primary_loss_phase_vs_direct_hip": None,
                                "bottleneck": {"class": "mixed_bound", "phase": "rns_gemm"},
                                "capture": str(scenarios / "b-direct.json"),
                                "promotion_blockers": ["not_accelerator_backend"],
                            },
                            {
                                "backend": "ck",
                                "accelerator_backend": True,
                                "scenario_promotion_scope": "proxy_evidence_only",
                                "promotion_blockers": ["scenario_scope_not_autotune_promotable"],
                            },
                        ],
                        "fastest_production_route": {
                            "backend": "hip-direct",
                            "accelerator_backend": False,
                            "scenario_promotion_scope": "release_review_candidate",
                            "selected_kernel": "direct_kernel",
                            "median_end_to_end_us": 100,
                            "speedup_vs_direct_hip": 1.0,
                            "primary_loss_phase_vs_direct_hip": None,
                            "bottleneck": {"class": "mixed_bound", "phase": "rns_gemm"},
                            "matrix_instruction_histogram": {},
                            "capture": str(scenarios / "b-direct.json"),
                        },
                        "fastest_accelerator_route": {
                            "backend": "ck",
                            "accelerator_backend": True,
                            "scenario_promotion_scope": "release_review_candidate",
                            "selected_kernel": "ck_kernel",
                            "median_end_to_end_us": 123,
                            "speedup_vs_direct_hip": 0.8,
                            "primary_loss_phase_vs_direct_hip": "pack",
                            "bottleneck": {"class": "pack_bound", "phase": "pack"},
                            "matrix_instruction_histogram": {},
                            "capture": str(scenarios / "c-ck.json"),
                        },
                    }
                ]
            },
        )
        _write(
            out / "isa-reports" / "ck_backend_kernels-gfx942-ck-isa-summary.json",
            {
                "backend": "ck",
                "target": "gfx942",
                "instruction_totals": {
                    "matrix_instruction_histogram": {"v_mfma_i32_16x16x32_i8": 2},
                    "matrix_instruction_families": ["mfma"],
                },
            },
        )

        lines = summary.build_summary(out)
        text = "\n".join(lines)
        assert "FAILED_CAPTURES 1" in text
        assert "rns8_create_plan: range error" in text
        assert "CHECKSUM_MISMATCH_GROUPS 1" in text
        assert "workflow-c" in text
        assert "workflow-a" not in text
        assert "workflow-b" not in text
        assert "not_faster_than_direct_hip 1" in text
        assert "not_accelerator_backend 1" in text
        assert "scenario_scope_not_autotune_promotable 1" in text
        assert "ACTIONABLE_PROMOTION_BLOCKER_COUNTS" in text
        assert "ACTIONABLE_PROMOTION_CANDIDATES 1" in text
        assert "ck semantics=bounded_i64 shape=64x64x64" in text
        assert "FASTEST_PRODUCTION_ROUTES 1" in text
        assert "production backend=hip-direct semantics=bounded_i64 shape=64x64x64" in text
        assert "FASTEST_ACCELERATOR_ROUTES 1" in text
        assert "accelerator backend=ck semantics=bounded_i64 shape=64x64x64" in text
        assert "matrix_isa=v_mfma_i32_16x16x32_i8:2" in text

    print("benchmark sweep failure summary self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
