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
                "summary": {
                    "next_work": [
                        {
                            "priority": "P1",
                            "work": "reduce_prepack_setup_or_reuse_steady_state_cost",
                            "reason": "reuse_not_faster_than_same_backend_setup_inclusive=1",
                        }
                    ],
                    "fastest_production_route_counts": {"hip-direct": 1},
                    "fastest_accelerator_route_counts": {"ck": 1},
                    "loss_phase_counts": {"pack": 1},
                    "bottleneck_counts": {"pack_bound": 1, "mixed_bound": 1},
                },
                "promotable_autotune_entries": [
                    {
                        "selected_backend": "ck",
                        "selected_kernel": "ck_kernel",
                        "median_end_to_end_us": 123,
                        "selection_end_to_end_us": 123,
                        "source_capture": str(scenarios / "c-ck.json"),
                    }
                ],
                "groups": [
                    {
                        "semantics": "bounded_i64",
                        "shape": {"m": 64, "n": 64, "k": 64},
                        "shape_family": "small_square",
                        "scenario_families": ["summary-test"],
                        "contract_key": "release-candidate-contract",
                        "required_baselines": ["cpu-reference", "hip-direct", "hip-vector-alu-int64"],
                        "missing_required_baselines": ["hip-vector-alu-int64"],
                        "scenario_promotion_scopes": ["release_review_candidate"],
                        "candidates": [
                            {
                                "backend": "ck",
                                "accelerator_backend": True,
                                "scenario_promotion_scope": "release_review_candidate",
                                "selected_kernel": "ck_kernel",
                                "median_end_to_end_us": 123,
                                "speedup_vs_direct_hip": 0.8,
                                "speedup_vs_vector_alu": 1.2,
                                "primary_loss_phase_vs_direct_hip": "pack",
                                "phase_medians_us": {
                                    "pack": 100.0,
                                    "pack_a": 70.0,
                                    "pack_b": 30.0,
                                    "rns_gemm": 20.0,
                                    "crt_export": 3.0,
                                    "end_to_end": 123.0,
                                },
                                "bottleneck": {"class": "pack_bound", "phase": "pack"},
                                "capture": str(scenarios / "c-ck.json"),
                                "promotion_blockers": ["not_faster_than_direct_hip"],
                                "prepacked_reuse_review": {
                                    "setup_inclusive_median_end_to_end_us": 150.0,
                                    "prepack_setup_us": 27.0,
                                    "same_backend_nonreuse_backend": "ck",
                                    "same_backend_nonreuse_median_end_to_end_us": 140.0,
                                    "best_nonreuse_backend": "hip-direct",
                                    "best_nonreuse_median_end_to_end_us": 100.0,
                                    "speedup_vs_same_backend_setup_inclusive": 0.9333333333333333,
                                    "speedup_vs_best_nonreuse_setup_inclusive": 0.6666666666666666,
                                },
                                "hip_graph_replay_review": {
                                    "setup_inclusive_median_end_to_end_us": 151.0,
                                    "graph_capture_us": 13.0,
                                    "graph_instantiate_us": 17.0,
                                    "graph_total_setup_us": 57.0,
                                    "baseline_backend": "hip-direct",
                                    "baseline_total_setup_us": 27.0,
                                    "baseline_setup_inclusive_median_end_to_end_us": 144.0,
                                    "break_even_repeat_count": 4,
                                    "declared_repeat_count": 9,
                                    "speedup_vs_non_graph_setup_inclusive": 0.9536423841059603,
                                },
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
                            "phase_medians_us": {
                                "pack": 100.0,
                                "pack_a": 70.0,
                                "pack_b": 30.0,
                                "rns_gemm": 20.0,
                                "crt_export": 3.0,
                                "end_to_end": 123.0,
                            },
                            "bottleneck": {"class": "pack_bound", "phase": "pack"},
                            "matrix_instruction_histogram": {},
                            "matrix_instruction_family": "mfma",
                            "matrix_instruction_shape": "16x16x32",
                            "matrix_instruction_dtype": "i8",
                            "matrix_instruction_sparsity": "dense",
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
        normalized = text.replace("\\", "/")
        assert "FAILED_CAPTURES 1" in text
        assert "CAPTURE_JSON_COUNT 6" in text
        assert "rns8_create_plan: range error" in text
        assert "CHECKSUM_MISMATCH_GROUPS 1" in text
        assert "workflow-c" in text
        assert "workflow-a" not in text
        assert "workflow-b" not in text
        assert "not_faster_than_direct_hip 1" in text
        assert "not_accelerator_backend 1" in text
        assert "scenario_scope_not_autotune_promotable 1" in text
        assert "PROMOTABLE_AUTOTUNE_ENTRIES 1" in text
        assert "backend=ck kernel=ck_kernel e2e=123 selection_e2e=123" in text
        assert "MISSING_REQUIRED_BASELINE_GROUPS 1" in text
        assert "missing=hip-vector-alu-int64" in text
        assert "present=ck,ck,hip-direct" in text
        assert "ROUTE_SUMMARY" in text
        assert "fastest_production hip-direct 1" in text
        assert "fastest_accelerator ck 1" in text
        assert "DIRECT_HIP_PRODUCTION_WINS 1" in text
        assert "shape_family=small_square" in text
        assert "LOSS_PHASE_COUNTS" in text
        assert "pack 1" in text
        assert "LOSS_PHASE_BY_BACKEND" in text
        assert "ck pack 1" in text
        assert "LOSS_PHASE_BY_SEMANTICS" in text
        assert "bounded_i64 pack 1" in text
        assert "LOSS_PHASE_BY_SHAPE_FAMILY" in text
        assert "small_square pack 1" in text
        assert "LOSS_PHASE_BY_SCENARIO_FAMILY" in text
        assert "summary-test pack 1" in text
        assert "BOTTLENECK_COUNTS" in text
        assert "pack_bound 1" in text
        assert "NEXT_WORK 1" in text
        assert "work=reduce_prepack_setup_or_reuse_steady_state_cost" in text
        assert "ACTIONABLE_PROMOTION_BLOCKER_COUNTS" in text
        assert "ACTIONABLE_PROMOTION_CANDIDATES 1" in text
        assert "review=rank-scenarios/all/review_report.json ck semantics=bounded_i64 shape=64x64x64" in normalized
        assert "details=pack_split=pack_a:70.0,pack_b:30.0 reuse_setup_e2e=150.0" in text
        assert "reuse_setup_e2e=150.0 prepack_setup=27.0 same_backend=ck" in text
        assert "reuse_vs_best=0.6666666666666666" in text
        assert "graph_total_setup=57.0" in text
        assert "baseline_total_setup=27.0" in text
        assert "graph_break_even_repeats=4" in text
        assert "graph_declared_repeats=9" in text
        assert "FASTEST_PRODUCTION_ROUTES 1" in text
        assert "production backend=hip-direct semantics=bounded_i64 shape=64x64x64" in text
        assert "FASTEST_ACCELERATOR_ROUTES 1" in text
        assert "accelerator backend=ck semantics=bounded_i64 shape=64x64x64" in text
        assert "pack_split=pack_a:70.0,pack_b:30.0 matrix_meta=mfma/16x16x32/i8/dense matrix_isa=" in text
        assert "matrix_isa=v_mfma_i32_16x16x32_i8:2" in text
        assert summary.clean_gate_failures(lines) == [
            "failed captures=1",
            "comparable checksum mismatch groups=1",
            "missing required baseline groups=1",
        ]

        clean_out = Path(tmp_name) / "cdna-rank79-clean-mi300x-test"
        clean_scenarios = clean_out / "rank-scenarios" / "all" / "scenarios" / "summary-test"
        _write(clean_scenarios / "clean-cpu.json", _capture("cpu-reference", 777, "clean-workflow"))
        _write(clean_scenarios / "clean-direct.json", _capture("hip-direct", 777, "clean-workflow"))
        _write(
            clean_out / "rank-scenarios" / "all" / "review_report.json",
            {
                "promotable_autotune_entries": [],
                "groups": [
                    {
                        "semantics": "bounded_i64",
                        "shape": {"m": 64, "n": 64, "k": 64},
                        "missing_required_baselines": [],
                        "candidates": [],
                    }
                ],
            },
        )
        clean_lines = summary.build_summary(clean_out)
        assert "FAILED_CAPTURES 0" in clean_lines
        assert "CAPTURE_JSON_COUNT 2" in clean_lines
        assert "CHECKSUM_MISMATCH_GROUPS 0" in clean_lines
        assert "REVIEW_REPORTS 1" in clean_lines
        assert "MISSING_REQUIRED_BASELINE_GROUPS 0" in clean_lines
        assert summary.clean_gate_failures(clean_lines) == []

    print("benchmark sweep failure summary self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
