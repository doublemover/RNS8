#!/usr/bin/env python3
"""Self-test HIP Graph replay reporting."""

from __future__ import annotations

import copy

import hip_graph_replay_report


def capture(
    *,
    graph: bool,
    median_us: float,
    setup_us: float | None,
    name: str,
    checksum: int = 1234,
    full_path: bool = False,
) -> dict:
    execution_mode = "hip_graph_replay_bounded_pack_gemm_export" if full_path and graph else (
        "hip_graph_replay_resident_rns_chain" if graph else ("persistent_resident_matrices" if full_path else "residue_current_rns_chain")
    )
    result = {
        "_path": f"{name}.json",
        "semantics": "bounded_i64",
        "m": 512,
        "n": 512,
        "k": 512,
        "tile_m": 128,
        "tile_n": 128,
        "bound_kind": "global_max_abs",
        "bound_mode": "global",
        "bound_source": "static_profile",
        "input_distribution": "signed_uniform_-16_16",
        "selected_prefix": 9,
        "requested_max_prefix": 9,
        "contract_prefix_policy": "minimum_proven" if full_path else "fixed_requested_residue_chain",
        "exact_wide_limb_count": None,
        "residue_chain_length": 1 if full_path else 3,
        "residue_output_mode": "host_export" if full_path else "residue_current_rns",
        "seed": 20260606,
        "warmups": 3,
        "repeats": 9,
        "backend_selected": "hip-direct",
        "checksum_u64": checksum,
        "reuse_packed_inputs": not full_path,
        "prepack_setup_us": int(setup_us) if setup_us is not None else None,
        "requested_next_op": (
            {"requested": "auto", "resolved": "final-export", "source": "benchmark_default"}
            if full_path
            else {"requested": "rns-gemm", "resolved": "rns-gemm", "source": "cli"}
        ),
        "exact_output_contract": {
            "limb_count": None,
            "output_domain_after_measured_repeats": "native_i64_u64_host" if full_path else "rns_residue_current",
        },
        "timing_summary_us": {
            "end_to_end": {"median": median_us},
            "rns_gemm": {"median": median_us},
        },
        "scenario_metadata": {
            "family": "hip-graph-replay",
            "name": name,
            "promotion_eligibility": "hip_graph_replay_evidence_only",
            "metadata": {
                "workflow_name": "hip_graph_replay",
                "graph_role": "graph_replay_candidate" if graph else "same_contract_non_graph_baseline",
            },
        },
        "timing_metadata": {
            "benchmark_execution_mode": execution_mode,
            "gpu_event_timing": not graph,
            "gpu_event_timing_reason": "hip_graph_replay_wall_clock_only" if graph else "requested",
            "gpu_event_timing_status": "not_requested_graph_replay" if graph else "available",
            "gpu_event_timing_source": None if graph else "hipEventElapsedTime",
            "gpu_event_phase_order": None if graph else ["pack", "rns_gemm", "crt_export"],
        },
        "hip_graph_replay": {
            "requested": graph,
            "available": graph,
            "used": graph,
            "status": "available" if graph else "not_requested",
            "capture_status": "replayed" if graph else "not_requested",
            "capture_us": 90 if graph else 0,
            "instantiate_us": 99 if graph else 0,
        },
    }
    if setup_us is not None:
        result["avg_prepack_setup_us"] = float(setup_us)
    if full_path and graph:
        result["hip_graph_replay"]["scope"] = "direct_hip_bounded_pack_gemm_export"
    elif graph:
        result["hip_graph_replay"]["scope"] = "direct_hip_reused_inputs_residue_current_rns_chain"
    return result


def main() -> int:
    baseline = capture(graph=False, median_us=1000.0, setup_us=90.0, name="baseline")
    graph = capture(graph=True, median_us=900.0, setup_us=90.0, name="graph")
    report = hip_graph_replay_report.build_hip_graph_replay_report_from_captures([baseline, graph])
    summary = report["summary"]
    comparison = report["comparisons"][0]
    assert summary["comparison_count"] == 1
    assert summary["candidate_workload_wins"] == 1
    assert summary["missing_baselines"] == 0
    assert comparison["decision"] == "candidate_workload_win"
    assert comparison["baseline_setup_inclusive_per_repeat_us"] == 1010.0
    assert comparison["graph_setup_inclusive_per_repeat_us"] == 931.0
    assert comparison["graph_total_setup_us"] == 279.0
    assert comparison["baseline_total_setup_us"] == 90.0
    assert comparison["graph_setup_overhead_vs_baseline_us"] == 189.0
    assert comparison["steady_state_delta_us"] == 100.0
    assert comparison["break_even_repeat_count"] == 2
    assert comparison["declared_repeat_count"] == 9
    assert comparison["declared_repeats_meet_break_even"] is True
    assert comparison["checksum_match"] is True

    slow_graph = capture(graph=True, median_us=1200.0, setup_us=90.0, name="slow-graph")
    slow_report = hip_graph_replay_report.build_hip_graph_replay_report_from_captures([baseline, slow_graph])
    assert slow_report["comparisons"][0]["decision"] == "deprioritize"

    missing_report = hip_graph_replay_report.build_hip_graph_replay_report_from_captures([graph])
    assert missing_report["summary"]["missing_baselines"] == 1
    assert missing_report["comparisons"][0]["decision"] == "keep_experimental"

    mismatch = copy.deepcopy(graph)
    mismatch["checksum_u64"] = 999
    mismatch_report = hip_graph_replay_report.build_hip_graph_replay_report_from_captures([baseline, mismatch])
    assert mismatch_report["comparisons"][0]["decision"] == "keep_experimental"
    assert "checksum_mismatch" in mismatch_report["comparisons"][0]["blockers"]

    full_baseline = capture(graph=False, median_us=800.0, setup_us=None, name="full-baseline", full_path=True)
    full_graph = capture(graph=True, median_us=720.0, setup_us=None, name="full-graph", full_path=True)
    full_report = hip_graph_replay_report.build_hip_graph_replay_report_from_captures([full_baseline, full_graph])
    full_comparison = full_report["comparisons"][0]
    assert full_comparison["decision"] == "candidate_workload_win"
    assert full_comparison["graph_scope"] == "direct_hip_bounded_pack_gemm_export"
    assert full_comparison["baseline_prepack_setup_us"] == 0.0
    assert full_comparison["graph_prepack_setup_us"] == 0.0
    assert full_comparison["baseline_setup_inclusive_per_repeat_us"] == 800.0
    assert full_comparison["graph_setup_inclusive_per_repeat_us"] == 741.0
    assert full_comparison["break_even_repeat_count"] == 3

    wrap_baseline = capture(graph=False, median_us=700.0, setup_us=None, name="wrap-baseline", full_path=True)
    wrap_graph = capture(graph=True, median_us=650.0, setup_us=None, name="wrap-graph", full_path=True)
    for item in [wrap_baseline, wrap_graph]:
        item["semantics"] = "wrap_u64_mod_2_64"
        item["bound_kind"] = "none"
        item["input_distribution"] = "unsigned_uniform_u64"
        item["selected_prefix"] = 0
        item["requested_max_prefix"] = 0
        item["contract_prefix_policy"] = "semantic_specific_no_rns_prefix"
    wrap_graph["timing_metadata"]["benchmark_execution_mode"] = "hip_graph_replay_wrap64_pack_gemm_export"
    wrap_graph["hip_graph_replay"]["scope"] = "direct_hip_wrap64_pack_gemm_export"
    wrap_report = hip_graph_replay_report.build_hip_graph_replay_report_from_captures([wrap_baseline, wrap_graph])
    wrap_comparison = wrap_report["comparisons"][0]
    assert wrap_comparison["decision"] == "candidate_workload_win"
    assert wrap_comparison["graph_scope"] == "direct_hip_wrap64_pack_gemm_export"

    print("hip graph replay report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
