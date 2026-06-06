#!/usr/bin/env python3
"""Self-test HIP Graph replay reporting."""

from __future__ import annotations

import copy

import hip_graph_replay_report


def capture(*, graph: bool, median_us: float, setup_us: float, name: str, checksum: int = 1234) -> dict:
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
        "contract_prefix_policy": "fixed_requested_residue_chain",
        "exact_wide_limb_count": None,
        "residue_chain_length": 3,
        "residue_output_mode": "residue_current_rns",
        "seed": 20260606,
        "warmups": 3,
        "repeats": 9,
        "backend_selected": "hip-direct",
        "checksum_u64": checksum,
        "prepack_setup_us": int(setup_us),
        "avg_prepack_setup_us": float(setup_us),
        "requested_next_op": {"requested": "rns-gemm", "resolved": "rns-gemm", "source": "cli"},
        "exact_output_contract": {
            "limb_count": None,
            "output_domain_after_measured_repeats": "rns_residue_current",
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
            "benchmark_execution_mode": "hip_graph_replay_resident_rns_chain" if graph else "residue_current_rns_chain",
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

    print("hip graph replay report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
