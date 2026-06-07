#!/usr/bin/env python3
"""Self-test strict wrap64 Direct-HIP tuning report gates."""

from __future__ import annotations

import copy

import wrap64_direct_hip_tuning_report as report
from test_benchmark_schema_support import as_hip_graph_full_wrap64_capture, as_wrap64_rocwmma_candidate_capture
from test_benchmark_sweep_support import mark_reused_pack, wrap64_capture


def with_shape(capture: dict, size: int) -> dict:
    item = copy.deepcopy(capture)
    item["m"] = size
    item["n"] = size
    item["k"] = size
    item["output_logical_ld"] = size
    return item


def mark_k_block_candidate(capture: dict, name: str) -> dict:
    item = copy.deepcopy(capture)
    item["_path"] = f"{name}.json"
    item["scenario_metadata"] = {
        **item.get("scenario_metadata", {}),
        "promotion_eligibility": "tile_shape_evidence_only",
        "k_block_policy": "fixed-safe-kblock-cdna-candidate",
    }
    item["tile_shape_variant"] = {
        "name": name,
        "tile_m": item["tile_m"],
        "tile_n": item["tile_n"],
        "tile_k": item["k"],
        "k_block_policy": "fixed-safe-kblock-cdna-candidate",
        "split_k_mode": "single_gpu_no_split_k",
        "accumulator_safety_key": "k_block_size=1024;k_block_cap=4096;safe_for_k_block=true",
        "selected_kernel_identity": item["selected_kernel"],
        "resource_report_key": f"tile_m={item['tile_m']};tile_n={item['tile_n']};tile_k={item['k']};kernel={item['selected_kernel']}",
        "shape_family_bucket": "medium",
        "resource_report_required": "isa_or_counter_for_non_default_k_block_policy",
        "stale_kernel_rejection": "selected_kernel_identity_must_match_capture",
    }
    return item


def main() -> int:
    cpu = with_shape(wrap64_capture("wrap64-byte-limb", 10000), 512)
    direct = with_shape(wrap64_capture("hip-direct", 2000), 512)
    reuse = mark_reused_pack(with_shape(wrap64_capture("hip-direct", 1600), 512))
    reuse["_path"] = "reuse.json"
    graph = as_hip_graph_full_wrap64_capture(direct)
    graph["_path"] = "graph.json"
    graph["raw_timings_us"]["end_to_end"] = [1200] * graph["repeats"]
    graph["timing_summary_us"]["end_to_end"]["median"] = 1200
    graph["avg_end_to_end_us"] = 1200.0
    matrix_candidate = with_shape(as_wrap64_rocwmma_candidate_capture(direct), 512)
    matrix_candidate["_path"] = "rocwmma-candidate.json"
    matrix_candidate["raw_timings_us"]["end_to_end"] = [2500] * matrix_candidate["repeats"]
    matrix_candidate["timing_summary_us"]["end_to_end"]["median"] = 2500
    matrix_candidate["avg_end_to_end_us"] = 2500.0
    k_block = mark_k_block_candidate(with_shape(wrap64_capture("hip-direct", 1500), 512), "cdna-wrap64-512-kblock")

    result = report.build_report_from_captures([cpu, direct, reuse, graph, matrix_candidate, k_block])
    summary = result["summary"]
    assert result["schema"] == "rns8_wrap64_direct_hip_tuning_report_v1"
    assert summary["capture_count"] == 6
    assert summary["group_count"] == 1
    assert summary["candidate_workload_wins"] == 3
    assert summary["deprioritized"] == 1
    rows = result["groups"][0]["comparisons"]
    by_role = {item["role"]: item for item in rows}
    assert by_role["direct_hip_v4_baseline"]["decision"] == "baseline"
    assert by_role["direct_hip_reuse_candidate"]["decision"] == "candidate_workload_win"
    assert by_role["hip_graph_full_path"]["decision"] == "candidate_workload_win"
    assert by_role["direct_hip_k_block_candidate"]["decision"] == "candidate_workload_win"
    assert by_role["direct_hip_k_block_candidate"]["k_block_policy"] == "fixed-safe-kblock-cdna-candidate"
    assert "single_gpu_k_block_candidate_only" in by_role["direct_hip_k_block_candidate"]["blockers"]
    assert "benchmark_only_graph_replay" in by_role["hip_graph_full_path"]["blockers"]
    assert by_role["matrix_engine_candidate"]["decision"] == "drop/deprioritize"
    assert "matrix_engine_candidate_not_default_route" in by_role["matrix_engine_candidate"]["blockers"]

    missing_direct = report.build_report_from_captures([cpu, reuse])
    missing_row = missing_direct["groups"][0]["comparisons"][1]
    assert missing_direct["summary"]["missing_direct_baseline_groups"] == 1
    assert missing_row["decision"] == "keep_experimental"
    assert "missing_direct_hip_v4_baseline" in missing_row["critical_blockers"]

    bad_checksum = copy.deepcopy(reuse)
    bad_checksum["checksum_u64"] = int(bad_checksum["checksum_u64"]) + 1
    mismatch = report.build_report_from_captures([cpu, direct, bad_checksum])
    mismatch_row = next(item for item in mismatch["groups"][0]["comparisons"] if item["role"] == "direct_hip_reuse_candidate")
    assert mismatch_row["decision"] == "keep_experimental"
    assert "checksum_mismatch_vs_wrap64_reference" in mismatch_row["critical_blockers"]

    print("wrap64 Direct-HIP tuning report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
