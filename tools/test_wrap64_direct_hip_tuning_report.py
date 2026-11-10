#!/usr/bin/env python3
"""Self-test strict wrap64 Direct-HIP tuning report gates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from itertools import permutations

import wrap64_direct_hip_tuning_report as report
from test_benchmark_schema_support_chains_wrap64 import (
    as_hip_graph_full_wrap64_capture,
    as_wrap64_rocwmma_candidate_capture,
)
from test_benchmark_sweep_support import mark_reused_pack, wrap64_capture
from test_diagnostic_companion import authored_capture_pair
from test_review_reference_support import add_synthetic_full_reference
from benchmark_schema import load_capture, validate_capture
from diagnostic_companion import build_binding


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
    k_block = mark_k_block_candidate(
        with_shape(wrap64_capture("hip-direct", 1500), 512), "cdna-wrap64-512-kblock"
    )

    result = report.build_report_from_captures(
        [cpu, direct, reuse, graph, matrix_candidate, k_block]
    )
    summary = result["summary"]
    assert result["schema"] == "rns8_wrap64_direct_hip_tuning_report_v1"
    assert summary["capture_count"] == 6
    assert summary["group_count"] >= 1
    # Single diagnostic records cannot establish production performance wins.
    assert summary["candidate_workload_wins"] == 0
    assert summary["deprioritized"] == 0
    rows = [row for group in result["groups"] for row in group["comparisons"]]
    by_role = {item["role"]: item for item in rows}
    assert by_role["direct_hip_v4_baseline"]["decision"] == "baseline"
    for role in (
        "direct_hip_reuse_candidate",
        "hip_graph_full_path",
        "direct_hip_k_block_candidate",
    ):
        assert by_role[role]["decision"] == "keep_experimental"
        assert by_role[role]["speedup_vs_direct_hip_v4"] is None
    assert (
        by_role["direct_hip_k_block_candidate"]["k_block_policy"]
        == "fixed-safe-kblock-cdna-candidate"
    )
    assert (
        "single_gpu_k_block_candidate_only" in by_role["direct_hip_k_block_candidate"]["blockers"]
    )
    assert "benchmark_only_graph_replay" in by_role["hip_graph_full_path"]["blockers"]
    assert by_role["matrix_engine_candidate"]["decision"] == "keep_experimental"
    assert (
        "matrix_engine_candidate_not_default_route"
        in by_role["matrix_engine_candidate"]["blockers"]
    )

    missing_direct = report.build_report_from_captures([cpu, reuse])
    missing_row = next(
        row
        for group in missing_direct["groups"]
        for row in group["comparisons"]
        if row["role"] == "direct_hip_reuse_candidate"
    )
    assert missing_direct["summary"]["missing_direct_baseline_groups"] >= 1
    assert missing_row["decision"] == "keep_experimental"
    assert "missing_direct_hip_v4_baseline" in missing_row["critical_blockers"]

    bad_checksum = copy.deepcopy(reuse)
    bad_checksum["checksum_u64"] = int(bad_checksum["checksum_u64"]) + 1
    mismatch = report.build_report_from_captures([cpu, direct, bad_checksum])
    mismatch_row = next(
        item
        for group in mismatch["groups"]
        for item in group["comparisons"]
        if item["role"] == "direct_hip_reuse_candidate"
    )
    assert mismatch_row["decision"] == "keep_experimental"
    assert "checksum_mismatch_vs_wrap64_reference" in mismatch_row["critical_blockers"]

    production_cohort_controls()

    print("wrap64 Direct-HIP tuning report self-test: PASS")
    return 0


def production_cohort_controls():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        captures, bindings, binding_paths = [], {}, []

        def collect(backend, costs, label=None):
            records = []
            for run, cost in enumerate(costs):
                lane = root / f"{label or backend}-{run}"
                lane.mkdir()
                fixture = add_synthetic_full_reference(wrap64_capture(backend, cost))
                _, _, paths = authored_capture_pair(lane, fixture)
                capture = load_capture(paths[0])
                validate_capture(capture, paths[0])
                capture["_path"] = str(paths[0])
                records.append(capture)
                if backend != "wrap64-byte-limb":
                    binding = build_binding(*paths)
                    bindings[str(paths[0].resolve())] = binding
                    target = lane / "binding.json"
                    target.write_text(json.dumps(binding), encoding="utf-8")
                    binding_paths.append(target)
            return records

        for backend, costs in (
            ("wrap64-byte-limb", (2000, 2000)),
            ("hip-direct", (1000, 1030)),
            ("rocwmma-wrap64-candidate", (700, 710)),
        ):
            captures.extend(collect(backend, costs))

        def candidate_row(value):
            return next(
                row
                for group in value["groups"]
                for row in group["comparisons"]
                if row["role"] == "matrix_engine_candidate"
            )

        def build(items, bindings=bindings):
            return report.build_report_from_captures(items, diagnostic_bindings=bindings)

        expected = candidate_row(build(captures))
        assert expected["decision"] == "candidate_workload_win", expected["critical_blockers"]
        assert expected["measurement_cohort"]["independent_runs"] == 2
        assert expected["median_end_to_end_us"] == 705
        assert expected["direct_hip_v4_median_end_to_end_us"] == 1015
        assert expected["numerical_reference"]["passed"]
        cpu_single = candidate_row(build(captures[1:]))
        assert cpu_single["numerical_reference"]["passed"]
        assert not cpu_single["wrap64_reference_timing_qualified"]
        assert cpu_single["wrap64_reference_median_end_to_end_us"] is None
        # Exhaust every ordering of the three two-process implementation blocks.
        blocks = [captures[:2], captures[2:4], captures[4:]]
        for ordering in permutations(blocks):
            ordered = [item for block in ordering for item in reversed(block)]
            assert candidate_row(build(ordered)) == expected
        repeated = candidate_row(build(captures + [copy.deepcopy(captures[4])]))
        assert repeated["measurement_cohort"]["independent_runs"] == 2
        assert repeated["median_end_to_end_us"] == expected["median_end_to_end_us"]
        singles = candidate_row(build([captures[0], captures[2], captures[4]]))
        assert singles["decision"] == "keep_experimental"
        assert singles["speedup_vs_direct_hip_v4"] is None
        duplicated = candidate_row(build(captures[:4] + [captures[4], copy.deepcopy(captures[4])]))
        assert duplicated["measurement_cohort"]["independent_runs"] == 1
        assert "insufficient_independent_processes" in duplicated["critical_blockers"]
        assert duplicated["speedup_vs_direct_hip_v4"] is None
        for field in ("device", "seed", "input_distribution"):
            changed = copy.deepcopy(captures)
            for candidate in changed[4:]:
                if field == "device":
                    candidate["device"]["name"] += " different synthetic device"
                elif field == "seed":
                    candidate["seed"] += 1
                else:
                    candidate["input_distribution"] += "-changed"
            row = candidate_row(build(changed))
            assert row["decision"] == "keep_experimental"
            assert row["speedup_vs_direct_hip_v4"] is None
            assert "missing_direct_hip_v4_baseline" in row["critical_blockers"]
        # The second candidate process needs its own current companion, too.
        missing = dict(bindings)
        missing.pop(str(Path(captures[5]["_path"]).resolve()))
        row = candidate_row(build(captures, missing))
        assert "matched_diagnostic_companion_missing" in row["critical_blockers"]
        actual = report.build_report([Path(c["_path"]) for c in captures], binding_paths)
        assert candidate_row(actual) == expected
        unstable_direct = collect("hip-direct", (1100, 700), "unstable-direct")
        unstable = captures[:2] + unstable_direct + captures[4:]
        blocks = [unstable[:2], unstable[2:4], unstable[4:]]
        for ordering in permutations(blocks):
            row = candidate_row(build([item for block in ordering for item in block]))
            assert row["decision"] == "keep_experimental"
            assert row["speedup_vs_direct_hip_v4"] is None
        renamed, renamed_bindings = [], {}
        for index, original in enumerate(captures):
            target = root / f"renamed-{len(captures) - index}.json"
            target.write_text(
                json.dumps(
                    {key: value for key, value in original.items() if not key.startswith("_")}
                ),
                encoding="utf-8",
            )
            renamed.append(load_capture(target))
            renamed[-1]["_path"] = str(target)
            binding = bindings.get(str(Path(original["_path"]).resolve()))
            if binding:
                renamed_bindings[str(target.resolve())] = build_binding(
                    target, binding["diagnostic_capture"]
                )
        row = candidate_row(build(renamed, renamed_bindings))
        for field in (
            "decision",
            "speedup_vs_direct_hip_v4",
            "median_end_to_end_us",
            "direct_hip_v4_median_end_to_end_us",
        ):
            assert row[field] == expected[field]


if __name__ == "__main__":
    raise SystemExit(main())
