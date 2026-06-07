#!/usr/bin/env python3
"""Run a compact RNS8 cleanup regression suite and write one report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "temp" / "golden-regression-suite"


@dataclass(frozen=True)
class SuiteCase:
    name: str
    category: str
    command: list[str]


DEFAULT_CASES = [
    SuiteCase("metadata_registry", "metadata_drift", [sys.executable, "tools/test_metadata_registry.py"]),
    SuiteCase(
        "benchmark_schema_semantic_contracts",
        "metadata_drift",
        [sys.executable, "tools/test_benchmark_schema_semantic_contracts.py"],
    ),
    SuiteCase(
        "benchmark_schema_reuse_timing",
        "metadata_drift",
        [sys.executable, "tools/test_benchmark_schema_reuse_timing.py"],
    ),
    SuiteCase(
        "benchmark_schema_gpu_events",
        "metadata_drift",
        [sys.executable, "tools/test_benchmark_schema_gpu_events.py"],
    ),
    SuiteCase(
        "benchmark_schema_output_policies",
        "metadata_drift",
        [sys.executable, "tools/test_benchmark_schema_output_policies.py"],
    ),
    SuiteCase("benchmark_schema", "metadata_drift", [sys.executable, "tools/test_benchmark_schema.py"]),
    SuiteCase("benchmark_sweep", "scenario_drift", [sys.executable, "tools/test_benchmark_sweep.py"]),
    SuiteCase("result_compare", "correctness_report", [sys.executable, "tools/test_result_compare.py"]),
    SuiteCase("many_small_grouped_report", "grouped_dispatch", [sys.executable, "tools/test_many_small_grouped_report.py"]),
    SuiteCase("reuse_contract_report", "reuse_contract", [sys.executable, "tools/test_reuse_contract_report.py"]),
    SuiteCase(
        "direct_hip_reuse_expansion_report",
        "reuse_contract",
        [sys.executable, "tools/test_direct_hip_reuse_expansion_report.py"],
    ),
    SuiteCase(
        "wrap64_direct_hip_tuning_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_wrap64_direct_hip_tuning_report.py"],
    ),
    SuiteCase(
        "direct_hip_resident_redesign_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_direct_hip_resident_redesign_report.py"],
    ),
    SuiteCase(
        "adaptive_grouped_scheduler_report",
        "scheduler_overlap",
        [sys.executable, "tools/test_adaptive_grouped_scheduler_report.py"],
    ),
    SuiteCase(
        "streaming_overlap_report",
        "scheduler_overlap",
        [sys.executable, "tools/test_streaming_overlap_report.py"],
    ),
    SuiteCase(
        "modulus_set_search",
        "performance_evidence_drift",
        [sys.executable, "tools/test_modulus_set_search.py"],
    ),
    SuiteCase(
        "modulus_set_autotune_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_modulus_set_autotune_report.py"],
    ),
    SuiteCase("hip_graph_replay_report", "scheduler_overlap", [sys.executable, "tools/test_hip_graph_replay_report.py"]),
    SuiteCase("rns_chain_report", "residue_chain", [sys.executable, "tools/test_rns_chain_report.py"]),
    SuiteCase(
        "exact_wide_chain_report",
        "residue_chain",
        [sys.executable, "tools/test_exact_wide_chain_report.py"],
    ),
    SuiteCase("perf_variance_report", "performance_evidence_drift", [sys.executable, "tools/test_perf_variance_report.py"]),
    SuiteCase("gpu_counter_report", "performance_evidence_drift", [sys.executable, "tools/test_gpu_counter_report.py"]),
    SuiteCase("gpu_isa_report", "performance_evidence_drift", [sys.executable, "tools/test_gpu_isa_report.py"]),
    SuiteCase("target_validation_report", "target_readiness", [sys.executable, "tools/test_target_validation_report.py"]),
    SuiteCase("multigpu_shard_report", "target_readiness", [sys.executable, "tools/test_multigpu_shard_report.py"]),
    SuiteCase("cdna_scripts", "target_readiness", [sys.executable, "tools/test_cdna_scripts.py"]),
    SuiteCase(
        "bounded_i64_1024_review",
        "performance_evidence_drift",
        [sys.executable, "tools/test_bounded_i64_1024_review.py"],
    ),
    SuiteCase("tile_shape_report", "performance_evidence_drift", [sys.executable, "tools/test_tile_shape_report.py"]),
    SuiteCase(
        "zero_skip_expansion_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_zero_skip_expansion_report.py"],
    ),
    SuiteCase(
        "verification_amortization_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_verification_amortization_report.py"],
    ),
    SuiteCase(
        "error_detection_policy_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_error_detection_policy_report.py"],
    ),
    SuiteCase(
        "finite_modulus_map_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_finite_modulus_map_report.py"],
    ),
    SuiteCase(
        "finite_distribution_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_finite_distribution_report.py"],
    ),
    SuiteCase(
        "shape_family_shadow_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_shape_family_shadow_report.py"],
    ),
    SuiteCase(
        "auto_shape_family_gate",
        "performance_evidence_drift",
        [sys.executable, "tools/test_auto_shape_family_gate.py"],
    ),
    SuiteCase("export_selector_report", "performance_evidence_drift", [sys.executable, "tools/test_export_selector_report.py"]),
    SuiteCase(
        "reconstruction_export_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_reconstruction_export_report.py"],
    ),
    SuiteCase(
        "gfx1100_pending_validation",
        "performance_evidence_drift",
        [sys.executable, "tools/test_gfx1100_pending_validation.py"],
    ),
    SuiteCase(
        "layout_search_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_layout_search_report.py"],
    ),
    SuiteCase(
        "vector_to_rns_chain_report",
        "performance_evidence_drift",
        [sys.executable, "tools/test_vector_to_rns_chain_report.py"],
    ),
    SuiteCase("claim_validation", "documentation_claims", [sys.executable, "tools/test_claim_validation.py"]),
]


def run_case(case: SuiteCase) -> dict[str, Any]:
    result = subprocess.run(case.command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "name": case.name,
        "category": case.category,
        "command": case.command,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "output": result.stdout,
    }


def write_report(results: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for item in results if item["passed"]),
            "failed": sum(1 for item in results if not item["passed"]),
            "categories": sorted({str(item["category"]) for item in results}),
        },
    }
    json_path = out_dir / "golden-regression-report.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / "golden-regression-report.md"
    lines = [
        "# RNS8 Golden Regression Suite",
        "",
        "| Case | Category | Result |",
        "|---|---|---|",
    ]
    for item in results:
        result = "PASS" if item["passed"] else f"FAIL ({item['returncode']})"
        lines.append(f"| {item['name']} | {item['category']} | {result} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    results = [run_case(case) for case in DEFAULT_CASES]
    report_path = write_report(results, args.out_dir)
    for item in results:
        result = "PASS" if item["passed"] else f"FAIL ({item['returncode']})"
        print(f"{item['name']}: {result}")
    print(report_path)
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
