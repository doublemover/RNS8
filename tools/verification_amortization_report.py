#!/usr/bin/env python3
"""Gate verification-amortized exact workload captures against CPU/reference coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from review_reference import ReviewIndex, load_review_index
from review_contract import REVIEW_SCHEMA_VERSION, FINAL_COMPARISON_STATUSES, COMPLETE_COMPARISON_STATUSES

from benchmark_schema import BenchmarkSchemaError
from report_capture_inputs import load_report_capture


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "temp" / "verification-amortization-reports"
CPU_REFERENCE_BACKENDS = {"cpu", "cpu-reference"}




def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _json_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("*.json")))
        elif item.exists():
            paths.append(item)
    return paths


def _load_captures(inputs: list[Path]) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    captures: list[tuple[Path, dict[str, Any]]] = []
    skipped: list[str] = []
    for path in _json_inputs(inputs):
        if path.name.endswith(".failed.json") or path.name in {
            "review_report.json",
            "scenario_manifest.json",
            "command-plan.json",
            "validation-summary.json",
            "verification-amortization-report.json",
        }:
            continue
        try:
            capture = load_report_capture(path, from_directory=False)
        except BenchmarkSchemaError as exc:
            skipped.append(f"{_relative(path)}: {exc}")
            continue
        amortization = capture.get("verification_amortization")
        if isinstance(amortization, dict) and amortization.get("enabled") is True:
            captures.append((path, capture))
    return captures, skipped






def _scenario_scope(capture: dict[str, Any]) -> str | None:
    scenario = capture.get("scenario_metadata")
    if not isinstance(scenario, dict):
        return None
    eligibility = scenario.get("promotion_eligibility")
    if isinstance(eligibility, str) and eligibility:
        return eligibility
    metadata = scenario.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("promotion_scope"), str):
        return metadata["promotion_scope"]
    return None


def _policy_blockers(capture: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    amortization = capture.get("verification_amortization")
    if not isinstance(amortization, dict):
        return ["verification_amortization_missing"]
    if amortization.get("enabled") is not True:
        blockers.append("verification_amortization_not_enabled")
    if amortization.get("final_exact_comparison_required") is not True:
        blockers.append("final_exact_comparison_not_required")
    status = amortization.get("final_exact_comparison_status")
    if status not in FINAL_COMPARISON_STATUSES:
        blockers.append("final_exact_comparison_status_missing_or_unknown")
    if amortization.get("promotion_eligible") is not False:
        blockers.append("verification_amortization_promotable")
    scope = _scenario_scope(capture)
    if scope in {None, "release_review_candidate"}:
        blockers.append("verification_amortization_scenario_not_tooling_only")
    return sorted(set(blockers))


def _row_blockers(capture: dict[str, Any], review: dict[str, Any] | None) -> list[str]:
    blockers = _policy_blockers(capture)
    if review is None:
        blockers.append("review_report_missing_for_amortized_capture")
    else:
        if review.get("review_schema_version") != REVIEW_SCHEMA_VERSION:
            blockers.append("review_schema_version_mismatch")
        if review.get("cpu_reference_present") is not True:
            blockers.append("cpu_reference_baseline_missing")
        if review.get("review_mode") == "release" and review.get("cpu_reference_release_review_capture") is not True:
            blockers.append("cpu_reference_release_review_missing")
        missing = review.get("missing_required_baselines") or []
        if any(str(item) in CPU_REFERENCE_BACKENDS for item in missing):
            blockers.append("cpu_reference_marked_missing_required_baseline")
        duplicates = review.get("duplicate_backends") or []
        if duplicates:
            blockers.append("duplicate_backend_records")
        if review.get("candidate_promotable") is True:
            blockers.append("amortized_capture_promotable_in_review")
        if review.get("candidate_cache_write_status") in {"eligible_after_review", "written", "pending"}:
            blockers.append("amortized_capture_cache_write_eligible")
    policy = capture.get("verification_amortization") or {}
    if policy.get("final_exact_comparison_status") not in COMPLETE_COMPARISON_STATUSES:
        blockers.append("final_exact_comparison_pending")
    if review is not None:
        blockers.extend(review.get("review_reader_blockers") or [])
    return sorted(set(blockers))


def _row_for_capture(path: Path, capture: dict[str, Any], review_index: ReviewIndex) -> dict[str, Any]:
    review = review_index.lookup(path, capture)
    amortization = capture.get("verification_amortization")
    if not isinstance(amortization, dict):
        amortization = {}
    blockers = _row_blockers(capture, review)
    numerical = review.get("numerical_reference") if review else None
    reference_passed = isinstance(numerical, dict) and numerical.get("passed") is True and not review.get("review_reader_blockers")
    return {
        "capture_path": _relative(path),
        "backend": capture.get("backend_selected"),
        "semantics": capture.get("semantics"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "policy": amortization.get("policy"),
        "reused_reference_structure": amortization.get("reused_reference_structure"),
        "final_exact_comparison_required": amortization.get("final_exact_comparison_required"),
        "final_exact_comparison_status": amortization.get("final_exact_comparison_status"),
        "promotion_eligible": amortization.get("promotion_eligible"),
        "scenario_promotion_scope": _scenario_scope(capture),
        "review_report": review.get("review_report") if isinstance(review, dict) else None,
        "review_mode": review.get("review_mode") if isinstance(review, dict) else None,
        "review_group_index": review.get("review_report_group_index") if isinstance(review, dict) else None,
        "review_cpu_reference_present": review.get("cpu_reference_present") if isinstance(review, dict) else None,
        "review_cpu_reference_release_review_capture": review.get("cpu_reference_release_review_capture")
        if isinstance(review, dict)
        else None,
        "review_candidate_backends": review.get("candidate_backends") if isinstance(review, dict) else [],
        "review_missing_required_baselines": review.get("missing_required_baselines") if isinstance(review, dict) else [],
        "review_duplicate_backends": review.get("duplicate_backends") if isinstance(review, dict) else [],
        "review_candidate_promotable": review.get("candidate_promotable") if isinstance(review, dict) else None,
        "review_candidate_cache_write_status": review.get("candidate_cache_write_status") if isinstance(review, dict) else None,
        "policy_metadata_valid": not _policy_blockers(capture),
        "reference_coverage_present": numerical.get("reference_coverage_present") is True if isinstance(numerical, dict) else False,
        "comparison_passed": reference_passed and amortization.get("final_exact_comparison_status") in COMPLETE_COMPARISON_STATUSES,
        "promotion_prohibited": amortization.get("promotion_eligible") is False,
        "numerical_reference": numerical,
        "ready": not blockers,
        "blockers": blockers,
    }


def build_report(inputs: list[Path]) -> dict[str, Any]:
    captures, skipped = _load_captures(inputs)
    review_index = load_review_index(inputs, captures)
    rows = [_row_for_capture(path, capture, review_index) for path, capture in captures]
    blocker_counts = Counter(blocker for row in rows for blocker in row["blockers"])
    policy_counts = Counter(str(row.get("policy")) for row in rows)
    return {
        "schema": "rns8_verification_amortization_report_v1",
        "policy": "tooling_only_final_exact_cpu_reference_required",
        "capture_count": len(rows),
        "ready_count": sum(1 for row in rows if row["ready"]),
        "blocked_count": sum(1 for row in rows if not row["ready"]),
        "rank38_gate_complete": bool(rows) and all(row["ready"] for row in rows),
        "policy_counts": dict(sorted(policy_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "skipped_json_count": len(skipped),
        "skipped_json": skipped[:20],
        "rows": rows,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "verification-amortization-report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "verification-amortization-report.md"
    lines = [
        "# Verification Amortization Report",
        "",
        f"- Gate complete: `{report['rank38_gate_complete']}`.",
        f"- Captures: `{report['capture_count']}`.",
        f"- Ready: `{report['ready_count']}`.",
        f"- Blocked: `{report['blocked_count']}`.",
        "",
        "| Capture | Backend | Policy | CPU Reference | Cache Status | Ready | Blockers |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| `{capture}` | `{backend}` | `{policy}` | `{cpu}` | `{cache}` | `{ready}` | `{blockers}` |".format(
                capture=row["capture_path"],
                backend=row["backend"],
                policy=row["policy"],
                cpu=row["review_cpu_reference_present"],
                cache=row["review_candidate_cache_write_status"],
                ready=row["ready"],
                blockers=", ".join(row["blockers"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="capture files or directories containing captures")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.inputs)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    if args.require_complete and not report["rank38_gate_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
