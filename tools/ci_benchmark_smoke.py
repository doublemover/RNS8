#!/usr/bin/env python3
"""Bounded non-promoting CI collection; never a numeric regression gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from benchmark_sweep_lib.cli import parse_args as parse_sweep_args
from benchmark_sweep_lib.commands import sweep_command_entries
from benchmark_sweep_lib.execution import execute_sweep_entries, validate_paths

POLICY = "non_promoting_benchmark_smoke_no_regression_or_performance_qualification"
EXPECTED_CAPTURES = 6


def sweep_arguments(bench: Path, out_root: Path) -> list[str]:
    return ["--bench", str(bench), "--out-root", str(out_root),
        "--case", "ci-smoke-512:512,512,512", "--backend", "cpu", "--backend", "hip-direct",
        "--semantics", "bounded-i64", "--semantics", "bounded-u64",
        "--semantics", "finite-u8-field", "--modulus", "251",
        "--warmups", "1", "--repeats", "3", "--seed", "20261024",
        "--review-mode", "smoke", "--capture-timeout-seconds", "180", "--progress"]


def collection_result(entries, capture_paths, stats: dict) -> dict:
    captures = validate_paths(capture_paths)
    completed = {str(path) for path in capture_paths}
    missing = sorted(str(entry.output) for entry in entries if str(entry.output) not in completed)
    return {"schema": "rns8_ci_benchmark_smoke_v1", "policy": POLICY,
        "promotion_eligible": False, "performance_regression_checked": False,
        "planned_captures": len(entries), "schema_valid_captures": len(captures),
        "missing_captures": missing, "execution": stats,
        "complete": not missing and len(captures) == len(entries) == EXPECTED_CAPTURES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="parse and serialize commands; never execute a benchmark")
    options = parser.parse_args(argv)
    args = parse_sweep_args(sweep_arguments(options.bench, options.out_root))
    entries = sweep_command_entries(args)
    if len(entries) != EXPECTED_CAPTURES:
        raise RuntimeError("smoke matrix drifted; review the bounded CI collection contract")
    args.out_root.mkdir(parents=True, exist_ok=True)
    plan = {"schema": "rns8_ci_benchmark_smoke_plan_v1", "policy": POLICY,
        "planned_captures": len(entries), "max_capture_seconds": args.capture_timeout_seconds,
        "commands": [{"name": e.name, "command": e.command, "output": str(e.output), "env": e.env} for e in entries]}
    (args.out_root / "smoke-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if options.plan_only:
        print(json.dumps(plan, indent=2)); return 0
    paths = []
    stats = execute_sweep_entries(entries, args, paths)
    report = collection_result(entries, paths, stats)
    (args.out_root / "smoke-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
