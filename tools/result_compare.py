#!/usr/bin/env python3
"""Compare two rns8-bench JSON result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TIMING_PHASES = {
    "planning": ["avg_planning_us", "plan_us"],
    "matrix_alloc": ["avg_matrix_alloc_us", "matrix_alloc_us"],
    "pack": ["avg_pack_us"],
    "rns_gemm": ["avg_rns_gemm_us"],
    "per_modulus_gemm_estimate": ["avg_per_modulus_gemm_estimate_us"],
    "crt_export": ["avg_crt_export_us"],
    "end_to_end": ["avg_end_to_end_us"],
}
CONTRACT_KEYS = ["backend_selected", "semantics", "m", "n", "k", "prefix", "seed"]


def load_result(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path}: failed to read benchmark JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: benchmark JSON root must be an object")
    return data


def schema_version(data: dict[str, Any]) -> int:
    value = data.get("schema_version", 1)
    return value if isinstance(value, int) else 1


def phase_timing(data: dict[str, Any], phase: str, path: Path) -> tuple[float, str]:
    summary = data.get("timing_summary_us")
    if isinstance(summary, dict):
        phase_summary = summary.get(phase)
        if isinstance(phase_summary, dict):
            value = phase_summary.get("avg")
            if isinstance(value, (int, float)):
                return float(value), f"timing_summary_us.{phase}.avg"

    for key in TIMING_PHASES[phase]:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value), key

    keys = ", ".join([f"timing_summary_us.{phase}.avg", *TIMING_PHASES[phase]])
    raise SystemExit(f"{path}: missing numeric timing for phase {phase}; looked for {keys}")


def compare(baseline: dict[str, Any], candidate: dict[str, Any], baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    contract = {
        key: {
            "baseline": baseline.get(key),
            "candidate": candidate.get(key),
            "match": baseline.get(key) == candidate.get(key),
        }
        for key in CONTRACT_KEYS
    }
    timings = {}
    for phase in TIMING_PHASES:
        base, base_source = phase_timing(baseline, phase, baseline_path)
        cand, cand_source = phase_timing(candidate, phase, candidate_path)
        timings[phase] = {
            "baseline": base,
            "candidate": cand,
            "delta": cand - base,
            "ratio": cand / base if base != 0 else None,
            "baseline_source": base_source,
            "candidate_source": cand_source,
        }

    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "schema": {
            "baseline_version": schema_version(baseline),
            "candidate_version": schema_version(candidate),
        },
        "matching_contract": all(item["match"] for item in contract.values()),
        "contract": contract,
        "timings": timings,
    }


def print_human(report: dict[str, Any]) -> None:
    print("RNS8 benchmark comparison")
    print("=========================")
    print(f"baseline:  {report['baseline']}")
    print(f"candidate: {report['candidate']}")
    print(
        "schema:    "
        f"baseline=v{report['schema']['baseline_version']} "
        f"candidate=v{report['schema']['candidate_version']}"
    )
    print(f"matching contract: {report['matching_contract']}")
    print()
    print("Contract")
    for key, item in report["contract"].items():
        status = "OK" if item["match"] else "DIFF"
        print(f"[{status}] {key}: {item['baseline']} -> {item['candidate']}")
    print()
    print("Timings")
    for phase, item in report["timings"].items():
        ratio = item["ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6g}"
        print(
            f"{phase}: baseline={item['baseline']:.6g} "
            f"candidate={item['candidate']:.6g} delta={item['delta']:.6g} ratio={ratio_text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="baseline rns8-bench JSON file")
    parser.add_argument("candidate", type=Path, help="candidate rns8-bench JSON file")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    baseline = load_result(args.baseline)
    candidate = load_result(args.candidate)
    report = compare(baseline, candidate, args.baseline, args.candidate)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0 if report["matching_contract"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
