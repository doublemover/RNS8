#!/usr/bin/env python3
"""Compare two rns8-bench JSON result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TIMING_KEYS = [
    "avg_pack_us",
    "avg_rns_gemm_us",
    "avg_per_modulus_gemm_estimate_us",
    "avg_crt_export_us",
    "avg_end_to_end_us",
]
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


def number(data: dict[str, Any], key: str, path: Path) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)):
        raise SystemExit(f"{path}: missing numeric field {key}")
    return float(value)


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
    for key in TIMING_KEYS:
        base = number(baseline, key, baseline_path)
        cand = number(candidate, key, candidate_path)
        timings[key] = {
            "baseline": base,
            "candidate": cand,
            "delta": cand - base,
            "ratio": cand / base if base != 0 else None,
        }

    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "matching_contract": all(item["match"] for item in contract.values()),
        "contract": contract,
        "timings": timings,
    }


def print_human(report: dict[str, Any]) -> None:
    print("RNS8 benchmark comparison")
    print("=========================")
    print(f"baseline:  {report['baseline']}")
    print(f"candidate: {report['candidate']}")
    print(f"matching contract: {report['matching_contract']}")
    print()
    print("Contract")
    for key, item in report["contract"].items():
        status = "OK" if item["match"] else "DIFF"
        print(f"[{status}] {key}: {item['baseline']} -> {item['candidate']}")
    print()
    print("Timings")
    for key, item in report["timings"].items():
        ratio = item["ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6g}"
        print(
            f"{key}: baseline={item['baseline']:.6g} "
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
