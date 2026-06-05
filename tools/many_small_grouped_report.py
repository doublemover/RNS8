#!/usr/bin/env python3
"""Compare many-small independent, host-batch, and grouped-dispatch captures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "many-small-grouped-reports"


def _median(capture: dict[str, Any]) -> float | None:
    summary = capture.get("timing_summary_us")
    if isinstance(summary, dict) and isinstance(summary.get("end_to_end"), dict):
        value = summary["end_to_end"].get("median")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def row_for_capture(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    grouped = capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    host_batch = capture.get("host_api_batch") if isinstance(capture.get("host_api_batch"), dict) else {}
    task_count = grouped.get("task_count") if grouped.get("requested") else host_batch.get("batch_size", 1)
    median = _median(capture)
    return {
        "path": str(path),
        "mode": (
            "grouped_dispatch"
            if grouped.get("requested")
            else "host_api_batch"
            if host_batch.get("enabled")
            else "independent_call"
        ),
        "task_count": task_count,
        "backend_selected": capture.get("backend_selected"),
        "semantics": capture.get("semantics"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "median_end_to_end_us": median,
        "median_per_task_end_to_end_us": (median / task_count) if median and task_count else None,
        "grouped_dispatch_status": grouped.get("capture_status"),
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows = [row_for_capture(path) for path in paths]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        shape = row["shape"]
        groups[(row["semantics"], shape["m"], shape["n"], shape["k"])].append(row)
    return {
        "schema_version": 1,
        "policy": "many_small_grouped_evidence_only_no_public_resident_lifetime_api",
        "groups": [
            {"key": key, "rows": sorted(value, key=lambda row: row.get("median_per_task_end_to_end_us") or float("inf"))}
            for key, value in sorted(groups.items(), key=lambda item: str(item[0]))
        ],
    }


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "many-small-grouped-report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.captures)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(write_report(report, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
