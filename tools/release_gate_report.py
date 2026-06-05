#!/usr/bin/env python3
"""Summarize release-gate and verification-amortization benchmark evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "release-gate-reports"


def _load(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    capture["_path"] = str(path)
    return capture


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blockers: dict[str, int] = {}
    for path in paths:
        capture = _load(path)
        gate = capture.get("release_gate") if isinstance(capture.get("release_gate"), dict) else {}
        amortization = (
            capture.get("verification_amortization")
            if isinstance(capture.get("verification_amortization"), dict)
            else {}
        )
        for blocker in gate.get("blockers", []) if isinstance(gate.get("blockers"), list) else []:
            blockers[blocker] = blockers.get(blocker, 0) + 1
        rows.append(
            {
                "capture_path": str(path),
                "backend": capture.get("backend_selected"),
                "semantics": capture.get("semantics"),
                "shape": [capture.get("m"), capture.get("n"), capture.get("k")],
                "gate": gate.get("name"),
                "review_status": gate.get("review_status"),
                "cache_eligible": gate.get("cache_eligible", False),
                "blockers": gate.get("blockers", []),
                "verification_policy": amortization.get("policy"),
                "final_exact_comparison_required": amortization.get("final_exact_comparison_required", True),
            }
        )
    return {
        "schema": "rns8_release_gate_report_v1",
        "capture_count": len(rows),
        "rows": rows,
        "blocker_counts": dict(sorted(blockers.items())),
        "policy": "release_gate_reports_are_review_inputs_not_raw_performance_claims",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    report = build_report(args.captures)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "release-gate-report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
