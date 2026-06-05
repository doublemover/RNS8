#!/usr/bin/env python3
"""Summarize resident lifetime and workspace arena benchmark evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "resident-workspace-reports"


def _load(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    capture["_path"] = str(path)
    return capture


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        capture = _load(path)
        resident = capture.get("resident_lifetime") if isinstance(capture.get("resident_lifetime"), dict) else {}
        arena = capture.get("workspace_arena") if isinstance(capture.get("workspace_arena"), dict) else {}
        rows.append(
            {
                "capture_path": str(path),
                "backend": capture.get("backend_selected"),
                "semantics": capture.get("semantics"),
                "shape": [capture.get("m"), capture.get("n"), capture.get("k")],
                "resident_enabled": resident.get("enabled", False),
                "output_domain": resident.get("output_domain"),
                "workspace_identity": resident.get("workspace_identity"),
                "arena_enabled": arena.get("enabled", False),
                "arena_size_bytes": arena.get("size_bytes"),
                "high_water_mark_bytes": arena.get("high_water_mark_bytes"),
                "suballocation_count": arena.get("suballocation_count"),
                "repeat_allocation_free": arena.get("measured_repeat_allocation_free"),
                "promotion_eligible": False,
            }
        )
    return {
        "schema": "rns8_resident_workspace_report_v1",
        "capture_count": len(rows),
        "rows": rows,
        "policy": "resident_lifetime_and_workspace_arena_evidence_only_until_stale_source_tests_and_allocation_proof_pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    report = build_report(args.captures)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "resident-workspace-report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
