#!/usr/bin/env python3
"""Summarize resident lifetime and workspace arena benchmark evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from report_capture_inputs import load_report_captures


DEFAULT_OUT_DIR = Path("temp") / "resident-workspace-reports"


def _counter_nonzero(counter: Any) -> bool:
    return (
        isinstance(counter, dict)
        and (
            int(counter.get("allocate_calls") or 0) != 0
            or int(counter.get("free_calls") or 0) != 0
            or int(counter.get("allocated_bytes") or 0) != 0
        )
    )


def _target_id(capture: dict[str, Any]) -> str | None:
    for source in (capture.get("target_variant"), capture.get("device"), capture):
        if isinstance(source, dict) and isinstance(source.get("target_id"), str):
            return source["target_id"]
        if isinstance(source, dict) and isinstance(source.get("target_arch"), str):
            return source["target_arch"]
    return None


def arena_blockers(capture: dict[str, Any], arena: dict[str, Any], resident: dict[str, Any]) -> list[str]:
    blockers = ["resident_workspace_report_evidence_only"]
    if not arena.get("enabled", False):
        blockers.append("workspace_arena_not_enabled")
    if arena.get("measured_repeat_allocation_free") is not True:
        blockers.append("measured_repeat_allocation_not_proven_free")
    repeat_delta = arena.get("measured_repeat_allocation_delta")
    if repeat_delta is None:
        repeat_delta = (capture.get("device_allocation") or {}).get("measured_repeat_delta")
    if repeat_delta is None:
        blockers.append("missing_measured_repeat_allocation_delta")
    elif _counter_nonzero(repeat_delta):
        blockers.append("measured_repeat_allocation_delta_nonzero")
    if not resident.get("workspace_identity"):
        blockers.append("missing_resident_workspace_identity")
    correctness = capture.get("correctness")
    if correctness is None:
        blockers.append("correctness_not_recorded")
    elif correctness != "ok":
        blockers.append("correctness_not_ok")
    return blockers


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for capture in load_report_captures(paths):
        resident = capture.get("resident_lifetime") if isinstance(capture.get("resident_lifetime"), dict) else {}
        arena = capture.get("workspace_arena") if isinstance(capture.get("workspace_arena"), dict) else {}
        blockers = arena_blockers(capture, arena, resident)
        rows.append(
            {
                "capture_path": capture.get("_path"),
                "backend": capture.get("backend_selected"),
                "semantics": capture.get("semantics"),
                "target_id": _target_id(capture),
                "shape": [capture.get("m"), capture.get("n"), capture.get("k")],
                "resident_enabled": resident.get("enabled", False),
                "output_domain": resident.get("output_domain"),
                "workspace_identity": resident.get("workspace_identity"),
                "arena_enabled": arena.get("enabled", False),
                "arena_identity": arena.get("arena_identity"),
                "arena_size_bytes": arena.get("size_bytes"),
                "high_water_mark_bytes": arena.get("high_water_mark_bytes"),
                "suballocation_count": arena.get("suballocation_count"),
                "repeat_allocation_free": arena.get("measured_repeat_allocation_free"),
                "setup_allocation_delta": arena.get("setup_allocation_delta"),
                "measured_repeat_allocation_delta": arena.get("measured_repeat_allocation_delta")
                or (capture.get("device_allocation") or {}).get("measured_repeat_delta"),
                "promotion_eligible": False,
                "promotion_blockers": blockers,
            }
        )
    return {
        "schema": "rns8_resident_workspace_report_v1",
        "capture_count": len(rows),
        "arena_ready_count": sum(
            1
            for row in rows
            if row["arena_enabled"]
            and row["repeat_allocation_free"] is True
            and "measured_repeat_allocation_delta_nonzero" not in row["promotion_blockers"]
            and "missing_measured_repeat_allocation_delta" not in row["promotion_blockers"]
        ),
        "blocked_count": sum(1 for row in rows if len(row["promotion_blockers"]) > 1),
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
