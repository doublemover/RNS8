#!/usr/bin/env python3
"""Summarize adaptive grouping, grouped dispatch, graph replay, and overlap evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "scheduler-overlap-reports"


def _load(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    capture["_path"] = str(path)
    return capture


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        capture = _load(path)
        grouped = capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
        adaptive = (
            capture.get("adaptive_grouped_scheduler")
            if isinstance(capture.get("adaptive_grouped_scheduler"), dict)
            else {}
        )
        graph = capture.get("hip_graph_replay") if isinstance(capture.get("hip_graph_replay"), dict) else {}
        overlap = capture.get("streaming_overlap") if isinstance(capture.get("streaming_overlap"), dict) else {}
        rows.append(
            {
                "capture_path": str(path),
                "backend": capture.get("backend_selected"),
                "semantics": capture.get("semantics"),
                "shape": [capture.get("m"), capture.get("n"), capture.get("k")],
                "grouped_task_count": grouped.get("task_count", 1),
                "grouped_status": grouped.get("capture_status"),
                "adaptive_grouped_requested": adaptive.get("requested", False),
                "adaptive_group_count": adaptive.get("group_count"),
                "adaptive_active_prefix_count": adaptive.get("active_prefix_count"),
                "adaptive_active_entry_count": adaptive.get("active_entry_count"),
                "adaptive_independent_launch_count_model": adaptive.get("independent_launch_count_model"),
                "adaptive_aggregate_launch_count_model": adaptive.get("aggregate_launch_count_model"),
                "adaptive_launch_reduction_ratio": adaptive.get("launch_reduction_ratio"),
                "adaptive_event_scope": adaptive.get("event_scope"),
                "adaptive_status": adaptive.get("capture_status"),
                "graph_requested": graph.get("requested", False),
                "graph_status": graph.get("capture_status"),
                "overlap_requested": overlap.get("requested", False),
                "overlap_status": overlap.get("capture_status"),
                "promotion_eligible": False,
            }
        )
    return {
        "schema": "rns8_scheduler_overlap_report_v1",
        "capture_count": len(rows),
        "rows": rows,
        "policy": "scheduler_graph_and_overlap_evidence_only_until_exact_equivalence_and_event_overlap_proof_pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    report = build_report(args.captures)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "scheduler-overlap-report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
