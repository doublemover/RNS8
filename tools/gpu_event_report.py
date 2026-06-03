#!/usr/bin/env python3
"""Validate an RNS8 benchmark capture and rank GPU event phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def phase_rows(capture: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = capture.get("timing_metadata")
    summary = capture.get("gpu_event_timing_summary_us")
    if not isinstance(metadata, dict) or not isinstance(summary, dict):
        return []
    phase_order = metadata.get("gpu_event_phase_order")
    if not isinstance(phase_order, list):
        return []

    medians: dict[str, float] = {}
    for phase in phase_order:
        item = summary.get(phase)
        if not isinstance(item, dict):
            continue
        median = _number(item.get("median"))
        if median is not None:
            medians[str(phase)] = median

    total = sum(medians.values())
    rows = []
    for phase in phase_order:
        phase_name = str(phase)
        item = summary.get(phase_name)
        if not isinstance(item, dict):
            continue
        median = medians.get(phase_name, 0.0)
        rows.append(
            {
                "phase": phase_name,
                "avg_us": _number(item.get("avg")),
                "median_us": median,
                "p95_us": _number(item.get("p95")),
                "share_of_declared_median_sum": (median / total) if total > 0.0 else 0.0,
            }
        )
    rows.sort(key=lambda row: float(row["median_us"] or 0.0), reverse=True)
    return rows


def report_for_capture(path: Path) -> dict[str, Any]:
    data = load_capture(path)
    validate_capture(data, path)
    metadata = data.get("timing_metadata")
    timing_enabled = isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True
    rows = phase_rows(data) if timing_enabled else []
    return {
        "path": str(path),
        "schema_version": data.get("schema_version"),
        "backend_requested": data.get("backend_requested"),
        "backend_selected": data.get("backend_selected"),
        "selected_kernel": data.get("selected_kernel"),
        "semantics": data.get("semantics"),
        "shape": {"m": data.get("m"), "n": data.get("n"), "k": data.get("k")},
        "repeats": data.get("repeats"),
        "gpu_event_timing": timing_enabled,
        "gpu_event_timing_status": metadata.get("gpu_event_timing_status") if isinstance(metadata, dict) else None,
        "gpu_event_timing_reason": metadata.get("gpu_event_timing_reason") if isinstance(metadata, dict) else None,
        "gpu_event_timing_source_scope": (
            metadata.get("gpu_event_timing_source_scope") if isinstance(metadata, dict) else None
        ),
        "ranked_phases": rows,
    }


def print_text_report(report: dict[str, Any], top: int) -> None:
    print(f"{report['path']}: schema v{report['schema_version']}")
    print(
        "backend: {backend} kernel: {kernel} semantics: {semantics} shape: {m}x{n}x{k} repeats: {repeats}".format(
            backend=report.get("backend_selected"),
            kernel=report.get("selected_kernel"),
            semantics=report.get("semantics"),
            m=report.get("shape", {}).get("m"),
            n=report.get("shape", {}).get("n"),
            k=report.get("shape", {}).get("k"),
            repeats=report.get("repeats"),
        )
    )
    if not report.get("gpu_event_timing"):
        print(
            "gpu events: unavailable "
            f"status={report.get('gpu_event_timing_status')} reason={report.get('gpu_event_timing_reason')}"
        )
        return
    print(f"gpu event scope: {report.get('gpu_event_timing_source_scope')}")
    print("rank | phase | median us | share | avg us | p95 us")
    print("---: | --- | ---: | ---: | ---: | ---:")
    for index, row in enumerate(report["ranked_phases"][:top], start=1):
        print(
            "{index} | {phase} | {median:.3f} | {share:.3%} | {avg:.3f} | {p95:.3f}".format(
                index=index,
                phase=row["phase"],
                median=float(row["median_us"] or 0.0),
                share=float(row["share_of_declared_median_sum"] or 0.0),
                avg=float(row["avg_us"] or 0.0),
                p95=float(row["p95_us"] or 0.0),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="schema-v4 benchmark JSON captures")
    parser.add_argument("--json", action="store_true", help="emit machine-readable reports")
    parser.add_argument("--top", type=int, default=20, help="number of ranked phases to print in text mode")
    parser.add_argument(
        "--fail-on-unavailable",
        action="store_true",
        help="return nonzero when a valid capture has no GPU event timing",
    )
    args = parser.parse_args()

    reports = []
    try:
        for path in args.captures:
            reports.append(report_for_capture(path))
    except BenchmarkSchemaError as exc:
        if args.json:
            print(json.dumps({"valid": False, "errors": str(exc).splitlines()}, indent=2, sort_keys=True))
        else:
            print(str(exc))
        return 1

    unavailable = [report for report in reports if not report.get("gpu_event_timing")]
    if args.json:
        print(json.dumps({"valid": True, "captures": reports}, indent=2, sort_keys=True))
    else:
        for index, report in enumerate(reports):
            if index:
                print()
            print_text_report(report, max(args.top, 0))
    return 1 if args.fail_on_unavailable and unavailable else 0


if __name__ == "__main__":
    raise SystemExit(main())
