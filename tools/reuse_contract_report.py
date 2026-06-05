#!/usr/bin/env python3
"""Build temp-only reuse contract reports from benchmark captures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "reuse-contract-reports"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _median_phase(capture: dict[str, Any], phase: str) -> float | None:
    summary = capture.get("timing_summary_us")
    if isinstance(summary, dict):
        item = summary.get(phase)
        if isinstance(item, dict):
            value = _number(item.get("median"))
            if value is not None:
                return value
    raw = capture.get("raw_timings_us")
    if isinstance(raw, dict) and isinstance(raw.get(phase), list):
        values = [_number(item) for item in raw[phase]]
        clean = [value for value in values if value is not None]
        return float(median(clean)) if clean else None
    return None


def _identity(capture: dict[str, Any], path: Path) -> dict[str, Any]:
    contract = capture.get("reuse_contract") if isinstance(capture.get("reuse_contract"), dict) else {}
    return {
        "path": str(path),
        "benchmark": capture.get("benchmark"),
        "backend_selected": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "semantics": capture.get("semantics"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "pack_mode": capture.get("pack_mode"),
        "residue_output_mode": capture.get("residue_output_mode"),
        "reuse_contract": contract,
        "median_end_to_end_us": _median_phase(capture, "end_to_end"),
        "median_pack_us": _median_phase(capture, "pack"),
        "median_gemm_us": _median_phase(capture, "rns_gemm"),
        "median_export_us": _median_phase(capture, "crt_export"),
    }


def _group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    shape = row["shape"]
    return (
        row.get("semantics"),
        shape.get("m"),
        shape.get("n"),
        shape.get("k"),
        row.get("backend_selected"),
        row.get("selected_kernel"),
        row.get("residue_output_mode"),
    )


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        capture = load_capture(path)
        validate_capture(capture, path)
        rows.append(_identity(capture, path))

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    groups = []
    for key, entries in sorted(grouped.items(), key=lambda item: str(item[0])):
        baseline = min(
            (entry for entry in entries if not entry.get("reuse_contract", {}).get("enabled")),
            key=lambda item: item.get("median_end_to_end_us") or float("inf"),
            default=None,
        )
        candidates = []
        for entry in entries:
            median_e2e = entry.get("median_end_to_end_us")
            baseline_e2e = baseline.get("median_end_to_end_us") if baseline else None
            speedup = (
                baseline_e2e / median_e2e
                if baseline_e2e and median_e2e and median_e2e > 0.0
                else None
            )
            setup_cost = _number(entry.get("reuse_contract", {}).get("setup_cost_us"))
            per_repeat_saved = (
                baseline_e2e - median_e2e
                if baseline_e2e is not None and median_e2e is not None
                else None
            )
            break_even = (
                math.ceil(setup_cost / per_repeat_saved)
                if setup_cost is not None and per_repeat_saved and per_repeat_saved > 0.0
                else None
            )
            candidates.append(
                {
                    **entry,
                    "steady_state_speedup_vs_group_baseline": speedup,
                    "setup_inclusive_break_even_repeats": break_even,
                }
            )
        groups.append({"key": key, "baseline_path": baseline.get("path") if baseline else None, "candidates": candidates})

    return {
        "schema_version": 1,
        "policy": "reuse_contract_evidence_only_not_autotune_promotion",
        "capture_count": len(rows),
        "groups": groups,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "reuse-contract-report.json"
    md_path = out_dir / "reuse-contract-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RNS8 Reuse Contract Report",
        "",
        f"- Captures: `{report['capture_count']}`",
        f"- Policy: `{report['policy']}`",
        "",
        "| group | candidate | median e2e us | speedup | break-even repeats |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for index, group in enumerate(report["groups"], start=1):
        for candidate in group["candidates"]:
            speedup = candidate.get("steady_state_speedup_vs_group_baseline")
            break_even = candidate.get("setup_inclusive_break_even_repeats")
            lines.append(
                "| {group} | `{path}` | {median} | {speedup} | {break_even} |".format(
                    group=index,
                    path=candidate.get("path"),
                    median=candidate.get("median_end_to_end_us"),
                    speedup=f"{speedup:.3f}" if isinstance(speedup, float) else "",
                    break_even=break_even if break_even is not None else "",
                )
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="schema-v4 benchmark JSON captures")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="ignored output directory")
    parser.add_argument("--json", action="store_true", help="print report JSON instead of writing files")
    args = parser.parse_args()
    report = build_report(args.captures)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        outputs = write_outputs(report, args.out_dir)
        for label, path in outputs.items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
