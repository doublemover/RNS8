#!/usr/bin/env python3
"""Validate benchmark captures and assemble temp-only GPU counter reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


COUNTER_REPORT_POLICY = (
    "explanation_evidence_only_not_a_correctness_gate_not_a_performance_claim"
)
DEFAULT_OUT_DIR = Path("temp") / "gpu-counter-reports"
COUNTER_VALUE_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str) and COUNTER_VALUE_RE.match(value.strip()):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _capture_identity(capture: dict[str, Any], path: Path) -> dict[str, Any]:
    target = capture.get("target_variant")
    if not isinstance(target, dict):
        target = {}
    metadata = capture.get("timing_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "path": str(path),
        "schema_version": capture.get("schema_version"),
        "benchmark": capture.get("benchmark"),
        "benchmark_execution_mode": capture.get("benchmark_execution_mode"),
        "backend_requested": capture.get("backend_requested"),
        "backend_selected": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "semantics": capture.get("semantics"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "repeats": capture.get("repeats"),
        "target_id": target.get("target_id"),
        "target_namespace": target.get("target_namespace"),
        "review_group_key": target.get("review_group_key"),
        "pack_layout": metadata.get("pack_layout"),
        "fusion_mode": metadata.get("fusion_mode"),
        "generated_reducer_identity": metadata.get("generated_reducer_identity"),
    }


def _top_numeric_metrics(metrics: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = []
    for name, value in metrics.items():
        number = _number(value)
        if number is None:
            continue
        rows.append({"metric": str(name), "value": number})
    rows.sort(key=lambda row: abs(float(row["value"])), reverse=True)
    return rows[: max(limit, 0)]


def _json_counter_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("counters", "metrics", "records", "rows"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [value]
        return [raw]
    raise RuntimeError(f"unsupported JSON counter shape: {path}")


def _csv_counter_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [dict(row) for row in rows]


def load_counter_file(path: Path, top_limit: int) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"counter input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        records = _json_counter_records(path)
    elif suffix == ".csv":
        records = _csv_counter_records(path)
    else:
        raise RuntimeError(f"counter input must be .json or .csv: {path}")

    numeric_totals: dict[str, float] = {}
    numeric_counts: dict[str, int] = {}
    nonnumeric_keys: set[str] = set()
    for record in records:
        for key, value in record.items():
            number = _number(value)
            if number is None:
                nonnumeric_keys.add(str(key))
                continue
            numeric_totals[str(key)] = numeric_totals.get(str(key), 0.0) + number
            numeric_counts[str(key)] = numeric_counts.get(str(key), 0) + 1

    numeric_averages = {
        key: numeric_totals[key] / numeric_counts[key] for key in sorted(numeric_totals)
    }
    return {
        "path": str(path),
        "format": suffix.lstrip("."),
        "record_count": len(records),
        "numeric_metric_count": len(numeric_averages),
        "nonnumeric_keys": sorted(nonnumeric_keys - set(numeric_averages)),
        "numeric_metric_average": numeric_averages,
        "top_numeric_metrics_by_abs_average": _top_numeric_metrics(numeric_averages, top_limit),
    }


def load_isa_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"ISA summary does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"ISA summary must be a JSON object: {path}")
    totals = data.get("instruction_totals")
    if not isinstance(totals, dict):
        totals = {}
    return {
        "path": str(path),
        "object": data.get("object"),
        "backend": data.get("backend"),
        "target": data.get("target"),
        "reported_symbol_count": data.get("reported_symbol_count"),
        "instruction_totals": totals,
        "linked_capture": data.get("capture"),
    }


def report_for_capture(
    capture_path: Path,
    counter_paths: list[Path],
    isa_paths: list[Path],
    top_limit: int,
) -> dict[str, Any]:
    capture = load_capture(capture_path)
    validate_capture(capture, capture_path)
    counter_reports = [load_counter_file(path, top_limit) for path in counter_paths]
    isa_reports = [load_isa_summary(path) for path in isa_paths]
    return {
        "schema_version": 1,
        "capture": _capture_identity(capture, capture_path),
        "policy": COUNTER_REPORT_POLICY,
        "counter_inputs": counter_reports,
        "isa_summaries": isa_reports,
        "review_notes": [
            "Counters and ISA summaries explain likely bottlenecks but do not replace exact CPU reference checks.",
            "Use host timings and HIP event timings for release comparisons; use counters to choose the next kernel experiment.",
            "Raw profiler exports, ISA dumps, and generated reports stay under temp/ and are not committed.",
        ],
    }


def safe_report_stem(capture_path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", capture_path.stem).strip("._")
    if not stem:
        stem = f"capture-{index}"
    return stem


def write_json_report(report: dict[str, Any], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}-gpu-counter-report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def write_markdown_report(report: dict[str, Any], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}-gpu-counter-report.md"
    capture = report["capture"]
    lines = [
        "# RNS8 GPU Counter Report",
        "",
        f"- Capture: `{capture['path']}`",
        f"- Backend: `{capture['backend_selected']}`",
        f"- Kernel: `{capture['selected_kernel']}`",
        f"- Semantics: `{capture['semantics']}`",
        f"- Shape: `{capture['shape']['m']}x{capture['shape']['n']}x{capture['shape']['k']}`",
        f"- Target: `{capture.get('target_id')}` / `{capture.get('target_namespace')}`",
        f"- Policy: `{report['policy']}`",
        "",
        "## Counter Inputs",
        "",
    ]
    if report["counter_inputs"]:
        for item in report["counter_inputs"]:
            lines.extend(
                [
                    f"### `{item['path']}`",
                    "",
                    f"- Records: `{item['record_count']}`",
                    f"- Numeric metrics: `{item['numeric_metric_count']}`",
                    "",
                    "| metric | average |",
                    "| --- | ---: |",
                ]
            )
            for row in item["top_numeric_metrics_by_abs_average"]:
                lines.append(f"| `{row['metric']}` | {float(row['value']):.6g} |")
            lines.append("")
    else:
        lines.extend(["No counter exports were attached.", ""])

    lines.extend(["## ISA Summaries", ""])
    if report["isa_summaries"]:
        lines.extend(["| object | backend | target | symbols | WMMA | MFMA | stores | LDS | waits |"])
        lines.extend(["| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for item in report["isa_summaries"]:
            totals = item.get("instruction_totals") or {}
            lines.append(
                "| `{object}` | `{backend}` | `{target}` | {symbols} | {wmma} | {mfma} | {stores} | {lds} | {waits} |".format(
                    object=item.get("object"),
                    backend=item.get("backend"),
                    target=item.get("target"),
                    symbols=item.get("reported_symbol_count"),
                    wmma=totals.get("wmma"),
                    mfma=totals.get("mfma"),
                    stores=totals.get("global_store"),
                    lds=totals.get("lds_mentions"),
                    waits=totals.get("wait_instructions"),
                )
            )
        lines.append("")
    else:
        lines.extend(["No ISA summaries were attached.", ""])

    lines.extend(["## Review Notes", ""])
    for note in report["review_notes"]:
        lines.append(f"- {note}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="schema-v4 benchmark JSON captures")
    parser.add_argument("--counter", action="append", type=Path, default=[], help="CSV or JSON profiler counter export")
    parser.add_argument("--isa-summary", action="append", type=Path, default=[], help="gpu_isa_report.py JSON summary")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="temp-only report directory")
    parser.add_argument("--top", type=int, default=20, help="top numeric counter metrics to include")
    parser.add_argument("--json", action="store_true", help="print combined JSON report to stdout")
    parser.add_argument("--no-markdown", action="store_true", help="write only JSON report files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [
        report_for_capture(capture_path, args.counter, args.isa_summary, args.top)
        for capture_path in args.captures
    ]
    outputs: list[Path] = []
    for index, report in enumerate(reports):
        stem = safe_report_stem(Path(report["capture"]["path"]), index)
        outputs.append(write_json_report(report, args.out_dir, stem))
        if not args.no_markdown:
            outputs.append(write_markdown_report(report, args.out_dir, stem))

    if args.json:
        print(json.dumps({"valid": True, "reports": reports, "outputs": [str(path) for path in outputs]}, indent=2, sort_keys=True))
    else:
        print("GPU counter report: PASS")
        for out_path in outputs:
            print(f"- {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkSchemaError as exc:
        print(f"GPU counter report: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001 - CLI should show the concise failure cause.
        print(f"GPU counter report: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
