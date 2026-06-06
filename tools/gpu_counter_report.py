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
from evidence_database_lib.work_model import classify_bottleneck, estimate_work, roofline_target_for_row


COUNTER_REPORT_POLICY = (
    "explanation_evidence_only_not_a_correctness_gate_not_a_performance_claim"
)
DEFAULT_OUT_DIR = Path("temp") / "gpu-counter-reports"
COUNTER_VALUE_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
RESOURCE_SIGNAL_FIELDS = [
    "vgpr",
    "sgpr",
    "lds_bytes",
    "scratch_bytes",
    "occupancy",
    "memory_pressure_signal",
    "wait_signal",
    "global_store_instruction_count",
    "lds_instruction_mentions",
    "wait_instruction_count",
    "matrix_instruction_count",
    "arithmetic_intensity_ops_per_byte",
    "measured_gops",
    "pack_bandwidth_gbs",
    "export_bandwidth_gbs",
]


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


def _event_medians(capture: dict[str, Any]) -> dict[str, float]:
    summary = capture.get("gpu_event_timing_summary_us")
    if not isinstance(summary, dict):
        return {}
    medians: dict[str, float] = {}
    for name, value in summary.items():
        if not isinstance(value, dict):
            continue
        median = _number(value.get("median"))
        if median is not None:
            medians[str(name)] = median
    return medians


def _top_numeric_metrics(metrics: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = []
    for name, value in metrics.items():
        number = _number(value)
        if number is None:
            continue
        rows.append({"metric": str(name), "value": number})
    rows.sort(key=lambda row: abs(float(row["value"])), reverse=True)
    return rows[: max(limit, 0)]


def _numeric_summary(values: list[float], total_count: int) -> dict[str, Any]:
    if not values:
        return {
            "present_count": 0,
            "missing_count": total_count,
            "min": None,
            "average": None,
            "max": None,
        }
    return {
        "present_count": len(values),
        "missing_count": max(total_count - len(values), 0),
        "min": min(values),
        "average": sum(values) / len(values),
        "max": max(values),
    }


def _first_metric(metrics: dict[str, float], patterns: tuple[str, ...]) -> float | None:
    lowered = {key.lower(): value for key, value in metrics.items()}
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for key, value in lowered.items():
            if regex.search(key):
                return value
    return None


def _max_metric(metrics: dict[str, float], patterns: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        values.extend(value for key, value in metrics.items() if regex.search(key))
    return max(values) if values else None


def _merged_counter_metrics(counter_reports: list[dict[str, Any]]) -> dict[str, float]:
    merged: dict[str, list[float]] = {}
    for report in counter_reports:
        averages = report.get("numeric_metric_average")
        if not isinstance(averages, dict):
            continue
        for key, value in averages.items():
            number = _number(value)
            if number is not None:
                merged.setdefault(str(key), []).append(number)
    return {key: sum(values) / len(values) for key, values in sorted(merged.items()) if values}


def _isa_resource_summary(isa_reports: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "global_store": 0,
        "lds_mentions": 0,
        "wait_instructions": 0,
        "wmma": 0,
        "mfma": 0,
    }
    for item in isa_reports:
        instruction_totals = item.get("instruction_totals") or {}
        if not isinstance(instruction_totals, dict):
            continue
        for key in totals:
            value = _number(instruction_totals.get(key))
            if value is not None:
                totals[key] += int(value)
    return totals


def _counter_resource_summary(
    capture: dict[str, Any],
    counter_reports: list[dict[str, Any]],
    isa_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _merged_counter_metrics(counter_reports)
    isa_totals = _isa_resource_summary(isa_reports)
    bottleneck = classify_bottleneck(capture)
    work = estimate_work(capture)
    row = {
        **bottleneck,
        "scenario_family": capture.get("scenario_family") or capture.get("benchmark"),
        "semantics": capture.get("semantics"),
        "target_id": (capture.get("target_variant") or {}).get("target_id")
        if isinstance(capture.get("target_variant"), dict)
        else (capture.get("device") or {}).get("gcn_arch")
        if isinstance(capture.get("device"), dict)
        else None,
    }
    return {
        "vgpr": _first_metric(metrics, (r"\bvgpr\b", r"vgpr_count", r"num_vgprs")),
        "sgpr": _first_metric(metrics, (r"\bsgpr\b", r"sgpr_count", r"num_sgprs")),
        "lds_bytes": _first_metric(metrics, (r"lds.*bytes", r"lds_size", r"group_segment")),
        "scratch_bytes": _max_metric(metrics, (r"scratch.*bytes", r"scratch_size", r"private_segment")),
        "occupancy": _first_metric(metrics, (r"occupancy", r"achieved_occupancy", r"wavefronts_per_cu")),
        "wait_signal": _max_metric(metrics, (r"wait", r"stall", r"idle")),
        "memory_pressure_signal": _max_metric(metrics, (r"tcc.*req", r"tcp.*req", r"dram", r"mem_", r"write_req")),
        "global_store_instruction_count": isa_totals["global_store"],
        "lds_instruction_mentions": isa_totals["lds_mentions"],
        "wait_instruction_count": isa_totals["wait_instructions"],
        "matrix_instruction_count": isa_totals["wmma"] + isa_totals["mfma"],
        "bottleneck_class": bottleneck.get("bottleneck_class"),
        "bottleneck_phase": bottleneck.get("bottleneck_phase"),
        "event_bottleneck_class": bottleneck.get("event_bottleneck_class"),
        "roofline_target": roofline_target_for_row(row),
        "arithmetic_intensity_ops_per_byte": work.get("arithmetic_intensity_ops_per_byte"),
        "measured_gops": work.get("measured_gops"),
        "pack_bandwidth_gbs": work.get("pack_bandwidth_gbs"),
        "export_bandwidth_gbs": work.get("export_bandwidth_gbs"),
    }


def _resource_signal_coverage(resource_summary: dict[str, Any]) -> dict[str, str]:
    coverage = {}
    for field in RESOURCE_SIGNAL_FIELDS:
        coverage[field] = "present" if _number(resource_summary.get(field)) is not None else "missing"
    return coverage


def _evidence_status(
    counter_reports: list[dict[str, Any]],
    isa_reports: list[dict[str, Any]],
    event_medians: dict[str, float],
    resource_summary: dict[str, Any],
) -> dict[str, Any]:
    isa_totals = _isa_resource_summary(isa_reports)
    isa_total_count = sum(int(value) for value in isa_totals.values())
    isa_status = "missing"
    if isa_reports and isa_total_count > 0:
        isa_status = "present"
    elif isa_reports:
        isa_status = "partial"

    resource_coverage = _resource_signal_coverage(resource_summary)
    missing_evidence = []
    if not counter_reports:
        missing_evidence.append("missing_profiler_counter_export")
    if isa_status == "missing":
        missing_evidence.append("missing_isa_summary")
    elif isa_status == "partial":
        missing_evidence.append("partial_isa_summary")
    if not event_medians:
        missing_evidence.append("missing_gpu_event_medians")
    if resource_coverage.get("occupancy") == "missing":
        missing_evidence.append("missing_occupancy_signal")
    if resource_coverage.get("vgpr") == "missing" and resource_coverage.get("sgpr") == "missing":
        missing_evidence.append("missing_register_pressure_signal")
    return {
        "profiler_counter_status": "present" if counter_reports else "missing",
        "isa_resource_status": isa_status,
        "gpu_event_timing_status": "present" if event_medians else "missing",
        "resource_signal_coverage": resource_coverage,
        "missing_evidence": sorted(set(missing_evidence)),
    }


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
    event_medians = _event_medians(capture)
    resource_summary = _counter_resource_summary(capture, counter_reports, isa_reports)
    evidence_status = _evidence_status(counter_reports, isa_reports, event_medians, resource_summary)
    return {
        "schema_version": 1,
        "capture": _capture_identity(capture, capture_path),
        "policy": COUNTER_REPORT_POLICY,
        "gpu_event_medians_us": event_medians,
        "counter_inputs": counter_reports,
        "isa_summaries": isa_reports,
        "resource_summary": resource_summary,
        "evidence_status": evidence_status,
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

    summary = report.get("resource_summary") or {}
    evidence = report.get("evidence_status") or {}
    lines.extend(
        [
            "## Resource Summary",
            "",
            "| roofline target | bottleneck | event bottleneck | VGPR | SGPR | LDS bytes | scratch bytes | occupancy | memory signal | wait signal |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| `{roofline}` | `{bottleneck}` | `{event}` | {vgpr} | {sgpr} | {lds} | {scratch} | {occupancy} | {memory} | {wait} |".format(
                roofline=summary.get("roofline_target"),
                bottleneck=summary.get("bottleneck_class"),
                event=summary.get("event_bottleneck_class"),
                vgpr=summary.get("vgpr"),
                sgpr=summary.get("sgpr"),
                lds=summary.get("lds_bytes"),
                scratch=summary.get("scratch_bytes"),
                occupancy=summary.get("occupancy"),
                memory=summary.get("memory_pressure_signal"),
                wait=summary.get("wait_signal"),
            ),
            "",
        ]
    )

    lines.extend(
        [
            "## Evidence Status",
            "",
            f"- Profiler counters: `{evidence.get('profiler_counter_status')}`",
            f"- ISA resources: `{evidence.get('isa_resource_status')}`",
            f"- GPU event timing: `{evidence.get('gpu_event_timing_status')}`",
            f"- Missing evidence: `{', '.join(evidence.get('missing_evidence') or [])}`",
            "",
        ]
    )

    lines.extend(["## Review Notes", ""])
    for note in report["review_notes"]:
        lines.append(f"- {note}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def build_batch_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for report in reports:
        capture = report["capture"]
        summary = report.get("resource_summary") or {}
        key = (
            str(capture.get("target_id") or "unknown"),
            str(summary.get("roofline_target") or "unclassified"),
            str(capture.get("semantics") or "unknown"),
        )
        groups.setdefault(key, []).append(report)
    group_rows = []
    for (target_id, roofline, semantics), rows in sorted(groups.items()):
        total_count = len(rows)
        signal_summary = {}
        for field in RESOURCE_SIGNAL_FIELDS:
            values = [
                number
                for number in (_number((row.get("resource_summary") or {}).get(field)) for row in rows)
                if number is not None
            ]
            signal_summary[field] = _numeric_summary(values, total_count)
        missing_evidence = sorted(
            {
                str(item)
                for row in rows
                for item in ((row.get("evidence_status") or {}).get("missing_evidence") or [])
                if isinstance(item, str) and item
            }
        )
        status_counts: dict[str, dict[str, int]] = {
            "profiler_counter_status": {},
            "isa_resource_status": {},
            "gpu_event_timing_status": {},
        }
        for row in rows:
            evidence = row.get("evidence_status") or {}
            for key, counts in status_counts.items():
                label = str(evidence.get(key) or "unknown")
                counts[label] = counts.get(label, 0) + 1
        group_rows.append(
            {
                "target_id": target_id,
                "roofline_target": roofline,
                "semantics": semantics,
                "capture_count": total_count,
                "backends": sorted({str(row["capture"].get("backend_selected")) for row in rows}),
                "selected_kernels": sorted({str(row["capture"].get("selected_kernel")) for row in rows}),
                "bottleneck_classes": sorted(
                    {str((row.get("resource_summary") or {}).get("bottleneck_class")) for row in rows}
                ),
                "event_bottlenecks": sorted(
                    {str((row.get("resource_summary") or {}).get("event_bottleneck_class")) for row in rows}
                ),
                "counter_evidence": "present"
                if any(row.get("counter_inputs") for row in rows)
                else "missing",
                "isa_evidence": "present" if any(row.get("isa_summaries") for row in rows) else "missing",
                "counter_input_count": sum(len(row.get("counter_inputs") or []) for row in rows),
                "isa_summary_count": sum(len(row.get("isa_summaries") or []) for row in rows),
                "gpu_event_capture_count": sum(1 for row in rows if row.get("gpu_event_medians_us")),
                "evidence_status_counts": status_counts,
                "missing_evidence": missing_evidence,
                "resource_signal_summary": signal_summary,
                "example_captures": [str(row["capture"].get("path")) for row in rows[:3]],
            }
        )
    return {
        "schema_version": 1,
        "policy": COUNTER_REPORT_POLICY,
        "report_count": len(reports),
        "group_count": len(group_rows),
        "summary": {
            "captures_with_counter_exports": sum(1 for row in reports if row.get("counter_inputs")),
            "captures_with_isa_summaries": sum(1 for row in reports if row.get("isa_summaries")),
            "captures_with_gpu_event_medians": sum(1 for row in reports if row.get("gpu_event_medians_us")),
            "groups_with_missing_counter_exports": sum(
                1 for row in group_rows if row.get("counter_evidence") == "missing"
            ),
            "groups_with_missing_isa_summaries": sum(1 for row in group_rows if row.get("isa_evidence") == "missing"),
        },
        "groups": group_rows,
        "reports": reports,
    }


def write_batch_reports(batch: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gpu-counter-batch-report.json"
    md_path = out_dir / "gpu-counter-batch-report.md"
    json_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RNS8 GPU Counter Batch Report",
        "",
        f"- Policy: `{batch['policy']}`",
        f"- Capture reports: `{batch['report_count']}`",
        "",
        "| target | roofline target | semantics | captures | backends | counter evidence | ISA evidence | event captures | missing evidence | bottlenecks |",
        "| --- | --- | --- | ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for group in batch["groups"]:
        lines.append(
            "| `{target}` | `{roofline}` | `{semantics}` | {count} | `{backends}` | `{counters}` | `{isa}` | {events} | `{missing}` | `{bottlenecks}` |".format(
                target=group["target_id"],
                roofline=group["roofline_target"],
                semantics=group["semantics"],
                count=group["capture_count"],
                backends=", ".join(group["backends"]),
                counters=group["counter_evidence"],
                isa=group["isa_evidence"],
                events=group.get("gpu_event_capture_count", 0),
                missing=", ".join(group.get("missing_evidence") or []),
                bottlenecks=", ".join(group["bottleneck_classes"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _path_lookup_keys(path: Path) -> list[str]:
    keys = [str(path), path.as_posix(), path.name, path.stem]
    try:
        resolved = path.resolve()
    except OSError:
        resolved = None
    if resolved is not None:
        keys.extend([str(resolved), resolved.as_posix()])
    deduped = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


def _resolve_manifest_paths(base_dir: Path, values: Any) -> list[Path]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        values = [values]
    if not isinstance(values, list):
        raise RuntimeError("attachment manifest path values must be strings or lists")
    paths = []
    for item in values:
        if not isinstance(item, str):
            raise RuntimeError("attachment manifest paths must be strings")
        path = Path(item)
        paths.append(path if path.is_absolute() else base_dir / path)
    return paths


def load_attachment_manifest(path: Path | None) -> dict[str, dict[str, list[Path]]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent
    raw_entries: list[dict[str, Any]]
    if isinstance(data, dict) and isinstance(data.get("captures"), list):
        raw_entries = [item for item in data["captures"] if isinstance(item, dict)]
    elif isinstance(data, dict):
        raw_entries = []
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            entry = dict(value)
            entry.setdefault("capture", key)
            raw_entries.append(entry)
    else:
        raise RuntimeError("attachment manifest must be a JSON object")

    manifest = {}
    for entry in raw_entries:
        capture = entry.get("capture")
        if not isinstance(capture, str) or not capture:
            raise RuntimeError("attachment manifest entries require a capture string")
        manifest[capture] = {
            "counters": _resolve_manifest_paths(base_dir, entry.get("counters") or entry.get("counter")),
            "isa_summaries": _resolve_manifest_paths(
                base_dir,
                entry.get("isa_summaries") or entry.get("isa_summary") or entry.get("isa"),
            ),
        }
    return manifest


def attachments_for_capture(
    capture_path: Path,
    manifest: dict[str, dict[str, list[Path]]],
    global_counters: list[Path],
    global_isa_summaries: list[Path],
) -> tuple[list[Path], list[Path]]:
    counters = list(global_counters)
    isa_summaries = list(global_isa_summaries)
    for key in _path_lookup_keys(capture_path):
        entry = manifest.get(key)
        if entry:
            counters.extend(entry.get("counters") or [])
            isa_summaries.extend(entry.get("isa_summaries") or [])
            break
    return counters, isa_summaries


def build_reports(
    capture_paths: list[Path],
    global_counters: list[Path],
    global_isa_summaries: list[Path],
    attachment_manifest: dict[str, dict[str, list[Path]]],
    top_limit: int,
) -> list[dict[str, Any]]:
    reports = []
    for capture_path in capture_paths:
        counters, isa_summaries = attachments_for_capture(
            capture_path,
            attachment_manifest,
            global_counters,
            global_isa_summaries,
        )
        reports.append(report_for_capture(capture_path, counters, isa_summaries, top_limit))
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="schema-v4 benchmark JSON captures")
    parser.add_argument("--counter", action="append", type=Path, default=[], help="CSV or JSON profiler counter export")
    parser.add_argument("--isa-summary", action="append", type=Path, default=[], help="gpu_isa_report.py JSON summary")
    parser.add_argument(
        "--attachment-manifest",
        type=Path,
        help="JSON map from capture path/name/stem to per-capture counter and ISA summary files",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="temp-only report directory")
    parser.add_argument("--top", type=int, default=20, help="top numeric counter metrics to include")
    parser.add_argument("--json", action="store_true", help="print combined JSON report to stdout")
    parser.add_argument("--no-markdown", action="store_true", help="write only JSON report files")
    parser.add_argument("--batch", action="store_true", help="also write a batch report grouped by target/roofline/semantic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = build_reports(
        args.captures,
        args.counter,
        args.isa_summary,
        load_attachment_manifest(args.attachment_manifest),
        args.top,
    )
    outputs: list[Path] = []
    for index, report in enumerate(reports):
        stem = safe_report_stem(Path(report["capture"]["path"]), index)
        outputs.append(write_json_report(report, args.out_dir, stem))
        if not args.no_markdown:
            outputs.append(write_markdown_report(report, args.out_dir, stem))
    batch = build_batch_report(reports)
    if args.batch:
        outputs.extend(Path(path) for path in write_batch_reports(batch, args.out_dir).values())

    if args.json:
        print(json.dumps({"valid": True, "batch": batch, "reports": reports, "outputs": [str(path) for path in outputs]}, indent=2, sort_keys=True))
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
