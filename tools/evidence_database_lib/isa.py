from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import BACKEND_ALIASES, ISA_COUNT_FIELDS, ISA_MAX_FIELDS
from .io import discover_isa_report_paths, normalized_capture_path

def normalized_backend(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    lowered = value.lower()
    return BACKEND_ALIASES.get(lowered, lowered)


def normalized_target(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value.lower() in {"none", "unknown"}:
        return None
    return value.lower()


def isa_index_key(backend: str, target: str | None) -> str:
    return f"{backend}|{target or '*'}"


def numeric_value(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) else None


def safe_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def sorted_strings(values: list[Any]) -> list[str]:
    return sorted({str(value) for value in values if value is not None and str(value)})


def summarize_isa_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    totals = report.get("instruction_totals") if isinstance(report.get("instruction_totals"), dict) else {}
    tools = report.get("tools") if isinstance(report.get("tools"), dict) else {}
    summary = {
        "isa_report_path": str(path),
        "isa_report_object": report.get("object"),
        "isa_report_backend": report.get("backend"),
        "isa_report_target": report.get("target"),
        "isa_report_symbol_count": report.get("reported_symbol_count"),
        "isa_device_symbol_count": report.get("device_symbol_count"),
        "isa_code_object_note": report.get("code_object_note"),
        "isa_rga_status": tools.get("rga_status"),
        "isa_rga_path": tools.get("rga"),
    }
    for source_key, row_key in ISA_COUNT_FIELDS.items():
        summary[row_key] = numeric_value(totals.get(source_key))
    for source_key, row_key in ISA_MAX_FIELDS.items():
        summary[row_key] = numeric_value(totals.get(source_key))
    return summary


def load_isa_index(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in discover_isa_report_paths(paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            continue
        backend = normalized_backend(report.get("backend"))
        if backend is None:
            continue
        target = normalized_target(report.get("target"))
        summary = summarize_isa_report(path, report)
        index[isa_index_key(backend, target)].append(summary)
        index[isa_index_key(backend, None)].append(summary)
    return {key: sorted(value, key=lambda item: str(item.get("isa_report_path"))) for key, value in index.items()}


def lookup_metadata(index: dict[str, dict[str, Any]], capture_path: str) -> dict[str, Any]:
    normalized = normalized_capture_path(capture_path)
    return index.get(normalized) or index.get(Path(capture_path).name.lower()) or {}


def aggregate_isa_resources(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {
            "isa_resource_reports": [],
            "isa_report_count": 0,
            "isa_report_paths": [],
            "isa_report_backends": [],
            "isa_report_targets": [],
            "isa_rga_statuses": [],
        }
    aggregate: dict[str, Any] = {
        "isa_resource_reports": reports,
        "isa_report_count": len(reports),
        "isa_report_paths": [report.get("isa_report_path") for report in reports],
        "isa_report_backends": sorted(
            {
                str(report.get("isa_report_backend"))
                for report in reports
                if report.get("isa_report_backend") is not None
            }
        ),
        "isa_report_targets": sorted(
            {
                str(report.get("isa_report_target"))
                for report in reports
                if report.get("isa_report_target") is not None
            }
        ),
        "isa_rga_statuses": sorted(
            {str(report.get("isa_rga_status")) for report in reports if report.get("isa_rga_status") is not None}
        ),
        "isa_code_object_notes": [
            report.get("isa_code_object_note") for report in reports if report.get("isa_code_object_note") is not None
        ],
    }
    for key in ("isa_report_symbol_count", "isa_device_symbol_count", *ISA_COUNT_FIELDS.values()):
        values = [numeric_value(report.get(key)) for report in reports]
        aggregate[key] = sum(value for value in values if value is not None)
    for key in ISA_MAX_FIELDS.values():
        values = [numeric_value(report.get(key)) for report in reports]
        aggregate[key] = max((value for value in values if value is not None), default=None)
    return aggregate


def lookup_isa_resources(index: dict[str, list[dict[str, Any]]], backend: Any, target: Any) -> dict[str, Any]:
    normalized = normalized_backend(backend)
    if normalized is None:
        return aggregate_isa_resources([])
    exact = index.get(isa_index_key(normalized, normalized_target(target)))
    if exact is not None:
        return aggregate_isa_resources(exact)
    return aggregate_isa_resources(index.get(isa_index_key(normalized, None), []))


