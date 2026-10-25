from __future__ import annotations

import json
import hashlib
import math
import re
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
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    # A target list, family, or missing target is not a concrete code object.
    return value if re.fullmatch(r"gfx[0-9a-f]+(?::[a-z0-9_+-]+)*", value) else None


def isa_index_key(backend: str, target: str | None) -> str:
    return f"{backend}|{target or '*'}"


def numeric_value(value: Any) -> int | float | None:
    return value if type(value) in (int, float) and math.isfinite(value) else None


def safe_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def sorted_strings(values: list[Any]) -> list[str]:
    return sorted({str(value) for value in values if value is not None and str(value)})


def summarize_isa_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    totals = report.get("instruction_totals") if isinstance(report.get("instruction_totals"), dict) else {}
    tools = report.get("tools") if isinstance(report.get("tools"), dict) else {}
    summary = {
        "isa_report_path": str(path),
        "isa_object_sha256": report.get("object_sha256"),
        "isa_code_object_sha256": report.get("code_object_sha256"),
        "isa_capture_binding": report.get("capture_binding"),
        "isa_symbols": report.get("symbols") if isinstance(report.get("symbols"), list) else [],
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
    histogram = totals.get("matrix_instruction_histogram")
    summary["isa_matrix_instruction_histogram"] = histogram if isinstance(histogram, dict) else {}
    families = totals.get("matrix_instruction_families")
    summary["isa_matrix_instruction_families"] = families if isinstance(families, list) else []
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
        if target is not None:
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
    histogram: dict[str, int] = {}
    families: set[str] = set()
    for report in reports:
        report_histogram = report.get("isa_matrix_instruction_histogram")
        if isinstance(report_histogram, dict):
            for name, count in report_histogram.items():
                try:
                    histogram[str(name)] = histogram.get(str(name), 0) + int(count)
                except (TypeError, ValueError):
                    continue
        report_families = report.get("isa_matrix_instruction_families")
        if isinstance(report_families, list):
            families.update(str(item) for item in report_families if item)
    aggregate["isa_matrix_instruction_histogram"] = dict(sorted(histogram.items()))
    aggregate["isa_matrix_instruction_families"] = sorted(families)
    return aggregate


def lookup_isa_inventory(index: dict[str, list[dict[str, Any]]], backend: Any, target: Any) -> dict[str, Any]:
    """Diagnostic inventory only; never the selected kernel's resource profile."""
    name, concrete = normalized_backend(backend), normalized_target(target)
    return aggregate_isa_resources(index.get(isa_index_key(name, concrete), []) if name else [])


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def lookup_isa_resources(index: dict[str, list[dict[str, Any]]], backend: Any, target: Any,
                         capture: dict[str, Any] | None = None) -> dict[str, Any]:
    name, concrete = normalized_backend(backend), normalized_target(target)
    inventory = lookup_isa_inventory(index, backend, target)
    reports = index.get(isa_index_key(name, concrete), []) if name and concrete else []
    qualified = []
    seen = {}
    reasons = set()
    artifact = (capture or {}).get("artifact_capture") or {}
    manifest_hash = hashlib.sha256(json.dumps((capture or {}).get("build_manifest"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not concrete: reasons.add("concrete_target_missing")
    if not capture: reasons.add("capture_identity_missing")
    if not reports: reasons.add("no_exact_target_report")
    for report in reports:
        binding = report.get("isa_capture_binding")
        if not isinstance(binding, dict): binding = {}
        kind = binding.get("artifact_kind")
        if (binding.get("schema") != "rns8_selected_static_isa_binding_v1" or
                binding.get("artifact_membership_verified") is not True or
                binding.get("selection_evidence") != "explicit_static_symbol_mapping" or
                kind not in ("executable_sha256", "shared_library_sha256") or
                not _sha(binding.get("artifact_sha256")) or
                binding.get("artifact_sha256") != artifact.get(kind) or
                artifact.get("unchanged_during_run") is not True or
                not _sha(report.get("isa_object_sha256")) or
                not _sha(report.get("isa_code_object_sha256")) or
                binding.get("code_object_sha256") != report.get("isa_code_object_sha256") or
                binding.get("build_manifest_sha256") != manifest_hash or
                normalized_target(binding.get("target")) != concrete or
                not (capture or {}).get("selected_kernel") or
                binding.get("selected_kernel") != (capture or {}).get("selected_kernel")):
            reasons.add("artifact_build_target_or_kernel_binding_mismatch")
            continue
        symbols = binding.get("symbols")
        available = {entry.get("symbol"): entry for entry in report.get("isa_symbols", []) if isinstance(entry, dict)}
        if (not isinstance(symbols, list) or not symbols or any(not isinstance(symbol, str) for symbol in symbols) or
                len(set(symbols)) != len(symbols) or any(symbol not in available for symbol in symbols)):
            reasons.add("selected_symbol_missing")
            continue
        for symbol in symbols:
            counts = available[symbol].get("counts") or {}
            key = (binding["artifact_sha256"], report["isa_code_object_sha256"], symbol)
            if key in seen:
                if seen[key] != counts:
                    return {**aggregate_isa_resources([]), "isa_inventory": inventory,
                            "isa_association": {"qualified": False, "reasons": ["conflicting_same_object_symbol_reports"]}}
                continue
            seen[key] = counts
            selected = {**report, "isa_selected_symbol": symbol, "isa_report_symbol_count": 1,
                        "isa_device_symbol_count": 1, "isa_symbols": [available[symbol]]}
            for source, destination in {**ISA_COUNT_FIELDS, **ISA_MAX_FIELDS}.items():
                selected[destination] = numeric_value(counts.get(source))
            selected["isa_matrix_instruction_histogram"] = counts.get("matrix_instruction_histogram") or {}
            selected["isa_matrix_instruction_families"] = counts.get("matrix_instruction_families") or []
            qualified.append(selected)
    return {**aggregate_isa_resources(qualified), "isa_inventory": inventory,
            "isa_association": {"qualified": bool(qualified),
                "scope": "selected_static_symbol_set_not_executed_work",
                "dynamic_execution_proven": False,
                "reasons": [] if qualified else sorted(reasons or {"selected_binding_missing"})}}
