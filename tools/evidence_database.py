#!/usr/bin/env python3
"""Build an ignored RNS8 benchmark evidence database from schema v4 captures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from benchmark_schema import load_capture, validate_capture


PHASES = ("pack", "rns_gemm", "crt_export", "end_to_end")
SKIP_JSON_NAMES = {
    "review_report.json",
    "scenario_manifest.json",
    "autotune.json",
    "autotune-cache.json",
    "evidence_database.json",
}
CSV_FIELDS = [
    "capture_path",
    "scenario_family",
    "scenario_name",
    "scenario_source_role",
    "scenario_evidence_role",
    "scenario_domain_family",
    "scenario_algebra_family",
    "scenario_workflow_name",
    "scenario_phase_label",
    "scenario_reuse_profile",
    "scenario_lowering_role",
    "scenario_output_domain_requirement",
    "scenario_large_shape_role",
    "scenario_promotion_scope",
    "scenario_grouping_role",
    "scenario_bridge_role",
    "scenario_modulus_role",
    "scenario_prime_or_composite",
    "scenario_metadata_json",
    "semantics",
    "backend",
    "selected_kernel",
    "target_id",
    "device_name",
    "m",
    "n",
    "k",
    "finite_modulus",
    "prefix",
    "selected_prefix",
    "exact_wide_limb_count",
    "pack_mode",
    "output_domain",
    "median_pack_us",
    "median_rns_gemm_us",
    "median_crt_export_us",
    "median_end_to_end_us",
    "bottleneck_class",
    "bottleneck_phase",
    "bottleneck_share",
    "roofline_target",
    "optimization_hint",
    "event_bottleneck_class",
    "event_bottleneck_share",
    "estimated_ops",
    "estimated_bytes",
    "arithmetic_intensity_ops_per_byte",
    "measured_gops",
    "pack_bandwidth_gbs",
    "export_bandwidth_gbs",
    "isa_evidence",
    "isa_report_count",
    "isa_report_paths",
    "isa_report_backends",
    "isa_report_targets",
    "isa_report_symbol_count",
    "isa_device_symbol_count",
    "isa_wmma_count",
    "isa_mfma_count",
    "isa_global_store_count",
    "isa_lds_mentions",
    "isa_wait_instructions",
    "isa_instruction_lines",
    "isa_vgpr_count",
    "isa_sgpr_count",
    "isa_occupancy",
    "isa_rga_statuses",
    "promotable",
    "promotion_blockers",
]
ISA_COUNT_FIELDS = {
    "wmma": "isa_wmma_count",
    "mfma": "isa_mfma_count",
    "global_store": "isa_global_store_count",
    "lds_mentions": "isa_lds_mentions",
    "wait_instructions": "isa_wait_instructions",
    "instruction_lines": "isa_instruction_lines",
}
ISA_MAX_FIELDS = {
    "vgpr_count": "isa_vgpr_count",
    "sgpr_count": "isa_sgpr_count",
    "occupancy": "isa_occupancy",
}
BACKEND_ALIASES = {
    "direct-hip": "direct-hip",
    "hip-direct": "direct-hip",
    "hip": "direct-hip",
    "hipblaslt": "hipblaslt",
    "ck": "ck",
    "rocwmma": "rocwmma",
    "wmma": "rocwmma",
    "vector-alu": "vector-alu",
    "vector-alu-int64": "vector-alu",
    "hip-vector-alu-int64": "vector-alu",
    "hip-vector-alu-int64-baseline": "vector-alu",
    "wrap64": "wrap64",
    "wrap64-byte-limb": "wrap64",
}
OPTIMIZATION_HINTS = {
    "compute_throughput": "Inspect selected kernels, ISA, occupancy, and accelerator choice before pack/export work.",
    "pack_bandwidth": "Inspect pack layout, prepack reuse, fusion, and H2D staging before GEMM tuning.",
    "export_bandwidth": "Inspect CRT/export, status traffic, compact output, and lazy output-domain policy.",
    "launch_api_overhead": "Inspect batching, HIP Graph replay, reuse contracts, and plan/workspace caching.",
    "transfer_bandwidth": "Inspect H2D/D2H staging, output layout, and host/device residency policy.",
    "status_overhead": "Inspect status memset/D2H policy and cases where overflow/status is impossible.",
    "mixed_phase_balance": "Compare phase-specific A/B runs before specializing a single phase.",
    "unclassified": "Capture stronger timing/event metadata before choosing an optimization target.",
}


def median_phase(capture: dict[str, Any], phase: str) -> float | None:
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    item = summary.get(phase)
    if not isinstance(item, dict):
        return None
    value = item.get("median")
    return float(value) if isinstance(value, (int, float)) else None


def event_medians(capture: dict[str, Any]) -> dict[str, float]:
    summary = capture.get("gpu_event_timing_summary_us")
    if not isinstance(summary, dict):
        return {}
    result: dict[str, float] = {}
    for name, item in summary.items():
        if not isinstance(item, dict):
            continue
        value = item.get("median")
        if isinstance(value, (int, float)):
            result[str(name)] = float(value)
    return result


def normalized_capture_path(path: Path | str) -> str:
    return str(Path(path)).replace("\\", "/").lower()


def is_skipped_json_candidate(path: Path) -> bool:
    name = path.name
    return (
        name in SKIP_JSON_NAMES
        or name.endswith(".failed.json")
        or name.endswith("autotune-cache.json")
        or name.endswith("-isa-summary.json")
    )


def discover_capture_paths(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.rglob("*.json")):
                if is_skipped_json_candidate(candidate):
                    continue
                discovered.append(candidate)
        else:
            discovered.append(path)
    return list(dict.fromkeys(discovered))


def discover_isa_report_paths(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*-isa-summary.json")))
        else:
            discovered.append(path)
    return list(dict.fromkeys(discovered))


def load_validated_captures(
    paths: list[Path],
    *,
    skip_invalid: bool = False,
    invalid_captures: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    for path in discover_capture_paths(paths):
        try:
            capture = load_capture(path)
            validate_capture(capture)
        except Exception as exc:
            if not skip_invalid:
                raise
            message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            if invalid_captures is not None:
                invalid_captures.append({"capture_path": str(path), "error": message})
            print(f"warning: skipped invalid capture {path}: {message}", file=sys.stderr)
            continue
        capture["_path"] = str(path)
        captures.append(capture)
    return captures


def load_scenario_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for entry in manifest.get("entries", []):
            if not isinstance(entry, dict):
                continue
            for key in (entry.get("capture_path"), entry.get("capture_name")):
                if isinstance(key, str) and key:
                    index[normalized_capture_path(key)] = entry
                    index[Path(key).name.lower()] = entry
    return index


def load_review_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for group in report.get("groups", []):
            if not isinstance(group, dict):
                continue
            for candidate in group.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                capture = candidate.get("capture")
                if not isinstance(capture, str) or not capture:
                    continue
                payload = {
                    "review_mode": group.get("review_mode"),
                    "release_review_satisfied": group.get("release_review_satisfied"),
                    "missing_required_baselines": group.get("missing_required_baselines") or [],
                    "promotable": candidate.get("promotable"),
                    "promotion_blockers": candidate.get("promotion_blockers") or [],
                    "primary_loss_phase_vs_direct_hip": candidate.get("primary_loss_phase_vs_direct_hip"),
                    "speedup_vs_direct_hip": candidate.get("speedup_vs_direct_hip"),
                    "speedup_vs_vector_alu": candidate.get("speedup_vs_vector_alu"),
                }
                index[normalized_capture_path(capture)] = payload
                index[Path(capture).name.lower()] = payload
    return index


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


def selected_residue_planes(capture: dict[str, Any]) -> int:
    for key in ("residue_planes_selected", "selected_prefix", "prefix"):
        value = capture.get(key)
        if isinstance(value, int) and value > 0:
            return value
    semantics = capture.get("semantics")
    if semantics in {"finite_ring_u8", "finite_field_u8", "wrap_u64_mod_2_64"}:
        return 1
    return 0


def native_input_bytes_per_element(capture: dict[str, Any]) -> int:
    semantics = capture.get("semantics")
    if semantics in {"finite_ring_u8", "finite_field_u8"}:
        return 1
    return 8


def output_bytes_per_element(capture: dict[str, Any]) -> int:
    semantics = capture.get("semantics")
    if semantics in {"finite_ring_u8", "finite_field_u8"}:
        return 1
    if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
        limb_count = capture.get("exact_wide_limb_count")
        return 8 * int(limb_count if isinstance(limb_count, int) and limb_count > 0 else 4)
    return 8


def estimate_work(capture: dict[str, Any]) -> dict[str, float | int | None]:
    m = int(capture.get("m", 0) or 0)
    n = int(capture.get("n", 0) or 0)
    k = int(capture.get("k", 0) or 0)
    semantics = capture.get("semantics")
    planes = selected_residue_planes(capture)
    if semantics == "wrap_u64_mod_2_64":
        logical_ops = 2 * m * n * k * 36
        residue_bytes = m * n * 8
    elif semantics in {"finite_ring_u8", "finite_field_u8"}:
        logical_ops = 2 * m * n * k
        residue_bytes = m * n
    else:
        logical_ops = 2 * m * n * k * max(planes, 1)
        residue_bytes = m * n * max(planes, 1)
    input_bytes = (m * k + k * n) * native_input_bytes_per_element(capture)
    output_bytes = m * n * output_bytes_per_element(capture)
    estimated_bytes = input_bytes + output_bytes + residue_bytes
    gemm_us = median_phase(capture, "rns_gemm")
    pack_us = median_phase(capture, "pack")
    export_us = median_phase(capture, "crt_export")
    return {
        "estimated_ops": logical_ops,
        "estimated_input_bytes": input_bytes,
        "estimated_output_bytes": output_bytes,
        "estimated_residue_bytes": residue_bytes,
        "estimated_bytes": estimated_bytes,
        "arithmetic_intensity_ops_per_byte": (logical_ops / estimated_bytes) if estimated_bytes else None,
        "measured_gops": (logical_ops / (gemm_us * 1000.0)) if gemm_us and gemm_us > 0 else None,
        "pack_bandwidth_gbs": (input_bytes / (pack_us * 1000.0)) if pack_us and pack_us > 0 else None,
        "export_bandwidth_gbs": (output_bytes / (export_us * 1000.0)) if export_us and export_us > 0 else None,
    }


def classify_event_bottleneck(events: dict[str, float]) -> tuple[str | None, float | None, dict[str, float]]:
    categories = {"status": 0.0, "transfer": 0.0, "pack": 0.0, "compute": 0.0, "export": 0.0, "other": 0.0}
    for name, value in events.items():
        lowered = name.lower()
        if "status" in lowered:
            categories["status"] += value
        elif "h2d" in lowered or "d2h" in lowered or "memcpy" in lowered:
            categories["transfer"] += value
        elif "pack" in lowered or "transpose" in lowered:
            categories["pack"] += value
        elif "gemm" in lowered or "matmul" in lowered or "kernel_group" in lowered:
            categories["compute"] += value
        elif "export" in lowered or "reduce" in lowered:
            categories["export"] += value
        else:
            categories["other"] += value
    total = sum(categories.values())
    if total <= 0:
        return None, None, categories
    category, value = max(categories.items(), key=lambda item: item[1])
    return f"{category}_event_bound", value / total, categories


def classify_bottleneck(capture: dict[str, Any]) -> dict[str, Any]:
    medians = {phase: median_phase(capture, phase) for phase in PHASES}
    end_to_end = medians.get("end_to_end")
    phase_values = {phase: medians[phase] for phase in ("pack", "rns_gemm", "crt_export") if medians[phase]}
    phase_shares = {
        phase: (value / end_to_end)
        for phase, value in phase_values.items()
        if end_to_end and end_to_end > 0 and value is not None
    }
    overhead = None
    if end_to_end and end_to_end > 0:
        overhead = max(0.0, end_to_end - sum(float(value) for value in phase_values.values())) / end_to_end
    phase = max(phase_shares, key=phase_shares.get) if phase_shares else None
    share = phase_shares.get(phase) if phase else None
    if overhead is not None and overhead >= 0.25 and (share is None or overhead > share):
        bottleneck_class = "launch_or_api_bound"
        bottleneck_phase = "unattributed_overhead"
        bottleneck_share = overhead
    elif phase is None or share is None:
        bottleneck_class = "unknown"
        bottleneck_phase = None
        bottleneck_share = None
    elif share < 0.40:
        bottleneck_class = "mixed_bound"
        bottleneck_phase = phase
        bottleneck_share = share
    else:
        bottleneck_class = {
            "pack": "pack_bound",
            "rns_gemm": "compute_bound",
            "crt_export": "export_bound",
        }[phase]
        bottleneck_phase = phase
        bottleneck_share = share
    event_class, event_share, event_categories = classify_event_bottleneck(event_medians(capture))
    return {
        "phase_medians_us": medians,
        "phase_shares": phase_shares,
        "unattributed_overhead_share": overhead,
        "bottleneck_class": bottleneck_class,
        "bottleneck_phase": bottleneck_phase,
        "bottleneck_share": bottleneck_share,
        "event_bottleneck_class": event_class,
        "event_bottleneck_share": event_share,
        "event_category_medians_us": event_categories,
    }


def roofline_target_for_row(row: dict[str, Any]) -> str:
    event_class = str(row.get("event_bottleneck_class") or "")
    if event_class == "status_event_bound":
        return "status_overhead"
    if event_class == "transfer_event_bound":
        return "transfer_bandwidth"
    bottleneck_class = str(row.get("bottleneck_class") or "")
    return {
        "compute_bound": "compute_throughput",
        "pack_bound": "pack_bandwidth",
        "export_bound": "export_bandwidth",
        "launch_or_api_bound": "launch_api_overhead",
        "mixed_bound": "mixed_phase_balance",
    }.get(bottleneck_class, "unclassified")


def optimization_hint_for_target(target: str) -> str:
    return OPTIMIZATION_HINTS.get(target, OPTIMIZATION_HINTS["unclassified"])


def bottleneck_time_us(row: dict[str, Any]) -> float:
    phase = row.get("bottleneck_phase")
    if isinstance(phase, str) and phase in PHASES:
        phase_value = safe_float(row.get(f"median_{phase}_us"))
        if phase_value is not None:
            return phase_value
    end_to_end = safe_float(row.get("median_end_to_end_us")) or 0.0
    share = safe_float(row.get("bottleneck_share"))
    if share is None:
        return end_to_end
    return end_to_end * max(0.0, min(share, 1.0))


def median_or_none(values: list[float]) -> float | None:
    return median(values) if values else None


def build_roofline_priority(rows: list[dict[str, Any]], *, limit: int = 24) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = str(row.get("roofline_target") or roofline_target_for_row(row))
        scenario = str(row.get("scenario_family") or "unlabeled")
        semantics = str(row.get("semantics") or "unknown")
        target_id = str(row.get("target_id") or "unknown")
        groups[(target, scenario, semantics, target_id)].append(row)

    priorities: list[dict[str, Any]] = []
    for (target, scenario, semantics, target_id), grouped_rows in groups.items():
        end_to_end_values = [
            value for value in (safe_float(row.get("median_end_to_end_us")) for row in grouped_rows) if value is not None
        ]
        bottleneck_values = [bottleneck_time_us(row) for row in grouped_rows]
        share_values = [value for value in (safe_float(row.get("bottleneck_share")) for row in grouped_rows) if value is not None]
        ai_values = [
            value
            for value in (safe_float(row.get("arithmetic_intensity_ops_per_byte")) for row in grouped_rows)
            if value is not None
        ]
        gops_values = [value for value in (safe_float(row.get("measured_gops")) for row in grouped_rows) if value is not None]
        pack_bw_values = [
            value for value in (safe_float(row.get("pack_bandwidth_gbs")) for row in grouped_rows) if value is not None
        ]
        export_bw_values = [
            value for value in (safe_float(row.get("export_bandwidth_gbs")) for row in grouped_rows) if value is not None
        ]
        top_rows = sorted(grouped_rows, key=bottleneck_time_us, reverse=True)[:3]
        priorities.append(
            {
                "rank": 0,
                "roofline_target": target,
                "optimization_hint": optimization_hint_for_target(target),
                "scenario_family": scenario,
                "semantics": semantics,
                "target_id": target_id,
                "captures": len(grouped_rows),
                "total_end_to_end_us": sum(end_to_end_values),
                "total_bottleneck_us": sum(bottleneck_values),
                "median_end_to_end_us": median_or_none(end_to_end_values),
                "median_bottleneck_share": median_or_none(share_values),
                "median_arithmetic_intensity_ops_per_byte": median_or_none(ai_values),
                "median_measured_gops": median_or_none(gops_values),
                "median_pack_bandwidth_gbs": median_or_none(pack_bw_values),
                "median_export_bandwidth_gbs": median_or_none(export_bw_values),
                "bottleneck_classes": sorted_strings([row.get("bottleneck_class") for row in grouped_rows]),
                "bottleneck_phases": sorted_strings([row.get("bottleneck_phase") for row in grouped_rows]),
                "event_bottlenecks": sorted_strings([row.get("event_bottleneck_class") for row in grouped_rows]),
                "backends": sorted_strings([row.get("backend") for row in grouped_rows]),
                "selected_kernels": sorted_strings([row.get("selected_kernel") for row in grouped_rows]),
                "example_captures": [str(row.get("capture_path")) for row in top_rows if row.get("capture_path")],
            }
        )

    priorities.sort(
        key=lambda item: (
            -float(item.get("total_bottleneck_us") or 0.0),
            str(item.get("roofline_target")),
            str(item.get("scenario_family")),
            str(item.get("semantics")),
            str(item.get("target_id")),
        )
    )
    for index, item in enumerate(priorities[:limit], start=1):
        item["rank"] = index
    return priorities[:limit]


def gpu_evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if normalized_target(row.get("target_id")) is not None and normalized_backend(row.get("backend")) != "cpu-reference"
    ]


def build_row(
    capture: dict[str, Any],
    scenario: dict[str, Any],
    review: dict[str, Any],
    isa_resources: dict[str, Any],
) -> dict[str, Any]:
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    timing_metadata = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
    work = estimate_work(capture)
    bottleneck = classify_bottleneck(capture)
    scenario_extra = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    row = {
        "capture_path": capture.get("_path"),
        "scenario_family": scenario.get("family", "unlabeled"),
        "scenario_name": scenario.get("name"),
        "scenario_evidence_scope": scenario.get("evidence_scope"),
        "scenario_rationale": scenario.get("rationale"),
        "scenario_source_role": scenario_extra.get("source_role"),
        "scenario_evidence_role": scenario_extra.get("evidence_role"),
        "scenario_domain_family": scenario_extra.get("domain_family"),
        "scenario_algebra_family": scenario_extra.get("algebra_family"),
        "scenario_workflow_name": scenario_extra.get("workflow_name"),
        "scenario_phase_label": scenario_extra.get("phase_label"),
        "scenario_reuse_profile": scenario_extra.get("reuse_profile"),
        "scenario_lowering_role": scenario_extra.get("lowering_role"),
        "scenario_output_domain_requirement": scenario_extra.get("output_domain_requirement"),
        "scenario_large_shape_role": scenario_extra.get("large_shape_role"),
        "scenario_promotion_scope": scenario_extra.get("promotion_scope"),
        "scenario_grouping_role": scenario_extra.get("grouping_role"),
        "scenario_bridge_role": scenario_extra.get("bridge_role"),
        "scenario_modulus_role": scenario_extra.get("modulus_role"),
        "scenario_prime_or_composite": scenario_extra.get("prime_or_composite"),
        "scenario_metadata": scenario_extra,
        "scenario_metadata_json": json.dumps(scenario_extra, sort_keys=True) if scenario_extra else None,
        "semantics": capture.get("semantics"),
        "backend": capture.get("backend_selected"),
        "backend_requested": capture.get("backend_requested"),
        "selected_kernel": capture.get("selected_kernel"),
        "m": capture.get("m"),
        "n": capture.get("n"),
        "k": capture.get("k"),
        "finite_modulus": capture.get("finite_modulus"),
        "prefix": capture.get("prefix"),
        "selected_prefix": capture.get("selected_prefix"),
        "exact_wide_limb_count": capture.get("exact_wide_limb_count"),
        "pack_mode": capture.get("pack_mode") or timing_metadata.get("pack_mode") or scenario.get("pack_mode"),
        "prepack_reuse_strategy": capture.get("prepack_reuse_strategy") or timing_metadata.get("prepack_reuse_strategy"),
        "output_domain": scenario.get("output_domain") or capture.get("residue_output_mode"),
        "target_id": device.get("gcn_arch"),
        "device_name": device.get("name"),
        "hip_runtime_version": device.get("hip_runtime_version"),
        "hip_driver_version": device.get("hip_driver_version"),
        "hip_sdk_or_rocm_version": (capture.get("hip_toolchain") or {}).get("hip_sdk_or_rocm_version")
        if isinstance(capture.get("hip_toolchain"), dict)
        else None,
        "accelerator_library": metadata.get("accelerator_library"),
        "workspace_bytes": metadata.get("workspace_required_bytes"),
        "isa_evidence": metadata.get("isa_evidence"),
        "autotune_key": metadata.get("autotune_key"),
        "checksum": capture.get("checksum"),
        "warmups": capture.get("warmups"),
        "repeats": capture.get("repeats"),
        "seed": capture.get("seed"),
        "review": review,
        **work,
        **bottleneck,
        **isa_resources,
    }
    for phase in PHASES:
        row[f"median_{phase}_us"] = row["phase_medians_us"].get(phase)
    row["roofline_target"] = roofline_target_for_row(row)
    row["optimization_hint"] = optimization_hint_for_target(str(row["roofline_target"]))
    row["promotable"] = review.get("promotable")
    row["promotion_blockers"] = review.get("promotion_blockers") or []
    return row


def build_database(
    captures: list[dict[str, Any]],
    *,
    scenario_index: dict[str, dict[str, Any]] | None = None,
    review_index: dict[str, dict[str, Any]] | None = None,
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
    invalid_captures: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    scenario_index = scenario_index or {}
    review_index = review_index or {}
    isa_index = isa_index or {}
    invalid_captures = invalid_captures or []
    rows = []
    for capture in captures:
        path = str(capture.get("_path", ""))
        device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
        rows.append(
            build_row(
                capture,
                lookup_metadata(scenario_index, path),
                lookup_metadata(review_index, path),
                lookup_isa_resources(isa_index, capture.get("backend_selected"), device.get("gcn_arch")),
            )
        )
    bottleneck_counts = Counter(str(row.get("bottleneck_class")) for row in rows)
    scenario_counts = Counter(str(row.get("scenario_family") or "unlabeled") for row in rows)
    backend_counts = Counter(str(row.get("backend")) for row in rows)
    isa_report_paths = sorted(
        {
            str(path)
            for row in rows
            for path in (row.get("isa_report_paths") or [])
            if path is not None
        }
    )
    roofline_priority = build_roofline_priority(rows)
    gpu_roofline_priority = build_roofline_priority(gpu_evidence_rows(rows))
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "capture_count": len(rows),
        "summary": {
            "bottleneck_counts": dict(sorted(bottleneck_counts.items())),
            "scenario_counts": dict(sorted(scenario_counts.items())),
            "backend_counts": dict(sorted(backend_counts.items())),
            "isa_report_count": len(isa_report_paths),
            "captures_with_isa_resources": sum(1 for row in rows if row.get("isa_report_count", 0) > 0),
            "roofline_priority": roofline_priority,
            "gpu_roofline_priority": gpu_roofline_priority,
            "skipped_invalid_capture_count": len(invalid_captures),
        },
        "skipped_invalid_captures": invalid_captures,
        "rows": rows,
    }


def csv_value(value: Any) -> str | int | float | None:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def write_outputs(database: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evidence_database.json"
    json_path.write_text(json.dumps(database, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_dir / "evidence_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in database["rows"]:
            writer.writerow({field: csv_value(row.get(field)) for field in CSV_FIELDS})

    markdown_path = out_dir / "evidence_summary.md"
    write_markdown(database, markdown_path)
    return {
        "evidence_database": str(json_path),
        "evidence_rows_csv": str(csv_path),
        "evidence_summary": str(markdown_path),
    }


def format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return ""
    return str(value)


def format_isa_brief(row: dict[str, Any]) -> str:
    if not row.get("isa_report_count"):
        return ""
    parts = [f"reports={row.get('isa_report_count')}"]
    for key, label in (
        ("isa_wmma_count", "wmma"),
        ("isa_mfma_count", "mfma"),
        ("isa_global_store_count", "stores"),
    ):
        value = row.get(key)
        if value is not None:
            parts.append(f"{label}={value}")
    return ";".join(parts)


def append_count_table(
    lines: list[str],
    *,
    heading: str,
    value_heading: str,
    counts: Counter[str],
) -> None:
    if not counts:
        return
    lines.extend(["", heading, "", f"| {value_heading} | captures |", "|---|---:|"])
    for name, count in sorted(counts.items()):
        lines.append(f"| {name} | {count} |")


def scenario_metadata_counts(database: dict[str, Any], row_key: str) -> Counter[str]:
    return Counter(str(row.get(row_key)) for row in database["rows"] if row.get(row_key))


def append_roofline_priority_table(lines: list[str], heading: str, priority: list[dict[str, Any]]) -> None:
    if not priority:
        return
    lines.extend(
        [
            "",
            heading,
            "",
            "| rank | target | scenario | semantics | target id | captures | bottleneck us | e2e us | median share | median GOP/s | median AI ops/B | backends | hint |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in priority:
        lines.append(
            "| {rank} | {target} | {scenario} | {semantics} | {target_id} | {captures} | {bottleneck_us} | {e2e_us} | {share} | {gops} | {ai} | {backends} | {hint} |".format(
                rank=item.get("rank"),
                target=item.get("roofline_target"),
                scenario=item.get("scenario_family"),
                semantics=item.get("semantics"),
                target_id=item.get("target_id"),
                captures=item.get("captures"),
                bottleneck_us=format_number(item.get("total_bottleneck_us")),
                e2e_us=format_number(item.get("total_end_to_end_us")),
                share=format_number(item.get("median_bottleneck_share")),
                gops=format_number(item.get("median_measured_gops")),
                ai=format_number(item.get("median_arithmetic_intensity_ops_per_byte")),
                backends=",".join(item.get("backends") or []),
                hint=item.get("optimization_hint"),
            )
        )


def write_markdown(database: dict[str, Any], path: Path) -> None:
    lines = [
        "# RNS8 Evidence Database Summary",
        "",
        f"- schema_version: `{database.get('schema_version')}`",
        f"- generated_utc: `{database.get('generated_utc')}`",
        f"- captures: `{database.get('capture_count')}`",
        f"- skipped_invalid_capture_count: `{database.get('summary', {}).get('skipped_invalid_capture_count', 0)}`",
        f"- isa_report_count: `{database.get('summary', {}).get('isa_report_count', 0)}`",
        f"- captures_with_isa_resources: `{database.get('summary', {}).get('captures_with_isa_resources', 0)}`",
        "",
        "## Bottlenecks",
        "",
        "| class | captures |",
        "|---|---:|",
    ]
    for name, count in database["summary"]["bottleneck_counts"].items():
        lines.append(f"| {name} | {count} |")
    skipped = database.get("skipped_invalid_captures") or []
    if skipped:
        lines.extend(["", "## Skipped Invalid Captures", "", "| capture | error |", "|---|---|"])
        for item in skipped[:40]:
            lines.append(f"| {item.get('capture_path')} | {item.get('error')} |")
        if len(skipped) > 40:
            lines.append(f"| ... | {len(skipped) - 40} additional skipped captures |")
    append_roofline_priority_table(
        lines,
        "## GPU Roofline Priority",
        database.get("summary", {}).get("gpu_roofline_priority") or [],
    )
    append_roofline_priority_table(
        lines,
        "## Roofline Priority",
        database.get("summary", {}).get("roofline_priority") or [],
    )
    lines.extend(["", "## Scenario Families", "", "| family | captures |", "|---|---:|"])
    for name, count in database["summary"]["scenario_counts"].items():
        lines.append(f"| {name} | {count} |")
    metadata_tables = (
        ("source_role", "scenario_source_role"),
        ("workflow_name", "scenario_workflow_name"),
        ("reuse_profile", "scenario_reuse_profile"),
        ("lowering_role", "scenario_lowering_role"),
        ("output_domain_requirement", "scenario_output_domain_requirement"),
        ("large_shape_role", "scenario_large_shape_role"),
        ("promotion_scope", "scenario_promotion_scope"),
        ("grouping_role", "scenario_grouping_role"),
        ("bridge_role", "scenario_bridge_role"),
        ("modulus_role", "scenario_modulus_role"),
        ("prime_or_composite", "scenario_prime_or_composite"),
    )
    if any(scenario_metadata_counts(database, row_key) for _, row_key in metadata_tables):
        lines.extend(["", "## Scenario Metadata"])
        for label, row_key in metadata_tables:
            append_count_table(
                lines,
                heading=f"### {label}",
                value_heading=label,
                counts=scenario_metadata_counts(database, row_key),
            )
    isa_rows = [row for row in database["rows"] if row.get("isa_report_count")]
    if isa_rows:
        lines.extend(
            [
                "",
                "## ISA Resources",
                "",
                "| backend | target | captures | reports | WMMA | MFMA | global stores | LDS | waits | instructions | VGPR | SGPR | occupancy |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in isa_rows:
            groups[(str(row.get("backend")), ",".join(row.get("isa_report_targets") or []))].append(row)
        for (backend, target), grouped_rows in sorted(groups.items()):
            representative = grouped_rows[0]
            lines.append(
                "| {backend} | {target} | {captures} | {reports} | {wmma} | {mfma} | {stores} | {lds} | {waits} | {instructions} | {vgpr} | {sgpr} | {occupancy} |".format(
                    backend=backend,
                    target=target,
                    captures=len(grouped_rows),
                    reports=representative.get("isa_report_count"),
                    wmma=format_number(representative.get("isa_wmma_count")),
                    mfma=format_number(representative.get("isa_mfma_count")),
                    stores=format_number(representative.get("isa_global_store_count")),
                    lds=format_number(representative.get("isa_lds_mentions")),
                    waits=format_number(representative.get("isa_wait_instructions")),
                    instructions=format_number(representative.get("isa_instruction_lines")),
                    vgpr=format_number(representative.get("isa_vgpr_count")),
                    sgpr=format_number(representative.get("isa_sgpr_count")),
                    occupancy=format_number(representative.get("isa_occupancy")),
                )
            )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| scenario | semantics | backend | kernel | shape | bottleneck | e2e us | GOP/s | AI ops/B | ISA | blockers |",
            "|---|---|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    rows = sorted(
        database["rows"],
        key=lambda row: (
            str(row.get("scenario_family")),
            str(row.get("semantics")),
            float(row.get("median_end_to_end_us") or 0.0),
        ),
    )
    for row in rows:
        shape = f"{row.get('m')}x{row.get('n')}x{row.get('k')}"
        blockers = ",".join(str(item) for item in row.get("promotion_blockers") or [])
        lines.append(
            "| {scenario} | {semantics} | {backend} | {kernel} | {shape} | {bottleneck} | {e2e} | {gops} | {ai} | {isa} | {blockers} |".format(
                scenario=row.get("scenario_family"),
                semantics=row.get("semantics"),
                backend=row.get("backend"),
                kernel=row.get("selected_kernel"),
                shape=shape,
                bottleneck=row.get("bottleneck_class"),
                e2e=format_number(row.get("median_end_to_end_us")),
                gops=format_number(row.get("measured_gops")),
                ai=format_number(row.get("arithmetic_intensity_ops_per_byte")),
                isa=format_isa_brief(row),
                blockers=blockers or "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True, help="capture file or directory")
    parser.add_argument("--review-report", type=Path, action="append", default=[], help="benchmark_sweep review_report.json")
    parser.add_argument("--scenario-manifest", type=Path, action="append", default=[], help="scenario_manifest.json")
    parser.add_argument("--isa-report", type=Path, action="append", default=[], help="ISA summary file or directory")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="skip invalid JSON captures while recording skipped paths in the output summary",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("temp") / "evidence-database",
        help="ignored output directory for evidence_database.json, CSV, and Markdown",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    invalid_captures: list[dict[str, str]] = []
    captures = load_validated_captures(
        args.capture,
        skip_invalid=args.skip_invalid,
        invalid_captures=invalid_captures,
    )
    scenario_index = load_scenario_index(args.scenario_manifest)
    review_index = load_review_index(args.review_report)
    isa_index = load_isa_index(args.isa_report)
    database = build_database(
        captures,
        scenario_index=scenario_index,
        review_index=review_index,
        isa_index=isa_index,
        invalid_captures=invalid_captures,
    )
    outputs = write_outputs(database, args.out_dir)
    print(json.dumps({"captures": len(captures), **outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
