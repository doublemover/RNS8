from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

from .config import OPTIMIZATION_HINTS, PHASES
from .io import event_medians, median_phase
from .isa import normalized_backend, normalized_target, safe_float, sorted_strings

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


