#!/usr/bin/env python3
"""Classify zero-tile and zero row/column skip expansion readiness."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


DIRECT_HIP_ROW_COL_KERNELS = {
    "direct_hip_tiled_active_prefix_zero_row_col_skip_rns_gemm_v1",
    "direct_hip_tiled_active_prefix_zero_tile_row_col_skip_rns_gemm_v1",
}
DIRECT_HIP_ZERO_OUTPUT_KERNELS = {
    "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3",
    "direct_hip_tiled_active_prefix_zero_tile_row_col_skip_rns_gemm_v1",
}


@dataclass(frozen=True)
class BackendCapability:
    scheduled_tiles: bool
    zero_output_tile_skip: bool
    zero_row_col_product_skip: bool
    row_col_notes: str


BACKEND_CAPABILITIES: dict[str, BackendCapability] = {
    "cpu-reference": BackendCapability(False, False, False, "CPU reference validates correctness only"),
    "hip-direct": BackendCapability(True, True, True, "Direct-HIP has device masks and row/column product skips"),
    "hip-vector-alu-int64": BackendCapability(False, False, False, "native vector path has no tiled RNS proof-mask lane"),
    "hipblaslt": BackendCapability(False, False, False, "hipBLASLt rejects scheduled/adaptive tiled RNS plans"),
    "ck": BackendCapability(True, True, False, "CK only handles whole zero-output tiles today"),
    "rocwmma": BackendCapability(True, True, False, "rocWMMA only handles whole zero-output tiles today"),
}


def expand_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.json")))
        else:
            expanded.append(path)
    return expanded


def load_validated_capture(path: Path) -> dict[str, Any]:
    try:
        data = load_capture(path)
        validate_capture(data, path)
    except BenchmarkSchemaError as exc:
        raise SystemExit(str(exc)) from exc
    data["_path"] = str(path)
    return data


def backend_id(capture: dict[str, Any]) -> str:
    selected = capture.get("backend_selected")
    return str(selected) if selected is not None else str(capture.get("backend_requested"))


def int_value(value: Any, default: int = 0) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


def schedule_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("schedule_metadata")
    return value if isinstance(value, dict) else {}


def tile_bounds_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("tile_bounds_u64")
    return value if isinstance(value, dict) else {}


def scenario_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("scenario_metadata")
    return value if isinstance(value, dict) else {}


def metadata_block(capture: dict[str, Any]) -> dict[str, Any]:
    scenario = scenario_metadata(capture)
    metadata = scenario.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def proof_source(capture: dict[str, Any]) -> str:
    metadata = metadata_block(capture)
    explicit = metadata.get("proof_source")
    if isinstance(explicit, str) and explicit:
        return explicit
    tile_source = tile_bounds_metadata(capture).get("source")
    if tile_source == "exact_seeded_input_prepass":
        return "scan_derived_exact_seeded_input_prepass"
    if capture.get("bound_source") == "input_scan":
        return "scan_derived_input_scan"
    return "unknown"


def is_caller_or_natural_sparse(capture: dict[str, Any]) -> bool:
    source = proof_source(capture)
    return source in {"caller_provided_zero_proofs", "naturally_sparse_workload_contract"}


def gpu_events_available(capture: dict[str, Any]) -> bool | None:
    backend = backend_id(capture)
    if backend in {"cpu-reference", "wrap64-byte-limb"}:
        return None
    timing = capture.get("timing_metadata")
    if not isinstance(timing, dict):
        return False
    return (
        timing.get("gpu_event_timing") is True
        and timing.get("gpu_event_timing_status") == "available"
        and isinstance(timing.get("gpu_event_phase_order"), list)
        and bool(timing.get("gpu_event_timing_source"))
    )


def classify_row_col_skip(capture: dict[str, Any]) -> tuple[str, list[str]]:
    schedule = schedule_metadata(capture)
    backend = backend_id(capture)
    selected_kernel = str(capture.get("selected_kernel") or "")
    count = int_value(schedule.get("zero_row_col_product_count"))
    capability = BACKEND_CAPABILITIES.get(
        backend,
        BackendCapability(False, False, False, "unknown backend has no declared zero-skip capability"),
    )
    blockers: list[str] = []
    if count <= 0:
        return "not_requested", blockers
    if not is_caller_or_natural_sparse(capture):
        blockers.append("proof_source_is_scan_derived_or_unknown")
    if gpu_events_available(capture) is False:
        blockers.append("missing_gpu_events")
    if capability.zero_row_col_product_skip:
        if backend == "hip-direct" and selected_kernel not in DIRECT_HIP_ROW_COL_KERNELS:
            blockers.append("selected_kernel_does_not_encode_row_col_skip")
            return "metadata_mismatch", blockers
        return ("candidate_row_col_skip" if not blockers else "experimental_row_col_skip"), blockers
    if capability.scheduled_tiles:
        blockers.append("backend_computes_full_tile_for_row_col_products")
        return "correct_full_tile_fallback", blockers
    blockers.append("backend_lacks_scheduled_tile_row_col_contract")
    return "unsupported", blockers


def classify_zero_output_skip(capture: dict[str, Any]) -> tuple[str, list[str]]:
    schedule = schedule_metadata(capture)
    backend = backend_id(capture)
    selected_kernel = str(capture.get("selected_kernel") or "")
    count = int_value(schedule.get("zero_output_tile_count"))
    capability = BACKEND_CAPABILITIES.get(
        backend,
        BackendCapability(False, False, False, "unknown backend has no declared zero-skip capability"),
    )
    blockers: list[str] = []
    if count <= 0:
        return "not_requested", blockers
    if not is_caller_or_natural_sparse(capture):
        blockers.append("proof_source_is_scan_derived_or_unknown")
    if gpu_events_available(capture) is False:
        blockers.append("missing_gpu_events")
    if capability.zero_output_tile_skip:
        if backend == "hip-direct" and selected_kernel not in DIRECT_HIP_ZERO_OUTPUT_KERNELS:
            blockers.append("selected_kernel_does_not_encode_zero_output_skip")
            return "metadata_mismatch", blockers
        return ("candidate_zero_output_skip" if not blockers else "experimental_zero_output_skip"), blockers
    blockers.append("backend_lacks_zero_output_tile_contract")
    return "unsupported", blockers


def compare_zero_skip_expansion(captures: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for capture in sorted(captures, key=lambda item: str(item.get("_path", ""))):
        schedule = schedule_metadata(capture)
        row_col_decision, row_col_blockers = classify_row_col_skip(capture)
        zero_output_decision, zero_output_blockers = classify_zero_output_skip(capture)
        backend = backend_id(capture)
        capability = BACKEND_CAPABILITIES.get(
            backend,
            BackendCapability(False, False, False, "unknown backend has no declared zero-skip capability"),
        )
        rows.append(
            {
                "capture": capture.get("_path"),
                "backend": backend,
                "selected_kernel": capture.get("selected_kernel"),
                "semantics": capture.get("semantics"),
                "shape": {
                    "m": capture.get("m"),
                    "n": capture.get("n"),
                    "k": capture.get("k"),
                },
                "proof_source": proof_source(capture),
                "caller_or_naturally_sparse": is_caller_or_natural_sparse(capture),
                "gpu_events_available": gpu_events_available(capture),
                "zero_output_tile_count": int_value(schedule.get("zero_output_tile_count")),
                "zero_row_col_product_count": int_value(schedule.get("zero_row_col_product_count")),
                "backend_capability": {
                    "scheduled_tiles": capability.scheduled_tiles,
                    "zero_output_tile_skip": capability.zero_output_tile_skip,
                    "zero_row_col_product_skip": capability.zero_row_col_product_skip,
                    "notes": capability.row_col_notes,
                },
                "zero_output_decision": zero_output_decision,
                "zero_output_blockers": zero_output_blockers,
                "row_col_decision": row_col_decision,
                "row_col_blockers": row_col_blockers,
            }
        )
    summary = {
        "captures": len(rows),
        "caller_or_naturally_sparse_captures": sum(1 for row in rows if row["caller_or_naturally_sparse"]),
        "scan_derived_or_unknown_captures": sum(1 for row in rows if not row["caller_or_naturally_sparse"]),
        "zero_output_requested": sum(1 for row in rows if row["zero_output_tile_count"] > 0),
        "row_col_requested": sum(1 for row in rows if row["zero_row_col_product_count"] > 0),
        "direct_hip_row_col_skip_candidates": sum(
            1 for row in rows if row["backend"] == "hip-direct" and row["row_col_decision"] == "candidate_row_col_skip"
        ),
        "experimental_direct_hip_row_col_skips": sum(
            1
            for row in rows
            if row["backend"] == "hip-direct" and row["row_col_decision"] == "experimental_row_col_skip"
        ),
        "accelerator_row_col_skip_candidates": sum(
            1
            for row in rows
            if row["backend"] in {"ck", "rocwmma", "hipblaslt"}
            and row["row_col_decision"] == "candidate_row_col_skip"
        ),
        "accelerator_correct_full_tile_fallbacks": sum(
            1
            for row in rows
            if row["backend"] in {"ck", "rocwmma"}
            and row["row_col_decision"] == "correct_full_tile_fallback"
        ),
        "unsupported_row_col_captures": sum(1 for row in rows if row["row_col_decision"] == "unsupported"),
        "rank22_expansion_ready": False,
    }
    summary["rank22_expansion_ready"] = (
        summary["caller_or_naturally_sparse_captures"] > 0
        and summary["accelerator_row_col_skip_candidates"] > 0
    )
    return {
        "summary": summary,
        "backend_capabilities": {
            backend: {
                "scheduled_tiles": capability.scheduled_tiles,
                "zero_output_tile_skip": capability.zero_output_tile_skip,
                "zero_row_col_product_skip": capability.zero_row_col_product_skip,
                "notes": capability.row_col_notes,
            }
            for backend, capability in sorted(BACKEND_CAPABILITIES.items())
        },
        "captures": rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Zero-Skip Expansion Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Backend Capability",
            "",
            "| Backend | Scheduled Tiles | Zero Tiles | Row/Column Product Skip | Notes |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for backend, capability in report["backend_capabilities"].items():
        lines.append(
            "| {backend} | {scheduled} | {zero_output} | {row_col} | {notes} |".format(
                backend=backend,
                scheduled=capability["scheduled_tiles"],
                zero_output=capability["zero_output_tile_skip"],
                row_col=capability["zero_row_col_product_skip"],
                notes=capability["notes"],
            )
        )
    lines.extend(
        [
            "",
            "## Captures",
            "",
            "| Backend | Shape | Proof Source | Zero Tiles | Row/Col Products | Row/Col Decision | Blockers |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in report["captures"]:
        shape = row["shape"]
        blockers = ",".join(row["row_col_blockers"]) or "none"
        lines.append(
            "| {backend} | {m}x{n}x{k} | {source} | {zero_tiles} | {row_col} | {decision} | {blockers} |".format(
                backend=row["backend"],
                m=shape.get("m"),
                n=shape.get("n"),
                k=shape.get("k"),
                source=row["proof_source"],
                zero_tiles=row["zero_output_tile_count"],
                row_col=row["zero_row_col_product_count"],
                decision=row["row_col_decision"],
                blockers=blockers,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="capture file or directory")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = expand_inputs(args.captures)
    captures = [load_validated_capture(path) for path in paths]
    report = compare_zero_skip_expansion(captures)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
