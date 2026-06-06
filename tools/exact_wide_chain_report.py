#!/usr/bin/env python3
"""Pair exact-wide residue-current chain captures with final-output controls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "exact-wide-chain-reports"
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
REFERENCE_BACKENDS = {"cpu-reference"}
REPORT_OUTPUT_NAMES = {
    "exact-wide-chain-report.json",
    "rns-chain-report.json",
    "review_report.json",
    "scenario_manifest.json",
}
EXACT_WIDE_SEMANTICS = {"exact_wide_signed", "exact_wide_unsigned"}


def expand_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(candidate for candidate in sorted(path.rglob("*.json")) if candidate.name not in REPORT_OUTPUT_NAMES)
        else:
            expanded.append(path)
    return expanded


def load_validated_capture(path: Path) -> dict[str, Any]:
    try:
        capture = load_capture(path)
        validate_capture(capture, path)
    except BenchmarkSchemaError as exc:
        raise SystemExit(str(exc)) from exc
    capture["_path"] = str(path)
    return capture


def backend_id(capture: dict[str, Any] | None) -> str:
    if capture is None:
        return ""
    return str(capture.get("backend_selected") or capture.get("backend_requested") or "")


def execution_mode(capture: dict[str, Any]) -> str:
    value = capture.get("benchmark_execution_mode")
    if isinstance(value, str):
        return value
    metadata = capture.get("timing_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("benchmark_execution_mode"), str):
        return metadata["benchmark_execution_mode"]
    return ""


def is_exact_wide_chain(capture: dict[str, Any]) -> bool:
    return (
        capture.get("semantics") in EXACT_WIDE_SEMANTICS
        and isinstance(capture.get("residue_chain_length"), int)
        and capture["residue_chain_length"] > 1
    )


def chain_output_mode(capture: dict[str, Any]) -> str | None:
    mode = execution_mode(capture)
    if mode == "residue_current_rns_chain" and capture.get("residue_output_mode") == "residue_current_rns":
        return "residue_current"
    if mode == "residue_chain_final_host_export" and capture.get("residue_chain_final_export") is True:
        return "final_output"
    return None


def release_satisfied(capture: dict[str, Any] | None) -> bool:
    return (
        capture is not None
        and isinstance(capture.get("warmups"), int)
        and isinstance(capture.get("repeats"), int)
        and capture["warmups"] >= RELEASE_MIN_WARMUPS
        and capture["repeats"] >= RELEASE_MIN_REPEATS
    )


def gpu_backend(capture: dict[str, Any] | None) -> bool:
    return capture is not None and backend_id(capture) not in REFERENCE_BACKENDS


def gpu_events_available(capture: dict[str, Any] | None) -> bool | None:
    if capture is None or not gpu_backend(capture):
        return None
    metadata = capture.get("timing_metadata")
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("gpu_event_timing") is True
        and metadata.get("gpu_event_timing_status") == "available"
        and isinstance(metadata.get("gpu_event_phase_order"), list)
        and bool(metadata.get("gpu_event_timing_source"))
    )


def timing_summary_value(capture: dict[str, Any] | None, phase: str, statistic: str) -> float | None:
    if capture is None:
        return None
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    phase_summary = summary.get(phase)
    if not isinstance(phase_summary, dict):
        return None
    value = phase_summary.get(statistic)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def median_end_to_end_us(capture: dict[str, Any] | None) -> float | None:
    return timing_summary_value(capture, "end_to_end", "median")


def speedup(numerator: dict[str, Any] | None, denominator: dict[str, Any] | None) -> float | None:
    numerator_us = median_end_to_end_us(numerator)
    denominator_us = median_end_to_end_us(denominator)
    if numerator_us is None or denominator_us in (None, 0.0):
        return None
    return numerator_us / denominator_us


def contract_key(capture: dict[str, Any]) -> str:
    fields = [
        "semantics",
        "m",
        "n",
        "k",
        "residue_chain_length",
        "exact_wide_limb_count",
        "prefix",
        "selected_prefix",
        "requested_max_prefix",
        "contract_prefix_policy",
        "seed",
        "input_distribution",
    ]
    return ";".join(f"{field}={capture.get(field)}" for field in fields)


def capture_summary(capture: dict[str, Any] | None) -> dict[str, Any] | None:
    if capture is None:
        return None
    return {
        "path": capture.get("_path"),
        "backend": backend_id(capture),
        "mode": chain_output_mode(capture),
        "median_end_to_end_us": median_end_to_end_us(capture),
        "release_review": release_satisfied(capture),
        "gpu_events_available": gpu_events_available(capture),
    }


def residue_lifetime_metadata(capture: dict[str, Any] | None) -> dict[str, Any]:
    if capture is None:
        return {"available": False, "reason": "missing_residue_current_capture"}
    plan_packing = capture.get("plan_packing")
    plan_lowering = capture.get("plan_lowering")
    output_policy = capture.get("output_policy")
    requested_next_op = capture.get("requested_next_op")
    checks = {
        "device_current_output": isinstance(plan_packing, dict) and plan_packing.get("output_device_current") is True,
        "host_not_current": isinstance(plan_packing, dict) and plan_packing.get("output_host_current") is False,
        "rns_continuation_available": isinstance(plan_lowering, dict)
        and plan_lowering.get("rns_continuation_available") is True,
        "per_repeat_export_deferred": isinstance(output_policy, dict)
        and output_policy.get("per_repeat_logical_export") is False,
        "final_checksum_export_after_repeats": isinstance(output_policy, dict)
        and output_policy.get("final_checksum_export_after_repeats") is True,
        "next_op_rns_gemm": isinstance(requested_next_op, dict) and requested_next_op.get("resolved") == "rns-gemm",
    }
    missing = [key for key, value in checks.items() if not value]
    return {
        "available": not missing,
        "checks": checks,
        "reason": "available" if not missing else "incomplete_residue_current_lifetime_metadata",
        "missing": missing,
    }


def decision_for(
    residue_current: dict[str, Any] | None,
    final_output: dict[str, Any] | None,
    cpu_final_output: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if residue_current is None:
        blockers.append("missing_residue_current_capture")
    if final_output is None:
        blockers.append("missing_same_backend_final_output_capture")
    if cpu_final_output is None:
        blockers.append("missing_cpu_final_output_baseline")
    for label, capture in [
        ("residue_current", residue_current),
        ("final_output", final_output),
        ("cpu_final_output", cpu_final_output),
    ]:
        if capture is not None and not release_satisfied(capture):
            blockers.append(f"{label}_not_release_review")
    for label, capture in [("residue_current", residue_current), ("final_output", final_output)]:
        if gpu_events_available(capture) is False:
            blockers.append(f"missing_{label}_gpu_events")
    lifetime = residue_lifetime_metadata(residue_current)
    if lifetime.get("available") is False:
        blockers.append(str(lifetime.get("reason") or "missing_residue_lifetime_metadata"))
    if speedup(final_output, residue_current) is None:
        blockers.append("missing_final_vs_residue_timing")
    if speedup(cpu_final_output, final_output) is None:
        blockers.append("missing_cpu_vs_final_output_timing")
    if blockers:
        return "keep_experimental", blockers
    return "paired_residue_current_chain_ready", []


def row_for_backend(key: str, captures: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    final_outputs = [capture for capture in captures if chain_output_mode(capture) == "final_output"]
    residue_current = next(
        (capture for capture in captures if chain_output_mode(capture) == "residue_current" and backend_id(capture) == backend),
        None,
    )
    final_output = next(
        (capture for capture in final_outputs if backend_id(capture) == backend),
        None,
    )
    cpu_final_output = next((capture for capture in final_outputs if backend_id(capture) in REFERENCE_BACKENDS), None)
    decision, blockers = decision_for(residue_current, final_output, cpu_final_output)
    representative = residue_current or final_output or cpu_final_output or {}
    return {
        "contract_key": key,
        "backend": backend,
        "semantics": representative.get("semantics"),
        "shape": {
            "m": representative.get("m"),
            "n": representative.get("n"),
            "k": representative.get("k"),
        },
        "chain_length": representative.get("residue_chain_length"),
        "limb_count": representative.get("exact_wide_limb_count"),
        "residue_current": capture_summary(residue_current),
        "final_output": capture_summary(final_output),
        "cpu_final_output": capture_summary(cpu_final_output),
        "final_output_vs_residue_current_ratio": speedup(final_output, residue_current),
        "cpu_vs_final_output_speedup": speedup(cpu_final_output, final_output),
        "residue_lifetime": residue_lifetime_metadata(residue_current),
        "decision": decision,
        "blockers": blockers,
    }


def build_report_from_captures(captures: list[dict[str, Any]]) -> dict[str, Any]:
    chain_captures = [
        capture for capture in captures if is_exact_wide_chain(capture) and chain_output_mode(capture) is not None
    ]
    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for capture in chain_captures:
        by_contract[contract_key(capture)].append(capture)

    groups: list[dict[str, Any]] = []
    for key, items in sorted(by_contract.items(), key=lambda item: item[0]):
        backends = sorted(
            {
                backend_id(capture)
                for capture in items
                if backend_id(capture) not in REFERENCE_BACKENDS and chain_output_mode(capture) == "residue_current"
            }
            | {
                backend_id(capture)
                for capture in items
                if backend_id(capture) not in REFERENCE_BACKENDS and chain_output_mode(capture) == "final_output"
            }
        )
        rows = [row_for_backend(key, items, backend) for backend in backends]
        groups.append(
            {
                "contract_key": key,
                "capture_count": len(items),
                "rows": rows,
            }
        )

    rows = [row for group in groups for row in group["rows"]]
    return {
        "schema_version": 1,
        "policy": "exact_wide_residue_current_chain_pairs_are_benchmark_evidence_only",
        "capture_count": len(captures),
        "exact_wide_chain_capture_count": len(chain_captures),
        "summary": {
            "groups": len(groups),
            "paired_rows": len(rows),
            "ready_pairs": sum(1 for row in rows if row["decision"] == "paired_residue_current_chain_ready"),
            "experimental": sum(1 for row in rows if row["decision"] == "keep_experimental"),
            "missing_residue_current": sum(
                1 for row in rows if "missing_residue_current_capture" in row["blockers"]
            ),
            "missing_final_output": sum(
                1 for row in rows if "missing_same_backend_final_output_capture" in row["blockers"]
            ),
            "missing_cpu_baseline": sum(1 for row in rows if "missing_cpu_final_output_baseline" in row["blockers"]),
        },
        "groups": groups,
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    captures = [load_validated_capture(path) for path in expand_inputs(paths)]
    return build_report_from_captures(captures)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "none"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Exact-Wide Residue-Current Chain Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "| backend | semantics | shape | chain | limb | residue-current us | final-output us | final/residue ratio | CPU/final speedup | decision | blockers |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for group in report["groups"]:
        for row in group["rows"]:
            shape = row["shape"]
            residue = row.get("residue_current") or {}
            final = row.get("final_output") or {}
            blockers = ",".join(row.get("blockers") or []) or "none"
            lines.append(
                "| {backend} | {semantics} | {m}x{n}x{k} | {chain} | {limb} | {residue_us} | {final_us} | {ratio} | {cpu_speedup} | {decision} | {blockers} |".format(
                    backend=row.get("backend"),
                    semantics=row.get("semantics"),
                    m=shape.get("m"),
                    n=shape.get("n"),
                    k=shape.get("k"),
                    chain=row.get("chain_length"),
                    limb=row.get("limb_count"),
                    residue_us=fmt(residue.get("median_end_to_end_us")),
                    final_us=fmt(final.get("median_end_to_end_us")),
                    ratio=fmt(row.get("final_output_vs_residue_current_ratio")),
                    cpu_speedup=fmt(row.get("cpu_vs_final_output_speedup")),
                    decision=row.get("decision"),
                    blockers=blockers,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "exact-wide-chain-report.json"
    md_path = out_dir / "exact-wide-chain-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="*")
    parser.add_argument("--capture", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [*args.captures, *args.capture]
    if not paths:
        raise SystemExit("at least one capture file or directory is required")
    report = build_report(paths)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        paths_written = write_report(report, args.out_dir)
        print(paths_written["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
