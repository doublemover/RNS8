#!/usr/bin/env python3
"""Compare final-output RNS-chain captures under the same output contract."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import result_compare
from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "rns-chain-reports"
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}
REPORT_OUTPUT_NAMES = {"rns-chain-report.json"}
REUSE_PACK_MODES = {"prepacked_reuse", "prepacked_reuse_a", "prepacked_reuse_b"}
NORMALIZED_CONTRACT_EXCLUDE = {
    "reuse_packed_inputs",
    "pack_mode",
    "prepack_reuse_operands",
    "prepack_reuse_strategy",
}


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
    selected = capture.get("backend_selected")
    if selected is not None:
        return str(selected)
    requested = capture.get("backend_requested")
    return str(requested) if requested is not None else ""


def benchmark_execution_mode(capture: dict[str, Any]) -> str:
    value = capture.get("benchmark_execution_mode")
    if isinstance(value, str):
        return value
    metadata = capture.get("timing_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("benchmark_execution_mode"), str):
        return metadata["benchmark_execution_mode"]
    return "persistent_resident"


def is_final_output_chain(capture: dict[str, Any]) -> bool:
    return (
        benchmark_execution_mode(capture)
        in {"residue_chain_final_host_export", "residue_chain_independent_final_host_export"}
        and capture.get("residue_chain_final_export") is True
        and isinstance(capture.get("residue_chain_length"), int)
        and capture["residue_chain_length"] > 1
    )


def is_independent_export_repack_chain(capture: dict[str, Any]) -> bool:
    return (
        benchmark_execution_mode(capture) == "residue_chain_independent_final_host_export"
        or capture.get("residue_chain_independent_final_export") is True
    )


def chain_mode(capture: dict[str, Any]) -> str:
    return "independent_export_repack" if is_independent_export_repack_chain(capture) else "resident_final_output"


def pack_mode(capture: dict[str, Any]) -> str:
    return result_compare.capture_pack_mode(capture)


def is_reuse_capture(capture: dict[str, Any]) -> bool:
    return pack_mode(capture) in REUSE_PACK_MODES or capture.get("reuse_packed_inputs") is True


def release_satisfied(capture: dict[str, Any]) -> bool:
    return (
        isinstance(capture.get("warmups"), int)
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


def avg_prepack_setup_us(capture: dict[str, Any]) -> float | None:
    value = capture.get("avg_prepack_setup_us")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def setup_inclusive_median_per_repeat_us(capture: dict[str, Any]) -> float | None:
    median = median_end_to_end_us(capture)
    if median is None:
        return None
    if not is_reuse_capture(capture):
        return median
    repeats = capture.get("repeats")
    setup_us = avg_prepack_setup_us(capture)
    if not isinstance(repeats, int) or repeats <= 0 or setup_us is None:
        return None
    return median + setup_us / repeats


def break_even_repeats(baseline_us: float | None, reuse_steady_us: float | None, setup_us: float | None) -> int | None:
    if baseline_us is None or reuse_steady_us is None or setup_us is None:
        return None
    savings = baseline_us - reuse_steady_us
    if savings <= 0.0:
        return None
    return max(1, math.floor(setup_us / savings) + 1)


def speedup(numerator: dict[str, Any] | None, denominator: dict[str, Any]) -> float | None:
    numerator_us = setup_inclusive_median_per_repeat_us(numerator) if numerator is not None else None
    denominator_us = setup_inclusive_median_per_repeat_us(denominator)
    if numerator_us is None or denominator_us in (None, 0.0):
        return None
    return numerator_us / denominator_us


def _nested(capture: dict[str, Any], path: str) -> Any:
    value: Any = capture
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _normalized_contract_value(capture: dict[str, Any], key: str) -> Any:
    if key in NORMALIZED_CONTRACT_EXCLUDE:
        return None
    value = result_compare.contract_value(capture, key)
    if key == "residue_output_mode" and value is None:
        return "host_export"
    if key == "output_policy.destination_layout" and value is None:
        return "contiguous_row_major"
    if key == "output_policy.status_handling" and value is None:
        return "required"
    if key == "exact_output_contract.requested_final_output" and value is None:
        semantics = capture.get("semantics")
        if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
            return "exact_wide_limb_host"
        if semantics in {"bounded_i64", "bounded_u64"}:
            return "native_i64_u64_host"
    if key == "exact_output_contract.limb_count" and value is None:
        return capture.get("exact_wide_limb_count")
    return value


def normalized_contract_key(capture: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in result_compare.CONTRACT_KEYS:
        if key in NORMALIZED_CONTRACT_EXCLUDE:
            continue
        parts.append(f"{key}={_normalized_contract_value(capture, key)}")
    extra_fields = [
        "residue_chain_length",
        "residue_chain_final_export",
        "requested_next_op.resolved",
        "output_policy.per_repeat_logical_export",
        "output_policy.final_checksum_export_after_repeats",
    ]
    for key in extra_fields:
        parts.append(f"{key}={_nested(capture, key)}")
    return ";".join(parts)


def capture_summary(capture: dict[str, Any] | None) -> dict[str, Any] | None:
    if capture is None:
        return None
    return {
        "path": capture.get("_path"),
        "backend": backend_id(capture),
        "pack_mode": pack_mode(capture),
        "chain_mode": chain_mode(capture),
        "reuse": is_reuse_capture(capture),
        "median_end_to_end_us": median_end_to_end_us(capture),
        "setup_inclusive_median_per_repeat_us": setup_inclusive_median_per_repeat_us(capture),
        "avg_prepack_setup_us": avg_prepack_setup_us(capture),
        "release_review": release_satisfied(capture),
        "gpu_events_available": gpu_events_available(capture),
    }


def fastest(captures: list[dict[str, Any]]) -> dict[str, Any] | None:
    timed = [capture for capture in captures if setup_inclusive_median_per_repeat_us(capture) is not None]
    if not timed:
        return None
    return min(timed, key=lambda capture: setup_inclusive_median_per_repeat_us(capture) or float("inf"))


def source_identity_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    if not is_reuse_capture(capture):
        return {"available": None, "reason": "not_reuse_capture"}
    allocation = capture.get("device_allocation")
    if not isinstance(allocation, dict):
        return {"available": False, "reason": "missing_device_allocation_metadata"}
    setup_scope = allocation.get("setup_scope")
    source_version_inputs = allocation.get("source_version_inputs")
    available = (
        setup_scope == "persistent_plan_workspace_prepacked_reuse"
        and isinstance(source_version_inputs, str)
        and bool(source_version_inputs)
    )
    return {
        "available": available,
        "setup_scope": setup_scope,
        "source_version_inputs": source_version_inputs,
        "reason": "available" if available else "incomplete_prepack_source_identity_metadata",
    }


def decision_for(
    candidate: dict[str, Any],
    cpu_baseline: dict[str, Any] | None,
    same_backend_nonreuse: dict[str, Any] | None,
    best_nonreuse: dict[str, Any] | None,
    independent_baseline: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    reuse = is_reuse_capture(candidate)
    independent = is_independent_export_repack_chain(candidate)
    if backend_id(candidate) in REFERENCE_BACKENDS:
        return "cpu_independent_export_repack_baseline" if independent else "cpu_final_output_baseline", []
    if independent:
        if not release_satisfied(candidate):
            blockers.append("not_release_review")
        if gpu_events_available(candidate) is False:
            blockers.append("missing_candidate_gpu_events")
        if setup_inclusive_median_per_repeat_us(candidate) is None:
            blockers.append("missing_candidate_setup_inclusive_timing")
        if blockers:
            return "keep_experimental", blockers
        return "independent_export_repack_baseline", []
    if cpu_baseline is None:
        return "missing_baseline", ["missing_cpu_final_output_baseline"]
    if best_nonreuse is None:
        return "missing_baseline", ["missing_nonreuse_final_output_baseline"]
    if independent_baseline is None:
        return "keep_experimental", ["missing_same_backend_independent_export_repack_baseline"]
    if reuse and same_backend_nonreuse is None:
        return "missing_baseline", ["missing_same_backend_nonreuse_chain_baseline"]
    inputs = [candidate, cpu_baseline, best_nonreuse, same_backend_nonreuse, independent_baseline]
    if any(capture is not None and not release_satisfied(capture) for capture in inputs):
        blockers.append("not_release_review")
    for label, capture in [
        ("candidate", candidate),
        ("best_nonreuse", best_nonreuse),
        ("same_backend_nonreuse", same_backend_nonreuse),
    ]:
        events = gpu_events_available(capture)
        if events is False:
            blockers.append(f"missing_{label}_gpu_events")
    if setup_inclusive_median_per_repeat_us(candidate) is None:
        blockers.append("missing_candidate_setup_inclusive_timing")
    if speedup(cpu_baseline, candidate) is None:
        blockers.append("missing_cpu_speedup_timing")
    if speedup(independent_baseline, candidate) is None:
        blockers.append("missing_independent_export_repack_speedup_timing")
    if reuse:
        if avg_prepack_setup_us(candidate) is None:
            blockers.append("missing_prepack_setup_timing")
        identity = source_identity_metadata(candidate)
        if identity.get("available") is False:
            blockers.append(str(identity.get("reason") or "missing_source_identity_metadata"))
        if speedup(same_backend_nonreuse, candidate) is None:
            blockers.append("missing_same_backend_reuse_speedup")
    if blockers:
        return "keep_experimental", blockers
    if speedup(cpu_baseline, candidate) is not None and speedup(cpu_baseline, candidate) <= 1.0:
        return "deprioritize", ["chain_not_faster_than_cpu_final_output"]
    independent_speedup = speedup(independent_baseline, candidate)
    if independent_speedup is not None and independent_speedup <= 1.0:
        return "deprioritize", ["chain_not_faster_than_independent_export_repack"]
    if reuse:
        same_backend_speedup = speedup(same_backend_nonreuse, candidate)
        if same_backend_speedup is not None and same_backend_speedup <= 1.0:
            return "deprioritize", ["reuse_not_faster_than_same_backend_nonreuse_setup_inclusive"]
        best_nonreuse_speedup = speedup(best_nonreuse, candidate)
        if best_nonreuse_speedup is not None and best_nonreuse_speedup <= 1.0:
            return "deprioritize", ["reuse_not_faster_than_best_nonreuse_setup_inclusive"]
        if (
            break_even_repeats(
                median_end_to_end_us(same_backend_nonreuse),
                median_end_to_end_us(candidate),
                avg_prepack_setup_us(candidate),
            )
            is None
        ):
            return "deprioritize", ["reuse_never_breaks_even_vs_same_backend"]
        return "candidate_reuse_chain_win", []
    return "candidate_final_output_chain_win", []


def row_for_capture(
    capture: dict[str, Any],
    cpu_baseline: dict[str, Any] | None,
    same_backend_nonreuse: dict[str, Any] | None,
    best_nonreuse: dict[str, Any] | None,
    independent_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    decision, blockers = decision_for(capture, cpu_baseline, same_backend_nonreuse, best_nonreuse, independent_baseline)
    same_backend_break_even = (
        break_even_repeats(
            median_end_to_end_us(same_backend_nonreuse),
            median_end_to_end_us(capture),
            avg_prepack_setup_us(capture),
        )
        if is_reuse_capture(capture)
        else None
    )
    return {
        "path": capture.get("_path"),
        "backend": backend_id(capture),
        "semantics": capture.get("semantics"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "chain_length": capture.get("residue_chain_length"),
        "chain_mode": chain_mode(capture),
        "pack_mode": pack_mode(capture),
        "reuse": is_reuse_capture(capture),
        "prepack_reuse_operands": capture.get("prepack_reuse_operands"),
        "repeats": capture.get("repeats"),
        "median_end_to_end_us": median_end_to_end_us(capture),
        "setup_inclusive_median_per_repeat_us": setup_inclusive_median_per_repeat_us(capture),
        "avg_prepack_setup_us": avg_prepack_setup_us(capture),
        "release_review": release_satisfied(capture),
        "gpu_events_available": gpu_events_available(capture),
        "cpu_baseline": capture_summary(cpu_baseline),
        "same_backend_nonreuse": capture_summary(same_backend_nonreuse),
        "best_nonreuse": capture_summary(best_nonreuse),
        "independent_export_repack_baseline": capture_summary(independent_baseline),
        "speedup_vs_cpu": speedup(cpu_baseline, capture),
        "speedup_vs_same_backend_nonreuse": speedup(same_backend_nonreuse, capture),
        "speedup_vs_best_nonreuse": speedup(best_nonreuse, capture),
        "speedup_vs_independent_export_repack": speedup(independent_baseline, capture),
        "break_even_repeats_same_backend": same_backend_break_even,
        "source_identity": source_identity_metadata(capture),
        "decision": decision,
        "blockers": blockers,
    }


def build_report_from_captures(captures: list[dict[str, Any]]) -> dict[str, Any]:
    chain_captures = [capture for capture in captures if is_final_output_chain(capture)]
    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for capture in chain_captures:
        by_contract[normalized_contract_key(capture)].append(capture)

    groups = []
    for key, items in sorted(by_contract.items(), key=lambda item: item[0]):
        cpu_baseline = fastest([capture for capture in items if backend_id(capture) in REFERENCE_BACKENDS])
        resident_nonreuse = [
            capture
            for capture in items
            if not is_reuse_capture(capture) and not is_independent_export_repack_chain(capture)
        ]
        independent_nonreuse = [
            capture
            for capture in items
            if not is_reuse_capture(capture) and is_independent_export_repack_chain(capture)
        ]
        best_nonreuse = fastest([capture for capture in resident_nonreuse if backend_id(capture) not in REFERENCE_BACKENDS])
        best_independent = fastest(
            [capture for capture in independent_nonreuse if backend_id(capture) not in REFERENCE_BACKENDS]
        )
        rows = []
        for capture in sorted(
            items,
            key=lambda item: (backend_id(item), chain_mode(item), pack_mode(item), str(item.get("_path"))),
        ):
            same_backend_nonreuse = fastest(
                [
                    item
                    for item in resident_nonreuse
                    if backend_id(item) == backend_id(capture) and item is not capture
                ]
            )
            same_backend_independent = fastest(
                [item for item in independent_nonreuse if backend_id(item) == backend_id(capture)]
            )
            rows.append(
                row_for_capture(capture, cpu_baseline, same_backend_nonreuse, best_nonreuse, same_backend_independent)
            )
        groups.append(
            {
                "contract_key": key,
                "summary": {
                    "capture_count": len(items),
                    "cpu_baseline": capture_summary(cpu_baseline),
                    "best_nonreuse": capture_summary(best_nonreuse),
                    "best_independent_export_repack": capture_summary(best_independent),
                },
                "rows": sorted(rows, key=lambda row: row.get("setup_inclusive_median_per_repeat_us") or float("inf")),
            }
        )

    candidate_rows = [
        row
        for group in groups
        for row in group["rows"]
        if row["decision"] in {"candidate_final_output_chain_win", "candidate_reuse_chain_win"}
    ]
    comparison_rows = [
        row
        for group in groups
        for row in group["rows"]
        if row["backend"] not in REFERENCE_BACKENDS
        and row["decision"] != "independent_export_repack_baseline"
    ]
    independent_rows = [
        row
        for group in groups
        for row in group["rows"]
        if row["decision"] == "independent_export_repack_baseline"
    ]
    return {
        "schema_version": 1,
        "policy": "rns_chain_final_output_same_contract_evidence_only_no_public_resident_lifetime_api",
        "capture_count": len(captures),
        "final_output_chain_capture_count": len(chain_captures),
        "summary": {
            "groups": len(groups),
            "gpu_or_reuse_comparisons": len(comparison_rows),
            "candidate_wins": len(candidate_rows),
            "reuse_candidate_wins": sum(1 for row in candidate_rows if row["decision"] == "candidate_reuse_chain_win"),
            "independent_export_repack_baselines": len(independent_rows),
            "deprioritized": sum(1 for row in comparison_rows if row["decision"] == "deprioritize"),
            "experimental": sum(1 for row in comparison_rows if row["decision"] == "keep_experimental"),
            "missing_baselines": sum(1 for row in comparison_rows if row["decision"] == "missing_baseline"),
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
        "# RNS Chain Final-Output Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "| backend | semantics | shape | chain | chain mode | pack mode | setup-inclusive us | CPU us | independent us | independent speedup | same-backend speedup | CPU speedup | break-even repeats | decision | blockers |",
            "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for group in report["groups"]:
        for row in group["rows"]:
            if row["backend"] in REFERENCE_BACKENDS:
                continue
            shape = row["shape"]
            cpu = row.get("cpu_baseline") or {}
            independent = row.get("independent_export_repack_baseline") or {}
            blockers = ",".join(row.get("blockers") or []) or "none"
            lines.append(
                "| {backend} | {semantics} | {m}x{n}x{k} | {chain} | {chain_mode} | {mode} | {setup} | {cpu} | {independent} | {independent_speedup} | {same_speedup} | {cpu_speedup} | {break_even} | {decision} | {blockers} |".format(
                    backend=row.get("backend"),
                    semantics=row.get("semantics"),
                    m=shape.get("m"),
                    n=shape.get("n"),
                    k=shape.get("k"),
                    chain=row.get("chain_length"),
                    chain_mode=row.get("chain_mode"),
                    mode=row.get("pack_mode"),
                    setup=fmt(row.get("setup_inclusive_median_per_repeat_us")),
                    cpu=fmt(cpu.get("setup_inclusive_median_per_repeat_us")),
                    independent=fmt(independent.get("setup_inclusive_median_per_repeat_us")),
                    independent_speedup=fmt(row.get("speedup_vs_independent_export_repack")),
                    same_speedup=fmt(row.get("speedup_vs_same_backend_nonreuse")),
                    cpu_speedup=fmt(row.get("speedup_vs_cpu")),
                    break_even=fmt(row.get("break_even_repeats_same_backend")),
                    decision=row.get("decision"),
                    blockers=blockers,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rns-chain-report.json"
    md_path = out_dir / "rns-chain-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        action="append",
        help="capture file or directory; directories are searched recursively for JSON",
    )
    parser.add_argument("captures", type=Path, nargs="*")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = list(args.captures)
    if args.capture:
        paths.extend(args.capture)
    if not paths:
        raise SystemExit("at least one capture file or directory is required")
    report = build_report(paths)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.out_md)
    if not args.json and not args.out_json and not args.out_md:
        paths_written = write_report(report, args.out_dir)
        print(paths_written["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
