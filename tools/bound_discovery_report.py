#!/usr/bin/env python3
"""Compare bound-discovery captures against static global-bound baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import result_compare
from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}
BOUNDED_SEMANTICS = {"bounded_i64", "bounded_u64"}
PHASES = ["pack", "rns_gemm", "crt_export", "end_to_end"]


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


def release_satisfied(capture: dict[str, Any]) -> bool:
    return (
        isinstance(capture.get("warmups"), int)
        and isinstance(capture.get("repeats"), int)
        and capture["warmups"] >= RELEASE_MIN_WARMUPS
        and capture["repeats"] >= RELEASE_MIN_REPEATS
    )


def gpu_backend(capture: dict[str, Any]) -> bool:
    return backend_id(capture) not in REFERENCE_BACKENDS


def gpu_events_available(capture: dict[str, Any]) -> bool:
    metadata = capture.get("timing_metadata")
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("gpu_event_timing") is True
        and metadata.get("gpu_event_timing_status") == "available"
        and isinstance(metadata.get("gpu_event_phase_order"), list)
        and bool(metadata.get("gpu_event_timing_source"))
    )


def timing_summary_value(capture: dict[str, Any], phase: str, statistic: str) -> float | None:
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    phase_summary = summary.get(phase)
    if not isinstance(phase_summary, dict):
        return None
    value = phase_summary.get(statistic)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def numeric_value(capture: dict[str, Any], key: str) -> float | None:
    value = capture.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def proof_counts(capture: dict[str, Any]) -> dict[str, int | None]:
    schedule = capture.get("schedule_metadata")
    if not isinstance(schedule, dict):
        return {
            "zero_a_rows": None,
            "zero_b_cols": None,
            "zero_row_col_products": None,
        }
    return {
        "zero_a_rows": schedule.get("zero_a_row_proof_count")
        if isinstance(schedule.get("zero_a_row_proof_count"), int)
        else None,
        "zero_b_cols": schedule.get("zero_b_col_proof_count")
        if isinstance(schedule.get("zero_b_col_proof_count"), int)
        else None,
        "zero_row_col_products": schedule.get("zero_row_col_product_count")
        if isinstance(schedule.get("zero_row_col_product_count"), int)
        else None,
    }


def has_zero_proof_coverage(capture: dict[str, Any]) -> bool:
    counts = proof_counts(capture)
    return any((value or 0) > 0 for value in counts.values())


def bound_variant(capture: dict[str, Any]) -> str | None:
    if capture.get("semantics") not in BOUNDED_SEMANTICS:
        return None
    bound_mode = capture.get("bound_mode", "global")
    bound_source = result_compare.capture_bound_source(capture)
    tile_bounds = capture.get("tile_bounds_u64")
    if bound_mode == "global" and bound_source == "static_profile":
        return "static_global"
    if bound_mode == "global" and bound_source == "input_scan":
        return "input_scan_global"
    if bound_mode == "per_tile" and bound_source == "input_scan" and isinstance(tile_bounds, dict):
        return "proof_mask_per_tile" if has_zero_proof_coverage(capture) else "tile_bound_per_tile"
    return None


def setup_scan_us(capture: dict[str, Any]) -> float | None:
    variant = bound_variant(capture)
    if variant == "static_global":
        return 0.0
    if variant == "input_scan_global":
        return numeric_value(capture, "avg_global_bound_scan_us")
    if variant in {"proof_mask_per_tile", "tile_bound_per_tile"}:
        return numeric_value(capture, "avg_tile_bound_scan_us")
    return None


def setup_inclusive_median_us(capture: dict[str, Any]) -> float | None:
    median = timing_summary_value(capture, "end_to_end", "median")
    setup = setup_scan_us(capture)
    if median is None or setup is None:
        return None
    return median + setup


def phase_comparison(baseline: dict[str, Any] | None, candidate: dict[str, Any], phase: str) -> dict[str, Any]:
    baseline_median = timing_summary_value(baseline, phase, "median") if baseline is not None else None
    candidate_median = timing_summary_value(candidate, phase, "median")
    speedup = (
        baseline_median / candidate_median
        if baseline_median is not None and candidate_median not in (None, 0.0)
        else None
    )
    return {
        "baseline_median_us": baseline_median,
        "candidate_median_us": candidate_median,
        "candidate_speedup_vs_baseline": speedup,
    }


def workload_key(capture: dict[str, Any]) -> str:
    parts = {
        "semantics": capture.get("semantics"),
        "m": capture.get("m"),
        "n": capture.get("n"),
        "k": capture.get("k"),
        "input_distribution": capture.get("input_distribution"),
        "seed": capture.get("seed"),
        "layout": capture.get("layout"),
        "output_logical_ld": capture.get("output_logical_ld", capture.get("n")),
        "output_ld_padding": capture.get("output_ld_padding", 0),
        "prefix": capture.get("prefix"),
        "requested_max_prefix": capture.get("requested_max_prefix"),
        "pack_mode": capture.get("pack_mode", "per_repeat_repack"),
        "residue_output_mode": capture.get("residue_output_mode", "host_export"),
    }
    return ";".join(f"{key}={value}" for key, value in parts.items())


def select_best_static(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        capture
        for capture in candidates
        if bound_variant(capture) == "static_global" and setup_inclusive_median_us(capture) is not None
    ]
    if not valid:
        return None
    return min(valid, key=lambda capture: setup_inclusive_median_us(capture) or float("inf"))


def decision_for(
    same_backend_static: dict[str, Any] | None,
    best_static: dict[str, Any] | None,
    candidate: dict[str, Any],
    speedup_vs_same_backend_static: float | None,
    speedup_vs_best_static: float | None,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if same_backend_static is None:
        return "missing_baseline", ["missing_same_backend_static_global_baseline"]
    if best_static is None:
        return "missing_baseline", ["missing_contract_static_global_baseline"]
    if not release_satisfied(candidate) or not release_satisfied(same_backend_static) or not release_satisfied(best_static):
        blockers.append("not_release_review")
    variant = bound_variant(candidate)
    if variant == "input_scan_global" and setup_scan_us(candidate) is None:
        blockers.append("missing_global_bound_scan_timing")
    if variant in {"proof_mask_per_tile", "tile_bound_per_tile"}:
        if setup_scan_us(candidate) is None:
            blockers.append("missing_tile_bound_scan_timing")
        if not isinstance(candidate.get("tile_bounds_u64"), dict):
            blockers.append("missing_tile_bounds_metadata")
        if variant != "proof_mask_per_tile":
            blockers.append("missing_zero_row_col_proof_coverage")
    if gpu_backend(candidate) and not gpu_events_available(candidate):
        blockers.append("missing_candidate_gpu_events")
    if gpu_backend(same_backend_static) and not gpu_events_available(same_backend_static):
        blockers.append("missing_same_backend_static_gpu_events")
    if gpu_backend(best_static) and not gpu_events_available(best_static):
        blockers.append("missing_best_static_gpu_events")
    if speedup_vs_same_backend_static is None or speedup_vs_best_static is None:
        blockers.append("missing_setup_inclusive_timing")
    if blockers:
        return "keep_experimental", blockers
    if speedup_vs_same_backend_static is not None and speedup_vs_same_backend_static <= 1.0:
        return "deprioritize", ["candidate_not_faster_than_same_backend_static_setup_inclusive"]
    if speedup_vs_best_static is not None and speedup_vs_best_static <= 1.0:
        return "deprioritize", ["candidate_not_faster_than_best_static_setup_inclusive"]
    return "candidate_win", []


def compare_bound_discovery(captures: list[dict[str, Any]]) -> dict[str, Any]:
    bounded = [capture for capture in captures if bound_variant(capture) is not None]
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_workload_backend: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []
    for capture in bounded:
        key = workload_key(capture)
        backend = backend_id(capture)
        by_workload[key].append(capture)
        by_workload_backend[(key, backend)].append(capture)
        if bound_variant(capture) != "static_global":
            candidates.append(capture)

    comparisons: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: str(item.get("_path"))):
        key = workload_key(candidate)
        backend = backend_id(candidate)
        same_backend_static = select_best_static(by_workload_backend.get((key, backend), []))
        best_static = select_best_static(by_workload.get(key, []))
        candidate_setup_inclusive = setup_inclusive_median_us(candidate)
        same_backend_setup_inclusive = (
            setup_inclusive_median_us(same_backend_static) if same_backend_static is not None else None
        )
        best_static_setup_inclusive = setup_inclusive_median_us(best_static) if best_static is not None else None
        speedup_vs_same_backend_static = (
            same_backend_setup_inclusive / candidate_setup_inclusive
            if same_backend_setup_inclusive is not None and candidate_setup_inclusive not in (None, 0.0)
            else None
        )
        speedup_vs_best_static = (
            best_static_setup_inclusive / candidate_setup_inclusive
            if best_static_setup_inclusive is not None and candidate_setup_inclusive not in (None, 0.0)
            else None
        )
        decision, blockers = decision_for(
            same_backend_static,
            best_static,
            candidate,
            speedup_vs_same_backend_static,
            speedup_vs_best_static,
        )
        counts = proof_counts(candidate)
        comparisons.append(
            {
                "workload_key": key,
                "variant": bound_variant(candidate),
                "backend": backend,
                "semantics": candidate.get("semantics"),
                "shape": {
                    "m": candidate.get("m"),
                    "n": candidate.get("n"),
                    "k": candidate.get("k"),
                },
                "input_distribution": candidate.get("input_distribution"),
                "candidate_capture": candidate.get("_path"),
                "same_backend_static_capture": same_backend_static.get("_path")
                if same_backend_static is not None
                else None,
                "best_static_capture": best_static.get("_path") if best_static is not None else None,
                "best_static_backend": backend_id(best_static) if best_static is not None else None,
                "candidate_setup_scan_us": setup_scan_us(candidate),
                "candidate_median_end_to_end_us": timing_summary_value(candidate, "end_to_end", "median"),
                "candidate_setup_inclusive_median_us": candidate_setup_inclusive,
                "same_backend_static_setup_inclusive_median_us": same_backend_setup_inclusive,
                "best_static_setup_inclusive_median_us": best_static_setup_inclusive,
                "speedup_vs_same_backend_static_setup_inclusive": speedup_vs_same_backend_static,
                "speedup_vs_best_static_setup_inclusive": speedup_vs_best_static,
                "release_review_triplet": bool(
                    same_backend_static is not None
                    and best_static is not None
                    and release_satisfied(candidate)
                    and release_satisfied(same_backend_static)
                    and release_satisfied(best_static)
                ),
                "candidate_gpu_events": gpu_events_available(candidate) if gpu_backend(candidate) else None,
                "same_backend_static_gpu_events": gpu_events_available(same_backend_static)
                if same_backend_static is not None and gpu_backend(same_backend_static)
                else None,
                "best_static_gpu_events": gpu_events_available(best_static)
                if best_static is not None and gpu_backend(best_static)
                else None,
                "proof_counts": counts,
                "phases": {
                    phase: phase_comparison(same_backend_static, candidate, phase)
                    for phase in PHASES
                },
                "decision": decision,
                "blockers": blockers,
            }
        )

    summary = {
        "bounded_captures": len(bounded),
        "static_global_captures": sum(1 for item in bounded if bound_variant(item) == "static_global"),
        "input_scan_global_captures": sum(1 for item in bounded if bound_variant(item) == "input_scan_global"),
        "proof_mask_per_tile_captures": sum(1 for item in bounded if bound_variant(item) == "proof_mask_per_tile"),
        "tile_bound_per_tile_without_proofs": sum(
            1 for item in bounded if bound_variant(item) == "tile_bound_per_tile"
        ),
        "comparisons": len(comparisons),
        "candidate_wins": sum(1 for item in comparisons if item["decision"] == "candidate_win"),
        "deprioritized": sum(1 for item in comparisons if item["decision"] == "deprioritize"),
        "experimental": sum(1 for item in comparisons if item["decision"] == "keep_experimental"),
        "missing_baselines": sum(1 for item in comparisons if item["decision"] == "missing_baseline"),
    }
    return {"summary": summary, "comparisons": comparisons}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Bound Discovery Comparison Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "| variant | backend | semantics | shape | setup scan us | candidate setup-inclusive us | same-backend static us | speedup vs same backend | best static | speedup vs best static | decision | blockers |",
            "|---|---|---|---|---:|---:|---:|---:|---|---:|---|---|",
        ]
    )
    for item in report["comparisons"]:
        shape = item["shape"]
        blockers = ",".join(item.get("blockers") or []) or "none"
        best_backend = item.get("best_static_backend")
        best_time = item.get("best_static_setup_inclusive_median_us")
        best_text = f"{best_backend} {best_time}" if best_backend is not None else "none"
        speedup_same = item.get("speedup_vs_same_backend_static_setup_inclusive")
        speedup_best = item.get("speedup_vs_best_static_setup_inclusive")
        lines.append(
            "| {variant} | {backend} | {semantics} | {m}x{n}x{k} | {setup} | {candidate} | {same} | {speedup_same} | {best} | {speedup_best} | {decision} | {blockers} |".format(
                variant=item.get("variant"),
                backend=item.get("backend"),
                semantics=item.get("semantics"),
                m=shape.get("m"),
                n=shape.get("n"),
                k=shape.get("k"),
                setup=item.get("candidate_setup_scan_us"),
                candidate=item.get("candidate_setup_inclusive_median_us"),
                same=item.get("same_backend_static_setup_inclusive_median_us"),
                speedup_same=None if speedup_same is None else round(float(speedup_same), 4),
                best=best_text,
                speedup_best=None if speedup_best is None else round(float(speedup_best), 4),
                decision=item.get("decision"),
                blockers=blockers,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        action="append",
        required=True,
        help="capture file or directory; directories are searched recursively for JSON",
    )
    parser.add_argument("--out-json", type=Path, help="write JSON report")
    parser.add_argument("--out-md", type=Path, help="write Markdown report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = expand_inputs(args.capture)
    captures = [load_validated_capture(path) for path in paths]
    report = compare_bound_discovery(captures)
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
