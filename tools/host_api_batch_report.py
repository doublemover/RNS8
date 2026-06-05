#!/usr/bin/env python3
"""Compare host API batch captures against independent-call baselines."""

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
PHASES = ["pack", "rns_gemm", "crt_export", "end_to_end"]
REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}


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


def benchmark_execution_mode(capture: dict[str, Any]) -> str:
    value = capture.get("benchmark_execution_mode")
    if isinstance(value, str):
        return value
    metadata = capture.get("timing_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("benchmark_execution_mode"), str):
        return metadata["benchmark_execution_mode"]
    return "persistent_resident"


def is_host_api_batch(capture: dict[str, Any]) -> bool:
    return benchmark_execution_mode(capture) == "benchmark_host_api_batch"


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


def host_batch_size(capture: dict[str, Any]) -> int:
    batch = capture.get("host_api_batch")
    if not isinstance(batch, dict):
        return 1
    value = batch.get("batch_size")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 1


def phase_comparison(baseline: dict[str, Any], batch: dict[str, Any], phase: str) -> dict[str, Any]:
    batch_size = host_batch_size(batch)
    baseline_median = timing_summary_value(baseline, phase, "median")
    batch_aggregate_median = timing_summary_value(batch, phase, "median")
    batch_per_task_median = (
        batch_aggregate_median / batch_size if batch_aggregate_median is not None and batch_size > 0 else None
    )
    speedup = (
        baseline_median / batch_per_task_median
        if baseline_median is not None and batch_per_task_median not in (None, 0.0)
        else None
    )
    return {
        "baseline_median_us": baseline_median,
        "batch_aggregate_median_us": batch_aggregate_median,
        "batch_per_task_median_us": batch_per_task_median,
        "per_task_speedup_vs_independent": speedup,
    }


def contract_key(capture: dict[str, Any]) -> str:
    parts = []
    for key in result_compare.CONTRACT_KEYS:
        value = result_compare.contract_value(capture, key)
        if key == "prepack_reuse_operands" and value is None:
            value = []
        parts.append(f"{key}={value}")
    return ";".join(parts)


def select_independent_baseline(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        capture
        for capture in candidates
        if not is_host_api_batch(capture) and timing_summary_value(capture, "end_to_end", "median") is not None
    ]
    if not valid:
        return None
    return min(valid, key=lambda capture: timing_summary_value(capture, "end_to_end", "median") or float("inf"))


def decision_for(
    same_backend_baseline: dict[str, Any] | None,
    best_independent_baseline: dict[str, Any] | None,
    batch: dict[str, Any],
    same_backend_phases: dict[str, dict[str, Any]],
    speedup_vs_best_independent: float | None,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if same_backend_baseline is None:
        return "missing_baseline", ["missing_independent_backend_baseline"]
    if best_independent_baseline is None:
        return "missing_baseline", ["missing_independent_contract_baseline"]
    if (
        not release_satisfied(same_backend_baseline)
        or not release_satisfied(best_independent_baseline)
        or not release_satisfied(batch)
    ):
        blockers.append("not_release_review")
    if gpu_backend(batch):
        if not gpu_events_available(batch):
            blockers.append("missing_host_batch_gpu_events")
        if gpu_backend(same_backend_baseline) and not gpu_events_available(same_backend_baseline):
            blockers.append("missing_independent_gpu_events")
        if gpu_backend(best_independent_baseline) and not gpu_events_available(best_independent_baseline):
            blockers.append("missing_best_independent_gpu_events")
    same_backend_speedup = same_backend_phases["end_to_end"].get("per_task_speedup_vs_independent")
    if same_backend_speedup is None or speedup_vs_best_independent is None:
        blockers.append("missing_end_to_end_timing")
    if blockers:
        return "keep_experimental", blockers
    if same_backend_speedup is not None and same_backend_speedup <= 1.0:
        return "deprioritize", ["host_batch_not_faster_than_same_backend_per_task"]
    if speedup_vs_best_independent is not None and speedup_vs_best_independent <= 1.0:
        return "deprioritize", ["host_batch_not_faster_than_best_independent_per_task"]
    return "candidate_win", []


def compare_host_batches(captures: list[dict[str, Any]]) -> dict[str, Any]:
    by_contract_backend: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    host_batches = []
    for capture in captures:
        key = contract_key(capture)
        backend = backend_id(capture)
        by_contract_backend[(key, backend)].append(capture)
        by_contract[key].append(capture)
        if is_host_api_batch(capture):
            host_batches.append(capture)

    comparisons = []
    for batch in sorted(host_batches, key=lambda item: str(item.get("_path"))):
        key = contract_key(batch)
        backend = backend_id(batch)
        same_backend_baseline = select_independent_baseline(by_contract_backend.get((key, backend), []))
        best_independent_baseline = select_independent_baseline(by_contract.get(key, []))
        phases = {
            phase: phase_comparison(same_backend_baseline, batch, phase)
            if same_backend_baseline is not None
            else phase_comparison({}, batch, phase)
            for phase in PHASES
        }
        batch_e2e_per_task = phases["end_to_end"].get("batch_per_task_median_us")
        best_independent_e2e = (
            timing_summary_value(best_independent_baseline, "end_to_end", "median")
            if best_independent_baseline is not None
            else None
        )
        speedup_vs_best_independent = (
            best_independent_e2e / batch_e2e_per_task
            if best_independent_e2e is not None and batch_e2e_per_task not in (None, 0.0)
            else None
        )
        decision, blockers = decision_for(
            same_backend_baseline,
            best_independent_baseline,
            batch,
            phases,
            speedup_vs_best_independent,
        )
        comparisons.append(
            {
                "contract_key": key,
                "backend": backend,
                "semantics": batch.get("semantics"),
                "finite_modulus": batch.get("finite_modulus"),
                "shape": {"m": batch.get("m"), "n": batch.get("n"), "k": batch.get("k")},
                "host_api_batch_size": host_batch_size(batch),
                "batch_capture": batch.get("_path"),
                "same_backend_independent_capture": same_backend_baseline.get("_path")
                if same_backend_baseline is not None
                else None,
                "best_independent_capture": best_independent_baseline.get("_path")
                if best_independent_baseline is not None
                else None,
                "best_independent_backend": backend_id(best_independent_baseline)
                if best_independent_baseline is not None
                else None,
                "best_independent_median_end_to_end_us": best_independent_e2e,
                "speedup_vs_best_independent": speedup_vs_best_independent,
                "release_review_pair": bool(
                    same_backend_baseline is not None
                    and best_independent_baseline is not None
                    and release_satisfied(same_backend_baseline)
                    and release_satisfied(best_independent_baseline)
                    and release_satisfied(batch)
                ),
                "host_batch_gpu_events": gpu_events_available(batch) if gpu_backend(batch) else None,
                "same_backend_independent_gpu_events": gpu_events_available(same_backend_baseline)
                if same_backend_baseline is not None and gpu_backend(same_backend_baseline)
                else None,
                "best_independent_gpu_events": gpu_events_available(best_independent_baseline)
                if best_independent_baseline is not None and gpu_backend(best_independent_baseline)
                else None,
                "phases": phases,
                "decision": decision,
                "blockers": blockers,
            }
        )

    summary = {
        "host_batch_captures": len(host_batches),
        "comparisons": len(comparisons),
        "candidate_wins": sum(1 for item in comparisons if item["decision"] == "candidate_win"),
        "deprioritized": sum(1 for item in comparisons if item["decision"] == "deprioritize"),
        "experimental": sum(1 for item in comparisons if item["decision"] == "keep_experimental"),
        "missing_baselines": sum(1 for item in comparisons if item["decision"] == "missing_baseline"),
    }
    return {"summary": summary, "comparisons": comparisons}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Host API Batch Comparison Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "| backend | semantics | shape | batch | same-backend independent us | batch per-task us | same-backend speedup | best independent | workload speedup | decision | blockers |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---|---|",
        ]
    )
    for item in report["comparisons"]:
        shape = item["shape"]
        phase = item["phases"]["end_to_end"]
        speedup = phase.get("per_task_speedup_vs_independent")
        blockers = ",".join(item.get("blockers") or []) or "none"
        best_backend = item.get("best_independent_backend")
        best_time = item.get("best_independent_median_end_to_end_us")
        best_text = f"{best_backend} {best_time}" if best_backend is not None else "none"
        workload_speedup = item.get("speedup_vs_best_independent")
        lines.append(
            "| {backend} | {semantics} | {m}x{n}x{k} | {batch} | {base} | {per_task} | {speedup} | {best} | {workload_speedup} | {decision} | {blockers} |".format(
                backend=item.get("backend"),
                semantics=item.get("semantics"),
                m=shape.get("m"),
                n=shape.get("n"),
                k=shape.get("k"),
                batch=item.get("host_api_batch_size"),
                base=phase.get("baseline_median_us"),
                per_task=phase.get("batch_per_task_median_us"),
                speedup=None if speedup is None else round(float(speedup), 4),
                best=best_text,
                workload_speedup=None if workload_speedup is None else round(float(workload_speedup), 4),
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
    report = compare_host_batches(captures)
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
