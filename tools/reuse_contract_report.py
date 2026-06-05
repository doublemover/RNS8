#!/usr/bin/env python3
"""Compare prepacked-reuse captures against non-reuse workload baselines."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import result_compare
from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
REUSE_PACK_MODES = {"prepacked_reuse", "prepacked_reuse_a", "prepacked_reuse_b"}
REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}
NORMALIZED_CONTRACT_EXCLUDE = {
    "reuse_packed_inputs",
    "pack_mode",
    "prepack_reuse_operands",
    "prepack_reuse_strategy",
}
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
    if selected is not None:
        return str(selected)
    requested = capture.get("backend_requested")
    return str(requested) if requested is not None else ""


def pack_mode(capture: dict[str, Any]) -> str:
    return result_compare.capture_pack_mode(capture)


def is_reuse_capture(capture: dict[str, Any]) -> bool:
    return pack_mode(capture) in REUSE_PACK_MODES or capture.get("reuse_packed_inputs") is True


def is_nonreuse_capture(capture: dict[str, Any]) -> bool:
    return not is_reuse_capture(capture)


def release_satisfied(capture: dict[str, Any]) -> bool:
    return (
        isinstance(capture.get("warmups"), int)
        and isinstance(capture.get("repeats"), int)
        and capture["warmups"] >= RELEASE_MIN_WARMUPS
        and capture["repeats"] >= RELEASE_MIN_REPEATS
    )


def gpu_backend(capture: dict[str, Any] | None) -> bool:
    if capture is None:
        return False
    return backend_id(capture) not in REFERENCE_BACKENDS


def gpu_events_available(capture: dict[str, Any] | None) -> bool:
    if capture is None:
        return False
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


def prepack_setup_us(capture: dict[str, Any]) -> float | None:
    value = capture.get("avg_prepack_setup_us")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def normalized_contract_key(capture: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in result_compare.CONTRACT_KEYS:
        if key in NORMALIZED_CONTRACT_EXCLUDE:
            continue
        value = result_compare.contract_value(capture, key)
        parts.append(f"{key}={value}")
    return ";".join(parts)


def select_fastest(captures: list[dict[str, Any]]) -> dict[str, Any] | None:
    timed = [capture for capture in captures if timing_summary_value(capture, "end_to_end", "median") is not None]
    if not timed:
        return None
    return min(timed, key=lambda capture: timing_summary_value(capture, "end_to_end", "median") or float("inf"))


def break_even_repeats(baseline_us: float | None, reuse_steady_us: float | None, setup_us: float | None) -> int | None:
    if baseline_us is None or reuse_steady_us is None or setup_us is None:
        return None
    savings = baseline_us - reuse_steady_us
    if savings <= 0.0:
        return None
    return max(1, math.floor(setup_us / savings) + 1)


def setup_inclusive_per_repeat(capture: dict[str, Any], phase: str) -> float | None:
    phase_median = timing_summary_value(capture, phase, "median")
    if phase_median is None:
        return None
    repeats = capture.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        return None
    setup = prepack_setup_us(capture)
    if setup is None:
        return None
    return phase_median + setup / repeats if phase == "end_to_end" else phase_median


def phase_comparison(baseline: dict[str, Any] | None, reuse: dict[str, Any], phase: str) -> dict[str, Any]:
    baseline_median = timing_summary_value(baseline, phase, "median")
    reuse_median = timing_summary_value(reuse, phase, "median")
    setup_inclusive = setup_inclusive_per_repeat(reuse, phase)
    speedup = (
        baseline_median / setup_inclusive
        if baseline_median is not None and setup_inclusive not in (None, 0.0)
        else None
    )
    steady_speedup = (
        baseline_median / reuse_median
        if baseline_median is not None and reuse_median not in (None, 0.0)
        else None
    )
    return {
        "baseline_median_us": baseline_median,
        "reuse_steady_median_us": reuse_median,
        "reuse_setup_inclusive_per_repeat_us": setup_inclusive,
        "steady_state_speedup": steady_speedup,
        "setup_inclusive_speedup": speedup,
    }


def source_identity_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    allocation = capture.get("device_allocation")
    if not isinstance(allocation, dict):
        return {
            "available": False,
            "setup_scope": None,
            "source_version_inputs": None,
            "reason": "missing_device_allocation_metadata",
        }
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
    same_backend_baseline: dict[str, Any] | None,
    best_nonreuse_baseline: dict[str, Any] | None,
    reuse: dict[str, Any],
    same_backend_phases: dict[str, dict[str, Any]],
    speedup_vs_best_nonreuse: float | None,
    break_even_same_backend: int | None,
    source_identity: dict[str, Any],
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if same_backend_baseline is None:
        return "missing_baseline", ["missing_same_backend_nonreuse_baseline"]
    if best_nonreuse_baseline is None:
        return "missing_baseline", ["missing_best_nonreuse_contract_baseline"]
    if not release_satisfied(reuse):
        blockers.append("reuse_capture_not_release_review")
    if not release_satisfied(same_backend_baseline):
        blockers.append("same_backend_baseline_not_release_review")
    if not release_satisfied(best_nonreuse_baseline):
        blockers.append("best_nonreuse_baseline_not_release_review")
    if prepack_setup_us(reuse) is None:
        blockers.append("missing_prepack_setup_timing")
    if not source_identity.get("available"):
        blockers.append(str(source_identity.get("reason") or "missing_source_identity_metadata"))
    if gpu_backend(reuse) and not gpu_events_available(reuse):
        blockers.append("missing_reuse_gpu_events")
    if gpu_backend(same_backend_baseline) and not gpu_events_available(same_backend_baseline):
        blockers.append("missing_same_backend_gpu_events")
    if gpu_backend(best_nonreuse_baseline) and not gpu_events_available(best_nonreuse_baseline):
        blockers.append("missing_best_nonreuse_gpu_events")

    end_to_end = same_backend_phases["end_to_end"]
    same_backend_speedup = end_to_end.get("setup_inclusive_speedup")
    steady_state_speedup = end_to_end.get("steady_state_speedup")
    repeats = reuse.get("repeats")
    if same_backend_speedup is None or speedup_vs_best_nonreuse is None:
        blockers.append("missing_end_to_end_timing")
    if not isinstance(repeats, int) or repeats <= 0:
        blockers.append("invalid_repeat_count")
    if blockers:
        return "keep_experimental", blockers
    if steady_state_speedup is not None and steady_state_speedup <= 1.0:
        return "deprioritize", ["reuse_not_faster_than_same_backend_steady_state"]
    if break_even_same_backend is None:
        return "deprioritize", ["reuse_never_breaks_even_vs_same_backend"]
    if isinstance(repeats, int) and repeats < break_even_same_backend:
        return "deprioritize", ["repeat_count_below_same_backend_break_even"]
    if same_backend_speedup is not None and same_backend_speedup <= 1.0:
        return "deprioritize", ["reuse_not_faster_than_same_backend_setup_inclusive"]
    if speedup_vs_best_nonreuse is not None and speedup_vs_best_nonreuse <= 1.0:
        return "deprioritize", ["reuse_not_faster_than_best_nonreuse_setup_inclusive"]
    return "candidate_workload_win", []


def compare_reuse_contracts(captures: list[dict[str, Any]]) -> dict[str, Any]:
    nonreuse_by_contract_backend: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    nonreuse_by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reuse_captures: list[dict[str, Any]] = []
    for capture in captures:
        key = normalized_contract_key(capture)
        if is_reuse_capture(capture):
            reuse_captures.append(capture)
        elif is_nonreuse_capture(capture):
            nonreuse_by_contract_backend[(key, backend_id(capture))].append(capture)
            nonreuse_by_contract[key].append(capture)

    comparisons: list[dict[str, Any]] = []
    for reuse in sorted(reuse_captures, key=lambda item: str(item.get("_path"))):
        key = normalized_contract_key(reuse)
        backend = backend_id(reuse)
        same_backend_baseline = select_fastest(nonreuse_by_contract_backend.get((key, backend), []))
        best_nonreuse_baseline = select_fastest(nonreuse_by_contract.get(key, []))
        same_backend_phases = {phase: phase_comparison(same_backend_baseline, reuse, phase) for phase in PHASES}
        best_nonreuse_per_repeat = setup_inclusive_per_repeat(reuse, "end_to_end")
        best_nonreuse_median = timing_summary_value(best_nonreuse_baseline, "end_to_end", "median")
        speedup_vs_best_nonreuse = (
            best_nonreuse_median / best_nonreuse_per_repeat
            if best_nonreuse_median is not None and best_nonreuse_per_repeat not in (None, 0.0)
            else None
        )
        break_even_same_backend = break_even_repeats(
            timing_summary_value(same_backend_baseline, "end_to_end", "median"),
            timing_summary_value(reuse, "end_to_end", "median"),
            prepack_setup_us(reuse),
        )
        break_even_best_nonreuse = break_even_repeats(
            best_nonreuse_median,
            timing_summary_value(reuse, "end_to_end", "median"),
            prepack_setup_us(reuse),
        )
        source_identity = source_identity_metadata(reuse)
        decision, blockers = decision_for(
            same_backend_baseline,
            best_nonreuse_baseline,
            reuse,
            same_backend_phases,
            speedup_vs_best_nonreuse,
            break_even_same_backend,
            source_identity,
        )
        comparisons.append(
            {
                "contract_key": key,
                "backend": backend,
                "semantics": reuse.get("semantics"),
                "finite_modulus": reuse.get("finite_modulus"),
                "shape": {"m": reuse.get("m"), "n": reuse.get("n"), "k": reuse.get("k")},
                "pack_mode": pack_mode(reuse),
                "prepack_reuse_operands": reuse.get("prepack_reuse_operands"),
                "prepack_reuse_strategy": reuse.get("prepack_reuse_strategy"),
                "repeats": reuse.get("repeats"),
                "prepack_setup_us": prepack_setup_us(reuse),
                "reuse_capture": reuse.get("_path"),
                "same_backend_nonreuse_capture": same_backend_baseline.get("_path")
                if same_backend_baseline is not None
                else None,
                "best_nonreuse_capture": best_nonreuse_baseline.get("_path")
                if best_nonreuse_baseline is not None
                else None,
                "best_nonreuse_backend": backend_id(best_nonreuse_baseline)
                if best_nonreuse_baseline is not None
                else None,
                "best_nonreuse_median_end_to_end_us": best_nonreuse_median,
                "speedup_vs_best_nonreuse_setup_inclusive": speedup_vs_best_nonreuse,
                "break_even_repeats_same_backend": break_even_same_backend,
                "break_even_repeats_best_nonreuse": break_even_best_nonreuse,
                "source_identity": source_identity,
                "release_review_pair": bool(
                    same_backend_baseline is not None
                    and best_nonreuse_baseline is not None
                    and release_satisfied(reuse)
                    and release_satisfied(same_backend_baseline)
                    and release_satisfied(best_nonreuse_baseline)
                ),
                "reuse_gpu_events": gpu_events_available(reuse) if gpu_backend(reuse) else None,
                "same_backend_nonreuse_gpu_events": gpu_events_available(same_backend_baseline)
                if gpu_backend(same_backend_baseline)
                else None,
                "best_nonreuse_gpu_events": gpu_events_available(best_nonreuse_baseline)
                if gpu_backend(best_nonreuse_baseline)
                else None,
                "phases": same_backend_phases,
                "decision": decision,
                "blockers": blockers,
            }
        )

    summary = {
        "reuse_captures": len(reuse_captures),
        "comparisons": len(comparisons),
        "candidate_workload_wins": sum(1 for item in comparisons if item["decision"] == "candidate_workload_win"),
        "deprioritized": sum(1 for item in comparisons if item["decision"] == "deprioritize"),
        "experimental": sum(1 for item in comparisons if item["decision"] == "keep_experimental"),
        "missing_baselines": sum(1 for item in comparisons if item["decision"] == "missing_baseline"),
    }
    return {"summary": summary, "comparisons": comparisons}


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "none"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Reuse Contract Comparison Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "| backend | semantics | shape | mode | operands | repeats | setup us | same-backend steady us | setup-inclusive us | same-backend speedup | best non-reuse | workload speedup | break-even repeats | decision | blockers |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|",
        ]
    )
    for item in report["comparisons"]:
        shape = item["shape"]
        phase = item["phases"]["end_to_end"]
        best_backend = item.get("best_nonreuse_backend")
        best_time = item.get("best_nonreuse_median_end_to_end_us")
        best_text = f"{best_backend} {fmt(best_time)}" if best_backend is not None else "none"
        blockers = ",".join(item.get("blockers") or []) or "none"
        lines.append(
            "| {backend} | {semantics} | {m}x{n}x{k} | {mode} | {operands} | {repeats} | {setup} | {steady} | {inclusive} | {same_speedup} | {best} | {workload_speedup} | {break_even} | {decision} | {blockers} |".format(
                backend=item.get("backend"),
                semantics=item.get("semantics"),
                m=shape.get("m"),
                n=shape.get("n"),
                k=shape.get("k"),
                mode=item.get("pack_mode"),
                operands=",".join(item.get("prepack_reuse_operands") or []),
                repeats=item.get("repeats"),
                setup=fmt(item.get("prepack_setup_us")),
                steady=fmt(phase.get("reuse_steady_median_us")),
                inclusive=fmt(phase.get("reuse_setup_inclusive_per_repeat_us")),
                same_speedup=fmt(phase.get("setup_inclusive_speedup")),
                best=best_text,
                workload_speedup=fmt(item.get("speedup_vs_best_nonreuse_setup_inclusive")),
                break_even=fmt(item.get("break_even_repeats_same_backend")),
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
    captures = [load_validated_capture(path) for path in expand_inputs(args.capture)]
    report = compare_reuse_contracts(captures)
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
