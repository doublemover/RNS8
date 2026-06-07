#!/usr/bin/env python3
"""Review benchmark-only streaming pack/GEMM/export overlap captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "streaming-overlap-reports"
STREAMING_SCOPE = "direct_hip_streaming_overlap_multistream_operation_groups"
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9


def _load(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    capture["_path"] = str(path)
    return capture


def _backend(capture: dict[str, Any]) -> str:
    return str(capture.get("backend_selected") or capture.get("backend_requested") or "")


def _median(capture: dict[str, Any], phase: str = "end_to_end") -> float | None:
    summary = capture.get("timing_summary_us")
    if isinstance(summary, dict):
        phase_summary = summary.get(phase)
        if isinstance(phase_summary, dict) and isinstance(phase_summary.get("median"), (int, float)):
            return float(phase_summary["median"])
    timings = capture.get("timings_us") or capture.get("raw_timings_us")
    if isinstance(timings, dict):
        values = timings.get(phase)
        if isinstance(values, list) and values:
            numeric = sorted(float(value) for value in values if isinstance(value, (int, float)))
            if numeric:
                return numeric[len(numeric) // 2]
    avg_key = f"avg_{phase}_us"
    if isinstance(capture.get(avg_key), (int, float)):
        return float(capture[avg_key])
    return None


def _streaming(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("streaming_overlap")
    return value if isinstance(value, dict) else {}


def _candidate(capture: dict[str, Any]) -> bool:
    streaming = _streaming(capture)
    return (
        streaming.get("requested") is True
        and streaming.get("capture_status") == "executed"
        and _backend(capture) == "hip-direct"
    )


def _base_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    schedule = capture.get("schedule_metadata") if isinstance(capture.get("schedule_metadata"), dict) else {}
    return (
        capture.get("semantics"),
        capture.get("bound_mode"),
        capture.get("m"),
        capture.get("n"),
        capture.get("k"),
        capture.get("tile_m"),
        capture.get("tile_n"),
        capture.get("prefix"),
        capture.get("input_distribution"),
        capture.get("residue_output_mode"),
        schedule.get("min_selected_prefix"),
        schedule.get("max_selected_prefix"),
        schedule.get("prefix_group_count"),
    )


def _direct_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    operands = capture.get("prepack_reuse_operands")
    if isinstance(operands, list):
        operand_key: Any = tuple(operands)
    else:
        operand_key = (
            "A" if capture.get("reuse_packed_a") is True else "",
            "B" if capture.get("reuse_packed_b") is True else "",
        )
    return _base_key(capture) + (
        capture.get("pack_mode"),
        operand_key,
        capture.get("prepack_reuse_strategy"),
    )


def _gpu_events_available(capture: dict[str, Any]) -> bool:
    timing = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    phases = timing.get("gpu_event_phase_order")
    timings = capture.get("gpu_event_timings_us")
    return (
        timing.get("gpu_event_timing") is True
        and timing.get("gpu_event_timing_status") == "available"
        and timing.get("gpu_event_timing_source") == "hipEventElapsedTime"
        and timing.get("gpu_event_timing_source_scope") == STREAMING_SCOPE
        and isinstance(phases, list)
        and all(
            phase in phases
            for phase in [
                "pack_h2d",
                "pack_kernel",
                "rns_gemm_kernel_group",
                "crt_export_kernel",
                "crt_export_d2h",
            ]
        )
        and isinstance(timings, dict)
        and all(isinstance(timings.get(phase), list) and timings[phase] for phase in phases)
    )


def _release_reviewed(capture: dict[str, Any]) -> bool:
    return (
        int(capture.get("warmups", 0) or 0) >= RELEASE_MIN_WARMUPS
        and int(capture.get("repeats", 0) or 0) >= RELEASE_MIN_REPEATS
    )


def _baseline_maps(captures: list[dict[str, Any]]) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    cpu: dict[tuple[Any, ...], dict[str, Any]] = {}
    direct: dict[tuple[Any, ...], dict[str, Any]] = {}
    for capture in captures:
        if _candidate(capture):
            continue
        backend = _backend(capture)
        streaming = _streaming(capture)
        if backend in {"cpu-reference", "cpu"}:
            cpu.setdefault(_base_key(capture), capture)
        elif backend == "hip-direct" and streaming.get("requested") is not True:
            direct.setdefault(_direct_key(capture), capture)
    return cpu, direct


def _speedup(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> float | None:
    candidate_median = _median(candidate)
    baseline_median = _median(baseline) if baseline is not None else None
    if candidate_median is None or baseline_median is None or candidate_median <= 0:
        return None
    return baseline_median / candidate_median


def _row(candidate: dict[str, Any], cpu: dict[str, Any] | None, direct: dict[str, Any] | None) -> dict[str, Any]:
    streaming = _streaming(candidate)
    blockers: list[str] = []
    if candidate.get("correctness") not in {"ok", None}:
        blockers.append("correctness_not_ok")
    if candidate.get("correctness") is None:
        blockers.append("correctness_not_recorded")
    if streaming.get("promotion_eligible") is not False:
        blockers.append("streaming_overlap_must_be_non_promoting")
    if streaming.get("stream_count", 0) < 3:
        blockers.append("missing_three_stream_pipeline")
    if streaming.get("buffer_count", 0) < 2:
        blockers.append("missing_double_buffering")
    if streaming.get("explicit_dependency_events") is not True:
        blockers.append("missing_dependency_events")
    if streaming.get("stage_event_scope") != STREAMING_SCOPE:
        blockers.append("wrong_stage_event_scope")
    if not _gpu_events_available(candidate):
        blockers.append("missing_required_gpu_events")
    if not _release_reviewed(candidate):
        blockers.append("not_release_reviewed")
    if cpu is None:
        blockers.append("missing_cpu_baseline")
    if direct is None:
        blockers.append("missing_serial_direct_hip_reuse_b_baseline")
    direct_speedup = _speedup(candidate, direct)
    if direct is not None and direct_speedup is None:
        blockers.append("missing_direct_hip_timing")
    if blockers:
        decision = "keep experimental"
    elif direct_speedup is not None and direct_speedup > 1.02:
        decision = "promote locally"
    elif direct_speedup is not None and direct_speedup < 0.98:
        decision = "drop/deprioritize"
    else:
        decision = "keep experimental"
        blockers.append("no_setup_inclusive_direct_hip_win")
    return {
        "capture_path": candidate.get("_path"),
        "semantics": candidate.get("semantics"),
        "shape": [candidate.get("m"), candidate.get("n"), candidate.get("k")],
        "backend": _backend(candidate),
        "selected_kernel": candidate.get("selected_kernel"),
        "pipeline": streaming.get("pipeline"),
        "stream_count": streaming.get("stream_count"),
        "buffer_count": streaming.get("buffer_count"),
        "measured_repeat_count": streaming.get("measured_repeat_count"),
        "batch_wall_us": streaming.get("batch_wall_us"),
        "per_repeat_pipeline_us": streaming.get("per_repeat_pipeline_us"),
        "candidate_median_us": _median(candidate),
        "cpu_baseline_path": cpu.get("_path") if cpu else None,
        "cpu_baseline_median_us": _median(cpu) if cpu else None,
        "direct_hip_baseline_path": direct.get("_path") if direct else None,
        "direct_hip_baseline_median_us": _median(direct) if direct else None,
        "speedup_vs_direct_hip": direct_speedup,
        "release_reviewed": _release_reviewed(candidate),
        "required_gpu_events": _gpu_events_available(candidate),
        "blockers": blockers,
        "decision": decision,
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    captures = [_load(path) for path in paths]
    candidates = [capture for capture in captures if _candidate(capture)]
    cpu, direct = _baseline_maps(captures)
    rows = [
        _row(candidate, cpu.get(_base_key(candidate)), direct.get(_direct_key(candidate)))
        for candidate in candidates
    ]
    return {
        "schema": "rns8_streaming_overlap_report_v1",
        "capture_count": len(captures),
        "candidate_count": len(candidates),
        "rows": rows,
        "decision_counts": {
            decision: sum(1 for row in rows if row["decision"] == decision)
            for decision in ["promote locally", "keep experimental", "drop/deprioritize"]
        },
        "policy": (
            "streaming overlap evidence requires schema-valid bounded Direct-HIP reuse-B captures, CPU and serial "
            "Direct-HIP same-contract baselines, required multi-stream GPU events, release warmups/repeats, exact "
            "correctness, and setup-inclusive end-to-end wins before local promotion; cache/default routing stays off"
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Streaming Overlap Report",
        "",
        f"- capture_count: `{report['capture_count']}`",
        f"- candidate_count: `{report['candidate_count']}`",
        f"- policy: {report['policy']}",
        "",
        "| Capture | Shape | Streams | Buffers | Pipeline Us | Direct Speedup | Decision | Blockers |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        speedup = row.get("speedup_vs_direct_hip")
        lines.append(
            "| {capture} | {shape} | {streams} | {buffers} | {pipeline} | {speedup} | {decision} | {blockers} |".format(
                capture=Path(row["capture_path"]).name if row.get("capture_path") else "unknown",
                shape="x".join(str(value) for value in row["shape"]),
                streams=row.get("stream_count"),
                buffers=row.get("buffer_count"),
                pipeline=row.get("per_repeat_pipeline_us"),
                speedup=f"{float(speedup):.3f}x" if isinstance(speedup, (int, float)) else "n/a",
                decision=row["decision"],
                blockers=", ".join(row["blockers"]) if row["blockers"] else "none",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = build_report(args.captures)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "streaming-overlap-report.json"
    markdown_path = args.out_dir / "streaming-overlap-report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, markdown_path)
    print(json_path)
    print(markdown_path)
    if args.require_complete and (
        report["candidate_count"] == 0 or any(row["decision"] != "promote locally" for row in report["rows"])
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
