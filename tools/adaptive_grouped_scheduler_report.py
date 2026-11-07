#!/usr/bin/env python3
"""Review adaptive prefix grouped scheduler benchmark captures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from report_capture_inputs import load_report_capture
from observation_cohorts import phase_median, evidence
from report_comparison import ComparisonPolicy, ReportCohorts
from diagnostic_companion import file_digest, load_bindings
from timing_eligibility import capture_identity


DEFAULT_OUT_DIR = Path("temp") / "adaptive-grouped-scheduler-reports"
DIRECT_GROUPED_KERNEL = "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3"
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
SCHEDULER_POLICY = ComparisonPolicy(
    "same_work_nonadaptive_vs_grouped_scheduler_v1",
    work_exclusions=("adaptive_grouped_scheduler.requested", "adaptive_grouped_scheduler.strategy"),
)


def _backend(capture: dict[str, Any]) -> str:
    return str(capture.get("backend_selected") or capture.get("backend_requested") or "")


def _median(capture: dict[str, Any], phase: str = "end_to_end") -> float | None:
    return phase_median(capture, phase)


def _same_contract_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    return (SCHEDULER_POLICY.work(capture),)


def _gpu_events_available(capture: dict[str, Any]) -> bool:
    timing = (
        capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    )
    events = capture.get("gpu_event_timings_us")
    phases = timing.get("gpu_event_phase_order")
    return (
        timing.get("gpu_event_timing") is True
        and timing.get("gpu_event_timing_status") == "available"
        and timing.get("gpu_event_timing_source") == "hipEventElapsedTime"
        and isinstance(phases, list)
        and "rns_gemm_kernel_group" in phases
        and isinstance(events, dict)
        and isinstance(events.get("rns_gemm_kernel_group"), list)
        and bool(events["rns_gemm_kernel_group"])
    )


def _release_reviewed(capture: dict[str, Any]) -> bool:
    return (
        int(capture.get("warmups", 0) or 0) >= RELEASE_MIN_WARMUPS
        and int(capture.get("repeats", 0) or 0) >= RELEASE_MIN_REPEATS
    )


def _load(path: Path) -> dict[str, Any]:
    return load_report_capture(path, from_directory=False)


def _adaptive_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("adaptive_grouped_scheduler")
    return value if isinstance(value, dict) else {}


def _candidate(capture: dict[str, Any]) -> bool:
    adaptive = _adaptive_metadata(capture)
    return (
        adaptive.get("requested") is True
        and _backend(capture) == "hip-direct"
        and capture.get("selected_kernel") == DIRECT_GROUPED_KERNEL
    )


def _baseline_map(
    captures: list[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]]:
    baselines: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = {}
    for capture in captures:
        if _candidate(capture):
            continue
        key = _same_contract_key(capture)
        backend = _backend(capture)
        adaptive = _adaptive_metadata(capture)
        if backend in {"cpu-reference", "cpu"}:
            baselines.setdefault(key, {}).setdefault("cpu", []).append(capture)
        elif backend == "hip-direct" and adaptive.get("requested") is not True:
            baselines.setdefault(key, {}).setdefault("direct_hip", []).append(capture)
    return baselines


def _diagnostic_events(candidate: dict, cohorts: ReportCohorts) -> bool:
    """GPU phases describe matched diagnostics, never production host timings."""
    receipts = cohorts.production.receipts(candidate)
    if len(receipts) != evidence(candidate).get("independent_runs") or not receipts:
        return False
    for receipt in receipts:
        binding = receipt.get("diagnostic_companion") or {}
        try:
            path = Path(binding["diagnostic_capture"])
            diagnostic = _load(path)
            if (
                binding.get("diagnostic_kind") != "hip_events"
                or file_digest(path) != binding["diagnostic_file_sha256"]
                or capture_identity(diagnostic) != binding["diagnostic_capture_sha256"]
                or not _gpu_events_available(diagnostic)
            ):
                return False
        except (KeyError, OSError, TypeError, ValueError):
            return False
    return True


def _scheduler_blockers(capture: dict) -> list[str]:
    adaptive = _adaptive_metadata(capture)
    schedule = capture.get("schedule_metadata") or {}
    blockers = []
    if adaptive.get("capture_status") != "executed":
        blockers.append("adaptive_grouped_scheduler_not_executed")
    if adaptive.get("promotion_eligible") is not False:
        blockers.append("adaptive_grouped_scheduler_must_be_non_promoting")
    if schedule.get("adaptive_execution_applied") is not True:
        blockers.append("adaptive_execution_not_applied")
    if int(adaptive.get("active_entry_count") or 0) <= 0:
        blockers.append("missing_active_entry_count")
    if int(adaptive.get("aggregate_launch_count_model") or 0) != 1:
        blockers.append("missing_aggregate_launch_model")
    ratio = adaptive.get("launch_reduction_ratio")
    if type(ratio) not in (int, float) or not math.isfinite(ratio):
        blockers.append("missing_launch_reduction_ratio")
    elif ratio <= 1:
        blockers.append("no_launch_reduction_model")
    return blockers


def _row(
    candidate: dict[str, Any], baselines: dict[str, list[dict]], cohorts: ReportCohorts
) -> dict[str, Any]:
    adaptive = _adaptive_metadata(candidate)
    schedule = (
        candidate.get("schedule_metadata")
        if isinstance(candidate.get("schedule_metadata"), dict)
        else {}
    )
    # CPU rows are arithmetic anchors here, not independently qualified timing
    # baselines. A CPU timing ratio would need the explicit CPU/GPU policy.
    numerical = cohorts.numerical(candidate)
    direct = cohorts.best(baselines.get("direct_hip", []), candidate, SCHEDULER_POLICY)
    comparison = cohorts.compare(candidate, direct, SCHEDULER_POLICY)
    direct_numerical = cohorts.numerical(direct) if direct is not None else None
    blockers: list[str] = list(comparison["blockers"])
    scheduler_observations = [
        dict(capture_sha256=capture_identity(raw), blockers=_scheduler_blockers(raw))
        for raw in cohorts.raw_observations(candidate)
    ]
    if not scheduler_observations:
        blockers.append("scheduler_process_observations_missing")
    blockers.extend(reason for item in scheduler_observations for reason in item["blockers"])
    gpu_events = _diagnostic_events(candidate, cohorts)
    if not gpu_events:
        blockers.append("missing_required_gpu_events")
    if not _release_reviewed(candidate):
        blockers.append("not_release_reviewed")
    if not numerical.reference_capture_sha256:
        blockers.append("missing_cpu_baseline")
    if direct is None:
        blockers.append("missing_direct_hip_baseline")
    direct_speedup = comparison["speedup"]
    margin = comparison["required_speedup_margin"]
    if direct is not None and direct_speedup is None:
        blockers.append("missing_direct_hip_timing")
    launch_reduction_ratio = adaptive.get("launch_reduction_ratio")
    if type(launch_reduction_ratio) not in (int, float) or not math.isfinite(
        float(launch_reduction_ratio)
    ):
        launch_reduction_ratio = None
    blockers = sorted(set(blockers))
    if blockers:
        decision = "keep experimental"
    elif direct_speedup is not None and margin is not None and direct_speedup > margin:
        decision = "promote locally"
    elif direct_speedup is not None and margin is not None and direct_speedup < 1 / margin:
        decision = "drop/deprioritize"
    else:
        decision = "keep experimental"
        blockers.append("no_setup_inclusive_direct_hip_win")
    return {
        "capture_path": candidate.get("_path"),
        "measurement_cohort": evidence(candidate),
        "numerical_reference": numerical.to_json(),
        "direct_numerical_reference": direct_numerical.to_json() if direct_numerical else None,
        "baseline_observation_cohorts": {
            name: [evidence(c) for c in values] for name, values in baselines.items()
        },
        "selected_direct_cohort": evidence(direct),
        "comparison": comparison,
        "required_speedup_margin": margin,
        "scheduler_process_observations": scheduler_observations,
        "semantics": candidate.get("semantics"),
        "shape": [candidate.get("m"), candidate.get("n"), candidate.get("k")],
        "backend": _backend(candidate),
        "selected_kernel": candidate.get("selected_kernel"),
        "schedule_groups": schedule.get("prefix_group_count"),
        "active_tile_count": adaptive.get("active_tile_count"),
        "active_entry_count": adaptive.get("active_entry_count"),
        "independent_launch_count_model": adaptive.get("independent_launch_count_model"),
        "aggregate_launch_count_model": adaptive.get("aggregate_launch_count_model"),
        "launch_reduction_ratio": launch_reduction_ratio,
        "event_scope": adaptive.get("event_scope"),
        "candidate_median_us": _median(candidate),
        "cpu_baseline_path": None,
        "cpu_baseline_median_us": None,
        "cpu_anchor_capture_sha256": list(numerical.reference_capture_sha256),
        "cpu_baseline_role": "arithmetic_anchor_only_no_cpu_timing_ratio",
        "direct_hip_baseline_path": direct.get("_path") if direct else None,
        "direct_hip_baseline_median_us": _median(direct) if direct else None,
        "speedup_vs_direct_hip": direct_speedup,
        "release_reviewed": _release_reviewed(candidate),
        "required_gpu_events": gpu_events,
        "gpu_event_evidence_scope": "matched_diagnostic_companions",
        "blockers": blockers,
        "decision": decision,
    }


def build_report(paths: list[Path], *, diagnostic_bindings: dict | None = None) -> dict[str, Any]:
    raw = [_load(path) for path in paths]
    cohorts = ReportCohorts(raw, diagnostic_bindings)
    captures = cohorts.observations
    candidates = [capture for capture in captures if _candidate(capture)]
    baselines = _baseline_map(captures)
    rows = [
        _row(candidate, baselines.get(_same_contract_key(candidate), {}), cohorts)
        for candidate in candidates
    ]
    return {
        "schema": "rns8_adaptive_grouped_scheduler_report_v1",
        "capture_count": len(raw),
        "implementation_cohort_count": len(captures),
        "candidate_count": len(candidates),
        "rows": rows,
        "decision_counts": {
            decision: sum(1 for row in rows if row["decision"] == decision)
            for decision in ["promote locally", "keep experimental", "drop/deprioritize"]
        },
        "policy": (
            "adaptive grouped scheduler evidence requires schema-valid captures, direct-HIP grouped kernel execution, "
            "CPU and non-requested direct-HIP baselines, required GPU events, release review warmups/repeats, "
            "and matched hook-disabled independent production processes with completed-output wins "
            "above their observed noise before local promotion. CPU rows are arithmetic anchors only."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Adaptive Grouped Scheduler Report",
        "",
        f"- capture_count: `{report['capture_count']}`",
        f"- candidate_count: `{report['candidate_count']}`",
        f"- policy: {report['policy']}",
        "",
        "| Capture | Shape | Groups | Active Entries | Launch Ratio | Direct Speedup | Decision | Blockers |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        speedup = row.get("speedup_vs_direct_hip")
        ratio = row.get("launch_reduction_ratio")
        lines.append(
            "| {capture} | {shape} | {groups} | {entries} | {ratio} | {speedup} | {decision} | {blockers} |".format(
                capture=Path(row["capture_path"]).name if row.get("capture_path") else "unknown",
                shape="x".join(str(value) for value in row["shape"]),
                groups=row.get("schedule_groups"),
                entries=row.get("active_entry_count"),
                ratio=f"{float(ratio):.2f}x" if isinstance(ratio, (int, float)) else "n/a",
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
    parser.add_argument("--diagnostic-binding", action="append", type=Path, default=[])
    args = parser.parse_args()
    report = build_report(args.captures, diagnostic_bindings=load_bindings(args.diagnostic_binding))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adaptive-grouped-scheduler-report.json"
    markdown_path = args.out_dir / "adaptive-grouped-scheduler-report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, markdown_path)
    print(json_path)
    print(markdown_path)
    if args.require_complete and (
        report["candidate_count"] == 0
        or any(row["decision"] != "promote locally" for row in report["rows"])
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
