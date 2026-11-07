#!/usr/bin/env python3
"""Compare many-small independent, host-batch, and grouped-dispatch captures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from report_inputs import load_validated_capture
from grouped_task_contract import (
    SINGLE_GROUPED_BUCKET_POLICY,
    grouped_task_descriptor_contract,
    grouped_task_descriptor_valid,
)
from diagnostic_companion import load_bindings
from measurement_contract import (
    host_observation_contract,
    output_completion_boundary,
    single_task_batch_contract,
    task_scaling_contract,
    uniform_task_sequence,
)
from observation_cohorts import (
    ProductionCohortIndex,
    aggregate_captures,
    canonical_work,
    comparison_environment,
    comparison_noise_margin,
    evidence,
    phase_median,
)
from reference_verdict import CPUReferenceIndex
from review_contract import release_capture_satisfied


DEFAULT_OUT_DIR = Path("temp") / "many-small-grouped-reports"
REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}
REPORT_OUTPUT_NAMES = {
    "many-small-grouped-report.json",
    "review_report.json",
    "scenario_manifest.json",
}


def expand_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                candidate
                for candidate in sorted(path.rglob("*.json"))
                if candidate.name not in REPORT_OUTPUT_NAMES
            )
        else:
            expanded.append(path)
    return expanded


def benchmark_execution_mode(capture: dict[str, Any]) -> str:
    value = capture.get("benchmark_execution_mode")
    if isinstance(value, str):
        return value
    metadata = capture.get("timing_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("benchmark_execution_mode"), str):
        return metadata["benchmark_execution_mode"]
    return "persistent_resident_matrices"


def is_grouped_dispatch(capture: dict[str, Any]) -> bool:
    grouped = (
        capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    )
    if grouped.get("requested") is True:
        return True
    return benchmark_execution_mode(capture) == "benchmark_grouped_dispatch_evidence"


def is_host_api_batch(capture: dict[str, Any]) -> bool:
    host_batch = (
        capture.get("host_api_batch") if isinstance(capture.get("host_api_batch"), dict) else {}
    )
    if host_batch.get("enabled") is True:
        return True
    return benchmark_execution_mode(capture) == "benchmark_host_api_batch"


def mode_for_capture(capture: dict[str, Any]) -> str:
    if is_grouped_dispatch(capture):
        return "grouped_dispatch"
    if is_host_api_batch(capture):
        return "host_api_batch"
    return "independent_call"


def backend_id(capture: dict[str, Any]) -> str:
    selected = capture.get("backend_selected")
    return str(selected) if selected is not None else str(capture.get("backend_requested"))


def task_count_for_capture(capture: dict[str, Any]) -> int:
    grouped = (
        capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    )
    if grouped.get("requested"):
        value = grouped.get("task_count")
        return (
            int(value)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
            else 1
        )
    host_batch = (
        capture.get("host_api_batch") if isinstance(capture.get("host_api_batch"), dict) else {}
    )
    if host_batch.get("enabled"):
        value = host_batch.get("batch_size")
        return (
            int(value)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
            else 1
        )
    return 1


def timing_summary_value(capture: dict[str, Any], phase: str, statistic: str) -> float | None:
    if statistic == "median":
        return phase_median(capture, phase)
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    phase_summary = summary.get(phase)
    if not isinstance(phase_summary, dict):
        return None
    value = phase_summary.get(statistic)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def median_end_to_end_us(capture: dict[str, Any]) -> float | None:
    return timing_summary_value(capture, "end_to_end", "median")


def median_per_task_end_to_end_us(capture: dict[str, Any]) -> float | None:
    median = median_end_to_end_us(capture)
    task_count = task_count_for_capture(capture)
    return median / task_count if median is not None and task_count > 0 else None


def release_satisfied(capture: dict[str, Any]) -> bool:
    return release_capture_satisfied(capture)


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


def checksum_value(capture: dict[str, Any] | None) -> int | None:
    if capture is None:
        return None
    value = capture.get("checksum_u64")
    if value is None:
        value = capture.get("checksum")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def checksum_matches(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool | None:
    left_checksum = checksum_value(left)
    right_checksum = checksum_value(right)
    if left_checksum is None or right_checksum is None:
        return None
    return left_checksum == right_checksum


def normalization(capture: dict[str, Any]) -> dict[str, Any]:
    """Whitelist the native uniform-task producer, not arbitrary average task costs."""
    if not is_grouped_dispatch(capture):
        result = task_scaling_contract(capture)
    else:
        grouped = capture.get("grouped_dispatch") or {}
        timing = capture.get("timing_metadata") or {}
        descriptor = grouped_task_descriptor_contract(capture)
        if (
            not grouped_task_descriptor_valid(capture)
            or uniform_task_sequence(capture) is None
            or descriptor.get("bucket_policy") != SINGLE_GROUPED_BUCKET_POLICY
            or grouped.get("capture_status") != "executed"
            or grouped.get("setup_scope")
            != "benchmark_grouped_dispatch_one_shared_plan_persistent_resident_tasks"
            or benchmark_execution_mode(capture) != "benchmark_grouped_dispatch_evidence"
            or not single_task_batch_contract(capture)
            or timing.get("host_result_kind") != "single_operation"
            or type(timing.get("inner_repetitions")) is not int
            or timing.get("inner_repetitions") != 1
            or timing.get("amortization_policy")
            != "explicit_independent_task_batch_with_per_task_output"
            or output_completion_boundary(capture) != "per_task_host_destinations"
            or host_observation_contract(capture) != {"unit": "completed_call_or_task_batch"}
        ):
            raise ValueError("matched uniform grouped tasks and completed per-task output required")
        result = dict(
            tasks=descriptor["task_count"],
            aggregate_units="microseconds_per_completed_task_batch",
            normalized_units="microseconds_per_completed_task",
            checksum_policy=descriptor["checksum_policy"],
        )
    return dict(result, scope="registered_uniform_host_task_throughput_scaling_v1")


def normalized_contract_key(capture: dict[str, Any]) -> str:
    contract = normalization(capture)
    # Both native batch lanes draw consecutive task inputs from the same seeded
    # generator. Only validated uniform composition permits this task-count scaling.
    return json.dumps(
        dict(
            scope=contract["scope"],
            task_work=json.loads(
                canonical_work(
                    capture,
                    exclude=(
                        "host_api_batch.work_contract",
                        "checksum_policy",
                        "grouped_dispatch.task_count",
                        "grouped_dispatch.task_descriptor_contract",
                    ),
                )
            ),
            checksum_comparison="own_complete_task_set_plus_equal_batch_checksum",
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def normalized_environment(capture: dict[str, Any]) -> str:
    contract = normalization(capture)
    environment = json.loads(comparison_environment(capture))
    environment["comparison_scope"] = contract["scope"]
    environment["host_batch_contract"] = {
        key: contract[key] for key in ("aggregate_units", "normalized_units")
    }
    environment["timing"]["host_result_kind"] = "explicit_per_task_throughput"
    environment["timing"]["inner_repetitions"] = "explicit_task_count_normalization"
    return json.dumps(environment, sort_keys=True, separators=(",", ":"), allow_nan=False)


def capture_summary(capture: dict[str, Any] | None) -> dict[str, Any] | None:
    if capture is None:
        return None
    grouped = (
        capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    )
    descriptor = grouped_task_descriptor_contract(capture)
    return {
        "path": capture.get("_path"),
        "measurement_cohort": evidence(capture),
        "qualified_production": not capture.get("_many_small_admission", ["admission_missing"]),
        "blockers": capture.get("_many_small_admission", ["admission_missing"]),
        "mode": mode_for_capture(capture),
        "backend": backend_id(capture),
        "task_count": task_count_for_capture(capture),
        "median_end_to_end_us": median_end_to_end_us(capture),
        "median_per_task_end_to_end_us": median_per_task_end_to_end_us(capture),
        "release_review": release_satisfied(capture),
        "gpu_events_available": gpu_events_available(capture),
        "grouped_dispatch_execution_strategy": grouped.get("execution_strategy"),
        "grouped_dispatch_batched_export_enabled": grouped.get("batched_export_enabled"),
        "grouped_task_descriptor_layout": descriptor.get("descriptor_layout"),
        "grouped_task_descriptor_bucket_policy": descriptor.get("bucket_policy"),
        "grouped_task_descriptor_bucket_count": descriptor.get("bucket_count"),
        "grouped_task_descriptor_device_policy": descriptor.get("device_descriptor_policy"),
        "grouped_task_descriptor_valid": grouped_task_descriptor_valid(capture)
        if is_grouped_dispatch(capture)
        else None,
    }


def fastest(captures: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        capture
        for capture in captures
        if not capture.get("_many_small_admission", ["admission_missing"])
        and median_per_task_end_to_end_us(capture) is not None
        and median_per_task_end_to_end_us(capture) > 0
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda capture: (
            median_per_task_end_to_end_us(capture),
            evidence(capture)["implementation_key"],
        ),
    )


def speedup(
    numerator_capture: dict[str, Any] | None, denominator_capture: dict[str, Any]
) -> float | None:
    numerator = (
        median_per_task_end_to_end_us(numerator_capture) if numerator_capture is not None else None
    )
    denominator = median_per_task_end_to_end_us(denominator_capture)
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def decision_for(
    candidate: dict[str, Any],
    same_backend_independent: dict[str, Any] | None,
    best_independent: dict[str, Any] | None,
    same_backend_host_batch: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    blockers: list[str] = list(candidate["_many_small_admission"])
    if best_independent is None:
        blockers.append("missing_independent_contract_baseline")
    if same_backend_independent is None:
        blockers.append("missing_same_backend_independent_baseline")
    if same_backend_host_batch is None:
        blockers.append("missing_same_backend_host_batch_baseline")
    elif task_count_for_capture(candidate) != task_count_for_capture(same_backend_host_batch):
        blockers.append("host_batch_task_count_mismatch")
    host_batch_checksum_match = checksum_matches(candidate, same_backend_host_batch)
    if same_backend_host_batch is not None and host_batch_checksum_match is None:
        blockers.append("missing_same_backend_host_batch_checksum")
    elif host_batch_checksum_match is False:
        blockers.append("checksum_mismatch_same_backend_host_batch")
    if not grouped_task_descriptor_valid(candidate):
        blockers.append("invalid_grouped_task_descriptor_contract")
    release_inputs = [
        candidate,
        best_independent,
        same_backend_independent,
        same_backend_host_batch,
    ]
    if any(capture is not None and not release_satisfied(capture) for capture in release_inputs):
        blockers.append("not_release_review")
    if speedup(best_independent, candidate) is None:
        blockers.append("missing_end_to_end_timing")
    if blockers:
        missing = (
            best_independent is None
            or same_backend_independent is None
            or same_backend_host_batch is None
        )
        return "missing_baseline" if missing else "keep_experimental", sorted(set(blockers))
    if speedup(best_independent, candidate) <= 1.0:
        return "deprioritize", ["grouped_not_faster_than_best_independent_per_task"]
    same_backend_speedup = speedup(same_backend_independent, candidate)
    if same_backend_speedup is not None and same_backend_speedup <= 1.0:
        return "deprioritize", ["grouped_not_faster_than_same_backend_independent_per_task"]
    host_batch_speedup = speedup(same_backend_host_batch, candidate)
    if host_batch_speedup is not None and host_batch_speedup <= 1.0:
        return "deprioritize", ["grouped_not_faster_than_same_backend_host_batch_per_task"]
    margins = [
        comparison_noise_margin(baseline, candidate)
        for baseline in (same_backend_independent, best_independent, same_backend_host_batch)
    ]
    if any(margin is None for margin in margins) or min(
        speedup(best_independent, candidate), same_backend_speedup, host_batch_speedup
    ) <= max(margins):
        return "keep_experimental", ["speedup_not_separated_from_observed_noise"]
    return "candidate_win", []


def row_for_capture(
    capture: dict[str, Any],
    same_backend_independent: dict[str, Any] | None,
    best_independent: dict[str, Any] | None,
    same_backend_host_batch: dict[str, Any] | None,
) -> dict[str, Any]:
    mode = mode_for_capture(capture)
    descriptor = grouped_task_descriptor_contract(capture)
    if mode == "grouped_dispatch":
        decision, blockers = decision_for(
            capture, same_backend_independent, best_independent, same_backend_host_batch
        )
    elif mode == "host_api_batch":
        decision, blockers = "host_batch_baseline", []
    else:
        decision, blockers = "independent_baseline", []
    if mode != "grouped_dispatch":
        blockers = capture["_many_small_admission"]
        if blockers:
            decision = "unqualified_baseline"
    comparable = not capture["_many_small_admission"] and all(
        item is not None
        for item in (same_backend_independent, best_independent, same_backend_host_batch)
    )
    if mode == "grouped_dispatch" and (
        checksum_matches(capture, same_backend_host_batch) is not True
        or task_count_for_capture(capture) != task_count_for_capture(same_backend_host_batch or {})
    ):
        comparable = False
    margins = (
        [
            comparison_noise_margin(item, capture)
            for item in (same_backend_independent, best_independent, same_backend_host_batch)
        ]
        if comparable
        else []
    )
    return {
        "measurement_cohort": evidence(capture),
        "comparison_scope": "per_task_throughput_scaling_not_equal_work_latency",
        "comparison_environment": capture.get("_many_small_environment"),
        "normalization": capture.get("_many_small_normalization"),
        "numerical_reference": capture["_many_small_numerical_reference"],
        "required_speedup_margin": max(margins)
        if margins and all(value is not None for value in margins)
        else None,
        "path": capture.get("_path"),
        "mode": mode,
        "backend": backend_id(capture),
        "semantics": capture.get("semantics"),
        "finite_modulus": capture.get("finite_modulus"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "task_count": task_count_for_capture(capture),
        "median_end_to_end_us": median_end_to_end_us(capture),
        "median_per_task_end_to_end_us": median_per_task_end_to_end_us(capture),
        "release_review": release_satisfied(capture),
        "gpu_events_available": gpu_events_available(capture),
        "grouped_dispatch_status": (
            capture.get("grouped_dispatch", {}).get("capture_status")
            if isinstance(capture.get("grouped_dispatch"), dict)
            else None
        ),
        "grouped_dispatch_execution_strategy": (
            capture.get("grouped_dispatch", {}).get("execution_strategy")
            if isinstance(capture.get("grouped_dispatch"), dict)
            else None
        ),
        "grouped_dispatch_batched_export_enabled": (
            capture.get("grouped_dispatch", {}).get("batched_export_enabled")
            if isinstance(capture.get("grouped_dispatch"), dict)
            else None
        ),
        "grouped_task_descriptor_layout": descriptor.get("descriptor_layout"),
        "grouped_task_descriptor_bucket_policy": descriptor.get("bucket_policy"),
        "grouped_task_descriptor_bucket_count": descriptor.get("bucket_count"),
        "grouped_task_descriptor_device_policy": descriptor.get("device_descriptor_policy"),
        "grouped_task_descriptor_valid": grouped_task_descriptor_valid(capture)
        if mode == "grouped_dispatch"
        else None,
        "same_backend_independent": capture_summary(same_backend_independent),
        "best_independent": capture_summary(best_independent),
        "same_backend_host_batch": capture_summary(same_backend_host_batch),
        "checksum": checksum_value(capture),
        "checksum_matches_same_backend_host_batch": checksum_matches(
            capture, same_backend_host_batch
        ),
        "same_backend_host_batch_task_count_matches": (
            None
            if same_backend_host_batch is None
            else task_count_for_capture(capture) == task_count_for_capture(same_backend_host_batch)
        ),
        "speedup_vs_same_backend_independent": speedup(same_backend_independent, capture)
        if comparable
        else None,
        "speedup_vs_best_independent": speedup(best_independent, capture) if comparable else None,
        "speedup_vs_same_backend_host_batch": speedup(same_backend_host_batch, capture)
        if comparable
        else None,
        "decision": decision,
        "blockers": blockers,
    }


def build_report_from_captures(
    captures: list[dict[str, Any]], *, diagnostic_bindings: dict | None = None
) -> dict[str, Any]:
    production = ProductionCohortIndex(captures, diagnostic_bindings)
    references = CPUReferenceIndex(captures)
    by_contract: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for capture in aggregate_captures(captures):
        numerical = references.verdict(capture)
        admission = production.blockers(capture) + list(numerical.blockers)
        if not release_satisfied(capture):
            admission.append("not_release_review")
        try:
            key = normalized_contract_key(capture)
            environment = normalized_environment(capture)
            capture["_many_small_normalization"] = normalization(capture)
        except ValueError as error:
            key = "unqualified:" + canonical_work(capture)
            environment = comparison_environment(capture)
            admission.append("task_scaling_contract_unqualified:" + str(error))
        capture["_many_small_environment"] = environment
        capture["_many_small_admission"] = sorted(set(admission))
        capture["_many_small_numerical_reference"] = numerical.to_json()
        by_contract[(key, environment)].append(capture)

    groups = []
    for (key, environment), items in sorted(by_contract.items()):
        independent = [
            capture for capture in items if mode_for_capture(capture) == "independent_call"
        ]
        host_batches = [
            capture for capture in items if mode_for_capture(capture) == "host_api_batch"
        ]
        grouped = [capture for capture in items if mode_for_capture(capture) == "grouped_dispatch"]
        best_host_batch = fastest(host_batches)
        best_grouped = fastest(grouped)
        rows = []
        for capture in sorted(
            items,
            key=lambda item: (
                mode_for_capture(item),
                backend_id(item),
                evidence(item)["implementation_key"],
            ),
        ):
            backend = backend_id(capture)
            best_independent = fastest(
                [item for item in independent if gpu_backend(item) == gpu_backend(capture)]
            )
            same_backend_independent = fastest(
                [item for item in independent if backend_id(item) == backend]
            )
            same_backend_host_batch = fastest(
                [
                    item
                    for item in host_batches
                    if backend_id(item) == backend
                    and task_count_for_capture(item) == task_count_for_capture(capture)
                ]
            )
            rows.append(
                row_for_capture(
                    capture, same_backend_independent, best_independent, same_backend_host_batch
                )
            )
        groups.append(
            {
                "contract_key": key,
                "comparison_environment": environment,
                "summary": {
                    "independent_count": len(independent),
                    "host_batch_count": len(host_batches),
                    "grouped_dispatch_count": len(grouped),
                    "best_independent": capture_summary(fastest(independent)),
                    "best_host_batch": capture_summary(best_host_batch),
                    "best_grouped_dispatch": capture_summary(best_grouped),
                },
                "rows": sorted(
                    rows, key=lambda row: row.get("median_per_task_end_to_end_us") or float("inf")
                ),
            }
        )

    comparisons = [
        row for group in groups for row in group["rows"] if row["mode"] == "grouped_dispatch"
    ]
    return {
        "schema_version": 2,
        "policy": "many_small_grouped_evidence_only_no_public_resident_lifetime_api",
        "summary": {
            "submitted_captures": len(captures),
            "groups": len(groups),
            "grouped_dispatch_comparisons": len(comparisons),
            "candidate_wins": sum(1 for row in comparisons if row["decision"] == "candidate_win"),
            "deprioritized": sum(1 for row in comparisons if row["decision"] == "deprioritize"),
            "experimental": sum(1 for row in comparisons if row["decision"] == "keep_experimental"),
            "missing_baselines": sum(
                1 for row in comparisons if row["decision"] == "missing_baseline"
            ),
        },
        "groups": groups,
    }


def build_report(paths: list[Path], *, diagnostic_bindings: dict | None = None) -> dict[str, Any]:
    captures = [load_validated_capture(path) for path in expand_inputs(paths)]
    return build_report_from_captures(captures, diagnostic_bindings=diagnostic_bindings)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Many-Small Grouped Dispatch Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "| backend | semantics | shape | tasks | strategy | descriptor | grouped per-task us | best independent | same-backend host-batch us | host-batch task count match | host-batch checksum match | speedup vs best independent | speedup vs host-batch | decision | blockers |",
            "|---|---|---|---:|---|---|---:|---|---:|---|---|---:|---:|---|---|",
        ]
    )
    for group in report["groups"]:
        for row in group["rows"]:
            if row["mode"] != "grouped_dispatch":
                continue
            shape = row["shape"]
            best = row.get("best_independent") or {}
            host_batch = row.get("same_backend_host_batch") or {}
            best_text = (
                f"{best.get('backend')} {best.get('median_per_task_end_to_end_us')}"
                if best
                else "none"
            )
            blockers = ",".join(row.get("blockers") or []) or "none"
            lines.append(
                "| {backend} | {semantics} | {m}x{n}x{k} | {tasks} | {strategy} | {descriptor} | {per_task} | {best} | {host_batch} | {task_count_match} | {checksum_match} | {speed_best} | {speed_batch} | {decision} | {blockers} |".format(
                    backend=row.get("backend"),
                    semantics=row.get("semantics"),
                    m=shape.get("m"),
                    n=shape.get("n"),
                    k=shape.get("k"),
                    tasks=row.get("task_count"),
                    strategy=row.get("grouped_dispatch_execution_strategy"),
                    descriptor=row.get("grouped_task_descriptor_device_policy"),
                    per_task=row.get("median_per_task_end_to_end_us"),
                    best=best_text,
                    host_batch=host_batch.get("median_per_task_end_to_end_us"),
                    task_count_match=row.get("same_backend_host_batch_task_count_matches"),
                    checksum_match=row.get("checksum_matches_same_backend_host_batch"),
                    speed_best=(
                        None
                        if row.get("speedup_vs_best_independent") is None
                        else round(float(row["speedup_vs_best_independent"]), 4)
                    ),
                    speed_batch=(
                        None
                        if row.get("speedup_vs_same_backend_host_batch") is None
                        else round(float(row["speedup_vs_same_backend_host_batch"]), 4)
                    ),
                    decision=row.get("decision"),
                    blockers=blockers,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "many-small-grouped-report.json"
    md_path = out_dir / "many-small-grouped-report.md"
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
    parser.add_argument("--diagnostic-binding", type=Path, action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = list(args.captures)
    if args.capture:
        paths.extend(args.capture)
    if not paths:
        raise SystemExit("at least one capture file or directory is required")
    report = build_report(paths, diagnostic_bindings=load_bindings(args.diagnostic_binding))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.out_md)
    if not args.json and not args.out_json and not args.out_md:
        paths_written = write_report(report, args.out_dir)
        print(paths_written["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
