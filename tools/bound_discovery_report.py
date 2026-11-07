#!/usr/bin/env python3
"""Compare bound-discovery captures against static global-bound baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import result_compare
from observation_cohorts import (
    canonical_work,
    phase_median,
    compound_median,
    evidence,
    qualified,
)
from report_inputs import expand_inputs, load_validated_capture
from report_comparison import ComparisonPolicy, ReportCohorts
from reference_verdict import checksum_u64, numerical_observation
from diagnostic_companion import load_bindings
from timing_eligibility import phase_blockers


RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}
BOUNDED_SEMANTICS = {"bounded_i64", "bounded_u64"}
PHASES = ["pack", "rns_gemm", "crt_export", "end_to_end"]
BOUND_WORK_EXCLUSIONS = tuple(
    key
    for key in result_compare.CONTRACT_KEYS
    if key.startswith(
        (
            "bound",
            "schedule_metadata.",
            "tile_bounds_u64.",
            "residue_planes_",
            "residue_count_policy.",
            "export_variant.selector_key",
        )
    )
    or key
    in {
        "prefix",
        "selected_prefix",
        "requested_max_prefix",
        "contract_prefix_policy",
        "tile_m",
        "tile_n",
    }
)


class BoundComparisonPolicy(ComparisonPolicy):
    def additional_timing_phases(self, capture: dict) -> tuple[str, ...]:
        variant = bound_variant(capture)
        if variant == "static_global":
            return ()
        return ("global_bound_scan",) if variant == "input_scan_global" else ("tile_bound_scan",)

    def pair_numerical_blockers(self, candidate: dict, baseline: dict) -> list[str]:
        # Bounds/prefixes are the intentional experiment. The native full-input
        # key binds identical integer inputs independently of those proof tags.
        records = [
            capture.get("native_reference_verification") for capture in (candidate, baseline)
        ]
        keys = [record.get("cache_key") if isinstance(record, dict) else None for record in records]
        blockers = []
        if not isinstance(keys[0], str) or len(keys[0]) != 64 or keys[0] != keys[1]:
            blockers.append("comparison_reference_input_key_mismatch_or_missing")
        if checksum_u64(candidate) is None or checksum_u64(candidate) != checksum_u64(baseline):
            blockers.append("comparison_checksum_missing_or_mismatched")
        return blockers


def comparison_policy(candidate: dict, baseline: dict | None = None) -> BoundComparisonPolicy:
    return BoundComparisonPolicy(
        "same_native_inputs_scan_plus_completed_call_v1",
        work_exclusions=BOUND_WORK_EXCLUSIONS,
        require_cpu_anchor=False,
        cpu_gpu_comparison=baseline is not None and gpu_backend(candidate) != gpu_backend(baseline),
    )


def companion_events(capture: dict, cohorts: ReportCohorts) -> bool:
    if not gpu_backend(capture):
        return True
    try:
        rows = cohorts.diagnostic_observations(capture)
        return bool(rows) and all(gpu_events_available(row["capture"]) for row in rows)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def native_input_key(capture: dict) -> tuple | None:
    record = capture.get("native_reference_verification")
    key = record.get("cache_key") if isinstance(record, dict) else None
    if not isinstance(key, str) or len(key) != 64:
        return None
    return (capture.get("semantics"), capture.get("m"), capture.get("n"), capture.get("k"), key)


def cpu_cross_check(capture: dict, references: dict) -> dict:
    rows = references.get(native_input_key(capture), [])
    blockers = []
    for row in rows:
        if not row["full_output_reference_covered"]:
            blockers.append("cpu_reference_full_output_pending")
        if row["checksum_u64"] is None:
            blockers.append("cpu_reference_checksum_missing")
        elif row["checksum_u64"] != checksum_u64(capture):
            blockers.append("checksum_mismatch_vs_reference")
    return dict(
        role="optional_explicit_cpu_native_input_cross_check",
        references=rows,
        blockers=sorted(set(blockers)),
    )


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


def numeric_value(capture: dict[str, Any], key: str) -> float | None:
    if "_observation_aggregate" in capture and key.startswith("avg_") and key.endswith("_us"):
        return phase_median(capture, key[4:-3])
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
        return phase_median(capture, "global_bound_scan")
    if variant in {"proof_mask_per_tile", "tile_bound_per_tile"}:
        return phase_median(capture, "tile_bound_scan")
    return None


def setup_inclusive_median_us(capture: dict[str, Any]) -> float | None:
    variant = bound_variant(capture)
    if variant == "static_global":
        return phase_median(capture)
    scan = "global_bound_scan" if variant == "input_scan_global" else "tile_bound_scan"
    return compound_median(capture, ["end_to_end", scan])


def phase_comparison(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any],
    phase: str,
    *,
    cohorts: ReportCohorts,
    comparison_qualified: bool,
) -> dict[str, Any]:
    baseline_median = (
        timing_summary_value(baseline, phase, "median") if baseline is not None else None
    )
    candidate_median = timing_summary_value(candidate, phase, "median")
    blockers = [
        reason
        for capture in (baseline, candidate)
        if capture is not None
        for raw in cohorts.raw_observations(capture)
        for reason in phase_blockers(raw, phase)
    ]
    speedup = (
        baseline_median / candidate_median
        if qualified(baseline)
        and qualified(candidate)
        and comparison_qualified
        and not blockers
        and baseline_median is not None
        and candidate_median not in (None, 0.0)
        else None
    )
    return {
        "baseline_median_us": baseline_median,
        "candidate_median_us": candidate_median,
        "candidate_speedup_vs_baseline": speedup,
        "timing_blockers": sorted(set(blockers)),
    }


def workload_key(capture: dict[str, Any]) -> str:
    return canonical_work(capture, exclude=BOUND_WORK_EXCLUSIONS)


def select_best_static(
    candidates: list[dict[str, Any]],
    candidate: dict[str, Any],
    cohorts: ReportCohorts,
) -> dict[str, Any] | None:
    valid = []
    work, environments = workload_key(candidate), {}
    for capture in candidates:
        policy = comparison_policy(candidate, capture)
        scope = policy.cpu_gpu_comparison
        if scope not in environments:
            environments[scope] = policy.environment(candidate)
        if (
            bound_variant(capture) == "static_global"
            and not cohorts.admission(capture, policy)
            and setup_inclusive_median_us(capture) is not None
            and workload_key(capture) == work
            and policy.environment(capture) == environments[scope]
            and not policy.pair_numerical_blockers(candidate, capture)
        ):
            valid.append(capture)
    if not valid:
        return None
    return min(
        valid,
        key=lambda capture: (
            setup_inclusive_median_us(capture),
            evidence(capture)["implementation_key"],
        ),
    )


def decision_for(
    same_backend_static: dict[str, Any] | None,
    best_static: dict[str, Any] | None,
    candidate: dict[str, Any],
    speedup_vs_same_backend_static: float | None,
    speedup_vs_best_static: float | None,
    *,
    admission_blockers: list[str],
    required_margin: float | None,
    cohorts: ReportCohorts,
) -> tuple[str, list[str]]:
    blockers: list[str] = list(admission_blockers)
    if same_backend_static is None:
        return "missing_baseline", sorted(
            set(blockers + ["missing_same_backend_static_global_baseline"])
        )
    if best_static is None:
        return "missing_baseline", sorted(
            set(blockers + ["missing_contract_static_global_baseline"])
        )
    if (
        not release_satisfied(candidate)
        or not release_satisfied(same_backend_static)
        or not release_satisfied(best_static)
    ):
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
    if not companion_events(candidate, cohorts):
        blockers.append("missing_candidate_gpu_events")
    if not companion_events(same_backend_static, cohorts):
        blockers.append("missing_same_backend_static_gpu_events")
    if not companion_events(best_static, cohorts):
        blockers.append("missing_best_static_gpu_events")
    if speedup_vs_same_backend_static is None or speedup_vs_best_static is None:
        blockers.append("missing_setup_inclusive_timing")
    if required_margin is None:
        blockers.append("comparison_variance_evidence_missing")
    if blockers:
        return "keep_experimental", blockers
    assert required_margin is not None
    assert speedup_vs_same_backend_static is not None and speedup_vs_best_static is not None
    if (
        speedup_vs_same_backend_static is not None
        and speedup_vs_same_backend_static < 1 / required_margin
    ):
        return "deprioritize", ["candidate_not_faster_than_same_backend_static_setup_inclusive"]
    if speedup_vs_best_static is not None and speedup_vs_best_static < 1 / required_margin:
        return "deprioritize", ["candidate_not_faster_than_best_static_setup_inclusive"]
    if min(speedup_vs_same_backend_static, speedup_vs_best_static) > required_margin:
        return "candidate_win", []
    return "keep_experimental", ["setup_inclusive_gain_not_above_observed_noise"]


def compare_bound_discovery(
    captures: list[dict[str, Any]], *, diagnostic_bindings: dict | None = None
) -> dict[str, Any]:
    raw = [capture for capture in captures if bound_variant(capture) is not None]
    cohorts = ReportCohorts(raw, diagnostic_bindings)
    cpu_references = defaultdict(list)
    seen_references = set()
    for capture in raw:
        if not gpu_backend(capture) and native_input_key(capture) is not None:
            observation = numerical_observation(capture)
            if observation["capture_sha256"] not in seen_references:
                cpu_references[native_input_key(capture)].append(observation)
                seen_references.add(observation["capture_sha256"])
    bounded = cohorts.observations
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
        same_backend_static = select_best_static(
            by_workload_backend.get((key, backend), []), candidate, cohorts
        )
        best_static = select_best_static(by_workload.get(key, []), candidate, cohorts)
        same_comparison = cohorts.compare(
            candidate, same_backend_static, comparison_policy(candidate, same_backend_static)
        )
        best_comparison = cohorts.compare(
            candidate, best_static, comparison_policy(candidate, best_static)
        )
        cpu_reference = cpu_cross_check(candidate, cpu_references)
        for comparison in (same_comparison, best_comparison):
            comparison["blockers"] = sorted(set(comparison["blockers"] + cpu_reference["blockers"]))
            comparison["qualified"] = not comparison["blockers"]
            if comparison["blockers"]:
                comparison["speedup"] = None
        candidate_setup_inclusive = setup_inclusive_median_us(candidate)
        same_backend_setup_inclusive = (
            setup_inclusive_median_us(same_backend_static)
            if same_backend_static is not None
            else None
        )
        best_static_setup_inclusive = (
            setup_inclusive_median_us(best_static) if best_static is not None else None
        )
        speedup_vs_same_backend_static = same_comparison["speedup"]
        speedup_vs_best_static = best_comparison["speedup"]
        margins = [row["required_speedup_margin"] for row in (same_comparison, best_comparison)]
        required_margin = max(margins) if all(value is not None for value in margins) else None
        decision, blockers = decision_for(
            same_backend_static,
            best_static,
            candidate,
            speedup_vs_same_backend_static,
            speedup_vs_best_static,
            admission_blockers=sorted(
                set(same_comparison["blockers"] + best_comparison["blockers"])
            ),
            required_margin=required_margin,
            cohorts=cohorts,
        )
        counts = proof_counts(candidate)
        comparisons.append(
            {
                "workload_key": key,
                "same_backend_comparison": same_comparison,
                "best_static_comparison": best_comparison,
                "required_speedup_margin": required_margin,
                "cpu_native_input_reference": cpu_reference,
                "measurement_cohort": evidence(candidate),
                "baseline_observation_cohorts": [
                    evidence(c)
                    for c in by_workload.get(key, [])
                    if bound_variant(c) == "static_global"
                ],
                "selected_same_backend_cohort": evidence(same_backend_static),
                "selected_best_static_cohort": evidence(best_static),
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
                "best_static_capture": best_static.get("_path")
                if best_static is not None
                else None,
                "best_static_backend": backend_id(best_static) if best_static is not None else None,
                "candidate_setup_scan_us": setup_scan_us(candidate),
                "candidate_median_end_to_end_us": timing_summary_value(
                    candidate, "end_to_end", "median"
                ),
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
                "candidate_gpu_events": companion_events(candidate, cohorts)
                if gpu_backend(candidate)
                else None,
                "same_backend_static_gpu_events": companion_events(same_backend_static, cohorts)
                if same_backend_static is not None and gpu_backend(same_backend_static)
                else None,
                "best_static_gpu_events": companion_events(best_static, cohorts)
                if best_static is not None and gpu_backend(best_static)
                else None,
                "proof_counts": counts,
                "phases": {
                    phase: phase_comparison(
                        same_backend_static,
                        candidate,
                        phase,
                        cohorts=cohorts,
                        comparison_qualified=same_comparison["qualified"],
                    )
                    for phase in PHASES
                },
                "decision": decision,
                "blockers": blockers,
            }
        )

    summary = {
        "bounded_captures": len(bounded),
        "static_global_captures": sum(
            1 for item in bounded if bound_variant(item) == "static_global"
        ),
        "input_scan_global_captures": sum(
            1 for item in bounded if bound_variant(item) == "input_scan_global"
        ),
        "proof_mask_per_tile_captures": sum(
            1 for item in bounded if bound_variant(item) == "proof_mask_per_tile"
        ),
        "tile_bound_per_tile_without_proofs": sum(
            1 for item in bounded if bound_variant(item) == "tile_bound_per_tile"
        ),
        "comparisons": len(comparisons),
        "candidate_wins": sum(1 for item in comparisons if item["decision"] == "candidate_win"),
        "deprioritized": sum(1 for item in comparisons if item["decision"] == "deprioritize"),
        "experimental": sum(1 for item in comparisons if item["decision"] == "keep_experimental"),
        "missing_baselines": sum(
            1 for item in comparisons if item["decision"] == "missing_baseline"
        ),
    }
    return {
        "timing_scope": "one_time_scan_plus_completed_call_estimate_not_full_cold_lifecycle",
        "promotion_eligible": False,
        "summary": summary,
        "comparisons": comparisons,
    }


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
    parser.add_argument("--diagnostic-binding", action="append", type=Path, default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = expand_inputs(args.capture)
    captures = [load_validated_capture(path) for path in paths]
    report = compare_bound_discovery(
        captures, diagnostic_bindings=load_bindings(args.diagnostic_binding)
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
