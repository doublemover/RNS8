#!/usr/bin/env python3
"""Review incremental result-cache research rows and public contract candidates."""

from __future__ import annotations

from review_contract import COMPLETE_COMPARISON_STATUSES, FINAL_COMPARISON_STATUSES
from diagnostic_companion import load_bindings
from observation_cohorts import partition_rows, phase_median
from report_comparison import ComparisonPolicy, ReportCohorts
from timing_eligibility import capture_identity

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark_sweep_lib.capture_metadata import backend_id
from report_capture_inputs import load_report_captures


DEFAULT_OUT_DIR = Path("temp") / "incremental-result-cache-reports"

PROMOTION_SPEEDUP_THRESHOLD = 1.10
COMPARISON_POLICY = ComparisonPolicy(
    "same_completed_output_result_cache_vs_full_recompute_v1",
    work_exclusions=("export_variant.selector_key",),
)


def _scenario(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("scenario_metadata")
    return value if isinstance(value, dict) else {}


def _cache(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("incremental_result_cache")
    return value if isinstance(value, dict) else {}


def _path(capture: dict[str, Any]) -> str:
    value = capture.get("_path")
    return str(value) if isinstance(value, str) else "<in-memory>"


def _scenario_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    scenario = _scenario(capture)
    value = scenario.get("metadata")
    return value if isinstance(value, dict) else {}


def _contract_key(capture: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    scenario = _scenario(capture)
    scenario_metadata = _scenario_metadata(capture)
    cache = _cache(capture)
    contract_name = scenario_metadata.get("result_cache_contract_group")
    if contract_name:
        return (
            str(scenario.get("family") or "ad_hoc"),
            str(contract_name),
            str(
                scenario_metadata.get("source_identity")
                or cache.get("source_identity_policy")
                or "missing"
            ),
            str(
                scenario_metadata.get("source_version")
                or cache.get("source_version_policy")
                or "missing"
            ),
            str(
                scenario_metadata.get("dirty_region_shape")
                or cache.get("dirty_region_policy")
                or "missing"
            ),
            str(
                scenario_metadata.get("result_lifetime")
                or cache.get("result_lifetime_policy")
                or "missing"
            ),
        )
    return (
        str(scenario.get("family") or "ad_hoc"),
        str(scenario.get("name") or "ad_hoc"),
        str(cache.get("source_identity_policy") or "missing"),
        str(cache.get("source_version_policy") or "missing"),
        str(cache.get("dirty_region_policy") or "missing"),
        str(cache.get("result_lifetime_policy") or "missing"),
    )


def _promotion_scope(capture: dict[str, Any]) -> str:
    return str(_scenario(capture).get("promotion_eligibility") or "")


def _public_contract(cache: dict[str, Any]) -> bool:
    return cache.get("public_contract_available") is True


def _candidate_role(capture: dict[str, Any]) -> str:
    return str(_cache(capture).get("candidate_role") or "comparison_candidate")


def _has_gpu_events(capture: dict[str, Any]) -> bool:
    timing = capture.get("timing_metadata")
    return isinstance(timing, dict) and timing.get("gpu_event_timing") is True


def row_for_capture(
    capture: dict[str, Any],
    cohorts: ReportCohorts,
) -> dict[str, Any]:
    cache = _cache(capture)
    numerical = cohorts.numerical(capture)
    blockers: list[str] = cohorts.admission(capture)
    if cache.get("enabled") is not True:
        blockers.append("incremental_result_cache_not_enabled")
    promotion_scope = _promotion_scope(capture)
    public_contract = _public_contract(cache)
    baseline_role = _candidate_role(capture) in {
        "cpu_reference_baseline",
        "same_backend_full_recompute_baseline",
        "full_recompute_baseline",
    }
    if public_contract:
        if promotion_scope != "result_cache_contract_candidate":
            blockers.append("scenario_not_result_cache_contract_candidate")
    elif promotion_scope == "result_cache_contract_candidate" and (
        baseline_role or _candidate_role(capture) == "comparison_candidate"
    ):
        pass
    elif promotion_scope != "result_cache_research_only":
        blockers.append("scenario_not_result_cache_research_only")
    for key in [
        "source_identity_policy",
        "source_version_policy",
        "dirty_region_policy",
        "result_lifetime_policy",
        "checksum_policy",
        "partial_recompute_policy",
    ]:
        if cache.get(key) in {None, "", "none"}:
            blockers.append(f"{key}_missing")
    if cache.get("final_exact_comparison_required") is not True:
        blockers.append("final_exact_comparison_not_required")
    if cache.get("final_exact_comparison_status") not in FINAL_COMPARISON_STATUSES:
        blockers.append("final_exact_comparison_status_missing")
    if cache.get("final_exact_comparison_status") not in COMPLETE_COMPARISON_STATUSES:
        blockers.append("final_exact_comparison_pending")
    if cache.get("default_gemm_unchanged") is not True:
        blockers.append("default_gemm_must_remain_unchanged")
    if public_contract:
        for key in [
            "a_matrix_instance_id",
            "b_matrix_instance_id",
            "a_source_version",
            "b_source_version",
            "result_cache_key_fingerprint",
            "dirty_region_count",
            "recomputed_region_count",
            "copied_from_cache_bytes",
            "cache_allocation_bytes",
        ]:
            if key not in cache:
                blockers.append(f"{key}_missing")
        if cache.get("stale_rejection_covered") is not True:
            blockers.append("stale_rejection_coverage_missing")
        if cache.get("runtime_routing_allowed") is not True:
            blockers.append("runtime_routing_allowed_must_be_true")
        if cache.get("cache_eligible") is not True:
            blockers.append("cache_eligible_must_be_true")
        if cache.get("promotion_eligible") is not True:
            blockers.append("promotion_eligible_must_be_true")
    else:
        if cache.get("public_contract_available") is not False:
            blockers.append("public_contract_must_remain_unavailable")
        for key in ["runtime_routing_allowed", "cache_eligible", "promotion_eligible"]:
            if cache.get(key) is not False:
                blockers.append(f"{key}_must_be_false")
    return {
        **cohorts.metadata(capture, COMPARISON_POLICY),
        "path": _path(capture),
        "backend": backend_id(capture),
        "semantics": capture.get("semantics"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "policy": cache.get("policy"),
        "source_identity_policy": cache.get("source_identity_policy"),
        "source_version_policy": cache.get("source_version_policy"),
        "dirty_region_policy": cache.get("dirty_region_policy"),
        "result_lifetime_policy": cache.get("result_lifetime_policy"),
        "checksum_policy": cache.get("checksum_policy"),
        "numerical_reference": numerical.to_json(),
        "partial_recompute_policy": cache.get("partial_recompute_policy"),
        "median_end_to_end_us": phase_median(capture),
        "promotion_scope": promotion_scope,
        "public_contract_available": public_contract,
        "candidate_role": _candidate_role(capture),
        "stale_rejection_covered": cache.get("stale_rejection_covered"),
        "has_gpu_events": _has_gpu_events(capture),
        "blockers": sorted(set(blockers)),
        "status": "blocked" if blockers else "research-ready",
    }


def _public_group_decision(
    rows: list[dict[str, Any]], captures: list[dict[str, Any]], cohorts: ReportCohorts
) -> tuple[str, list[str], dict[str, Any]]:
    by_identity = {row["capture_sha256"]: row for row in rows}
    cpu_rows = [
        capture
        for capture in captures
        if backend_id(capture) in {"cpu", "cpu-reference"}
        and not by_identity[capture_identity(capture)]["blockers"]
    ]
    full_rows = [
        capture
        for capture in captures
        if backend_id(capture) == "hip-direct"
        and _candidate_role(capture)
        in {"same_backend_full_recompute_baseline", "full_recompute_baseline"}
        and not by_identity[capture_identity(capture)]["blockers"]
    ]
    candidates = [
        capture
        for capture in captures
        if backend_id(capture) == "hip-direct" and _public_contract(_cache(capture))
    ]
    comparisons = []
    for candidate in candidates:
        baseline = cohorts.best(full_rows, candidate, COMPARISON_POLICY)
        pair = cohorts.compare(candidate, baseline, COMPARISON_POLICY)
        blockers = list(pair["blockers"]) + by_identity[capture_identity(candidate)]["blockers"]
        if not cpu_rows:
            blockers.append("cpu_baseline_missing")
        if baseline is None:
            blockers.append("same_backend_full_recompute_baseline_missing")
        ratio = pair["speedup"] if not blockers else None
        noise = pair["required_speedup_margin"]
        if ratio is not None:
            if ratio < PROMOTION_SPEEDUP_THRESHOLD:
                blockers.append("speedup_below_1_10x")
            if noise is None or ratio <= noise:
                blockers.append("speedup_not_separated_from_observed_noise")
        comparisons.append(
            dict(
                candidate=_path(candidate),
                baseline=_path(baseline) if baseline is not None else None,
                comparison=pair,
                speedup_vs_full_recompute=ratio,
                required_speedup_margin=max(PROMOTION_SPEEDUP_THRESHOLD, noise)
                if noise is not None
                else None,
                blockers=sorted(set(blockers)),
            )
        )
    ready = [item for item in comparisons if not item["blockers"]]
    qualified_ratios = [
        item["speedup_vs_full_recompute"]
        for item in comparisons
        if item["speedup_vs_full_recompute"] is not None
    ]
    details = dict(
        cpu_baseline_count=len(cpu_rows),
        full_recompute_baseline_count=len(full_rows),
        public_candidate_count=len(candidates),
        comparisons=comparisons,
        best_speedup_vs_full_recompute=max(qualified_ratios, default=None),
        speedup_threshold=PROMOTION_SPEEDUP_THRESHOLD,
    )
    if ready:
        return "promote", [], details
    blockers = sorted({value for item in comparisons for value in item["blockers"]})
    if not candidates:
        blockers.append("public_result_cache_candidate_missing")
    return "keep experimental" if candidates else "blocked", blockers, details


def build_report_from_captures(
    captures: list[dict[str, Any]], *, diagnostic_bindings: dict | None = None
) -> dict[str, Any]:
    captures = [
        capture
        for capture in captures
        if isinstance(capture.get("incremental_result_cache"), dict)
        and capture.get("incremental_result_cache", {}).get("enabled") is True
    ]
    cohorts = ReportCohorts(captures, diagnostic_bindings)
    observations = {capture_identity(capture): capture for capture in cohorts.observations}
    scoped = defaultdict(list)
    for capture in cohorts.observations:
        key = _contract_key(capture)
        scoped[key].append(row_for_capture(capture, cohorts))
    grouped = []
    for key, scoped_rows in sorted(scoped.items()):
        for partition, rows in partition_rows(scoped_rows):
            grouped.append((key, partition, rows))
    groups: list[dict[str, Any]] = []
    promotable: list[dict[str, Any]] = []
    for key, partition, rows in grouped:
        blockers = sorted({blocker for row in rows for blocker in row["blockers"]})
        public_group = any(row["public_contract_available"] for row in rows)
        if public_group:
            decision, promotion_blockers, details = _public_group_decision(
                rows, [observations[row["capture_sha256"]] for row in rows], cohorts
            )
            promotion_scope = "public_result_cache_contract_candidate"
            status = decision
        else:
            promotion_blockers = sorted(
                set(blockers)
                | {
                    "public_source_identity_contract_missing",
                    "public_source_version_contract_missing",
                    "public_result_lifetime_contract_missing",
                }
            )
            decision = "blocked_public_incremental_cache_contract_required"
            promotion_scope = "not_promotable_until_public_lifetime_version_contract_exists"
            status = "blocked" if blockers else "research-ready"
            details = {}
        group = {
            "work_contract_key": partition[0],
            "source_identity": partition[1],
            "comparison_environment": partition[2],
            "family": key[0],
            "name": key[1],
            "source_identity_policy": key[2],
            "source_version_policy": key[3],
            "dirty_region_policy": key[4],
            "result_lifetime_policy": key[5],
            "promotion_decision": decision,
            "promotion_scope": promotion_scope,
            "promotion_blockers": promotion_blockers,
            "status": status,
            "blockers": blockers,
            "public_contract_available": public_group,
            "details": details,
            "rows": rows,
        }
        groups.append(group)
        if decision == "promote":
            promotable.append(group)
    blocker_counts = Counter(blocker for group in groups for blocker in group["blockers"])
    promotion_blocker_counts = Counter(
        blocker for group in groups for blocker in group["promotion_blockers"]
    )
    any_public_group = any(group["public_contract_available"] for group in groups)
    promotion_status = (
        "promote"
        if promotable
        else (
            "blocked_or_experimental"
            if any_public_group
            else ("blocked_public_incremental_cache_contract_required" if groups else "no_captures")
        )
    )
    return {
        "schema": "rns8_incremental_result_cache_report_v1",
        "policy": "qualified_same_work_production_cohorts_require_1_10x_and_observed_noise_separation",
        "rank75_gate_complete": bool(groups) and not any(group["blockers"] for group in groups),
        "rank78_gate_complete": bool(promotable),
        "promotable_result_cache_candidate_count": len(promotable),
        "promotable_result_cache_candidates": [
            {
                "family": group["family"],
                "name": group["name"],
                "work_contract_key": group["work_contract_key"],
                "source_identity": group["source_identity"],
                "comparison_environment": group["comparison_environment"],
                "comparisons": group["details"].get("comparisons", []),
                "best_speedup_vs_full_recompute": group["details"].get(
                    "best_speedup_vs_full_recompute"
                ),
            }
            for group in promotable
        ],
        "promotion_status": promotion_status,
        "capture_count": len(captures),
        "observation_cohort_count": len(cohorts.observations),
        "group_count": len(groups),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "promotion_blocker_counts": dict(sorted(promotion_blocker_counts.items())),
        "groups": groups,
    }


def build_report(paths: list[Path], *, diagnostic_bindings: dict | None = None) -> dict[str, Any]:
    return build_report_from_captures(
        load_report_captures(paths), diagnostic_bindings=diagnostic_bindings
    )


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "incremental-result-cache-report.json"
    md_path = out_dir / "incremental-result-cache-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Incremental Result Cache Report",
        "",
        f"- Policy: `{report['policy']}`",
        f"- Rank 75 gate complete: `{report['rank75_gate_complete']}`",
        f"- Rank 78 gate complete: `{report.get('rank78_gate_complete')}`",
        f"- Promotion status: `{report['promotion_status']}`",
        "",
        "| workload | status | promotion decision | source version | dirty region | blockers |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for group in report["groups"]:
        lines.append(
            "| `{name}` | `{status}` | `{decision}` | `{version}` | `{dirty}` | `{blockers}` |".format(
                name=group["name"],
                status=group["status"],
                decision=group["promotion_decision"],
                version=group["source_version_policy"],
                dirty=group["dirty_region_policy"],
                blockers=", ".join(group["blockers"]) or "none",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--diagnostic-binding", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = build_report(args.captures, diagnostic_bindings=load_bindings(args.diagnostic_binding))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_report(report, args.out_dir).items():
            print(f"{label}: {path}")
    if args.require_complete and not report["rank75_gate_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
