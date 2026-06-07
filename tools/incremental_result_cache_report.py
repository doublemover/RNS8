#!/usr/bin/env python3
"""Review incremental result-cache research rows and public contract candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark_sweep_lib.capture_metadata import backend_id, median_phase
from report_capture_inputs import load_report_captures


DEFAULT_OUT_DIR = Path("temp") / "incremental-result-cache-reports"
FINAL_COMPARISON_STATUSES = {
    "checksum_recorded_reference_required",
    "reference_required",
    "exact_cpu_reference_compared",
    "passed",
}
PROMOTION_SPEEDUP_THRESHOLD = 1.10


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
            str(scenario_metadata.get("source_identity") or cache.get("source_identity_policy") or "missing"),
            str(scenario_metadata.get("source_version") or cache.get("source_version_policy") or "missing"),
            str(scenario_metadata.get("dirty_region_shape") or cache.get("dirty_region_policy") or "missing"),
            str(scenario_metadata.get("result_lifetime") or cache.get("result_lifetime_policy") or "missing"),
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


def _end_to_end(capture: dict[str, Any]) -> float | None:
    value = median_phase(capture, "end_to_end")
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _has_gpu_events(capture: dict[str, Any]) -> bool:
    timing = capture.get("timing_metadata")
    return isinstance(timing, dict) and timing.get("gpu_event_timing") is True


def row_for_capture(capture: dict[str, Any]) -> dict[str, Any]:
    cache = _cache(capture)
    scenario = _scenario(capture)
    blockers: list[str] = []
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
        "partial_recompute_policy": cache.get("partial_recompute_policy"),
        "median_end_to_end_us": median_phase(capture, "end_to_end"),
        "promotion_scope": promotion_scope,
        "public_contract_available": public_contract,
        "candidate_role": _candidate_role(capture),
        "stale_rejection_covered": cache.get("stale_rejection_covered"),
        "has_gpu_events": _has_gpu_events(capture),
        "blockers": sorted(set(blockers)),
        "status": "blocked" if blockers else "research-ready",
    }


def _public_group_decision(rows: list[dict[str, Any]], captures: list[dict[str, Any]]) -> tuple[str, list[str], dict[str, Any]]:
    blockers: list[str] = []
    cpu_rows = [capture for capture in captures if backend_id(capture) in {"cpu", "cpu-reference"}]
    full_rows = [
        capture
        for capture in captures
        if backend_id(capture) == "hip-direct"
        and _candidate_role(capture) in {"same_backend_full_recompute_baseline", "full_recompute_baseline"}
    ]
    candidate_rows = [
        capture
        for capture in captures
        if backend_id(capture) == "hip-direct" and _public_contract(_cache(capture))
    ]
    if not cpu_rows:
        blockers.append("cpu_baseline_missing")
    if not full_rows:
        blockers.append("same_backend_full_recompute_baseline_missing")
    if not candidate_rows:
        blockers.append("public_result_cache_candidate_missing")
    for row in rows:
        blockers.extend(row["blockers"])
    best_speedup = None
    if full_rows and candidate_rows:
        full_median = min((_end_to_end(capture) for capture in full_rows), default=None)
        candidate_median = min((_end_to_end(capture) for capture in candidate_rows), default=None)
        if full_median and candidate_median:
            best_speedup = full_median / candidate_median
            if best_speedup < PROMOTION_SPEEDUP_THRESHOLD:
                blockers.append("speedup_below_1_10x")
        else:
            blockers.append("median_end_to_end_missing")
    for capture in candidate_rows:
        cache = _cache(capture)
        if cache.get("final_exact_comparison_status") not in FINAL_COMPARISON_STATUSES:
            blockers.append("candidate_exact_comparison_missing")
        if cache.get("stale_rejection_covered") is not True:
            blockers.append("candidate_stale_rejection_coverage_missing")
        if not _has_gpu_events(capture):
            blockers.append("candidate_gpu_events_missing")
    blockers = sorted(set(blockers))
    details = {
        "cpu_baseline_count": len(cpu_rows),
        "full_recompute_baseline_count": len(full_rows),
        "public_candidate_count": len(candidate_rows),
        "best_speedup_vs_full_recompute": best_speedup,
        "speedup_threshold": PROMOTION_SPEEDUP_THRESHOLD,
    }
    if blockers:
        return "blocked" if "public_result_cache_candidate_missing" in blockers else "keep experimental", blockers, details
    return "promote", blockers, details


def build_report(paths: list[Path]) -> dict[str, Any]:
    captures = [
        capture
        for capture in load_report_captures(paths)
        if isinstance(capture.get("incremental_result_cache"), dict)
        and capture.get("incremental_result_cache", {}).get("enabled") is True
    ]
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_captures: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        key = _contract_key(capture)
        grouped[key].append(row_for_capture(capture))
        grouped_captures[key].append(capture)
    groups: list[dict[str, Any]] = []
    promotable: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        blockers = sorted({blocker for row in rows for blocker in row["blockers"]})
        public_group = any(row["public_contract_available"] for row in rows)
        if public_group:
            decision, promotion_blockers, details = _public_group_decision(rows, grouped_captures[key])
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
    promotion_blocker_counts = Counter(blocker for group in groups for blocker in group["promotion_blockers"])
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
        "policy": "research_rows_blocked_public_contract_candidates_require_1_10x_same_backend_speedup",
        "rank75_gate_complete": bool(groups) and not any(group["blockers"] for group in groups if not group["public_contract_available"]),
        "rank78_gate_complete": bool(promotable),
        "promotable_result_cache_candidate_count": len(promotable),
        "promotable_result_cache_candidates": [
            {
                "family": group["family"],
                "name": group["name"],
                "best_speedup_vs_full_recompute": group["details"].get("best_speedup_vs_full_recompute"),
            }
            for group in promotable
        ],
        "promotion_status": promotion_status,
        "capture_count": len(captures),
        "group_count": len(groups),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "promotion_blocker_counts": dict(sorted(promotion_blocker_counts.items())),
        "groups": groups,
    }


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
    args = parser.parse_args()
    report = build_report(args.captures)
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
