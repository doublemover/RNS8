#!/usr/bin/env python3
"""Verify AUTO shape-family recommendations stay exact-cache and non-routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import install_autotune_cache


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "temp" / "auto-shape-family-gate"
REQUIRED_BOUNDARY_FIELDS = {
    "target_id",
    "target_family",
    "semantic_contract",
    "signedness",
    "finite_modulus",
    "layout",
    "output_contract",
    "export_selector",
    "limb_count",
}
EXPECTED_SHADOW_POLICY = "non_routing_shape_family_recommendations_require_exact_review_before_AUTO"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _cache_key_set(cache_path: Path) -> set[str]:
    return {
        str(entry["key"])
        for entry in install_autotune_cache.read_cache(cache_path)
        if isinstance(entry.get("key"), str) and entry["key"]
    }


def _runtime_exact_cache_guard(source_root: Path) -> dict[str, Any]:
    source_path = source_root / "src" / "core" / "autotune_cache.cpp"
    blockers: list[str] = []
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "source": str(source_path),
            "exact_lookup_call_present": False,
            "shape_family_runtime_lookup_absent": False,
            "ready": False,
            "blockers": [f"runtime_source_unreadable:{exc}"],
        }
    exact_lookup = "find_exact_autotune_entry(snapshot, key)" in text
    family_lookup_absent = "shape_family" not in text and "same_family" not in text
    if not exact_lookup:
        blockers.append("runtime_exact_lookup_call_missing")
    if not family_lookup_absent:
        blockers.append("runtime_shape_family_lookup_present")
    return {
        "source": str(source_path),
        "exact_lookup_call_present": exact_lookup,
        "shape_family_runtime_lookup_absent": family_lookup_absent,
        "ready": not blockers,
        "blockers": blockers,
    }


def _basis_key_blockers(recommendation: dict[str, Any], cache_keys: set[str]) -> list[str]:
    blockers: list[str] = []
    basis = recommendation.get("basis_cache_key")
    if recommendation.get("would_recommend") is True:
        if not isinstance(basis, str) or not basis:
            blockers.append("recommended_basis_cache_key_missing")
        elif basis not in cache_keys:
            blockers.append("recommended_basis_cache_key_not_in_reviewed_cache")
    for rejected in recommendation.get("rejected_boundary_candidates") or []:
        if not isinstance(rejected, dict):
            continue
        rejected_basis = rejected.get("basis_cache_key")
        if isinstance(rejected_basis, str) and rejected_basis and rejected_basis not in cache_keys:
            blockers.append("rejected_boundary_basis_cache_key_not_in_reviewed_cache")
    return blockers


def _recommendation_blockers(recommendation: dict[str, Any], cache_keys: set[str]) -> list[str]:
    blockers: list[str] = []
    rec_blockers = recommendation.get("promotion_blockers") or []
    if recommendation.get("runtime_routing_allowed") is not False:
        blockers.append("shape_family_runtime_routing_not_disabled")
    if recommendation.get("promotion_eligible") is not False:
        blockers.append("shape_family_promotion_not_disabled")
    if "shape_family_shadow_only_no_routing_change" not in rec_blockers:
        blockers.append("shape_family_shadow_no_routing_blocker_missing")
    if recommendation.get("recommendation_is_exact_cache_hit") is True:
        if recommendation.get("recommendation_boundary_status") != "exact_reviewed_cache_entry":
            blockers.append("exact_hit_boundary_status_mismatch")
        if "exact_cache_hit_already_owned_by_current_AUTO_cache" not in rec_blockers:
            blockers.append("exact_hit_current_auto_cache_blocker_missing")
    elif recommendation.get("would_recommend") is True:
        if recommendation.get("recommendation_boundary_status") != "same_boundary_family_shadow_representative":
            blockers.append("family_representative_boundary_status_mismatch")
        for required in ("exact_query_not_reviewed", "representative_matrix_requires_same_target_layout_contract_review"):
            if required not in rec_blockers:
                blockers.append(f"family_representative_blocker_missing:{required}")
    else:
        if recommendation.get("recommendation_boundary_status") != "missing_same_boundary_family_reviewed_entry":
            blockers.append("missing_family_boundary_status_mismatch")
        if "missing_same_family_reviewed_entry" not in rec_blockers:
            blockers.append("missing_family_blocker_missing")
        if recommendation.get("rejected_boundary_candidates") and "same_family_entries_rejected_by_boundary" not in rec_blockers:
            blockers.append("boundary_rejection_blocker_missing")
    blockers.extend(_basis_key_blockers(recommendation, cache_keys))
    return sorted(set(blockers))


def _report_rows(path: Path, report: dict[str, Any], cache_keys: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    if report.get("schema") != "rns8_shape_family_shadow_report_v2":
        blockers.append("shape_family_report_schema_mismatch")
    if report.get("policy") != EXPECTED_SHADOW_POLICY:
        blockers.append("shape_family_report_policy_mismatch")
    fields = {str(item) for item in report.get("boundary_fields", []) if isinstance(item, str)}
    missing_fields = sorted(REQUIRED_BOUNDARY_FIELDS - fields)
    blockers.extend(f"boundary_field_missing:{field}" for field in missing_fields)
    rows: list[dict[str, Any]] = []
    for index, recommendation in enumerate(report.get("recommendations", [])):
        if not isinstance(recommendation, dict):
            blockers.append(f"recommendation_not_object:{index}")
            continue
        row_blockers = _recommendation_blockers(recommendation, cache_keys)
        query = recommendation.get("query") if isinstance(recommendation.get("query"), dict) else {}
        rows.append(
            {
                "report": str(path),
                "index": index,
                "family_key": recommendation.get("family_key"),
                "query": query,
                "would_recommend": recommendation.get("would_recommend"),
                "recommendation_is_exact_cache_hit": recommendation.get("recommendation_is_exact_cache_hit"),
                "recommendation_boundary_status": recommendation.get("recommendation_boundary_status"),
                "runtime_routing_allowed": recommendation.get("runtime_routing_allowed"),
                "promotion_eligible": recommendation.get("promotion_eligible"),
                "basis_cache_key": recommendation.get("basis_cache_key"),
                "recommended_backend": recommendation.get("recommended_backend"),
                "recommended_kernel": recommendation.get("recommended_kernel"),
                "rejected_boundary_candidate_count": len(recommendation.get("rejected_boundary_candidates") or []),
                "blockers": row_blockers,
            }
        )
        blockers.extend(f"recommendation_{index}:{blocker}" for blocker in row_blockers)
    return rows, blockers


def build_report(
    cache_path: Path,
    shape_family_shadow_reports: list[Path],
    *,
    source_root: Path = REPO_ROOT,
    require_recommendations: bool = False,
) -> dict[str, Any]:
    cache_keys = _cache_key_set(cache_path)
    runtime_guard = _runtime_exact_cache_guard(source_root)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = list(runtime_guard["blockers"])
    report_summaries: list[dict[str, Any]] = []
    for path in shape_family_shadow_reports:
        report = _read_json(path)
        report_rows, report_blockers = _report_rows(path, report, cache_keys)
        rows.extend(report_rows)
        blockers.extend(f"{path}:{blocker}" for blocker in report_blockers)
        report_summaries.append(
            {
                "path": str(path),
                "recommendation_count": len(report.get("recommendations", [])),
                "family_count": report.get("family_count"),
                "reviewed_cache_entry_count": report.get("reviewed_cache_entry_count"),
            }
        )
    if require_recommendations and not rows:
        blockers.append("no_shape_family_recommendations")
    exact_hits = [row for row in rows if row["recommendation_is_exact_cache_hit"] is True]
    non_exact = [
        row
        for row in rows
        if row["would_recommend"] is True and row["recommendation_is_exact_cache_hit"] is not True
    ]
    missing = [row for row in rows if row["would_recommend"] is not True]
    boundary_rejected = [row for row in rows if row["rejected_boundary_candidate_count"]]
    ready = not blockers
    return {
        "schema": "rns8_auto_shape_family_gate_v1",
        "policy": "exact_cache_runtime_shape_family_shadow_advisory_only",
        "cache_path": str(cache_path),
        "reviewed_cache_entry_count": len(cache_keys),
        "shape_family_shadow_report_count": len(shape_family_shadow_reports),
        "runtime_exact_cache_guard": runtime_guard,
        "required_boundary_fields": sorted(REQUIRED_BOUNDARY_FIELDS),
        "report_summaries": report_summaries,
        "recommendation_count": len(rows),
        "exact_cache_hit_count": len(exact_hits),
        "non_exact_recommendation_count": len(non_exact),
        "missing_same_boundary_count": len(missing),
        "boundary_rejected_recommendation_count": len(boundary_rejected),
        "rank36_gate_complete": ready,
        "blockers": sorted(set(blockers)),
        "recommendations": rows,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "auto-shape-family-gate.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "auto-shape-family-gate.md"
    lines = [
        "# AUTO Shape-Family Gate",
        "",
        f"- Gate complete: `{report['rank36_gate_complete']}`.",
        f"- Reviewed cache entries: `{report['reviewed_cache_entry_count']}`.",
        f"- Shape-family reports: `{report['shape_family_shadow_report_count']}`.",
        f"- Recommendations: `{report['recommendation_count']}`.",
        f"- Blockers: `{', '.join(report['blockers'])}`.",
        "",
        "| Query | Boundary Status | Exact Hit | Would Recommend | Backend | Runtime Routing | Promotion | Blockers |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in report["recommendations"]:
        query = row.get("query") or {}
        shape = f"{query.get('m')}x{query.get('n')}x{query.get('k')}"
        lines.append(
            "| `{semantic}` `{shape}` `{target}` | `{status}` | `{exact}` | `{would}` | `{backend}` | `{routing}` | `{promotion}` | `{blockers}` |".format(
                semantic=query.get("semantic_contract"),
                shape=shape,
                target=query.get("target_id"),
                status=row.get("recommendation_boundary_status"),
                exact=row.get("recommendation_is_exact_cache_hit"),
                would=row.get("would_recommend"),
                backend=row.get("recommended_backend"),
                routing=row.get("runtime_routing_allowed"),
                promotion=row.get("promotion_eligible"),
                blockers=", ".join(row.get("blockers") or []),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="reviewed autotune cache JSON")
    parser.add_argument(
        "--shape-family-shadow-report",
        type=Path,
        action="append",
        default=[],
        help="shape_family_shadow_report.py JSON output",
    )
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--require-recommendations", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.cache,
        args.shape_family_shadow_report,
        source_root=args.source_root,
        require_recommendations=args.require_recommendations,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    if args.require_complete and not report["rank36_gate_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
