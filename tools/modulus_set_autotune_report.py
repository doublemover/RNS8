#!/usr/bin/env python3
"""Gate modulus-set and residue-count autotune experiments.

The benchmark can tag experimental modulus-set and residue-count captures, but
the runtime currently executes the checked-in default ladder. This report keeps
that boundary explicit: search reports can make experiments ready for
non-promoting evidence, but cannot promote a new default ladder or cache entry.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "modulus-set-autotune-reports"
NON_CAPTURE_JSON_NAMES = {
    "modulus-set-autotune-report.json",
    "modulus-set-search-report.json",
    "review_report.json",
    "scenario_manifest.json",
    "validation-summary.json",
}


def _as_list(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                child
                for child in sorted(path.rglob("*.json"))
                if child.name not in NON_CAPTURE_JSON_NAMES
            )
        else:
            expanded.append(path)
    return expanded


def _load_search_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path}: failed to read modulus-set search report: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") not in {1, 2}:
        raise SystemExit(f"{path}: modulus-set search report must be a schema v1/v2 object")
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit(f"{path}: modulus-set search report must contain candidates")
    return data


def _candidate_index(search_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in search_report.get("candidates", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            index[name] = item
            index[f"experimental:{name}"] = item
    return index


def _load_capture(path: Path) -> dict[str, Any]:
    try:
        capture = load_capture(path)
    except Exception as exc:  # noqa: BLE001 - keep report tolerant over mixed roots.
        return {
            "_path": str(path),
            "_schema_status": "invalid",
            "_schema_errors": [f"failed to load capture: {exc}"],
        }
    capture["_path"] = str(path)
    try:
        validate_capture(capture, path)
    except BenchmarkSchemaError as exc:
        capture["_schema_status"] = "invalid"
        capture["_schema_errors"] = [str(exc)]
    else:
        capture["_schema_status"] = "valid"
        capture["_schema_errors"] = []
    return capture


def _nested(capture: dict[str, Any], key: str) -> Any:
    value: Any = capture
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _timing_median(capture: dict[str, Any]) -> float | None:
    summary = capture.get("timing_summary_us")
    if isinstance(summary, dict):
        end_to_end = summary.get("end_to_end")
        if isinstance(end_to_end, dict) and isinstance(end_to_end.get("median"), (int, float)):
            return float(end_to_end["median"])
    raw = capture.get("raw_timings_us")
    if isinstance(raw, dict) and isinstance(raw.get("end_to_end"), list):
        values = sorted(float(value) for value in raw["end_to_end"] if isinstance(value, (int, float)))
        if values:
            return values[len(values) // 2]
    return None


def _backend(capture: dict[str, Any]) -> str:
    value = capture.get("backend_selected") or capture.get("backend_requested")
    return str(value) if isinstance(value, str) else "unknown"


def _modulus_name(capture: dict[str, Any]) -> str:
    value = _nested(capture, "modulus_set.name")
    return str(value) if isinstance(value, str) and value else "missing"


def _workload_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    keys = [
        "semantics",
        "m",
        "n",
        "k",
        "bound_kind",
        "bound_mode",
        "bound",
        "tile_m",
        "tile_n",
        "seed",
        "input_distribution",
        "output_policy.destination_layout",
        "output_policy.status_handling",
    ]
    return tuple(_nested(capture, key) for key in keys)


def _anchor_for(capture: dict[str, Any], captures: list[dict[str, Any]]) -> dict[str, Any] | None:
    key = _workload_key(capture)
    default_rows = [
        item
        for item in captures
        if _workload_key(item) == key
        and _modulus_name(item) == "default"
        and item.get("_schema_status") == "valid"
    ]
    same_backend = [item for item in default_rows if _backend(item) == _backend(capture)]
    rows = same_backend or default_rows
    if not rows:
        return None
    return min(rows, key=lambda item: _timing_median(item) if _timing_median(item) is not None else float("inf"))


def _speedup(candidate: dict[str, Any], anchor: dict[str, Any] | None) -> float | None:
    if anchor is None:
        return None
    candidate_median = _timing_median(candidate)
    anchor_median = _timing_median(anchor)
    if candidate_median is None or anchor_median is None or candidate_median <= 0:
        return None
    return anchor_median / candidate_median


def _candidate_blockers(
    capture: dict[str, Any],
    search_candidates: dict[str, dict[str, Any]],
    all_captures: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any] | None]:
    blockers: list[str] = []
    if capture.get("_schema_status") != "valid":
        blockers.append("schema_invalid")
    modulus_set = capture.get("modulus_set")
    residue_policy = capture.get("residue_count_policy")
    if not isinstance(modulus_set, dict):
        blockers.append("missing_modulus_set_metadata")
        return blockers, None
    if not isinstance(residue_policy, dict):
        blockers.append("missing_residue_count_policy_metadata")
    name = _modulus_name(capture)
    experimental = name != "default"
    search_candidate = search_candidates.get(name)
    if experimental:
        if search_candidate is None:
            blockers.append("experimental_modulus_set_missing_search_candidate")
        else:
            if search_candidate.get("pairwise_coprime") is not True:
                blockers.append("search_candidate_not_pairwise_coprime")
            if search_candidate.get("satisfies_min_bits") is not True:
                blockers.append("search_candidate_insufficient_range_bits")
        if not isinstance(modulus_set.get("cache_promotion_blocker"), str):
            blockers.append("experimental_modulus_set_missing_cache_blocker")
        if isinstance(residue_policy, dict):
            if residue_policy.get("autotune_scope") != "evidence_only_non_promoting":
                blockers.append("experimental_residue_count_scope_not_non_promoting")
            if not isinstance(residue_policy.get("cache_promotion_blocker"), str):
                blockers.append("experimental_residue_policy_missing_cache_blocker")
        if _anchor_for(capture, all_captures) is None:
            blockers.append("missing_default_same_workload_anchor")
        blockers.append("runtime_ladder_not_selectable")
        blockers.append("default_change_requires_spec_cache_schema_proof")
    else:
        if modulus_set.get("experimental") is not False:
            blockers.append("default_modulus_set_marked_experimental")
        if isinstance(residue_policy, dict) and residue_policy.get("autotune_scope") not in {
            "current_exact_cache",
            "evidence_only_non_promoting",
        }:
            blockers.append("unknown_residue_count_autotune_scope")
    return blockers, search_candidate


def build_report(capture_paths: list[Path], search_report: dict[str, Any]) -> dict[str, Any]:
    captures = [_load_capture(path) for path in _as_list(capture_paths)]
    search_candidates = _candidate_index(search_report)
    rows: list[dict[str, Any]] = []
    summary = Counter()
    for capture in captures:
        blockers, search_candidate = _candidate_blockers(capture, search_candidates, captures)
        anchor = _anchor_for(capture, captures) if _modulus_name(capture) != "default" else None
        evidence_blockers = [
            blocker
            for blocker in blockers
            if blocker not in {"runtime_ladder_not_selectable", "default_change_requires_spec_cache_schema_proof"}
        ]
        if _modulus_name(capture) == "default" and capture.get("_schema_status") == "valid":
            decision = "comparison_anchor"
        elif not evidence_blockers:
            decision = "ready_non_promoting_evidence"
        else:
            decision = "blocked"
        summary[decision] += 1
        for blocker in blockers:
            summary[f"blocker:{blocker}"] += 1
        row = {
            "path": capture.get("_path"),
            "schema_status": capture.get("_schema_status"),
            "backend": _backend(capture),
            "semantics": capture.get("semantics"),
            "shape": {
                "m": capture.get("m"),
                "n": capture.get("n"),
                "k": capture.get("k"),
            },
            "modulus_set": _modulus_name(capture),
            "experimental": _modulus_name(capture) != "default",
            "prefix_policy": capture.get("contract_prefix_policy"),
            "selected_prefix": capture.get("selected_prefix"),
            "requested_prefix": capture.get("requested_max_prefix"),
            "residue_autotune_scope": _nested(capture, "residue_count_policy.autotune_scope"),
            "search_candidate": search_candidate.get("name") if isinstance(search_candidate, dict) else None,
            "search_disposition": search_candidate.get("disposition") if isinstance(search_candidate, dict) else None,
            "anchor_path": anchor.get("_path") if isinstance(anchor, dict) else None,
            "anchor_backend": _backend(anchor) if isinstance(anchor, dict) else None,
            "median_end_to_end_us": _timing_median(capture),
            "speedup_vs_anchor": _speedup(capture, anchor),
            "promotion_eligible": False,
            "runtime_routing_allowed": False if _modulus_name(capture) != "default" else None,
            "decision": decision,
            "blockers": blockers,
        }
        rows.append(row)
    return {
        "schema_version": 1,
        "policy": "modulus_set_residue_count_evidence_only",
        "runtime_ladder_changed": False,
        "default_change_gate": search_report.get(
            "default_change_gate",
            "spec_cache_schema_proof_and_same_target_review_required",
        ),
        "search_report_policy": search_report.get("policy"),
        "search_dimensions": search_report.get("search_dimensions", []),
        "summary": dict(summary),
        "rows": rows,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "modulus-set-autotune-report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "modulus-set-autotune-report.md"
    lines = [
        "# Modulus-Set Autotune Report",
        "",
        f"- Policy: `{report['policy']}`",
        f"- Runtime ladder changed: `{str(report['runtime_ladder_changed']).lower()}`",
        f"- Default-change gate: `{report['default_change_gate']}`",
        "",
        "| Decision | Backend | Semantics | Shape | Modulus Set | Prefix | Speedup Vs Anchor | Blockers |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        shape = row["shape"]
        speedup = row.get("speedup_vs_anchor")
        lines.append(
            "| {decision} | {backend} | {semantics} | {m}x{n}x{k} | `{modulus}` | {prefix} | {speedup} | {blockers} |".format(
                decision=row["decision"],
                backend=row["backend"],
                semantics=row["semantics"],
                m=shape["m"],
                n=shape["n"],
                k=shape["k"],
                modulus=row["modulus_set"],
                prefix=row["selected_prefix"],
                speedup=f"{speedup:.3f}x" if isinstance(speedup, float) else "",
                blockers=", ".join(row["blockers"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--search-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    search_report = _load_search_report(args.search_report)
    report = build_report(args.captures, search_report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    path = write_outputs(report, args.out_dir)
    if not args.json:
        print(path)
    if args.require_complete and report["summary"].get("blocked", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
