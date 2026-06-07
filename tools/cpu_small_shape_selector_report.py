#!/usr/bin/env python3
"""Review CPU small-shape selector evidence without changing AUTO routing."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark_sweep_lib.capture_metadata import backend_id, median_phase
from report_capture_inputs import load_report_captures


DEFAULT_OUT_DIR = Path("temp") / "cpu-small-shape-selector-reports"
PROMOTION_MARGIN = 1.10


def _scenario(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("scenario_metadata")
    return value if isinstance(value, dict) else {}


def _selector(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("cpu_small_shape_selector")
    return value if isinstance(value, dict) else {}


def _path(capture: dict[str, Any]) -> str:
    value = capture.get("_path")
    return str(value) if isinstance(value, str) else "<in-memory>"


def _output_contract(capture: dict[str, Any]) -> str:
    output_policy = capture.get("output_policy")
    if isinstance(output_policy, dict) and isinstance(output_policy.get("contract"), str):
        return str(output_policy["contract"])
    scenario = _scenario(capture)
    return str(scenario.get("output_domain") or "unknown")


def _finite_modulus(capture: dict[str, Any]) -> str:
    scenario = _scenario(capture)
    for value in (scenario.get("modulus"), capture.get("finite_modulus")):
        if value is not None:
            return str(value)
    return "none"


def _target_boundary(capture: dict[str, Any]) -> str:
    target_variant = capture.get("target_variant")
    if isinstance(target_variant, dict):
        target_id = target_variant.get("target_id")
        namespace = target_variant.get("target_namespace")
        if isinstance(target_id, str) and target_id:
            return f"{namespace or 'unknown'}:{target_id}"
    return str(capture.get("target_id") or "cpu")


def _group_key(capture: dict[str, Any]) -> tuple[str, str, str, str, str, int, int, int]:
    scenario = _scenario(capture)
    shape = scenario.get("shape") if isinstance(scenario.get("shape"), dict) else {}
    return (
        str(scenario.get("family") or "ad_hoc"),
        str(scenario.get("name") or "ad_hoc"),
        str(capture.get("semantics") or scenario.get("semantics") or "unknown"),
        _finite_modulus(capture),
        _output_contract(capture),
        int(shape.get("m") or capture.get("m") or 0),
        int(shape.get("n") or capture.get("n") or 0),
        int(shape.get("k") or capture.get("k") or 0),
    )


def row_for_capture(capture: dict[str, Any]) -> dict[str, Any]:
    selector = _selector(capture)
    backend = backend_id(capture)
    blockers: list[str] = []
    if selector.get("enabled") is not True:
        blockers.append("cpu_small_shape_selector_not_enabled")
    if selector.get("cpu_reference_required") is not True:
        blockers.append("cpu_reference_not_required")
    if selector.get("release_review_required") is not True:
        blockers.append("release_review_not_required")
    for key in ("runtime_routing_allowed", "cache_eligible", "promotion_eligible"):
        if selector.get(key) is not False:
            blockers.append(f"{key}_must_be_false")
    median = median_phase(capture, "end_to_end")
    if median is None:
        blockers.append("missing_end_to_end_median")
    return {
        "path": _path(capture),
        "backend": backend,
        "target_boundary": _target_boundary(capture),
        "candidate_role": selector.get("candidate_role"),
        "policy": selector.get("policy"),
        "boundary_key": selector.get("boundary_key"),
        "median_end_to_end_us": median,
        "checksum": capture.get("checksum"),
        "blockers": sorted(set(blockers)),
    }


def _group_report(key: tuple[str, str, str, str, str, int, int, int], rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = sorted({blocker for row in rows for blocker in row["blockers"]})
    cpu_rows = [row for row in rows if row["backend"] in {"cpu-reference", "cpu"}]
    gpu_rows = [row for row in rows if row["backend"] not in {"cpu-reference", "cpu"}]
    if not cpu_rows:
        blockers.append("cpu_reference_baseline_missing")
    cpu_median = min(
        (float(row["median_end_to_end_us"]) for row in cpu_rows if isinstance(row["median_end_to_end_us"], (int, float))),
        default=None,
    )
    fastest_gpu = min(
        (row for row in gpu_rows if isinstance(row["median_end_to_end_us"], (int, float))),
        key=lambda row: float(row["median_end_to_end_us"]),
        default=None,
    )
    fastest_gpu_median = (
        float(fastest_gpu["median_end_to_end_us"]) if fastest_gpu and fastest_gpu["median_end_to_end_us"] is not None else None
    )
    cpu_checksum = cpu_rows[0].get("checksum") if cpu_rows else None
    for row in rows:
        if cpu_checksum is not None and row.get("checksum") not in {None, cpu_checksum}:
            blockers.append("checksum_mismatch_vs_cpu")
    target_boundaries = sorted({str(row["target_boundary"]) for row in rows})
    if len(target_boundaries) > 2:
        blockers.append("mixed_target_boundary_review")
    if cpu_median is None or fastest_gpu_median is None:
        recommendation = "blocked"
        speedup = None
        promotion_decision = "blocked"
        promoted_backend = None
        promotion_speedup = None
    else:
        ratio = cpu_median / fastest_gpu_median if fastest_gpu_median > 0 else None
        if ratio is not None and abs(cpu_median - fastest_gpu_median) / min(cpu_median, fastest_gpu_median) <= 0.05:
            recommendation = "threshold_boundary"
        elif cpu_median <= fastest_gpu_median:
            recommendation = "cpu_wins"
        else:
            recommendation = "gpu_wins"
        speedup = ratio
        if blockers:
            promotion_decision = "blocked"
            promoted_backend = None
            promotion_speedup = None
        elif recommendation == "gpu_wins" and ratio is not None and ratio >= PROMOTION_MARGIN:
            promotion_decision = "promote_gpu_selector_threshold"
            promoted_backend = fastest_gpu.get("backend") if fastest_gpu else None
            promotion_speedup = ratio
        elif recommendation == "cpu_wins" and ratio is not None and ratio > 0 and (1.0 / ratio) >= PROMOTION_MARGIN:
            promotion_decision = "promote_cpu_selector_threshold"
            promoted_backend = "cpu-reference"
            promotion_speedup = 1.0 / ratio
        elif recommendation == "threshold_boundary":
            promotion_decision = "keep_threshold_boundary"
            promoted_backend = None
            promotion_speedup = None
        else:
            promotion_decision = "keep_experimental"
            promoted_backend = None
            promotion_speedup = None
    return {
        "family": key[0],
        "name": key[1],
        "semantics": key[2],
        "finite_modulus": key[3],
        "output_contract": key[4],
        "shape": {"m": key[5], "n": key[6], "k": key[7]},
        "cpu_median_end_to_end_us": cpu_median,
        "fastest_gpu_backend": fastest_gpu.get("backend") if fastest_gpu else None,
        "fastest_gpu_median_end_to_end_us": fastest_gpu_median,
        "cpu_vs_fastest_gpu_ratio": speedup,
        "recommendation": recommendation,
        "selector_explanation": "reviewed_selector_threshold_recommendation_only",
        "promotion_decision": promotion_decision,
        "promoted_backend": promoted_backend,
        "promotion_speedup_vs_next_best": promotion_speedup,
        "promotion_margin_required": PROMOTION_MARGIN,
        "promotion_scope": "local_selector_threshold_recommendation_not_cache_entry",
        "target_boundaries": target_boundaries,
        "status": "blocked" if blockers else "reviewed",
        "blockers": sorted(set(blockers)),
        "rows": rows,
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    captures = [
        capture
        for capture in load_report_captures(paths)
        if isinstance(capture.get("cpu_small_shape_selector"), dict)
        and capture.get("cpu_small_shape_selector", {}).get("enabled") is True
    ]
    grouped: dict[tuple[str, str, str, str, str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        grouped[_group_key(capture)].append(row_for_capture(capture))
    groups = [_group_report(key, rows) for key, rows in sorted(grouped.items(), key=lambda item: str(item[0]))]
    blocker_counts = Counter(blocker for group in groups for blocker in group["blockers"])
    promotable_thresholds = [
        {
            "name": group["name"],
            "semantics": group["semantics"],
            "shape": group["shape"],
            "recommendation": group["recommendation"],
            "promotion_decision": group["promotion_decision"],
            "promoted_backend": group["promoted_backend"],
            "promotion_speedup_vs_next_best": group["promotion_speedup_vs_next_best"],
            "target_boundaries": group["target_boundaries"],
        }
        for group in groups
        if str(group.get("promotion_decision", "")).startswith("promote_")
    ]
    return {
        "schema": "rns8_cpu_small_shape_selector_report_v1",
        "policy": "reviewed_selector_threshold_recommendations_no_cache_or_default_route_install",
        "rank69_gate_complete": bool(groups) and not any(group["blockers"] for group in groups),
        "promotable_selector_threshold_count": len(promotable_thresholds),
        "promotable_selector_thresholds": promotable_thresholds,
        "capture_count": len(captures),
        "group_count": len(groups),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "groups": groups,
    }


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cpu-small-shape-selector-report.json"
    md_path = out_dir / "cpu-small-shape-selector-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# CPU Small-Shape Selector Report",
        "",
        f"- Policy: `{report['policy']}`",
        f"- Rank 69 gate complete: `{report['rank69_gate_complete']}`",
        f"- Promotable selector thresholds: `{report['promotable_selector_threshold_count']}`",
        "",
        "| workload | shape | recommendation | promoted backend | speedup | blockers |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for group in report["groups"]:
        shape = group["shape"]
        speedup = group["promotion_speedup_vs_next_best"]
        lines.append(
            "| `{name}` | `{m}x{n}x{k}` | `{recommendation}` | `{backend}` | `{speedup}` | `{blockers}` |".format(
                name=group["name"],
                m=shape["m"],
                n=shape["n"],
                k=shape["k"],
                recommendation=group["recommendation"],
                backend=group["promoted_backend"] or "none",
                speedup=f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "n/a",
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
    if args.require_complete and not report["rank69_gate_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
