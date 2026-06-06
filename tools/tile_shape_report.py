#!/usr/bin/env python3
"""Group tile-shape benchmark captures with timing and resource evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "tile-shape-reports"


def _median(capture: dict[str, Any], phase: str) -> float | None:
    item = (capture.get("timing_summary_us") or {}).get(phase) if isinstance(capture.get("timing_summary_us"), dict) else None
    if isinstance(item, dict) and isinstance(item.get("median"), (int, float)):
        return float(item["median"])
    return None


def _target_id(capture: dict[str, Any]) -> str | None:
    for source in (capture.get("target_variant"), capture.get("device"), capture):
        if isinstance(source, dict) and isinstance(source.get("target_id"), str):
            return source["target_id"]
        if isinstance(source, dict) and isinstance(source.get("target_arch"), str):
            return source["target_arch"]
    return None


def row_for_capture(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    variant = capture.get("tile_shape_variant") if isinstance(capture.get("tile_shape_variant"), dict) else {}
    safety = (capture.get("backend_metadata") or {}).get("accumulator_safety") or {}
    k_block_policy = variant.get("k_block_policy", "auto")
    blockers = ["tile_shape_report_evidence_only"]
    if k_block_policy != "auto":
        blockers.append("non_default_k_block_policy_requires_release_ab_and_counter_report")
    if not variant.get("resource_report_key"):
        blockers.append("missing_resource_report_key")
    if not variant.get("accumulator_safety_key"):
        blockers.append("missing_accumulator_safety_key")
    if safety.get("safe_for_k_block") is not True:
        blockers.append("accumulator_safety_not_proven")
    return {
        "path": str(path),
        "semantics": capture.get("semantics"),
        "backend_selected": capture.get("backend_selected"),
        "target_id": _target_id(capture),
        "selected_kernel": capture.get("selected_kernel"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "tile_m": variant.get("tile_m", capture.get("tile_m")),
        "tile_n": variant.get("tile_n", capture.get("tile_n")),
        "tile_k": variant.get("tile_k", capture.get("k_block_size")),
        "k_block_policy": k_block_policy,
        "split_k_mode": variant.get("split_k_mode", "single_gpu_no_split_k"),
        "accumulator_safety_key": variant.get("accumulator_safety_key"),
        "resource_report_required": variant.get("resource_report_required"),
        "variant_name": variant.get("name", "legacy"),
        "shape_family_bucket": variant.get("shape_family_bucket"),
        "resource_report_key": variant.get("resource_report_key"),
        "median_end_to_end_us": _median(capture, "end_to_end"),
        "median_pack_us": _median(capture, "pack"),
        "median_gemm_us": _median(capture, "rns_gemm"),
        "median_export_us": _median(capture, "crt_export"),
        "promotion_eligible": False,
        "promotion_blockers": blockers,
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows = [row_for_capture(path) for path in paths]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        shape = row["shape"]
        groups[(row["semantics"], row["backend_selected"], row["target_id"], shape["m"], shape["n"], shape["k"])].append(row)
    return {
        "schema_version": 1,
        "policy": "tile_shape_and_k_block_evidence_only_requires_same_contract_resource_counter_review",
        "groups": [
            {"key": key, "rows": sorted(value, key=lambda row: (row.get("median_end_to_end_us") or float("inf")))}
            for key, value in sorted(groups.items(), key=lambda item: str(item[0]))
        ],
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tile-shape-report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.captures)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
