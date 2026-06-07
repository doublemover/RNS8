#!/usr/bin/env python3
"""Summarize export/reconstruction selector evidence from benchmark captures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from report_capture_inputs import load_report_captures


DEFAULT_OUT_DIR = Path("temp") / "export-selector-reports"


def _timing_median(capture: dict[str, Any], phase: str) -> float | None:
    summary = capture.get("timing_summary_us")
    item = summary.get(phase) if isinstance(summary, dict) else None
    if isinstance(item, dict) and isinstance(item.get("median"), (int, float)):
        return float(item["median"])
    return None


def _target_id(capture: dict[str, Any], export: dict[str, Any]) -> str | None:
    for source in (export, capture.get("target_variant"), capture.get("device"), capture):
        if isinstance(source, dict) and isinstance(source.get("target_id"), str):
            return source["target_id"]
        if isinstance(source, dict) and isinstance(source.get("target_arch"), str):
            return source["target_arch"]
    return None


def _reviewable_export_variant(capture: dict[str, Any], export: dict[str, Any]) -> bool:
    name = export.get("name")
    if name == "default":
        return True
    status_policy = export.get("selector_status_policy")
    status_reviewable = status_policy == "range_checked_status_buffer" or (
        status_policy == "none"
        and isinstance(export.get("status_elision_reason"), str)
        and bool(export.get("status_elision_reason"))
    )
    return (
        name == "exact-wide-fixed-limb-export"
        and capture.get("semantics") in {"exact_wide_signed", "exact_wide_unsigned"}
        and export.get("semantic_contract") == capture.get("semantics")
        and export.get("output_layout") == "fixed_u64_limbs"
        and isinstance(export.get("limb_count"), int)
        and status_reviewable
        and export.get("d2h_policy") in {"host_ld_padded", "compact_contiguous"}
        and export.get("final_output_mode") == "final_host_output"
    )


def _blockers(capture: dict[str, Any]) -> list[str]:
    export = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
    blockers: list[str] = []
    if not export.get("selector_key"):
        blockers.append("missing_selector_key")
    if not export.get("stale_entry_reason"):
        blockers.append("missing_stale_entry_reason")
    selector_promotion_eligible = export.get("promotion_eligible") is True
    if selector_promotion_eligible and not _reviewable_export_variant(capture, export):
        blockers.append("non_default_export_variant_not_reviewable_for_promotion")
    if not selector_promotion_eligible:
        blocker = export.get("promotion_blocker")
        blockers.append(str(blocker) if isinstance(blocker, str) and blocker else "non_promoting_export_variant")
    correctness = capture.get("correctness")
    if correctness is None:
        blockers.append("correctness_not_recorded")
    elif correctness != "ok":
        blockers.append("correctness_not_ok")
    if capture.get("gpu_events", {}).get("requested") and not capture.get("gpu_events", {}).get("complete", False):
        blockers.append("gpu_events_incomplete")
    if export.get("selector_key") and export.get("selected_kernel"):
        if f"selected_kernel={export['selected_kernel']}" not in export["selector_key"]:
            blockers.append("selector_key_selected_kernel_mismatch")
    return blockers


def row_for_capture(capture: dict[str, Any]) -> dict[str, Any]:
    export = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
    reconstruction = (
        capture.get("reconstruction_variant") if isinstance(capture.get("reconstruction_variant"), dict) else {}
    )
    target_id = _target_id(capture, export)
    return {
        "capture_path": capture.get("_path"),
        "semantics": capture.get("semantics"),
        "backend": capture.get("backend_selected"),
        "target_id": target_id,
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "export_variant": export.get("name"),
        "reconstruction_variant": reconstruction.get("name"),
        "selector_key": export.get("selector_key"),
        "selector_policy": export.get("selector_policy"),
        "output_layout": export.get("output_layout"),
        "limb_count": export.get("limb_count"),
        "signedness": export.get("signedness"),
        "status_policy": export.get("selector_status_policy"),
        "d2h_policy": export.get("d2h_policy"),
        "final_output_mode": export.get("final_output_mode"),
        "selected_kernel": export.get("selected_kernel"),
        "cache_visibility": export.get("cache_visibility"),
        "stale_entry_reason": export.get("stale_entry_reason"),
        "status_elision_reason": export.get("status_elision_reason"),
        "median_end_to_end_us": _timing_median(capture, "end_to_end"),
        "median_export_us": _timing_median(capture, "crt_export"),
        "selector_promotion_eligible": export.get("promotion_eligible") is True,
        "promotion_eligible": export.get("promotion_eligible") is True,
        "promotion_blockers": _blockers(capture),
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows = [row_for_capture(capture) for capture in load_report_captures(paths)]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        shape = row["shape"]
        groups[
            (
                row["semantics"],
                row["backend"],
                row["target_id"],
                shape["m"],
                shape["n"],
                shape["k"],
                row["output_layout"],
                row["limb_count"],
                row["signedness"],
                row["final_output_mode"],
            )
        ].append(row)
    return {
        "schema": "rns8_export_selector_report_v1",
        "policy": "same_target_same_contract_export_selector_review_surface",
        "capture_count": len(rows),
        "groups": [
            {
                "key": {
                    "semantics": key[0],
                    "backend": key[1],
                    "target_id": key[2],
                    "m": key[3],
                    "n": key[4],
                    "k": key[5],
                    "output_layout": key[6],
                    "limb_count": key[7],
                    "signedness": key[8],
                    "final_output_mode": key[9],
                },
                "rows": sorted(value, key=lambda row: (row.get("median_end_to_end_us") or float("inf"))),
            }
            for key, value in sorted(groups.items(), key=lambda item: str(item[0]))
        ],
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "export-selector-report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(json_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
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
