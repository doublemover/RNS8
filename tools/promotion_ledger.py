#!/usr/bin/env python3
"""Audit reviewed benchmark evidence against an autotune promotion ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "promotion-ledgers"
MIN_SPEEDUP_MARGIN = 1.02


def path_key(path: str | Path) -> str:
    return str(Path(path).resolve())


def load_cache_entries(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    entries = data.get("entries")
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    return []


def load_review_entries(paths: list[Path]) -> dict[str, dict[str, Any]]:
    reviewed: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        for group in data.get("groups", []):
            if not isinstance(group, dict):
                continue
            for candidate in group.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                capture_path = candidate.get("capture")
                if isinstance(capture_path, str) and capture_path:
                    reviewed[path_key(capture_path)] = candidate
    return reviewed


def load_variance_entries(paths: list[Path]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        for entry in data.get("entries", []):
            if not isinstance(entry, dict):
                continue
            capture_key = entry.get("capture_key")
            capture_path = entry.get("capture")
            if isinstance(capture_key, str) and capture_key:
                entries[capture_key] = entry
            elif isinstance(capture_path, str) and capture_path:
                entries[path_key(capture_path)] = entry
    return entries


def feature_lane_requested(item: dict[str, Any]) -> bool:
    if item.get("requested") is True or item.get("enabled") is True or item.get("used") is True:
        return True
    status = item.get("capture_status") or item.get("status")
    return isinstance(status, str) and status not in {"not_requested", "not_applicable"}


def capture_entry(
    path: Path,
    reviewed_entry: dict[str, Any] | None = None,
    variance_entry: dict[str, Any] | None = None,
    *,
    variance_report_supplied: bool = False,
) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    baseline = capture.get("comparison_baseline") if isinstance(capture.get("comparison_baseline"), dict) else {}
    blockers: list[str] = []
    key = metadata.get("autotune_key")
    if not key:
        blockers.append("missing_autotune_key")

    reviewed_blockers: list[str] = []
    reviewed_promotable = False
    reviewed_speedup = None
    if reviewed_entry is not None:
        reviewed_blockers = [
            str(item)
            for item in reviewed_entry.get("promotion_blockers", [])
            if isinstance(item, str) and item
        ]
        reviewed_promotable = reviewed_entry.get("promotable") is True and not reviewed_blockers
        reviewed_speedup = reviewed_entry.get("speedup_vs_direct_hip")
        if not isinstance(reviewed_speedup, (int, float)):
            reviewed_speedup = reviewed_entry.get("speedup_vs_vector_alu")
        blockers.extend(f"review_blocker:{item}" for item in reviewed_blockers)

    raw_performance_validated = metadata.get("performance_validated") is True
    if not raw_performance_validated and not reviewed_promotable:
        blockers.append("not_performance_validated")

    raw_baseline_reviewed = baseline.get("status") == "reviewed_release_same_contract_baseline"
    if not raw_baseline_reviewed and not reviewed_promotable:
        blockers.append("missing_release_reviewed_baseline")
    speedup = baseline.get("speedup_vs_baseline_median_end_to_end")
    if isinstance(reviewed_speedup, (int, float)):
        speedup = reviewed_speedup
    if not isinstance(speedup, (int, float)) or speedup < MIN_SPEEDUP_MARGIN:
        blockers.append("missing_or_narrow_speedup_margin")
    variance_required_margin = None
    variance_observed_noise = None
    variance_ready = None
    if variance_report_supplied and variance_entry is None:
        blockers.append("missing_variance_gate_entry")
    elif variance_entry is not None:
        variance_ready = variance_entry.get("promotion_ready") is True
        variance_required_margin = variance_entry.get("required_speedup_margin")
        variance_observed_noise = variance_entry.get("observed_max_relative_noise")
        for blocker in variance_entry.get("blockers", []):
            if isinstance(blocker, str) and blocker:
                blockers.append(f"variance_blocker:{blocker}")
        if not variance_ready and not variance_entry.get("blockers"):
            blockers.append("variance_gate_not_ready")
        if (
            isinstance(speedup, (int, float))
            and isinstance(variance_required_margin, (int, float))
            and speedup <= variance_required_margin
        ):
            blockers.append("speedup_inside_variance_margin")
    for object_name in ("modulus_set", "export_variant", "reconstruction_variant", "grouped_dispatch", "hip_graph_replay"):
        item = capture.get(object_name)
        if (
            isinstance(item, dict)
            and item.get("promotion_eligible") is False
            and (object_name not in {"grouped_dispatch", "hip_graph_replay"} or feature_lane_requested(item))
        ):
            blockers.append(f"{object_name}_non_promoting")
        if object_name == "modulus_set" and isinstance(item, dict) and item.get("cache_promotion_blocker"):
            blockers.append(str(item.get("cache_promotion_blocker")))
    return {
        "path": str(path),
        "autotune_key": key,
        "backend_selected": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "validation_status": metadata.get("capability_status"),
        "performance_validated": raw_performance_validated or reviewed_promotable,
        "review_report_promotable": reviewed_promotable,
        "speedup_margin": speedup,
        "variance_gate_available": variance_entry is not None,
        "variance_gate_ready": variance_ready,
        "variance_required_speedup_margin": variance_required_margin,
        "variance_observed_max_relative_noise": variance_observed_noise,
        "promotion_blockers": sorted(set(blockers)),
    }


def build_ledger(
    captures: list[Path],
    cache_path: Path | None,
    review_reports: list[Path] | None = None,
    variance_reports: list[Path] | None = None,
) -> dict[str, Any]:
    reviewed = load_review_entries(review_reports or [])
    variance = load_variance_entries(variance_reports or [])
    variance_report_supplied = bool(variance_reports)
    entries = [
        capture_entry(
            path,
            reviewed.get(path_key(path)),
            variance.get(path_key(path)),
            variance_report_supplied=variance_report_supplied,
        )
        for path in captures
    ]
    cache_entries = load_cache_entries(cache_path)
    cache_keys = {entry.get("key") for entry in cache_entries if isinstance(entry.get("key"), str)}
    for entry in entries:
        key = entry.get("autotune_key")
        entry["installed_cache_entry"] = bool(key and key in cache_keys)
        if key and key not in cache_keys:
            entry["promotion_blockers"].append("missing_installed_cache_entry")
    return {
        "schema_version": 1,
        "policy": "reviewed_release_evidence_required_for_autotune_promotion",
        "cache_path": str(cache_path) if cache_path else None,
        "review_report_count": len(review_reports or []),
        "variance_report_count": len(variance_reports or []),
        "cache_entry_count": len(cache_entries),
        "entries": entries,
        "blocked_count": sum(1 for entry in entries if entry["promotion_blockers"]),
    }


def write_outputs(ledger: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "promotion-ledger.json"
    md_path = out_dir / "promotion-ledger.md"
    json_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RNS8 Promotion Ledger",
        "",
        f"- Policy: `{ledger['policy']}`",
        f"- Cache entries: `{ledger['cache_entry_count']}`",
        f"- Blocked entries: `{ledger['blocked_count']}`",
        "",
        "| capture | backend | kernel | cache entry | variance gate | required speedup | blockers |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for entry in ledger["entries"]:
        lines.append(
            "| `{path}` | `{backend}` | `{kernel}` | `{cache}` | `{variance}` | `{required}` | `{blockers}` |".format(
                path=entry["path"],
                backend=entry.get("backend_selected"),
                kernel=entry.get("selected_kernel"),
                cache=entry.get("installed_cache_entry"),
                variance=entry.get("variance_gate_ready"),
                required=entry.get("variance_required_speedup_margin"),
                blockers=", ".join(entry.get("promotion_blockers") or []),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="reviewed schema-v4 benchmark captures")
    parser.add_argument("--cache", type=Path, help="installed or candidate autotune cache JSON")
    parser.add_argument(
        "--review-report",
        type=Path,
        action="append",
        default=[],
        help="benchmark_sweep review_report.json that proves fastest promotable same-contract candidates",
    )
    parser.add_argument(
        "--variance-report",
        type=Path,
        action="append",
        default=[],
        help="perf_variance_report.py output that proves the win clears observed repeatability noise",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ledger = build_ledger(args.captures, args.cache, args.review_report, args.variance_report)
    if args.json:
        print(json.dumps(ledger, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(ledger, args.out_dir).items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
