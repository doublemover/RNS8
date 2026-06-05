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


def capture_entry(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    baseline = capture.get("comparison_baseline") if isinstance(capture.get("comparison_baseline"), dict) else {}
    speedup = baseline.get("speedup_vs_baseline_median_end_to_end")
    blockers: list[str] = []
    key = metadata.get("autotune_key")
    if not key:
        blockers.append("missing_autotune_key")
    if metadata.get("performance_validated") is not True:
        blockers.append("not_performance_validated")
    if baseline.get("status") != "reviewed_release_same_contract_baseline":
        blockers.append("missing_release_reviewed_baseline")
    if not isinstance(speedup, (int, float)) or speedup < MIN_SPEEDUP_MARGIN:
        blockers.append("missing_or_narrow_speedup_margin")
    for object_name in ("modulus_set", "export_variant", "reconstruction_variant", "grouped_dispatch", "hip_graph_replay"):
        item = capture.get(object_name)
        if isinstance(item, dict) and item.get("promotion_eligible") is False:
            blockers.append(f"{object_name}_non_promoting")
        if object_name == "modulus_set" and isinstance(item, dict) and item.get("cache_promotion_blocker"):
            blockers.append(str(item.get("cache_promotion_blocker")))
    return {
        "path": str(path),
        "autotune_key": key,
        "backend_selected": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "validation_status": metadata.get("capability_status"),
        "performance_validated": metadata.get("performance_validated"),
        "speedup_margin": speedup,
        "promotion_blockers": sorted(set(blockers)),
    }


def build_ledger(captures: list[Path], cache_path: Path | None) -> dict[str, Any]:
    entries = [capture_entry(path) for path in captures]
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
        "| capture | backend | kernel | cache entry | blockers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in ledger["entries"]:
        lines.append(
            "| `{path}` | `{backend}` | `{kernel}` | `{cache}` | `{blockers}` |".format(
                path=entry["path"],
                backend=entry.get("backend_selected"),
                kernel=entry.get("selected_kernel"),
                cache=entry.get("installed_cache_entry"),
                blockers=", ".join(entry.get("promotion_blockers") or []),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="reviewed schema-v4 benchmark captures")
    parser.add_argument("--cache", type=Path, help="installed or candidate autotune cache JSON")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ledger = build_ledger(args.captures, args.cache)
    if args.json:
        print(json.dumps(ledger, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(ledger, args.out_dir).items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
