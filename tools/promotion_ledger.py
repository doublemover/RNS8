#!/usr/bin/env python3
"""Audit reviewed benchmark evidence against an autotune promotion ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture
import target_validation_report


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


def load_target_validation_groups(paths: list[Path]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        for group in data.get("groups", []):
            if not isinstance(group, dict):
                continue
            key = group.get("target_validation_group") or group.get("review_group_key")
            if isinstance(key, str) and key:
                groups[key] = group
    return groups


def feature_lane_requested(item: dict[str, Any]) -> bool:
    if item.get("requested") is True or item.get("enabled") is True or item.get("used") is True:
        return True
    status = item.get("capture_status") or item.get("status")
    return isinstance(status, str) and status not in {"not_requested", "not_applicable"}


def selector_key_hash(selector_key: Any) -> str | None:
    if not isinstance(selector_key, str) or not selector_key:
        return None
    return hashlib.sha256(selector_key.encode("utf-8")).hexdigest()[:16]


def capture_entry(
    path: Path,
    reviewed_entry: dict[str, Any] | None = None,
    variance_entry: dict[str, Any] | None = None,
    *,
    variance_report_supplied: bool = False,
    variance_gate_required: bool = False,
    target_validation_entry: dict[str, Any] | None = None,
    target_validation_report_supplied: bool = False,
) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    target_row = target_validation_report.capture_target(path)
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    baseline = capture.get("comparison_baseline") if isinstance(capture.get("comparison_baseline"), dict) else {}
    export_variant = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
    reconstruction_variant = (
        capture.get("reconstruction_variant") if isinstance(capture.get("reconstruction_variant"), dict) else {}
    )
    export_variant_name = str(export_variant.get("name") or "default")
    reconstruction_variant_name = str(reconstruction_variant.get("name") or "default_garner")
    export_selector_key = export_variant.get("selector_key")
    cache_scope = (
        "runtime_exact_autotune"
        if export_variant_name == "default" and reconstruction_variant_name == "default_garner"
        else "selector_review_only_non_default"
    )
    shape = {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")}
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
    if (variance_gate_required or variance_report_supplied) and variance_entry is None:
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
    target_validation_ready = None
    target_cache_eligible = None
    target_cache_blockers: list[str] = []
    if target_validation_report_supplied and target_validation_entry is None:
        blockers.append("missing_target_validation_group")
    elif target_validation_entry is not None:
        eligibility = target_validation_entry.get("cache_eligibility")
        if not isinstance(eligibility, dict):
            eligibility = {}
        target_cache_eligible = eligibility.get("eligible") is True
        target_cache_blockers = [
            str(item)
            for item in eligibility.get("blockers", [])
            if isinstance(item, str) and item
        ]
        target_validation_ready = target_cache_eligible and not target_cache_blockers
        blockers.extend(f"target_validation_blocker:{item}" for item in target_cache_blockers)
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
        if object_name == "export_variant" and isinstance(item, dict):
            if not item.get("selector_key"):
                blockers.append("export_selector_key_missing")
            if not item.get("stale_entry_reason"):
                blockers.append("export_selector_stale_reason_missing")
    tile_variant = capture.get("tile_shape_variant")
    if isinstance(tile_variant, dict):
        if tile_variant.get("k_block_policy") not in {None, "auto"}:
            blockers.append("non_default_k_block_policy_requires_same_target_counter_review")
        if tile_variant.get("accumulator_safety_key") is None:
            blockers.append("tile_shape_accumulator_safety_key_missing")
    arena = capture.get("workspace_arena")
    if isinstance(arena, dict) and arena.get("enabled"):
        if arena.get("measured_repeat_allocation_free") is not True:
            blockers.append("workspace_arena_repeat_allocation_not_free")
        repeat_delta = arena.get("measured_repeat_allocation_delta")
        if isinstance(repeat_delta, dict) and (
            int(repeat_delta.get("allocate_calls") or 0) != 0
            or int(repeat_delta.get("free_calls") or 0) != 0
            or int(repeat_delta.get("allocated_bytes") or 0) != 0
        ):
            blockers.append("workspace_arena_repeat_allocation_delta_nonzero")
    if cache_scope != "runtime_exact_autotune":
        blockers.append("selector_review_only_not_runtime_cache_route")
    return {
        "path": str(path),
        "autotune_key": key,
        "backend_selected": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "semantic_contract": capture.get("semantics"),
        "shape": shape,
        "target_id": target_row.get("target_id"),
        "target_class": target_row.get("target_class"),
        "target_family": target_row.get("target_family"),
        "host_os": target_row.get("host_os"),
        "target_validation_group": target_row.get("target_validation_group"),
        "target_validation_gate_available": target_validation_entry is not None,
        "target_validation_gate_ready": target_validation_ready,
        "target_cache_eligible": target_cache_eligible,
        "target_cache_blockers": sorted(set(target_cache_blockers)),
        "validation_status": metadata.get("capability_status"),
        "performance_validated": raw_performance_validated or reviewed_promotable,
        "review_report_promotable": reviewed_promotable,
        "speedup_margin": speedup,
        "variance_gate_available": variance_entry is not None,
        "variance_gate_ready": variance_ready,
        "variance_required_speedup_margin": variance_required_margin,
        "variance_observed_max_relative_noise": variance_observed_noise,
        "promotion_blockers": sorted(set(blockers)),
        "export_variant": export_variant_name,
        "reconstruction_variant": reconstruction_variant_name,
        "export_selector_key": export_selector_key,
        "export_selector_hash": selector_key_hash(export_selector_key),
        "export_selector_policy": export_variant.get("selector_policy"),
        "export_cache_visibility": export_variant.get("cache_visibility"),
        "export_stale_entry_reason": export_variant.get("stale_entry_reason"),
        "cache_scope": cache_scope,
        "shape_family_recommendation_status": "exact_cache_only_no_family_routing",
    }


def stale_invalidation_reasons(entry: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for blocker in entry.get("promotion_blockers", []):
        text = str(blocker)
        if "target" in text:
            reasons.append("target_id_or_target_validation_mismatch")
        elif "variance" in text or "margin" in text or "speedup" in text:
            reasons.append("margin_below_required_threshold")
        elif "kernel" in text:
            reasons.append("selected_kernel_mismatch")
        elif "epilogue" in text:
            reasons.append("epilogue_mismatch")
        elif "workspace" in text:
            reasons.append("workspace_mismatch")
        elif (
            "export_selector" in text
            or "export_variant" in text
            or "reconstruction_variant" in text
            or "selector_review" in text
        ):
            reasons.append("export_selector_contract_mismatch")
        elif "schema" in text:
            reasons.append("schema_mismatch")
        elif "review" in text or "evidence" in text or "performance" in text:
            reasons.append("evidence_missing_or_not_promoted")
    if not reasons and entry.get("promotion_blockers"):
        reasons.append("promotion_blockers_present")
    return sorted(set(reasons))


def cache_coverage(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        shape = entry.get("shape") if isinstance(entry.get("shape"), dict) else {}
        shape_family = "unknown"
        if all(isinstance(shape.get(key), int) for key in ("m", "n", "k")):
            dims = [int(shape[key]) for key in ("m", "n", "k")]
            max_dim = max(dims)
            shape_family = "small" if max_dim <= 128 else "medium" if max_dim <= 1024 else "large"
        key = (
            str(entry.get("semantic_contract") or "unknown"),
            shape_family,
            str(entry.get("backend_selected") or "unknown"),
            str(entry.get("target_class") or "unknown"),
            str(entry.get("target_id") or "unknown"),
            str(entry.get("export_variant") or "default"),
            str(entry.get("reconstruction_variant") or "default_garner"),
        )
        groups.setdefault(key, []).append(entry)
    rows: list[dict[str, Any]] = []
    for (semantic, shape_family, backend, target_class, target_id, export_variant, reconstruction_variant), grouped in sorted(
        groups.items()
    ):
        rows.append(
            {
                "semantic_contract": semantic,
                "shape_family": shape_family,
                "backend": backend,
                "target_class": target_class,
                "target_id": target_id,
                "export_variant": export_variant,
                "reconstruction_variant": reconstruction_variant,
                "entry_count": len(grouped),
                "installed_count": sum(1 for item in grouped if item.get("installed_cache_entry") is True),
                "eligible_count": sum(1 for item in grouped if not item.get("promotion_blockers")),
                "blocked_count": sum(1 for item in grouped if item.get("promotion_blockers")),
            }
        )
    return rows


def build_ledger(
    captures: list[Path],
    cache_path: Path | None,
    review_reports: list[Path] | None = None,
    variance_reports: list[Path] | None = None,
    target_validation_reports: list[Path] | None = None,
    *,
    require_variance_gate: bool = False,
) -> dict[str, Any]:
    reviewed = load_review_entries(review_reports or [])
    variance = load_variance_entries(variance_reports or [])
    target_groups = load_target_validation_groups(target_validation_reports or [])
    variance_report_supplied = bool(variance_reports)
    entries = [
        (
            lambda target_row: capture_entry(
                path,
                reviewed.get(path_key(path)),
                variance.get(path_key(path)),
                variance_report_supplied=variance_report_supplied,
                variance_gate_required=require_variance_gate,
                target_validation_entry=target_groups.get(str(target_row.get("target_validation_group"))),
                target_validation_report_supplied=bool(target_validation_reports),
            )
        )(target_validation_report.capture_target(path))
        for path in captures
    ]
    cache_entries = load_cache_entries(cache_path)
    cache_keys = {entry.get("key") for entry in cache_entries if isinstance(entry.get("key"), str)}
    for entry in entries:
        key = entry.get("autotune_key")
        entry["installed_cache_entry"] = bool(key and key in cache_keys)
        if key and key not in cache_keys:
            entry["promotion_blockers"].append("missing_installed_cache_entry")
        entry["stale_invalidation_reasons"] = stale_invalidation_reasons(entry)
    return {
        "schema_version": 1,
        "policy": "reviewed_release_evidence_required_for_autotune_promotion",
        "cache_path": str(cache_path) if cache_path else None,
        "review_report_count": len(review_reports or []),
        "variance_report_count": len(variance_reports or []),
        "require_variance_gate": require_variance_gate,
        "target_validation_report_count": len(target_validation_reports or []),
        "cache_entry_count": len(cache_entries),
        "entries": entries,
        "cache_coverage": cache_coverage(entries),
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
        "| capture | backend | target | cache entry | variance gate | target gate | required speedup | blockers |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for entry in ledger["entries"]:
        lines.append(
            "| `{path}` | `{backend}` | `{target}` | `{cache}` | `{variance}` | `{target_gate}` | `{required}` | `{blockers}` |".format(
                path=entry["path"],
                backend=entry.get("backend_selected"),
                target=entry.get("target_validation_group"),
                cache=entry.get("installed_cache_entry"),
                variance=entry.get("variance_gate_ready"),
                target_gate=entry.get("target_validation_gate_ready"),
                required=entry.get("variance_required_speedup_margin"),
                blockers=", ".join(entry.get("promotion_blockers") or []),
            )
        )
    lines.extend(
        [
            "",
            "## Cache Coverage",
            "",
            "| semantic | shape family | backend | target | export variant | reconstruction | entries | installed | eligible | blocked |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in ledger.get("cache_coverage", []):
        lines.append(
            "| `{semantic}` | `{shape}` | `{backend}` | `{target}` | `{export}` | `{reconstruction}` | {entries} | {installed} | {eligible} | {blocked} |".format(
                semantic=row["semantic_contract"],
                shape=row["shape_family"],
                backend=row["backend"],
                target=row["target_id"],
                export=row.get("export_variant", "default"),
                reconstruction=row.get("reconstruction_variant", "default_garner"),
                entries=row["entry_count"],
                installed=row["installed_count"],
                eligible=row["eligible_count"],
                blocked=row["blocked_count"],
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
    parser.add_argument(
        "--require-variance-gate",
        action="store_true",
        help="block every promotion ledger row that does not have a matching ready perf_variance_report entry",
    )
    parser.add_argument(
        "--target-validation-report",
        type=Path,
        action="append",
        default=[],
        help="target_validation_report.py output that proves matching OS/target/toolchain cache eligibility",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ledger = build_ledger(
        args.captures,
        args.cache,
        args.review_report,
        args.variance_report,
        args.target_validation_report,
        require_variance_gate=args.require_variance_gate,
    )
    if args.json:
        print(json.dumps(ledger, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(ledger, args.out_dir).items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
