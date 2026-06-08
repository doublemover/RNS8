#!/usr/bin/env python3
"""Review tile-shape benchmark captures with baseline and resource gates."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from report_capture_inputs import load_report_captures


DEFAULT_OUT_DIR = Path("temp") / "tile-shape-reports"
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
DEFAULT_TILE_M = 128
DEFAULT_TILE_N = 128


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _median(capture: dict[str, Any] | None, phase: str) -> float | None:
    if not capture:
        return None
    item = (capture.get("timing_summary_us") or {}).get(phase) if isinstance(capture.get("timing_summary_us"), dict) else None
    if isinstance(item, dict) and isinstance(item.get("median"), (int, float)):
        return float(item["median"])
    values = (capture.get("raw_timings_us") or {}).get(phase) if isinstance(capture.get("raw_timings_us"), dict) else None
    if isinstance(values, list) and values:
        numeric = sorted(float(value) for value in values if isinstance(value, (int, float)))
        if numeric:
            return numeric[len(numeric) // 2]
    avg_key = f"avg_{phase}_us"
    if isinstance(capture.get(avg_key), (int, float)):
        return float(capture[avg_key])
    return None


def _target_id(capture: dict[str, Any]) -> str | None:
    for source in (capture.get("target_variant"), capture.get("device"), capture):
        if isinstance(source, dict) and isinstance(source.get("target_id"), str):
            return source["target_id"]
        if isinstance(source, dict) and isinstance(source.get("target_arch"), str):
            return source["target_arch"]
        if isinstance(source, dict) and isinstance(source.get("gcn_arch"), str):
            return source["gcn_arch"]
    return None


def _backend(capture: dict[str, Any]) -> str:
    return str(capture.get("backend_selected") or capture.get("backend_requested") or "")


def _tile_variant(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("tile_shape_variant")
    return value if isinstance(value, dict) else {}


def _backend_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("backend_metadata")
    return value if isinstance(value, dict) else {}


def _schedule(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("schedule_metadata")
    return value if isinstance(value, dict) else {}


def _is_release_reviewed(capture: dict[str, Any]) -> bool:
    return int(capture.get("warmups", 0) or 0) >= RELEASE_MIN_WARMUPS and int(capture.get("repeats", 0) or 0) >= RELEASE_MIN_REPEATS


def _gpu_events_available(capture: dict[str, Any]) -> bool:
    timing = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    events = capture.get("gpu_event_timings_us")
    phases = timing.get("gpu_event_phase_order")
    return (
        timing.get("gpu_event_timing") is True
        and timing.get("gpu_event_timing_status") == "available"
        and timing.get("gpu_event_timing_source") == "hipEventElapsedTime"
        and isinstance(phases, list)
        and bool(phases)
        and isinstance(events, dict)
        and any(isinstance(events.get(phase), list) and events.get(phase) for phase in phases)
    )


def _contract_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    schedule = _schedule(capture)
    export_variant = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
    reconstruction_variant = (
        capture.get("reconstruction_variant") if isinstance(capture.get("reconstruction_variant"), dict) else {}
    )
    return (
        capture.get("semantics"),
        capture.get("bound_mode"),
        capture.get("bound_kind"),
        capture.get("m"),
        capture.get("n"),
        capture.get("k"),
        capture.get("finite_modulus"),
        capture.get("exact_wide_limb_count"),
        capture.get("output_logical_ld"),
        capture.get("output_ld_padding"),
        capture.get("pack_mode"),
        capture.get("prepack_reuse_strategy"),
        capture.get("selected_prefix"),
        capture.get("requested_max_prefix"),
        capture.get("contract_prefix_policy"),
        schedule.get("min_required_prefix"),
        schedule.get("max_required_prefix"),
        schedule.get("prefix_group_count"),
        export_variant.get("name", "default"),
        reconstruction_variant.get("name", "default_garner"),
    )


def _tile_shape_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    variant = _tile_variant(capture)
    return (
        variant.get("name", "legacy"),
        variant.get("tile_m", capture.get("tile_m")),
        variant.get("tile_n", capture.get("tile_n")),
        variant.get("tile_k", capture.get("k_block_size")),
        variant.get("k_block_policy", "auto"),
    )


def _is_default_tile_anchor(capture: dict[str, Any]) -> bool:
    variant = _tile_variant(capture)
    return (
        _backend(capture) == "hip-direct"
        and capture.get("tile_m") == DEFAULT_TILE_M
        and capture.get("tile_n") == DEFAULT_TILE_N
        and variant.get("k_block_policy", "auto") == "auto"
        and variant.get("name", "default") in {"default", "direct-hip-default-128x128"}
    )


def _is_tile_candidate(capture: dict[str, Any]) -> bool:
    variant = _tile_variant(capture)
    k_block_policy = variant.get("k_block_policy", "auto")
    non_default_tile = capture.get("tile_m") != DEFAULT_TILE_M or capture.get("tile_n") != DEFAULT_TILE_N
    non_default_k_block = k_block_policy not in {"", "auto"}
    return (
        _backend(capture) == "hip-direct"
        and bool(variant)
        and variant.get("name") not in {None, "", "default", "direct-hip-default-128x128"}
        and (non_default_tile or non_default_k_block)
    )


def _resource_report_key(capture: dict[str, Any]) -> str:
    value = _tile_variant(capture).get("resource_report_key")
    return value if isinstance(value, str) else ""


def _autotune_key(capture: dict[str, Any]) -> str:
    value = _backend_metadata(capture).get("autotune_key")
    return value if isinstance(value, str) else ""


def _autotune_key_has_tile_identity(capture: dict[str, Any]) -> bool:
    key = f";{_autotune_key(capture)};"
    tile_k = _tile_variant(capture).get("tile_k", capture.get("k_block_size"))
    selected_kernel = capture.get("selected_kernel")
    required = [
        f";tile_m={capture.get('tile_m')};",
        f";tile_n={capture.get('tile_n')};",
        f";k_block_size={tile_k};",
    ]
    if isinstance(selected_kernel, str) and selected_kernel:
        required.append(f";kernel={selected_kernel};")
    return all(item in key for item in required)


def _selected_kernel_has_tile_report_identity(capture: dict[str, Any]) -> bool:
    variant = _tile_variant(capture)
    selected_kernel = capture.get("selected_kernel")
    resource_key = _resource_report_key(capture)
    if not isinstance(selected_kernel, str) or not selected_kernel:
        return False
    return (
        variant.get("selected_kernel_identity") == selected_kernel
        and f"kernel={selected_kernel}" in resource_key
        and f"tile_m={capture.get('tile_m')}" in resource_key
        and f"tile_n={capture.get('tile_n')}" in resource_key
        and f"tile_k={variant.get('tile_k', capture.get('k_block_size'))}" in resource_key
    )


def _path_lookup_keys(path: Path) -> list[str]:
    keys = [str(path), path.as_posix(), path.name, path.stem]
    try:
        resolved = path.resolve()
    except OSError:
        resolved = None
    if resolved is not None:
        keys.extend([str(resolved), resolved.as_posix()])
    deduped: list[str] = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


def _resource_from_path(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"resource report must be a JSON object: {path}")
    if "resource_summary" in data or "evidence_status" in data:
        return data
    if "instruction_totals" in data:
        totals = data.get("instruction_totals") if isinstance(data.get("instruction_totals"), dict) else {}
        return {
            "resource_summary": {
                "vgpr": totals.get("vgpr_count"),
                "sgpr": totals.get("sgpr_count"),
                "lds_bytes": totals.get("lds_bytes"),
                "scratch_bytes": totals.get("scratch_bytes"),
                "lds_instruction_mentions": totals.get("lds_mentions"),
                "occupancy": totals.get("occupancy"),
                "global_store_instruction_count": totals.get("global_store"),
                "wait_instruction_count": totals.get("wait_instructions"),
                "matrix_instruction_count": totals.get("matrix_instruction_count")
                if totals.get("matrix_instruction_count") is not None
                else (totals.get("wmma") or 0) + (totals.get("mfma") or 0),
                "dense_integer_matrix_instruction_count": totals.get("dense_integer_matrix_instruction_count"),
                "sparse_integer_matrix_instruction_count": totals.get("sparse_integer_matrix_instruction_count"),
                "matrix_instruction_histogram": totals.get("matrix_instruction_histogram") or {},
            },
            "evidence_status": {
                "profiler_counter_status": "missing",
                "isa_resource_status": "present",
                "gpu_event_timing_status": "not_from_resource_report",
                "missing_evidence": ["missing_profiler_counter_export"],
            },
            "source": str(path),
        }
    raise RuntimeError(f"unsupported resource report shape: {path}")


def load_resource_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("tile resource manifest must be a JSON object")
    raw_entries: list[dict[str, Any]]
    if isinstance(data.get("captures"), list):
        raw_entries = [item for item in data["captures"] if isinstance(item, dict)]
    else:
        raw_entries = []
        for key, value in data.items():
            if isinstance(value, dict):
                entry = dict(value)
                entry.setdefault("capture", key)
                raw_entries.append(entry)
    manifest: dict[str, dict[str, Any]] = {}
    for entry in raw_entries:
        capture = entry.get("capture")
        if not isinstance(capture, str) or not capture:
            raise RuntimeError("tile resource manifest entries require a capture string")
        reports: list[dict[str, Any]] = []
        for key in ("gpu_counter_report", "counter_report", "isa_summary", "resource_report"):
            value = entry.get(key)
            if isinstance(value, str):
                report_path = Path(value)
                if not report_path.is_absolute():
                    report_path = path.parent / report_path
                reports.append(_resource_from_path(report_path))
        inline = entry.get("resource_summary")
        if isinstance(inline, dict):
            reports.append({"resource_summary": inline, "evidence_status": entry.get("evidence_status", {})})
        manifest[capture] = {"reports": reports}
    return manifest


def _resource_for_capture(capture: dict[str, Any], manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    capture_path = Path(str(capture.get("_path") or ""))
    for key in _path_lookup_keys(capture_path):
        entry = manifest.get(key)
        if entry:
            reports = entry.get("reports")
            if isinstance(reports, list) and reports:
                return _merge_resource_reports([item for item in reports if isinstance(item, dict)])
    return {}


def _merge_resource_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    missing: set[str] = set()
    profiler_present = False
    isa_present = False
    for report in reports:
        report_summary = report.get("resource_summary") if isinstance(report.get("resource_summary"), dict) else {}
        for key, value in report_summary.items():
            if summary.get(key) is None:
                summary[key] = value
        evidence = report.get("evidence_status") if isinstance(report.get("evidence_status"), dict) else {}
        profiler_present = profiler_present or evidence.get("profiler_counter_status") == "present"
        isa_present = isa_present or evidence.get("isa_resource_status") == "present"
        for item in evidence.get("missing_evidence") or []:
            if isinstance(item, str):
                missing.add(item)
    return {
        "resource_summary": summary,
        "evidence_status": {
            "profiler_counter_status": "present" if profiler_present else "missing",
            "isa_resource_status": "present" if isa_present else "missing",
            "missing_evidence": sorted(missing),
        },
    }


def _resource_status(resource: dict[str, Any]) -> dict[str, Any]:
    summary = resource.get("resource_summary") if isinstance(resource.get("resource_summary"), dict) else {}
    evidence = resource.get("evidence_status") if isinstance(resource.get("evidence_status"), dict) else {}
    evidence_present = (
        evidence.get("profiler_counter_status") == "present"
        or evidence.get("isa_resource_status") in {"present", "partial"}
    )
    register_present = _number(summary.get("vgpr")) is not None or _number(summary.get("sgpr")) is not None
    lds_present = _number(summary.get("lds_bytes")) is not None or _number(summary.get("lds_instruction_mentions")) is not None
    occupancy_present = _number(summary.get("occupancy")) is not None
    return {
        "resource_summary": summary,
        "evidence_status": evidence,
        "resource_evidence_present": evidence_present,
        "register_pressure_present": register_present,
        "lds_signal_present": lds_present,
        "occupancy_present": occupancy_present,
        "complete_for_promotion": bool(evidence_present and register_present and lds_present and occupancy_present),
    }


def _baseline_maps(captures: list[dict[str, Any]]) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    cpu: dict[tuple[Any, ...], dict[str, Any]] = {}
    direct: dict[tuple[Any, ...], dict[str, Any]] = {}
    for capture in captures:
        key = _contract_key(capture)
        backend = _backend(capture)
        if backend in {"cpu-reference", "cpu", "wrap64-byte-limb"}:
            cpu.setdefault(key, capture)
        elif _is_default_tile_anchor(capture):
            direct[(key, _target_id(capture))] = capture
    return cpu, direct


def _speedup(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> float | None:
    candidate_median = _median(candidate, "end_to_end")
    baseline_median = _median(baseline, "end_to_end") if baseline else None
    if candidate_median is None or baseline_median is None or candidate_median <= 0:
        return None
    return baseline_median / candidate_median


def row_for_candidate(
    candidate: dict[str, Any],
    cpu: dict[str, Any] | None,
    direct: dict[str, Any] | None,
    resource: dict[str, Any],
) -> dict[str, Any]:
    variant = _tile_variant(candidate)
    resource_status = _resource_status(resource)
    core_blockers: list[str] = []
    resource_blockers: list[str] = []
    if not _is_release_reviewed(candidate):
        core_blockers.append("candidate_not_release_reviewed")
    if cpu is None:
        core_blockers.append("missing_cpu_baseline")
    if direct is None:
        core_blockers.append("missing_default_direct_hip_tile_baseline")
    elif not _is_release_reviewed(direct):
        core_blockers.append("default_direct_hip_tile_baseline_not_release_reviewed")
    if direct is not None and not _gpu_events_available(direct):
        core_blockers.append("default_direct_hip_baseline_missing_gpu_events")
    if not _gpu_events_available(candidate):
        core_blockers.append("candidate_missing_gpu_events")
    if _median(candidate, "end_to_end") is None:
        core_blockers.append("candidate_missing_end_to_end_timing")
    if direct is not None and _median(direct, "end_to_end") is None:
        core_blockers.append("default_direct_hip_baseline_missing_end_to_end_timing")
    if not _autotune_key_has_tile_identity(candidate):
        core_blockers.append("autotune_key_missing_tile_identity")
    if not _selected_kernel_has_tile_report_identity(candidate):
        core_blockers.append("selected_kernel_missing_tile_report_identity")
    if variant.get("k_block_policy", "auto") not in {"", "auto"}:
        if variant.get("split_k_mode") != "single_gpu_no_split_k":
            core_blockers.append("k_block_split_mode_not_single_gpu")
        if not isinstance(variant.get("accumulator_safety_key"), str) or not variant.get("accumulator_safety_key"):
            core_blockers.append("missing_accumulator_safety_key")
        if not isinstance(variant.get("resource_report_required"), str) or not variant.get("resource_report_required"):
            core_blockers.append("missing_resource_report_requirement")
    if not resource_status["resource_evidence_present"]:
        resource_blockers.append("missing_counter_or_isa_resource_evidence")
    if not resource_status["register_pressure_present"]:
        resource_blockers.append("missing_vgpr_sgpr_signal")
    if not resource_status["lds_signal_present"]:
        resource_blockers.append("missing_lds_signal")
    if not resource_status["occupancy_present"]:
        resource_blockers.append("missing_occupancy_signal")
    speedup = _speedup(candidate, direct)
    if direct is not None and speedup is None:
        core_blockers.append("missing_default_direct_hip_speedup")

    blockers = core_blockers + resource_blockers
    if core_blockers:
        decision = "keep experimental"
    elif speedup is not None and speedup < 0.98:
        decision = "drop/deprioritize"
    elif resource_blockers:
        decision = "keep experimental"
    elif speedup is not None and speedup > 1.02:
        decision = "promote locally"
    else:
        decision = "keep experimental"
        blockers.append("no_setup_inclusive_default_tile_win")

    return {
        "capture_path": candidate.get("_path"),
        "semantics": candidate.get("semantics"),
        "backend": _backend(candidate),
        "target_id": _target_id(candidate),
        "shape": {"m": candidate.get("m"), "n": candidate.get("n"), "k": candidate.get("k")},
        "variant_name": variant.get("name", "legacy"),
        "tile_m": variant.get("tile_m", candidate.get("tile_m")),
        "tile_n": variant.get("tile_n", candidate.get("tile_n")),
        "tile_k": variant.get("tile_k", candidate.get("k_block_size")),
        "k_block_policy": variant.get("k_block_policy", "auto"),
        "split_k_mode": variant.get("split_k_mode", "single_gpu_no_split_k"),
        "shape_family_bucket": variant.get("shape_family_bucket"),
        "selected_kernel": candidate.get("selected_kernel"),
        "autotune_key": _autotune_key(candidate),
        "resource_report_key": _resource_report_key(candidate),
        "candidate_median_end_to_end_us": _median(candidate, "end_to_end"),
        "candidate_median_gemm_us": _median(candidate, "rns_gemm"),
        "candidate_median_pack_us": _median(candidate, "pack"),
        "candidate_median_export_us": _median(candidate, "crt_export"),
        "cpu_baseline_path": cpu.get("_path") if cpu else None,
        "cpu_baseline_median_end_to_end_us": _median(cpu, "end_to_end") if cpu else None,
        "default_direct_hip_baseline_path": direct.get("_path") if direct else None,
        "default_direct_hip_median_end_to_end_us": _median(direct, "end_to_end") if direct else None,
        "speedup_vs_default_direct_hip_tile": speedup,
        "release_reviewed": _is_release_reviewed(candidate),
        "candidate_gpu_events": _gpu_events_available(candidate),
        "default_direct_hip_gpu_events": _gpu_events_available(direct) if direct else False,
        "autotune_key_has_tile_identity": _autotune_key_has_tile_identity(candidate),
        "selected_kernel_has_tile_report_identity": _selected_kernel_has_tile_report_identity(candidate),
        "resource_status": resource_status,
        "promotion_eligible": decision == "promote locally",
        "promotion_blockers": blockers,
        "decision": decision,
    }


def build_report(paths: list[Path], resource_manifest: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    resources = resource_manifest or {}
    captures = load_report_captures(paths)
    candidates = [capture for capture in captures if _is_tile_candidate(capture)]
    cpu_map, direct_map = _baseline_maps(captures)
    rows = [
        row_for_candidate(
            candidate,
            cpu_map.get(_contract_key(candidate)),
            direct_map.get((_contract_key(candidate), _target_id(candidate))),
            _resource_for_capture(candidate, resources),
        )
        for candidate in candidates
    ]
    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        shape = row["shape"]
        grouped_rows[(row["semantics"], row["target_id"], shape["m"], shape["n"], shape["k"])].append(row)
    return {
        "schema": "rns8_tile_shape_report_v2",
        "capture_count": len(captures),
        "candidate_count": len(candidates),
        "default_direct_hip_anchor_count": len(direct_map),
        "cpu_anchor_count": len(cpu_map),
        "policy": (
            "tile-shape promotion requires schema-valid release captures, same-contract CPU and default "
            "Direct-HIP tile anchors, required GPU events, selected-kernel/resource identity, autotune keys "
            "with tile_m/tile_n/tile_k, and counter or ISA resource evidence before local promotion"
        ),
        "decision_counts": {
            decision: sum(1 for row in rows if row["decision"] == decision)
            for decision in ["promote locally", "keep experimental", "drop/deprioritize"]
        },
        "groups": [
            {
                "key": key,
                "rows": sorted(value, key=lambda row: row.get("candidate_median_end_to_end_us") or float("inf")),
            }
            for key, value in sorted(grouped_rows.items(), key=lambda item: str(item[0]))
        ],
        "rows": rows,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tile-shape-report.json"
    md_path = out_dir / "tile-shape-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Tile Shape Report",
        "",
        f"- capture_count: `{report['capture_count']}`",
        f"- candidate_count: `{report['candidate_count']}`",
        f"- policy: {report['policy']}",
        "",
        "| Capture | Shape | Tile | Kernel | Direct Speedup | Resource | Decision | Blockers |",
        "|---|---:|---:|---|---:|---|---|---|",
    ]
    for row in report["rows"]:
        speedup = row.get("speedup_vs_default_direct_hip_tile")
        status = row.get("resource_status") or {}
        resource = "complete" if status.get("complete_for_promotion") else "incomplete"
        lines.append(
            "| {capture} | {shape} | {tile} | `{kernel}` | {speedup} | {resource} | {decision} | {blockers} |".format(
                capture=Path(str(row.get("capture_path") or "unknown")).name,
                shape="{m}x{n}x{k}".format(**row["shape"]),
                tile=f"{row.get('tile_m')}x{row.get('tile_n')}x{row.get('tile_k')}",
                kernel=row.get("selected_kernel"),
                speedup=f"{float(speedup):.3f}x" if isinstance(speedup, (int, float)) else "n/a",
                resource=resource,
                decision=row["decision"],
                blockers=", ".join(row.get("promotion_blockers") or []) or "none",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--resource-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = build_report(args.captures, load_resource_manifest(args.resource_manifest))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    if args.require_complete and (
        report["candidate_count"] == 0 or any(row["decision"] == "keep experimental" for row in report["rows"])
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
