#!/usr/bin/env python3
"""Classify strict wrap64 Direct-HIP tuning captures for queue rank 68."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from report_capture_inputs import load_report_captures


DEFAULT_OUT_DIR = Path("temp") / "wrap64-direct-hip-tuning-reports"
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
DIRECT_V4_KERNEL = "direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4"
GRAPH_EXECUTION_MODE = "hip_graph_replay_wrap64_pack_gemm_export"
def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _median(capture: dict[str, Any] | None, phase: str) -> float | None:
    if not capture:
        return None
    summary = capture.get("timing_summary_us") if isinstance(capture.get("timing_summary_us"), dict) else {}
    item = summary.get(phase)
    if isinstance(item, dict) and _number(item.get("median")) is not None:
        return float(item["median"])
    raw = capture.get("raw_timings_us") if isinstance(capture.get("raw_timings_us"), dict) else {}
    values = raw.get(phase)
    if isinstance(values, list):
        numeric = sorted(float(value) for value in values if _number(value) is not None)
        if numeric:
            return numeric[len(numeric) // 2]
    avg = capture.get(f"avg_{phase}_us")
    return _number(avg)


def _target_id(capture: dict[str, Any]) -> str:
    for source in (capture.get("target_variant"), capture.get("device"), capture):
        if isinstance(source, dict):
            for key in ("target_id", "target_arch", "gcn_arch"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
    return "unknown"


def _execution_mode(capture: dict[str, Any]) -> str:
    value = capture.get("benchmark_execution_mode")
    if isinstance(value, str) and value:
        return value
    metadata = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    value = metadata.get("benchmark_execution_mode")
    return value if isinstance(value, str) and value else "unknown"


def _backend_id(capture: dict[str, Any]) -> str:
    if capture.get("backend_requested") == "rocwmma-wrap64-candidate":
        return "rocwmma-wrap64-candidate"
    value = capture.get("backend_selected") or capture.get("backend_requested")
    return str(value or "unknown")


def _pack_mode(capture: dict[str, Any]) -> str:
    value = capture.get("pack_mode")
    if isinstance(value, str) and value:
        return value
    metadata = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    value = metadata.get("pack_mode")
    return value if isinstance(value, str) and value else "per_repeat_repack"


def _tile_shape_variant(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("tile_shape_variant")
    return value if isinstance(value, dict) else {}


def _k_block_policy(capture: dict[str, Any]) -> str:
    variant = _tile_shape_variant(capture)
    value = variant.get("k_block_policy")
    if isinstance(value, str) and value:
        return value
    scenario = capture.get("scenario_metadata")
    if isinstance(scenario, dict):
        value = scenario.get("k_block_policy")
        if isinstance(value, str) and value:
            return value
    return "auto"


def _scenario_scope(capture: dict[str, Any]) -> str | None:
    scenario = capture.get("scenario_metadata")
    if not isinstance(scenario, dict):
        return None
    value = scenario.get("promotion_eligibility")
    if isinstance(value, str) and value:
        return value
    metadata = scenario.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("promotion_scope")
        if isinstance(value, str) and value:
            return value
    return None


def _release_reviewed(capture: dict[str, Any]) -> bool:
    return int(capture.get("warmups", 0) or 0) >= RELEASE_MIN_WARMUPS and int(capture.get("repeats", 0) or 0) >= RELEASE_MIN_REPEATS


def _gpu_events_available(capture: dict[str, Any]) -> bool:
    metadata = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    events = capture.get("gpu_event_timings_us")
    phases = metadata.get("gpu_event_phase_order")
    return (
        metadata.get("gpu_event_timing") is True
        and metadata.get("gpu_event_timing_status") == "available"
        and metadata.get("gpu_event_timing_source") == "hipEventElapsedTime"
        and isinstance(phases, list)
        and bool(phases)
        and isinstance(events, dict)
        and any(isinstance(events.get(phase), list) and events.get(phase) for phase in phases)
    )


def _isa_evidence(capture: dict[str, Any]) -> str | None:
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    value = metadata.get("isa_evidence")
    return value if isinstance(value, str) and value else None


def _critical_blockers(blockers: list[str]) -> list[str]:
    non_critical = {
        "benchmark_only_graph_replay",
        "explicit_reuse_contract_only",
        "matrix_engine_candidate_not_default_route",
        "scenario_scope_not_autotune_promotable",
        "single_gpu_k_block_candidate_only",
    }
    return [item for item in blockers if item not in non_critical]


def _role(capture: dict[str, Any]) -> str:
    backend = _backend_id(capture)
    kernel = str(capture.get("selected_kernel") or "")
    execution = _execution_mode(capture)
    pack = _pack_mode(capture)
    if backend == "wrap64-byte-limb":
        return "cpu_byte_limb_reference"
    if backend == "rocwmma-wrap64-candidate":
        return "matrix_engine_candidate"
    if execution == GRAPH_EXECUTION_MODE:
        return "hip_graph_full_path"
    if backend == "hip-direct" and _k_block_policy(capture) not in {"", "auto"}:
        return "direct_hip_k_block_candidate"
    if backend == "hip-direct" and "colpair" in kernel:
        return "direct_hip_colpair_candidate"
    if backend == "hip-direct" and pack != "per_repeat_repack":
        return "direct_hip_reuse_candidate"
    if backend == "hip-direct" and kernel == DIRECT_V4_KERNEL:
        return "direct_hip_v4_baseline"
    if backend == "hip-direct":
        return "direct_hip_other_candidate"
    return "other"


def _group_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    return (
        capture.get("semantics"),
        capture.get("m"),
        capture.get("n"),
        capture.get("k"),
        capture.get("output_logical_ld"),
        int(capture.get("output_ld_padding", 0) or 0),
    )


def _candidate_key(capture: dict[str, Any]) -> tuple[str, str, str, str, str]:
    variant = _tile_shape_variant(capture)
    variant_name = variant.get("name") if isinstance(variant.get("name"), str) else "default"
    policy = _k_block_policy(capture)
    return (_role(capture), _backend_id(capture), str(capture.get("selected_kernel") or "unknown"), variant_name, policy)


def _capture_row(capture: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": capture.get("_path"),
        "role": _role(capture),
        "backend": _backend_id(capture),
        "selected_kernel": capture.get("selected_kernel"),
        "execution_mode": _execution_mode(capture),
        "pack_mode": _pack_mode(capture),
        "tile_shape_variant": _tile_shape_variant(capture),
        "k_block_policy": _k_block_policy(capture),
        "median_end_to_end_us": _median(capture, "end_to_end"),
        "median_gemm_us": _median(capture, "rns_gemm"),
        "median_export_us": _median(capture, "crt_export"),
        "release_reviewed": _release_reviewed(capture),
        "gpu_events_available": _gpu_events_available(capture),
        "isa_evidence": _isa_evidence(capture),
        "scenario_scope": _scenario_scope(capture),
        "checksum_u64": capture.get("checksum_u64"),
    }


def _comparison(
    candidate: dict[str, Any],
    direct: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    row = _capture_row(candidate)
    blockers: list[str] = []
    role = row["role"]
    candidate_median = row["median_end_to_end_us"]
    direct_median = _median(direct, "end_to_end") if direct else None
    reference_checksum = reference.get("checksum_u64") if reference else None
    if reference is None:
        blockers.append("missing_wrap64_byte_limb_reference")
    if direct is None:
        blockers.append("missing_direct_hip_v4_baseline")
    if not row["release_reviewed"]:
        blockers.append("not_release_reviewed")
    if role != "hip_graph_full_path" and row["backend"] not in {"wrap64-byte-limb"} and not row["gpu_events_available"]:
        blockers.append("missing_required_gpu_events")
    if row["backend"] not in {"wrap64-byte-limb"} and not row["isa_evidence"]:
        blockers.append("missing_isa_evidence")
    if reference_checksum is not None and candidate.get("checksum_u64") != reference_checksum:
        blockers.append("checksum_mismatch_vs_wrap64_reference")
    if role == "hip_graph_full_path":
        blockers.append("benchmark_only_graph_replay")
    if role == "direct_hip_reuse_candidate":
        blockers.append("explicit_reuse_contract_only")
    if role == "direct_hip_k_block_candidate":
        blockers.append("single_gpu_k_block_candidate_only")
    if role == "matrix_engine_candidate":
        blockers.append("matrix_engine_candidate_not_default_route")
    scope = row["scenario_scope"]
    if scope and scope != "release_review_candidate":
        blockers.append("scenario_scope_not_autotune_promotable")
    speedup = direct_median / candidate_median if direct_median and candidate_median else None
    critical = _critical_blockers(blockers)
    if role == "cpu_byte_limb_reference":
        decision = "reference"
    elif role == "direct_hip_v4_baseline":
        decision = "baseline"
    elif critical:
        decision = "keep_experimental"
    elif speedup is not None and speedup >= 1.03:
        decision = "candidate_workload_win"
    elif speedup is not None:
        decision = "drop/deprioritize"
    else:
        decision = "keep_experimental"
    row.update(
        {
            "speedup_vs_direct_hip_v4": speedup,
            "direct_hip_v4_median_end_to_end_us": direct_median,
            "wrap64_reference_median_end_to_end_us": _median(reference, "end_to_end") if reference else None,
            "blockers": blockers,
            "critical_blockers": critical,
            "decision": decision,
        }
    )
    return row


def build_report_from_captures(captures: list[dict[str, Any]]) -> dict[str, Any]:
    wrap_captures = [item for item in captures if item.get("semantics") == "wrap_u64_mod_2_64"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for capture in wrap_captures:
        grouped[_group_key(capture)].append(capture)

    groups: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_role[_role(item)].append(item)
        reference = by_role.get("cpu_byte_limb_reference", [None])[0]
        direct_baselines = by_role.get("direct_hip_v4_baseline", [])
        direct = next((item for item in direct_baselines if _scenario_scope(item) == "release_review_candidate"), None)
        if direct is None:
            direct = direct_baselines[0] if direct_baselines else None
        candidate_rows = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for item in sorted(items, key=lambda capture: (_role(capture), _backend_id(capture), str(capture.get("_path")))):
            key_for_item = _candidate_key(item)
            if key_for_item in seen and _role(item) not in {"direct_hip_v4_baseline", "cpu_byte_limb_reference"}:
                continue
            seen.add(key_for_item)
            row = _comparison(item, direct, reference)
            candidate_rows.append(row)
            decisions[row["decision"]] += 1
        groups.append(
            {
                "key": {
                    "semantics": key[0],
                    "m": key[1],
                    "n": key[2],
                    "k": key[3],
                    "output_logical_ld": key[4],
                    "output_ld_padding": key[5],
                    "target_ids": sorted({_target_id(item) for item in items if _backend_id(item) != "wrap64-byte-limb"}),
                },
                "capture_count": len(items),
                "has_wrap64_reference": reference is not None,
                "has_direct_hip_v4_baseline": direct is not None,
                "comparisons": candidate_rows,
            }
        )

    summary = {
        "capture_count": len(wrap_captures),
        "group_count": len(groups),
        "candidate_workload_wins": decisions.get("candidate_workload_win", 0),
        "deprioritized": decisions.get("drop/deprioritize", 0),
        "experimental": decisions.get("keep_experimental", 0),
        "baselines": decisions.get("baseline", 0),
        "references": decisions.get("reference", 0),
        "missing_reference_groups": sum(1 for group in groups if not group["has_wrap64_reference"]),
        "missing_direct_baseline_groups": sum(1 for group in groups if not group["has_direct_hip_v4_baseline"]),
        "decisions": dict(sorted(decisions.items())),
    }
    return {
        "schema": "rns8_wrap64_direct_hip_tuning_report_v1",
        "policy": "strict_wrap64_same_contract_release_event_isa_gate",
        "summary": summary,
        "groups": groups,
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    captures = load_report_captures(paths)
    return build_report_from_captures(captures)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "none"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Wrap64 Direct-HIP Tuning Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        if key == "decisions":
            continue
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Comparisons", "", "| Shape | Role | Backend | Kernel | Median us | Speedup vs Direct-HIP v4 | Decision | Blockers |", "|---|---|---|---|---:|---:|---|---|"])
    for group in report["groups"]:
        shape = group["key"]
        shape_text = f"{shape['m']}x{shape['n']}x{shape['k']}"
        for item in group["comparisons"]:
            blockers = ",".join(item.get("blockers") or []) or "none"
            lines.append(
                f"| {shape_text} | {item.get('role')} | {item.get('backend')} | {item.get('selected_kernel')} | "
                f"{_fmt(item.get('median_end_to_end_us'))} | {_fmt(item.get('speedup_vs_direct_hip_v4'))} | "
                f"{item.get('decision')} | {blockers} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "wrap64-direct-hip-tuning-report.json"
    md_path = out_dir / "wrap64-direct-hip-tuning-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    return {"json": str(json_path), "markdown": str(md_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path, help="schema-v4 capture files or directories")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true", help="print report JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.captures)
    outputs = write_outputs(report, args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in outputs.items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
