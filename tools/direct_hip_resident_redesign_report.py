#!/usr/bin/env python3
"""Classify Direct-HIP resident redesign candidates after colpair rejection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture
from promotion_ledger import path_key


DEFAULT_OUT_DIR = Path("temp") / "direct-hip-resident-redesign-reports"
DEFAULT_MARGIN = 1.02
DEFAULT_KERNEL = "direct_hip_tiled_active_prefix_rns_gemm_v2"
GROUPED_ACTIVE_SCHEDULE_KERNEL = "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3"
REJECTED_COLPAIR_KERNEL = "direct_hip_active_prefix_colpair_grouped_rns_gemm_v2"
NON_CAPTURE_JSON_NAMES = {
    "direct-hip-resident-redesign-report.json",
    "direct-hip-prefix-fusion-report.json",
    "review_report.json",
    "scenario_manifest.json",
    "validation-summary.json",
}
REQUIRED_REDESIGN_DIMENSIONS = {
    "data_layout",
    "tile_shape",
    "export_interaction",
    "schedule_upload",
    "workspace_reuse",
}
EVIDENCE_BLOCKERS = {
    "missing_resident_default_baseline",
    "candidate_schema_invalid",
    "baseline_schema_invalid",
    "candidate_not_release_review",
    "baseline_not_release_review",
    "candidate_missing_required_gpu_events",
    "baseline_missing_required_gpu_events",
    "checksum_mismatch_with_resident_default",
    "missing_end_to_end_median",
    "missing_resource_explanation",
}


def _schema_status(capture: dict[str, Any]) -> str:
    value = capture.get("_schema_status")
    return str(value) if isinstance(value, str) else "unknown"


def _schema_errors(capture: dict[str, Any]) -> list[str]:
    errors = capture.get("_schema_errors")
    return [str(error) for error in errors] if isinstance(errors, list) else []


def _capture_path(capture: dict[str, Any] | None) -> str | None:
    if not capture:
        return None
    value = capture.get("_path")
    return str(value) if value is not None else None


def _selected_kernel(capture: dict[str, Any] | None) -> str:
    if not capture:
        return ""
    value = capture.get("selected_kernel")
    if isinstance(value, str) and value:
        return value
    metadata = capture.get("backend_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("selected_kernel"), str):
        return str(metadata["selected_kernel"])
    return ""


def _release_capture_satisfied(capture: dict[str, Any] | None) -> bool:
    return bool(
        capture
        and isinstance(capture.get("warmups"), int)
        and isinstance(capture.get("repeats"), int)
        and capture["warmups"] >= 3
        and capture["repeats"] >= 9
    )


def _gpu_events_available(capture: dict[str, Any] | None) -> bool:
    metadata = capture.get("timing_metadata") if capture else None
    events = capture.get("gpu_event_timings_us") if capture else None
    return bool(
        isinstance(metadata, dict)
        and metadata.get("gpu_event_timing") is True
        and metadata.get("gpu_event_timing_status") == "available"
        and isinstance(metadata.get("gpu_event_phase_order"), list)
        and metadata.get("gpu_event_phase_order")
        and isinstance(events, dict)
        and events
    )


def _timing_value(capture: dict[str, Any] | None, phase: str, statistic: str = "median") -> float | None:
    if capture is None:
        return None
    summary = capture.get("timing_summary_us")
    if isinstance(summary, dict):
        phase_summary = summary.get(phase)
        if isinstance(phase_summary, dict) and isinstance(phase_summary.get(statistic), (int, float)):
            return float(phase_summary[statistic])
    raw = capture.get("raw_timings_us")
    if isinstance(raw, dict) and isinstance(raw.get(phase), list) and raw[phase]:
        values = sorted(float(value) for value in raw[phase] if isinstance(value, (int, float)))
        if values:
            return values[len(values) // 2]
    return None


def _checksum_value(capture: dict[str, Any] | None) -> Any:
    if capture is None:
        return None
    if capture.get("checksum_u64") is not None:
        return capture.get("checksum_u64")
    return capture.get("checksum")


def _same_checksum(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    left_value = _checksum_value(left)
    right_value = _checksum_value(right)
    return left_value is not None and right_value is not None and left_value == right_value


def _target_id(capture: dict[str, Any]) -> str | None:
    for source in (capture.get("target_variant"), capture.get("device"), capture):
        if isinstance(source, dict):
            for key in ("target_id", "target_arch", "gcn_arch"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _selected_prefix(capture: dict[str, Any]) -> Any:
    return capture.get("selected_prefix", capture.get("prefix"))


def _requested_prefix(capture: dict[str, Any]) -> Any:
    return capture.get("requested_max_prefix", capture.get("prefix"))


def _prefix_policy(capture: dict[str, Any]) -> Any:
    return capture.get("contract_prefix_policy")


def _input_profile(capture: dict[str, Any]) -> Any:
    for key in ("input_distribution", "input_profile", "case_name"):
        value = capture.get(key)
        if value is not None:
            return value
    return None


def _resident_direct_capture(capture: dict[str, Any]) -> bool:
    return (
        capture.get("backend_selected") == "hip-direct"
        and capture.get("benchmark_execution_mode") == "persistent_resident_matrices"
        and capture.get("semantics") in {"bounded_i64", "bounded_u64"}
    )


def _comparison_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    output_policy = capture.get("output_policy") if isinstance(capture.get("output_policy"), dict) else {}
    return (
        capture.get("semantics"),
        capture.get("m"),
        capture.get("n"),
        capture.get("k"),
        _selected_prefix(capture),
        _requested_prefix(capture),
        _prefix_policy(capture),
        _input_profile(capture),
        capture.get("seed"),
        output_policy.get("output_domain") or capture.get("output_domain"),
    )


def _shape(capture: dict[str, Any] | None) -> dict[str, Any]:
    if not capture:
        return {}
    return {
        "m": capture.get("m"),
        "n": capture.get("n"),
        "k": capture.get("k"),
        "selected_prefix": _selected_prefix(capture),
        "requested_prefix": _requested_prefix(capture),
    }


def _speedup(baseline_us: float | None, candidate_us: float | None) -> float | None:
    if baseline_us is None or candidate_us in (None, 0.0):
        return None
    return baseline_us / candidate_us


def _resident_redesign_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture.get("resident_redesign")
    return metadata if isinstance(metadata, dict) else {}


def _redesign_dimensions(capture: dict[str, Any]) -> set[str]:
    metadata = _resident_redesign_metadata(capture)
    raw = metadata.get("dimensions")
    if isinstance(raw, list):
        return {str(item) for item in raw if isinstance(item, str) and item}
    return set()


def _resource_report(capture: dict[str, Any], counter_reports: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    report = counter_reports.get(path_key(str(capture.get("_path", ""))))
    if isinstance(report, dict):
        summary = report.get("resource_summary")
        if isinstance(summary, dict) and summary:
            return summary
    metadata = _resident_redesign_metadata(capture)
    summary = metadata.get("resource_summary")
    return summary if isinstance(summary, dict) and summary else None


def _candidate_label(capture: dict[str, Any]) -> str:
    metadata = _resident_redesign_metadata(capture)
    value = metadata.get("candidate")
    if isinstance(value, str) and value:
        return value
    kernel = _selected_kernel(capture)
    if kernel == REJECTED_COLPAIR_KERNEL:
        return "selected_prefix_colpair_rejected_route"
    tile_variant = capture.get("tile_shape_variant")
    if isinstance(tile_variant, dict) and tile_variant.get("name") not in {None, "default"}:
        return f"tile_shape:{tile_variant.get('name')}"
    return kernel or "unknown_candidate"


def _candidate_capture(capture: dict[str, Any]) -> bool:
    if not _resident_direct_capture(capture):
        return False
    if _selected_kernel(capture) != DEFAULT_KERNEL:
        return True
    metadata = _resident_redesign_metadata(capture)
    if metadata.get("candidate"):
        return True
    tile_variant = capture.get("tile_shape_variant")
    return isinstance(tile_variant, dict) and tile_variant.get("name") not in {None, "default"}


def _baseline_capture(capture: dict[str, Any]) -> bool:
    return _resident_direct_capture(capture) and _selected_kernel(capture) == DEFAULT_KERNEL and not _candidate_capture(capture)


def _counter_reports(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    reports = data.get("reports") if isinstance(data.get("reports"), list) else [data]
    rows: dict[str, dict[str, Any]] = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        capture = report.get("capture")
        if isinstance(capture, dict) and isinstance(capture.get("path"), str):
            rows[path_key(capture["path"])] = report
    return rows


def compare_candidate(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    counter_reports: dict[str, dict[str, Any]],
    margin: float,
) -> dict[str, Any]:
    candidate_median = _timing_value(candidate, "end_to_end")
    baseline_median = _timing_value(baseline, "end_to_end")
    candidate_gemm = _timing_value(candidate, "rns_gemm")
    baseline_gemm = _timing_value(baseline, "rns_gemm")
    candidate_export = _timing_value(candidate, "crt_export")
    baseline_export = _timing_value(baseline, "crt_export")
    speedup = _speedup(baseline_median, candidate_median)
    gemm_speedup = _speedup(baseline_gemm, candidate_gemm)
    export_speedup = _speedup(baseline_export, candidate_export)
    blockers: list[str] = []

    if baseline is None:
        blockers.append("missing_resident_default_baseline")
    if _schema_status(candidate) != "valid":
        blockers.append("candidate_schema_invalid")
    if baseline is not None and _schema_status(baseline) != "valid":
        blockers.append("baseline_schema_invalid")
    if not _release_capture_satisfied(candidate):
        blockers.append("candidate_not_release_review")
    if baseline is not None and not _release_capture_satisfied(baseline):
        blockers.append("baseline_not_release_review")
    if not _gpu_events_available(candidate):
        blockers.append("candidate_missing_required_gpu_events")
    if baseline is not None and not _gpu_events_available(baseline):
        blockers.append("baseline_missing_required_gpu_events")
    if baseline is not None and not _same_checksum(candidate, baseline):
        blockers.append("checksum_mismatch_with_resident_default")
    if candidate_median is None or baseline_median is None:
        blockers.append("missing_end_to_end_median")

    dimensions = _redesign_dimensions(candidate)
    missing_dimensions = sorted(REQUIRED_REDESIGN_DIMENSIONS - dimensions)
    for dimension in missing_dimensions:
        blockers.append(f"missing_redesign_dimension:{dimension}")
    resource_summary = _resource_report(candidate, counter_reports)
    if resource_summary is None:
        blockers.append("missing_resource_explanation")
    if gemm_speedup is not None and gemm_speedup > 1.0 and speedup is not None and speedup <= 1.0:
        blockers.append("gemm_only_win_end_to_end_loss")
    if speedup is not None and speedup <= 1.0:
        blockers.append("end_to_end_not_faster")
    elif speedup is not None and speedup < margin:
        blockers.append("does_not_clear_speedup_margin")

    evidence_complete = not any(blocker in EVIDENCE_BLOCKERS for blocker in blockers)
    if evidence_complete and not blockers:
        decision = "route_candidate"
    elif evidence_complete and speedup is not None and speedup < 1.0:
        decision = "drop/deprioritize"
    elif evidence_complete and "gemm_only_win_end_to_end_loss" in blockers:
        decision = "drop/deprioritize"
    else:
        decision = "keep_experimental"

    return {
        "candidate": _candidate_label(candidate),
        "candidate_capture": _capture_path(candidate),
        "baseline_capture": _capture_path(baseline),
        "semantics": candidate.get("semantics"),
        "target_id": _target_id(candidate),
        "shape": _shape(candidate),
        "candidate_kernel": _selected_kernel(candidate),
        "baseline_kernel": _selected_kernel(baseline),
        "candidate_schema_status": _schema_status(candidate),
        "baseline_schema_status": _schema_status(baseline) if baseline else None,
        "candidate_schema_errors": _schema_errors(candidate)[:5],
        "baseline_schema_errors": _schema_errors(baseline)[:5] if baseline else [],
        "candidate_median_end_to_end_us": candidate_median,
        "baseline_median_end_to_end_us": baseline_median,
        "speedup_vs_resident_default": speedup,
        "candidate_median_rns_gemm_us": candidate_gemm,
        "baseline_median_rns_gemm_us": baseline_gemm,
        "speedup_vs_resident_default_rns_gemm": gemm_speedup,
        "candidate_median_crt_export_us": candidate_export,
        "baseline_median_crt_export_us": baseline_export,
        "speedup_vs_resident_default_export": export_speedup,
        "release_review_satisfied": _release_capture_satisfied(candidate) and _release_capture_satisfied(baseline),
        "candidate_gpu_events_available": _gpu_events_available(candidate),
        "baseline_gpu_events_available": _gpu_events_available(baseline),
        "checksum_match": _same_checksum(candidate, baseline),
        "redesign_dimensions": sorted(dimensions),
        "missing_redesign_dimensions": missing_dimensions,
        "resource_explanation_available": resource_summary is not None,
        "resource_summary": resource_summary,
        "evidence_complete": evidence_complete,
        "decision": decision,
        "promotion_eligible": decision == "route_candidate",
        "blockers": sorted(set(blockers)),
    }


def build_direct_hip_resident_redesign_report(
    captures: list[dict[str, Any]],
    *,
    counter_reports: dict[str, dict[str, Any]] | None = None,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    counters = counter_reports or {}
    baselines: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    candidates: list[dict[str, Any]] = []
    ignored = 0
    for capture in captures:
        if _baseline_capture(capture):
            baselines.setdefault(_comparison_key(capture), []).append(capture)
        elif _candidate_capture(capture):
            candidates.append(capture)
        else:
            ignored += 1

    comparisons = []
    for candidate in sorted(candidates, key=lambda item: (_comparison_key(item), _candidate_label(item))):
        baseline_list = baselines.get(_comparison_key(candidate), [])
        baseline = sorted(baseline_list, key=lambda item: str(_capture_path(item) or ""))[0] if baseline_list else None
        comparisons.append(compare_candidate(candidate, baseline, counter_reports=counters, margin=margin))

    decisions = Counter(str(item["decision"]) for item in comparisons)
    summary = {
        "capture_count": len(captures),
        "candidate_count": len(candidates),
        "resident_default_baseline_count": sum(len(items) for items in baselines.values()),
        "ignored_capture_count": ignored,
        "comparison_count": len(comparisons),
        "route_candidate_count": decisions.get("route_candidate", 0),
        "deprioritized_count": decisions.get("drop/deprioritize", 0),
        "experimental_count": decisions.get("keep_experimental", 0),
        "rejected_colpair_count": sum(
            1 for item in comparisons if item.get("candidate_kernel") == REJECTED_COLPAIR_KERNEL
        ),
        "gemm_only_win_blocked_count": sum(
            1 for item in comparisons if "gemm_only_win_end_to_end_loss" in item.get("blockers", [])
        ),
        "evidence_complete_count": sum(1 for item in comparisons if item.get("evidence_complete") is True),
        "rank51_gate_complete": bool(comparisons) and all(item.get("evidence_complete") is True for item in comparisons),
        "decisions": dict(sorted(decisions.items())),
    }
    return {
        "schema_version": 1,
        "policy": "direct_hip_resident_redesign_requires_end_to_end_events_checksum_and_resource_evidence",
        "required_redesign_dimensions": sorted(REQUIRED_REDESIGN_DIMENSIONS),
        "required_speedup_margin": margin,
        "summary": summary,
        "comparisons": comparisons,
    }


def expand_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                item
                for item in sorted(path.rglob("*.json"))
                if item.name not in NON_CAPTURE_JSON_NAMES and not item.name.endswith("-report.json")
            )
        else:
            expanded.append(path)
    return expanded


def load_capture_with_schema_status(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    try:
        validate_capture(capture, path)
        capture["_schema_status"] = "valid"
        capture["_schema_errors"] = []
    except BenchmarkSchemaError as exc:
        capture["_schema_status"] = "invalid"
        capture["_schema_errors"] = str(exc).splitlines()
    capture["_path"] = str(path)
    return capture


def build_report(paths: list[Path], *, counter_report: Path | None = None, margin: float = DEFAULT_MARGIN) -> dict[str, Any]:
    captures = [load_capture_with_schema_status(path) for path in expand_inputs(paths)]
    return build_direct_hip_resident_redesign_report(
        captures, counter_reports=_counter_reports(counter_report), margin=margin
    )


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "none"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Direct-HIP Resident Redesign Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| candidate | semantics | shape | end-to-end speedup | GEMM speedup | decision | blockers |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in report["comparisons"]:
        shape = row.get("shape") if isinstance(row.get("shape"), dict) else {}
        blockers = ",".join(row.get("blockers") or []) or "none"
        lines.append(
            "| {candidate} | {semantics} | {m}x{n}x{k} prefix {prefix} | {speedup} | {gemm} | {decision} | {blockers} |".format(
                candidate=row.get("candidate"),
                semantics=row.get("semantics"),
                m=shape.get("m"),
                n=shape.get("n"),
                k=shape.get("k"),
                prefix=shape.get("selected_prefix"),
                speedup=fmt(row.get("speedup_vs_resident_default")),
                gemm=fmt(row.get("speedup_vs_resident_default_rns_gemm")),
                decision=row.get("decision"),
                blockers=blockers,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "direct-hip-resident-redesign-report.json"
    md_path = out_dir / "direct-hip-resident-redesign-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--counter-report", type=Path)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.captures, counter_report=args.counter_report, margin=args.margin)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, output in write_outputs(report, args.out_dir).items():
            print(f"{label}: {output}")
    if args.require_complete and not report["summary"].get("rank51_gate_complete"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
