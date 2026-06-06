#!/usr/bin/env python3
"""Summarize Direct-HIP prefix-9/prefix-20 fusion evidence for queue rank 10."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "direct-hip-prefix-fusion-reports"
NON_CAPTURE_JSON_NAMES = {
    "direct-hip-prefix-fusion-report.json",
    "export-selector-report.json",
    "review_report.json",
    "scenario_manifest.json",
    "validation-summary.json",
}

ONESHOT_V1_KERNEL = "direct_hip_prefix9_native_input_grouped_rns_gemm_v1"
ONESHOT_V2_COLPAIR_KERNEL = "direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2"
RESIDENT_COLPAIR_KERNEL = "direct_hip_active_prefix_colpair_grouped_rns_gemm_v2"
RESIDENT_DEFAULT_KERNEL = "direct_hip_tiled_active_prefix_rns_gemm_v2"
PREFIX20_EXPORT_VARIANT = "prefix20-fixed-export-candidate"


def _timing_summary_value(capture: dict[str, Any] | None, phase: str, statistic: str) -> float | None:
    if capture is None:
        return None
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    phase_summary = summary.get(phase)
    if not isinstance(phase_summary, dict):
        return None
    value = phase_summary.get(statistic)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _schema_status(capture: dict[str, Any]) -> str:
    status = capture.get("_schema_status")
    return status if isinstance(status, str) else "unknown"


def _schema_errors(capture: dict[str, Any]) -> list[str]:
    errors = capture.get("_schema_errors")
    return [str(item) for item in errors] if isinstance(errors, list) else []


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
    return bool(
        isinstance(metadata, dict)
        and metadata.get("gpu_event_timing") is True
        and metadata.get("gpu_event_timing_status") == "available"
        and isinstance(metadata.get("gpu_event_phase_order"), list)
        and metadata.get("gpu_event_phase_order")
    )


def _checksum_value(capture: dict[str, Any] | None) -> Any:
    if capture is None:
        return None
    if capture.get("checksum_u64") is not None:
        return capture.get("checksum_u64")
    return capture.get("checksum")


def _selected_kernel(capture: dict[str, Any]) -> str:
    value = capture.get("selected_kernel")
    if isinstance(value, str) and value:
        return value
    metadata = capture.get("backend_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("selected_kernel"), str):
        return str(metadata["selected_kernel"])
    return ""


def _bounded_one_shot_capture(capture: dict[str, Any]) -> bool:
    return (
        capture.get("backend_selected") == "hip-direct"
        and capture.get("benchmark_execution_mode") == "public_oneshot_transient_native_inputs"
        and capture.get("semantics") in {"bounded_i64", "bounded_u64"}
    )


def _selected_prefix(capture: dict[str, Any]) -> Any:
    value = capture.get("selected_prefix", capture.get("prefix"))
    if value is None and _bounded_one_shot_capture(capture) and _selected_kernel(capture) == ONESHOT_V1_KERNEL:
        return 9
    return value


def _requested_prefix(capture: dict[str, Any]) -> Any:
    value = capture.get("requested_max_prefix", capture.get("prefix"))
    if value is None and _bounded_one_shot_capture(capture) and _selected_kernel(capture) == ONESHOT_V1_KERNEL:
        return 9
    return value


def _prefix_policy(capture: dict[str, Any]) -> Any:
    value = capture.get("contract_prefix_policy")
    if value is None and _bounded_one_shot_capture(capture) and _selected_kernel(capture) == ONESHOT_V1_KERNEL:
        return "fixed_requested"
    return value


def _capture_path(capture: dict[str, Any] | None) -> str | None:
    if capture is None:
        return None
    value = capture.get("_path")
    return str(value) if value is not None else None


def _shape(capture: dict[str, Any]) -> dict[str, Any]:
    return {
        "m": capture.get("m"),
        "n": capture.get("n"),
        "k": capture.get("k"),
        "selected_prefix": _selected_prefix(capture),
    }


def _comparison_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    return (
        capture.get("semantics"),
        capture.get("m"),
        capture.get("n"),
        capture.get("k"),
        _selected_prefix(capture),
        _requested_prefix(capture),
        _prefix_policy(capture),
        capture.get("benchmark_execution_mode"),
        capture.get("input_distribution"),
        capture.get("seed"),
    )


def _resident_key(capture: dict[str, Any]) -> tuple[Any, ...]:
    return (
        capture.get("semantics"),
        capture.get("m"),
        capture.get("n"),
        capture.get("k"),
        capture.get("input_distribution"),
        capture.get("seed"),
    )


def _speedup(baseline_us: float | None, candidate_us: float | None) -> float | None:
    if baseline_us is None or candidate_us in (None, 0.0):
        return None
    return baseline_us / candidate_us


def _same_checksum(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    left = _checksum_value(a)
    right = _checksum_value(b)
    return left is not None and right is not None and left == right


def _shape_label(capture: dict[str, Any]) -> str:
    return f"{capture.get('m')}x{capture.get('n')}x{capture.get('k')}"


def _path_contains(path: str | None, token: str) -> bool:
    return bool(path and token.lower() in path.lower())


def _one_shot_baseline_sort_key(capture: dict[str, Any]) -> tuple[int, str]:
    path = _capture_path(capture)
    if _path_contains(path, "before"):
        return (0, path or "")
    return (1, path or "")


def _resident_baseline_sort_key(capture: dict[str, Any]) -> tuple[int, str]:
    path = _capture_path(capture)
    if _path_contains(path, "before-rerun"):
        return (0, path or "")
    if _path_contains(path, "before"):
        return (1, path or "")
    return (2, path or "")


def _one_shot_capture(capture: dict[str, Any]) -> bool:
    return _bounded_one_shot_capture(capture) and _selected_prefix(capture) == 9


def _resident_capture(capture: dict[str, Any]) -> bool:
    return (
        capture.get("backend_selected") == "hip-direct"
        and capture.get("benchmark_execution_mode") == "persistent_resident_matrices"
        and capture.get("semantics") in {"bounded_i64", "bounded_u64"}
    )


def compare_one_shot(candidate: dict[str, Any], baseline: dict[str, Any] | None, resident: dict[str, Any] | None) -> dict[str, Any]:
    candidate_median = _timing_summary_value(candidate, "end_to_end", "median")
    baseline_median = _timing_summary_value(baseline, "end_to_end", "median")
    resident_median = _timing_summary_value(resident, "end_to_end", "median")
    blockers: list[str] = []
    if baseline is None:
        blockers.append("missing_legacy_v1_one_shot_baseline")
    elif not _release_capture_satisfied(baseline):
        blockers.append("baseline_not_release_review")
    if not _release_capture_satisfied(candidate):
        blockers.append("candidate_not_release_review")
    if _schema_status(candidate) != "valid":
        blockers.append("candidate_schema_invalid")
    if baseline is not None and _schema_status(baseline) != "valid":
        blockers.append("baseline_legacy_schema")
    if not _gpu_events_available(candidate):
        blockers.append("candidate_missing_required_gpu_events")
    if baseline is not None and not _same_checksum(candidate, baseline):
        blockers.append("checksum_mismatch_with_v1_baseline")
    if candidate_median is None or baseline_median is None:
        blockers.append("missing_end_to_end_median")

    speedup = _speedup(baseline_median, candidate_median)
    if (
        "candidate_schema_invalid" in blockers
        or "candidate_missing_required_gpu_events" in blockers
        or "checksum_mismatch_with_v1_baseline" in blockers
    ):
        decision = "keep_experimental"
    elif speedup is not None and speedup > 1.0 and "checksum_mismatch_with_v1_baseline" not in blockers:
        decision = "candidate_one_shot_win"
    elif speedup is None:
        decision = "keep_experimental"
    else:
        decision = "deprioritize"

    return {
        "kind": "prefix9_public_one_shot_colpair",
        "semantics": candidate.get("semantics"),
        "shape": _shape(candidate),
        "candidate_capture": _capture_path(candidate),
        "baseline_capture": _capture_path(baseline),
        "resident_reference_capture": _capture_path(resident),
        "candidate_kernel": _selected_kernel(candidate),
        "baseline_kernel": _selected_kernel(baseline) if baseline else None,
        "resident_reference_kernel": _selected_kernel(resident) if resident else None,
        "candidate_schema_status": _schema_status(candidate),
        "baseline_schema_status": _schema_status(baseline) if baseline else None,
        "baseline_schema_errors": _schema_errors(baseline)[:5] if baseline else [],
        "candidate_median_end_to_end_us": candidate_median,
        "baseline_median_end_to_end_us": baseline_median,
        "resident_reference_median_end_to_end_us": resident_median,
        "speedup_vs_legacy_v1": speedup,
        "resident_reference_speedup_vs_candidate": _speedup(candidate_median, resident_median),
        "resident_reference_faster": bool(resident_median is not None and candidate_median is not None and resident_median < candidate_median),
        "release_review_satisfied": _release_capture_satisfied(candidate) and _release_capture_satisfied(baseline),
        "candidate_gpu_events_available": _gpu_events_available(candidate),
        "checksum_match": _same_checksum(candidate, baseline),
        "decision": decision,
        "blockers": blockers,
        "promotion_eligible": False,
    }


def compare_resident_colpair(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    candidate_median = _timing_summary_value(candidate, "end_to_end", "median")
    baseline_median = _timing_summary_value(baseline, "end_to_end", "median")
    speedup = _speedup(baseline_median, candidate_median)
    blockers: list[str] = []
    if baseline is None:
        blockers.append("missing_resident_default_baseline")
    if not _release_capture_satisfied(candidate):
        blockers.append("candidate_not_release_review")
    if baseline is not None and not _release_capture_satisfied(baseline):
        blockers.append("baseline_not_release_review")
    if _schema_status(candidate) != "valid":
        blockers.append("candidate_schema_invalid")
    if baseline is not None and _schema_status(baseline) != "valid":
        blockers.append("baseline_schema_invalid")
    if not _gpu_events_available(candidate):
        blockers.append("candidate_missing_required_gpu_events")
    if baseline is not None and not _same_checksum(candidate, baseline):
        blockers.append("checksum_mismatch_with_default_baseline")
    if candidate_median is None or baseline_median is None:
        blockers.append("missing_end_to_end_median")

    if blockers:
        decision = "keep_experimental"
    elif speedup is not None and speedup > 1.0:
        decision = "candidate_resident_win"
    else:
        decision = "deprioritize"

    return {
        "kind": "prefix9_resident_selected_prefix_colpair",
        "semantics": candidate.get("semantics"),
        "shape": _shape(candidate),
        "candidate_capture": _capture_path(candidate),
        "baseline_capture": _capture_path(baseline),
        "candidate_kernel": _selected_kernel(candidate),
        "baseline_kernel": _selected_kernel(baseline) if baseline else None,
        "candidate_schema_status": _schema_status(candidate),
        "baseline_schema_status": _schema_status(baseline) if baseline else None,
        "candidate_median_end_to_end_us": candidate_median,
        "baseline_median_end_to_end_us": baseline_median,
        "speedup_vs_resident_default": speedup,
        "release_review_satisfied": _release_capture_satisfied(candidate) and _release_capture_satisfied(baseline),
        "candidate_gpu_events_available": _gpu_events_available(candidate),
        "checksum_match": _same_checksum(candidate, baseline),
        "decision": decision,
        "blockers": blockers,
        "promotion_eligible": False,
    }


def _export_selector_groups(report: dict[str, Any]) -> list[dict[str, Any]]:
    groups = report.get("groups")
    return groups if isinstance(groups, list) else []


def summarize_prefix20_export_selector(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in _export_selector_groups(report):
        group_rows = group.get("rows")
        if not isinstance(group_rows, list):
            continue
        for row in group_rows:
            if not isinstance(row, dict) or row.get("export_variant") != PREFIX20_EXPORT_VARIANT:
                continue
            rows.append(
                {
                    "kind": "prefix20_fixed_export_selector",
                    "semantics": row.get("semantics"),
                    "shape": row.get("shape"),
                    "backend": row.get("backend"),
                    "target_id": row.get("target_id"),
                    "capture": row.get("capture_path"),
                    "selected_kernel": row.get("selected_kernel"),
                    "export_variant": row.get("export_variant"),
                    "reconstruction_variant": row.get("reconstruction_variant"),
                    "median_end_to_end_us": row.get("median_end_to_end_us"),
                    "median_export_us": row.get("median_export_us"),
                    "selector_promotion_eligible": row.get("selector_promotion_eligible"),
                    "promotion_blockers": row.get("promotion_blockers") if isinstance(row.get("promotion_blockers"), list) else [],
                    "decision": "keep_experimental",
                    "promotion_eligible": False,
                }
            )
    return rows


def build_direct_hip_prefix_fusion_report(
    captures: list[dict[str, Any]], export_selector_reports: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    oneshot_v1: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    oneshot_v2: list[dict[str, Any]] = []
    resident_references: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    resident_defaults: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    resident_colpairs: list[dict[str, Any]] = []

    for capture in captures:
        kernel = _selected_kernel(capture)
        if _one_shot_capture(capture) and kernel == ONESHOT_V1_KERNEL:
            oneshot_v1.setdefault(_comparison_key(capture), []).append(capture)
        elif _one_shot_capture(capture) and kernel == ONESHOT_V2_COLPAIR_KERNEL:
            oneshot_v2.append(capture)
        elif _resident_capture(capture) and kernel == RESIDENT_COLPAIR_KERNEL:
            resident_colpairs.append(capture)
        elif _resident_capture(capture) and kernel == RESIDENT_DEFAULT_KERNEL:
            resident_defaults.setdefault(_resident_key(capture), []).append(capture)
        elif _resident_capture(capture):
            resident_references.setdefault(_resident_key(capture), []).append(capture)

    comparisons: list[dict[str, Any]] = []
    for candidate in sorted(oneshot_v2, key=lambda item: (_shape_label(item), str(item.get("semantics")))):
        baselines = sorted(oneshot_v1.get(_comparison_key(candidate), []), key=_one_shot_baseline_sort_key)
        resident = None
        references = resident_references.get(_resident_key(candidate), []) + resident_defaults.get(_resident_key(candidate), [])
        if references:
            resident = sorted(references, key=_resident_baseline_sort_key)[0]
        comparisons.append(compare_one_shot(candidate, baselines[0] if baselines else None, resident))

    for candidate in sorted(resident_colpairs, key=lambda item: (_shape_label(item), str(item.get("semantics")))):
        baselines = sorted(resident_defaults.get(_resident_key(candidate), []), key=_resident_baseline_sort_key)
        comparisons.append(compare_resident_colpair(candidate, baselines[0] if baselines else None))

    prefix20_rows: list[dict[str, Any]] = []
    for report in export_selector_reports or []:
        prefix20_rows.extend(summarize_prefix20_export_selector(report))

    decision_counts = Counter(str(item["decision"]) for item in comparisons)
    prefix20_decisions = Counter(str(item["decision"]) for item in prefix20_rows)
    summary = {
        "capture_count": len(captures),
        "one_shot_colpair_comparisons": sum(1 for item in comparisons if item["kind"] == "prefix9_public_one_shot_colpair"),
        "resident_colpair_comparisons": sum(
            1 for item in comparisons if item["kind"] == "prefix9_resident_selected_prefix_colpair"
        ),
        "prefix20_export_selector_rows": len(prefix20_rows),
        "candidate_one_shot_wins": decision_counts.get("candidate_one_shot_win", 0),
        "candidate_resident_wins": decision_counts.get("candidate_resident_win", 0),
        "deprioritized": decision_counts.get("deprioritize", 0),
        "experimental": decision_counts.get("keep_experimental", 0) + prefix20_decisions.get("keep_experimental", 0),
        "legacy_before_captures": sum(1 for item in captures if _schema_status(item) == "legacy_schema"),
        "valid_current_schema_captures": sum(1 for item in captures if _schema_status(item) == "valid"),
        "promotion_eligible": False,
        "decisions": dict(sorted((decision_counts + prefix20_decisions).items())),
    }
    return {
        "schema_version": 1,
        "policy": "direct_hip_prefix_fusion_rank10_evidence_only",
        "summary": summary,
        "comparisons": comparisons,
        "prefix20_export_selector_rows": prefix20_rows,
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
        capture["_schema_status"] = "legacy_schema"
        capture["_schema_errors"] = str(exc).splitlines()
    capture["_path"] = str(path)
    return capture


def load_json_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def build_report(paths: list[Path], export_selector_report_paths: list[Path] | None = None) -> dict[str, Any]:
    captures = [load_capture_with_schema_status(path) for path in expand_inputs(paths)]
    selector_reports = [load_json_file(path) for path in export_selector_report_paths or []]
    return build_direct_hip_prefix_fusion_report(captures, selector_reports)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "none"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Direct-HIP Prefix Fusion Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Prefix-9 Comparisons",
            "",
            "| kind | semantics | shape | candidate us | baseline us | speedup | decision | blockers |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for item in report["comparisons"]:
        shape = item.get("shape", {})
        blockers = ",".join(item.get("blockers") or []) or "none"
        speedup = item.get("speedup_vs_legacy_v1", item.get("speedup_vs_resident_default"))
        lines.append(
            "| {kind} | {semantics} | {m}x{n}x{k} prefix {prefix} | {candidate} | {baseline} | {speedup} | {decision} | {blockers} |".format(
                kind=item.get("kind"),
                semantics=item.get("semantics"),
                m=shape.get("m"),
                n=shape.get("n"),
                k=shape.get("k"),
                prefix=shape.get("selected_prefix"),
                candidate=fmt(item.get("candidate_median_end_to_end_us")),
                baseline=fmt(item.get("baseline_median_end_to_end_us")),
                speedup=fmt(speedup),
                decision=item.get("decision"),
                blockers=blockers,
            )
        )
    lines.extend(
        [
            "",
            "## Prefix-20 Export Selector Rows",
            "",
            "| backend | semantics | shape | variant | median us | export us | decision | blockers |",
            "|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for item in report["prefix20_export_selector_rows"]:
        shape = item.get("shape") if isinstance(item.get("shape"), dict) else {}
        blockers = ",".join(item.get("promotion_blockers") or []) or "none"
        lines.append(
            "| {backend} | {semantics} | {m}x{n}x{k} | {variant} | {median} | {export} | {decision} | {blockers} |".format(
                backend=item.get("backend"),
                semantics=item.get("semantics"),
                m=shape.get("m"),
                n=shape.get("n"),
                k=shape.get("k"),
                variant=item.get("export_variant"),
                median=fmt(item.get("median_end_to_end_us")),
                export=fmt(item.get("median_export_us")),
                decision=item.get("decision"),
                blockers=blockers,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "direct-hip-prefix-fusion-report.json"
    md_path = out_dir / "direct-hip-prefix-fusion-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    return {"json": str(json_path), "markdown": str(md_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="*", help="schema-v4 benchmark JSON captures or directories")
    parser.add_argument("--capture", type=Path, action="append", help="additional capture file or directory")
    parser.add_argument(
        "--export-selector-report",
        type=Path,
        action="append",
        help="export-selector-report.json to summarize prefix-20 selector rows",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="write JSON and Markdown reports here")
    parser.add_argument("--json", action="store_true", help="print report JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [*(args.capture or []), *args.captures]
    if not paths and not args.export_selector_report:
        raise SystemExit("direct_hip_prefix_fusion_report requires at least one capture path or export selector report")
    report = build_report(paths, args.export_selector_report)
    outputs = write_outputs(report, args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in outputs.items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
