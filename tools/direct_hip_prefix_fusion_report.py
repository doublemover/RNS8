#!/usr/bin/env python3
"""Review Direct-HIP prefix fusion using qualified independent process cohorts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from diagnostic_companion import load_bindings
from json_evidence import load_evidence_json
from observation_cohorts import phase_median
from report_capture_inputs import load_report_inventory, partition_report_inventory
from report_comparison import ComparisonPolicy, ReportCohorts, PHYSICAL_TILE_WORK_EXCLUSIONS

DEFAULT_OUT_DIR = Path("temp") / "direct-hip-prefix-fusion-reports"
PREFIX_POLICY = ComparisonPolicy(
    "matched_prefix_fusion_completed_output",
    work_exclusions=PHYSICAL_TILE_WORK_EXCLUSIONS,
    amortize_prepack_setup=True,
)

ONESHOT_V1_KERNEL = "direct_hip_prefix9_native_input_grouped_rns_gemm_v1"
ONESHOT_V2_COLPAIR_KERNEL = "direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2"
RESIDENT_COLPAIR_KERNEL = "direct_hip_active_prefix_colpair_grouped_rns_gemm_v2"
RESIDENT_DEFAULT_KERNEL = "direct_hip_tiled_active_prefix_rns_gemm_v2"
PREFIX20_EXPORT_VARIANT = "prefix20-fixed-export-candidate"


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
    return capture.get("selected_prefix", capture.get("prefix"))


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


def _one_shot_capture(capture: dict[str, Any]) -> bool:
    return _bounded_one_shot_capture(capture) and _selected_prefix(capture) == 9


def _resident_capture(capture: dict[str, Any]) -> bool:
    return (
        capture.get("backend_selected") == "hip-direct"
        and capture.get("benchmark_execution_mode") == "persistent_resident_matrices"
        and capture.get("semantics") in {"bounded_i64", "bounded_u64"}
    )


def _comparison_row(
    candidate: dict, baseline: dict | None, cohorts: ReportCohorts, *, oneshot: bool
) -> dict:
    comparison = cohorts.compare(candidate, baseline, PREFIX_POLICY)
    speedup, margin = comparison["speedup"], comparison["required_speedup_margin"]
    blockers = list(comparison["blockers"])
    if baseline is None:
        blockers.append(
            "missing_legacy_v1_one_shot_baseline"
            if oneshot
            else "missing_resident_default_baseline"
        )
    if blockers:
        decision = "keep_experimental"
    elif speedup > margin:
        decision = "candidate_one_shot_win" if oneshot else "candidate_resident_win"
    elif speedup < 1 / margin:
        decision = "deprioritize"
    else:
        decision = "keep_experimental"
        blockers.append("no_completed_output_win_above_observed_noise")
    return dict(
        kind="prefix9_public_one_shot_colpair" if oneshot else "resident_selected_prefix_colpair",
        semantics=candidate.get("semantics"),
        shape=_shape(candidate),
        candidate_capture=_capture_path(candidate),
        baseline_capture=_capture_path(baseline),
        candidate_kernel=_selected_kernel(candidate),
        baseline_kernel=_selected_kernel(baseline) if baseline else None,
        candidate_schema_status="valid",
        baseline_schema_status="valid" if baseline else None,
        candidate_median_end_to_end_us=phase_median(candidate),
        baseline_median_end_to_end_us=phase_median(baseline) if baseline else None,
        **{"speedup_vs_legacy_v1" if oneshot else "speedup_vs_resident_default": speedup},
        observation=cohorts.metadata(candidate, PREFIX_POLICY),
        comparison=comparison,
        resident_reference_comparison="different_execution_boundary_no_timing_ratio"
        if oneshot
        else None,
        decision=decision,
        blockers=sorted(set(blockers)),
        promotion_eligible=False,
    )


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
                    "reported_selector_promotion_eligible": row.get("selector_promotion_eligible"),
                    "selector_promotion_eligible": False,
                    "qualification_scope": "external_derived_report_inventory_only",
                    "promotion_blockers": row.get("promotion_blockers")
                    if isinstance(row.get("promotion_blockers"), list)
                    else [],
                    "decision": "keep_experimental",
                    "promotion_eligible": False,
                }
            )
    return rows


def build_direct_hip_prefix_fusion_report(
    captures: list[dict],
    export_selector_reports: list[dict] | None = None,
    *,
    diagnostic_bindings: dict | None = None,
) -> dict:
    valid, legacy = partition_report_inventory(captures)
    cohorts = ReportCohorts(valid, diagnostic_bindings)
    oneshot_v1, oneshot_v2, resident_defaults, resident_colpairs = [], [], [], []
    for capture in cohorts.observations:
        kernel = _selected_kernel(capture)
        if _one_shot_capture(capture) and kernel == ONESHOT_V1_KERNEL:
            oneshot_v1.append(capture)
        elif _one_shot_capture(capture) and kernel == ONESHOT_V2_COLPAIR_KERNEL:
            oneshot_v2.append(capture)
        elif _resident_capture(capture) and kernel == RESIDENT_COLPAIR_KERNEL:
            resident_colpairs.append(capture)
        elif _resident_capture(capture) and kernel == RESIDENT_DEFAULT_KERNEL:
            resident_defaults.append(capture)
    comparisons = [
        _comparison_row(
            candidate, cohorts.best(anchors, candidate, PREFIX_POLICY), cohorts, oneshot=oneshot
        )
        for candidates, anchors, oneshot in (
            (oneshot_v2, oneshot_v1, True),
            (resident_colpairs, resident_defaults, False),
        )
        for candidate in candidates
    ]
    prefix20_rows = [
        row
        for report in export_selector_reports or []
        for row in summarize_prefix20_export_selector(report)
    ]
    decisions = Counter(item["decision"] for item in comparisons)
    return dict(
        schema_version=2,
        policy="full_work_same_source_device_process_cohorts_production_companions_and_cpu_references",
        summary=dict(
            capture_count=len(captures),
            valid_current_schema_captures=len(valid),
            legacy_capture_count=len(legacy),
            observation_count=len(cohorts.observations),
            one_shot_colpair_comparisons=len(oneshot_v2),
            resident_colpair_comparisons=len(resident_colpairs),
            prefix20_export_selector_rows=len(prefix20_rows),
            candidate_one_shot_wins=decisions["candidate_one_shot_win"],
            candidate_resident_wins=decisions["candidate_resident_win"],
            deprioritized=decisions["deprioritize"],
            experimental=decisions["keep_experimental"] + len(prefix20_rows),
            promotion_eligible=False,
            decisions=dict(sorted(decisions.items())),
        ),
        observations=[cohorts.metadata(view, PREFIX_POLICY) for view in cohorts.observations],
        legacy_inventory=legacy,
        comparisons=comparisons,
        prefix20_export_selector_rows=prefix20_rows,
    )


def build_report(
    paths: list[Path],
    export_selector_report_paths: list[Path] | None = None,
    *,
    diagnostic_bindings: dict | None = None,
) -> dict:
    selector_reports = []
    for path in export_selector_report_paths or []:
        report = load_evidence_json(path)
        if not isinstance(report, dict):
            raise ValueError("export selector inventory must be a JSON object")
        selector_reports.append(report)
    return build_direct_hip_prefix_fusion_report(
        load_report_inventory(paths),
        selector_reports,
        diagnostic_bindings=diagnostic_bindings,
    )


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
    parser.add_argument(
        "captures", type=Path, nargs="*", help="schema-v4 benchmark JSON captures or directories"
    )
    parser.add_argument(
        "--capture", type=Path, action="append", help="additional capture file or directory"
    )
    parser.add_argument(
        "--export-selector-report",
        type=Path,
        action="append",
        help="export-selector-report.json to summarize prefix-20 selector rows",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="write JSON and Markdown reports here"
    )
    parser.add_argument("--diagnostic-binding", type=Path, action="append", default=[])
    parser.add_argument("--json", action="store_true", help="print report JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [*(args.capture or []), *args.captures]
    if not paths and not args.export_selector_report:
        raise SystemExit(
            "direct_hip_prefix_fusion_report requires at least one capture path or export selector report"
        )
    report = build_report(
        paths,
        args.export_selector_report,
        diagnostic_bindings=load_bindings(args.diagnostic_binding),
    )
    outputs = write_outputs(report, args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in outputs.items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
