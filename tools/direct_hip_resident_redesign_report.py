#!/usr/bin/env python3
"""Review Direct-HIP resident redesign using matched process and resource evidence."""

from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
from diagnostic_companion import load_bindings
from observation_cohorts import phase_median
from report_capture_inputs import load_report_inventory, partition_report_inventory
from report_comparison import ComparisonPolicy, ReportCohorts, PHYSICAL_TILE_WORK_EXCLUSIONS
from selected_kernel_resources import (
    SelectedStaticResources,
    load_static_counter_report,
    selected_resource_status,
)

DEFAULT_OUT_DIR = Path("temp") / "direct-hip-resident-redesign-reports"
DEFAULT_MARGIN = 1.03
DEFAULT_KERNEL = "direct_hip_tiled_active_prefix_rns_gemm_v2"
GROUPED_ACTIVE_SCHEDULE_KERNEL = "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3"
REJECTED_COLPAIR_KERNEL = "direct_hip_active_prefix_colpair_grouped_rns_gemm_v2"
REQUIRED_REDESIGN_DIMENSIONS = {
    "data_layout",
    "tile_shape",
    "export_interaction",
    "schedule_upload",
    "workspace_reuse",
}
RESIDENT_POLICY = ComparisonPolicy(
    "matched_direct_hip_resident_redesign_completed_output",
    work_exclusions=PHYSICAL_TILE_WORK_EXCLUSIONS,
    amortize_prepack_setup=True,
)


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


def _resident_direct_capture(capture: dict[str, Any]) -> bool:
    return (
        capture.get("backend_selected") == "hip-direct"
        and capture.get("benchmark_execution_mode") == "persistent_resident_matrices"
        and capture.get("semantics") in {"bounded_i64", "bounded_u64"}
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


def _resident_redesign_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    metadata = capture.get("resident_redesign")
    return metadata if isinstance(metadata, dict) else {}


def _redesign_dimensions(capture: dict[str, Any]) -> set[str]:
    metadata = _resident_redesign_metadata(capture)
    raw = metadata.get("dimensions")
    if isinstance(raw, list):
        return {str(item) for item in raw if isinstance(item, str) and item}
    return set()


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
    return (
        _resident_direct_capture(capture)
        and _selected_kernel(capture) == DEFAULT_KERNEL
        and not _candidate_capture(capture)
    )


def compare_candidate(
    candidate: dict,
    baseline: dict | None,
    *,
    cohorts: ReportCohorts,
    resources: SelectedStaticResources,
    margin: float,
) -> dict:
    comparison = cohorts.compare(candidate, baseline, RESIDENT_POLICY)
    speedup = comparison["speedup"]
    observed_margin = comparison["required_speedup_margin"]
    required_margin = max(margin, observed_margin) if observed_margin is not None else None
    raw_observations = cohorts.raw_observations(candidate)
    resource_observations = [
        dict(
            capture_path=raw.get("_path"),
            status=selected_resource_status(
                raw, resources, backend="hip-direct", target=_target_id(raw)
            ),
            declared_resource_inventory=_resident_redesign_metadata(raw).get("resource_summary"),
            redesign_dimensions=sorted(_redesign_dimensions(raw)),
        )
        for raw in raw_observations
    ]
    dimensions = (
        set.intersection(*(_redesign_dimensions(raw) for raw in raw_observations))
        if raw_observations
        else set()
    )
    missing_dimensions = sorted(REQUIRED_REDESIGN_DIMENSIONS - dimensions)
    resources_complete = bool(resource_observations) and all(
        row["status"]["complete_for_promotion"] for row in resource_observations
    )
    blockers = list(comparison["blockers"])
    if baseline is None:
        blockers.append("missing_resident_default_baseline")
    if not resources_complete:
        blockers.append("missing_selected_static_resource_explanation")
    blockers.extend("missing_redesign_dimension:" + name for name in missing_dimensions)
    evidence_complete = comparison["qualified"] and resources_complete and not missing_dimensions
    if evidence_complete and speedup > required_margin:
        decision = "route_candidate"
    elif comparison["qualified"] and speedup < 1 / required_margin:
        decision = "drop/deprioritize"
        blockers.append("end_to_end_not_faster_above_observed_noise")
    else:
        decision = "keep_experimental"
        if comparison["qualified"] and speedup <= required_margin:
            blockers.append("does_not_clear_speedup_margin")
    return dict(
        candidate=_candidate_label(candidate),
        candidate_capture=_capture_path(candidate),
        baseline_capture=_capture_path(baseline),
        semantics=candidate.get("semantics"),
        target_id=_target_id(candidate),
        shape=_shape(candidate),
        candidate_kernel=_selected_kernel(candidate),
        baseline_kernel=_selected_kernel(baseline),
        candidate_median_end_to_end_us=phase_median(candidate),
        baseline_median_end_to_end_us=phase_median(baseline) if baseline else None,
        speedup_vs_resident_default=speedup,
        required_speedup_margin=required_margin,
        phase_observations=dict(
            candidate={
                phase: phase_median(candidate, phase)
                for phase in ("pack", "rns_gemm", "crt_export")
            },
            baseline={
                phase: phase_median(baseline, phase) for phase in ("pack", "rns_gemm", "crt_export")
            }
            if baseline
            else None,
            scope="descriptive_host_component_medians_no_independent_phase_speedup",
        ),
        redesign_dimensions=sorted(dimensions),
        redesign_dimensions_scope="declared_research_checklist_not_measured_change",
        missing_redesign_dimensions=missing_dimensions,
        resource_observations=resource_observations,
        resource_explanation_available=resources_complete,
        observation=cohorts.metadata(candidate, RESIDENT_POLICY),
        comparison=comparison,
        evidence_complete=evidence_complete,
        decision=decision,
        promotion_eligible=decision == "route_candidate",
        blockers=sorted(set(blockers)),
    )


def build_direct_hip_resident_redesign_report(
    captures: list[dict],
    *,
    resources: SelectedStaticResources | None = None,
    diagnostic_bindings: dict | None = None,
    margin: float = DEFAULT_MARGIN,
) -> dict:
    if type(margin) not in (int, float) or not math.isfinite(margin) or margin < DEFAULT_MARGIN:
        raise ValueError("resident redesign margin must be finite and at least 1.03")
    resources = resources or SelectedStaticResources()
    valid, legacy = partition_report_inventory(captures)
    cohorts = ReportCohorts(valid, diagnostic_bindings)
    baselines = [view for view in cohorts.observations if _baseline_capture(view)]
    candidates = [view for view in cohorts.observations if _candidate_capture(view)]
    comparisons = [
        compare_candidate(
            candidate,
            cohorts.best(baselines, candidate, RESIDENT_POLICY),
            cohorts=cohorts,
            resources=resources,
            margin=margin,
        )
        for candidate in candidates
    ]
    resources.store.finish()
    decisions = Counter(row["decision"] for row in comparisons)
    return dict(
        schema_version=2,
        policy="matched_work_source_device_independent_processes_full_references_companions_selected_static_resources",
        required_redesign_dimensions=sorted(REQUIRED_REDESIGN_DIMENSIONS),
        required_speedup_margin=margin,
        summary=dict(
            capture_count=len(captures),
            valid_capture_count=len(valid),
            legacy_capture_count=len(legacy),
            observation_count=len(cohorts.observations),
            candidate_count=len(candidates),
            resident_default_baseline_count=len(baselines),
            ignored_capture_count=len(cohorts.observations) - len(candidates) - len(baselines),
            comparison_count=len(comparisons),
            route_candidate_count=decisions["route_candidate"],
            deprioritized_count=decisions["drop/deprioritize"],
            experimental_count=decisions["keep_experimental"],
            rejected_colpair_count=sum(
                row["candidate_kernel"] == REJECTED_COLPAIR_KERNEL for row in comparisons
            ),
            evidence_complete_count=sum(row["evidence_complete"] for row in comparisons),
            rank51_gate_complete=bool(comparisons)
            and all(row["evidence_complete"] for row in comparisons),
            decisions=dict(sorted(decisions.items())),
        ),
        comparisons=comparisons,
        observations=[cohorts.metadata(view, RESIDENT_POLICY) for view in cohorts.observations],
        legacy_inventory=legacy,
        resource_inventory=resources.inventory,
    )


def build_report(
    paths: list[Path],
    *,
    counter_report: Path | None = None,
    diagnostic_bindings: dict | None = None,
    margin: float = DEFAULT_MARGIN,
) -> dict:
    return build_direct_hip_resident_redesign_report(
        load_report_inventory(paths),
        resources=load_static_counter_report(counter_report),
        diagnostic_bindings=diagnostic_bindings,
        margin=margin,
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
            "| candidate | semantics | shape | end-to-end speedup | decision | blockers |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in report["comparisons"]:
        shape = row.get("shape") if isinstance(row.get("shape"), dict) else {}
        blockers = ",".join(row.get("blockers") or []) or "none"
        lines.append(
            "| {candidate} | {semantics} | {m}x{n}x{k} prefix {prefix} | {speedup} | {decision} | {blockers} |".format(
                candidate=row.get("candidate"),
                semantics=row.get("semantics"),
                m=shape.get("m"),
                n=shape.get("n"),
                k=shape.get("k"),
                prefix=shape.get("selected_prefix"),
                speedup=fmt(row.get("speedup_vs_resident_default")),
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
    parser.add_argument("--diagnostic-binding", type=Path, action="append", default=[])
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.captures,
        counter_report=args.counter_report,
        diagnostic_bindings=load_bindings(args.diagnostic_binding),
        margin=args.margin,
    )
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
