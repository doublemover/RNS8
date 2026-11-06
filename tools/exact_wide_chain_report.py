#!/usr/bin/env python3
"""Pair exact-wide residue-current chain captures with final-output controls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from diagnostic_companion import load_bindings
from observation_cohorts import phase_median
from report_capture_inputs import load_report_captures
from report_comparison import ComparisonPolicy, ReportCohorts

# This is a boundary-cost diagnostic, never a same-output host speedup.
BOUNDARY_POLICY = ComparisonPolicy(
    "same_exact_chain_resident_vs_final_export_boundary_diagnostic_v1",
    work_exclusions=(
        "requested_next_op.resolved",
        "residue_output_mode",
        "exact_output_contract.requested_final_output",
        "export_variant.final_output_mode",
        "export_variant.selector_key",
        "export_variant.d2h_policy",
        "export_variant.selector_status_policy",
        "output_policy.status_handling",
        "exact_output_contract.status_policy",
    ),
    require_cpu_anchor=False,
)
CPU_POLICY = ComparisonPolicy(
    "same_completed_exact_chain_same_host_build_cpu_vs_gpu_v1",
    work_exclusions=("export_variant.selector_key",),
    require_cpu_anchor=False,
    cpu_gpu_comparison=True,
)


DEFAULT_OUT_DIR = Path("temp") / "exact-wide-chain-reports"
REFERENCE_BACKENDS = {"cpu-reference"}
EXACT_WIDE_SEMANTICS = {"exact_wide_signed", "exact_wide_unsigned"}


def backend_id(capture: dict[str, Any] | None) -> str:
    if capture is None:
        return ""
    return str(capture.get("backend_selected") or capture.get("backend_requested") or "")


def execution_mode(capture: dict[str, Any]) -> str:
    value = capture.get("benchmark_execution_mode")
    if isinstance(value, str):
        return value
    metadata = capture.get("timing_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("benchmark_execution_mode"), str):
        return metadata["benchmark_execution_mode"]
    return ""


def is_exact_wide_chain(capture: dict[str, Any]) -> bool:
    return (
        capture.get("semantics") in EXACT_WIDE_SEMANTICS
        and type(capture.get("residue_chain_length")) is int
        and capture["residue_chain_length"] > 1
    )


def chain_output_mode(capture: dict[str, Any]) -> str | None:
    mode = execution_mode(capture)
    if (
        mode == "residue_current_rns_chain"
        and capture.get("residue_output_mode") == "residue_current_rns"
    ):
        return "residue_current"
    if (
        mode == "residue_chain_final_host_export"
        and capture.get("residue_chain_final_export") is True
    ):
        return "final_output"
    return None


def capture_summary(
    capture: dict | None, cohorts: ReportCohorts, policy: ComparisonPolicy
) -> dict | None:
    if capture is None:
        return None
    return dict(
        path=capture.get("_path"),
        backend=backend_id(capture),
        mode=chain_output_mode(capture),
        median_end_to_end_us=phase_median(capture),
        **cohorts.metadata(capture, policy),
    )


def residue_lifetime_metadata(capture: dict[str, Any] | None) -> dict[str, Any]:
    if capture is None:
        return {"available": False, "reason": "missing_residue_current_capture"}
    plan_packing = capture.get("plan_packing")
    plan_lowering = capture.get("plan_lowering")
    output_policy = capture.get("output_policy")
    requested_next_op = capture.get("requested_next_op")
    checks = {
        "device_current_output": isinstance(plan_packing, dict)
        and plan_packing.get("output_device_current") is True,
        "host_not_current": isinstance(plan_packing, dict)
        and plan_packing.get("output_host_current") is False,
        "rns_continuation_available": isinstance(plan_lowering, dict)
        and plan_lowering.get("rns_continuation_available") is True,
        "per_repeat_export_deferred": isinstance(output_policy, dict)
        and output_policy.get("per_repeat_logical_export") is False,
        "final_checksum_export_after_repeats": isinstance(output_policy, dict)
        and output_policy.get("final_checksum_export_after_repeats") is True,
        "next_op_rns_gemm": isinstance(requested_next_op, dict)
        and requested_next_op.get("resolved") == "rns-gemm",
    }
    missing = [key for key, value in checks.items() if not value]
    return {
        "available": not missing,
        "checks": checks,
        "reason": "available" if not missing else "incomplete_residue_current_lifetime_metadata",
        "missing": missing,
    }


def row_for_backend(key: str, captures: list[dict], backend: str, cohorts: ReportCohorts) -> dict:
    gpu = [capture for capture in captures if backend_id(capture) == backend]
    current = [capture for capture in gpu if chain_output_mode(capture) == "residue_current"]
    finals = [capture for capture in gpu if chain_output_mode(capture) == "final_output"]
    anchor = next(iter(current or finals))
    residue = cohorts.best(current, anchor, BOUNDARY_POLICY) or next(iter(current), None)
    final = cohorts.best(finals, anchor, BOUNDARY_POLICY) or next(iter(finals), None)
    cpu_finals = [
        capture
        for capture in captures
        if backend_id(capture) in REFERENCE_BACKENDS
        and chain_output_mode(capture) == "final_output"
    ]
    cpu = cohorts.best(cpu_finals, final or anchor, CPU_POLICY)
    # Preserve rejected observations without selecting a fastest raw file.
    cpu = cpu or next(iter(cpu_finals), None)
    boundary = cohorts.compare(residue or anchor, final, BOUNDARY_POLICY)
    completed = cohorts.compare(final or anchor, cpu, CPU_POLICY)
    lifetime = residue_lifetime_metadata(residue)
    blockers = list(boundary["blockers"]) + ["cpu:" + item for item in completed["blockers"]]
    if residue is None:
        blockers.append("missing_residue_current_capture")
    if final is None:
        blockers.append("missing_same_backend_final_output_capture")
    if cpu is None:
        blockers.append("missing_cpu_final_output_baseline")
    if not lifetime["available"]:
        blockers.append(lifetime["reason"])
    blockers = sorted(set(blockers))
    return dict(
        contract_key=key,
        backend=backend,
        semantics=anchor.get("semantics"),
        shape={field: anchor.get(field) for field in ("m", "n", "k")},
        chain_length=anchor.get("residue_chain_length"),
        limb_count=anchor.get("exact_wide_limb_count"),
        residue_current=capture_summary(residue, cohorts, BOUNDARY_POLICY),
        final_output=capture_summary(final, cohorts, BOUNDARY_POLICY),
        cpu_final_output=capture_summary(cpu, cohorts, CPU_POLICY),
        final_output_vs_residue_current_ratio=boundary["speedup"] if residue is not None else None,
        boundary_ratio_kind="different_completion_boundaries_not_host_output_speedup",
        cpu_vs_final_output_speedup=completed["speedup"] if final is not None else None,
        boundary_comparison=boundary,
        cpu_comparison=completed,
        residue_lifetime=lifetime,
        observations=[cohorts.metadata(view, BOUNDARY_POLICY) for view in gpu],
        cpu_observations=[cohorts.metadata(view, CPU_POLICY) for view in cpu_finals],
        decision="keep_experimental" if blockers else "paired_residue_current_chain_ready",
        blockers=blockers,
        promotion_eligible=False,
    )


def build_report_from_captures(
    captures: list[dict[str, Any]], *, diagnostic_bindings: dict | None = None
) -> dict[str, Any]:
    cohorts = ReportCohorts(captures, diagnostic_bindings)
    selected = [
        view
        for view in cohorts.observations
        if is_exact_wide_chain(view) and chain_output_mode(view) is not None
    ]
    by_contract = defaultdict(list)
    for view in selected:
        by_contract[(BOUNDARY_POLICY.work(view), CPU_POLICY.environment(view))].append(view)
    groups = []
    for (work, environment), items in sorted(by_contract.items()):
        lanes = sorted(
            {
                (backend_id(view), BOUNDARY_POLICY.environment(view))
                for view in items
                if backend_id(view) not in REFERENCE_BACKENDS
            }
        )
        rows = []
        for backend, device_environment in lanes:
            lane = [
                view
                for view in items
                if backend_id(view) in REFERENCE_BACKENDS
                or (
                    backend_id(view) == backend
                    and BOUNDARY_POLICY.environment(view) == device_environment
                )
            ]
            rows.append(row_for_backend(work, lane, backend, cohorts))
        groups.append(
            dict(
                contract_key=work,
                comparison_environment=environment,
                capture_count=sum(
                    len(view["_observation_aggregate"]["submitted_observations"]) for view in items
                ),
                rows=rows,
            )
        )
    rows = [row for group in groups for row in group["rows"]]
    return dict(
        schema_version=2,
        policy="exact_wide_chain_independent_production_cohorts_and_complete_cpu_oracles_v2",
        boundary_ratio_scope="resident_vs_final_export_diagnostic_not_same_output_speedup",
        capture_count=len(captures),
        exact_wide_chain_capture_count=sum(is_exact_wide_chain(view) for view in captures),
        summary=dict(
            groups=len(groups),
            paired_rows=len(rows),
            ready_pairs=sum(
                row["decision"] == "paired_residue_current_chain_ready" for row in rows
            ),
            experimental=sum(row["decision"] == "keep_experimental" for row in rows),
            missing_residue_current=sum(
                "missing_residue_current_capture" in row["blockers"] for row in rows
            ),
            missing_final_output=sum(
                "missing_same_backend_final_output_capture" in row["blockers"] for row in rows
            ),
            missing_cpu_baseline=sum(
                "missing_cpu_final_output_baseline" in row["blockers"] for row in rows
            ),
        ),
        groups=groups,
        promotion_eligible=False,
    )


def build_report(paths: list[Path], *, diagnostic_bindings: dict | None = None) -> dict[str, Any]:
    return build_report_from_captures(
        load_report_captures(paths), diagnostic_bindings=diagnostic_bindings
    )


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "none"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Exact-Wide Residue-Current Chain Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "| backend | semantics | shape | chain | limb | residue-current us | final-output us | final/resident boundary ratio | CPU/final host speedup | decision | blockers |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for group in report["groups"]:
        for row in group["rows"]:
            shape = row["shape"]
            residue = row.get("residue_current") or {}
            final = row.get("final_output") or {}
            blockers = ",".join(row.get("blockers") or []) or "none"
            lines.append(
                "| {backend} | {semantics} | {m}x{n}x{k} | {chain} | {limb} | {residue_us} | {final_us} | {ratio} | {cpu_speedup} | {decision} | {blockers} |".format(
                    backend=row.get("backend"),
                    semantics=row.get("semantics"),
                    m=shape.get("m"),
                    n=shape.get("n"),
                    k=shape.get("k"),
                    chain=row.get("chain_length"),
                    limb=row.get("limb_count"),
                    residue_us=fmt(residue.get("median_end_to_end_us")),
                    final_us=fmt(final.get("median_end_to_end_us")),
                    ratio=fmt(row.get("final_output_vs_residue_current_ratio")),
                    cpu_speedup=fmt(row.get("cpu_vs_final_output_speedup")),
                    decision=row.get("decision"),
                    blockers=blockers,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "exact-wide-chain-report.json"
    md_path = out_dir / "exact-wide-chain-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="*")
    parser.add_argument("--capture", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--diagnostic-binding", type=Path, action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [*args.captures, *args.capture]
    if not paths:
        raise SystemExit("at least one capture file or directory is required")
    report = build_report(paths, diagnostic_bindings=load_bindings(args.diagnostic_binding))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        paths_written = write_report(report, args.out_dir)
        print(paths_written["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
