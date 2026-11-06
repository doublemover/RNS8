#!/usr/bin/env python3
"""Compare final-output RNS-chain captures under the same output contract."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import result_compare
from diagnostic_companion import load_bindings
from observation_cohorts import phase_median, evidence
from report_capture_inputs import load_report_captures
from report_comparison import ComparisonPolicy, ReportCohorts
from timing_eligibility import capture_identity


DEFAULT_OUT_DIR = Path("temp") / "rns-chain-reports"
REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}
REUSE_PACK_MODES = {"prepacked_reuse", "prepacked_reuse_a", "prepacked_reuse_b"}
NORMALIZED_CONTRACT_EXCLUDE = {
    "export_variant.selector_key",
    "reuse_packed_inputs",
    "pack_mode",
    "prepack_reuse_operands",
    "prepack_reuse_strategy",
    "resident_lifetime.enabled",
}


FINAL_POLICY = ComparisonPolicy(
    "same_completed_rns_chain_setup_inclusive_reuse_vs_independent_materialization_v2",
    work_exclusions=tuple(sorted(NORMALIZED_CONTRACT_EXCLUDE)),
    require_cpu_anchor=False,
    amortize_prepack_setup=True,
)
CPU_POLICY = ComparisonPolicy(
    "same_completed_rns_chain_setup_inclusive_same_host_build_cpu_vs_gpu_v2",
    work_exclusions=FINAL_POLICY.work_exclusions,
    require_cpu_anchor=False,
    cpu_gpu_comparison=True,
    amortize_prepack_setup=True,
)


def backend_id(capture: dict[str, Any] | None) -> str:
    if capture is None:
        return ""
    selected = capture.get("backend_selected")
    if selected is not None:
        return str(selected)
    requested = capture.get("backend_requested")
    return str(requested) if requested is not None else ""


def benchmark_execution_mode(capture: dict[str, Any]) -> str:
    value = capture.get("benchmark_execution_mode")
    if isinstance(value, str):
        return value
    metadata = capture.get("timing_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("benchmark_execution_mode"), str):
        return metadata["benchmark_execution_mode"]
    return "persistent_resident"


def is_final_output_chain(capture: dict[str, Any]) -> bool:
    return (
        benchmark_execution_mode(capture)
        in {"residue_chain_final_host_export", "residue_chain_independent_final_host_export"}
        and capture.get("residue_chain_final_export") is True
        and type(capture.get("residue_chain_length")) is int
        and capture["residue_chain_length"] > 1
    )


def is_independent_export_repack_chain(capture: dict[str, Any]) -> bool:
    return (
        benchmark_execution_mode(capture) == "residue_chain_independent_final_host_export"
        or capture.get("residue_chain_independent_final_export") is True
    )


def chain_mode(capture: dict[str, Any]) -> str:
    return (
        "independent_export_repack"
        if is_independent_export_repack_chain(capture)
        else "resident_final_output"
    )


def pack_mode(capture: dict[str, Any]) -> str:
    return result_compare.capture_pack_mode(capture)


def is_reuse_capture(capture: dict[str, Any]) -> bool:
    return pack_mode(capture) in REUSE_PACK_MODES or capture.get("reuse_packed_inputs") is True


def median_end_to_end_us(capture: dict | None) -> float | None:
    return phase_median(capture) if capture is not None else None


def setup_inclusive_median_per_repeat_us(capture: dict | None) -> float | None:
    return FINAL_POLICY.measurement(capture)["median_us"] if capture is not None else None


def break_even_repeats(
    baseline_us: float | None, reuse_steady_us: float | None, setup_us: float | None
) -> int | None:
    if baseline_us is None or reuse_steady_us is None or setup_us is None:
        return None
    savings = baseline_us - reuse_steady_us
    if savings <= 0.0:
        return None
    return max(1, math.floor(setup_us / savings) + 1)


def normalized_contract_key(capture: dict) -> str:
    return FINAL_POLICY.work(capture)


def capture_summary(
    capture: dict | None, cohorts: ReportCohorts, policy=FINAL_POLICY
) -> dict | None:
    if capture is None:
        return None
    return dict(
        path=capture.get("_path"),
        backend=backend_id(capture),
        pack_mode=pack_mode(capture),
        chain_mode=chain_mode(capture),
        median_end_to_end_us=phase_median(capture),
        setup_inclusive_median_per_repeat_us=policy.measurement(capture)["median_us"],
        **cohorts.metadata(capture, policy),
    )


def select_baseline(
    values: list[dict], candidate: dict, cohorts: ReportCohorts, policy=FINAL_POLICY
) -> dict | None:
    pool = [
        view
        for view in values
        if policy.processor_pair(candidate, view)
        and policy.work(view) == policy.work(candidate)
        and policy.environment(view) == policy.environment(candidate)
    ]
    return cohorts.best(pool, candidate, policy) or next(iter(pool), None)


def source_identity_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    if not is_reuse_capture(capture):
        return {"available": None, "reason": "not_reuse_capture"}
    allocation = capture.get("device_allocation")
    if not isinstance(allocation, dict):
        return {"available": False, "reason": "missing_device_allocation_metadata"}
    setup_scope = allocation.get("setup_scope")
    source_version_inputs = allocation.get("source_version_inputs")
    available = (
        setup_scope == "persistent_plan_workspace_prepacked_reuse"
        and isinstance(source_version_inputs, str)
        and bool(source_version_inputs)
    )
    return {
        "available": available,
        "setup_scope": setup_scope,
        "source_version_inputs": source_version_inputs,
        "reason": "available" if available else "incomplete_prepack_source_identity_metadata",
    }


def row_for_capture(capture: dict, items: list[dict], cohorts: ReportCohorts) -> dict:
    backend = backend_id(capture)
    cpu = backend in REFERENCE_BACKENDS
    independent = is_independent_export_repack_chain(capture)
    reuse = is_reuse_capture(capture)
    nonreuse = [
        view
        for view in items
        if not is_reuse_capture(view)
        and not is_independent_export_repack_chain(view)
        and backend_id(view) not in REFERENCE_BACKENDS
    ]
    independent_views = [
        view
        for view in items
        if is_independent_export_repack_chain(view)
        and backend_id(view) == backend
        and not is_reuse_capture(view)
    ]
    cpu_views = [view for view in items if backend_id(view) in REFERENCE_BACKENDS]
    cpu_baseline = select_baseline(cpu_views, capture, cohorts, CPU_POLICY) if not cpu else None
    same = select_baseline(
        [view for view in nonreuse if backend_id(view) == backend and view is not capture],
        capture,
        cohorts,
    )
    best = select_baseline(nonreuse, capture, cohorts)
    independent_baseline = select_baseline(independent_views, capture, cohorts)
    baselines = dict(
        cpu=cpu_baseline,
        same_backend_nonreuse=same,
        best_nonreuse=best,
        independent_export_repack=independent_baseline,
    )
    comparisons = {
        name: cohorts.compare(capture, value, CPU_POLICY if name == "cpu" else FINAL_POLICY)
        for name, value in baselines.items()
    }
    blockers = cohorts.admission(capture, FINAL_POLICY)
    source = source_identity_metadata(capture)
    source_observations = [
        dict(capture_sha256=capture_identity(raw), **source_identity_metadata(raw))
        for raw in cohorts.raw_observations(capture)
    ]
    source["observations"] = source_observations
    source_failures = sorted(
        {item["reason"] for item in source_observations if item.get("available") is False}
    )
    if reuse:
        blockers.extend(source_failures)
    if reuse and any(item.get("available") is False for item in source_observations):
        source["available"] = False
        source["reason"] = "reuse_process_source_identity_incomplete"
    if reuse and source.get("available") is False:
        for comparison in comparisons.values():
            comparison["blockers"] = sorted(set(comparison["blockers"] + [source["reason"]]))
            comparison["qualified"] = False
            comparison["speedup"] = None
    if cpu or independent:
        decision = (
            "cpu_independent_export_repack_baseline"
            if cpu and independent
            else "cpu_final_output_baseline"
            if cpu
            else "independent_export_repack_baseline"
        )
        if blockers:
            decision = "keep_experimental"
    else:
        required = ["cpu", "independent_export_repack"]
        if reuse:
            required.extend(["same_backend_nonreuse", "best_nonreuse"])
            if source.get("available") is False:
                blockers.append(source["reason"])
        missing = []
        for name in required:
            comparison = comparisons[name]
            blockers.extend(name + ":" + item for item in comparison["blockers"])
            if baselines[name] is None:
                missing.append(name)
        if "cpu" in missing:
            blockers.append("missing_cpu_final_output_baseline")
        if "independent_export_repack" in missing:
            blockers.append("missing_same_backend_independent_export_repack_baseline")
        if "same_backend_nonreuse" in missing:
            blockers.append("missing_same_backend_nonreuse_chain_baseline")
        if blockers:
            decision = "keep_experimental"
        elif any(comparisons[name]["speedup"] <= 1 for name in required):
            decision = "deprioritize"
            blockers.append("setup_inclusive_chain_does_not_beat_required_baselines")
        elif any(
            comparisons[name]["speedup"] <= comparisons[name]["required_speedup_margin"]
            for name in required
        ):
            decision = "keep_experimental"
            blockers.append("speedup_not_separated_from_observed_noise")
        else:
            decision = "candidate_reuse_chain_win" if reuse else "candidate_final_output_chain_win"
    projected_break_even = None
    if reuse and comparisons["same_backend_nonreuse"]["qualified"]:
        projected_break_even = break_even_repeats(
            median_end_to_end_us(same),
            phase_median(capture),
            phase_median(capture, "prepack_setup"),
        )
    return dict(
        path=capture.get("_path"),
        backend=backend,
        semantics=capture.get("semantics"),
        shape={key: capture.get(key) for key in ("m", "n", "k")},
        chain_length=capture.get("residue_chain_length"),
        chain_mode=chain_mode(capture),
        pack_mode=pack_mode(capture),
        reuse=reuse,
        prepack_reuse_operands=capture.get("prepack_reuse_operands"),
        repeats=capture.get("repeats"),
        median_end_to_end_us=phase_median(capture),
        setup_inclusive_median_per_repeat_us=setup_inclusive_median_per_repeat_us(capture),
        avg_prepack_setup_us=phase_median(capture, "prepack_setup"),
        observation=cohorts.metadata(capture, FINAL_POLICY),
        cpu_baseline=capture_summary(cpu_baseline, cohorts, CPU_POLICY),
        same_backend_nonreuse=capture_summary(same, cohorts),
        best_nonreuse=capture_summary(best, cohorts),
        independent_export_repack_baseline=capture_summary(independent_baseline, cohorts),
        speedup_vs_cpu=comparisons["cpu"]["speedup"],
        speedup_vs_same_backend_nonreuse=comparisons["same_backend_nonreuse"]["speedup"],
        speedup_vs_best_nonreuse=comparisons["best_nonreuse"]["speedup"],
        speedup_vs_independent_export_repack=comparisons["independent_export_repack"]["speedup"],
        comparisons=comparisons,
        source_identity=source,
        break_even_repeats_same_backend=projected_break_even,
        break_even_scope="projection_from_observed_cohort_medians_not_application_lifetime_qualification",
        decision=decision,
        blockers=sorted(set(blockers)),
        promotion_eligible=False,
    )


def build_report_from_captures(
    captures: list[dict], *, diagnostic_bindings: dict | None = None
) -> dict:
    cohorts = ReportCohorts(captures, diagnostic_bindings)
    selected = [view for view in cohorts.observations if is_final_output_chain(view)]
    grouped = defaultdict(list)
    for view in selected:
        grouped[(normalized_contract_key(view), CPU_POLICY.environment(view))].append(view)
    groups = []
    for (work, environment), items in sorted(grouped.items()):
        rows = [row_for_capture(view, items, cohorts) for view in items]
        groups.append(
            dict(
                contract_key=work,
                comparison_environment=environment,
                summary=dict(
                    capture_count=sum(evidence(view)["submitted_records"] for view in items),
                    implementation_count=len(items),
                ),
                rows=rows,
            )
        )
    rows = [row for group in groups for row in group["rows"]]
    candidates = [
        row
        for row in rows
        if row["decision"] in {"candidate_final_output_chain_win", "candidate_reuse_chain_win"}
    ]
    comparison_rows = [
        row
        for row in rows
        if row["backend"] not in REFERENCE_BACKENDS
        and row["chain_mode"] != "independent_export_repack"
    ]
    return dict(
        schema_version=2,
        policy="rns_chain_complete_output_independent_production_cohorts_and_observed_setup_v2",
        capture_count=len(captures),
        final_output_chain_capture_count=sum(is_final_output_chain(view) for view in captures),
        summary=dict(
            groups=len(groups),
            gpu_or_reuse_comparisons=len(comparison_rows),
            candidate_wins=len(candidates),
            reuse_candidate_wins=sum(row["reuse"] for row in candidates),
            independent_export_repack_baselines=sum(
                row["decision"] == "independent_export_repack_baseline" for row in rows
            ),
            deprioritized=sum(row["decision"] == "deprioritize" for row in comparison_rows),
            experimental=sum(row["decision"] == "keep_experimental" for row in comparison_rows),
            missing_baselines=sum(
                row["cpu_baseline"] is None or row["independent_export_repack_baseline"] is None
                for row in comparison_rows
            ),
        ),
        groups=groups,
        promotion_eligible=False,
    )


def build_report(paths: list[Path], *, diagnostic_bindings: dict | None = None) -> dict:
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
        "# RNS Chain Final-Output Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "| backend | semantics | shape | chain | chain mode | pack mode | setup-inclusive us | CPU us | independent us | independent speedup | same-backend speedup | CPU speedup | break-even repeats | decision | blockers |",
            "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for group in report["groups"]:
        for row in group["rows"]:
            if row["backend"] in REFERENCE_BACKENDS:
                continue
            shape = row["shape"]
            cpu = row.get("cpu_baseline") or {}
            independent = row.get("independent_export_repack_baseline") or {}
            blockers = ",".join(row.get("blockers") or []) or "none"
            lines.append(
                "| {backend} | {semantics} | {m}x{n}x{k} | {chain} | {chain_mode} | {mode} | {setup} | {cpu} | {independent} | {independent_speedup} | {same_speedup} | {cpu_speedup} | {break_even} | {decision} | {blockers} |".format(
                    backend=row.get("backend"),
                    semantics=row.get("semantics"),
                    m=shape.get("m"),
                    n=shape.get("n"),
                    k=shape.get("k"),
                    chain=row.get("chain_length"),
                    chain_mode=row.get("chain_mode"),
                    mode=row.get("pack_mode"),
                    setup=fmt(row.get("setup_inclusive_median_per_repeat_us")),
                    cpu=fmt(cpu.get("setup_inclusive_median_per_repeat_us")),
                    independent=fmt(independent.get("setup_inclusive_median_per_repeat_us")),
                    independent_speedup=fmt(row.get("speedup_vs_independent_export_repack")),
                    same_speedup=fmt(row.get("speedup_vs_same_backend_nonreuse")),
                    cpu_speedup=fmt(row.get("speedup_vs_cpu")),
                    break_even=fmt(row.get("break_even_repeats_same_backend")),
                    decision=row.get("decision"),
                    blockers=blockers,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rns-chain-report.json"
    md_path = out_dir / "rns-chain-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        action="append",
        help="capture file or directory; directories are searched recursively for JSON",
    )
    parser.add_argument("captures", type=Path, nargs="*")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--diagnostic-binding", type=Path, action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = list(args.captures)
    if args.capture:
        paths.extend(args.capture)
    if not paths:
        raise SystemExit("at least one capture file or directory is required")
    report = build_report(paths, diagnostic_bindings=load_bindings(args.diagnostic_binding))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.out_md)
    if not args.json and not args.out_json and not args.out_md:
        paths_written = write_report(report, args.out_dir)
        print(paths_written["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
