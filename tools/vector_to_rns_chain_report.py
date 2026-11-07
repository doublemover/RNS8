#!/usr/bin/env python3
"""Classify vector/native producer to Direct-HIP RNS consumer chain captures."""

from __future__ import annotations

from reference_verdict import reference_verdict, checksum_u64
from diagnostic_companion import load_bindings
from observation_cohorts import phase_median, comparison_environment
from report_comparison import ComparisonPolicy, ReportCohorts

import argparse
import json
from pathlib import Path
from typing import Any

from report_capture_inputs import load_report_captures


PROMOTION_SPEEDUP_THRESHOLD = 1.02
COMPARISON_POLICY = ComparisonPolicy(
    "same_completed_bounded_two_product_chain_device_handoff_vs_host_repack_v1",
    require_cpu_anchor=False,
)


def timing_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("timing_metadata")
    return value if isinstance(value, dict) else {}


def scenario_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = capture.get("scenario_metadata")
    return value if isinstance(value, dict) else {}


def scenario_inner_metadata(capture: dict[str, Any]) -> dict[str, Any]:
    value = scenario_metadata(capture).get("metadata")
    return value if isinstance(value, dict) else {}


def control_mode(capture: dict[str, Any]) -> str:
    metadata = timing_metadata(capture)
    value = metadata.get("vector_to_rns_chain_control_mode")
    if isinstance(value, str) and value:
        return value
    scenario_value = scenario_inner_metadata(capture).get("chain_control_mode")
    return str(scenario_value or "")


def target_id(capture: dict[str, Any]) -> str:
    target = capture.get("target_variant")
    if isinstance(target, dict) and target.get("target_id"):
        return str(target["target_id"])
    return str(capture.get("target_id") or "")


def timing_median(capture: dict[str, Any], phase: str) -> float | None:
    return phase_median(capture, phase)


def event_median(capture: dict[str, Any], phase: str) -> float | None:
    return phase_median(capture, phase, event=True)


def chain_key(capture: dict[str, Any]) -> tuple[str, str]:
    return COMPARISON_POLICY.work(capture), comparison_environment(capture)


def vector_chain_captures(captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        capture
        for capture in captures
        if capture.get("benchmark_execution_mode")
        in {
            "vector_native_to_direct_rns_chain",
            "vector_native_host_export_repack_direct_rns_chain",
        }
    ]


def classify_pair(
    candidate: dict[str, Any] | None, control: dict[str, Any] | None, cohorts: ReportCohorts
) -> dict[str, Any]:
    blockers: list[str] = []
    if candidate is None:
        blockers.append("missing_fused_device_candidate")
    if control is None:
        blockers.append("missing_host_export_repack_control")
    if candidate is None or control is None:
        return {
            "disposition": "keep experimental",
            "blockers": blockers,
        }

    candidate_median = timing_median(candidate, "end_to_end")
    control_median = timing_median(control, "end_to_end")
    if candidate_median is None:
        blockers.append("missing_candidate_end_to_end_median")
    if control_median is None:
        blockers.append("missing_control_end_to_end_median")
    numerical = reference_verdict(candidate, [control])
    control_numerical = reference_verdict(control, [candidate])
    blockers.extend(numerical.blockers)
    blockers.extend("control:" + item for item in control_numerical.blockers)
    candidate_checksum, control_checksum = checksum_u64(candidate), checksum_u64(control)
    checksum_match = (
        candidate_checksum is not None
        and control_checksum is not None
        and candidate_checksum == control_checksum
    )
    if not checksum_match:
        blockers.append("final_checksum_mismatch_or_missing")
    comparison = cohorts.compare(candidate, control, COMPARISON_POLICY)
    blockers.extend(comparison["blockers"])
    blockers = sorted(set(blockers))
    speedup = comparison["speedup"] if not blockers else None
    margin = comparison["required_speedup_margin"]
    required = max(PROMOTION_SPEEDUP_THRESHOLD, margin) if margin is not None else None
    if blockers:
        disposition = "keep experimental"
    elif speedup is not None and required is not None and speedup >= required and speedup > margin:
        disposition = "local promote"
    elif speedup is not None and speedup < 1.0:
        disposition = "drop/deprioritize"
    else:
        disposition = "keep experimental"
        blockers.append("speedup_not_separated_from_observed_noise")

    return {
        "disposition": disposition,
        "blockers": blockers,
        "comparison": comparison,
        "required_speedup_margin": required,
        "candidate_end_to_end_median_us": candidate_median,
        "control_end_to_end_median_us": control_median,
        "speedup_vs_host_repack_control": speedup,
        "checksum_match": checksum_match,
        "numerical_reference": numerical.to_json(),
        "control_numerical_reference": control_numerical.to_json(),
        "candidate_pack_median_us": timing_median(candidate, "pack"),
        "candidate_chain_median_us": timing_median(candidate, "rns_gemm"),
        "candidate_export_median_us": timing_median(candidate, "crt_export"),
        "control_pack_median_us": timing_median(control, "pack"),
        "control_chain_median_us": timing_median(control, "rns_gemm"),
        "control_export_median_us": timing_median(control, "crt_export"),
        "candidate_conversion_event_us": event_median(
            candidate,
            "native_i64_to_rns_kernel"
            if candidate.get("semantics") == "bounded_i64"
            else "native_u64_to_rns_kernel",
        ),
        "control_vector_output_d2h_us": event_median(control, "vector_alu_output_d2h"),
        "control_host_repack_us": event_median(control, "vector_to_rns_host_repack_a"),
    }


def build_report(
    captures: list[dict[str, Any]], *, diagnostic_bindings: dict | None = None
) -> dict[str, Any]:
    cohorts = ReportCohorts(captures, diagnostic_bindings)
    selected = vector_chain_captures(cohorts.observations)
    groups = {}
    for capture in selected:
        groups.setdefault(chain_key(capture), {}).setdefault(control_mode(capture), []).append(
            capture
        )

    rows = []
    for key, group in sorted(groups.items()):
        candidates = group.get("fused_device_native_to_rns", [])
        controls = group.get("host_export_repack_control", [])
        all_views = [view for views in group.values() for view in views]
        anchor = next(iter(candidates or controls or all_views))
        candidate = cohorts.best(candidates, anchor, COMPARISON_POLICY)
        control = cohorts.best(controls, anchor, COMPARISON_POLICY)
        # Unqualified views remain descriptive; never select a fastest raw file.
        candidate = candidate or next(iter(candidates), None)
        control = control or next(iter(controls), None)
        classified = classify_pair(candidate, control, cohorts)
        observations = [cohorts.metadata(view, COMPARISON_POLICY) for view in all_views]
        representative = candidate or control or anchor
        rows.append(
            dict(
                target_id=target_id(representative),
                semantics=representative.get("semantics"),
                shape={field: representative.get(field) for field in ("m", "n", "k")},
                pack_mode=representative.get("pack_mode"),
                backend_selected=representative.get("backend_selected"),
                backend_requested=representative.get("backend_requested"),
                work_contract=key[0],
                comparison_environment=key[1],
                observations=observations,
                candidate_capture=candidate.get("_path") if candidate is not None else None,
                control_capture=control.get("_path") if control is not None else None,
                scenario_name=scenario_metadata(representative).get("name"),
                **classified,
            )
        )
    counts = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    return dict(
        schema="rns8_vector_to_rns_chain_report_v1",
        capture_count=len(vector_chain_captures(captures)),
        observation_cohort_count=len(selected),
        group_count=len(rows),
        disposition_counts=counts,
        rows=rows,
    )


def write_outputs(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vector-to-rns-chain-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Vector/Native-To-RNS Chain Report",
        "",
        f"- Captures: {report['capture_count']}",
        f"- Groups: {report['group_count']}",
        f"- Dispositions: {json.dumps(report['disposition_counts'], sort_keys=True)}",
        "",
        "| Semantics | Shape | Pack Mode | Candidate us | Control us | Speedup | Disposition | Blockers |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        shape = row["shape"]
        speedup = row.get("speedup_vs_host_repack_control")
        lines.append(
            "| {semantics} | {shape} | {pack} | {candidate} | {control} | {speedup} | {disp} | {blockers} |".format(
                semantics=row["semantics"],
                shape=f"{shape['m']}x{shape['n']}x{shape['k']}",
                pack=row["pack_mode"],
                candidate=f"{row.get('candidate_end_to_end_median_us'):.3f}"
                if isinstance(row.get("candidate_end_to_end_median_us"), float)
                else "",
                control=f"{row.get('control_end_to_end_median_us'):.3f}"
                if isinstance(row.get("control_end_to_end_median_us"), float)
                else "",
                speedup=f"{speedup:.3f}x" if isinstance(speedup, float) else "",
                disp=row["disposition"],
                blockers=", ".join(row.get("blockers") or []),
            )
        )
    (out_dir / "vector-to-rns-chain-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("temp/vector-to-rns-chain-report"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--diagnostic-binding", type=Path, action="append", default=[])
    args = parser.parse_args()

    captures = load_report_captures(args.captures)
    report = build_report(captures, diagnostic_bindings=load_bindings(args.diagnostic_binding))
    write_outputs(report, args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(args.out_dir / "vector-to-rns-chain-report.md")


if __name__ == "__main__":
    main()
