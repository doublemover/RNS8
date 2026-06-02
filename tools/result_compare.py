#!/usr/bin/env python3
"""Compare two rns8-bench JSON result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, schema_version, validate_capture


TIMING_PHASES = {
    "planning": ["avg_planning_us", "plan_us"],
    "matrix_alloc": ["avg_matrix_alloc_us", "matrix_alloc_us"],
    "pack": ["avg_pack_us"],
    "rns_gemm": ["avg_rns_gemm_us"],
    "per_modulus_gemm_estimate": ["avg_per_modulus_gemm_estimate_us"],
    "crt_export": ["avg_crt_export_us"],
    "end_to_end": ["avg_end_to_end_us"],
}
GPU_EVENT_PHASES = [
    "pack_h2d",
    "pack_kernel",
    "pack",
    "rns_gemm_kernel_group",
    "rns_gemm",
    "crt_export_status_memset",
    "crt_export_kernel",
    "crt_export_status_d2h",
    "crt_export_d2h",
    "crt_export",
]
CONTRACT_KEYS = [
    "benchmark",
    "backend_requested",
    "backend_selected",
    "semantics",
    "bound_kind",
    "bound",
    "m",
    "n",
    "k",
    "prefix",
    "k_block_size",
    "epilogue_type",
    "packed_layout_version",
    "seed",
    "warmups",
    "repeats",
    "input_distribution",
    "timing_source",
    "compiler.id",
    "compiler.version",
    "configured_amdgpu_targets",
    "device.gcn_arch",
    "device.hip_runtime_version",
    "device.hip_driver_version",
]


def load_result(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path}: failed to read benchmark JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: benchmark JSON root must be an object")
    try:
        validate_capture(data, path)
    except BenchmarkSchemaError as exc:
        raise SystemExit(str(exc)) from exc
    return data


def dotted_get(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def phase_timing(data: dict[str, Any], phase: str, path: Path) -> tuple[float, str]:
    summary = data.get("timing_summary_us")
    if isinstance(summary, dict):
        phase_summary = summary.get(phase)
        if isinstance(phase_summary, dict):
            value = phase_summary.get("avg")
            if isinstance(value, (int, float)):
                return float(value), f"timing_summary_us.{phase}.avg"

    for key in TIMING_PHASES[phase]:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value), key

    keys = ", ".join([f"timing_summary_us.{phase}.avg", *TIMING_PHASES[phase]])
    raise SystemExit(f"{path}: missing numeric timing for phase {phase}; looked for {keys}")


def phase_applicable(data: dict[str, Any], phase: str) -> bool:
    if phase == "per_modulus_gemm_estimate":
        value = data.get("per_modulus_gemm_estimate_applicable")
        return value if isinstance(value, bool) else True
    return True


def timing_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("timing_metadata")
    return metadata if isinstance(metadata, dict) else {}


def gpu_event_phase_timing(data: dict[str, Any], phase: str, path: Path) -> tuple[float, str]:
    summary = data.get("gpu_event_timing_summary_us")
    if isinstance(summary, dict):
        phase_summary = summary.get(phase)
        if isinstance(phase_summary, dict):
            value = phase_summary.get("avg")
            if isinstance(value, (int, float)):
                return float(value), f"gpu_event_timing_summary_us.{phase}.avg"
    raise SystemExit(f"{path}: missing numeric GPU event timing for phase {phase}")


def compare_gpu_events(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    base_meta = timing_metadata(baseline)
    cand_meta = timing_metadata(candidate)
    base_enabled = base_meta.get("gpu_event_timing") is True
    cand_enabled = cand_meta.get("gpu_event_timing") is True
    base_source = base_meta.get("gpu_event_timing_source")
    cand_source = cand_meta.get("gpu_event_timing_source")
    base_scope = base_meta.get("gpu_event_timing_source_scope")
    cand_scope = cand_meta.get("gpu_event_timing_source_scope")
    source_match = base_source == cand_source
    scope_match = base_scope == cand_scope
    comparable = base_enabled and cand_enabled and source_match and scope_match

    event_timings = {}
    if comparable:
        for phase in GPU_EVENT_PHASES:
            base, base_source_key = gpu_event_phase_timing(baseline, phase, baseline_path)
            cand, cand_source_key = gpu_event_phase_timing(candidate, phase, candidate_path)
            event_timings[phase] = {
                "baseline": base,
                "candidate": cand,
                "delta": cand - base,
                "ratio": cand / base if base != 0 else None,
                "baseline_source": base_source_key,
                "candidate_source": cand_source_key,
            }

    reason = "comparable"
    if not comparable:
        if not base_enabled or not cand_enabled:
            reason = "gpu_event_timing_not_enabled_for_both_captures"
        elif not source_match:
            reason = "gpu_event_timing_source_mismatch"
        elif not scope_match:
            reason = "gpu_event_timing_source_scope_mismatch"

    return {
        "comparable": comparable,
        "reason": reason,
        "baseline_enabled": base_enabled,
        "candidate_enabled": cand_enabled,
        "baseline_source": base_source,
        "candidate_source": cand_source,
        "baseline_source_scope": base_scope,
        "candidate_source_scope": cand_scope,
        "timings": event_timings,
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any], baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    contract = {
        key: {
            "baseline": dotted_get(baseline, key),
            "candidate": dotted_get(candidate, key),
            "match": dotted_get(baseline, key) == dotted_get(candidate, key),
        }
        for key in CONTRACT_KEYS
    }
    timings = {}
    for phase in TIMING_PHASES:
        base, base_source = phase_timing(baseline, phase, baseline_path)
        cand, cand_source = phase_timing(candidate, phase, candidate_path)
        timings[phase] = {
            "baseline": base,
            "candidate": cand,
            "delta": cand - base,
            "ratio": cand / base if base != 0 else None,
            "baseline_source": base_source,
            "candidate_source": cand_source,
            "baseline_applicable": phase_applicable(baseline, phase),
            "candidate_applicable": phase_applicable(candidate, phase),
        }

    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "schema": {
            "baseline_version": schema_version(baseline),
            "candidate_version": schema_version(candidate),
        },
        "matching_contract": all(item["match"] for item in contract.values()),
        "contract": contract,
        "timings": timings,
        "gpu_event_timings": compare_gpu_events(baseline, candidate, baseline_path, candidate_path),
    }


def print_human(report: dict[str, Any]) -> None:
    print("RNS8 benchmark comparison")
    print("=========================")
    print(f"baseline:  {report['baseline']}")
    print(f"candidate: {report['candidate']}")
    print(
        "schema:    "
        f"baseline=v{report['schema']['baseline_version']} "
        f"candidate=v{report['schema']['candidate_version']}"
    )
    print(f"matching contract: {report['matching_contract']}")
    print()
    print("Contract")
    for key, item in report["contract"].items():
        status = "OK" if item["match"] else "DIFF"
        print(f"[{status}] {key}: {item['baseline']} -> {item['candidate']}")
    print()
    print("Timings")
    for phase, item in report["timings"].items():
        ratio = item["ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6g}"
        applicability = (
            "" if item["baseline_applicable"] and item["candidate_applicable"] else " [not applicable to one or both captures]"
        )
        print(
            f"{phase}: baseline={item['baseline']:.6g} "
            f"candidate={item['candidate']:.6g} delta={item['delta']:.6g} ratio={ratio_text}{applicability}"
        )
    print()
    gpu_events = report["gpu_event_timings"]
    print(f"GPU event timings: {gpu_events['reason']}")
    if gpu_events["comparable"]:
        for phase, item in gpu_events["timings"].items():
            ratio = item["ratio"]
            ratio_text = "n/a" if ratio is None else f"{ratio:.6g}"
            print(
                f"{phase}: baseline={item['baseline']:.6g} "
                f"candidate={item['candidate']:.6g} delta={item['delta']:.6g} ratio={ratio_text}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="baseline rns8-bench JSON file")
    parser.add_argument("candidate", type=Path, help="candidate rns8-bench JSON file")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    baseline = load_result(args.baseline)
    candidate = load_result(args.candidate)
    report = compare(baseline, candidate, args.baseline, args.candidate)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0 if report["matching_contract"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
