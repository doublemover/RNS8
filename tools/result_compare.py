#!/usr/bin/env python3
"""Compare two rns8-bench JSON result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, schema_version, validate_capture


TIMING_PHASES = [
    "global_bound_scan",
    "planning",
    "scheduling",
    "tile_bound_scan",
    "matrix_alloc",
    "prepack_setup",
    "pack",
    "rns_gemm",
    "per_modulus_gemm_estimate",
    "crt_export",
    "end_to_end",
]
CONTRACT_KEYS = [
    "semantics",
    "bound_kind",
    "bound_mode",
    "bound",
    "m",
    "n",
    "k",
    "prefix",
    "bound_source",
    "selected_prefix",
    "requested_max_prefix",
    "contract_prefix_policy",
    "residue_planes_requested",
    "residue_planes_selected",
    "residue_planes_skipped",
    "tile_m",
    "tile_n",
    "k_block_size",
    "schedule_metadata.bound_kind",
    "schedule_metadata.effective_bound",
    "schedule_metadata.lhs_bound",
    "schedule_metadata.rhs_bound",
    "schedule_metadata.bound_contract",
    "schedule_metadata.tile_rows",
    "schedule_metadata.tile_cols",
    "schedule_metadata.tile_count",
    "schedule_metadata.min_required_prefix",
    "schedule_metadata.max_required_prefix",
    "schedule_metadata.min_selected_prefix",
    "schedule_metadata.max_selected_prefix",
    "schedule_metadata.prefix_group_count",
    "schedule_metadata.adaptive_prefix_active",
    "schedule_metadata.adaptive_skip_active",
    "schedule_metadata.adaptive_execution_applied",
    "schedule_metadata.zero_a_row_proof_count",
    "schedule_metadata.zero_b_col_proof_count",
    "schedule_metadata.zero_row_col_product_count",
    "schedule_metadata.planner_zero_a_row_count",
    "schedule_metadata.planner_zero_b_col_count",
    "schedule_metadata.planner_zero_row_col_product_count",
    "schedule_metadata.range_bit_length",
    "tile_bounds_u64.source",
    "tile_bounds_u64.pattern",
    "tile_bounds_u64.order",
    "tile_bounds_u64.count",
    "tile_bounds_u64.min",
    "tile_bounds_u64.max",
    "tile_bounds_u64.hash_u64",
    "packed_layout_version",
    "reuse_packed_inputs",
    "pack_mode",
    "requested_next_op.resolved",
    "output_policy.destination_layout",
    "output_policy.status_handling",
    "plan_packing.input_domain_name",
    "plan_packing.output_domain_name",
    "plan_packing.uses_transient_pack_workspace",
    "plan_packing.uses_matrix_engine_pack_layout",
    "plan_lowering.lowering_path",
    "timing_metadata.pack_layout",
    "timing_metadata.fusion_mode",
    "timing_metadata.residue_group_width",
    "timing_metadata.residue_group_layout",
    "timing_metadata.generated_reducer_identity",
    "prepack_reuse_operands",
    "prepack_reuse_strategy",
    "device_allocation.setup_scope",
    "seed",
    "input_distribution",
]
GPU_COMPATIBILITY_KEYS = [
    "compiler.id",
    "compiler.version",
    "configured_amdgpu_targets",
    "hip_toolchain.enabled",
    "hip_toolchain.hip_root",
    "hip_toolchain.hipcc_path",
    "hip_toolchain.hipcc_version",
    "hip_toolchain.hip_sdk_or_rocm_version",
    "device.gcn_arch",
    "device.hip_runtime_version",
    "device.hip_driver_version",
    "target_variant.target_id",
    "target_variant.target_namespace",
    "target_variant.review_group_key",
]
REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}
BACKEND_EVIDENCE_KEYS = [
    "benchmark",
    "backend_requested",
    "backend_selected",
    "selected_kernel",
    "backend_metadata.source",
    "backend_metadata.selected_kernel",
    "backend_metadata.accelerator_backend",
    "backend_metadata.correctness_backend",
    "backend_metadata.matrix_engine_backend",
    "backend_metadata.compiled_kernel_available",
    "backend_metadata.exact_differential_validated",
    "backend_metadata.performance_validated",
    "backend_metadata.accelerator_library",
    "backend_metadata.accelerator_version",
    "backend_metadata.capability_status",
    "backend_metadata.epilogue_mode",
    "backend_metadata.workspace_mode",
    "backend_metadata.workspace_required_bytes",
    "backend_metadata.isa_evidence",
    "backend_metadata.autotune_key",
    "auto_selector.selected_key",
    "auto_selector.validated_hit",
    "epilogue_type",
    "warmups",
    "repeats",
    "per_modulus_gemm_estimate_applicable",
    "timing_source",
    "timing_metadata.gpu_event_timing_source_scope",
    "timing_metadata.phase_availability.scheduling.scope",
    "timing_metadata.phase_availability.tile_bound_scan.scope",
    "timing_metadata.phase_availability.reduction.scope",
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


def capture_pack_mode(data: dict[str, Any]) -> str:
    metadata = data.get("timing_metadata")
    mode = metadata.get("pack_mode") if isinstance(metadata, dict) else None
    if mode is None:
        mode = data.get("pack_mode")
    if isinstance(mode, str):
        return mode
    return "prepacked_reuse" if data.get("reuse_packed_inputs") is True else "per_repeat_repack"


def capture_prepack_reuse_strategy(data: dict[str, Any]) -> str:
    metadata = data.get("timing_metadata")
    strategy = metadata.get("prepack_reuse_strategy") if isinstance(metadata, dict) else None
    if strategy is None:
        strategy = data.get("prepack_reuse_strategy")
    if isinstance(strategy, str):
        return strategy
    return "persistent_matrix_residency" if data.get("reuse_packed_inputs") is True else "none"


def capture_bound_source(data: dict[str, Any]) -> str:
    value = data.get("bound_source")
    return value if isinstance(value, str) else "static_profile"


def contract_value(data: dict[str, Any], key: str) -> Any:
    if key == "reuse_packed_inputs":
        return data.get("reuse_packed_inputs") is True
    if key == "pack_mode":
        return capture_pack_mode(data)
    if key == "prepack_reuse_strategy":
        return capture_prepack_reuse_strategy(data)
    if key == "bound_source":
        return capture_bound_source(data)
    return dotted_get(data, key)


def phase_timing(data: dict[str, Any], phase: str, path: Path) -> tuple[float, str]:
    if phase == "per_modulus_gemm_estimate":
        value = data.get("avg_per_modulus_gemm_estimate_us")
        if isinstance(value, (int, float)):
            return float(value), "avg_per_modulus_gemm_estimate_us"
        raise SystemExit(f"{path}: missing numeric timing for phase avg_per_modulus_gemm_estimate_us")
    if phase == "prepack_setup":
        value = data.get("avg_prepack_setup_us")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), "avg_prepack_setup_us"
        raise SystemExit(f"{path}: missing numeric timing for phase avg_prepack_setup_us")

    summary = data.get("timing_summary_us")
    if isinstance(summary, dict):
        phase_summary = summary.get(phase)
        if isinstance(phase_summary, dict):
            value = phase_summary.get("avg")
            if isinstance(value, (int, float)):
                return float(value), f"timing_summary_us.{phase}.avg"

    raise SystemExit(f"{path}: missing numeric timing for phase timing_summary_us.{phase}.avg")


def phase_applicable(data: dict[str, Any], phase: str) -> bool:
    if phase == "per_modulus_gemm_estimate":
        value = data.get("per_modulus_gemm_estimate_applicable")
        return value if isinstance(value, bool) else True
    if phase == "prepack_setup":
        value = data.get("avg_prepack_setup_us")
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if phase == "tile_bound_scan":
        value = data.get("avg_tile_bound_scan_us")
        summary = data.get("timing_summary_us")
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(summary, dict)
            and isinstance(summary.get("tile_bound_scan"), dict)
        )
    if phase == "global_bound_scan":
        value = data.get("avg_global_bound_scan_us")
        summary = data.get("timing_summary_us")
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(summary, dict)
            and isinstance(summary.get("global_bound_scan"), dict)
        )
    return True


def timing_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("timing_metadata")
    return metadata if isinstance(metadata, dict) else {}


def selected_backend(data: dict[str, Any]) -> str:
    backend = data.get("backend_selected")
    return str(backend) if backend is not None else ""


def is_gpu_capture(data: dict[str, Any]) -> bool:
    backend = selected_backend(data)
    if backend in REFERENCE_BACKENDS:
        return False
    hip_toolchain = data.get("hip_toolchain")
    if isinstance(hip_toolchain, dict) and hip_toolchain.get("enabled") is False:
        return False
    device = data.get("device")
    if isinstance(device, dict):
        gcn_arch = str(device.get("gcn_arch") or "").strip().lower()
        return gcn_arch not in {"", "none", "cpu", "unknown", "not_applicable", "n/a", "null"}
    return backend not in {"", "cpu-reference"}


def gpu_event_phase_timing(data: dict[str, Any], phase: str, path: Path) -> tuple[float, str]:
    summary = data.get("gpu_event_timing_summary_us")
    if isinstance(summary, dict):
        phase_summary = summary.get(phase)
        if isinstance(phase_summary, dict):
            value = phase_summary.get("avg")
            if isinstance(value, (int, float)):
                return float(value), f"gpu_event_timing_summary_us.{phase}.avg"
    raise SystemExit(f"{path}: missing numeric GPU event timing for phase {phase}")


def gpu_event_phase_order(data: dict[str, Any]) -> list[str]:
    metadata = timing_metadata(data)
    phase_order = metadata.get("gpu_event_phase_order")
    if isinstance(phase_order, list) and all(isinstance(item, str) for item in phase_order):
        return list(phase_order)
    return []


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
    base_phases = gpu_event_phase_order(baseline)
    cand_phases = gpu_event_phase_order(candidate)
    source_match = base_source == cand_source
    scope_match = base_scope == cand_scope
    phases_match = base_phases == cand_phases
    comparable = base_enabled and cand_enabled and source_match and scope_match and phases_match

    event_timings = {}
    if comparable:
        for phase in base_phases:
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
        elif not phases_match:
            reason = "gpu_event_phase_order_mismatch"

    return {
        "comparable": comparable,
        "reason": reason,
        "baseline_enabled": base_enabled,
        "candidate_enabled": cand_enabled,
        "baseline_source": base_source,
        "candidate_source": cand_source,
        "baseline_source_scope": base_scope,
        "candidate_source_scope": cand_scope,
        "baseline_phase_order": base_phases,
        "candidate_phase_order": cand_phases,
        "timings": event_timings,
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any], baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    contract = {
        key: {
            "baseline": contract_value(baseline, key),
            "candidate": contract_value(candidate, key),
            "match": contract_value(baseline, key) == contract_value(candidate, key),
        }
        for key in CONTRACT_KEYS
    }
    backend_evidence = {
        key: {
            "baseline": dotted_get(baseline, key),
            "candidate": dotted_get(candidate, key),
            "match": dotted_get(baseline, key) == dotted_get(candidate, key),
        }
        for key in BACKEND_EVIDENCE_KEYS
    }
    baseline_gpu = is_gpu_capture(baseline)
    candidate_gpu = is_gpu_capture(candidate)
    gpu_compatibility = {
        key: {
            "baseline": dotted_get(baseline, key),
            "candidate": dotted_get(candidate, key),
            "match": dotted_get(baseline, key) == dotted_get(candidate, key),
        }
        for key in GPU_COMPATIBILITY_KEYS
    }
    gpu_compatibility_required = baseline_gpu and candidate_gpu
    gpu_compatible = (not gpu_compatibility_required) or all(item["match"] for item in gpu_compatibility.values())
    timings = {}
    for phase in TIMING_PHASES:
        base_applicable = phase_applicable(baseline, phase)
        cand_applicable = phase_applicable(candidate, phase)
        base, base_source = phase_timing(baseline, phase, baseline_path) if base_applicable else (None, "not_applicable")
        cand, cand_source = phase_timing(candidate, phase, candidate_path) if cand_applicable else (None, "not_applicable")
        delta = cand - base if base is not None and cand is not None else None
        ratio = cand / base if base is not None and base != 0 and cand is not None else None
        timings[phase] = {
            "baseline": base,
            "candidate": cand,
            "delta": delta,
            "ratio": ratio,
            "baseline_source": base_source,
            "candidate_source": cand_source,
            "baseline_applicable": base_applicable,
            "candidate_applicable": cand_applicable,
        }

    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "schema": {
            "baseline_version": schema_version(baseline),
            "candidate_version": schema_version(candidate),
        },
        "matching_contract": all(item["match"] for item in contract.values()),
        "gpu_compatible": gpu_compatible,
        "gpu_compatibility_required": gpu_compatibility_required,
        "contract": contract,
        "gpu_compatibility": gpu_compatibility,
        "backend_evidence": backend_evidence,
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
    print(f"gpu compatible:   {report['gpu_compatible']} (required={report['gpu_compatibility_required']})")
    print()
    print("Contract")
    for key, item in report["contract"].items():
        status = "OK" if item["match"] else "DIFF"
        print(f"[{status}] {key}: {item['baseline']} -> {item['candidate']}")
    print()
    print("GPU Compatibility")
    for key, item in report["gpu_compatibility"].items():
        status = "OK" if item["match"] else "DIFF"
        print(f"[{status}] {key}: {item['baseline']} -> {item['candidate']}")
    print()
    print("Backend Evidence")
    for key, item in report["backend_evidence"].items():
        status = "OK" if item["match"] else "DIFF"
        print(f"[{status}] {key}: {item['baseline']} -> {item['candidate']}")
    print()
    print("Timings")
    for phase, item in report["timings"].items():
        ratio = item["ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6g}"
        base_text = "n/a" if item["baseline"] is None else f"{item['baseline']:.6g}"
        cand_text = "n/a" if item["candidate"] is None else f"{item['candidate']:.6g}"
        delta_text = "n/a" if item["delta"] is None else f"{item['delta']:.6g}"
        applicability = (
            "" if item["baseline_applicable"] and item["candidate_applicable"] else " [not applicable to one or both captures]"
        )
        print(
            f"{phase}: baseline={base_text} "
            f"candidate={cand_text} delta={delta_text} ratio={ratio_text}{applicability}"
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
    return 0 if report["matching_contract"] and report["gpu_compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
