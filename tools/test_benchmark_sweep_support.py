#!/usr/bin/env python3
"""Self-test benchmark sweep review and promotion helpers."""

from __future__ import annotations

import copy
import argparse
import json
import sys
import tempfile
from pathlib import Path

import benchmark_sweep
from benchmark_schema import load_capture, validate_capture
from metadata_registry_constants import (
    GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_AND_BOUNDED_EXPORT_KERNELS_BATCHED_D2H,
    GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_AND_FINITE_EXPORT_KERNEL_BATCHED_D2H,
)
from test_benchmark_schema import as_host_api_batch_capture


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def set_phase(capture: dict, end_to_end: int) -> None:
    repeats = capture.get("repeats", 2)
    for phase in benchmark_sweep.PHASES:
        value = end_to_end if phase == "end_to_end" else int(capture["timing_summary_us"][phase]["median"])
        capture["raw_timings_us"][phase] = [value] * repeats
        capture["timing_summary_us"][phase] = {
            "avg": float(value),
            "median": float(value),
            "p95": float(value),
        }
        avg_field = {
            "pack": "avg_pack_us",
            "rns_gemm": "avg_rns_gemm_us",
            "crt_export": "avg_crt_export_us",
            "end_to_end": "avg_end_to_end_us",
        }[phase]
        capture[avg_field] = float(value)
    timings = capture.get("gpu_event_timings_us")
    summaries = capture.get("gpu_event_timing_summary_us")
    if isinstance(timings, dict) and isinstance(summaries, dict):
        for phase, values in list(timings.items()):
            current = summaries.get(phase, {})
            value = float(current.get("median", values[-1] if values else 0.0))
            timings[phase] = [value] * repeats
            summaries[phase] = {
                "avg": value,
                "median": value,
                "p95": value,
            }


def int32_accumulator_safety(capture: dict, cap: int = 65536) -> dict:
    finite = capture.get("semantics") in {"finite_ring_u8", "finite_field_u8"}
    return {
        "input_domain": "centered_i8_finite_u8_residues" if finite else "centered_i8_rns_residue_planes",
        "signedness": "signed_i8x_signed_i8",
        "accumulator_type": "int32",
        "modulus_policy": "finite_u8_modulus" if finite else "selected_rns_modulus_ladder",
        "modulus": capture.get("finite_modulus") if finite else 0,
        "uses_int32_inner_product": True,
        "k_block_size": min(capture["k"], cap),
        "k_block_cap": cap,
        "max_lhs_abs": 128,
        "max_rhs_abs": 128,
        "max_product": 128 * 128,
        "safe_for_k_block": True,
        "status": "safe_int32_k_block_split",
    }


def vector_accumulator_safety(capture: dict) -> dict:
    semantics = capture.get("semantics")
    return {
        "input_domain": "native_i64_values" if semantics == "bounded_i64" else "native_u64_values",
        "signedness": "signed_i64x_signed_i64" if semantics == "bounded_i64" else "unsigned_u64x_unsigned_u64",
        "accumulator_type": "software_192bit_limb",
        "modulus_policy": "native_exact_integer_output",
        "modulus": 0,
        "uses_int32_inner_product": False,
        "k_block_size": capture["k"],
        "k_block_cap": 0,
        "max_lhs_abs": 0,
        "max_rhs_abs": 0,
        "max_product": 0,
        "safe_for_k_block": True,
        "status": "exact_192bit_limb_no_int32_k_cap",
    }


def wrap64_candidate_accumulator_safety(capture: dict) -> dict:
    return {
        "input_domain": "compact_u8_byte_limb_pairs",
        "signedness": "unsigned_u8x_unsigned_u8",
        "accumulator_type": "int32_then_int64_diagonal",
        "modulus_policy": "mod_2_64_wraparound_byte_limb",
        "modulus": 0,
        "uses_int32_inner_product": True,
        "k_block_size": capture["k"],
        "k_block_cap": 32768,
        "max_lhs_abs": 255,
        "max_rhs_abs": 255,
        "max_product": 255 * 255,
        "safe_for_k_block": True,
        "status": "safe_int32_byte_limb_gemm36_k_block",
    }


def apply_accumulator_safety(capture: dict, safety: dict) -> None:
    capture["backend_metadata"]["accumulator_safety"] = safety
    capture["k_block_size"] = safety["k_block_size"]


def with_accumulator_key_fields(key: str, capture: dict) -> str:
    safety = capture["backend_metadata"]["accumulator_safety"]
    parts = [
        part
        for part in key.split(";")
        if part.split("=", 1)[0]
        not in {
            "target_id",
            "accumulator_type",
            "accumulator_signedness",
            "accumulator_modulus_policy",
            "k_block_size",
            "k_block_cap",
        }
    ]
    target = capture.get("device", {}).get("gcn_arch", "cpu")
    if capture.get("backend_selected") not in {
        "hip-direct",
        "hipblaslt",
        "ck",
        "rocwmma",
        "amdgpu-builtins",
        "hip-vector-alu-int64",
    }:
        target = "cpu"
    if target in {"", "none", "unknown"}:
        target = "cpu"
    insert_target_at = 1 if parts and parts[0].startswith("backend=") else 0
    parts = parts[:insert_target_at] + [f"target_id={target}"] + parts[insert_target_at:]
    insert_at = next((index for index, part in enumerate(parts) if part.startswith("kernel=")), len(parts))
    additions = [
        f"accumulator_type={safety['accumulator_type']}",
        f"accumulator_signedness={safety['signedness']}",
        f"accumulator_modulus_policy={safety['modulus_policy']}",
        f"k_block_size={safety['k_block_size']}",
        f"k_block_cap={safety['k_block_cap']}",
    ]
    return ";".join(parts[:insert_at] + additions + parts[insert_at:])


def finite_capture(backend: str, end_to_end: int) -> dict:
    capture = copy.deepcopy(load_capture(FIXTURE_DIR / "v4_finite_ring_u8_ck.json"))
    capture["_path"] = f"{backend}.json"
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    capture["warmups"] = benchmark_sweep.RELEASE_MIN_WARMUPS
    capture["repeats"] = benchmark_sweep.RELEASE_MIN_REPEATS
    metadata = capture["backend_metadata"]
    metadata["source"] = "rns8_get_plan_backend_info"
    if backend == "cpu-reference":
        capture["selected_kernel"] = "cpu_reference_finite_u8_gemm_v1"
        metadata["selected_kernel"] = "cpu_reference_finite_u8_gemm_v1"
        metadata["accelerator_backend"] = False
        metadata["matrix_engine_backend"] = False
        metadata["accelerator_library"] = None
        metadata["capability_status"] = "implemented_correctness_backend"
        metadata["epilogue_mode"] = "fused_centered_residue_then_canonical_u8_export"
        metadata["workspace_mode"] = "host_reference_workspace"
        metadata["workspace_required_bytes"] = 0
        metadata["isa_evidence"] = "not_applicable_cpu"
        apply_accumulator_safety(capture, int32_accumulator_safety(capture))
        metadata["autotune_key"] = metadata["autotune_key"].replace("backend=ck", "backend=cpu-reference")
        capture["device"] = {
            "device_id": -1,
            "name": "CPU reference",
            "gcn_arch": "none",
            "hip_available": 0,
            "hip_runtime_version": 0,
            "hip_driver_version": 0,
            "global_mem_bytes": 0,
        }
    elif backend == "hip-direct":
        capture["selected_kernel"] = "direct_hip_tiled_finite_u8_gemm_v1"
        metadata["selected_kernel"] = "direct_hip_tiled_finite_u8_gemm_v1"
        metadata["accelerator_backend"] = False
        metadata["matrix_engine_backend"] = False
        metadata["accelerator_library"] = "HIP runtime"
        metadata["capability_status"] = "implemented_correctness_backend"
        metadata["epilogue_mode"] = "fused_centered_residue_then_canonical_u8_export"
        metadata["workspace_mode"] = "resident_device_buffers"
        metadata["workspace_required_bytes"] = 0
        metadata["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
        apply_accumulator_safety(capture, int32_accumulator_safety(capture))
        metadata["autotune_key"] = metadata["autotune_key"].replace("backend=ck", "backend=hip-direct")
    metadata["autotune_key"] = with_accumulator_key_fields(metadata["autotune_key"], capture)
    set_phase(capture, end_to_end)
    return capture


def remove_gpu_events(capture: dict) -> None:
    timing = capture["timing_metadata"]
    timing["gpu_event_timing"] = False
    timing["gpu_event_timing_reason"] = "backend_event_capture_incomplete"
    timing["gpu_event_timing_status"] = "unavailable_missing_expected_events"
    timing["gpu_event_timing_source"] = None
    timing["gpu_event_timing_source_scope"] = None
    timing["gpu_event_timing_caveat"] = None
    timing["gpu_event_phase_order"] = None
    timing["gpu_event_timing_unavailable_reasons"] = [
        "rns_gemm missing backend HIP event label test_missing_event"
    ]
    capture["gpu_event_timings_us"] = None
    capture["gpu_event_timing_summary_us"] = None


def bounded_capture(backend: str, end_to_end: int) -> dict:
    capture = copy.deepcopy(load_capture(FIXTURE_DIR / "v4_bounded_i64_ck.json"))
    capture["_path"] = f"{backend}-bounded.json"
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    capture["warmups"] = benchmark_sweep.RELEASE_MIN_WARMUPS
    capture["repeats"] = benchmark_sweep.RELEASE_MIN_REPEATS
    metadata = capture["backend_metadata"]
    metadata["source"] = "rns8_get_plan_backend_info"
    if backend == "cpu-reference":
        capture["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
        metadata["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
        metadata["accelerator_backend"] = False
        metadata["matrix_engine_backend"] = False
        metadata["accelerator_library"] = None
        metadata["capability_status"] = "implemented_correctness_backend"
        metadata["epilogue_mode"] = "fused_centered_residue_then_crt_export"
        metadata["workspace_mode"] = "host_reference_workspace"
        metadata["workspace_required_bytes"] = 0
        metadata["isa_evidence"] = "not_applicable_cpu"
        apply_accumulator_safety(capture, int32_accumulator_safety(capture))
        metadata["autotune_key"] = metadata["autotune_key"].replace("backend=ck", "backend=cpu-reference").replace(
            "kernel=ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3",
            "kernel=cpu_reference_scalar_rns_gemm_v1",
        ).replace("epilogue=ck_fused_i32_to_centered_residue_then_crt_export", "epilogue=fused_centered_residue_then_crt_export")
        capture["device"] = {
            "device_id": -1,
            "name": "CPU reference",
            "gcn_arch": "none",
            "hip_available": 0,
            "hip_runtime_version": 0,
            "hip_driver_version": 0,
            "global_mem_bytes": 0,
        }
    elif backend == "hip-direct":
        capture["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
        metadata["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
        metadata["accelerator_backend"] = False
        metadata["matrix_engine_backend"] = False
        metadata["accelerator_library"] = "HIP runtime"
        metadata["capability_status"] = "implemented_correctness_backend"
        metadata["epilogue_mode"] = "fused_centered_residue_then_crt_export"
        metadata["workspace_mode"] = "resident_device_buffers"
        metadata["workspace_required_bytes"] = 0
        metadata["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
        apply_accumulator_safety(capture, int32_accumulator_safety(capture))
        metadata["autotune_key"] = metadata["autotune_key"].replace("backend=ck", "backend=hip-direct").replace(
            "kernel=ck_wmma_cshuffle_i8_i32_default_moduli_static_centered_epilogue_v3",
            "kernel=direct_hip_tiled_rns_gemm_v1",
        ).replace("epilogue=ck_fused_i32_to_centered_residue_then_crt_export", "epilogue=fused_centered_residue_then_crt_export")
    elif backend == "hip-vector-alu-int64":
        vector = copy.deepcopy(load_capture(FIXTURE_DIR / "v4_bounded_i64_vector_alu.json"))
        vector["_path"] = capture["_path"]
        vector["m"] = capture["m"]
        vector["n"] = capture["n"]
        vector["k"] = capture["k"]
        vector["bound"] = capture["bound"]
        vector["k_block_size"] = capture["k_block_size"]
        vector["warmups"] = benchmark_sweep.RELEASE_MIN_WARMUPS
        vector["repeats"] = benchmark_sweep.RELEASE_MIN_REPEATS
        vector["seed"] = capture["seed"]
        vector["git_commit"] = capture["git_commit"]
        vector["compiler"] = copy.deepcopy(capture["compiler"])
        vector["configured_amdgpu_targets"] = capture["configured_amdgpu_targets"]
        vector["hip_toolchain"] = copy.deepcopy(capture["hip_toolchain"])
        vector["device"] = copy.deepcopy(capture["device"])
        vector["checksum_u64"] = capture["checksum_u64"]
        apply_accumulator_safety(vector, vector_accumulator_safety(vector))
        vector["backend_metadata"]["autotune_key"] = (
            "backend=hip-vector-alu-int64;semantics=bounded_i64;m=64;n=128;k=64;prefix=9;"
            "tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
            "kernel=hip_vector_alu_i64_exact_192b_v1;epilogue=direct_int64_export"
        )
        vector["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
            vector["backend_metadata"]["autotune_key"], vector
        )
        capture = vector
    if backend != "hip-vector-alu-int64":
        metadata["autotune_key"] = with_accumulator_key_fields(metadata["autotune_key"], capture)
    set_phase(capture, end_to_end)
    return capture


def exact_wide_capture(backend: str, end_to_end: int) -> dict:
    capture = bounded_capture(backend, end_to_end)
    capture["_path"] = f"{backend}-exact-wide.json"
    capture["benchmark"] = "rns8_exact_wide_persistent_rns"
    capture["semantics"] = "exact_wide_signed"
    capture["bound_kind"] = "none"
    capture["bound_mode"] = "global"
    capture["bound"] = 0
    capture["prefix"] = 20
    capture["finite_modulus"] = None
    capture["tile_bounds_u64"] = None
    capture["epilogue_type"] = "exact_wide_signed_limb_export"
    capture["exact_wide_limb_count"] = benchmark_sweep.DEFAULT_EXACT_WIDE_LIMB_COUNT
    capture["input_distribution"] = "signed_uniform_-16_16"
    capture["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_correctness",
    ]
    capture["schedule_metadata"]["min_selected_prefix"] = 20
    capture["schedule_metadata"]["max_selected_prefix"] = 20
    capture["schedule_metadata"]["prefix_group_count"] = 1
    capture["schedule_metadata"]["adaptive_execution_applied"] = False
    metadata = capture["backend_metadata"]
    if backend == "ck":
        metadata["epilogue_mode"] = "ck_fused_i32_to_centered_residue_rns_output"
    elif backend == "hipblaslt":
        metadata["epilogue_mode"] = "separate_i32_scratch_reduce_rns_output"
    elif backend == "rocwmma":
        metadata["epilogue_mode"] = "rocwmma_fused_i32_to_centered_residue_rns_output"
    else:
        metadata["epilogue_mode"] = "fused_centered_residue_rns_output"
    metadata["autotune_key"] = with_accumulator_key_fields(
        (
        f"backend={backend};semantics=exact_wide_signed;m={capture['m']};n={capture['n']};k={capture['k']};"
        "prefix=20;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        f"kernel={capture['selected_kernel']};epilogue={metadata['epilogue_mode']}"
        ),
        capture,
    )
    return capture


def wrap64_capture(backend: str, end_to_end: int) -> dict:
    capture = copy.deepcopy(load_capture(FIXTURE_DIR / "v4_wrap64_hip.json"))
    capture["_path"] = f"{backend}-wrap64.json"
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    capture["warmups"] = benchmark_sweep.RELEASE_MIN_WARMUPS
    capture["repeats"] = benchmark_sweep.RELEASE_MIN_REPEATS
    metadata = capture["backend_metadata"]
    metadata["source"] = "rns8_get_plan_backend_info"
    metadata["accelerator_backend"] = False
    metadata["matrix_engine_backend"] = False
    metadata["performance_validated"] = False
    metadata["autotune_key"] = metadata["autotune_key"].replace("backend=hip-direct", f"backend={backend}")
    if backend == "wrap64-byte-limb":
        capture["selected_kernel"] = "cpu_wrap64_byte_limb_reference_v1"
        metadata["selected_kernel"] = "cpu_wrap64_byte_limb_reference_v1"
        metadata["selected_backend"] = "wrap64-byte-limb"
        metadata["accelerator_library"] = None
        metadata["capability_status"] = "implemented_cpu_wrap64_byte_limb_reference"
        metadata["workspace_mode"] = "host_byte_limb_reference_workspace"
        metadata["workspace_required_bytes"] = 0
        metadata["isa_evidence"] = "not_applicable_cpu"
        metadata["autotune_key"] = metadata["autotune_key"].replace(
            "kernel=direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4",
            "kernel=cpu_wrap64_byte_limb_reference_v1",
        )
        capture["device"] = {
            "device_id": -1,
            "name": "CPU wrap64 byte-limb reference",
            "gcn_arch": "none",
            "hip_available": 0,
            "hip_runtime_version": 0,
            "hip_driver_version": 0,
            "global_mem_bytes": 0,
        }
    elif backend == benchmark_sweep.WRAP64_ROCWMMA_CANDIDATE_BACKEND:
        capture["backend_requested"] = benchmark_sweep.WRAP64_ROCWMMA_CANDIDATE_BACKEND
        capture["backend_selected"] = "rocwmma"
        capture["selected_kernel"] = "rocwmma_wrap64_byte_gemm36_candidate_v0"
        capture["tile_m"] = 16
        capture["tile_n"] = 16
        capture["schedule_metadata"].update(
            {
                "source": "rns8_bench_wrap64_rocwmma_candidate_static_schedule",
                "tile_m": 16,
                "tile_n": 16,
                "tile_rows": 1,
                "tile_cols": 1,
                "tile_count": 1,
            }
        )
        metadata.update(
            {
                "source": "rns8_bench_wrap64_rocwmma_candidate",
                "selected_kernel": "rocwmma_wrap64_byte_gemm36_candidate_v0",
                "accelerator_backend": True,
                "correctness_backend": False,
                "matrix_engine_backend": True,
                "accelerator_library": "rocWMMA",
                "accelerator_version": "repo-local release/rocm-rel-7.1",
                "capability_status": "internal_wrap64_matrix_engine_candidate",
                "workspace_mode": "benchmark_owned_compact_byte_limb_device_buffers",
                "workspace_required_bytes": 640,
                "isa_evidence": "rocwmma_wrap64_byte_gemm36_matrix_isa_gate_no_divide",
                "autotune_key": (
                    "backend=rocwmma-wrap64-candidate;semantics=wrap_u64_mod_2_64;m=4;n=4;k=8;"
                    "prefix=0;tile_m=16;tile_n=16;groups=0;adaptive_prefix=0;adaptive_skip=0;"
                    "kernel=rocwmma_wrap64_byte_gemm36_candidate_v0;epilogue=low64_wrap_export"
                ),
            }
        )
        apply_accumulator_safety(capture, wrap64_candidate_accumulator_safety(capture))
        metadata["autotune_key"] = with_accumulator_key_fields(metadata["autotune_key"], capture)
        renamed = {
            "wrap64_byte_gemm36_tiled_2d_kernel": "wrap64_rocwmma_candidate_gemm36_kernel_group",
        }
        phase_order = capture["timing_metadata"].get("gpu_event_phase_order")
        if isinstance(phase_order, list):
            capture["timing_metadata"]["gpu_event_phase_order"] = [renamed.get(item, item) for item in phase_order]
        for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
            values = capture.get(field)
            if isinstance(values, dict):
                for old, new in renamed.items():
                    if old in values:
                        values[new] = values.pop(old)
        capture["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_internal_rocwmma_wrap64_candidate_hooks"
        capture["timing_metadata"]["gpu_event_timing_source_scope"] = (
            "rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups"
        )
        capture["timing_metadata"]["phase_availability"]["scheduling"] = {
            "timed": True,
            "timing_key": "scheduling",
            "scope": "benchmark_static_wrap64_rocwmma_candidate_schedule",
            "reason": "measured with host steady_clock around fixed 16x16 candidate schedule metadata initialization",
        }
    set_phase(capture, end_to_end)
    return capture


def mark_reused_pack(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    reused["reuse_packed_inputs"] = True
    reused["pack_mode"] = "prepacked_reuse"
    reused["prepack_reuse_operands"] = ["A", "B"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 11
    reused["avg_prepack_setup_us"] = 11.0
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A", "B"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    return reused


def mark_reused_a_pack(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    reused["reuse_packed_inputs"] = True
    reused["pack_mode"] = "prepacked_reuse_a"
    reused["prepack_reuse_operands"] = ["A"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 11
    reused["avg_prepack_setup_us"] = 11.0
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse_a"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    return reused


