#!/usr/bin/env python3
"""Self-test benchmark sweep review and promotion helpers."""

from __future__ import annotations

import copy
import argparse
import json
import tempfile
from pathlib import Path

import benchmark_sweep
from benchmark_schema import load_capture, validate_capture


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
    if capture.get("backend_selected") not in {"hip-direct", "hipblaslt", "ck", "rocwmma", "hip-vector-alu-int64"}:
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
            "kernel=ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
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
            "kernel=ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
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
                "isa_evidence": "rocwmma_wrap64_byte_gemm36_wmma_isa_gate_no_int32_global_store_no_divide",
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


def main() -> int:
    parsed = benchmark_sweep.parse_case("rect:64,128,256")
    assert parsed.name == "rect"
    assert (parsed.m, parsed.n, parsed.k) == (64, 128, 256)
    adaptive = benchmark_sweep.parse_case("adaptive:65,65,64,64,64", adaptive=True)
    assert adaptive.bound_mode == "per-tile"
    assert adaptive.input_profile == "uniform-small"
    assert adaptive.require_adaptive is True
    adaptive_profile = benchmark_sweep.parse_case("adaptive-bands:256,256,512,64,64,adaptive-bands", adaptive=True)
    assert adaptive_profile.input_profile == "adaptive-bands"
    assert benchmark_sweep.backend_allowed_for("wrap-u64", parsed, "wrap64-byte-limb") is True
    assert benchmark_sweep.backend_allowed_for("bounded-i64", parsed, "wrap64-byte-limb") is False
    assert benchmark_sweep.backend_allowed_for("finite-u8-ring", parsed, "hip-vector-alu-int64") is False
    assert benchmark_sweep.backend_allowed_for("exact-wide-signed", parsed, "ck") is True
    assert benchmark_sweep.backend_allowed_for("exact-wide-unsigned", parsed, "hip-vector-alu-int64") is False
    assert benchmark_sweep.backend_allowed_for("exact-wide-signed", adaptive, "ck") is False
    assert benchmark_sweep.backend_allowed_for("bounded-u64", adaptive, "hipblaslt") is False
    assert benchmark_sweep.cli_backend("rocwmma") == "rocwmma"
    assert benchmark_sweep.cli_backend("hip-vector-alu-int64") == "hip-vector-alu-int64-runtime"
    assert benchmark_sweep.cli_backend("hip-direct") == "hip-direct"

    catalog = benchmark_sweep.scenario_catalog()
    for scenario_name in [
        "generated-prefix-reducers",
        "large-release-validation",
        "layout-search",
        "multi-modulus-pack",
        "residue-channel-fusion",
        "fused-pack-gemm-small",
    ]:
        assert scenario_name in catalog
        assert catalog[scenario_name]
    fusion_item = catalog["residue-channel-fusion"][0]
    assert fusion_item.residue_channel_fusion is True
    assert fusion_item.next_op_hint == "final-export"
    scenario_base_args = argparse.Namespace(
        warmups=1,
        repeats=2,
        seed=11,
        reuse_packed_inputs=False,
        reuse_packed_a=False,
        reuse_packed_b=False,
        residue_chain_length=1,
        output_ld_padding=0,
        prefix_policy=None,
        max_prefix=None,
        bound_source=None,
        next_op_hint=None,
        residue_channel_fusion=False,
    )
    fusion_args = benchmark_sweep.scenario_args_for_item(scenario_base_args, fusion_item)
    fusion_command = benchmark_sweep.command_for(
        Path("rns8-bench"),
        "hip-direct",
        fusion_item.semantics,
        fusion_item.case,
        None,
        None,
        fusion_args,
    )
    assert "--residue-channel-fusion" in fusion_command
    assert "--next-op-hint" in fusion_command and "final-export" in fusion_command
    assert "--prefix-policy" in fusion_command and "fixed-requested" in fusion_command
    assert "--max-prefix" in fusion_command and "9" in fusion_command

    wrap64_args = argparse.Namespace(
        bench=Path("rns8-bench"),
        bench_for=[],
        out_root=Path("temp") / "wrap64-release",
        backends=None,
        semantics=["wrap-u64"],
        case=None,
        adaptive_case=None,
        shapes=None,
        modulus=None,
        exact_wide_limbs=None,
        include_exact_wide_limb_variants=False,
        residue_chain_length=1,
        include_default_adaptive=False,
        include_adaptive_workloads=False,
        adaptive_only=False,
        include_wrap64=False,
        include_wrap64_rocwmma_candidate=False,
        include_exact_wide=False,
        reuse_packed_inputs=False,
        reuse_packed_a=False,
        reuse_packed_b=False,
        release_matrix=True,
        include_exploratory_large=False,
        review_mode="release",
        warmups=benchmark_sweep.RELEASE_MIN_WARMUPS,
        repeats=benchmark_sweep.RELEASE_MIN_REPEATS,
        seed=20260602,
        write_autotune_cache=False,
        autotune_cache=None,
    )
    wrap64_commands = benchmark_sweep.sweep_commands(wrap64_args)
    assert len(wrap64_commands) == len(benchmark_sweep.PROMOTABLE_RELEASE_SHAPES) * 2
    assert wrap64_commands[0][0] == "wrap-u64-wrap64-64-64x64x64-wrap64-byte-limb.json"
    assert wrap64_commands[1][0] == "wrap-u64-wrap64-64-64x64x64-hip-direct.json"
    assert all("--semantics" in command and "wrap-u64" in command for _name, command, _output in wrap64_commands)
    wrap64_args.include_wrap64_rocwmma_candidate = True
    candidate_commands = benchmark_sweep.sweep_commands(wrap64_args)
    assert len(candidate_commands) == len(benchmark_sweep.PROMOTABLE_RELEASE_SHAPES) * 3
    candidate_name, candidate_command, _candidate_output = candidate_commands[2]
    assert candidate_name == "wrap-u64-wrap64-64-64x64x64-rocwmma-wrap64-candidate.json"
    assert "--backend" in candidate_command and "rocwmma-wrap64-candidate" in candidate_command
    assert "--tile-m" in candidate_command and "16" in candidate_command
    wrap64_args.include_wrap64_rocwmma_candidate = False
    wrap64_args.reuse_packed_inputs = True
    reuse_commands = benchmark_sweep.sweep_commands(wrap64_args)
    assert reuse_commands[0][0] == "wrap-u64-wrap64-64-64x64x64-reuse-packed-wrap64-byte-limb.json"
    assert all("--reuse-packed-inputs" in command for _name, command, _output in reuse_commands)
    wrap64_args.reuse_packed_inputs = False
    wrap64_args.reuse_packed_a = True
    reuse_a_commands = benchmark_sweep.sweep_commands(wrap64_args)
    assert reuse_a_commands[0][0] == "wrap-u64-wrap64-64-64x64x64-reuse-packed-a-wrap64-byte-limb.json"
    assert all("--reuse-packed-a" in command for _name, command, _output in reuse_a_commands)
    wrap64_args.reuse_packed_a = False
    wrap64_args.reuse_packed_b = True
    reuse_b_commands = benchmark_sweep.sweep_commands(wrap64_args)
    assert reuse_b_commands[0][0] == "wrap-u64-wrap64-64-64x64x64-reuse-packed-b-wrap64-byte-limb.json"
    assert all("--reuse-packed-b" in command for _name, command, _output in reuse_b_commands)
    wrap64_args.reuse_packed_b = False
    wrap64_args.adaptive_only = True
    try:
        benchmark_sweep.sweep_commands(wrap64_args)
    except SystemExit as exc:
        assert "--adaptive-only requires" in str(exc)
    else:
        raise AssertionError("adaptive-only without adaptive cases should fail even when wrap64 is requested")
    wrap64_args.adaptive_only = False

    exact_args = argparse.Namespace(
        bench=Path("rns8-bench"),
        bench_for=[],
        out_root=Path("temp") / "exact-wide",
        backends=["cpu", "hip-direct"],
        semantics=["exact_wide_signed"],
        case=["small:16,16,16"],
        adaptive_case=None,
        shapes=None,
        modulus=None,
        exact_wide_limbs=None,
        include_exact_wide_limb_variants=False,
        residue_chain_length=1,
        include_default_adaptive=False,
        include_adaptive_workloads=False,
        adaptive_only=False,
        include_wrap64=False,
        include_wrap64_rocwmma_candidate=False,
        include_exact_wide=False,
        reuse_packed_inputs=False,
        reuse_packed_a=False,
        reuse_packed_b=False,
        release_matrix=False,
        include_exploratory_large=False,
        review_mode="smoke",
        warmups=1,
        repeats=2,
        seed=7,
        write_autotune_cache=False,
        autotune_cache=None,
    )
    exact_commands = benchmark_sweep.sweep_commands(exact_args)
    assert [name for name, _command, _output in exact_commands] == [
        "exact-wide-signed-small-16x16x16-cpu.json",
        "exact-wide-signed-small-16x16x16-hip-direct.json",
    ]
    assert all("--semantics" in command and "exact-wide-signed" in command for _name, command, _output in exact_commands)
    assert all("--exact-wide-limbs" in command and "4" in command for _name, command, _output in exact_commands)

    exact_variant_args = copy.copy(exact_args)
    exact_variant_args.backends = ["cpu"]
    exact_variant_args.include_exact_wide_limb_variants = True
    exact_variant_commands = benchmark_sweep.sweep_commands(exact_variant_args)
    assert len(exact_variant_commands) == len(benchmark_sweep.EXACT_WIDE_LIMB_VARIANTS)
    assert [name for name, _command, _output in exact_variant_commands] == [
        "exact-wide-signed-small-16x16x16-limbs1-cpu.json",
        "exact-wide-signed-small-16x16x16-limbs2-cpu.json",
        "exact-wide-signed-small-16x16x16-limbs3-cpu.json",
        "exact-wide-signed-small-16x16x16-cpu.json",
        "exact-wide-signed-small-16x16x16-limbs8-cpu.json",
        "exact-wide-signed-small-16x16x16-limbs16-cpu.json",
        "exact-wide-signed-small-16x16x16-limbs32-cpu.json",
    ]
    assert exact_variant_commands[0][1][exact_variant_commands[0][1].index("--exact-wide-limbs") + 1] == "1"
    assert exact_variant_commands[-1][1][exact_variant_commands[-1][1].index("--exact-wide-limbs") + 1] == "32"

    scenario_args = copy.copy(exact_args)
    scenario_args.out_root = Path("temp") / "scenario"
    scenario_args.backends = ["hip-direct"]
    scenario_args.semantics = None
    scenario_args.case = None
    scenario_args.scenario = ["repeated-b"]
    scenario_entries = benchmark_sweep.sweep_command_entries(scenario_args)
    assert len(scenario_entries) == 2
    assert [entry.scenario["name"] for entry in scenario_entries] == ["bounded-i64-512", "bounded-i64-1024"]
    assert all(entry.scenario["family"] == "repeated-b" for entry in scenario_entries)
    assert all(entry.scenario["pack_mode"] == "prepacked_reuse_b" for entry in scenario_entries)
    assert all("--reuse-packed-b" in entry.command for entry in scenario_entries)
    assert all("--prefix-policy" in entry.command and "fixed-requested" in entry.command for entry in scenario_entries)
    assert all("--max-prefix" in entry.command and "9" in entry.command for entry in scenario_entries)
    assert all("scenarios" in entry.output.parts and "repeated-b" in entry.output.parts for entry in scenario_entries)
    assert scenario_entries[0].name.startswith("repeated-b-bounded-i64-512-")

    skinny_args = copy.copy(scenario_args)
    skinny_args.backends = ["hip-vector-alu-int64"]
    skinny_args.scenario = ["skinny-gemv"]
    skinny_entries = benchmark_sweep.sweep_command_entries(skinny_args)
    assert len(skinny_entries) == 3
    assert [entry.scenario["name"] for entry in skinny_entries] == [
        "bounded-i64-n1-512",
        "bounded-u64-n1-1024",
        "bounded-i64-n1-longk-256",
    ]
    assert all(entry.scenario["family"] == "skinny-gemv" for entry in skinny_entries)
    assert all(entry.scenario["backend"] == "hip-vector-alu-int64" for entry in skinny_entries)
    assert all(entry.scenario["shape"]["n"] == 1 for entry in skinny_entries)
    assert all("--backend" in entry.command for entry in skinny_entries)
    assert all(benchmark_sweep.cli_backend("hip-vector-alu-int64") in entry.command for entry in skinny_entries)
    assert all("scenarios" in entry.output.parts and "skinny-gemv" in entry.output.parts for entry in skinny_entries)
    assert skinny_entries[0].name.startswith("skinny-gemv-bounded-i64-n1-512-")
    assert skinny_entries[2].scenario.get("metadata", {}).get("workflow_name") == "gemv_n1_long_k"
    assert skinny_entries[2].command[skinny_entries[2].command.index("--k") + 1] == "4096"

    many_small_args = copy.copy(scenario_args)
    many_small_args.backends = ["hip-direct"]
    many_small_args.scenario = ["many-small"]
    many_small_entries = benchmark_sweep.sweep_command_entries(many_small_args)
    assert len(many_small_entries) == 8
    assert all(entry.scenario["family"] == "many-small" for entry in many_small_entries)
    assert {entry.scenario["name"] for entry in many_small_entries} == {
        "bounded-i64-32-proxy",
        "bounded-i64-32-oneshot-proxy",
        "bounded-i64-128-proxy",
        "bounded-u64-64-proxy",
        "bounded-u64-skinny-n1-proxy",
        "exact-wide-signed-64-proxy",
        "finite-ring-64-proxy",
    }
    assert any("--oneshot" in entry.command for entry in many_small_entries)
    assert sorted(
        entry.scenario["modulus"] for entry in many_small_entries if entry.scenario["semantics"] == "finite-u8-ring"
    ) == [251, 255]
    assert all(entry.scenario.get("metadata", {}).get("evidence_role") == "proxy_single_shape_repeat" for entry in many_small_entries)
    assert any(entry.scenario.get("metadata", {}).get("grouping_role") == "pre_grouped_baseline" for entry in many_small_entries)
    assert any(
        entry.scenario.get("metadata", {}).get("bridge_role") == "native_vector_to_rns_candidate"
        for entry in many_small_entries
    )
    assert any(
        entry.scenario["shape"]["n"] == 1 and entry.scenario.get("metadata", {}).get("phase_label") == "pre_grouped_skinny_proxy"
        for entry in many_small_entries
    )
    assert any(entry.scenario.get("exact_wide_limb_count") == 4 for entry in many_small_entries)

    rns_chain_args = copy.copy(scenario_args)
    rns_chain_args.backends = ["hip-direct"]
    rns_chain_args.scenario = ["rns-chain"]
    rns_chain_entries = benchmark_sweep.sweep_command_entries(rns_chain_args)
    assert [entry.scenario["name"] for entry in rns_chain_entries] == [
        "bounded-i64-chain3",
        "bounded-i64-chain3-reuse-b",
        "bounded-i64-chain4-256",
        "bounded-i64-chain4-256-reuse-b",
        "bounded-u64-chain3-256",
        "bounded-u64-chain3-256-reuse-b",
        "exact-wide-signed-chain3",
        "exact-wide-signed-chain3-reuse-b",
        "exact-wide-unsigned-chain3-256",
        "exact-wide-unsigned-chain3-256-reuse-b",
    ]
    assert all(entry.scenario["output_domain"] == "residue_current_rns" for entry in rns_chain_entries)
    assert all(entry.scenario.get("metadata", {}).get("output_domain_requirement") == "lazy_export" for entry in rns_chain_entries)
    assert any("--residue-chain-length" in entry.command and "4" in entry.command for entry in rns_chain_entries)
    assert any(entry.scenario.get("metadata", {}).get("chain_depth") == 4 for entry in rns_chain_entries)
    assert {entry.scenario["pack_mode"] for entry in rns_chain_entries} == {
        "per_repeat_repack",
        "prepacked_reuse_b",
    }
    assert any("--reuse-packed-b" in entry.command for entry in rns_chain_entries)
    assert any(
        entry.scenario.get("metadata", {}).get("reuse_contract") == "stable_chain_rhs_prepacked_before_warmups"
        for entry in rns_chain_entries
    )

    adaptive_bands_args = copy.copy(scenario_args)
    adaptive_bands_args.backends = None
    adaptive_bands_args.scenario = ["adaptive-bands"]
    adaptive_bands_entries = benchmark_sweep.sweep_command_entries(adaptive_bands_args)
    assert len(adaptive_bands_entries) == 15
    assert {entry.scenario["family"] for entry in adaptive_bands_entries} == {"adaptive-bands"}
    assert {
        entry.scenario["backend"]
        for entry in adaptive_bands_entries
        if entry.scenario["name"] == "bounded-i64-256"
    } == {"cpu", "hip-direct", "hip-vector-alu-int64", "ck", "rocwmma"}
    assert all("--bound-mode" in entry.command and "per-tile" in entry.command for entry in adaptive_bands_entries)
    assert all("--require-adaptive-execution" in entry.command for entry in adaptive_bands_entries)

    large_args = copy.copy(scenario_args)
    large_args.backends = ["hip-direct"]
    large_args.scenario = ["large-exploratory"]
    large_entries = benchmark_sweep.sweep_command_entries(large_args)
    assert len(large_entries) == 22
    assert {entry.scenario["name"] for entry in large_entries} == {
        "bounded-i64-2048",
        "bounded-i64-2048-reuse-b",
        "bounded-u64-2048",
        "bounded-u64-2048-reuse-b",
        "bounded-i64-4096",
        "bounded-i64-4096-reuse-b",
        "bounded-u64-4096",
        "bounded-u64-4096-reuse-b",
        "exact-wide-signed-2048",
        "exact-wide-unsigned-2048",
        "exact-wide-signed-4096",
        "exact-wide-unsigned-4096",
        "finite-ring-2048",
        "finite-field-2048",
        "finite-ring-4096",
        "finite-field-4096",
        "wrap64-2048",
        "wrap64-4096",
    }
    assert any(entry.scenario["shape"]["m"] == 4096 for entry in large_entries)
    assert sorted(
        entry.scenario["modulus"] for entry in large_entries if entry.scenario["name"] == "finite-ring-2048"
    ) == [251, 255, 256]
    assert all(
        entry.scenario["exact_wide_limb_count"] == 4
        for entry in large_entries
        if entry.scenario["semantics"].startswith("exact-wide")
    )
    assert any(
        entry.scenario.get("metadata", {}).get("large_shape_role") == "wrap64_direct_hip_throughput_probe"
        for entry in large_entries
    )
    assert any(
        entry.scenario.get("metadata", {}).get("reuse_contract") == "large_stable_rhs_prepacked_before_warmups"
        for entry in large_entries
    )

    layout_args = copy.copy(scenario_args)
    layout_args.backends = ["hip-direct"]
    layout_args.scenario = ["layout-search"]
    layout_entries = benchmark_sweep.sweep_command_entries(layout_args)
    assert len(layout_entries) == 9
    assert {entry.scenario["family"] for entry in layout_entries} == {"layout-search"}
    assert {entry.scenario["name"] for entry in layout_entries} == {
        "bounded-i64-prefix9-final-export",
        "bounded-i64-prefix9-rns-next",
        "exact-wide-signed-prefix20-final-export",
        "exact-wide-signed-prefix20-rns-next",
        "finite-ring-hot-modulus-layout",
        "finite-field-hot-prime-layout",
        "wrap64-direct-byte-layout",
    }
    assert sorted(
        entry.scenario["modulus"]
        for entry in layout_entries
        if entry.scenario["name"] == "finite-ring-hot-modulus-layout"
    ) == [251, 255, 256]
    assert any("--next-op-hint" in entry.command and "rns-gemm" in entry.command for entry in layout_entries)
    assert any("--prefix-policy" in entry.command and "fixed-requested" in entry.command for entry in layout_entries)
    assert any("--max-prefix" in entry.command and "20" in entry.command for entry in layout_entries)
    assert any(entry.scenario["semantics"] == "wrap-u64" for entry in layout_entries)
    assert any(
        entry.scenario.get("metadata", {}).get("layout_role") == "exact_wide_prefix20_next_rns_gemm"
        for entry in layout_entries
    )
    assert all(
        entry.scenario.get("metadata", {}).get("workflow_name") == "end_to_end_layout_search"
        for entry in layout_entries
    )

    large_bounded_args = copy.copy(scenario_args)
    large_bounded_args.backends = ["hip-direct"]
    large_bounded_args.scenario = ["large-exploratory"]
    large_bounded_args.semantics = ["bounded-i64", "bounded-u64"]
    large_bounded_entries = benchmark_sweep.sweep_command_entries(large_bounded_args)
    assert [entry.scenario["name"] for entry in large_bounded_entries] == [
        "bounded-i64-2048",
        "bounded-i64-2048-reuse-b",
        "bounded-u64-2048",
        "bounded-u64-2048-reuse-b",
        "bounded-i64-4096",
        "bounded-i64-4096-reuse-b",
        "bounded-u64-4096",
        "bounded-u64-4096-reuse-b",
    ]
    assert {entry.scenario["semantics"] for entry in large_bounded_entries} == {"bounded-i64", "bounded-u64"}
    assert {entry.scenario["pack_mode"] for entry in large_bounded_entries} == {
        "per_repeat_repack",
        "prepacked_reuse_b",
    }
    assert any("--reuse-packed-b" in entry.command for entry in large_bounded_entries)

    large_validation_args = copy.copy(scenario_args)
    large_validation_args.backends = None
    large_validation_args.scenario = ["large-release-validation"]
    large_validation_entries = benchmark_sweep.sweep_command_entries(large_validation_args)
    assert len(large_validation_entries) == 56
    assert {entry.scenario["family"] for entry in large_validation_entries} == {"large-release-validation"}
    assert {entry.scenario["name"] for entry in large_validation_entries} == {
        "bounded-i64-2048-required-baselines",
        "bounded-u64-2048-required-baselines",
        "bounded-i64-2048-reuse-b-required-baselines",
        "bounded-u64-2048-reuse-b-required-baselines",
        "exact-wide-signed-2048-required-baselines",
        "exact-wide-unsigned-2048-required-baselines",
        "finite-ring-2048-hot-required-baselines",
        "finite-field-2048-hot-required-baselines",
        "wrap64-2048-required-baselines",
    }
    assert {
        entry.scenario["backend"]
        for entry in large_validation_entries
        if entry.scenario["name"] == "bounded-i64-2048-required-baselines"
    } == {"cpu", "hip-direct", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma"}
    assert {
        entry.scenario["backend"]
        for entry in large_validation_entries
        if entry.scenario["name"] == "wrap64-2048-required-baselines"
    } == {"wrap64-byte-limb", "hip-direct"}
    assert sorted(
        entry.scenario["modulus"]
        for entry in large_validation_entries
        if entry.scenario["name"] == "finite-ring-2048-hot-required-baselines"
        and entry.scenario["backend"] == "cpu"
    ) == [251, 255, 256]
    assert all(
        entry.scenario.get("metadata", {}).get("workflow_name") == "large_shape_release_validation"
        for entry in large_validation_entries
    )
    assert any("--reuse-packed-b" in entry.command for entry in large_validation_entries)
    assert any(
        entry.scenario.get("metadata", {}).get("validation_contract")
        == "same_contract_cpu_direct_vector_accelerator_release_review"
        for entry in large_validation_entries
    )

    finite_generic_args = copy.copy(scenario_args)
    finite_generic_args.backends = ["hip-direct", "ck", "rocwmma"]
    finite_generic_args.scenario = ["finite-generic-moduli"]
    finite_generic_entries = benchmark_sweep.sweep_command_entries(finite_generic_args)
    assert len(finite_generic_entries) == 5
    assert {entry.scenario["backend"] for entry in finite_generic_entries} == {"hip-direct"}
    assert {entry.scenario["modulus"] for entry in finite_generic_entries} == {127, 253}
    assert {entry.scenario["name"] for entry in finite_generic_entries} == {
        "ring-prime-127-512",
        "field-prime-127-512",
        "ring-composite-253-512",
        "ring-prime-127-2048",
        "ring-composite-253-2048",
    }
    assert all(
        entry.scenario.get("metadata", {}).get("workflow_name") == "finite_u8_generic_modulus"
        for entry in finite_generic_entries
    )
    assert any(
        entry.scenario.get("metadata", {}).get("prime_or_composite") == "composite"
        for entry in finite_generic_entries
    )
    assert any(
        entry.scenario["shape"]["m"] == 2048
        and entry.scenario.get("metadata", {}).get("large_shape_role") == "finite_generic_modulus_probe"
        for entry in finite_generic_entries
    )

    bridge_args = copy.copy(scenario_args)
    bridge_args.backends = None
    bridge_args.bench_for = ["hip-direct=hip-direct-release-bench"]
    bridge_args.scenario = ["native-to-rns-bridge"]
    bridge_entries = benchmark_sweep.sweep_command_entries(bridge_args)
    assert len(bridge_entries) == 4
    assert {entry.scenario["name"] for entry in bridge_entries} == {
        "bounded-i64-64",
        "bounded-u64-64",
        "bounded-i64-128",
        "bounded-u64-128",
    }
    assert {entry.scenario["backend"] for entry in bridge_entries} == {"auto"}
    assert all(entry.command[0] == "hip-direct-release-bench" for entry in bridge_entries)
    assert all(entry.command[entry.command.index("--backend") + 1] == "auto" for entry in bridge_entries)
    assert all("--native-to-rns-bridge" in entry.command for entry in bridge_entries)
    assert all("--reuse-packed-inputs" not in entry.command for entry in bridge_entries)
    assert all(entry.scenario["native_to_rns_bridge"] is True for entry in bridge_entries)
    assert all(
        entry.scenario.get("metadata", {}).get("workflow_name") == "native_to_rns_bridge"
        for entry in bridge_entries
    )
    assert any(
        entry.scenario.get("metadata", {}).get("conversion_event_required") == "native_u64_to_rns_kernel"
        for entry in bridge_entries
    )

    vector_chain_args = copy.copy(scenario_args)
    vector_chain_args.backends = None
    vector_chain_args.bench_for = ["hip-direct=hip-direct-release-bench"]
    vector_chain_args.scenario = ["vector-to-rns-chain"]
    vector_chain_entries = benchmark_sweep.sweep_command_entries(vector_chain_args)
    assert len(vector_chain_entries) == 8
    assert {entry.scenario["name"] for entry in vector_chain_entries} == {
        "bounded-i64-64",
        "bounded-u64-64",
        "bounded-i64-128",
        "bounded-u64-128",
        "bounded-i64-64-reuse-consumer-b",
        "bounded-u64-64-reuse-consumer-b",
        "bounded-i64-128-reuse-consumer-b",
        "bounded-u64-128-reuse-consumer-b",
    }
    assert {entry.scenario["backend"] for entry in vector_chain_entries} == {"auto"}
    assert all(entry.command[0] == "hip-direct-release-bench" for entry in vector_chain_entries)
    assert all(entry.command[entry.command.index("--backend") + 1] == "auto" for entry in vector_chain_entries)
    assert all("--vector-to-rns-chain" in entry.command for entry in vector_chain_entries)
    assert all("--native-to-rns-bridge" not in entry.command for entry in vector_chain_entries)
    assert all(entry.scenario["native_to_rns_bridge"] is False for entry in vector_chain_entries)
    assert all(entry.scenario["vector_to_rns_chain"] is True for entry in vector_chain_entries)
    assert {
        entry.scenario["pack_mode"]
        for entry in vector_chain_entries
    } == {"per_repeat_repack", "prepacked_reuse_b"}
    assert any("--reuse-packed-b" in entry.command for entry in vector_chain_entries)
    assert any(
        entry.scenario.get("metadata", {}).get("reuse_contract") == "consumer_b_prepacked_before_warmups"
        for entry in vector_chain_entries
    )
    assert all(
        entry.scenario.get("metadata", {}).get("workflow_name") == "vector_to_rns_chain"
        for entry in vector_chain_entries
    )
    assert any(
        entry.scenario.get("metadata", {}).get("conversion_event_required") == "native_u64_to_rns_kernel"
        for entry in vector_chain_entries
    )

    algebra_args = copy.copy(scenario_args)
    algebra_args.backends = ["ck"]
    algebra_args.scenario = ["computational-algebra-proxies"]
    algebra_entries = benchmark_sweep.sweep_command_entries(algebra_args)
    assert len(algebra_entries) == 5
    assert all(entry.scenario["family"] == "computational-algebra-proxies" for entry in algebra_entries)
    assert {entry.scenario.get("metadata", {}).get("source_role") for entry in algebra_entries} == {
        "computational_algebra_proxy"
    }
    assert any(entry.scenario.get("metadata", {}).get("workflow_name") == "F4" for entry in algebra_entries)
    assert any(entry.scenario["semantics"] == "exact-wide-signed" for entry in algebra_entries)

    fhe_args = copy.copy(scenario_args)
    fhe_args.backends = ["hip-direct"]
    fhe_args.scenario = ["fhe-lattice-proxies"]
    fhe_entries = benchmark_sweep.sweep_command_entries(fhe_args)
    assert len(fhe_entries) == 4
    assert all(entry.scenario["family"] == "fhe-lattice-proxies" for entry in fhe_entries)
    assert {entry.scenario.get("metadata", {}).get("source_role") for entry in fhe_entries} == {
        "fhe_lattice_proxy"
    }
    assert any(entry.scenario.get("metadata", {}).get("workflow_name") == "ntt_intt_pressure" for entry in fhe_entries)
    assert any("--reuse-packed-b" in entry.command for entry in fhe_entries)
    assert any("--residue-chain-length" in entry.command and "4" in entry.command for entry in fhe_entries)

    with tempfile.TemporaryDirectory() as tmp:
        manifest_paths = benchmark_sweep.write_scenario_manifest(scenario_entries, scenario_args, Path(tmp))
        assert manifest_paths is not None
        manifest = json.loads(Path(manifest_paths["scenario_manifest"]).read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert manifest["scenario_families"] == ["repeated-b"]
        assert manifest["capture_count"] == 2
        assert manifest["entries"][0]["output_domain"] == "host_export"
        assert Path(manifest_paths["scenario_markdown"]).exists()

    bad_scenario_args = copy.copy(scenario_args)
    bad_scenario_args.include_oneshot = True
    try:
        benchmark_sweep.sweep_commands(bad_scenario_args)
    except SystemExit as exc:
        assert "--scenario cannot be combined" in str(exc)
    else:
        raise AssertionError("scenario mode should reject manual include flags")

    exact_chain_args = copy.copy(exact_args)
    exact_chain_args.backends = ["cpu"]
    exact_chain_args.residue_chain_length = 3
    exact_chain_commands = benchmark_sweep.sweep_commands(exact_chain_args)
    assert [name for name, _command, _output in exact_chain_commands] == [
        "exact-wide-signed-small-16x16x16-chain3-cpu.json",
    ]
    exact_chain_command = exact_chain_commands[0][1]
    assert "--residue-chain-length" in exact_chain_command
    assert exact_chain_command[exact_chain_command.index("--residue-chain-length") + 1] == "3"

    vector_args = copy.copy(exact_args)
    vector_args.out_root = Path("temp") / "vector-runtime"
    vector_args.backends = ["hip-vector-alu-int64"]
    vector_args.semantics = ["bounded-i64"]
    vector_args.case = ["small:16,16,16"]
    vector_commands = benchmark_sweep.sweep_commands(vector_args)
    vector_name, vector_command, _vector_output = vector_commands[0]
    assert vector_name == "bounded-i64-small-16x16x16-hip-vector-alu-int64.json"
    assert vector_command[vector_command.index("--backend") + 1] == "hip-vector-alu-int64-runtime"

    padded_output_args = copy.copy(vector_args)
    padded_output_args.backends = ["hip-direct"]
    padded_output_args.output_ld_padding = 7
    padded_output_commands = benchmark_sweep.sweep_commands(padded_output_args)
    padded_name, padded_command, _padded_output = padded_output_commands[0]
    assert padded_name == "bounded-i64-small-16x16x16-outpad7-hip-direct.json"
    assert "--output-ld-padding" in padded_command
    assert padded_command[padded_command.index("--output-ld-padding") + 1] == "7"

    input_scan_args = copy.copy(vector_args)
    input_scan_args.bound_source = "input-scan"
    input_scan_commands = benchmark_sweep.sweep_commands(input_scan_args)
    input_scan_command = input_scan_commands[0][1]
    assert "--bound-source" in input_scan_command
    assert input_scan_command[input_scan_command.index("--bound-source") + 1] == "input-scan"

    exact_input_scan_args = copy.copy(exact_args)
    exact_input_scan_args.bound_source = "input-scan"
    try:
        benchmark_sweep.sweep_commands(exact_input_scan_args)
    except SystemExit as exc:
        assert "--bound-source input-scan is only valid for bounded RNS sweeps" in str(exc)
    else:
        raise AssertionError("input-scan bound discovery should reject non-bounded sweeps")

    oneshot_args = copy.copy(exact_args)
    oneshot_args.out_root = Path("temp") / "oneshot"
    oneshot_args.backends = ["hip-direct"]
    oneshot_args.semantics = ["bounded-i64"]
    oneshot_args.case = ["small:16,16,16"]
    oneshot_args.include_oneshot = True
    oneshot_args.oneshot_only = False
    oneshot_commands = benchmark_sweep.sweep_commands(oneshot_args)
    assert [name for name, _command, _output in oneshot_commands] == [
        "bounded-i64-small-16x16x16-hip-direct.json",
        "bounded-i64-small-16x16x16-oneshot-hip-direct.json",
    ]
    assert "--oneshot" not in oneshot_commands[0][1]
    assert "--oneshot" in oneshot_commands[1][1]

    oneshot_only_args = copy.copy(oneshot_args)
    oneshot_only_args.oneshot_only = True
    oneshot_only_commands = benchmark_sweep.sweep_commands(oneshot_only_args)
    assert [name for name, _command, _output in oneshot_only_commands] == [
        "bounded-i64-small-16x16x16-oneshot-hip-direct.json",
    ]
    assert "--oneshot" in oneshot_only_commands[0][1]

    finite_oneshot_args = copy.copy(exact_args)
    finite_oneshot_args.out_root = Path("temp") / "finite-oneshot"
    finite_oneshot_args.backends = ["hip-direct"]
    finite_oneshot_args.semantics = ["finite-u8-ring"]
    finite_oneshot_args.case = ["small:16,16,16"]
    finite_oneshot_args.modulus = [255]
    finite_oneshot_args.include_oneshot = True
    finite_oneshot_args.oneshot_only = False
    finite_oneshot_commands = benchmark_sweep.sweep_commands(finite_oneshot_args)
    assert [name for name, _command, _output in finite_oneshot_commands] == [
        "finite-u8-ring-small-16x16x16-mod255-hip-direct.json",
        "finite-u8-ring-small-16x16x16-mod255-oneshot-hip-direct.json",
    ]
    assert "--oneshot" not in finite_oneshot_commands[0][1]
    assert "--oneshot" in finite_oneshot_commands[1][1]

    finite_oneshot_only_args = copy.copy(finite_oneshot_args)
    finite_oneshot_only_args.oneshot_only = True
    finite_oneshot_only_commands = benchmark_sweep.sweep_commands(finite_oneshot_only_args)
    assert [name for name, _command, _output in finite_oneshot_only_commands] == [
        "finite-u8-ring-small-16x16x16-mod255-oneshot-hip-direct.json",
    ]
    assert "--oneshot" in finite_oneshot_only_commands[0][1]

    exact_include_args = argparse.Namespace(
        bench=Path("rns8-bench"),
        bench_for=[],
        out_root=Path("temp") / "exact-wide-release",
        backends=None,
        semantics=None,
        case=["small:16,16,16"],
        adaptive_case=None,
        shapes=None,
        modulus=None,
        exact_wide_limbs=None,
        include_exact_wide_limb_variants=False,
        residue_chain_length=1,
        include_default_adaptive=False,
        include_adaptive_workloads=False,
        adaptive_only=False,
        include_wrap64=False,
        include_wrap64_rocwmma_candidate=False,
        include_exact_wide=True,
        reuse_packed_inputs=False,
        reuse_packed_a=False,
        reuse_packed_b=False,
        release_matrix=False,
        include_exploratory_large=False,
        review_mode="smoke",
        warmups=1,
        repeats=2,
        seed=7,
        write_autotune_cache=False,
        autotune_cache=None,
    )
    exact_include_commands = benchmark_sweep.sweep_commands(exact_include_args)
    exact_include_names = [name for name, _command, _output in exact_include_commands]
    assert "exact-wide-signed-small-16x16x16-cpu.json" in exact_include_names
    assert "exact-wide-unsigned-small-16x16x16-rocwmma.json" in exact_include_names
    assert len(exact_include_commands) == 10 + (2 * len(benchmark_sweep.BOUNDED_BACKENDS))

    adaptive_only_args = argparse.Namespace(
        bench=Path("rns8-bench"),
        bench_for=[],
        out_root=Path("temp") / "adaptive-only",
        backends=["cpu"],
        semantics=["bounded-i64"],
        case=None,
        adaptive_case=None,
        shapes=None,
        modulus=None,
        exact_wide_limbs=None,
        include_exact_wide_limb_variants=False,
        residue_chain_length=1,
        include_default_adaptive=True,
        include_adaptive_workloads=False,
        adaptive_only=True,
        include_wrap64=False,
        include_wrap64_rocwmma_candidate=False,
        include_exact_wide=False,
        reuse_packed_inputs=False,
        reuse_packed_a=False,
        reuse_packed_b=False,
        release_matrix=True,
        include_exploratory_large=False,
        review_mode="release",
        warmups=benchmark_sweep.RELEASE_MIN_WARMUPS,
        repeats=benchmark_sweep.RELEASE_MIN_REPEATS,
        seed=20260602,
        write_autotune_cache=False,
        autotune_cache=None,
    )
    commands = benchmark_sweep.sweep_commands(adaptive_only_args)
    assert len(commands) == 2
    assert all("--require-adaptive-execution" in command for _name, command, _output in commands)
    assert all("--bound-mode" in command and "per-tile" in command for _name, command, _output in commands)
    assert [name for name, _command, _output in commands] == [
        "bounded-i64-tiny-adaptive-65x65x64-cpu.json",
        "bounded-i64-medium-adaptive-1024x1024x1024-cpu.json",
    ]
    adaptive_input_scan_args = copy.copy(adaptive_only_args)
    adaptive_input_scan_args.bound_source = "input-scan"
    adaptive_input_scan_commands = benchmark_sweep.sweep_commands(adaptive_input_scan_args)
    assert len(adaptive_input_scan_commands) == 2
    assert all("--bound-source" in command for _name, command, _output in adaptive_input_scan_commands)
    assert all(
        command[command.index("--bound-source") + 1] == "input-scan"
        for _name, command, _output in adaptive_input_scan_commands
    )
    assert all("--bound-mode" in command and "per-tile" in command for _name, command, _output in adaptive_input_scan_commands)

    adaptive_only_args.include_default_adaptive = False
    try:
        benchmark_sweep.sweep_commands(adaptive_only_args)
    except SystemExit as exc:
        assert "--adaptive-only requires" in str(exc)
    else:
        raise AssertionError("adaptive-only without adaptive cases should fail")
    adaptive_only_args.include_adaptive_workloads = True
    workload_commands = benchmark_sweep.sweep_commands(adaptive_only_args)
    assert len(workload_commands) == len(benchmark_sweep.ADAPTIVE_WORKLOAD_CASES)
    assert all("--input-profile" in command and "adaptive-bands" in command for _name, command, _output in workload_commands)
    assert workload_commands[0][0] == "bounded-i64-banded-adaptive-256-256x256x512-cpu.json"
    adaptive_only_args.include_adaptive_workloads = False

    ck = finite_capture("ck", 190)
    direct = finite_capture("hip-direct", 300)
    cpu = finite_capture("cpu-reference", 500)
    smoke_report = benchmark_sweep.review_captures([ck, direct, cpu])
    assert smoke_report["schema_version"] == 3
    assert smoke_report["review_mode"] == "smoke"
    assert smoke_report["promotable_autotune_entries"] == []
    assert "not_release_review" in smoke_report["groups"][0]["candidates"][0]["promotion_blockers"]

    report = benchmark_sweep.review_captures([ck, direct, cpu], review_mode="release")
    benchmark_sweep.attach_cache_write_status(report, False, Path("unused.json"), 0)
    assert report["schema_version"] == 3
    assert report["review_mode"] == "release"
    assert report["group_count"] == 1
    assert len(report["promotable_autotune_entries"]) == 1
    assert report["promotable_autotune_entries"][0]["selected_backend"] == "ck"
    group = report["groups"][0]
    assert group["missing_required_baselines"] == []
    assert group["release_review_satisfied"] is True
    assert group["source_metadata"]["target_ids"] == ["gfx1100"]
    assert group["source_metadata"]["configured_amdgpu_targets"] == ["gfx1100"]
    assert group["source_metadata"]["hip_runtime_versions"] == ["70260201"]
    assert group["source_metadata"]["hip_driver_versions"] == ["70260201"]
    assert group["source_metadata"]["compilers"] == ["msvc 1944.194435227"]
    assert group["source_metadata"]["git_commits"] == ["fixture"]
    assert group["source_metadata"]["seeds"] == [13]
    assert group["source_metadata"]["warmups"] == [benchmark_sweep.RELEASE_MIN_WARMUPS]
    assert group["source_metadata"]["repeats"] == [benchmark_sweep.RELEASE_MIN_REPEATS]
    assert group["missing_gpu_targets"] == []
    assert group["gpu_target_identity_complete"] is True
    assert group["gpu_target_compatible"] is True
    assert group["missing_hip_toolchain_versions"] == []
    assert group["hip_toolchain_version_complete"] is True
    assert group["hip_toolchain_version_compatible"] is True
    assert group["missing_configured_gpu_targets"] == []
    assert group["configured_target_identity_complete"] is True
    assert group["configured_target_compatible"] is True
    assert group["missing_hip_runtime_versions"] == []
    assert group["hip_runtime_version_complete"] is True
    assert group["hip_runtime_version_compatible"] is True

    bounded_ck = bounded_capture("ck", 700)
    bounded_direct = bounded_capture("hip-direct", 300)
    bounded_cpu = bounded_capture("cpu-reference", 5000)
    bounded_vector = bounded_capture("hip-vector-alu-int64", 900)
    bounded_ck["timing_metadata"]["pack_layout"] = "matrix_engine_transient_pack_layout"
    bounded_ck["target_variant"] = {
        "target_id": "gfx1100",
        "target_namespace": "gfx1100",
        "review_group_key": "gfx1100/target=gfx1100/backend=ck",
    }
    bounded_direct["timing_metadata"]["generated_reducer_identity"] = (
        "direct_hip_fixed_prefix_2_generated_reducer_v1"
    )
    bounded_direct["target_variant"] = {
        "target_id": "gfx1100",
        "target_namespace": "gfx1100",
        "review_group_key": "gfx1100/target=gfx1100/backend=hip-direct",
    }
    bounded_cpu["target_variant"] = {
        "target_id": "cpu",
        "target_namespace": "cpu",
        "review_group_key": "cpu/target=cpu/backend=cpu-reference",
    }
    bounded_vector["benchmark_execution_mode"] = "public_runtime_vector_alu_native_buffers"
    bounded_vector["requested_next_op"] = {
        "requested": "native-gemm",
        "resolved": "native-gemm",
        "source": "benchmark_default",
    }
    bounded_vector["timing_metadata"]["pack_layout"] = "native_i64_row_major"
    bounded_vector["target_variant"] = {
        "target_id": "gfx1100",
        "target_namespace": "gfx1100",
        "review_group_key": "gfx1100/target=gfx1100/backend=hip-vector-alu-int64",
    }
    implementation_split_report = benchmark_sweep.review_captures(
        [bounded_ck, bounded_direct, bounded_cpu, bounded_vector],
        review_mode="release",
    )
    assert implementation_split_report["group_count"] == 1
    implementation_split_group = implementation_split_report["groups"][0]
    assert implementation_split_group["missing_required_baselines"] == []
    assert {
        candidate["backend"] for candidate in implementation_split_group["candidates"]
    } == {"ck", "cpu-reference", "hip-direct", "hip-vector-alu-int64"}

    eventless_ck = finite_capture("ck", 190)
    remove_gpu_events(eventless_ck)
    eventless_report = benchmark_sweep.review_captures([eventless_ck, direct, cpu], review_mode="release")
    eventless_group = eventless_report["groups"][0]
    assert eventless_report["promotable_autotune_entries"] == []
    assert eventless_group["fastest_promotable"] is None
    eventless_candidate = next(item for item in eventless_group["candidates"] if item["backend"] == "ck")
    assert "missing_required_gpu_events" in eventless_candidate["promotion_blockers"]

    cpu_faster = finite_capture("cpu-reference", 100)
    cpu_faster_report = benchmark_sweep.review_captures([ck, direct, cpu_faster], review_mode="release")
    cpu_faster_group = cpu_faster_report["groups"][0]
    assert cpu_faster_report["promotable_autotune_entries"] == []
    assert cpu_faster_group["fastest_promotable"] is None
    cpu_faster_candidate = next(item for item in cpu_faster_group["candidates"] if item["backend"] == "ck")
    assert "not_faster_than_cpu_reference" in cpu_faster_candidate["promotion_blockers"]
    assert group["missing_hip_driver_versions"] == []
    assert group["hip_driver_version_complete"] is True
    assert group["hip_driver_version_compatible"] is True
    assert group["missing_compiler_identities"] == []
    assert group["compiler_identity_complete"] is True
    assert group["compiler_identity_compatible"] is True
    assert group["missing_git_commits"] == []
    assert group["git_commit_identity_complete"] is True
    assert group["git_commit_identity_compatible"] is True
    assert group["missing_warmup_counts"] == []
    assert group["warmup_count_complete"] is True
    assert group["warmup_count_compatible"] is True
    assert group["missing_repeat_counts"] == []
    assert group["repeat_count_complete"] is True
    assert group["repeat_count_compatible"] is True
    assert group["duplicate_backends"] == []
    assert group["finite_modulus"] == 255
    assert group["fastest_promotable"]["backend"] == "ck"
    assert group["candidates"][0]["promotion_blockers"] == []
    assert group["candidates"][0]["bottleneck"]["class"] in {
        "compute_bound",
        "export_bound",
        "launch_or_api_bound",
        "mixed_bound",
        "pack_bound",
        "unknown",
    }

    missing_target_ck = copy.deepcopy(ck)
    missing_target_ck["device"]["gcn_arch"] = "unknown"
    missing_target_report = benchmark_sweep.review_captures(
        [missing_target_ck, direct, cpu],
        review_mode="release",
    )
    missing_target_group = missing_target_report["groups"][0]
    assert missing_target_report["promotable_autotune_entries"] == []
    assert missing_target_group["missing_gpu_targets"] == ["ck"]
    assert missing_target_group["gpu_target_identity_complete"] is False
    assert missing_target_group["gpu_target_compatible"] is False
    missing_target_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_target_group["candidates"]
    }
    assert "missing_gpu_target_id" in missing_target_blockers["ck"]

    mismatched_target_ck = copy.deepcopy(ck)
    mismatched_target_ck["device"]["gcn_arch"] = "gfx1101"
    mismatched_target_report = benchmark_sweep.review_captures(
        [mismatched_target_ck, direct, cpu],
        review_mode="release",
    )
    mismatched_target_group = mismatched_target_report["groups"][0]
    assert mismatched_target_report["promotable_autotune_entries"] == []
    assert mismatched_target_group["missing_gpu_targets"] == []
    assert mismatched_target_group["gpu_target_identity_complete"] is True
    assert mismatched_target_group["gpu_target_compatible"] is False
    mismatched_target_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_target_group["candidates"]
    }
    assert "gpu_target_mismatch" in mismatched_target_blockers["ck"]

    missing_version_direct = copy.deepcopy(direct)
    missing_version_direct["hip_toolchain"]["hip_sdk_or_rocm_version"] = None
    missing_version_report = benchmark_sweep.review_captures(
        [ck, missing_version_direct, cpu],
        review_mode="release",
    )
    missing_version_group = missing_version_report["groups"][0]
    assert missing_version_report["promotable_autotune_entries"] == []
    assert missing_version_group["missing_hip_toolchain_versions"] == ["hip-direct"]
    assert missing_version_group["hip_toolchain_version_complete"] is False
    assert missing_version_group["hip_toolchain_version_compatible"] is False
    missing_version_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_version_group["candidates"]
    }
    assert "missing_hip_toolchain_version" in missing_version_blockers["ck"]

    mismatched_version_ck = copy.deepcopy(ck)
    mismatched_version_ck["hip_toolchain"]["hip_sdk_or_rocm_version"] = "70260299"
    mismatched_version_report = benchmark_sweep.review_captures(
        [mismatched_version_ck, direct, cpu],
        review_mode="release",
    )
    mismatched_version_group = mismatched_version_report["groups"][0]
    assert mismatched_version_report["promotable_autotune_entries"] == []
    assert mismatched_version_group["missing_hip_toolchain_versions"] == []
    assert mismatched_version_group["hip_toolchain_version_complete"] is True
    assert mismatched_version_group["hip_toolchain_version_compatible"] is False
    mismatched_version_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_version_group["candidates"]
    }
    assert "hip_toolchain_version_mismatch" in mismatched_version_blockers["ck"]

    missing_configured_direct = copy.deepcopy(direct)
    missing_configured_direct["configured_amdgpu_targets"] = "unknown"
    missing_configured_report = benchmark_sweep.review_captures(
        [ck, missing_configured_direct, cpu],
        review_mode="release",
    )
    missing_configured_group = missing_configured_report["groups"][0]
    assert missing_configured_report["promotable_autotune_entries"] == []
    assert missing_configured_group["missing_configured_gpu_targets"] == ["hip-direct"]
    assert missing_configured_group["configured_target_identity_complete"] is False
    assert missing_configured_group["configured_target_compatible"] is False
    missing_configured_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_configured_group["candidates"]
    }
    assert "missing_configured_gpu_target" in missing_configured_blockers["ck"]

    mismatched_configured_ck = copy.deepcopy(ck)
    mismatched_configured_ck["configured_amdgpu_targets"] = "gfx1101"
    mismatched_configured_report = benchmark_sweep.review_captures(
        [mismatched_configured_ck, direct, cpu],
        review_mode="release",
    )
    mismatched_configured_group = mismatched_configured_report["groups"][0]
    assert mismatched_configured_report["promotable_autotune_entries"] == []
    assert mismatched_configured_group["missing_configured_gpu_targets"] == []
    assert mismatched_configured_group["configured_target_identity_complete"] is True
    assert mismatched_configured_group["configured_target_compatible"] is False
    mismatched_configured_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_configured_group["candidates"]
    }
    assert "configured_gpu_target_mismatch" in mismatched_configured_blockers["ck"]

    missing_runtime_direct = copy.deepcopy(direct)
    missing_runtime_direct["device"]["hip_runtime_version"] = 0
    missing_runtime_report = benchmark_sweep.review_captures(
        [ck, missing_runtime_direct, cpu],
        review_mode="release",
    )
    missing_runtime_group = missing_runtime_report["groups"][0]
    assert missing_runtime_report["promotable_autotune_entries"] == []
    assert missing_runtime_group["missing_hip_runtime_versions"] == ["hip-direct"]
    assert missing_runtime_group["hip_runtime_version_complete"] is False
    assert missing_runtime_group["hip_runtime_version_compatible"] is False
    missing_runtime_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_runtime_group["candidates"]
    }
    assert "missing_hip_runtime_version" in missing_runtime_blockers["ck"]

    mismatched_runtime_ck = copy.deepcopy(ck)
    mismatched_runtime_ck["device"]["hip_runtime_version"] = 70260299
    mismatched_runtime_report = benchmark_sweep.review_captures(
        [mismatched_runtime_ck, direct, cpu],
        review_mode="release",
    )
    mismatched_runtime_group = mismatched_runtime_report["groups"][0]
    assert mismatched_runtime_report["promotable_autotune_entries"] == []
    assert mismatched_runtime_group["missing_hip_runtime_versions"] == []
    assert mismatched_runtime_group["hip_runtime_version_complete"] is True
    assert mismatched_runtime_group["hip_runtime_version_compatible"] is False
    mismatched_runtime_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_runtime_group["candidates"]
    }
    assert "hip_runtime_version_mismatch" in mismatched_runtime_blockers["ck"]

    missing_driver_direct = copy.deepcopy(direct)
    missing_driver_direct["device"]["hip_driver_version"] = 0
    missing_driver_report = benchmark_sweep.review_captures(
        [ck, missing_driver_direct, cpu],
        review_mode="release",
    )
    missing_driver_group = missing_driver_report["groups"][0]
    assert missing_driver_report["promotable_autotune_entries"] == []
    assert missing_driver_group["missing_hip_driver_versions"] == ["hip-direct"]
    assert missing_driver_group["hip_driver_version_complete"] is False
    assert missing_driver_group["hip_driver_version_compatible"] is False
    missing_driver_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_driver_group["candidates"]
    }
    assert "missing_hip_driver_version" in missing_driver_blockers["ck"]

    mismatched_driver_ck = copy.deepcopy(ck)
    mismatched_driver_ck["device"]["hip_driver_version"] = 70260299
    mismatched_driver_report = benchmark_sweep.review_captures(
        [mismatched_driver_ck, direct, cpu],
        review_mode="release",
    )
    mismatched_driver_group = mismatched_driver_report["groups"][0]
    assert mismatched_driver_report["promotable_autotune_entries"] == []
    assert mismatched_driver_group["missing_hip_driver_versions"] == []
    assert mismatched_driver_group["hip_driver_version_complete"] is True
    assert mismatched_driver_group["hip_driver_version_compatible"] is False
    mismatched_driver_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_driver_group["candidates"]
    }
    assert "hip_driver_version_mismatch" in mismatched_driver_blockers["ck"]

    missing_compiler_direct = copy.deepcopy(direct)
    missing_compiler_direct["compiler"]["version"] = ""
    missing_compiler_report = benchmark_sweep.review_captures(
        [ck, missing_compiler_direct, cpu],
        review_mode="release",
    )
    missing_compiler_group = missing_compiler_report["groups"][0]
    assert missing_compiler_report["promotable_autotune_entries"] == []
    assert missing_compiler_group["missing_compiler_identities"] == ["hip-direct"]
    assert missing_compiler_group["compiler_identity_complete"] is False
    assert missing_compiler_group["compiler_identity_compatible"] is False
    missing_compiler_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_compiler_group["candidates"]
    }
    assert "missing_compiler_identity" in missing_compiler_blockers["ck"]

    mismatched_compiler_ck = copy.deepcopy(ck)
    mismatched_compiler_ck["compiler"]["version"] = "1944.999999"
    mismatched_compiler_report = benchmark_sweep.review_captures(
        [mismatched_compiler_ck, direct, cpu],
        review_mode="release",
    )
    mismatched_compiler_group = mismatched_compiler_report["groups"][0]
    assert mismatched_compiler_report["promotable_autotune_entries"] == []
    assert mismatched_compiler_group["missing_compiler_identities"] == []
    assert mismatched_compiler_group["compiler_identity_complete"] is True
    assert mismatched_compiler_group["compiler_identity_compatible"] is False
    mismatched_compiler_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_compiler_group["candidates"]
    }
    assert "compiler_identity_mismatch" in mismatched_compiler_blockers["ck"]

    missing_git_direct = copy.deepcopy(direct)
    missing_git_direct["git_commit"] = "unknown"
    missing_git_report = benchmark_sweep.review_captures(
        [ck, missing_git_direct, cpu],
        review_mode="release",
    )
    missing_git_group = missing_git_report["groups"][0]
    assert missing_git_report["promotable_autotune_entries"] == []
    assert missing_git_group["missing_git_commits"] == ["hip-direct"]
    assert missing_git_group["git_commit_identity_complete"] is False
    assert missing_git_group["git_commit_identity_compatible"] is False
    missing_git_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_git_group["candidates"]
    }
    assert "missing_git_commit" in missing_git_blockers["ck"]

    mismatched_git_ck = copy.deepcopy(ck)
    mismatched_git_ck["git_commit"] = "different-fixture"
    mismatched_git_report = benchmark_sweep.review_captures(
        [mismatched_git_ck, direct, cpu],
        review_mode="release",
    )
    mismatched_git_group = mismatched_git_report["groups"][0]
    assert mismatched_git_report["promotable_autotune_entries"] == []
    assert mismatched_git_group["missing_git_commits"] == []
    assert mismatched_git_group["git_commit_identity_complete"] is True
    assert mismatched_git_group["git_commit_identity_compatible"] is False
    mismatched_git_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_git_group["candidates"]
    }
    assert "git_commit_mismatch" in mismatched_git_blockers["ck"]

    missing_warmups_direct = copy.deepcopy(direct)
    missing_warmups_direct["warmups"] = 0
    missing_warmups_report = benchmark_sweep.review_captures(
        [ck, missing_warmups_direct, cpu],
        review_mode="release",
    )
    missing_warmups_group = missing_warmups_report["groups"][0]
    assert missing_warmups_report["promotable_autotune_entries"] == []
    assert missing_warmups_group["missing_warmup_counts"] == ["hip-direct"]
    assert missing_warmups_group["warmup_count_complete"] is False
    assert missing_warmups_group["warmup_count_compatible"] is False
    missing_warmups_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_warmups_group["candidates"]
    }
    assert "missing_warmup_count" in missing_warmups_blockers["ck"]

    mismatched_warmups_ck = copy.deepcopy(ck)
    mismatched_warmups_ck["warmups"] = benchmark_sweep.RELEASE_MIN_WARMUPS + 1
    mismatched_warmups_report = benchmark_sweep.review_captures(
        [mismatched_warmups_ck, direct, cpu],
        review_mode="release",
    )
    mismatched_warmups_group = mismatched_warmups_report["groups"][0]
    assert mismatched_warmups_report["promotable_autotune_entries"] == []
    assert mismatched_warmups_group["missing_warmup_counts"] == []
    assert mismatched_warmups_group["warmup_count_complete"] is True
    assert mismatched_warmups_group["warmup_count_compatible"] is False
    mismatched_warmups_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_warmups_group["candidates"]
    }
    assert "warmup_count_mismatch" in mismatched_warmups_blockers["ck"]

    missing_repeats_direct = copy.deepcopy(direct)
    missing_repeats_direct["repeats"] = 0
    missing_repeats_report = benchmark_sweep.review_captures(
        [ck, missing_repeats_direct, cpu],
        review_mode="release",
    )
    missing_repeats_group = missing_repeats_report["groups"][0]
    assert missing_repeats_report["promotable_autotune_entries"] == []
    assert missing_repeats_group["missing_repeat_counts"] == ["hip-direct"]
    assert missing_repeats_group["repeat_count_complete"] is False
    assert missing_repeats_group["repeat_count_compatible"] is False
    missing_repeats_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in missing_repeats_group["candidates"]
    }
    assert "missing_repeat_count" in missing_repeats_blockers["ck"]

    mismatched_repeats_ck = copy.deepcopy(ck)
    mismatched_repeats_ck["repeats"] = benchmark_sweep.RELEASE_MIN_REPEATS + 1
    mismatched_repeats_report = benchmark_sweep.review_captures(
        [mismatched_repeats_ck, direct, cpu],
        review_mode="release",
    )
    mismatched_repeats_group = mismatched_repeats_report["groups"][0]
    assert mismatched_repeats_report["promotable_autotune_entries"] == []
    assert mismatched_repeats_group["missing_repeat_counts"] == []
    assert mismatched_repeats_group["repeat_count_complete"] is True
    assert mismatched_repeats_group["repeat_count_compatible"] is False
    mismatched_repeats_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in mismatched_repeats_group["candidates"]
    }
    assert "repeat_count_mismatch" in mismatched_repeats_blockers["ck"]

    duplicate_ck_report = benchmark_sweep.review_captures(
        [ck, copy.deepcopy(ck), direct, cpu],
        review_mode="release",
    )
    duplicate_ck_group = duplicate_ck_report["groups"][0]
    assert duplicate_ck_report["promotable_autotune_entries"] == []
    assert duplicate_ck_group["duplicate_backends"] == ["ck"]
    duplicate_ck_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in duplicate_ck_group["candidates"]
    }
    assert "duplicate_backend_capture" in duplicate_ck_blockers["ck"]

    reuse_report = benchmark_sweep.review_captures(
        [mark_reused_pack(ck), mark_reused_pack(direct), mark_reused_pack(cpu)],
        review_mode="release",
    )
    reuse_group = reuse_report["groups"][0]
    assert reuse_group["source_metadata"]["pack_modes"] == ["prepacked_reuse"]
    assert reuse_group["source_metadata"]["prepack_reuse_strategies"] == ["persistent_matrix_residency"]
    assert reuse_group["source_metadata"]["prepack_reuse_operands"] == ["A/B"]
    assert reuse_report["promotable_autotune_entries"] == []
    reuse_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in reuse_group["candidates"]
    }
    assert "prepacked_reuse_not_autotune_promotable" in reuse_blockers["ck"]

    reuse_a_report = benchmark_sweep.review_captures(
        [mark_reused_a_pack(ck), mark_reused_a_pack(direct), mark_reused_a_pack(cpu)],
        review_mode="release",
    )
    reuse_a_group = reuse_a_report["groups"][0]
    assert reuse_a_group["source_metadata"]["pack_modes"] == ["prepacked_reuse_a"]
    assert reuse_a_group["source_metadata"]["prepack_reuse_strategies"] == ["persistent_matrix_residency"]
    assert reuse_a_group["source_metadata"]["prepack_reuse_operands"] == ["A"]
    reuse_a_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in reuse_a_group["candidates"]
    }
    assert "prepacked_reuse_not_autotune_promotable" in reuse_a_blockers["ck"]

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "finite-autotune.json"
        promoted = benchmark_sweep.write_promoted_cache_entries(report, [ck, direct, cpu], cache_path)
        benchmark_sweep.attach_cache_write_status(report, True, cache_path, promoted)
        assert promoted == 1
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        entry = cache["entries"][0]
        assert entry["finite_modulus"] == 255
        assert ";finite_modulus=255;" in f";{entry['key']};"

    blocked = benchmark_sweep.review_captures([ck], review_mode="release")
    benchmark_sweep.attach_cache_write_status(blocked, False, Path("unused.json"), 0)
    assert blocked["promotable_autotune_entries"] == []
    assert blocked["cache_write"]["status"] == "not_requested"
    assert blocked["groups"][0]["missing_required_baselines"] == ["cpu-reference", "hip-direct"]

    wrap64_direct = wrap64_capture("hip-direct", 200)
    wrap64_cpu = wrap64_capture("wrap64-byte-limb", 500)
    wrap64_report = benchmark_sweep.review_captures([wrap64_direct, wrap64_cpu], review_mode="release")
    assert wrap64_report["promotable_autotune_entries"] == []
    wrap64_group = wrap64_report["groups"][0]
    assert wrap64_group["semantics"] == "wrap_u64_mod_2_64"
    assert wrap64_group["missing_required_baselines"] == []
    assert wrap64_group["release_review_satisfied"] is True
    assert wrap64_group["fastest_promotable"] is None
    blockers_by_backend = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in wrap64_group["candidates"]
    }
    assert "not_accelerator_backend" in blockers_by_backend["hip-direct"]
    assert "not_accelerator_backend" in blockers_by_backend["wrap64-byte-limb"]
    assert "not_faster_than_direct_hip" in blockers_by_backend["wrap64-byte-limb"]

    wrap64_candidate = wrap64_capture(benchmark_sweep.WRAP64_ROCWMMA_CANDIDATE_BACKEND, 150)
    validate_capture(wrap64_candidate)
    wrap64_candidate_report = benchmark_sweep.review_captures(
        [wrap64_direct, wrap64_cpu, wrap64_candidate], review_mode="release"
    )
    candidate_group = wrap64_candidate_report["groups"][0]
    assert candidate_group["missing_required_baselines"] == []
    assert wrap64_candidate_report["promotable_autotune_entries"] == []
    candidate_blockers = {
        candidate["backend"]: candidate["promotion_blockers"] for candidate in candidate_group["candidates"]
    }
    assert "internal_candidate_not_public_backend" in candidate_blockers[
        benchmark_sweep.WRAP64_ROCWMMA_CANDIDATE_BACKEND
    ]

    exact_ck = exact_wide_capture("ck", 170)
    exact_direct = exact_wide_capture("hip-direct", 300)
    exact_cpu = exact_wide_capture("cpu-reference", 520)
    exact_report = benchmark_sweep.review_captures([exact_ck, exact_direct, exact_cpu], review_mode="release")
    exact_group = exact_report["groups"][0]
    assert exact_group["semantics"] == "exact_wide_signed"
    assert exact_group["required_baselines"] == ["cpu-reference", "hip-direct"]
    assert exact_group["missing_required_baselines"] == []
    assert len(exact_report["promotable_autotune_entries"]) == 1
    assert exact_report["promotable_autotune_entries"][0]["selected_backend"] == "ck"

    exact_blocked = benchmark_sweep.review_captures([exact_ck], review_mode="release")
    assert exact_blocked["groups"][0]["missing_required_baselines"] == ["cpu-reference", "hip-direct"]

    with tempfile.TemporaryDirectory() as temp_dir:
        bounded_ck = bounded_capture("ck", 180)
        bounded_direct = bounded_capture("hip-direct", 300)
        bounded_vector = bounded_capture("hip-vector-alu-int64", 240)
        bounded_cpu = bounded_capture("cpu-reference", 500)
        bounded_report = benchmark_sweep.review_captures(
            [bounded_ck, bounded_direct, bounded_vector, bounded_cpu], review_mode="release"
        )
        assert len(bounded_report["promotable_autotune_entries"]) == 1
        assert bounded_report["promotable_autotune_entries"][0]["selected_backend"] == "ck"
        assert bounded_report["promotable_autotune_entries"][0]["target_id"] == "gfx1100"
        assert bounded_report["groups"][0]["fastest_promotable"]["backend"] == "ck"
        cache_path = Path(temp_dir) / "autotune.json"
        promoted = benchmark_sweep.write_promoted_cache_entries(
            bounded_report, [bounded_ck, bounded_direct, bounded_vector, bounded_cpu], cache_path
        )
        benchmark_sweep.attach_cache_write_status(bounded_report, True, cache_path, promoted)
        assert promoted == 1
        assert bounded_report["cache_write"]["status"] == "written"
        assert bounded_report["groups"][0]["fastest_promotable"]["cache_write_status"] == "written"
        assert bounded_report["groups"][0]["candidates"][1]["cache_write_status"] != "written"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        entry = cache["entries"][0]
        assert entry["performance_validated"] is True
        assert entry["validation_status"] == "reviewed_release_same_contract_fastest_windows_gfx1100"
        assert entry["epilogue"] == bounded_ck["backend_metadata"]["epilogue_mode"]
        assert f";epilogue={entry['epilogue']}" in entry["key"]

    validate_capture(load_capture(FIXTURE_DIR / "v4_finite_ring_u8_ck.json"))
    validate_capture(load_capture(FIXTURE_DIR / "v4_finite_field_u8_rocwmma.json"))
    print("benchmark sweep self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
