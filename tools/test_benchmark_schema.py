#!/usr/bin/env python3
"""Self-test benchmark schema validation fixtures."""

from __future__ import annotations

import copy
from pathlib import Path

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def expect_valid(name: str) -> dict:
    path = FIXTURE_DIR / name
    data = load_capture(path)
    validate_capture(data, path)
    return data


def expect_invalid(data: dict, needle: str) -> None:
    try:
        validate_capture(data)
    except BenchmarkSchemaError as exc:
        message = str(exc)
        if needle not in message:
            raise AssertionError(f"expected validation error containing {needle!r}, got {message!r}") from exc
        return
    raise AssertionError(f"expected validation error containing {needle!r}")


def zero_summary() -> dict:
    return {"avg": 0.0, "median": 0.0, "p95": 0.0}


def as_direct_hip_finite_capture(
    capture: dict, modulus: int, kernel: str, isa_evidence: str
) -> dict:
    direct = copy.deepcopy(capture)
    metadata = direct["backend_metadata"]
    direct["backend_requested"] = "hip-direct"
    direct["backend_selected"] = "hip-direct"
    direct["selected_kernel"] = kernel
    direct["finite_modulus"] = modulus
    direct["backend_metadata"]["selected_kernel"] = kernel
    metadata["accelerator_backend"] = False
    metadata["matrix_engine_backend"] = False
    metadata["accelerator_library"] = "HIP runtime"
    metadata["accelerator_version"] = None
    metadata["capability_status"] = "implemented_correctness_backend"
    metadata["epilogue_mode"] = "fused_centered_residue_then_canonical_u8_export"
    metadata["workspace_mode"] = "resident_device_buffers"
    metadata["workspace_required_bytes"] = 0
    metadata["isa_evidence"] = isa_evidence
    metadata["autotune_key"] = (
        f"backend=hip-direct;semantics={direct['semantics']};m={direct['m']};n={direct['n']};k={direct['k']};"
        f"finite_modulus={modulus};prefix=0;tile_m={direct['tile_m']};tile_n={direct['tile_n']};"
        f"groups=0;adaptive_prefix=0;adaptive_skip=0;kernel={kernel};"
        "epilogue=fused_centered_residue_then_canonical_u8_export"
    )
    direct["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    return direct


def as_reused_pack_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    repeats = reused["repeats"]
    reused["reuse_packed_inputs"] = True
    reused["pack_mode"] = "prepacked_reuse"
    reused["prepack_reuse_operands"] = ["A", "B"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 123
    reused["avg_prepack_setup_us"] = 123.0
    reused["avg_pack_us"] = 0.0
    reused["raw_timings_us"]["pack"] = [0] * repeats
    reused["timing_summary_us"]["pack"] = zero_summary()
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A", "B"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["phase_notes"]["pack"] = (
        "zero-valued per-repeat phase; A and B were packed once into persistent matrices before warmups"
    )
    reused["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat rns_gemm plus crt_export host timing; excludes one-time prepack_setup_us"
    )
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "A and B were packed once before warmups and reused for every measured repeat",
    }
    for phase in ["pack_h2d", "pack_kernel", "finite_pack_h2d", "finite_pack_kernel", "pack"]:
        timings = reused.get("gpu_event_timings_us")
        summaries = reused.get("gpu_event_timing_summary_us")
        if isinstance(timings, dict) and phase in timings:
            timings[phase] = [0.0] * repeats
        if isinstance(summaries, dict) and phase in summaries:
            summaries[phase] = zero_summary()
    return reused


def as_hipblaslt_reused_ab_capture(capture: dict) -> dict:
    reused = as_reused_pack_capture(capture)
    phase = "hipblaslt_pack_transpose_centered"
    phase_order = reused["timing_metadata"].get("gpu_event_phase_order")
    if isinstance(phase_order, list) and phase in phase_order:
        phase_order.remove(phase)
    for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
        values = reused.get(field)
        if isinstance(values, dict):
            values.pop(phase, None)
    return reused


def as_reused_a_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    reused["reuse_packed_inputs"] = True
    reused["pack_mode"] = "prepacked_reuse_a"
    reused["prepack_reuse_operands"] = ["A"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 77
    reused["avg_prepack_setup_us"] = 77.0
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse_a"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["phase_notes"]["pack"] = (
        "per-repeat host timing for packing B; A was packed once into a persistent matrix before warmups"
    )
    reused["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat pack of non-reused input plus rns_gemm plus crt_export host timing; excludes one-time "
        "prepack_setup_us"
    )
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "prepacked A once before warmups and reused for every measured repeat",
    }
    return reused


def as_exact_wide_capture(capture: dict) -> dict:
    exact = copy.deepcopy(capture)
    exact["benchmark"] = "rns8_exact_wide_persistent_rns"
    exact["semantics"] = "exact_wide_signed"
    exact["bound_kind"] = "none"
    exact["bound_mode"] = "global"
    exact["bound"] = 0
    exact["prefix"] = 20
    exact["finite_modulus"] = None
    exact["tile_bounds_u64"] = None
    exact["epilogue_type"] = "exact_wide_signed_limb_export"
    exact["exact_wide_limb_count"] = 4
    exact["residue_chain_length"] = 1
    exact["residue_output_mode"] = "host_export"
    exact["input_distribution"] = "signed_uniform_-16_16"
    exact["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_correctness",
    ]
    exact["backend_metadata"]["epilogue_mode"] = "ck_fused_i32_to_centered_residue_rns_output"
    exact["backend_metadata"]["autotune_key"] = (
        "backend=ck;semantics=exact_wide_signed;m=64;n=128;k=64;prefix=20;tile_m=128;tile_n=128;"
        "groups=1;adaptive_prefix=0;adaptive_skip=0;kernel=ck_wmma_cshuffle_i8_i32_centered_epilogue_v1;"
        "epilogue=ck_fused_i32_to_centered_residue_rns_output"
    )
    exact["schedule_metadata"]["min_selected_prefix"] = 20
    exact["schedule_metadata"]["max_selected_prefix"] = 20
    exact["schedule_metadata"]["prefix_group_count"] = 1
    exact["schedule_metadata"]["adaptive_execution_applied"] = False
    exact["avg_per_modulus_gemm_estimate_us"] = float(exact["avg_rns_gemm_us"]) / 20.0
    exact["timing_note"] = (
        "host wall-clock timings for persistent exact-wide RNS packing, RNS GEMM, and fixed-width little-endian "
        "limb export; GPU event timing names exact-wide export operation groups when backend hooks are available"
    )
    exact["timing_metadata"]["phase_notes"]["crt_export"] = (
        "per-repeat host timing for fixed-width exact-wide limb export"
    )

    renamed = {
        "crt_export_status_memset": "exact_wide_export_status_memset",
        "crt_export_kernel": "exact_wide_export_kernel",
        "crt_export_status_d2h": "exact_wide_export_status_d2h",
        "crt_export_d2h": "exact_wide_export_d2h",
    }
    phase_order = exact["timing_metadata"].get("gpu_event_phase_order")
    if isinstance(phase_order, list):
        exact["timing_metadata"]["gpu_event_phase_order"] = [renamed.get(item, item) for item in phase_order]
    for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
        values = exact.get(field)
        if isinstance(values, dict):
            for old, new in renamed.items():
                if old in values:
                    values[new] = values.pop(old)
    return exact


def as_residue_current_chain_capture(capture: dict) -> dict:
    chain = as_exact_wide_capture(capture)
    repeats = chain["repeats"]
    chain["m"] = 64
    chain["n"] = 64
    chain["k"] = 64
    chain["epilogue_type"] = "residue_current_rns_output"
    chain["residue_chain_length"] = 3
    chain["residue_output_mode"] = "residue_current_rns"
    chain["timing_note"] = (
        "host wall-clock timings for an exact-wide residue-current RNS GEMM chain; each measured repeat runs "
        "3 resident RNS GEMM calls before host export, raw_timings_us.crt_export is intentionally zero, and one "
        "final fixed-width limb export runs after measured repeats only to produce checksum_u64"
    )
    chain["timing_metadata"]["gpu_event_timing"] = False
    chain["timing_metadata"]["gpu_event_timing_reason"] = "not_supported_for_residue_current_chain_mode"
    chain["timing_metadata"]["gpu_event_timing_status"] = "not_requested_for_residue_current_chain_mode"
    chain["timing_metadata"]["gpu_event_timing_source"] = None
    chain["timing_metadata"]["gpu_event_timing_source_scope"] = None
    chain["timing_metadata"]["gpu_event_timing_caveat"] = None
    chain["timing_metadata"]["gpu_event_phase_order"] = None
    chain["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for 3 chained rns8_gemm_rns calls that keep the intermediate output resident "
        "in RNS form"
    )
    chain["timing_metadata"]["phase_notes"]["crt_export"] = (
        "zero-valued per-repeat phase; residue-current chain mode defers host limb export until one final checksum "
        "export after measured repeats"
    )
    chain["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat pack plus chained rns_gemm host timing; excludes the final checksum-only limb export"
    )
    chain["raw_timings_us"]["crt_export"] = [0 for _ in range(repeats)]
    chain["timing_summary_us"]["crt_export"] = {"avg": 0.0, "median": 0.0, "p95": 0.0}
    chain["avg_crt_export_us"] = 0.0
    chain["gpu_event_timings_us"] = None
    chain["gpu_event_timing_summary_us"] = None
    return chain


def as_wrap64_wmma_candidate_capture(capture: dict) -> dict:
    candidate = copy.deepcopy(capture)
    repeats = candidate["repeats"]
    candidate["backend_requested"] = "rocwmma-wrap64-candidate"
    candidate["backend_selected"] = "wmma"
    candidate["selected_kernel"] = "rocwmma_wrap64_byte_gemm36_candidate_v0"
    candidate["tile_m"] = 16
    candidate["tile_n"] = 16
    candidate["command_line"] = (
        "rns8-bench --backend rocwmma-wrap64-candidate --semantics wrap-u64 "
        "--m 4 --n 4 --k 8 --tile-m 16 --tile-n 16"
    )
    candidate["backend_metadata"].update(
        {
            "source": "rns8_bench_wrap64_wmma_candidate",
            "selected_kernel": "rocwmma_wrap64_byte_gemm36_candidate_v0",
            "accelerator_backend": True,
            "correctness_backend": False,
            "matrix_engine_backend": True,
            "compiled_kernel_available": True,
            "exact_differential_validated": True,
            "performance_validated": False,
            "accelerator_library": "rocWMMA",
            "accelerator_version": "repo-local release/rocm-rel-7.1",
            "capability_status": "internal_wrap64_matrix_engine_candidate",
            "epilogue_mode": "low64_wrap_export",
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
    candidate["schedule_metadata"].update(
        {
            "source": "rns8_bench_wrap64_wmma_candidate_static_schedule",
            "tile_m": 16,
            "tile_n": 16,
            "tile_rows": 1,
            "tile_cols": 1,
            "tile_count": 1,
        }
    )
    candidate["timing_note"] = (
        "host wall-clock timings for the internal rocWMMA wrap64 byte-GEMM36 candidate; GPU event timing uses "
        "direct-HIP byte-limb pack/export labels plus one candidate operation-group label and this path is not "
        "public or AUTO-selected"
    )
    candidate["timing_metadata"]["gpu_event_timing_reason"] = (
        "captured_by_internal_rocwmma_wrap64_candidate_hooks"
    )
    candidate["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups"
    )
    candidate["timing_metadata"]["gpu_event_timing_caveat"] = (
        "HIP event timings record direct-HIP byte-limb pack/export operation groups plus one internal rocWMMA "
        "wrap64 byte-GEMM36 candidate operation group; host wall-clock timings remain required for CPU scheduling "
        "overhead, API dispatch, allocations, and synchronous host-side overhead not represented on the HIP stream"
    )
    candidate["timing_metadata"]["phase_notes"].update(
        {
            "planning": "one-time benchmark-owned metadata initialization for the internal rocWMMA wrap64 candidate",
            "scheduling": "one-time fixed 16x16 WMMA candidate schedule derivation from the matrix shape",
            "matrix_alloc": "one-time benchmark-owned compact byte-limb HIP device buffer allocation host timing",
            "pack": "per-repeat host timing for direct-HIP packing of A and B into compact byte-limb device buffers",
            "rns_gemm": "per-repeat host timing for the internal rocWMMA wrap64 byte-GEMM36 candidate",
            "crt_export": "per-repeat host timing for direct-HIP low-64-bit byte-limb export",
        }
    )
    candidate["timing_metadata"]["phase_availability"]["scheduling"] = {
        "timed": True,
        "timing_key": "scheduling",
        "scope": "benchmark_static_wrap64_wmma_candidate_schedule",
        "reason": "measured with host steady_clock around fixed 16x16 candidate schedule metadata initialization",
    }
    renamed = {
        "wrap64_byte_gemm36_tiled_2d_kernel": "wrap64_wmma_candidate_gemm36_kernel_group",
    }
    phase_order = candidate["timing_metadata"].get("gpu_event_phase_order")
    if isinstance(phase_order, list):
        candidate["timing_metadata"]["gpu_event_phase_order"] = [renamed.get(item, item) for item in phase_order]
    for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
        values = candidate.get(field)
        if isinstance(values, dict):
            for old, new in renamed.items():
                if old in values:
                    values[new] = values.pop(old)
    assert len(candidate["gpu_event_timings_us"]["wrap64_wmma_candidate_gemm36_kernel_group"]) == repeats
    return candidate


def main() -> int:
    v4_wrap64_hip = expect_valid("v4_wrap64_hip.json")
    v4_adaptive_u64 = expect_valid("v4_bounded_u64_adaptive_hip.json")
    v4_adaptive_i64 = expect_valid("v4_bounded_i64_adaptive_hip.json")
    v4_hipblaslt_i64 = expect_valid("v4_bounded_i64_hipblaslt.json")
    v4_ck_i64 = expect_valid("v4_bounded_i64_ck.json")
    v4_ck_adaptive_u64 = expect_valid("v4_bounded_u64_adaptive_ck.json")
    v4_wmma_i64 = expect_valid("v4_bounded_i64_rocwmma.json")
    v4_wmma_adaptive_u64 = expect_valid("v4_bounded_u64_adaptive_rocwmma.json")
    v4_vector_i64 = expect_valid("v4_bounded_i64_vector_alu.json")
    v4_vector_u64 = expect_valid("v4_bounded_u64_vector_alu.json")
    v4_finite_ring_ck = expect_valid("v4_finite_ring_u8_ck.json")
    v4_finite_field_wmma = expect_valid("v4_finite_field_u8_rocwmma.json")
    bounded = v4_adaptive_i64
    wrap64 = v4_wrap64_hip
    v4_wrap64_wmma_candidate = as_wrap64_wmma_candidate_capture(v4_wrap64_hip)
    validate_capture(v4_wrap64_wmma_candidate)

    v4_cpu_adaptive_i64 = copy.deepcopy(v4_adaptive_i64)
    v4_cpu_adaptive_i64["backend_requested"] = "cpu-reference"
    v4_cpu_adaptive_i64["backend_selected"] = "cpu-reference"
    v4_cpu_adaptive_i64["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
    v4_cpu_adaptive_i64["backend_metadata"]["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
    v4_cpu_adaptive_i64["backend_metadata"]["accelerator_library"] = None
    v4_cpu_adaptive_i64["backend_metadata"]["workspace_mode"] = "host_reference_workspace"
    v4_cpu_adaptive_i64["backend_metadata"]["workspace_required_bytes"] = 0
    v4_cpu_adaptive_i64["backend_metadata"]["isa_evidence"] = "not_applicable_cpu"
    v4_cpu_adaptive_i64["backend_metadata"]["autotune_key"] = (
        "backend=cpu-reference;semantics=bounded_i64;m=65;n=65;k=64;prefix=9;tile_m=64;tile_n=64;"
        "groups=4;adaptive_prefix=1;adaptive_skip=1;kernel=cpu_reference_scalar_rns_gemm_v1;"
        "epilogue=fused_centered_residue_then_crt_export"
    )
    v4_cpu_adaptive_i64["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_vector_alu_int64",
        "same_contract_direct_hip_correctness",
    ]
    v4_cpu_adaptive_i64["device"] = {
        "device_id": -1,
        "name": "CPU reference",
        "gcn_arch": "none",
        "hip_available": 0,
        "hip_runtime_version": 0,
        "hip_driver_version": 0,
        "global_mem_bytes": 0,
    }
    v4_cpu_adaptive_i64["timing_note"] = (
        "host wall-clock timings for the CPU adaptive per-tile bounded reference path; no GPU event timing is "
        "requested for this backend"
    )
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing"] = False
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_reason"] = "not_supported_for_selected_backend"
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_status"] = "not_requested_for_selected_backend"
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_source"] = None
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_source_scope"] = None
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_caveat"] = None
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_phase_order"] = None
    v4_cpu_adaptive_i64["gpu_event_timings_us"] = None
    v4_cpu_adaptive_i64["gpu_event_timing_summary_us"] = None
    validate_capture(v4_cpu_adaptive_i64)

    reused_ck_i64 = as_reused_pack_capture(v4_ck_i64)
    validate_capture(reused_ck_i64)
    reused_a_ck_i64 = as_reused_a_capture(v4_ck_i64)
    validate_capture(reused_a_ck_i64)
    reused_hipblaslt_i64 = as_hipblaslt_reused_ab_capture(v4_hipblaslt_i64)
    validate_capture(reused_hipblaslt_i64)

    exact_wide_ck = as_exact_wide_capture(v4_ck_i64)
    validate_capture(exact_wide_ck)
    exact_wide_no_status = copy.deepcopy(exact_wide_ck)
    exact_wide_no_status["exact_wide_export_status_check"] = "elided_full_width_device_reconstruction"
    exact_wide_no_status["gpu_event_timings_us"]["exact_wide_export_status_memset"] = [
        0.0 for _ in range(exact_wide_no_status["repeats"])
    ]
    exact_wide_no_status["gpu_event_timings_us"]["exact_wide_export_status_d2h"] = [
        0.0 for _ in range(exact_wide_no_status["repeats"])
    ]
    exact_wide_no_status["gpu_event_timing_summary_us"]["exact_wide_export_status_memset"] = zero_summary()
    exact_wide_no_status["gpu_event_timing_summary_us"]["exact_wide_export_status_d2h"] = zero_summary()
    validate_capture(exact_wide_no_status)
    exact_chain_ck = as_residue_current_chain_capture(v4_ck_i64)
    validate_capture(exact_chain_ck)

    bad_exact_bound = copy.deepcopy(exact_wide_ck)
    bad_exact_bound["bound_kind"] = "global_max_abs"
    expect_invalid(bad_exact_bound, "exact-wide captures must use bound_kind=none and bound=0")

    bad_exact_epilogue = copy.deepcopy(exact_wide_ck)
    bad_exact_epilogue["epilogue_type"] = "crt_export"
    expect_invalid(bad_exact_epilogue, "exact_wide_signed_limb_export")

    bad_exact_limb_count = copy.deepcopy(exact_wide_ck)
    bad_exact_limb_count["exact_wide_limb_count"] = 33
    expect_invalid(bad_exact_limb_count, "exact_wide_limb_count in [1, 32]")

    missing_exact_limb_count = copy.deepcopy(exact_wide_ck)
    del missing_exact_limb_count["exact_wide_limb_count"]
    expect_invalid(missing_exact_limb_count, "exact_wide_limb_count in [1, 32]")

    bad_exact_backend_epilogue = copy.deepcopy(exact_wide_ck)
    bad_exact_backend_epilogue["backend_metadata"]["epilogue_mode"] = "ck_fused_i32_to_centered_residue_then_crt_export"
    expect_invalid(bad_exact_backend_epilogue, "ck_fused_i32_to_centered_residue_rns_output")

    bad_exact_status_check = copy.deepcopy(exact_wide_no_status)
    bad_exact_status_check["exact_wide_export_status_check"] = "required_for_range_check"
    expect_invalid(bad_exact_status_check, "exact_wide_export_status_check")

    bad_exact_status_elision_events = copy.deepcopy(exact_wide_no_status)
    bad_exact_status_elision_events["gpu_event_timings_us"]["exact_wide_export_status_d2h"][0] = 1.0
    expect_invalid(bad_exact_status_elision_events, "status-elided captures")

    bad_chain_export = copy.deepcopy(exact_chain_ck)
    bad_chain_export["raw_timings_us"]["crt_export"][0] = 1
    bad_chain_export["timing_summary_us"]["crt_export"]["avg"] = 1
    bad_chain_export["avg_crt_export_us"] = 1
    expect_invalid(bad_chain_export, "residue-current chain captures must report raw_timings_us.crt_export")

    bad_chain_gpu_events = copy.deepcopy(exact_chain_ck)
    bad_chain_gpu_events["timing_metadata"]["gpu_event_timing"] = True
    bad_chain_gpu_events["timing_metadata"]["gpu_event_timing_source"] = "hipEventElapsedTime"
    bad_chain_gpu_events["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
    )
    bad_chain_gpu_events["timing_metadata"]["gpu_event_phase_order"] = ["pack", "rns_gemm"]
    bad_chain_repeats = bad_chain_gpu_events["repeats"]
    bad_chain_gpu_events["gpu_event_timings_us"] = {
        "pack": [1.0 for _ in range(bad_chain_repeats)],
        "rns_gemm": [2.0 for _ in range(bad_chain_repeats)],
    }
    bad_chain_gpu_events["gpu_event_timing_summary_us"] = {
        "pack": {"avg": 1.0, "median": 1.0, "p95": 1.0},
        "rns_gemm": {"avg": 2.0, "median": 2.0, "p95": 2.0},
    }
    expect_invalid(bad_chain_gpu_events, "residue-current chain captures must not claim GPU event timings")

    bad_chain_mode = copy.deepcopy(exact_chain_ck)
    bad_chain_mode["residue_output_mode"] = "host_export"
    expect_invalid(bad_chain_mode, "residue_output_mode=residue_current_rns")

    bad_chain_shape = copy.deepcopy(exact_chain_ck)
    bad_chain_shape["n"] = 128
    expect_invalid(bad_chain_shape, "square m=n=k shapes")

    bad_reused_pack = copy.deepcopy(reused_ck_i64)
    bad_reused_pack["raw_timings_us"]["pack"][0] = 1
    expect_invalid(bad_reused_pack, "zero-valued repeats")

    bad_reused_prepack = copy.deepcopy(reused_ck_i64)
    bad_reused_prepack["prepack_setup_us"] = None
    expect_invalid(bad_reused_prepack, "prepack_setup_us")

    bad_reused_mode = copy.deepcopy(reused_ck_i64)
    bad_reused_mode["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    expect_invalid(bad_reused_mode, "timing_metadata.pack_mode")

    bad_reused_operands = copy.deepcopy(reused_a_ck_i64)
    bad_reused_operands["prepack_reuse_operands"] = ["B"]
    expect_invalid(bad_reused_operands, "prepack_reuse_operands")

    bad_reused_metadata_operands = copy.deepcopy(reused_a_ck_i64)
    bad_reused_metadata_operands["timing_metadata"]["prepack_reuse_operands"] = ["A", "B"]
    expect_invalid(bad_reused_metadata_operands, "timing_metadata.prepack_reuse_operands")

    bad_reused_strategy = copy.deepcopy(reused_ck_i64)
    bad_reused_strategy["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
    bad_reused_strategy["timing_metadata"]["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
    expect_invalid(bad_reused_strategy, "pack_mode=prepacked_reuse_b")

    bad_reused_strategy_backend = copy.deepcopy(reused_ck_i64)
    bad_reused_strategy_backend["pack_mode"] = "prepacked_reuse_b"
    bad_reused_strategy_backend["prepack_reuse_operands"] = ["B"]
    bad_reused_strategy_backend["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
    bad_reused_strategy_backend["timing_metadata"]["pack_mode"] = "prepacked_reuse_b"
    bad_reused_strategy_backend["timing_metadata"]["prepack_reuse_operands"] = ["B"]
    bad_reused_strategy_backend["timing_metadata"]["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
    expect_invalid(bad_reused_strategy_backend, "backend_selected=wmma")

    bad_reused_metadata_strategy = copy.deepcopy(reused_ck_i64)
    bad_reused_metadata_strategy["timing_metadata"]["prepack_reuse_strategy"] = "none"
    expect_invalid(bad_reused_metadata_strategy, "timing_metadata.prepack_reuse_strategy")

    bad_hipblaslt_full_reuse_stale_event = copy.deepcopy(reused_hipblaslt_i64)
    bad_hipblaslt_full_reuse_stale_event["gpu_event_timings_us"]["hipblaslt_pack_transpose_centered"] = [
        0.0
    ] * bad_hipblaslt_full_reuse_stale_event["repeats"]
    bad_hipblaslt_full_reuse_stale_event["gpu_event_timing_summary_us"][
        "hipblaslt_pack_transpose_centered"
    ] = zero_summary()
    expect_invalid(
        bad_hipblaslt_full_reuse_stale_event,
        "undeclared phase hipblaslt_pack_transpose_centered",
    )

    bad_repack_prepack = copy.deepcopy(v4_ck_i64)
    bad_repack_prepack["reuse_packed_inputs"] = False
    bad_repack_prepack["pack_mode"] = "per_repeat_repack"
    bad_repack_prepack["prepack_setup_us"] = 1
    bad_repack_prepack["avg_prepack_setup_us"] = 1.0
    bad_repack_prepack["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    expect_invalid(bad_repack_prepack, "prepack_setup_us=null")

    bad_repack_strategy = copy.deepcopy(v4_ck_i64)
    bad_repack_strategy["prepack_reuse_strategy"] = "persistent_matrix_residency"
    bad_repack_strategy["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    expect_invalid(bad_repack_strategy, "prepack_reuse_strategy=none")

    bad_length = copy.deepcopy(bounded)
    bad_length["raw_timings_us"]["pack"].pop()
    expect_invalid(bad_length, "raw_timings_us.pack length")

    bad_summary = copy.deepcopy(bounded)
    bad_summary["gpu_event_timing_summary_us"]["crt_export"]["avg"] = 999.0
    expect_invalid(bad_summary, "gpu_event_timing_summary_us.crt_export.avg")

    bad_event_source = copy.deepcopy(bounded)
    bad_event_source["timing_metadata"]["gpu_event_timing_source"] = "std::chrono::steady_clock"
    expect_invalid(bad_event_source, "gpu_event_timing_source must be hipEventElapsedTime")

    bad_event_scope = copy.deepcopy(bounded)
    bad_event_scope["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_unknown_scope"
    expect_invalid(bad_event_scope, "known direct-HIP scope")

    bad_hipblaslt_scope = copy.deepcopy(v4_hipblaslt_i64)
    bad_hipblaslt_scope["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
    expect_invalid(bad_hipblaslt_scope, "known hipBLASLt scope")

    bad_ck_library = copy.deepcopy(v4_ck_i64)
    bad_ck_library["backend_metadata"]["accelerator_library"] = "HIP runtime"
    expect_invalid(bad_ck_library, "Composable Kernel")

    bad_ck_kernel = copy.deepcopy(v4_ck_adaptive_u64)
    bad_ck_kernel["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    bad_ck_kernel["backend_metadata"]["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    expect_invalid(bad_ck_kernel, "per-tile adaptive ck captures")

    bad_ck_events = copy.deepcopy(v4_ck_adaptive_u64)
    bad_ck_events["timing_metadata"]["gpu_event_timing_source_scope"] = "ck_default_stream"
    expect_invalid(bad_ck_events, "accelerator_backend_default_stream_deep_kernel_events")

    bad_wmma_library = copy.deepcopy(v4_wmma_i64)
    bad_wmma_library["backend_metadata"]["accelerator_library"] = "HIP runtime"
    expect_invalid(bad_wmma_library, "rocWMMA")

    bad_wmma_kernel = copy.deepcopy(v4_wmma_adaptive_u64)
    bad_wmma_kernel["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    bad_wmma_kernel["backend_metadata"]["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    expect_invalid(bad_wmma_kernel, "per-tile adaptive wmma captures")

    bad_wmma_events = copy.deepcopy(v4_wmma_adaptive_u64)
    bad_wmma_events["timing_metadata"]["gpu_event_timing_source_scope"] = "rocwmma_default_stream"
    expect_invalid(bad_wmma_events, "accelerator_backend_default_stream_deep_kernel_events")

    bad_hip_target = copy.deepcopy(v4_ck_i64)
    bad_hip_target["device"]["gcn_arch"] = "unknown"
    expect_invalid(bad_hip_target, "HIP backend captures must include non-placeholder device.gcn_arch")

    bad_hip_available = copy.deepcopy(v4_wmma_i64)
    bad_hip_available["device"]["hip_available"] = 0
    expect_invalid(bad_hip_available, "HIP backend captures must use device.hip_available=1")

    bad_vector_source = copy.deepcopy(v4_vector_i64)
    bad_vector_source["backend_metadata"]["source"] = "rns8_get_plan_backend_info"
    expect_invalid(bad_vector_source, "rns8_bench_vector_alu_baseline")

    bad_vector_accelerator = copy.deepcopy(v4_vector_u64)
    bad_vector_accelerator["backend_metadata"]["accelerator_backend"] = True
    expect_invalid(bad_vector_accelerator, "accelerator_backend")

    bad_vector_epilogue = copy.deepcopy(v4_vector_i64)
    bad_vector_epilogue["epilogue_type"] = "crt_export"
    expect_invalid(bad_vector_epilogue, "direct_int64_export")

    bad_vector_prereq = copy.deepcopy(v4_vector_u64)
    bad_vector_prereq["comparison_baseline"]["required_before_speedup_claim"] = ["same_contract_cpu_reference"]
    expect_invalid(bad_vector_prereq, "same_contract_direct_hip_correctness")

    bad_finite_ring_modulus = copy.deepcopy(v4_finite_ring_ck)
    bad_finite_ring_modulus["finite_modulus"] = 1
    expect_invalid(bad_finite_ring_modulus, "finite_ring_u8 finite_modulus")

    bad_finite_field_modulus = copy.deepcopy(v4_finite_field_wmma)
    bad_finite_field_modulus["finite_modulus"] = 255
    expect_invalid(bad_finite_field_modulus, "finite_field_u8 finite_modulus")

    bad_finite_prefix = copy.deepcopy(v4_finite_ring_ck)
    bad_finite_prefix["prefix"] = 9
    expect_invalid(bad_finite_prefix, "finite-u8 captures must use prefix=0")

    bad_finite_key = copy.deepcopy(v4_finite_ring_ck)
    bad_finite_key["backend_metadata"]["autotune_key"] = bad_finite_key["backend_metadata"]["autotune_key"].replace(
        ";finite_modulus=255",
        "",
    )
    expect_invalid(bad_finite_key, "finite-u8 backend_metadata.autotune_key must include finite_modulus")

    bad_finite_epilogue = copy.deepcopy(v4_finite_field_wmma)
    bad_finite_epilogue["epilogue_type"] = "crt_export"
    expect_invalid(bad_finite_epilogue, "canonical_u8_export")

    direct_finite_specialized = as_direct_hip_finite_capture(
        v4_finite_ring_ck,
        255,
        "direct_hip_tiled_finite_u8_gemm_mod255_v1",
        "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide",
    )
    validate_capture(direct_finite_specialized)

    bad_direct_finite_specialized_isa = copy.deepcopy(direct_finite_specialized)
    bad_direct_finite_specialized_isa["backend_metadata"]["isa_evidence"] = (
        "rns8_hip_direct_reciprocal_isa_gate"
    )
    expect_invalid(
        bad_direct_finite_specialized_isa,
        "direct-HIP finite-u8 specialized captures",
    )

    bad_direct_finite_specialized_kernel = copy.deepcopy(direct_finite_specialized)
    bad_direct_finite_specialized_kernel["selected_kernel"] = (
        "direct_hip_tiled_finite_u8_gemm_v1"
    )
    bad_direct_finite_specialized_kernel["backend_metadata"]["selected_kernel"] = (
        "direct_hip_tiled_finite_u8_gemm_v1"
    )
    expect_invalid(
        bad_direct_finite_specialized_kernel,
        "direct-HIP finite-u8 modulus 255 captures",
    )

    direct_finite_generic = as_direct_hip_finite_capture(
        v4_finite_ring_ck,
        127,
        "direct_hip_tiled_finite_u8_gemm_v1",
        "rns8_hip_direct_reciprocal_isa_gate",
    )
    validate_capture(direct_finite_generic)

    bad_direct_finite_generic_isa = copy.deepcopy(direct_finite_generic)
    bad_direct_finite_generic_isa["backend_metadata"]["isa_evidence"] = (
        "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide"
    )
    expect_invalid(
        bad_direct_finite_generic_isa,
        "direct-HIP generic finite-u8 captures",
    )

    missing_event_phase_order = copy.deepcopy(bounded)
    del missing_event_phase_order["timing_metadata"]["gpu_event_phase_order"]
    expect_invalid(missing_event_phase_order, "gpu_event_phase_order must be an array of strings when events are available")

    undeclared_event_phase = copy.deepcopy(bounded)
    undeclared_event_phase["gpu_event_timings_us"]["old_event_scope_phase"] = [1.0, 1.0, 1.0]
    expect_invalid(undeclared_event_phase, "undeclared phase old_event_scope_phase")

    duplicate_event_phase_order = copy.deepcopy(v4_vector_i64)
    duplicate_event_phase_order["timing_metadata"]["gpu_event_phase_order"].append("crt_export")
    expect_invalid(duplicate_event_phase_order, "gpu_event_phase_order must not contain duplicates")

    incomplete_vector_events = copy.deepcopy(v4_vector_i64)
    incomplete_vector_events["timing_metadata"]["gpu_event_phase_order"].remove("vector_alu_status_d2h")
    del incomplete_vector_events["gpu_event_timings_us"]["vector_alu_status_d2h"]
    del incomplete_vector_events["gpu_event_timing_summary_us"]["vector_alu_status_d2h"]
    expect_invalid(incomplete_vector_events, "vector-ALU GPU event phase set is incomplete")

    stale_deep_scope = copy.deepcopy(v4_ck_adaptive_u64)
    stale_deep_scope["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export"
    )
    expect_invalid(stale_deep_scope, "deep accelerator GPU event labels require")

    undeclared_deep_phase = copy.deepcopy(v4_ck_adaptive_u64)
    undeclared_deep_phase["timing_metadata"]["gpu_event_phase_order"].insert(6, "ck_prefix_99_fake_kernel")
    undeclared_deep_phase["gpu_event_timings_us"]["ck_prefix_99_fake_kernel"] = [1.0, 1.0]
    undeclared_deep_phase["gpu_event_timing_summary_us"]["ck_prefix_99_fake_kernel"] = zero_summary()
    expect_invalid(undeclared_deep_phase, "deep accelerator GPU event phase set contains undeclared phases")

    bad_schedule_tile = copy.deepcopy(bounded)
    bad_schedule_tile["tile_m"] = 96
    bad_schedule_tile["schedule_metadata"]["tile_m"] = 96
    expect_invalid(bad_schedule_tile, "tile_m must be a power of two")

    bad_schedule_prefix = copy.deepcopy(bounded)
    bad_schedule_prefix["schedule_metadata"]["max_selected_prefix"] = bad_schedule_prefix["prefix"]
    expect_invalid(bad_schedule_prefix, "adaptive_skip_active must match")

    bad_wrap_prefix = copy.deepcopy(wrap64)
    bad_wrap_prefix["prefix"] = 9
    expect_invalid(bad_wrap_prefix, "wrap64 captures must use prefix=0")

    bad_wrap_backend = copy.deepcopy(wrap64)
    bad_wrap_backend["backend_selected"] = "cpu-reference"
    expect_invalid(bad_wrap_backend, "rocWMMA candidate")

    bad_wrap64_hip_phase = copy.deepcopy(v4_wrap64_hip)
    bad_wrap64_hip_phase["gpu_event_timing_summary_us"]["wrap64_export_d2h"]["avg"] = 999.0
    expect_invalid(bad_wrap64_hip_phase, "gpu_event_timing_summary_us.wrap64_export_d2h.avg")

    bad_candidate_schedule_source = copy.deepcopy(v4_wrap64_wmma_candidate)
    bad_candidate_schedule_source["schedule_metadata"]["source"] = "rns8_get_plan_schedule_info"
    expect_invalid(bad_candidate_schedule_source, "rns8_bench_wrap64_wmma_candidate_static_schedule")

    bad_candidate_scope = copy.deepcopy(v4_wrap64_wmma_candidate)
    bad_candidate_scope["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export"
    )
    expect_invalid(bad_candidate_scope, "rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups")

    bad_candidate_correctness_flag = copy.deepcopy(v4_wrap64_wmma_candidate)
    bad_candidate_correctness_flag["backend_metadata"]["correctness_backend"] = True
    expect_invalid(bad_candidate_correctness_flag, "correctness_backend=False")

    bad_baseline_prereq = copy.deepcopy(v4_adaptive_u64)
    bad_baseline_prereq["comparison_baseline"]["required_before_speedup_claim"] = ["same_contract_cpu_reference"]
    expect_invalid(bad_baseline_prereq, "same_contract_direct_hip_vector_alu_int64")

    bad_speedup_claim = copy.deepcopy(v4_ck_i64)
    bad_speedup_claim["comparison_baseline"]["speedup_claimed"] = True
    expect_invalid(bad_speedup_claim, "speedup claims require a reviewed same-contract comparison baseline")

    legacy_reviewed_speedup = copy.deepcopy(v4_ck_i64)
    legacy_reviewed_speedup["comparison_baseline"]["status"] = "reviewed_same_contract_baseline"
    legacy_reviewed_speedup["comparison_baseline"]["speedup_claimed"] = True
    legacy_reviewed_speedup["comparison_baseline"]["selected_reference"] = "hip-direct"
    validate_capture(legacy_reviewed_speedup)

    bad_performance_promotion = copy.deepcopy(v4_wmma_i64)
    bad_performance_promotion["backend_metadata"]["performance_validated"] = True
    expect_invalid(
        bad_performance_promotion,
        "performance_validated captures require comparison_baseline.status=reviewed_release_same_contract_baseline",
    )

    bad_legacy_performance_promotion = copy.deepcopy(v4_wmma_i64)
    bad_legacy_performance_promotion["comparison_baseline"]["status"] = "reviewed_same_contract_baseline"
    bad_legacy_performance_promotion["comparison_baseline"]["selected_reference"] = "hip-direct"
    bad_legacy_performance_promotion["backend_metadata"]["performance_validated"] = True
    expect_invalid(
        bad_legacy_performance_promotion,
        "performance_validated captures require comparison_baseline.status=reviewed_release_same_contract_baseline",
    )

    bad_legacy_derived_tops = copy.deepcopy(v4_ck_i64)
    bad_legacy_derived_tops["comparison_baseline"]["status"] = "reviewed_same_contract_baseline"
    bad_legacy_derived_tops["comparison_baseline"]["selected_reference"] = "hip-direct"
    bad_legacy_derived_tops["derived_tops_equivalent"] = 123.0
    expect_invalid(
        bad_legacy_derived_tops,
        "derived_tops_equivalent requires a reviewed release same-contract comparison baseline",
    )

    release_performance_promotion = copy.deepcopy(v4_wmma_i64)
    release_performance_promotion["comparison_baseline"]["status"] = "reviewed_release_same_contract_baseline"
    release_performance_promotion["comparison_baseline"]["speedup_claimed"] = True
    release_performance_promotion["comparison_baseline"]["selected_reference"] = "hip-direct"
    release_performance_promotion["backend_metadata"]["performance_validated"] = True
    release_performance_promotion["derived_tops_equivalent"] = 123.0
    validate_capture(release_performance_promotion)

    release_performance_capture_without_speedup_claim = copy.deepcopy(v4_wmma_i64)
    release_performance_capture_without_speedup_claim["backend_requested"] = "auto"
    release_performance_capture_without_speedup_claim["comparison_baseline"]["status"] = (
        "reviewed_release_same_contract_baseline"
    )
    release_performance_capture_without_speedup_claim["comparison_baseline"]["speedup_claimed"] = False
    release_performance_capture_without_speedup_claim["comparison_baseline"]["selected_reference"] = None
    release_performance_capture_without_speedup_claim["backend_metadata"]["performance_validated"] = True
    validate_capture(release_performance_capture_without_speedup_claim)

    bad_current_version = copy.deepcopy(v4_adaptive_u64)
    bad_current_version["schema_version"] = 3
    expect_invalid(bad_current_version, "expected 4")

    missing_current_version = copy.deepcopy(v4_adaptive_u64)
    del missing_current_version["schema_version"]
    expect_invalid(missing_current_version, "missing required field schema_version")

    bad_schedule_summary = copy.deepcopy(v4_adaptive_u64)
    bad_schedule_summary["raw_timings_us"]["scheduling"] = [6]
    expect_invalid(bad_schedule_summary, "timing_summary_us.scheduling.avg")

    missing_tile_bound_scan = copy.deepcopy(v4_adaptive_u64)
    del missing_tile_bound_scan["raw_timings_us"]["tile_bound_scan"]
    expect_invalid(missing_tile_bound_scan, "raw_timings_us.tile_bound_scan must be an array")

    bad_tile_bound_scan_summary = copy.deepcopy(v4_adaptive_u64)
    bad_tile_bound_scan_summary["raw_timings_us"]["tile_bound_scan"] = [11]
    expect_invalid(bad_tile_bound_scan_summary, "timing_summary_us.tile_bound_scan.avg")

    bad_tile_bound_scan_availability = copy.deepcopy(v4_adaptive_u64)
    del bad_tile_bound_scan_availability["timing_metadata"]["phase_availability"]["tile_bound_scan"]
    expect_invalid(
        bad_tile_bound_scan_availability,
        "phase_availability.tile_bound_scan must be an object for per-tile captures",
    )

    bad_reduction_scope = copy.deepcopy(v4_wrap64_hip)
    bad_reduction_scope["timing_metadata"]["phase_availability"]["reduction"]["scope"] = "fused_into_rns_gemm"
    expect_invalid(bad_reduction_scope, "phase_availability.reduction.scope")

    bad_v4_bound = copy.deepcopy(v4_adaptive_u64)
    bad_v4_bound["bound"] = 1
    expect_invalid(bad_v4_bound, "per-tile adaptive captures must use bound=0")

    bad_v4_tile_count = copy.deepcopy(v4_adaptive_u64)
    bad_v4_tile_count["tile_bounds_u64"]["count"] = 3
    expect_invalid(bad_v4_tile_count, "tile_bounds_u64.count must match")

    bad_v4_skip_flag = copy.deepcopy(v4_adaptive_u64)
    bad_v4_skip_flag["schedule_metadata"]["adaptive_skip_active"] = False
    expect_invalid(bad_v4_skip_flag, "adaptive_skip_active must match")

    bad_v4_per_modulus = copy.deepcopy(v4_adaptive_u64)
    bad_v4_per_modulus["per_modulus_gemm_estimate_applicable"] = True
    expect_invalid(bad_v4_per_modulus, "fixed-prefix contract")

    bad_v4_toolchain = copy.deepcopy(v4_adaptive_u64)
    del bad_v4_toolchain["hip_toolchain"]
    expect_invalid(bad_v4_toolchain, "missing required field hip_toolchain")

    bad_v4_toolchain_enabled = copy.deepcopy(v4_adaptive_u64)
    bad_v4_toolchain_enabled["hip_toolchain"]["enabled"] = False
    expect_invalid(bad_v4_toolchain_enabled, "HIP backend captures must set hip_toolchain.enabled=true")

    bad_v4_scope = copy.deepcopy(v4_adaptive_u64)
    bad_v4_scope["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
    expect_invalid(bad_v4_scope, "bounded_adaptive")

    bad_v4_wrap64_kernel = copy.deepcopy(v4_wrap64_hip)
    bad_v4_wrap64_kernel["selected_kernel"] = "direct_hip_wrap64_comba_correctness_v1"
    expect_invalid(bad_v4_wrap64_kernel, "byte_gemm36")

    bad_event_nullability = copy.deepcopy(wrap64)
    bad_event_nullability["timing_metadata"]["gpu_event_timing"] = False
    bad_event_nullability["timing_metadata"]["gpu_event_timing_source"] = None
    bad_event_nullability["timing_metadata"]["gpu_event_timing_source_scope"] = None
    bad_event_nullability["gpu_event_timings_us"] = {"pack": [1.0, 2.0]}
    bad_event_nullability["gpu_event_timing_summary_us"] = None
    expect_invalid(bad_event_nullability, "gpu_event_timings_us must be null")

    bad_event_phase_order_nullability = copy.deepcopy(wrap64)
    bad_event_phase_order_nullability["timing_metadata"]["gpu_event_timing"] = False
    bad_event_phase_order_nullability["timing_metadata"]["gpu_event_timing_source"] = None
    bad_event_phase_order_nullability["timing_metadata"]["gpu_event_timing_source_scope"] = None
    bad_event_phase_order_nullability["gpu_event_timings_us"] = None
    bad_event_phase_order_nullability["gpu_event_timing_summary_us"] = None
    expect_invalid(bad_event_phase_order_nullability, "gpu_event_phase_order must be null")

    print("benchmark schema self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
