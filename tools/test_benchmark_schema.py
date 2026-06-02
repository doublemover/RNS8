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
    bad_ck_events["timing_metadata"]["gpu_event_timing"] = True
    bad_ck_events["timing_metadata"]["gpu_event_timing_source"] = "hipEventElapsedTime"
    bad_ck_events["timing_metadata"]["gpu_event_timing_source_scope"] = "ck_default_stream"
    bad_ck_events["timing_metadata"]["gpu_event_phase_order"] = ["pack"]
    bad_ck_events["gpu_event_timings_us"] = {"pack": [1.0, 1.0]}
    bad_ck_events["gpu_event_timing_summary_us"] = {"pack": {"avg": 1.0, "median": 1.0, "p95": 1.0}}
    expect_invalid(bad_ck_events, "CK per-tile adaptive captures must report unavailable GPU event timings")

    bad_wmma_library = copy.deepcopy(v4_wmma_i64)
    bad_wmma_library["backend_metadata"]["accelerator_library"] = "HIP runtime"
    expect_invalid(bad_wmma_library, "rocWMMA")

    bad_wmma_kernel = copy.deepcopy(v4_wmma_adaptive_u64)
    bad_wmma_kernel["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    bad_wmma_kernel["backend_metadata"]["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    expect_invalid(bad_wmma_kernel, "per-tile adaptive wmma captures")

    bad_wmma_events = copy.deepcopy(v4_wmma_adaptive_u64)
    bad_wmma_events["timing_metadata"]["gpu_event_timing"] = True
    bad_wmma_events["timing_metadata"]["gpu_event_timing_source"] = "hipEventElapsedTime"
    bad_wmma_events["timing_metadata"]["gpu_event_timing_source_scope"] = "rocwmma_default_stream"
    bad_wmma_events["timing_metadata"]["gpu_event_phase_order"] = ["pack"]
    bad_wmma_events["gpu_event_timings_us"] = {"pack": [1.0, 1.0]}
    bad_wmma_events["gpu_event_timing_summary_us"] = {"pack": {"avg": 1.0, "median": 1.0, "p95": 1.0}}
    expect_invalid(
        bad_wmma_events,
        "rocWMMA per-tile adaptive captures must report unavailable GPU event timings",
    )

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

    bad_finite_epilogue = copy.deepcopy(v4_finite_field_wmma)
    bad_finite_epilogue["epilogue_type"] = "crt_export"
    expect_invalid(bad_finite_epilogue, "canonical_u8_export")

    missing_event_phase_order = copy.deepcopy(bounded)
    del missing_event_phase_order["timing_metadata"]["gpu_event_phase_order"]
    expect_invalid(missing_event_phase_order, "gpu_event_phase_order must be an array of strings when events are available")

    undeclared_event_phase = copy.deepcopy(bounded)
    undeclared_event_phase["gpu_event_timings_us"]["old_event_scope_phase"] = [1.0, 1.0, 1.0]
    expect_invalid(undeclared_event_phase, "undeclared phase old_event_scope_phase")

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
    expect_invalid(bad_wrap_backend, "wrap64 captures must select wrap64-byte-limb or hip-direct backend")

    bad_wrap64_hip_phase = copy.deepcopy(v4_wrap64_hip)
    bad_wrap64_hip_phase["gpu_event_timing_summary_us"]["wrap64_export_d2h"]["avg"] = 999.0
    expect_invalid(bad_wrap64_hip_phase, "gpu_event_timing_summary_us.wrap64_export_d2h.avg")

    bad_baseline_prereq = copy.deepcopy(v4_adaptive_u64)
    bad_baseline_prereq["comparison_baseline"]["required_before_speedup_claim"] = ["same_contract_cpu_reference"]
    expect_invalid(bad_baseline_prereq, "same_contract_direct_hip_vector_alu_int64")

    bad_speedup_claim = copy.deepcopy(v4_ck_i64)
    bad_speedup_claim["comparison_baseline"]["speedup_claimed"] = True
    expect_invalid(bad_speedup_claim, "speedup claims require a reviewed same-contract comparison baseline")

    bad_performance_promotion = copy.deepcopy(v4_wmma_i64)
    bad_performance_promotion["backend_metadata"]["performance_validated"] = True
    expect_invalid(
        bad_performance_promotion,
        "performance_validated captures require comparison_baseline.status=reviewed_same_contract_baseline",
    )

    bad_current_version = copy.deepcopy(v4_adaptive_u64)
    bad_current_version["schema_version"] = 3
    expect_invalid(bad_current_version, "expected 4")

    missing_current_version = copy.deepcopy(v4_adaptive_u64)
    del missing_current_version["schema_version"]
    expect_invalid(missing_current_version, "missing required field schema_version")

    bad_schedule_summary = copy.deepcopy(v4_adaptive_u64)
    bad_schedule_summary["raw_timings_us"]["scheduling"] = [6]
    expect_invalid(bad_schedule_summary, "timing_summary_us.scheduling.avg")

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
