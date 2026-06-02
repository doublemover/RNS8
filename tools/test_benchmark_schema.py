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
    bounded = expect_valid("v2_bounded_hip.json")
    wrap64 = expect_valid("v2_wrap64.json")
    wrap64_hip = expect_valid("v2_wrap64_hip.json")
    v3_bounded = expect_valid("v3_bounded_hip.json")
    v3_wrap64_hip = expect_valid("v3_wrap64_hip.json")
    v4_wrap64_hip = expect_valid("v4_wrap64_hip.json")
    v4_adaptive_u64 = expect_valid("v4_bounded_u64_adaptive_hip.json")
    expect_valid("v4_bounded_i64_adaptive_hip.json")
    expect_valid("v1_legacy.json")

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

    bad_schedule_tile = copy.deepcopy(bounded)
    bad_schedule_tile["tile_m"] = 96
    bad_schedule_tile["schedule_metadata"]["tile_m"] = 96
    expect_invalid(bad_schedule_tile, "tile_m must be a power of two")

    bad_schedule_prefix = copy.deepcopy(bounded)
    bad_schedule_prefix["schedule_metadata"]["min_selected_prefix"] = 8
    expect_invalid(bad_schedule_prefix, "fixed selected schedule prefix equal to prefix")

    bad_wrap_prefix = copy.deepcopy(wrap64)
    bad_wrap_prefix["prefix"] = 9
    expect_invalid(bad_wrap_prefix, "wrap64 captures must use prefix=0")

    bad_wrap_backend = copy.deepcopy(wrap64)
    bad_wrap_backend["backend_selected"] = "cpu-reference"
    expect_invalid(bad_wrap_backend, "wrap64 captures must select wrap64-byte-limb or hip-direct backend")

    bad_wrap64_hip_phase = copy.deepcopy(wrap64_hip)
    bad_wrap64_hip_phase["gpu_event_timing_summary_us"]["wrap64_export_d2h"]["avg"] = 999.0
    expect_invalid(bad_wrap64_hip_phase, "gpu_event_timing_summary_us.wrap64_export_d2h.avg")

    bad_v3_schedule = copy.deepcopy(v3_bounded)
    bad_v3_schedule["raw_timings_us"]["scheduling"] = [6]
    expect_invalid(bad_v3_schedule, "timing_summary_us.scheduling.avg")

    bad_v3_reduction_scope = copy.deepcopy(v3_wrap64_hip)
    bad_v3_reduction_scope["timing_metadata"]["phase_availability"]["reduction"]["scope"] = "fused_into_rns_gemm"
    expect_invalid(bad_v3_reduction_scope, "phase_availability.reduction.scope")

    bad_v3_adaptive = copy.deepcopy(v3_bounded)
    bad_v3_adaptive["schedule_metadata"]["adaptive_execution_applied"] = True
    expect_invalid(bad_v3_adaptive, "adaptive_execution_applied must remain false")

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
    expect_invalid(bad_v4_toolchain_enabled, "hip-direct captures must set hip_toolchain.enabled=true")

    bad_v4_scope = copy.deepcopy(v4_adaptive_u64)
    bad_v4_scope["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
    expect_invalid(bad_v4_scope, "bounded_adaptive")

    bad_v4_wrap64_kernel = copy.deepcopy(v4_wrap64_hip)
    bad_v4_wrap64_kernel["selected_kernel"] = "direct_hip_wrap64_comba_correctness_v1"
    expect_invalid(bad_v4_wrap64_kernel, "byte_gemm36")

    bad_event_nullability = copy.deepcopy(wrap64)
    bad_event_nullability["gpu_event_timings_us"] = {"pack": [1.0, 2.0]}
    expect_invalid(bad_event_nullability, "gpu_event_timings_us must be null")

    print("benchmark schema self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
