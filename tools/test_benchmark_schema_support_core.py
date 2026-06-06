#!/usr/bin/env python3
"""Self-test benchmark schema validation fixtures."""

from __future__ import annotations

import copy
from pathlib import Path

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture
from metadata_registry_constants import (
    GROUPED_DISPATCH_EXECUTION_STRATEGIES,
    GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS,
    GROUPED_TASK_DEVICE_DESCRIPTOR_POLICIES,
)


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


def summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"avg": sum(values) / len(values), "median": ordered[len(ordered) // 2], "p95": ordered[-1]}


def ck_deep_chain_event_phases(prefix_count: int) -> list[str]:
    phases = [
        "pack_h2d",
        "pack_kernel",
        "pack",
        "rns_gemm_kernel_group",
        "ck_pack_a_kernel",
        "ck_pack_b_kernel",
        "ck_wmma_cshuffle_matmul",
        "ck_copy_centered_kernel",
        "ck_add_centered_kernel",
    ]
    for index in range(prefix_count):
        phases.extend(
            [
                f"ck_prefix_{index:02d}_pack_a",
                f"ck_prefix_{index:02d}_pack_b",
                f"ck_prefix_{index:02d}_matmul",
                f"ck_prefix_{index:02d}_copy_centered",
                f"ck_prefix_{index:02d}_add_centered",
            ]
        )
    phases.append("rns_gemm")
    return phases


def add_ck_chain_gpu_events(capture: dict, prefix_count: int) -> None:
    phases = ck_deep_chain_event_phases(prefix_count)
    repeats = capture["repeats"]
    capture["timing_metadata"]["gpu_event_timing"] = True
    capture["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_residue_current_chain_backend_hooks"
    capture["timing_metadata"]["gpu_event_timing_status"] = "available"
    capture["timing_metadata"]["gpu_event_timing_source"] = "hipEventElapsedTime"
    capture["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
    )
    capture["timing_metadata"]["gpu_event_timing_caveat"] = (
        "HIP event timings record per-repeat pack operation groups and chained rns_gemm backend operation groups; "
        "the final checksum-only export is outside measured repeats and absent from gpu_event_phase_order"
    )
    capture["timing_metadata"]["gpu_event_phase_order"] = phases
    capture["gpu_event_timings_us"] = {phase: [1.0 for _ in range(repeats)] for phase in phases}
    capture["gpu_event_timing_summary_us"] = {
        phase: summary(values) for phase, values in capture["gpu_event_timings_us"].items()
    }


def add_prefix_policy_fields(capture: dict, policy: str) -> dict:
    requested = capture["prefix"]
    selected = capture["schedule_metadata"]["max_selected_prefix"]
    skipped = max(requested - selected, 0)
    capture["selected_prefix"] = selected
    capture["requested_max_prefix"] = requested
    capture["contract_prefix_policy"] = policy
    capture["residue_planes_requested"] = requested
    capture["residue_planes_selected"] = selected
    capture["residue_planes_skipped"] = skipped
    capture["residue_plane_skip_fraction"] = float(skipped) / float(requested) if requested else 0.0
    return capture


def add_global_bound_scan_fields(capture: dict) -> dict:
    bound = capture["bound"]
    capture["bound_source"] = "input_scan"
    capture["bound_discovery"] = {
        "source": "input_row_column_abs_summary",
        "static_bound": bound,
        "selected_bound": bound,
        "discovered_global_bound": bound,
        "candidate_row_sum_col_max": bound,
        "candidate_row_max_col_sum": bound,
        "row_abs_sum_max": max(bound, 1),
        "row_abs_max": 16,
        "col_abs_sum_max": max(bound, 1),
        "col_abs_max": 16,
        "zero_row_count": 0,
        "zero_col_count": 0,
    }
    capture["global_bound_scan_us"] = 7
    capture["avg_global_bound_scan_us"] = 7.0
    capture["raw_timings_us"]["global_bound_scan"] = [7]
    capture["timing_summary_us"] = {
        "global_bound_scan": {"avg": 7.0, "median": 7.0, "p95": 7.0},
        **capture["timing_summary_us"],
    }
    phase_order = capture["timing_metadata"]["phase_order"]
    capture["timing_metadata"]["phase_order"] = ["global_bound_scan", *phase_order]
    capture["timing_metadata"]["phase_notes"] = {
        "global_bound_scan": "one-time exact seeded input prepass that computes row/column absolute-summary global bounds before plan creation",
        **capture["timing_metadata"]["phase_notes"],
    }
    capture["timing_metadata"]["phase_availability"] = {
        "global_bound_scan": {
            "timed": True,
            "timing_key": "global_bound_scan",
            "scope": "input_row_column_abs_summary",
            "reason": "measured with host steady_clock around seeded input row/column absolute-summary bound discovery before plan creation",
        },
        **capture["timing_metadata"]["phase_availability"],
    }
    return capture


def as_host_api_batch_capture(capture: dict) -> dict:
    batch = copy.deepcopy(capture)
    batch_size = 4
    batch["benchmark"] = "rns8_host_api_batch_persistent_resident"
    batch["benchmark_execution_mode"] = "benchmark_host_api_batch"
    batch["command_line"] = f'{batch["command_line"]} --host-api-batch-size {batch_size}'
    batch["timing_note"] = (
        "host wall-clock timings for benchmark-owned host API batching; raw timings are aggregate batch totals"
    )
    batch["reuse_packed_inputs"] = False
    batch["pack_mode"] = "per_repeat_repack"
    batch["prepack_reuse_operands"] = []
    batch["prepack_reuse_strategy"] = "none"
    batch["prepack_setup_us"] = None
    batch["avg_prepack_setup_us"] = None
    batch["host_api_batch"] = {
        "enabled": True,
        "batch_size": batch_size,
        "tasks_per_measured_repeat": batch_size,
        "total_measured_tasks": batch_size * batch["repeats"],
        "setup_scope": "one_shared_plan_per_capture_one_resident_matrix_workspace_triplet_per_task",
        "timing_policy": "aggregate_batch_totals_per_measured_repeat",
        "checksum_policy": "fnv1a_over_final_task_output_checksums",
    }
    batch["per_modulus_gemm_estimate_applicable"] = False
    batch["avg_per_modulus_gemm_estimate_us"] = batch["avg_rns_gemm_us"]
    for field, source in [
        ("avg_pack_per_task_us", "avg_pack_us"),
        ("avg_rns_gemm_per_task_us", "avg_rns_gemm_us"),
        ("avg_crt_export_per_task_us", "avg_crt_export_us"),
        ("avg_end_to_end_per_task_us", "avg_end_to_end_us"),
    ]:
        batch[field] = batch[source] / float(batch_size)
    metadata = batch["timing_metadata"]
    metadata["benchmark_execution_mode"] = "benchmark_host_api_batch"
    metadata["pack_mode"] = "per_repeat_repack"
    metadata["prepack_reuse_operands"] = []
    metadata["prepack_reuse_strategy"] = "none"
    metadata["host_api_batch_enabled"] = True
    metadata["host_api_batch_size"] = batch_size
    metadata["phase_notes"]["pack"] = (
        "per-repeat aggregate host timing for packing A and B for independent resident tasks"
    )
    metadata["phase_notes"]["rns_gemm"] = (
        "per-repeat aggregate host timing for independent resident rns8_gemm calls"
    )
    metadata["phase_notes"]["crt_export"] = (
        "per-repeat aggregate host timing for CRT export/reconstruction of every batch task"
    )
    metadata["phase_notes"]["end_to_end"] = (
        "per-repeat aggregate pack plus rns_gemm plus crt_export host timing for independent batch tasks"
    )
    return batch


def as_grouped_dispatch_capture(capture: dict) -> dict:
    grouped = copy.deepcopy(capture)
    task_count = 8
    kernel = "direct_hip_tiled_rns_gemm_v1"
    epilogue = "fused_centered_residue_then_crt_export"
    grouped["benchmark"] = "rns8_grouped_dispatch_persistent_resident"
    grouped["benchmark_execution_mode"] = "benchmark_grouped_dispatch_evidence"
    grouped["backend_requested"] = "hip-direct"
    grouped["backend_selected"] = "hip-direct"
    grouped["selected_kernel"] = kernel
    metadata = grouped["backend_metadata"]
    metadata["source"] = "rns8_get_plan_backend_info"
    metadata["selected_kernel"] = kernel
    metadata["accelerator_backend"] = False
    metadata["matrix_engine_backend"] = False
    metadata["accelerator_library"] = "HIP runtime"
    metadata["accelerator_version"] = "7.1"
    metadata["capability_status"] = "implemented_correctness_backend"
    metadata["epilogue_mode"] = epilogue
    metadata["workspace_mode"] = "resident_device_buffers"
    metadata["workspace_required_bytes"] = 0
    metadata["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(grouped)
    metadata["autotune_key"] = with_accumulator_key_fields(
        (
            f"backend=hip-direct;semantics={grouped['semantics']};m={grouped['m']};n={grouped['n']};"
            f"k={grouped['k']};prefix={grouped['prefix']};tile_m={grouped['tile_m']};"
            f"tile_n={grouped['tile_n']};groups=1;adaptive_prefix=0;adaptive_skip=0;"
            "execution=benchmark_grouped_dispatch_evidence;"
            f"kernel={kernel};epilogue={epilogue}"
        ),
        grouped,
    )
    grouped["command_line"] = f'{grouped["command_line"]} --backend hip-direct --grouped-dispatch {task_count}'
    grouped["timing_note"] = (
        "host wall-clock timings for benchmark-owned grouped dispatch evidence; raw timings are aggregate grouped totals"
    )
    grouped["reuse_packed_inputs"] = False
    grouped["pack_mode"] = "per_repeat_repack"
    grouped["bound_mode"] = "global"
    grouped["bound_source"] = "static_profile"
    grouped["prepack_reuse_operands"] = []
    grouped["prepack_reuse_strategy"] = "none"
    grouped["prepack_setup_us"] = None
    grouped["avg_prepack_setup_us"] = None
    grouped["host_api_batch"] = {
        "enabled": False,
        "batch_size": 1,
        "tasks_per_measured_repeat": 1,
        "total_measured_tasks": grouped["repeats"],
        "setup_scope": "single_task_default_benchmark_mode",
        "timing_policy": "single_call_totals_per_measured_repeat",
        "checksum_policy": "single_final_output_checksum",
    }
    grouped_output_domain = "native_i64_u64_host"
    if grouped["semantics"] in {"finite_ring_u8", "finite_field_u8"}:
        grouped_output_domain = "finite_u8_host"
    elif grouped["semantics"] in {"exact_wide_signed", "exact_wide_unsigned"}:
        grouped_output_domain = "exact_wide_limb_host"
    grouped_selected_prefix = grouped.get("selected_prefix", grouped["prefix"])
    grouped["grouped_dispatch"] = {
        "requested": True,
        "task_count": task_count,
        "descriptor_identity": (
            f"same_shape_m={grouped['m']};n={grouped['n']};k={grouped['k']};"
            f"semantics={grouped['semantics']}"
        ),
        "source_hash": str(grouped["seed"]),
        "output_hash": "final_checksum_u64",
        "setup_scope": "benchmark_grouped_dispatch_one_shared_plan_persistent_resident_tasks",
        "execution_strategy": "host_phase_loop_per_task_export",
        "batched_export_enabled": False,
        "device_output_slab_bytes": 0,
        "capture_status": "executed",
        "unsupported_reason": None,
        "promotion_eligible": False,
        "task_descriptor_contract": {
            "schema_version": 1,
            "descriptor_layout": "same_shape_resident_task_triplets_v1",
            "bucket_policy": "single_same_shape_bucket",
            "bucket_count": 1,
            "task_count": task_count,
            "same_shape_required": True,
            "shape_key": (
                f"m={grouped['m']};n={grouped['n']};k={grouped['k']};"
                f"tile_m={grouped['tile_m']};tile_n={grouped['tile_n']};"
                f"prefix={grouped_selected_prefix}"
            ),
            "semantics": grouped["semantics"],
            "output_domain": grouped_output_domain,
            "source_version_policy": "per_task_monotonic_source_version_repack",
            "workspace_policy": "one_workspace_per_task_shared_plan",
            "checksum_policy": "combined_per_task_checksum_u64",
            "status_policy": "fail_fast_per_task_operation_status",
            "device_descriptor_policy": "host_resident_task_loop",
            "promotion_eligible": False,
        },
    }
    grouped["per_modulus_gemm_estimate_applicable"] = False
    grouped["avg_per_modulus_gemm_estimate_us"] = grouped["avg_rns_gemm_us"]
    for field, source in [
        ("avg_pack_per_task_us", "avg_pack_us"),
        ("avg_rns_gemm_per_task_us", "avg_rns_gemm_us"),
        ("avg_crt_export_per_task_us", "avg_crt_export_us"),
        ("avg_end_to_end_per_task_us", "avg_end_to_end_us"),
    ]:
        grouped[field] = grouped[source] / float(task_count)
    add_helper_lane_fields(grouped)
    metadata = grouped["timing_metadata"]
    metadata["benchmark_execution_mode"] = "benchmark_grouped_dispatch_evidence"
    metadata["pack_mode"] = "per_repeat_repack"
    metadata["prepack_reuse_operands"] = []
    metadata["prepack_reuse_strategy"] = "none"
    metadata["host_api_batch_enabled"] = False
    metadata["host_api_batch_size"] = 1
    metadata["grouped_dispatch_enabled"] = True
    metadata["grouped_dispatch_task_count"] = task_count
    metadata["grouped_dispatch_execution_strategy"] = "host_phase_loop_per_task_export"
    metadata["grouped_dispatch_batched_export_enabled"] = False
    metadata["grouped_dispatch_device_output_slab_bytes"] = 0
    metadata["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
    metadata["generated_reducer_identity"] = "direct_hip_fixed_prefix_9_generated_reducer_v1"
    metadata["phase_notes"]["pack"] = (
        "per-repeat aggregate grouped-dispatch host timing for packing A and B for independent resident tasks"
    )
    metadata["phase_notes"]["rns_gemm"] = (
        "per-repeat aggregate grouped-dispatch host timing for independent resident rns8_gemm calls"
    )
    metadata["phase_notes"]["crt_export"] = (
        "per-repeat aggregate grouped-dispatch host timing for CRT export/reconstruction of every grouped task"
    )
    metadata["phase_notes"]["end_to_end"] = (
        "per-repeat aggregate grouped-dispatch pack plus rns_gemm plus crt_export host timing for independent grouped tasks"
    )
    return grouped


def as_native_to_rns_bridge_capture(capture: dict, conversion_event: str) -> dict:
    bridge = copy.deepcopy(capture)
    bridge["benchmark"] = "rns8_bounded_gemm_native_to_rns_bridge"
    bridge["benchmark_execution_mode"] = "auto_native_to_rns_bridge"
    bridge["backend_requested"] = "auto"
    bridge["backend_selected"] = "hip-direct"
    bridge["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    bridge["bound_kind"] = "global_max_abs" if bridge["semantics"] == "bounded_i64" else "global_max_unsigned"
    bridge["bound_mode"] = "global"
    bridge["bound"] = 16384
    bridge["tile_bounds_u64"] = None
    bridge["reuse_packed_inputs"] = False
    bridge["pack_mode"] = "per_repeat_repack"
    bridge["prepack_reuse_operands"] = []
    bridge["prepack_reuse_strategy"] = "none"
    bridge["prepack_setup_us"] = None
    bridge["avg_prepack_setup_us"] = None
    bridge["command_line"] = (
        "rns8-bench --backend auto --semantics bounded-i64 --m 65 --n 65 --k 64 "
        "--native-to-rns-bridge --warmups 1 --repeats 3 --seed 7"
    )
    bridge["timing_note"] = (
        "host wall-clock timings for an explicit AUTO/direct-HIP native-to-RNS bridge benchmark"
    )
    bridge.pop("tile_bound_scan_us", None)
    bridge.pop("avg_tile_bound_scan_us", None)

    schedule = bridge["schedule_metadata"]
    schedule["tile_rows"] = 1
    schedule["tile_cols"] = 1
    schedule["tile_count"] = 1
    schedule["min_required_prefix"] = bridge["prefix"]
    schedule["max_required_prefix"] = bridge["prefix"]
    schedule["min_selected_prefix"] = bridge["prefix"]
    schedule["max_selected_prefix"] = bridge["prefix"]
    schedule["prefix_group_count"] = 1
    schedule["adaptive_prefix_active"] = False
    schedule["adaptive_skip_active"] = False
    schedule["adaptive_execution_applied"] = False
    schedule["range_bit_length"] = 32
    schedule["zero_output_tile_count"] = 0
    schedule["zero_output_tile_fraction"] = 0.0
    schedule["zero_output_skip_active"] = False

    metadata = bridge["timing_metadata"]
    metadata["benchmark_execution_mode"] = "auto_native_to_rns_bridge"
    metadata["pack_mode"] = "per_repeat_repack"
    metadata["native_to_rns_bridge_forced"] = True
    metadata["prepack_reuse_operands"] = []
    metadata["prepack_reuse_strategy"] = "none"
    metadata["gpu_event_timing_reason"] = "captured_by_direct_hip_native_to_rns_bridge_hooks"
    metadata["gpu_event_timing_source_scope"] = "direct_hip_native_to_rns_bridge_default_stream_operation_groups"
    metadata["gpu_event_timing_caveat"] = (
        "HIP event timings record direct-HIP pack/export operation groups plus forced native-to-RNS conversion"
    )
    metadata["phase_order"] = [
        "planning",
        "scheduling",
        "matrix_alloc",
        "pack",
        "rns_gemm",
        "crt_export",
        "end_to_end",
    ]
    metadata["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for forced native-to-RNS device conversion of A and B followed by direct-HIP rns8_gemm_rns"
    )
    metadata["phase_notes"]["end_to_end"] = (
        "per-repeat pack plus native-to-RNS input conversion plus direct-HIP rns_gemm plus crt_export host timing"
    )
    metadata["phase_availability"].pop("tile_bound_scan", None)
    metadata["phase_availability"]["prepack_setup"] = {
        "timed": False,
        "timing_key": None,
        "scope": "not_requested_per_repeat_repack",
        "reason": "benchmark mode packs A and B inside every measured repeat",
    }
    metadata["phase_availability"]["native_to_rns_bridge"] = {
        "timed": True,
        "timing_key": "rns_gemm",
        "scope": "device_native_to_rns_conversion_inside_rns_gemm",
        "reason": "measured as an explicit native-to-RNS GPU event phase inside rns_gemm",
    }

    bridge["raw_timings_us"].pop("tile_bound_scan", None)
    bridge["timing_summary_us"].pop("tile_bound_scan", None)
    bridge["per_modulus_gemm_estimate_applicable"] = True
    bridge["avg_per_modulus_gemm_estimate_us"] = bridge["avg_rns_gemm_us"] / float(bridge["prefix"])

    backend_metadata = bridge["backend_metadata"]
    backend_metadata["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    backend_metadata["workspace_mode"] = "resident_device_buffers"
    backend_metadata["autotune_key"] = (
        "backend=hip-direct;target_id=gfx1100;semantics=bounded_i64;m=65;n=65;k=64;"
        "prefix=9;tile_m=64;tile_n=64;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "accumulator_type=int32;accumulator_signedness=signed_i8x_signed_i8;"
        "accumulator_modulus_policy=selected_rns_modulus_ladder;k_block_size=64;k_block_cap=65536;"
        "kernel=direct_hip_tiled_rns_gemm_v1;epilogue=fused_centered_residue_then_crt_export"
    )

    phases = [
        "pack_h2d",
        "pack_kernel",
        "pack",
        conversion_event,
        "rns_gemm_kernel_group",
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
    ]
    metadata["gpu_event_phase_order"] = phases
    bridge["gpu_event_timings_us"][conversion_event] = [2.0 for _ in range(bridge["repeats"])]
    bridge["gpu_event_timings_us"]["rns_gemm"] = [
        value + 2.0 for value in bridge["gpu_event_timings_us"]["rns_gemm_kernel_group"]
    ]
    bridge["gpu_event_timings_us"] = {phase: bridge["gpu_event_timings_us"][phase] for phase in phases}
    bridge["gpu_event_timing_summary_us"] = {
        phase: summary(values) for phase, values in bridge["gpu_event_timings_us"].items()
    }
    return bridge


def as_vector_to_rns_chain_capture(
    capture: dict,
    conversion_event: str,
    vector_kernel: str,
    *,
    reuse_consumer_b: bool = False,
) -> dict:
    chain = as_native_to_rns_bridge_capture(capture, conversion_event)
    chain["benchmark"] = "rns8_bounded_gemm_vector_to_rns_chain"
    chain["benchmark_execution_mode"] = "vector_native_to_direct_rns_chain"
    if reuse_consumer_b:
        chain["reuse_packed_inputs"] = True
        chain["pack_mode"] = "prepacked_reuse_b"
        chain["prepack_reuse_operands"] = ["B"]
        chain["prepack_reuse_strategy"] = "persistent_matrix_residency"
        chain["prepack_setup_us"] = 321
        chain["avg_prepack_setup_us"] = 321.0
    chain["command_line"] = (
        "rns8-bench --backend auto --semantics bounded-i64 --m 65 --n 65 --k 64 "
        "--vector-to-rns-chain"
        + (" --reuse-packed-b" if reuse_consumer_b else "")
        + " --warmups 1 --repeats 3 --seed 7"
    )
    chain["timing_note"] = "host wall-clock timings for a vector-native-to-Direct-RNS chain"
    chain["per_modulus_gemm_estimate_applicable"] = False
    chain["avg_per_modulus_gemm_estimate_us"] = chain["avg_rns_gemm_us"]

    metadata = chain["timing_metadata"]
    metadata["benchmark_execution_mode"] = "vector_native_to_direct_rns_chain"
    metadata["pack_mode"] = chain["pack_mode"]
    metadata["native_to_rns_bridge_forced"] = False
    metadata["vector_to_rns_chain"] = True
    metadata["vector_to_rns_chain_producer_backend"] = "hip-vector-alu-int64"
    metadata["vector_to_rns_chain_consumer_backend"] = "hip-direct"
    metadata["vector_to_rns_chain_consumer_k"] = chain["n"]
    metadata["gpu_event_timing_reason"] = "captured_by_vector_native_to_direct_rns_chain_hooks"
    metadata["gpu_event_timing_source_scope"] = (
        "direct_hip_vector_native_to_rns_chain_default_stream_operation_groups"
    )
    metadata["gpu_event_timing_caveat"] = (
        "HIP event timings record vector producer, native-to-RNS materialization, and Direct-HIP consumer events"
    )
    metadata["prepack_reuse_operands"] = chain["prepack_reuse_operands"]
    metadata["prepack_reuse_strategy"] = chain["prepack_reuse_strategy"]
    metadata["phase_notes"]["pack"] = (
        "per-repeat host timing for copying vector producer A/B into native HIP buffers"
        + (
            "; the Direct-HIP consumer B input was packed once before warmups"
            if reuse_consumer_b
            else " and packing the second Direct-HIP RNS input"
        )
    )
    metadata["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for vector-ALU native GEMM, native-to-RNS materialization, and Direct-HIP consumer RNS GEMM"
    )
    metadata["phase_notes"]["end_to_end"] = (
        "per-repeat vector producer pack plus Direct-HIP input pack, vector GEMM, native-to-RNS materialization, Direct-HIP consumer GEMM, and final CRT export host timing"
    )
    metadata["phase_availability"].pop("native_to_rns_bridge", None)
    metadata["phase_availability"]["vector_to_rns_chain"] = {
        "timed": True,
        "timing_key": "rns_gemm",
        "scope": "vector_native_output_to_direct_rns_consumer",
        "reason": "measured as vector, native-to-RNS, and Direct-HIP GPU event phases inside rns_gemm",
    }
    metadata["phase_availability"]["prepack_setup"] = {
        "timed": reuse_consumer_b,
        "timing_key": "prepack_setup_us" if reuse_consumer_b else None,
        "scope": "one_time_before_warmups" if reuse_consumer_b else "not_requested_per_repeat_repack",
        "reason": (
            "prepacked B once before warmups and reused for every measured repeat"
            if reuse_consumer_b
            else "benchmark mode packs A and B inside every measured repeat"
        ),
    }

    repeats = chain["repeats"]
    timings = chain["gpu_event_timings_us"]
    timings["vector_alu_pack_a_h2d"] = [1.0 for _ in range(repeats)]
    timings["vector_alu_pack_b_h2d"] = [1.5 for _ in range(repeats)]
    if reuse_consumer_b:
        timings["pack_h2d"] = [0.0 for _ in range(repeats)]
        timings["pack_kernel"] = [0.0 for _ in range(repeats)]
    timings["pack"] = [
        a + b + h2d + kernel
        for a, b, h2d, kernel in zip(
            timings["vector_alu_pack_a_h2d"],
            timings["vector_alu_pack_b_h2d"],
            timings["pack_h2d"],
            timings["pack_kernel"],
        )
    ]
    timings["vector_alu_status_memset"] = [0.25 for _ in range(repeats)]
    timings[vector_kernel] = [4.0 for _ in range(repeats)]
    timings["vector_alu_status_d2h"] = [0.5 for _ in range(repeats)]
    timings["rns_gemm"] = [
        memset + vector + status + conversion + direct
        for memset, vector, status, conversion, direct in zip(
            timings["vector_alu_status_memset"],
            timings[vector_kernel],
            timings["vector_alu_status_d2h"],
            timings[conversion_event],
            timings["rns_gemm_kernel_group"],
        )
    ]
    phases = [
        "vector_alu_pack_a_h2d",
        "vector_alu_pack_b_h2d",
        "pack_h2d",
        "pack_kernel",
        "pack",
        "vector_alu_status_memset",
        vector_kernel,
        "vector_alu_status_d2h",
        conversion_event,
        "rns_gemm_kernel_group",
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
    ]
    metadata["gpu_event_phase_order"] = phases
    chain["gpu_event_timings_us"] = {phase: timings[phase] for phase in phases}
    chain["gpu_event_timing_summary_us"] = {
        phase: summary(values) for phase, values in chain["gpu_event_timings_us"].items()
    }
    return chain


def add_per_tile_input_scan_fields(capture: dict) -> dict:
    capture["bound_source"] = "input_scan"
    capture["bound_discovery"] = {
        "source": "input_exact_tile_bounds",
        "static_bound": 0,
        "selected_bound": 0,
        "discovered_global_bound": None,
        "candidate_row_sum_col_max": None,
        "candidate_row_max_col_sum": None,
        "row_abs_sum_max": None,
        "row_abs_max": None,
        "col_abs_sum_max": None,
        "col_abs_max": None,
        "zero_row_count": None,
        "zero_col_count": None,
    }
    capture["command_line"] = f"{capture['command_line']} --bound-source input-scan"
    return capture


