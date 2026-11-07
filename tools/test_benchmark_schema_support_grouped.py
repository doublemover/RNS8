"""Independent schema fixture builders with explicit dependencies."""

from __future__ import annotations

import copy
from test_benchmark_schema_support_metadata import (
    add_helper_lane_fields,
    apply_int32_accumulator_contract,
    with_accumulator_key_fields,
)


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
    grouped["command_line"] = (
        f"{grouped['command_line']} --backend hip-direct --grouped-dispatch {task_count}"
    )
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
            "matrix_ownership_policy": "benchmark_owns_all_task_triplets_until_capture_end",
            "descriptor_reuse_policy": "reuse_after_shape_workspace_source_validation",
            "stride_policy": "matrix_ld_matches_logical_shape_host_output_ld_explicit",
            "output_currentness_policy": "device_residue_current_after_grouped_gemm_host_output_after_export",
            "lifetime_policy": "task_matrices_and_workspaces_destroyed_after_capture",
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
