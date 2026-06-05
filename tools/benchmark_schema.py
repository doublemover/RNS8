#!/usr/bin/env python3
"""Validate rns8-bench JSON capture files."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 4
WRAP64_HIP_U32_ACCUMULATOR_MAX_K = 4096
WRAP64_HIP_COLPAIR_MIN_DIMENSION = 256
WRAP64_LOW_PRODUCT_DIAGONALS = 8
WRAP64_HIP_U32_KERNEL = "direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4"
WRAP64_HIP_U64_KERNEL = "direct_hip_wrap64_byte_gemm36_u64acc_tiled_2d_v4"
WRAP64_HIP_U32_COLPAIR_KERNEL = "direct_hip_wrap64_byte_gemm36_u32acc_colpair_2d_v5"
BASELINE_STATUS_REQUIRED_NOT_RECORDED = "required_not_recorded"
BASELINE_STATUS_REVIEWED = "reviewed_same_contract_baseline"
BASELINE_STATUS_RELEASE_REVIEWED = "reviewed_release_same_contract_baseline"
BASELINE_STATUS_MISSING_REVIEWED = "missing_reviewed_same_contract_baseline"
REVIEWED_BASELINE_STATUSES = {
    BASELINE_STATUS_REVIEWED,
    BASELINE_STATUS_RELEASE_REVIEWED,
}
COMPARISON_BASELINE_STATUSES = {
    BASELINE_STATUS_REQUIRED_NOT_RECORDED,
    BASELINE_STATUS_REVIEWED,
    BASELINE_STATUS_RELEASE_REVIEWED,
    BASELINE_STATUS_MISSING_REVIEWED,
}
TIMING_PHASES = ["planning", "scheduling", "matrix_alloc", "pack", "rns_gemm", "crt_export", "end_to_end"]
GLOBAL_BOUND_TIMING_PHASE = "global_bound_scan"
PER_TILE_TIMING_PHASE = "tile_bound_scan"
REPEATED_TIMING_PHASES = {"pack", "rns_gemm", "crt_export", "end_to_end"}
TILE_SCHEDULE_ZERO_OUTPUT = 0x00000001
TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT = 0x00000002
TILE_SCHEDULE_KNOWN_FLAGS = TILE_SCHEDULE_ZERO_OUTPUT | TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT
BOUND_SOURCES = {"static_profile", "input_scan"}
BOUND_DISCOVERY_SOURCES = {
    "static_profile_contract",
    "input_row_column_abs_summary",
    "input_exact_tile_bounds",
}


def wrap64_hip_shape_supports_colpair_kernel(m_value: int, n_value: int, k_value: int) -> bool:
    if not all(type(value) is int for value in [m_value, n_value, k_value]):
        return False
    return (
        0 < k_value <= WRAP64_HIP_U32_ACCUMULATOR_MAX_K
        and m_value >= WRAP64_HIP_COLPAIR_MIN_DIMENSION
        and n_value >= WRAP64_HIP_COLPAIR_MIN_DIMENSION
    )


def wrap64_hip_allowed_kernels(m_value: int, n_value: int, k_value: int) -> set[str]:
    if wrap64_hip_shape_supports_colpair_kernel(m_value, n_value, k_value):
        return {WRAP64_HIP_U32_KERNEL, WRAP64_HIP_U32_COLPAIR_KERNEL}
    return (
        {WRAP64_HIP_U32_KERNEL}
        if type(k_value) is int and 0 < k_value <= WRAP64_HIP_U32_ACCUMULATOR_MAX_K
        else {WRAP64_HIP_U64_KERNEL}
    )


def wrap64_hip_expected_gemm_event_label(selected_kernel: Any) -> str:
    return (
        "wrap64_byte_gemm36_colpair_2d_kernel"
        if selected_kernel == WRAP64_HIP_U32_COLPAIR_KERNEL
        else "wrap64_byte_gemm36_tiled_2d_kernel"
    )


def output_destination_layout(padding: Any) -> str:
    return "contiguous_row_major" if padding == 0 else "padded_row_major"


BOUND_KINDS = {
    "none",
    "global_max_abs",
    "global_max_unsigned",
    "per_tile_max_abs",
    "per_tile_max_unsigned",
    "input_range_and_k",
}
PACK_MODES = {"per_repeat_repack", "prepacked_reuse", "prepacked_reuse_a", "prepacked_reuse_b"}
OUTPUT_DESTINATION_LAYOUTS = {"contiguous_row_major", "padded_row_major"}
NEXT_OP_HINTS = {"auto", "final-export", "rns-gemm", "native-gemm", "native-to-rns", "reuse-b"}
STATUS_HANDLING = {"required", "structurally_elided", "not_applicable"}
PACK_LAYOUTS = {
    "resident_rns_residue_planes",
    "wrap64_byte_limb_planes",
    "finite_u8_centered_residue",
    "native_i64_row_major",
    "native_u64_row_major",
    "native_i8_row_major_uniform_small",
    "native_i8_row_major_residue_channel_width3",
    "matrix_engine_transient_pack_layout",
    "transient_backend_pack_layout",
}
FUSION_MODES = {"none", "residue_channel_width3_experimental_benchmark_only"}
RESIDUE_GROUP_LAYOUTS = {
    "one_modulus_per_residue_plane",
    "first_prefix9_moduli_contiguous_width3_groups",
}
TARGET_NAMESPACES = {"cpu", "gfx1100", "gfx11xx", "gfx12xx", "gfx9xx_gfx94x", "unknown"}
OUTPUT_CONTRACT_DOMAINS = {
    "rns_residue_current",
    "finite_u8_host",
    "wrap64_u64_host",
    "exact_wide_limb_host",
    "native_i64_u64_host",
}
REUSE_OPERAND_ROLES = {"none", "A", "B", "A+B"}
GRAPH_REPLAY_STATUSES = {
    "not_requested",
    "unsupported_stream_capture_not_executed",
    "captured",
    "replayed",
}
GROUPED_DISPATCH_STATUSES = {
    "not_requested",
    "metadata_only_unsupported_for_execution_path",
    "executed",
}
STREAMING_OVERLAP_STATUSES = {
    "not_requested",
    "metadata_only_unsupported_for_execution_path",
    "executed",
}
RELEASE_GATE_REVIEW_STATUSES = {
    "not_requested",
    "pending_reviewed_summary",
    "reviewed_blocked",
    "reviewed_passed",
}
WORKLOAD_PROXY_FAMILIES = {
    "not_requested",
    "fhe_lattice_proxy",
    "dense_exact_arithmetic_proxy",
}
SELECTOR_REJECTION_REASONS = {
    "unsupported semantics",
    "per-tile unsupported",
    "backend not compiled",
    "probe failed",
    "no exact entry",
    "unvalidated entry",
    "identity/runtime mismatch",
    "workspace mismatch",
    "slower than selected",
}
GENERATED_REDUCER_RE = re.compile(
    r"^(not_applicable|direct_hip_fixed_prefix_(?:[1-9]|20)_generated_reducer_v1|"
    r"direct_hip_finite_modulus_\d+_fixed_reducer_v1)$"
)
DIRECT_HIP_EXPORT_STAGING_POLICIES = {
    "not_applicable",
    "disabled_by_RNS8_HIP_PINNED_EXPORT_STAGING",
    "forced_for_large_outputs_by_RNS8_HIP_PINNED_EXPORT_STAGING",
    "wrap64_forced_only_pending_padded_staging_evidence",
    "exact_wide_signed_forced_only_local_gfx1100_padded_staging_loses",
    "large_padded_outputs_only_default",
}
PREPACK_REUSE_STRATEGIES = {"none", "persistent_matrix_residency", "rocwmma_reusable_b_cache"}
PACK_MODE_OPERANDS = {
    "per_repeat_repack": [],
    "prepacked_reuse": ["A", "B"],
    "prepacked_reuse_a": ["A"],
    "prepacked_reuse_b": ["B"],
}
PREFIX_POLICY_FIELDS = {
    "selected_prefix",
    "requested_max_prefix",
    "contract_prefix_policy",
    "residue_planes_requested",
    "residue_planes_selected",
    "residue_planes_skipped",
    "residue_plane_skip_fraction",
}
CONTRACT_PREFIX_POLICIES = {
    "minimum_proven",
    "fixed_requested",
    "fixed_requested_residue_chain",
    "per_tile_minimum",
    "semantic_specific_no_rns_prefix",
}
RNS_PREFIX_SEMANTICS = {"bounded_i64", "bounded_u64", "exact_wide_signed", "exact_wide_unsigned"}
NON_RNS_PREFIX_SEMANTICS = {"finite_ring_u8", "finite_field_u8", "wrap_u64_mod_2_64"}
DIRECT_HIP_GPU_EVENT_SCOPES = {
    "direct_hip_default_stream_backend_operation_groups",
    "direct_hip_bounded_adaptive_default_stream_backend_operation_groups",
    "direct_hip_native_to_rns_bridge_default_stream_operation_groups",
    "direct_hip_vector_native_to_rns_chain_default_stream_operation_groups",
    "direct_hip_oneshot_default_stream_operation_groups",
    "direct_hip_oneshot_resident_fallback_default_stream_operation_groups",
    "direct_hip_wrap64_byte_gemm36_default_stream_backend_operation_groups",
}
HIPBLASLT_GPU_EVENT_SCOPES = {
    "hipblaslt_baseline_default_stream_backend_operation_groups",
}
ACCELERATOR_GPU_EVENT_SCOPES = {
    "accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export",
    "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export",
    "rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups",
}
VECTOR_ALU_GPU_EVENT_SCOPES = {
    "vector_alu_default_stream_native_int64_operation_groups",
}
OLD_ACCELERATOR_GPU_EVENT_SCOPE = "accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export"
DEEP_ACCELERATOR_GPU_EVENT_SCOPE = (
    "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
)
VECTOR_ALU_GPU_EVENT_SCOPE = "vector_alu_default_stream_native_int64_operation_groups"
HIP_RESIDENT_BACKENDS = {"hip-direct", "hipblaslt", "ck", "rocwmma", "hip-vector-alu-int64"}
CURRENT_CORRECTNESS_BACKENDS = {"cpu-reference", "hip-direct", "wrap64-byte-limb"}
BACKEND_SELECTED_VALUES = HIP_RESIDENT_BACKENDS | {"cpu-reference", "wrap64-byte-limb"}
BACKEND_REQUESTED_VALUES = BACKEND_SELECTED_VALUES | {"auto", "rocwmma-wrap64-candidate"}
PLACEHOLDER_GPU_TARGET_IDS = {"", "none", "cpu", "unknown", "not_applicable", "n/a", "null"}
VECTOR_ALU_SELECTED_KERNELS = {
    "hip_vector_alu_i64_exact_192b_v1",
    "hip_vector_alu_i64_gemv_n1_exact_192b_v1",
    "hip_vector_alu_u64_exact_192b_v1",
    "hip_vector_alu_u64_gemv_n1_exact_192b_v1",
}
CK_SELECTED_KERNELS = {
    "ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
    "ck_wmma_cshuffle_tiled_i8_i32_mod251_255_256_centered_epilogue_v2",
    "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1",
    "ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2",
    "ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2",
    "ck_wmma_cshuffle_finite_u8_mod256_centered_epilogue_v2",
}
ROCWMMA_SELECTED_KERNELS = {
    "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2",
    "rocwmma_i8_i32_signed_tiled_mod251_255_256_hot_residue_v2",
    "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1",
    "rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2",
    "rocwmma_i8_i32_signed_finite_u8_mod255_hot_residue_v2",
    "rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2",
}
DIRECT_HIP_FINITE_GENERIC_KERNEL = "direct_hip_tiled_finite_u8_gemm_v1"
DIRECT_HIP_FINITE_SPECIALIZED_KERNELS = {
    251: "direct_hip_tiled_finite_u8_gemm_mod251_v1",
    255: "direct_hip_tiled_finite_u8_gemm_mod255_v1",
    256: "direct_hip_tiled_finite_u8_gemm_mod256_v1",
}
DIRECT_HIP_FINITE_ONESHOT_GENERIC_KERNEL = "direct_hip_native_finite_u8_gemm_v1"
DIRECT_HIP_FINITE_ONESHOT_SPECIALIZED_KERNELS = {
    251: "direct_hip_native_finite_u8_gemm_mod251_v1",
    255: "direct_hip_native_finite_u8_gemm_mod255_v1",
    256: "direct_hip_native_finite_u8_gemm_mod256_v1",
}
CK_FINITE_GENERIC_KERNEL = "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1"
CK_FINITE_SPECIALIZED_KERNELS = {
    251: "ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2",
    255: "ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2",
    256: "ck_wmma_cshuffle_finite_u8_mod256_centered_epilogue_v2",
}
ROCWMMA_FINITE_GENERIC_KERNEL = "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1"
ROCWMMA_FINITE_SPECIALIZED_KERNELS = {
    251: "rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2",
    255: "rocwmma_i8_i32_signed_finite_u8_mod255_hot_residue_v2",
    256: "rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2",
}
DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_GENERIC_KERNEL = "direct_hip_native_a_finite_u8_gemm_v1"
DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_SPECIALIZED_KERNELS = {
    251: "direct_hip_native_a_finite_u8_gemm_mod251_v1",
    255: "direct_hip_native_a_finite_u8_gemm_mod255_v1",
    256: "direct_hip_native_a_finite_u8_gemm_mod256_v1",
}
DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE = "rns8_hip_direct_reciprocal_isa_gate"
DIRECT_HIP_BOUNDED_ONESHOT_KERNEL_V1 = "direct_hip_prefix9_native_input_grouped_rns_gemm_v1"
DIRECT_HIP_BOUNDED_ONESHOT_KERNEL_LARGE_COLPAIR_V2 = (
    "direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2"
)
DIRECT_HIP_BOUNDED_ONESHOT_KERNEL_U64_LARGE_V2 = DIRECT_HIP_BOUNDED_ONESHOT_KERNEL_LARGE_COLPAIR_V2
DIRECT_HIP_BOUNDED_ONESHOT_EPILOGUE = "native_input_centered_residue_then_crt_export"
DIRECT_HIP_BOUNDED_ONESHOT_WORKSPACE = "transient_native_inputs_to_resident_rns_output"
DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_KERNEL = "direct_hip_tiled_active_prefix_rns_gemm_v2"
DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_EPILOGUE = "fused_centered_residue_then_crt_export"
DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_WORKSPACE = "resident_device_buffers"
DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_KERNELS = {
    "bounded_i64": "direct_hip_native_a_i64_prefix9_reuse_b_grouped_rns_gemm_v1",
    "bounded_u64": "direct_hip_native_a_u64_prefix9_reuse_b_grouped_rns_gemm_v1",
}
DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_U64_LARGE_COLPAIR_KERNEL = (
    "direct_hip_native_a_u64_colpair_prefix9_reuse_b_grouped_rns_gemm_v2"
)
DIRECT_HIP_BOUNDED_UNIFORM_SMALL_NATIVE_A_REUSE_B_KERNELS = {
    "bounded_i64": "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_b_grouped_rns_gemm_v2",
    "bounded_u64": "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_b_grouped_rns_gemm_v2",
}
DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_KERNELS = {
    "bounded_i64": "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_a_grouped_rns_gemm_v1",
    "bounded_u64": "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_a_grouped_rns_gemm_v1",
}
DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_KERNELS = {
    "bounded_i64": "direct_hip_uniform_small_i8_ab_colpair_prefix9_transient_grouped_rns_gemm_v1",
    "bounded_u64": "direct_hip_uniform_small_i8_ab_colpair_prefix9_transient_grouped_rns_gemm_v1",
}
DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_KERNELS = {
    "bounded_i64": "direct_hip_uniform_small_i8_ab_colpair_prefix9_residue_channel_width3_experimental_v0",
    "bounded_u64": "direct_hip_uniform_small_i8_ab_colpair_prefix9_residue_channel_width3_experimental_v0",
}
DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_U64_LARGE_COLPAIR_KERNEL = (
    "direct_hip_native_b_u64_colpair_prefix9_reuse_a_grouped_rns_gemm_v1"
)
DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_EPILOGUE = "native_a_centered_resident_b_residue_then_crt_export"
DIRECT_HIP_BOUNDED_UNIFORM_SMALL_NATIVE_A_REUSE_B_EPILOGUE = (
    "uniform_small_i8_ab_resident_b_residue_then_crt_export"
)
DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_EPILOGUE = "uniform_small_i8_ab_resident_a_residue_then_crt_export"
DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_EPILOGUE = "uniform_small_i8_ab_transient_residue_then_crt_export"
DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_EPILOGUE = (
    "width3_residue_fusion_transient_then_crt_export"
)
DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_EPILOGUE = "resident_a_native_b_centered_residue_then_crt_export"
DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_WORKSPACE = "transient_native_a_resident_rns_b_output"
DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_WORKSPACE = "transient_i8_b_resident_i8_a_rns_output"
DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_WORKSPACE = "transient_i8_a_transient_i8_b_rns_output"
DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_WORKSPACE = (
    "width3_residue_fusion_transient_i8_inputs"
)
DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_WORKSPACE = "transient_native_b_resident_rns_a_output"
DIRECT_HIP_FINITE_ONESHOT_EPILOGUE = "native_u8_centered_residue_then_canonical_u8_export"
DIRECT_HIP_FINITE_ONESHOT_WORKSPACE = "transient_native_u8_inputs_to_resident_finite_output"
DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_EPILOGUE = "native_a_centered_resident_b_residue_then_canonical_u8_export"
DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_WORKSPACE = "transient_native_u8_a_resident_finite_b_output"
DIRECT_HIP_FINITE_SPECIALIZED_ISA_EVIDENCE = (
    "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide"
)
DIRECT_HIP_ADAPTIVE_KERNEL_V2 = "direct_hip_tiled_active_prefix_rns_gemm_v2"
DIRECT_HIP_ADAPTIVE_ZERO_SKIP_KERNEL_V3 = "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3"
DIRECT_HIP_ADAPTIVE_ZERO_ROW_COL_SKIP_KERNEL_V1 = "direct_hip_tiled_active_prefix_zero_row_col_skip_rns_gemm_v1"
DIRECT_HIP_ADAPTIVE_ZERO_TILE_ROW_COL_SKIP_KERNEL_V1 = (
    "direct_hip_tiled_active_prefix_zero_tile_row_col_skip_rns_gemm_v1"
)
WRAP64_ROCWMMA_CANDIDATE_KERNEL = "rocwmma_wrap64_byte_gemm36_candidate_v0"
CK_PREFIX_EVENT_RE = re.compile(r"^ck_prefix_(\d{2})_(pack_a|pack_b|matmul|copy_centered|add_centered)$")
ROCWMMA_PREFIX_EVENT_RE = re.compile(
    r"^rocwmma_prefix_(\d{2})_(pack_a|pack_b|matmul|pack_a_prepacked_b|matmul_prepacked_b)$"
)
CK_DEEP_GPU_EVENT_LABELS = {
    "ck_pack_a_kernel",
    "ck_pack_b_kernel",
    "ck_wmma_cshuffle_matmul",
    "ck_copy_centered_kernel",
    "ck_add_centered_kernel",
    "ck_zero_output_tile_memset",
}
ROCWMMA_DEEP_GPU_EVENT_LABELS = {
    "rocwmma_pack_a_kernel",
    "rocwmma_pack_b_kernel",
    "rocwmma_matmul_kernel",
    "rocwmma_pack_a_prepacked_b_kernel",
    "rocwmma_matmul_prepacked_b_kernel",
    "rocwmma_zero_output_tile_memset",
}
VECTOR_ALU_GPU_EVENT_LABELS = {
    "vector_alu_pack_a_h2d",
    "vector_alu_pack_b_h2d",
    "vector_alu_status_memset",
    "vector_alu_i64_kernel",
    "vector_alu_u64_kernel",
    "vector_alu_status_d2h",
    "vector_alu_output_d2h",
}
BENCHMARK_EXECUTION_MODES = {
    "persistent_resident_matrices",
    "public_oneshot_transient_native_inputs",
    "benchmark_owned_vector_alu_native_buffers",
    "public_runtime_vector_alu_native_buffers",
    "auto_native_to_rns_bridge",
    "vector_native_to_direct_rns_chain",
    "benchmark_host_api_batch",
    "hip_graph_replay_resident_rns_chain",
    "transient_native_a_resident_b_reuse",
    "transient_native_b_resident_a_reuse",
    "transient_uniform_small_i8_ab_inputs",
    "residue_channel_fusion_native_inputs",
    "transient_uniform_small_i8_a_resident_i8_b_reuse",
    "transient_uniform_small_i8_b_resident_i8_a_reuse",
    "internal_wrap64_rocwmma_candidate",
    "benchmark_grouped_dispatch_evidence",
    "benchmark_hip_graph_replay_evidence",
}
INT32_MAX = 2_147_483_647
UINT32_MAX = 4_294_967_295
DIRECT_HIP_ONESHOT_GPU_EVENT_PHASES = [
    "oneshot_native_input_h2d",
    "rns_gemm_kernel_group",
    "rns_gemm",
    "crt_export_status_memset",
    "crt_export_kernel",
    "crt_export_status_d2h",
    "crt_export_d2h",
    "crt_export",
    "oneshot_api_gpu",
]
DIRECT_HIP_ONESHOT_RESIDENT_FALLBACK_GPU_EVENT_PHASES = [
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
    "oneshot_api_gpu",
]
DIRECT_HIP_FINITE_ONESHOT_GPU_EVENT_PHASES = [
    "oneshot_native_input_h2d",
    "finite_native_gemm_kernel",
    "rns_gemm",
    "finite_export_kernel",
    "finite_export_d2h",
    "crt_export",
    "oneshot_api_gpu",
]


class BenchmarkSchemaError(ValueError):
    """Raised when a benchmark capture does not match the expected schema."""


def load_capture(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkSchemaError(f"{path}: failed to read benchmark JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BenchmarkSchemaError(f"{path}: benchmark JSON root must be an object")
    return data


def schema_version(data: dict[str, Any]) -> int:
    if "schema_version" not in data:
        raise BenchmarkSchemaError("missing required field schema_version")
    value = data["schema_version"]
    if not _is_int(value):
        raise BenchmarkSchemaError("schema_version must be an integer")
    return int(value)


def validation_errors(data: dict[str, Any], path: str | Path = "<memory>") -> list[str]:
    validator = _Validator(data, str(path))
    validator.validate()
    return validator.errors


def validate_capture(data: dict[str, Any], path: str | Path = "<memory>") -> dict[str, Any]:
    errors = validation_errors(data, path)
    if errors:
        raise BenchmarkSchemaError("\n".join(errors))
    return {"schema_version": schema_version(data)}


def validate_capture_file(path: Path) -> dict[str, Any]:
    return validate_capture(load_capture(path), path)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _has_concrete_gpu_target_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() not in PLACEHOLDER_GPU_TARGET_IDS


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(p * float(len(ordered) - 1) + 0.999999)
    return float(ordered[index])


def _average(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1.0e-5, abs_tol=1.0e-2)


def _is_prime_modulus(value: int) -> bool:
    if value < 2 or value > 251:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


class _Validator:
    def __init__(self, data: dict[str, Any], path: str) -> None:
        self.data = data
        self.path = path
        self.errors: list[str] = []
        self.version = 1

    def validate(self) -> None:
        if "schema_version" not in self.data:
            self._error("missing required field schema_version")
            return
        version_value = self.data["schema_version"]
        if not _is_int(version_value):
            self._error("schema_version must be an integer")
            return
        version = int(version_value)
        if version != SCHEMA_VERSION:
            self._error(f"unsupported schema_version {version}; expected {SCHEMA_VERSION}")
            return
        self.version = version
        self._validate_v4()

    def _error(self, message: str) -> None:
        self.errors.append(f"{self.path}: {message}")

    def _is_wrap64_rocwmma_candidate(self) -> bool:
        return (
            self.data.get("semantics") == "wrap_u64_mod_2_64"
            and self.data.get("backend_selected") == "rocwmma"
            and self.data.get("selected_kernel") == WRAP64_ROCWMMA_CANDIDATE_KERNEL
            and self.data.get("backend_requested") == "rocwmma-wrap64-candidate"
        )

    def _is_vector_alu_runtime_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-vector-alu-int64"
            and self.data.get("benchmark") == "rns8_bounded_gemm_hip_vector_alu_int64_runtime"
        )

    def _benchmark_execution_mode(self) -> str:
        mode = self.data.get("benchmark_execution_mode")
        if mode is None:
            metadata = self.data.get("timing_metadata")
            if isinstance(metadata, dict):
                mode = metadata.get("benchmark_execution_mode")
        if isinstance(mode, str):
            return mode
        if self._is_wrap64_rocwmma_candidate():
            return "internal_wrap64_rocwmma_candidate"
        if self._is_vector_alu_runtime_capture():
            return "public_runtime_vector_alu_native_buffers"
        if self.data.get("benchmark") == "rns8_hip_graph_replay_resident_rns_chain":
            return "hip_graph_replay_resident_rns_chain"
        if self.data.get("benchmark") in {"rns8_bounded_gemm_public_oneshot", "rns8_finite_u8_public_oneshot"}:
            return "public_oneshot_transient_native_inputs"
        if self.data.get("backend_selected") == "hip-vector-alu-int64":
            return "benchmark_owned_vector_alu_native_buffers"
        return "persistent_resident_matrices"

    def _is_bounded_oneshot_capture(self) -> bool:
        return (
            self._benchmark_execution_mode() == "public_oneshot_transient_native_inputs"
            or self.data.get("benchmark") == "rns8_bounded_gemm_public_oneshot"
        ) and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}

    def _is_direct_hip_bounded_native_input_oneshot_capture(self) -> bool:
        return (
            self._is_bounded_oneshot_capture()
            and self.data.get("backend_selected") == "hip-direct"
            and self.data.get("selected_prefix") == self.data.get("prefix") == 9
        )

    def _is_direct_hip_bounded_resident_fallback_oneshot_capture(self) -> bool:
        selected = self.data.get("selected_prefix")
        requested = self.data.get("prefix")
        schedule = self.data.get("schedule_metadata")
        schedule_selected = schedule.get("max_selected_prefix") if isinstance(schedule, dict) else None
        return (
            self._is_bounded_oneshot_capture()
            and self.data.get("backend_selected") == "hip-direct"
            and _is_int(selected)
            and _is_int(requested)
            and selected > 0
            and selected < requested
            and selected == schedule_selected
        )

    def _is_finite_oneshot_capture(self) -> bool:
        return (
            self._benchmark_execution_mode() == "public_oneshot_transient_native_inputs"
            or self.data.get("benchmark") == "rns8_finite_u8_public_oneshot"
        ) and self.data.get("semantics") in {"finite_ring_u8", "finite_field_u8"}

    def _is_public_oneshot_capture(self) -> bool:
        return self._is_bounded_oneshot_capture() or self._is_finite_oneshot_capture()

    def _is_direct_hip_bounded_native_a_reuse_b_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self.data.get("pack_mode") == "prepacked_reuse_b"
            and self.data.get("prepack_reuse_strategy") == "persistent_matrix_residency"
            and self._benchmark_execution_mode()
            in {"transient_native_a_resident_b_reuse", "transient_uniform_small_i8_a_resident_i8_b_reuse"}
        )

    def _is_direct_hip_bounded_native_a_reuse_b_uniform_small(self) -> bool:
        semantics = self.data.get("semantics")
        distribution = self.data.get("input_distribution")
        return (
            (semantics == "bounded_i64" and distribution == "signed_uniform_-16_16")
            or (semantics == "bounded_u64" and distribution == "unsigned_uniform_0_16")
        )

    def _is_direct_hip_bounded_native_a_reuse_b_u64_large_colpair(self) -> bool:
        return (
            self.data.get("semantics") == "bounded_u64"
            and not self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
            and _is_int(self.data.get("m"))
            and _is_int(self.data.get("n"))
            and _is_int(self.data.get("k"))
            and self.data.get("m") >= 512
            and self.data.get("n") >= 512
            and self.data.get("k") >= 512
        )

    def _is_direct_hip_bounded_uniform_small_reuse_a_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self.data.get("pack_mode") == "prepacked_reuse_a"
            and self.data.get("prepack_reuse_strategy") == "persistent_matrix_residency"
            and self._benchmark_execution_mode() == "transient_uniform_small_i8_b_resident_i8_a_reuse"
            and self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
        )

    def _is_direct_hip_bounded_uniform_small_transient_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self.data.get("pack_mode") == "per_repeat_repack"
            and self.data.get("prepack_reuse_strategy") == "none"
            and self._benchmark_execution_mode() == "transient_uniform_small_i8_ab_inputs"
            and self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
        )

    def _is_direct_hip_native_to_rns_bridge_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self._benchmark_execution_mode() == "auto_native_to_rns_bridge"
        )

    def _is_direct_hip_vector_to_rns_chain_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self._benchmark_execution_mode() == "vector_native_to_direct_rns_chain"
        )

    def _is_direct_hip_bounded_residue_channel_fusion_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self.data.get("benchmark") == "rns8_bounded_gemm_residue_channel_fusion_experiment"
            and self._benchmark_execution_mode() == "residue_channel_fusion_native_inputs"
            and self.data.get("pack_mode") == "per_repeat_repack"
            and self.data.get("prepack_reuse_strategy") == "none"
        )

    def _is_host_api_batch_capture(self) -> bool:
        return self._benchmark_execution_mode() == "benchmark_host_api_batch"

    def _is_grouped_dispatch_capture(self) -> bool:
        return self._benchmark_execution_mode() == "benchmark_grouped_dispatch_evidence"

    def _is_hip_graph_replay_capture(self) -> bool:
        return self._benchmark_execution_mode() == "hip_graph_replay_resident_rns_chain"

    def _is_direct_hip_bounded_native_b_reuse_a_u64_large_colpair_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") == "bounded_u64"
            and self.data.get("pack_mode") == "prepacked_reuse_a"
            and self.data.get("prepack_reuse_strategy") == "persistent_matrix_residency"
            and self._benchmark_execution_mode() == "transient_native_b_resident_a_reuse"
            and not self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
            and _is_int(self.data.get("m"))
            and _is_int(self.data.get("n"))
            and _is_int(self.data.get("k"))
            and self.data.get("m") >= 512
            and self.data.get("n") >= 512
            and self.data.get("k") >= 512
        )

    def _is_direct_hip_finite_native_a_reuse_b_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"finite_ring_u8", "finite_field_u8"}
            and self.data.get("pack_mode") == "prepacked_reuse_b"
            and self.data.get("prepack_reuse_strategy") == "persistent_matrix_residency"
        )

    def _require(self, key: str, kind: str) -> Any:
        if key not in self.data:
            self._error(f"missing required field {key}")
            return None
        value = self.data[key]
        if kind == "str" and not isinstance(value, str):
            self._error(f"{key} must be a string")
        elif kind == "int" and not _is_int(value):
            self._error(f"{key} must be an integer")
        elif kind == "number" and not _is_number(value):
            self._error(f"{key} must be a finite number")
        elif kind == "bool" and not isinstance(value, bool):
            self._error(f"{key} must be a boolean")
        elif kind == "dict" and not isinstance(value, dict):
            self._error(f"{key} must be an object")
        return value

    def _validate_v4(self) -> None:
        for key in [
            "benchmark",
            "backend_requested",
            "backend_selected",
            "semantics",
            "bound_kind",
            "epilogue_type",
            "input_distribution",
            "command_line",
            "git_commit",
            "configured_amdgpu_targets",
            "timing_source",
            "timing_note",
        ]:
            self._require(key, "str")
        selected_kernel = self.data.get("selected_kernel")
        if selected_kernel is not None and not isinstance(selected_kernel, str):
            self._error("selected_kernel must be a string or null")
        execution_mode = self.data.get("benchmark_execution_mode")
        if execution_mode is not None:
            if execution_mode not in BENCHMARK_EXECUTION_MODES:
                self._error(f"benchmark_execution_mode must be one of {sorted(BENCHMARK_EXECUTION_MODES)}")
            elif (
                self.data.get("benchmark") in {"rns8_bounded_gemm_public_oneshot", "rns8_finite_u8_public_oneshot"}
                and execution_mode != "public_oneshot_transient_native_inputs"
            ):
                self._error("one-shot benchmark captures must use benchmark_execution_mode=public_oneshot_transient_native_inputs")
        selected_backend = self.data.get("backend_selected")
        if isinstance(selected_backend, str) and selected_backend not in BACKEND_SELECTED_VALUES:
            self._error(f"backend_selected must be one of {sorted(BACKEND_SELECTED_VALUES)}")
        requested_backend = self.data.get("backend_requested")
        if isinstance(requested_backend, str) and requested_backend not in BACKEND_REQUESTED_VALUES:
            self._error(f"backend_requested must be one of {sorted(BACKEND_REQUESTED_VALUES)}")
        self._require("bound_mode", "str")
        for key in [
            "bound",
            "m",
            "n",
            "k",
            "prefix",
            "tile_m",
            "tile_n",
            "k_block_size",
            "seed",
            "warmups",
            "repeats",
            "checksum_u64",
        ]:
            self._require(key, "int")
        self._validate_nonnegative_ints()
        self._validate_nested_metadata()
        self._validate_backend_metadata()
        self._validate_helper_lane_metadata()
        self._validate_starfoundry_metadata()
        self._validate_host_api_batch_metadata()
        self._validate_grouped_dispatch_metadata()
        self._validate_hip_graph_replay_metadata()
        self._validate_comparison_baseline()
        self._validate_schedule_metadata()
        self._validate_bound_discovery_metadata()
        self._validate_prefix_policy_metadata()
        self._validate_semantic_contract()
        raw_timings = self._validate_raw_timings()
        self._validate_pack_reuse_fields(raw_timings)
        self._validate_residue_current_timings(raw_timings)
        self._validate_bounded_oneshot_timings(raw_timings)
        self._validate_all_zero_direct_hip_adaptive_timings(raw_timings)
        self._validate_timing_summaries(raw_timings, "timing_summary_us", self._timing_phases())
        self._validate_top_level_averages(raw_timings)
        self._validate_gpu_events()

    def _expected_accumulator_contract(self) -> dict[str, Any]:
        selected_backend = self.data.get("backend_selected")
        semantics = self.data.get("semantics")
        k = self.data.get("k")
        k_value = int(k) if _is_int(k) and k > 0 else 0
        if selected_backend == "hip-vector-alu-int64":
            return {
                "uses_int32_inner_product": False,
                "k_block_size": k_value,
                "k_block_cap": 0,
                "max_lhs_abs": 0,
                "max_rhs_abs": 0,
                "max_product": 0,
                "accumulator_type": "software_192bit_limb",
                "signedness": "signed_i64x_signed_i64"
                if semantics == "bounded_i64"
                else "unsigned_u64x_unsigned_u64",
                "input_domain": "native_i64_values" if semantics == "bounded_i64" else "native_u64_values",
                "modulus_policy": "native_exact_integer_output",
                "modulus": 0,
                "status": "exact_192bit_limb_no_int32_k_cap",
            }
        if self._is_wrap64_rocwmma_candidate():
            return {
                "uses_int32_inner_product": True,
                "k_block_size": k_value,
                "k_block_cap": 32768,
                "max_lhs_abs": 255,
                "max_rhs_abs": 255,
                "max_product": 255 * 255,
                "accumulator_type": "int32_then_int64_diagonal",
                "signedness": "unsigned_u8x_unsigned_u8",
                "input_domain": "compact_u8_byte_limb_pairs",
                "modulus_policy": "mod_2_64_wraparound_byte_limb",
                "modulus": 0,
                "status": "safe_int32_byte_limb_gemm36_k_block",
            }
        if semantics == "wrap_u64_mod_2_64" and selected_backend == "hip-direct" and 0 < k_value <= WRAP64_HIP_U32_ACCUMULATOR_MAX_K:
            return {
                "uses_int32_inner_product": False,
                "k_block_size": k_value,
                "k_block_cap": WRAP64_HIP_U32_ACCUMULATOR_MAX_K,
                "max_lhs_abs": 255,
                "max_rhs_abs": 255,
                "max_product": 255 * 255,
                "accumulator_type": "uint32_low_diagonal_then_uint64_carry",
                "signedness": "unsigned_byte_limb",
                "input_domain": "uint8_byte_limb_pairs",
                "modulus_policy": "mod_2_64_wraparound_byte_limb",
                "modulus": 0,
                "status": "safe_uint32_byte_limb_gemm36_k_block",
            }
        if semantics == "wrap_u64_mod_2_64":
            return {
                "uses_int32_inner_product": False,
                "k_block_size": k_value,
                "k_block_cap": 0,
                "max_lhs_abs": 0,
                "max_rhs_abs": 0,
                "max_product": 0,
                "accumulator_type": "uint64_wraparound_byte_limb",
                "signedness": "unsigned_byte_limb",
                "input_domain": "uint8_byte_limb_pairs",
                "modulus_policy": "mod_2_64_wraparound_byte_limb",
                "modulus": 0,
                "status": "exact_mod_2_64_byte_limb_no_int32_k_cap",
            }
        cap = 32768 if selected_backend == "ck" else 65536
        modulus = self.data.get("finite_modulus") if semantics in {"finite_ring_u8", "finite_field_u8"} else 0
        if not _is_int(modulus):
            modulus = 0
        return {
            "uses_int32_inner_product": True,
            "k_block_size": min(k_value, cap) if k_value > 0 else 0,
            "k_block_cap": cap,
            "max_lhs_abs": 128,
            "max_rhs_abs": 128,
            "max_product": 128 * 128,
            "accumulator_type": "int32",
            "signedness": "signed_i8x_signed_i8",
            "input_domain": "centered_i8_finite_u8_residues"
            if semantics in {"finite_ring_u8", "finite_field_u8"}
            else "centered_i8_rns_residue_planes",
            "modulus_policy": "finite_u8_modulus"
            if semantics in {"finite_ring_u8", "finite_field_u8"}
            else "selected_rns_modulus_ladder",
            "modulus": int(modulus),
            "status": "safe_int32_k_block_split",
        }

    def _validate_accumulator_safety(self, metadata: dict[str, Any]) -> None:
        safety = metadata.get("accumulator_safety")
        if not isinstance(safety, dict):
            self._error("backend_metadata.accumulator_safety must be an object")
            return
        for key in ["input_domain", "signedness", "accumulator_type", "modulus_policy", "status"]:
            if not isinstance(safety.get(key), str) or not safety.get(key):
                self._error(f"backend_metadata.accumulator_safety.{key} must be a nonempty string")
        for key in [
            "k_block_size",
            "k_block_cap",
            "modulus",
            "max_lhs_abs",
            "max_rhs_abs",
            "max_product",
        ]:
            value = safety.get(key)
            if not _is_int(value) or value < 0:
                self._error(f"backend_metadata.accumulator_safety.{key} must be a nonnegative integer")
        for key in ["uses_int32_inner_product", "safe_for_k_block"]:
            if not isinstance(safety.get(key), bool):
                self._error(f"backend_metadata.accumulator_safety.{key} must be a boolean")
        expected = self._expected_accumulator_contract()
        for key, value in expected.items():
            if safety.get(key) != value:
                self._error(f"backend_metadata.accumulator_safety.{key} must be {value}")
        if self.data.get("k_block_size") != safety.get("k_block_size"):
            self._error("k_block_size must match backend_metadata.accumulator_safety.k_block_size")
        if safety.get("max_product") != safety.get("max_lhs_abs") * safety.get("max_rhs_abs"):
            self._error("backend_metadata.accumulator_safety.max_product must equal max_lhs_abs*max_rhs_abs")
        if safety.get("uses_int32_inner_product") is True:
            k_block_size = safety.get("k_block_size")
            k_block_cap = safety.get("k_block_cap")
            max_product = safety.get("max_product")
            if _is_int(k_block_size) and _is_int(k_block_cap) and k_block_size > k_block_cap:
                self._error("int32 accumulator k_block_size must not exceed k_block_cap")
            if _is_int(k_block_size) and _is_int(max_product) and k_block_size > 0:
                if max_product * k_block_size > INT32_MAX:
                    self._error("int32 accumulator contract exceeds int32 range")
            if safety.get("safe_for_k_block") is not True:
                self._error("int32 accumulator captures must set safe_for_k_block=true")
        else:
            uint32_diagonal_accumulator = safety.get("accumulator_type") == "uint32_low_diagonal_then_uint64_carry"
            if uint32_diagonal_accumulator:
                k_block_size = safety.get("k_block_size")
                k_block_cap = safety.get("k_block_cap")
                max_product = safety.get("max_product")
                if _is_int(k_block_size) and _is_int(k_block_cap) and k_block_size > k_block_cap:
                    self._error("uint32 diagonal accumulator k_block_size must not exceed k_block_cap")
                if _is_int(k_block_size) and _is_int(max_product) and k_block_size > 0:
                    if max_product * WRAP64_LOW_PRODUCT_DIAGONALS * k_block_size > UINT32_MAX:
                        self._error("uint32 diagonal accumulator contract exceeds uint32 range")
            elif safety.get("k_block_cap") != 0:
                self._error("non-int32 accumulator captures must use k_block_cap=0")
            if safety.get("safe_for_k_block") is not True:
                self._error("non-int32 accumulator captures must set safe_for_k_block=true")
        autotune_key = metadata.get("autotune_key")
        if isinstance(autotune_key, str):
            normalized_key = f";{autotune_key};"
            selected_backend = self.data.get("backend_selected")
            expected_target_id: str | None = None
            if selected_backend in HIP_RESIDENT_BACKENDS:
                device = self.data.get("device")
                target_id = device.get("gcn_arch") if isinstance(device, dict) else None
                if _has_concrete_gpu_target_id(target_id):
                    expected_target_id = str(target_id)
            elif selected_backend in BACKEND_SELECTED_VALUES:
                expected_target_id = "cpu"
            if expected_target_id is not None and f";target_id={expected_target_id};" not in normalized_key:
                self._error(f"backend_metadata.autotune_key must include target_id={expected_target_id}")
            required_key_fields = {
                "accumulator_type": safety.get("accumulator_type"),
                "accumulator_signedness": safety.get("signedness"),
                "accumulator_modulus_policy": safety.get("modulus_policy"),
                "k_block_size": safety.get("k_block_size"),
                "k_block_cap": safety.get("k_block_cap"),
            }
            for key, value in required_key_fields.items():
                if f";{key}={value};" not in normalized_key:
                    self._error(f"backend_metadata.autotune_key must include {key}={value}")
            schedule = self.data.get("schedule_metadata")
            if isinstance(schedule, dict) and _is_int(schedule.get("zero_row_col_product_count")):
                if schedule.get("zero_row_col_product_count") > 0:
                    for key, value in {
                        "schedule_flags": schedule.get("flags"),
                        "zero_a_rows": schedule.get("zero_a_row_proof_count"),
                        "zero_b_cols": schedule.get("zero_b_col_proof_count"),
                        "zero_row_col_products": schedule.get("zero_row_col_product_count"),
                    }.items():
                        if f";{key}={value};" not in normalized_key:
                            self._error(f"backend_metadata.autotune_key must include {key}={value}")

    def _validate_nonnegative_ints(self) -> None:
        for key in [
            "bound",
            "prefix",
            "tile_m",
            "tile_n",
            "k_block_size",
            "seed",
            "warmups",
            "repeats",
            "checksum_u64",
            "output_ld_padding",
        ]:
            value = self.data.get(key)
            if _is_int(value) and value < 0:
                self._error(f"{key} must be nonnegative")
        for key in ["m", "n", "k"]:
            value = self.data.get(key)
            if _is_int(value) and value <= 0:
                self._error(f"{key} must be positive")
        repeats = self.data.get("repeats")
        if _is_int(repeats) and repeats <= 0:
            self._error("repeats must be positive")

    def _validate_output_layout_metadata(self, metadata: dict[str, Any]) -> None:
        present = any(key in self.data for key in ["output_logical_ld", "output_ld_padding"]) or any(
            key in metadata
            for key in [
                "benchmark_output_destination_layout",
                "benchmark_output_logical_ld",
                "benchmark_output_ld_padding",
            ]
        )
        if not present:
            return

        n = self.data.get("n")
        output_ld = self.data.get("output_logical_ld")
        padding = self.data.get("output_ld_padding")
        if not _is_int(output_ld) or output_ld <= 0:
            self._error("output_logical_ld must be a positive integer")
            return
        if not _is_int(padding) or padding < 0:
            self._error("output_ld_padding must be a nonnegative integer")
            return
        if _is_int(n) and output_ld != n + padding:
            self._error("output_logical_ld must equal n + output_ld_padding")

        layout = metadata.get("benchmark_output_destination_layout")
        if layout not in OUTPUT_DESTINATION_LAYOUTS:
            self._error(
                f"timing_metadata.benchmark_output_destination_layout must be one of {sorted(OUTPUT_DESTINATION_LAYOUTS)}"
            )
        else:
            expected_layout = "contiguous_row_major" if padding == 0 else "padded_row_major"
            if layout != expected_layout:
                self._error(f"timing_metadata.benchmark_output_destination_layout must be {expected_layout}")
        if metadata.get("benchmark_output_logical_ld") != output_ld:
            self._error("timing_metadata.benchmark_output_logical_ld must match output_logical_ld")
        if metadata.get("benchmark_output_ld_padding") != padding:
            self._error("timing_metadata.benchmark_output_ld_padding must match output_ld_padding")
        staging_policy = metadata.get("direct_hip_export_staging_policy")
        if staging_policy not in DIRECT_HIP_EXPORT_STAGING_POLICIES:
            self._error(
                "timing_metadata.direct_hip_export_staging_policy must be one of "
                f"{sorted(DIRECT_HIP_EXPORT_STAGING_POLICIES)}"
            )

    def _validate_hip_toolchain(self) -> None:
        toolchain = self._require("hip_toolchain", "dict")
        if not isinstance(toolchain, dict):
            return
        enabled = toolchain.get("enabled")
        if not isinstance(enabled, bool):
            self._error("hip_toolchain.enabled must be a boolean")
        for key in ["hip_root", "hipcc_path", "hipcc_version", "hip_sdk_or_rocm_version", "version_source"]:
            value = toolchain.get(key)
            if value is not None and not isinstance(value, str):
                self._error(f"hip_toolchain.{key} must be a string or null")
        if enabled is False:
            for key in ["hipcc_path", "hipcc_version", "version_source"]:
                if toolchain.get(key) is not None:
                    self._error(f"hip_toolchain.{key} must be null when hip_toolchain.enabled is false")
        if self.data.get("backend_selected") in HIP_RESIDENT_BACKENDS:
            if enabled is not True:
                self._error("HIP backend captures must set hip_toolchain.enabled=true")
            for key in ["hip_root", "hipcc_path", "hipcc_version", "version_source"]:
                value = toolchain.get(key)
                if not isinstance(value, str) or not value:
                    self._error(f"HIP backend captures must include nonempty hip_toolchain.{key}")
            if toolchain.get("version_source") != "hipcc --version":
                self._error("HIP backend captures must use hip_toolchain.version_source=hipcc --version")

    def _validate_nested_metadata(self) -> None:
        compiler = self._require("compiler", "dict")
        if isinstance(compiler, dict):
            for key in ["id", "version"]:
                if not isinstance(compiler.get(key), str):
                    self._error(f"compiler.{key} must be a string")
        self._validate_hip_toolchain()
        device = self._require("device", "dict")
        if isinstance(device, dict):
            for key in ["device_id", "hip_available", "hip_runtime_version", "hip_driver_version", "global_mem_bytes"]:
                if not _is_int(device.get(key)):
                    self._error(f"device.{key} must be an integer")
            for key in ["name", "gcn_arch"]:
                if not isinstance(device.get(key), str):
                    self._error(f"device.{key} must be a string")
            if self.data.get("backend_selected") in HIP_RESIDENT_BACKENDS:
                if not _has_concrete_gpu_target_id(device.get("gcn_arch")):
                    self._error("HIP backend captures must include non-placeholder device.gcn_arch")
                if device.get("hip_available") != 1:
                    self._error("HIP backend captures must use device.hip_available=1")
                device_id = device.get("device_id")
                if _is_int(device_id) and device_id < 0:
                    self._error("HIP backend captures must use a nonnegative device.device_id")
        metadata = self._require("timing_metadata", "dict")
        if isinstance(metadata, dict):
            if metadata.get("unit") != "microseconds":
                self._error("timing_metadata.unit must be microseconds")
            for key in ["source", "source_scope", "gpu_event_timing_reason"]:
                if not isinstance(metadata.get(key), str):
                    self._error(f"timing_metadata.{key} must be a string")
            if "gpu_event_timing_status" in metadata and not isinstance(metadata.get("gpu_event_timing_status"), str):
                self._error("timing_metadata.gpu_event_timing_status must be a string")
            if not isinstance(metadata.get("gpu_event_timing"), bool):
                self._error("timing_metadata.gpu_event_timing must be a boolean")
            self._validate_output_layout_metadata(metadata)
            metadata_mode = metadata.get("benchmark_execution_mode")
            if metadata_mode is not None:
                if metadata_mode not in BENCHMARK_EXECUTION_MODES:
                    self._error(
                        f"timing_metadata.benchmark_execution_mode must be one of {sorted(BENCHMARK_EXECUTION_MODES)}"
                    )
                elif self.data.get("benchmark_execution_mode") is not None and metadata_mode != self.data.get(
                    "benchmark_execution_mode"
                ):
                    self._error("timing_metadata.benchmark_execution_mode must match benchmark_execution_mode")
            phase_order = metadata.get("phase_order")
            expected_phases = self._timing_phases()
            if phase_order != expected_phases:
                self._error(f"timing_metadata.phase_order must be {expected_phases}")
            self._validate_phase_availability(metadata)
            gpu_phase_order = metadata.get("gpu_event_phase_order")
            if gpu_phase_order is not None:
                if not isinstance(gpu_phase_order, list) or not all(isinstance(item, str) for item in gpu_phase_order):
                    self._error("timing_metadata.gpu_event_phase_order must be an array of strings")
                elif len(set(gpu_phase_order)) != len(gpu_phase_order):
                    self._error("timing_metadata.gpu_event_phase_order must not contain duplicates")

    def _validate_counter_snapshot(self, label: str, value: Any) -> None:
        if not isinstance(value, dict):
            self._error(f"{label} must be an object")
            return
        for key in ["allocate_calls", "free_calls", "allocated_bytes"]:
            item = value.get(key)
            if not _is_int(item) or item < 0:
                self._error(f"{label}.{key} must be a nonnegative integer")

    def _validate_helper_lane_metadata(self) -> None:
        metadata = self.data.get("timing_metadata")
        if isinstance(metadata, dict):
            pack_layout = metadata.get("pack_layout")
            if pack_layout is not None and pack_layout not in PACK_LAYOUTS:
                self._error(f"timing_metadata.pack_layout must be one of {sorted(PACK_LAYOUTS)}")
            fusion_mode = metadata.get("fusion_mode")
            if fusion_mode is not None and fusion_mode not in FUSION_MODES:
                self._error(f"timing_metadata.fusion_mode must be one of {sorted(FUSION_MODES)}")
            residue_width = metadata.get("residue_group_width")
            if residue_width is not None and (not _is_int(residue_width) or residue_width <= 0):
                self._error("timing_metadata.residue_group_width must be a positive integer")
            residue_layout = metadata.get("residue_group_layout")
            if residue_layout is not None and residue_layout not in RESIDUE_GROUP_LAYOUTS:
                self._error(f"timing_metadata.residue_group_layout must be one of {sorted(RESIDUE_GROUP_LAYOUTS)}")
            reducer = metadata.get("generated_reducer_identity")
            if reducer is not None:
                if not isinstance(reducer, str) or GENERATED_REDUCER_RE.match(reducer) is None:
                    self._error("timing_metadata.generated_reducer_identity must be a declared reducer identity")
                if reducer == "generic" or reducer == "direct_hip_generic_reducer":
                    self._error("timing_metadata.generated_reducer_identity must not use stale generic identities")
            if fusion_mode == "residue_channel_width3_experimental_benchmark_only":
                if self._benchmark_execution_mode() != "residue_channel_fusion_native_inputs":
                    self._error("residue-channel fusion metadata requires benchmark_execution_mode=residue_channel_fusion_native_inputs")
                if pack_layout != "native_i8_row_major_residue_channel_width3":
                    self._error("residue-channel fusion captures must use pack_layout=native_i8_row_major_residue_channel_width3")
                if residue_width != 3:
                    self._error("residue-channel fusion captures must use residue_group_width=3")

            batch_enabled = metadata.get("host_api_batch_enabled")
            batch_size = metadata.get("host_api_batch_size")
            if batch_enabled is not None and not isinstance(batch_enabled, bool):
                self._error("timing_metadata.host_api_batch_enabled must be a boolean")
            if batch_size is not None and (not _is_int(batch_size) or batch_size <= 0):
                self._error("timing_metadata.host_api_batch_size must be a positive integer")

        plan_packing = self.data.get("plan_packing")
        if plan_packing is not None:
            if not isinstance(plan_packing, dict):
                self._error("plan_packing must be an object or null")
            else:
                for key in ["source", "backend", "semantics", "input_domain_name", "output_domain_name", "next_op_hint"]:
                    if not isinstance(plan_packing.get(key), str):
                        self._error(f"plan_packing.{key} must be a string")
                for key in [
                    "uses_resident_matrix_inputs",
                    "uses_transient_pack_workspace",
                    "uses_matrix_engine_pack_layout",
                    "reusable_prepack_cache_available",
                    "production_prepack_cache_available",
                    "output_host_current",
                    "output_device_current",
                ]:
                    if not isinstance(plan_packing.get(key), bool):
                        self._error(f"plan_packing.{key} must be a boolean")
                for key in [
                    "next_op_flags",
                    "a_pack_workspace_bytes",
                    "b_pack_workspace_bytes",
                    "accumulator_workspace_bytes",
                    "library_workspace_bytes",
                    "total_transient_workspace_bytes",
                ]:
                    value = plan_packing.get(key)
                    if not _is_int(value) or value < 0:
                        self._error(f"plan_packing.{key} must be a nonnegative integer")

        plan_lowering = self.data.get("plan_lowering")
        if plan_lowering is not None:
            if not isinstance(plan_lowering, dict):
                self._error("plan_lowering must be an object or null")
            else:
                for key in [
                    "source",
                    "operation",
                    "semantic_contract",
                    "backend_family",
                    "input_domain",
                    "output_domain",
                    "desired_output",
                    "schedule_strategy",
                    "packing_strategy",
                    "reuse_strategy",
                    "conversion_strategy",
                    "lowering_path",
                ]:
                    if not isinstance(plan_lowering.get(key), str):
                        self._error(f"plan_lowering.{key} must be a string")
                for key in [
                    "final_export_available",
                    "rns_continuation_available",
                    "native_continuation_available",
                    "native_to_rns_available",
                    "reusable_b_prepack_available",
                ]:
                    if not isinstance(plan_lowering.get(key), bool):
                        self._error(f"plan_lowering.{key} must be a boolean")

        requested_next_op = self.data.get("requested_next_op")
        if requested_next_op is not None:
            if not isinstance(requested_next_op, dict):
                self._error("requested_next_op must be an object")
            else:
                if requested_next_op.get("requested") not in NEXT_OP_HINTS:
                    self._error(f"requested_next_op.requested must be one of {sorted(NEXT_OP_HINTS)}")
                if requested_next_op.get("resolved") not in NEXT_OP_HINTS - {"auto"}:
                    self._error(f"requested_next_op.resolved must be one of {sorted(NEXT_OP_HINTS - {'auto'})}")
                if requested_next_op.get("source") not in {"cli", "benchmark_default"}:
                    self._error("requested_next_op.source must be cli or benchmark_default")

        output_policy = self.data.get("output_policy")
        if output_policy is not None:
            if not isinstance(output_policy, dict):
                self._error("output_policy must be an object")
            else:
                if output_policy.get("destination_layout") not in OUTPUT_DESTINATION_LAYOUTS:
                    self._error(f"output_policy.destination_layout must be one of {sorted(OUTPUT_DESTINATION_LAYOUTS)}")
                if output_policy.get("destination_layout") != output_destination_layout(self.data.get("output_ld_padding", 0)):
                    self._error("output_policy.destination_layout must match output_ld_padding")
                for key in ["logical_ld", "ld_padding"]:
                    value = output_policy.get(key)
                    if not _is_int(value) or value < 0:
                        self._error(f"output_policy.{key} must be a nonnegative integer")
                if _is_int(self.data.get("output_logical_ld")) and output_policy.get("logical_ld") != self.data.get("output_logical_ld"):
                    self._error("output_policy.logical_ld must match output_logical_ld")
                if output_policy.get("ld_padding") != self.data.get("output_ld_padding"):
                    self._error("output_policy.ld_padding must match output_ld_padding")
                if output_policy.get("status_handling") not in STATUS_HANDLING:
                    self._error(f"output_policy.status_handling must be one of {sorted(STATUS_HANDLING)}")
                if not isinstance(output_policy.get("status_event_policy"), str):
                    self._error("output_policy.status_event_policy must be a string")

        if self._is_residue_current_chain_capture():
            if not isinstance(requested_next_op, dict):
                self._error("residue-current chain captures must declare requested_next_op")
            elif requested_next_op.get("resolved") != "rns-gemm":
                self._error("residue-current chain captures must declare requested_next_op.resolved=rns-gemm")
            if not isinstance(output_policy, dict):
                self._error("residue-current chain captures must declare output_policy")
            else:
                if output_policy.get("per_repeat_logical_export") is not False:
                    self._error("residue-current chain captures must declare output_policy.per_repeat_logical_export=false")
                if output_policy.get("final_checksum_export_after_repeats") is not True:
                    self._error(
                        "residue-current chain captures must declare "
                        "output_policy.final_checksum_export_after_repeats=true"
                    )
                if output_policy.get("status_handling") != "structurally_elided":
                    self._error(
                        "residue-current chain captures must declare "
                        "output_policy.status_handling=structurally_elided"
                    )

        target_variant = self.data.get("target_variant")
        helper_lane_surface_present = any(
            key in self.data
            for key in [
                "plan_packing",
                "plan_lowering",
                "requested_next_op",
                "output_policy",
                "auto_selector",
                "device_allocation",
                "reuse_contract",
                "exact_output_contract",
                "export_variant",
                "reconstruction_variant",
                "modulus_set",
                "residue_count_policy",
                "tile_shape_variant",
                "grouped_dispatch",
                "adaptive_grouped_scheduler",
                "hip_graph_replay",
                "resident_lifetime",
                "workspace_arena",
                "streaming_overlap",
                "workload_proxy",
                "release_gate",
                "verification_amortization",
            ]
        )
        if (
            target_variant is None
            and helper_lane_surface_present
            and self.data.get("backend_selected") in HIP_RESIDENT_BACKENDS
        ):
            self._error("HIP helper-lane captures must include target_variant")
        if target_variant is not None:
            if not isinstance(target_variant, dict):
                self._error("target_variant must be an object")
            else:
                target_id = target_variant.get("target_id")
                namespace = target_variant.get("target_namespace")
                if not isinstance(target_id, str):
                    self._error("target_variant.target_id must be a string")
                if namespace not in TARGET_NAMESPACES:
                    self._error(f"target_variant.target_namespace must be one of {sorted(TARGET_NAMESPACES)}")
                if self.data.get("backend_selected") in HIP_RESIDENT_BACKENDS:
                    if not _has_concrete_gpu_target_id(target_id):
                        self._error("HIP captures with target_variant must include concrete target_variant.target_id")
                    if namespace == "unknown":
                        self._error("HIP captures with target_variant must use a concrete target_namespace")
                for key in ["review_group_key", "configured_amdgpu_targets"]:
                    if not isinstance(target_variant.get(key), str):
                        self._error(f"target_variant.{key} must be a string")

        auto_selector = self.data.get("auto_selector")
        if auto_selector is not None:
            if not isinstance(auto_selector, dict):
                self._error("auto_selector must be an object")
            else:
                for key in ["source", "requested_backend", "selected_backend", "fallback_reason"]:
                    if not isinstance(auto_selector.get(key), str):
                        self._error(f"auto_selector.{key} must be a string")
                selected_key = auto_selector.get("selected_key")
                if selected_key is not None and not isinstance(selected_key, str):
                    self._error("auto_selector.selected_key must be a string or null")
                vocabulary = auto_selector.get("rejection_reason_vocabulary")
                if not isinstance(vocabulary, list) or set(vocabulary) != SELECTOR_REJECTION_REASONS:
                    self._error("auto_selector.rejection_reason_vocabulary must match fixed rejection reasons")
                rejected = auto_selector.get("rejected_candidates")
                if rejected is not None:
                    if not isinstance(rejected, list):
                        self._error("auto_selector.rejected_candidates must be an array")
                    else:
                        for index, item in enumerate(rejected):
                            if not isinstance(item, dict):
                                self._error(f"auto_selector.rejected_candidates[{index}] must be an object")
                                continue
                            if item.get("reason") not in SELECTOR_REJECTION_REASONS:
                                self._error(
                                    f"auto_selector.rejected_candidates[{index}].reason must be a fixed rejection reason"
                                )

        device_allocation = self.data.get("device_allocation")
        if device_allocation is not None:
            if not isinstance(device_allocation, dict):
                self._error("device_allocation must be an object")
            else:
                if not isinstance(device_allocation.get("tracking_available"), bool):
                    self._error("device_allocation.tracking_available must be a boolean")
                for key in ["source", "setup_scope", "source_version_inputs"]:
                    if not isinstance(device_allocation.get(key), str):
                        self._error(f"device_allocation.{key} must be a string")
                self._validate_counter_snapshot("device_allocation.before", device_allocation.get("before"))
                if device_allocation.get("after_warmups") is not None:
                    self._validate_counter_snapshot("device_allocation.after_warmups", device_allocation.get("after_warmups"))
                self._validate_counter_snapshot("device_allocation.after_repeats", device_allocation.get("after_repeats"))
                self._validate_counter_snapshot("device_allocation.setup_delta", device_allocation.get("setup_delta"))
                if device_allocation.get("measured_repeat_delta") is not None:
                    self._validate_counter_snapshot(
                        "device_allocation.measured_repeat_delta",
                        device_allocation.get("measured_repeat_delta"),
                    )

    def _validate_starfoundry_metadata(self) -> None:
        reuse = self.data.get("reuse_contract")
        if reuse is not None:
            if not isinstance(reuse, dict):
                self._error("reuse_contract must be an object")
            else:
                if not isinstance(reuse.get("enabled"), bool):
                    self._error("reuse_contract.enabled must be a boolean")
                if reuse.get("operand_role") not in REUSE_OPERAND_ROLES:
                    self._error(f"reuse_contract.operand_role must be one of {sorted(REUSE_OPERAND_ROLES)}")
                for key in [
                    "source_version_inputs",
                    "setup_scope",
                    "output_domain",
                    "next_op",
                    "target_fingerprint",
                    "backend_fingerprint",
                    "workspace_fingerprint",
                ]:
                    if not isinstance(reuse.get(key), str):
                        self._error(f"reuse_contract.{key} must be a string")
                if reuse.get("output_domain") not in OUTPUT_CONTRACT_DOMAINS:
                    self._error(f"reuse_contract.output_domain must be one of {sorted(OUTPUT_CONTRACT_DOMAINS)}")
                setup_cost = reuse.get("setup_cost_us")
                if setup_cost is not None and (not _is_number(setup_cost) or setup_cost < 0):
                    self._error("reuse_contract.setup_cost_us must be a nonnegative number or null")
                repeats = reuse.get("measured_repeat_count")
                if not _is_int(repeats) or repeats <= 0:
                    self._error("reuse_contract.measured_repeat_count must be a positive integer")
                break_even = reuse.get("break_even_repeat_count")
                if break_even is not None and (not _is_number(break_even) or break_even <= 0):
                    self._error("reuse_contract.break_even_repeat_count must be positive or null")
                if not isinstance(reuse.get("promotion_eligible"), bool):
                    self._error("reuse_contract.promotion_eligible must be a boolean")
                invalidation = reuse.get("invalidation_reasons")
                if not isinstance(invalidation, list) or not all(isinstance(item, str) for item in invalidation):
                    self._error("reuse_contract.invalidation_reasons must be an array of strings")

        exact_output = self.data.get("exact_output_contract")
        if exact_output is not None:
            if not isinstance(exact_output, dict):
                self._error("exact_output_contract must be an object")
            else:
                if exact_output.get("requested_final_output") not in OUTPUT_CONTRACT_DOMAINS:
                    self._error(
                        f"exact_output_contract.requested_final_output must be one of {sorted(OUTPUT_CONTRACT_DOMAINS)}"
                    )
                limb_count = exact_output.get("limb_count")
                if self.data.get("semantics") in {"exact_wide_signed", "exact_wide_unsigned"}:
                    if not _is_int(limb_count) or not 1 <= limb_count <= 32:
                        self._error("exact_output_contract.limb_count must be in [1, 32] for exact-wide captures")
                elif limb_count is not None:
                    self._error("exact_output_contract.limb_count must be null outside exact-wide captures")
                if exact_output.get("status_policy") not in STATUS_HANDLING:
                    self._error(f"exact_output_contract.status_policy must be one of {sorted(STATUS_HANDLING)}")
                if exact_output.get("output_domain_after_measured_repeats") not in OUTPUT_CONTRACT_DOMAINS:
                    self._error("exact_output_contract.output_domain_after_measured_repeats must be a known output domain")
                if not isinstance(exact_output.get("final_checksum_export_after_repeats"), bool):
                    self._error("exact_output_contract.final_checksum_export_after_repeats must be a boolean")

        export_variant = self.data.get("export_variant")
        if export_variant is not None:
            if not isinstance(export_variant, dict):
                self._error("export_variant must be an object")
            else:
                if not isinstance(export_variant.get("name"), str) or not export_variant.get("name"):
                    self._error("export_variant.name must be a nonempty string")
                for key in ["source", "status_policy", "constants_placement"]:
                    if not isinstance(export_variant.get(key), str):
                        self._error(f"export_variant.{key} must be a string")
                if export_variant.get("status_policy") not in STATUS_HANDLING:
                    self._error(f"export_variant.status_policy must be one of {sorted(STATUS_HANDLING)}")
                limb_count = export_variant.get("limb_count")
                if limb_count is not None and (not _is_int(limb_count) or not 1 <= limb_count <= 32):
                    self._error("export_variant.limb_count must be in [1, 32] or null")
                if not isinstance(export_variant.get("promotion_eligible"), bool):
                    self._error("export_variant.promotion_eligible must be a boolean")
                if export_variant.get("name") != "default" and export_variant.get("promotion_eligible") is True:
                    self._error("experimental export_variant captures must set promotion_eligible=false")
                blocker = export_variant.get("promotion_blocker")
                if export_variant.get("name") != "default" and not isinstance(blocker, str):
                    self._error("experimental export_variant captures must declare promotion_blocker")

        reconstruction = self.data.get("reconstruction_variant")
        if reconstruction is not None:
            if not isinstance(reconstruction, dict):
                self._error("reconstruction_variant must be an object")
            else:
                for key in ["name", "family", "controller"]:
                    if not isinstance(reconstruction.get(key), str) or not reconstruction.get(key):
                        self._error(f"reconstruction_variant.{key} must be a nonempty string")
                prefix_count = reconstruction.get("prefix_count")
                if not _is_int(prefix_count) or prefix_count < 0:
                    self._error("reconstruction_variant.prefix_count must be a nonnegative integer")
                if not isinstance(reconstruction.get("promotion_eligible"), bool):
                    self._error("reconstruction_variant.promotion_eligible must be a boolean")
                if reconstruction.get("name") != "default_garner" and reconstruction.get("promotion_eligible") is True:
                    self._error("experimental reconstruction_variant captures must set promotion_eligible=false")
                blocker = reconstruction.get("promotion_blocker")
                if reconstruction.get("name") != "default_garner" and not isinstance(blocker, str):
                    self._error("experimental reconstruction_variant captures must declare promotion_blocker")

        modulus_set = self.data.get("modulus_set")
        if modulus_set is not None:
            if not isinstance(modulus_set, dict):
                self._error("modulus_set must be an object")
            else:
                name = modulus_set.get("name")
                if not isinstance(name, str) or (name != "default" and not name.startswith("experimental:")):
                    self._error("modulus_set.name must be default or experimental:NAME")
                if not isinstance(modulus_set.get("experimental"), bool):
                    self._error("modulus_set.experimental must be a boolean")
                for key in ["source", "execution_ladder", "pairwise_coprime_proof", "reducer_cost_hint"]:
                    if not isinstance(modulus_set.get(key), str):
                        self._error(f"modulus_set.{key} must be a string")
                for key in ["product_bits", "prefix_count"]:
                    value = modulus_set.get(key)
                    if not _is_int(value) or value < 0:
                        self._error(f"modulus_set.{key} must be a nonnegative integer")
                if name == "default" and modulus_set.get("experimental") is not False:
                    self._error("modulus_set.experimental must be false for default")
                if name != "default" and not isinstance(modulus_set.get("cache_promotion_blocker"), str):
                    self._error("experimental modulus_set captures must declare cache_promotion_blocker")

        residue_policy = self.data.get("residue_count_policy")
        if residue_policy is not None:
            if not isinstance(residue_policy, dict):
                self._error("residue_count_policy must be an object")
            else:
                for key in ["policy", "autotune_scope"]:
                    if not isinstance(residue_policy.get(key), str):
                        self._error(f"residue_count_policy.{key} must be a string")
                for key in ["requested_prefix", "selected_prefix", "minimum_range_prefix", "redundant_residue_count"]:
                    value = residue_policy.get(key)
                    if not _is_int(value) or value < 0:
                        self._error(f"residue_count_policy.{key} must be a nonnegative integer")
                if _is_int(self.data.get("selected_prefix")) and residue_policy.get("selected_prefix") != self.data.get("selected_prefix"):
                    self._error("residue_count_policy.selected_prefix must match selected_prefix")
                if _is_int(self.data.get("requested_max_prefix")) and residue_policy.get("requested_prefix") != self.data.get("requested_max_prefix"):
                    self._error("residue_count_policy.requested_prefix must match requested_max_prefix")

        tile_variant = self.data.get("tile_shape_variant")
        if tile_variant is not None:
            if not isinstance(tile_variant, dict):
                self._error("tile_shape_variant must be an object")
            else:
                if not isinstance(tile_variant.get("name"), str) or not tile_variant.get("name"):
                    self._error("tile_shape_variant.name must be a nonempty string")
                for key in ["resource_report_key", "shape_family_bucket", "stale_kernel_rejection"]:
                    if not isinstance(tile_variant.get(key), str):
                        self._error(f"tile_shape_variant.{key} must be a string")
                for key in ["tile_m", "tile_n", "tile_k"]:
                    value = tile_variant.get(key)
                    if not _is_int(value) or value <= 0:
                        self._error(f"tile_shape_variant.{key} must be a positive integer")
                if tile_variant.get("tile_m") != self.data.get("tile_m"):
                    self._error("tile_shape_variant.tile_m must match tile_m")
                if tile_variant.get("tile_n") != self.data.get("tile_n"):
                    self._error("tile_shape_variant.tile_n must match tile_n")
                if _is_int(self.data.get("k_block_size")) and tile_variant.get("tile_k") != self.data.get("k_block_size"):
                    self._error("tile_shape_variant.tile_k must match k_block_size")
                identity = tile_variant.get("selected_kernel_identity")
                selected_kernel = self.data.get("selected_kernel")
                if identity is not None and identity != selected_kernel:
                    self._error("tile_shape_variant.selected_kernel_identity must match selected_kernel")

        grouped = self.data.get("grouped_dispatch")
        if grouped is not None:
            if not isinstance(grouped, dict):
                self._error("grouped_dispatch must be an object")
            else:
                if not isinstance(grouped.get("requested"), bool):
                    self._error("grouped_dispatch.requested must be a boolean")
                task_count = grouped.get("task_count")
                if not _is_int(task_count) or task_count <= 0:
                    self._error("grouped_dispatch.task_count must be a positive integer")
                for key in ["descriptor_identity", "source_hash", "output_hash", "setup_scope"]:
                    if not isinstance(grouped.get(key), str):
                        self._error(f"grouped_dispatch.{key} must be a string")
                if grouped.get("capture_status") not in GROUPED_DISPATCH_STATUSES:
                    self._error(f"grouped_dispatch.capture_status must be one of {sorted(GROUPED_DISPATCH_STATUSES)}")
                if task_count and task_count > 1 and grouped.get("requested") is not True:
                    self._error("grouped_dispatch.requested must be true when task_count > 1")
                if task_count and task_count > 1 and grouped.get("promotion_eligible") is not False:
                    self._error("grouped_dispatch task_count > 1 must set promotion_eligible=false")

        graph = self.data.get("hip_graph_replay")
        if graph is not None:
            if not isinstance(graph, dict):
                self._error("hip_graph_replay must be an object")
            else:
                if not isinstance(graph.get("requested"), bool):
                    self._error("hip_graph_replay.requested must be a boolean")
                for key in ["descriptor_identity", "plan_identity", "setup_scope"]:
                    if not isinstance(graph.get(key), str):
                        self._error(f"hip_graph_replay.{key} must be a string")
                if graph.get("capture_status") not in GRAPH_REPLAY_STATUSES:
                    self._error(f"hip_graph_replay.capture_status must be one of {sorted(GRAPH_REPLAY_STATUSES)}")
                if graph.get("requested") is True and graph.get("promotion_eligible") is not False:
                    self._error("hip_graph_replay requested captures must set promotion_eligible=false")

        adaptive = self.data.get("adaptive_grouped_scheduler")
        if adaptive is not None:
            if not isinstance(adaptive, dict):
                self._error("adaptive_grouped_scheduler must be an object")
            else:
                if not isinstance(adaptive.get("requested"), bool):
                    self._error("adaptive_grouped_scheduler.requested must be a boolean")
                for key in ["strategy", "descriptor_identity", "selected_prefix_histogram"]:
                    if not isinstance(adaptive.get(key), str):
                        self._error(f"adaptive_grouped_scheduler.{key} must be a string")
                for key in ["group_count", "active_tile_count", "zero_tile_count"]:
                    value = adaptive.get(key)
                    if not _is_int(value) or value < 0:
                        self._error(f"adaptive_grouped_scheduler.{key} must be a nonnegative integer")
                if adaptive.get("capture_status") not in GROUPED_DISPATCH_STATUSES:
                    self._error(
                        f"adaptive_grouped_scheduler.capture_status must be one of {sorted(GROUPED_DISPATCH_STATUSES)}"
                    )
                if adaptive.get("requested") is True and adaptive.get("promotion_eligible") is not False:
                    self._error("adaptive_grouped_scheduler requested captures must set promotion_eligible=false")

        resident = self.data.get("resident_lifetime")
        if resident is not None:
            if not isinstance(resident, dict):
                self._error("resident_lifetime must be an object")
            else:
                if not isinstance(resident.get("enabled"), bool):
                    self._error("resident_lifetime.enabled must be a boolean")
                for key in [
                    "matrix_roles",
                    "source_version_policy",
                    "current_storage_state",
                    "output_domain",
                    "workspace_identity",
                    "stale_source_rejection",
                ]:
                    if not isinstance(resident.get(key), str):
                        self._error(f"resident_lifetime.{key} must be a string")
                if resident.get("output_domain") not in OUTPUT_CONTRACT_DOMAINS:
                    self._error("resident_lifetime.output_domain must be a known output domain")
                if resident.get("promotion_eligible") is not False:
                    self._error("resident_lifetime captures must set promotion_eligible=false")

        arena = self.data.get("workspace_arena")
        if arena is not None:
            if not isinstance(arena, dict):
                self._error("workspace_arena must be an object")
            else:
                if not isinstance(arena.get("enabled"), bool):
                    self._error("workspace_arena.enabled must be a boolean")
                for key in ["arena_identity", "source_version_policy", "stream_safety"]:
                    if not isinstance(arena.get(key), str):
                        self._error(f"workspace_arena.{key} must be a string")
                for key in ["size_bytes", "high_water_mark_bytes", "suballocation_count"]:
                    value = arena.get(key)
                    if not _is_int(value) or value < 0:
                        self._error(f"workspace_arena.{key} must be a nonnegative integer")
                if not isinstance(arena.get("measured_repeat_allocation_free"), bool):
                    self._error("workspace_arena.measured_repeat_allocation_free must be a boolean")
                if arena.get("promotion_eligible") is not False:
                    self._error("workspace_arena captures must set promotion_eligible=false")

        overlap = self.data.get("streaming_overlap")
        if overlap is not None:
            if not isinstance(overlap, dict):
                self._error("streaming_overlap must be an object")
            else:
                if not isinstance(overlap.get("requested"), bool):
                    self._error("streaming_overlap.requested must be a boolean")
                for key in ["pipeline", "buffering", "dependency_contract", "transfer_policy"]:
                    if not isinstance(overlap.get(key), str):
                        self._error(f"streaming_overlap.{key} must be a string")
                if overlap.get("capture_status") not in STREAMING_OVERLAP_STATUSES:
                    self._error(f"streaming_overlap.capture_status must be one of {sorted(STREAMING_OVERLAP_STATUSES)}")
                if overlap.get("requested") is True and overlap.get("promotion_eligible") is not False:
                    self._error("streaming_overlap requested captures must set promotion_eligible=false")

        proxy = self.data.get("workload_proxy")
        if proxy is not None:
            if not isinstance(proxy, dict):
                self._error("workload_proxy must be an object")
            else:
                if not isinstance(proxy.get("enabled"), bool):
                    self._error("workload_proxy.enabled must be a boolean")
                if proxy.get("family") not in WORKLOAD_PROXY_FAMILIES:
                    self._error(f"workload_proxy.family must be one of {sorted(WORKLOAD_PROXY_FAMILIES)}")
                for key in ["label", "tower_role", "reuse_profile", "transform_role", "output_domain_requirement"]:
                    if not isinstance(proxy.get(key), str):
                        self._error(f"workload_proxy.{key} must be a string")
                if proxy.get("output_domain_requirement") not in OUTPUT_CONTRACT_DOMAINS:
                    self._error("workload_proxy.output_domain_requirement must be a known output domain")
                if proxy.get("compatibility_claim") is not False:
                    self._error("workload_proxy.compatibility_claim must be false")

        release_gate = self.data.get("release_gate")
        if release_gate is not None:
            if not isinstance(release_gate, dict):
                self._error("release_gate must be an object")
            else:
                if not isinstance(release_gate.get("name"), str) or not release_gate.get("name"):
                    self._error("release_gate.name must be a nonempty string")
                if not isinstance(release_gate.get("requested"), bool):
                    self._error("release_gate.requested must be a boolean")
                for key in ["classification_tier", "cpu_reference_policy", "memory_cap_policy", "resume_policy"]:
                    if not isinstance(release_gate.get(key), str):
                        self._error(f"release_gate.{key} must be a string")
                if release_gate.get("review_status") not in RELEASE_GATE_REVIEW_STATUSES:
                    self._error(f"release_gate.review_status must be one of {sorted(RELEASE_GATE_REVIEW_STATUSES)}")
                if not isinstance(release_gate.get("cache_eligible"), bool):
                    self._error("release_gate.cache_eligible must be a boolean")
                blockers = release_gate.get("blockers")
                if not isinstance(blockers, list) or not all(isinstance(item, str) for item in blockers):
                    self._error("release_gate.blockers must be an array of strings")

        amortization = self.data.get("verification_amortization")
        if amortization is not None:
            if not isinstance(amortization, dict):
                self._error("verification_amortization must be an object")
            else:
                if not isinstance(amortization.get("enabled"), bool):
                    self._error("verification_amortization.enabled must be a boolean")
                for key in ["policy", "reused_reference_structure", "final_exact_comparison_status"]:
                    if not isinstance(amortization.get(key), str):
                        self._error(f"verification_amortization.{key} must be a string")
                if amortization.get("final_exact_comparison_required") is not True:
                    self._error("verification_amortization.final_exact_comparison_required must be true")
                if amortization.get("promotion_eligible") is not False:
                    self._error("verification_amortization captures must set promotion_eligible=false")

    def _validate_grouped_dispatch_metadata(self) -> None:
        grouped = self.data.get("grouped_dispatch")
        is_grouped = self._is_grouped_dispatch_capture()
        if grouped is None:
            if is_grouped:
                self._error("benchmark_grouped_dispatch_evidence captures must include grouped_dispatch metadata")
            return
        if not isinstance(grouped, dict):
            return

        task_count = grouped.get("task_count")
        if is_grouped:
            if self.data.get("benchmark") != "rns8_grouped_dispatch_persistent_resident":
                self._error(
                    "benchmark_grouped_dispatch_evidence captures must use "
                    "benchmark=rns8_grouped_dispatch_persistent_resident"
                )
            if self.data.get("backend_selected") != "hip-direct":
                self._error("benchmark_grouped_dispatch_evidence captures must use backend_selected=hip-direct")
            if grouped.get("requested") is not True:
                self._error("benchmark_grouped_dispatch_evidence captures must set grouped_dispatch.requested=true")
            if not _is_int(task_count) or task_count <= 1:
                self._error("benchmark_grouped_dispatch_evidence captures must use grouped_dispatch.task_count > 1")
            if grouped.get("capture_status") != "executed":
                self._error("benchmark_grouped_dispatch_evidence captures must set grouped_dispatch.capture_status=executed")
            if grouped.get("unsupported_reason") is not None:
                self._error("benchmark_grouped_dispatch_evidence captures must set grouped_dispatch.unsupported_reason=null")
            if grouped.get("promotion_eligible") is not False:
                self._error("benchmark_grouped_dispatch_evidence captures must set grouped_dispatch.promotion_eligible=false")
            if self.data.get("semantics") == "wrap_u64_mod_2_64":
                self._error("benchmark_grouped_dispatch_evidence captures must not use wrap_u64_mod_2_64")
            if self.data.get("reuse_packed_inputs") is not False:
                self._error("benchmark_grouped_dispatch_evidence captures must not use packed-input reuse")
            if self.data.get("pack_mode") != "per_repeat_repack":
                self._error("benchmark_grouped_dispatch_evidence captures must use pack_mode=per_repeat_repack")
            if self._residue_chain_length() != 1 or self._residue_output_mode() != "host_export":
                self._error("benchmark_grouped_dispatch_evidence captures must use host_export residue_chain_length=1")
            if self.data.get("bound_mode") != "global":
                self._error("benchmark_grouped_dispatch_evidence captures must use bound_mode=global")
            if self.data.get("bound_source") not in {None, "static_profile"}:
                self._error("benchmark_grouped_dispatch_evidence captures must use bound_source=static_profile")

            batch = self.data.get("host_api_batch")
            if not isinstance(batch, dict):
                self._error("benchmark_grouped_dispatch_evidence captures must include disabled host_api_batch metadata")
            elif batch.get("enabled") is not False or batch.get("batch_size") != 1:
                self._error("benchmark_grouped_dispatch_evidence captures must keep host_api_batch disabled")

            metadata = self.data.get("timing_metadata")
            if isinstance(metadata, dict):
                if metadata.get("grouped_dispatch_enabled") is not True:
                    self._error("timing_metadata.grouped_dispatch_enabled must be true for grouped dispatch captures")
                if _is_int(task_count) and metadata.get("grouped_dispatch_task_count") != task_count:
                    self._error(
                        "timing_metadata.grouped_dispatch_task_count must match grouped_dispatch.task_count"
                    )
                notes = metadata.get("phase_notes")
                if not isinstance(notes, dict):
                    self._error("benchmark_grouped_dispatch_evidence captures must include timing_metadata.phase_notes")
                else:
                    for phase in ["pack", "rns_gemm", "crt_export", "end_to_end"]:
                        note = notes.get(phase)
                        if not isinstance(note, str) or "aggregate" not in note or "grouped" not in note:
                            self._error(
                                f"benchmark_grouped_dispatch_evidence phase note {phase} "
                                "must describe aggregate grouped timing"
                            )

    def _validate_host_api_batch_metadata(self) -> None:
        batch = self.data.get("host_api_batch")
        is_batch = self._is_host_api_batch_capture()
        if batch is None:
            if is_batch:
                self._error("benchmark_host_api_batch captures must include host_api_batch metadata")
            return
        if not isinstance(batch, dict):
            self._error("host_api_batch must be an object")
            return

        enabled = batch.get("enabled")
        if not isinstance(enabled, bool):
            self._error("host_api_batch.enabled must be a boolean")
            return
        if enabled != is_batch:
            expected = "true" if is_batch else "false"
            self._error(f"host_api_batch.enabled must be {expected} for this benchmark_execution_mode")

        batch_size = batch.get("batch_size")
        tasks_per_repeat = batch.get("tasks_per_measured_repeat")
        total_tasks = batch.get("total_measured_tasks")
        repeats = self.data.get("repeats")
        for key, value in [
            ("batch_size", batch_size),
            ("tasks_per_measured_repeat", tasks_per_repeat),
            ("total_measured_tasks", total_tasks),
        ]:
            if not _is_int(value) or value <= 0:
                self._error(f"host_api_batch.{key} must be a positive integer")
        if _is_int(batch_size) and _is_int(tasks_per_repeat) and tasks_per_repeat != batch_size:
            self._error("host_api_batch.tasks_per_measured_repeat must equal host_api_batch.batch_size")
        if _is_int(batch_size) and _is_int(repeats) and _is_int(total_tasks) and total_tasks != batch_size * repeats:
            self._error("host_api_batch.total_measured_tasks must equal batch_size * repeats")

        expected_batch_size = 1 if not is_batch else None
        if expected_batch_size is not None and batch_size != expected_batch_size:
            self._error("non-batch captures must use host_api_batch.batch_size=1")
        if is_batch and _is_int(batch_size) and batch_size <= 1:
            self._error("benchmark_host_api_batch captures must use host_api_batch.batch_size > 1")

        expected_setup = (
            "one_shared_plan_per_capture_one_resident_matrix_workspace_triplet_per_task"
            if is_batch
            else "single_task_default_benchmark_mode"
        )
        expected_timing = (
            "aggregate_batch_totals_per_measured_repeat" if is_batch else "single_call_totals_per_measured_repeat"
        )
        expected_checksum = (
            "fnv1a_over_final_task_output_checksums" if is_batch else "single_final_output_checksum"
        )
        for key, expected in [
            ("setup_scope", expected_setup),
            ("timing_policy", expected_timing),
            ("checksum_policy", expected_checksum),
        ]:
            if batch.get(key) != expected:
                self._error(f"host_api_batch.{key} must be {expected}")

        metadata = self.data.get("timing_metadata")
        if isinstance(metadata, dict):
            if metadata.get("host_api_batch_enabled") is not enabled:
                self._error("timing_metadata.host_api_batch_enabled must match host_api_batch.enabled")
            if metadata.get("host_api_batch_size") != batch_size:
                self._error("timing_metadata.host_api_batch_size must match host_api_batch.batch_size")

        if is_batch:
            if self.data.get("benchmark") != "rns8_host_api_batch_persistent_resident":
                self._error("benchmark_host_api_batch captures must use benchmark=rns8_host_api_batch_persistent_resident")
            if self.data.get("semantics") == "wrap_u64_mod_2_64":
                self._error("benchmark_host_api_batch captures must not use wrap_u64_mod_2_64")
            if self.data.get("reuse_packed_inputs") is not False:
                self._error("benchmark_host_api_batch captures must not use packed-input reuse")
            if self.data.get("pack_mode") != "per_repeat_repack":
                self._error("benchmark_host_api_batch captures must use pack_mode=per_repeat_repack")
            if self._residue_chain_length() != 1 or self._residue_output_mode() != "host_export":
                self._error("benchmark_host_api_batch captures must use host_export residue_chain_length=1")
            if self.data.get("bound_mode") != "global":
                self._error("benchmark_host_api_batch captures must use bound_mode=global")
            if self.data.get("bound_source") not in {None, "static_profile"}:
                self._error("benchmark_host_api_batch captures must use bound_source=static_profile")
            metadata = self.data.get("timing_metadata")
            if isinstance(metadata, dict):
                notes = metadata.get("phase_notes")
                if not isinstance(notes, dict):
                    self._error("benchmark_host_api_batch captures must include timing_metadata.phase_notes")
                else:
                    for phase in ["pack", "rns_gemm", "crt_export", "end_to_end"]:
                        note = notes.get(phase)
                        if not isinstance(note, str) or "aggregate" not in note:
                            self._error(f"benchmark_host_api_batch phase note {phase} must describe aggregate batch timing")

        for field, source in [
            ("avg_pack_per_task_us", "avg_pack_us"),
            ("avg_rns_gemm_per_task_us", "avg_rns_gemm_us"),
            ("avg_crt_export_per_task_us", "avg_crt_export_us"),
            ("avg_end_to_end_per_task_us", "avg_end_to_end_us"),
        ]:
            value = self.data.get(field)
            source_value = self.data.get(source)
            if value is None:
                continue
            if not _is_number(value):
                self._error(f"{field} must be a finite number")
                continue
            denominator = batch_size
            denominator_name = "host_api_batch.batch_size"
            grouped = self.data.get("grouped_dispatch")
            if (
                self._is_grouped_dispatch_capture()
                and isinstance(grouped, dict)
                and _is_int(grouped.get("task_count"))
            ):
                denominator = grouped.get("task_count")
                denominator_name = "grouped_dispatch.task_count"
            if _is_number(source_value) and _is_int(denominator):
                expected = float(source_value) / float(denominator)
                if not _close(float(value), expected):
                    self._error(f"{field} must equal {source} / {denominator_name}")

    def _validate_hip_graph_replay_metadata(self) -> None:
        graph = self.data.get("hip_graph_replay")
        is_graph = self._is_hip_graph_replay_capture()
        if graph is None:
            if is_graph:
                self._error("hip_graph_replay_resident_rns_chain captures must include hip_graph_replay metadata")
            return
        if not isinstance(graph, dict):
            self._error("hip_graph_replay must be an object")
            return

        for key in ["requested", "available", "used"]:
            if not isinstance(graph.get(key), bool):
                self._error(f"hip_graph_replay.{key} must be a boolean")
        for key in ["status", "scope", "timing_policy", "setup_policy", "final_export_policy"]:
            if not isinstance(graph.get(key), str):
                self._error(f"hip_graph_replay.{key} must be a string")
        for key in [
            "capture_us",
            "instantiate_us",
            "graph_launches_per_measured_repeat",
            "total_graph_launches",
            "captured_chain_length",
        ]:
            if not _is_int(graph.get(key)) or graph.get(key) < 0:
                self._error(f"hip_graph_replay.{key} must be a nonnegative integer")
        caveat = graph.get("caveat")
        if caveat is not None and not isinstance(caveat, str):
            self._error("hip_graph_replay.caveat must be a string or null")

        metadata = self.data.get("timing_metadata")
        repeats = self.data.get("repeats")
        warmups = self.data.get("warmups")
        chain_length = self._residue_chain_length()

        if not is_graph:
            for key in ["requested", "available", "used"]:
                if graph.get(key) is not False:
                    self._error(f"non-graph captures must set hip_graph_replay.{key}=false")
            if graph.get("status") != "not_requested":
                self._error("non-graph captures must set hip_graph_replay.status=not_requested")
            if graph.get("scope") != "not_applicable":
                self._error("non-graph captures must set hip_graph_replay.scope=not_applicable")
            for key in [
                "capture_us",
                "instantiate_us",
                "graph_launches_per_measured_repeat",
                "total_graph_launches",
                "captured_chain_length",
            ]:
                if graph.get(key) != 0:
                    self._error(f"non-graph captures must set hip_graph_replay.{key}=0")
            for key in ["timing_policy", "setup_policy", "final_export_policy"]:
                if graph.get(key) != "not_applicable":
                    self._error(f"non-graph captures must set hip_graph_replay.{key}=not_applicable")
            if isinstance(metadata, dict):
                enabled = metadata.get("hip_graph_replay_enabled")
                if enabled is not None and enabled is not False:
                    self._error("non-graph captures must set timing_metadata.hip_graph_replay_enabled=false")
            return

        if self.data.get("benchmark") != "rns8_hip_graph_replay_resident_rns_chain":
            self._error("hip_graph_replay captures must use benchmark=rns8_hip_graph_replay_resident_rns_chain")
        if self.data.get("backend_selected") != "hip-direct" or self.data.get("backend_requested") != "hip-direct":
            self._error("hip_graph_replay captures must request and select backend hip-direct")
        if self.data.get("semantics") not in {"bounded_i64", "bounded_u64", "exact_wide_signed", "exact_wide_unsigned"}:
            self._error("hip_graph_replay captures must use bounded or exact-wide RNS semantics")
        if self.data.get("reuse_packed_inputs") is not True:
            self._error("hip_graph_replay captures must use reuse_packed_inputs=true")
        if self.data.get("pack_mode") != "prepacked_reuse":
            self._error("hip_graph_replay captures must use pack_mode=prepacked_reuse")
        if self.data.get("prepack_reuse_operands") != ["A", "B"]:
            self._error("hip_graph_replay captures must reuse operands A and B")
        if self.data.get("prepack_reuse_strategy") != "persistent_matrix_residency":
            self._error("hip_graph_replay captures must use prepack_reuse_strategy=persistent_matrix_residency")
        if chain_length <= 1 or self._residue_output_mode() != "residue_current_rns":
            self._error("hip_graph_replay captures must use residue-current chain output")
        if self.data.get("bound_mode") != "global":
            self._error("hip_graph_replay captures must use bound_mode=global")
        if self.data.get("bound_source") not in {None, "static_profile"}:
            self._error("hip_graph_replay captures must use static-profile bounds")

        for key in ["requested", "available", "used"]:
            if graph.get(key) is not True:
                self._error(f"hip_graph_replay captures must set hip_graph_replay.{key}=true")
        if graph.get("status") != "available":
            self._error("hip_graph_replay captures must set hip_graph_replay.status=available")
        if graph.get("scope") != "direct_hip_reused_inputs_residue_current_rns_chain":
            self._error(
                "hip_graph_replay captures must set hip_graph_replay.scope="
                "direct_hip_reused_inputs_residue_current_rns_chain"
            )
        if graph.get("graph_launches_per_measured_repeat") != 1:
            self._error("hip_graph_replay captures must launch one graph per measured repeat")
        if graph.get("captured_chain_length") != chain_length:
            self._error("hip_graph_replay.captured_chain_length must match residue_chain_length")
        if _is_int(repeats) and _is_int(warmups) and graph.get("total_graph_launches") != repeats + warmups:
            self._error("hip_graph_replay.total_graph_launches must equal warmups + repeats")
        if graph.get("timing_policy") != "raw_timings_us.rns_gemm_and_end_to_end_measure_one_hipGraphLaunch_plus_stream_sync":
            self._error("hip_graph_replay.timing_policy is stale or unsupported")
        if graph.get("setup_policy") != "A_B_prepack_before_capture_capture_and_instantiate_before_warmups":
            self._error("hip_graph_replay.setup_policy is stale or unsupported")
        if graph.get("final_export_policy") != "one_final_logical_export_after_measured_repeats_for_checksum_only":
            self._error("hip_graph_replay.final_export_policy is stale or unsupported")
        if not isinstance(caveat, str) or "resident Direct-HIP RNS GEMM launches only" not in caveat:
            self._error("hip_graph_replay.caveat must describe graph replay scope")

        if not isinstance(metadata, dict):
            return
        if metadata.get("hip_graph_replay_enabled") is not True:
            self._error("hip_graph_replay captures must set timing_metadata.hip_graph_replay_enabled=true")
        if metadata.get("hip_graph_replay_status") != graph.get("status"):
            self._error("timing_metadata.hip_graph_replay_status must match hip_graph_replay.status")
        if metadata.get("hip_graph_replay_scope") != graph.get("scope"):
            self._error("timing_metadata.hip_graph_replay_scope must match hip_graph_replay.scope")
        if metadata.get("hip_graph_capture_us") != graph.get("capture_us"):
            self._error("timing_metadata.hip_graph_capture_us must match hip_graph_replay.capture_us")
        if metadata.get("hip_graph_instantiate_us") != graph.get("instantiate_us"):
            self._error("timing_metadata.hip_graph_instantiate_us must match hip_graph_replay.instantiate_us")
        if metadata.get("hip_graph_total_launches") != graph.get("total_graph_launches"):
            self._error("timing_metadata.hip_graph_total_launches must match hip_graph_replay.total_graph_launches")
        if metadata.get("gpu_event_timing") is not False:
            self._error("hip_graph_replay captures must report gpu_event_timing=false")
        if metadata.get("gpu_event_timing_reason") != "hip_graph_replay_wall_clock_only":
            self._error("hip_graph_replay captures must report gpu_event_timing_reason=hip_graph_replay_wall_clock_only")
        if metadata.get("gpu_event_timing_status") != "not_requested_graph_replay":
            self._error("hip_graph_replay captures must report gpu_event_timing_status=not_requested_graph_replay")

    def _validate_backend_metadata(self) -> None:
        metadata = self._require("backend_metadata", "dict")
        if not isinstance(metadata, dict):
            return
        selected_backend = self.data.get("backend_selected")
        expected_source = (
            "rns8_bench_wrap64_rocwmma_candidate"
            if self._is_wrap64_rocwmma_candidate()
            else "rns8_bench_public_oneshot_api"
            if self._is_public_oneshot_capture()
            else "rns8_bench_residue_channel_fusion_path"
            if self._is_direct_hip_bounded_residue_channel_fusion_capture()
            else "rns8_bench_uniform_small_i8_ab_transient_path"
            if self._is_direct_hip_bounded_uniform_small_transient_capture()
            else "rns8_bench_uniform_small_i8_ab_reuse_b_path"
            if self._is_direct_hip_bounded_native_a_reuse_b_capture()
            and self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
            else "rns8_bench_uniform_small_i8_ab_reuse_a_path"
            if self._is_direct_hip_bounded_uniform_small_reuse_a_capture()
            else "rns8_bench_native_b_reuse_a_path"
            if self._is_direct_hip_bounded_native_b_reuse_a_u64_large_colpair_capture()
            else "rns8_bench_native_a_reuse_b_path"
            if self._is_direct_hip_bounded_native_a_reuse_b_capture()
            or self._is_direct_hip_finite_native_a_reuse_b_capture()
            else (
                "rns8_get_plan_backend_info"
                if self._is_vector_alu_runtime_capture()
                else "rns8_bench_vector_alu_baseline"
                if selected_backend == "hip-vector-alu-int64"
                else "rns8_get_plan_backend_info"
            )
        )
        if metadata.get("source") != expected_source:
            self._error(f"backend_metadata.source must be {expected_source}")
        selected_kernel = metadata.get("selected_kernel")
        if not isinstance(selected_kernel, str) or not selected_kernel:
            self._error("backend_metadata.selected_kernel must be a nonempty string")
        if self.data.get("selected_kernel") != selected_kernel:
            self._error("selected_kernel must match backend_metadata.selected_kernel")
        for key in [
            "accelerator_backend",
            "correctness_backend",
            "matrix_engine_backend",
            "compiled_kernel_available",
            "exact_differential_validated",
            "performance_validated",
        ]:
            if not isinstance(metadata.get(key), bool):
                self._error(f"backend_metadata.{key} must be a boolean")
        for key in [
            "accelerator_library",
            "accelerator_version",
            "capability_status",
            "epilogue_mode",
            "workspace_mode",
            "isa_evidence",
            "autotune_key",
        ]:
            value = metadata.get(key)
            if value is not None and not isinstance(value, str):
                self._error(f"backend_metadata.{key} must be a string or null")
        for key in ["capability_status", "epilogue_mode", "workspace_mode", "isa_evidence", "autotune_key"]:
            value = metadata.get(key)
            if not isinstance(value, str) or not value:
                self._error(f"backend_metadata.{key} must be a nonempty string")
        workspace_bytes = metadata.get("workspace_required_bytes")
        if not _is_int(workspace_bytes) or workspace_bytes < 0:
            self._error("backend_metadata.workspace_required_bytes must be a nonnegative integer")
        if selected_backend in CURRENT_CORRECTNESS_BACKENDS:
            if metadata.get("accelerator_backend") is not False:
                self._error("current correctness backends must set backend_metadata.accelerator_backend=false")
            if metadata.get("correctness_backend") is not True:
                self._error("current correctness backends must set backend_metadata.correctness_backend=true")
            if metadata.get("matrix_engine_backend") is not False:
                self._error("current correctness backends must set backend_metadata.matrix_engine_backend=false")
            if metadata.get("compiled_kernel_available") is not True:
                self._error("current correctness backends must set backend_metadata.compiled_kernel_available=true")
            if metadata.get("exact_differential_validated") is not True:
                self._error("current correctness backends must set backend_metadata.exact_differential_validated=true")
            if metadata.get("performance_validated") is not False:
                self._error("current correctness backends must set backend_metadata.performance_validated=false")
            if metadata.get("capability_status") != "implemented_correctness_backend":
                self._error("current correctness backends must use capability_status=implemented_correctness_backend")
        if selected_backend == "hipblaslt":
            expected = {
                "selected_kernel": "hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2",
                "accelerator_library": "hipBLASLt",
                "capability_status": "implemented_baseline_backend",
                "workspace_mode": "resident_device_buffers_with_hipblaslt_scratch",
                "isa_evidence": "hipblaslt_library_int8_matmul_specialized_reduce_251_255_256",
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    self._error(f"hipBLASLt captures must use backend_metadata.{key}={value}")
            bool_expected = {
                "accelerator_backend": True,
                "correctness_backend": True,
                "matrix_engine_backend": True,
                "compiled_kernel_available": True,
                "exact_differential_validated": True,
            }
            for key, value in bool_expected.items():
                if metadata.get(key) is not value:
                    self._error(f"hipBLASLt captures must use backend_metadata.{key}={value}")
            epilogue = metadata.get("epilogue_mode")
            if epilogue not in {
                "separate_i32_scratch_reduce_then_crt_export",
                "separate_i32_scratch_reduce_rns_output",
                "separate_i32_scratch_reduce_then_canonical_u8_export",
            }:
                self._error("hipBLASLt captures must report a separate INT32 scratch reduction epilogue")
        if selected_backend == "hip-vector-alu-int64":
            runtime_capture = self._is_vector_alu_runtime_capture()
            expected = {
                "accelerator_library": "HIP runtime",
                "capability_status": (
                    "implemented_native_bounded_vector_backend"
                    if runtime_capture
                    else "benchmark_only_vector_alu_baseline"
                ),
                "epilogue_mode": "direct_int64_export",
                "workspace_mode": (
                    "native_device_i64_u64_buffers" if runtime_capture else "benchmark_owned_device_buffers"
                ),
                "isa_evidence": "source_level_192bit_limb_accumulator_no_matrix_engine",
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    self._error(f"hip-vector-alu-int64 captures must use backend_metadata.{key}={value}")
            if metadata.get("selected_kernel") not in VECTOR_ALU_SELECTED_KERNELS:
                self._error("hip-vector-alu-int64 captures must report a known vector-ALU selected_kernel")
            gemv_n1 = self.data.get("n") == 1 and self.data.get("k", 0) >= 4096
            expected_kernel = (
                "hip_vector_alu_i64_gemv_n1_exact_192b_v1"
                if self.data.get("semantics") == "bounded_i64" and gemv_n1
                else "hip_vector_alu_i64_exact_192b_v1"
                if self.data.get("semantics") == "bounded_i64"
                else "hip_vector_alu_u64_gemv_n1_exact_192b_v1"
                if gemv_n1
                else "hip_vector_alu_u64_exact_192b_v1"
            )
            if metadata.get("selected_kernel") != expected_kernel:
                self._error(f"hip-vector-alu-int64 captures must use selected_kernel={expected_kernel}")
            bool_expected = {
                "accelerator_backend": False,
                "correctness_backend": True,
                "matrix_engine_backend": False,
                "compiled_kernel_available": True,
                "exact_differential_validated": True,
                "performance_validated": False,
            }
            for key, value in bool_expected.items():
                if metadata.get(key) is not value:
                    self._error(f"hip-vector-alu-int64 captures must use backend_metadata.{key}={value}")

    def _validate_comparison_baseline(self) -> None:
        baseline = self._require("comparison_baseline", "dict")
        if not isinstance(baseline, dict):
            return
        status = baseline.get("status")
        if status not in COMPARISON_BASELINE_STATUSES:
            self._error("comparison_baseline.status must describe reviewed or missing same-contract baseline evidence")
        if not isinstance(baseline.get("speedup_claimed"), bool):
            self._error("comparison_baseline.speedup_claimed must be a boolean")
        selected_reference = baseline.get("selected_reference")
        if selected_reference is not None and not isinstance(selected_reference, str):
            self._error("comparison_baseline.selected_reference must be a string or null")
        required = baseline.get("required_before_speedup_claim")
        if not isinstance(required, list) or not all(isinstance(item, str) and item for item in required):
            self._error("comparison_baseline.required_before_speedup_claim must be a nonempty string array")
        if not isinstance(baseline.get("reason"), str) or not baseline.get("reason"):
            self._error("comparison_baseline.reason must be a nonempty string")
        raw_metadata = self.data.get("backend_metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        selected_backend = self.data.get("backend_selected")
        performance_validated = metadata.get("performance_validated") is True
        derived_tops = self.data.get("derived_tops_equivalent")
        if baseline.get("speedup_claimed") is True:
            if status not in REVIEWED_BASELINE_STATUSES or not isinstance(selected_reference, str) or not selected_reference:
                self._error("speedup claims require a reviewed same-contract comparison baseline")
        if performance_validated and status != BASELINE_STATUS_RELEASE_REVIEWED:
            self._error(
                "performance_validated captures require "
                "comparison_baseline.status=reviewed_release_same_contract_baseline"
            )
        if derived_tops is not None and status != BASELINE_STATUS_RELEASE_REVIEWED:
            self._error("derived_tops_equivalent requires a reviewed release same-contract comparison baseline")
        semantics = self.data.get("semantics")
        if semantics in {"bounded_i64", "bounded_u64"} and isinstance(required, list):
            expected = ["same_contract_cpu_reference"]
            if selected_backend == "hip-vector-alu-int64":
                expected.append("same_contract_direct_hip_correctness")
            else:
                expected.append("same_contract_direct_hip_vector_alu_int64")
                if selected_backend != "hip-direct":
                    expected.append("same_contract_direct_hip_correctness")
            if self._is_bounded_oneshot_capture():
                expected.append("same_contract_direct_hip_persistent_rns")
            for item in expected:
                if item not in required:
                    self._error(f"bounded captures require comparison baseline prerequisite {item}")
        if semantics == "wrap_u64_mod_2_64" and isinstance(required, list):
            for item in ["same_contract_cpu_wrap64_byte_limb_reference", "same_contract_direct_hip_wrap64_byte_gemm36"]:
                if item not in required:
                    self._error(f"wrap64 captures require comparison baseline prerequisite {item}")
        if semantics in {"finite_ring_u8", "finite_field_u8"} and isinstance(required, list):
            if "same_contract_cpu_reference" not in required:
                self._error("finite-u8 captures require comparison baseline prerequisite same_contract_cpu_reference")
            if selected_backend != "hip-direct" and "same_contract_direct_hip_correctness" not in required:
                self._error("finite-u8 captures require comparison baseline prerequisite same_contract_direct_hip_correctness")
            if self._is_finite_oneshot_capture() and "same_contract_direct_hip_persistent_finite_u8" not in required:
                self._error(
                    "finite-u8 one-shot captures require comparison baseline prerequisite "
                    "same_contract_direct_hip_persistent_finite_u8"
                )
        if semantics in {"exact_wide_signed", "exact_wide_unsigned"} and isinstance(required, list):
            if "same_contract_cpu_reference" not in required:
                self._error("exact-wide captures require comparison baseline prerequisite same_contract_cpu_reference")
            if selected_backend != "hip-direct" and "same_contract_direct_hip_correctness" not in required:
                self._error(
                    "exact-wide captures require comparison baseline prerequisite same_contract_direct_hip_correctness"
                )
        if selected_backend == "ck":
            expected = {
                "accelerator_library": "Composable Kernel",
                "accelerator_version": "repo-local release/rocm-rel-7.1",
                "capability_status": "implemented_opt_in_ck_backend",
                "workspace_mode": "resident_device_buffers_with_ck_canonical_pack_workspace",
                "isa_evidence": "ck_wmma_cshuffle_int8_matrix_isa_gate_no_int32_global_store_no_divide",
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    self._error(f"CK captures must use backend_metadata.{key}={value}")
            if metadata.get("selected_kernel") not in CK_SELECTED_KERNELS:
                self._error("CK captures must report a known CK selected_kernel")
            bool_expected = {
                "accelerator_backend": True,
                "correctness_backend": True,
                "matrix_engine_backend": True,
                "compiled_kernel_available": True,
                "exact_differential_validated": True,
            }
            for key, value in bool_expected.items():
                if metadata.get(key) is not value:
                    self._error(f"CK captures must use backend_metadata.{key}={value}")
            epilogue = metadata.get("epilogue_mode")
            if epilogue not in {
                "ck_fused_i32_to_centered_residue_then_crt_export",
                "ck_fused_i32_to_centered_residue_rns_output",
                "ck_fused_i32_to_centered_residue_then_canonical_u8_export",
            }:
                self._error("CK captures must report a fused CK centered-residue epilogue")
        if selected_backend == "rocwmma":
            if self._is_wrap64_rocwmma_candidate():
                expected = {
                    "selected_kernel": WRAP64_ROCWMMA_CANDIDATE_KERNEL,
                    "accelerator_library": "rocWMMA",
                    "accelerator_version": "repo-local release/rocm-rel-7.1",
                    "capability_status": "internal_wrap64_matrix_engine_candidate",
                    "epilogue_mode": "low64_wrap_export",
                    "workspace_mode": "benchmark_owned_compact_byte_limb_device_buffers",
                    "isa_evidence": "rocwmma_wrap64_byte_gemm36_wmma_isa_gate_no_int32_global_store_no_divide",
                }
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        self._error(f"rocWMMA wrap64 candidate captures must use backend_metadata.{key}={value}")
                bool_expected = {
                    "accelerator_backend": True,
                    "correctness_backend": False,
                    "matrix_engine_backend": True,
                    "compiled_kernel_available": True,
                    "exact_differential_validated": True,
                    "performance_validated": False,
                }
                for key, value in bool_expected.items():
                    if metadata.get(key) is not value:
                        self._error(f"rocWMMA wrap64 candidate captures must use backend_metadata.{key}={value}")
            else:
                expected = {
                    "accelerator_library": "rocWMMA",
                    "accelerator_version": "repo-local release/rocm-rel-7.1",
                    "capability_status": "implemented_opt_in_rocwmma_backend",
                    "workspace_mode": "resident_device_buffers_with_rocwmma_pack_workspace",
                    "isa_evidence": "rocwmma_i8_wmma_isa_gate_no_int32_global_store_no_divide",
                }
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        self._error(f"rocWMMA captures must use backend_metadata.{key}={value}")
                if metadata.get("selected_kernel") not in ROCWMMA_SELECTED_KERNELS:
                    self._error("rocWMMA captures must report a known rocWMMA selected_kernel")
                bool_expected = {
                    "accelerator_backend": True,
                    "correctness_backend": True,
                    "matrix_engine_backend": True,
                    "compiled_kernel_available": True,
                    "exact_differential_validated": True,
                }
                for key, value in bool_expected.items():
                    if metadata.get(key) is not value:
                        self._error(f"rocWMMA captures must use backend_metadata.{key}={value}")
                epilogue = metadata.get("epilogue_mode")
                if epilogue not in {
                    "rocwmma_fused_i32_to_centered_residue_then_crt_export",
                    "rocwmma_fused_i32_to_centered_residue_rns_output",
                    "rocwmma_fused_i32_to_centered_residue_then_canonical_u8_export",
                }:
                    self._error("rocWMMA captures must report a fused rocWMMA centered-residue epilogue")
        if selected_backend == "hip-direct" and metadata.get("accelerator_library") != "HIP runtime":
            self._error("hip-direct captures must use backend_metadata.accelerator_library=HIP runtime")
        if selected_backend not in HIP_RESIDENT_BACKENDS and metadata.get("accelerator_library") not in {None, ""}:
            self._error("non-HIP correctness captures must not report an accelerator library")
        self._validate_accumulator_safety(metadata)

    def _validate_phase_availability(self, metadata: dict[str, Any]) -> None:
        availability = metadata.get("phase_availability")
        if not isinstance(availability, dict):
            self._error("timing_metadata.phase_availability must be an object")
            return
        scheduling = availability.get("scheduling")
        if not isinstance(scheduling, dict):
            self._error("timing_metadata.phase_availability.scheduling must be an object")
        else:
            if scheduling.get("timed") is not True:
                self._error("timing_metadata.phase_availability.scheduling.timed must be true")
            if scheduling.get("timing_key") != "scheduling":
                self._error("timing_metadata.phase_availability.scheduling.timing_key must be scheduling")
            expected_scope = (
                "benchmark_static_wrap64_rocwmma_candidate_schedule"
                if self._is_wrap64_rocwmma_candidate()
                else "one_time_schedule_info_query"
            )
            if scheduling.get("scope") != expected_scope:
                self._error(f"timing_metadata.phase_availability.scheduling.scope must be {expected_scope}")
            if not isinstance(scheduling.get("reason"), str) or not scheduling.get("reason"):
                self._error("timing_metadata.phase_availability.scheduling.reason must be a nonempty string")

        global_bound_scan = availability.get(GLOBAL_BOUND_TIMING_PHASE)
        input_scan = self.data.get("bound_source") == "input_scan"
        global_input_scan = input_scan and self.data.get("bound_mode", "global") == "global"
        if global_input_scan and not isinstance(global_bound_scan, dict):
            self._error("timing_metadata.phase_availability.global_bound_scan must be an object for input_scan captures")
        elif global_bound_scan is not None:
            if not isinstance(global_bound_scan, dict):
                self._error("timing_metadata.phase_availability.global_bound_scan must be an object")
            else:
                expected_timed = global_input_scan
                expected_key = GLOBAL_BOUND_TIMING_PHASE if global_input_scan else None
                expected_scope = (
                    "input_row_column_abs_summary" if global_input_scan else "not_applicable_static_profile"
                )
                if global_bound_scan.get("timed") is not expected_timed:
                    self._error(
                        "timing_metadata.phase_availability.global_bound_scan.timed must be "
                        f"{str(expected_timed).lower()}"
                    )
                if global_bound_scan.get("timing_key") != expected_key:
                    self._error(
                        "timing_metadata.phase_availability.global_bound_scan.timing_key must be "
                        f"{expected_key}"
                    )
                if global_bound_scan.get("scope") != expected_scope:
                    self._error(
                        "timing_metadata.phase_availability.global_bound_scan.scope must be "
                        f"{expected_scope}"
                    )
                if not isinstance(global_bound_scan.get("reason"), str) or not global_bound_scan.get("reason"):
                    self._error("timing_metadata.phase_availability.global_bound_scan.reason must be a nonempty string")

        per_tile = self.data.get("bound_mode") == "per_tile"
        tile_bound_scan = availability.get(PER_TILE_TIMING_PHASE)
        if per_tile and not isinstance(tile_bound_scan, dict):
            self._error("timing_metadata.phase_availability.tile_bound_scan must be an object for per-tile captures")
        elif tile_bound_scan is not None:
            if not isinstance(tile_bound_scan, dict):
                self._error("timing_metadata.phase_availability.tile_bound_scan must be an object")
            else:
                expected_timed = per_tile
                expected_key = PER_TILE_TIMING_PHASE if per_tile else None
                expected_scope = "exact_seeded_input_prepass" if per_tile else "not_applicable_global_bound"
                if tile_bound_scan.get("timed") is not expected_timed:
                    self._error(
                        "timing_metadata.phase_availability.tile_bound_scan.timed must be "
                        f"{str(expected_timed).lower()}"
                    )
                if tile_bound_scan.get("timing_key") != expected_key:
                    self._error(
                        "timing_metadata.phase_availability.tile_bound_scan.timing_key must be "
                        f"{expected_key}"
                    )
                if tile_bound_scan.get("scope") != expected_scope:
                    self._error(
                        "timing_metadata.phase_availability.tile_bound_scan.scope must be "
                        f"{expected_scope}"
                    )
                if not isinstance(tile_bound_scan.get("reason"), str) or not tile_bound_scan.get("reason"):
                    self._error("timing_metadata.phase_availability.tile_bound_scan.reason must be a nonempty string")

        reuse_packed = self.data.get("reuse_packed_inputs") is True
        prepack = availability.get("prepack_setup")
        if reuse_packed and not isinstance(prepack, dict):
            self._error("timing_metadata.phase_availability.prepack_setup must be an object for prepacked reuse")
        elif prepack is not None:
            if not isinstance(prepack, dict):
                self._error("timing_metadata.phase_availability.prepack_setup must be an object")
            else:
                expected_timed = reuse_packed
                expected_key = "prepack_setup_us" if reuse_packed else None
                expected_scope = "one_time_before_warmups" if reuse_packed else "not_requested_per_repeat_repack"
                if prepack.get("timed") is not expected_timed:
                    self._error(
                        f"timing_metadata.phase_availability.prepack_setup.timed must be {str(expected_timed).lower()}"
                    )
                if prepack.get("timing_key") != expected_key:
                    self._error(
                        f"timing_metadata.phase_availability.prepack_setup.timing_key must be {expected_key}"
                    )
                if prepack.get("scope") != expected_scope:
                    self._error(
                        f"timing_metadata.phase_availability.prepack_setup.scope must be {expected_scope}"
                    )
                if not isinstance(prepack.get("reason"), str) or not prepack.get("reason"):
                    self._error("timing_metadata.phase_availability.prepack_setup.reason must be a nonempty string")

        reduction = availability.get("reduction")
        if not isinstance(reduction, dict):
            self._error("timing_metadata.phase_availability.reduction must be an object")
            return
        if reduction.get("timed") is not False:
            self._error("timing_metadata.phase_availability.reduction.timed must be false")
        if reduction.get("timing_key") is not None:
            self._error("timing_metadata.phase_availability.reduction.timing_key must be null")
        if self.data.get("semantics") == "wrap_u64_mod_2_64":
            expected_scope = "not_applicable_wrap64_byte_limb"
        elif self.data.get("backend_selected") == "hipblaslt":
            expected_scope = "separate_hipblaslt_i32_scratch_residue_reduce"
        elif self.data.get("backend_selected") == "hip-vector-alu-int64":
            expected_scope = (
                "not_applicable_native_vector_output"
                if self._is_vector_alu_runtime_capture()
                else "not_applicable_direct_int64_export"
            )
        else:
            expected_scope = "fused_into_rns_gemm"
        if reduction.get("scope") != expected_scope:
            self._error(f"timing_metadata.phase_availability.reduction.scope must be {expected_scope}")
        if not isinstance(reduction.get("reason"), str) or not reduction.get("reason"):
            self._error("timing_metadata.phase_availability.reduction.reason must be a nonempty string")

    def _timing_phases(self) -> list[str]:
        phases = list(TIMING_PHASES)
        if self.data.get("bound_source") == "input_scan" and self.data.get("bound_mode", "global") == "global":
            phases.insert(0, GLOBAL_BOUND_TIMING_PHASE)
        if self.data.get("bound_mode") == "per_tile":
            phases.insert(phases.index("matrix_alloc"), PER_TILE_TIMING_PHASE)
        return phases

    def _residue_chain_length(self) -> int:
        value = self.data.get("residue_chain_length", 1)
        if not _is_int(value) or value < 1:
            self._error("residue_chain_length must be a positive integer")
            return 1
        return int(value)

    def _residue_output_mode(self) -> str:
        value = self.data.get("residue_output_mode", "host_export")
        if not isinstance(value, str):
            self._error("residue_output_mode must be a string")
            return "host_export"
        if value not in {"host_export", "residue_current_rns"}:
            self._error("residue_output_mode must be host_export or residue_current_rns")
            return "host_export"
        return value

    def _is_residue_current_chain_capture(self) -> bool:
        return (
            self._residue_chain_length() > 1
            or self._residue_output_mode() == "residue_current_rns"
            or self.data.get("epilogue_type") == "residue_current_rns_output"
        )

    def _validate_tile_value(self, key: str, value: Any) -> None:
        if not _is_int(value):
            self._error(f"{key} must be an integer")
            return
        if self._is_wrap64_rocwmma_candidate():
            if value != 16:
                self._error(f"{key} must be 16 for rocWMMA wrap64 candidate captures")
            return
        if value < 64 or value > 512 or (value & (value - 1)) != 0:
            self._error(f"{key} must be a power of two from 64 through 512")

    def _validate_schedule_metadata(self) -> None:
        self._validate_tile_value("tile_m", self.data.get("tile_m"))
        self._validate_tile_value("tile_n", self.data.get("tile_n"))
        schedule = self._require("schedule_metadata", "dict")
        if not isinstance(schedule, dict):
            return
        expected_source = (
            "rns8_bench_wrap64_rocwmma_candidate_static_schedule"
            if self._is_wrap64_rocwmma_candidate()
            else "rns8_get_plan_schedule_info"
        )
        if schedule.get("source") != expected_source:
            self._error(f"schedule_metadata.source must be {expected_source}")
        schedule_bound_kind = schedule.get("bound_kind")
        if schedule_bound_kind is not None:
            if not isinstance(schedule_bound_kind, str) or schedule_bound_kind not in BOUND_KINDS:
                self._error(f"schedule_metadata.bound_kind must be one of {sorted(BOUND_KINDS)}")
            elif schedule_bound_kind != self.data.get("bound_kind"):
                self._error("schedule_metadata.bound_kind must match bound_kind")
        for key in ["effective_bound", "lhs_bound", "rhs_bound"]:
            value = schedule.get(key)
            if value is not None and (not _is_int(value) or value < 0):
                self._error(f"schedule_metadata.{key} must be a nonnegative integer")
        bound_contract = schedule.get("bound_contract")
        if bound_contract is not None and not isinstance(bound_contract, str):
            self._error("schedule_metadata.bound_contract must be a string")
        if schedule_bound_kind == "input_range_and_k":
            if schedule.get("bound_contract") != "input_range_and_k_derived_output_bound":
                self._error("input-range schedules must declare the derived output bound contract")
            for key in ["effective_bound", "lhs_bound", "rhs_bound"]:
                if not _is_int(schedule.get(key)):
                    self._error(f"input-range schedules require schedule_metadata.{key}")
        elif schedule_bound_kind is not None:
            for key in ["lhs_bound", "rhs_bound"]:
                if _is_int(schedule.get(key)) and schedule.get(key) != 0:
                    self._error(f"non-input-range schedules must use schedule_metadata.{key}=0")
        for key in [
            "tile_m",
            "tile_n",
            "tile_rows",
            "tile_cols",
            "tile_count",
            "min_required_prefix",
            "max_required_prefix",
            "min_selected_prefix",
            "max_selected_prefix",
            "prefix_group_count",
            "range_bit_length",
        ]:
            if not _is_int(schedule.get(key)):
                self._error(f"schedule_metadata.{key} must be an integer")
        for key in ["adaptive_prefix_active", "adaptive_skip_active", "adaptive_execution_applied"]:
            if not isinstance(schedule.get(key), bool):
                self._error(f"schedule_metadata.{key} must be a boolean")
        if schedule.get("tile_m") != self.data.get("tile_m"):
            self._error("schedule_metadata.tile_m must match tile_m")
        if schedule.get("tile_n") != self.data.get("tile_n"):
            self._error("schedule_metadata.tile_n must match tile_n")
        tile_rows = schedule.get("tile_rows")
        tile_cols = schedule.get("tile_cols")
        tile_count = schedule.get("tile_count")
        if _is_int(tile_rows) and _is_int(tile_cols) and _is_int(tile_count):
            if tile_rows <= 0 or tile_cols <= 0 or tile_count != tile_rows * tile_cols:
                self._error("schedule_metadata tile grid must have positive rows/cols and matching tile_count")
        min_required = schedule.get("min_required_prefix")
        max_required = schedule.get("max_required_prefix")
        min_selected = schedule.get("min_selected_prefix")
        max_selected = schedule.get("max_selected_prefix")
        if _is_int(min_required) and _is_int(max_required) and min_required > max_required:
            self._error("schedule_metadata min_required_prefix must be <= max_required_prefix")
        if _is_int(min_selected) and _is_int(max_selected) and min_selected > max_selected:
            self._error("schedule_metadata min_selected_prefix must be <= max_selected_prefix")
        flags = schedule.get("flags")
        zero_count = schedule.get("zero_output_tile_count")
        zero_fraction = schedule.get("zero_output_tile_fraction")
        zero_planes = schedule.get("zero_output_selected_residue_planes")
        zero_active = schedule.get("zero_output_skip_active")
        zero_a_rows = schedule.get("zero_a_row_proof_count")
        zero_b_cols = schedule.get("zero_b_col_proof_count")
        zero_row_col_products = schedule.get("zero_row_col_product_count")
        planner_zero_a_rows = schedule.get("planner_zero_a_row_count")
        planner_zero_b_cols = schedule.get("planner_zero_b_col_count")
        planner_zero_row_col_products = schedule.get("planner_zero_row_col_product_count")
        if flags is not None:
            if not _is_int(flags) or flags < 0:
                self._error("schedule_metadata.flags must be a nonnegative integer")
            elif flags & ~TILE_SCHEDULE_KNOWN_FLAGS:
                self._error("schedule_metadata.flags contains unknown tile schedule flags")
        for key, value in [
            ("zero_a_row_proof_count", zero_a_rows),
            ("zero_b_col_proof_count", zero_b_cols),
            ("zero_row_col_product_count", zero_row_col_products),
            ("planner_zero_a_row_count", planner_zero_a_rows),
            ("planner_zero_b_col_count", planner_zero_b_cols),
            ("planner_zero_row_col_product_count", planner_zero_row_col_products),
        ]:
            if not _is_int(value) or value < 0:
                self._error(f"schedule_metadata.{key} must be a nonnegative integer")
        if zero_count is not None:
            if not _is_int(zero_count) or zero_count < 0:
                self._error("schedule_metadata.zero_output_tile_count must be a nonnegative integer")
            elif _is_int(tile_count) and zero_count > tile_count:
                self._error("schedule_metadata.zero_output_tile_count must be <= tile_count")
        if zero_fraction is not None:
            if not _is_number(zero_fraction) or zero_fraction < 0.0 or zero_fraction > 1.0:
                self._error("schedule_metadata.zero_output_tile_fraction must be between 0 and 1")
            elif _is_int(zero_count) and _is_int(tile_count) and tile_count > 0:
                expected = zero_count / tile_count
                if abs(float(zero_fraction) - expected) > 0.000001:
                    self._error("schedule_metadata.zero_output_tile_fraction must match zero_output_tile_count/tile_count")
        if zero_planes is not None:
            if not _is_int(zero_planes) or zero_planes < 0:
                self._error("schedule_metadata.zero_output_selected_residue_planes must be a nonnegative integer")
        if zero_active is not None:
            if not isinstance(zero_active, bool):
                self._error("schedule_metadata.zero_output_skip_active must be a boolean")
            elif _is_int(zero_count) and zero_active != (zero_count > 0):
                self._error("schedule_metadata.zero_output_skip_active must match zero_output_tile_count > 0")
        if _is_int(zero_count):
            if zero_count > 0 and (not _is_int(flags) or (flags & TILE_SCHEDULE_ZERO_OUTPUT) == 0):
                self._error("schedule_metadata zero_output_tile_count requires ZERO_OUTPUT schedule flag")
            if zero_count == 0 and _is_int(flags) and (flags & TILE_SCHEDULE_ZERO_OUTPUT) != 0:
                self._error("schedule_metadata ZERO_OUTPUT flag requires zero_output_tile_count > 0")
        elif _is_int(flags) and (flags & TILE_SCHEDULE_ZERO_OUTPUT) != 0:
            self._error("schedule_metadata ZERO_OUTPUT flag requires zero_output_tile_count")
        if _is_int(zero_planes) and _is_int(zero_count) and zero_count == 0 and zero_planes != 0:
            self._error("schedule_metadata.zero_output_selected_residue_planes must be zero when no zero tiles are skipped")
        if _is_int(zero_a_rows) and _is_int(planner_zero_a_rows) and zero_a_rows != planner_zero_a_rows:
            self._error("schedule_metadata planner_zero_a_row_count must match zero_a_row_proof_count")
        if _is_int(zero_b_cols) and _is_int(planner_zero_b_cols) and zero_b_cols != planner_zero_b_cols:
            self._error("schedule_metadata planner_zero_b_col_count must match zero_b_col_proof_count")
        if (
            _is_int(zero_row_col_products)
            and _is_int(planner_zero_row_col_products)
            and zero_row_col_products != planner_zero_row_col_products
        ):
            self._error("schedule_metadata planner_zero_row_col_product_count must match zero_row_col_product_count")
        row_col_counts_valid = (
            _is_int(zero_a_rows)
            and _is_int(zero_b_cols)
            and _is_int(zero_row_col_products)
            and _is_int(planner_zero_a_rows)
            and _is_int(planner_zero_b_cols)
            and _is_int(planner_zero_row_col_products)
        )
        if row_col_counts_valid:
            m_value = self.data.get("m")
            n_value = self.data.get("n")
            if _is_int(m_value) and zero_a_rows > m_value:
                self._error("schedule_metadata.zero_a_row_proof_count must be <= m")
            if _is_int(n_value) and zero_b_cols > n_value:
                self._error("schedule_metadata.zero_b_col_proof_count must be <= n")
            if _is_int(m_value) and _is_int(n_value):
                expected_products = zero_a_rows * n_value + (m_value - zero_a_rows) * zero_b_cols
                if zero_row_col_products != expected_products:
                    self._error("schedule_metadata.zero_row_col_product_count must match zero row/column product coverage")
                if zero_row_col_products > m_value * n_value:
                    self._error("schedule_metadata.zero_row_col_product_count must be <= m*n")
            row_col_flag = _is_int(flags) and (flags & TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT) != 0
            if zero_row_col_products > 0 and not row_col_flag:
                self._error("schedule_metadata zero_row_col_product_count requires ZERO_ROW_COL_PRODUCT schedule flag")
            if zero_row_col_products == 0 and row_col_flag:
                self._error("schedule_metadata ZERO_ROW_COL_PRODUCT flag requires zero_row_col_product_count > 0")
            per_tile_bounded = (
                self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
                and self.data.get("bound_mode") == "per_tile"
            )
            if not per_tile_bounded and (
                zero_a_rows != 0
                or zero_b_cols != 0
                or zero_row_col_products != 0
                or planner_zero_a_rows != 0
                or planner_zero_b_cols != 0
                or planner_zero_row_col_products != 0
                or row_col_flag
            ):
                self._error("non-per-tile bounded captures must use zero row/column proof counts of 0")

    def _validate_bound_discovery_metadata(self) -> None:
        bound_source = self.data.get("bound_source")
        discovery = self.data.get("bound_discovery")
        semantics = self.data.get("semantics")
        bound_mode = self.data.get("bound_mode", "global")
        if bound_source is not None:
            if not isinstance(bound_source, str) or bound_source not in BOUND_SOURCES:
                self._error(f"bound_source must be one of {sorted(BOUND_SOURCES)}")
        if discovery is None:
            if bound_source == "input_scan":
                self._error("input_scan captures must include bound_discovery")
            return
        if semantics not in {"bounded_i64", "bounded_u64"}:
            self._error("bound_discovery is only valid for bounded captures")
            return
        if not isinstance(discovery, dict):
            self._error("bound_discovery must be an object or null")
            return

        source = discovery.get("source")
        if not isinstance(source, str) or source not in BOUND_DISCOVERY_SOURCES:
            self._error(f"bound_discovery.source must be one of {sorted(BOUND_DISCOVERY_SOURCES)}")
        static_bound = discovery.get("static_bound")
        selected_bound = discovery.get("selected_bound")
        top_bound = self.data.get("bound")
        for key in ["static_bound", "selected_bound"]:
            value = discovery.get(key)
            if not _is_int(value) or value < 0:
                self._error(f"bound_discovery.{key} must be a nonnegative integer")
        if _is_int(selected_bound) and _is_int(top_bound) and selected_bound != top_bound:
            self._error("bound_discovery.selected_bound must match bound")
        if _is_int(static_bound) and _is_int(selected_bound) and selected_bound > static_bound and static_bound != 0:
            self._error("bound_discovery.selected_bound must not exceed static_bound")

        if source == "static_profile_contract":
            if bound_source not in {None, "static_profile"}:
                self._error("static_profile_contract captures must use bound_source=static_profile")
            for key in [
                "discovered_global_bound",
                "candidate_row_sum_col_max",
                "candidate_row_max_col_sum",
                "row_abs_sum_max",
                "row_abs_max",
                "col_abs_sum_max",
                "col_abs_max",
                "zero_row_count",
                "zero_col_count",
            ]:
                if discovery.get(key) is not None:
                    self._error(f"static_profile_contract captures must use bound_discovery.{key}=null")
            return

        if source == "input_exact_tile_bounds":
            if bound_source != "input_scan":
                self._error("input_exact_tile_bounds captures must use bound_source=input_scan")
            if bound_mode != "per_tile":
                self._error("input_exact_tile_bounds captures must use bound_mode=per_tile")
            for key in [
                "discovered_global_bound",
                "candidate_row_sum_col_max",
                "candidate_row_max_col_sum",
                "row_abs_sum_max",
                "row_abs_max",
                "col_abs_sum_max",
                "col_abs_max",
                "zero_row_count",
                "zero_col_count",
            ]:
                if discovery.get(key) is not None:
                    self._error(f"input_exact_tile_bounds captures must use bound_discovery.{key}=null")
            tile_bounds = self.data.get("tile_bounds_u64")
            if not isinstance(tile_bounds, dict):
                self._error("input_exact_tile_bounds captures must include tile_bounds_u64")
            elif tile_bounds.get("source") != "exact_seeded_input_prepass":
                self._error("input_exact_tile_bounds captures must use tile_bounds_u64.source=exact_seeded_input_prepass")
            if _is_int(selected_bound) and selected_bound != 0:
                self._error("input_exact_tile_bounds captures must use selected_bound=0")
            return

        if bound_source != "input_scan":
            self._error("input_row_column_abs_summary captures must use bound_source=input_scan")
        if bound_mode != "global":
            self._error("input_row_column_abs_summary captures must use bound_mode=global")
        for key in [
            "discovered_global_bound",
            "candidate_row_sum_col_max",
            "candidate_row_max_col_sum",
            "row_abs_sum_max",
            "row_abs_max",
            "col_abs_sum_max",
            "col_abs_max",
            "zero_row_count",
            "zero_col_count",
        ]:
            value = discovery.get(key)
            if not _is_int(value) or value < 0:
                self._error(f"bound_discovery.{key} must be a nonnegative integer")
        discovered = discovery.get("discovered_global_bound")
        candidate_a = discovery.get("candidate_row_sum_col_max")
        candidate_b = discovery.get("candidate_row_max_col_sum")
        if _is_int(discovered) and _is_int(candidate_a) and _is_int(candidate_b):
            expected = min(candidate_a, candidate_b)
            if _is_int(static_bound) and static_bound != 0:
                expected = min(expected, static_bound)
            if discovered != expected:
                self._error("bound_discovery.discovered_global_bound must equal the minimum safe candidate bound")
        if _is_int(discovered) and _is_int(selected_bound) and discovered != selected_bound:
            self._error("input_scan bound_discovery.discovered_global_bound must match selected_bound")
        if _is_int(discovery.get("zero_row_count")) and _is_int(self.data.get("m")):
            if discovery.get("zero_row_count") > self.data.get("m"):
                self._error("bound_discovery.zero_row_count must be <= m")
        if _is_int(discovery.get("zero_col_count")) and _is_int(self.data.get("n")):
            if discovery.get("zero_col_count") > self.data.get("n"):
                self._error("bound_discovery.zero_col_count must be <= n")

    def _validate_prefix_policy_metadata(self) -> None:
        present = [field for field in PREFIX_POLICY_FIELDS if field in self.data]
        if present and len(present) != len(PREFIX_POLICY_FIELDS):
            missing = sorted(PREFIX_POLICY_FIELDS - set(present))
            self._error(f"prefix policy metadata fields must be complete; missing {missing}")
            return
        if not present:
            return

        semantics = self.data.get("semantics")
        bound_mode = self.data.get("bound_mode", "global")
        schedule = self.data.get("schedule_metadata")
        prefix = self.data.get("prefix")
        selected = self.data.get("selected_prefix")
        requested = self.data.get("requested_max_prefix")
        policy = self.data.get("contract_prefix_policy")
        planes_requested = self.data.get("residue_planes_requested")
        planes_selected = self.data.get("residue_planes_selected")
        planes_skipped = self.data.get("residue_planes_skipped")
        skip_fraction = self.data.get("residue_plane_skip_fraction")

        for key in [
            "selected_prefix",
            "requested_max_prefix",
            "residue_planes_requested",
            "residue_planes_selected",
            "residue_planes_skipped",
        ]:
            if not _is_int(self.data.get(key)) or self.data.get(key) < 0:
                self._error(f"{key} must be a nonnegative integer")
        if not _is_number(skip_fraction) or float(skip_fraction) < 0.0 or float(skip_fraction) > 1.0:
            self._error("residue_plane_skip_fraction must be a number in [0, 1]")
        if not isinstance(policy, str) or policy not in CONTRACT_PREFIX_POLICIES:
            self._error(f"contract_prefix_policy must be one of {sorted(CONTRACT_PREFIX_POLICIES)}")
        if _is_int(prefix) and _is_int(requested) and requested != prefix:
            self._error("requested_max_prefix must match prefix")

        if semantics in NON_RNS_PREFIX_SEMANTICS:
            if selected != 0 or requested != 0 or planes_requested != 0 or planes_selected != 0 or planes_skipped != 0:
                self._error("non-RNS captures must report zero prefix policy plane counts")
            if policy != "semantic_specific_no_rns_prefix":
                self._error("non-RNS captures must use contract_prefix_policy=semantic_specific_no_rns_prefix")
            if _is_number(skip_fraction) and not _close(float(skip_fraction), 0.0):
                self._error("non-RNS captures must use residue_plane_skip_fraction=0")
            return

        if semantics in RNS_PREFIX_SEMANTICS:
            if _is_int(prefix) and prefix <= 0:
                self._error("RNS captures with prefix policy metadata must use prefix>0")
            if _is_int(selected) and _is_int(prefix) and selected > prefix:
                self._error("selected_prefix must be <= prefix")
            if isinstance(schedule, dict):
                if schedule.get("max_selected_prefix") != selected:
                    self._error("selected_prefix must match schedule_metadata.max_selected_prefix")
                if (
                    _is_int(selected)
                    and selected > 0
                    and _is_int(schedule.get("min_selected_prefix"))
                    and schedule.get("min_selected_prefix") > selected
                ):
                    self._error("schedule_metadata.min_selected_prefix must be <= selected_prefix")
            expected_skipped = max(int(requested) - int(selected), 0) if _is_int(requested) and _is_int(selected) else None
            if expected_skipped is not None and planes_skipped != expected_skipped:
                self._error("residue_planes_skipped must equal requested_max_prefix - selected_prefix")
            if _is_int(requested) and planes_requested != requested:
                self._error("residue_planes_requested must match requested_max_prefix")
            if _is_int(selected) and planes_selected != selected:
                self._error("residue_planes_selected must match selected_prefix")
            if _is_int(requested) and requested > 0 and expected_skipped is not None:
                expected_fraction = float(expected_skipped) / float(requested)
                if _is_number(skip_fraction) and not _close(float(skip_fraction), expected_fraction):
                    self._error("residue_plane_skip_fraction must match skipped/requested")
            if policy == "semantic_specific_no_rns_prefix":
                self._error("RNS captures must not use semantic_specific_no_rns_prefix")
            if policy == "per_tile_minimum" and bound_mode != "per_tile":
                self._error("contract_prefix_policy=per_tile_minimum requires bound_mode=per_tile")
            if policy in {"minimum_proven", "fixed_requested_residue_chain"} and bound_mode != "global":
                self._error(f"contract_prefix_policy={policy} requires bound_mode=global")
            if policy in {"fixed_requested", "fixed_requested_residue_chain"} and _is_int(selected) and _is_int(prefix):
                if selected != prefix or planes_skipped != 0:
                    self._error(f"contract_prefix_policy={policy} requires selected_prefix=prefix")
            if policy == "minimum_proven" and isinstance(schedule, dict):
                if schedule.get("adaptive_execution_applied") is True:
                    self._error("global minimum_proven captures must not apply adaptive execution")
                if schedule.get("prefix_group_count") != 1:
                    self._error("global minimum_proven captures must use one uniform selected prefix group")
            return

        self._error(f"prefix policy metadata is not supported for semantics {semantics}")

    def _validate_semantic_contract(self) -> None:
        semantics = self.data.get("semantics")
        prefix = self.data.get("prefix")
        packed_layout = self.data.get("packed_layout_version")
        schedule = self.data.get("schedule_metadata")
        backend_metadata = self.data.get("backend_metadata")
        k = self.data.get("k")
        k_value = int(k) if _is_int(k) and k > 0 else 0
        bound_mode = self.data.get("bound_mode", "global")
        residue_chain_length = self._residue_chain_length()
        residue_output_mode = self._residue_output_mode()
        status_check = self.data.get("exact_wide_export_status_check")
        prefix_policy = self.data.get("contract_prefix_policy")
        if bound_mode not in {"global", "per_tile"}:
            self._error("bound_mode must be global or per_tile")
        if status_check is not None and semantics not in {"exact_wide_signed", "exact_wide_unsigned"}:
            self._error("exact_wide_export_status_check must be null outside exact-wide captures")
        rns_chain_semantics = {"bounded_i64", "bounded_u64", "exact_wide_signed", "exact_wide_unsigned"}
        if residue_chain_length > 1 and semantics not in rns_chain_semantics:
            self._error("residue_chain_length > 1 captures must use bounded or exact-wide RNS semantics")
        if residue_output_mode == "residue_current_rns" and residue_chain_length <= 1:
            self._error("residue_output_mode=residue_current_rns requires residue_chain_length > 1")
        if residue_chain_length == 1 and residue_output_mode != "host_export":
            self._error("residue_chain_length=1 captures must use residue_output_mode=host_export")
        if semantics == "wrap_u64_mod_2_64":
            is_candidate = self._is_wrap64_rocwmma_candidate()
            if self.data.get("backend_selected") not in {"wrap64-byte-limb", "hip-direct"} and not is_candidate:
                self._error(
                    "wrap64 captures must select wrap64-byte-limb, hip-direct, or rocWMMA candidate backend"
                )
            if bound_mode != "global":
                self._error("wrap64 captures must use bound_mode=global")
            if self.data.get("backend_selected") == "hip-direct":
                allowed_kernels = wrap64_hip_allowed_kernels(self.data.get("m"), self.data.get("n"), k_value)
                if self.data.get("selected_kernel") not in allowed_kernels:
                    self._error(
                        "direct-HIP wrap64 captures must use selected_kernel in "
                        f"{sorted(allowed_kernels)}"
                    )
                if isinstance(backend_metadata, dict):
                    if backend_metadata.get("epilogue_mode") != "low64_wrap_export":
                        self._error("direct-HIP wrap64 captures must use backend_metadata.epilogue_mode=low64_wrap_export")
                    if backend_metadata.get("workspace_mode") != "resident_device_buffers":
                        self._error("direct-HIP wrap64 captures must use backend_metadata.workspace_mode=resident_device_buffers")
                    expected_isa = "wrap64_byte_gemm36_isa_gate_no_variable_divide_no_matrix_engine"
                    if backend_metadata.get("isa_evidence") != expected_isa:
                        self._error(f"direct-HIP wrap64 captures must use backend_metadata.isa_evidence={expected_isa}")
            if is_candidate:
                if self.data.get("selected_kernel") != WRAP64_ROCWMMA_CANDIDATE_KERNEL:
                    self._error(f"rocWMMA wrap64 candidate captures must use selected_kernel={WRAP64_ROCWMMA_CANDIDATE_KERNEL}")
                if isinstance(backend_metadata, dict):
                    if backend_metadata.get("epilogue_mode") != "low64_wrap_export":
                        self._error("rocWMMA wrap64 candidate captures must use backend_metadata.epilogue_mode=low64_wrap_export")
                    if backend_metadata.get("workspace_mode") != "benchmark_owned_compact_byte_limb_device_buffers":
                        self._error(
                            "rocWMMA wrap64 candidate captures must use "
                            "backend_metadata.workspace_mode=benchmark_owned_compact_byte_limb_device_buffers"
                        )
            if self.data.get("bound_kind") != "none" or self.data.get("bound") != 0:
                self._error("wrap64 captures must use bound_kind=none and bound=0")
            if self.data.get("tile_bounds_u64") is not None:
                self._error("wrap64 captures must use tile_bounds_u64=null")
            if prefix != 0:
                self._error("wrap64 captures must use prefix=0")
            if packed_layout != "byte_limb_v1":
                self._error("wrap64 captures must use packed_layout_version=byte_limb_v1")
            if self.data.get("epilogue_type") != "low64_wrap_export":
                self._error("wrap64 captures must use low64_wrap_export epilogue")
            if isinstance(schedule, dict):
                for key in ["min_required_prefix", "max_required_prefix", "min_selected_prefix", "max_selected_prefix"]:
                    if schedule.get(key) != 0:
                        self._error(f"wrap64 captures must use schedule_metadata.{key}=0")
                if schedule.get("prefix_group_count") != 0:
                    self._error("wrap64 captures must use schedule_metadata.prefix_group_count=0")
            if self.data.get("backend_selected") == "hip-direct":
                metadata = self.data.get("timing_metadata")
                if isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                    expected_scope = "direct_hip_wrap64_byte_gemm36_default_stream_backend_operation_groups"
                    if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                        self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
            if is_candidate:
                metadata = self.data.get("timing_metadata")
                if isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                    expected_scope = "rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups"
                    if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                        self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
        elif semantics in {"bounded_i64", "bounded_u64"}:
            oneshot_capture = self._is_bounded_oneshot_capture()
            native_a_reuse_b_capture = self._is_direct_hip_bounded_native_a_reuse_b_capture()
            if oneshot_capture:
                if self.data.get("benchmark") != "rns8_bounded_gemm_public_oneshot":
                    self._error("one-shot captures must use benchmark=rns8_bounded_gemm_public_oneshot")
                if self._benchmark_execution_mode() != "public_oneshot_transient_native_inputs":
                    self._error("one-shot captures must use benchmark_execution_mode=public_oneshot_transient_native_inputs")
                if self.data.get("backend_selected") not in {"cpu-reference", "hip-direct"}:
                    self._error("one-shot bounded captures must select cpu-reference or hip-direct")
                if self.data.get("backend_requested") not in {"cpu-reference", "cpu", "hip-direct"}:
                    self._error("one-shot bounded captures must request cpu or hip-direct")
                if bound_mode != "global":
                    self._error("one-shot bounded captures must use bound_mode=global")
                if residue_chain_length != 1 or residue_output_mode != "host_export":
                    self._error("one-shot bounded captures must use host_export residue_chain_length=1")
                if self.data.get("reuse_packed_inputs") is True:
                    self._error("one-shot bounded captures must not use packed-input reuse")
                if self.data.get("pack_mode") not in {None, "per_repeat_repack"}:
                    self._error("one-shot bounded captures must use pack_mode=per_repeat_repack")
                if self.data.get("prepack_reuse_strategy") not in {None, "none"}:
                    self._error("one-shot bounded captures must use prepack_reuse_strategy=none")
                if self.data.get("backend_selected") == "hip-direct":
                    native_input_oneshot = self._is_direct_hip_bounded_native_input_oneshot_capture()
                    resident_fallback_oneshot = self._is_direct_hip_bounded_resident_fallback_oneshot_capture()
                    if not native_input_oneshot and not resident_fallback_oneshot:
                        self._error(
                            "direct-HIP one-shot bounded captures must use either prefix-9 native-input metadata "
                            "or selected-prefix resident fallback metadata"
                        )
                    expected_kernel = DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_KERNEL
                    expected_epilogue = DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_EPILOGUE
                    expected_workspace = DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_WORKSPACE
                    if native_input_oneshot:
                        expected_kernel = DIRECT_HIP_BOUNDED_ONESHOT_KERNEL_V1
                        large_oneshot_shape = all(int(self.data.get(dim, 0)) >= 512 for dim in ("m", "n", "k"))
                        if large_oneshot_shape:
                            expected_kernel = DIRECT_HIP_BOUNDED_ONESHOT_KERNEL_LARGE_COLPAIR_V2
                        expected_epilogue = DIRECT_HIP_BOUNDED_ONESHOT_EPILOGUE
                        expected_workspace = DIRECT_HIP_BOUNDED_ONESHOT_WORKSPACE
                    if self.data.get("selected_kernel") != expected_kernel:
                        self._error(
                            "direct-HIP one-shot bounded captures must use "
                            f"selected_kernel={expected_kernel}"
                        )
                    if isinstance(backend_metadata, dict):
                        if backend_metadata.get("epilogue_mode") != expected_epilogue:
                            self._error(
                                "direct-HIP one-shot bounded captures must use "
                                f"backend_metadata.epilogue_mode={expected_epilogue}"
                            )
                        if backend_metadata.get("workspace_mode") != expected_workspace:
                            self._error(
                                "direct-HIP one-shot bounded captures must use "
                                f"backend_metadata.workspace_mode={expected_workspace}"
                            )
                        if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                            self._error(
                                "direct-HIP one-shot bounded captures must use "
                                f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                            )
                    metadata = self.data.get("timing_metadata")
                    if native_input_oneshot and isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                        expected_scope = "direct_hip_oneshot_default_stream_operation_groups"
                        if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                            self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
                    if resident_fallback_oneshot and isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                        expected_scope = "direct_hip_oneshot_resident_fallback_default_stream_operation_groups"
                        if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                            self._error(
                                "direct-HIP one-shot captures must use "
                                f"timing_metadata.gpu_event_timing_source_scope={expected_scope} "
                                "for resident fallback metadata"
                            )
            if native_a_reuse_b_capture:
                if bound_mode != "global":
                    self._error("direct-HIP bounded native-A reuse-B captures must use bound_mode=global")
                if residue_chain_length != 1 or residue_output_mode != "host_export":
                    self._error("direct-HIP bounded native-A reuse-B captures must use host_export residue_chain_length=1")
                uniform_small = self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
                expected_kernel = (
                    DIRECT_HIP_BOUNDED_UNIFORM_SMALL_NATIVE_A_REUSE_B_KERNELS[semantics]
                    if uniform_small
                    else DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_KERNELS[semantics]
                )
                if self._is_direct_hip_bounded_native_a_reuse_b_u64_large_colpair():
                    expected_kernel = DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_U64_LARGE_COLPAIR_KERNEL
                expected_epilogue = (
                    DIRECT_HIP_BOUNDED_UNIFORM_SMALL_NATIVE_A_REUSE_B_EPILOGUE
                    if uniform_small
                    else DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_EPILOGUE
                )
                expected_input_profile = "uniform-small" if uniform_small else "adaptive-bands"
                if self.data.get("selected_kernel") != expected_kernel:
                    self._error(
                        "direct-HIP bounded native-A reuse-B captures must use "
                        f"selected_kernel={expected_kernel}"
                    )
                if isinstance(backend_metadata, dict):
                    if backend_metadata.get("epilogue_mode") != expected_epilogue:
                        self._error(
                            "direct-HIP bounded native-A reuse-B captures must use "
                            f"backend_metadata.epilogue_mode={expected_epilogue}"
                        )
                    if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_WORKSPACE:
                        self._error(
                            "direct-HIP bounded native-A reuse-B captures must use "
                            f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_WORKSPACE}"
                        )
                    if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                        self._error(
                            "direct-HIP bounded native-A reuse-B captures must use "
                            f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                        )
                    autotune_key = backend_metadata.get("autotune_key")
                    if isinstance(autotune_key, str):
                        normalized_key = f";{autotune_key};"
                        required_parts = {
                            "kernel": expected_kernel,
                            "epilogue": expected_epilogue,
                            "input_profile": expected_input_profile,
                        }
                        for key, value in required_parts.items():
                            if f";{key}={value};" not in normalized_key:
                                self._error(
                                    "direct-HIP bounded native-A reuse-B backend_metadata.autotune_key "
                                    f"must include {key}={value}"
                                )
            if self._is_direct_hip_bounded_uniform_small_reuse_a_capture():
                if bound_mode != "global":
                    self._error("direct-HIP bounded uniform-small reuse-A captures must use bound_mode=global")
                if residue_chain_length != 1 or residue_output_mode != "host_export":
                    self._error(
                        "direct-HIP bounded uniform-small reuse-A captures must use host_export residue_chain_length=1"
                    )
                expected_kernel = DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_KERNELS[semantics]
                expected_epilogue = DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_EPILOGUE
                if self.data.get("selected_kernel") != expected_kernel:
                    self._error(
                        "direct-HIP bounded uniform-small reuse-A captures must use "
                        f"selected_kernel={expected_kernel}"
                    )
                if isinstance(backend_metadata, dict):
                    if backend_metadata.get("epilogue_mode") != expected_epilogue:
                        self._error(
                            "direct-HIP bounded uniform-small reuse-A captures must use "
                            f"backend_metadata.epilogue_mode={expected_epilogue}"
                        )
                    if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_WORKSPACE:
                        self._error(
                            "direct-HIP bounded uniform-small reuse-A captures must use "
                            f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_WORKSPACE}"
                        )
                    if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                        self._error(
                            "direct-HIP bounded uniform-small reuse-A captures must use "
                            f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                        )
                    autotune_key = backend_metadata.get("autotune_key")
                    if isinstance(autotune_key, str):
                        normalized_key = f";{autotune_key};"
                        required_parts = {
                            "kernel": expected_kernel,
                            "epilogue": expected_epilogue,
                            "input_profile": "uniform-small",
                        }
                        for key, value in required_parts.items():
                            if f";{key}={value};" not in normalized_key:
                                self._error(
                                    "direct-HIP bounded uniform-small reuse-A backend_metadata.autotune_key "
                                    f"must include {key}={value}"
                                )
            if self._is_direct_hip_bounded_residue_channel_fusion_capture():
                if bound_mode != "global":
                    self._error("direct-HIP residue-channel fusion captures must use bound_mode=global")
                if residue_chain_length != 1 or residue_output_mode != "host_export":
                    self._error(
                        "direct-HIP residue-channel fusion captures must use host_export residue_chain_length=1"
                    )
                if self.data.get("reuse_packed_inputs") is True:
                    self._error("direct-HIP residue-channel fusion captures must not use packed-input reuse")
                expected_kernel = DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_KERNELS[semantics]
                expected_epilogue = DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_EPILOGUE
                if self.data.get("selected_kernel") != expected_kernel:
                    self._error(
                        "direct-HIP residue-channel fusion captures must use "
                        f"selected_kernel={expected_kernel}"
                    )
                if isinstance(backend_metadata, dict):
                    if backend_metadata.get("epilogue_mode") != expected_epilogue:
                        self._error(
                            "direct-HIP residue-channel fusion captures must use "
                            f"backend_metadata.epilogue_mode={expected_epilogue}"
                        )
                    if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_WORKSPACE:
                        self._error(
                            "direct-HIP residue-channel fusion captures must use "
                            f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_WORKSPACE}"
                        )
                    if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                        self._error(
                            "direct-HIP residue-channel fusion captures must use "
                            f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                        )
                    autotune_key = backend_metadata.get("autotune_key")
                    if isinstance(autotune_key, str):
                        normalized_key = f";{autotune_key};"
                        required_parts = {
                            "kernel": expected_kernel,
                            "epilogue": expected_epilogue,
                            "input_profile": "uniform-small",
                            "execution": "residue_channel_fusion_native_inputs",
                        }
                        for key, value in required_parts.items():
                            if f";{key}={value};" not in normalized_key:
                                self._error(
                                    "direct-HIP residue-channel fusion backend_metadata.autotune_key "
                                    f"must include {key}={value}"
                                )
            if self._is_direct_hip_bounded_uniform_small_transient_capture():
                if bound_mode != "global":
                    self._error("direct-HIP bounded uniform-small transient captures must use bound_mode=global")
                if residue_chain_length != 1 or residue_output_mode != "host_export":
                    self._error(
                        "direct-HIP bounded uniform-small transient captures must use host_export residue_chain_length=1"
                    )
                if self.data.get("reuse_packed_inputs") is True:
                    self._error("direct-HIP bounded uniform-small transient captures must not use packed-input reuse")
                expected_kernel = DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_KERNELS[semantics]
                expected_epilogue = DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_EPILOGUE
                if self.data.get("selected_kernel") != expected_kernel:
                    self._error(
                        "direct-HIP bounded uniform-small transient captures must use "
                        f"selected_kernel={expected_kernel}"
                    )
                if isinstance(backend_metadata, dict):
                    if backend_metadata.get("epilogue_mode") != expected_epilogue:
                        self._error(
                            "direct-HIP bounded uniform-small transient captures must use "
                            f"backend_metadata.epilogue_mode={expected_epilogue}"
                        )
                    if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_WORKSPACE:
                        self._error(
                            "direct-HIP bounded uniform-small transient captures must use "
                            f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_WORKSPACE}"
                        )
                    if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                        self._error(
                            "direct-HIP bounded uniform-small transient captures must use "
                            f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                        )
                    autotune_key = backend_metadata.get("autotune_key")
                    if isinstance(autotune_key, str):
                        normalized_key = f";{autotune_key};"
                        required_parts = {
                            "kernel": expected_kernel,
                            "epilogue": expected_epilogue,
                            "input_profile": "uniform-small",
                            "execution": "transient_uniform_small_i8_ab_inputs",
                        }
                        for key, value in required_parts.items():
                            if f";{key}={value};" not in normalized_key:
                                self._error(
                                    "direct-HIP bounded uniform-small transient backend_metadata.autotune_key "
                                    f"must include {key}={value}"
                                )
            if self._is_direct_hip_bounded_native_b_reuse_a_u64_large_colpair_capture():
                if bound_mode != "global":
                    self._error("direct-HIP bounded native-B reuse-A captures must use bound_mode=global")
                if residue_chain_length != 1 or residue_output_mode != "host_export":
                    self._error("direct-HIP bounded native-B reuse-A captures must use host_export residue_chain_length=1")
                expected_kernel = DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_U64_LARGE_COLPAIR_KERNEL
                expected_epilogue = DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_EPILOGUE
                if self.data.get("selected_kernel") != expected_kernel:
                    self._error(
                        "direct-HIP bounded native-B reuse-A captures must use "
                        f"selected_kernel={expected_kernel}"
                    )
                if isinstance(backend_metadata, dict):
                    if backend_metadata.get("epilogue_mode") != expected_epilogue:
                        self._error(
                            "direct-HIP bounded native-B reuse-A captures must use "
                            f"backend_metadata.epilogue_mode={expected_epilogue}"
                        )
                    if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_WORKSPACE:
                        self._error(
                            "direct-HIP bounded native-B reuse-A captures must use "
                            f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_WORKSPACE}"
                        )
                    if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                        self._error(
                            "direct-HIP bounded native-B reuse-A captures must use "
                            f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                        )
                    autotune_key = backend_metadata.get("autotune_key")
                    if isinstance(autotune_key, str):
                        normalized_key = f";{autotune_key};"
                        required_parts = {
                            "kernel": expected_kernel,
                            "epilogue": expected_epilogue,
                            "input_profile": "adaptive-bands",
                        }
                        for key, value in required_parts.items():
                            if f";{key}={value};" not in normalized_key:
                                self._error(
                                    "direct-HIP bounded native-B reuse-A backend_metadata.autotune_key "
                                    f"must include {key}={value}"
                                )
            if _is_int(prefix) and prefix <= 0:
                self._error(f"{semantics} captures must use a positive prefix")
            expected_native_layout = (
                "native_i64_rowmajor_v1" if semantics == "bounded_i64" else "native_u64_rowmajor_v1"
            )
            if self._is_vector_alu_runtime_capture():
                if packed_layout != expected_native_layout:
                    self._error(f"{semantics} runtime vector captures must use packed_layout_version={expected_native_layout}")
            elif packed_layout is not None:
                self._error(f"{semantics} captures must use packed_layout_version=null")
            if self._is_direct_hip_native_to_rns_bridge_capture():
                metadata = self.data.get("timing_metadata")
                if self.data.get("benchmark") != "rns8_bounded_gemm_native_to_rns_bridge":
                    self._error("native-to-RNS bridge captures must use benchmark=rns8_bounded_gemm_native_to_rns_bridge")
                if self.data.get("backend_requested") != "auto":
                    self._error("native-to-RNS bridge captures must use backend_requested=auto")
                if self.data.get("pack_mode") != "per_repeat_repack":
                    self._error("native-to-RNS bridge captures must use pack_mode=per_repeat_repack")
                if self.data.get("reuse_packed_inputs") is not False:
                    self._error("native-to-RNS bridge captures must not use packed-input reuse")
                if isinstance(metadata, dict) and metadata.get("native_to_rns_bridge_forced") is not True:
                    self._error("native-to-RNS bridge captures must set timing_metadata.native_to_rns_bridge_forced=true")
            if self._is_direct_hip_vector_to_rns_chain_capture():
                metadata = self.data.get("timing_metadata")
                if self.data.get("benchmark") != "rns8_bounded_gemm_vector_to_rns_chain":
                    self._error(
                        "vector-to-RNS chain captures must use benchmark=rns8_bounded_gemm_vector_to_rns_chain"
                    )
                if self.data.get("backend_requested") != "auto":
                    self._error("vector-to-RNS chain captures must use backend_requested=auto")
                pack_mode = self.data.get("pack_mode")
                if pack_mode not in {"per_repeat_repack", "prepacked_reuse_b"}:
                    self._error("vector-to-RNS chain captures must use pack_mode=per_repeat_repack or prepacked_reuse_b")
                expected_reuse = pack_mode == "prepacked_reuse_b"
                if self.data.get("reuse_packed_inputs") is not expected_reuse:
                    self._error("vector-to-RNS chain captures must set reuse_packed_inputs to match consumer-B reuse")
                expected_strategy = "persistent_matrix_residency" if expected_reuse else "none"
                if self.data.get("prepack_reuse_strategy") != expected_strategy:
                    self._error(
                        f"vector-to-RNS chain captures must use prepack_reuse_strategy={expected_strategy}"
                    )
                if isinstance(metadata, dict):
                    if metadata.get("vector_to_rns_chain") is not True:
                        self._error("vector-to-RNS chain captures must set timing_metadata.vector_to_rns_chain=true")
                    if metadata.get("native_to_rns_bridge_forced") is not False:
                        self._error(
                            "vector-to-RNS chain captures must set timing_metadata.native_to_rns_bridge_forced=false"
                        )
                    if metadata.get("pack_mode") != pack_mode:
                        self._error("vector-to-RNS chain captures must keep timing_metadata.pack_mode in sync")
                    if metadata.get("prepack_reuse_strategy") != expected_strategy:
                        self._error(
                            "vector-to-RNS chain captures must keep timing_metadata.prepack_reuse_strategy in sync"
                        )
            if residue_chain_length > 1:
                expected_epilogue_type = "residue_current_rns_output"
                if residue_output_mode != "residue_current_rns":
                    self._error("bounded residue-current chains must use residue_output_mode=residue_current_rns")
                if bound_mode != "global":
                    self._error("bounded residue-current chains must use bound_mode=global")
                if self.data.get("backend_selected") in {"hip-vector-alu-int64"}:
                    self._error("bounded residue-current chains must not select hip-vector-alu-int64")
                if self.data.get("m") != self.data.get("n") or self.data.get("n") != self.data.get("k"):
                    self._error("bounded residue-current chains must use square m=n=k shapes")
            else:
                expected_epilogue_type = (
                    "direct_int64_export" if self.data.get("backend_selected") == "hip-vector-alu-int64" else "crt_export"
                )
                if residue_output_mode != "host_export":
                    self._error("bounded host-export captures must use residue_output_mode=host_export")
            if self.data.get("epilogue_type") != expected_epilogue_type:
                self._error(f"{semantics} captures must use {expected_epilogue_type} epilogue")
            if bound_mode == "global":
                expected_bound_kind = "global_max_abs" if semantics == "bounded_i64" else "global_max_unsigned"
                if self.data.get("bound_kind") != expected_bound_kind:
                    self._error(f"{semantics} captures must use bound_kind={expected_bound_kind}")
                if self.data.get("tile_bounds_u64") is not None:
                    self._error(f"{semantics} global captures must use tile_bounds_u64=null")
                if self.data.get("backend_selected") == "hip-direct" and not oneshot_capture:
                    metadata = self.data.get("timing_metadata")
                    if isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                        if self._is_direct_hip_native_to_rns_bridge_capture():
                            expected_scope = "direct_hip_native_to_rns_bridge_default_stream_operation_groups"
                        elif self._is_direct_hip_vector_to_rns_chain_capture():
                            expected_scope = "direct_hip_vector_native_to_rns_chain_default_stream_operation_groups"
                        else:
                            expected_scope = "direct_hip_default_stream_backend_operation_groups"
                        if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                            self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
                if self.data.get("backend_selected") == "hipblaslt":
                    metadata = self.data.get("timing_metadata")
                    if isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                        expected_scope = "hipblaslt_baseline_default_stream_backend_operation_groups"
                        if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                            self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
                    if isinstance(backend_metadata, dict):
                        if backend_metadata.get("epilogue_mode") != "separate_i32_scratch_reduce_then_crt_export":
                            self._error(
                                "hipBLASLt bounded captures must use "
                                "backend_metadata.epilogue_mode=separate_i32_scratch_reduce_then_crt_export"
                            )
                if isinstance(schedule, dict) and _is_int(prefix):
                    if prefix_policy == "minimum_proven":
                        if schedule.get("min_selected_prefix") != schedule.get("max_selected_prefix"):
                            self._error(f"{semantics} minimum_proven global captures must use one selected prefix")
                        if schedule.get("max_selected_prefix") > prefix:
                            self._error(f"{semantics} selected schedule prefix must be <= prefix")
                    elif schedule.get("min_selected_prefix") != prefix or schedule.get("max_selected_prefix") != prefix:
                        self._error(f"{semantics} captures must use fixed selected schedule prefix equal to prefix")
                    if schedule.get("prefix_group_count") != 1:
                        self._error(f"{semantics} captures must use one fixed prefix group")
                    if schedule.get("adaptive_execution_applied") is True:
                        self._error(f"{semantics} global captures must not apply adaptive execution")
            elif bound_mode == "per_tile":
                expected_bound_kind = "per_tile_max_abs" if semantics == "bounded_i64" else "per_tile_max_unsigned"
                if self.data.get("bound_kind") != expected_bound_kind:
                    self._error(f"{semantics} per-tile captures must use bound_kind={expected_bound_kind}")
                if self.data.get("backend_selected") not in {
                    "cpu-reference",
                    "hip-direct",
                    "ck",
                    "rocwmma",
                    "hip-vector-alu-int64",
                }:
                    self._error(
                        "per-tile adaptive captures must select cpu-reference, hip-direct, ck, rocwmma, "
                        "or hip-vector-alu-int64 backend"
                    )
                if self.data.get("bound") != 0:
                    self._error("per-tile adaptive captures must use bound=0")
                self._validate_v4_tile_bounds(semantics, schedule)
                if prefix_policy == "fixed_requested":
                    if isinstance(schedule, dict) and _is_int(prefix):
                        if schedule.get("min_selected_prefix") != prefix or schedule.get("max_selected_prefix") != prefix:
                            self._error("fixed-requested per-tile captures must use selected schedule prefix equal to prefix")
                        if schedule.get("prefix_group_count") != 1:
                            self._error("fixed-requested per-tile captures must use one selected prefix group")
                        if schedule.get("adaptive_prefix_active") is not False or schedule.get("adaptive_skip_active") is not False:
                            self._error("fixed-requested per-tile captures must not set adaptive prefix flags")
                        if schedule.get("adaptive_execution_applied") is not False:
                            self._error("fixed-requested per-tile captures must not apply adaptive execution")
                else:
                    self._validate_v4_adaptive_schedule(prefix, schedule)
        elif semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
            if self.data.get("backend_selected") not in {"cpu-reference", "hip-direct", "hipblaslt", "ck", "rocwmma"}:
                self._error("exact-wide captures must select cpu-reference, hip-direct, hipblaslt, ck, or rocwmma backend")
            if bound_mode != "global":
                self._error("exact-wide captures must use bound_mode=global")
            if self.data.get("bound_kind") != "none" or self.data.get("bound") != 0:
                self._error("exact-wide captures must use bound_kind=none and bound=0")
            if self.data.get("tile_bounds_u64") is not None:
                self._error("exact-wide captures must use tile_bounds_u64=null")
            if _is_int(prefix) and prefix <= 0:
                self._error("exact-wide captures must use a positive prefix")
            if packed_layout is not None:
                self._error("exact-wide captures must use packed_layout_version=null")
            if residue_chain_length > 1:
                expected_epilogue_type = "residue_current_rns_output"
                if residue_output_mode != "residue_current_rns":
                    self._error("exact-wide residue-current chains must use residue_output_mode=residue_current_rns")
                if self.data.get("m") != self.data.get("n") or self.data.get("n") != self.data.get("k"):
                    self._error("exact-wide residue-current chains must use square m=n=k shapes")
            else:
                expected_epilogue_type = (
                    "exact_wide_signed_limb_export"
                    if semantics == "exact_wide_signed"
                    else "exact_wide_unsigned_limb_export"
                )
                if residue_output_mode != "host_export":
                    self._error("exact-wide host-export captures must use residue_output_mode=host_export")
            if self.data.get("epilogue_type") != expected_epilogue_type:
                self._error(f"exact-wide captures must use {expected_epilogue_type} epilogue")
            if self.data.get("finite_modulus") is not None:
                self._error("exact-wide captures must use finite_modulus=null")
            limb_count = self.data.get("exact_wide_limb_count")
            if not _is_int(limb_count) or limb_count < 1 or limb_count > 32:
                self._error("exact-wide captures must use exact_wide_limb_count in [1, 32]")
            elif status_check is not None:
                expected_status_check = (
                    "required_for_range_check"
                    if (
                        (semantics == "exact_wide_signed" and limb_count < 3)
                        or (semantics == "exact_wide_unsigned" and limb_count < 3)
                    )
                    else "elided_full_width_device_reconstruction"
                )
                if status_check != expected_status_check:
                    self._error(f"exact_wide_export_status_check must be {expected_status_check}")
                if status_check == "elided_full_width_device_reconstruction":
                    event_timings = self.data.get("gpu_event_timings_us")
                    if isinstance(event_timings, dict):
                        for phase in ("exact_wide_export_status_memset", "exact_wide_export_status_d2h"):
                            values = event_timings.get(phase)
                            if isinstance(values, list) and any(
                                _is_number(value) and float(value) != 0.0 for value in values
                            ):
                                self._error(
                                    f"exact-wide status-elided captures must report gpu_event_timings_us.{phase} as zero"
                                )
            if isinstance(schedule, dict) and _is_int(prefix):
                if prefix_policy == "minimum_proven":
                    if schedule.get("min_selected_prefix") != schedule.get("max_selected_prefix"):
                        self._error("exact-wide minimum_proven captures must use one selected prefix")
                    if schedule.get("max_selected_prefix") > prefix:
                        self._error("exact-wide selected schedule prefix must be <= prefix")
                elif schedule.get("min_selected_prefix") != prefix or schedule.get("max_selected_prefix") != prefix:
                    self._error("exact-wide captures must use fixed selected schedule prefix equal to prefix")
                if schedule.get("prefix_group_count") != 1:
                    self._error("exact-wide captures must use one fixed prefix group")
                if schedule.get("adaptive_execution_applied") is True:
                    self._error("exact-wide captures must not apply adaptive execution")
            if isinstance(backend_metadata, dict):
                backend = self.data.get("backend_selected")
                expected_epilogues = {
                    "hipblaslt": "separate_i32_scratch_reduce_rns_output",
                    "ck": "ck_fused_i32_to_centered_residue_rns_output",
                    "rocwmma": "rocwmma_fused_i32_to_centered_residue_rns_output",
                }
                expected_backend_epilogue = expected_epilogues.get(str(backend))
                if (
                    expected_backend_epilogue is not None
                    and backend_metadata.get("epilogue_mode") != expected_backend_epilogue
                ):
                    self._error(
                        f"exact-wide {backend} captures must use "
                        f"backend_metadata.epilogue_mode={expected_backend_epilogue}"
                    )
            metadata = self.data.get("timing_metadata")
            if isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                if self.data.get("backend_selected") == "hip-direct":
                    expected_scope = "direct_hip_default_stream_backend_operation_groups"
                    if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                        self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
                if self.data.get("backend_selected") == "hipblaslt":
                    expected_scope = "hipblaslt_baseline_default_stream_backend_operation_groups"
                    if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                        self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
        elif semantics in {"finite_ring_u8", "finite_field_u8"}:
            finite_oneshot_capture = self._is_finite_oneshot_capture()
            if finite_oneshot_capture:
                if self.data.get("benchmark") != "rns8_finite_u8_public_oneshot":
                    self._error("finite one-shot captures must use benchmark=rns8_finite_u8_public_oneshot")
                if self._benchmark_execution_mode() != "public_oneshot_transient_native_inputs":
                    self._error("finite one-shot captures must use benchmark_execution_mode=public_oneshot_transient_native_inputs")
                if self.data.get("backend_selected") not in {"cpu-reference", "hip-direct"}:
                    self._error("finite one-shot captures must select cpu-reference or hip-direct")
                if self.data.get("backend_requested") not in {"cpu-reference", "cpu", "hip-direct"}:
                    self._error("finite one-shot captures must request cpu or hip-direct")
                if self.data.get("reuse_packed_inputs") is True:
                    self._error("finite one-shot captures must not use packed-input reuse")
                if self.data.get("pack_mode") not in {None, "per_repeat_repack"}:
                    self._error("finite one-shot captures must use pack_mode=per_repeat_repack")
                if self.data.get("prepack_reuse_strategy") not in {None, "none"}:
                    self._error("finite one-shot captures must use prepack_reuse_strategy=none")
            if self.data.get("backend_selected") not in {"cpu-reference", "hip-direct", "hipblaslt", "ck", "rocwmma"}:
                self._error("finite-u8 captures must select cpu-reference, hip-direct, hipblaslt, ck, or rocwmma backend")
            if bound_mode != "global":
                self._error("finite-u8 captures must use bound_mode=global")
            if self.data.get("bound_kind") != "none" or self.data.get("bound") != 0:
                self._error("finite-u8 captures must use bound_kind=none and bound=0")
            if self.data.get("tile_bounds_u64") is not None:
                self._error("finite-u8 captures must use tile_bounds_u64=null")
            if prefix != 0:
                self._error("finite-u8 captures must use prefix=0")
            if packed_layout is not None:
                self._error("finite-u8 captures must use packed_layout_version=null")
            if self.data.get("epilogue_type") != "canonical_u8_export":
                self._error("finite-u8 captures must use canonical_u8_export epilogue")
            modulus = self.data.get("finite_modulus")
            if not _is_int(modulus):
                self._error("finite-u8 captures must include integer finite_modulus")
            elif semantics == "finite_ring_u8" and (modulus < 2 or modulus > 256):
                self._error("finite_ring_u8 finite_modulus must be in [2, 256]")
            elif semantics == "finite_field_u8" and not _is_prime_modulus(modulus):
                self._error("finite_field_u8 finite_modulus must be prime and <= 251")
            raw_metadata = self.data.get("backend_metadata")
            backend_metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
            autotune_key = backend_metadata.get("autotune_key")
            if isinstance(autotune_key, str) and _is_int(modulus):
                required_field = f";finite_modulus={modulus};"
                normalized_key = f";{autotune_key};"
                if required_field not in normalized_key:
                    self._error("finite-u8 backend_metadata.autotune_key must include finite_modulus")
            if self.data.get("backend_selected") == "hip-direct" and _is_int(modulus):
                native_a_reuse_b_capture = self._is_direct_hip_finite_native_a_reuse_b_capture()
                specialized_kernel = (
                    DIRECT_HIP_FINITE_ONESHOT_SPECIALIZED_KERNELS.get(modulus)
                    if finite_oneshot_capture
                    else DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_SPECIALIZED_KERNELS.get(modulus)
                    if native_a_reuse_b_capture
                    else DIRECT_HIP_FINITE_SPECIALIZED_KERNELS.get(modulus)
                )
                generic_kernel = (
                    DIRECT_HIP_FINITE_ONESHOT_GENERIC_KERNEL
                    if finite_oneshot_capture
                    else DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_GENERIC_KERNEL
                    if native_a_reuse_b_capture
                    else DIRECT_HIP_FINITE_GENERIC_KERNEL
                )
                if specialized_kernel is not None:
                    if self.data.get("selected_kernel") != specialized_kernel:
                        self._error(
                            f"direct-HIP finite-u8 modulus {modulus} captures "
                            f"must use selected_kernel={specialized_kernel}"
                        )
                    if backend_metadata.get("isa_evidence") != DIRECT_HIP_FINITE_SPECIALIZED_ISA_EVIDENCE:
                        self._error(
                            "direct-HIP finite-u8 specialized captures must use "
                            f"backend_metadata.isa_evidence={DIRECT_HIP_FINITE_SPECIALIZED_ISA_EVIDENCE}"
                        )
                else:
                    if self.data.get("selected_kernel") != generic_kernel:
                        self._error(
                            "direct-HIP generic finite-u8 captures must use "
                            f"selected_kernel={generic_kernel}"
                        )
                    if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                        self._error(
                            "direct-HIP generic finite-u8 captures must use "
                            f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                        )
                if finite_oneshot_capture:
                    if backend_metadata.get("epilogue_mode") != DIRECT_HIP_FINITE_ONESHOT_EPILOGUE:
                        self._error(
                            "direct-HIP finite one-shot captures must use "
                            f"backend_metadata.epilogue_mode={DIRECT_HIP_FINITE_ONESHOT_EPILOGUE}"
                        )
                    if backend_metadata.get("workspace_mode") != DIRECT_HIP_FINITE_ONESHOT_WORKSPACE:
                        self._error(
                            "direct-HIP finite one-shot captures must use "
                            f"backend_metadata.workspace_mode={DIRECT_HIP_FINITE_ONESHOT_WORKSPACE}"
                        )
                elif native_a_reuse_b_capture:
                    if backend_metadata.get("epilogue_mode") != DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_EPILOGUE:
                        self._error(
                            "direct-HIP finite native-A reuse-B captures must use "
                            f"backend_metadata.epilogue_mode={DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_EPILOGUE}"
                        )
                    if backend_metadata.get("workspace_mode") != DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_WORKSPACE:
                        self._error(
                            "direct-HIP finite native-A reuse-B captures must use "
                            f"backend_metadata.workspace_mode={DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_WORKSPACE}"
                        )
            if self.data.get("backend_selected") == "ck" and _is_int(modulus):
                expected_kernel = CK_FINITE_SPECIALIZED_KERNELS.get(modulus, CK_FINITE_GENERIC_KERNEL)
                if self.data.get("selected_kernel") != expected_kernel:
                    self._error(
                        f"CK finite-u8 modulus {modulus} captures must use selected_kernel={expected_kernel}"
                    )
            if self.data.get("backend_selected") == "rocwmma" and _is_int(modulus):
                expected_kernel = ROCWMMA_FINITE_SPECIALIZED_KERNELS.get(modulus, ROCWMMA_FINITE_GENERIC_KERNEL)
                if self.data.get("selected_kernel") != expected_kernel:
                    self._error(
                        f"rocWMMA finite-u8 modulus {modulus} captures must use selected_kernel={expected_kernel}"
                    )
            if isinstance(schedule, dict):
                for key in ["min_required_prefix", "max_required_prefix", "min_selected_prefix", "max_selected_prefix"]:
                    if schedule.get(key) != 0:
                        self._error(f"finite-u8 captures must use schedule_metadata.{key}=0")
                if schedule.get("prefix_group_count") != 0:
                    self._error("finite-u8 captures must use schedule_metadata.prefix_group_count=0")
                if schedule.get("adaptive_execution_applied") is True:
                    self._error("finite-u8 captures must not apply adaptive execution")
            metadata = self.data.get("timing_metadata")
            if isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                if self.data.get("backend_selected") == "hip-direct":
                    expected_scope = (
                        "direct_hip_oneshot_default_stream_operation_groups"
                        if finite_oneshot_capture
                        else "direct_hip_default_stream_backend_operation_groups"
                    )
                    if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                        self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
                if self.data.get("backend_selected") == "hipblaslt":
                    expected_scope = "hipblaslt_baseline_default_stream_backend_operation_groups"
                    if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                        self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
        elif isinstance(semantics, str):
            self._error(f"unsupported benchmark semantics {semantics}")

        applicable = self.data.get("per_modulus_gemm_estimate_applicable")
        if applicable is not None and not isinstance(applicable, bool):
            self._error("per_modulus_gemm_estimate_applicable must be a boolean")
        elif isinstance(applicable, bool) and _is_int(prefix):
            expected_applicable = (
                prefix > 0
                and not self._is_public_oneshot_capture()
                and not (semantics in {"bounded_i64", "bounded_u64"} and bound_mode == "per_tile")
                and not self._is_direct_hip_vector_to_rns_chain_capture()
                and not self._is_host_api_batch_capture()
                and not self._is_grouped_dispatch_capture()
                and not self._is_hip_graph_replay_capture()
                and self.data.get("backend_selected") != "hip-vector-alu-int64"
            )
            if applicable != expected_applicable:
                self._error("per_modulus_gemm_estimate_applicable must match the fixed-prefix contract")

    def _validate_v4_tile_bounds(self, semantics: Any, schedule: Any) -> None:
        tile_bounds = self.data.get("tile_bounds_u64")
        if not isinstance(tile_bounds, dict):
            self._error("tile_bounds_u64 must be an object for per-tile adaptive captures")
            return
        for key in ["source", "pattern", "order"]:
            if not isinstance(tile_bounds.get(key), str) or not tile_bounds.get(key):
                self._error(f"tile_bounds_u64.{key} must be a nonempty string")
        expected_pattern = "exact_output_tile_max_abs_v1" if semantics == "bounded_i64" else "exact_output_tile_max_unsigned_v1"
        if tile_bounds.get("source") != "exact_seeded_input_prepass":
            self._error("tile_bounds_u64.source must be exact_seeded_input_prepass")
        if tile_bounds.get("pattern") != expected_pattern:
            self._error(f"tile_bounds_u64.pattern must be {expected_pattern}")
        if tile_bounds.get("order") != "row_major_output_tiles":
            self._error("tile_bounds_u64.order must be row_major_output_tiles")
        for key in ["count", "min", "max", "hash_u64"]:
            if not _is_int(tile_bounds.get(key)):
                self._error(f"tile_bounds_u64.{key} must be an integer")
        count = tile_bounds.get("count")
        minimum = tile_bounds.get("min")
        maximum = tile_bounds.get("max")
        if _is_int(count) and count <= 0:
            self._error("tile_bounds_u64.count must be positive")
        if _is_int(minimum) and minimum < 0:
            self._error("tile_bounds_u64.min must be nonnegative")
        if _is_int(maximum) and maximum < 0:
            self._error("tile_bounds_u64.max must be nonnegative")
        if _is_int(minimum) and _is_int(maximum) and minimum > maximum:
            self._error("tile_bounds_u64.min must be <= max")
        if semantics == "bounded_i64" and _is_int(maximum) and maximum > 2**63:
            self._error("bounded_i64 tile_bounds_u64.max must be <= 2^63")
        if _is_int(tile_bounds.get("hash_u64")) and tile_bounds.get("hash_u64") < 0:
            self._error("tile_bounds_u64.hash_u64 must be nonnegative")
        if isinstance(schedule, dict) and _is_int(count) and _is_int(schedule.get("tile_count")):
            if count != schedule.get("tile_count"):
                self._error("tile_bounds_u64.count must match schedule_metadata.tile_count")

    def _validate_v4_adaptive_schedule(self, prefix: Any, schedule: Any) -> None:
        if not isinstance(schedule, dict):
            return
        selected_backend_for_schedule = self.data.get("backend_selected")
        vector_runtime_comparator = (
            selected_backend_for_schedule == "hip-vector-alu-int64"
            and self._is_vector_alu_runtime_capture()
        )
        if vector_runtime_comparator:
            if schedule.get("adaptive_execution_applied") is not False:
                self._error(
                    "per-tile adaptive vector runtime captures must set "
                    "schedule_metadata.adaptive_execution_applied=false"
                )
        elif schedule.get("adaptive_execution_applied") is not True:
            self._error("per-tile adaptive captures must set schedule_metadata.adaptive_execution_applied=true")
        selected_kernel = self.data.get("selected_kernel")
        if not isinstance(selected_kernel, str) or not selected_kernel:
            self._error("per-tile adaptive captures must report selected_kernel")
        else:
            selected_backend = self.data.get("backend_selected")
            zero_output_tiles = (
                _is_int(schedule.get("zero_output_tile_count")) and schedule.get("zero_output_tile_count") > 0
            )
            zero_row_col_products = (
                _is_int(schedule.get("zero_row_col_product_count"))
                and schedule.get("zero_row_col_product_count") > 0
            )
            if zero_output_tiles and zero_row_col_products:
                direct_hip_expected_kernel = DIRECT_HIP_ADAPTIVE_ZERO_TILE_ROW_COL_SKIP_KERNEL_V1
            elif zero_output_tiles:
                direct_hip_expected_kernel = DIRECT_HIP_ADAPTIVE_ZERO_SKIP_KERNEL_V3
            elif zero_row_col_products:
                direct_hip_expected_kernel = DIRECT_HIP_ADAPTIVE_ZERO_ROW_COL_SKIP_KERNEL_V1
            else:
                direct_hip_expected_kernel = DIRECT_HIP_ADAPTIVE_KERNEL_V2
            expected_kernels = {
                "hip-direct": direct_hip_expected_kernel,
                "ck": "ck_wmma_cshuffle_tiled_i8_i32_mod251_255_256_centered_epilogue_v2",
                "rocwmma": "rocwmma_i8_i32_signed_tiled_mod251_255_256_hot_residue_v2",
            }
            expected_kernel = expected_kernels.get(selected_backend)
            if expected_kernel is not None and selected_kernel != expected_kernel:
                self._error(f"per-tile adaptive {selected_backend} captures must use selected_kernel={expected_kernel}")
            if selected_backend == "hip-vector-alu-int64" and selected_kernel not in VECTOR_ALU_SELECTED_KERNELS:
                self._error("per-tile adaptive hip-vector-alu-int64 captures must use a known vector-ALU selected_kernel")
        prefix_group_count = schedule.get("prefix_group_count")
        max_selected = schedule.get("max_selected_prefix")
        min_selected = schedule.get("min_selected_prefix")
        adaptive_prefix_expected = _is_int(prefix_group_count) and prefix_group_count > 1
        if isinstance(schedule.get("adaptive_prefix_active"), bool) and schedule.get("adaptive_prefix_active") != adaptive_prefix_expected:
            self._error("schedule_metadata.adaptive_prefix_active must match prefix_group_count > 1")
        adaptive_skip_expected = _is_int(max_selected) and _is_int(prefix) and max_selected < prefix
        if isinstance(schedule.get("adaptive_skip_active"), bool) and schedule.get("adaptive_skip_active") != adaptive_skip_expected:
            self._error("schedule_metadata.adaptive_skip_active must match max_selected_prefix < prefix")
        if _is_int(prefix_group_count) and prefix_group_count <= 0:
            self._error("per-tile adaptive captures must use at least one prefix group")
        if _is_int(min_selected) and min_selected <= 0:
            self._error("schedule_metadata.min_selected_prefix must be positive for per-tile adaptive captures")
        if _is_int(max_selected) and _is_int(prefix) and max_selected > prefix:
            self._error("schedule_metadata.max_selected_prefix must be <= prefix")
        if schedule.get("adaptive_prefix_active") is not True and schedule.get("adaptive_skip_active") is not True:
            self._error("per-tile adaptive captures must apply prefix grouping or prefix skipping")
        metadata = self.data.get("timing_metadata")
        if isinstance(metadata, dict):
            if self.data.get("backend_selected") == "hip-direct":
                if metadata.get("gpu_event_timing") is not True:
                    self._error("direct-HIP per-tile adaptive captures must include HIP event timings")
                expected_scope = "direct_hip_bounded_adaptive_default_stream_backend_operation_groups"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
            elif self.data.get("backend_selected") == "ck":
                if metadata.get("gpu_event_timing") is not True:
                    self._error("CK per-tile adaptive captures must include HIP event operation-group timings")
                expected_scope = "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
            elif self.data.get("backend_selected") == "rocwmma":
                if metadata.get("gpu_event_timing") is not True:
                    self._error("rocWMMA per-tile adaptive captures must include HIP event operation-group timings")
                expected_scope = "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
            elif self.data.get("backend_selected") == "hip-vector-alu-int64":
                if metadata.get("gpu_event_timing") is not True:
                    self._error("vector-ALU per-tile adaptive captures must include HIP event operation-group timings")
                expected_scope = "vector_alu_default_stream_native_int64_operation_groups"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
        backend_metadata = self.data.get("backend_metadata")
        if isinstance(backend_metadata, dict):
            expected_epilogues = {
                "ck": "ck_fused_i32_to_centered_residue_then_crt_export",
                "rocwmma": "rocwmma_fused_i32_to_centered_residue_then_crt_export",
                "hip-vector-alu-int64": "direct_int64_export",
            }
            expected_epilogue = expected_epilogues.get(
                self.data.get("backend_selected"), "fused_centered_residue_then_crt_export"
            )
            if backend_metadata.get("epilogue_mode") != expected_epilogue:
                self._error(
                    f"per-tile adaptive captures must use backend_metadata.epilogue_mode={expected_epilogue}"
                )
            expected_workspaces = {
                "cpu-reference": "host_reference_workspace",
                "ck": "resident_device_buffers_with_ck_canonical_pack_workspace",
                "rocwmma": "resident_device_buffers_with_rocwmma_pack_workspace",
                "hip-vector-alu-int64": (
                    "native_device_i64_u64_buffers"
                    if self._is_vector_alu_runtime_capture()
                    else "benchmark_owned_device_buffers"
                ),
            }
            expected_workspace = expected_workspaces.get(
                self.data.get("backend_selected"), "resident_device_buffers_with_active_prefix_tiled_schedule"
            )
            if backend_metadata.get("workspace_mode") != expected_workspace:
                self._error(
                    f"per-tile adaptive captures must use backend_metadata.workspace_mode={expected_workspace}"
                )
            expected_isas = {
                "cpu-reference": "not_applicable_cpu",
                "ck": "ck_wmma_cshuffle_int8_matrix_isa_gate_no_int32_global_store_no_divide",
                "rocwmma": "rocwmma_i8_wmma_isa_gate_no_int32_global_store_no_divide",
                "hip-vector-alu-int64": "source_level_192bit_limb_accumulator_no_matrix_engine",
            }
            expected_isa = expected_isas.get(
                self.data.get("backend_selected"), "rns8_hip_direct_reciprocal_isa_gate"
            )
            if backend_metadata.get("isa_evidence") != expected_isa:
                self._error(
                    f"per-tile adaptive captures must use backend_metadata.isa_evidence={expected_isa}"
                )

    def _validate_pack_reuse_fields(self, raw_timings: dict[str, list[float]]) -> None:
        reuse_value = self.data.get("reuse_packed_inputs", False)
        if "reuse_packed_inputs" in self.data and not isinstance(reuse_value, bool):
            self._error("reuse_packed_inputs must be a boolean")
        reuse_packed = reuse_value is True

        pack_mode = self.data.get("pack_mode")
        if pack_mode is not None:
            if pack_mode not in PACK_MODES:
                self._error(f"pack_mode must be one of {sorted(PACK_MODES)}")
            elif reuse_packed and pack_mode == "per_repeat_repack":
                self._error("pack_mode must describe a prepacked reuse mode")
            elif not reuse_packed and pack_mode != "per_repeat_repack":
                self._error("pack_mode must be per_repeat_repack")

        metadata = self.data.get("timing_metadata")
        if isinstance(metadata, dict):
            metadata_mode = metadata.get("pack_mode")
            if metadata_mode is not None:
                if metadata_mode not in PACK_MODES:
                    self._error(f"timing_metadata.pack_mode must be one of {sorted(PACK_MODES)}")
                elif reuse_packed and metadata_mode == "per_repeat_repack":
                    self._error("timing_metadata.pack_mode must describe a prepacked reuse mode")
                elif not reuse_packed and metadata_mode != "per_repeat_repack":
                    self._error("timing_metadata.pack_mode must be per_repeat_repack")
                if pack_mode is not None and metadata_mode != pack_mode:
                    self._error("timing_metadata.pack_mode must match pack_mode")
            metadata_operands = metadata.get("prepack_reuse_operands")
            if metadata_operands is not None and isinstance(pack_mode, str) and pack_mode in PACK_MODE_OPERANDS:
                if metadata_operands != PACK_MODE_OPERANDS[pack_mode]:
                    self._error("timing_metadata.prepack_reuse_operands must match pack_mode")
            metadata_strategy = metadata.get("prepack_reuse_strategy")
            if metadata_strategy is not None:
                if metadata_strategy not in PREPACK_REUSE_STRATEGIES:
                    self._error(
                        f"timing_metadata.prepack_reuse_strategy must be one of {sorted(PREPACK_REUSE_STRATEGIES)}"
                    )

        operands = self.data.get("prepack_reuse_operands")
        if operands is not None and isinstance(pack_mode, str) and pack_mode in PACK_MODE_OPERANDS:
            if operands != PACK_MODE_OPERANDS[pack_mode]:
                self._error("prepack_reuse_operands must match pack_mode")
        strategy = self.data.get("prepack_reuse_strategy")
        if strategy is not None:
            if strategy not in PREPACK_REUSE_STRATEGIES:
                self._error(f"prepack_reuse_strategy must be one of {sorted(PREPACK_REUSE_STRATEGIES)}")
            metadata = self.data.get("timing_metadata")
            if isinstance(metadata, dict) and metadata.get("prepack_reuse_strategy") is not None:
                if metadata.get("prepack_reuse_strategy") != strategy:
                    self._error("timing_metadata.prepack_reuse_strategy must match prepack_reuse_strategy")
            if reuse_packed and strategy == "none":
                self._error("prepacked reuse captures must not use prepack_reuse_strategy=none")
            if not reuse_packed and strategy != "none":
                self._error("per-repeat repack captures must use prepack_reuse_strategy=none")
            if strategy == "rocwmma_reusable_b_cache":
                if pack_mode != "prepacked_reuse_b":
                    self._error("rocwmma_reusable_b_cache captures must use pack_mode=prepacked_reuse_b")
                if operands is not None and operands != ["B"]:
                    self._error("rocwmma_reusable_b_cache captures must reuse only operand B")
                if self.data.get("backend_selected") != "rocwmma":
                    self._error("rocwmma_reusable_b_cache captures must select backend_selected=rocwmma")

        prepack_setup = self.data.get("prepack_setup_us")
        avg_prepack_setup = self.data.get("avg_prepack_setup_us")
        if reuse_packed:
            if not _is_int(prepack_setup) or prepack_setup < 0:
                self._error("prepacked reuse captures must include nonnegative integer prepack_setup_us")
            if not _is_number(avg_prepack_setup):
                self._error("prepacked reuse captures must include avg_prepack_setup_us")
            elif _is_int(prepack_setup) and not _close(float(avg_prepack_setup), float(prepack_setup)):
                self._error("avg_prepack_setup_us must match prepack_setup_us")
            if pack_mode == "prepacked_reuse":
                pack_values = raw_timings.get("pack")
                if pack_values is not None and any(value != 0.0 for value in pack_values):
                    self._error("prepacked reuse captures must report raw_timings_us.pack as zero-valued repeats")
                event_timings = self.data.get("gpu_event_timings_us")
                if isinstance(event_timings, dict):
                    for phase in ["pack_h2d", "pack_kernel", "finite_pack_h2d", "finite_pack_kernel", "pack"]:
                        values = event_timings.get(phase)
                        if isinstance(values, list) and any(_is_number(value) and float(value) != 0.0 for value in values):
                            self._error(f"prepacked reuse captures must report gpu_event_timings_us.{phase} as zero")
        else:
            if "prepack_setup_us" in self.data and prepack_setup is not None:
                self._error("per-repeat repack captures must use prepack_setup_us=null")
            if "avg_prepack_setup_us" in self.data and avg_prepack_setup is not None:
                self._error("per-repeat repack captures must use avg_prepack_setup_us=null")

    def _validate_residue_current_timings(self, raw_timings: dict[str, list[float]]) -> None:
        if not self._is_residue_current_chain_capture():
            return
        values = raw_timings.get("crt_export")
        if not isinstance(values, list) or any(value != 0.0 for value in values):
            self._error("residue-current chain captures must report raw_timings_us.crt_export as zero-valued repeats")
        avg_export = self.data.get("avg_crt_export_us")
        if _is_number(avg_export) and float(avg_export) != 0.0:
            self._error("residue-current chain captures must report avg_crt_export_us=0")

    def _validate_bounded_oneshot_timings(self, raw_timings: dict[str, list[float]]) -> None:
        if not self._is_public_oneshot_capture():
            return
        for phase, field in [("pack", "avg_pack_us"), ("crt_export", "avg_crt_export_us"), ("matrix_alloc", "avg_matrix_alloc_us")]:
            values = raw_timings.get(phase)
            if not isinstance(values, list) or any(value != 0.0 for value in values):
                self._error(f"public one-shot captures must report raw_timings_us.{phase} as zero-valued")
            average_value = self.data.get(field)
            if _is_number(average_value) and float(average_value) != 0.0:
                self._error(f"public one-shot captures must report {field}=0")
        gemm_values = raw_timings.get("rns_gemm")
        e2e_values = raw_timings.get("end_to_end")
        if isinstance(gemm_values, list) and isinstance(e2e_values, list) and gemm_values != e2e_values:
            self._error("public one-shot captures must report raw_timings_us.rns_gemm equal to end_to_end")

    def _is_all_zero_direct_hip_adaptive_capture(self) -> bool:
        schedule = self.data.get("schedule_metadata")
        if not isinstance(schedule, dict):
            return False
        tile_count = schedule.get("tile_count")
        zero_count = schedule.get("zero_output_tile_count")
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and schedule.get("adaptive_execution_applied") is True
            and _is_int(tile_count)
            and tile_count > 0
            and _is_int(zero_count)
            and zero_count == tile_count
        )

    def _validate_all_zero_direct_hip_adaptive_timings(self, raw_timings: dict[str, list[float]]) -> None:
        if not self._is_all_zero_direct_hip_adaptive_capture():
            return
        values = raw_timings.get("pack")
        if not isinstance(values, list) or any(value != 0.0 for value in values):
            self._error(
                "all-zero direct-HIP adaptive captures must report raw_timings_us.pack as zero-valued repeats"
            )
        avg_pack = self.data.get("avg_pack_us")
        if _is_number(avg_pack) and float(avg_pack) != 0.0:
            self._error("all-zero direct-HIP adaptive captures must report avg_pack_us=0")

    def _validate_raw_timings(self) -> dict[str, list[float]]:
        raw = self._require("raw_timings_us", "dict")
        repeats = self.data.get("repeats")
        result: dict[str, list[float]] = {}
        if not isinstance(raw, dict) or not _is_int(repeats):
            return result
        for phase in self._timing_phases():
            values = raw.get(phase)
            if not isinstance(values, list):
                self._error(f"raw_timings_us.{phase} must be an array")
                continue
            expected_length = repeats if phase in REPEATED_TIMING_PHASES else 1
            if len(values) != expected_length:
                self._error(f"raw_timings_us.{phase} length {len(values)} does not match expected {expected_length}")
            parsed: list[float] = []
            for index, value in enumerate(values):
                if not _is_int(value) or value < 0:
                    self._error(f"raw_timings_us.{phase}[{index}] must be a nonnegative integer")
                else:
                    parsed.append(float(value))
            result[phase] = parsed
        return result

    def _validate_timing_summaries(
        self,
        raw_values: dict[str, list[float]],
        summary_key: str,
        phases: list[str],
    ) -> None:
        summary = self._require(summary_key, "dict")
        if not isinstance(summary, dict):
            return
        for phase in phases:
            item = summary.get(phase)
            if not isinstance(item, dict):
                self._error(f"{summary_key}.{phase} must be an object")
                continue
            for key in ["avg", "median", "p95"]:
                if not _is_number(item.get(key)):
                    self._error(f"{summary_key}.{phase}.{key} must be a finite number")
            values = raw_values.get(phase)
            if values is None:
                continue
            expected = {
                "avg": _average(values),
                "median": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
            for key, expected_value in expected.items():
                actual = item.get(key)
                if _is_number(actual) and not _close(float(actual), expected_value):
                    self._error(
                        f"{summary_key}.{phase}.{key}={actual} does not match raw {key} {expected_value}"
                    )

    def _validate_top_level_averages(self, raw_timings: dict[str, list[float]]) -> None:
        fields = [
            ("avg_planning_us", "planning"),
            ("avg_matrix_alloc_us", "matrix_alloc"),
            ("avg_pack_us", "pack"),
            ("avg_rns_gemm_us", "rns_gemm"),
            ("avg_crt_export_us", "crt_export"),
            ("avg_end_to_end_us", "end_to_end"),
        ]
        if GLOBAL_BOUND_TIMING_PHASE in self._timing_phases():
            fields.insert(0, ("avg_global_bound_scan_us", GLOBAL_BOUND_TIMING_PHASE))
        fields.insert(1, ("avg_scheduling_us", "scheduling"))
        if PER_TILE_TIMING_PHASE in self._timing_phases():
            fields.insert(2, ("avg_tile_bound_scan_us", PER_TILE_TIMING_PHASE))
        for field, phase in fields:
            value = self._require(field, "number")
            values = raw_timings.get(phase)
            if _is_number(value) and values is not None and not _close(float(value), _average(values)):
                self._error(f"{field}={value} does not match raw average {_average(values)}")
        schedule_query = self._require("schedule_query_us", "number")
        scheduling_values = raw_timings.get("scheduling")
        if _is_number(schedule_query) and scheduling_values is not None and not _close(
            float(schedule_query), _average(scheduling_values)
        ):
            self._error(f"schedule_query_us={schedule_query} does not match raw average {_average(scheduling_values)}")
        if PER_TILE_TIMING_PHASE in raw_timings:
            tile_bound_scan = self._require("tile_bound_scan_us", "number")
            tile_bound_values = raw_timings.get(PER_TILE_TIMING_PHASE)
            if _is_number(tile_bound_scan) and tile_bound_values is not None and not _close(
                float(tile_bound_scan), _average(tile_bound_values)
            ):
                self._error(
                    f"tile_bound_scan_us={tile_bound_scan} does not match raw average "
                    f"{_average(tile_bound_values)}"
                )
        if GLOBAL_BOUND_TIMING_PHASE in raw_timings:
            global_bound_scan = self._require("global_bound_scan_us", "number")
            global_bound_values = raw_timings.get(GLOBAL_BOUND_TIMING_PHASE)
            if _is_number(global_bound_scan) and global_bound_values is not None and not _close(
                float(global_bound_scan), _average(global_bound_values)
            ):
                self._error(
                    f"global_bound_scan_us={global_bound_scan} does not match raw average "
                    f"{_average(global_bound_values)}"
                )
        prefix = self.data.get("selected_prefix", self.data.get("prefix"))
        applicable = self.data.get("per_modulus_gemm_estimate_applicable")
        per_modulus = self._require("avg_per_modulus_gemm_estimate_us", "number")
        gemm_values = raw_timings.get("rns_gemm")
        if (
            _is_number(per_modulus)
            and _is_int(prefix)
            and gemm_values is not None
            and applicable is not False
        ):
            expected = _average(gemm_values) / float(prefix) if prefix > 0 else _average(gemm_values)
            if not _close(float(per_modulus), expected):
                self._error(f"avg_per_modulus_gemm_estimate_us={per_modulus} does not match expected {expected}")

    def _gpu_event_selected_prefix_count(self) -> int:
        semantics = self.data.get("semantics")
        if semantics in {"finite_ring_u8", "finite_field_u8", "wrap_u64_mod_2_64"}:
            return 0
        schedule = self.data.get("schedule_metadata")
        if isinstance(schedule, dict):
            max_selected = schedule.get("max_selected_prefix")
            if _is_int(max_selected) and max_selected > 0:
                return int(max_selected)
        prefix = self.data.get("prefix")
        return int(prefix) if _is_int(prefix) and prefix > 0 else 0

    def _uses_rocwmma_prepacked_b_cache(self) -> bool:
        if self.data.get("prepack_reuse_strategy") == "rocwmma_reusable_b_cache":
            return True
        metadata = self.data.get("timing_metadata")
        return isinstance(metadata, dict) and metadata.get("prepack_reuse_strategy") == "rocwmma_reusable_b_cache"

    @staticmethod
    def _prefix_event_label(prefix: str, index: int, suffix: str) -> str:
        return f"{prefix}{index:02d}_{suffix}"

    def _ck_deep_gpu_event_phases(self, prefix_count: int, zero_output_tiles: bool) -> list[str]:
        phases = [
            "ck_pack_a_kernel",
            "ck_pack_b_kernel",
            "ck_wmma_cshuffle_matmul",
            "ck_copy_centered_kernel",
            "ck_add_centered_kernel",
        ]
        if zero_output_tiles:
            phases.append("ck_zero_output_tile_memset")
        for index in range(prefix_count):
            phases.extend(
                [
                    self._prefix_event_label("ck_prefix_", index, "pack_a"),
                    self._prefix_event_label("ck_prefix_", index, "pack_b"),
                    self._prefix_event_label("ck_prefix_", index, "matmul"),
                    self._prefix_event_label("ck_prefix_", index, "copy_centered"),
                    self._prefix_event_label("ck_prefix_", index, "add_centered"),
                ]
            )
        return phases

    def _rocwmma_deep_gpu_event_phases(
        self,
        prefix_count: int,
        use_prepacked_b: bool,
        zero_output_tiles: bool,
    ) -> list[str]:
        if use_prepacked_b:
            phases = ["rocwmma_pack_a_prepacked_b_kernel", "rocwmma_matmul_prepacked_b_kernel"]
            if zero_output_tiles:
                phases.append("rocwmma_zero_output_tile_memset")
            for index in range(prefix_count):
                phases.extend(
                    [
                        self._prefix_event_label("rocwmma_prefix_", index, "pack_a_prepacked_b"),
                        self._prefix_event_label("rocwmma_prefix_", index, "matmul_prepacked_b"),
                    ]
            )
            return phases
        phases = ["rocwmma_pack_a_kernel", "rocwmma_pack_b_kernel", "rocwmma_matmul_kernel"]
        if zero_output_tiles:
            phases.append("rocwmma_zero_output_tile_memset")
        for index in range(prefix_count):
            phases.extend(
                [
                    self._prefix_event_label("rocwmma_prefix_", index, "pack_a"),
                    self._prefix_event_label("rocwmma_prefix_", index, "pack_b"),
                    self._prefix_event_label("rocwmma_prefix_", index, "matmul"),
                ]
            )
        return phases

    def _expected_vector_gpu_event_phases(self) -> list[str]:
        kernel = (
            "vector_alu_i64_kernel"
            if self.data.get("semantics") == "bounded_i64"
            or self.data.get("selected_kernel") == "hip_vector_alu_i64_exact_192b_v1"
            else "vector_alu_u64_kernel"
        )
        return [
            "vector_alu_pack_a_h2d",
            "vector_alu_pack_b_h2d",
            "pack",
            "vector_alu_status_memset",
            kernel,
            "rns_gemm",
            "vector_alu_status_d2h",
            "vector_alu_output_d2h",
            "crt_export",
        ]

    def _expected_accelerator_deep_gpu_event_phases(self) -> list[str] | None:
        backend = self.data.get("backend_selected")
        if backend not in {"ck", "rocwmma"} or self._is_wrap64_rocwmma_candidate():
            return None
        semantics = self.data.get("semantics")
        use_prepacked_b = backend == "rocwmma" and self._uses_rocwmma_prepacked_b_cache()
        gemm_group = "rns_gemm_prepacked_b_kernel_group" if use_prepacked_b else "rns_gemm_kernel_group"
        if semantics in {"finite_ring_u8", "finite_field_u8"}:
            phases = ["finite_pack_h2d", "finite_pack_kernel", "pack", "rns_gemm_kernel_group"]
        elif semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
            phases = ["pack_h2d", "pack_kernel", "pack", gemm_group]
        else:
            phases = ["pack_h2d", "pack_kernel", "pack", gemm_group]
        prefix_count = self._gpu_event_selected_prefix_count()
        schedule = self.data.get("schedule_metadata")
        zero_output_tiles = (
            isinstance(schedule, dict)
            and _is_int(schedule.get("zero_output_tile_count"))
            and schedule.get("zero_output_tile_count") > 0
        )
        if backend == "ck":
            phases.extend(self._ck_deep_gpu_event_phases(prefix_count, zero_output_tiles))
        else:
            phases.extend(self._rocwmma_deep_gpu_event_phases(prefix_count, use_prepacked_b, zero_output_tiles))
        phases.append("rns_gemm")
        if self._is_residue_current_chain_capture():
            return phases
        if semantics in {"finite_ring_u8", "finite_field_u8"}:
            phases.extend(["finite_export_kernel", "finite_export_d2h", "crt_export"])
        elif semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
            phases.extend(
                [
                    "exact_wide_export_status_memset",
                    "exact_wide_export_kernel",
                    "exact_wide_export_status_d2h",
                    "exact_wide_export_d2h",
                    "crt_export",
                ]
            )
        else:
            phases.extend(
                [
                    "crt_export_status_memset",
                    "crt_export_kernel",
                    "crt_export_status_d2h",
                    "crt_export_d2h",
                    "crt_export",
                ]
            )
        return phases

    @staticmethod
    def _is_deep_accelerator_gpu_event_label(phase: str) -> bool:
        return (
            phase in CK_DEEP_GPU_EVENT_LABELS
            or phase in ROCWMMA_DEEP_GPU_EVENT_LABELS
            or CK_PREFIX_EVENT_RE.match(phase) is not None
            or ROCWMMA_PREFIX_EVENT_RE.match(phase) is not None
        )

    def _validate_expected_gpu_event_phases(self, scope: Any, phases: list[str]) -> None:
        backend = self.data.get("backend_selected")
        if self._is_finite_oneshot_capture() and backend == "hip-direct":
            expected = DIRECT_HIP_FINITE_ONESHOT_GPU_EVENT_PHASES
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(f"direct-HIP finite one-shot GPU event phase set is incomplete; missing {', '.join(missing)}")
                if extra:
                    self._error(f"direct-HIP finite one-shot GPU event phase set contains undeclared phases: {', '.join(extra)}")
                if not missing and not extra:
                    self._error("direct-HIP finite one-shot GPU event phase order must match the public API operation order")
            return
        if self._is_direct_hip_bounded_native_input_oneshot_capture() and backend == "hip-direct":
            expected = DIRECT_HIP_ONESHOT_GPU_EVENT_PHASES
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(f"direct-HIP one-shot GPU event phase set is incomplete; missing {', '.join(missing)}")
                if extra:
                    self._error(f"direct-HIP one-shot GPU event phase set contains undeclared phases: {', '.join(extra)}")
                if not missing and not extra:
                    self._error("direct-HIP one-shot GPU event phase order must match the public API operation order")
            return
        if self._is_direct_hip_bounded_resident_fallback_oneshot_capture() and backend == "hip-direct":
            expected = DIRECT_HIP_ONESHOT_RESIDENT_FALLBACK_GPU_EVENT_PHASES
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(
                        "direct-HIP one-shot resident fallback GPU event phase set is incomplete; "
                        f"missing {', '.join(missing)}"
                    )
                if extra:
                    self._error(
                        "direct-HIP one-shot resident fallback GPU event phase set contains undeclared phases: "
                        f"{', '.join(extra)}"
                    )
                if not missing and not extra:
                    self._error(
                        "direct-HIP one-shot resident fallback GPU event phase order must match the resident pack/GEMM/export order"
                    )
            return
        if self._is_direct_hip_bounded_residue_channel_fusion_capture():
            expected = [
                "bounded_uniform_small_i8_a_h2d",
                "bounded_uniform_small_i8_b_h2d",
                "pack",
                "bounded_uniform_small_i8_ab_transient_gemm_kernel_group",
                "rns_gemm",
                "crt_export_status_memset",
                "crt_export_kernel",
                "crt_export_status_d2h",
                "crt_export_d2h",
                "crt_export",
            ]
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(
                        "direct-HIP residue-channel fusion GPU event phase set is incomplete; "
                        f"missing {', '.join(missing)}"
                    )
                if extra:
                    self._error(
                        "direct-HIP residue-channel fusion GPU event phase set contains undeclared phases: "
                        f"{', '.join(extra)}"
                    )
                if not missing and not extra:
                    self._error("direct-HIP residue-channel fusion GPU event phase order must match the operation order")
            return
        if self._is_direct_hip_bounded_uniform_small_transient_capture():
            expected = [
                "bounded_uniform_small_i8_a_h2d",
                "bounded_uniform_small_i8_b_h2d",
                "pack",
                "bounded_uniform_small_i8_ab_transient_gemm_kernel_group",
                "rns_gemm",
                "crt_export_status_memset",
                "crt_export_kernel",
                "crt_export_status_d2h",
                "crt_export_d2h",
                "crt_export",
            ]
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(
                        "direct-HIP bounded uniform-small transient GPU event phase set is incomplete; "
                        f"missing {', '.join(missing)}"
                    )
                if extra:
                    self._error(
                        "direct-HIP bounded uniform-small transient GPU event phase set contains undeclared phases: "
                        f"{', '.join(extra)}"
                    )
                if not missing and not extra:
                    self._error(
                        "direct-HIP bounded uniform-small transient GPU event phase order must match the operation order"
                    )
            return
        if self._is_direct_hip_bounded_native_a_reuse_b_capture():
            gemm_event = (
                "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
                if self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
                else "bounded_native_a_reuse_b_gemm_kernel_group"
            )
            if self._is_direct_hip_bounded_native_a_reuse_b_u64_large_colpair():
                gemm_event = "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
            expected = [
                "pack_h2d",
                "pack_kernel",
                "pack",
                gemm_event,
                "rns_gemm",
                "crt_export_status_memset",
                "crt_export_kernel",
                "crt_export_status_d2h",
                "crt_export_d2h",
                "crt_export",
            ]
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(
                        "direct-HIP bounded native-A reuse-B GPU event phase set is incomplete; "
                        f"missing {', '.join(missing)}"
                    )
                if extra:
                    self._error(
                        "direct-HIP bounded native-A reuse-B GPU event phase set contains undeclared phases: "
                        f"{', '.join(extra)}"
                    )
                if not missing and not extra:
                    self._error(
                        "direct-HIP bounded native-A reuse-B GPU event phase order must match the operation order"
                    )
            return
        if self._is_direct_hip_bounded_uniform_small_reuse_a_capture():
            expected = [
                "pack_h2d",
                "pack_kernel",
                "pack",
                "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group",
                "rns_gemm",
                "crt_export_status_memset",
                "crt_export_kernel",
                "crt_export_status_d2h",
                "crt_export_d2h",
                "crt_export",
            ]
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(
                        "direct-HIP bounded uniform-small reuse-A GPU event phase set is incomplete; "
                        f"missing {', '.join(missing)}"
                    )
                if extra:
                    self._error(
                        "direct-HIP bounded uniform-small reuse-A GPU event phase set contains undeclared phases: "
                        f"{', '.join(extra)}"
                    )
                if not missing and not extra:
                    self._error(
                        "direct-HIP bounded uniform-small reuse-A GPU event phase order must match the operation order"
                    )
            return
        if self._is_direct_hip_bounded_native_b_reuse_a_u64_large_colpair_capture():
            expected = [
                "pack_h2d",
                "pack_kernel",
                "pack",
                "bounded_native_b_colpair_reuse_a_gemm_kernel_group",
                "rns_gemm",
                "crt_export_status_memset",
                "crt_export_kernel",
                "crt_export_status_d2h",
                "crt_export_d2h",
                "crt_export",
            ]
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(
                        "direct-HIP bounded native-B reuse-A GPU event phase set is incomplete; "
                        f"missing {', '.join(missing)}"
                    )
                if extra:
                    self._error(
                        "direct-HIP bounded native-B reuse-A GPU event phase set contains undeclared phases: "
                        f"{', '.join(extra)}"
                    )
                if not missing and not extra:
                    self._error(
                        "direct-HIP bounded native-B reuse-A GPU event phase order must match the operation order"
                    )
            return
        if self._is_direct_hip_native_to_rns_bridge_capture():
            conversion_event = (
                "native_i64_to_rns_kernel"
                if self.data.get("semantics") == "bounded_i64"
                else "native_u64_to_rns_kernel"
            )
            expected = [
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
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(
                        "direct-HIP native-to-RNS bridge GPU event phase set is incomplete; "
                        f"missing {', '.join(missing)}"
                    )
                if extra:
                    self._error(
                        "direct-HIP native-to-RNS bridge GPU event phase set contains undeclared phases: "
                        f"{', '.join(extra)}"
                    )
                if not missing and not extra:
                    self._error("direct-HIP native-to-RNS bridge GPU event phase order must match the operation order")
            return
        if self._is_direct_hip_vector_to_rns_chain_capture():
            conversion_event = (
                "native_i64_to_rns_kernel"
                if self.data.get("semantics") == "bounded_i64"
                else "native_u64_to_rns_kernel"
            )
            vector_kernel = (
                "vector_alu_i64_kernel"
                if self.data.get("semantics") == "bounded_i64"
                else "vector_alu_u64_kernel"
            )
            expected = [
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
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(
                        "direct-HIP vector-to-RNS chain GPU event phase set is incomplete; "
                        f"missing {', '.join(missing)}"
                    )
                if extra:
                    self._error(
                        "direct-HIP vector-to-RNS chain GPU event phase set contains undeclared phases: "
                        f"{', '.join(extra)}"
                    )
                if not missing and not extra:
                    self._error("direct-HIP vector-to-RNS chain GPU event phase order must match the operation order")
            return
        if self.data.get("semantics") == "wrap_u64_mod_2_64" and backend == "hip-direct":
            expected = [
                "pack_h2d",
                "pack_kernel",
                "pack",
                wrap64_hip_expected_gemm_event_label(self.data.get("selected_kernel")),
                "rns_gemm",
                "wrap64_export_kernel",
                "wrap64_export_d2h",
                "crt_export",
            ]
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(f"direct-HIP wrap64 GPU event phase set is incomplete; missing {', '.join(missing)}")
                if extra:
                    self._error(
                        f"direct-HIP wrap64 GPU event phase set contains undeclared phases: {', '.join(extra)}"
                    )
                if not missing and not extra:
                    self._error("direct-HIP wrap64 GPU event phase order must match the operation order")
            return
        if backend == "hip-vector-alu-int64":
            expected = self._expected_vector_gpu_event_phases()
            if phases != expected:
                missing = [phase for phase in expected if phase not in phases]
                extra = [phase for phase in phases if phase not in expected]
                if missing:
                    self._error(f"vector-ALU GPU event phase set is incomplete; missing {', '.join(missing)}")
                if extra:
                    self._error(f"vector-ALU GPU event phase set contains undeclared phases: {', '.join(extra)}")
                if not missing and not extra:
                    self._error("vector-ALU GPU event phase order must match the native int64 operation order")
            return
        if backend not in {"ck", "rocwmma"} or self._is_wrap64_rocwmma_candidate():
            return
        deep_labels = [phase for phase in phases if self._is_deep_accelerator_gpu_event_label(phase)]
        if scope == OLD_ACCELERATOR_GPU_EVENT_SCOPE:
            if deep_labels:
                self._error(
                    "deep accelerator GPU event labels require "
                    f"timing_metadata.gpu_event_timing_source_scope={DEEP_ACCELERATOR_GPU_EVENT_SCOPE}"
                )
            return
        if scope != DEEP_ACCELERATOR_GPU_EVENT_SCOPE:
            return
        expected = self._expected_accelerator_deep_gpu_event_phases()
        if expected is None or phases == expected:
            return
        missing = [phase for phase in expected if phase not in phases]
        extra = [phase for phase in phases if phase not in expected]
        if missing:
            self._error(f"deep accelerator GPU event phase set is incomplete; missing {', '.join(missing)}")
        if extra:
            self._error(f"deep accelerator GPU event phase set contains undeclared phases: {', '.join(extra)}")
        if not missing and not extra:
            self._error("deep accelerator GPU event phase order must match the selected backend operation order")

    def _expected_status_event_labels(self) -> list[str]:
        backend = self.data.get("backend_selected")
        semantics = self.data.get("semantics")
        if backend == "hip-vector-alu-int64":
            return ["vector_alu_status_memset", "vector_alu_status_d2h"]
        if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
            return ["exact_wide_export_status_memset", "exact_wide_export_status_d2h"]
        if semantics in {"bounded_i64", "bounded_u64"}:
            return ["crt_export_status_memset", "crt_export_status_d2h"]
        return []

    def _known_status_event_labels(self) -> set[str]:
        return {
            "crt_export_status_memset",
            "crt_export_status_d2h",
            "exact_wide_export_status_memset",
            "exact_wide_export_status_d2h",
            "vector_alu_status_memset",
            "vector_alu_status_d2h",
        }

    def _validate_status_event_consistency(self, phases: list[str], parsed: dict[str, list[float]]) -> None:
        output_policy = self.data.get("output_policy")
        if not isinstance(output_policy, dict):
            return
        handling = output_policy.get("status_handling")
        phase_set = set(phases)
        expected = self._expected_status_event_labels()
        known_present = sorted(self._known_status_event_labels() & phase_set)
        if handling == "required":
            for label in expected:
                if label not in phase_set:
                    self._error(f"output_policy.status_handling=required requires GPU event phase {label}")
            return
        if handling == "not_applicable":
            for label in known_present:
                self._error(f"output_policy.status_handling=not_applicable forbids GPU event phase {label}")
            return
        if handling == "structurally_elided":
            for label in known_present:
                values = parsed.get(label, [])
                if any(value != 0.0 for value in values):
                    self._error(
                        "output_policy.status_handling=structurally_elided requires "
                        f"gpu_event_timings_us.{label} to be zero-filled when present"
                    )

    def _validate_gpu_events(self) -> None:
        metadata = self.data.get("timing_metadata")
        if not isinstance(metadata, dict):
            return
        enabled = metadata.get("gpu_event_timing")
        repeats = self.data.get("repeats")
        if not isinstance(enabled, bool) or not _is_int(repeats):
            return
        selected_backend = self.data.get("backend_selected")
        residue_current_chain = self._is_residue_current_chain_capture()
        if selected_backend in {"ck", "rocwmma", "hip-vector-alu-int64"} and enabled is not True and not residue_current_chain:
            self._error(f"{selected_backend} captures must include HIP event operation-group timings")
        timings = self.data.get("gpu_event_timings_us")
        summary = self.data.get("gpu_event_timing_summary_us")
        if not enabled:
            if timings is not None:
                self._error("gpu_event_timings_us must be null when gpu_event_timing is false")
            if summary is not None:
                self._error("gpu_event_timing_summary_us must be null when gpu_event_timing is false")
            if metadata.get("gpu_event_phase_order") is not None:
                self._error("timing_metadata.gpu_event_phase_order must be null when events are unavailable")
            if metadata.get("gpu_event_timing_source") is not None:
                self._error("timing_metadata.gpu_event_timing_source must be null when events are unavailable")
            if metadata.get("gpu_event_timing_source_scope") is not None:
                self._error("timing_metadata.gpu_event_timing_source_scope must be null when events are unavailable")
            return
        source = metadata.get("gpu_event_timing_source")
        scope = metadata.get("gpu_event_timing_source_scope")
        if not isinstance(source, str):
            self._error("timing_metadata.gpu_event_timing_source must be a string when events are available")
        elif source != "hipEventElapsedTime":
            self._error("timing_metadata.gpu_event_timing_source must be hipEventElapsedTime")
        if not isinstance(scope, str):
            self._error("timing_metadata.gpu_event_timing_source_scope must be a string when events are available")
        elif selected_backend == "hip-direct" and scope not in DIRECT_HIP_GPU_EVENT_SCOPES:
            expected = ", ".join(sorted(DIRECT_HIP_GPU_EVENT_SCOPES))
            self._error(f"timing_metadata.gpu_event_timing_source_scope must be a known direct-HIP scope: {expected}")
        elif selected_backend == "hipblaslt" and scope not in HIPBLASLT_GPU_EVENT_SCOPES:
            expected = ", ".join(sorted(HIPBLASLT_GPU_EVENT_SCOPES))
            self._error(f"timing_metadata.gpu_event_timing_source_scope must be a known hipBLASLt scope: {expected}")
        elif selected_backend in {"ck", "rocwmma"} and scope not in ACCELERATOR_GPU_EVENT_SCOPES:
            expected = ", ".join(sorted(ACCELERATOR_GPU_EVENT_SCOPES))
            self._error(f"timing_metadata.gpu_event_timing_source_scope must be a known accelerator scope: {expected}")
        elif selected_backend == "hip-vector-alu-int64" and scope not in VECTOR_ALU_GPU_EVENT_SCOPES:
            expected = ", ".join(sorted(VECTOR_ALU_GPU_EVENT_SCOPES))
            self._error(f"timing_metadata.gpu_event_timing_source_scope must be a known vector-ALU scope: {expected}")
        if not isinstance(timings, dict):
            self._error("gpu_event_timings_us must be an object when gpu_event_timing is true")
            return
        phases = self._gpu_event_phases(metadata)
        if not phases:
            return
        self._validate_expected_gpu_event_phases(scope, phases)
        phase_set = set(phases)
        timing_keys = set(timings.keys())
        if timing_keys != phase_set:
            for phase in sorted(phase_set - timing_keys):
                self._error(f"gpu_event_timings_us.{phase} must be an array")
            for phase in sorted(timing_keys - phase_set):
                self._error(f"gpu_event_timings_us contains undeclared phase {phase}")
        if isinstance(summary, dict):
            summary_keys = set(summary.keys())
            if summary_keys != phase_set:
                for phase in sorted(phase_set - summary_keys):
                    self._error(f"gpu_event_timing_summary_us.{phase} must be an object")
                for phase in sorted(summary_keys - phase_set):
                    self._error(f"gpu_event_timing_summary_us contains undeclared phase {phase}")
        parsed: dict[str, list[float]] = {}
        for phase in phases:
            values = timings.get(phase)
            if not isinstance(values, list):
                self._error(f"gpu_event_timings_us.{phase} must be an array")
                continue
            if len(values) != repeats:
                self._error(f"gpu_event_timings_us.{phase} length {len(values)} does not match repeats {repeats}")
            parsed_values: list[float] = []
            for index, value in enumerate(values):
                if not _is_number(value) or float(value) < 0.0:
                    self._error(f"gpu_event_timings_us.{phase}[{index}] must be a nonnegative finite number")
                else:
                    parsed_values.append(float(value))
            parsed[phase] = parsed_values
        self._validate_status_event_consistency(phases, parsed)
        self._validate_timing_summaries(parsed, "gpu_event_timing_summary_us", phases)
        if self._is_all_zero_direct_hip_adaptive_capture():
            for phase in ["pack_h2d", "pack_kernel", "pack"]:
                values = parsed.get(phase)
                if isinstance(values, list) and any(value != 0.0 for value in values):
                    self._error(
                        f"all-zero direct-HIP adaptive captures must report gpu_event_timings_us.{phase} as zero"
                    )

    def _gpu_event_phases(self, metadata: dict[str, Any]) -> list[str]:
        phase_order = metadata.get("gpu_event_phase_order")
        if not isinstance(phase_order, list) or not all(isinstance(item, str) for item in phase_order):
            self._error("timing_metadata.gpu_event_phase_order must be an array of strings when events are available")
            return []
        if len(set(phase_order)) != len(phase_order):
            self._error("timing_metadata.gpu_event_phase_order must not contain duplicates")
            return []
        return list(phase_order)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="benchmark JSON capture files")
    parser.add_argument("--json", action="store_true", help="emit machine-readable validation result")
    args = parser.parse_args()

    results = []
    all_errors: list[str] = []
    for path in args.captures:
        try:
            data = load_capture(path)
            result = validate_capture(data, path)
            results.append({"path": str(path), "valid": True, **result})
        except BenchmarkSchemaError as exc:
            messages = str(exc).splitlines()
            all_errors.extend(messages)
            results.append({"path": str(path), "valid": False, "errors": messages})

    if args.json:
        print(json.dumps({"valid": not all_errors, "captures": results}, indent=2, sort_keys=True))
    else:
        for item in results:
            if item["valid"]:
                print(f"{item['path']}: valid schema v{item['schema_version']}")
            else:
                for message in item["errors"]:
                    print(message)
    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
