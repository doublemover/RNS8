from __future__ import annotations

import math
import re
from typing import Any

from metadata_registry_constants import (
    ACCELERATOR_GPU_EVENT_SCOPES,
    BACKEND_REQUESTED_VALUES,
    BACKEND_SELECTED_VALUES,
    BENCHMARK_EXECUTION_MODES,
    BOUNDED_EXPORT_KERNELS,
    BOUNDED_I64_EXPORT_KERNELS,
    BOUNDED_U64_EXPORT_KERNELS,
    BOUND_DISCOVERY_SOURCES,
    BOUND_KINDS,
    BOUND_SOURCES,
    CK_SELECTED_KERNELS,
    COMPARISON_BASELINE_STATUSES,
    CONTRACT_PREFIX_POLICIES,
    DIRECT_HIP_GPU_EVENT_SCOPES,
    EXACT_WIDE_EXPORT_KERNELS,
    EXACT_WIDE_SIGNED_EXPORT_KERNELS,
    EXACT_WIDE_UNSIGNED_EXPORT_KERNELS,
    GENERATED_REDUCER_IDENTITIES,
    GENERATED_REDUCER_IDENTITY_PATTERNS,
    HIPBLASLT_GPU_EVENT_SCOPES,
    HIP_RESIDENT_BACKENDS,
    NON_RNS_PREFIX_SEMANTICS,
    PACK_MODES,
    PLACEHOLDER_GPU_TARGET_IDS,
    PREPACK_REUSE_STRATEGIES,
    ROCWMMA_SELECTED_KERNELS,
    RNS_PREFIX_SEMANTICS,
    VECTOR_ALU_GPU_EVENT_SCOPES,
    VECTOR_ALU_SELECTED_KERNELS,
)

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
REVIEWED_BASELINE_STATUSES = {BASELINE_STATUS_REVIEWED, BASELINE_STATUS_RELEASE_REVIEWED}
TIMING_PHASES = ["planning", "scheduling", "matrix_alloc", "pack", "rns_gemm", "crt_export", "end_to_end"]
GLOBAL_BOUND_TIMING_PHASE = "global_bound_scan"
PER_TILE_TIMING_PHASE = "tile_bound_scan"
REPEATED_TIMING_PHASES = {"pack", "rns_gemm", "crt_export", "end_to_end"}
OPTIONAL_REPEATED_TIMING_PHASES = {"pack_a", "pack_b"}
TILE_SCHEDULE_ZERO_OUTPUT = 0x00000001
TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT = 0x00000002
TILE_SCHEDULE_KNOWN_FLAGS = TILE_SCHEDULE_ZERO_OUTPUT | TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT


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


GENERATED_REDUCER_RE = re.compile(
    r"^(?:"
    + "|".join(
        [re.escape(identity) for identity in sorted(GENERATED_REDUCER_IDENTITIES)]
        + [f"(?:{pattern})" for pattern in sorted(GENERATED_REDUCER_IDENTITY_PATTERNS)]
    )
    + r")$"
)
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
OLD_ACCELERATOR_GPU_EVENT_SCOPE = "accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export"
DEEP_ACCELERATOR_GPU_EVENT_SCOPE = (
    "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
)
VECTOR_ALU_GPU_EVENT_SCOPE = "vector_alu_default_stream_native_int64_operation_groups"
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
CK_FINITE_GENERIC_KERNEL = "ck_wmma_cshuffle_finite_u8_static_modulus_centered_epilogue_v2"
CK_FINITE_SPECIALIZED_KERNELS = {
    251: "ck_wmma_cshuffle_finite_u8_mod251_centered_epilogue_v2",
    255: "ck_wmma_cshuffle_finite_u8_mod255_centered_epilogue_v2",
    256: "ck_wmma_cshuffle_finite_u8_mod256_centered_epilogue_v2",
}
CK_FINITE_STATIC_MODULI = {
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    217,
    223,
    227,
    229,
    233,
    239,
    241,
    243,
    247,
    251,
    253,
    255,
    256,
}
ROCWMMA_FINITE_GENERIC_KERNEL = "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1"
ROCWMMA_FINITE_SPECIALIZED_KERNELS = {
    251: "rocwmma_i8_i32_signed_finite_u8_mod251_hot_residue_v2",
    255: "rocwmma_i8_i32_signed_finite_u8_mod255_hot_residue_v2",
    256: "rocwmma_i8_i32_signed_finite_u8_mod256_hot_residue_v2",
}
ROCWMMA_FINITE_STATIC_MODULI = CK_FINITE_STATIC_MODULI
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
DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_KERNEL = "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3"
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
DIRECT_HIP_BOUNDED_SKINNY_GEMV_N1_KERNELS = {
    "bounded_i64": "direct_hip_prefix9_rns_gemv_n1_i64_v1",
    "bounded_u64": "direct_hip_prefix9_rns_gemv_n1_u64_v1",
}
DIRECT_HIP_BOUNDED_SKINNY_GEMV_SMALL_N_KERNELS = {
    "bounded_i64": "direct_hip_prefix9_rns_gemv_small_n_i64_v1",
    "bounded_u64": "direct_hip_prefix9_rns_gemv_small_n_u64_v1",
}
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
DIRECT_HIP_BOUNDED_SKINNY_GEMV_N1_EPILOGUE = "resident_rns_gemv_n1_centered_residue_then_crt_export"
DIRECT_HIP_BOUNDED_SKINNY_GEMV_SMALL_N_EPILOGUE = "resident_rns_gemv_small_n_centered_residue_then_crt_export"
DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_WORKSPACE = "transient_native_a_resident_rns_b_output"
DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_WORKSPACE = "transient_i8_b_resident_i8_a_rns_output"
DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_WORKSPACE = "transient_i8_a_transient_i8_b_rns_output"
DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_WORKSPACE = (
    "width3_residue_fusion_transient_i8_inputs"
)
DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_WORKSPACE = "transient_native_b_resident_rns_a_output"
DIRECT_HIP_BOUNDED_SKINNY_GEMV_N1_WORKSPACE = "resident_rns_inputs_skinny_n1_output"
DIRECT_HIP_BOUNDED_SKINNY_GEMV_SMALL_N_WORKSPACE = "resident_rns_inputs_skinny_n_le4_output"
DIRECT_HIP_FINITE_ONESHOT_EPILOGUE = "native_u8_centered_residue_then_canonical_u8_export"
DIRECT_HIP_FINITE_ONESHOT_WORKSPACE = "transient_native_u8_inputs_to_resident_finite_output"
DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_EPILOGUE = "native_a_centered_resident_b_residue_then_canonical_u8_export"
DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_WORKSPACE = "transient_native_u8_a_resident_finite_b_output"
DIRECT_HIP_FINITE_SPECIALIZED_ISA_EVIDENCE = (
    "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide"
)
DIRECT_HIP_ADAPTIVE_KERNEL_V2 = "direct_hip_tiled_active_prefix_rns_gemm_v2"
DIRECT_HIP_ADAPTIVE_GROUPED_SCHEDULE_KERNEL_V3 = "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3"
DIRECT_HIP_ADAPTIVE_ZERO_SKIP_KERNEL_V3 = "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3"
DIRECT_HIP_ADAPTIVE_ZERO_ROW_COL_SKIP_KERNEL_V1 = "direct_hip_tiled_active_prefix_zero_row_col_skip_rns_gemm_v1"
DIRECT_HIP_ADAPTIVE_ZERO_TILE_ROW_COL_SKIP_KERNEL_V1 = (
    "direct_hip_tiled_active_prefix_zero_tile_row_col_skip_rns_gemm_v1"
)
WRAP64_ROCWMMA_CANDIDATE_KERNEL = "rocwmma_wrap64_byte_gemm36_candidate_v0"
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



__all__ = [name for name in globals() if not name.startswith("__")]
