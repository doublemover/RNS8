"""GPU event phase helpers for benchmark schema validation."""

from __future__ import annotations

import re

from metadata_registry_constants import CK_DEEP_GPU_EVENT_LABELS, ROCWMMA_DEEP_GPU_EVENT_LABELS


CK_PREFIX_EVENT_RE = re.compile(r"^ck_prefix_(\d{2})_(pack_a|pack_b|matmul|copy_centered|add_centered)$")
ROCWMMA_PREFIX_EVENT_RE = re.compile(
    r"^rocwmma_prefix_(\d{2})_(pack_a|pack_b|matmul|pack_a_prepacked_b|matmul_prepacked_b)$"
)


def prefix_event_label(prefix: str, index: int, suffix: str) -> str:
    return f"{prefix}{index:02d}_{suffix}"


def ck_deep_gpu_event_phases(prefix_count: int, zero_output_tiles: bool) -> list[str]:
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
                prefix_event_label("ck_prefix_", index, "pack_a"),
                prefix_event_label("ck_prefix_", index, "pack_b"),
                prefix_event_label("ck_prefix_", index, "matmul"),
                prefix_event_label("ck_prefix_", index, "copy_centered"),
                prefix_event_label("ck_prefix_", index, "add_centered"),
            ]
        )
    return phases


def rocwmma_deep_gpu_event_phases(
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
                    prefix_event_label("rocwmma_prefix_", index, "pack_a_prepacked_b"),
                    prefix_event_label("rocwmma_prefix_", index, "matmul_prepacked_b"),
                ]
            )
        return phases
    phases = ["rocwmma_pack_a_kernel", "rocwmma_pack_b_kernel", "rocwmma_matmul_kernel"]
    if zero_output_tiles:
        phases.append("rocwmma_zero_output_tile_memset")
    for index in range(prefix_count):
        phases.extend(
            [
                prefix_event_label("rocwmma_prefix_", index, "pack_a"),
                prefix_event_label("rocwmma_prefix_", index, "pack_b"),
                prefix_event_label("rocwmma_prefix_", index, "matmul"),
            ]
        )
    return phases


def vector_gpu_event_phases(semantics: object, selected_kernel: object) -> list[str]:
    if selected_kernel == "hip_vector_alu_i64_gemv_n1_exact_192b_v1":
        kernel = "vector_alu_i64_gemv_n1_kernel"
    elif selected_kernel == "hip_vector_alu_u64_gemv_n1_exact_192b_v1":
        kernel = "vector_alu_u64_gemv_n1_kernel"
    elif semantics == "bounded_i64" or selected_kernel == "hip_vector_alu_i64_exact_192b_v1":
        kernel = "vector_alu_i64_kernel"
    else:
        kernel = "vector_alu_u64_kernel"
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


def is_deep_accelerator_gpu_event_label(phase: str) -> bool:
    return (
        phase in CK_DEEP_GPU_EVENT_LABELS
        or phase in ROCWMMA_DEEP_GPU_EVENT_LABELS
        or CK_PREFIX_EVENT_RE.match(phase) is not None
        or ROCWMMA_PREFIX_EVENT_RE.match(phase) is not None
    )
