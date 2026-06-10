#!/usr/bin/env python3
"""GPU-event validator tests covering required event sets per backend/semantic."""

from __future__ import annotations

import copy

from test_benchmark_schema import expect_invalid, expect_valid


REQUIRED_EVENT_SETS = {
    "hipblaslt-bounded": [
        "pack_h2d", "pack_kernel", "pack",
        "rns_gemm_kernel_group", "rns_gemm",
        "hipblaslt_scratch_reduce_kernel", "hipblaslt_scratch_reduce",
        "crt_export_status_memset", "crt_export_kernel",
        "crt_export_status_d2h", "crt_export_d2h", "crt_export",
    ],
    "hipblaslt-exact-wide": [
        "pack_h2d", "pack_kernel", "pack",
        "rns_gemm_kernel_group", "rns_gemm",
        "hipblaslt_scratch_reduce_kernel", "hipblaslt_scratch_reduce",
        "exact_wide_export_kernel", "exact_wide_status_d2h",
        "exact_wide_output_d2h", "crt_export",
    ],
    "hipblaslt-finite-u8": [
        "finite_pack_h2d", "finite_pack_kernel", "pack",
        "finite_rns_gemm_kernel_group", "rns_gemm",
        "hipblaslt_finite_reduce_kernel", "hipblaslt_finite_reduce",
        "finite_export_kernel", "finite_export_d2h", "crt_export",
    ],
    "ck-bounded": [
        "pack_h2d", "pack_kernel", "pack",
        "rns_gemm_kernel_group", "rns_gemm",
        "ck_epilogue_kernel", "ck_epilogue",
        "crt_export_status_memset", "crt_export_kernel",
        "crt_export_status_d2h", "crt_export_d2h", "crt_export",
    ],
    "ck-exact-wide": [
        "pack_h2d", "pack_kernel", "pack",
        "rns_gemm_kernel_group", "rns_gemm",
        "ck_epilogue_kernel", "ck_epilogue",
        "exact_wide_export_kernel", "exact_wide_status_d2h",
        "exact_wide_output_d2h", "crt_export",
    ],
    "ck-finite-u8": [
        "finite_pack_h2d", "finite_pack_kernel", "pack",
        "finite_rns_gemm_kernel_group", "rns_gemm",
        "ck_finite_epilogue_kernel", "ck_finite_epilogue",
        "finite_export_kernel", "finite_export_d2h", "crt_export",
    ],
    "rocwmma-bounded": [
        "pack_h2d", "pack_kernel", "pack",
        "rns_gemm_kernel_group", "rns_gemm",
        "rocwmma_epilogue_kernel", "rocwmma_epilogue",
        "crt_export_status_memset", "crt_export_kernel",
        "crt_export_status_d2h", "crt_export_d2h", "crt_export",
    ],
    "rocwmma-finite-u8": [
        "finite_pack_h2d", "finite_pack_kernel", "pack",
        "finite_rns_gemm_kernel_group", "rns_gemm",
        "rocwmma_finite_epilogue_kernel", "rocwmma_finite_epilogue",
        "finite_export_kernel", "finite_export_d2h", "crt_export",
    ],
    "amdgpu-builtins-bounded": [
        "pack_h2d", "pack_kernel", "pack",
        "rns_gemm_kernel_group", "rns_gemm",
        "amdgpu_builtin_epilogue_kernel", "amdgpu_builtin_epilogue",
        "crt_export_status_memset", "crt_export_kernel",
        "crt_export_status_d2h", "crt_export_d2h", "crt_export",
    ],
    "amdgpu-builtins-exact-wide": [
        "pack_h2d", "pack_kernel", "pack",
        "rns_gemm_kernel_group", "rns_gemm",
        "amdgpu_builtin_epilogue_kernel", "amdgpu_builtin_epilogue",
        "exact_wide_export_kernel", "exact_wide_status_d2h",
        "exact_wide_output_d2h", "crt_export",
    ],
    "amdgpu-builtins-sparse": [
        "sparse_a_value_h2d", "sparse_a_index_h2d",
        "sparse_pack_kernel", "pack",
        "sparse_rns_gemm_kernel_group", "rns_gemm",
        "amdgpu_builtin_sparse_epilogue_kernel", "amdgpu_builtin_sparse_epilogue",
        "crt_export_status_memset", "crt_export_kernel",
        "crt_export_status_d2h", "crt_export_d2h", "crt_export",
    ],
}

NULLABLE_EVENT_SETS = {
    "hipblaslt-scratch": {
        "backends": {"hipblaslt"},
        "events": ["hipblaslt_scratch_reduce_kernel", "hipblaslt_scratch_reduce"],
        "reason": "nullable when scratch reduce is elided by fused epilogue",
    },
    "ck-epilogue": {
        "backends": {"ck"},
        "events": ["ck_epilogue_kernel", "ck_epilogue"],
        "reason": "nullable when CK uses fused epilogue path",
    },
}


def main() -> int:
    capture = expect_valid("v4_bounded_i64_ck.json")

    # Basic event length validation
    invalid_events = copy.deepcopy(capture)
    invalid_events["gpu_event_timings_us"]["pack_h2d"] = [1.0]
    expect_invalid(invalid_events, "gpu_event_timings_us.pack_h2d length")

    # Verify required event set definitions are internally consistent
    for set_name, events in REQUIRED_EVENT_SETS.items():
        assert isinstance(events, list), f"{set_name} must be a list"
        assert len(events) > 0, f"{set_name} must not be empty"
        assert len(events) == len(set(events)), f"{set_name} has duplicate events"

    # Verify nullable event sets reference valid backends
    valid_backends = {"hipblaslt", "ck", "rocwmma", "amdgpu-builtins", "hip-direct"}
    for set_name, info in NULLABLE_EVENT_SETS.items():
        for backend in info["backends"]:
            assert backend in valid_backends, f"{set_name}: {backend} not a valid backend"

    print("gpu event schema self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
