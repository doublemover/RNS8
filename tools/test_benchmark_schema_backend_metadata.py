#!/usr/bin/env python3
"""Focused schema self-tests for backend metadata validators."""

from __future__ import annotations

import copy

from benchmark_schema import validate_capture
from test_benchmark_schema import expect_invalid, expect_valid


def main() -> int:
    ck = expect_valid("v4_bounded_i64_ck.json")
    hipblaslt = expect_valid("v4_bounded_i64_hipblaslt.json")
    vector = expect_valid("v4_bounded_i64_vector_alu.json")

    validate_capture(ck)

    bad_source = copy.deepcopy(ck)
    bad_source["backend_metadata"]["source"] = "stale_manual_source"
    expect_invalid(bad_source, "backend_metadata.source must be rns8_get_plan_backend_info")

    bad_kernel_match = copy.deepcopy(ck)
    bad_kernel_match["backend_metadata"]["selected_kernel"] = "ck_unknown_kernel"
    expect_invalid(bad_kernel_match, "selected_kernel must match backend_metadata.selected_kernel")

    bad_workspace_bytes = copy.deepcopy(ck)
    bad_workspace_bytes["backend_metadata"]["workspace_required_bytes"] = -1
    expect_invalid(bad_workspace_bytes, "backend_metadata.workspace_required_bytes must be a nonnegative integer")

    validate_capture(hipblaslt)

    bad_hipblaslt_epilogue = copy.deepcopy(hipblaslt)
    bad_hipblaslt_epilogue["backend_metadata"]["epilogue_mode"] = "fused_centered_residue_then_crt_export"
    expect_invalid(
        bad_hipblaslt_epilogue,
        "hipBLASLt captures must report a separate INT32 scratch reduction epilogue",
    )

    validate_capture(vector)

    bad_vector_source = copy.deepcopy(vector)
    bad_vector_source["backend_metadata"]["source"] = "rns8_get_plan_backend_info"
    expect_invalid(bad_vector_source, "backend_metadata.source must be rns8_bench_vector_alu_baseline")

    bad_vector_capability = copy.deepcopy(vector)
    bad_vector_capability["backend_metadata"]["capability_status"] = "implemented_native_bounded_vector_backend"
    expect_invalid(
        bad_vector_capability,
        "hip-vector-alu-int64 captures must use backend_metadata.capability_status=benchmark_only_vector_alu_baseline",
    )

    print("benchmark schema backend metadata self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
