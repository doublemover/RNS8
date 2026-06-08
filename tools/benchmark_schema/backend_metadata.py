"""Backend metadata validators for benchmark schema captures."""

from __future__ import annotations

from typing import Any

from metadata_registry_constants import (
    CURRENT_CORRECTNESS_BACKENDS,
    HIP_RESIDENT_BACKENDS,
    SELECTED_KERNELS,
    VECTOR_ALU_SELECTED_KERNELS,
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _amdgpu_builtin_matrix_family(selected_kernel: str) -> str | None:
    if "_mfma_" in selected_kernel:
        return "mfma"
    if "_smfmac_" in selected_kernel:
        return "smfmac"
    if "_swmmac_" in selected_kernel:
        return "swmmac"
    if "_wmma_" in selected_kernel:
        return "wmma"
    return None


def _amdgpu_builtin_matrix_shape(selected_kernel: str) -> str | None:
    for shape in ("32x32x32", "32x32x16", "16x16x64", "16x16x32", "16x16x16"):
        if shape in selected_kernel:
            return shape
    return None


def _amdgpu_builtin_matrix_dtype(selected_kernel: str) -> str | None:
    for dtype in ("iu4", "iu8", "i8"):
        if f"_{dtype}" in selected_kernel:
            return dtype
    return None


def _amdgpu_builtin_matrix_sparsity(family: str | None) -> str | None:
    if family is None:
        return None
    return "structured_4_2" if family in {"smfmac", "swmmac"} else "dense"


def validate_backend_metadata(self: Any) -> None:
    metadata = self._require("backend_metadata", "dict")
    if not isinstance(metadata, dict):
        return
    selected_backend = self.data.get("backend_selected")
    selected_kernel_for_source = self.data.get("selected_kernel")
    sparse_a_capture = (
        self.data.get("benchmark") == "rns8_finite_u8_explicit_sparse_a_4_to_2_persistent_residue"
        or (isinstance(selected_kernel_for_source, str) and "sparse_a" in selected_kernel_for_source)
    )
    expected_source = (
        "rns8_bench_wrap64_rocwmma_candidate"
        if self._is_wrap64_rocwmma_candidate()
        else "rns8_bench_explicit_sparse_a_4_to_2_path"
        if sparse_a_capture
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
        else "rns8_bench_skinny_gemv_n1_path"
        if self._is_direct_hip_bounded_skinny_gemv_n1_capture()
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
    if selected_backend == "amdgpu-builtins":
        if metadata.get("selected_kernel") not in SELECTED_KERNELS or not str(
            metadata.get("selected_kernel")
        ).startswith("amdgpu_builtin_"):
            self._error("amdgpu-builtins captures must report a registered amdgpu_builtin_* selected_kernel")
        amdgpu_kernel = str(metadata.get("selected_kernel") or "")
        expected_matrix = {
            "matrix_instruction_family": _amdgpu_builtin_matrix_family(amdgpu_kernel),
            "matrix_instruction_shape": _amdgpu_builtin_matrix_shape(amdgpu_kernel),
            "matrix_instruction_dtype": _amdgpu_builtin_matrix_dtype(amdgpu_kernel),
        }
        expected_matrix["matrix_instruction_sparsity"] = _amdgpu_builtin_matrix_sparsity(
            expected_matrix["matrix_instruction_family"]
        )
        for key, value in expected_matrix.items():
            if metadata.get(key) != value:
                self._error(f"amdgpu-builtins captures must use backend_metadata.{key}={value}")
        if metadata.get("accelerator_library") != "AMDGPU builtins":
            self._error("amdgpu-builtins captures must use accelerator_library=AMDGPU builtins")
        if metadata.get("capability_status") != "implemented_opt_in_amdgpu_builtin_backend":
            self._error(
                "amdgpu-builtins captures must use capability_status=implemented_opt_in_amdgpu_builtin_backend"
            )
        if not str(metadata.get("epilogue_mode", "")).startswith("amdgpu_builtin_"):
            self._error("amdgpu-builtins captures must report an amdgpu_builtin_* epilogue")
        expected_workspace = (
            "resident_sparse_a_explicit_4_to_2_contract_dense_b_finite_u8"
            if "sparse_a" in amdgpu_kernel
            else "resident_device_buffers_direct_amdgpu_builtin_matrix_core_no_dense_pack_workspace"
        )
        if metadata.get("workspace_mode") != expected_workspace:
            self._error(
                f"amdgpu-builtins captures must use workspace_mode={expected_workspace}"
            )
        expected_isa = (
            "amdgpu_builtin_sparse_a_matrix_core_isa_gate_no_divide"
            if "sparse_a" in amdgpu_kernel
            else "amdgpu_builtin_matrix_isa_gate_no_divide"
        )
        if metadata.get("isa_evidence") != expected_isa:
            self._error(f"amdgpu-builtins captures must use isa_evidence={expected_isa}")
        bool_expected = {
            "accelerator_backend": True,
            "correctness_backend": True,
            "matrix_engine_backend": True,
            "compiled_kernel_available": True,
            "exact_differential_validated": True,
        }
        for key, value in bool_expected.items():
            if metadata.get(key) is not value:
                self._error(f"amdgpu-builtins captures must use backend_metadata.{key}={value}")
