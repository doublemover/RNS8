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
PER_TILE_TIMING_PHASE = "tile_bound_scan"
REPEATED_TIMING_PHASES = {"pack", "rns_gemm", "crt_export", "end_to_end"}
PACK_MODES = {"per_repeat_repack", "prepacked_reuse", "prepacked_reuse_a", "prepacked_reuse_b"}
PREPACK_REUSE_STRATEGIES = {"none", "persistent_matrix_residency", "rocwmma_reusable_b_cache"}
PACK_MODE_OPERANDS = {
    "per_repeat_repack": [],
    "prepacked_reuse": ["A", "B"],
    "prepacked_reuse_a": ["A"],
    "prepacked_reuse_b": ["B"],
}
DIRECT_HIP_GPU_EVENT_SCOPES = {
    "direct_hip_default_stream_backend_operation_groups",
    "direct_hip_bounded_adaptive_default_stream_backend_operation_groups",
    "direct_hip_oneshot_default_stream_operation_groups",
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
    "hip_vector_alu_u64_exact_192b_v1",
}
CK_SELECTED_KERNELS = {
    "ck_wmma_cshuffle_i8_i32_centered_epilogue_v1",
    "ck_wmma_cshuffle_tiled_i8_i32_centered_epilogue_v1",
    "ck_wmma_cshuffle_finite_u8_centered_epilogue_v1",
}
ROCWMMA_SELECTED_KERNELS = {
    "rocwmma_i8_i32_signed_hot_residue_v1",
    "rocwmma_i8_i32_signed_tiled_hot_residue_v1",
    "rocwmma_i8_i32_signed_finite_u8_hot_residue_v1",
}
DIRECT_HIP_FINITE_GENERIC_KERNEL = "direct_hip_tiled_finite_u8_gemm_v1"
DIRECT_HIP_FINITE_SPECIALIZED_KERNELS = {
    251: "direct_hip_tiled_finite_u8_gemm_mod251_v1",
    255: "direct_hip_tiled_finite_u8_gemm_mod255_v1",
    256: "direct_hip_tiled_finite_u8_gemm_mod256_v1",
}
DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE = "rns8_hip_direct_reciprocal_isa_gate"
DIRECT_HIP_BOUNDED_ONESHOT_KERNEL = "direct_hip_prefix9_native_input_grouped_rns_gemm_v1"
DIRECT_HIP_BOUNDED_ONESHOT_EPILOGUE = "native_input_centered_residue_then_crt_export"
DIRECT_HIP_BOUNDED_ONESHOT_WORKSPACE = "transient_native_inputs_to_resident_rns_output"
DIRECT_HIP_FINITE_SPECIALIZED_ISA_EVIDENCE = (
    "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide"
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
}
ROCWMMA_DEEP_GPU_EVENT_LABELS = {
    "rocwmma_pack_a_kernel",
    "rocwmma_pack_b_kernel",
    "rocwmma_matmul_kernel",
    "rocwmma_pack_a_prepacked_b_kernel",
    "rocwmma_matmul_prepacked_b_kernel",
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
    "internal_wrap64_rocwmma_candidate",
}
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
        if self.data.get("backend_selected") == "hip-vector-alu-int64":
            return "benchmark_owned_vector_alu_native_buffers"
        return "persistent_resident_matrices"

    def _is_bounded_oneshot_capture(self) -> bool:
        return (
            self._benchmark_execution_mode() == "public_oneshot_transient_native_inputs"
            or self.data.get("benchmark") == "rns8_bounded_gemm_public_oneshot"
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
            elif self.data.get("benchmark") == "rns8_bounded_gemm_public_oneshot" and execution_mode != (
                "public_oneshot_transient_native_inputs"
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
        self._validate_comparison_baseline()
        self._validate_schedule_metadata()
        self._validate_semantic_contract()
        raw_timings = self._validate_raw_timings()
        self._validate_pack_reuse_fields(raw_timings)
        self._validate_residue_current_timings(raw_timings)
        self._validate_bounded_oneshot_timings(raw_timings)
        self._validate_timing_summaries(raw_timings, "timing_summary_us", self._timing_phases())
        self._validate_top_level_averages(raw_timings)
        self._validate_gpu_events()

    def _validate_nonnegative_ints(self) -> None:
        for key in ["bound", "prefix", "tile_m", "tile_n", "k_block_size", "seed", "warmups", "repeats", "checksum_u64"]:
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

    def _validate_backend_metadata(self) -> None:
        metadata = self._require("backend_metadata", "dict")
        if not isinstance(metadata, dict):
            return
        selected_backend = self.data.get("backend_selected")
        expected_source = (
            "rns8_bench_wrap64_rocwmma_candidate"
            if self._is_wrap64_rocwmma_candidate()
            else "rns8_bench_public_oneshot_api"
            if self._is_bounded_oneshot_capture()
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
                "selected_kernel": "hipblaslt_int8_i32_scratch_reduce_baseline_v1",
                "accelerator_library": "hipBLASLt",
                "capability_status": "implemented_baseline_backend",
                "workspace_mode": "resident_device_buffers_with_hipblaslt_scratch",
                "isa_evidence": "hipblaslt_library_int8_matmul_baseline",
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
    def _validate_semantic_contract(self) -> None:
        semantics = self.data.get("semantics")
        prefix = self.data.get("prefix")
        packed_layout = self.data.get("packed_layout_version")
        schedule = self.data.get("schedule_metadata")
        backend_metadata = self.data.get("backend_metadata")
        bound_mode = self.data.get("bound_mode", "global")
        residue_chain_length = self._residue_chain_length()
        residue_output_mode = self._residue_output_mode()
        status_check = self.data.get("exact_wide_export_status_check")
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
                expected_kernel = "direct_hip_wrap64_byte_gemm36_tiled_2d_v3"
                if self.data.get("selected_kernel") != expected_kernel:
                    self._error(f"v4 direct-HIP wrap64 captures must use selected_kernel={expected_kernel}")
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
                    if self.data.get("selected_kernel") != DIRECT_HIP_BOUNDED_ONESHOT_KERNEL:
                        self._error(
                            "direct-HIP one-shot bounded captures must use "
                            f"selected_kernel={DIRECT_HIP_BOUNDED_ONESHOT_KERNEL}"
                        )
                    if isinstance(backend_metadata, dict):
                        if backend_metadata.get("epilogue_mode") != DIRECT_HIP_BOUNDED_ONESHOT_EPILOGUE:
                            self._error(
                                "direct-HIP one-shot bounded captures must use "
                                f"backend_metadata.epilogue_mode={DIRECT_HIP_BOUNDED_ONESHOT_EPILOGUE}"
                            )
                        if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_ONESHOT_WORKSPACE:
                            self._error(
                                "direct-HIP one-shot bounded captures must use "
                                f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_ONESHOT_WORKSPACE}"
                            )
                        if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                            self._error(
                                "direct-HIP one-shot bounded captures must use "
                                f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                            )
                    metadata = self.data.get("timing_metadata")
                    if isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                        expected_scope = "direct_hip_oneshot_default_stream_operation_groups"
                        if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                            self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
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
                    if schedule.get("min_selected_prefix") != prefix or schedule.get("max_selected_prefix") != prefix:
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
                        (semantics == "exact_wide_signed" and limb_count < 4)
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
                if schedule.get("min_selected_prefix") != prefix or schedule.get("max_selected_prefix") != prefix:
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
                specialized_kernel = DIRECT_HIP_FINITE_SPECIALIZED_KERNELS.get(modulus)
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
                    if self.data.get("selected_kernel") != DIRECT_HIP_FINITE_GENERIC_KERNEL:
                        self._error(
                            "direct-HIP generic finite-u8 captures must use "
                            f"selected_kernel={DIRECT_HIP_FINITE_GENERIC_KERNEL}"
                        )
                    if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                        self._error(
                            "direct-HIP generic finite-u8 captures must use "
                            f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
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
                    expected_scope = "direct_hip_default_stream_backend_operation_groups"
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
                and not self._is_bounded_oneshot_capture()
                and not (semantics in {"bounded_i64", "bounded_u64"} and bound_mode == "per_tile")
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
        if schedule.get("adaptive_execution_applied") is not True:
            self._error("per-tile adaptive captures must set schedule_metadata.adaptive_execution_applied=true")
        selected_kernel = self.data.get("selected_kernel")
        if not isinstance(selected_kernel, str) or not selected_kernel:
            self._error("per-tile adaptive captures must report selected_kernel")
        else:
            selected_backend = self.data.get("backend_selected")
            expected_kernels = {
                "hip-direct": "direct_hip_tiled_rns_gemm_v1",
                "ck": "ck_wmma_cshuffle_tiled_i8_i32_centered_epilogue_v1",
                "rocwmma": "rocwmma_i8_i32_signed_tiled_hot_residue_v1",
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
                "hip-vector-alu-int64": "benchmark_owned_device_buffers",
            }
            expected_workspace = expected_workspaces.get(
                self.data.get("backend_selected"), "resident_device_buffers_with_tiled_schedule"
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
        if not self._is_bounded_oneshot_capture():
            return
        for phase, field in [("pack", "avg_pack_us"), ("crt_export", "avg_crt_export_us"), ("matrix_alloc", "avg_matrix_alloc_us")]:
            values = raw_timings.get(phase)
            if not isinstance(values, list) or any(value != 0.0 for value in values):
                self._error(f"one-shot bounded captures must report raw_timings_us.{phase} as zero-valued")
            average_value = self.data.get(field)
            if _is_number(average_value) and float(average_value) != 0.0:
                self._error(f"one-shot bounded captures must report {field}=0")
        gemm_values = raw_timings.get("rns_gemm")
        e2e_values = raw_timings.get("end_to_end")
        if isinstance(gemm_values, list) and isinstance(e2e_values, list) and gemm_values != e2e_values:
            self._error("one-shot bounded captures must report raw_timings_us.rns_gemm equal to end_to_end")

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
        prefix = self.data.get("prefix")
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

    def _ck_deep_gpu_event_phases(self, prefix_count: int) -> list[str]:
        phases = [
            "ck_pack_a_kernel",
            "ck_pack_b_kernel",
            "ck_wmma_cshuffle_matmul",
            "ck_copy_centered_kernel",
            "ck_add_centered_kernel",
        ]
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

    def _rocwmma_deep_gpu_event_phases(self, prefix_count: int, use_prepacked_b: bool) -> list[str]:
        if use_prepacked_b:
            phases = ["rocwmma_pack_a_prepacked_b_kernel", "rocwmma_matmul_prepacked_b_kernel"]
            for index in range(prefix_count):
                phases.extend(
                    [
                        self._prefix_event_label("rocwmma_prefix_", index, "pack_a_prepacked_b"),
                        self._prefix_event_label("rocwmma_prefix_", index, "matmul_prepacked_b"),
                    ]
                )
            return phases
        phases = ["rocwmma_pack_a_kernel", "rocwmma_pack_b_kernel", "rocwmma_matmul_kernel"]
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
        if backend == "ck":
            phases.extend(self._ck_deep_gpu_event_phases(prefix_count))
        else:
            phases.extend(self._rocwmma_deep_gpu_event_phases(prefix_count, use_prepacked_b))
        phases.append("rns_gemm")
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
        if self._is_bounded_oneshot_capture() and backend == "hip-direct":
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
        if residue_current_chain and enabled is True:
            self._error("residue-current chain captures must not claim GPU event timings")
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
        self._validate_timing_summaries(parsed, "gpu_event_timing_summary_us", phases)

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
