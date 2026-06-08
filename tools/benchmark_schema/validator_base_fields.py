from __future__ import annotations

from typing import Any

from .backend_metadata import validate_backend_metadata as validate_backend_metadata_impl
from .contract_metadata import validate_contract_metadata as validate_contract_metadata_impl
from .core_shared import *
from .execution_modes import (
    validate_grouped_dispatch_metadata as validate_grouped_dispatch_metadata_impl,
    validate_hip_graph_replay_metadata as validate_hip_graph_replay_metadata_impl,
    validate_host_api_batch_metadata as validate_host_api_batch_metadata_impl,
)
from .helper_metadata import validate_helper_lane_metadata as validate_helper_lane_metadata_impl

class ValidatorBaseFieldsMixin:
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
        from .output_policies import validate_output_layout_metadata

        validate_output_layout_metadata(self, metadata)

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
            for key in ["device_index", "visible_device_count", "node_gpu_count"]:
                if key in device and not _is_int(device.get(key)):
                    self._error(f"device.{key} must be an integer")
            for key in ["name", "gcn_arch"]:
                if not isinstance(device.get(key), str):
                    self._error(f"device.{key} must be a string")
            for key in ["device_name", "target_arch", "target_cache_key", "target_instance_id"]:
                if key in device and not isinstance(device.get(key), str):
                    self._error(f"device.{key} must be a string")
            if "device_index" in device and device.get("device_index") != device.get("device_id"):
                self._error("device.device_index must match device.device_id")
            if "device_name" in device and device.get("device_name") != device.get("name"):
                self._error("device.device_name must match device.name")
            if "target_arch" in device and device.get("target_arch") != device.get("gcn_arch"):
                self._error("device.target_arch must match device.gcn_arch")
            if self.data.get("backend_selected") in HIP_RESIDENT_BACKENDS:
                if not _has_concrete_gpu_target_id(device.get("gcn_arch")):
                    self._error("HIP backend captures must include non-placeholder device.gcn_arch")
                if device.get("hip_available") != 1:
                    self._error("HIP backend captures must use device.hip_available=1")
                device_id = device.get("device_id")
                if _is_int(device_id) and device_id < 0:
                    self._error("HIP backend captures must use a nonnegative device.device_id")
                for key in ["visible_device_count", "node_gpu_count"]:
                    if key in device and _is_int(device.get(key)) and device.get(key) < 0:
                        self._error(f"HIP backend captures must use a nonnegative device.{key}")
        runtime_environment = self.data.get("runtime_environment")
        if runtime_environment is not None:
            if not isinstance(runtime_environment, dict):
                self._error("runtime_environment must be an object")
            else:
                for key in [
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "GPU_DEVICE_ORDINAL",
                    "ROCM_PATH",
                    "HIP_PATH",
                    "LD_LIBRARY_PATH",
                    "RNS8_MULTI_GPU_MODE",
                    "RNS8_RANK",
                    "RNS8_WORLD_SIZE",
                ]:
                    value = runtime_environment.get(key)
                    if value is not None and not isinstance(value, str):
                        self._error(f"runtime_environment.{key} must be a string or null")
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
        validate_helper_lane_metadata_impl(self)

    def _validate_starfoundry_metadata(self) -> None:
        validate_contract_metadata_impl(self)

    def _validate_grouped_dispatch_metadata(self) -> None:
        validate_grouped_dispatch_metadata_impl(self)

    def _validate_host_api_batch_metadata(self) -> None:
        validate_host_api_batch_metadata_impl(self)

    def _validate_hip_graph_replay_metadata(self) -> None:
        validate_hip_graph_replay_metadata_impl(self)

    def _validate_backend_metadata(self) -> None:
        validate_backend_metadata_impl(self)

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
                "isa_evidence": "ck_cshuffle_int8_matrix_isa_gate_no_int32_global_store_no_divide",
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
                    "isa_evidence": "rocwmma_wrap64_byte_gemm36_matrix_isa_gate_no_divide",
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
                    "isa_evidence": "rocwmma_i8_matrix_isa_gate_no_divide",
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

