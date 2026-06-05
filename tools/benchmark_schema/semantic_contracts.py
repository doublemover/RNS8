"""Focused validators split out of benchmark_schema.core."""

from __future__ import annotations

from typing import Any

from .core import (
    CK_FINITE_GENERIC_KERNEL,
    CK_FINITE_SPECIALIZED_KERNELS,
    DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_EPILOGUE,
    DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_KERNELS,
    DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_U64_LARGE_COLPAIR_KERNEL,
    DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_WORKSPACE,
    DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_EPILOGUE,
    DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_U64_LARGE_COLPAIR_KERNEL,
    DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_WORKSPACE,
    DIRECT_HIP_BOUNDED_ONESHOT_EPILOGUE,
    DIRECT_HIP_BOUNDED_ONESHOT_KERNEL_LARGE_COLPAIR_V2,
    DIRECT_HIP_BOUNDED_ONESHOT_KERNEL_V1,
    DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_EPILOGUE,
    DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_KERNEL,
    DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_WORKSPACE,
    DIRECT_HIP_BOUNDED_ONESHOT_WORKSPACE,
    DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_EPILOGUE,
    DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_KERNELS,
    DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_WORKSPACE,
    DIRECT_HIP_BOUNDED_UNIFORM_SMALL_NATIVE_A_REUSE_B_EPILOGUE,
    DIRECT_HIP_BOUNDED_UNIFORM_SMALL_NATIVE_A_REUSE_B_KERNELS,
    DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_EPILOGUE,
    DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_KERNELS,
    DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_WORKSPACE,
    DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_EPILOGUE,
    DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_KERNELS,
    DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_WORKSPACE,
    DIRECT_HIP_FINITE_GENERIC_KERNEL,
    DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_EPILOGUE,
    DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_GENERIC_KERNEL,
    DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_SPECIALIZED_KERNELS,
    DIRECT_HIP_FINITE_NATIVE_A_REUSE_B_WORKSPACE,
    DIRECT_HIP_FINITE_ONESHOT_EPILOGUE,
    DIRECT_HIP_FINITE_ONESHOT_GENERIC_KERNEL,
    DIRECT_HIP_FINITE_ONESHOT_SPECIALIZED_KERNELS,
    DIRECT_HIP_FINITE_ONESHOT_WORKSPACE,
    DIRECT_HIP_FINITE_SPECIALIZED_ISA_EVIDENCE,
    DIRECT_HIP_FINITE_SPECIALIZED_KERNELS,
    DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE,
    ROCWMMA_FINITE_GENERIC_KERNEL,
    ROCWMMA_FINITE_SPECIALIZED_KERNELS,
    WRAP64_ROCWMMA_CANDIDATE_KERNEL,
    _is_int,
    _is_number,
    _is_prime_modulus,
    wrap64_hip_allowed_kernels,
)

def validate_semantic_contract(self) -> None:
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
    residue_chain_final_export = self.data.get("residue_chain_final_export")
    residue_chain_independent_final_export = self.data.get("residue_chain_independent_final_export")
    status_check = self.data.get("exact_wide_export_status_check")
    prefix_policy = self.data.get("contract_prefix_policy")
    metadata = self.data.get("timing_metadata")
    if bound_mode not in {"global", "per_tile"}:
        self._error("bound_mode must be global or per_tile")
    if residue_chain_final_export is not None and not isinstance(residue_chain_final_export, bool):
        self._error("residue_chain_final_export must be a boolean")
    if residue_chain_independent_final_export is not None and not isinstance(
        residue_chain_independent_final_export, bool
    ):
        self._error("residue_chain_independent_final_export must be a boolean")
    if status_check is not None and semantics not in {"exact_wide_signed", "exact_wide_unsigned"}:
        self._error("exact_wide_export_status_check must be null outside exact-wide captures")
    rns_chain_semantics = {"bounded_i64", "bounded_u64", "exact_wide_signed", "exact_wide_unsigned"}
    if residue_chain_length > 1 and semantics not in rns_chain_semantics:
        self._error("residue_chain_length > 1 captures must use bounded or exact-wide RNS semantics")
    if residue_output_mode == "residue_current_rns" and residue_chain_length <= 1:
        self._error("residue_output_mode=residue_current_rns requires residue_chain_length > 1")
    if residue_chain_length == 1 and residue_output_mode != "host_export":
        self._error("residue_chain_length=1 captures must use residue_output_mode=host_export")
    if residue_chain_length > 1 and isinstance(residue_chain_final_export, bool):
        expected_final_export = residue_output_mode == "host_export"
        if residue_chain_final_export is not expected_final_export:
            self._error("residue_chain_final_export must match residue_output_mode for chain captures")
    if residue_chain_independent_final_export is True:
        if semantics not in {"bounded_i64", "bounded_u64", "exact_wide_signed", "exact_wide_unsigned"}:
            self._error("independent final-output residue-chain captures require bounded or exact-wide RNS semantics")
        if residue_chain_length <= 1 or residue_output_mode != "host_export":
            self._error("independent final-output residue-chain captures must use host_export residue_chain_length > 1")
        if residue_chain_final_export is not True:
            self._error("independent final-output residue-chain captures must set residue_chain_final_export=true")
        if self._benchmark_execution_mode() != "residue_chain_independent_final_host_export":
            self._error(
                "independent final-output residue-chain captures must use "
                "benchmark_execution_mode=residue_chain_independent_final_host_export"
            )
        if self.data.get("pack_mode") != "per_repeat_repack":
            self._error("independent final-output residue-chain captures must use pack_mode=per_repeat_repack")
        if self.data.get("reuse_packed_inputs") is not False:
            self._error("independent final-output residue-chain captures must not use packed-input reuse")
        if self.data.get("prepack_reuse_operands") not in (None, []):
            self._error("independent final-output residue-chain captures must not declare reused operands")
        if self.data.get("prepack_reuse_strategy") not in {None, "none"}:
            self._error("independent final-output residue-chain captures must use prepack_reuse_strategy=none")
        if isinstance(metadata, dict):
            if metadata.get("residue_chain_independent_final_export") is not True:
                self._error(
                    "independent final-output residue-chain captures must set "
                    "timing_metadata.residue_chain_independent_final_export=true"
                )
            if metadata.get("residue_chain_final_export") is not True:
                self._error(
                    "independent final-output residue-chain captures must set "
                    "timing_metadata.residue_chain_final_export=true"
                )
    elif self._benchmark_execution_mode() == "residue_chain_independent_final_host_export":
        self._error(
            "benchmark_execution_mode=residue_chain_independent_final_host_export requires "
            "residue_chain_independent_final_export=true"
        )
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
            if residue_output_mode == "residue_current_rns":
                expected_epilogue_type = "residue_current_rns_output"
            else:
                expected_epilogue_type = "crt_export"
                if self._benchmark_execution_mode() not in {
                    "residue_chain_final_host_export",
                    "residue_chain_independent_final_host_export",
                }:
                    self._error(
                        "bounded residue-chain final-export captures must use "
                        "benchmark_execution_mode=residue_chain_final_host_export or "
                        "residue_chain_independent_final_host_export"
                    )
            if bound_mode != "global":
                self._error("bounded residue chains must use bound_mode=global")
            if self.data.get("backend_selected") in {"hip-vector-alu-int64"}:
                self._error("bounded residue chains must not select hip-vector-alu-int64")
            if self.data.get("m") != self.data.get("n") or self.data.get("n") != self.data.get("k"):
                self._error("bounded residue chains must use square m=n=k shapes")
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
            if residue_output_mode == "residue_current_rns":
                expected_epilogue_type = "residue_current_rns_output"
            else:
                expected_epilogue_type = (
                    "exact_wide_signed_limb_export"
                    if semantics == "exact_wide_signed"
                    else "exact_wide_unsigned_limb_export"
                )
                if self._benchmark_execution_mode() not in {
                    "residue_chain_final_host_export",
                    "residue_chain_independent_final_host_export",
                }:
                    self._error(
                        "exact-wide residue-chain final-export captures must use "
                        "benchmark_execution_mode=residue_chain_final_host_export or "
                        "residue_chain_independent_final_host_export"
                    )
            if self.data.get("m") != self.data.get("n") or self.data.get("n") != self.data.get("k"):
                self._error("exact-wide residue chains must use square m=n=k shapes")
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
