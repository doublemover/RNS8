from __future__ import annotations

from typing import Any

from .core_shared import *

def validate_finite_u8_contract(self, ctx: dict[str, Any]) -> None:
    semantics = ctx['semantics']
    prefix = ctx['prefix']
    packed_layout = ctx['packed_layout']
    schedule = ctx['schedule']
    backend_metadata = ctx['backend_metadata']
    k_value = ctx['k_value']
    bound_mode = ctx['bound_mode']
    residue_chain_length = ctx['residue_chain_length']
    residue_output_mode = ctx['residue_output_mode']
    status_check = ctx['status_check']
    prefix_policy = ctx['prefix_policy']
    metadata = ctx['metadata']
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
    distribution = self.data.get("input_distribution")
    allowed_distributions = {
        "u8_binary_0_1",
        "u8_sparse_90pct_zero_uniform_nonzero",
        "u8_low_hamming_powers_of_two_mod_q",
        "u8_small_centered_minus2_2_mod_q",
        "u8_full_uniform_0_modulus_minus_1",
        "u8_uniform_0_modulus_minus_1",
    }
    if distribution not in allowed_distributions:
        self._error("finite-u8 captures must use a registered finite input_distribution")
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
