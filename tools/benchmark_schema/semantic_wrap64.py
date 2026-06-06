from __future__ import annotations

from typing import Any

from .core_shared import *

def validate_wrap64_contract(self, ctx: dict[str, Any]) -> None:
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
