from __future__ import annotations

from typing import Any

from .core_shared import *

def validate_exact_wide_contract(self, ctx: dict[str, Any]) -> None:
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
