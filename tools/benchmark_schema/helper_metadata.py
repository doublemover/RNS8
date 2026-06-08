"""Helper-lane, output-policy, and selector metadata validators."""

from __future__ import annotations

from typing import Any

from metadata_registry_constants import (
    FUSION_MODES,
    GROUPED_DISPATCH_EXECUTION_STRATEGIES,
    HIP_RESIDENT_BACKENDS,
    NEXT_OP_HINTS,
    OUTPUT_DESTINATION_LAYOUTS,
    PACK_LAYOUTS,
    RESIDUE_GROUP_LAYOUTS,
    SELECTOR_REJECTION_REASONS,
    STATUS_HANDLING,
    TARGET_NAMESPACES,
)

from .core_shared import GENERATED_REDUCER_RE


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _has_concrete_gpu_target_id(value: Any) -> bool:
    return isinstance(value, str) and value and not value.startswith("unknown") and value != "not_recorded"


def output_destination_layout(padding: Any) -> str:
    return "contiguous_row_major" if padding == 0 else "padded_row_major"


def validate_helper_lane_metadata(self: Any) -> None:
    metadata = self.data.get("timing_metadata")
    if isinstance(metadata, dict):
        pack_layout = metadata.get("pack_layout")
        if pack_layout is not None and pack_layout not in PACK_LAYOUTS:
            self._error(f"timing_metadata.pack_layout must be one of {sorted(PACK_LAYOUTS)}")
        fusion_mode = metadata.get("fusion_mode")
        if fusion_mode is not None and fusion_mode not in FUSION_MODES:
            self._error(f"timing_metadata.fusion_mode must be one of {sorted(FUSION_MODES)}")
        residue_width = metadata.get("residue_group_width")
        if residue_width is not None and (not _is_int(residue_width) or residue_width <= 0):
            self._error("timing_metadata.residue_group_width must be a positive integer")
        residue_layout = metadata.get("residue_group_layout")
        if residue_layout is not None and residue_layout not in RESIDUE_GROUP_LAYOUTS:
            self._error(f"timing_metadata.residue_group_layout must be one of {sorted(RESIDUE_GROUP_LAYOUTS)}")
        reducer = metadata.get("generated_reducer_identity")
        if reducer is not None:
            if not isinstance(reducer, str) or GENERATED_REDUCER_RE.match(reducer) is None:
                self._error("timing_metadata.generated_reducer_identity must be a declared reducer identity")
            if reducer == "generic" or reducer == "direct_hip_generic_reducer":
                self._error("timing_metadata.generated_reducer_identity must not use stale generic identities")
        if fusion_mode == "residue_channel_width3_experimental_benchmark_only":
            if self._benchmark_execution_mode() != "residue_channel_fusion_native_inputs":
                self._error("residue-channel fusion metadata requires benchmark_execution_mode=residue_channel_fusion_native_inputs")
            if pack_layout != "native_i8_row_major_residue_channel_width3":
                self._error("residue-channel fusion captures must use pack_layout=native_i8_row_major_residue_channel_width3")
            if residue_width != 3:
                self._error("residue-channel fusion captures must use residue_group_width=3")

        batch_enabled = metadata.get("host_api_batch_enabled")
        batch_size = metadata.get("host_api_batch_size")
        if batch_enabled is not None and not isinstance(batch_enabled, bool):
            self._error("timing_metadata.host_api_batch_enabled must be a boolean")
        if batch_size is not None and (not _is_int(batch_size) or batch_size <= 0):
            self._error("timing_metadata.host_api_batch_size must be a positive integer")
        grouped_strategy = metadata.get("grouped_dispatch_execution_strategy")
        if grouped_strategy is not None and grouped_strategy not in GROUPED_DISPATCH_EXECUTION_STRATEGIES:
            self._error(
                "timing_metadata.grouped_dispatch_execution_strategy must be a known grouped strategy"
            )
        grouped_batched_export = metadata.get("grouped_dispatch_batched_export_enabled")
        if grouped_batched_export is not None and not isinstance(grouped_batched_export, bool):
            self._error("timing_metadata.grouped_dispatch_batched_export_enabled must be a boolean")
        grouped_slab_bytes = metadata.get("grouped_dispatch_device_output_slab_bytes")
        if grouped_slab_bytes is not None and (not _is_int(grouped_slab_bytes) or grouped_slab_bytes < 0):
            self._error("timing_metadata.grouped_dispatch_device_output_slab_bytes must be a nonnegative integer")

    plan_packing = self.data.get("plan_packing")
    if plan_packing is not None:
        if not isinstance(plan_packing, dict):
            self._error("plan_packing must be an object or null")
        else:
            for key in ["source", "backend", "semantics", "input_domain_name", "output_domain_name", "next_op_hint"]:
                if not isinstance(plan_packing.get(key), str):
                    self._error(f"plan_packing.{key} must be a string")
            for key in [
                "uses_resident_matrix_inputs",
                "uses_transient_pack_workspace",
                "uses_matrix_engine_pack_layout",
                "reusable_prepack_cache_available",
                "production_prepack_cache_available",
                "output_host_current",
                "output_device_current",
                "source_versioned_inputs",
                "same_source_version_pack_elision_available",
            ]:
                if not isinstance(plan_packing.get(key), bool):
                    self._error(f"plan_packing.{key} must be a boolean")
            for key in [
                "flags",
                "next_op_flags",
                "a_pack_workspace_bytes",
                "b_pack_workspace_bytes",
                "accumulator_workspace_bytes",
                "library_workspace_bytes",
                "total_transient_workspace_bytes",
            ]:
                value = plan_packing.get(key)
                if not _is_int(value) or value < 0:
                    self._error(f"plan_packing.{key} must be a nonnegative integer")

    plan_lowering = self.data.get("plan_lowering")
    if plan_lowering is not None:
        if not isinstance(plan_lowering, dict):
            self._error("plan_lowering must be an object or null")
        else:
            for key in [
                "source",
                "operation",
                "semantic_contract",
                "backend_family",
                "input_domain",
                "output_domain",
                "desired_output",
                "schedule_strategy",
                "packing_strategy",
                "reuse_strategy",
                "conversion_strategy",
                "lowering_path",
            ]:
                if not isinstance(plan_lowering.get(key), str):
                    self._error(f"plan_lowering.{key} must be a string")
            for key in [
                "final_export_available",
                "rns_continuation_available",
                "native_continuation_available",
                "native_to_rns_available",
                "reusable_b_prepack_available",
            ]:
                if not isinstance(plan_lowering.get(key), bool):
                    self._error(f"plan_lowering.{key} must be a boolean")

    requested_next_op = self.data.get("requested_next_op")
    if requested_next_op is not None:
        if not isinstance(requested_next_op, dict):
            self._error("requested_next_op must be an object")
        else:
            if requested_next_op.get("requested") not in NEXT_OP_HINTS:
                self._error(f"requested_next_op.requested must be one of {sorted(NEXT_OP_HINTS)}")
            if requested_next_op.get("resolved") not in NEXT_OP_HINTS - {"auto"}:
                self._error(f"requested_next_op.resolved must be one of {sorted(NEXT_OP_HINTS - {'auto'})}")
            if requested_next_op.get("source") not in {"cli", "benchmark_default"}:
                self._error("requested_next_op.source must be cli or benchmark_default")

    output_policy = self.data.get("output_policy")
    if output_policy is not None:
        if not isinstance(output_policy, dict):
            self._error("output_policy must be an object")
        else:
            if output_policy.get("destination_layout") not in OUTPUT_DESTINATION_LAYOUTS:
                self._error(f"output_policy.destination_layout must be one of {sorted(OUTPUT_DESTINATION_LAYOUTS)}")
            if output_policy.get("destination_layout") != output_destination_layout(self.data.get("output_ld_padding", 0)):
                self._error("output_policy.destination_layout must match output_ld_padding")
            for key in ["logical_ld", "ld_padding"]:
                value = output_policy.get(key)
                if not _is_int(value) or value < 0:
                    self._error(f"output_policy.{key} must be a nonnegative integer")
            if _is_int(self.data.get("output_logical_ld")) and output_policy.get("logical_ld") != self.data.get("output_logical_ld"):
                self._error("output_policy.logical_ld must match output_logical_ld")
            if output_policy.get("ld_padding") != self.data.get("output_ld_padding"):
                self._error("output_policy.ld_padding must match output_ld_padding")
            if output_policy.get("status_handling") not in STATUS_HANDLING:
                self._error(f"output_policy.status_handling must be one of {sorted(STATUS_HANDLING)}")
            if not isinstance(output_policy.get("status_event_policy"), str):
                self._error("output_policy.status_event_policy must be a string")

    if self._is_residue_current_chain_capture():
        if not isinstance(requested_next_op, dict):
            self._error("residue-current chain captures must declare requested_next_op")
        elif requested_next_op.get("resolved") != "rns-gemm":
            self._error("residue-current chain captures must declare requested_next_op.resolved=rns-gemm")
        if not isinstance(output_policy, dict):
            self._error("residue-current chain captures must declare output_policy")
        else:
            if output_policy.get("per_repeat_logical_export") is not False:
                self._error("residue-current chain captures must declare output_policy.per_repeat_logical_export=false")
            if output_policy.get("final_checksum_export_after_repeats") is not True:
                self._error(
                    "residue-current chain captures must declare "
                    "output_policy.final_checksum_export_after_repeats=true"
                )
            if output_policy.get("status_handling") != "structurally_elided":
                self._error(
                    "residue-current chain captures must declare "
                    "output_policy.status_handling=structurally_elided"
                )
    if self._is_residue_chain_final_export_capture():
        if not isinstance(requested_next_op, dict):
            self._error("residue-chain final-export captures must declare requested_next_op")
        elif requested_next_op.get("resolved") != "final-export":
            self._error("residue-chain final-export captures must declare requested_next_op.resolved=final-export")
        if not isinstance(output_policy, dict):
            self._error("residue-chain final-export captures must declare output_policy")
        else:
            if output_policy.get("per_repeat_logical_export") is not True:
                self._error(
                    "residue-chain final-export captures must declare "
                    "output_policy.per_repeat_logical_export=true"
                )
            if output_policy.get("final_checksum_export_after_repeats") is not False:
                self._error(
                    "residue-chain final-export captures must declare "
                    "output_policy.final_checksum_export_after_repeats=false"
                )

    target_variant = self.data.get("target_variant")
    helper_lane_surface_present = any(
        key in self.data
        for key in [
            "plan_packing",
            "plan_lowering",
            "requested_next_op",
            "output_policy",
            "auto_selector",
            "device_allocation",
            "reuse_contract",
            "exact_output_contract",
            "export_variant",
            "reconstruction_variant",
            "modulus_set",
            "residue_count_policy",
            "tile_shape_variant",
            "grouped_dispatch",
            "adaptive_grouped_scheduler",
            "hip_graph_replay",
            "resident_lifetime",
            "workspace_arena",
            "streaming_overlap",
            "workload_proxy",
            "release_gate",
            "verification_amortization",
            "error_detection_policy",
            "cpu_small_shape_selector",
            "incremental_result_cache",
        ]
    )
    if (
        target_variant is None
        and helper_lane_surface_present
        and self.data.get("backend_selected") in HIP_RESIDENT_BACKENDS
    ):
        self._error("HIP helper-lane captures must include target_variant")
    if target_variant is not None:
        if not isinstance(target_variant, dict):
            self._error("target_variant must be an object")
        else:
            target_id = target_variant.get("target_id")
            namespace = target_variant.get("target_namespace")
            if not isinstance(target_id, str):
                self._error("target_variant.target_id must be a string")
            if namespace not in TARGET_NAMESPACES:
                self._error(f"target_variant.target_namespace must be one of {sorted(TARGET_NAMESPACES)}")
            if self.data.get("backend_selected") in HIP_RESIDENT_BACKENDS:
                if not _has_concrete_gpu_target_id(target_id):
                    self._error("HIP captures with target_variant must include concrete target_variant.target_id")
                if namespace == "unknown":
                    self._error("HIP captures with target_variant must use a concrete target_namespace")
            for key in ["review_group_key", "configured_amdgpu_targets"]:
                if not isinstance(target_variant.get(key), str):
                    self._error(f"target_variant.{key} must be a string")
            for key in ["target_arch", "target_cache_key", "target_instance_id", "device_name"]:
                if key in target_variant and not isinstance(target_variant.get(key), str):
                    self._error(f"target_variant.{key} must be a string")
            for key in ["device_index", "visible_device_count", "node_gpu_count"]:
                if key in target_variant and not _is_int(target_variant.get(key)):
                    self._error(f"target_variant.{key} must be an integer")

    auto_selector = self.data.get("auto_selector")
    if auto_selector is not None:
        if not isinstance(auto_selector, dict):
            self._error("auto_selector must be an object")
        else:
            for key in ["source", "requested_backend", "selected_backend", "fallback_reason"]:
                if not isinstance(auto_selector.get(key), str):
                    self._error(f"auto_selector.{key} must be a string")
            selected_key = auto_selector.get("selected_key")
            if selected_key is not None and not isinstance(selected_key, str):
                self._error("auto_selector.selected_key must be a string or null")
            vocabulary = auto_selector.get("rejection_reason_vocabulary")
            if not isinstance(vocabulary, list) or set(vocabulary) != SELECTOR_REJECTION_REASONS:
                self._error("auto_selector.rejection_reason_vocabulary must match fixed rejection reasons")
            rejected = auto_selector.get("rejected_candidates")
            if rejected is not None:
                if not isinstance(rejected, list):
                    self._error("auto_selector.rejected_candidates must be an array")
                else:
                    for index, item in enumerate(rejected):
                        if not isinstance(item, dict):
                            self._error(f"auto_selector.rejected_candidates[{index}] must be an object")
                            continue
                        if item.get("reason") not in SELECTOR_REJECTION_REASONS:
                            self._error(
                                f"auto_selector.rejected_candidates[{index}].reason must be a fixed rejection reason"
                            )

    device_allocation = self.data.get("device_allocation")
    if device_allocation is not None:
        if not isinstance(device_allocation, dict):
            self._error("device_allocation must be an object")
        else:
            if not isinstance(device_allocation.get("tracking_available"), bool):
                self._error("device_allocation.tracking_available must be a boolean")
            for key in ["source", "setup_scope", "source_version_inputs"]:
                if not isinstance(device_allocation.get(key), str):
                    self._error(f"device_allocation.{key} must be a string")
            self._validate_counter_snapshot("device_allocation.before", device_allocation.get("before"))
            if device_allocation.get("after_warmups") is not None:
                self._validate_counter_snapshot("device_allocation.after_warmups", device_allocation.get("after_warmups"))
            self._validate_counter_snapshot("device_allocation.after_repeats", device_allocation.get("after_repeats"))
            self._validate_counter_snapshot("device_allocation.setup_delta", device_allocation.get("setup_delta"))
            if device_allocation.get("measured_repeat_delta") is not None:
                self._validate_counter_snapshot(
                    "device_allocation.measured_repeat_delta",
                    device_allocation.get("measured_repeat_delta"),
                )
