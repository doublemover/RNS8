"""Execution-mode validators for benchmark schema captures."""

from __future__ import annotations

import math
from typing import Any

from metadata_registry_constants import (
    GRAPH_REPLAY_STATUSES,
    GROUPED_DISPATCH_BATCHED_EXACT_WIDE_EXPORT_STRATEGIES,
    GROUPED_STRATEGY_DEVICE_DESCRIPTOR_POLICIES,
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1.0e-5, abs_tol=1.0e-2)


def validate_grouped_dispatch_metadata(self: Any) -> None:
    grouped = self.data.get("grouped_dispatch")
    is_grouped = self._is_grouped_dispatch_capture()
    if grouped is None:
        if is_grouped:
            self._error("benchmark_grouped_dispatch_evidence captures must include grouped_dispatch metadata")
        return
    if not isinstance(grouped, dict):
        return

    task_count = grouped.get("task_count")
    if is_grouped:
        if self.data.get("benchmark") != "rns8_grouped_dispatch_persistent_resident":
            self._error(
                "benchmark_grouped_dispatch_evidence captures must use "
                "benchmark=rns8_grouped_dispatch_persistent_resident"
            )
        if self.data.get("backend_selected") != "hip-direct":
            self._error("benchmark_grouped_dispatch_evidence captures must use backend_selected=hip-direct")
        if grouped.get("requested") is not True:
            self._error("benchmark_grouped_dispatch_evidence captures must set grouped_dispatch.requested=true")
        if not _is_int(task_count) or task_count <= 1:
            self._error("benchmark_grouped_dispatch_evidence captures must use grouped_dispatch.task_count > 1")
        if grouped.get("capture_status") != "executed":
            self._error("benchmark_grouped_dispatch_evidence captures must set grouped_dispatch.capture_status=executed")
        if grouped.get("unsupported_reason") is not None:
            self._error("benchmark_grouped_dispatch_evidence captures must set grouped_dispatch.unsupported_reason=null")
        if grouped.get("promotion_eligible") is not False:
            self._error("benchmark_grouped_dispatch_evidence captures must set grouped_dispatch.promotion_eligible=false")
        strategy = grouped.get("execution_strategy")
        batched_export = grouped.get("batched_export_enabled")
        slab_bytes = grouped.get("device_output_slab_bytes")
        if strategy is not None and strategy == "not_requested":
            self._error("benchmark_grouped_dispatch_evidence captures must declare an executed grouped strategy")
        if batched_export is True:
            if strategy not in GROUPED_DISPATCH_BATCHED_EXACT_WIDE_EXPORT_STRATEGIES:
                self._error("grouped_dispatch batched export requires the batched exact-wide export strategy")
            if self.data.get("semantics") not in {"exact_wide_signed", "exact_wide_unsigned"}:
                self._error("grouped_dispatch batched export is only valid for exact-wide semantics")
            if self.data.get("exact_wide_export_status_check") != "elided_full_width_device_reconstruction":
                self._error("grouped_dispatch batched export requires structurally elided exact-wide status")
            if not _is_int(slab_bytes) or slab_bytes <= 0:
                self._error("grouped_dispatch batched export requires a positive device_output_slab_bytes")
        elif strategy in GROUPED_DISPATCH_BATCHED_EXACT_WIDE_EXPORT_STRATEGIES:
            self._error("grouped_dispatch batched export strategy requires batched_export_enabled=true")
        if self.data.get("semantics") == "wrap_u64_mod_2_64":
            self._error("benchmark_grouped_dispatch_evidence captures must not use wrap_u64_mod_2_64")
        if self.data.get("reuse_packed_inputs") is not False:
            self._error("benchmark_grouped_dispatch_evidence captures must not use packed-input reuse")
        if self.data.get("pack_mode") != "per_repeat_repack":
            self._error("benchmark_grouped_dispatch_evidence captures must use pack_mode=per_repeat_repack")
        if self._residue_chain_length() != 1 or self._residue_output_mode() != "host_export":
            self._error("benchmark_grouped_dispatch_evidence captures must use host_export residue_chain_length=1")
        if self.data.get("bound_mode") != "global":
            self._error("benchmark_grouped_dispatch_evidence captures must use bound_mode=global")
        if self.data.get("bound_source") not in {None, "static_profile"}:
            self._error("benchmark_grouped_dispatch_evidence captures must use bound_source=static_profile")

        batch = self.data.get("host_api_batch")
        if not isinstance(batch, dict):
            self._error("benchmark_grouped_dispatch_evidence captures must include disabled host_api_batch metadata")
        elif batch.get("enabled") is not False or batch.get("batch_size") != 1:
            self._error("benchmark_grouped_dispatch_evidence captures must keep host_api_batch disabled")

        task_descriptor = grouped.get("task_descriptor_contract")
        if not isinstance(task_descriptor, dict):
            self._error("benchmark_grouped_dispatch_evidence captures must include task_descriptor_contract")
        else:
            semantics = self.data.get("semantics")
            if task_descriptor.get("descriptor_layout") != "same_shape_resident_task_triplets_v1":
                self._error("grouped task descriptor must use same_shape_resident_task_triplets_v1")
            if task_descriptor.get("bucket_policy") != "single_same_shape_bucket":
                self._error("grouped task descriptor must use single_same_shape_bucket")
            if task_descriptor.get("bucket_count") != 1:
                self._error("grouped task descriptor bucket_count must be 1")
            if _is_int(task_count) and task_descriptor.get("task_count") != task_count:
                self._error("grouped task descriptor task_count must match grouped_dispatch.task_count")
            if task_descriptor.get("same_shape_required") is not True:
                self._error("grouped task descriptor must require same_shape")
            selected_prefix = self.data.get("selected_prefix", self.data.get("prefix"))
            expected_shape_key = (
                f"m={self.data.get('m')};n={self.data.get('n')};k={self.data.get('k')};"
                f"tile_m={self.data.get('tile_m')};tile_n={self.data.get('tile_n')};"
                f"prefix={selected_prefix}"
            )
            if task_descriptor.get("shape_key") != expected_shape_key:
                self._error("grouped task descriptor shape_key must match capture shape/tile/prefix")
            if task_descriptor.get("semantics") != semantics:
                self._error("grouped task descriptor semantics must match capture semantics")
            expected_output_domain = "native_i64_u64_host"
            if semantics in {"finite_ring_u8", "finite_field_u8"}:
                expected_output_domain = "finite_u8_host"
            elif semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
                expected_output_domain = "exact_wide_limb_host"
            if task_descriptor.get("output_domain") != expected_output_domain:
                self._error("grouped task descriptor output_domain must match capture output contract")
            if task_descriptor.get("source_version_policy") != "per_task_monotonic_source_version_repack":
                self._error("grouped task descriptor source_version_policy must require per-task repack versions")
            if task_descriptor.get("workspace_policy") != "one_workspace_per_task_shared_plan":
                self._error("grouped task descriptor workspace_policy must be one workspace per task")
            if task_descriptor.get("matrix_ownership_policy") != "benchmark_owns_all_task_triplets_until_capture_end":
                self._error("grouped task descriptor matrix_ownership_policy must be benchmark-owned")
            if task_descriptor.get("descriptor_reuse_policy") != "reuse_after_shape_workspace_source_validation":
                self._error("grouped task descriptor descriptor_reuse_policy must require validated reuse")
            if task_descriptor.get("stride_policy") != "matrix_ld_matches_logical_shape_host_output_ld_explicit":
                self._error("grouped task descriptor stride_policy must declare explicit matrix/output strides")
            if (
                task_descriptor.get("output_currentness_policy")
                != "device_residue_current_after_grouped_gemm_host_output_after_export"
            ):
                self._error("grouped task descriptor output_currentness_policy must bind device-current outputs")
            if task_descriptor.get("lifetime_policy") != "task_matrices_and_workspaces_destroyed_after_capture":
                self._error("grouped task descriptor lifetime_policy must describe capture lifetime")
            if task_descriptor.get("checksum_policy") != "combined_per_task_checksum_u64":
                self._error("grouped task descriptor checksum_policy must combine per-task checksums")
            if task_descriptor.get("status_policy") != "fail_fast_per_task_operation_status":
                self._error("grouped task descriptor status_policy must fail fast on per-task operation status")
            expected_device_policy = GROUPED_STRATEGY_DEVICE_DESCRIPTOR_POLICIES.get(
                strategy,
                "host_resident_task_loop",
            )
            if task_descriptor.get("device_descriptor_policy") != expected_device_policy:
                self._error("grouped task descriptor device_descriptor_policy must match execution strategy")

        metadata = self.data.get("timing_metadata")
        if isinstance(metadata, dict):
            if metadata.get("grouped_dispatch_enabled") is not True:
                self._error("timing_metadata.grouped_dispatch_enabled must be true for grouped dispatch captures")
            if _is_int(task_count) and metadata.get("grouped_dispatch_task_count") != task_count:
                self._error(
                    "timing_metadata.grouped_dispatch_task_count must match grouped_dispatch.task_count"
                )
            if strategy is not None and metadata.get("grouped_dispatch_execution_strategy") not in {None, strategy}:
                self._error("timing_metadata.grouped_dispatch_execution_strategy must match grouped_dispatch")
            if batched_export is not None and metadata.get("grouped_dispatch_batched_export_enabled") not in {None, batched_export}:
                self._error("timing_metadata.grouped_dispatch_batched_export_enabled must match grouped_dispatch")
            if _is_int(slab_bytes) and metadata.get("grouped_dispatch_device_output_slab_bytes") not in {None, slab_bytes}:
                self._error("timing_metadata.grouped_dispatch_device_output_slab_bytes must match grouped_dispatch")
            notes = metadata.get("phase_notes")
            if not isinstance(notes, dict):
                self._error("benchmark_grouped_dispatch_evidence captures must include timing_metadata.phase_notes")
            else:
                for phase in ["pack", "rns_gemm", "crt_export", "end_to_end"]:
                    note = notes.get(phase)
                    if not isinstance(note, str) or "aggregate" not in note or "grouped" not in note:
                        self._error(
                            f"benchmark_grouped_dispatch_evidence phase note {phase} "
                            "must describe aggregate grouped timing"
                        )


def validate_host_api_batch_metadata(self: Any) -> None:
    batch = self.data.get("host_api_batch")
    is_batch = self._is_host_api_batch_capture()
    if batch is None:
        if is_batch:
            self._error("benchmark_host_api_batch captures must include host_api_batch metadata")
        return
    if not isinstance(batch, dict):
        self._error("host_api_batch must be an object")
        return

    enabled = batch.get("enabled")
    if not isinstance(enabled, bool):
        self._error("host_api_batch.enabled must be a boolean")
        return
    if enabled != is_batch:
        expected = "true" if is_batch else "false"
        self._error(f"host_api_batch.enabled must be {expected} for this benchmark_execution_mode")

    batch_size = batch.get("batch_size")
    tasks_per_repeat = batch.get("tasks_per_measured_repeat")
    total_tasks = batch.get("total_measured_tasks")
    repeats = self.data.get("repeats")
    for key, value in [
        ("batch_size", batch_size),
        ("tasks_per_measured_repeat", tasks_per_repeat),
        ("total_measured_tasks", total_tasks),
    ]:
        if not _is_int(value) or value <= 0:
            self._error(f"host_api_batch.{key} must be a positive integer")
    if _is_int(batch_size) and _is_int(tasks_per_repeat) and tasks_per_repeat != batch_size:
        self._error("host_api_batch.tasks_per_measured_repeat must equal host_api_batch.batch_size")
    if _is_int(batch_size) and _is_int(repeats) and _is_int(total_tasks) and total_tasks != batch_size * repeats:
        self._error("host_api_batch.total_measured_tasks must equal batch_size * repeats")

    expected_batch_size = 1 if not is_batch else None
    if expected_batch_size is not None and batch_size != expected_batch_size:
        self._error("non-batch captures must use host_api_batch.batch_size=1")
    if is_batch and _is_int(batch_size) and batch_size <= 1:
        self._error("benchmark_host_api_batch captures must use host_api_batch.batch_size > 1")

    expected_setup = (
        "one_shared_plan_per_capture_one_resident_matrix_workspace_triplet_per_task"
        if is_batch
        else "single_task_default_benchmark_mode"
    )
    expected_timing = (
        "aggregate_batch_totals_per_measured_repeat" if is_batch else "single_call_totals_per_measured_repeat"
    )
    expected_checksum = (
        "fnv1a_over_final_task_output_checksums" if is_batch else "single_final_output_checksum"
    )
    for key, expected in [
        ("setup_scope", expected_setup),
        ("timing_policy", expected_timing),
        ("checksum_policy", expected_checksum),
    ]:
        if batch.get(key) != expected:
            self._error(f"host_api_batch.{key} must be {expected}")

    metadata = self.data.get("timing_metadata")
    if isinstance(metadata, dict):
        if metadata.get("host_api_batch_enabled") is not enabled:
            self._error("timing_metadata.host_api_batch_enabled must match host_api_batch.enabled")
        if metadata.get("host_api_batch_size") != batch_size:
            self._error("timing_metadata.host_api_batch_size must match host_api_batch.batch_size")

    if is_batch:
        if self.data.get("benchmark") != "rns8_host_api_batch_persistent_resident":
            self._error("benchmark_host_api_batch captures must use benchmark=rns8_host_api_batch_persistent_resident")
        if self.data.get("semantics") == "wrap_u64_mod_2_64":
            self._error("benchmark_host_api_batch captures must not use wrap_u64_mod_2_64")
        if self.data.get("reuse_packed_inputs") is not False:
            self._error("benchmark_host_api_batch captures must not use packed-input reuse")
        if self.data.get("pack_mode") != "per_repeat_repack":
            self._error("benchmark_host_api_batch captures must use pack_mode=per_repeat_repack")
        if self._residue_chain_length() != 1 or self._residue_output_mode() != "host_export":
            self._error("benchmark_host_api_batch captures must use host_export residue_chain_length=1")
        if self.data.get("bound_mode") != "global":
            self._error("benchmark_host_api_batch captures must use bound_mode=global")
        if self.data.get("bound_source") not in {None, "static_profile"}:
            self._error("benchmark_host_api_batch captures must use bound_source=static_profile")
        metadata = self.data.get("timing_metadata")
        if isinstance(metadata, dict):
            notes = metadata.get("phase_notes")
            if not isinstance(notes, dict):
                self._error("benchmark_host_api_batch captures must include timing_metadata.phase_notes")
            else:
                for phase in ["pack", "rns_gemm", "crt_export", "end_to_end"]:
                    note = notes.get(phase)
                    if not isinstance(note, str) or "aggregate" not in note:
                        self._error(f"benchmark_host_api_batch phase note {phase} must describe aggregate batch timing")

    for field, source in [
        ("avg_pack_per_task_us", "avg_pack_us"),
        ("avg_rns_gemm_per_task_us", "avg_rns_gemm_us"),
        ("avg_crt_export_per_task_us", "avg_crt_export_us"),
        ("avg_end_to_end_per_task_us", "avg_end_to_end_us"),
    ]:
        value = self.data.get(field)
        source_value = self.data.get(source)
        if value is None:
            continue
        if not _is_number(value):
            self._error(f"{field} must be a finite number")
            continue
        denominator = batch_size
        denominator_name = "host_api_batch.batch_size"
        grouped = self.data.get("grouped_dispatch")
        if (
            self._is_grouped_dispatch_capture()
            and isinstance(grouped, dict)
            and _is_int(grouped.get("task_count"))
        ):
            denominator = grouped.get("task_count")
            denominator_name = "grouped_dispatch.task_count"
        if _is_number(source_value) and _is_int(denominator):
            expected = float(source_value) / float(denominator)
            if not _close(float(value), expected):
                self._error(f"{field} must equal {source} / {denominator_name}")


def validate_hip_graph_replay_metadata(self: Any) -> None:
    graph = self.data.get("hip_graph_replay")
    is_graph = self._is_hip_graph_replay_capture()
    if graph is None:
        if is_graph:
            self._error("hip_graph_replay_resident_rns_chain captures must include hip_graph_replay metadata")
        return
    if not isinstance(graph, dict):
        self._error("hip_graph_replay must be an object")
        return

    for key in ["requested", "available", "used"]:
        if not isinstance(graph.get(key), bool):
            self._error(f"hip_graph_replay.{key} must be a boolean")
    for key in ["status", "scope", "timing_policy", "setup_policy", "final_export_policy"]:
        if not isinstance(graph.get(key), str):
            self._error(f"hip_graph_replay.{key} must be a string")
    for key in [
        "capture_us",
        "instantiate_us",
        "graph_launches_per_measured_repeat",
        "total_graph_launches",
        "captured_chain_length",
    ]:
        if not _is_int(graph.get(key)) or graph.get(key) < 0:
            self._error(f"hip_graph_replay.{key} must be a nonnegative integer")
    caveat = graph.get("caveat")
    if caveat is not None and not isinstance(caveat, str):
        self._error("hip_graph_replay.caveat must be a string or null")

    metadata = self.data.get("timing_metadata")
    repeats = self.data.get("repeats")
    warmups = self.data.get("warmups")
    chain_length = self._residue_chain_length()

    if not is_graph:
        for key in ["requested", "available", "used"]:
            if graph.get(key) is not False:
                self._error(f"non-graph captures must set hip_graph_replay.{key}=false")
        if graph.get("status") != "not_requested":
            self._error("non-graph captures must set hip_graph_replay.status=not_requested")
        if graph.get("scope") != "not_applicable":
            self._error("non-graph captures must set hip_graph_replay.scope=not_applicable")
        for key in [
            "capture_us",
            "instantiate_us",
            "graph_launches_per_measured_repeat",
            "total_graph_launches",
            "captured_chain_length",
        ]:
            if graph.get(key) != 0:
                self._error(f"non-graph captures must set hip_graph_replay.{key}=0")
        for key in ["timing_policy", "setup_policy", "final_export_policy"]:
            if graph.get(key) != "not_applicable":
                self._error(f"non-graph captures must set hip_graph_replay.{key}=not_applicable")
        if isinstance(metadata, dict):
            enabled = metadata.get("hip_graph_replay_enabled")
            if enabled is not None and enabled is not False:
                self._error("non-graph captures must set timing_metadata.hip_graph_replay_enabled=false")
        return

    if self.data.get("benchmark") != "rns8_hip_graph_replay_resident_rns_chain":
        self._error("hip_graph_replay captures must use benchmark=rns8_hip_graph_replay_resident_rns_chain")
    if self.data.get("backend_selected") != "hip-direct" or self.data.get("backend_requested") != "hip-direct":
        self._error("hip_graph_replay captures must request and select backend hip-direct")
    if self.data.get("semantics") not in {"bounded_i64", "bounded_u64", "exact_wide_signed", "exact_wide_unsigned"}:
        self._error("hip_graph_replay captures must use bounded or exact-wide RNS semantics")
    if self.data.get("reuse_packed_inputs") is not True:
        self._error("hip_graph_replay captures must use reuse_packed_inputs=true")
    if self.data.get("pack_mode") != "prepacked_reuse":
        self._error("hip_graph_replay captures must use pack_mode=prepacked_reuse")
    if self.data.get("prepack_reuse_operands") != ["A", "B"]:
        self._error("hip_graph_replay captures must reuse operands A and B")
    if self.data.get("prepack_reuse_strategy") != "persistent_matrix_residency":
        self._error("hip_graph_replay captures must use prepack_reuse_strategy=persistent_matrix_residency")
    if chain_length <= 1 or self._residue_output_mode() != "residue_current_rns":
        self._error("hip_graph_replay captures must use residue-current chain output")
    if self.data.get("bound_mode") != "global":
        self._error("hip_graph_replay captures must use bound_mode=global")
    if self.data.get("bound_source") not in {None, "static_profile"}:
        self._error("hip_graph_replay captures must use static-profile bounds")

    for key in ["requested", "available", "used"]:
        if graph.get(key) is not True:
            self._error(f"hip_graph_replay captures must set hip_graph_replay.{key}=true")
    if graph.get("status") != "available":
        self._error("hip_graph_replay captures must set hip_graph_replay.status=available")
    if graph.get("scope") != "direct_hip_reused_inputs_residue_current_rns_chain":
        self._error(
            "hip_graph_replay captures must set hip_graph_replay.scope="
            "direct_hip_reused_inputs_residue_current_rns_chain"
        )
    if graph.get("graph_launches_per_measured_repeat") != 1:
        self._error("hip_graph_replay captures must launch one graph per measured repeat")
    if graph.get("captured_chain_length") != chain_length:
        self._error("hip_graph_replay.captured_chain_length must match residue_chain_length")
    if _is_int(repeats) and _is_int(warmups) and graph.get("total_graph_launches") != repeats + warmups:
        self._error("hip_graph_replay.total_graph_launches must equal warmups + repeats")
    if graph.get("timing_policy") != "raw_timings_us.rns_gemm_and_end_to_end_measure_one_hipGraphLaunch_plus_stream_sync":
        self._error("hip_graph_replay.timing_policy is stale or unsupported")
    if graph.get("setup_policy") != "A_B_prepack_before_capture_capture_and_instantiate_before_warmups":
        self._error("hip_graph_replay.setup_policy is stale or unsupported")
    if graph.get("final_export_policy") != "one_final_logical_export_after_measured_repeats_for_checksum_only":
        self._error("hip_graph_replay.final_export_policy is stale or unsupported")
    if not isinstance(caveat, str) or "resident Direct-HIP RNS GEMM launches only" not in caveat:
        self._error("hip_graph_replay.caveat must describe graph replay scope")

    if not isinstance(metadata, dict):
        return
    if metadata.get("hip_graph_replay_enabled") is not True:
        self._error("hip_graph_replay captures must set timing_metadata.hip_graph_replay_enabled=true")
    if metadata.get("hip_graph_replay_status") != graph.get("status"):
        self._error("timing_metadata.hip_graph_replay_status must match hip_graph_replay.status")
    if metadata.get("hip_graph_replay_scope") != graph.get("scope"):
        self._error("timing_metadata.hip_graph_replay_scope must match hip_graph_replay.scope")
    if metadata.get("hip_graph_capture_us") != graph.get("capture_us"):
        self._error("timing_metadata.hip_graph_capture_us must match hip_graph_replay.capture_us")
    if metadata.get("hip_graph_instantiate_us") != graph.get("instantiate_us"):
        self._error("timing_metadata.hip_graph_instantiate_us must match hip_graph_replay.instantiate_us")
    if metadata.get("hip_graph_total_launches") != graph.get("total_graph_launches"):
        self._error("timing_metadata.hip_graph_total_launches must match hip_graph_replay.total_graph_launches")
    if metadata.get("gpu_event_timing") is not False:
        self._error("hip_graph_replay captures must report gpu_event_timing=false")
    if metadata.get("gpu_event_timing_reason") != "hip_graph_replay_wall_clock_only":
        self._error("hip_graph_replay captures must report gpu_event_timing_reason=hip_graph_replay_wall_clock_only")
    if metadata.get("gpu_event_timing_status") != "not_requested_graph_replay":
        self._error("hip_graph_replay captures must report gpu_event_timing_status=not_requested_graph_replay")
