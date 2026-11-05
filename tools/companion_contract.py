"""Versioned observation projection for immutable diagnostic companions.

Only reviewed observation leaves are excluded. New fields, including duration
controls and unknown nested fields, continue to participate in compatibility.
Raw file/capture hashes bind every original observation separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA = "rns8_companion_common_contract_v2"


@dataclass(frozen=True)
class ObservationSection:
    path: tuple[str, ...]
    fields: frozenset[str]


TOP_LEVEL_OBSERVATIONS = frozenset((
    "profiler_ranges",
    "execution_environment_witness",
    "process_id",
    "plan_us", "global_bound_scan_us", "tile_bound_scan_us", "schedule_query_us", "matrix_alloc_us",
    "prepack_setup_us", "raw_timings_us", "timing_summary_us", "gpu_event_timings_us",
    "gpu_event_timing_summary_us", "avg_planning_us", "avg_global_bound_scan_us", "avg_scheduling_us",
    "avg_tile_bound_scan_us", "avg_matrix_alloc_us", "avg_prepack_setup_us", "avg_pack_us",
    "avg_pack_a_us", "avg_pack_b_us", "avg_pack_per_task_us", "avg_rns_gemm_us",
    "avg_rns_gemm_per_task_us", "avg_per_modulus_gemm_estimate_us", "avg_crt_export_us",
    "avg_crt_export_per_task_us", "avg_end_to_end_us", "avg_end_to_end_per_task_us",
))

OBSERVATION_SECTIONS = (
    ObservationSection(("hip_graph_replay", "caller_transaction"),
        frozenset(("additional_preparation_us", "stream_preparation_us", "teardown_us", "phase_clocks"))),
    ObservationSection(("hip_graph_replay", "caller_transaction", "first_replay"),
        frozenset(("total_us", "input_staging_us", "replay_us", "caller_delivery_us"))),
    ObservationSection(("prepack_setup_breakdown_us",),
                       frozenset(("pack_a", "pack_b", "runtime_cache", "unclassified"))),
    ObservationSection(("streaming_overlap",),
                       frozenset(("first_use_batch_wall_us", "batch_wall_us", "per_repeat_pipeline_us"))),
    ObservationSection(("reuse_contract",), frozenset((
        "setup_cost_us", "setup_amortized_us", "repeat_median_end_to_end_us",
        "setup_inclusive_median_end_to_end_us", "break_even_repeat_count"))),
    ObservationSection(("reuse_contract", "setup_breakdown_us"),
                       frozenset(("pack_a", "pack_b", "runtime_cache", "unclassified"))),
    ObservationSection(("native_reference_verification",), frozenset(("elapsed_us",))),
    ObservationSection(("exact_wide_magnitude",), frozenset(("admission_us", "oracle_us"))),
    ObservationSection(("hip_graph_replay",), frozenset(("capture_us", "instantiate_us"))),
    ObservationSection(("timing_metadata",), frozenset((
        "host_instrumentation_state", "phase_states", "timer_pair_median_us",
        "timer_pair_p95_us", "censor_threshold_us", "physical_accuracy_ns",
        "gpu_event_timing", "gpu_event_timing_status", "gpu_event_timing_reason",
        "gpu_event_timing_source", "gpu_event_timing_source_scope", "gpu_event_phase_order",
        "hip_graph_capture_us", "hip_graph_instantiate_us"))),
)


def remove_observations(identity: dict[str, Any]) -> None:
    """Mutate a detached projection, never either original capture."""
    for section in OBSERVATION_SECTIONS:
        node = identity
        for key in section.path:
            node = node.get(key)
            if not isinstance(node, dict):
                break
        if isinstance(node, dict):
            for key in section.fields:
                node.pop(key, None)
    # Keep call counts, source versions, allocation facts and future fields.
    for section, member, fields in (
        ("streaming_overlap", "batches", ("duration_us", "setup_us")),
        ("sustained_production", "observations", ("duration_us", "per_call_us")),
    ):
        node = identity.get(section)
        records = node.get(member) if isinstance(node, dict) else None
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    for field in fields:
                        record.pop(field, None)
    graph = identity.get("hip_graph_replay")
    transaction = graph.get("caller_transaction") if isinstance(graph, dict) else None
    observations = transaction.get("observations") if isinstance(transaction, dict) else None
    if isinstance(observations, list):
        for record in observations:
            if isinstance(record, dict):
                for field in ("total_us", "input_staging_us", "replay_us", "caller_delivery_us"):
                    record.pop(field, None)
    identity["common_identity_contract"] = SCHEMA
