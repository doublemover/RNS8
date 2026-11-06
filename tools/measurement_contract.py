"""Normalized workload identities shared by comparisons and execution cohorts."""

from __future__ import annotations
from typing import Any
from graph_caller_contract import full_path_graph, SCHEMA as GRAPH_CALLER_SCHEMA


def profile_ranges_instrumented(capture: dict[str, Any]) -> bool:
    ranges = capture.get("profiler_ranges")
    return ranges is not None and (
        not isinstance(ranges, dict)
        or ranges.get("schema") != "rns8_profiler_ranges_v1"
        or ranges.get("provider_status") not in ("disabled_at_build", "disabled_at_runtime")
    )


def exact_proof_policy(capture: dict[str, Any]) -> str:
    # Before the opt-in policy existed, the documented behavior was eager.
    policy = capture.get("exact_proof_policy", "eager_moments_v1")
    if policy not in ("eager_moments_v1", "scalar_first_v1"):
        raise ValueError("unknown exact proof-policy identity")
    return policy


def host_batch_contract(capture: dict[str, Any]) -> dict[str, Any]:
    batch = capture.get("host_api_batch") or {}
    timing = capture.get("timing_metadata") or {}
    size = batch.get("batch_size", 1)
    tasks = batch.get("tasks_per_measured_repeat", size)
    return {
        "batch_size": size,
        "tasks_per_measured_repeat": tasks,
        "inner_repetitions": timing.get("inner_repetitions", size),
        "aggregate_units": "microseconds_per_measured_repeat",
        "per_task_units": "microseconds_per_task",
    }


def task_scaling_contract(capture: dict[str, Any]) -> dict[str, Any]:
    """Admit the registered host API batch's explicit per-task throughput boundary."""
    batch = capture.get("host_api_batch") or {}
    timing = capture.get("timing_metadata") or {}
    enabled = batch.get("enabled") is True
    tasks = batch.get("tasks_per_measured_repeat", 1)
    if type(tasks) is not int or not 1 <= tasks <= 65536:
        raise ValueError("bounded explicit host task count required")
    if enabled:
        if (
            type(batch.get("batch_size")) is not int
            or batch.get("batch_size") != tasks
            or tasks < 2
            or capture.get("benchmark_execution_mode") != "benchmark_host_api_batch"
            or batch.get("setup_scope")
            != "one_shared_plan_per_capture_one_resident_matrix_workspace_triplet_per_task"
            or batch.get("timing_policy") != "aggregate_batch_totals_per_measured_repeat"
            or batch.get("checksum_policy") != "fnv1a_over_final_task_output_checksums"
        ):
            raise ValueError("registered complete host task-batch contract required")
    elif batch or tasks != 1 or capture.get("benchmark_execution_mode") != "persistent_resident":
        raise ValueError("independent baseline must describe one complete task")
    if (
        timing.get("host_result_kind") != ("amortized_batch" if enabled else "single_operation")
        or type(timing.get("inner_repetitions")) is not int
        or timing.get("inner_repetitions") != tasks
        or timing.get("amortization_policy")
        != "explicit_independent_task_batch_with_per_task_output"
        or output_completion_boundary(capture) != "per_task_host_destinations"
        or host_observation_contract(capture) != {"unit": "completed_call_or_task_batch"}
    ):
        raise ValueError("per-task host completion and native amortization evidence required")
    return dict(
        scope="registered_host_task_throughput_scaling_v1",
        tasks=tasks,
        aggregate_units="microseconds_per_completed_task_batch",
        normalized_units="microseconds_per_completed_task",
        checksum_policy=batch.get("checksum_policy", "single_final_output_checksum"),
    )


def host_instrumentation(capture: dict[str, Any]) -> str:
    timing = capture.get("timing_metadata") or {}
    # Historical hook-enabled captures remain diagnostic, never silently become
    # production observations when this newly explicit field is absent.
    state = timing.get("host_instrumentation_state")
    if state is None:
        return (
            "diagnostic_gpu_events" if timing.get("gpu_event_timing") is True else "hooks_disabled"
        )
    if state not in ("diagnostic_gpu_events", "hooks_disabled"):
        raise ValueError("unknown host instrumentation state")
    if state == "hooks_disabled" and timing.get("gpu_event_timing") is True:
        raise ValueError("hook-disabled host observations cannot contain enabled diagnostic hooks")
    return state


def output_completion_boundary(capture: dict[str, Any]) -> str:
    if full_path_graph(capture):
        transaction = capture["hip_graph_replay"].get("caller_transaction")
        if (
            not isinstance(transaction, dict)
            or transaction.get("schema") != GRAPH_CALLER_SCHEMA
            or transaction.get("completion_boundary") != "per_task_host_destinations"
        ):
            return "legacy_prepared_pinned_graph_output"
    policy = capture.get("output_policy") or {}
    value = policy.get("completion_boundary")
    if value is not None:
        if value not in ("compact_task_major_host_slab", "per_task_host_destinations"):
            raise ValueError("unknown output completion boundary")
        return value
    grouped = capture.get("grouped_dispatch") or {}
    timing = capture.get("timing_metadata") or {}
    # Legacy grouped slab captures completed the compact slab during timing,
    # even when their old destination_layout field incorrectly promised padding.
    if (
        grouped.get("batched_export_enabled") is True
        or timing.get("grouped_dispatch_batched_export_enabled") is True
    ):
        return "compact_task_major_host_slab"
    return "per_task_host_destinations"


def streaming_executed(capture: dict[str, Any]) -> bool:
    overlap = capture.get("streaming_overlap") or {}
    return overlap.get("requested") is True and overlap.get("capture_status") == "executed"


def host_observation_contract(capture: dict[str, Any]) -> dict[str, Any]:
    if not streaming_executed(capture):
        return {"unit": "completed_call_or_task_batch"}
    overlap = capture["streaming_overlap"]
    if overlap.get("observation_contract_version") != 2:
        return {
            "unit": "legacy_single_pipeline_batch_repeated_quotient",
            "calls": overlap.get("measured_repeat_count"),
        }
    return {
        "unit": "completed_pipeline_batch_mean_per_call_v2",
        "calls_per_batch": capture.get("repeats"),
        "fill_drain_policy": overlap.get("fill_drain_policy"),
        "per_call_latency_available": False,
    }


def host_observation_count(capture: dict[str, Any]) -> int:
    import math

    if streaming_executed(capture):
        overlap = capture["streaming_overlap"]
        if overlap.get("observation_contract_version") != 2:
            return int(
                type(overlap.get("batch_wall_us")) in (int, float)
                and math.isfinite(overlap["batch_wall_us"])
                and overlap["batch_wall_us"] > 0
            )
        batches = overlap.get("batches")
        return len(batches) if isinstance(batches, list) else 0
    values = (capture.get("raw_timings_us") or {}).get("end_to_end")
    if isinstance(values, list):
        return sum(type(v) in (int, float) and math.isfinite(v) for v in values)
    repeats = capture.get("repeats")
    return repeats if type(repeats) is int and repeats >= 0 else 0
