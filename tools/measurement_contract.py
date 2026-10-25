"""Normalized workload identities shared by comparisons and execution cohorts."""
from __future__ import annotations
from typing import Any


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
    return {"batch_size": size, "tasks_per_measured_repeat": tasks,
            "inner_repetitions": timing.get("inner_repetitions", size),
            "aggregate_units": "microseconds_per_measured_repeat",
            "per_task_units": "microseconds_per_task"}


def host_instrumentation(capture: dict[str, Any]) -> str:
    timing = capture.get("timing_metadata") or {}
    # Historical hook-enabled captures remain diagnostic, never silently become
    # production observations when this newly explicit field is absent.
    state = timing.get("host_instrumentation_state")
    if state is None:
        return "diagnostic_gpu_events" if timing.get("gpu_event_timing") is True else "hooks_disabled"
    if state not in ("diagnostic_gpu_events", "hooks_disabled"):
        raise ValueError("unknown host instrumentation state")
    if state == "hooks_disabled" and timing.get("gpu_event_timing") is True:
        raise ValueError("hook-disabled host observations cannot contain enabled diagnostic hooks")
    return state


def output_completion_boundary(capture: dict[str, Any]) -> str:
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
    if grouped.get("batched_export_enabled") is True or timing.get("grouped_dispatch_batched_export_enabled") is True:
        return "compact_task_major_host_slab"
    return "per_task_host_destinations"


def streaming_executed(capture: dict[str, Any]) -> bool:
    overlap=capture.get("streaming_overlap") or {}
    return overlap.get("requested") is True and overlap.get("capture_status")=="executed"


def host_observation_contract(capture: dict[str, Any]) -> dict[str, Any]:
    if not streaming_executed(capture): return {"unit":"completed_call_or_task_batch"}
    overlap=capture["streaming_overlap"]
    if overlap.get("observation_contract_version")!=2:
        return {"unit":"legacy_single_pipeline_batch_repeated_quotient","calls":overlap.get("measured_repeat_count")}
    return {"unit":"completed_pipeline_batch_mean_per_call_v2","calls_per_batch":capture.get("repeats"),
            "fill_drain_policy":overlap.get("fill_drain_policy"),"per_call_latency_available":False}


def host_observation_count(capture: dict[str, Any]) -> int:
    import math
    if streaming_executed(capture):
        overlap=capture["streaming_overlap"]
        if overlap.get("observation_contract_version")!=2:
            return int(type(overlap.get("batch_wall_us")) in (int,float) and math.isfinite(overlap["batch_wall_us"]) and overlap["batch_wall_us"]>0)
        batches=overlap.get("batches")
        return len(batches) if isinstance(batches,list) else 0
    values=(capture.get("raw_timings_us") or {}).get("end_to_end")
    if isinstance(values,list):return sum(type(v) in (int,float) and math.isfinite(v) for v in values)
    repeats=capture.get("repeats")
    return repeats if type(repeats) is int and repeats>=0 else 0
