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
