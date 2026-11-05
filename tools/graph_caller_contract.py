"""Explicit caller-host boundaries and disjoint setup for full-path graphs."""
from __future__ import annotations

import math
from typing import Any

SCHEMA = "rns8_graph_caller_transaction_v1"
FULL_SCOPES = frozenset(("direct_hip_bounded_pack_gemm_export", "direct_hip_exact_wide_pack_gemm_export",
                        "direct_hip_finite_u8_pack_gemm_export", "direct_hip_wrap64_pack_gemm_export"))
SETUP_FIELDS = ("additional_preparation_us", "stream_preparation_us")
OBSERVATION_FIELDS = ("total_us", "input_staging_us", "replay_us", "caller_delivery_us")


def full_path_graph(capture: dict[str, Any]) -> bool:
    graph = capture.get("hip_graph_replay")
    return isinstance(graph, dict) and isinstance(graph.get("scope"), str) and graph["scope"] in FULL_SCOPES


def finite_nonnegative(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def caller_transaction_blockers(capture: dict[str, Any]) -> list[str]:
    if not full_path_graph(capture):
        return []
    graph = capture["hip_graph_replay"]
    transaction = graph.get("caller_transaction")
    if not isinstance(transaction, dict) or transaction.get("schema") != SCHEMA:
        return ["graph_caller_transaction_contract_missing"]
    blockers = []
    if (transaction.get("completion_boundary") != "per_task_host_destinations"
            or transaction.get("input_staging") != "every_completed_call"
            or transaction.get("caller_output_delivery") != "every_completed_call"
            or transaction.get("common_setup_excluded") != ["planning_workspace", "matrix_allocation"]):
        blockers.append("graph_caller_transaction_boundary_mismatch")
    detailed = transaction.get("phase_clocks")
    if type(detailed) is not bool:
        blockers.append("graph_caller_transaction_clock_policy_missing")
    timing = capture.get("timing_metadata")
    timing = timing if isinstance(timing, dict) else {}
    if detailed is True and timing.get("host_instrumentation_state") == "hooks_disabled":
        blockers.append("graph_phase_clocks_inside_production_interval")
    if any(not finite_nonnegative(transaction.get(key)) for key in (*SETUP_FIELDS, "teardown_us")):
        blockers.append("graph_caller_transaction_setup_incomplete")
    dimensions = [capture.get(key) for key in ("m", "n", "k")]
    if any(type(value) is not int or value <= 0 for value in dimensions):
        blockers.append("graph_caller_transaction_geometry_missing")
    else:
        m, n, k = dimensions
        finite = capture.get("semantics") in ("finite_ring_u8", "finite_field_u8")
        width = 1 if finite else 8
        limbs = capture.get("exact_wide_limb_count", 4) if capture.get("semantics") in ("exact_wide_signed", "exact_wide_unsigned") else 1
        if type(limbs) is not int or limbs < 1:
            blockers.append("graph_caller_transaction_geometry_missing")
        else:
            expected = (m*k*width, k*n*width, m*n*width*limbs)
            for key, value in zip(("input_a_bytes_per_call", "input_b_bytes_per_call", "output_bytes_per_call"), expected):
                if type(transaction.get(key)) is not int or transaction[key] != value:
                    blockers.append("graph_caller_transaction_payload_mismatch")
    records = transaction.get("observations")
    repeats = capture.get("repeats")
    raw = capture.get("raw_timings_us") or {}
    samples = raw.get("end_to_end") if isinstance(raw, dict) else None
    if (type(repeats) is not int or repeats < 1 or not isinstance(records, list) or len(records) != repeats
            or not isinstance(samples, list) or len(samples) != repeats):
        blockers.append("graph_caller_transaction_observations_missing")
        records, samples = [], []
    for record, sample in zip(records, samples or ()):
        if not finite_nonnegative(sample) or not valid_observation(record, detailed) or record.get("total_us") != sample:
            blockers.append("graph_caller_transaction_observation_mismatch")
    first = transaction.get("first_replay")
    if not valid_observation(first, detailed):
        blockers.append("graph_first_caller_replay_missing")
    elif capture.get("warmups") == 0 and records and first != records[0]:
        blockers.append("graph_first_caller_replay_mismatch")
    return sorted(set(blockers))


def valid_observation(record: Any, detailed: bool) -> bool:
    if not isinstance(record, dict) or not finite_nonnegative(record.get("total_us")):
        return False
    phases = [record.get(key) for key in OBSERVATION_FIELDS[1:]]
    if not detailed:
        return all(value is None for value in phases)
    if any(not finite_nonnegative(value) for value in phases):
        return False
    total = sum(phases)
    return finite_nonnegative(total) and math.isclose(total, record["total_us"], rel_tol=1e-9, abs_tol=1e-9)


def graph_setup_us(capture: dict[str, Any]) -> float | None:
    graph = capture.get("hip_graph_replay")
    if not isinstance(graph, dict):
        return None
    values = [graph.get("capture_us"), graph.get("instantiate_us")]
    if full_path_graph(capture):
        if caller_transaction_blockers(capture):
            return None
        values.extend(graph["caller_transaction"][key] for key in SETUP_FIELDS)
    if any(not finite_nonnegative(value) for value in values):
        return None
    total = sum(values)
    return total if finite_nonnegative(total) else None


def graph_phase_diagnostic(capture: dict[str, Any]) -> bool:
    graph = capture.get("hip_graph_replay")
    timing = capture.get("timing_metadata")
    if not isinstance(graph, dict) or not isinstance(timing, dict):
        return False
    transaction = graph.get("caller_transaction") or {}
    return (full_path_graph(capture) and not caller_transaction_blockers(capture)
            and transaction.get("phase_clocks") is True
            and timing.get("host_instrumentation_state") == "diagnostic_gpu_events"
            and timing.get("gpu_event_timing") is False
            and timing.get("gpu_event_timing_status") == "not_requested_graph_replay"
            and timing.get("gpu_event_timing_reason") == "hip_graph_replay_wall_clock_only")
