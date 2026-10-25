"""Shared host measurement admission; descriptive statistics are not evidence."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from measurement_contract import exact_proof_policy, host_batch_contract, host_instrumentation, output_completion_boundary, host_observation_contract


def phase_blockers(capture: dict, phase: str = "end_to_end") -> list[str]:
    timing = capture.get("timing_metadata") or {}
    if timing.get("host_timing_contract_version") != 2:
        return ["legacy_host_timing_unqualified"]
    blockers = []
    observation=host_observation_contract(capture)
    if observation["unit"]=="legacy_single_pipeline_batch_repeated_quotient":
        blockers.append("legacy_repeated_batch_quotients_not_repeat_observations")
    if timing.get("host_result_kind") not in {"single_operation", "amortized_batch", "pipeline_batch_throughput"}:
        blockers.append("host_result_kind_unqualified")
    if timing.get("host_result_kind")=="pipeline_batch_throughput" and observation["unit"]!="completed_pipeline_batch_mean_per_call_v2":
        blockers.append("pipeline_batch_observation_contract_missing")
    if (timing.get("phase_states") or {}).get(phase) != "measured":
        blockers.append(f"{phase}_timing_not_measured")
    threshold = timing.get("censor_threshold_us")
    samples = (capture.get("raw_timings_us") or {}).get(phase)
    if phase=="end_to_end" and observation["unit"]=="completed_pipeline_batch_mean_per_call_v2":
        samples=[b.get("duration_us") for b in (capture.get("streaming_overlap") or {}).get("batches",[]) if isinstance(b,dict)]
    if (type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0
            or not isinstance(samples, list) or not samples):
        blockers.append(f"{phase}_resolution_evidence_missing")
    elif any(type(x) not in (int, float) or not math.isfinite(x) or x <= threshold for x in samples):
        blockers.append(f"{phase}_samples_below_resolution")
    return blockers


def pair_blockers(baseline: dict, candidate: dict, phase: str = "end_to_end") -> list[str]:
    blockers = [f"baseline:{x}" for x in phase_blockers(baseline, phase)]
    blockers += [f"candidate:{x}" for x in phase_blockers(candidate, phase)]
    if ((baseline.get("timing_metadata") or {}).get("host_result_kind") !=
            (candidate.get("timing_metadata") or {}).get("host_result_kind")):
        blockers.append("host_result_kind_mismatch")
    if host_observation_contract(baseline) != host_observation_contract(candidate):
        blockers.append("host_observation_contract_mismatch")
    if host_instrumentation(baseline) != host_instrumentation(candidate):
        blockers.append("host_instrumentation_state_mismatch")
    if output_completion_boundary(baseline) != output_completion_boundary(candidate):
        blockers.append("output_completion_boundary_mismatch")
    if exact_proof_policy(baseline) != exact_proof_policy(candidate):
        blockers.append("exact_proof_policy_mismatch")
    if host_batch_contract(baseline) != host_batch_contract(candidate):
        blockers.append("host_batch_workload_mismatch")
    return blockers


def capture_identity(capture: dict) -> str:
    # Review loaders add private path annotations; they are not artifact data.
    data = {k: v for k, v in capture.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def comparison_record(baseline: dict, candidate: dict, baseline_path: str | Path,
                      phase: str = "end_to_end") -> dict:
    blockers = pair_blockers(baseline, candidate, phase)
    return {"version": 1, "phase": phase, "baseline_capture": str(Path(baseline_path).resolve()),
            "baseline_sha256": capture_identity(baseline), "candidate_sha256": capture_identity(candidate),
            "eligible": not blockers, "blockers": blockers}


def bound_comparison_blockers(record: dict | None, candidate: dict) -> list[str]:
    if not isinstance(record, dict) or record.get("version") != 1 or record.get("phase") != "end_to_end":
        return ["timing_comparison_binding_missing"]
    if record.get("candidate_sha256") != capture_identity(candidate):
        return ["timing_comparison_candidate_mismatch"]
    path = record.get("baseline_capture")
    if not isinstance(path, str) or not path:
        return ["timing_comparison_baseline_missing"]
    from benchmark_schema import load_capture, validate_capture
    try:
        baseline = load_capture(Path(path))
        validate_capture(baseline, Path(path))
        if record.get("baseline_sha256") != capture_identity(baseline):
            return ["timing_comparison_baseline_mismatch"]
    except (OSError, ValueError, TypeError):
        return ["timing_comparison_baseline_unreadable"]
    # Recompute from the bound captures, never trust the report's eligible bit.
    return pair_blockers(baseline, candidate)
