"""Order-invariant process observations; report views never rewrite raw captures."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import statistics
from typing import Any, Iterable

from measurement_contract import exact_proof_policy, host_batch_contract, host_instrumentation, host_observation_contract, host_observation_count
from perf_variance_report import execution_cohort, execution_id, DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD, DEFAULT_MAX_WITHIN_CAPTURE_REL_SPREAD, DEFAULT_MIN_REPEATS
from result_compare import CONTRACT_KEYS, contract_value
from timing_eligibility import capture_identity, phase_blockers


def canonical_work(capture: dict, *, exclude: Iterable[str] = ()) -> str:
    excluded = set(exclude)
    contract = {key: contract_value(capture, key) for key in CONTRACT_KEYS if key not in excluded}
    # These describe the caller-visible boundary, not physical kernel tiling.
    contract["layout"] = capture.get("layout")
    contract["checksum_policy"] = (capture.get("host_api_batch") or {}).get("checksum_policy", "single_final_output_checksum")
    return json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)


def comparison_environment(capture: dict, *, cpu_anchor: bool = False) -> str:
    manifest = capture.get("build_manifest") or {}
    identity = {"source_commit": manifest.get("source_commit"), "source_digest": manifest.get("source_digest"),
                "exact_proof_policy": exact_proof_policy(capture),
                "host_instrumentation_state": host_instrumentation(capture),
                "host_observation_contract": host_observation_contract(capture),
                "host_batch_contract": host_batch_contract(capture)}
    if not cpu_anchor:
        identity.update({key: capture.get(key) for key in ("compiler", "hip_toolchain", "configured_amdgpu_targets",
            "device", "cpu_parallel", "clock_policy")})
        timing = capture.get("timing_metadata") or {}
        identity["timing"] = {key: timing.get(key) for key in ("host_timing_contract_version", "host_result_kind",
            "clock_period_numerator", "clock_period_denominator", "inner_repetitions", "amortization_policy")}
        # Backend flags may differ between explicitly compared implementations;
        # all other configuration text, including experimental policies, remains.
        configuration = manifest.get("configuration") or ""
        identity["configuration"] = "\n".join(line for line in configuration.splitlines()
            if not line.startswith(("ck=", "rocwmma=", "hipblaslt=", "builtins=", "dependency_tree.")))
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def phase_median(capture: dict, phase: str = "end_to_end", *, event: bool = False) -> float | None:
    aggregate = capture.get("_observation_aggregate")
    key = "event_medians_us" if event else "phase_medians_us"
    if isinstance(aggregate, dict): return aggregate[key].get(phase)
    summary = capture.get("gpu_event_timing_summary_us" if event else "timing_summary_us") or {}
    return _number((summary.get(phase) or {}).get("median"))


def aggregate_captures(captures: list[dict], *, minimum_runs: int = 2) -> list[dict]:
    """One report-only view per exact implementation/work/environment cohort.

    Distinct execution IDs are required; copied records do not add weight.
    Unknown provenance stays descriptive and cannot produce a qualified winner.
    The original sample arrays are never replaced by invented process samples.
    """
    groups = defaultdict(list)
    for capture in captures:
        if "_observation_aggregate" in capture:
            raise ValueError("process observations must be aggregated exactly once")
        cohort = execution_cohort(capture)
        fallback = comparison_environment(capture)
        timing = capture.get("timing_metadata") or {}
        implementation = {"locality_schedule":capture.get("locality_schedule"), "hipblaslt_metadata_policy": (capture.get("hipblaslt_metadata") or {}).get("policy", "legacy_per_block") if capture.get("backend_selected")=="hipblaslt" else None, "scenario": capture.get("scenario_metadata"), "plan_packing": capture.get("plan_packing"),
            "autotune_key": (capture.get("backend_metadata") or {}).get("autotune_key"),
            "physical_policy": {key: timing.get(key) for key in ("pack_layout", "fusion_mode", "residue_group_width",
                "residue_group_layout", "generated_reducer_identity")}}
        identity = (canonical_work(capture), capture.get("backend_selected"), capture.get("selected_kernel"),
                    capture.get("benchmark_execution_mode"), implementation, cohort or ("unqualified:" + fallback))
        groups[json.dumps(identity, sort_keys=True)].append(capture)
    output = []
    for key, items in sorted(groups.items()):
        items = sorted(items, key=lambda c: (capture_identity(c), str(c.get("_path", ""))))
        unique = {}; conflicts = set(); paths = defaultdict(set)
        for capture in items:
            run = execution_id(capture)
            identity = run or ("unqualified:" + capture_identity(capture))
            paths[identity].add(str(capture.get("_path", "<memory>")))
            if identity in unique and capture_identity(unique[identity]) != capture_identity(capture): conflicts.add(identity)
            unique.setdefault(identity, capture)
        observations = list(unique.values())
        cohort = execution_cohort(items[0])
        independent = len({execution_id(c) for c in observations if execution_id(c)}) if cohort else 0
        blockers = set()
        if not cohort: blockers.add("independent_execution_provenance_unavailable")
        if independent < minimum_runs: blockers.add("insufficient_independent_processes")
        if conflicts: blockers.add("conflicting_reused_execution_id")
        for capture in observations:
            blockers.update(phase_blockers(capture))
            samples = (capture.get("raw_timings_us") or {}).get("end_to_end", [])
            if host_observation_count(capture) < DEFAULT_MIN_REPEATS: blockers.add("insufficient_repeat_samples")
            if type(capture.get("warmups")) is not int or capture["warmups"] < 3:
                blockers.add("insufficient_warmups")
            summary = (capture.get("timing_summary_us") or {}).get("end_to_end") or {}
            median, p95 = _number(summary.get("median")), _number(summary.get("p95"))
            if median is None or median <= 0 or p95 is None:
                blockers.add("missing_within_capture_spread")
            elif max(0.0, (p95 - median) / median) > DEFAULT_MAX_WITHIN_CAPTURE_REL_SPREAD:
                blockers.add("high_within_capture_variance")
        phases = set().union(*((c.get("timing_summary_us") or {}).keys() for c in observations))
        events = set().union(*((c.get("gpu_event_timing_summary_us") or {}).keys() for c in observations))
        medians = {}; spread = {}; event_medians = {}
        for phase in sorted(phases):
            values = [phase_median(c, phase) for c in observations]
            if any(v is None for v in values): medians[phase] = None; continue
            center = statistics.median(values)
            medians[phase] = center
            spread[phase] = {"minimum": min(values), "maximum": max(values),
                "relative_range": (max(values)-min(values))/center if center > 0 else None}
        for phase in sorted(events):
            values = [phase_median(c, phase, event=True) for c in observations]
            event_medians[phase] = statistics.median(values) if all(v is not None for v in values) else None
        if (spread.get("end_to_end") or {}).get("relative_range", math.inf) is None:
            blockers.add("nonpositive_process_median")
        elif (spread.get("end_to_end") or {}).get("relative_range", math.inf) > DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD:
            blockers.add("process_spread_exceeds_ten_percent")
        view = deepcopy(items[0])
        view["_observation_aggregate"] = {"schema": "rns8_process_observations_v1", "cohort": cohort,
            "implementation_key": hashlib.sha256(key.encode()).hexdigest(), "submitted_records": len(items),
            "unique_records": len({capture_identity(c) for c in items}), "independent_runs": independent, "minimum_runs": minimum_runs,
            "maximum_process_relative_range": DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD,
            "conflicting_execution_ids": sorted(conflicts),
            "submitted_observations": [{"execution_id": execution_id(c), "path": c.get("_path"),
                                        "capture_sha256": capture_identity(c)} for c in items],
            "qualified": not blockers, "blockers": sorted(blockers), "phase_medians_us": medians,
            "event_medians_us": event_medians, "process_spread_us": spread,
            "estimator": "median_of_independent_process_medians", "observations": [
                {"execution_id": execution_id(c), "capture_sha256": capture_identity(c),
                 "paths": sorted(paths[execution_id(c) or ("unqualified:"+capture_identity(c))]),
                 "phase_medians_us": {p: phase_median(c,p) for p in sorted(phases)}} for c in observations]}
        output.append(view)
    return output


def evidence(capture: dict | None) -> dict:
    return (capture or {}).get("_observation_aggregate") or {"qualified": False, "blockers": ["unaggregated_observations"]}


def qualified(capture: dict | None) -> bool:
    return evidence(capture).get("qualified") is True


def best_implementation(captures: Iterable[dict], *, candidate: dict | None = None,
                        cpu_anchor: bool = False) -> dict | None:
    valid = [c for c in captures if qualified(c) and phase_median(c) is not None and
             (candidate is None or comparison_environment(c, cpu_anchor=cpu_anchor) ==
              comparison_environment(candidate, cpu_anchor=cpu_anchor))]
    return min(valid, key=lambda c: (phase_median(c), evidence(c)["implementation_key"]), default=None)


def row_identity(capture: dict, *, exclude: Iterable[str] = ()) -> dict:
    work = canonical_work(capture, exclude=exclude)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    return {"work_contract": json.loads(work), "work_contract_key": digest(work),
            "comparison_environment": digest(comparison_environment(capture)),
            "source_identity": digest(comparison_environment(capture, cpu_anchor=True)),
            "measurement_cohort": evidence(capture)}


def partition_rows(rows: list[dict]) -> list[tuple[tuple[str, str, str], list[dict]]]:
    """Share same-source CPU anchors only with identical arithmetic/work contracts."""
    work_groups = defaultdict(list)
    for row in rows: work_groups[(row["work_contract_key"], row["source_identity"])].append(row)
    result = []
    for (work, source), items in sorted(work_groups.items()):
        cpu = [r for r in items if r["backend"] in ("cpu", "cpu-reference", "wrap64-byte-limb")]
        gpu = [r for r in items if r not in cpu]
        environments = sorted({r["comparison_environment"] for r in gpu}) or ["cpu_only"]
        for environment in environments:
            members = cpu + [r for r in gpu if r["comparison_environment"] == environment]
            result.append(((work, source, environment), sorted(members,
                key=lambda r: (r["backend"], r["measurement_cohort"].get("implementation_key", "")))))
    return result


def best_row(rows: list[dict], backend: str | None = None) -> dict | None:
    valid = [r for r in rows if (backend is None or r["backend"] == backend)
             and r.get("measurement_cohort", {}).get("qualified") is True
             and _number(r.get("median_end_to_end_us")) is not None]
    return min(valid, key=lambda r: (r["median_end_to_end_us"],
               r["measurement_cohort"]["implementation_key"]), default=None)


def compound_median(capture: dict, phases: list[str]) -> float | None:
    observations = evidence(capture).get("observations")
    if not observations:
        values = [phase_median(capture, phase) for phase in phases]
        return sum(values) if all(v is not None for v in values) else None
    sums = []
    for observation in observations:
        values = [observation["phase_medians_us"].get(phase) for phase in phases]
        if any(v is None for v in values): return None
        sums.append(sum(values))
    return statistics.median(sums)
