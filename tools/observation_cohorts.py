"""Order-invariant process observations; report views never rewrite raw captures."""

from __future__ import annotations
from benchmark_schema.rocwmma_packing import packing_policy
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import statistics
from typing import Any, Iterable

from measurement_contract import (
    exact_proof_policy,
    host_batch_contract,
    host_instrumentation,
    host_observation_contract,
    host_observation_count,
    task_scaling_contract,
)
from perf_variance_report import (
    execution_cohort,
    execution_id,
    DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD,
    DEFAULT_MAX_WITHIN_CAPTURE_REL_SPREAD,
    DEFAULT_MIN_REPEATS,
    DEFAULT_MIN_NOISE_SPEEDUP_MARGIN,
)
from result_compare import CONTRACT_KEYS, contract_value
from timing_eligibility import capture_identity, phase_blockers
from reference_verdict import numerical_observation
from benchmark_request import request_environment_identity
from execution_host_identity import hardware_cohort_identity


def canonical_work(capture: dict, *, exclude: Iterable[str] = ()) -> str:
    excluded = set(exclude)
    contract = {key: contract_value(capture, key) for key in CONTRACT_KEYS if key not in excluded}
    # These describe the caller-visible boundary, not physical kernel tiling.
    contract["layout"] = capture.get("layout")
    if "checksum_policy" not in excluded:
        contract["checksum_policy"] = (capture.get("host_api_batch") or {}).get(
            "checksum_policy", "single_final_output_checksum"
        )
    return json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)


def comparison_environment(
    capture: dict,
    *,
    cpu_anchor: bool = False,
    cpu_gpu_comparison: bool = False,
    task_scaling: bool = False,
) -> str:
    """Separate arithmetic anchors from same-host CPU/GPU timing comparisons.

    An arithmetic anchor need not share a compiler or timer. Timing comparisons
    may omit GPU-only identity, but retain every build and host timing field.
    """
    if cpu_anchor and cpu_gpu_comparison:
        raise ValueError("arithmetic-anchor and CPU/GPU timing scopes are exclusive")
    manifest = capture.get("build_manifest") or {}
    identity = {
        "observed_hardware": hardware_cohort_identity(
            capture, cpu_anchor=cpu_anchor or cpu_gpu_comparison
        ),
        "source_commit": manifest.get("source_commit"),
        "source_digest": manifest.get("source_digest"),
        "environment_identity": request_environment_identity(capture.get("request_identity"))[0],
        "exact_proof_policy": exact_proof_policy(capture),
        "host_instrumentation_state": host_instrumentation(capture),
        "host_observation_contract": host_observation_contract(capture),
        "host_batch_contract": host_batch_contract(capture),
    }
    if task_scaling:
        scaling = task_scaling_contract(capture)
        identity["comparison_scope"] = scaling["scope"]
        identity["host_batch_contract"] = dict(
            aggregate_units=scaling["aggregate_units"], normalized_units=scaling["normalized_units"]
        )
    if not cpu_anchor:
        identity.update(
            {
                key: capture.get(key)
                for key in (
                    "compiler",
                    "hip_toolchain",
                    "configured_amdgpu_targets",
                    "device",
                    "cpu_parallel",
                    "clock_policy",
                )
            }
        )
        timing = capture.get("timing_metadata") or {}
        identity["timing"] = {
            key: timing.get(key)
            for key in (
                "host_timing_contract_version",
                "host_result_kind",
                "clock_period_numerator",
                "clock_period_denominator",
                "inner_repetitions",
                "amortization_policy",
            )
        }
        if task_scaling:
            identity["timing"]["host_result_kind"] = "explicit_per_task_throughput"
            identity["timing"]["inner_repetitions"] = "explicit_task_count_normalization"
        # Backend flags may differ between explicitly compared implementations;
        # all other configuration text, including experimental policies, remains.
        configuration = manifest.get("configuration") or ""
        identity["configuration"] = "\n".join(
            line
            for line in configuration.splitlines()
            if not line.startswith(
                ("ck=", "rocwmma=", "hipblaslt=", "builtins=", "dependency_tree.")
            )
        )
        if cpu_gpu_comparison:
            identity.pop("device", None)
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)


def task_scaling_work(capture: dict) -> str:
    scaling = task_scaling_contract(capture)
    return json.dumps(
        dict(
            scope=scaling["scope"],
            task_work=json.loads(
                canonical_work(capture, exclude=("host_api_batch.work_contract", "checksum_policy"))
            ),
            checksum_comparison="own_complete_task_set_not_independent_single_checksum",
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def prepack_setup_observation(capture: dict) -> float | None:
    """Read the native one-time scalar; never invent repeated setup samples."""
    value = _number(capture.get("prepack_setup_us"))
    average_alias = _number(capture.get("avg_prepack_setup_us"))
    metadata = capture.get("timing_metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("phase_availability"), dict):
        return None
    availability = metadata["phase_availability"].get("prepack_setup")
    if not isinstance(availability, dict):
        return None
    if (
        value is None
        or value < 0
        or average_alias != value
        or availability.get("timed") is not True
        or availability.get("timing_key") != "prepack_setup_us"
        or availability.get("scope") != "one_time_before_warmups"
    ):
        return None
    return value


def phase_median(capture: dict, phase: str = "end_to_end", *, event: bool = False) -> float | None:
    aggregate = capture.get("_observation_aggregate")
    key = "event_medians_us" if event else "phase_medians_us"
    if isinstance(aggregate, dict):
        return aggregate[key].get(phase)
    if not event and phase == "prepack_setup":
        return prepack_setup_observation(capture)
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
        implementation = {
            "incremental_result_cache": {
                field: value
                for field, value in (capture.get("incremental_result_cache") or {}).items()
                if field
                not in (
                    "a_matrix_instance_id",
                    "b_matrix_instance_id",
                    "result_cache_key_fingerprint",
                )
            },
            "grouped_dispatch_strategy": (capture.get("grouped_dispatch") or {}).get(
                "execution_strategy"
            ),
            "locality_schedule": capture.get("locality_schedule"),
            "hipblaslt_metadata_policy": (capture.get("hipblaslt_metadata") or {}).get(
                "policy", "legacy_per_block"
            )
            if capture.get("backend_selected") == "hipblaslt"
            else None,
            "scenario": capture.get("scenario_metadata"),
            "plan_packing": capture.get("plan_packing"),
            "hipblaslt_prepack_policy": (capture.get("hipblaslt_operand_prepack") or {}).get(
                "policy", "both"
            )
            if capture.get("backend_selected") == "hipblaslt"
            else None,
            "rocwmma_packing_policy": packing_policy(capture),
            "autotune_key": (capture.get("backend_metadata") or {}).get("autotune_key"),
            "physical_policy": {
                key: timing.get(key)
                for key in (
                    "pack_layout",
                    "fusion_mode",
                    "residue_group_width",
                    "residue_group_layout",
                    "generated_reducer_identity",
                )
            },
        }
        identity = (
            canonical_work(capture),
            capture.get("backend_selected"),
            capture.get("selected_kernel"),
            capture.get("benchmark_execution_mode"),
            implementation,
            cohort or ("unqualified:" + fallback),
        )
        groups[json.dumps(identity, sort_keys=True)].append(capture)
    output = []
    for key, items in sorted(groups.items()):
        items = sorted(items, key=lambda c: (capture_identity(c), str(c.get("_path", ""))))
        unique = {}
        conflicts = set()
        paths = defaultdict(set)
        for capture in items:
            run = execution_id(capture)
            identity = run or ("unqualified:" + capture_identity(capture))
            paths[identity].add(str(capture.get("_path", "<memory>")))
            if identity in unique and capture_identity(unique[identity]) != capture_identity(
                capture
            ):
                conflicts.add(identity)
            unique.setdefault(identity, capture)
        observations = list(unique.values())
        cohort = execution_cohort(items[0])
        independent = (
            len({execution_id(c) for c in observations if execution_id(c)}) if cohort else 0
        )
        blockers = set()
        if not cohort:
            blockers.add("independent_execution_provenance_unavailable")
        if independent < minimum_runs:
            blockers.add("insufficient_independent_processes")
        if conflicts:
            blockers.add("conflicting_reused_execution_id")
        within_spreads = []
        for capture in observations:
            environment_error = request_environment_identity(capture.get("request_identity"))[1]
            if environment_error:
                blockers.add(environment_error)
            blockers.update(phase_blockers(capture))
            if host_observation_count(capture) < DEFAULT_MIN_REPEATS:
                blockers.add("insufficient_repeat_samples")
            if type(capture.get("warmups")) is not int or capture["warmups"] < 3:
                blockers.add("insufficient_warmups")
            summary = (capture.get("timing_summary_us") or {}).get("end_to_end") or {}
            median, p95 = _number(summary.get("median")), _number(summary.get("p95"))
            if median is None or median <= 0 or p95 is None:
                blockers.add("missing_within_capture_spread")
            elif max(0.0, (p95 - median) / median) > DEFAULT_MAX_WITHIN_CAPTURE_REL_SPREAD:
                blockers.add("high_within_capture_variance")
            if median is not None and median > 0 and p95 is not None:
                within_spreads.append(max(0.0, (p95 - median) / median))
        phases = set().union(*((c.get("timing_summary_us") or {}).keys() for c in observations))
        if any(
            c.get("prepack_setup_us") is not None or c.get("reuse_packed_inputs") is True
            for c in observations
        ):
            phases.add("prepack_setup")
        events = set().union(
            *((c.get("gpu_event_timing_summary_us") or {}).keys() for c in observations)
        )
        medians = {}
        spread = {}
        event_medians = {}
        for phase in sorted(phases):
            values = [phase_median(c, phase) for c in observations]
            if any(v is None for v in values):
                medians[phase] = None
                continue
            center = statistics.median(values)
            medians[phase] = center
            spread[phase] = {
                "minimum": min(values),
                "maximum": max(values),
                "relative_range": (max(values) - min(values)) / center if center > 0 else None,
            }
        for phase in sorted(events):
            values = [phase_median(c, phase, event=True) for c in observations]
            event_medians[phase] = (
                statistics.median(values) if all(v is not None for v in values) else None
            )
        if (spread.get("end_to_end") or {}).get("relative_range", math.inf) is None:
            blockers.add("nonpositive_process_median")
        elif (spread.get("end_to_end") or {}).get(
            "relative_range", math.inf
        ) > DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD:
            blockers.add("process_spread_exceeds_ten_percent")
        view = deepcopy(items[0])
        numerical, numerical_errors = {}, []
        for capture in items:
            try:
                fact = numerical_observation(capture)
                numerical[fact["capture_sha256"]] = fact
            except (KeyError, TypeError, ValueError, OverflowError):
                numerical_errors.append("numerical_capture_identity_invalid")
        view["_observation_aggregate"] = {
            "schema": "rns8_process_observations_v1",
            "cohort": cohort,
            "numerical_observations": list(numerical.values()),
            "numerical_observation_errors": numerical_errors,
            "environment_identity": request_environment_identity(items[0].get("request_identity"))[
                0
            ],
            "implementation_key": hashlib.sha256(key.encode()).hexdigest(),
            "submitted_records": len(items),
            "unique_records": len({capture_identity(c) for c in items}),
            "independent_runs": independent,
            "minimum_runs": minimum_runs,
            "maximum_process_relative_range": DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD,
            "maximum_observed_within_capture_relative_spread": max(within_spreads, default=None),
            "conflicting_execution_ids": sorted(conflicts),
            "submitted_observations": [
                {
                    "execution_id": execution_id(c),
                    "path": c.get("_path"),
                    "capture_sha256": capture_identity(c),
                }
                for c in items
            ],
            "qualified": not blockers,
            "blockers": sorted(blockers),
            "phase_medians_us": medians,
            "event_medians_us": event_medians,
            "process_spread_us": spread,
            "estimator": "median_of_independent_process_medians",
            "observations": [
                {
                    "execution_id": execution_id(c),
                    "capture_sha256": capture_identity(c),
                    "paths": sorted(
                        paths[execution_id(c) or ("unqualified:" + capture_identity(c))]
                    ),
                    "phase_medians_us": {p: phase_median(c, p) for p in sorted(phases)},
                    "measured_repeat_count": c.get("repeats"),
                }
                for c in observations
            ],
        }
        output.append(view)
    return output


def evidence(capture: dict | None) -> dict:
    return (capture or {}).get("_observation_aggregate") or {
        "qualified": False,
        "blockers": ["unaggregated_observations"],
    }


def measurement_identity(capture: dict) -> str:
    """Bind the process estimator and inputs without giving path aliases weight."""
    aggregate = evidence(capture)
    projection = {
        key: aggregate.get(key)
        for key in (
            "schema",
            "cohort",
            "implementation_key",
            "independent_runs",
            "qualified",
            "blockers",
            "phase_medians_us",
            "event_medians_us",
            "process_spread_us",
            "maximum_observed_within_capture_relative_spread",
            "estimator",
        )
    }
    projection["observations"] = sorted(
        (
            {key: value for key, value in observation.items() if key != "paths"}
            for observation in aggregate.get("observations", [])
        ),
        key=lambda observation: (
            str(observation.get("execution_id")),
            str(observation.get("capture_sha256")),
        ),
    )
    return hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def qualified(capture: dict | None) -> bool:
    return evidence(capture).get("qualified") is True


def comparison_noise_margin(baseline: dict | None, candidate: dict | None) -> float | None:
    return cohort_noise_margin(evidence(baseline), evidence(candidate))


def cohort_noise_margin(baseline: dict, candidate: dict) -> float | None:
    noise = [DEFAULT_MIN_NOISE_SPEEDUP_MARGIN]
    for aggregate in (baseline, candidate):
        if aggregate.get("qualified") is not True:
            return None
        for value in (
            aggregate.get("maximum_observed_within_capture_relative_spread"),
            (aggregate.get("process_spread_us", {}).get("end_to_end") or {}).get("relative_range"),
        ):
            number = _number(value)
            if number is None or number < 0:
                return None
            noise.append(number)
    return 1.0 + max(noise)


class ProductionCohortIndex:
    """Check every unique production observation, sharing expensive bindings."""

    def __init__(self, captures: Iterable[dict], diagnostic_bindings: dict | None = None):
        self._captures = defaultdict(dict)
        for capture in captures:
            self._captures[capture_identity(capture)].setdefault(str(capture.get("_path")), capture)
        self._bindings = diagnostic_bindings or {}
        self._checked: dict[str, list[str]] = {}
        self._accepted = {}

    def blockers(self, capture: dict) -> list[str]:
        from diagnostic_companion import binding_for_capture, production_evidence

        aggregate = evidence(capture)
        blockers = list(aggregate.get("blockers", []))
        if not qualified(capture):
            blockers.append("independent_production_cohort_unqualified")
        observations = aggregate.get("observations")
        if not isinstance(observations, list) or not observations:
            return sorted(set(blockers + ["production_observations_missing"]))
        for observation in observations:
            identity = observation.get("capture_sha256")
            aliases = self._captures.get(identity)
            if not aliases:
                blockers.append("production_observation_content_unavailable")
                continue
            if identity not in self._checked:
                errors = []
                for _, raw in sorted(aliases.items()):
                    binding = binding_for_capture(raw, self._bindings)
                    found = production_evidence(raw, binding)["blockers"]
                    if not found:
                        # Identical content remains one observation. A copied
                        # filename cannot erase an intact original binding.
                        self._accepted[identity] = dict(
                            capture_sha256=identity,
                            source_capture=str(raw["_path"])
                            if raw.get("_path") is not None
                            else None,
                            diagnostic_companion=binding,
                        )
                        errors = []
                        break
                    errors.extend(found)
                self._checked[identity] = sorted(set(errors))
            blockers.extend(self._checked[identity])
        return sorted(set(blockers))

    def receipts(self, capture: dict) -> list[dict]:
        self.blockers(capture)
        return [
            dict(self._accepted[identity])
            for identity in sorted(
                {fact["capture_sha256"] for fact in evidence(capture).get("observations", [])}
            )
            if identity in self._accepted
        ]


def best_implementation(
    captures: Iterable[dict], *, candidate: dict | None = None, cpu_gpu_comparison: bool = False
) -> dict | None:
    valid = [
        c
        for c in captures
        if qualified(c)
        and phase_median(c) is not None
        and (
            candidate is None
            or comparison_environment(c, cpu_gpu_comparison=cpu_gpu_comparison)
            == comparison_environment(candidate, cpu_gpu_comparison=cpu_gpu_comparison)
        )
    ]
    return min(
        valid, key=lambda c: (phase_median(c), evidence(c)["implementation_key"]), default=None
    )


def row_identity(capture: dict, *, exclude: Iterable[str] = ()) -> dict:
    work = canonical_work(capture, exclude=exclude)

    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()

    return {
        "work_contract": json.loads(work),
        "work_contract_key": digest(work),
        "comparison_environment": digest(comparison_environment(capture)),
        "source_identity": digest(comparison_environment(capture, cpu_gpu_comparison=True)),
        "measurement_cohort": evidence(capture),
    }


def partition_rows(rows: list[dict]) -> list[tuple[tuple[str, str, str], list[dict]]]:
    """Share CPU timings only within the same host/build/timing/work contract."""
    work_groups = defaultdict(list)
    for row in rows:
        work_groups[(row["work_contract_key"], row["source_identity"])].append(row)
    result = []
    for (work, source), items in sorted(work_groups.items()):
        cpu = [r for r in items if r["backend"] in ("cpu", "cpu-reference", "wrap64-byte-limb")]
        gpu = [r for r in items if r not in cpu]
        environments = sorted({r["comparison_environment"] for r in gpu}) or ["cpu_only"]
        for environment in environments:
            members = cpu + [r for r in gpu if r["comparison_environment"] == environment]
            result.append(
                (
                    (work, source, environment),
                    sorted(
                        members,
                        key=lambda r: (
                            r["backend"],
                            r["measurement_cohort"].get("implementation_key", ""),
                        ),
                    ),
                )
            )
    return result


def best_row(rows: list[dict], backend: str | None = None) -> dict | None:
    valid = [
        r
        for r in rows
        if (backend is None or r["backend"] == backend)
        and r.get("measurement_cohort", {}).get("qualified") is True
        and (
            "numerical_reference" not in r
            or (
                isinstance(r["numerical_reference"], dict)
                and r["numerical_reference"].get("passed") is True
            )
        )
        and _number(r.get("median_end_to_end_us")) is not None
    ]
    return min(
        valid,
        key=lambda r: (r["median_end_to_end_us"], r["measurement_cohort"]["implementation_key"]),
        default=None,
    )


def compound_median(capture: dict, phases: list[str]) -> float | None:
    return phase_sum_measurement(capture, tuple(phases))["median_us"]


def phase_sum_measurement(capture: dict, phases: tuple[str, ...]) -> dict:
    """Sum named phase medians within each process before estimating its cohort.

    This is explicit component accounting, not a new raw whole-call sample or
    an inferred distribution of correlated phases. One-time phases are charged
    once per estimate; this policy does not amortize them over repeat counts.
    """
    if not phases or len(phases) > 16 or len(set(phases)) != len(phases):
        raise ValueError("bounded unique phase-sum contract required")
    observations = evidence(capture).get("observations")
    blockers = []
    if not observations:
        values = [phase_median(capture, phase) for phase in phases]
        total = (
            _number(sum(v for v in values if v is not None))
            if all(v is not None for v in values)
            else None
        )
        return dict(
            policy="explicit_process_phase_median_sum",
            phases=list(phases),
            median_us=total,
            process_relative_range=None,
            observations=[],
            blockers=["phase_sum_process_observations_missing"],
        )
    sums, records = [], []
    for observation in observations:
        values = [_number(observation["phase_medians_us"].get(phase)) for phase in phases]
        if any(value is None or value <= 0 for value in values):
            blockers.append("phase_sum_positive_observation_missing")
            continue
        total = _number(sum(value for value in values if value is not None))
        if total is None or total <= 0:
            blockers.append("phase_sum_nonfinite_total")
            continue
        sums.append(total)
        records.append(
            dict(
                capture_sha256=observation.get("capture_sha256"),
                phase_medians_us=dict(zip(phases, values)),
                total_us=total,
            )
        )
    duration = _number(statistics.median(sums)) if sums and not blockers else None
    spread = (max(sums) - min(sums)) / duration if duration is not None and duration > 0 else None
    if spread is None or not math.isfinite(spread):
        blockers.append("phase_sum_process_spread_unavailable")
    elif spread > DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD:
        blockers.append("phase_sum_process_spread_exceeds_ten_percent")
    return dict(
        policy="explicit_process_phase_median_sum",
        phases=list(phases),
        median_us=duration,
        process_relative_range=spread,
        observations=records,
        blockers=sorted(set(blockers)),
    )


def amortized_prepack_measurement(capture: dict) -> dict:
    """Charge each process's observed preparation over its own completed repeats."""
    observations = evidence(capture).get("observations")
    values, records, blockers = [], [], []
    if not observations:
        blockers.append("prepack_setup_process_observations_missing")
    for observation in observations or []:
        phases = observation.get("phase_medians_us") or {}
        duration, setup = _number(phases.get("end_to_end")), _number(phases.get("prepack_setup"))
        repeats = observation.get("measured_repeat_count")
        if (
            duration is None
            or duration <= 0
            or setup is None
            or setup < 0
            or type(repeats) is not int
            or repeats <= 0
        ):
            blockers.append("prepack_setup_process_measurement_invalid")
            continue
        inclusive = duration + setup / repeats
        if not math.isfinite(inclusive):
            blockers.append("prepack_setup_process_measurement_invalid")
            continue
        values.append(inclusive)
        records.append(
            dict(
                capture_sha256=observation["capture_sha256"],
                repeat_count=repeats,
                steady_median_us=duration,
                setup_us=setup,
                setup_inclusive_us=inclusive,
            )
        )
    complete = bool(values) and not blockers
    median = _number(statistics.median(values)) if complete else None
    if complete and median is None:
        blockers.append("prepack_setup_process_measurement_invalid")
    spread = (max(values) - min(values)) / median if median is not None else None
    if spread is not None and spread > DEFAULT_MAX_RUN_TO_RUN_REL_SPREAD:
        blockers.append("setup_inclusive_process_spread_exceeds_ten_percent")
    return dict(
        policy="per_process_prepack_setup_amortized_over_own_measured_repeats_v1",
        units="microseconds_per_completed_call_including_amortized_prepack_setup",
        median_us=median,
        process_relative_range=spread,
        observations=records,
        blockers=sorted(set(blockers)),
    )
