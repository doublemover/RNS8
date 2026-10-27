"""Typed offline calibration evidence; never installs a runtime route.

Independent process runs, not within-run repetitions, are the statistical unit.
Intervals use a conservative Student-t critical value (df=4) for >=5 runs.
Normal independent run means are an explicit assumption, not a confidence
guarantee for correlated/thermally drifting measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import re
from statistics import mean, stdev

from analytical_planner import Rate


@dataclass(frozen=True)
class Identity:
    target: str
    device: str
    runtime: str
    compiler: str
    artifact_sha256: str
    isa_sha256: str
    configuration_sha256: str
    clocks_policy: str
    power_policy: str
    instruction: str
    kernel_symbol: str
    dataset_sha256: str
    cache_state: str
    output_contract: str
    timing_source: str
    profiling_mode: str
    host_instrumentation_state: str = "not_applicable_instruction_microbenchmark"


@dataclass(frozen=True)
class Quantity:
    name: str
    value: int | None
    unit: str
    origin: str  # measured, modeled, or unavailable


@dataclass(frozen=True)
class Run:
    run_id: str
    elapsed_ns: tuple[float | None, ...]
    resolution_ns: float
    work_units: int
    warmup_count: int
    exact_reference_passed: bool
    isa_verified: bool
    counters: tuple[Quantity, ...] = ()


@dataclass(frozen=True)
class Calibration:
    identity: Identity
    work_kind: str
    unit: str
    runs: tuple[Run, ...]


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate(calibration: Calibration) -> None:
    identity = asdict(calibration.identity)
    if any(type(value) is not str or not value.strip() for value in identity.values()):
        raise ValueError("complete calibration identity is required")
    for key, value in identity.items():
        if key.endswith("_sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"invalid digest: {key}")
    if calibration.identity.cache_state not in {"hot", "cold", "mixed"}:
        raise ValueError("cache state must be explicit")
    if calibration.identity.timing_source not in {"hip_event", "host_monotonic"}:
        raise ValueError("unsupported timing source")
    if calibration.identity.profiling_mode not in {"none", "application_replay", "kernel_replay"}:
        raise ValueError("unsupported profiling mode")
    if calibration.identity.host_instrumentation_state not in {"not_applicable_instruction_microbenchmark", "hooks_disabled", "diagnostic_gpu_events"}:
        raise ValueError("unknown host instrumentation state")
    if not calibration.work_kind or not calibration.unit or not calibration.runs:
        raise ValueError("work definition and raw runs are required")
    ids = set()
    for run in calibration.runs:
        if type(run.run_id) is not str or not run.run_id or run.run_id in ids:
            raise ValueError("each independent run requires a unique ID")
        ids.add(run.run_id)
        if not _positive(run.resolution_ns) or type(run.work_units) is not int or run.work_units <= 0:
            raise ValueError("positive timing resolution and work count required")
        if type(run.warmup_count) is not int or run.warmup_count < 0 or not run.elapsed_ns:
            raise ValueError("invalid warmup/sample counts")
        if type(run.exact_reference_passed) is not bool or type(run.isa_verified) is not bool:
            raise ValueError("qualification observations must be boolean")
        for value in run.elapsed_ns:
            if value is not None and (not _positive(value) or value < run.resolution_ns):
                raise ValueError("use null for censored samples; never zero or sub-resolution times")
        names = set()
        for counter in run.counters:
            if not counter.name or counter.name in names or not counter.unit:
                raise ValueError("counter names must be unique with explicit units")
            names.add(counter.name)
            if counter.origin not in {"measured", "modeled", "unavailable"}:
                raise ValueError("counter origin required")
            if counter.origin == "unavailable":
                if counter.value is not None:
                    raise ValueError("unavailable counters must remain null")
            elif type(counter.value) is not int or counter.value < 0:
                raise ValueError("observed/modeled counters require nonnegative integer values")


def identity_hash(identity: Identity) -> str:
    return hashlib.sha256(json.dumps(asdict(identity), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def independent_run_interval(samples, resolution_ns):
    """Statistical interval only; this does not admit a calibration or an ISA.

    Runs are the statistical unit. Resolution must be supplied explicitly; a
    nominal clock tick is only a nominal bound and must be labelled by callers.
    """
    if len(samples) < 5 or not _positive(resolution_ns) or any(
            not run or any(not _positive(v) for v in run) for run in samples):
        raise ValueError("five independent uncensored runs and resolution required")
    means = [mean(run) for run in samples]
    center = mean(means)
    half = 2.776445106 * stdev(means) / math.sqrt(len(means)) + resolution_ns
    return {"lower_ns": center-half, "mean_ns": center, "upper_ns": center+half,
            "independent_run_stddev_ns": stdev(means), "nominal_confidence": 0.95,
            "method": "conservative_student_t_run_means_plus_resolution",
            "assumption": "independent_approximately_normal_stationary_run_means"}


def summarize(calibration: Calibration) -> dict:
    validate(calibration)
    failures = []
    if len(calibration.runs) < 5:
        failures.append("fewer_than_five_independent_runs")
    if any(value is None for run in calibration.runs for value in run.elapsed_ns):
        failures.append("censored_samples")
    if any(not run.exact_reference_passed for run in calibration.runs):
        failures.append("exact_reference_not_passed")
    if any(not run.isa_verified for run in calibration.runs):
        failures.append("isa_not_verified")
    if calibration.identity.host_instrumentation_state == "diagnostic_gpu_events":
        failures.append("diagnostic_host_hooks_not_uninstrumented_calibration")
    if calibration.work_kind in {"public_pipeline", "public_end_to_end"} and calibration.identity.host_instrumentation_state != "hooks_disabled":
        failures.append("public_pipeline_requires_explicit_hook_disabled_boundary")
    if calibration.identity.profiling_mode != "none":
        failures.append("profiled_time_not_uninstrumented_calibration")
    if len({run.work_units for run in calibration.runs}) != 1:
        failures.append("mismatched_work_count")
    interval = None
    if not failures:
        # t(0.975,4)=2.77645; using it for higher df is conservative.
        # Add the largest quantization interval instead of assuming it averages away.
        candidate = independent_run_interval([run.elapsed_ns for run in calibration.runs],
                                             max(run.resolution_ns for run in calibration.runs))
        if candidate["lower_ns"] <= 0:
            failures.append("uncertainty_reaches_zero")
        else:
            interval = candidate
    return {"schema": "rns8_calibration_v1", "identity_sha256": identity_hash(calibration.identity),
            "raw": asdict(calibration), "interval": interval, "admission_failures": failures,
            "rate_export_eligible": not failures, "promotion_eligible": False}


def planner_rate(calibration: Calibration, expected_identity: Identity, cohort: str) -> Rate:
    if calibration.identity != expected_identity:
        raise ValueError("calibration identity changed; remeasure before reuse")
    result = summarize(calibration)
    if not result["rate_export_eligible"] or not cohort:
        raise ValueError(f"calibration rejected: {result['admission_failures']}")
    units = calibration.runs[0].work_units
    interval = result["interval"]
    return Rate(Fraction(units) / Fraction(str(interval["upper_ns"])),
                Fraction(units) / Fraction(str(interval["lower_ns"])), calibration.unit, cohort)


def held_out_report(predictions: list[tuple[str, float, float, Calibration]],
                    training_run_ids: set[str], relative_tolerance: float) -> dict:
    if type(relative_tolerance) not in (int, float) or not math.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
        raise ValueError("held-out tolerance must be explicitly within [0,1)")
    if not predictions or len({item[0] for item in predictions}) != len(predictions):
        raise ValueError("unique held-out cases required")
    rows = []
    seen = set(training_run_ids)
    for name, lower, upper, calibration in predictions:
        if not name or not _positive(lower) or not _positive(upper) or lower > upper:
            raise ValueError("invalid predicted interval")
        ids = {run.run_id for run in calibration.runs}
        if ids & seen:
            raise ValueError("held-out runs overlap training or another held-out case")
        seen.update(ids)
        result = summarize(calibration)
        if not result["rate_export_eligible"]:
            raise ValueError("unqualified held-out measurements")
        measured = result["interval"]
        # Require the entire measurement interval inside the declared expanded
        # prediction; interval overlap alone is not model validation.
        passed = measured["lower_ns"] >= lower*(1-relative_tolerance) and measured["upper_ns"] <= upper*(1+relative_tolerance)
        rows.append({"case": name, "predicted_lower_ns": lower, "predicted_upper_ns": upper,
                     "measured": measured, "within_tolerance": passed,
                     "identity_sha256": result["identity_sha256"]})
    return {"schema": "rns8_calibration_discrepancy_v1", "relative_tolerance": relative_tolerance,
            "cases": rows, "held_out_passed": all(row["within_tolerance"] for row in rows),
            "promotion_eligible": False}


def planner_rates(calibrations: tuple[Calibration, ...], expected: dict[str, Identity]) -> dict[str, Rate]:
    """Export one immutable same-environment bundle with no invented cohort ID."""
    if not calibrations or len({item.work_kind for item in calibrations}) != len(calibrations):
        raise ValueError("calibrations require unique work kinds")
    if set(expected) != {item.work_kind for item in calibrations}:
        raise ValueError("expected identities must cover exactly the supplied rates")
    fields = ("target", "device", "runtime", "compiler", "artifact_sha256", "clocks_policy", "power_policy")
    environments = {tuple(getattr(item.identity, key) for key in fields) for item in calibrations}
    if len(environments) != 1:
        raise ValueError("cannot mix calibration environments or binaries")
    records = sorted((item.work_kind, identity_hash(item.identity), sorted(run.run_id for run in item.runs))
                     for item in calibrations)
    cohort = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
    return {item.work_kind: planner_rate(item, expected[item.work_kind], cohort) for item in calibrations}


def repeated_b_break_even(prepare_ns: Fraction, packed_ns: Fraction, unpacked_ns: Fraction) -> int | None:
    """Modeled strict break-even; measured profitability needs complete paired runs."""
    if prepare_ns < 0 or packed_ns <= 0 or unpacked_ns <= 0:
        raise ValueError("invalid cost")
    saving = unpacked_ns-packed_ns
    return None if saving <= 0 else int(prepare_ns//saving)+1
