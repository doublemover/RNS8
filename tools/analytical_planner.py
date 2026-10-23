"""Offline resource-constrained exact-pipeline predictor, never a promotion path.

Stages are already executable kernel/copy candidates. Rates come from a named
calibration cohort; missing conversion or instruction rates reject a candidate.
Each stage reserves all declared resources until completion. This conservative
list schedule models dependencies and contention instead of summing overlapping
stages or ignoring conversion work.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from arithmetic_certificate import ArithmeticCertificate, certificate_hash, verify


@dataclass(frozen=True)
class Rate:
    lower_units_per_ns: Fraction
    upper_units_per_ns: Fraction
    unit: str
    cohort: str


@dataclass(frozen=True)
class Stage:
    name: str
    kernel_symbol: str
    dependencies: tuple[str, ...]
    resources: tuple[str, ...]
    work: tuple[tuple[str, int], ...]
    launch_ns: int
    tiles: int = 1
    resident_tiles: int = 1
    bytes_allocated: int = 0
    retain_output: bool = False


def predict(stages: tuple[Stage, ...], rates: dict[str, Rate], certificate: ArithmeticCertificate,
            workspace_budget: int, target: str) -> dict:
    proof = verify(certificate)
    if not proof["algebraically_verified"]:
        raise ValueError(f"arithmetic obligations failed: {proof['proof_failures']}")
    if not target or workspace_budget < 0:
        raise ValueError("target and nonnegative workspace budget are required")
    names = {stage.name for stage in stages}
    if not stages or len(names) != len(stages) or "" in names:
        raise ValueError("stages must have unique nonempty identities")
    cohorts = set()
    for stage in stages:
        if not stage.kernel_symbol or not stage.resources or len(set(stage.resources)) != len(stage.resources):
            raise ValueError("stage requires a concrete symbol and unique resources")
        if not set(stage.dependencies) <= names or stage.name in stage.dependencies:
            raise ValueError("unknown or self dependency")
        if (stage.launch_ns < 0 or stage.tiles <= 0 or stage.resident_tiles <= 0 or stage.bytes_allocated < 0
                or len({kind for kind, _ in stage.work}) != len(stage.work)):
            raise ValueError("invalid work/resource descriptor")
        for kind, count in stage.work:
            rate = rates.get(kind)
            if type(count) is not int or count < 0 or rate is None:
                raise ValueError(f"missing or invalid calibrated work term: {kind}")
            if (not rate.cohort or not rate.unit or rate.lower_units_per_ns <= 0
                    or rate.upper_units_per_ns < rate.lower_units_per_ns):
                raise ValueError(f"invalid calibrated rate: {kind}")
            cohorts.add(rate.cohort)
    if len(cohorts) != 1:
        raise ValueError("candidate requires one complete same-environment calibration cohort")

    def schedule(optimistic: bool):
        pending = list(stages)
        finished, started, resource_end, terms = {}, {}, {}, {}
        while pending:
            ready = [stage for stage in pending if all(dep in finished for dep in stage.dependencies)]
            if not ready:
                raise ValueError("cyclic execution graph")
            stage = min(ready, key=lambda item: item.name)
            start = max([Fraction(0)] + [finished[dep] for dep in stage.dependencies]
                        + [resource_end.get(resource, Fraction(0)) for resource in stage.resources])
            # Each work count describes one resident wave. Explicit wave count
            # accounts for quantization and underfilled final scheduling waves.
            waves = (stage.tiles + stage.resident_tiles - 1) // stage.resident_tiles
            components = {kind: Fraction(count) / (rates[kind].upper_units_per_ns if optimistic
                                                  else rates[kind].lower_units_per_ns)
                          for kind, count in stage.work}
            duration = stage.launch_ns + waves * max(components.values(), default=Fraction(0))
            started[stage.name] = start
            finished[stage.name] = start + duration
            for resource in stage.resources:
                resource_end[resource] = start + duration
            terms[stage.name] = {"start_ns": float(start), "end_ns": float(start + duration),
                                 "waves": waves, "work_terms_ns": {k: float(v) for k, v in components.items()}}
            pending.remove(stage)
        return started, finished, terms

    low_start, low_end, low_terms = schedule(True)
    high_start, high_end, high_terms = schedule(False)
    # For the fixed dependency/resource order, max-plus starts/completions are
    # monotone in every stage duration. Each possible lifetime is contained in
    # [earliest producer start, latest last-consumer completion]. Intersecting
    # these envelopes is conservative but sound even for interior rate choices;
    # neither endpoint schedule alone bounds allocation overlap.
    events = []
    end = max(high_end.values())
    for stage in stages:
        consumers = [high_end[consumer.name] for consumer in stages if stage.name in consumer.dependencies]
        release = end if stage.retain_output else max(consumers + [high_end[stage.name]])
        events.append((low_start[stage.name], 0, stage.bytes_allocated))
        events.append((release, 1, -stage.bytes_allocated))
    live = peak = 0
    for _, _, delta in sorted(events):
        live += delta
        peak = max(peak, live)
    if peak > workspace_budget:
        raise ValueError(f"workspace proof exceeds budget: {peak} > {workspace_budget}")
    return {"schema": "rns8_analytical_pipeline_v1", "target": target,
            "certificate_sha256": certificate_hash(certificate), "calibration_cohort": next(iter(cohorts)),
            "latency_lower_ns": float(max(low_end.values())), "latency_upper_ns": float(end),
            "peak_workspace_bytes": peak, "optimistic_schedule": low_terms, "conservative_schedule": high_terms,
            "workspace_admission": "robust_lifetime_envelopes",
            "workspace_bound_tight": False,
            "model_assumptions": ["exclusive_declared_resources", "work_per_resident_wave",
                                  "intra_stage_work_terms_overlap", "fixed_named_topological_order",
                                  "closed_lifetime_envelopes_include_equal_time_boundaries",
                                  "one_allocation_per_producer_retained_through_all_consumers"],
            "instruction_qualification_required": True, "runtime_route_changed": False,
            "promotion_eligible": False}


def pareto_frontier(candidates: list[dict]) -> list[dict]:
    """Retain nondominated latency/workspace predictions; no measured-win label."""
    return [candidate for candidate in candidates if not any(
        other is not candidate and other["latency_upper_ns"] <= candidate["latency_upper_ns"]
        and other["peak_workspace_bytes"] <= candidate["peak_workspace_bytes"]
        and (other["latency_upper_ns"] < candidate["latency_upper_ns"]
             or other["peak_workspace_bytes"] < candidate["peak_workspace_bytes"])
        for other in candidates)]
