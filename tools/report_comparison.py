"""Report-facing admission, implementation selection and matched comparison facts."""

from __future__ import annotations

from dataclasses import dataclass
import json

from execution_host_identity import hardware_cohort_identity

from observation_cohorts import (
    ProductionCohortIndex,
    aggregate_captures,
    canonical_work,
    comparison_environment,
    comparison_noise_margin,
    evidence,
    phase_median,
    row_identity,
)
from reference_verdict import CPUReferenceIndex, reference_verdict
from review_contract import release_capture_satisfied
from timing_eligibility import capture_identity


@dataclass(frozen=True)
class ComparisonPolicy:
    name: str
    work_exclusions: tuple[str, ...] = ()
    require_cpu_anchor: bool = True
    cpu_gpu_comparison: bool = False

    def work(self, capture: dict) -> str:
        return canonical_work(capture, exclude=self.work_exclusions)

    def environment(self, capture: dict) -> str:
        if not self.cpu_gpu_comparison:
            return comparison_environment(capture)
        identity = json.loads(comparison_environment(capture))
        # CPU/GPU timings can join only an observed OS instance and otherwise
        # identical source/build/host clock policy. Device data is GPU-only.
        identity["observed_hardware"] = hardware_cohort_identity(capture, cpu_anchor=True)
        identity.pop("device", None)
        return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def processor_pair(self, candidate: dict, baseline: dict) -> bool:
        cpu = ("cpu", "cpu-reference", "wrap64-byte-limb")
        opposite = (candidate.get("backend_selected") in cpu) != (
            baseline.get("backend_selected") in cpu
        )
        return opposite if self.cpu_gpu_comparison else not opposite


class ReportCohorts:
    """Retain raw evidence and aggregate once before any implementation selection."""

    def __init__(self, captures: list[dict], diagnostic_bindings: dict | None = None):
        self.observations = aggregate_captures(captures)
        self.production = ProductionCohortIndex(captures, diagnostic_bindings)
        self.references = CPUReferenceIndex(captures)
        self._numerical = {}
        self._admission = {}

    def numerical(self, capture: dict, *, require_cpu_anchor: bool = True):
        key = (
            capture_identity(capture),
            evidence(capture).get("implementation_key"),
            require_cpu_anchor,
        )
        if key not in self._numerical:
            self._numerical[key] = self.references.verdict(
                capture, require_anchor=require_cpu_anchor
            )
        return self._numerical[key]

    def admission(self, capture: dict, policy: ComparisonPolicy | None = None) -> list[str]:
        anchor = policy.require_cpu_anchor if policy is not None else True
        key = (capture_identity(capture), evidence(capture).get("implementation_key"), anchor)
        if key not in self._admission:
            blockers = self.production.blockers(capture) + list(
                self.numerical(capture, require_cpu_anchor=anchor).blockers
            )
            if not release_capture_satisfied(capture):
                blockers.append("not_release_review")
            self._admission[key] = tuple(sorted(set(blockers)))
        return list(self._admission[key])

    def metadata(self, capture: dict, policy: ComparisonPolicy) -> dict:
        return dict(
            row_identity(capture, exclude=policy.work_exclusions),
            capture_sha256=capture_identity(capture),
            comparison_scope=policy.name,
            intentional_work_differences=list(policy.work_exclusions),
            cpu_gpu_comparison=policy.cpu_gpu_comparison,
            comparison_environment_contract=policy.environment(capture),
            reference_requirement="separate_cpu_anchor_and_full_output"
            if policy.require_cpu_anchor
            else "embedded_full_cpu_output_reference",
            numerical_reference=self.numerical(
                capture, require_cpu_anchor=policy.require_cpu_anchor
            ).to_json(),
            admission_blockers=self.admission(capture, policy),
            production_observations=self.production.receipts(capture),
        )

    def best(self, captures: list[dict], candidate: dict, policy: ComparisonPolicy) -> dict | None:
        valid = []
        work, environment = policy.work(candidate), policy.environment(candidate)
        for capture in captures:
            if (
                not policy.processor_pair(candidate, capture)
                or policy.work(capture) != work
                or policy.environment(capture) != environment
                or self.admission(capture, policy)
            ):
                continue
            duration = phase_median(capture)
            if duration is not None and duration > 0:
                valid.append(capture)
        return min(
            valid,
            key=lambda capture: (phase_median(capture), evidence(capture)["implementation_key"]),
            default=None,
        )

    def compare(self, candidate: dict, baseline: dict | None, policy: ComparisonPolicy) -> dict:
        blockers = self.admission(candidate, policy)
        work_match = baseline is not None and policy.work(candidate) == policy.work(baseline)
        environment_match = baseline is not None and policy.environment(
            candidate
        ) == policy.environment(baseline)
        if baseline is None:
            blockers.append("qualified_baseline_missing")
        else:
            blockers.extend("baseline:" + value for value in self.admission(baseline, policy))
            if not work_match:
                blockers.append("comparison_work_contract_mismatch")
            if not environment_match:
                blockers.append("comparison_environment_mismatch")
            if not policy.processor_pair(candidate, baseline):
                blockers.append("comparison_processor_class_mismatch")
            # Embedded CPU oracles still need the same mathematical input key
            # and output checksum across the two physical implementations.
            if not policy.require_cpu_anchor:
                blockers.extend(reference_verdict(candidate, [baseline]).blockers)
                blockers.extend(
                    "baseline:" + value
                    for value in reference_verdict(baseline, [candidate]).blockers
                )
        left = phase_median(baseline) if baseline is not None else None
        right = phase_median(candidate)
        if left is None or right is None or left <= 0 or right <= 0:
            blockers.append("positive_completed_output_timing_missing")
        margin = comparison_noise_margin(candidate, baseline) if baseline is not None else None
        if margin is None:
            blockers.append("qualified_noise_margin_missing")
        blockers = sorted(set(blockers))
        return dict(
            comparison_scope=policy.name,
            intentional_work_differences=list(policy.work_exclusions),
            cpu_gpu_comparison=policy.cpu_gpu_comparison,
            matching_work=work_match,
            matching_environment=environment_match,
            candidate_measurement_cohort=evidence(candidate),
            baseline_measurement_cohort=evidence(baseline),
            candidate_production_observations=self.production.receipts(candidate),
            baseline_production_observations=self.production.receipts(baseline)
            if baseline is not None
            else [],
            reference_requirement="separate_cpu_anchor_and_full_output"
            if policy.require_cpu_anchor
            else "embedded_full_cpu_output_reference",
            candidate_numerical_reference=self.numerical(
                candidate, require_cpu_anchor=policy.require_cpu_anchor
            ).to_json(),
            baseline_numerical_reference=self.numerical(
                baseline, require_cpu_anchor=policy.require_cpu_anchor
            ).to_json()
            if baseline is not None
            else None,
            candidate_capture_sha256=capture_identity(candidate),
            baseline_capture_sha256=capture_identity(baseline) if baseline is not None else None,
            required_speedup_margin=margin,
            speedup=left / right if not blockers else None,
            qualified=not blockers,
            blockers=blockers,
        )
