"""Numerical evidence for schema-validated captures, separate from timing/promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Iterable

from benchmark_schema.public_timing import validate_native_reference
from benchmark_sweep_lib.capture_metadata import capture_checksum_policy
from result_compare import contract_value
from timing_eligibility import capture_identity

SCHEMA = "rns8_numerical_reference_verdict_v1"
FULL_REFERENCE_METHODS = frozenset(
    ("cached_full_integer_reference", "boost_cpp_int_full_integer_reference")
)
NUMERICAL_KEYS = (
    "semantics",
    "finite_modulus",
    "bound_kind",
    "bound_mode",
    "bound",
    "m",
    "n",
    "k",
    "seed",
    "input_distribution",
    "input_profile",
    "residue_chain_length",
    "exact_wide_limb_count",
    "requested_next_op.resolved",
    "host_api_batch.work_contract",
    "grouped_dispatch.task_descriptor_contract",
)


def checksum_u64(capture: dict | None) -> int | None:
    if not isinstance(capture, dict):
        return None
    value = capture.get("checksum_u64")
    return value if type(value) is int and 0 <= value < 2**64 else None


def numerical_contract(capture: dict) -> str:
    for key in ("m", "n", "k"):
        if type(capture.get(key)) is not int or capture[key] < 1:
            raise ValueError("numerical work requires positive integer " + key)
    if not isinstance(capture.get("semantics"), str) or not capture["semantics"]:
        raise ValueError("numerical work semantics missing")
    if type(capture.get("seed")) is not int or not 0 <= capture["seed"] < 2**64:
        raise ValueError("numerical input seed missing")
    work = {key: contract_value(capture, key) for key in NUMERICAL_KEYS}
    work["input_profile"] = capture.get("input_profile", "uniform-small")
    work["checksum_policy"] = capture_checksum_policy(capture)
    reference = capture.get("native_reference_verification")
    work["reference_input_key"] = (
        reference.get("cache_key")
        if isinstance(reference, dict)
        and reference.get("method") in FULL_REFERENCE_METHODS
        and _hex256(reference.get("cache_key"))
        else None
    )
    # Physical tiling, padding, instrumentation and preparation do not change
    # the mathematical result. Their independent timing contracts still apply.
    return json.dumps(work, sort_keys=True, separators=(",", ":"), allow_nan=False)


def numerical_observation(capture: dict) -> dict:
    """Facts for one raw record; no CPU-backend-name correctness inference."""
    contract = numerical_contract(capture)
    record = capture.get("native_reference_verification")
    coverage = (
        isinstance(record, dict)
        and record.get("method") in FULL_REFERENCE_METHODS
        and all(isinstance(record.get(key), str) for key in ("cache_key", "payload_sha256"))
        and _hex256(record.get("cache_key"))
        and not validate_native_reference(capture)
    )
    return dict(
        capture_sha256=capture_identity(capture),
        work_sha256=hashlib.sha256(contract.encode()).hexdigest(),
        checksum_u64=checksum_u64(capture),
        full_output_reference_covered=bool(coverage),
        reference_method=record.get("method") if isinstance(record, dict) else None,
    )


def _hex256(value) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _observations(capture: dict) -> tuple[list[dict], list[str]]:
    if not isinstance(capture, dict):
        return [], ["numerical_capture_identity_invalid"]
    try:
        raw = numerical_observation(capture)
    except (KeyError, TypeError, ValueError, OverflowError):
        return [], ["numerical_capture_identity_invalid"]
    aggregate = capture.get("_observation_aggregate")
    if aggregate is None:
        return [raw], []
    if not isinstance(aggregate, dict) or aggregate.get("schema") != "rns8_process_observations_v1":
        return [], ["numerical_cohort_evidence_missing"]
    if aggregate.get("numerical_observation_errors"):
        return [], ["numerical_cohort_evidence_invalid"]
    facts, submitted = (
        aggregate.get("numerical_observations"),
        aggregate.get("submitted_observations"),
    )
    if not isinstance(facts, list) or not facts or not isinstance(submitted, list):
        return [], ["numerical_cohort_evidence_missing"]
    if any(
        not isinstance(record, dict) or not _hex256(record.get("capture_sha256"))
        for record in submitted
    ):
        return [], ["numerical_cohort_content_binding_missing"]
    expected = {record["capture_sha256"] for record in submitted}
    actual = set()
    for fact in facts:
        if (
            not isinstance(fact, dict)
            or set(fact) != set(raw)
            or not _hex256(fact.get("capture_sha256"))
            or not _hex256(fact.get("work_sha256"))
            or type(fact.get("full_output_reference_covered")) is not bool
            or (
                fact.get("reference_method") is not None
                and not isinstance(fact["reference_method"], str)
            )
            or (
                fact["full_output_reference_covered"]
                and fact.get("reference_method") not in FULL_REFERENCE_METHODS
            )
            or (fact.get("checksum_u64") is not None and checksum_u64(fact) is None)
        ):
            return [], ["numerical_cohort_evidence_invalid"]
        if fact["capture_sha256"] in actual or fact["work_sha256"] != raw["work_sha256"]:
            return [], ["numerical_cohort_identity_conflict"]
        actual.add(fact["capture_sha256"])
        if fact["capture_sha256"] == raw["capture_sha256"] and fact != raw:
            return [], ["numerical_cohort_identity_conflict"]
    if actual != expected or raw["capture_sha256"] not in actual:
        return [], ["numerical_cohort_content_binding_missing"]
    return facts, []


@dataclass(frozen=True)
class ReferenceVerdict:
    schema: str
    candidate_capture_sha256: str | None
    reference_capture_sha256: tuple[str, ...]
    work_sha256: str | None
    checksum_comparison: str
    reference_coverage_present: bool
    full_output_comparison: str
    blockers: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.full_output_comparison == "passed" and not self.blockers

    def to_json(self) -> dict:
        result = asdict(self)
        result["reference_capture_sha256"] = list(self.reference_capture_sha256)
        result["blockers"] = list(self.blockers)
        result["passed"] = self.passed
        return result


def reference_verdict(candidate: dict, references: Iterable[dict] = ()) -> ReferenceVerdict:
    facts, errors = _observations(candidate)
    blockers = list(errors)
    candidate_hash = work_hash = None
    values = set()
    coverage = False
    if facts:
        candidate_hash = capture_identity(candidate)
        work_hash = facts[0]["work_sha256"]
        values = {fact["checksum_u64"] for fact in facts if fact["checksum_u64"] is not None}
        if any(fact["checksum_u64"] is None for fact in facts):
            blockers.append("candidate_checksum_missing")
        if len(values) > 1:
            blockers.append("candidate_checksum_conflict")
        coverage = all(fact["full_output_reference_covered"] for fact in facts)
        if not coverage:
            blockers.append("full_output_reference_pending")
    reference_hashes, reference_values = set(), set()
    supplied = matched = 0
    for reference in references:
        supplied += 1
        other, other_errors = _observations(reference)
        if other_errors:
            blockers.extend("reference:" + error for error in other_errors)
            continue
        if not other or not work_hash or any(fact["work_sha256"] != work_hash for fact in other):
            continue
        matched += 1
        for fact in other:
            reference_hashes.add(fact["capture_sha256"])
            if fact["checksum_u64"] is None:
                blockers.append("reference_checksum_missing")
            else:
                reference_values.add(fact["checksum_u64"])
    if supplied and not matched:
        blockers.append("reference_work_contract_missing")
    if len(reference_values) > 1:
        blockers.append("reference_checksum_conflict")
    mismatch = bool(values and reference_values and values != reference_values) or len(values) > 1
    if mismatch:
        blockers.append("checksum_mismatch_vs_reference")
    checksum_state = (
        "mismatch" if mismatch else "matched" if values and reference_values else "unavailable"
    )
    full_state = "mismatch" if mismatch else "passed" if coverage and not blockers else "pending"
    return ReferenceVerdict(
        SCHEMA,
        candidate_hash,
        tuple(sorted(reference_hashes)),
        work_hash,
        checksum_state,
        coverage,
        full_state,
        tuple(sorted(set(blockers))),
    )


class CPUReferenceIndex:
    """Index every CPU observation by mathematical work, independently of timing."""

    def __init__(self, captures: Iterable[dict]) -> None:
        self._references: dict[str, list[dict]] = {}
        for capture in captures:
            backend = capture.get("backend_selected") or capture.get("backend_requested")
            if backend not in ("cpu", "cpu-reference"):
                continue
            try:
                key = numerical_contract(capture)
            except (TypeError, ValueError, KeyError):
                continue
            self._references.setdefault(key, []).append(capture)

    def verdict(self, candidate: dict) -> ReferenceVerdict:
        try:
            references = self._references.get(numerical_contract(candidate), ())
        except (TypeError, ValueError, KeyError):
            references = ()
        verdict = reference_verdict(candidate, references)
        if verdict.reference_capture_sha256:
            return verdict
        blockers = tuple(sorted(set(verdict.blockers) | {"cpu_reference_work_coverage_missing"}))
        return replace(
            verdict,
            blockers=blockers,
            full_output_comparison="mismatch"
            if verdict.full_output_comparison == "mismatch"
            else "pending",
        )
