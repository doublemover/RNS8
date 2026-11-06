"""Content-bound generic review observations, separate from cache installation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from benchmark_schema import load_capture, validate_capture
from benchmark_sweep_lib.capture_metadata import capture_contract_key
from json_evidence import parse_evidence_json
from observation_cohorts import canonical_work, comparison_environment, evidence, phase_median
from report_comparison import ReportCohorts
from review_contract import REVIEW_SCHEMA_VERSION
from timing_eligibility import capture_identity
from diagnostic_companion import binding_for_capture

SCHEMA = "rns8_review_capture_receipt_v1"
MAX_INPUTS = 4096
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_review_report(path: Path) -> tuple[dict, str]:
    if not path.is_file() or not 1 <= path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("bounded nonempty review file required")
    raw = path.read_bytes()
    if not 1 <= len(raw) <= MAX_FILE_BYTES:
        raise ValueError("review file exceeded bounded read")
    data = parse_evidence_json(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("review report object required")
    return data, hashlib.sha256(raw).hexdigest()


def make_review_receipt(
    candidate: dict, captures: list[dict], required: list[str], bindings=None
) -> dict | None:
    backends = set(required) | {candidate.get("backend_selected"), "cpu", "cpu-reference"}
    captures = [capture for capture in captures if capture.get("backend_selected") in backends]
    if not 1 <= len(captures) <= MAX_INPUTS or any(
        not capture.get("_path") for capture in captures
    ):
        return None
    inputs = {}
    for capture in captures:
        path = str(Path(capture["_path"]).resolve())
        record = dict(
            capture=path,
            capture_sha256=capture_identity(capture),
            diagnostic_companion=binding_for_capture(capture, bindings),
        )
        if path in inputs and inputs[path] != record:
            return None
        inputs[path] = record
    result = dict(
        schema=SCHEMA,
        capture_sha256=capture_identity(candidate),
        work_sha256=digest(canonical_work(candidate)),
        environment_sha256=digest(comparison_environment(candidate)),
        backend=candidate.get("backend_selected"),
        selected_kernel=candidate.get("selected_kernel"),
        required_baselines=sorted(set(required)),
        inputs=sorted(
            inputs.values(), key=lambda value: (value["capture_sha256"], value["capture"])
        ),
    )
    result["identity_sha256"] = digest(result)
    return result


@dataclass(frozen=True)
class ReviewReceiptVerdict:
    bound: bool
    qualified: bool
    blockers: tuple[str, ...]
    measurement_cohort: dict | None = None

    def to_json(self) -> dict:
        value = asdict(self)
        value["blockers"] = list(self.blockers)
        return value


class ReviewReceiptReader:
    """One bounded read snapshot per summary; never treat filenames as authority."""

    def __init__(self):
        self._reports = {}
        self._captures = {}
        self._cohorts = {}
        self._total_bytes = 0

    def _capture(self, path: Path) -> dict:
        path = path.resolve(strict=True)
        key = str(path)
        if key not in self._captures:
            size = path.stat().st_size
            if not path.is_file() or not 1 <= size <= MAX_FILE_BYTES:
                raise ValueError("bounded regular reviewed capture required")
            self._total_bytes += size
            if self._total_bytes > MAX_TOTAL_BYTES:
                raise ValueError("review capture inventory exceeded byte budget")
            capture = load_capture(path)
            validate_capture(capture, path)
            capture["_path"] = key
            self._captures[key] = capture
        return self._captures[key]

    def validate(self, capture: dict, entry: dict) -> ReviewReceiptVerdict:
        blockers = []
        if entry.get("review_receipt_conflict"):
            blockers.append("conflicting_review_receipts")
        if (
            type(entry.get("review_schema_version")) is not int
            or entry.get("review_schema_version") != REVIEW_SCHEMA_VERSION
        ):
            blockers.append("unsupported_review_schema")
        row = entry.get("candidate") or {}
        receipt = row.get("review_capture_receipt")
        if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
            blockers.append("review_capture_receipt_missing_or_unsupported")
            return ReviewReceiptVerdict(False, False, tuple(blockers))
        expected_fields = {
            "schema",
            "capture_sha256",
            "work_sha256",
            "environment_sha256",
            "backend",
            "selected_kernel",
            "required_baselines",
            "inputs",
            "identity_sha256",
        }
        if set(receipt) != expected_fields or receipt.get("identity_sha256") != digest(
            {key: value for key, value in receipt.items() if key != "identity_sha256"}
        ):
            blockers.append("review_receipt_identity_invalid")
        identity = capture_identity(capture)
        if receipt.get("capture_sha256") != identity or row.get("capture_sha256") != identity:
            blockers.append("review_capture_content_mismatch")
        if (
            receipt.get("work_sha256") != digest(canonical_work(capture))
            or receipt.get("environment_sha256") != digest(comparison_environment(capture))
            or receipt.get("backend") != capture.get("backend_selected")
            or row.get("backend") != capture.get("backend_selected")
            or receipt.get("selected_kernel") != capture.get("selected_kernel")
            or row.get("selected_kernel") != capture.get("selected_kernel")
            or entry.get("contract_key") != capture_contract_key(capture)
            or row.get("median_end_to_end_us") != phase_median(capture)
            or row.get("checksum") != capture.get("checksum_u64")
        ):
            blockers.append("review_capture_contract_mismatch")
        required = receipt.get("required_baselines")
        if not isinstance(required, list) or not all(isinstance(value, str) for value in required):
            blockers.append("review_required_baselines_invalid")
        elif not isinstance(entry.get("required_baselines"), list) or not all(
            isinstance(value, str) for value in entry["required_baselines"]
        ):
            blockers.append("review_declared_baselines_invalid")
        elif required != sorted(set(entry["required_baselines"])):
            blockers.append("review_required_baselines_mismatch")
        try:
            path = str(Path(entry["review_report"]).resolve(strict=True))
            if path not in self._reports:
                self._reports[path] = read_review_report(Path(path))[1]
            if self._reports[path] != entry.get("review_report_sha256"):
                blockers.append("review_report_content_changed")
        except (KeyError, OSError, TypeError, ValueError):
            blockers.append("review_report_unreadable")
        inputs = receipt.get("inputs")
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= MAX_INPUTS:
            blockers.append("review_capture_inventory_invalid")
        if blockers:
            return ReviewReceiptVerdict(False, False, tuple(sorted(set(blockers))))
        captures, bindings, paths = [], {}, set()
        try:
            for record in inputs:
                if not isinstance(record, dict) or set(record) != {
                    "capture",
                    "capture_sha256",
                    "diagnostic_companion",
                }:
                    raise ValueError("review input record shape")
                path = Path(record["capture"])
                if not path.is_absolute() or str(path) in paths:
                    raise ValueError("review input paths must be absolute and unique")
                paths.add(str(path))
                other = self._capture(path)
                if capture_identity(other) != record["capture_sha256"]:
                    raise ValueError("review input content changed")
                captures.append(other)
                if record["diagnostic_companion"] is not None:
                    bindings[str(path.resolve())] = record["diagnostic_companion"]
        except (KeyError, OSError, TypeError, ValueError):
            return ReviewReceiptVerdict(False, False, ("review_input_content_missing_or_changed",))
        cohort_key = digest(inputs)
        if cohort_key not in self._cohorts:
            self._cohorts[cohort_key] = ReportCohorts(captures, bindings)
        cohorts = self._cohorts[cohort_key]
        candidates = [
            view
            for view in cohorts.observations
            if identity
            in {fact["capture_sha256"] for fact in evidence(view).get("submitted_observations", [])}
        ]
        if len(candidates) != 1:
            return ReviewReceiptVerdict(
                False, False, ("review_candidate_observation_missing_or_conflicting",)
            )
        view = candidates[0]
        blockers.extend(cohorts.admission(view))
        bound = not any(
            blocker.startswith("matched_diagnostic_companion_")
            for other in cohorts.observations
            for blocker in cohorts.production.blockers(other)
        )
        if not bound:
            blockers.append("review_diagnostic_binding_unavailable_or_changed")
        for backend in required:
            cpu = backend in ("cpu", "cpu-reference", "wrap64-byte-limb")
            matched = [
                other
                for other in cohorts.observations
                if other.get("backend_selected") == backend
                and canonical_work(other) == canonical_work(capture)
                and comparison_environment(other, cpu_anchor=cpu)
                == comparison_environment(capture, cpu_anchor=cpu)
                and not cohorts.admission(other)
            ]
            if not matched:
                blockers.append("review_qualified_baseline_missing:" + backend)
        return ReviewReceiptVerdict(
            bound, not blockers, tuple(sorted(set(blockers))), evidence(view)
        )
