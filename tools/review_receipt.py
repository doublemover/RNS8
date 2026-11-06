"""Content-bound generic review observations, separate from cache installation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from benchmark_schema import validate_capture
from benchmark_request import file_digest
from evidence_database_lib.isa import isa_index_from_reports
from benchmark_sweep_lib.capture_metadata import (
    capture_contract_key,
    correctness_anchor_reference_capture,
)
from json_evidence import parse_evidence_json
from observation_cohorts import (
    canonical_work,
    comparison_environment,
    evidence,
    phase_median,
    measurement_identity,
)
from report_comparison import ComparisonPolicy, ReportCohorts
from review_contract import REVIEW_SCHEMA_VERSION
from timing_eligibility import capture_identity
from diagnostic_companion import binding_for_capture

SCHEMA = "rns8_review_capture_receipt_v2"
MAX_INPUTS = 4096
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024

RECEIPT_POLICY = ComparisonPolicy("review_receipt_completed_output", amortize_prepack_setup=True)
CPU_RECEIPT_POLICY = ComparisonPolicy(
    "review_receipt_same_host_cpu_gpu", amortize_prepack_setup=True, cpu_gpu_comparison=True
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read_document(path: Path) -> tuple[dict, str, int]:
    if not path.is_file() or not 1 <= path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("bounded nonempty review file required")
    raw = path.read_bytes()
    if not 1 <= len(raw) <= MAX_FILE_BYTES:
        raise ValueError("review file exceeded bounded read")
    data = parse_evidence_json(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("review report object required")
    return data, hashlib.sha256(raw).hexdigest(), len(raw)


def read_review_report(path: Path) -> tuple[dict, str]:
    data, identity, _ = _read_document(path)
    return data, identity


def isa_input_references(index: dict | None) -> list[dict]:
    paths = {
        str(Path(report["isa_report_path"]).resolve())
        for reports in (index or {}).values()
        for report in reports
        if isinstance(report.get("isa_report_path"), str)
        and Path(report["isa_report_path"]).is_file()
    }
    return [dict(path=path, sha256=file_digest(Path(path))) for path in sorted(paths)]


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
        measurement_identity_sha256=measurement_identity(candidate),
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
        self._documents = {}
        self._document_hashes = {}
        self._total_bytes = 0

    def _document(self, path: Path) -> dict:
        path = path.resolve(strict=True)
        key = str(path)
        if key not in self._documents:
            if self._total_bytes + path.stat().st_size > MAX_TOTAL_BYTES:
                raise ValueError("review input inventory exceeded byte budget")
            data, identity, size = _read_document(path)
            self._total_bytes += size
            if self._total_bytes > MAX_TOTAL_BYTES:
                raise ValueError("review input inventory exceeded byte budget")
            self._documents[key] = data
            self._document_hashes[key] = identity
        return self._documents[key]

    def _capture(self, path: Path) -> dict:
        path = path.resolve(strict=True)
        key = str(path)
        if key not in self._captures:
            capture = self._document(path)
            validate_capture(capture, path)
            self._captures[key] = {**capture, "_path": key}
        return self._captures[key]

    def isa_index(self, references: list[dict]) -> dict:
        if not isinstance(references, list) or len(references) > MAX_INPUTS:
            raise ValueError("bounded ISA input inventory required")
        reports = []
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
                raise ValueError("ISA input reference shape")
            path = Path(reference["path"]).resolve(strict=True)
            data = self._document(path)
            if self._document_hashes[str(path)] != reference["sha256"]:
                raise ValueError("ISA input content changed")
            reports.append((path, data))
        return isa_index_from_reports(reports)

    def finish(self) -> None:
        for path, expected in self._document_hashes.items():
            if file_digest(Path(path)) != expected:
                raise ValueError("review input changed during publication: " + path)

    def input_records(self, receipt: dict) -> tuple[list[dict], dict]:
        inputs = receipt.get("inputs")
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= MAX_INPUTS:
            raise ValueError("bounded review input records required")
        captures, bindings, paths = [], {}, set()
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
        return captures, bindings

    def input_cohorts(self, receipt: dict) -> ReportCohorts:
        key = digest(receipt.get("inputs"))
        if key not in self._cohorts:
            captures, bindings = self.input_records(receipt)
            self._cohorts[key] = ReportCohorts(captures, bindings)
        return self._cohorts[key]

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
            "measurement_identity_sha256",
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
                self._document(Path(path))
                self._reports[path] = self._document_hashes[path]
            if self._reports[path] != entry.get("review_report_sha256"):
                blockers.append("review_report_content_changed")
            self.isa_index(entry.get("isa_input_references", []))
        except (KeyError, OSError, TypeError, ValueError):
            blockers.append("review_report_unreadable")
        inputs = receipt.get("inputs")
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= MAX_INPUTS:
            blockers.append("review_capture_inventory_invalid")
        if blockers:
            return ReviewReceiptVerdict(False, False, tuple(sorted(set(blockers))))
        try:
            cohorts = self.input_cohorts(receipt)
        except (KeyError, OSError, TypeError, ValueError):
            return ReviewReceiptVerdict(False, False, ("review_input_content_missing_or_changed",))
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
        if (
            receipt.get("measurement_identity_sha256") != measurement_identity(view)
            or row.get("measurement_identity_sha256") != measurement_identity(view)
            or row.get("median_end_to_end_us") != phase_median(view)
        ):
            blockers.append("review_process_measurement_mismatch")
        blockers.extend(cohorts.admission(view, RECEIPT_POLICY))
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
                and (
                    cohorts.numerical(other).passed
                    if cpu
                    else not cohorts.admission(other, RECEIPT_POLICY)
                )
            ]
            if not matched:
                blockers.append("review_qualified_baseline_missing:" + backend)
                continue
            if (
                view.get("backend_selected") in {"cpu", "cpu-reference", "wrap64-byte-limb"}
                or view.get("backend_selected") == backend
            ):
                continue
            policy = CPU_RECEIPT_POLICY if cpu else RECEIPT_POLICY
            timed = [
                other
                for other in matched
                if not cpu or not correctness_anchor_reference_capture(other)
            ]
            baseline = cohorts.best(timed, view, policy)
            # CPU arithmetic anchors do not implicitly provide CPU timing evidence.
            if cpu and baseline is None:
                continue
            comparison = cohorts.compare(view, baseline, policy)
            blockers.extend(backend + ":" + reason for reason in comparison["blockers"])
            if (
                comparison["qualified"]
                and comparison["speedup"] <= comparison["required_speedup_margin"]
            ):
                blockers.append(backend + ":speedup_inside_process_noise_margin")
        return ReviewReceiptVerdict(
            bound, not blockers, tuple(sorted(set(blockers))), evidence(view)
        )
