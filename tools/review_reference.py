"""Read current review sidecars without dropping their numerical verdict/bindings."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
from typing import Iterable

from benchmark_schema import validate_capture
from benchmark_sweep_lib.capture_metadata import backend_id, capture_contract_key
from review_contract import REVIEW_SCHEMA_VERSION, REVIEW_MODES, release_capture_satisfied
from json_evidence import JsonValue, load_evidence_json, same_evidence_value
from reference_verdict import SCHEMA as NUMERICAL_REFERENCE_SCHEMA, checksum_u64, reference_verdict
from report_capture_inputs import expand_report_inputs, load_report_capture
from timing_eligibility import capture_identity

ROOT = Path(__file__).resolve().parents[1]
CPU_BACKENDS = frozenset(("cpu", "cpu-reference"))
type ReviewEntry = dict[str, JsonValue]


def normalized_path(path: str | Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    text = str(value.resolve()).replace("\\", "/")
    return text.casefold() if os.name == "nt" else text


def _text(value: JsonValue) -> str | None:
    return value if isinstance(value, str) and value else None


def _boolean(value: JsonValue) -> bool | None:
    return value if type(value) is bool else None


def _strings(value: JsonValue) -> list[str] | None:
    return (
        list(value)
        if isinstance(value, list) and all(isinstance(item, str) and item for item in value)
        else None
    )


def _hex(value: JsonValue) -> str | None:
    return (
        value
        if isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
        else None
    )


class ReviewIndex:
    def __init__(self) -> None:
        self._entries: dict[str, list[ReviewEntry]] = {}

    def add(self, path: str | Path, entry: ReviewEntry) -> None:
        keys = [normalized_path(path)]
        identity = entry.get("capture_sha256")
        if isinstance(identity, str):
            keys.append("sha256:" + identity)
        for key in keys:
            records = self._entries.setdefault(key, [])
            if entry not in records:
                records.append(entry)

    def lookup(self, path: Path, capture: dict) -> ReviewEntry | None:
        identity = capture_identity(capture)
        records = self._entries.get(normalized_path(path)) or self._entries.get(
            "sha256:" + identity
        )
        if not records:
            return None
        entry = deepcopy(records[0])
        blockers = set(entry["review_reader_blockers"])

        def decision(record):
            return {
                key: value
                for key, value in record.items()
                if key not in ("review_report", "review_report_group_index")
            }

        if any(
            not same_evidence_value(decision(record), decision(records[0]))
            for record in records[1:]
        ):
            blockers.add("review_sidecar_conflict")
        if entry.get("capture_sha256") != identity:
            blockers.add("review_capture_content_mismatch")
        if entry.get("contract_key") != capture_contract_key(capture):
            blockers.add("review_comparison_contract_mismatch")
        entry["review_reader_blockers"] = sorted(blockers)
        return entry


def load_review_index(
    inputs: Iterable[Path], captures: Iterable[tuple[Path, dict]] = ()
) -> ReviewIndex:
    index = ReviewIndex()
    snapshots: dict[str, dict | None] = {}
    by_hash: dict[str, list[tuple[Path, dict]]] = {}
    for path, capture in captures:
        raw = {key: value for key, value in capture.items() if not key.startswith("_")}
        validate_capture(raw, path)
        raw["_path"] = str(path)
        snapshots[normalized_path(path)] = raw
        by_hash.setdefault(capture_identity(raw), []).append((path, raw))

    def aliases(raw: str, report: Path) -> list[Path]:
        path = Path(raw.replace("\\", "/"))
        return [path] if path.is_absolute() else [ROOT / path, report.parent / path]

    def bound(
        candidate: dict[str, JsonValue], report: Path
    ) -> tuple[dict | None, list[Path], list[str]]:
        raw, expected = _text(candidate.get("capture")), _hex(candidate.get("capture_sha256"))
        if not raw:
            return None, [], ["review_capture_path_missing"]
        paths = aliases(raw, report)
        existing = []
        for path in paths:
            key = normalized_path(path)
            if key not in snapshots:
                try:
                    snapshots[key] = (
                        load_report_capture(path, from_directory=False) if path.is_file() else None
                    )
                except (OSError, ValueError, TypeError):
                    snapshots[key] = None
            if snapshots[key] is not None:
                existing.append(snapshots[key])
        if not expected:
            return (existing[0] if existing else None), paths, ["review_capture_binding_missing"]
        matched = [capture for capture in existing if capture_identity(capture) == expected]
        if matched:
            return matched[0], paths, []
        if existing:
            return existing[0], paths, ["review_capture_content_mismatch"]
        relocated = by_hash.get(expected)
        if relocated:
            return relocated[0][1], [*paths, *(path for path, _ in relocated)], []
        return None, paths, ["review_capture_reference_unavailable"]

    report_paths = sorted(
        {
            path.resolve()
            for path, _ in expand_report_inputs(list(inputs))
            if path.name == "review_report.json"
        }
    )
    for report_path in report_paths:
        try:
            report = load_evidence_json(report_path)
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(report, dict):
            continue
        schema = report.get("schema_version")
        schema_valid = type(schema) is int and schema == REVIEW_SCHEMA_VERSION
        groups = report.get("groups")
        if not isinstance(groups, list):
            continue
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict) or not isinstance(group.get("candidates"), list):
                continue
            candidates = [
                candidate for candidate in group["candidates"] if isinstance(candidate, dict)
            ]
            bindings = [bound(candidate, report_path) for candidate in candidates]
            cpu_bindings = [
                (candidate, capture)
                for candidate, (capture, _, errors) in zip(candidates, bindings)
                if _text(candidate.get("backend")) in CPU_BACKENDS
                and capture is not None
                and not errors
                and backend_id(capture) in CPU_BACKENDS
            ]
            cpu = [capture for _, capture in cpu_bindings]
            backends = sorted(
                {
                    candidate["backend"]
                    for candidate in candidates
                    if isinstance(candidate.get("backend"), str)
                }
            )
            for candidate, (capture, paths, binding_errors) in zip(candidates, bindings):
                if not paths:
                    continue
                blockers = list(binding_errors)
                if not schema_valid:
                    blockers.append("review_schema_version_mismatch")
                if (
                    "numerical_reference_contract" in report
                    and report["numerical_reference_contract"] != NUMERICAL_REFERENCE_SCHEMA
                ):
                    blockers.append("review_numerical_contract_mismatch")
                if not cpu:
                    blockers.append("review_cpu_reference_binding_missing")
                if group.get("checksum_consistent") is not True:
                    blockers.append("review_checksum_comparison_not_passed")
                if candidate.get("checksum_matches_reference") is not True:
                    blockers.append("review_candidate_comparison_not_passed")
                reference_values = {checksum_u64(item) for item in cpu}
                if (
                    not reference_values
                    or None in reference_values
                    or len(reference_values) != 1
                    or type(group.get("checksum_reference")) is not int
                    or group["checksum_reference"] not in reference_values
                    or type(candidate.get("checksum_reference")) is not int
                    or candidate["checksum_reference"] not in reference_values
                ):
                    blockers.append("review_reference_checksum_mismatch")
                required = _strings(group.get("required_baselines"))
                missing = _strings(group.get("missing_required_baselines"))
                duplicates = _strings(group.get("duplicate_backends"))
                promotion = _strings(candidate.get("promotion_blockers"))
                if (
                    any(value is None for value in (required, missing, duplicates, promotion))
                    or _boolean(candidate.get("promotable")) is None
                ):
                    blockers.append("review_metadata_missing_or_malformed")
                if _boolean(group.get("release_review_satisfied")) is None:
                    blockers.append("review_metadata_missing_or_malformed")
                if (
                    _text(group.get("review_mode")) or _text(report.get("review_mode"))
                ) not in REVIEW_MODES:
                    blockers.append("review_mode_missing_or_unknown")
                if _text(candidate.get("cache_write_status")) is None:
                    blockers.append("review_metadata_missing_or_malformed")
                if promotion and "checksum_mismatch_vs_reference" in promotion:
                    blockers.append("review_candidate_comparison_not_passed")
                contract = _text(group.get("contract_key"))
                if capture is not None:
                    if _boolean(
                        candidate.get("release_review_capture")
                    ) != release_capture_satisfied(capture):
                        blockers.append("review_release_capture_mismatch")
                    if contract != capture_contract_key(capture):
                        blockers.append("review_comparison_contract_mismatch")
                    if candidate.get("backend") != backend_id(capture):
                        blockers.append("review_capture_backend_mismatch")
                    reported = candidate.get("checksum")
                    expected = checksum_u64(capture)
                    if type(reported) is not int or expected is None or reported != expected:
                        blockers.append("review_capture_checksum_mismatch")
                verdict = reference_verdict(capture or {}, cpu)
                if "numerical_reference" in candidate and not same_evidence_value(
                    candidate["numerical_reference"], verdict.to_json()
                ):
                    blockers.append("review_numerical_verdict_conflict")
                blockers.extend(verdict.blockers)
                entry: ReviewEntry = dict(
                    review_report=str(report_path),
                    review_schema_version=schema if type(schema) is int else None,
                    review_mode=_text(group.get("review_mode")) or _text(report.get("review_mode")),
                    review_report_group_index=group_index,
                    release_review_satisfied=_boolean(group.get("release_review_satisfied")),
                    required_baselines=required or [],
                    missing_required_baselines=missing or [],
                    duplicate_backends=duplicates or [],
                    candidate_backends=backends,
                    cpu_reference_present=bool(cpu),
                    cpu_reference_release_review_capture=any(
                        candidate.get("release_review_capture") is True
                        and release_capture_satisfied(capture)
                        for candidate, capture in cpu_bindings
                    ),
                    promotable_candidate_count=sum(
                        candidate.get("promotable") is True for candidate in candidates
                    ),
                    promotable_candidate_backends=sorted(
                        {
                            candidate["backend"]
                            for candidate in candidates
                            if candidate.get("promotable") is True
                            and isinstance(candidate.get("backend"), str)
                        }
                    ),
                    candidate_backend=_text(candidate.get("backend")),
                    candidate_promotable=candidate.get("promotable") is True,
                    candidate_cache_write_status=_text(candidate.get("cache_write_status")),
                    candidate_promotion_blockers=promotion or [],
                    candidate_promotion_reason=_text(candidate.get("promotion_reason")),
                    candidate_scenario_promotion_scope=_text(
                        candidate.get("scenario_promotion_scope")
                    ),
                    capture_sha256=_hex(candidate.get("capture_sha256")),
                    contract_key=contract,
                    numerical_reference=verdict.to_json(),
                    review_reader_blockers=sorted(set(blockers)),
                )
                for path in paths:
                    index.add(path, entry)
    return index
