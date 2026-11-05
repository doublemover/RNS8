#!/usr/bin/env python3
"""Validate rns8-bench JSON capture files."""

from __future__ import annotations

from json_evidence import load_evidence_json
from first_call_setup import inventory_errors
from graph_caller_contract import caller_transaction_blockers
from pathlib import Path
from typing import Any
from build_manifest import manifest_diagnostic_shape_valid

from .core_shared import (
    SCHEMA_VERSION,
    _is_int,
)
from .validator_base_fields import ValidatorBaseFieldsMixin
from .validator_predicates import ValidatorPredicatesMixin
from .validator_schedule import ValidatorScheduleMixin
from .validator_timing_events import ValidatorTimingEventsMixin


class BenchmarkSchemaError(ValueError):
    """Raised when a benchmark capture does not match the expected schema."""


def load_capture(path: Path) -> dict[str, Any]:
    try:
        data = load_evidence_json(path)
    except (OSError, ValueError) as exc:
        raise BenchmarkSchemaError(f"{path}: failed to read benchmark JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BenchmarkSchemaError(f"{path}: benchmark JSON root must be an object")
    return data


def schema_version(data: dict[str, Any]) -> int:
    if "schema_version" not in data:
        raise BenchmarkSchemaError("missing required field schema_version")
    value = data["schema_version"]
    if not _is_int(value):
        raise BenchmarkSchemaError("schema_version must be an integer")
    return int(value)


def validation_errors(data: dict[str, Any], path: str | Path = "<memory>") -> list[str]:
    if not isinstance(data, dict):
        return [f"{path}: benchmark JSON root must be an object"]
    validator = _Validator(data, str(path))
    validator.validate()
    return validator.errors


def validate_capture(data: dict[str, Any], path: str | Path = "<memory>") -> dict[str, Any]:
    errors = validation_errors(data, path)
    if errors:
        raise BenchmarkSchemaError("\n".join(errors))
    return {"schema_version": schema_version(data)}


def validate_capture_file(path: Path) -> dict[str, Any]:
    return validate_capture(load_capture(path), path)


class _Validator(
    ValidatorPredicatesMixin,
    ValidatorBaseFieldsMixin,
    ValidatorScheduleMixin,
    ValidatorTimingEventsMixin,
):
    def __init__(self, data: dict[str, Any], path: str) -> None:
        self.data = data
        self.path = path
        self.errors: list[str] = []
        self.version = 1

    def validate(self) -> None:
        if "schema_version" not in self.data:
            self._error("missing required field schema_version")
            return
        version_value = self.data["schema_version"]
        if not _is_int(version_value):
            self._error("schema_version must be an integer")
            return
        version = int(version_value)
        if version != SCHEMA_VERSION:
            self._error(f"unsupported schema_version {version}; expected {SCHEMA_VERSION}")
            return
        self.version = version
        self._validate_v4()
        if self.errors:
            return
        graph = self.data.get("hip_graph_replay") or {}
        if isinstance(graph, dict) and graph.get("caller_transaction") is not None:
            for error in caller_transaction_blockers(self.data):
                self._error(error)
        for error in inventory_errors(self.data):
            self._error(error)
        manifest = self.data.get("build_manifest")
        if manifest is not None:
            if not isinstance(manifest, dict):
                self._error("build_manifest must be an object")
            elif (
                type(manifest.get("version")) is not int
                or manifest["version"] not in (1, 2, 3)
                or (
                    manifest["version"] >= 2
                    and not isinstance(manifest.get("external_dependencies"), dict)
                )
                or (
                    manifest["version"] == 3
                    and (
                        not isinstance(manifest.get("dependency_identity"), dict)
                        or not isinstance(manifest.get("external_files"), dict)
                        or not isinstance(manifest.get("dependency_receipts"), dict)
                    )
                )
                or type(manifest.get("source_dirty")) is not bool
                or not isinstance(manifest.get("configuration"), str)
                or not isinstance(manifest.get("source_digest"), str)
                or len(manifest["source_digest"]) != 64
                or manifest.get("source_commit") != self.data.get("git_commit")
            ):
                self._error("build_manifest has incomplete or inconsistent source identity")
            elif manifest["version"] == 3 and not manifest_diagnostic_shape_valid(manifest):
                self._error("build_manifest has inconsistent selected dependency content identity")

    def _error(self, message: str) -> None:
        self.errors.append(f"{self.path}: {message}")
