#!/usr/bin/env python3
"""Validate rns8-bench JSON capture files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core_shared import *
from .validator_base_fields import ValidatorBaseFieldsMixin
from .validator_predicates import ValidatorPredicatesMixin
from .validator_schedule import ValidatorScheduleMixin
from .validator_timing_events import ValidatorTimingEventsMixin

class BenchmarkSchemaError(ValueError):
    """Raised when a benchmark capture does not match the expected schema."""


def load_capture(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
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

    def _error(self, message: str) -> None:
        self.errors.append(f"{self.path}: {message}")
