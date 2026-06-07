#!/usr/bin/env python3
"""Shared benchmark-capture input handling for report tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


def looks_like_benchmark_capture(data: dict[str, Any]) -> bool:
    return data.get("schema_version") == 4 and isinstance(data.get("benchmark"), str)


def expand_report_inputs(paths: list[Path]) -> list[tuple[Path, bool]]:
    expanded: list[tuple[Path, bool]] = []
    for path in paths:
        if path.is_dir():
            expanded.extend((item, True) for item in sorted(path.rglob("*.json")))
        else:
            expanded.append((path, False))
    return expanded


def load_report_capture(path: Path, *, from_directory: bool) -> dict[str, Any] | None:
    try:
        capture = load_capture(path)
    except BenchmarkSchemaError:
        if from_directory:
            return None
        raise
    if from_directory and not looks_like_benchmark_capture(capture):
        return None
    validate_capture(capture, path)
    capture["_path"] = str(path)
    return capture


def load_report_captures(paths: list[Path]) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    for path, from_directory in expand_report_inputs(paths):
        capture = load_report_capture(path, from_directory=from_directory)
        if capture is not None:
            captures.append(capture)
    return captures


def load_report_capture_paths(paths: list[Path]) -> list[Path]:
    capture_paths: list[Path] = []
    for path, from_directory in expand_report_inputs(paths):
        capture = load_report_capture(path, from_directory=from_directory)
        if capture is not None:
            capture_paths.append(path)
    return capture_paths
