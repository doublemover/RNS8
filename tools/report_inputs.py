"""Shared strict input handling for benchmark comparison reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


def expand_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.json")))
        else:
            expanded.append(path)
    return expanded


def load_validated_capture(path: Path) -> dict[str, Any]:
    """Reject invalid evidence before attaching the report-only source path."""
    try:
        capture = load_capture(path)
        validate_capture(capture, path)
    except BenchmarkSchemaError as exc:
        raise SystemExit(str(exc)) from exc
    capture["_path"] = str(path)
    return capture
