from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture

from .config import SKIP_JSON_NAMES

def median_phase(capture: dict[str, Any], phase: str) -> float | None:
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    item = summary.get(phase)
    if not isinstance(item, dict):
        return None
    value = item.get("median")
    return float(value) if isinstance(value, (int, float)) else None


def event_medians(capture: dict[str, Any]) -> dict[str, float]:
    summary = capture.get("gpu_event_timing_summary_us")
    if not isinstance(summary, dict):
        return {}
    result: dict[str, float] = {}
    for name, item in summary.items():
        if not isinstance(item, dict):
            continue
        value = item.get("median")
        if isinstance(value, (int, float)):
            result[str(name)] = float(value)
    return result


def normalized_capture_path(path: Path | str) -> str:
    return str(Path(path)).replace("\\", "/").lower()


def is_skipped_json_candidate(path: Path) -> bool:
    name = path.name
    return (
        name in SKIP_JSON_NAMES
        or name.endswith(".failed.json")
        or name.endswith("autotune-cache.json")
        or name.endswith("-isa-summary.json")
    )


def discover_capture_paths(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.rglob("*.json")):
                if is_skipped_json_candidate(candidate):
                    continue
                discovered.append(candidate)
        else:
            discovered.append(path)
    return list(dict.fromkeys(discovered))


def discover_isa_report_paths(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*-isa-summary.json")))
        else:
            discovered.append(path)
    return list(dict.fromkeys(discovered))


def load_validated_captures(
    paths: list[Path],
    *,
    skip_invalid: bool = False,
    invalid_captures: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    for path in discover_capture_paths(paths):
        try:
            capture = load_capture(path)
            validate_capture(capture)
        except Exception as exc:
            if not skip_invalid:
                raise
            message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            if invalid_captures is not None:
                invalid_captures.append({"capture_path": str(path), "error": message})
            print(f"warning: skipped invalid capture {path}: {message}", file=sys.stderr)
            continue
        capture["_path"] = str(path)
        captures.append(capture)
    return captures


def load_scenario_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for entry in manifest.get("entries", []):
            if not isinstance(entry, dict):
                continue
            for key in (entry.get("capture_path"), entry.get("capture_name")):
                if isinstance(key, str) and key:
                    index[normalized_capture_path(key)] = entry
                    index[Path(key).name.lower()] = entry
    return index


def load_review_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for group in report.get("groups", []):
            if not isinstance(group, dict):
                continue
            for candidate in group.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                capture = candidate.get("capture")
                if not isinstance(capture, str) or not capture:
                    continue
                payload = {
                    "review_mode": group.get("review_mode"),
                    "release_review_satisfied": group.get("release_review_satisfied"),
                    "missing_required_baselines": group.get("missing_required_baselines") or [],
                    "promotable": candidate.get("promotable"),
                    "promotion_blockers": candidate.get("promotion_blockers") or [],
                    "primary_loss_phase_vs_direct_hip": candidate.get("primary_loss_phase_vs_direct_hip"),
                    "speedup_vs_direct_hip": candidate.get("speedup_vs_direct_hip"),
                    "speedup_vs_vector_alu": candidate.get("speedup_vs_vector_alu"),
                }
                index[normalized_capture_path(capture)] = payload
                index[Path(capture).name.lower()] = payload
    return index


