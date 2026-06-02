#!/usr/bin/env python3
"""Validate rns8-bench JSON capture files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
TIMING_PHASES = ["planning", "matrix_alloc", "pack", "rns_gemm", "crt_export", "end_to_end"]
REPEATED_TIMING_PHASES = {"pack", "rns_gemm", "crt_export", "end_to_end"}
GPU_EVENT_PHASES = [
    "pack_h2d",
    "pack_kernel",
    "pack",
    "rns_gemm_kernel_group",
    "rns_gemm",
    "crt_export_status_memset",
    "crt_export_kernel",
    "crt_export_status_d2h",
    "crt_export_d2h",
    "crt_export",
]


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
    value = data.get("schema_version", 1)
    if not _is_int(value):
        raise BenchmarkSchemaError("schema_version must be an integer when present")
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


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(p * float(len(ordered) - 1) + 0.999999)
    return float(ordered[index])


def _average(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1.0e-5, abs_tol=1.0e-3)


class _Validator:
    def __init__(self, data: dict[str, Any], path: str) -> None:
        self.data = data
        self.path = path
        self.errors: list[str] = []

    def validate(self) -> None:
        version_value = self.data.get("schema_version", 1)
        if not _is_int(version_value):
            self._error("schema_version must be an integer when present")
            return
        version = int(version_value)
        if version <= 1:
            self._validate_v1_legacy()
            return
        if version != SCHEMA_VERSION:
            self._error(f"unsupported schema_version {version}; expected {SCHEMA_VERSION}")
            return
        self._validate_v2()

    def _error(self, message: str) -> None:
        self.errors.append(f"{self.path}: {message}")

    def _require(self, key: str, kind: str) -> Any:
        if key not in self.data:
            self._error(f"missing required field {key}")
            return None
        value = self.data[key]
        if kind == "str" and not isinstance(value, str):
            self._error(f"{key} must be a string")
        elif kind == "int" and not _is_int(value):
            self._error(f"{key} must be an integer")
        elif kind == "number" and not _is_number(value):
            self._error(f"{key} must be a finite number")
        elif kind == "bool" and not isinstance(value, bool):
            self._error(f"{key} must be a boolean")
        elif kind == "dict" and not isinstance(value, dict):
            self._error(f"{key} must be an object")
        return value

    def _validate_v1_legacy(self) -> None:
        for key in ["benchmark", "backend_requested", "backend_selected", "semantics", "m", "n", "k", "repeats"]:
            if key not in self.data:
                self._error(f"legacy v1 capture missing {key}")
        for phase, keys in {
            "planning": ["avg_planning_us", "plan_us"],
            "matrix_alloc": ["avg_matrix_alloc_us", "matrix_alloc_us"],
            "pack": ["avg_pack_us"],
            "rns_gemm": ["avg_rns_gemm_us"],
            "per_modulus_gemm_estimate": ["avg_per_modulus_gemm_estimate_us"],
            "crt_export": ["avg_crt_export_us"],
            "end_to_end": ["avg_end_to_end_us"],
        }.items():
            if not any(_is_number(self.data.get(key)) for key in keys):
                self._error(f"legacy v1 capture missing numeric timing for {phase}")

    def _validate_v2(self) -> None:
        for key in [
            "benchmark",
            "backend_requested",
            "backend_selected",
            "semantics",
            "bound_kind",
            "epilogue_type",
            "input_distribution",
            "command_line",
            "git_commit",
            "configured_amdgpu_targets",
            "timing_source",
            "timing_note",
        ]:
            self._require(key, "str")
        for key in [
            "bound",
            "m",
            "n",
            "k",
            "prefix",
            "tile_m",
            "tile_n",
            "k_block_size",
            "seed",
            "warmups",
            "repeats",
            "checksum_u64",
        ]:
            self._require(key, "int")
        self._validate_nonnegative_ints()
        self._validate_nested_metadata()
        self._validate_semantic_contract()
        raw_timings = self._validate_raw_timings()
        self._validate_timing_summaries(raw_timings, "timing_summary_us", TIMING_PHASES)
        self._validate_top_level_averages(raw_timings)
        self._validate_gpu_events()

    def _validate_nonnegative_ints(self) -> None:
        for key in ["bound", "prefix", "tile_m", "tile_n", "k_block_size", "seed", "warmups", "repeats", "checksum_u64"]:
            value = self.data.get(key)
            if _is_int(value) and value < 0:
                self._error(f"{key} must be nonnegative")
        for key in ["m", "n", "k"]:
            value = self.data.get(key)
            if _is_int(value) and value <= 0:
                self._error(f"{key} must be positive")
        repeats = self.data.get("repeats")
        if _is_int(repeats) and repeats <= 0:
            self._error("repeats must be positive")

    def _validate_nested_metadata(self) -> None:
        compiler = self._require("compiler", "dict")
        if isinstance(compiler, dict):
            for key in ["id", "version"]:
                if not isinstance(compiler.get(key), str):
                    self._error(f"compiler.{key} must be a string")
        device = self._require("device", "dict")
        if isinstance(device, dict):
            for key in ["device_id", "hip_available", "hip_runtime_version", "hip_driver_version", "global_mem_bytes"]:
                if not _is_int(device.get(key)):
                    self._error(f"device.{key} must be an integer")
            for key in ["name", "gcn_arch"]:
                if not isinstance(device.get(key), str):
                    self._error(f"device.{key} must be a string")
        metadata = self._require("timing_metadata", "dict")
        if isinstance(metadata, dict):
            if metadata.get("unit") != "microseconds":
                self._error("timing_metadata.unit must be microseconds")
            for key in ["source", "source_scope", "gpu_event_timing_reason"]:
                if not isinstance(metadata.get(key), str):
                    self._error(f"timing_metadata.{key} must be a string")
            if "gpu_event_timing_status" in metadata and not isinstance(metadata.get("gpu_event_timing_status"), str):
                self._error("timing_metadata.gpu_event_timing_status must be a string")
            if not isinstance(metadata.get("gpu_event_timing"), bool):
                self._error("timing_metadata.gpu_event_timing must be a boolean")
            phase_order = metadata.get("phase_order")
            if phase_order != TIMING_PHASES:
                self._error(f"timing_metadata.phase_order must be {TIMING_PHASES}")

    def _validate_semantic_contract(self) -> None:
        semantics = self.data.get("semantics")
        prefix = self.data.get("prefix")
        packed_layout = self.data.get("packed_layout_version")
        if semantics == "wrap_u64_mod_2_64":
            if self.data.get("backend_selected") != "wrap64-byte-limb":
                self._error("wrap64 captures must select wrap64-byte-limb backend")
            if self.data.get("bound_kind") != "none" or self.data.get("bound") != 0:
                self._error("wrap64 captures must use bound_kind=none and bound=0")
            if prefix != 0:
                self._error("wrap64 captures must use prefix=0")
            if packed_layout != "byte_limb_v1":
                self._error("wrap64 captures must use packed_layout_version=byte_limb_v1")
            if self.data.get("epilogue_type") != "low64_wrap_export":
                self._error("wrap64 captures must use low64_wrap_export epilogue")
        elif semantics in {"bounded_i64", "bounded_u64"}:
            expected_bound_kind = "global_max_abs" if semantics == "bounded_i64" else "global_max_unsigned"
            if self.data.get("bound_kind") != expected_bound_kind:
                self._error(f"{semantics} captures must use bound_kind={expected_bound_kind}")
            if _is_int(prefix) and prefix <= 0:
                self._error(f"{semantics} captures must use a positive prefix")
            if packed_layout is not None:
                self._error(f"{semantics} captures must use packed_layout_version=null")
            if self.data.get("epilogue_type") != "crt_export":
                self._error(f"{semantics} captures must use crt_export epilogue")
        elif isinstance(semantics, str):
            self._error(f"unsupported benchmark semantics {semantics}")

        applicable = self.data.get("per_modulus_gemm_estimate_applicable")
        if applicable is not None and not isinstance(applicable, bool):
            self._error("per_modulus_gemm_estimate_applicable must be a boolean")
        elif isinstance(applicable, bool) and _is_int(prefix) and applicable != (prefix > 0):
            self._error("per_modulus_gemm_estimate_applicable must match prefix > 0")

    def _validate_raw_timings(self) -> dict[str, list[float]]:
        raw = self._require("raw_timings_us", "dict")
        repeats = self.data.get("repeats")
        result: dict[str, list[float]] = {}
        if not isinstance(raw, dict) or not _is_int(repeats):
            return result
        for phase in TIMING_PHASES:
            values = raw.get(phase)
            if not isinstance(values, list):
                self._error(f"raw_timings_us.{phase} must be an array")
                continue
            expected_length = repeats if phase in REPEATED_TIMING_PHASES else 1
            if len(values) != expected_length:
                self._error(f"raw_timings_us.{phase} length {len(values)} does not match expected {expected_length}")
            parsed: list[float] = []
            for index, value in enumerate(values):
                if not _is_int(value) or value < 0:
                    self._error(f"raw_timings_us.{phase}[{index}] must be a nonnegative integer")
                else:
                    parsed.append(float(value))
            result[phase] = parsed
        return result

    def _validate_timing_summaries(
        self,
        raw_values: dict[str, list[float]],
        summary_key: str,
        phases: list[str],
    ) -> None:
        summary = self._require(summary_key, "dict")
        if not isinstance(summary, dict):
            return
        for phase in phases:
            item = summary.get(phase)
            if not isinstance(item, dict):
                self._error(f"{summary_key}.{phase} must be an object")
                continue
            for key in ["avg", "median", "p95"]:
                if not _is_number(item.get(key)):
                    self._error(f"{summary_key}.{phase}.{key} must be a finite number")
            values = raw_values.get(phase)
            if values is None:
                continue
            expected = {
                "avg": _average(values),
                "median": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
            for key, expected_value in expected.items():
                actual = item.get(key)
                if _is_number(actual) and not _close(float(actual), expected_value):
                    self._error(
                        f"{summary_key}.{phase}.{key}={actual} does not match raw {key} {expected_value}"
                    )

    def _validate_top_level_averages(self, raw_timings: dict[str, list[float]]) -> None:
        for field, phase in [
            ("avg_planning_us", "planning"),
            ("avg_matrix_alloc_us", "matrix_alloc"),
            ("avg_pack_us", "pack"),
            ("avg_rns_gemm_us", "rns_gemm"),
            ("avg_crt_export_us", "crt_export"),
            ("avg_end_to_end_us", "end_to_end"),
        ]:
            value = self._require(field, "number")
            values = raw_timings.get(phase)
            if _is_number(value) and values is not None and not _close(float(value), _average(values)):
                self._error(f"{field}={value} does not match raw average {_average(values)}")
        prefix = self.data.get("prefix")
        per_modulus = self._require("avg_per_modulus_gemm_estimate_us", "number")
        gemm_values = raw_timings.get("rns_gemm")
        if _is_number(per_modulus) and _is_int(prefix) and gemm_values is not None:
            expected = _average(gemm_values) / float(prefix) if prefix > 0 else _average(gemm_values)
            if not _close(float(per_modulus), expected):
                self._error(f"avg_per_modulus_gemm_estimate_us={per_modulus} does not match expected {expected}")

    def _validate_gpu_events(self) -> None:
        metadata = self.data.get("timing_metadata")
        if not isinstance(metadata, dict):
            return
        enabled = metadata.get("gpu_event_timing")
        repeats = self.data.get("repeats")
        if not isinstance(enabled, bool) or not _is_int(repeats):
            return
        timings = self.data.get("gpu_event_timings_us")
        summary = self.data.get("gpu_event_timing_summary_us")
        if not enabled:
            if timings is not None:
                self._error("gpu_event_timings_us must be null when gpu_event_timing is false")
            if summary is not None:
                self._error("gpu_event_timing_summary_us must be null when gpu_event_timing is false")
            if metadata.get("gpu_event_timing_source") is not None:
                self._error("timing_metadata.gpu_event_timing_source must be null when events are unavailable")
            if metadata.get("gpu_event_timing_source_scope") is not None:
                self._error("timing_metadata.gpu_event_timing_source_scope must be null when events are unavailable")
            return
        if not isinstance(metadata.get("gpu_event_timing_source"), str):
            self._error("timing_metadata.gpu_event_timing_source must be a string when events are available")
        if not isinstance(metadata.get("gpu_event_timing_source_scope"), str):
            self._error("timing_metadata.gpu_event_timing_source_scope must be a string when events are available")
        if not isinstance(timings, dict):
            self._error("gpu_event_timings_us must be an object when gpu_event_timing is true")
            return
        parsed: dict[str, list[float]] = {}
        for phase in GPU_EVENT_PHASES:
            values = timings.get(phase)
            if not isinstance(values, list):
                self._error(f"gpu_event_timings_us.{phase} must be an array")
                continue
            if len(values) != repeats:
                self._error(f"gpu_event_timings_us.{phase} length {len(values)} does not match repeats {repeats}")
            parsed_values: list[float] = []
            for index, value in enumerate(values):
                if not _is_number(value) or float(value) < 0.0:
                    self._error(f"gpu_event_timings_us.{phase}[{index}] must be a nonnegative finite number")
                else:
                    parsed_values.append(float(value))
            parsed[phase] = parsed_values
        self._validate_timing_summaries(parsed, "gpu_event_timing_summary_us", GPU_EVENT_PHASES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+", help="benchmark JSON capture files")
    parser.add_argument("--json", action="store_true", help="emit machine-readable validation result")
    args = parser.parse_args()

    results = []
    all_errors: list[str] = []
    for path in args.captures:
        try:
            data = load_capture(path)
            result = validate_capture(data, path)
            results.append({"path": str(path), "valid": True, **result})
        except BenchmarkSchemaError as exc:
            messages = str(exc).splitlines()
            all_errors.extend(messages)
            results.append({"path": str(path), "valid": False, "errors": messages})

    if args.json:
        print(json.dumps({"valid": not all_errors, "captures": results}, indent=2, sort_keys=True))
    else:
        for item in results:
            if item["valid"]:
                print(f"{item['path']}: valid schema v{item['schema_version']}")
            else:
                for message in item["errors"]:
                    print(message)
    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
