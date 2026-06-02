#!/usr/bin/env python3
"""Validate rns8-bench JSON capture files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 4
SUPPORTED_SCHEMA_VERSIONS = {2, 3, SCHEMA_VERSION}
TIMING_PHASES_V2 = ["planning", "matrix_alloc", "pack", "rns_gemm", "crt_export", "end_to_end"]
TIMING_PHASES_V3 = ["planning", "scheduling", "matrix_alloc", "pack", "rns_gemm", "crt_export", "end_to_end"]
REPEATED_TIMING_PHASES = {"pack", "rns_gemm", "crt_export", "end_to_end"}
DEFAULT_GPU_EVENT_PHASES = [
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
AGGREGATE_GPU_EVENT_PHASES = ["pack", "rns_gemm", "crt_export"]


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
        self.version = 1

    def validate(self) -> None:
        version_value = self.data.get("schema_version", 1)
        if not _is_int(version_value):
            self._error("schema_version must be an integer when present")
            return
        version = int(version_value)
        if version <= 1:
            self._validate_v1_legacy()
            return
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            expected = ", ".join(str(item) for item in sorted(SUPPORTED_SCHEMA_VERSIONS))
            self._error(f"unsupported schema_version {version}; expected one of {expected}")
            return
        self.version = version
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
        selected_kernel = self.data.get("selected_kernel")
        if selected_kernel is not None and not isinstance(selected_kernel, str):
            self._error("selected_kernel must be a string or null")
        if self.version >= 4:
            self._require("bound_mode", "str")
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
        self._validate_schedule_metadata()
        self._validate_semantic_contract()
        raw_timings = self._validate_raw_timings()
        self._validate_timing_summaries(raw_timings, "timing_summary_us", self._timing_phases())
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
            expected_phases = self._timing_phases()
            if phase_order != expected_phases:
                self._error(f"timing_metadata.phase_order must be {expected_phases}")
            if self.version >= 3 or "phase_availability" in metadata:
                self._validate_phase_availability(metadata)
            gpu_phase_order = metadata.get("gpu_event_phase_order")
            if gpu_phase_order is not None:
                if not isinstance(gpu_phase_order, list) or not all(isinstance(item, str) for item in gpu_phase_order):
                    self._error("timing_metadata.gpu_event_phase_order must be an array of strings")
                elif len(set(gpu_phase_order)) != len(gpu_phase_order):
                    self._error("timing_metadata.gpu_event_phase_order must not contain duplicates")

    def _validate_phase_availability(self, metadata: dict[str, Any]) -> None:
        availability = metadata.get("phase_availability")
        if not isinstance(availability, dict):
            self._error("timing_metadata.phase_availability must be an object")
            return
        scheduling = availability.get("scheduling")
        if not isinstance(scheduling, dict):
            self._error("timing_metadata.phase_availability.scheduling must be an object")
        else:
            if scheduling.get("timed") is not True:
                self._error("timing_metadata.phase_availability.scheduling.timed must be true")
            if scheduling.get("timing_key") != "scheduling":
                self._error("timing_metadata.phase_availability.scheduling.timing_key must be scheduling")
            if scheduling.get("scope") != "one_time_schedule_info_query":
                self._error("timing_metadata.phase_availability.scheduling.scope must be one_time_schedule_info_query")
            if not isinstance(scheduling.get("reason"), str) or not scheduling.get("reason"):
                self._error("timing_metadata.phase_availability.scheduling.reason must be a nonempty string")

        reduction = availability.get("reduction")
        if not isinstance(reduction, dict):
            self._error("timing_metadata.phase_availability.reduction must be an object")
            return
        if reduction.get("timed") is not False:
            self._error("timing_metadata.phase_availability.reduction.timed must be false")
        if reduction.get("timing_key") is not None:
            self._error("timing_metadata.phase_availability.reduction.timing_key must be null")
        expected_scope = (
            "not_applicable_wrap64_byte_limb"
            if self.data.get("semantics") == "wrap_u64_mod_2_64"
            else "fused_into_rns_gemm"
        )
        if reduction.get("scope") != expected_scope:
            self._error(f"timing_metadata.phase_availability.reduction.scope must be {expected_scope}")
        if not isinstance(reduction.get("reason"), str) or not reduction.get("reason"):
            self._error("timing_metadata.phase_availability.reduction.reason must be a nonempty string")

    def _timing_phases(self) -> list[str]:
        return TIMING_PHASES_V3 if self.version >= 3 else TIMING_PHASES_V2

    def _validate_tile_value(self, key: str, value: Any) -> None:
        if not _is_int(value):
            self._error(f"{key} must be an integer")
            return
        if value < 64 or value > 512 or (value & (value - 1)) != 0:
            self._error(f"{key} must be a power of two from 64 through 512")

    def _validate_schedule_metadata(self) -> None:
        self._validate_tile_value("tile_m", self.data.get("tile_m"))
        self._validate_tile_value("tile_n", self.data.get("tile_n"))
        schedule = self._require("schedule_metadata", "dict")
        if not isinstance(schedule, dict):
            return
        if schedule.get("source") != "rns8_get_plan_schedule_info":
            self._error("schedule_metadata.source must be rns8_get_plan_schedule_info")
        for key in [
            "tile_m",
            "tile_n",
            "tile_rows",
            "tile_cols",
            "tile_count",
            "min_required_prefix",
            "max_required_prefix",
            "min_selected_prefix",
            "max_selected_prefix",
            "prefix_group_count",
            "range_bit_length",
        ]:
            if not _is_int(schedule.get(key)):
                self._error(f"schedule_metadata.{key} must be an integer")
        for key in ["adaptive_prefix_active", "adaptive_skip_active", "adaptive_execution_applied"]:
            if not isinstance(schedule.get(key), bool):
                self._error(f"schedule_metadata.{key} must be a boolean")
        if schedule.get("tile_m") != self.data.get("tile_m"):
            self._error("schedule_metadata.tile_m must match tile_m")
        if schedule.get("tile_n") != self.data.get("tile_n"):
            self._error("schedule_metadata.tile_n must match tile_n")
        tile_rows = schedule.get("tile_rows")
        tile_cols = schedule.get("tile_cols")
        tile_count = schedule.get("tile_count")
        if _is_int(tile_rows) and _is_int(tile_cols) and _is_int(tile_count):
            if tile_rows <= 0 or tile_cols <= 0 or tile_count != tile_rows * tile_cols:
                self._error("schedule_metadata tile grid must have positive rows/cols and matching tile_count")
        min_required = schedule.get("min_required_prefix")
        max_required = schedule.get("max_required_prefix")
        min_selected = schedule.get("min_selected_prefix")
        max_selected = schedule.get("max_selected_prefix")
        if _is_int(min_required) and _is_int(max_required) and min_required > max_required:
            self._error("schedule_metadata min_required_prefix must be <= max_required_prefix")
        if _is_int(min_selected) and _is_int(max_selected) and min_selected > max_selected:
            self._error("schedule_metadata min_selected_prefix must be <= max_selected_prefix")
        if self.version < 4 and schedule.get("adaptive_execution_applied") is True:
            self._error(
                "schedule_metadata.adaptive_execution_applied must remain false until adaptive benchmark capture support is implemented"
            )

    def _validate_semantic_contract(self) -> None:
        semantics = self.data.get("semantics")
        prefix = self.data.get("prefix")
        packed_layout = self.data.get("packed_layout_version")
        schedule = self.data.get("schedule_metadata")
        bound_mode = self.data.get("bound_mode", "global")
        if self.version >= 4 and bound_mode not in {"global", "per_tile"}:
            self._error("bound_mode must be global or per_tile")
        if semantics == "wrap_u64_mod_2_64":
            if self.data.get("backend_selected") not in {"wrap64-byte-limb", "hip-direct"}:
                self._error("wrap64 captures must select wrap64-byte-limb or hip-direct backend")
            if self.version >= 4 and bound_mode != "global":
                self._error("wrap64 captures must use bound_mode=global")
            if self.data.get("bound_kind") != "none" or self.data.get("bound") != 0:
                self._error("wrap64 captures must use bound_kind=none and bound=0")
            if self.version >= 4 and self.data.get("tile_bounds_u64") is not None:
                self._error("wrap64 captures must use tile_bounds_u64=null")
            if prefix != 0:
                self._error("wrap64 captures must use prefix=0")
            if packed_layout != "byte_limb_v1":
                self._error("wrap64 captures must use packed_layout_version=byte_limb_v1")
            if self.data.get("epilogue_type") != "low64_wrap_export":
                self._error("wrap64 captures must use low64_wrap_export epilogue")
            if isinstance(schedule, dict):
                for key in ["min_required_prefix", "max_required_prefix", "min_selected_prefix", "max_selected_prefix"]:
                    if schedule.get(key) != 0:
                        self._error(f"wrap64 captures must use schedule_metadata.{key}=0")
                if schedule.get("prefix_group_count") != 0:
                    self._error("wrap64 captures must use schedule_metadata.prefix_group_count=0")
        elif semantics in {"bounded_i64", "bounded_u64"}:
            if _is_int(prefix) and prefix <= 0:
                self._error(f"{semantics} captures must use a positive prefix")
            if packed_layout is not None:
                self._error(f"{semantics} captures must use packed_layout_version=null")
            if self.data.get("epilogue_type") != "crt_export":
                self._error(f"{semantics} captures must use crt_export epilogue")
            if bound_mode == "global":
                expected_bound_kind = "global_max_abs" if semantics == "bounded_i64" else "global_max_unsigned"
                if self.data.get("bound_kind") != expected_bound_kind:
                    self._error(f"{semantics} captures must use bound_kind={expected_bound_kind}")
                if self.version >= 4 and self.data.get("tile_bounds_u64") is not None:
                    self._error(f"{semantics} global captures must use tile_bounds_u64=null")
                if isinstance(schedule, dict) and _is_int(prefix):
                    if schedule.get("min_selected_prefix") != prefix or schedule.get("max_selected_prefix") != prefix:
                        self._error(f"{semantics} captures must use fixed selected schedule prefix equal to prefix")
                    if schedule.get("prefix_group_count") != 1:
                        self._error(f"{semantics} captures must use one fixed prefix group")
                    if schedule.get("adaptive_execution_applied") is True:
                        self._error(f"{semantics} global captures must not apply adaptive execution")
            elif bound_mode == "per_tile":
                if self.version < 4:
                    self._error("per_tile bound_mode requires schema_version 4")
                expected_bound_kind = "per_tile_max_abs" if semantics == "bounded_i64" else "per_tile_max_unsigned"
                if self.data.get("bound_kind") != expected_bound_kind:
                    self._error(f"{semantics} per-tile captures must use bound_kind={expected_bound_kind}")
                if self.data.get("backend_selected") != "hip-direct":
                    self._error("per-tile adaptive captures must select hip-direct backend")
                if self.data.get("bound") != 0:
                    self._error("per-tile adaptive captures must use bound=0")
                self._validate_v4_tile_bounds(semantics, schedule)
                self._validate_v4_adaptive_schedule(prefix, schedule)
        elif isinstance(semantics, str):
            self._error(f"unsupported benchmark semantics {semantics}")

        applicable = self.data.get("per_modulus_gemm_estimate_applicable")
        if applicable is not None and not isinstance(applicable, bool):
            self._error("per_modulus_gemm_estimate_applicable must be a boolean")
        elif isinstance(applicable, bool) and _is_int(prefix):
            expected_applicable = prefix > 0 and not (semantics in {"bounded_i64", "bounded_u64"} and bound_mode == "per_tile")
            if applicable != expected_applicable:
                self._error("per_modulus_gemm_estimate_applicable must match the fixed-prefix contract")

    def _validate_v4_tile_bounds(self, semantics: Any, schedule: Any) -> None:
        tile_bounds = self.data.get("tile_bounds_u64")
        if not isinstance(tile_bounds, dict):
            self._error("tile_bounds_u64 must be an object for per-tile adaptive captures")
            return
        for key in ["source", "pattern", "order"]:
            if not isinstance(tile_bounds.get(key), str) or not tile_bounds.get(key):
                self._error(f"tile_bounds_u64.{key} must be a nonempty string")
        expected_pattern = "exact_output_tile_max_abs_v1" if semantics == "bounded_i64" else "exact_output_tile_max_unsigned_v1"
        if tile_bounds.get("source") != "exact_seeded_input_prepass":
            self._error("tile_bounds_u64.source must be exact_seeded_input_prepass")
        if tile_bounds.get("pattern") != expected_pattern:
            self._error(f"tile_bounds_u64.pattern must be {expected_pattern}")
        if tile_bounds.get("order") != "row_major_output_tiles":
            self._error("tile_bounds_u64.order must be row_major_output_tiles")
        for key in ["count", "min", "max", "hash_u64"]:
            if not _is_int(tile_bounds.get(key)):
                self._error(f"tile_bounds_u64.{key} must be an integer")
        count = tile_bounds.get("count")
        minimum = tile_bounds.get("min")
        maximum = tile_bounds.get("max")
        if _is_int(count) and count <= 0:
            self._error("tile_bounds_u64.count must be positive")
        if _is_int(minimum) and minimum < 0:
            self._error("tile_bounds_u64.min must be nonnegative")
        if _is_int(maximum) and maximum < 0:
            self._error("tile_bounds_u64.max must be nonnegative")
        if _is_int(minimum) and _is_int(maximum) and minimum > maximum:
            self._error("tile_bounds_u64.min must be <= max")
        if semantics == "bounded_i64" and _is_int(maximum) and maximum > 2**63:
            self._error("bounded_i64 tile_bounds_u64.max must be <= 2^63")
        if _is_int(tile_bounds.get("hash_u64")) and tile_bounds.get("hash_u64") < 0:
            self._error("tile_bounds_u64.hash_u64 must be nonnegative")
        if isinstance(schedule, dict) and _is_int(count) and _is_int(schedule.get("tile_count")):
            if count != schedule.get("tile_count"):
                self._error("tile_bounds_u64.count must match schedule_metadata.tile_count")

    def _validate_v4_adaptive_schedule(self, prefix: Any, schedule: Any) -> None:
        if not isinstance(schedule, dict):
            return
        if schedule.get("adaptive_execution_applied") is not True:
            self._error("per-tile adaptive captures must set schedule_metadata.adaptive_execution_applied=true")
        selected_kernel = self.data.get("selected_kernel")
        if not isinstance(selected_kernel, str) or not selected_kernel:
            self._error("per-tile adaptive captures must report selected_kernel")
        elif selected_kernel != "direct_hip_tiled_rns_gemm_v1":
            self._error("per-tile adaptive captures must use selected_kernel=direct_hip_tiled_rns_gemm_v1")
        prefix_group_count = schedule.get("prefix_group_count")
        max_selected = schedule.get("max_selected_prefix")
        min_selected = schedule.get("min_selected_prefix")
        adaptive_prefix_expected = _is_int(prefix_group_count) and prefix_group_count > 1
        if isinstance(schedule.get("adaptive_prefix_active"), bool) and schedule.get("adaptive_prefix_active") != adaptive_prefix_expected:
            self._error("schedule_metadata.adaptive_prefix_active must match prefix_group_count > 1")
        adaptive_skip_expected = _is_int(max_selected) and _is_int(prefix) and max_selected < prefix
        if isinstance(schedule.get("adaptive_skip_active"), bool) and schedule.get("adaptive_skip_active") != adaptive_skip_expected:
            self._error("schedule_metadata.adaptive_skip_active must match max_selected_prefix < prefix")
        if _is_int(prefix_group_count) and prefix_group_count <= 0:
            self._error("per-tile adaptive captures must use at least one prefix group")
        if _is_int(min_selected) and min_selected <= 0:
            self._error("schedule_metadata.min_selected_prefix must be positive for per-tile adaptive captures")
        if _is_int(max_selected) and _is_int(prefix) and max_selected > prefix:
            self._error("schedule_metadata.max_selected_prefix must be <= prefix")
        if schedule.get("adaptive_prefix_active") is not True and schedule.get("adaptive_skip_active") is not True:
            self._error("per-tile adaptive captures must apply prefix grouping or prefix skipping")
        metadata = self.data.get("timing_metadata")
        if isinstance(metadata, dict):
            if metadata.get("gpu_event_timing") is not True:
                self._error("per-tile adaptive captures must include HIP event timings")
            expected_scope = "direct_hip_bounded_adaptive_default_stream_backend_operation_groups"
            if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")

    def _validate_raw_timings(self) -> dict[str, list[float]]:
        raw = self._require("raw_timings_us", "dict")
        repeats = self.data.get("repeats")
        result: dict[str, list[float]] = {}
        if not isinstance(raw, dict) or not _is_int(repeats):
            return result
        for phase in self._timing_phases():
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
        fields = [
            ("avg_planning_us", "planning"),
            ("avg_matrix_alloc_us", "matrix_alloc"),
            ("avg_pack_us", "pack"),
            ("avg_rns_gemm_us", "rns_gemm"),
            ("avg_crt_export_us", "crt_export"),
            ("avg_end_to_end_us", "end_to_end"),
        ]
        if self.version >= 3:
            fields.insert(1, ("avg_scheduling_us", "scheduling"))
        for field, phase in fields:
            value = self._require(field, "number")
            values = raw_timings.get(phase)
            if _is_number(value) and values is not None and not _close(float(value), _average(values)):
                self._error(f"{field}={value} does not match raw average {_average(values)}")
        if self.version >= 3:
            schedule_query = self._require("schedule_query_us", "number")
            scheduling_values = raw_timings.get("scheduling")
            if _is_number(schedule_query) and scheduling_values is not None and not _close(
                float(schedule_query), _average(scheduling_values)
            ):
                self._error(
                    f"schedule_query_us={schedule_query} does not match raw average {_average(scheduling_values)}"
                )
        prefix = self.data.get("prefix")
        applicable = self.data.get("per_modulus_gemm_estimate_applicable")
        per_modulus = self._require("avg_per_modulus_gemm_estimate_us", "number")
        gemm_values = raw_timings.get("rns_gemm")
        if (
            _is_number(per_modulus)
            and _is_int(prefix)
            and gemm_values is not None
            and applicable is not False
        ):
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
        phases = self._gpu_event_phases(metadata, timings)
        parsed: dict[str, list[float]] = {}
        for phase in phases:
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
        self._validate_timing_summaries(parsed, "gpu_event_timing_summary_us", phases)

    def _gpu_event_phases(self, metadata: dict[str, Any], timings: dict[str, Any]) -> list[str]:
        phase_order = metadata.get("gpu_event_phase_order")
        if isinstance(phase_order, list) and all(isinstance(item, str) for item in phase_order):
            phases = list(phase_order)
        else:
            phases = list(DEFAULT_GPU_EVENT_PHASES)
        for phase in AGGREGATE_GPU_EVENT_PHASES:
            if phase in timings and phase not in phases:
                phases.append(phase)
        return phases


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
