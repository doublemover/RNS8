from __future__ import annotations

from typing import Any

from .core_shared import *
from .gpu_events import (
    ck_deep_gpu_event_phases,
    is_deep_accelerator_gpu_event_label,
    prefix_event_label,
    rocwmma_deep_gpu_event_phases,
    vector_gpu_event_phases,
)

class ValidatorTimingEventsMixin:
    def _validate_pack_reuse_fields(self, raw_timings: dict[str, list[float]]) -> None:
        from .reuse_timing import validate_pack_reuse_fields

        validate_pack_reuse_fields(self, raw_timings)

    def _validate_residue_current_timings(self, raw_timings: dict[str, list[float]]) -> None:
        from .reuse_timing import validate_residue_current_timings

        validate_residue_current_timings(self, raw_timings)

    def _validate_bounded_oneshot_timings(self, raw_timings: dict[str, list[float]]) -> None:
        from .reuse_timing import validate_bounded_oneshot_timings

        validate_bounded_oneshot_timings(self, raw_timings)

    def _is_all_zero_direct_hip_adaptive_capture(self) -> bool:
        from .reuse_timing import is_all_zero_direct_hip_adaptive_capture

        return is_all_zero_direct_hip_adaptive_capture(self)

    def _validate_all_zero_direct_hip_adaptive_timings(self, raw_timings: dict[str, list[float]]) -> None:
        from .reuse_timing import validate_all_zero_direct_hip_adaptive_timings

        validate_all_zero_direct_hip_adaptive_timings(self, raw_timings)

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
        if GLOBAL_BOUND_TIMING_PHASE in self._timing_phases():
            fields.insert(0, ("avg_global_bound_scan_us", GLOBAL_BOUND_TIMING_PHASE))
        fields.insert(1, ("avg_scheduling_us", "scheduling"))
        if PER_TILE_TIMING_PHASE in self._timing_phases():
            fields.insert(2, ("avg_tile_bound_scan_us", PER_TILE_TIMING_PHASE))
        for field, phase in fields:
            value = self._require(field, "number")
            values = raw_timings.get(phase)
            if _is_number(value) and values is not None and not _close(float(value), _average(values)):
                self._error(f"{field}={value} does not match raw average {_average(values)}")
        schedule_query = self._require("schedule_query_us", "number")
        scheduling_values = raw_timings.get("scheduling")
        if _is_number(schedule_query) and scheduling_values is not None and not _close(
            float(schedule_query), _average(scheduling_values)
        ):
            self._error(f"schedule_query_us={schedule_query} does not match raw average {_average(scheduling_values)}")
        if PER_TILE_TIMING_PHASE in raw_timings:
            tile_bound_scan = self._require("tile_bound_scan_us", "number")
            tile_bound_values = raw_timings.get(PER_TILE_TIMING_PHASE)
            if _is_number(tile_bound_scan) and tile_bound_values is not None and not _close(
                float(tile_bound_scan), _average(tile_bound_values)
            ):
                self._error(
                    f"tile_bound_scan_us={tile_bound_scan} does not match raw average "
                    f"{_average(tile_bound_values)}"
                )
        if GLOBAL_BOUND_TIMING_PHASE in raw_timings:
            global_bound_scan = self._require("global_bound_scan_us", "number")
            global_bound_values = raw_timings.get(GLOBAL_BOUND_TIMING_PHASE)
            if _is_number(global_bound_scan) and global_bound_values is not None and not _close(
                float(global_bound_scan), _average(global_bound_values)
            ):
                self._error(
                    f"global_bound_scan_us={global_bound_scan} does not match raw average "
                    f"{_average(global_bound_values)}"
                )
        prefix = self.data.get("selected_prefix", self.data.get("prefix"))
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

    def _gpu_event_selected_prefix_count(self) -> int:
        semantics = self.data.get("semantics")
        if semantics in {"finite_ring_u8", "finite_field_u8", "wrap_u64_mod_2_64"}:
            return 0
        schedule = self.data.get("schedule_metadata")
        if isinstance(schedule, dict):
            max_selected = schedule.get("max_selected_prefix")
            if _is_int(max_selected) and max_selected > 0:
                return int(max_selected)
        prefix = self.data.get("prefix")
        return int(prefix) if _is_int(prefix) and prefix > 0 else 0

    def _uses_rocwmma_prepacked_b_cache(self) -> bool:
        if self.data.get("prepack_reuse_strategy") == "rocwmma_reusable_b_cache":
            return True
        metadata = self.data.get("timing_metadata")
        return isinstance(metadata, dict) and metadata.get("prepack_reuse_strategy") == "rocwmma_reusable_b_cache"

    @staticmethod
    def _prefix_event_label(prefix: str, index: int, suffix: str) -> str:
        return prefix_event_label(prefix, index, suffix)

    def _ck_deep_gpu_event_phases(self, prefix_count: int, zero_output_tiles: bool) -> list[str]:
        return ck_deep_gpu_event_phases(prefix_count, zero_output_tiles)

    def _rocwmma_deep_gpu_event_phases(
        self,
        prefix_count: int,
        use_prepacked_b: bool,
        zero_output_tiles: bool,
    ) -> list[str]:
        return rocwmma_deep_gpu_event_phases(prefix_count, use_prepacked_b, zero_output_tiles)

    def _expected_vector_gpu_event_phases(self) -> list[str]:
        return vector_gpu_event_phases(self.data.get("semantics"), self.data.get("selected_kernel"))

    def _expected_accelerator_deep_gpu_event_phases(self) -> list[str] | None:
        backend = self.data.get("backend_selected")
        if backend not in {"ck", "rocwmma"} or self._is_wrap64_rocwmma_candidate():
            return None
        semantics = self.data.get("semantics")
        use_prepacked_b = backend == "rocwmma" and self._uses_rocwmma_prepacked_b_cache()
        gemm_group = "rns_gemm_prepacked_b_kernel_group" if use_prepacked_b else "rns_gemm_kernel_group"
        if semantics in {"finite_ring_u8", "finite_field_u8"}:
            phases = ["finite_pack_h2d", "finite_pack_kernel", "pack", "rns_gemm_kernel_group"]
        elif semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
            phases = ["pack_h2d", "pack_kernel", "pack", gemm_group]
        else:
            phases = ["pack_h2d", "pack_kernel", "pack", gemm_group]
        prefix_count = self._gpu_event_selected_prefix_count()
        schedule = self.data.get("schedule_metadata")
        zero_output_tiles = (
            isinstance(schedule, dict)
            and _is_int(schedule.get("zero_output_tile_count"))
            and schedule.get("zero_output_tile_count") > 0
        )
        if backend == "ck":
            phases.extend(self._ck_deep_gpu_event_phases(prefix_count, zero_output_tiles))
        else:
            phases.extend(self._rocwmma_deep_gpu_event_phases(prefix_count, use_prepacked_b, zero_output_tiles))
        phases.append("rns_gemm")
        if self._is_residue_current_chain_capture():
            return phases
        if semantics in {"finite_ring_u8", "finite_field_u8"}:
            phases.extend(["finite_export_kernel", "finite_export_d2h", "crt_export"])
        elif semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
            phases.extend(
                [
                    "exact_wide_export_status_memset",
                    "exact_wide_export_kernel",
                    "exact_wide_export_status_d2h",
                    "exact_wide_export_d2h",
                    "crt_export",
                ]
            )
        else:
            phases.extend(
                [
                    "crt_export_status_memset",
                    "crt_export_kernel",
                    "crt_export_status_d2h",
                    "crt_export_d2h",
                    "crt_export",
                ]
            )
        return phases

    @staticmethod
    def _is_deep_accelerator_gpu_event_label(phase: str) -> bool:
        return is_deep_accelerator_gpu_event_label(phase)

    def _validate_expected_gpu_event_phases(self, scope: Any, phases: list[str]) -> None:
        from .gpu_event_validation import validate_expected_gpu_event_phases

        validate_expected_gpu_event_phases(self, scope, phases)

    def _expected_status_event_labels(self) -> list[str]:
        from .gpu_event_validation import expected_status_event_labels

        return expected_status_event_labels(self)

    def _known_status_event_labels(self) -> set[str]:
        from .gpu_event_validation import known_status_event_labels

        return known_status_event_labels(self)

    def _validate_status_event_consistency(self, phases: list[str], parsed: dict[str, list[float]]) -> None:
        from .gpu_event_validation import validate_status_event_consistency

        validate_status_event_consistency(self, phases, parsed)

    def _validate_gpu_events(self) -> None:
        from .gpu_event_validation import validate_gpu_events

        validate_gpu_events(self)

    def _gpu_event_phases(self, metadata: dict[str, Any]) -> list[str]:
        from .gpu_event_validation import gpu_event_phases

        return gpu_event_phases(self, metadata)
