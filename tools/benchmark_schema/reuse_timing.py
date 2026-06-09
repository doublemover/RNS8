"""Focused validators split out of benchmark_schema.core."""

from __future__ import annotations

from typing import Any

from .core import (
    PACK_MODES,
    PACK_MODE_OPERANDS,
    PREPACK_REUSE_STRATEGIES,
    REPEATED_TIMING_PHASES,
    _close,
    _is_int,
    _is_number,
)

PREPACK_SETUP_BREAKDOWN_KEYS = ("pack_a", "pack_b", "runtime_cache", "unclassified")


def _validate_prepack_setup_breakdown(self: Any, value: Any, *, field: str, setup_cost: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        self._error(f"{field} must be an object or null")
        return None
    breakdown: dict[str, float] = {}
    for key in PREPACK_SETUP_BREAKDOWN_KEYS:
        item = value.get(key)
        if not _is_number(item) or item < 0:
            self._error(f"{field}.{key} must be a nonnegative number")
            return None
        breakdown[key] = float(item)
    extra = sorted(str(key) for key in value if key not in PREPACK_SETUP_BREAKDOWN_KEYS)
    if extra:
        self._error(f"{field} has unknown keys: {extra}")
    if _is_number(setup_cost):
        total = sum(breakdown.values())
        if not _close(total, float(setup_cost)):
            self._error(f"{field} must sum to prepack setup cost")
    return breakdown


def validate_pack_reuse_fields(self, raw_timings: dict[str, list[float]]) -> None:
    reuse_value = self.data.get("reuse_packed_inputs", False)
    if "reuse_packed_inputs" in self.data and not isinstance(reuse_value, bool):
        self._error("reuse_packed_inputs must be a boolean")
    reuse_packed = reuse_value is True

    pack_mode = self.data.get("pack_mode")
    if pack_mode is not None:
        if pack_mode not in PACK_MODES:
            self._error(f"pack_mode must be one of {sorted(PACK_MODES)}")
        elif reuse_packed and pack_mode == "per_repeat_repack":
            self._error("pack_mode must describe a prepacked reuse mode")
        elif not reuse_packed and pack_mode != "per_repeat_repack":
            self._error("pack_mode must be per_repeat_repack")

    metadata = self.data.get("timing_metadata")
    if isinstance(metadata, dict):
        metadata_mode = metadata.get("pack_mode")
        if metadata_mode is not None:
            if metadata_mode not in PACK_MODES:
                self._error(f"timing_metadata.pack_mode must be one of {sorted(PACK_MODES)}")
            elif reuse_packed and metadata_mode == "per_repeat_repack":
                self._error("timing_metadata.pack_mode must describe a prepacked reuse mode")
            elif not reuse_packed and metadata_mode != "per_repeat_repack":
                self._error("timing_metadata.pack_mode must be per_repeat_repack")
            if pack_mode is not None and metadata_mode != pack_mode:
                self._error("timing_metadata.pack_mode must match pack_mode")
        metadata_operands = metadata.get("prepack_reuse_operands")
        if metadata_operands is not None and isinstance(pack_mode, str) and pack_mode in PACK_MODE_OPERANDS:
            if metadata_operands != PACK_MODE_OPERANDS[pack_mode]:
                self._error("timing_metadata.prepack_reuse_operands must match pack_mode")
        metadata_strategy = metadata.get("prepack_reuse_strategy")
        if metadata_strategy is not None:
            if metadata_strategy not in PREPACK_REUSE_STRATEGIES:
                self._error(
                    f"timing_metadata.prepack_reuse_strategy must be one of {sorted(PREPACK_REUSE_STRATEGIES)}"
                )

    operands = self.data.get("prepack_reuse_operands")
    if operands is not None and isinstance(pack_mode, str) and pack_mode in PACK_MODE_OPERANDS:
        if operands != PACK_MODE_OPERANDS[pack_mode]:
            self._error("prepack_reuse_operands must match pack_mode")
    strategy = self.data.get("prepack_reuse_strategy")
    if strategy is not None:
        if strategy not in PREPACK_REUSE_STRATEGIES:
            self._error(f"prepack_reuse_strategy must be one of {sorted(PREPACK_REUSE_STRATEGIES)}")
        metadata = self.data.get("timing_metadata")
        if isinstance(metadata, dict) and metadata.get("prepack_reuse_strategy") is not None:
            if metadata.get("prepack_reuse_strategy") != strategy:
                self._error("timing_metadata.prepack_reuse_strategy must match prepack_reuse_strategy")
        if reuse_packed and strategy == "none":
            self._error("prepacked reuse captures must not use prepack_reuse_strategy=none")
        if not reuse_packed and strategy != "none":
            self._error("per-repeat repack captures must use prepack_reuse_strategy=none")
        if strategy == "rocwmma_reusable_b_cache":
            if pack_mode != "prepacked_reuse_b":
                self._error("rocwmma_reusable_b_cache captures must use pack_mode=prepacked_reuse_b")
            if operands is not None and operands != ["B"]:
                self._error("rocwmma_reusable_b_cache captures must reuse only operand B")
            if self.data.get("backend_selected") != "rocwmma":
                self._error("rocwmma_reusable_b_cache captures must select backend_selected=rocwmma")

    prepack_setup = self.data.get("prepack_setup_us")
    avg_prepack_setup = self.data.get("avg_prepack_setup_us")
    prepack_setup_breakdown = self.data.get("prepack_setup_breakdown_us")
    if reuse_packed:
        if not _is_int(prepack_setup) or prepack_setup < 0:
            self._error("prepacked reuse captures must include nonnegative integer prepack_setup_us")
        if not _is_number(avg_prepack_setup):
            self._error("prepacked reuse captures must include avg_prepack_setup_us")
        elif _is_int(prepack_setup) and not _close(float(avg_prepack_setup), float(prepack_setup)):
            self._error("avg_prepack_setup_us must match prepack_setup_us")
        breakdown = (
            _validate_prepack_setup_breakdown(
                self,
                prepack_setup_breakdown,
                field="prepack_setup_breakdown_us",
                setup_cost=prepack_setup,
            )
            if prepack_setup_breakdown is not None
            else None
        )
        if breakdown is not None and pack_mode == "prepacked_reuse_a" and breakdown["pack_b"] != 0.0:
            self._error("prepacked_reuse_a captures must report prepack_setup_breakdown_us.pack_b=0")
        elif breakdown is not None and pack_mode == "prepacked_reuse_b" and breakdown["pack_a"] != 0.0:
            self._error("prepacked_reuse_b captures must report prepack_setup_breakdown_us.pack_a=0")
        if pack_mode == "prepacked_reuse":
            pack_values = raw_timings.get("pack")
            if pack_values is not None and any(value != 0.0 for value in pack_values):
                self._error("prepacked reuse captures must report raw_timings_us.pack as zero-valued repeats")
            event_timings = self.data.get("gpu_event_timings_us")
            if isinstance(event_timings, dict):
                for phase in ["pack_h2d", "pack_kernel", "finite_pack_h2d", "finite_pack_kernel", "pack"]:
                    values = event_timings.get(phase)
                    if isinstance(values, list) and any(_is_number(value) and float(value) != 0.0 for value in values):
                        self._error(f"prepacked reuse captures must report gpu_event_timings_us.{phase} as zero")
    else:
        if "prepack_setup_us" in self.data and prepack_setup is not None:
            self._error("per-repeat repack captures must use prepack_setup_us=null")
        if "avg_prepack_setup_us" in self.data and avg_prepack_setup is not None:
            self._error("per-repeat repack captures must use avg_prepack_setup_us=null")
        if "prepack_setup_breakdown_us" in self.data and prepack_setup_breakdown is not None:
            self._error("per-repeat repack captures must use prepack_setup_breakdown_us=null")

def validate_residue_current_timings(self, raw_timings: dict[str, list[float]]) -> None:
    if not self._is_residue_current_chain_capture():
        return
    values = raw_timings.get("crt_export")
    if not isinstance(values, list) or any(value != 0.0 for value in values):
        self._error("residue-current chain captures must report raw_timings_us.crt_export as zero-valued repeats")
    avg_export = self.data.get("avg_crt_export_us")
    if _is_number(avg_export) and float(avg_export) != 0.0:
        self._error("residue-current chain captures must report avg_crt_export_us=0")

def validate_bounded_oneshot_timings(self, raw_timings: dict[str, list[float]]) -> None:
    if not self._is_public_oneshot_capture():
        return
    for phase, field in [("pack", "avg_pack_us"), ("crt_export", "avg_crt_export_us"), ("matrix_alloc", "avg_matrix_alloc_us")]:
        values = raw_timings.get(phase)
        if not isinstance(values, list) or any(value != 0.0 for value in values):
            self._error(f"public one-shot captures must report raw_timings_us.{phase} as zero-valued")
        average_value = self.data.get(field)
        if _is_number(average_value) and float(average_value) != 0.0:
            self._error(f"public one-shot captures must report {field}=0")
    gemm_values = raw_timings.get("rns_gemm")
    e2e_values = raw_timings.get("end_to_end")
    if isinstance(gemm_values, list) and isinstance(e2e_values, list) and gemm_values != e2e_values:
        self._error("public one-shot captures must report raw_timings_us.rns_gemm equal to end_to_end")

def is_all_zero_direct_hip_adaptive_capture(self) -> bool:
    schedule = self.data.get("schedule_metadata")
    if not isinstance(schedule, dict):
        return False
    tile_count = schedule.get("tile_count")
    zero_count = schedule.get("zero_output_tile_count")
    return (
        self.data.get("backend_selected") == "hip-direct"
        and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
        and schedule.get("adaptive_execution_applied") is True
        and _is_int(tile_count)
        and tile_count > 0
        and _is_int(zero_count)
        and zero_count == tile_count
    )

def validate_all_zero_direct_hip_adaptive_timings(self, raw_timings: dict[str, list[float]]) -> None:
    if not self._is_all_zero_direct_hip_adaptive_capture():
        return
    values = raw_timings.get("pack")
    if not isinstance(values, list) or any(value != 0.0 for value in values):
        self._error(
            "all-zero direct-HIP adaptive captures must report raw_timings_us.pack as zero-valued repeats"
        )
    avg_pack = self.data.get("avg_pack_us")
    if _is_number(avg_pack) and float(avg_pack) != 0.0:
        self._error("all-zero direct-HIP adaptive captures must report avg_pack_us=0")
