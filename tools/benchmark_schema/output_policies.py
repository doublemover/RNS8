"""Output layout and export-staging policy validators."""

from __future__ import annotations

from typing import Any

from metadata_registry_constants import DIRECT_HIP_EXPORT_STAGING_POLICIES, OUTPUT_DESTINATION_LAYOUTS


def _is_int(value: Any) -> bool:
    return type(value) is int


def validate_output_layout_metadata(self, metadata: dict[str, Any]) -> None:
    present = any(key in self.data for key in ["output_logical_ld", "output_ld_padding"]) or any(
        key in metadata
        for key in [
            "benchmark_output_destination_layout",
            "benchmark_output_logical_ld",
            "benchmark_output_ld_padding",
        ]
    )
    if not present:
        return

    n = self.data.get("n")
    output_ld = self.data.get("output_logical_ld")
    padding = self.data.get("output_ld_padding")
    if not _is_int(output_ld) or output_ld <= 0:
        self._error("output_logical_ld must be a positive integer")
        return
    if not _is_int(padding) or padding < 0:
        self._error("output_ld_padding must be a nonnegative integer")
        return
    if _is_int(n) and output_ld != n + padding:
        self._error("output_logical_ld must equal n + output_ld_padding")

    layout = metadata.get("benchmark_output_destination_layout")
    if layout not in OUTPUT_DESTINATION_LAYOUTS:
        self._error(
            f"timing_metadata.benchmark_output_destination_layout must be one of {sorted(OUTPUT_DESTINATION_LAYOUTS)}"
        )
    else:
        expected_layout = "contiguous_row_major" if padding == 0 else "padded_row_major"
        if layout != expected_layout:
            self._error(f"timing_metadata.benchmark_output_destination_layout must be {expected_layout}")
    if metadata.get("benchmark_output_logical_ld") != output_ld:
        self._error("timing_metadata.benchmark_output_logical_ld must match output_logical_ld")
    if metadata.get("benchmark_output_ld_padding") != padding:
        self._error("timing_metadata.benchmark_output_ld_padding must match output_ld_padding")
    staging_policy = metadata.get("direct_hip_export_staging_policy")
    if staging_policy not in DIRECT_HIP_EXPORT_STAGING_POLICIES:
        self._error(
            "timing_metadata.direct_hip_export_staging_policy must be one of "
            f"{sorted(DIRECT_HIP_EXPORT_STAGING_POLICIES)}"
        )
