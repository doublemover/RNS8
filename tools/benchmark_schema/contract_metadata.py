"""Contract and variant metadata validators for benchmark schema captures."""

from __future__ import annotations

import math
from typing import Any

from metadata_registry_constants import (
    GRAPH_REPLAY_STATUSES,
    GROUPED_DISPATCH_EXECUTION_STRATEGIES,
    GROUPED_DISPATCH_STATUSES,
    GROUPED_TASK_BUCKET_POLICIES,
    GROUPED_TASK_CHECKSUM_POLICIES,
    GROUPED_TASK_DESCRIPTOR_REUSE_POLICIES,
    GROUPED_TASK_DESCRIPTOR_LAYOUTS,
    GROUPED_TASK_DEVICE_DESCRIPTOR_POLICIES,
    GROUPED_TASK_LIFETIME_POLICIES,
    GROUPED_TASK_MATRIX_OWNERSHIP_POLICIES,
    GROUPED_TASK_OUTPUT_CURRENTNESS_POLICIES,
    GROUPED_TASK_SOURCE_VERSION_POLICIES,
    GROUPED_TASK_STATUS_POLICIES,
    GROUPED_TASK_STRIDE_POLICIES,
    GROUPED_TASK_WORKSPACE_POLICIES,
    OUTPUT_CONTRACT_DOMAINS,
    RELEASE_GATE_REVIEW_STATUSES,
    REUSE_OPERAND_ROLES,
    STATUS_HANDLING,
    STREAMING_OVERLAP_STATUSES,
    WORKLOAD_PROXY_FAMILIES,
)

EXPORT_OUTPUT_LAYOUTS = {
    "unknown",
    "scalar_i64",
    "scalar_u64",
    "finite_u8",
    "fixed_u64_limbs",
}

EXPORT_SELECTOR_STATUS_POLICIES = {
    "unknown",
    "none",
    "range_checked_status_buffer",
}

EXPORT_D2H_POLICIES = {
    "unknown",
    "host_ld_padded",
    "compact_contiguous",
    "device_residue_current",
}

EXPORT_FINAL_OUTPUT_MODES = {
    "final_host_output",
    "residue_chain_no_final_export",
    "chain_internal_residue_output",
}

REVIEWABLE_EXPORT_VARIANTS = {
    "default",
    "exact-wide-fixed-limb-export",
}

ERROR_DETECTION_MODES = {
    "not_requested",
    "deterministic_error_check",
    "probabilistic_product_check",
    "redundant_residue_check",
    "certificate_check",
}

ERROR_DETECTION_FINAL_STATUSES = {
    "checksum_recorded_reference_required",
    "reference_required",
    "exact_cpu_reference_compared",
    "passed",
}

CPU_SELECTOR_ROLES = {
    "not_requested",
    "cpu_baseline",
    "comparison_candidate",
    "unsupported_accelerator",
}

RESIDENT_REDESIGN_DIMENSIONS = {
    "data_layout",
    "tile_shape",
    "export_interaction",
    "schedule_upload",
    "workspace_reuse",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


PREPACK_SETUP_BREAKDOWN_KEYS = ("pack_a", "pack_b", "runtime_cache", "unclassified")


def _validate_setup_breakdown(
    self: Any,
    value: Any,
    *,
    field: str,
    setup_cost: Any,
) -> dict[str, float] | None:
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
        if abs(total - float(setup_cost)) > 1.0e-6:
            self._error(f"{field} must sum to reuse_contract.setup_cost_us")
    return breakdown


def _prepack_cache_key_fields(cache_key: str) -> dict[str, str] | None:
    if not isinstance(cache_key, str) or not cache_key.startswith("prepack-v2;"):
        return None
    fields: dict[str, str] = {}
    for part in cache_key.split(";")[1:]:
        if "=" not in part:
            return None
        key, value = part.split("=", 1)
        if not key or key in fields:
            return None
        fields[key] = value
    return fields


def validate_contract_metadata(self: Any) -> None:
    reuse = self.data.get("reuse_contract")
    if reuse is not None:
        if not isinstance(reuse, dict):
            self._error("reuse_contract must be an object")
        else:
            if not isinstance(reuse.get("enabled"), bool):
                self._error("reuse_contract.enabled must be a boolean")
            if reuse.get("operand_role") not in REUSE_OPERAND_ROLES:
                self._error(f"reuse_contract.operand_role must be one of {sorted(REUSE_OPERAND_ROLES)}")
            for key in [
                "source_version_inputs",
                "setup_scope",
                "output_domain",
                "next_op",
                "target_fingerprint",
                "backend_fingerprint",
                "workspace_fingerprint",
            ]:
                if not isinstance(reuse.get(key), str):
                    self._error(f"reuse_contract.{key} must be a string")
            if reuse.get("output_domain") not in OUTPUT_CONTRACT_DOMAINS:
                self._error(f"reuse_contract.output_domain must be one of {sorted(OUTPUT_CONTRACT_DOMAINS)}")
            setup_cost = reuse.get("setup_cost_us")
            if setup_cost is not None and (not _is_number(setup_cost) or setup_cost < 0):
                self._error("reuse_contract.setup_cost_us must be a nonnegative number or null")
            setup_breakdown = _validate_setup_breakdown(
                self,
                reuse.get("setup_breakdown_us"),
                field="reuse_contract.setup_breakdown_us",
                setup_cost=setup_cost,
            )
            top_level_breakdown = self.data.get("prepack_setup_breakdown_us")
            if setup_breakdown is not None and top_level_breakdown is not None and setup_breakdown != _validate_setup_breakdown(
                self,
                top_level_breakdown,
                field="prepack_setup_breakdown_us",
                setup_cost=setup_cost,
            ):
                self._error("reuse_contract.setup_breakdown_us must match prepack_setup_breakdown_us")
            for key in [
                "setup_amortized_us",
                "repeat_median_end_to_end_us",
                "setup_inclusive_median_end_to_end_us",
            ]:
                value = reuse.get(key)
                if value is not None and (not _is_number(value) or value < 0):
                    self._error(f"reuse_contract.{key} must be a nonnegative number or null")
            if "setup_inclusive_policy" in reuse and reuse.get("setup_inclusive_policy") != (
                "one_time_setup_amortized_over_measured_repeats"
            ):
                self._error(
                    "reuse_contract.setup_inclusive_policy must be "
                    "one_time_setup_amortized_over_measured_repeats"
                )
            repeats = reuse.get("measured_repeat_count")
            if not _is_int(repeats) or repeats <= 0:
                self._error("reuse_contract.measured_repeat_count must be a positive integer")
            break_even = reuse.get("break_even_repeat_count")
            if break_even is not None and (not _is_number(break_even) or break_even <= 0):
                self._error("reuse_contract.break_even_repeat_count must be positive or null")
            if not isinstance(reuse.get("promotion_eligible"), bool):
                self._error("reuse_contract.promotion_eligible must be a boolean")
            for key in [
                "production_runtime_prepack_cache_available",
                "setup_inclusive_cache_promotion_candidate",
            ]:
                value = reuse.get(key)
                if value is not None and not isinstance(value, bool):
                    self._error(f"reuse_contract.{key} must be a boolean when present")
            invalidation = reuse.get("invalidation_reasons")
            if not isinstance(invalidation, list) or not all(isinstance(item, str) for item in invalidation):
                self._error("reuse_contract.invalidation_reasons must be an array of strings")
            runtime_cache = reuse.get("runtime_prepack_cache")
            if self.data.get("prepack_reuse_strategy") == "rocwmma_reusable_b_cache" and runtime_cache is None:
                self._error("rocwmma reusable B-cache captures must include reuse_contract.runtime_prepack_cache")
            if runtime_cache is not None:
                if not isinstance(runtime_cache, dict):
                    self._error("reuse_contract.runtime_prepack_cache must be an object or null")
                else:
                    for key in [
                        "source",
                        "backend",
                        "semantics",
                        "operand_role",
                        "matrix_layout_version",
                        "operand_layout_version",
                        "cache_scope",
                        "cache_key",
                        "detail",
                    ]:
                        if not isinstance(runtime_cache.get(key), str):
                            self._error(f"reuse_contract.runtime_prepack_cache.{key} must be a string")
                    for key in [
                        "cache_key_valid",
                        "reusable_prepack_cache_available",
                        "production_prepack_cache_available",
                    ]:
                        if not isinstance(runtime_cache.get(key), bool):
                            self._error(f"reuse_contract.runtime_prepack_cache.{key} must be a boolean")
                    for key in [
                        "hip_device_id",
                        "matrix_rows",
                        "matrix_cols",
                        "k",
                        "max_prefix",
                        "finite_modulus",
                        "source_version",
                        "plan_fingerprint",
                        "cache_key_hash",
                        "device_bytes",
                        "operand_pack_bytes",
                    ]:
                        value = runtime_cache.get(key)
                        if not _is_int(value):
                            self._error(f"reuse_contract.runtime_prepack_cache.{key} must be an integer")
                    if runtime_cache.get("source") != "rns8_get_prepack_cache_info":
                        self._error("reuse_contract.runtime_prepack_cache.source must be rns8_get_prepack_cache_info")
                    if runtime_cache.get("backend") != "rocwmma":
                        self._error("runtime prepack cache captures must report backend=rocwmma")
                    if runtime_cache.get("operand_role") != "B":
                        self._error("runtime prepack cache captures must report operand_role=B")
                    if runtime_cache.get("cache_key_valid") is not True:
                        self._error("runtime prepack cache captures must report cache_key_valid=true")
                    if runtime_cache.get("reusable_prepack_cache_available") is not True:
                        self._error("runtime prepack cache captures must report reusable_prepack_cache_available=true")
                    source_version = runtime_cache.get("source_version")
                    if _is_int(source_version) and source_version <= 0:
                        self._error("runtime prepack cache captures must report nonzero source_version")
                    if runtime_cache.get("backend") != self.data.get("backend_selected"):
                        self._error("runtime prepack cache backend must match backend_selected")
                    if runtime_cache.get("semantics") != self.data.get("semantics"):
                        self._error("runtime prepack cache semantics must match capture semantics")
                    if runtime_cache.get("matrix_rows") != self.data.get("k"):
                        self._error("runtime prepack cache matrix_rows must match capture k for B operand")
                    if runtime_cache.get("matrix_cols") != self.data.get("n"):
                        self._error("runtime prepack cache matrix_cols must match capture n for B operand")
                    cache_max_prefix = runtime_cache.get("max_prefix")
                    residue_policy = self.data.get("residue_count_policy") or {}
                    capture_prefix = residue_policy.get("selected_prefix", self.data.get("prefix"))
                    if cache_max_prefix is not None and capture_prefix is not None:
                        if cache_max_prefix < capture_prefix:
                            self._error("runtime prepack cache max_prefix must be >= selected prefix")
                    elif cache_max_prefix != capture_prefix:
                        self._error("runtime prepack cache max_prefix must match selected prefix")
                    expected_modulus = self.data.get("finite_modulus") or 0
                    if runtime_cache.get("finite_modulus") != expected_modulus:
                        self._error("runtime prepack cache finite_modulus must match capture finite_modulus")
                    if runtime_cache.get("cache_key_hash") == 0:
                        self._error("runtime prepack cache cache_key_hash must be nonzero")
                    cache_key_fields = _prepack_cache_key_fields(runtime_cache.get("cache_key", ""))
                    if cache_key_fields is None:
                        self._error("runtime prepack cache cache_key must be a prepack-v2 serialized identity")
                    else:
                        strict_cache_fields = {
                            "backend": runtime_cache.get("backend"),
                            "semantics": runtime_cache.get("semantics"),
                            "operand": runtime_cache.get("operand_role"),
                            "m": self.data.get("m"),
                            "n": self.data.get("n"),
                            "k": self.data.get("k"),
                            "source_version": runtime_cache.get("source_version"),
                            "hip_device_id": runtime_cache.get("hip_device_id"),
                            "matrix_rows": runtime_cache.get("matrix_rows"),
                            "matrix_cols": runtime_cache.get("matrix_cols"),
                            "prefix": runtime_cache.get("max_prefix"),
                            "finite_modulus": runtime_cache.get("finite_modulus"),
                            "matrix_layout": runtime_cache.get("matrix_layout_version"),
                        }
                        for key, expected in strict_cache_fields.items():
                            if cache_key_fields.get(key) != str(expected):
                                self._error(f"runtime prepack cache cache_key must include {key}={expected}")
                        soft_cache_fields = {
                            "operand_layout": runtime_cache.get("operand_layout_version"),
                            "plan_fingerprint": runtime_cache.get("plan_fingerprint"),
                            "hash": runtime_cache.get("cache_key_hash"),
                        }
                        for key, expected in soft_cache_fields.items():
                            actual = cache_key_fields.get(key)
                            if actual is not None and str(expected) not in (actual, '') and not str(expected).startswith(actual.rstrip('_')) and not actual.startswith(str(expected)):
                                self._error(f"runtime prepack cache cache_key {key}={actual} != expected {expected}")
                        for key in [
                            "target_id",
                            "kernel",
                            "prepack_kernel",
                            "prefix_schedule_hash",
                            "tile_m",
                            "tile_n",
                            "operand_tile_m",
                            "operand_tile_n",
                            "k_block_size",
                            "k_block_cap",
                        ]:
                            if not cache_key_fields.get(key):
                                self._error(f"runtime prepack cache cache_key must include nonempty {key}")
                    device_bytes = runtime_cache.get("device_bytes")
                    operand_pack_bytes = runtime_cache.get("operand_pack_bytes")
                    if _is_int(device_bytes) and device_bytes <= 0:
                        self._error("runtime prepack cache device_bytes must be positive")
                    if _is_int(operand_pack_bytes) and operand_pack_bytes <= 0:
                        self._error("runtime prepack cache operand_pack_bytes must be positive")
                    if _is_int(device_bytes) and _is_int(operand_pack_bytes) and operand_pack_bytes > device_bytes:
                        self._error("runtime prepack cache operand_pack_bytes must not exceed device_bytes")
                    if (
                        runtime_cache.get("production_prepack_cache_available") is True
                        and self.data.get("semantics") not in {"bounded_i64", "bounded_u64"}
                    ):
                        self._error("runtime production prepack cache is allowed only for bounded i64/u64 captures")

    exact_output = self.data.get("exact_output_contract")
    if exact_output is not None:
        if not isinstance(exact_output, dict):
            self._error("exact_output_contract must be an object")
        else:
            if exact_output.get("requested_final_output") not in OUTPUT_CONTRACT_DOMAINS:
                self._error(
                    f"exact_output_contract.requested_final_output must be one of {sorted(OUTPUT_CONTRACT_DOMAINS)}"
                )
            limb_count = exact_output.get("limb_count")
            if self.data.get("semantics") in {"exact_wide_signed", "exact_wide_unsigned"}:
                if not _is_int(limb_count) or not 1 <= limb_count <= 32:
                    self._error("exact_output_contract.limb_count must be in [1, 32] for exact-wide captures")
            elif limb_count is not None:
                self._error("exact_output_contract.limb_count must be null outside exact-wide captures")
            if exact_output.get("status_policy") not in STATUS_HANDLING:
                self._error(f"exact_output_contract.status_policy must be one of {sorted(STATUS_HANDLING)}")
            if exact_output.get("output_domain_after_measured_repeats") not in OUTPUT_CONTRACT_DOMAINS:
                self._error("exact_output_contract.output_domain_after_measured_repeats must be a known output domain")
            if not isinstance(exact_output.get("final_checksum_export_after_repeats"), bool):
                self._error("exact_output_contract.final_checksum_export_after_repeats must be a boolean")

    export_variant = self.data.get("export_variant")
    if export_variant is not None:
        if not isinstance(export_variant, dict):
            self._error("export_variant must be an object")
        else:
            if not isinstance(export_variant.get("name"), str) or not export_variant.get("name"):
                self._error("export_variant.name must be a nonempty string")
            for key in ["source", "status_policy", "constants_placement"]:
                if not isinstance(export_variant.get(key), str):
                    self._error(f"export_variant.{key} must be a string")
            if export_variant.get("status_policy") not in STATUS_HANDLING:
                self._error(f"export_variant.status_policy must be one of {sorted(STATUS_HANDLING)}")
            selector_source = export_variant.get("selector_source")
            if selector_source is not None and not isinstance(selector_source, str):
                self._error("export_variant.selector_source must be a string when present")
            for key in [
                "selector_policy",
                "semantic_contract",
                "backend",
                "target_id",
                "prefix_contract",
                "signedness",
                "cache_visibility",
            ]:
                if key in export_variant and not isinstance(export_variant.get(key), str):
                    self._error(f"export_variant.{key} must be a string when present")
            selector_key = export_variant.get("selector_key")
            if selector_key is not None and not isinstance(selector_key, str):
                self._error("export_variant.selector_key must be a string or null")
            final_output_mode = export_variant.get("final_output_mode")
            if final_output_mode is not None and final_output_mode not in EXPORT_FINAL_OUTPUT_MODES:
                self._error(f"export_variant.final_output_mode must be one of {sorted(EXPORT_FINAL_OUTPUT_MODES)}")
            for key in ["stale_entry_reason", "status_elision_reason"]:
                if export_variant.get(key) is not None and not isinstance(export_variant.get(key), str):
                    self._error(f"export_variant.{key} must be a string or null")
            output_layout = export_variant.get("output_layout")
            if output_layout is not None and output_layout not in EXPORT_OUTPUT_LAYOUTS:
                self._error(f"export_variant.output_layout must be one of {sorted(EXPORT_OUTPUT_LAYOUTS)}")
            selector_status = export_variant.get("selector_status_policy")
            if selector_status is not None and selector_status not in EXPORT_SELECTOR_STATUS_POLICIES:
                self._error(
                    f"export_variant.selector_status_policy must be one of {sorted(EXPORT_SELECTOR_STATUS_POLICIES)}"
                )
            d2h_policy = export_variant.get("d2h_policy")
            if d2h_policy is not None and d2h_policy not in EXPORT_D2H_POLICIES:
                self._error(f"export_variant.d2h_policy must be one of {sorted(EXPORT_D2H_POLICIES)}")
            selected_export_kernel = export_variant.get("selected_kernel")
            if selected_export_kernel is not None and not isinstance(selected_export_kernel, str):
                self._error("export_variant.selected_kernel must be a string or null")
            for key in ["requires_tile_metadata", "all_zero_tiled_output"]:
                if key in export_variant and not isinstance(export_variant.get(key), bool):
                    self._error(f"export_variant.{key} must be a boolean when present")
            limb_count = export_variant.get("limb_count")
            if limb_count is not None and (not _is_int(limb_count) or not 1 <= limb_count <= 32):
                self._error("export_variant.limb_count must be in [1, 32] or null")
            if not isinstance(export_variant.get("promotion_eligible"), bool):
                self._error("export_variant.promotion_eligible must be a boolean")
            if (
                selected_export_kernel
                and selector_key
                and f"selected_kernel={selected_export_kernel}" not in selector_key
            ):
                self._error("export_variant.selector_key must include selected_kernel")
            if (
                export_variant.get("semantic_contract")
                and selector_key
                and f"semantics={export_variant['semantic_contract']}" not in selector_key
            ):
                self._error("export_variant.selector_key must include semantic_contract")
            reviewable_fixed_limb = (
                export_variant.get("name") == "exact-wide-fixed-limb-export"
                and self.data.get("semantics") in {"exact_wide_signed", "exact_wide_unsigned"}
                and export_variant.get("semantic_contract") == self.data.get("semantics")
                and output_layout == "fixed_u64_limbs"
                and _is_int(limb_count)
                and (
                    export_variant.get("selector_status_policy") == "range_checked_status_buffer"
                    or (
                        export_variant.get("selector_status_policy") == "none"
                        and isinstance(export_variant.get("status_elision_reason"), str)
                        and bool(export_variant.get("status_elision_reason"))
                    )
                )
                and export_variant.get("d2h_policy") in {"host_ld_padded", "compact_contiguous"}
                and final_output_mode == "final_host_output"
            )
            if export_variant.get("promotion_eligible") is True:
                if export_variant.get("name") not in REVIEWABLE_EXPORT_VARIANTS or (
                    export_variant.get("name") != "default" and not reviewable_fixed_limb
                ):
                    self._error(
                        "export_variant.promotion_eligible=true is allowed only for default or exact-wide fixed-limb selector captures"
                    )
            blocker = export_variant.get("promotion_blocker")
            if export_variant.get("promotion_eligible") is True and blocker is not None:
                self._error("promotion-eligible export_variant captures must set promotion_blocker=null")
            if export_variant.get("promotion_eligible") is False and export_variant.get("name") != "default" and not isinstance(blocker, str):
                self._error("non-promoting export_variant captures must declare promotion_blocker")

    reconstruction = self.data.get("reconstruction_variant")
    if reconstruction is not None:
        if not isinstance(reconstruction, dict):
            self._error("reconstruction_variant must be an object")
        else:
            for key in ["name", "family", "controller"]:
                if not isinstance(reconstruction.get(key), str) or not reconstruction.get(key):
                    self._error(f"reconstruction_variant.{key} must be a nonempty string")
            prefix_count = reconstruction.get("prefix_count")
            if not _is_int(prefix_count) or prefix_count < 0:
                self._error("reconstruction_variant.prefix_count must be a nonnegative integer")
            if not isinstance(reconstruction.get("promotion_eligible"), bool):
                self._error("reconstruction_variant.promotion_eligible must be a boolean")
            if reconstruction.get("name") != "default_garner" and reconstruction.get("promotion_eligible") is True:
                self._error("experimental reconstruction_variant captures must set promotion_eligible=false")
            blocker = reconstruction.get("promotion_blocker")
            if reconstruction.get("name") != "default_garner" and not isinstance(blocker, str):
                self._error("experimental reconstruction_variant captures must declare promotion_blocker")

    modulus_set = self.data.get("modulus_set")
    if modulus_set is not None:
        if not isinstance(modulus_set, dict):
            self._error("modulus_set must be an object")
        else:
            name = modulus_set.get("name")
            if not isinstance(name, str) or (name != "default" and not name.startswith("experimental:")):
                self._error("modulus_set.name must be default or experimental:NAME")
            if not isinstance(modulus_set.get("experimental"), bool):
                self._error("modulus_set.experimental must be a boolean")
            for key in ["source", "execution_ladder", "pairwise_coprime_proof", "reducer_cost_hint"]:
                if not isinstance(modulus_set.get(key), str):
                    self._error(f"modulus_set.{key} must be a string")
            for key in ["runtime_selectable", "search_report_required"]:
                if key in modulus_set and not isinstance(modulus_set.get(key), bool):
                    self._error(f"modulus_set.{key} must be a boolean when present")
            if "default_change_gate" in modulus_set and not isinstance(modulus_set.get("default_change_gate"), str):
                self._error("modulus_set.default_change_gate must be a string when present")
            for key in ["product_bits", "prefix_count"]:
                value = modulus_set.get(key)
                if not _is_int(value) or value < 0:
                    self._error(f"modulus_set.{key} must be a nonnegative integer")
            if name == "default" and modulus_set.get("experimental") is not False:
                self._error("modulus_set.experimental must be false for default")
            if name != "default":
                if not isinstance(modulus_set.get("cache_promotion_blocker"), str):
                    self._error("experimental modulus_set captures must declare cache_promotion_blocker")
                if modulus_set.get("runtime_selectable") is not False:
                    self._error("experimental modulus_set captures must set runtime_selectable=false")
                if modulus_set.get("search_report_required") is not True:
                    self._error("experimental modulus_set captures must set search_report_required=true")
                if modulus_set.get("default_change_gate") != "spec_cache_schema_proof_and_same_target_review_required":
                    self._error("experimental modulus_set captures must declare the default-change gate")

    residue_policy = self.data.get("residue_count_policy")
    if residue_policy is not None:
        if not isinstance(residue_policy, dict):
            self._error("residue_count_policy must be an object")
        else:
            for key in ["policy", "autotune_scope"]:
                if not isinstance(residue_policy.get(key), str):
                    self._error(f"residue_count_policy.{key} must be a string")
            for key in ["requested_prefix", "selected_prefix", "minimum_range_prefix", "redundant_residue_count"]:
                value = residue_policy.get(key)
                if not _is_int(value) or value < 0:
                    self._error(f"residue_count_policy.{key} must be a nonnegative integer")
            if "promotion_eligible" in residue_policy and not isinstance(residue_policy.get("promotion_eligible"), bool):
                self._error("residue_count_policy.promotion_eligible must be a boolean when present")
            if _is_int(self.data.get("selected_prefix")) and residue_policy.get("selected_prefix") != self.data.get("selected_prefix"):
                self._error("residue_count_policy.selected_prefix must match selected_prefix")
            if _is_int(self.data.get("requested_max_prefix")) and residue_policy.get("requested_prefix") != self.data.get("requested_max_prefix"):
                self._error("residue_count_policy.requested_prefix must match requested_max_prefix")
            expected_redundant = max(0, residue_policy.get("selected_prefix", 0) - residue_policy.get("minimum_range_prefix", 0))
            if residue_policy.get("redundant_residue_count") != expected_redundant:
                self._error("residue_count_policy.redundant_residue_count must equal selected_prefix - minimum_range_prefix")
            if residue_policy.get("autotune_scope") == "evidence_only_non_promoting":
                if residue_policy.get("promotion_eligible") is not False:
                    self._error("evidence-only residue_count_policy captures must set promotion_eligible=false")
                if not isinstance(residue_policy.get("cache_promotion_blocker"), str):
                    self._error("evidence-only residue_count_policy captures must declare cache_promotion_blocker")
            if residue_policy.get("autotune_scope") == "current_exact_cache" and residue_policy.get("promotion_eligible") is False:
                self._error("current_exact_cache residue_count_policy captures must not force promotion_eligible=false")

    tile_variant = self.data.get("tile_shape_variant")
    if tile_variant is not None:
        if not isinstance(tile_variant, dict):
            self._error("tile_shape_variant must be an object")
        else:
            if not isinstance(tile_variant.get("name"), str) or not tile_variant.get("name"):
                self._error("tile_shape_variant.name must be a nonempty string")
            for key in [
                "resource_report_key",
                "shape_family_bucket",
                "stale_kernel_rejection",
            ]:
                if not isinstance(tile_variant.get(key), str):
                    self._error(f"tile_shape_variant.{key} must be a string")
            for key in [
                "k_block_policy",
                "split_k_mode",
                "accumulator_safety_key",
                "resource_report_required",
            ]:
                if key in tile_variant and not isinstance(tile_variant.get(key), str):
                    self._error(f"tile_shape_variant.{key} must be a string")
            for key in ["tile_m", "tile_n", "tile_k"]:
                value = tile_variant.get(key)
                if not _is_int(value) or value <= 0:
                    self._error(f"tile_shape_variant.{key} must be a positive integer")
            if tile_variant.get("tile_m") != self.data.get("tile_m"):
                self._error("tile_shape_variant.tile_m must match tile_m")
            if tile_variant.get("tile_n") != self.data.get("tile_n"):
                self._error("tile_shape_variant.tile_n must match tile_n")
            if _is_int(self.data.get("k_block_size")) and tile_variant.get("tile_k") != self.data.get("k_block_size"):
                self._error("tile_shape_variant.tile_k must match k_block_size")
            identity = tile_variant.get("selected_kernel_identity")
            selected_kernel = self.data.get("selected_kernel")
            if identity is not None and identity != selected_kernel:
                self._error("tile_shape_variant.selected_kernel_identity must match selected_kernel")
            safety_key = tile_variant.get("accumulator_safety_key")
            if safety_key and f"k_block_size={tile_variant.get('tile_k')}" not in safety_key:
                self._error("tile_shape_variant.accumulator_safety_key must include tile_k k_block_size")

    grouped = self.data.get("grouped_dispatch")
    if grouped is not None:
        if not isinstance(grouped, dict):
            self._error("grouped_dispatch must be an object")
        else:
            if not isinstance(grouped.get("requested"), bool):
                self._error("grouped_dispatch.requested must be a boolean")
            task_count = grouped.get("task_count")
            if not _is_int(task_count) or task_count <= 0:
                self._error("grouped_dispatch.task_count must be a positive integer")
            for key in ["descriptor_identity", "source_hash", "output_hash", "setup_scope"]:
                if not isinstance(grouped.get(key), str):
                    self._error(f"grouped_dispatch.{key} must be a string")
            strategy = grouped.get("execution_strategy")
            if strategy is not None and strategy not in GROUPED_DISPATCH_EXECUTION_STRATEGIES:
                self._error("grouped_dispatch.execution_strategy must be a known grouped strategy")
            batched_export = grouped.get("batched_export_enabled")
            if batched_export is not None and not isinstance(batched_export, bool):
                self._error("grouped_dispatch.batched_export_enabled must be a boolean")
            slab_bytes = grouped.get("device_output_slab_bytes")
            if slab_bytes is not None and (not _is_int(slab_bytes) or slab_bytes < 0):
                self._error("grouped_dispatch.device_output_slab_bytes must be a nonnegative integer")
            if grouped.get("capture_status") not in GROUPED_DISPATCH_STATUSES:
                self._error(f"grouped_dispatch.capture_status must be one of {sorted(GROUPED_DISPATCH_STATUSES)}")
            if task_count and task_count > 1 and grouped.get("requested") is not True:
                self._error("grouped_dispatch.requested must be true when task_count > 1")
            if task_count and task_count > 1 and grouped.get("promotion_eligible") is not False:
                self._error("grouped_dispatch task_count > 1 must set promotion_eligible=false")
            task_descriptor = grouped.get("task_descriptor_contract")
            if task_descriptor is not None:
                if not isinstance(task_descriptor, dict):
                    self._error("grouped_dispatch.task_descriptor_contract must be an object")
                else:
                    if task_descriptor.get("schema_version") != 1:
                        self._error("grouped_dispatch.task_descriptor_contract.schema_version must be 1")
                    descriptor_layout = task_descriptor.get("descriptor_layout")
                    if descriptor_layout not in GROUPED_TASK_DESCRIPTOR_LAYOUTS:
                        self._error("grouped_dispatch.task_descriptor_contract.descriptor_layout must be known")
                    bucket_policy = task_descriptor.get("bucket_policy")
                    if bucket_policy not in GROUPED_TASK_BUCKET_POLICIES:
                        self._error("grouped_dispatch.task_descriptor_contract.bucket_policy must be known")
                    bucket_count = task_descriptor.get("bucket_count")
                    if not _is_int(bucket_count) or bucket_count < 0:
                        self._error("grouped_dispatch.task_descriptor_contract.bucket_count must be nonnegative")
                    descriptor_task_count = task_descriptor.get("task_count")
                    if not _is_int(descriptor_task_count) or descriptor_task_count <= 0:
                        self._error("grouped_dispatch.task_descriptor_contract.task_count must be positive")
                    if not isinstance(task_descriptor.get("same_shape_required"), bool):
                        self._error("grouped_dispatch.task_descriptor_contract.same_shape_required must be a boolean")
                    for key in ["shape_key", "semantics"]:
                        if not isinstance(task_descriptor.get(key), str):
                            self._error(f"grouped_dispatch.task_descriptor_contract.{key} must be a string")
                    if task_descriptor.get("output_domain") not in OUTPUT_CONTRACT_DOMAINS:
                        self._error("grouped_dispatch.task_descriptor_contract.output_domain must be a known output domain")
                    if task_descriptor.get("source_version_policy") not in GROUPED_TASK_SOURCE_VERSION_POLICIES:
                        self._error("grouped_dispatch.task_descriptor_contract.source_version_policy must be known")
                    if task_descriptor.get("workspace_policy") not in GROUPED_TASK_WORKSPACE_POLICIES:
                        self._error("grouped_dispatch.task_descriptor_contract.workspace_policy must be known")
                    if task_descriptor.get("matrix_ownership_policy") not in GROUPED_TASK_MATRIX_OWNERSHIP_POLICIES:
                        self._error(
                            "grouped_dispatch.task_descriptor_contract.matrix_ownership_policy must be known"
                        )
                    if task_descriptor.get("descriptor_reuse_policy") not in GROUPED_TASK_DESCRIPTOR_REUSE_POLICIES:
                        self._error(
                            "grouped_dispatch.task_descriptor_contract.descriptor_reuse_policy must be known"
                        )
                    if task_descriptor.get("stride_policy") not in GROUPED_TASK_STRIDE_POLICIES:
                        self._error("grouped_dispatch.task_descriptor_contract.stride_policy must be known")
                    if task_descriptor.get("output_currentness_policy") not in GROUPED_TASK_OUTPUT_CURRENTNESS_POLICIES:
                        self._error(
                            "grouped_dispatch.task_descriptor_contract.output_currentness_policy must be known"
                        )
                    if task_descriptor.get("lifetime_policy") not in GROUPED_TASK_LIFETIME_POLICIES:
                        self._error("grouped_dispatch.task_descriptor_contract.lifetime_policy must be known")
                    if task_descriptor.get("checksum_policy") not in GROUPED_TASK_CHECKSUM_POLICIES:
                        self._error("grouped_dispatch.task_descriptor_contract.checksum_policy must be known")
                    if task_descriptor.get("status_policy") not in GROUPED_TASK_STATUS_POLICIES:
                        self._error("grouped_dispatch.task_descriptor_contract.status_policy must be known")
                    if task_descriptor.get("device_descriptor_policy") not in GROUPED_TASK_DEVICE_DESCRIPTOR_POLICIES:
                        self._error("grouped_dispatch.task_descriptor_contract.device_descriptor_policy must be known")
                    if task_descriptor.get("promotion_eligible") is not False:
                        self._error("grouped_dispatch.task_descriptor_contract.promotion_eligible must be false")
                    buckets = task_descriptor.get("buckets")
                    if buckets is not None:
                        if not isinstance(buckets, list):
                            self._error("grouped_dispatch.task_descriptor_contract.buckets must be an array")
                        else:
                            for index, bucket in enumerate(buckets):
                                if not isinstance(bucket, dict):
                                    self._error(
                                        "grouped_dispatch.task_descriptor_contract.buckets entries must be objects"
                                    )
                                    continue
                                if bucket.get("bucket_index") != index:
                                    self._error(
                                        "grouped_dispatch.task_descriptor_contract.buckets bucket_index must be contiguous"
                                    )
                                for key in ["task_offset", "task_count"]:
                                    value = bucket.get(key)
                                    if not _is_int(value) or value < 0:
                                        self._error(
                                            f"grouped_dispatch.task_descriptor_contract.buckets.{key} "
                                            "must be a nonnegative integer"
                                        )
                                for key in ["shape_key", "semantics"]:
                                    if not isinstance(bucket.get(key), str):
                                        self._error(
                                            f"grouped_dispatch.task_descriptor_contract.buckets.{key} must be a string"
                                        )
                                if bucket.get("output_domain") not in OUTPUT_CONTRACT_DOMAINS:
                                    self._error(
                                        "grouped_dispatch.task_descriptor_contract.buckets.output_domain "
                                        "must be a known output domain"
                                    )

    graph = self.data.get("hip_graph_replay")
    if graph is not None:
        if not isinstance(graph, dict):
            self._error("hip_graph_replay must be an object")
        else:
            if not isinstance(graph.get("requested"), bool):
                self._error("hip_graph_replay.requested must be a boolean")
            for key in ["descriptor_identity", "plan_identity", "setup_scope"]:
                if not isinstance(graph.get(key), str):
                    self._error(f"hip_graph_replay.{key} must be a string")
            if graph.get("capture_status") not in GRAPH_REPLAY_STATUSES:
                self._error(f"hip_graph_replay.capture_status must be one of {sorted(GRAPH_REPLAY_STATUSES)}")
            if graph.get("requested") is True and graph.get("promotion_eligible") is not False:
                self._error("hip_graph_replay requested captures must set promotion_eligible=false")
            if graph.get("requested") is True:
                metadata = self.data.get("backend_metadata")
                expected_plan_identity = metadata.get("autotune_key") if isinstance(metadata, dict) else None
                if isinstance(expected_plan_identity, str) and graph.get("plan_identity") != expected_plan_identity:
                    self._error("hip_graph_replay.plan_identity must match backend_metadata.autotune_key")
                expected_descriptor = (
                    f"fixed_plan_workspace_descriptor:m={self.data.get('m')};"
                    f"n={self.data.get('n')};k={self.data.get('k')}"
                )
                if graph.get("descriptor_identity") != expected_descriptor:
                    self._error("hip_graph_replay.descriptor_identity must match capture m/n/k")

    adaptive = self.data.get("adaptive_grouped_scheduler")
    if adaptive is not None:
        if not isinstance(adaptive, dict):
            self._error("adaptive_grouped_scheduler must be an object")
        else:
            if not isinstance(adaptive.get("requested"), bool):
                self._error("adaptive_grouped_scheduler.requested must be a boolean")
            for key in ["strategy", "descriptor_identity", "selected_prefix_histogram"]:
                if not isinstance(adaptive.get(key), str):
                    self._error(f"adaptive_grouped_scheduler.{key} must be a string")
            for key in [
                "group_count",
                "active_prefix_count",
                "active_tile_count",
                "active_entry_count",
                "zero_tile_count",
                "independent_launch_count_model",
                "aggregate_launch_count_model",
            ]:
                value = adaptive.get(key)
                if not _is_int(value) or value < 0:
                    self._error(f"adaptive_grouped_scheduler.{key} must be a nonnegative integer")
            if not _is_number(adaptive.get("launch_reduction_ratio")) or adaptive.get("launch_reduction_ratio") < 0:
                self._error("adaptive_grouped_scheduler.launch_reduction_ratio must be a nonnegative number")
            if not isinstance(adaptive.get("event_scope"), str):
                self._error("adaptive_grouped_scheduler.event_scope must be a string")
            if adaptive.get("capture_status") not in GROUPED_DISPATCH_STATUSES:
                self._error(
                    f"adaptive_grouped_scheduler.capture_status must be one of {sorted(GROUPED_DISPATCH_STATUSES)}"
                )
            if adaptive.get("requested") is True and adaptive.get("promotion_eligible") is not False:
                self._error("adaptive_grouped_scheduler requested captures must set promotion_eligible=false")
            if adaptive.get("requested") is True:
                if adaptive.get("strategy") != "prefix_tile_zero_mask_grouped_descriptors":
                    self._error("adaptive_grouped_scheduler requested captures must declare the grouped descriptor strategy")
                if adaptive.get("group_count", 0) <= 0:
                    self._error("adaptive_grouped_scheduler requested captures must have a positive group_count")
                if adaptive.get("active_tile_count", 0) <= 0:
                    self._error("adaptive_grouped_scheduler requested captures must have a positive active_tile_count")
                if self.data.get("backend_selected") == "hip-direct" and self.data.get(
                    "selected_kernel"
                ) == "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3":
                    if adaptive.get("capture_status") != "executed":
                        self._error(
                            "adaptive_grouped_scheduler direct-HIP grouped-kernel captures must set "
                            "capture_status=executed"
                        )
                    if adaptive.get("unsupported_reason") is not None:
                        self._error(
                            "adaptive_grouped_scheduler executed captures must set unsupported_reason=null"
                        )
                    if adaptive.get("active_entry_count", 0) <= 0:
                        self._error(
                            "adaptive_grouped_scheduler executed captures must have a positive active_entry_count"
                        )
                    if adaptive.get("aggregate_launch_count_model") != 1:
                        self._error(
                            "adaptive_grouped_scheduler executed captures must model one aggregate GEMM event group"
                        )
                    if adaptive.get("event_scope") != "aggregate_rns_gemm_kernel_group_per_measured_repeat":
                        self._error(
                            "adaptive_grouped_scheduler executed captures must declare the aggregate GEMM event scope"
                        )

    resident_redesign = self.data.get("resident_redesign")
    if resident_redesign is not None:
        if not isinstance(resident_redesign, dict):
            self._error("resident_redesign must be an object")
        else:
            enabled = resident_redesign.get("enabled")
            if not isinstance(enabled, bool):
                self._error("resident_redesign.enabled must be a boolean")
            candidate = resident_redesign.get("candidate")
            if enabled is True:
                if not isinstance(candidate, str) or not candidate:
                    self._error("resident_redesign.candidate must be a nonempty string when enabled")
            elif candidate is not None:
                self._error("resident_redesign.candidate must be null when disabled")
            dimensions = resident_redesign.get("dimensions")
            if not isinstance(dimensions, list) or not all(isinstance(item, str) and item for item in dimensions):
                self._error("resident_redesign.dimensions must be an array of nonempty strings")
            else:
                unknown = sorted(set(dimensions) - RESIDENT_REDESIGN_DIMENSIONS)
                if unknown:
                    self._error(f"resident_redesign.dimensions contains unknown values: {unknown}")
                if enabled is True and not dimensions:
                    self._error("resident_redesign.dimensions must be nonempty when enabled")
                if enabled is False and dimensions:
                    self._error("resident_redesign.dimensions must be empty when disabled")
            if not isinstance(resident_redesign.get("policy"), str) or not resident_redesign.get("policy"):
                self._error("resident_redesign.policy must be a nonempty string")
            if resident_redesign.get("resource_evidence_required") is not True:
                self._error("resident_redesign.resource_evidence_required must be true")
            if resident_redesign.get("promotion_eligible") is not False:
                self._error("resident_redesign captures must set promotion_eligible=false")
            blocker = resident_redesign.get("cache_promotion_blocker")
            if enabled is True:
                if not isinstance(blocker, str) or not blocker:
                    self._error("resident_redesign.cache_promotion_blocker must be a nonempty string when enabled")
            elif blocker is not None:
                self._error("resident_redesign.cache_promotion_blocker must be null when disabled")

    resident = self.data.get("resident_lifetime")
    if resident is not None:
        if not isinstance(resident, dict):
            self._error("resident_lifetime must be an object")
        else:
            if not isinstance(resident.get("enabled"), bool):
                self._error("resident_lifetime.enabled must be a boolean")
            for key in [
                "matrix_roles",
                "source_version_policy",
                "current_storage_state",
                "output_domain",
                "workspace_identity",
                "stale_source_rejection",
            ]:
                if not isinstance(resident.get(key), str):
                    self._error(f"resident_lifetime.{key} must be a string")
            if resident.get("output_domain") not in OUTPUT_CONTRACT_DOMAINS:
                self._error("resident_lifetime.output_domain must be a known output domain")
            if resident.get("promotion_eligible") is not False:
                self._error("resident_lifetime captures must set promotion_eligible=false")

    arena = self.data.get("workspace_arena")
    if arena is not None:
        if not isinstance(arena, dict):
            self._error("workspace_arena must be an object")
        else:
            if not isinstance(arena.get("enabled"), bool):
                self._error("workspace_arena.enabled must be a boolean")
            for key in ["arena_identity", "source_version_policy", "stream_safety"]:
                if not isinstance(arena.get(key), str):
                    self._error(f"workspace_arena.{key} must be a string")
            for key in ["size_bytes", "high_water_mark_bytes", "suballocation_count"]:
                value = arena.get(key)
                if not _is_int(value) or value < 0:
                    self._error(f"workspace_arena.{key} must be a nonnegative integer")
            if not isinstance(arena.get("measured_repeat_allocation_free"), bool):
                self._error("workspace_arena.measured_repeat_allocation_free must be a boolean")
            for key in ["setup_allocation_delta", "measured_repeat_allocation_delta"]:
                if arena.get(key) is not None:
                    self._validate_counter_snapshot(f"workspace_arena.{key}", arena.get(key))
            if arena.get("promotion_eligible") is not False:
                self._error("workspace_arena captures must set promotion_eligible=false")

    overlap = self.data.get("streaming_overlap")
    if overlap is not None:
        if not isinstance(overlap, dict):
            self._error("streaming_overlap must be an object")
        else:
            if not isinstance(overlap.get("requested"), bool):
                self._error("streaming_overlap.requested must be a boolean")
            for key in ["pipeline", "buffering", "dependency_contract", "transfer_policy"]:
                if not isinstance(overlap.get(key), str):
                    self._error(f"streaming_overlap.{key} must be a string")
            for key in ["stream_count", "buffer_count", "measured_repeat_count", "batch_wall_us", "per_repeat_pipeline_us"]:
                value = overlap.get(key)
                if not _is_int(value) or value < 0:
                    self._error(f"streaming_overlap.{key} must be a nonnegative integer")
            if not isinstance(overlap.get("explicit_dependency_events"), bool):
                self._error("streaming_overlap.explicit_dependency_events must be a boolean")
            if not isinstance(overlap.get("stage_event_scope"), str):
                self._error("streaming_overlap.stage_event_scope must be a string")
            if overlap.get("capture_status") not in STREAMING_OVERLAP_STATUSES:
                self._error(f"streaming_overlap.capture_status must be one of {sorted(STREAMING_OVERLAP_STATUSES)}")
            if overlap.get("requested") is True and overlap.get("promotion_eligible") is not False:
                self._error("streaming_overlap requested captures must set promotion_eligible=false")
            if overlap.get("capture_status") == "executed":
                stream_count = overlap.get("stream_count")
                buffer_count = overlap.get("buffer_count")
                measured_repeat_count = overlap.get("measured_repeat_count")
                batch_wall_us = overlap.get("batch_wall_us")
                per_repeat_pipeline_us = overlap.get("per_repeat_pipeline_us")
                if overlap.get("requested") is not True:
                    self._error("executed streaming_overlap captures must set requested=true")
                if _is_int(stream_count) and stream_count < 3:
                    self._error("executed streaming_overlap captures must use at least three streams")
                if _is_int(buffer_count) and buffer_count < 2:
                    self._error("executed streaming_overlap captures must use at least two buffers")
                if _is_int(measured_repeat_count) and measured_repeat_count != self.data.get("repeats"):
                    self._error("streaming_overlap.measured_repeat_count must match repeats when executed")
                if _is_int(batch_wall_us) and batch_wall_us <= 0:
                    self._error("executed streaming_overlap captures must record batch_wall_us > 0")
                if _is_int(per_repeat_pipeline_us) and per_repeat_pipeline_us <= 0:
                    self._error("executed streaming_overlap captures must record per_repeat_pipeline_us > 0")
                if overlap.get("explicit_dependency_events") is not True:
                    self._error("executed streaming_overlap captures must set explicit_dependency_events=true")
                if overlap.get("stage_event_scope") != "direct_hip_streaming_overlap_multistream_operation_groups":
                    self._error(
                        "executed streaming_overlap captures must use "
                        "stage_event_scope=direct_hip_streaming_overlap_multistream_operation_groups"
                    )
                if overlap.get("unsupported_reason") is not None:
                    self._error("executed streaming_overlap captures must set unsupported_reason=null")
            elif overlap.get("requested") is True:
                if not isinstance(overlap.get("unsupported_reason"), str) or not overlap.get("unsupported_reason"):
                    self._error("requested non-executed streaming_overlap captures must include unsupported_reason")

    proxy = self.data.get("workload_proxy")
    if proxy is not None:
        if not isinstance(proxy, dict):
            self._error("workload_proxy must be an object")
        else:
            if not isinstance(proxy.get("enabled"), bool):
                self._error("workload_proxy.enabled must be a boolean")
            if proxy.get("family") not in WORKLOAD_PROXY_FAMILIES:
                self._error(f"workload_proxy.family must be one of {sorted(WORKLOAD_PROXY_FAMILIES)}")
            for key in ["label", "tower_role", "reuse_profile", "transform_role", "output_domain_requirement"]:
                if not isinstance(proxy.get(key), str):
                    self._error(f"workload_proxy.{key} must be a string")
            if proxy.get("output_domain_requirement") not in OUTPUT_CONTRACT_DOMAINS:
                self._error("workload_proxy.output_domain_requirement must be a known output domain")
            if proxy.get("compatibility_claim") is not False:
                self._error("workload_proxy.compatibility_claim must be false")

    release_gate = self.data.get("release_gate")
    if release_gate is not None:
        if not isinstance(release_gate, dict):
            self._error("release_gate must be an object")
        else:
            if not isinstance(release_gate.get("name"), str) or not release_gate.get("name"):
                self._error("release_gate.name must be a nonempty string")
            if not isinstance(release_gate.get("requested"), bool):
                self._error("release_gate.requested must be a boolean")
            for key in ["classification_tier", "cpu_reference_policy", "memory_cap_policy", "resume_policy"]:
                if not isinstance(release_gate.get(key), str):
                    self._error(f"release_gate.{key} must be a string")
            if release_gate.get("review_status") not in RELEASE_GATE_REVIEW_STATUSES:
                self._error(f"release_gate.review_status must be one of {sorted(RELEASE_GATE_REVIEW_STATUSES)}")
            if not isinstance(release_gate.get("cache_eligible"), bool):
                self._error("release_gate.cache_eligible must be a boolean")
            blockers = release_gate.get("blockers")
            if not isinstance(blockers, list) or not all(isinstance(item, str) for item in blockers):
                self._error("release_gate.blockers must be an array of strings")

    amortization = self.data.get("verification_amortization")
    if amortization is not None:
        if not isinstance(amortization, dict):
            self._error("verification_amortization must be an object")
        else:
            if not isinstance(amortization.get("enabled"), bool):
                self._error("verification_amortization.enabled must be a boolean")
            for key in ["policy", "reused_reference_structure", "final_exact_comparison_status"]:
                if not isinstance(amortization.get(key), str):
                    self._error(f"verification_amortization.{key} must be a string")
            if amortization.get("final_exact_comparison_required") is not True:
                self._error("verification_amortization.final_exact_comparison_required must be true")
            if amortization.get("promotion_eligible") is not False:
                self._error("verification_amortization captures must set promotion_eligible=false")

    error_detection = self.data.get("error_detection_policy")
    if error_detection is not None:
        if not isinstance(error_detection, dict):
            self._error("error_detection_policy must be an object")
        else:
            if not isinstance(error_detection.get("enabled"), bool):
                self._error("error_detection_policy.enabled must be a boolean")
            for key in ["policy", "mode", "verification_basis", "false_negative_policy", "final_exact_comparison_status"]:
                if not isinstance(error_detection.get(key), str):
                    self._error(f"error_detection_policy.{key} must be a string")
            if error_detection.get("mode") not in ERROR_DETECTION_MODES:
                self._error(f"error_detection_policy.mode must be one of {sorted(ERROR_DETECTION_MODES)}")
            rounds = error_detection.get("verification_rounds")
            if not _is_int(rounds) or rounds < 0:
                self._error("error_detection_policy.verification_rounds must be a nonnegative integer")
            for key in [
                "rng_seed_recorded",
                "final_exact_comparison_required",
                "research_only",
                "default_exact_api_unchanged",
                "runtime_routing_allowed",
                "cache_eligible",
                "promotion_eligible",
            ]:
                if not isinstance(error_detection.get(key), bool):
                    self._error(f"error_detection_policy.{key} must be a boolean")
            if error_detection.get("enabled") is True:
                if not error_detection.get("policy") or error_detection.get("policy") == "none":
                    self._error("enabled error_detection_policy.policy must be a nonempty non-none string")
                if error_detection.get("mode") == "not_requested":
                    self._error("enabled error_detection_policy.mode must not be not_requested")
                if error_detection.get("verification_basis") in {"", "none"}:
                    self._error("enabled error_detection_policy.verification_basis must describe the verification basis")
                if error_detection.get("false_negative_policy") in {"", "none"}:
                    self._error("enabled error_detection_policy.false_negative_policy must describe false-negative policy")
                if error_detection.get("final_exact_comparison_required") is not True:
                    self._error("error_detection_policy.final_exact_comparison_required must be true")
                if error_detection.get("final_exact_comparison_status") not in ERROR_DETECTION_FINAL_STATUSES:
                    self._error(
                        "error_detection_policy.final_exact_comparison_status must record exact/reference status"
                    )
                if error_detection.get("research_only") is not True:
                    self._error("enabled error_detection_policy captures must set research_only=true")
                if error_detection.get("default_exact_api_unchanged") is not True:
                    self._error("enabled error_detection_policy captures must keep default_exact_api_unchanged=true")
                if error_detection.get("runtime_routing_allowed") is not False:
                    self._error("enabled error_detection_policy captures must set runtime_routing_allowed=false")
                if error_detection.get("cache_eligible") is not False:
                    self._error("enabled error_detection_policy captures must set cache_eligible=false")
                if error_detection.get("promotion_eligible") is not False:
                    self._error("enabled error_detection_policy captures must set promotion_eligible=false")
                if error_detection.get("mode") == "probabilistic_product_check":
                    if error_detection.get("rng_seed_recorded") is not True:
                        self._error("probabilistic error_detection_policy captures must set rng_seed_recorded=true")
                    if not _is_int(rounds) or rounds <= 0:
                        self._error(
                            "probabilistic error_detection_policy captures must set positive verification_rounds"
                        )

    cpu_selector = self.data.get("cpu_small_shape_selector")
    if cpu_selector is not None:
        if not isinstance(cpu_selector, dict):
            self._error("cpu_small_shape_selector must be an object")
        else:
            if not isinstance(cpu_selector.get("enabled"), bool):
                self._error("cpu_small_shape_selector.enabled must be a boolean")
            for key in ["policy", "candidate_role", "boundary_key", "threshold_scope", "selector_explanation"]:
                if not isinstance(cpu_selector.get(key), str):
                    self._error(f"cpu_small_shape_selector.{key} must be a string")
            if cpu_selector.get("candidate_role") not in CPU_SELECTOR_ROLES:
                self._error(
                    f"cpu_small_shape_selector.candidate_role must be one of {sorted(CPU_SELECTOR_ROLES)}"
                )
            for key in [
                "cpu_reference_required",
                "release_review_required",
                "runtime_routing_allowed",
                "cache_eligible",
                "promotion_eligible",
            ]:
                if not isinstance(cpu_selector.get(key), bool):
                    self._error(f"cpu_small_shape_selector.{key} must be a boolean")
            if cpu_selector.get("enabled") is True:
                if cpu_selector.get("policy") in {"", "none"}:
                    self._error("enabled cpu_small_shape_selector.policy must be a nonempty non-none string")
                if cpu_selector.get("candidate_role") == "not_requested":
                    self._error("enabled cpu_small_shape_selector.candidate_role must not be not_requested")
                if not cpu_selector.get("boundary_key"):
                    self._error("enabled cpu_small_shape_selector.boundary_key must be nonempty")
                if cpu_selector.get("cpu_reference_required") is not True:
                    self._error("enabled cpu_small_shape_selector captures must require a CPU reference")
                if cpu_selector.get("release_review_required") is not True:
                    self._error("enabled cpu_small_shape_selector captures must require release review")
                if cpu_selector.get("runtime_routing_allowed") is not False:
                    self._error("enabled cpu_small_shape_selector captures must set runtime_routing_allowed=false")
                if cpu_selector.get("cache_eligible") is not False:
                    self._error("enabled cpu_small_shape_selector captures must set cache_eligible=false")
                if cpu_selector.get("promotion_eligible") is not False:
                    self._error("enabled cpu_small_shape_selector captures must set promotion_eligible=false")

    incremental = self.data.get("incremental_result_cache")
    if incremental is not None:
        if not isinstance(incremental, dict):
            self._error("incremental_result_cache must be an object")
        else:
            if not isinstance(incremental.get("enabled"), bool):
                self._error("incremental_result_cache.enabled must be a boolean")
            for key in [
                "policy",
                "source_identity_policy",
                "source_version_policy",
                "dirty_region_policy",
                "result_lifetime_policy",
                "checksum_policy",
                "partial_recompute_policy",
                "final_exact_comparison_status",
            ]:
                if not isinstance(incremental.get(key), str):
                    self._error(f"incremental_result_cache.{key} must be a string")
            for key in [
                "final_exact_comparison_required",
                "public_contract_available",
                "default_gemm_unchanged",
                "runtime_routing_allowed",
                "cache_eligible",
                "promotion_eligible",
            ]:
                if not isinstance(incremental.get(key), bool):
                    self._error(f"incremental_result_cache.{key} must be a boolean")
            if incremental.get("enabled") is True:
                scenario = self.data.get("scenario_metadata") or {}
                promotion_scope = scenario.get("promotion_eligibility")
                public_contract = incremental.get("public_contract_available") is True
                if incremental.get("policy") in {"", "none"}:
                    self._error("enabled incremental_result_cache.policy must be a nonempty non-none string")
                for key in [
                    "source_identity_policy",
                    "source_version_policy",
                    "dirty_region_policy",
                    "result_lifetime_policy",
                    "checksum_policy",
                    "partial_recompute_policy",
                ]:
                    if incremental.get(key) in {"", "none"}:
                        self._error(f"enabled incremental_result_cache.{key} must describe the contract")
                if incremental.get("final_exact_comparison_required") is not True:
                    self._error("incremental_result_cache.final_exact_comparison_required must be true")
                if incremental.get("final_exact_comparison_status") not in ERROR_DETECTION_FINAL_STATUSES:
                    self._error("incremental_result_cache.final_exact_comparison_status must record exact/reference status")
                if incremental.get("default_gemm_unchanged") is not True:
                    self._error("enabled incremental_result_cache captures must keep default_gemm_unchanged=true")
                if public_contract:
                    if promotion_scope != "result_cache_contract_candidate":
                        self._error(
                            "public incremental_result_cache captures must use result_cache_contract_candidate eligibility"
                        )
                    if incremental.get("runtime_routing_allowed") is not True:
                        self._error("public incremental_result_cache captures must set runtime_routing_allowed=true")
                    if incremental.get("cache_eligible") is not True:
                        self._error("public incremental_result_cache captures must set cache_eligible=true")
                    if incremental.get("promotion_eligible") is not True:
                        self._error("public incremental_result_cache captures must set promotion_eligible=true")
                    for key in [
                        "a_matrix_instance_id",
                        "b_matrix_instance_id",
                        "a_source_version",
                        "b_source_version",
                        "result_cache_key_fingerprint",
                        "dirty_region_count",
                        "recomputed_region_count",
                        "copied_from_cache_bytes",
                        "cache_allocation_bytes",
                    ]:
                        if key not in incremental:
                            self._error(f"public incremental_result_cache.{key} must be present")
                    if incremental.get("stale_rejection_covered") is not True:
                        self._error("public incremental_result_cache.stale_rejection_covered must be true")
                    regions = incremental.get("dirty_regions")
                    if not isinstance(regions, list) or not regions:
                        self._error("public incremental_result_cache.dirty_regions must be a nonempty list")
                    else:
                        for index, region in enumerate(regions):
                            if not isinstance(region, dict):
                                self._error(f"incremental_result_cache.dirty_regions[{index}] must be an object")
                                continue
                            for key in ["row_offset", "col_offset", "row_extent", "col_extent"]:
                                if not isinstance(region.get(key), int):
                                    self._error(f"incremental_result_cache.dirty_regions[{index}].{key} must be an integer")
                            row_offset = region.get("row_offset")
                            col_offset = region.get("col_offset")
                            row_extent = region.get("row_extent")
                            col_extent = region.get("col_extent")
                            if (
                                isinstance(row_offset, int)
                                and isinstance(row_extent, int)
                                and (row_offset < 0 or row_extent <= 0 or row_offset + row_extent > self.data.get("m", 0))
                            ):
                                self._error(f"incremental_result_cache.dirty_regions[{index}] rows out of bounds")
                            if (
                                isinstance(col_offset, int)
                                and isinstance(col_extent, int)
                                and (col_offset < 0 or col_extent <= 0 or col_offset + col_extent > self.data.get("n", 0))
                            ):
                                self._error(f"incremental_result_cache.dirty_regions[{index}] columns out of bounds")
                else:
                    if incremental.get("runtime_routing_allowed") is not False:
                        self._error("research incremental_result_cache captures must set runtime_routing_allowed=false")
                    if incremental.get("cache_eligible") is not False:
                        self._error("research incremental_result_cache captures must set cache_eligible=false")
                    if incremental.get("promotion_eligible") is not False:
                        self._error("research incremental_result_cache captures must set promotion_eligible=false")
