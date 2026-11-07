"""Synthetic native-format chain metadata for report controls; never measured evidence."""

from __future__ import annotations

import copy
from test_benchmark_schema_support_core import expect_valid
from test_benchmark_schema_support_chains_wrap64 import (
    as_residue_current_chain_capture,
    as_residue_chain_final_export_capture,
    as_exact_wide_residue_chain_independent_final_export_capture,
)
from test_benchmark_sweep_support import exact_wide_capture, set_phase
from test_review_reference_support import add_synthetic_full_reference


def exact_chain_capture(mode: str, duration: float) -> dict:
    base = expect_valid("v4_bounded_i64_ck.json")
    builder = (
        as_residue_current_chain_capture
        if mode == "resident"
        else as_exact_wide_residue_chain_independent_final_export_capture
        if mode == "independent"
        else as_residue_chain_final_export_capture
    )
    fixture = builder(base)
    if mode == "resident":
        fixture["plan_packing"] = dict(
            source="rns8_get_plan_packing_info",
            backend="ck",
            semantics="exact_wide_signed",
            input_domain_name="resident_rns",
            output_domain_name="resident_rns",
            next_op_hint="rns-gemm",
            uses_resident_matrix_inputs=True,
            uses_transient_pack_workspace=False,
            uses_matrix_engine_pack_layout=True,
            reusable_prepack_cache_available=False,
            production_prepack_cache_available=False,
            output_host_current=False,
            output_device_current=True,
            source_versioned_inputs=True,
            same_source_version_pack_elision_available=True,
            flags=0,
            next_op_flags=0,
            a_pack_workspace_bytes=0,
            b_pack_workspace_bytes=0,
            accumulator_workspace_bytes=0,
            library_workspace_bytes=0,
            total_transient_workspace_bytes=0,
        )
        fixture["plan_lowering"] = dict(
            source="rns8_get_plan_lowering_info",
            operation="gemm",
            semantic_contract="exact_wide_signed",
            backend_family="ck",
            input_domain="resident_rns",
            output_domain="resident_rns",
            desired_output="resident_rns",
            schedule_strategy="fixed_prefix",
            packing_strategy="resident_rns",
            reuse_strategy="none",
            conversion_strategy="deferred_export",
            lowering_path="synthetic_fixture",
            final_export_available=True,
            rns_continuation_available=True,
            native_continuation_available=False,
            native_to_rns_available=False,
            reusable_b_prepack_available=False,
        )
    if mode == "cpu":
        cpu = exact_wide_capture("cpu-reference", duration)
        for field in (
            "backend_requested",
            "backend_selected",
            "selected_kernel",
            "backend_metadata",
            "device",
            "target_variant",
            "int32_accumulator_safety",
        ):
            if field in cpu:
                fixture[field] = copy.deepcopy(cpu[field])
    fixture["warmups"], fixture["repeats"] = 3, 9
    set_phase(fixture, duration)
    fixture["avg_per_modulus_gemm_estimate_us"] = fixture["avg_rns_gemm_us"] / 20
    fixture["checksum_u64"] = 12345
    return add_synthetic_full_reference(fixture)
