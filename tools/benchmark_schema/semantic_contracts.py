"""Focused semantic contract validators for benchmark_schema.core."""

from __future__ import annotations

import re

from evidence_scalars import text_choice

from .core_shared import (
    _is_int,
)
from .semantic_bounded import validate_bounded_contract
from .semantic_context import SemanticContext
from .semantic_exact_wide import validate_exact_wide_contract
from .semantic_finite import validate_finite_u8_contract
from .semantic_wrap64 import validate_wrap64_contract


def _validate_intermediate_materialization(self) -> None:
    value = self.data.get("exact_wide_intermediate_materialization")
    if value is None:
        return
    label = "exact_wide_intermediate_materialization"
    fields = {
        "schema",
        "limb_count",
        "prefix",
        "basis_product",
        "logical_ld",
        "host_payload_bytes",
        "exports_per_repeat",
        "final_output_limb_count",
        "scope",
    }
    if not isinstance(value, dict) or set(value) != fields:
        self._error(label + " must contain the native materialization fields")
        return
    if (
        self.data.get("semantics") not in {"exact_wide_signed", "exact_wide_unsigned"}
        or self.data.get("residue_chain_independent_final_export") is not True
        or self._residue_chain_length() <= 1
    ):
        self._error(label + " is only valid for independent exact-wide chains")
    for key in fields - {"schema", "basis_product", "scope"}:
        if not _is_int(value.get(key)) or value[key] < 1:
            self._error(label + " requires positive native integer " + key)
            return
    product_text = value.get("basis_product")
    if (
        not isinstance(product_text, str)
        or re.fullmatch(r"[1-9][0-9]{0,616}", product_text) is None
    ):
        self._error(label + " basis_product must be bounded positive decimal")
        return
    product = int(product_text)
    limbs = ((product - 1).bit_length() + 63) // 64
    if product <= 1 or limbs > 32 or value["limb_count"] != limbs:
        self._error(label + " limb_count must cover the full declared CRT representative")
    if (
        value["schema"] != "rns8_exact_wide_intermediate_materialization_v1"
        or value["scope"] != "private_full_selected_crt_representative_not_final_output_width"
        or value["prefix"] != self.data.get("selected_prefix", self.data.get("prefix"))
        or value["logical_ld"] != self.data.get("n")
        or value["exports_per_repeat"] != self._residue_chain_length() - 1
        or value["final_output_limb_count"] != self.data.get("exact_wide_limb_count")
    ):
        self._error(label + " has stale schema, width, prefix or chain geometry")
    m, n = self.data.get("m"), self.data.get("n")
    if _is_int(m) and _is_int(n) and value["host_payload_bytes"] != m * n * limbs * 8:
        self._error(label + " host_payload_bytes must match the compact private limb window")


def validate_semantic_contract(self) -> None:
    _validate_intermediate_materialization(self)
    semantics = self.data.get("semantics")
    prefix = self.data.get("prefix")
    packed_layout = self.data.get("packed_layout_version")
    schedule = self.data.get("schedule_metadata")
    backend_metadata = self.data.get("backend_metadata")
    k = self.data.get("k")
    k_value = int(k) if _is_int(k) and k > 0 else 0
    bound_mode = self.data.get("bound_mode", "global")
    residue_chain_length = self._residue_chain_length()
    residue_output_mode = self._residue_output_mode()
    residue_chain_final_export = self.data.get("residue_chain_final_export")
    residue_chain_independent_final_export = self.data.get("residue_chain_independent_final_export")
    status_check = self.data.get("exact_wide_export_status_check")
    prefix_policy = self.data.get("contract_prefix_policy")
    metadata = self.data.get("timing_metadata")
    if not text_choice(bound_mode, {"global", "per_tile"}):
        self._error("bound_mode must be global or per_tile")
    if residue_chain_final_export is not None and not isinstance(residue_chain_final_export, bool):
        self._error("residue_chain_final_export must be a boolean")
    if residue_chain_independent_final_export is not None and not isinstance(
        residue_chain_independent_final_export, bool
    ):
        self._error("residue_chain_independent_final_export must be a boolean")
    if status_check is not None and not text_choice(
        semantics, {"exact_wide_signed", "exact_wide_unsigned"}
    ):
        self._error("exact_wide_export_status_check must be null outside exact-wide captures")
    rns_chain_semantics = {"bounded_i64", "bounded_u64", "exact_wide_signed", "exact_wide_unsigned"}
    if residue_chain_length > 1 and semantics not in rns_chain_semantics:
        self._error(
            "residue_chain_length > 1 captures must use bounded or exact-wide RNS semantics"
        )
    if residue_output_mode == "residue_current_rns" and residue_chain_length <= 1:
        self._error("residue_output_mode=residue_current_rns requires residue_chain_length > 1")
    if residue_chain_length == 1 and residue_output_mode != "host_export":
        self._error("residue_chain_length=1 captures must use residue_output_mode=host_export")
    if residue_chain_length > 1 and isinstance(residue_chain_final_export, bool):
        expected_final_export = residue_output_mode == "host_export"
        if residue_chain_final_export is not expected_final_export:
            self._error(
                "residue_chain_final_export must match residue_output_mode for chain captures"
            )
    if residue_chain_independent_final_export is True:
        if not text_choice(
            semantics, {"bounded_i64", "bounded_u64", "exact_wide_signed", "exact_wide_unsigned"}
        ):
            self._error(
                "independent final-output residue-chain captures require bounded or exact-wide RNS semantics"
            )
        if residue_chain_length <= 1 or residue_output_mode != "host_export":
            self._error(
                "independent final-output residue-chain captures must use host_export residue_chain_length > 1"
            )
        if residue_chain_final_export is not True:
            self._error(
                "independent final-output residue-chain captures must set residue_chain_final_export=true"
            )
        if self._benchmark_execution_mode() != "residue_chain_independent_final_host_export":
            self._error(
                "independent final-output residue-chain captures must use "
                "benchmark_execution_mode=residue_chain_independent_final_host_export"
            )
        if self.data.get("pack_mode") != "per_repeat_repack":
            self._error(
                "independent final-output residue-chain captures must use pack_mode=per_repeat_repack"
            )
        if self.data.get("reuse_packed_inputs") is not False:
            self._error(
                "independent final-output residue-chain captures must not use packed-input reuse"
            )
        if self.data.get("prepack_reuse_operands") not in (None, []):
            self._error(
                "independent final-output residue-chain captures must not declare reused operands"
            )
        if self.data.get("prepack_reuse_strategy") not in {None, "none"}:
            self._error(
                "independent final-output residue-chain captures must use prepack_reuse_strategy=none"
            )
        if isinstance(metadata, dict):
            if metadata.get("residue_chain_independent_final_export") is not True:
                self._error(
                    "independent final-output residue-chain captures must set "
                    "timing_metadata.residue_chain_independent_final_export=true"
                )
            if metadata.get("residue_chain_final_export") is not True:
                self._error(
                    "independent final-output residue-chain captures must set "
                    "timing_metadata.residue_chain_final_export=true"
                )
    elif self._benchmark_execution_mode() == "residue_chain_independent_final_host_export":
        self._error(
            "benchmark_execution_mode=residue_chain_independent_final_host_export requires "
            "residue_chain_independent_final_export=true"
        )
    ctx = SemanticContext(
        semantics=semantics,
        prefix=prefix,
        packed_layout=packed_layout,
        schedule=schedule,
        backend_metadata=backend_metadata,
        k_value=k_value,
        bound_mode=bound_mode,
        residue_chain_length=residue_chain_length,
        residue_output_mode=residue_output_mode,
        status_check=status_check,
        prefix_policy=prefix_policy,
        metadata=metadata,
    )
    if semantics == "wrap_u64_mod_2_64":
        validate_wrap64_contract(self, ctx)
    elif text_choice(semantics, {"bounded_i64", "bounded_u64"}):
        validate_bounded_contract(self, ctx)
    elif text_choice(semantics, {"exact_wide_signed", "exact_wide_unsigned"}):
        validate_exact_wide_contract(self, ctx)
    elif text_choice(semantics, {"finite_ring_u8", "finite_field_u8"}):
        validate_finite_u8_contract(self, ctx)
    elif isinstance(semantics, str):
        self._error(f"unsupported benchmark semantics {semantics}")

    applicable = self.data.get("per_modulus_gemm_estimate_applicable")
    if applicable is not None and not isinstance(applicable, bool):
        self._error("per_modulus_gemm_estimate_applicable must be a boolean")
    elif isinstance(applicable, bool) and _is_int(prefix):
        expected_applicable = (
            prefix > 0
            and not self._is_public_oneshot_capture()
            and not (
                text_choice(semantics, {"bounded_i64", "bounded_u64"}) and bound_mode == "per_tile"
            )
            and not self._is_direct_hip_vector_to_rns_chain_capture()
            and not self._is_host_api_batch_capture()
            and not self._is_grouped_dispatch_capture()
            and not self._is_hip_graph_replay_capture()
            and not self._is_streaming_overlap_capture()
            and self.data.get("backend_selected") != "hip-vector-alu-int64"
        )
        if applicable != expected_applicable:
            self._error("per_modulus_gemm_estimate_applicable must match the fixed-prefix contract")
