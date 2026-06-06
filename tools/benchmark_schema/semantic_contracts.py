"""Focused semantic contract validators for benchmark_schema.core."""

from __future__ import annotations

from .core_shared import _is_int
from .semantic_bounded import validate_bounded_contract
from .semantic_exact_wide import validate_exact_wide_contract
from .semantic_finite import validate_finite_u8_contract
from .semantic_wrap64 import validate_wrap64_contract

def validate_semantic_contract(self) -> None:
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
    if bound_mode not in {"global", "per_tile"}:
        self._error("bound_mode must be global or per_tile")
    if residue_chain_final_export is not None and not isinstance(residue_chain_final_export, bool):
        self._error("residue_chain_final_export must be a boolean")
    if residue_chain_independent_final_export is not None and not isinstance(
        residue_chain_independent_final_export, bool
    ):
        self._error("residue_chain_independent_final_export must be a boolean")
    if status_check is not None and semantics not in {"exact_wide_signed", "exact_wide_unsigned"}:
        self._error("exact_wide_export_status_check must be null outside exact-wide captures")
    rns_chain_semantics = {"bounded_i64", "bounded_u64", "exact_wide_signed", "exact_wide_unsigned"}
    if residue_chain_length > 1 and semantics not in rns_chain_semantics:
        self._error("residue_chain_length > 1 captures must use bounded or exact-wide RNS semantics")
    if residue_output_mode == "residue_current_rns" and residue_chain_length <= 1:
        self._error("residue_output_mode=residue_current_rns requires residue_chain_length > 1")
    if residue_chain_length == 1 and residue_output_mode != "host_export":
        self._error("residue_chain_length=1 captures must use residue_output_mode=host_export")
    if residue_chain_length > 1 and isinstance(residue_chain_final_export, bool):
        expected_final_export = residue_output_mode == "host_export"
        if residue_chain_final_export is not expected_final_export:
            self._error("residue_chain_final_export must match residue_output_mode for chain captures")
    if residue_chain_independent_final_export is True:
        if semantics not in {"bounded_i64", "bounded_u64", "exact_wide_signed", "exact_wide_unsigned"}:
            self._error("independent final-output residue-chain captures require bounded or exact-wide RNS semantics")
        if residue_chain_length <= 1 or residue_output_mode != "host_export":
            self._error("independent final-output residue-chain captures must use host_export residue_chain_length > 1")
        if residue_chain_final_export is not True:
            self._error("independent final-output residue-chain captures must set residue_chain_final_export=true")
        if self._benchmark_execution_mode() != "residue_chain_independent_final_host_export":
            self._error(
                "independent final-output residue-chain captures must use "
                "benchmark_execution_mode=residue_chain_independent_final_host_export"
            )
        if self.data.get("pack_mode") != "per_repeat_repack":
            self._error("independent final-output residue-chain captures must use pack_mode=per_repeat_repack")
        if self.data.get("reuse_packed_inputs") is not False:
            self._error("independent final-output residue-chain captures must not use packed-input reuse")
        if self.data.get("prepack_reuse_operands") not in (None, []):
            self._error("independent final-output residue-chain captures must not declare reused operands")
        if self.data.get("prepack_reuse_strategy") not in {None, "none"}:
            self._error("independent final-output residue-chain captures must use prepack_reuse_strategy=none")
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
    ctx = {
        "semantics": semantics,
        "prefix": prefix,
        "packed_layout": packed_layout,
        "schedule": schedule,
        "backend_metadata": backend_metadata,
        "k_value": k_value,
        "bound_mode": bound_mode,
        "residue_chain_length": residue_chain_length,
        "residue_output_mode": residue_output_mode,
        "status_check": status_check,
        "prefix_policy": prefix_policy,
        "metadata": metadata,
    }
    if semantics == "wrap_u64_mod_2_64":
        validate_wrap64_contract(self, ctx)
    elif semantics in {"bounded_i64", "bounded_u64"}:
        validate_bounded_contract(self, ctx)
    elif semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
        validate_exact_wide_contract(self, ctx)
    elif semantics in {"finite_ring_u8", "finite_field_u8"}:
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
            and not (semantics in {"bounded_i64", "bounded_u64"} and bound_mode == "per_tile")
            and not self._is_direct_hip_vector_to_rns_chain_capture()
            and not self._is_host_api_batch_capture()
            and not self._is_grouped_dispatch_capture()
            and not self._is_hip_graph_replay_capture()
            and not self._is_streaming_overlap_capture()
            and self.data.get("backend_selected") != "hip-vector-alu-int64"
        )
        if applicable != expected_applicable:
            self._error("per_modulus_gemm_estimate_applicable must match the fixed-prefix contract")
