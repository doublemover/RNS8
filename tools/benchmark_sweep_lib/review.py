from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evidence_database_lib.isa import lookup_isa_resources

from .capture_metadata import (
    backend_id,
    backend_family_id,
    candidate_source_metadata,
    backend_requires_gpu_target,
    capture_backend_metadata,
    capture_bound_source,
    capture_compiler,
    capture_contract_key,
    capture_device,
    capture_execution_mode,
    capture_export_variant_name,
    capture_grouped_dispatch_strategy,
    capture_grouped_dispatch_task_count,
    capture_host_api_batch_size,
    capture_hip_toolchain,
    capture_pack_mode,
    capture_reconstruction_variant_name,
    capture_timing_metadata,
    capture_checksum_policy,
    group_source_metadata,
    median_phase,
    normalized_compiler_identity,
    normalized_identity_text,
    normalized_positive_int,
    normalized_target_id,
    selected_kernel,
)
from .config import (
    PHASES,
    RELEASE_MIN_REPEATS,
    RELEASE_MIN_WARMUPS,
    REVIEW_SCHEMA_VERSION,
    WRAP64_ROCWMMA_CANDIDATE_BACKEND,
)

DIAGNOSTIC_PHASES = tuple(PHASES) + ("pack_a", "pack_b")


def required_baselines(semantics: Any) -> list[str]:
    if semantics in {"bounded_i64", "bounded_u64"}:
        return ["cpu-reference", "hip-direct", "hip-vector-alu-int64"]
    if semantics in {"finite_ring_u8", "finite_field_u8"}:
        return ["cpu-reference", "hip-direct"]
    if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
        return ["cpu-reference", "hip-direct"]
    if semantics == "wrap_u64_mod_2_64":
        return ["wrap64-byte-limb", "hip-direct"]
    return []


def residue_chain_group(items: list[dict[str, Any]]) -> bool:
    for item in items:
        value = item.get("residue_chain_length", 1)
        if isinstance(value, int) and not isinstance(value, bool) and value > 1:
            return True
    return False


REUSE_EVIDENCE_PROMOTION_SCOPES = {"explicit_reuse_contract_only", "reuse_contract_evidence_only"}
GRAPH_EVIDENCE_PROMOTION_SCOPES = {"hip_graph_replay_evidence_only"}
CORRECTNESS_ANCHOR_REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}
REUSE_PACK_MODES = {"prepacked_reuse", "prepacked_reuse_a", "prepacked_reuse_b"}
GRAPH_REPLAY_EXECUTION_MODES = {
    "hip_graph_replay_resident_rns_chain",
    "hip_graph_replay_bounded_pack_gemm_export",
    "hip_graph_replay_finite_u8_pack_gemm_export",
    "hip_graph_replay_wrap64_pack_gemm_export",
}
AMDGPU_BUILTIN_MATRIX_FAMILIES = {"mfma", "smfmac", "wmma", "swmmac"}
FINAL_OUTPUT_EXPORT_SEMANTICS = {
    "bounded_i64",
    "bounded_u64",
    "exact_wide_signed",
    "exact_wide_unsigned",
}
ROUTE_METADATA_BLOCKERS = {
    "missing_direct_hip_skinny_gemv_kernel_identity",
    "missing_direct_hip_skinny_gemv_execution_mode",
    "missing_vector_alu_skinny_gemv_kernel_identity",
    "missing_vector_alu_skinny_gemv_execution_mode",
    "missing_skinny_gemv_gpu_event_phase",
    "missing_native_to_rns_bridge_forced_metadata",
    "missing_native_to_rns_handoff_scope",
    "missing_native_to_rns_phase_availability",
    "missing_vector_to_rns_chain_metadata",
    "vector_to_rns_chain_control_mode_mismatch",
    "missing_vector_to_rns_chain_producer_backend",
    "missing_vector_to_rns_chain_consumer_backend",
    "missing_vector_to_rns_chain_phase_scope",
    "missing_sparse_a_4_to_2_scenario_contract",
    "sparse_a_k_not_divisible_by_4",
    "missing_sparse_a_4_to_2_input_distribution",
    "missing_sparse_a_runtime_kernel_identity",
    "dense_sparse_a_baseline_used_sparse_kernel",
    "missing_sparse_a_matrix_instruction_sparsity",
    "missing_sparse_a_matrix_instruction_family",
    "missing_sparse_a_4_to_2_autotune_contract",
    "amdgpu_builtin_tile_variant_kernel_mismatch",
    "amdgpu_builtin_tile_variant_family_mismatch",
    "amdgpu_builtin_tile_variant_shape_mismatch",
    "amdgpu_builtin_tile_variant_dtype_mismatch",
}
NATIVE_TO_RNS_EXECUTION_MODES = {
    "auto_native_to_rns_bridge",
    "vector_native_to_direct_rns_chain",
    "vector_native_host_export_repack_direct_rns_chain",
}


def _truthy_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _autotune_key_has_all(metadata: dict[str, Any], tokens: tuple[str, ...]) -> bool:
    key = metadata.get("autotune_key")
    return isinstance(key, str) and all(token in key for token in tokens)


def requires_amdgpu_builtin_matrix_isa_proof(backend_family: str, metadata: dict[str, Any]) -> bool:
    family = metadata.get("matrix_instruction_family")
    return backend_family == "amdgpu-builtins" and isinstance(family, str) and family in AMDGPU_BUILTIN_MATRIX_FAMILIES


def expected_amdgpu_builtin_matrix_mnemonic(capture: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    kernel = str(selected_kernel(capture) or metadata.get("selected_kernel") or "")
    kernel_tokens = (
        ("smfmac_i32_32x32x32_i8", "v_smfmac_i32_32x32x32_i8"),
        ("smfmac_i32_16x16x64_i8", "v_smfmac_i32_16x16x64_i8"),
        ("mfma_i32_32x32x16_i8", "v_mfma_i32_32x32x16_i8"),
        ("mfma_i32_16x16x32_i8", "v_mfma_i32_16x16x32_i8"),
        ("swmmac_i32_16x16x32_iu4", "v_swmmac_i32_16x16x32_iu4"),
        ("swmmac_i32_16x16x32_iu8", "v_swmmac_i32_16x16x32_iu8"),
        ("wmma_i32_16x16x16_iu4", "v_wmma_i32_16x16x16_iu4"),
        ("wmma_i32_16x16x16_iu8", "v_wmma_i32_16x16x16_iu8"),
    )
    for token, mnemonic in kernel_tokens:
        if token in kernel:
            return mnemonic
    family = metadata.get("matrix_instruction_family")
    shape = metadata.get("matrix_instruction_shape")
    dtype = metadata.get("matrix_instruction_dtype")
    if (
        isinstance(family, str)
        and family in AMDGPU_BUILTIN_MATRIX_FAMILIES
        and isinstance(shape, str)
        and shape
        and isinstance(dtype, str)
        and dtype
    ):
        return f"v_{family}_i32_{shape}_{dtype}"
    return None


def final_output_export_metadata_blockers(
    capture: dict[str, Any],
    *,
    semantics: Any,
    accelerator: bool,
) -> list[str]:
    if not accelerator or semantics not in FINAL_OUTPUT_EXPORT_SEMANTICS:
        return []
    exact_output = capture.get("exact_output_contract")
    if isinstance(exact_output, dict) and exact_output.get("output_domain_after_measured_repeats") == "rns_residue_current":
        return []
    if capture.get("residue_output_mode", "host_export") != "host_export":
        return []

    blockers: list[str] = []
    if not isinstance(exact_output, dict):
        blockers.append("missing_final_output_contract_metadata")
    else:
        if exact_output.get("requested_final_output") not in {"native_i64_u64_host", "exact_wide_limb_host"}:
            blockers.append("final_output_contract_not_host_output")
        if not isinstance(exact_output.get("kernel_identity"), str) or not exact_output.get("kernel_identity"):
            blockers.append("missing_final_output_kernel_identity")

    export_variant = capture.get("export_variant")
    if not isinstance(export_variant, dict):
        blockers.append("missing_export_variant_metadata")
    else:
        if not isinstance(export_variant.get("selected_kernel"), str) or not export_variant.get("selected_kernel"):
            blockers.append("missing_export_kernel_identity")
        if export_variant.get("final_output_mode") != "final_host_output":
            blockers.append("export_variant_not_final_host_output")
        if export_variant.get("output_layout") in {None, "unknown"}:
            blockers.append("missing_export_output_layout")
        if export_variant.get("d2h_policy") in {None, "unknown"}:
            blockers.append("missing_export_d2h_policy")

    reconstruction_variant = capture.get("reconstruction_variant")
    if not isinstance(reconstruction_variant, dict):
        blockers.append("missing_reconstruction_variant_metadata")
    else:
        if (
            not isinstance(reconstruction_variant.get("kernel_identity"), str)
            or not reconstruction_variant.get("kernel_identity")
        ):
            blockers.append("missing_reconstruction_kernel_identity")
        prefix_count = reconstruction_variant.get("prefix_count")
        if not isinstance(prefix_count, int) or isinstance(prefix_count, bool) or prefix_count < 0:
            blockers.append("missing_reconstruction_prefix_count")
    return blockers


def skinny_gemv_metadata_blockers(capture: dict[str, Any], *, semantics: Any, backend_family: str) -> list[str]:
    if semantics not in {"bounded_i64", "bounded_u64"} or capture.get("n") != 1:
        return []
    if backend_family not in {"hip-direct", "hip-vector-alu-int64"}:
        return []

    blockers: list[str] = []
    kernel = selected_kernel(capture)
    mode = capture_execution_mode(capture)
    if backend_family == "hip-direct":
        if "gemv_n1" not in kernel:
            blockers.append("missing_direct_hip_skinny_gemv_kernel_identity")
        if mode != "direct_hip_skinny_gemv_n1_resident_rns":
            blockers.append("missing_direct_hip_skinny_gemv_execution_mode")
    else:
        if "gemv_n1" not in kernel:
            blockers.append("missing_vector_alu_skinny_gemv_kernel_identity")
        if "gemv_n1" not in mode:
            blockers.append("missing_vector_alu_skinny_gemv_execution_mode")

    timing = capture_timing_metadata(capture)
    phase_order = timing.get("gpu_event_phase_order")
    if not isinstance(phase_order, list) or not any("gemv" in str(phase) for phase in phase_order):
        blockers.append("missing_skinny_gemv_gpu_event_phase")
    return blockers


def native_to_rns_handoff_metadata_blockers(
    capture: dict[str, Any],
    *,
    semantics: Any,
    backend_family: str,
) -> list[str]:
    mode = capture_execution_mode(capture)
    if semantics not in {"bounded_i64", "bounded_u64"} or backend_family != "hip-direct":
        return []
    if mode not in NATIVE_TO_RNS_EXECUTION_MODES:
        return []

    blockers: list[str] = []
    timing = capture_timing_metadata(capture)
    phase_availability = timing.get("phase_availability")
    if not isinstance(phase_availability, dict):
        blockers.append("missing_native_to_rns_phase_availability")

    if mode == "auto_native_to_rns_bridge":
        if timing.get("native_to_rns_bridge_forced") is not True:
            blockers.append("missing_native_to_rns_bridge_forced_metadata")
        phase = phase_availability.get("native_to_rns_bridge") if isinstance(phase_availability, dict) else None
        if not isinstance(phase, dict) or phase.get("scope") != "device_native_to_rns_conversion_inside_rns_gemm":
            blockers.append("missing_native_to_rns_handoff_scope")
    else:
        if timing.get("vector_to_rns_chain") is not True:
            blockers.append("missing_vector_to_rns_chain_metadata")
        expected_control = (
            "host_export_repack_control"
            if mode == "vector_native_host_export_repack_direct_rns_chain"
            else "fused_device_native_to_rns"
        )
        if timing.get("vector_to_rns_chain_control_mode") != expected_control:
            blockers.append("vector_to_rns_chain_control_mode_mismatch")
        if timing.get("vector_to_rns_chain_producer_backend") != "hip-vector-alu-int64":
            blockers.append("missing_vector_to_rns_chain_producer_backend")
        if timing.get("vector_to_rns_chain_consumer_backend") != "hip-direct":
            blockers.append("missing_vector_to_rns_chain_consumer_backend")
        phase = phase_availability.get("vector_to_rns_chain") if isinstance(phase_availability, dict) else None
        if not isinstance(phase, dict):
            blockers.append("missing_vector_to_rns_chain_phase_scope")
    return blockers


def sparse_a_4_to_2_contract_blockers(
    capture: dict[str, Any],
    *,
    backend: str,
    backend_family: str,
    metadata: dict[str, Any],
) -> list[str]:
    scenario = capture.get("scenario_metadata")
    sparse_scenario = isinstance(scenario, dict) and scenario.get("sparse_a_4_to_2") is True
    kernel = selected_kernel(capture)
    sparse_runtime = backend.endswith("-sparse-a-runtime")
    dense_sparse_baseline = backend.endswith("-dense-sparse-a-input")
    if not sparse_scenario and "sparse_a" not in kernel and not sparse_runtime and not dense_sparse_baseline:
        return []

    blockers: list[str] = []
    if not sparse_scenario:
        blockers.append("missing_sparse_a_4_to_2_scenario_contract")
    if not _truthy_int(capture.get("k")) or capture["k"] % 4 != 0:
        blockers.append("sparse_a_k_not_divisible_by_4")
    if "sparse_a_4_to_2" not in str(capture.get("input_distribution", "")):
        blockers.append("missing_sparse_a_4_to_2_input_distribution")
    if sparse_runtime and "sparse_a" not in kernel:
        blockers.append("missing_sparse_a_runtime_kernel_identity")
    if dense_sparse_baseline and "sparse_a" in kernel:
        blockers.append("dense_sparse_a_baseline_used_sparse_kernel")
    if backend_family == "amdgpu-builtins" and sparse_runtime:
        if metadata.get("matrix_instruction_sparsity") != "structured_4_2":
            blockers.append("missing_sparse_a_matrix_instruction_sparsity")
        if metadata.get("matrix_instruction_family") not in {"smfmac", "swmmac"}:
            blockers.append("missing_sparse_a_matrix_instruction_family")

    required_key_tokens = (
        "sparse_contract=a_4_to_2_structured_k_v1",
        "sparse_operand=A",
        "sparse_group_size=4",
        "sparse_nonzeros_per_group=2",
        "sparse_index_layout=canonical_2bit_k_groups_v1",
        "dense_operand=B",
    )
    if not _autotune_key_has_all(metadata, required_key_tokens):
        blockers.append("missing_sparse_a_4_to_2_autotune_contract")
    return blockers


def amdgpu_builtin_tile_variant_blockers(capture: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    if backend_family_id(backend_id(capture)) != "amdgpu-builtins":
        return []
    tile_variant = capture.get("tile_shape_variant")
    if not isinstance(tile_variant, dict):
        return []
    name = tile_variant.get("name")
    if not isinstance(name, str):
        return []

    expectations = {
        "amdgpu-cdna3-mfma-16x16x32": ("mfma_i32_16x16x32_i8", "mfma", "16x16x32", "i8"),
        "amdgpu-cdna3-mfma-32x32x16": ("mfma_i32_32x32x16_i8", "mfma", "32x32x16", "i8"),
    }
    expected = expectations.get(name)
    if expected is None:
        return []
    kernel_token, family, shape, dtype = expected
    blockers: list[str] = []
    if kernel_token not in selected_kernel(capture):
        blockers.append("amdgpu_builtin_tile_variant_kernel_mismatch")
    if metadata.get("matrix_instruction_family") != family:
        blockers.append("amdgpu_builtin_tile_variant_family_mismatch")
    if metadata.get("matrix_instruction_shape") != shape:
        blockers.append("amdgpu_builtin_tile_variant_shape_mismatch")
    if metadata.get("matrix_instruction_dtype") != dtype:
        blockers.append("amdgpu_builtin_tile_variant_dtype_mismatch")
    return blockers


def correctness_anchor_reference_capture(capture: dict[str, Any]) -> bool:
    if backend_family_id(backend_id(capture)) not in CORRECTNESS_ANCHOR_REFERENCE_BACKENDS:
        return False
    cpu_parallel = capture.get("cpu_parallel")
    if not isinstance(cpu_parallel, dict):
        return False
    return (
        cpu_parallel.get("correctness_anchor") is True
        or cpu_parallel.get("reference_mode") == "correctness-anchor"
    )


def autotune_promotable_scope(capture: dict[str, Any]) -> bool:
    scope = capture_scenario_promotion_scope(capture)
    return scope is None or scope == "release_review_candidate"


def reuse_evidence_group(items: list[dict[str, Any]]) -> bool:
    if not items:
        return False
    if not all(capture_pack_mode(item) != "per_repeat_repack" for item in items):
        return False
    scopes = {capture_scenario_promotion_scope(item) for item in items}
    return bool(scopes & REUSE_EVIDENCE_PROMOTION_SCOPES)


def required_baselines_for_group(semantics: Any, items: list[dict[str, Any]]) -> list[str]:
    if not any(autotune_promotable_scope(item) for item in items):
        return []
    if reuse_evidence_group(items) or any(capture_prepacked_reuse(item) for item in items):
        return []
    scopes = {capture_scenario_promotion_scope(item) for item in items}
    if scopes & GRAPH_EVIDENCE_PROMOTION_SCOPES or any(capture_hip_graph_replay(item) for item in items):
        return []
    required = required_baselines(semantics)
    if semantics in {"bounded_i64", "bounded_u64"} and residue_chain_group(items):
        required = [backend for backend in required if backend != "hip-vector-alu-int64"]
    return required


def phase_ratios(item: dict[str, Any], direct: dict[str, Any] | None, vector: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in DIAGNOSTIC_PHASES:
        value = median_phase(item, phase)
        direct_value = median_phase(direct, phase) if direct else None
        vector_value = median_phase(vector, phase) if vector else None
        result[phase] = {
            "median_us": value,
            "speedup_vs_direct_hip": (direct_value / value) if direct_value and value else None,
            "speedup_vs_vector_alu": (vector_value / value) if vector_value and value else None,
        }
    return result


def phase_medians_for_capture(capture: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for phase in DIAGNOSTIC_PHASES:
        value = median_phase(capture, phase)
        if value is not None:
            result[phase] = value
    return result


def scenario_values(items: list[dict[str, Any]], key: str) -> list[str]:
    values = set()
    for item in items:
        scenario = item.get("scenario_metadata")
        if not isinstance(scenario, dict):
            continue
        value = scenario.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    return sorted(values)


def shape_family(m: Any, n: Any, k: Any) -> str:
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (m, n, k)):
        return "unknown"
    if n == 1:
        return "skinny_n1"
    if n <= 16:
        return "skinny_n_le16"
    if m == n == k:
        if m <= 128:
            return "small_square"
        if m <= 512:
            return "medium_square"
        if m <= 1024:
            return "large_square"
        return "very_large_square"
    if m == n:
        return "square_mn_rectangular_k"
    if max(m, n, k) >= 2048:
        return "large_rectangular"
    return "rectangular"


def phase_ratio_summary_for_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    diagnostics = candidate.get("phase_diagnostics")
    phase_speedups: dict[str, Any] = {}
    slowest_phase = None
    slowest_ratio = 0.0
    if isinstance(diagnostics, dict):
        for phase in PHASES:
            item = diagnostics.get(phase)
            if not isinstance(item, dict):
                continue
            speedup = item.get("speedup_vs_direct_hip")
            phase_speedups[phase] = speedup
            if isinstance(speedup, (int, float)) and speedup > 0:
                candidate_over_direct = 1.0 / float(speedup)
                if candidate_over_direct > 1.0 and candidate_over_direct > slowest_ratio:
                    slowest_ratio = candidate_over_direct
                    slowest_phase = phase
    return {
        "backend": candidate.get("backend"),
        "selected_kernel": candidate.get("selected_kernel"),
        "slowest_phase_vs_direct_hip": slowest_phase,
        "slowest_phase_candidate_over_direct": slowest_ratio if slowest_phase else None,
        "phase_speedups_vs_direct_hip": phase_speedups,
    }


def compact_candidate_route(group: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    shape = group.get("shape") if isinstance(group.get("shape"), dict) else {}
    return {
        "semantics": group.get("semantics"),
        "shape": shape,
        "shape_family": group.get("shape_family"),
        "scenario_families": group.get("scenario_families") or [],
        "backend": candidate.get("backend"),
        "selected_kernel": candidate.get("selected_kernel"),
        "median_end_to_end_us": candidate.get("median_end_to_end_us"),
        "selection_end_to_end_us": candidate.get("selection_end_to_end_us"),
        "speedup_vs_direct_hip": candidate.get("speedup_vs_direct_hip"),
        "primary_loss_phase_vs_direct_hip": candidate.get("primary_loss_phase_vs_direct_hip"),
        "bottleneck": candidate.get("bottleneck"),
        "capture": candidate.get("capture"),
    }


def nested_counter(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = {}
    for (outer, inner), count in counter.most_common():
        nested.setdefault(outer, {})[inner] = count
    return nested


def group_scenario_families(group: dict[str, Any]) -> list[str]:
    families = group.get("scenario_families")
    if isinstance(families, list):
        result = [str(item) for item in families if isinstance(item, str) and item]
        if result:
            return result
    return ["unknown"]


def review_next_work(
    *,
    missing_baseline_count: int,
    checksum_mismatch_count: int,
    actionable_blockers: Counter[str],
    loss_phase_counts: Counter[str],
    bottleneck_counts: Counter[str],
    pack_split_counts: Counter[str],
    pack_dominant_operand_counts: Counter[str],
    pack_diagnostics: list[dict[str, Any]],
    direct_hip_wins: int,
    promotable_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if checksum_mismatch_count:
        rows.append(
            {
                "priority": "P0",
                "work": "fix_checksum_mismatches_before_performance_promotion",
                "reason": f"{checksum_mismatch_count} review groups have candidate checksum mismatches",
            }
        )
    if missing_baseline_count:
        rows.append(
            {
                "priority": "P0",
                "work": "fix_missing_required_baselines_or_reclassify_invalid_scenarios",
                "reason": f"{missing_baseline_count} review groups are missing required baselines",
            }
        )
    for blocker, work in [
        ("reuse_not_faster_than_same_backend_setup_inclusive", "reduce_prepack_setup_or_reuse_steady_state_cost"),
        ("reuse_not_faster_than_best_nonreuse_setup_inclusive", "reduce_prepack_setup_or_raise_declared_reuse_count_with_contract_evidence"),
        ("graph_not_faster_than_non_graph_setup_inclusive", "improve_graph_replay_break_even_or_keep_graph_benchmark_only"),
        ("missing_graph_setup_inclusive_timing", "fix_graph_setup_inclusive_timing_metadata"),
        ("missing_amdgpu_builtin_matrix_isa_histogram", "attach_compiled_matrix_isa_reports_before_builtin_promotion"),
        (
            "missing_selected_amdgpu_builtin_matrix_instruction",
            "compile_selected_amdgpu_builtin_kernel_with_expected_matrix_instruction",
        ),
        ("missing_final_output_contract_metadata", "attach_final_output_contract_metadata_before_promotion"),
        ("missing_final_output_kernel_identity", "attach_final_output_kernel_identity_before_promotion"),
        ("missing_export_variant_metadata", "attach_export_variant_metadata_before_promotion"),
        ("missing_export_kernel_identity", "attach_export_kernel_identity_before_promotion"),
        ("export_variant_not_final_host_output", "reclassify_residue_current_or_capture_final_host_output"),
        ("missing_export_output_layout", "attach_export_output_layout_before_promotion"),
        ("missing_export_d2h_policy", "attach_export_d2h_policy_before_promotion"),
        ("missing_reconstruction_variant_metadata", "attach_reconstruction_variant_metadata_before_promotion"),
        ("missing_reconstruction_kernel_identity", "attach_reconstruction_kernel_identity_before_promotion"),
        ("missing_reconstruction_prefix_count", "attach_reconstruction_prefix_count_before_promotion"),
        ("missing_direct_hip_skinny_gemv_kernel_identity", "select_direct_hip_skinny_gemv_kernel_before_n1_promotion"),
        ("missing_direct_hip_skinny_gemv_execution_mode", "route_n1_direct_hip_captures_through_skinny_gemv_mode"),
        ("missing_vector_alu_skinny_gemv_kernel_identity", "select_vector_alu_skinny_gemv_kernel_before_n1_promotion"),
        ("missing_vector_alu_skinny_gemv_execution_mode", "route_n1_vector_alu_captures_through_gemv_mode"),
        ("missing_skinny_gemv_gpu_event_phase", "attach_skinny_gemv_gpu_event_phase_timing"),
        ("missing_native_to_rns_bridge_forced_metadata", "attach_native_to_rns_bridge_forced_metadata"),
        ("missing_native_to_rns_handoff_scope", "attach_native_to_rns_handoff_phase_scope"),
        ("missing_native_to_rns_phase_availability", "attach_native_to_rns_phase_availability_metadata"),
        ("missing_vector_to_rns_chain_metadata", "attach_vector_to_rns_chain_metadata"),
        ("vector_to_rns_chain_control_mode_mismatch", "fix_vector_to_rns_chain_control_mode_metadata"),
        ("missing_vector_to_rns_chain_producer_backend", "attach_vector_to_rns_chain_producer_backend"),
        ("missing_vector_to_rns_chain_consumer_backend", "attach_vector_to_rns_chain_consumer_backend"),
        ("missing_vector_to_rns_chain_phase_scope", "attach_vector_to_rns_chain_phase_scope"),
        ("missing_sparse_a_4_to_2_scenario_contract", "route_sparse_a_captures_through_explicit_sparse_contract_scenarios"),
        ("sparse_a_k_not_divisible_by_4", "fix_sparse_a_shapes_to_k_divisible_by_4"),
        ("missing_sparse_a_4_to_2_input_distribution", "attach_sparse_a_4_to_2_input_distribution_metadata"),
        ("missing_sparse_a_runtime_kernel_identity", "compile_sparse_a_matrix_core_runtime_kernel"),
        ("dense_sparse_a_baseline_used_sparse_kernel", "separate_dense_sparse_input_baseline_from_sparse_runtime_kernel"),
        ("missing_sparse_a_matrix_instruction_sparsity", "attach_sparse_a_matrix_instruction_sparsity_metadata"),
        ("missing_sparse_a_matrix_instruction_family", "attach_sparse_a_smfmac_or_swmmac_family_metadata"),
        ("missing_sparse_a_4_to_2_autotune_contract", "attach_sparse_a_4_to_2_autotune_contract_metadata"),
        ("amdgpu_builtin_tile_variant_kernel_mismatch", "align_amdgpu_builtin_tile_variant_with_selected_kernel"),
        ("amdgpu_builtin_tile_variant_family_mismatch", "align_amdgpu_builtin_tile_variant_family_metadata"),
        ("amdgpu_builtin_tile_variant_shape_mismatch", "align_amdgpu_builtin_tile_variant_shape_metadata"),
        ("amdgpu_builtin_tile_variant_dtype_mismatch", "align_amdgpu_builtin_tile_variant_dtype_metadata"),
        ("not_faster_than_direct_hip", "optimize_accelerator_loss_phase_or_keep_direct_hip_production_winner"),
        ("not_faster_than_vector_alu", "specialize_native_vector_or_small_shape_path_before_matrix_engine_promotion"),
    ]:
        count = actionable_blockers.get(blocker, 0)
        if count:
            rows.append({"priority": "P1", "work": work, "reason": f"{blocker}={count}"})
    has_unresolved_work = (
        checksum_mismatch_count
        or missing_baseline_count
        or actionable_blockers
        or loss_phase_counts
        or direct_hip_wins
        or promotable_count == 0
    )
    if has_unresolved_work:
        split_missing = pack_split_counts.get("split_missing", 0)
        if split_missing:
            rows.append(
                {
                    "priority": "P1",
                    "work": "attach_pack_a_b_phase_split_timing_before_pack_kernel_work",
                    "reason": f"{split_missing} pack-bearing candidates still lack A/B split timing",
                }
            )
        for operand, work in [
            ("A", "optimize_a_side_pack_kernel_or_prepack_a_reuse"),
            ("B", "optimize_b_side_pack_kernel_or_prepack_b_reuse"),
            ("balanced", "optimize_balanced_pack_path_or_fuse_native_pack_gemm"),
        ]:
            count = pack_dominant_operand_counts.get(operand, 0)
            if count:
                rows.append(
                    {
                        "priority": "P1",
                        "work": work,
                        "reason": f"pack_dominant_operand:{operand}={count}",
                    }
                )
        missing_elision = sum(
            1
            for row in pack_diagnostics
            if row.get("source_versioned_inputs") is True
            and row.get("same_source_version_pack_elision_available") is not True
        )
        if missing_elision:
            rows.append(
                {
                    "priority": "P1",
                    "work": "enable_source_versioned_pack_elision_for_repeated_inputs",
                    "reason": f"{missing_elision} source-versioned pack rows cannot elide same-version repack",
                }
            )
        native_pack_rows = sum(
            1
            for row in pack_diagnostics
            if "native" in str(row.get("pack_layout") or "").lower()
            or any("native" in str(family).lower() for family in row.get("scenario_families") or [])
        )
        if native_pack_rows:
            rows.append(
                {
                    "priority": "P1",
                    "work": "implement_fused_native_pack_plus_gemm_for_native_input_workloads",
                    "reason": f"{native_pack_rows} pack-heavy rows involve native input or native-to-RNS layouts",
                }
            )
    for phase, count in loss_phase_counts.most_common(3):
        rows.append(
            {
                "priority": "P1",
                "work": f"optimize_{phase}_phase",
                "reason": f"{count} actionable accelerator candidates report {phase} as primary loss phase",
            }
        )
    if has_unresolved_work:
        for bottleneck, count in bottleneck_counts.most_common(3):
            if bottleneck == "unknown":
                continue
            rows.append(
                {
                    "priority": "P1",
                    "work": f"address_{bottleneck}",
                    "reason": f"{count} fastest routes or actionable candidates classify as {bottleneck}",
                }
            )
    if direct_hip_wins and promotable_count == 0:
        rows.append(
            {
                "priority": "P1",
                "work": "treat_direct_hip_as_current_production_winner_and_target_accelerator_loss_phases",
                "reason": f"Direct HIP is fastest production route in {direct_hip_wins} groups and no accelerator entries promoted",
            }
        )
    return rows


def build_review_summary(groups: list[dict[str, Any]], promotable_entries: list[dict[str, Any]]) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    actionable_blockers: Counter[str] = Counter()
    loss_phase_counts: Counter[str] = Counter()
    loss_phase_by_backend: Counter[tuple[str, str]] = Counter()
    loss_phase_by_semantics: Counter[tuple[str, str]] = Counter()
    loss_phase_by_shape_family: Counter[tuple[str, str]] = Counter()
    loss_phase_by_scenario_family: Counter[tuple[str, str]] = Counter()
    loss_rows: list[dict[str, Any]] = []
    bottleneck_counts: Counter[str] = Counter()
    pack_split_counts: Counter[str] = Counter()
    pack_dominant_operand_counts: Counter[str] = Counter()
    pack_diagnostic_rows: list[dict[str, Any]] = []
    production_counts: Counter[str] = Counter()
    accelerator_counts: Counter[str] = Counter()
    direct_hip_winners: list[dict[str, Any]] = []
    setup_sensitive: list[dict[str, Any]] = []
    missing_baseline_count = 0
    checksum_mismatch_count = 0

    for group in groups:
        if group.get("missing_required_baselines"):
            missing_baseline_count += 1
        if group.get("checksum_mismatches"):
            checksum_mismatch_count += 1
        production = group.get("fastest_production_route")
        if isinstance(production, dict):
            production_counts.update([str(production.get("backend"))])
            bottleneck = production.get("bottleneck") if isinstance(production.get("bottleneck"), dict) else {}
            bottleneck_class = str(bottleneck.get("class") or "unknown")
            bottleneck_counts.update([bottleneck_class])
            if production.get("backend") == "hip-direct":
                direct_hip_winners.append(compact_candidate_route(group, production))
        accelerator = group.get("fastest_accelerator_route")
        if isinstance(accelerator, dict):
            accelerator_counts.update([str(accelerator.get("backend"))])
        for candidate in group.get("candidates", []):
            blockers = candidate.get("promotion_blockers") if isinstance(candidate.get("promotion_blockers"), list) else []
            blocker_counts.update(str(item) for item in blockers)
            pack_diagnostics = candidate.get("pack_diagnostics")
            if isinstance(pack_diagnostics, dict):
                split_state = "split_available" if pack_diagnostics.get("split_available") is True else "split_missing"
                pack_split_counts.update([split_state])
                dominant = pack_diagnostics.get("dominant_operand")
                if isinstance(dominant, str) and dominant:
                    pack_dominant_operand_counts.update([dominant])
                share = pack_diagnostics.get("pack_share_of_end_to_end")
                if isinstance(share, (int, float)) and share > 0:
                    pack_diagnostic_rows.append(
                        {
                            "semantics": group.get("semantics"),
                            "shape": group.get("shape"),
                            "shape_family": group.get("shape_family"),
                            "scenario_families": group.get("scenario_families"),
                            "backend": candidate.get("backend"),
                            "selected_kernel": candidate.get("selected_kernel"),
                            "capture": candidate.get("capture"),
                            "pack_median_us": pack_diagnostics.get("pack_median_us"),
                            "pack_a_median_us": pack_diagnostics.get("pack_a_median_us"),
                            "pack_b_median_us": pack_diagnostics.get("pack_b_median_us"),
                            "pack_share_of_end_to_end": share,
                            "split_available": pack_diagnostics.get("split_available"),
                            "dominant_operand": dominant,
                            "pack_mode": pack_diagnostics.get("pack_mode"),
                            "pack_layout": pack_diagnostics.get("pack_layout"),
                            "source_versioned_inputs": pack_diagnostics.get("source_versioned_inputs"),
                            "same_source_version_pack_elision_available": pack_diagnostics.get(
                                "same_source_version_pack_elision_available"
                            ),
                        }
                    )
            route_metadata_actionable = any(str(item) in ROUTE_METADATA_BLOCKERS for item in blockers)
            if (
                (candidate.get("accelerator_backend") is True or route_metadata_actionable)
                and candidate.get("scenario_promotion_scope") in {None, "release_review_candidate"}
            ):
                actionable = [
                    str(item)
                    for item in blockers
                    if str(item) not in {"not_accelerator_backend", "scenario_scope_not_autotune_promotable"}
                ]
                actionable_blockers.update(actionable)
                phase = candidate.get("primary_loss_phase_vs_direct_hip")
                if isinstance(phase, str) and phase:
                    backend = str(candidate.get("backend") or "unknown")
                    semantics = str(group.get("semantics") or "unknown")
                    family = str(group.get("shape_family") or "unknown")
                    loss_phase_counts.update([phase])
                    loss_phase_by_backend.update([(backend, phase)])
                    loss_phase_by_semantics.update([(semantics, phase)])
                    loss_phase_by_shape_family.update([(family, phase)])
                    loss_phase_by_scenario_family.update((scenario_family, phase) for scenario_family in group_scenario_families(group))
                    loss_rows.append(compact_candidate_route(group, candidate))
                bottleneck = candidate.get("bottleneck") if isinstance(candidate.get("bottleneck"), dict) else {}
                bottleneck_class = str(bottleneck.get("class") or "unknown")
                bottleneck_counts.update([bottleneck_class])
            if isinstance(candidate.get("prepacked_reuse_review"), dict) or isinstance(candidate.get("hip_graph_replay_review"), dict):
                setup_sensitive.append(
                    {
                        **compact_candidate_route(group, candidate),
                        "prepacked_reuse_review": candidate.get("prepacked_reuse_review"),
                        "hip_graph_replay_review": candidate.get("hip_graph_replay_review"),
                        "promotion_blockers": candidate.get("promotion_blockers") or [],
                    }
                )

    return {
        "group_count": len(groups),
        "promotable_autotune_entry_count": len(promotable_entries),
        "missing_required_baseline_group_count": missing_baseline_count,
        "checksum_mismatch_group_count": checksum_mismatch_count,
        "review_blocker_counts": dict(blocker_counts.most_common()),
        "actionable_blocker_counts": dict(actionable_blockers.most_common()),
        "loss_phase_counts": dict(loss_phase_counts.most_common()),
        "loss_phase_by_backend": nested_counter(loss_phase_by_backend),
        "loss_phase_by_semantics": nested_counter(loss_phase_by_semantics),
        "loss_phase_by_shape_family": nested_counter(loss_phase_by_shape_family),
        "loss_phase_by_scenario_family": nested_counter(loss_phase_by_scenario_family),
        "bottleneck_counts": dict(bottleneck_counts.most_common()),
        "fastest_production_route_counts": dict(production_counts.most_common()),
        "fastest_accelerator_route_counts": dict(accelerator_counts.most_common()),
        "direct_hip_production_wins": direct_hip_winners,
        "loss_phase_examples": loss_rows[:40],
        "setup_sensitive_candidates": setup_sensitive[:40],
        "pack_split_counts": dict(pack_split_counts.most_common()),
        "pack_dominant_operand_counts": dict(pack_dominant_operand_counts.most_common()),
        "pack_diagnostics": sorted(
            pack_diagnostic_rows,
            key=lambda row: (
                row.get("pack_share_of_end_to_end")
                if isinstance(row.get("pack_share_of_end_to_end"), (int, float))
                else 0.0
            ),
            reverse=True,
        )[:40],
        "next_work": review_next_work(
            missing_baseline_count=missing_baseline_count,
            checksum_mismatch_count=checksum_mismatch_count,
            actionable_blockers=actionable_blockers,
            loss_phase_counts=loss_phase_counts,
            bottleneck_counts=bottleneck_counts,
            pack_split_counts=pack_split_counts,
            pack_dominant_operand_counts=pack_dominant_operand_counts,
            pack_diagnostics=pack_diagnostic_rows,
            direct_hip_wins=len(direct_hip_winners),
            promotable_count=len(promotable_entries),
        ),
    }


def release_capture_satisfied(capture: dict[str, Any]) -> bool:
    return (
        isinstance(capture.get("warmups"), int)
        and isinstance(capture.get("repeats"), int)
        and capture["warmups"] >= RELEASE_MIN_WARMUPS
        and capture["repeats"] >= RELEASE_MIN_REPEATS
    )


def capture_checksum(capture: dict[str, Any] | None) -> Any:
    if not capture:
        return None
    if capture.get("checksum_u64") is not None:
        return capture.get("checksum_u64")
    return capture.get("checksum")


def capture_prepacked_reuse(capture: dict[str, Any]) -> bool:
    return capture_pack_mode(capture) in REUSE_PACK_MODES or capture.get("reuse_packed_inputs") is True


def capture_hip_graph_replay(capture: dict[str, Any]) -> bool:
    return capture_execution_mode(capture) in GRAPH_REPLAY_EXECUTION_MODES


def numeric_capture_value(capture: dict[str, Any], key: str) -> float | None:
    value = capture.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def graph_numeric_value(capture: dict[str, Any], key: str) -> float | None:
    graph = capture.get("hip_graph_replay")
    if not isinstance(graph, dict):
        return None
    value = graph.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def reuse_contract_numeric_value(capture: dict[str, Any], key: str) -> float | None:
    reuse = capture.get("reuse_contract")
    if not isinstance(reuse, dict):
        return None
    value = reuse.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def setup_cost_us(capture: dict[str, Any], *, include_graph_setup: bool) -> float | None:
    setup = 0.0
    if capture_prepacked_reuse(capture):
        prepack = reuse_contract_numeric_value(capture, "setup_cost_us")
        if prepack is None:
            prepack = numeric_capture_value(capture, "avg_prepack_setup_us")
        if prepack is None:
            prepack = numeric_capture_value(capture, "prepack_setup_us")
        if prepack is None:
            return None
        setup += prepack
    if include_graph_setup:
        capture_us = graph_numeric_value(capture, "capture_us")
        instantiate_us = graph_numeric_value(capture, "instantiate_us")
        if capture_us is None or instantiate_us is None:
            return None
        setup += capture_us + instantiate_us
    return setup


def setup_inclusive_end_to_end_us(capture: dict[str, Any], *, include_graph_setup: bool = False) -> float | None:
    if capture_prepacked_reuse(capture) and not include_graph_setup:
        recorded = reuse_contract_numeric_value(capture, "setup_inclusive_median_end_to_end_us")
        if recorded is not None:
            return recorded
    median = median_phase(capture, "end_to_end")
    repeats = normalized_positive_int(capture.get("repeats"))
    if median is None or repeats is None:
        return None
    setup = setup_cost_us(capture, include_graph_setup=include_graph_setup)
    if setup is None:
        return None
    return median + setup / float(repeats)


def break_even_repeat_count(
    *,
    baseline_median_us: float | None,
    candidate_median_us: float | None,
    baseline_setup_us: float | None,
    candidate_setup_us: float | None,
) -> int | None:
    if (
        baseline_median_us is None
        or candidate_median_us is None
        or baseline_setup_us is None
        or candidate_setup_us is None
    ):
        return None
    if candidate_median_us == baseline_median_us:
        return 1 if candidate_setup_us < baseline_setup_us else None
    if candidate_median_us > baseline_median_us:
        return None
    steady_state_delta = baseline_median_us - candidate_median_us
    setup_delta = candidate_setup_us - baseline_setup_us
    if setup_delta <= 0:
        return 1
    return max(1, math.floor(setup_delta / steady_state_delta) + 1)


def selection_end_to_end_us(capture: dict[str, Any]) -> float | None:
    if capture_hip_graph_replay(capture):
        return setup_inclusive_end_to_end_us(capture, include_graph_setup=True)
    if capture_prepacked_reuse(capture):
        return setup_inclusive_end_to_end_us(capture)
    return median_phase(capture, "end_to_end")


def setup_comparison_key(capture: dict[str, Any], *, include_pack_mode: bool) -> str:
    tile_bounds = capture.get("tile_bounds_u64")
    tile_hash = tile_bounds.get("hash_u64") if isinstance(tile_bounds, dict) else None
    wrap64_contract = capture.get("semantics") == "wrap_u64_mod_2_64"
    tile_m = "wrap64_semantic_contract" if wrap64_contract else capture.get("tile_m")
    tile_n = "wrap64_semantic_contract" if wrap64_contract else capture.get("tile_n")
    timing_metadata = capture_timing_metadata(capture)
    output_policy = capture.get("output_policy")
    requested_next_op = capture.get("requested_next_op")
    residue_output_mode = capture.get("residue_output_mode", "host_export")
    next_op_contract = (
        "host_export"
        if residue_output_mode == "host_export"
        else requested_next_op.get("resolved")
        if isinstance(requested_next_op, dict)
        else None
    )
    parts = [
        f"checksum_policy={capture_checksum_policy(capture)}",
        f"host_api_batch_size={capture_host_api_batch_size(capture)}",
        f"grouped_dispatch_task_count={capture_grouped_dispatch_task_count(capture)}",
        f"grouped_dispatch_strategy={capture_grouped_dispatch_strategy(capture)}",
        f"semantics={capture.get('semantics')}",
        f"finite_modulus={capture.get('finite_modulus')}",
        f"bound_kind={capture.get('bound_kind')}",
        f"bound_mode={capture.get('bound_mode')}",
        f"bound={capture.get('bound')}",
        f"bound_source={capture_bound_source(capture)}",
        f"m={capture.get('m')}",
        f"n={capture.get('n')}",
        f"k={capture.get('k')}",
        f"output_logical_ld={capture.get('output_logical_ld', capture.get('n'))}",
        f"output_ld_padding={capture.get('output_ld_padding', 0)}",
        f"prefix={capture.get('prefix')}",
        f"layout={capture.get('layout')}",
        f"tile_m={tile_m}",
        f"tile_n={tile_n}",
        f"k_block={capture.get('k_block_size')}",
        f"exact_wide_limb_count={capture.get('exact_wide_limb_count')}",
        f"residue_chain_length={capture.get('residue_chain_length', 1)}",
        f"residue_output_mode={residue_output_mode}",
        f"seed={capture.get('seed')}",
        f"input_distribution={capture.get('input_distribution')}",
        f"next_op_contract={next_op_contract}",
        f"output_policy={output_policy.get('destination_layout') if isinstance(output_policy, dict) else None}",
        f"status_handling={output_policy.get('status_handling') if isinstance(output_policy, dict) else None}",
        f"export_variant={capture_export_variant_name(capture)}",
        f"reconstruction_variant={capture_reconstruction_variant_name(capture)}",
        f"fusion_mode={timing_metadata.get('fusion_mode') if isinstance(timing_metadata, dict) else None}",
        f"residue_group_width={timing_metadata.get('residue_group_width') if isinstance(timing_metadata, dict) else None}",
        f"tile_hash={tile_hash}",
    ]
    if include_pack_mode:
        parts.extend(
            [
                f"reuse_packed_inputs={capture.get('reuse_packed_inputs') is True}",
                f"pack_mode={capture_pack_mode(capture)}",
            ]
        )
    return ";".join(str(part) for part in parts)


def fastest_capture(captures: list[dict[str, Any]], *, include_graph_setup: bool = False) -> dict[str, Any] | None:
    timed = [
        capture
        for capture in captures
        if setup_inclusive_end_to_end_us(capture, include_graph_setup=include_graph_setup) is not None
    ]
    if not timed:
        return None
    return min(
        timed,
        key=lambda capture: setup_inclusive_end_to_end_us(capture, include_graph_setup=include_graph_setup)
        or float("inf"),
    )


def graph_status_available(capture: dict[str, Any]) -> bool:
    graph = capture.get("hip_graph_replay")
    return (
        isinstance(graph, dict)
        and graph.get("status") == "available"
        and graph.get("capture_status") == "replayed"
    )


def build_setup_baseline_indexes(
    captures: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, str], list[dict[str, Any]]],
]:
    nonreuse_by_key_backend: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    nonreuse_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    graph_baseline_by_key_backend: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        family = backend_family_id(backend_id(capture))
        if not correctness_anchor_reference_capture(capture) and not capture_hip_graph_replay(capture):
            graph_key = setup_comparison_key(capture, include_pack_mode=True)
            graph_baseline_by_key_backend[(graph_key, family)].append(capture)
        if capture_prepacked_reuse(capture) or capture_hip_graph_replay(capture):
            continue
        if correctness_anchor_reference_capture(capture):
            continue
        key = setup_comparison_key(capture, include_pack_mode=False)
        nonreuse_by_key_backend[(key, family)].append(capture)
        if family != "cpu-reference":
            nonreuse_by_key[key].append(capture)
    return nonreuse_by_key_backend, nonreuse_by_key, graph_baseline_by_key_backend


def checksum_match_blockers(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    label: str,
) -> list[str]:
    if baseline is None:
        return []
    candidate_checksum = capture_checksum(candidate)
    baseline_checksum = capture_checksum(baseline)
    if candidate_checksum is None or baseline_checksum is None:
        return []
    if candidate_checksum != baseline_checksum:
        return [f"checksum_mismatch_vs_{label}"]
    return []


def prepacked_reuse_review(
    capture: dict[str, Any],
    same_backend_baseline: dict[str, Any] | None,
    best_nonreuse_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_setup = setup_inclusive_end_to_end_us(capture)
    candidate_median = median_phase(capture, "end_to_end")
    same_backend_time = median_phase(same_backend_baseline, "end_to_end") if same_backend_baseline else None
    best_nonreuse_time = median_phase(best_nonreuse_baseline, "end_to_end") if best_nonreuse_baseline else None
    prepack_setup = setup_cost_us(capture, include_graph_setup=False)
    declared_repeats = normalized_positive_int(capture.get("repeats"))
    setup_amortized = (
        prepack_setup / float(declared_repeats)
        if prepack_setup is not None and declared_repeats is not None
        else None
    )
    blockers: list[str] = []
    if same_backend_baseline is None:
        blockers.append("missing_same_backend_nonreuse_baseline")
    elif not release_capture_satisfied(same_backend_baseline):
        blockers.append("same_backend_nonreuse_not_release_review")
    if best_nonreuse_baseline is None:
        blockers.append("missing_best_nonreuse_contract_baseline")
    elif not release_capture_satisfied(best_nonreuse_baseline):
        blockers.append("best_nonreuse_not_release_review")
    if candidate_setup is None:
        blockers.append("missing_prepack_setup_inclusive_timing")
    if same_backend_time is not None and candidate_setup is not None and candidate_setup >= same_backend_time:
        blockers.append("reuse_not_faster_than_same_backend_setup_inclusive")
    if best_nonreuse_time is not None and candidate_setup is not None and candidate_setup >= best_nonreuse_time:
        blockers.append("reuse_not_faster_than_best_nonreuse_setup_inclusive")
    blockers.extend(checksum_match_blockers(capture, same_backend_baseline, label="same_backend_nonreuse"))
    blockers.extend(checksum_match_blockers(capture, best_nonreuse_baseline, label="best_nonreuse"))
    return {
        "setup_inclusive_median_end_to_end_us": candidate_setup,
        "candidate_median_end_to_end_us": candidate_median,
        "prepack_setup_us": prepack_setup,
        "declared_repeat_count": declared_repeats,
        "setup_amortized_us": setup_amortized,
        "setup_share_of_setup_inclusive": (
            setup_amortized / candidate_setup
            if setup_amortized is not None and candidate_setup not in (None, 0.0)
            else None
        ),
        "same_backend_nonreuse_backend": backend_family_id(backend_id(same_backend_baseline)) if same_backend_baseline else None,
        "same_backend_nonreuse_capture": same_backend_baseline.get("_path") if same_backend_baseline else None,
        "same_backend_nonreuse_median_end_to_end_us": same_backend_time,
        "best_nonreuse_backend": backend_family_id(backend_id(best_nonreuse_baseline)) if best_nonreuse_baseline else None,
        "best_nonreuse_capture": best_nonreuse_baseline.get("_path") if best_nonreuse_baseline else None,
        "best_nonreuse_median_end_to_end_us": best_nonreuse_time,
        "speedup_vs_same_backend_setup_inclusive": (
            same_backend_time / candidate_setup
            if same_backend_time is not None and candidate_setup not in (None, 0.0)
            else None
        ),
        "speedup_vs_best_nonreuse_setup_inclusive": (
            best_nonreuse_time / candidate_setup
            if best_nonreuse_time is not None and candidate_setup not in (None, 0.0)
            else None
        ),
        "blockers": sorted(set(blockers)),
    }


def hip_graph_replay_review(
    capture: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    graph_setup = setup_inclusive_end_to_end_us(capture, include_graph_setup=True)
    baseline_setup = setup_inclusive_end_to_end_us(baseline) if baseline else None
    graph_median = median_phase(capture, "end_to_end")
    baseline_median = median_phase(baseline, "end_to_end") if baseline else None
    graph_total_setup = setup_cost_us(capture, include_graph_setup=True)
    baseline_total_setup = setup_cost_us(baseline, include_graph_setup=False) if baseline else None
    break_even_repeats = break_even_repeat_count(
        baseline_median_us=baseline_median,
        candidate_median_us=graph_median,
        baseline_setup_us=baseline_total_setup,
        candidate_setup_us=graph_total_setup,
    )
    raw_repeats = capture.get("repeats")
    declared_repeats = raw_repeats if isinstance(raw_repeats, int) and not isinstance(raw_repeats, bool) and raw_repeats > 0 else None
    blockers: list[str] = []
    if baseline is None:
        blockers.append("missing_same_contract_non_graph_baseline")
    elif not release_capture_satisfied(baseline):
        blockers.append("non_graph_baseline_not_release_review")
    if graph_setup is None or baseline_setup is None:
        blockers.append("missing_graph_setup_inclusive_timing")
    if not graph_status_available(capture):
        blockers.append("graph_replay_not_available")
    if graph_setup is not None and baseline_setup is not None and graph_setup >= baseline_setup:
        blockers.append("graph_not_faster_than_non_graph_setup_inclusive")
    blockers.extend(checksum_match_blockers(capture, baseline, label="non_graph_baseline"))
    return {
        "setup_inclusive_median_end_to_end_us": graph_setup,
        "candidate_median_end_to_end_us": graph_median,
        "graph_capture_us": graph_numeric_value(capture, "capture_us"),
        "graph_instantiate_us": graph_numeric_value(capture, "instantiate_us"),
        "graph_total_setup_us": graph_total_setup,
        "graph_setup_amortized_us": (
            graph_total_setup / float(declared_repeats)
            if graph_total_setup is not None and declared_repeats is not None
            else None
        ),
        "graph_setup_share_of_setup_inclusive": (
            (graph_total_setup / float(declared_repeats)) / graph_setup
            if graph_total_setup is not None and declared_repeats is not None and graph_setup not in (None, 0.0)
            else None
        ),
        "baseline_backend": backend_family_id(backend_id(baseline)) if baseline else None,
        "baseline_capture": baseline.get("_path") if baseline else None,
        "baseline_median_end_to_end_us": baseline_median,
        "baseline_total_setup_us": baseline_total_setup,
        "baseline_setup_inclusive_median_end_to_end_us": baseline_setup,
        "baseline_setup_share_of_setup_inclusive": (
            (baseline_total_setup / float(declared_repeats)) / baseline_setup
            if baseline_total_setup is not None and declared_repeats is not None and baseline_setup not in (None, 0.0)
            else None
        ),
        "steady_state_delta_us": (
            baseline_median - graph_median if baseline_median is not None and graph_median is not None else None
        ),
        "graph_setup_overhead_vs_baseline_us": (
            graph_total_setup - baseline_total_setup
            if graph_total_setup is not None and baseline_total_setup is not None
            else None
        ),
        "break_even_repeat_count": break_even_repeats,
        "declared_repeat_count": declared_repeats,
        "declared_repeats_meet_break_even": (
            declared_repeats >= break_even_repeats
            if declared_repeats is not None and break_even_repeats is not None
            else False
        ),
        "speedup_vs_non_graph_setup_inclusive": (
            baseline_setup / graph_setup if baseline_setup is not None and graph_setup not in (None, 0.0) else None
        ),
        "blockers": sorted(set(blockers)),
    }


def reference_checksum_for_group(by_backend: dict[str, dict[str, Any]]) -> tuple[str | None, Any]:
    for backend in ("cpu-reference", "wrap64-byte-limb", "hip-direct"):
        checksum = capture_checksum(by_backend.get(backend))
        if checksum is not None:
            return backend, checksum
    for backend, capture in sorted(by_backend.items()):
        checksum = capture_checksum(capture)
        if checksum is not None:
            return backend, checksum
    return None, None


def promotion_blockers(
    *,
    missing: list[str],
    semantics: Any,
    release_review_satisfied: bool,
    gpu_target_identity_complete: bool,
    gpu_target_compatible: bool,
    configured_target_identity_complete: bool,
    configured_target_compatible: bool,
    hip_toolchain_version_complete: bool,
    hip_toolchain_version_compatible: bool,
    hip_runtime_version_complete: bool,
    hip_runtime_version_compatible: bool,
    hip_driver_version_complete: bool,
    hip_driver_version_compatible: bool,
    compiler_identity_complete: bool,
    compiler_identity_compatible: bool,
    git_commit_identity_complete: bool,
    git_commit_identity_compatible: bool,
    warmup_count_complete: bool,
    warmup_count_compatible: bool,
    repeat_count_complete: bool,
    repeat_count_compatible: bool,
    duplicate_backends: list[str],
    accelerator: bool,
    internal_candidate: bool,
    prepacked_reuse: bool,
    oneshot_capture: bool,
    host_api_batch_capture: bool,
    hip_graph_replay_capture: bool,
    gpu_events_available: bool,
    end_to_end: float | None,
    cpu: float | None,
    direct: float | None,
    vector: float | None,
) -> list[str]:
    blockers: list[str] = []
    if missing:
        blockers.append("missing_required_baselines")
    if not release_review_satisfied:
        blockers.append("not_release_review")
    if not gpu_target_identity_complete:
        blockers.append("missing_gpu_target_id")
    elif not gpu_target_compatible:
        blockers.append("gpu_target_mismatch")
    if not configured_target_identity_complete:
        blockers.append("missing_configured_gpu_target")
    elif not configured_target_compatible:
        blockers.append("configured_gpu_target_mismatch")
    if not hip_toolchain_version_complete:
        blockers.append("missing_hip_toolchain_version")
    elif not hip_toolchain_version_compatible:
        blockers.append("hip_toolchain_version_mismatch")
    if not hip_runtime_version_complete:
        blockers.append("missing_hip_runtime_version")
    elif not hip_runtime_version_compatible:
        blockers.append("hip_runtime_version_mismatch")
    if not hip_driver_version_complete:
        blockers.append("missing_hip_driver_version")
    elif not hip_driver_version_compatible:
        blockers.append("hip_driver_version_mismatch")
    if not compiler_identity_complete:
        blockers.append("missing_compiler_identity")
    elif not compiler_identity_compatible:
        blockers.append("compiler_identity_mismatch")
    if not git_commit_identity_complete:
        blockers.append("missing_git_commit")
    elif not git_commit_identity_compatible:
        blockers.append("git_commit_mismatch")
    if not warmup_count_complete:
        blockers.append("missing_warmup_count")
    elif not warmup_count_compatible:
        blockers.append("warmup_count_mismatch")
    if not repeat_count_complete:
        blockers.append("missing_repeat_count")
    elif not repeat_count_compatible:
        blockers.append("repeat_count_mismatch")
    if duplicate_backends:
        blockers.append("duplicate_backend_capture")
    if not accelerator:
        blockers.append("not_accelerator_backend")
    if internal_candidate:
        blockers.append("internal_candidate_not_public_backend")
    if prepacked_reuse:
        blockers.append("prepacked_reuse_not_autotune_promotable")
    if oneshot_capture:
        blockers.append("oneshot_api_capture_not_autotune_promotable")
    if host_api_batch_capture:
        blockers.append("host_api_batch_not_autotune_promotable")
    if hip_graph_replay_capture:
        blockers.append("hip_graph_replay_not_autotune_promotable")
    if accelerator and not gpu_events_available:
        blockers.append("missing_required_gpu_events")
    if accelerator:
        if end_to_end is None:
            blockers.append("missing_end_to_end_timing")
        if cpu is not None and end_to_end is not None and end_to_end >= cpu:
            blockers.append("not_faster_than_cpu_reference")
        if direct is None:
            blockers.append("missing_direct_hip_timing")
        elif end_to_end is not None and end_to_end >= direct:
            blockers.append("not_faster_than_direct_hip")
        if vector is not None and end_to_end is not None and end_to_end >= vector:
            blockers.append("not_faster_than_vector_alu")
    return blockers


def primary_loss_phase(item: dict[str, Any], direct: dict[str, Any] | None) -> str | None:
    if direct is None:
        return None
    worst_phase = None
    worst_ratio = 0.0
    for phase in PHASES:
        value = median_phase(item, phase)
        baseline = median_phase(direct, phase)
        if value is None or baseline is None or baseline <= 0:
            continue
        ratio = value / baseline
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_phase = phase
    return worst_phase


def bottleneck_classification(capture: dict[str, Any]) -> dict[str, Any]:
    end_to_end = median_phase(capture, "end_to_end")
    phase_values = {
        phase: value
        for phase in ("pack", "rns_gemm", "crt_export")
        if (value := median_phase(capture, phase)) is not None and value > 0
    }
    if not end_to_end or end_to_end <= 0 or not phase_values:
        return {"class": "unknown", "phase": None, "share": None}
    shares = {phase: value / end_to_end for phase, value in phase_values.items()}
    overhead_share = max(0.0, end_to_end - sum(phase_values.values())) / end_to_end
    phase, share = max(shares.items(), key=lambda item: item[1])
    if overhead_share >= 0.25 and overhead_share > share:
        return {"class": "launch_or_api_bound", "phase": "unattributed_overhead", "share": overhead_share}
    if share < 0.40:
        return {"class": "mixed_bound", "phase": phase, "share": share}
    return {
        "class": {"pack": "pack_bound", "rns_gemm": "compute_bound", "crt_export": "export_bound"}[phase],
        "phase": phase,
        "share": share,
    }


def pack_phase_diagnostics(capture: dict[str, Any]) -> dict[str, Any]:
    end_to_end = median_phase(capture, "end_to_end")
    pack = median_phase(capture, "pack")
    pack_a = median_phase(capture, "pack_a")
    pack_b = median_phase(capture, "pack_b")
    split_available = pack_a is not None and pack_b is not None
    dominant_operand = None
    if split_available:
        if pack_a > pack_b * 1.10:
            dominant_operand = "A"
        elif pack_b > pack_a * 1.10:
            dominant_operand = "B"
        else:
            dominant_operand = "balanced"
    plan_packing = capture.get("plan_packing")
    plan_packing_dict = plan_packing if isinstance(plan_packing, dict) else {}
    return {
        "pack_median_us": pack,
        "pack_a_median_us": pack_a,
        "pack_b_median_us": pack_b,
        "pack_share_of_end_to_end": (pack / end_to_end) if pack is not None and end_to_end else None,
        "split_available": split_available,
        "dominant_operand": dominant_operand,
        "pack_mode": capture_pack_mode(capture),
        "pack_layout": capture_timing_metadata(capture).get("pack_layout"),
        "source_versioned_inputs": plan_packing_dict.get("source_versioned_inputs"),
        "same_source_version_pack_elision_available": plan_packing_dict.get(
            "same_source_version_pack_elision_available"
        ),
        "a_pack_workspace_bytes": plan_packing_dict.get("a_pack_workspace_bytes"),
        "b_pack_workspace_bytes": plan_packing_dict.get("b_pack_workspace_bytes"),
    }


def capture_gpu_events_available(capture: dict[str, Any]) -> bool:
    timing = capture_timing_metadata(capture)
    return (
        timing.get("gpu_event_timing") is True
        and timing.get("gpu_event_timing_status") == "available"
        and bool(timing.get("gpu_event_timing_source"))
        and isinstance(timing.get("gpu_event_phase_order"), list)
    )


def capture_scenario_promotion_scope(capture: dict[str, Any]) -> str | None:
    scenario = capture.get("scenario_metadata")
    if not isinstance(scenario, dict):
        return None
    eligibility = scenario.get("promotion_eligibility")
    if isinstance(eligibility, str) and eligibility:
        return eligibility
    metadata = scenario.get("metadata")
    if isinstance(metadata, dict):
        scope = metadata.get("promotion_scope")
        if isinstance(scope, str) and scope:
            return scope
    return None


def scenario_promotion_blockers(capture: dict[str, Any]) -> list[str]:
    scope = capture_scenario_promotion_scope(capture)
    if scope is None or scope == "release_review_candidate":
        return []
    return ["scenario_scope_not_autotune_promotable"]


def matrix_instruction_histogram(
    capture: dict[str, Any],
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    direct = capture.get("matrix_instruction_histogram")
    if isinstance(direct, dict):
        return direct
    for key in ("gpu_isa_report", "isa_report", "backend_isa_report", "compiled_isa"):
        report = capture.get(key)
        if not isinstance(report, dict):
            continue
        for nested_key in (
            "matrix_instruction_histogram",
            "matrix_instruction_counts",
            "mnemonic_histogram",
        ):
            histogram = report.get(nested_key)
            if isinstance(histogram, dict):
                return histogram
    if isa_index is not None:
        target = capture_device(capture).get("gcn_arch") or capture.get("configured_amdgpu_targets")
        resources = lookup_isa_resources(isa_index, backend_family_id(backend_id(capture)), target)
        histogram = resources.get("isa_matrix_instruction_histogram")
        if isinstance(histogram, dict):
            return histogram
    return {}


def review_route_candidate(candidate: dict[str, Any]) -> bool:
    scope = candidate.get("scenario_promotion_scope")
    blockers = {str(item) for item in candidate.get("promotion_blockers", [])}
    fatal = {
        "missing_required_baselines",
        "not_release_review",
        "missing_gpu_target_id",
        "gpu_target_mismatch",
        "missing_configured_gpu_target",
        "configured_gpu_target_mismatch",
        "missing_hip_toolchain_version",
        "hip_toolchain_version_mismatch",
        "missing_hip_runtime_version",
        "hip_runtime_version_mismatch",
        "missing_hip_driver_version",
        "hip_driver_version_mismatch",
        "missing_compiler_identity",
        "compiler_identity_mismatch",
        "missing_git_commit",
        "git_commit_mismatch",
        "missing_warmup_count",
        "warmup_count_mismatch",
        "missing_repeat_count",
        "repeat_count_mismatch",
        "duplicate_backend_capture",
        "internal_candidate_not_public_backend",
        "prepacked_reuse_not_autotune_promotable",
        "oneshot_api_capture_not_autotune_promotable",
        "host_api_batch_not_autotune_promotable",
        "hip_graph_replay_not_autotune_promotable",
        "missing_checksum",
        "missing_reference_checksum",
        "checksum_mismatch_vs_reference",
        "scenario_scope_not_autotune_promotable",
        "missing_direct_hip_skinny_gemv_kernel_identity",
        "missing_direct_hip_skinny_gemv_execution_mode",
        "missing_vector_alu_skinny_gemv_kernel_identity",
        "missing_vector_alu_skinny_gemv_execution_mode",
        "missing_skinny_gemv_gpu_event_phase",
        "missing_native_to_rns_bridge_forced_metadata",
        "missing_native_to_rns_handoff_scope",
        "missing_native_to_rns_phase_availability",
        "missing_vector_to_rns_chain_metadata",
        "vector_to_rns_chain_control_mode_mismatch",
        "missing_vector_to_rns_chain_producer_backend",
        "missing_vector_to_rns_chain_consumer_backend",
        "missing_vector_to_rns_chain_phase_scope",
        "missing_sparse_a_4_to_2_scenario_contract",
        "sparse_a_k_not_divisible_by_4",
        "missing_sparse_a_4_to_2_input_distribution",
        "missing_sparse_a_runtime_kernel_identity",
        "dense_sparse_a_baseline_used_sparse_kernel",
        "missing_sparse_a_matrix_instruction_sparsity",
        "missing_sparse_a_matrix_instruction_family",
        "missing_sparse_a_4_to_2_autotune_contract",
        "amdgpu_builtin_tile_variant_kernel_mismatch",
        "amdgpu_builtin_tile_variant_family_mismatch",
        "amdgpu_builtin_tile_variant_shape_mismatch",
        "amdgpu_builtin_tile_variant_dtype_mismatch",
    }
    return (
        scope in {None, "release_review_candidate"}
        and candidate.get("release_review_capture") is True
        and candidate.get("checksum_matches_reference") is True
        and candidate.get("selection_end_to_end_us") is not None
        and not (blockers & fatal)
    )


def fastest_route(candidates: list[dict[str, Any]], *, accelerator_only: bool) -> dict[str, Any] | None:
    route_candidates = [
        candidate
        for candidate in candidates
        if review_route_candidate(candidate)
        and (candidate.get("accelerator_backend") is True if accelerator_only else candidate.get("backend") != "cpu-reference")
    ]
    if not route_candidates:
        return None
    return min(route_candidates, key=lambda item: item["selection_end_to_end_us"])


def review_captures(
    captures: list[dict[str, Any]],
    *,
    review_mode: str = "smoke",
    isa_index: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if review_mode not in {"smoke", "release"}:
        raise ValueError(f"unsupported review mode: {review_mode}")
    nonreuse_by_key_backend, nonreuse_by_key, graph_baseline_by_key_backend = build_setup_baseline_indexes(captures)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        grouped[capture_contract_key(capture)].append(capture)

    groups = []
    promotable_entries = []
    for key, items in sorted(grouped.items()):
        backend_counts: dict[str, int] = defaultdict(int)
        for item in items:
            backend_counts[backend_id(item)] += 1
        duplicate_backends = sorted(backend for backend, count in backend_counts.items() if count > 1)
        by_backend: dict[str, dict[str, Any]] = {}
        for item in items:
            by_backend.setdefault(backend_id(item), item)
        semantics = items[0].get("semantics")
        required = required_baselines_for_group(semantics, items)
        missing = [backend for backend in required if backend not in by_backend]
        cpu_capture = by_backend.get("cpu-reference")
        direct_capture = by_backend.get("hip-direct")
        vector_capture = by_backend.get("hip-vector-alu-int64")
        phase_medians = {
            f"{backend_id(item)}/{selected_kernel(item)}": phase_medians_for_capture(item)
            for item in items
        }
        gpu_targets = {
            backend: normalized_target_id(capture.get("device", {}).get("gcn_arch"))
            for backend, capture in by_backend.items()
            if backend_requires_gpu_target(backend)
        }
        missing_gpu_targets = sorted(backend for backend, target in gpu_targets.items() if target is None)
        gpu_target_identity_complete = not missing_gpu_targets
        gpu_target_values = {value for value in gpu_targets.values() if value}
        gpu_target_compatible = gpu_target_identity_complete and len(gpu_target_values) <= 1
        configured_gpu_targets = {
            backend: normalized_target_id(capture.get("configured_amdgpu_targets"))
            for backend, capture in by_backend.items()
            if backend_requires_gpu_target(backend)
        }
        missing_configured_gpu_targets = sorted(
            backend for backend, target in configured_gpu_targets.items() if target is None
        )
        configured_target_identity_complete = not missing_configured_gpu_targets
        configured_target_values = {target for target in configured_gpu_targets.values() if target}
        configured_target_compatible = configured_target_identity_complete and len(configured_target_values) <= 1
        hip_toolchain_versions = {
            backend: normalized_target_id(capture_hip_toolchain(capture).get("hip_sdk_or_rocm_version"))
            for backend, capture in by_backend.items()
            if backend_requires_gpu_target(backend)
        }
        missing_hip_toolchain_versions = sorted(
            backend for backend, version in hip_toolchain_versions.items() if version is None
        )
        hip_toolchain_version_complete = not missing_hip_toolchain_versions
        hip_toolchain_version_values = {version for version in hip_toolchain_versions.values() if version}
        hip_toolchain_version_compatible = (
            hip_toolchain_version_complete and len(hip_toolchain_version_values) <= 1
        )
        hip_runtime_versions = {
            backend: normalized_positive_int(capture_device(capture).get("hip_runtime_version"))
            for backend, capture in by_backend.items()
            if backend_requires_gpu_target(backend)
        }
        missing_hip_runtime_versions = sorted(
            backend for backend, version in hip_runtime_versions.items() if version is None
        )
        hip_runtime_version_complete = not missing_hip_runtime_versions
        hip_runtime_version_values = {version for version in hip_runtime_versions.values() if version}
        hip_runtime_version_compatible = hip_runtime_version_complete and len(hip_runtime_version_values) <= 1
        hip_driver_versions = {
            backend: normalized_positive_int(capture_device(capture).get("hip_driver_version"))
            for backend, capture in by_backend.items()
            if backend_requires_gpu_target(backend)
        }
        missing_hip_driver_versions = sorted(
            backend for backend, version in hip_driver_versions.items() if version is None
        )
        hip_driver_version_complete = not missing_hip_driver_versions
        hip_driver_version_values = {version for version in hip_driver_versions.values() if version}
        hip_driver_version_compatible = hip_driver_version_complete and len(hip_driver_version_values) <= 1
        compiler_identities = {backend: normalized_compiler_identity(capture) for backend, capture in by_backend.items()}
        missing_compiler_identities = sorted(
            backend for backend, identity in compiler_identities.items() if identity is None
        )
        compiler_identity_complete = not missing_compiler_identities
        compiler_identity_values = {identity for identity in compiler_identities.values() if identity}
        compiler_identity_compatible = compiler_identity_complete and len(compiler_identity_values) <= 1
        git_commits = {
            backend: normalized_identity_text(capture.get("git_commit")) for backend, capture in by_backend.items()
        }
        missing_git_commits = sorted(backend for backend, commit in git_commits.items() if commit is None)
        git_commit_identity_complete = not missing_git_commits
        git_commit_values = {commit for commit in git_commits.values() if commit}
        git_commit_identity_compatible = git_commit_identity_complete and len(git_commit_values) <= 1
        release_timing_by_backend = {
            backend: capture
            for backend, capture in by_backend.items()
            if not correctness_anchor_reference_capture(capture)
        }
        warmup_counts = {
            backend: normalized_positive_int(capture.get("warmups"))
            for backend, capture in release_timing_by_backend.items()
        }
        missing_warmup_counts = sorted(backend for backend, count in warmup_counts.items() if count is None)
        warmup_count_complete = not missing_warmup_counts
        warmup_count_values = {count for count in warmup_counts.values() if count}
        warmup_count_compatible = warmup_count_complete and len(warmup_count_values) <= 1
        repeat_counts = {
            backend: normalized_positive_int(capture.get("repeats"))
            for backend, capture in release_timing_by_backend.items()
        }
        missing_repeat_counts = sorted(backend for backend, count in repeat_counts.items() if count is None)
        repeat_count_complete = not missing_repeat_counts
        repeat_count_values = {count for count in repeat_counts.values() if count}
        repeat_count_compatible = repeat_count_complete and len(repeat_count_values) <= 1
        release_review_satisfied = review_mode == "release" and bool(release_timing_by_backend) and all(
            release_capture_satisfied(item) for item in release_timing_by_backend.values()
        )
        checksum_by_backend = {backend: capture_checksum(capture) for backend, capture in by_backend.items()}
        checksum_reference_backend, checksum_reference = reference_checksum_for_group(by_backend)
        missing_checksums = sorted(backend for backend, checksum in checksum_by_backend.items() if checksum is None)
        checksum_mismatches = sorted(
            backend
            for backend, checksum in checksum_by_backend.items()
            if checksum_reference is not None and checksum is not None and checksum != checksum_reference
        )
        checksum_consistent = checksum_reference is not None and not missing_checksums and not checksum_mismatches
        scenario_promotion_scopes = sorted(
            {
                scope
                for item in items
                if (scope := capture_scenario_promotion_scope(item)) is not None
            }
        )
        candidates = []
        for item in items:
            backend = backend_id(item)
            backend_family = backend_family_id(backend)
            metadata = capture_backend_metadata(item)
            accelerator = metadata.get("accelerator_backend") is True
            internal_candidate = backend == WRAP64_ROCWMMA_CANDIDATE_BACKEND
            execution_mode = capture_execution_mode(item)
            oneshot_capture = execution_mode == "public_oneshot_transient_native_inputs"
            host_api_batch_capture = execution_mode == "benchmark_host_api_batch"
            hip_graph_replay_capture = capture_hip_graph_replay(item)
            end_to_end = median_phase(item, "end_to_end")
            selection_e2e = selection_end_to_end_us(item)
            timed_cpu_capture = (
                cpu_capture
                if cpu_capture is not None and not correctness_anchor_reference_capture(cpu_capture)
                else None
            )
            setup_key = setup_comparison_key(item, include_pack_mode=False)
            graph_key = setup_comparison_key(item, include_pack_mode=True)
            same_backend_nonreuse = fastest_capture(nonreuse_by_key_backend.get((setup_key, backend_family), []))
            best_nonreuse = fastest_capture(nonreuse_by_key.get(setup_key, []))
            graph_non_graph_baseline = fastest_capture(
                graph_baseline_by_key_backend.get((graph_key, backend_family), []),
            )
            direct_baseline = direct_capture or fastest_capture(nonreuse_by_key_backend.get((setup_key, "hip-direct"), []))
            vector_baseline = vector_capture or fastest_capture(
                nonreuse_by_key_backend.get((setup_key, "hip-vector-alu-int64"), [])
            )
            cpu = median_phase(timed_cpu_capture, "end_to_end") if timed_cpu_capture else None
            direct = median_phase(direct_baseline, "end_to_end") if direct_baseline else None
            vector = median_phase(vector_baseline, "end_to_end") if vector_baseline else None
            prepack_review = (
                prepacked_reuse_review(item, same_backend_nonreuse, best_nonreuse)
                if capture_prepacked_reuse(item)
                else None
            )
            graph_review = (
                hip_graph_replay_review(item, graph_non_graph_baseline)
                if hip_graph_replay_capture
                else None
            )
            matrix_histogram = matrix_instruction_histogram(item, isa_index)
            if autotune_promotable_scope(item):
                blockers = promotion_blockers(
                    missing=missing,
                    semantics=semantics,
                    release_review_satisfied=release_review_satisfied,
                    gpu_target_identity_complete=gpu_target_identity_complete,
                    gpu_target_compatible=gpu_target_compatible,
                    configured_target_identity_complete=configured_target_identity_complete,
                    configured_target_compatible=configured_target_compatible,
                    hip_toolchain_version_complete=hip_toolchain_version_complete,
                    hip_toolchain_version_compatible=hip_toolchain_version_compatible,
                    hip_runtime_version_complete=hip_runtime_version_complete,
                    hip_runtime_version_compatible=hip_runtime_version_compatible,
                    hip_driver_version_complete=hip_driver_version_complete,
                    hip_driver_version_compatible=hip_driver_version_compatible,
                    compiler_identity_complete=compiler_identity_complete,
                    compiler_identity_compatible=compiler_identity_compatible,
                    git_commit_identity_complete=git_commit_identity_complete,
                    git_commit_identity_compatible=git_commit_identity_compatible,
                    warmup_count_complete=warmup_count_complete,
                    warmup_count_compatible=warmup_count_compatible,
                    repeat_count_complete=repeat_count_complete,
                    repeat_count_compatible=repeat_count_compatible,
                    duplicate_backends=duplicate_backends,
                    accelerator=accelerator,
                    internal_candidate=internal_candidate,
                    prepacked_reuse=False,
                    oneshot_capture=oneshot_capture,
                    host_api_batch_capture=host_api_batch_capture,
                    hip_graph_replay_capture=False,
                    gpu_events_available=capture_gpu_events_available(item),
                    end_to_end=selection_e2e,
                    cpu=cpu,
                    direct=direct,
                    vector=vector if semantics in {"bounded_i64", "bounded_u64"} else None,
                )
                if prepack_review is not None:
                    blockers.extend(prepack_review["blockers"])
                if graph_review is not None:
                    blockers.extend(graph_review["blockers"])
            else:
                blockers = []
            expected_matrix_mnemonic = expected_amdgpu_builtin_matrix_mnemonic(item, metadata)
            if requires_amdgpu_builtin_matrix_isa_proof(backend_family, metadata):
                if not matrix_histogram:
                    blockers.append("missing_amdgpu_builtin_matrix_isa_histogram")
                elif (
                    not isinstance(expected_matrix_mnemonic, str)
                    or not expected_matrix_mnemonic
                    or matrix_histogram.get(expected_matrix_mnemonic, 0) <= 0
                ):
                    blockers.append("missing_selected_amdgpu_builtin_matrix_instruction")
            blockers.extend(
                final_output_export_metadata_blockers(
                    item,
                    semantics=semantics,
                    accelerator=accelerator,
                )
            )
            blockers.extend(
                skinny_gemv_metadata_blockers(
                    item,
                    semantics=semantics,
                    backend_family=backend_family,
                )
            )
            blockers.extend(
                native_to_rns_handoff_metadata_blockers(
                    item,
                    semantics=semantics,
                    backend_family=backend_family,
                )
            )
            blockers.extend(
                sparse_a_4_to_2_contract_blockers(
                    item,
                    backend=backend,
                    backend_family=backend_family,
                    metadata=metadata,
                )
            )
            blockers.extend(amdgpu_builtin_tile_variant_blockers(item, metadata))
            item_checksum = capture_checksum(item)
            if item_checksum is None:
                blockers.append("missing_checksum")
            elif checksum_reference is None:
                blockers.append("missing_reference_checksum")
            elif item_checksum != checksum_reference:
                blockers.append("checksum_mismatch_vs_reference")
            blockers.extend(scenario_promotion_blockers(item))
            promotable = not blockers
            candidate = {
                "backend": backend,
                "selected_kernel": selected_kernel(item),
                "capture": item.get("_path"),
                "source_metadata": candidate_source_metadata(item),
                "scenario_promotion_scope": capture_scenario_promotion_scope(item),
                "accelerator_backend": accelerator,
                "release_review_capture": release_capture_satisfied(item),
                "checksum": item_checksum,
                "checksum_reference_backend": checksum_reference_backend,
                "checksum_reference": checksum_reference,
                "checksum_matches_reference": (
                    item_checksum is not None
                    and checksum_reference is not None
                    and item_checksum == checksum_reference
                ),
                "median_end_to_end_us": end_to_end,
                "selection_end_to_end_us": selection_e2e,
                "setup_comparison_key": setup_key,
                "phase_diagnostics": phase_ratios(item, direct_baseline, vector_baseline),
                "phase_medians_us": phase_medians_for_capture(item),
                "pack_diagnostics": pack_phase_diagnostics(item),
                "tile_shape_variant": (
                    item.get("tile_shape_variant", {}).get("name")
                    if isinstance(item.get("tile_shape_variant"), dict)
                    else None
                ),
                "speedup_vs_direct_hip": (direct / selection_e2e) if direct and selection_e2e else None,
                "speedup_vs_vector_alu": (vector / selection_e2e) if vector and selection_e2e else None,
                "exact_output_contract": (
                    item.get("exact_output_contract") if isinstance(item.get("exact_output_contract"), dict) else None
                ),
                "export_variant": item.get("export_variant") if isinstance(item.get("export_variant"), dict) else None,
                "reconstruction_variant": (
                    item.get("reconstruction_variant") if isinstance(item.get("reconstruction_variant"), dict) else None
                ),
                "prepacked_reuse_review": prepack_review,
                "runtime_prepack_cache": (
                    item.get("reuse_contract", {}).get("runtime_prepack_cache")
                    if isinstance(item.get("reuse_contract"), dict)
                    else None
                ),
                "hip_graph_replay_review": graph_review,
                "matrix_instruction_histogram": matrix_histogram,
                "matrix_instruction_family": metadata.get("matrix_instruction_family"),
                "matrix_instruction_shape": metadata.get("matrix_instruction_shape"),
                "matrix_instruction_dtype": metadata.get("matrix_instruction_dtype"),
                "matrix_instruction_sparsity": metadata.get("matrix_instruction_sparsity"),
                "expected_matrix_instruction_mnemonic": expected_matrix_mnemonic,
                "promotable": promotable,
                "promotion_blockers": blockers,
                "promotion_reason": "beats_required_same_contract_gpu_baselines" if promotable else "blocked",
                "primary_loss_phase_vs_direct_hip": None if promotable else primary_loss_phase(item, direct_baseline),
                "bottleneck": bottleneck_classification(item),
                "cache_write_status": "eligible_after_review" if promotable else "not_eligible",
            }
            candidates.append(candidate)

        fastest_production_route = fastest_route(candidates, accelerator_only=False)
        fastest_accelerator_route = fastest_route(candidates, accelerator_only=True)

        fastest = None
        promotable_candidates = [item for item in candidates if item["promotable"]]
        if promotable_candidates:
            fastest = min(promotable_candidates, key=lambda item: item["selection_end_to_end_us"])
            for item in candidates:
                if item is not fastest and item["promotable"]:
                    item["promotable"] = False
                    item["promotion_blockers"] = ["not_fastest_promotable_accelerator"]
                    item["promotion_reason"] = "blocked"
                    item["cache_write_status"] = "not_eligible"
            source = by_backend.get(fastest["backend"])
            if source is not None:
                metadata = source.get("backend_metadata") if isinstance(source.get("backend_metadata"), dict) else {}
                promotable_entries.append(
                    {
                        "source_capture": source.get("_path"),
                        "autotune_key": metadata.get("autotune_key"),
                        "selected_backend": fastest["backend"],
                        "selected_kernel": fastest["selected_kernel"],
                        "median_end_to_end_us": fastest["median_end_to_end_us"],
                        "selection_end_to_end_us": fastest["selection_end_to_end_us"],
                        "target_id": candidate_source_metadata(source).get("target_id"),
                        "hip_sdk_or_rocm_version": candidate_source_metadata(source).get("hip_sdk_or_rocm_version"),
                        "accelerator_library": metadata.get("accelerator_library"),
                        "accelerator_version": metadata.get("accelerator_version"),
                        "workspace_bytes": metadata.get("workspace_required_bytes", 0),
                        "winner_rationale": "fastest_promotable_same_contract_accelerator",
                        "cache_write_status": "pending",
                    }
                )

        group_shape = {"m": items[0].get("m"), "n": items[0].get("n"), "k": items[0].get("k")}
        groups.append(
            {
                "contract_key": key,
                "semantics": semantics,
                "finite_modulus": items[0].get("finite_modulus"),
                "shape": group_shape,
                "shape_family": shape_family(group_shape.get("m"), group_shape.get("n"), group_shape.get("k")),
                "capture_count": len(items),
                "scenario_families": scenario_values(items, "family"),
                "scenario_names": scenario_values(items, "name"),
                "source_metadata": group_source_metadata(items),
                "review_mode": review_mode,
                "release_review_satisfied": release_review_satisfied,
                "release_review_requirements": {
                    "min_warmups": RELEASE_MIN_WARMUPS,
                    "min_repeats": RELEASE_MIN_REPEATS,
                },
                "required_baselines": required,
                "missing_required_baselines": missing,
                "gpu_targets": gpu_targets,
                "missing_gpu_targets": missing_gpu_targets,
                "gpu_target_identity_complete": gpu_target_identity_complete,
                "gpu_target_compatible": gpu_target_compatible,
                "configured_gpu_targets": configured_gpu_targets,
                "missing_configured_gpu_targets": missing_configured_gpu_targets,
                "configured_target_identity_complete": configured_target_identity_complete,
                "configured_target_compatible": configured_target_compatible,
                "hip_toolchain_versions": hip_toolchain_versions,
                "missing_hip_toolchain_versions": missing_hip_toolchain_versions,
                "hip_toolchain_version_complete": hip_toolchain_version_complete,
                "hip_toolchain_version_compatible": hip_toolchain_version_compatible,
                "hip_runtime_versions": hip_runtime_versions,
                "missing_hip_runtime_versions": missing_hip_runtime_versions,
                "hip_runtime_version_complete": hip_runtime_version_complete,
                "hip_runtime_version_compatible": hip_runtime_version_compatible,
                "hip_driver_versions": hip_driver_versions,
                "missing_hip_driver_versions": missing_hip_driver_versions,
                "hip_driver_version_complete": hip_driver_version_complete,
                "hip_driver_version_compatible": hip_driver_version_compatible,
                "compiler_identities": compiler_identities,
                "missing_compiler_identities": missing_compiler_identities,
                "compiler_identity_complete": compiler_identity_complete,
                "compiler_identity_compatible": compiler_identity_compatible,
                "git_commits": git_commits,
                "missing_git_commits": missing_git_commits,
                "git_commit_identity_complete": git_commit_identity_complete,
                "git_commit_identity_compatible": git_commit_identity_compatible,
                "warmup_counts": warmup_counts,
                "missing_warmup_counts": missing_warmup_counts,
                "warmup_count_complete": warmup_count_complete,
                "warmup_count_compatible": warmup_count_compatible,
                "repeat_counts": repeat_counts,
                "missing_repeat_counts": missing_repeat_counts,
                "repeat_count_complete": repeat_count_complete,
                "repeat_count_compatible": repeat_count_compatible,
                "duplicate_backends": duplicate_backends,
                "checksum_reference_backend": checksum_reference_backend,
                "checksum_reference": checksum_reference,
                "checksum_by_backend": checksum_by_backend,
                "missing_checksums": missing_checksums,
                "checksum_mismatches": checksum_mismatches,
                "checksum_consistent": checksum_consistent,
                "scenario_promotion_scopes": scenario_promotion_scopes,
                "phase_medians_us": phase_medians,
                "phase_ratio_summary": [phase_ratio_summary_for_candidate(candidate) for candidate in candidates],
                "fastest_promotable": fastest,
                "fastest_production_route": fastest_production_route,
                "fastest_accelerator_route": fastest_accelerator_route,
                "candidates": candidates,
            }
        )

    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_mode": review_mode,
        "release_review_requirements": {
            "min_warmups": RELEASE_MIN_WARMUPS,
            "min_repeats": RELEASE_MIN_REPEATS,
        },
        "reviewed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "group_count": len(groups),
        "groups": groups,
        "promotable_autotune_entries": promotable_entries,
        "summary": build_review_summary(groups, promotable_entries),
        "cache_write": {
            "requested": False,
            "path": None,
            "entries_written": 0,
            "status": "not_requested",
        },
    }


def cache_entry_from_capture(capture: dict[str, Any], validation_status: str) -> dict[str, Any]:
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    medians = capture.get("timing_summary_us") if isinstance(capture.get("timing_summary_us"), dict) else {}
    schedule = capture.get("schedule_metadata") if isinstance(capture.get("schedule_metadata"), dict) else {}
    tile_bounds = capture.get("tile_bounds_u64") if isinstance(capture.get("tile_bounds_u64"), dict) else {}
    export_variant = capture.get("export_variant") if isinstance(capture.get("export_variant"), dict) else {}
    reconstruction_variant = (
        capture.get("reconstruction_variant") if isinstance(capture.get("reconstruction_variant"), dict) else {}
    )
    export_variant_name = str(export_variant.get("name") or "default")
    reconstruction_variant_name = str(reconstruction_variant.get("name") or "default_garner")
    export_selector_key = export_variant.get("selector_key")
    export_selector_hash = (
        hashlib.sha256(export_selector_key.encode("utf-8")).hexdigest()[:16]
        if isinstance(export_selector_key, str) and export_selector_key
        else None
    )
    default_export_contract = export_variant_name == "default" and reconstruction_variant_name == "default_garner"
    selected_prefix = capture.get("selected_prefix", schedule.get("max_selected_prefix"))
    requested_max_prefix = capture.get("requested_max_prefix", capture.get("prefix"))
    prefix_policy = capture.get("contract_prefix_policy", "legacy_v4_unspecified")
    bound_source = capture_bound_source(capture)
    prefix_schedule_hash = (
        f"tile_rows={schedule.get('tile_rows')};tile_cols={schedule.get('tile_cols')};"
        f"selected_prefix={selected_prefix};requested_max_prefix={requested_max_prefix};"
        f"prefix_policy={prefix_policy};bound_source={bound_source};"
        f"groups={schedule.get('prefix_group_count')};"
        f"adaptive_prefix={int(bool(schedule.get('adaptive_prefix_active')))};"
        f"adaptive_skip={int(bool(schedule.get('adaptive_skip_active')))};"
        f"schedule_flags={schedule.get('flags', 0)};"
        f"zero_output_tiles={schedule.get('zero_output_tile_count', 0)};"
        f"zero_a_rows={schedule.get('zero_a_row_proof_count', 0)};"
        f"zero_b_cols={schedule.get('zero_b_col_proof_count', 0)};"
        f"zero_row_col_products={schedule.get('zero_row_col_product_count', 0)};"
        f"tile_bound_hash={tile_bounds.get('hash_u64', 0)}"
    )
    hip_toolchain = capture.get("hip_toolchain") if isinstance(capture.get("hip_toolchain"), dict) else {}
    device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
    version = metadata.get("accelerator_version") or hip_toolchain.get("hip_sdk_or_rocm_version") or "unknown"
    key = metadata.get("autotune_key")
    if isinstance(key, str) and key and not default_export_contract and export_selector_hash:
        key = (
            f"{key};export_variant={export_variant_name};"
            f"reconstruction_variant={reconstruction_variant_name};"
            f"export_selector_hash={export_selector_hash}"
        )

    def median(phase: str) -> float:
        item = medians.get(phase) if isinstance(medians, dict) else None
        if isinstance(item, dict) and isinstance(item.get("median"), (int, float)):
            return float(item["median"])
        return 0.0

    return {
        "key": key,
        "selected_backend": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "target_id": normalized_target_id(device.get("gcn_arch")) or "cpu",
        "hip_sdk_or_library_version": version,
        "semantic_contract": capture.get("semantics"),
        "finite_modulus": capture.get("finite_modulus") or 0,
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "layout": capture.get("layout"),
        "prefix_schedule_hash": prefix_schedule_hash,
        "k_block_size": capture.get("k_block_size"),
        "tile_m": capture.get("tile_m"),
        "tile_n": capture.get("tile_n"),
        "epilogue": metadata.get("epilogue_mode"),
        "kernel_family": metadata.get("selected_kernel") or capture.get("selected_kernel"),
        "workspace_bytes": metadata.get("workspace_required_bytes", 0),
        "export_variant": export_variant_name,
        "reconstruction_variant": reconstruction_variant_name,
        "export_selector_key": export_selector_key,
        "export_selector_hash": export_selector_hash,
        "export_selector_policy": export_variant.get("selector_policy"),
        "export_cache_visibility": export_variant.get("cache_visibility"),
        "export_stale_entry_reason": export_variant.get("stale_entry_reason"),
        "cache_scope": "runtime_exact_autotune" if default_export_contract else "selector_review_only_non_default",
        "pack_mode": capture_pack_mode(capture),
        "reuse_packed_inputs": capture_prepacked_reuse(capture),
        "prepack_reuse_operands": capture.get("prepack_reuse_operands"),
        "prepack_reuse_strategy": capture.get("prepack_reuse_strategy"),
        "prepack_setup_us": numeric_capture_value(capture, "avg_prepack_setup_us"),
        "setup_inclusive_median_end_to_end_us": selection_end_to_end_us(capture),
        "measured_medians_us": {
            "pack": median("pack"),
            "pack_a": median("pack_a"),
            "pack_b": median("pack_b"),
            "rns_gemm": median("rns_gemm"),
            "crt_export": median("crt_export"),
            "end_to_end": median("end_to_end"),
        },
        "performance_validated": True,
        "validation_status": validation_status,
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def reviewed_release_status_for_target(target_id: str | None) -> str:
    target = str(target_id or "")
    if target == "gfx1100":
        return "reviewed_release_same_contract_fastest_windows_gfx1100"
    if target in {"gfx90a", "gfx942", "gfx950"}:
        return f"reviewed_release_same_contract_fastest_linux_{target}"
    if target.startswith("gfx"):
        return f"reviewed_release_same_contract_fastest_target_{target}"
    return "reviewed_release_unknown_target"


def write_promoted_cache_entries(report: dict[str, Any], captures: list[dict[str, Any]], path: Path) -> int:
    promotable = report.get("promotable_autotune_entries")
    if not isinstance(promotable, list) or not promotable:
        return 0
    by_path = {str(capture.get("_path")): capture for capture in captures}
    entries = []
    for item in promotable:
        if not isinstance(item, dict):
            continue
        capture = by_path.get(str(item.get("source_capture")))
        if not capture:
            continue
        device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
        target_id = normalized_target_id(device.get("gcn_arch")) or "cpu"
        entry = cache_entry_from_capture(capture, reviewed_release_status_for_target(target_id))
        if entry.get("key"):
            entries.append(entry)
    if not entries:
        return 0

    existing: dict[str, Any] = {"schema_version": 1, "entries": []}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {"schema_version": 1, "entries": []}
    existing_entries = existing.get("entries")
    if not isinstance(existing_entries, list):
        existing_entries = []
    by_key = {entry.get("key"): entry for entry in existing_entries if isinstance(entry, dict) and entry.get("key")}
    for entry in entries:
        by_key[entry["key"]] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "entries": list(by_key.values())}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for item in promotable:
        if isinstance(item, dict):
            item["cache_write_status"] = "written"
    return len(entries)


def attach_cache_write_status(report: dict[str, Any], requested: bool, path: Path, entries_written: int) -> None:
    report["cache_write"] = {
        "requested": requested,
        "path": str(path) if requested else None,
        "entries_written": entries_written,
        "status": "written" if requested and entries_written else "no_promotable_entries" if requested else "not_requested",
    }
    if not requested:
        for item in report.get("promotable_autotune_entries", []):
            if isinstance(item, dict):
                item["cache_write_status"] = "not_requested"
        return
    written_sources = {
        str(item.get("source_capture"))
        for item in report.get("promotable_autotune_entries", [])
        if isinstance(item, dict) and item.get("cache_write_status") == "written" and item.get("source_capture")
    }
    for group in report.get("groups", []):
        if not isinstance(group, dict):
            continue
        for candidate in group.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("capture")) in written_sources:
                candidate["cache_write_status"] = "written"
            elif candidate.get("promotable"):
                candidate["cache_write_status"] = "pending"


