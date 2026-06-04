#!/usr/bin/env python3
"""Self-test benchmark schema validation fixtures."""

from __future__ import annotations

import copy
from pathlib import Path

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def expect_valid(name: str) -> dict:
    path = FIXTURE_DIR / name
    data = load_capture(path)
    validate_capture(data, path)
    return data


def expect_invalid(data: dict, needle: str) -> None:
    try:
        validate_capture(data)
    except BenchmarkSchemaError as exc:
        message = str(exc)
        if needle not in message:
            raise AssertionError(f"expected validation error containing {needle!r}, got {message!r}") from exc
        return
    raise AssertionError(f"expected validation error containing {needle!r}")


def zero_summary() -> dict:
    return {"avg": 0.0, "median": 0.0, "p95": 0.0}


def summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"avg": sum(values) / len(values), "median": ordered[len(ordered) // 2], "p95": ordered[-1]}


def add_prefix_policy_fields(capture: dict, policy: str) -> dict:
    requested = capture["prefix"]
    selected = capture["schedule_metadata"]["max_selected_prefix"]
    skipped = max(requested - selected, 0)
    capture["selected_prefix"] = selected
    capture["requested_max_prefix"] = requested
    capture["contract_prefix_policy"] = policy
    capture["residue_planes_requested"] = requested
    capture["residue_planes_selected"] = selected
    capture["residue_planes_skipped"] = skipped
    capture["residue_plane_skip_fraction"] = float(skipped) / float(requested) if requested else 0.0
    return capture


def add_global_bound_scan_fields(capture: dict) -> dict:
    bound = capture["bound"]
    capture["bound_source"] = "input_scan"
    capture["bound_discovery"] = {
        "source": "input_row_column_abs_summary",
        "static_bound": bound,
        "selected_bound": bound,
        "discovered_global_bound": bound,
        "candidate_row_sum_col_max": bound,
        "candidate_row_max_col_sum": bound,
        "row_abs_sum_max": max(bound, 1),
        "row_abs_max": 16,
        "col_abs_sum_max": max(bound, 1),
        "col_abs_max": 16,
        "zero_row_count": 0,
        "zero_col_count": 0,
    }
    capture["global_bound_scan_us"] = 7
    capture["avg_global_bound_scan_us"] = 7.0
    capture["raw_timings_us"]["global_bound_scan"] = [7]
    capture["timing_summary_us"] = {
        "global_bound_scan": {"avg": 7.0, "median": 7.0, "p95": 7.0},
        **capture["timing_summary_us"],
    }
    phase_order = capture["timing_metadata"]["phase_order"]
    capture["timing_metadata"]["phase_order"] = ["global_bound_scan", *phase_order]
    capture["timing_metadata"]["phase_notes"] = {
        "global_bound_scan": "one-time exact seeded input prepass that computes row/column absolute-summary global bounds before plan creation",
        **capture["timing_metadata"]["phase_notes"],
    }
    capture["timing_metadata"]["phase_availability"] = {
        "global_bound_scan": {
            "timed": True,
            "timing_key": "global_bound_scan",
            "scope": "input_row_column_abs_summary",
            "reason": "measured with host steady_clock around seeded input row/column absolute-summary bound discovery before plan creation",
        },
        **capture["timing_metadata"]["phase_availability"],
    }
    return capture


def add_per_tile_input_scan_fields(capture: dict) -> dict:
    capture["bound_source"] = "input_scan"
    capture["bound_discovery"] = {
        "source": "input_exact_tile_bounds",
        "static_bound": 0,
        "selected_bound": 0,
        "discovered_global_bound": None,
        "candidate_row_sum_col_max": None,
        "candidate_row_max_col_sum": None,
        "row_abs_sum_max": None,
        "row_abs_max": None,
        "col_abs_sum_max": None,
        "col_abs_max": None,
        "zero_row_count": None,
        "zero_col_count": None,
    }
    capture["command_line"] = f"{capture['command_line']} --bound-source input-scan"
    return capture


def int32_accumulator_safety(capture: dict, cap: int = 65536) -> dict:
    k_block = min(capture["k"], cap)
    finite = capture.get("semantics") in {"finite_ring_u8", "finite_field_u8"}
    return {
        "input_domain": "centered_i8_finite_u8_residues" if finite else "centered_i8_rns_residue_planes",
        "signedness": "signed_i8x_signed_i8",
        "accumulator_type": "int32",
        "modulus_policy": "finite_u8_modulus" if finite else "selected_rns_modulus_ladder",
        "modulus": capture.get("finite_modulus") if finite else 0,
        "uses_int32_inner_product": True,
        "k_block_size": k_block,
        "k_block_cap": cap,
        "max_lhs_abs": 128,
        "max_rhs_abs": 128,
        "max_product": 128 * 128,
        "safe_for_k_block": True,
        "status": "safe_int32_k_block_split",
    }


def apply_int32_accumulator_contract(capture: dict, cap: int = 65536) -> dict:
    safety = int32_accumulator_safety(capture, cap)
    capture["backend_metadata"]["accumulator_safety"] = safety
    capture["k_block_size"] = safety["k_block_size"]
    return capture


def with_accumulator_key_fields(key: str, capture: dict) -> str:
    safety = capture["backend_metadata"]["accumulator_safety"]
    parts = [
        part
        for part in key.split(";")
        if part.split("=", 1)[0]
        not in {
            "target_id",
            "accumulator_type",
            "accumulator_signedness",
            "accumulator_modulus_policy",
            "k_block_size",
            "k_block_cap",
        }
    ]
    target = capture.get("device", {}).get("gcn_arch", "cpu")
    if capture.get("backend_selected") not in {"hip-direct", "hipblaslt", "ck", "rocwmma", "hip-vector-alu-int64"}:
        target = "cpu"
    if target in {"", "none", "unknown"}:
        target = "cpu"
    insert_target_at = 1 if parts and parts[0].startswith("backend=") else 0
    parts = parts[:insert_target_at] + [f"target_id={target}"] + parts[insert_target_at:]
    insert_at = next((index for index, part in enumerate(parts) if part.startswith("kernel=")), len(parts))
    additions = [
        f"accumulator_type={safety['accumulator_type']}",
        f"accumulator_signedness={safety['signedness']}",
        f"accumulator_modulus_policy={safety['modulus_policy']}",
        f"k_block_size={safety['k_block_size']}",
        f"k_block_cap={safety['k_block_cap']}",
    ]
    return ";".join(parts[:insert_at] + additions + parts[insert_at:])


def as_direct_hip_finite_capture(
    capture: dict, modulus: int, kernel: str, isa_evidence: str
) -> dict:
    direct = copy.deepcopy(capture)
    metadata = direct["backend_metadata"]
    direct["backend_requested"] = "hip-direct"
    direct["backend_selected"] = "hip-direct"
    direct["selected_kernel"] = kernel
    direct["finite_modulus"] = modulus
    direct["backend_metadata"]["selected_kernel"] = kernel
    metadata["accelerator_backend"] = False
    metadata["matrix_engine_backend"] = False
    metadata["accelerator_library"] = "HIP runtime"
    metadata["accelerator_version"] = None
    metadata["capability_status"] = "implemented_correctness_backend"
    metadata["epilogue_mode"] = "fused_centered_residue_then_canonical_u8_export"
    metadata["workspace_mode"] = "resident_device_buffers"
    metadata["workspace_required_bytes"] = 0
    metadata["isa_evidence"] = isa_evidence
    apply_int32_accumulator_contract(direct)
    metadata["autotune_key"] = with_accumulator_key_fields(
        (
        f"backend=hip-direct;semantics={direct['semantics']};m={direct['m']};n={direct['n']};k={direct['k']};"
        f"finite_modulus={modulus};prefix=0;tile_m={direct['tile_m']};tile_n={direct['tile_n']};"
        f"groups=0;adaptive_prefix=0;adaptive_skip=0;kernel={kernel};"
        "epilogue=fused_centered_residue_then_canonical_u8_export"
        ),
        direct,
    )
    direct["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    return direct


def as_direct_hip_oneshot_capture(capture: dict) -> dict:
    oneshot = copy.deepcopy(capture)
    repeats = oneshot["repeats"]
    kernel = "direct_hip_prefix9_native_input_grouped_rns_gemm_v1"
    epilogue = "native_input_centered_residue_then_crt_export"
    oneshot["benchmark"] = "rns8_bounded_gemm_public_oneshot"
    oneshot["benchmark_execution_mode"] = "public_oneshot_transient_native_inputs"
    oneshot["backend_requested"] = "hip-direct"
    oneshot["backend_selected"] = "hip-direct"
    oneshot["selected_kernel"] = kernel
    oneshot["backend_metadata"]["source"] = "rns8_bench_public_oneshot_api"
    oneshot["backend_metadata"]["selected_kernel"] = kernel
    oneshot["backend_metadata"]["accelerator_backend"] = False
    oneshot["backend_metadata"]["matrix_engine_backend"] = False
    oneshot["backend_metadata"]["accelerator_library"] = "HIP runtime"
    oneshot["backend_metadata"]["accelerator_version"] = "7.1"
    oneshot["backend_metadata"]["capability_status"] = "implemented_correctness_backend"
    oneshot["backend_metadata"]["epilogue_mode"] = epilogue
    oneshot["backend_metadata"]["workspace_mode"] = "transient_native_inputs_to_resident_rns_output"
    oneshot["backend_metadata"]["workspace_required_bytes"] = 0
    oneshot["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(oneshot)
    oneshot["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;prefix=9;tile_m=128;tile_n=128;"
        "groups=1;adaptive_prefix=0;adaptive_skip=0;execution=public_oneshot_transient_native_inputs;"
        f"kernel={kernel};epilogue={epilogue}"
        ),
        oneshot,
    )
    oneshot["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_vector_alu_int64",
        "same_contract_direct_hip_persistent_rns",
    ]
    oneshot["timing_note"] = (
        "host wall-clock timings for the public bounded one-shot API; raw_timings_us.rns_gemm and "
        "raw_timings_us.end_to_end both measure one complete call"
    )
    oneshot["timing_metadata"]["benchmark_execution_mode"] = "public_oneshot_transient_native_inputs"
    oneshot["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_oneshot_api_hooks"
    oneshot["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_oneshot_default_stream_operation_groups"
    oneshot["timing_metadata"]["gpu_event_phase_order"] = [
        "oneshot_native_input_h2d",
        "rns_gemm_kernel_group",
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
        "oneshot_api_gpu",
    ]
    oneshot["timing_metadata"]["phase_notes"]["matrix_alloc"] = (
        "zero-valued external phase; transient API allocations are inside the measured one-shot call"
    )
    oneshot["timing_metadata"]["phase_notes"]["pack"] = (
        "zero-valued external phase; native input copies are inside the measured one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for one complete public bounded one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["crt_export"] = (
        "zero-valued external phase; logical output export is inside the measured one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "same measured duration as rns_gemm for one complete public bounded one-shot API call"
    )
    oneshot["matrix_alloc_us"] = 0
    oneshot["avg_matrix_alloc_us"] = 0.0
    oneshot["avg_pack_us"] = 0.0
    oneshot["avg_crt_export_us"] = 0.0
    oneshot["avg_rns_gemm_us"] = 1000.0
    oneshot["avg_end_to_end_us"] = 1000.0
    oneshot["per_modulus_gemm_estimate_applicable"] = False
    oneshot["avg_per_modulus_gemm_estimate_us"] = 1000.0
    oneshot["raw_timings_us"]["matrix_alloc"] = [0]
    oneshot["raw_timings_us"]["pack"] = [0] * repeats
    oneshot["raw_timings_us"]["rns_gemm"] = [900, 1100]
    oneshot["raw_timings_us"]["crt_export"] = [0] * repeats
    oneshot["raw_timings_us"]["end_to_end"] = [900, 1100]
    oneshot["timing_summary_us"]["matrix_alloc"] = zero_summary()
    oneshot["timing_summary_us"]["pack"] = zero_summary()
    oneshot["timing_summary_us"]["rns_gemm"] = {"avg": 1000.0, "median": 1100.0, "p95": 1100.0}
    oneshot["timing_summary_us"]["crt_export"] = zero_summary()
    oneshot["timing_summary_us"]["end_to_end"] = {"avg": 1000.0, "median": 1100.0, "p95": 1100.0}
    event_values = {
        "oneshot_native_input_h2d": [10.0, 12.0],
        "rns_gemm_kernel_group": [100.0, 110.0],
        "rns_gemm": [100.0, 110.0],
        "crt_export_status_memset": [0.0, 0.0],
        "crt_export_kernel": [20.0, 22.0],
        "crt_export_status_d2h": [1.0, 1.0],
        "crt_export_d2h": [8.0, 9.0],
        "crt_export": [29.0, 32.0],
        "oneshot_api_gpu": [139.0, 154.0],
    }
    oneshot["gpu_event_timings_us"] = event_values
    oneshot["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    return oneshot


def as_direct_hip_finite_oneshot_capture(capture: dict) -> dict:
    oneshot = as_direct_hip_finite_capture(
        capture,
        255,
        "direct_hip_native_finite_u8_gemm_mod255_v1",
        "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide",
    )
    repeats = oneshot["repeats"]
    epilogue = "native_u8_centered_residue_then_canonical_u8_export"
    oneshot["benchmark"] = "rns8_finite_u8_public_oneshot"
    oneshot["benchmark_execution_mode"] = "public_oneshot_transient_native_inputs"
    oneshot["backend_metadata"]["source"] = "rns8_bench_public_oneshot_api"
    oneshot["backend_metadata"]["epilogue_mode"] = epilogue
    oneshot["backend_metadata"]["workspace_mode"] = "transient_native_u8_inputs_to_resident_finite_output"
    oneshot["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=finite_ring_u8;m=64;n=128;k=64;finite_modulus=255;"
        "tile_m=128;tile_n=128;execution=public_oneshot_transient_native_inputs;"
        "kernel=direct_hip_native_finite_u8_gemm_mod255_v1;"
        f"epilogue={epilogue}"
        ),
        oneshot,
    )
    oneshot["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_persistent_finite_u8",
    ]
    oneshot["timing_note"] = (
        "host wall-clock timings for the public finite-u8 one-shot API; raw_timings_us.rns_gemm and "
        "raw_timings_us.end_to_end both measure one complete call"
    )
    oneshot["timing_metadata"]["benchmark_execution_mode"] = "public_oneshot_transient_native_inputs"
    oneshot["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_oneshot_api_hooks"
    oneshot["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_oneshot_default_stream_operation_groups"
    oneshot["timing_metadata"]["gpu_event_phase_order"] = [
        "oneshot_native_input_h2d",
        "finite_native_gemm_kernel",
        "rns_gemm",
        "finite_export_kernel",
        "finite_export_d2h",
        "crt_export",
        "oneshot_api_gpu",
    ]
    oneshot["timing_metadata"]["phase_notes"]["matrix_alloc"] = (
        "zero-valued external phase; transient API allocations are inside the measured one-shot call"
    )
    oneshot["timing_metadata"]["phase_notes"]["pack"] = (
        "zero-valued external phase; native input copies are inside the measured one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for one complete public one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["crt_export"] = (
        "zero-valued external phase; logical output export is inside the measured one-shot API call"
    )
    oneshot["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "same measured duration as rns_gemm for one complete public one-shot API call"
    )
    oneshot["matrix_alloc_us"] = 0
    oneshot["avg_matrix_alloc_us"] = 0.0
    oneshot["avg_pack_us"] = 0.0
    oneshot["avg_crt_export_us"] = 0.0
    oneshot["avg_rns_gemm_us"] = 1000.0
    oneshot["avg_end_to_end_us"] = 1000.0
    oneshot["per_modulus_gemm_estimate_applicable"] = False
    oneshot["avg_per_modulus_gemm_estimate_us"] = 1000.0
    oneshot["raw_timings_us"]["matrix_alloc"] = [0]
    oneshot["raw_timings_us"]["pack"] = [0] * repeats
    oneshot["raw_timings_us"]["rns_gemm"] = [900, 1100][:repeats]
    oneshot["raw_timings_us"]["crt_export"] = [0] * repeats
    oneshot["raw_timings_us"]["end_to_end"] = [900, 1100][:repeats]
    oneshot["timing_summary_us"]["matrix_alloc"] = zero_summary()
    oneshot["timing_summary_us"]["pack"] = zero_summary()
    oneshot["timing_summary_us"]["rns_gemm"] = summary(oneshot["raw_timings_us"]["rns_gemm"])
    oneshot["timing_summary_us"]["crt_export"] = zero_summary()
    oneshot["timing_summary_us"]["end_to_end"] = summary(oneshot["raw_timings_us"]["end_to_end"])
    event_values = {
        "oneshot_native_input_h2d": [10.0, 12.0][:repeats],
        "finite_native_gemm_kernel": [100.0, 110.0][:repeats],
        "rns_gemm": [100.0, 110.0][:repeats],
        "finite_export_kernel": [20.0, 22.0][:repeats],
        "finite_export_d2h": [8.0, 9.0][:repeats],
        "crt_export": [28.0, 31.0][:repeats],
        "oneshot_api_gpu": [138.0, 153.0][:repeats],
    }
    oneshot["gpu_event_timings_us"] = event_values
    oneshot["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    return oneshot


def as_direct_hip_finite_native_a_reuse_b_capture(capture: dict) -> dict:
    reused = as_direct_hip_finite_capture(
        capture,
        255,
        "direct_hip_native_a_finite_u8_gemm_mod255_v1",
        "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide",
    )
    repeats = reused["repeats"]
    epilogue = "native_a_centered_resident_b_residue_then_canonical_u8_export"
    reused["benchmark_execution_mode"] = "transient_native_a_resident_b_reuse"
    reused["backend_metadata"]["source"] = "rns8_bench_native_a_reuse_b_path"
    reused["backend_metadata"]["epilogue_mode"] = epilogue
    reused["backend_metadata"]["workspace_mode"] = "transient_native_u8_a_resident_finite_b_output"
    reused["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=finite_ring_u8;m=64;n=128;k=64;finite_modulus=255;"
        "tile_m=128;tile_n=128;execution=transient_native_a_resident_b_reuse;"
        "kernel=direct_hip_native_a_finite_u8_gemm_mod255_v1;"
        f"epilogue={epilogue}"
        ),
        reused,
    )
    reused["pack_mode"] = "prepacked_reuse_b"
    reused["reuse_packed_inputs"] = True
    reused["prepack_reuse_operands"] = ["B"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 300
    reused["avg_prepack_setup_us"] = 300.0
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "B was packed once before warmups and reused for every measured repeat",
    }
    reused["timing_metadata"]["benchmark_execution_mode"] = "transient_native_a_resident_b_reuse"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["B"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["gpu_event_phase_order"] = [
        "finite_pack_h2d",
        "finite_pack_kernel",
        "pack",
        "finite_native_a_gemm_kernel",
        "rns_gemm",
        "finite_export_kernel",
        "finite_export_d2h",
        "crt_export",
    ]
    event_values = {
        "finite_pack_h2d": [12.0, 13.0][:repeats],
        "finite_pack_kernel": [0.0, 0.0][:repeats],
        "pack": [12.0, 13.0][:repeats],
        "finite_native_a_gemm_kernel": [80.0, 82.0][:repeats],
        "rns_gemm": [80.0, 82.0][:repeats],
        "finite_export_kernel": [20.0, 22.0][:repeats],
        "finite_export_d2h": [8.0, 9.0][:repeats],
        "crt_export": [28.0, 31.0][:repeats],
    }
    reused["gpu_event_timings_us"] = event_values
    reused["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    reused["raw_timings_us"]["pack"] = [100, 110][:repeats]
    reused["raw_timings_us"]["rns_gemm"] = [200, 210][:repeats]
    reused["raw_timings_us"]["crt_export"] = [90, 100][:repeats]
    reused["raw_timings_us"]["end_to_end"] = [390, 420][:repeats]
    reused["timing_summary_us"]["pack"] = summary(reused["raw_timings_us"]["pack"])
    reused["timing_summary_us"]["rns_gemm"] = summary(reused["raw_timings_us"]["rns_gemm"])
    reused["timing_summary_us"]["crt_export"] = summary(reused["raw_timings_us"]["crt_export"])
    reused["timing_summary_us"]["end_to_end"] = summary(reused["raw_timings_us"]["end_to_end"])
    reused["avg_pack_us"] = reused["timing_summary_us"]["pack"]["avg"]
    reused["avg_rns_gemm_us"] = reused["timing_summary_us"]["rns_gemm"]["avg"]
    reused["avg_crt_export_us"] = reused["timing_summary_us"]["crt_export"]["avg"]
    reused["avg_end_to_end_us"] = reused["timing_summary_us"]["end_to_end"]["avg"]
    return reused


def as_direct_hip_bounded_native_a_reuse_b_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    repeats = reused["repeats"]
    kernel = "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_b_grouped_rns_gemm_v2"
    epilogue = "uniform_small_i8_ab_resident_b_residue_then_crt_export"
    gemm_event = "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    reused["benchmark_execution_mode"] = "transient_uniform_small_i8_a_resident_i8_b_reuse"
    reused["backend_requested"] = "hip-direct"
    reused["backend_selected"] = "hip-direct"
    reused["selected_kernel"] = kernel
    reused["backend_metadata"]["source"] = "rns8_bench_uniform_small_i8_ab_reuse_b_path"
    reused["backend_metadata"]["selected_kernel"] = kernel
    reused["backend_metadata"]["accelerator_backend"] = False
    reused["backend_metadata"]["matrix_engine_backend"] = False
    reused["backend_metadata"]["accelerator_library"] = "HIP runtime"
    reused["backend_metadata"]["accelerator_version"] = "7.1"
    reused["backend_metadata"]["capability_status"] = "implemented_correctness_backend"
    reused["backend_metadata"]["epilogue_mode"] = epilogue
    reused["backend_metadata"]["workspace_mode"] = "transient_native_a_resident_rns_b_output"
    reused["backend_metadata"]["workspace_required_bytes"] = 0
    reused["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(reused)
    reused["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
        "input_profile=uniform-small;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_uniform_small_i8_a_resident_i8_b_reuse;"
        f"kernel={kernel};epilogue={epilogue}"
        ),
        reused,
    )
    reused["pack_mode"] = "prepacked_reuse_b"
    reused["reuse_packed_inputs"] = True
    reused["prepack_reuse_operands"] = ["B"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 400
    reused["avg_prepack_setup_us"] = 400.0
    reused["timing_metadata"]["benchmark_execution_mode"] = "transient_uniform_small_i8_a_resident_i8_b_reuse"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["B"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    reused["timing_metadata"]["gpu_event_phase_order"] = [
        "pack_h2d",
        "pack_kernel",
        "pack",
        gemm_event,
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
    ]
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "B was packed once before warmups and reused for every measured repeat",
    }
    event_values = {
        "pack_h2d": [18.0, 19.0][:repeats],
        "pack_kernel": [0.0, 0.0][:repeats],
        "pack": [18.0, 19.0][:repeats],
        gemm_event: [150.0, 151.0][:repeats],
        "rns_gemm": [150.0, 151.0][:repeats],
        "crt_export_status_memset": [0.5, 0.5][:repeats],
        "crt_export_kernel": [30.0, 31.0][:repeats],
        "crt_export_status_d2h": [1.0, 1.0][:repeats],
        "crt_export_d2h": [10.0, 11.0][:repeats],
        "crt_export": [41.5, 43.5][:repeats],
    }
    reused["gpu_event_timings_us"] = event_values
    reused["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    reused["raw_timings_us"]["pack"] = [120, 125][:repeats]
    reused["raw_timings_us"]["rns_gemm"] = [260, 270][:repeats]
    reused["raw_timings_us"]["crt_export"] = [95, 100][:repeats]
    reused["raw_timings_us"]["end_to_end"] = [475, 495][:repeats]
    reused["timing_summary_us"]["pack"] = summary(reused["raw_timings_us"]["pack"])
    reused["timing_summary_us"]["rns_gemm"] = summary(reused["raw_timings_us"]["rns_gemm"])
    reused["timing_summary_us"]["crt_export"] = summary(reused["raw_timings_us"]["crt_export"])
    reused["timing_summary_us"]["end_to_end"] = summary(reused["raw_timings_us"]["end_to_end"])
    reused["avg_pack_us"] = reused["timing_summary_us"]["pack"]["avg"]
    reused["avg_rns_gemm_us"] = reused["timing_summary_us"]["rns_gemm"]["avg"]
    reused["avg_crt_export_us"] = reused["timing_summary_us"]["crt_export"]["avg"]
    reused["avg_end_to_end_us"] = reused["timing_summary_us"]["end_to_end"]["avg"]
    reused["avg_per_modulus_gemm_estimate_us"] = reused["avg_rns_gemm_us"] / reused["prefix"]
    return reused


def as_direct_hip_bounded_uniform_small_reuse_a_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    repeats = reused["repeats"]
    kernel = "direct_hip_uniform_small_i8_ab_colpair_prefix9_reuse_a_grouped_rns_gemm_v1"
    epilogue = "uniform_small_i8_ab_resident_a_residue_then_crt_export"
    gemm_event = "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"
    reused["benchmark_execution_mode"] = "transient_uniform_small_i8_b_resident_i8_a_reuse"
    reused["backend_requested"] = "hip-direct"
    reused["backend_selected"] = "hip-direct"
    reused["selected_kernel"] = kernel
    reused["backend_metadata"]["source"] = "rns8_bench_uniform_small_i8_ab_reuse_a_path"
    reused["backend_metadata"]["selected_kernel"] = kernel
    reused["backend_metadata"]["accelerator_backend"] = False
    reused["backend_metadata"]["matrix_engine_backend"] = False
    reused["backend_metadata"]["accelerator_library"] = "HIP runtime"
    reused["backend_metadata"]["accelerator_version"] = "7.1"
    reused["backend_metadata"]["capability_status"] = "implemented_correctness_backend"
    reused["backend_metadata"]["epilogue_mode"] = epilogue
    reused["backend_metadata"]["workspace_mode"] = "transient_i8_b_resident_i8_a_rns_output"
    reused["backend_metadata"]["workspace_required_bytes"] = 0
    reused["backend_metadata"]["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
    apply_int32_accumulator_contract(reused)
    reused["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
        "input_profile=uniform-small;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_uniform_small_i8_b_resident_i8_a_reuse;"
        f"kernel={kernel};epilogue={epilogue}"
        ),
        reused,
    )
    reused["pack_mode"] = "prepacked_reuse_a"
    reused["reuse_packed_inputs"] = True
    reused["prepack_reuse_operands"] = ["A"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 320
    reused["avg_prepack_setup_us"] = 320.0
    reused["timing_metadata"]["benchmark_execution_mode"] = "transient_uniform_small_i8_b_resident_i8_a_reuse"
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse_a"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    reused["timing_metadata"]["gpu_event_phase_order"] = [
        "pack_h2d",
        "pack_kernel",
        "pack",
        gemm_event,
        "rns_gemm",
        "crt_export_status_memset",
        "crt_export_kernel",
        "crt_export_status_d2h",
        "crt_export_d2h",
        "crt_export",
    ]
    reused["timing_metadata"]["phase_notes"]["pack"] = (
        "per-repeat host timing for copying uniform-small B; A was copied once before warmups"
    )
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "A was copied once before warmups and reused for every measured repeat",
    }
    event_values = {
        "pack_h2d": [17.0, 18.0][:repeats],
        "pack_kernel": [0.0, 0.0][:repeats],
        "pack": [17.0, 18.0][:repeats],
        gemm_event: [145.0, 146.0][:repeats],
        "rns_gemm": [145.0, 146.0][:repeats],
        "crt_export_status_memset": [0.5, 0.5][:repeats],
        "crt_export_kernel": [30.0, 31.0][:repeats],
        "crt_export_status_d2h": [1.0, 1.0][:repeats],
        "crt_export_d2h": [10.0, 11.0][:repeats],
        "crt_export": [41.5, 43.5][:repeats],
    }
    reused["gpu_event_timings_us"] = event_values
    reused["gpu_event_timing_summary_us"] = {key: summary(value) for key, value in event_values.items()}
    reused["raw_timings_us"]["pack"] = [118, 123][:repeats]
    reused["raw_timings_us"]["rns_gemm"] = [255, 265][:repeats]
    reused["raw_timings_us"]["crt_export"] = [95, 100][:repeats]
    reused["raw_timings_us"]["end_to_end"] = [468, 488][:repeats]
    reused["timing_summary_us"]["pack"] = summary(reused["raw_timings_us"]["pack"])
    reused["timing_summary_us"]["rns_gemm"] = summary(reused["raw_timings_us"]["rns_gemm"])
    reused["timing_summary_us"]["crt_export"] = summary(reused["raw_timings_us"]["crt_export"])
    reused["timing_summary_us"]["end_to_end"] = summary(reused["raw_timings_us"]["end_to_end"])
    reused["avg_pack_us"] = reused["timing_summary_us"]["pack"]["avg"]
    reused["avg_rns_gemm_us"] = reused["timing_summary_us"]["rns_gemm"]["avg"]
    reused["avg_crt_export_us"] = reused["timing_summary_us"]["crt_export"]["avg"]
    reused["avg_end_to_end_us"] = reused["timing_summary_us"]["end_to_end"]["avg"]
    reused["avg_per_modulus_gemm_estimate_us"] = reused["avg_rns_gemm_us"] / reused["prefix"]
    return reused


def as_reused_pack_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    repeats = reused["repeats"]
    reused["reuse_packed_inputs"] = True
    reused["pack_mode"] = "prepacked_reuse"
    reused["prepack_reuse_operands"] = ["A", "B"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 123
    reused["avg_prepack_setup_us"] = 123.0
    reused["avg_pack_us"] = 0.0
    reused["raw_timings_us"]["pack"] = [0] * repeats
    reused["timing_summary_us"]["pack"] = zero_summary()
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A", "B"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["phase_notes"]["pack"] = (
        "zero-valued per-repeat phase; A and B were packed once into persistent matrices before warmups"
    )
    reused["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat rns_gemm plus crt_export host timing; excludes one-time prepack_setup_us"
    )
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "A and B were packed once before warmups and reused for every measured repeat",
    }
    for phase in ["pack_h2d", "pack_kernel", "finite_pack_h2d", "finite_pack_kernel", "pack"]:
        timings = reused.get("gpu_event_timings_us")
        summaries = reused.get("gpu_event_timing_summary_us")
        if isinstance(timings, dict) and phase in timings:
            timings[phase] = [0.0] * repeats
        if isinstance(summaries, dict) and phase in summaries:
            summaries[phase] = zero_summary()
    return reused


def as_hipblaslt_reused_ab_capture(capture: dict) -> dict:
    reused = as_reused_pack_capture(capture)
    phase = "hipblaslt_pack_transpose_centered"
    phase_order = reused["timing_metadata"].get("gpu_event_phase_order")
    if isinstance(phase_order, list) and phase in phase_order:
        phase_order.remove(phase)
    for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
        values = reused.get(field)
        if isinstance(values, dict):
            values.pop(phase, None)
    return reused


def as_reused_a_capture(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    reused["reuse_packed_inputs"] = True
    reused["pack_mode"] = "prepacked_reuse_a"
    reused["prepack_reuse_operands"] = ["A"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 77
    reused["avg_prepack_setup_us"] = 77.0
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse_a"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["A"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"]["phase_notes"]["pack"] = (
        "per-repeat host timing for packing B; A was packed once into a persistent matrix before warmups"
    )
    reused["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat pack of non-reused input plus rns_gemm plus crt_export host timing; excludes one-time "
        "prepack_setup_us"
    )
    reused["timing_metadata"]["phase_availability"]["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "prepacked A once before warmups and reused for every measured repeat",
    }
    return reused


def as_exact_wide_capture(capture: dict) -> dict:
    exact = copy.deepcopy(capture)
    exact["benchmark"] = "rns8_exact_wide_persistent_rns"
    exact["semantics"] = "exact_wide_signed"
    exact["bound_kind"] = "none"
    exact["bound_mode"] = "global"
    exact["bound"] = 0
    exact["prefix"] = 20
    exact["finite_modulus"] = None
    exact["tile_bounds_u64"] = None
    exact["epilogue_type"] = "exact_wide_signed_limb_export"
    exact["exact_wide_limb_count"] = 4
    exact["residue_chain_length"] = 1
    exact["residue_output_mode"] = "host_export"
    exact["input_distribution"] = "signed_uniform_-16_16"
    exact["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_correctness",
    ]
    exact["backend_metadata"]["epilogue_mode"] = "ck_fused_i32_to_centered_residue_rns_output"
    exact["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=ck;semantics=exact_wide_signed;m=64;n=128;k=64;prefix=20;tile_m=128;tile_n=128;"
        "groups=1;adaptive_prefix=0;adaptive_skip=0;kernel=ck_wmma_cshuffle_i8_i32_centered_epilogue_v1;"
        "epilogue=ck_fused_i32_to_centered_residue_rns_output"
        ),
        exact,
    )
    exact["schedule_metadata"]["min_selected_prefix"] = 20
    exact["schedule_metadata"]["max_selected_prefix"] = 20
    exact["schedule_metadata"]["prefix_group_count"] = 1
    exact["schedule_metadata"]["adaptive_execution_applied"] = False
    exact["avg_per_modulus_gemm_estimate_us"] = float(exact["avg_rns_gemm_us"]) / 20.0
    exact["timing_note"] = (
        "host wall-clock timings for persistent exact-wide RNS packing, RNS GEMM, and fixed-width little-endian "
        "limb export; GPU event timing names exact-wide export operation groups when backend hooks are available"
    )
    exact["timing_metadata"]["phase_notes"]["crt_export"] = (
        "per-repeat host timing for fixed-width exact-wide limb export"
    )

    renamed = {
        "crt_export_status_memset": "exact_wide_export_status_memset",
        "crt_export_kernel": "exact_wide_export_kernel",
        "crt_export_status_d2h": "exact_wide_export_status_d2h",
        "crt_export_d2h": "exact_wide_export_d2h",
    }
    phase_order = exact["timing_metadata"].get("gpu_event_phase_order")
    if isinstance(phase_order, list):
        exact["timing_metadata"]["gpu_event_phase_order"] = [renamed.get(item, item) for item in phase_order]
    for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
        values = exact.get(field)
        if isinstance(values, dict):
            for old, new in renamed.items():
                if old in values:
                    values[new] = values.pop(old)
    return exact

def as_residue_current_chain_capture(capture: dict) -> dict:
    chain = as_exact_wide_capture(capture)
    repeats = chain["repeats"]
    chain["m"] = 64
    chain["n"] = 64
    chain["k"] = 64
    chain["epilogue_type"] = "residue_current_rns_output"
    chain["residue_chain_length"] = 3
    chain["residue_output_mode"] = "residue_current_rns"
    chain["timing_note"] = (
        "host wall-clock timings for an exact-wide residue-current RNS GEMM chain; each measured repeat runs "
        "3 resident RNS GEMM calls before host export, raw_timings_us.crt_export is intentionally zero, and one "
        "final fixed-width limb export runs after measured repeats only to produce checksum_u64"
    )
    chain["timing_metadata"]["gpu_event_timing"] = False
    chain["timing_metadata"]["gpu_event_timing_reason"] = "not_supported_for_residue_current_chain_mode"
    chain["timing_metadata"]["gpu_event_timing_status"] = "not_requested_for_residue_current_chain_mode"
    chain["timing_metadata"]["gpu_event_timing_source"] = None
    chain["timing_metadata"]["gpu_event_timing_source_scope"] = None
    chain["timing_metadata"]["gpu_event_timing_caveat"] = None
    chain["timing_metadata"]["gpu_event_phase_order"] = None
    chain["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for 3 chained rns8_gemm_rns calls that keep the intermediate output resident "
        "in RNS form"
    )
    chain["timing_metadata"]["phase_notes"]["crt_export"] = (
        "zero-valued per-repeat phase; residue-current chain mode defers host limb export until one final checksum "
        "export after measured repeats"
    )
    chain["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat pack plus chained rns_gemm host timing; excludes the final checksum-only limb export"
    )
    chain["raw_timings_us"]["crt_export"] = [0 for _ in range(repeats)]
    chain["timing_summary_us"]["crt_export"] = {"avg": 0.0, "median": 0.0, "p95": 0.0}
    chain["avg_crt_export_us"] = 0.0
    chain["gpu_event_timings_us"] = None
    chain["gpu_event_timing_summary_us"] = None
    return chain


def as_bounded_residue_current_chain_capture(capture: dict) -> dict:
    chain = copy.deepcopy(capture)
    repeats = chain["repeats"]
    chain["m"] = 64
    chain["n"] = 64
    chain["k"] = 64
    chain["bound_mode"] = "global"
    chain["bound_kind"] = "global_max_abs"
    chain["bound"] = 1099511627776
    chain["tile_bounds_u64"] = None
    chain["schedule_metadata"]["tile_rows"] = 1
    chain["schedule_metadata"]["tile_cols"] = 1
    chain["schedule_metadata"]["tile_count"] = 1
    chain["schedule_metadata"]["min_required_prefix"] = 9
    chain["schedule_metadata"]["max_required_prefix"] = 9
    chain["schedule_metadata"]["min_selected_prefix"] = 9
    chain["schedule_metadata"]["max_selected_prefix"] = 9
    chain["schedule_metadata"]["prefix_group_count"] = 1
    chain["schedule_metadata"]["adaptive_prefix_active"] = False
    chain["schedule_metadata"]["adaptive_skip_active"] = False
    chain["schedule_metadata"]["adaptive_execution_applied"] = False
    chain["schedule_metadata"]["range_bit_length"] = 41
    chain["epilogue_type"] = "residue_current_rns_output"
    chain["residue_chain_length"] = 3
    chain["residue_output_mode"] = "residue_current_rns"
    chain["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=ck;semantics=bounded_i64;m=64;n=64;k=64;prefix=9;tile_m=128;tile_n=128;"
        "groups=1;adaptive_prefix=0;adaptive_skip=0;kernel=ck_wmma_cshuffle_i8_i32_centered_epilogue_v1;"
        "epilogue=ck_fused_i32_to_centered_residue_then_crt_export"
        ),
        chain,
    )
    chain["timing_note"] = (
        "host wall-clock timings for a residue-current RNS GEMM chain; each measured repeat runs 3 resident "
        "RNS GEMM calls before host export, raw_timings_us.crt_export is intentionally zero, and one final "
        "logical export runs after measured repeats only to produce checksum_u64"
    )
    chain["timing_metadata"]["gpu_event_timing"] = False
    chain["timing_metadata"]["gpu_event_timing_reason"] = "not_supported_for_residue_current_chain_mode"
    chain["timing_metadata"]["gpu_event_timing_status"] = "not_requested_for_residue_current_chain_mode"
    chain["timing_metadata"]["gpu_event_timing_source"] = None
    chain["timing_metadata"]["gpu_event_timing_source_scope"] = None
    chain["timing_metadata"]["gpu_event_timing_caveat"] = None
    chain["timing_metadata"]["gpu_event_phase_order"] = None
    chain["timing_metadata"]["phase_notes"]["rns_gemm"] = (
        "per-repeat host timing for 3 chained rns8_gemm_rns calls that keep the intermediate output resident "
        "in RNS form"
    )
    chain["timing_metadata"]["phase_notes"]["crt_export"] = (
        "zero-valued per-repeat phase; residue-current chain mode defers host logical export until one final "
        "checksum export after measured repeats"
    )
    chain["timing_metadata"]["phase_notes"]["end_to_end"] = (
        "per-repeat pack plus chained rns_gemm host timing; excludes the final checksum-only logical export"
    )
    chain["raw_timings_us"]["crt_export"] = [0 for _ in range(repeats)]
    chain["timing_summary_us"]["crt_export"] = {"avg": 0.0, "median": 0.0, "p95": 0.0}
    chain["avg_crt_export_us"] = 0.0
    chain["gpu_event_timings_us"] = None
    chain["gpu_event_timing_summary_us"] = None
    return chain


def as_wrap64_rocwmma_candidate_capture(capture: dict) -> dict:
    candidate = copy.deepcopy(capture)
    repeats = candidate["repeats"]
    candidate["backend_requested"] = "rocwmma-wrap64-candidate"
    candidate["backend_selected"] = "rocwmma"
    candidate["selected_kernel"] = "rocwmma_wrap64_byte_gemm36_candidate_v0"
    candidate["tile_m"] = 16
    candidate["tile_n"] = 16
    candidate["command_line"] = (
        "rns8-bench --backend rocwmma-wrap64-candidate --semantics wrap-u64 "
        "--m 4 --n 4 --k 8 --tile-m 16 --tile-n 16"
    )
    candidate["backend_metadata"].update(
        {
            "source": "rns8_bench_wrap64_rocwmma_candidate",
            "selected_kernel": "rocwmma_wrap64_byte_gemm36_candidate_v0",
            "accelerator_backend": True,
            "correctness_backend": False,
            "matrix_engine_backend": True,
            "compiled_kernel_available": True,
            "exact_differential_validated": True,
            "performance_validated": False,
            "accelerator_library": "rocWMMA",
            "accelerator_version": "repo-local release/rocm-rel-7.1",
            "capability_status": "internal_wrap64_matrix_engine_candidate",
            "epilogue_mode": "low64_wrap_export",
            "workspace_mode": "benchmark_owned_compact_byte_limb_device_buffers",
            "workspace_required_bytes": 640,
            "isa_evidence": "rocwmma_wrap64_byte_gemm36_wmma_isa_gate_no_int32_global_store_no_divide",
            "autotune_key": (
                "backend=rocwmma-wrap64-candidate;target_id=gfx1100;semantics=wrap_u64_mod_2_64;m=4;n=4;k=8;"
                "prefix=0;tile_m=16;tile_n=16;groups=0;adaptive_prefix=0;adaptive_skip=0;"
                "accumulator_type=int32_then_int64_diagonal;accumulator_signedness=unsigned_u8x_unsigned_u8;"
                "accumulator_modulus_policy=mod_2_64_wraparound_byte_limb;k_block_size=8;k_block_cap=32768;"
                "kernel=rocwmma_wrap64_byte_gemm36_candidate_v0;epilogue=low64_wrap_export"
            ),
            "accumulator_safety": {
                "input_domain": "compact_u8_byte_limb_pairs",
                "signedness": "unsigned_u8x_unsigned_u8",
                "accumulator_type": "int32_then_int64_diagonal",
                "modulus_policy": "mod_2_64_wraparound_byte_limb",
                "modulus": 0,
                "uses_int32_inner_product": True,
                "k_block_size": 8,
                "k_block_cap": 32768,
                "max_lhs_abs": 255,
                "max_rhs_abs": 255,
                "max_product": 255 * 255,
                "safe_for_k_block": True,
                "status": "safe_int32_byte_limb_gemm36_k_block",
            },
        }
    )
    candidate["k_block_size"] = 8
    candidate["schedule_metadata"].update(
        {
            "source": "rns8_bench_wrap64_rocwmma_candidate_static_schedule",
            "tile_m": 16,
            "tile_n": 16,
            "tile_rows": 1,
            "tile_cols": 1,
            "tile_count": 1,
        }
    )
    candidate["timing_note"] = (
        "host wall-clock timings for the internal rocWMMA wrap64 byte-GEMM36 candidate; GPU event timing uses "
        "direct-HIP byte-limb pack/export labels plus one candidate operation-group label and this path is not "
        "public or AUTO-selected"
    )
    candidate["timing_metadata"]["gpu_event_timing_reason"] = (
        "captured_by_internal_rocwmma_wrap64_candidate_hooks"
    )
    candidate["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups"
    )
    candidate["timing_metadata"]["gpu_event_timing_caveat"] = (
        "HIP event timings record direct-HIP byte-limb pack/export operation groups plus one internal rocWMMA "
        "wrap64 byte-GEMM36 candidate operation group; host wall-clock timings remain required for CPU scheduling "
        "overhead, API dispatch, allocations, and synchronous host-side overhead not represented on the HIP stream"
    )
    candidate["timing_metadata"]["phase_notes"].update(
        {
            "planning": "one-time benchmark-owned metadata initialization for the internal rocWMMA wrap64 candidate",
            "scheduling": "one-time fixed 16x16 WMMA candidate schedule derivation from the matrix shape",
            "matrix_alloc": "one-time benchmark-owned compact byte-limb HIP device buffer allocation host timing",
            "pack": "per-repeat host timing for direct-HIP packing of A and B into compact byte-limb device buffers",
            "rns_gemm": "per-repeat host timing for the internal rocWMMA wrap64 byte-GEMM36 candidate",
            "crt_export": "per-repeat host timing for direct-HIP low-64-bit byte-limb export",
        }
    )
    candidate["timing_metadata"]["phase_availability"]["scheduling"] = {
        "timed": True,
        "timing_key": "scheduling",
        "scope": "benchmark_static_wrap64_rocwmma_candidate_schedule",
        "reason": "measured with host steady_clock around fixed 16x16 candidate schedule metadata initialization",
    }
    renamed = {
        "wrap64_byte_gemm36_tiled_2d_kernel": "wrap64_rocwmma_candidate_gemm36_kernel_group",
    }
    phase_order = candidate["timing_metadata"].get("gpu_event_phase_order")
    if isinstance(phase_order, list):
        candidate["timing_metadata"]["gpu_event_phase_order"] = [renamed.get(item, item) for item in phase_order]
    for field in ["gpu_event_timings_us", "gpu_event_timing_summary_us"]:
        values = candidate.get(field)
        if isinstance(values, dict):
            for old, new in renamed.items():
                if old in values:
                    values[new] = values.pop(old)
    assert len(candidate["gpu_event_timings_us"]["wrap64_rocwmma_candidate_gemm36_kernel_group"]) == repeats
    return candidate


def main() -> int:
    v4_wrap64_hip = expect_valid("v4_wrap64_hip.json")
    v4_adaptive_u64 = expect_valid("v4_bounded_u64_adaptive_hip.json")
    v4_adaptive_i64 = expect_valid("v4_bounded_i64_adaptive_hip.json")
    v4_hipblaslt_i64 = expect_valid("v4_bounded_i64_hipblaslt.json")
    v4_ck_i64 = expect_valid("v4_bounded_i64_ck.json")
    v4_ck_adaptive_u64 = expect_valid("v4_bounded_u64_adaptive_ck.json")
    v4_rocwmma_i64 = expect_valid("v4_bounded_i64_rocwmma.json")
    v4_rocwmma_adaptive_u64 = expect_valid("v4_bounded_u64_adaptive_rocwmma.json")
    v4_vector_i64 = expect_valid("v4_bounded_i64_vector_alu.json")
    v4_vector_u64 = expect_valid("v4_bounded_u64_vector_alu.json")
    v4_finite_ring_ck = expect_valid("v4_finite_ring_u8_ck.json")
    v4_finite_field_rocwmma = expect_valid("v4_finite_field_u8_rocwmma.json")
    bounded = v4_adaptive_i64

    missing_accumulator_safety = copy.deepcopy(v4_ck_i64)
    del missing_accumulator_safety["backend_metadata"]["accumulator_safety"]
    expect_invalid(missing_accumulator_safety, "backend_metadata.accumulator_safety must be an object")

    stale_ck_accumulator_cap = copy.deepcopy(v4_ck_i64)
    stale_ck_accumulator_cap["backend_metadata"]["accumulator_safety"]["k_block_cap"] = 65536
    stale_ck_accumulator_cap["backend_metadata"]["autotune_key"] = stale_ck_accumulator_cap["backend_metadata"][
        "autotune_key"
    ].replace("k_block_cap=32768", "k_block_cap=65536")
    expect_invalid(stale_ck_accumulator_cap, "backend_metadata.accumulator_safety.k_block_cap must be 32768")

    missing_accumulator_key_field = copy.deepcopy(v4_ck_i64)
    missing_accumulator_key_field["backend_metadata"]["autotune_key"] = missing_accumulator_key_field[
        "backend_metadata"
    ]["autotune_key"].replace("accumulator_type=int32;", "")
    expect_invalid(missing_accumulator_key_field, "backend_metadata.autotune_key must include accumulator_type=int32")

    unsafe_accumulator_flag = copy.deepcopy(v4_ck_i64)
    unsafe_accumulator_flag["backend_metadata"]["accumulator_safety"]["safe_for_k_block"] = False
    expect_invalid(unsafe_accumulator_flag, "int32 accumulator captures must set safe_for_k_block=true")

    prefixed_adaptive = add_prefix_policy_fields(copy.deepcopy(v4_adaptive_i64), "per_tile_minimum")
    validate_capture(prefixed_adaptive)

    incomplete_prefix_policy = copy.deepcopy(prefixed_adaptive)
    del incomplete_prefix_policy["residue_planes_skipped"]
    expect_invalid(incomplete_prefix_policy, "prefix policy metadata fields must be complete")

    stale_selected_prefix = copy.deepcopy(prefixed_adaptive)
    stale_selected_prefix["selected_prefix"] = stale_selected_prefix["prefix"]
    expect_invalid(stale_selected_prefix, "selected_prefix must match")

    bad_prefix_skip_fraction = copy.deepcopy(prefixed_adaptive)
    bad_prefix_skip_fraction["residue_plane_skip_fraction"] = 0.0
    expect_invalid(bad_prefix_skip_fraction, "residue_plane_skip_fraction must match")

    bad_prefix_policy_scope = copy.deepcopy(prefixed_adaptive)
    bad_prefix_policy_scope["contract_prefix_policy"] = "minimum_proven"
    expect_invalid(bad_prefix_policy_scope, "contract_prefix_policy=minimum_proven requires bound_mode=global")

    scanned_bound = add_global_bound_scan_fields(copy.deepcopy(v4_hipblaslt_i64))
    validate_capture(scanned_bound)

    stale_scanned_bound = copy.deepcopy(scanned_bound)
    stale_scanned_bound["bound_discovery"]["discovered_global_bound"] += 1
    expect_invalid(stale_scanned_bound, "bound_discovery.discovered_global_bound must equal")

    incomplete_scanned_timing = copy.deepcopy(scanned_bound)
    del incomplete_scanned_timing["timing_metadata"]["phase_availability"]["global_bound_scan"]
    expect_invalid(incomplete_scanned_timing, "phase_availability.global_bound_scan must be an object")

    per_tile_input_scan = add_per_tile_input_scan_fields(copy.deepcopy(v4_adaptive_i64))
    validate_capture(per_tile_input_scan)

    per_tile_scan_without_tile_bounds = copy.deepcopy(per_tile_input_scan)
    per_tile_scan_without_tile_bounds["tile_bounds_u64"] = None
    expect_invalid(per_tile_scan_without_tile_bounds, "input_exact_tile_bounds captures must include tile_bounds_u64")

    stale_per_tile_scan_global_field = copy.deepcopy(per_tile_input_scan)
    stale_per_tile_scan_global_field["bound_discovery"]["discovered_global_bound"] = 1
    expect_invalid(
        stale_per_tile_scan_global_field,
        "input_exact_tile_bounds captures must use bound_discovery.discovered_global_bound=null",
    )

    stale_per_tile_scan_global_availability = copy.deepcopy(per_tile_input_scan)
    stale_per_tile_scan_global_availability["timing_metadata"]["phase_availability"]["global_bound_scan"] = {
        "timed": True,
        "timing_key": "global_bound_scan",
        "scope": "input_row_column_abs_summary",
        "reason": "stale global scan metadata from a non-per-tile input-scan capture",
    }
    expect_invalid(stale_per_tile_scan_global_availability, "phase_availability.global_bound_scan.timed must be false")
    wrap64 = v4_wrap64_hip
    v4_wrap64_rocwmma_candidate = as_wrap64_rocwmma_candidate_capture(v4_wrap64_hip)
    validate_capture(v4_wrap64_rocwmma_candidate)

    stale_rocwmma_backend_spelling = copy.deepcopy(v4_rocwmma_i64)
    stale_rocwmma_backend_spelling["backend_requested"] = "wmma"
    stale_rocwmma_backend_spelling["backend_selected"] = "wmma"
    stale_rocwmma_backend_spelling["backend_metadata"]["autotune_key"] = stale_rocwmma_backend_spelling[
        "backend_metadata"
    ]["autotune_key"].replace("backend=rocwmma;", "backend=wmma;")
    expect_invalid(stale_rocwmma_backend_spelling, "backend_selected must be one of")

    stale_candidate_request_spelling = copy.deepcopy(v4_wrap64_rocwmma_candidate)
    stale_candidate_request_spelling["backend_requested"] = "wrap64-wmma-candidate"
    stale_candidate_request_spelling["backend_selected"] = "wmma"
    stale_candidate_request_spelling["backend_metadata"]["autotune_key"] = stale_candidate_request_spelling[
        "backend_metadata"
    ]["autotune_key"].replace("backend=rocwmma-wrap64-candidate;", "backend=wrap64-wmma-candidate;")
    expect_invalid(stale_candidate_request_spelling, "backend_selected must be one of")

    v4_cpu_adaptive_i64 = copy.deepcopy(v4_adaptive_i64)
    v4_cpu_adaptive_i64["backend_requested"] = "cpu-reference"
    v4_cpu_adaptive_i64["backend_selected"] = "cpu-reference"
    v4_cpu_adaptive_i64["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
    v4_cpu_adaptive_i64["backend_metadata"]["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
    v4_cpu_adaptive_i64["backend_metadata"]["accelerator_library"] = None
    v4_cpu_adaptive_i64["backend_metadata"]["workspace_mode"] = "host_reference_workspace"
    v4_cpu_adaptive_i64["backend_metadata"]["workspace_required_bytes"] = 0
    v4_cpu_adaptive_i64["backend_metadata"]["isa_evidence"] = "not_applicable_cpu"
    v4_cpu_adaptive_i64["backend_metadata"]["autotune_key"] = (
        "backend=cpu-reference;target_id=cpu;semantics=bounded_i64;m=65;n=65;k=64;prefix=9;tile_m=64;tile_n=64;"
        "groups=4;adaptive_prefix=1;adaptive_skip=1;accumulator_type=int32;"
        "accumulator_signedness=signed_i8x_signed_i8;"
        "accumulator_modulus_policy=selected_rns_modulus_ladder;k_block_size=64;k_block_cap=65536;"
        "kernel=cpu_reference_scalar_rns_gemm_v1;"
        "epilogue=fused_centered_residue_then_crt_export"
    )
    v4_cpu_adaptive_i64["comparison_baseline"]["required_before_speedup_claim"] = [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_vector_alu_int64",
        "same_contract_direct_hip_correctness",
    ]
    v4_cpu_adaptive_i64["device"] = {
        "device_id": -1,
        "name": "CPU reference",
        "gcn_arch": "none",
        "hip_available": 0,
        "hip_runtime_version": 0,
        "hip_driver_version": 0,
        "global_mem_bytes": 0,
    }
    v4_cpu_adaptive_i64["timing_note"] = (
        "host wall-clock timings for the CPU adaptive per-tile bounded reference path; no GPU event timing is "
        "requested for this backend"
    )
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing"] = False
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_reason"] = "not_supported_for_selected_backend"
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_status"] = "not_requested_for_selected_backend"
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_source"] = None
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_source_scope"] = None
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_timing_caveat"] = None
    v4_cpu_adaptive_i64["timing_metadata"]["gpu_event_phase_order"] = None
    v4_cpu_adaptive_i64["gpu_event_timings_us"] = None
    v4_cpu_adaptive_i64["gpu_event_timing_summary_us"] = None
    validate_capture(v4_cpu_adaptive_i64)

    reused_ck_i64 = as_reused_pack_capture(v4_ck_i64)
    validate_capture(reused_ck_i64)
    reused_a_ck_i64 = as_reused_a_capture(v4_ck_i64)
    validate_capture(reused_a_ck_i64)
    reused_hipblaslt_i64 = as_hipblaslt_reused_ab_capture(v4_hipblaslt_i64)
    validate_capture(reused_hipblaslt_i64)
    direct_hip_oneshot_i64 = as_direct_hip_oneshot_capture(v4_ck_i64)
    validate_capture(direct_hip_oneshot_i64)
    direct_hip_finite_oneshot = as_direct_hip_finite_oneshot_capture(v4_finite_ring_ck)
    validate_capture(direct_hip_finite_oneshot)
    direct_hip_finite_native_a_reuse_b = as_direct_hip_finite_native_a_reuse_b_capture(v4_finite_ring_ck)
    validate_capture(direct_hip_finite_native_a_reuse_b)
    direct_hip_bounded_native_a_reuse_b = as_direct_hip_bounded_native_a_reuse_b_capture(v4_ck_i64)
    validate_capture(direct_hip_bounded_native_a_reuse_b)
    direct_hip_bounded_uniform_small_reuse_a = as_direct_hip_bounded_uniform_small_reuse_a_capture(v4_ck_i64)
    validate_capture(direct_hip_bounded_uniform_small_reuse_a)
    adaptive_direct_hip_bounded_native_a = copy.deepcopy(direct_hip_bounded_native_a_reuse_b)
    centered_kernel = "direct_hip_native_a_i64_prefix9_reuse_b_grouped_rns_gemm_v1"
    centered_epilogue = "native_a_centered_resident_b_residue_then_crt_export"
    adaptive_direct_hip_bounded_native_a["input_distribution"] = "signed_adaptive_bands_-16_16"
    adaptive_direct_hip_bounded_native_a["benchmark_execution_mode"] = "transient_native_a_resident_b_reuse"
    adaptive_direct_hip_bounded_native_a["selected_kernel"] = centered_kernel
    adaptive_direct_hip_bounded_native_a["backend_metadata"]["source"] = "rns8_bench_native_a_reuse_b_path"
    adaptive_direct_hip_bounded_native_a["backend_metadata"]["selected_kernel"] = centered_kernel
    adaptive_direct_hip_bounded_native_a["backend_metadata"]["epilogue_mode"] = centered_epilogue
    adaptive_direct_hip_bounded_native_a["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
        "input_profile=adaptive-bands;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_native_a_resident_b_reuse;"
        f"kernel={centered_kernel};epilogue={centered_epilogue}"
        ),
        adaptive_direct_hip_bounded_native_a,
    )
    adaptive_direct_hip_bounded_native_a["timing_metadata"][
        "benchmark_execution_mode"
    ] = "transient_native_a_resident_b_reuse"
    adaptive_direct_hip_bounded_native_a["timing_metadata"]["gpu_event_phase_order"] = [
        "bounded_native_a_reuse_b_gemm_kernel_group"
        if phase == "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
        else phase
        for phase in adaptive_direct_hip_bounded_native_a["timing_metadata"]["gpu_event_phase_order"]
    ]
    adaptive_direct_hip_bounded_native_a["gpu_event_timings_us"]["bounded_native_a_reuse_b_gemm_kernel_group"] = (
        adaptive_direct_hip_bounded_native_a["gpu_event_timings_us"].pop(
            "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
        )
    )
    adaptive_direct_hip_bounded_native_a["gpu_event_timing_summary_us"][
        "bounded_native_a_reuse_b_gemm_kernel_group"
    ] = adaptive_direct_hip_bounded_native_a["gpu_event_timing_summary_us"].pop(
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    )
    validate_capture(adaptive_direct_hip_bounded_native_a)

    large_u64_colpair_native_a = copy.deepcopy(adaptive_direct_hip_bounded_native_a)
    large_u64_colpair_kernel = "direct_hip_native_a_u64_colpair_prefix9_reuse_b_grouped_rns_gemm_v2"
    large_u64_colpair_native_a["semantics"] = "bounded_u64"
    large_u64_colpair_native_a["bound_kind"] = "global_max_unsigned"
    large_u64_colpair_native_a["input_distribution"] = "unsigned_adaptive_bands_0_16"
    large_u64_colpair_native_a["m"] = 512
    large_u64_colpair_native_a["n"] = 512
    large_u64_colpair_native_a["k"] = 512
    large_u64_colpair_native_a["selected_kernel"] = large_u64_colpair_kernel
    large_u64_colpair_native_a["backend_metadata"]["selected_kernel"] = large_u64_colpair_kernel
    apply_int32_accumulator_contract(large_u64_colpair_native_a)
    large_u64_colpair_native_a["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_u64;m=512;n=512;k=512;bound=16384;"
        "input_profile=adaptive-bands;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_native_a_resident_b_reuse;"
        f"kernel={large_u64_colpair_kernel};epilogue={centered_epilogue}"
        ),
        large_u64_colpair_native_a,
    )
    large_u64_colpair_native_a["timing_metadata"]["gpu_event_phase_order"] = [
        "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
        if phase == "bounded_native_a_reuse_b_gemm_kernel_group"
        else phase
        for phase in large_u64_colpair_native_a["timing_metadata"]["gpu_event_phase_order"]
    ]
    large_u64_colpair_native_a["gpu_event_timings_us"][
        "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
    ] = large_u64_colpair_native_a["gpu_event_timings_us"].pop("bounded_native_a_reuse_b_gemm_kernel_group")
    large_u64_colpair_native_a["gpu_event_timing_summary_us"][
        "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
    ] = large_u64_colpair_native_a["gpu_event_timing_summary_us"].pop("bounded_native_a_reuse_b_gemm_kernel_group")
    validate_capture(large_u64_colpair_native_a)

    stale_large_u64_colpair_kernel = copy.deepcopy(large_u64_colpair_native_a)
    stale_large_u64_kernel = "direct_hip_native_a_u64_prefix9_reuse_b_grouped_rns_gemm_v1"
    stale_large_u64_colpair_kernel["selected_kernel"] = stale_large_u64_kernel
    stale_large_u64_colpair_kernel["backend_metadata"]["selected_kernel"] = stale_large_u64_kernel
    stale_large_u64_colpair_kernel["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_u64;m=512;n=512;k=512;bound=16384;"
        "input_profile=adaptive-bands;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_native_a_resident_b_reuse;"
        f"kernel={stale_large_u64_kernel};epilogue={centered_epilogue}"
        ),
        stale_large_u64_colpair_kernel,
    )
    expect_invalid(
        stale_large_u64_colpair_kernel,
        "direct-HIP bounded native-A reuse-B captures must use selected_kernel",
    )

    stale_large_u64_colpair_phase = copy.deepcopy(large_u64_colpair_native_a)
    stale_large_u64_colpair_phase["timing_metadata"]["gpu_event_phase_order"] = [
        "bounded_native_a_reuse_b_gemm_kernel_group"
        if phase == "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
        else phase
        for phase in stale_large_u64_colpair_phase["timing_metadata"]["gpu_event_phase_order"]
    ]
    stale_large_u64_colpair_phase["gpu_event_timings_us"]["bounded_native_a_reuse_b_gemm_kernel_group"] = (
        stale_large_u64_colpair_phase["gpu_event_timings_us"].pop(
            "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
        )
    )
    stale_large_u64_colpair_phase["gpu_event_timing_summary_us"]["bounded_native_a_reuse_b_gemm_kernel_group"] = (
        stale_large_u64_colpair_phase["gpu_event_timing_summary_us"].pop(
            "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
        )
    )
    expect_invalid(
        stale_large_u64_colpair_phase,
        "direct-HIP bounded native-A reuse-B GPU event phase set is incomplete",
    )

    bad_bounded_native_a_phase = copy.deepcopy(direct_hip_bounded_native_a_reuse_b)
    bad_bounded_native_a_phase["timing_metadata"]["gpu_event_phase_order"] = [
        "rns_gemm_kernel_group" if phase == "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group" else phase
        for phase in bad_bounded_native_a_phase["timing_metadata"]["gpu_event_phase_order"]
    ]
    bad_bounded_native_a_phase["gpu_event_timings_us"]["rns_gemm_kernel_group"] = (
        bad_bounded_native_a_phase["gpu_event_timings_us"].pop(
            "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
        )
    )
    bad_bounded_native_a_phase["gpu_event_timing_summary_us"]["rns_gemm_kernel_group"] = (
        bad_bounded_native_a_phase["gpu_event_timing_summary_us"].pop(
            "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
        )
    )
    expect_invalid(
        bad_bounded_native_a_phase,
        "direct-HIP bounded native-A reuse-B GPU event phase set is incomplete",
    )
    bad_bounded_uniform_small_reuse_a_phase = copy.deepcopy(direct_hip_bounded_uniform_small_reuse_a)
    bad_bounded_uniform_small_reuse_a_phase["timing_metadata"]["gpu_event_phase_order"] = [
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
        if phase == "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"
        else phase
        for phase in bad_bounded_uniform_small_reuse_a_phase["timing_metadata"]["gpu_event_phase_order"]
    ]
    bad_bounded_uniform_small_reuse_a_phase["gpu_event_timings_us"][
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    ] = bad_bounded_uniform_small_reuse_a_phase["gpu_event_timings_us"].pop(
        "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"
    )
    bad_bounded_uniform_small_reuse_a_phase["gpu_event_timing_summary_us"][
        "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
    ] = bad_bounded_uniform_small_reuse_a_phase["gpu_event_timing_summary_us"].pop(
        "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group"
    )
    expect_invalid(
        bad_bounded_uniform_small_reuse_a_phase,
        "direct-HIP bounded uniform-small reuse-A GPU event phase set is incomplete",
    )
    stale_generic_bounded_native_a = copy.deepcopy(direct_hip_bounded_native_a_reuse_b)
    stale_kernel = "direct_hip_native_a_i64_prefix9_reuse_b_grouped_rns_gemm_v1"
    stale_epilogue = "native_a_centered_resident_b_residue_then_crt_export"
    stale_generic_bounded_native_a["selected_kernel"] = stale_kernel
    stale_generic_bounded_native_a["backend_metadata"]["selected_kernel"] = stale_kernel
    stale_generic_bounded_native_a["backend_metadata"]["epilogue_mode"] = stale_epilogue
    stale_generic_bounded_native_a["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_native_a_resident_b_reuse;"
        f"kernel={stale_kernel};epilogue={stale_epilogue}"
        ),
        stale_generic_bounded_native_a,
    )
    expect_invalid(
        stale_generic_bounded_native_a,
        "direct-HIP bounded native-A reuse-B captures must use selected_kernel",
    )
    stale_v1_uniform_small_bounded_native_a = copy.deepcopy(direct_hip_bounded_native_a_reuse_b)
    stale_v1_kernel = "direct_hip_uniform_small_i8_ab_prefix9_reuse_b_grouped_rns_gemm_v1"
    stale_v1_uniform_small_bounded_native_a["selected_kernel"] = stale_v1_kernel
    stale_v1_uniform_small_bounded_native_a["backend_metadata"]["selected_kernel"] = stale_v1_kernel
    stale_v1_uniform_small_bounded_native_a["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        (
        "backend=hip-direct;semantics=bounded_i64;m=64;n=128;k=64;bound=16384;"
        "input_profile=uniform-small;"
        "prefix=9;tile_m=128;tile_n=128;groups=1;adaptive_prefix=0;adaptive_skip=0;"
        "execution=transient_uniform_small_i8_a_resident_i8_b_reuse;"
        f"kernel={stale_v1_kernel};epilogue=uniform_small_i8_ab_resident_b_residue_then_crt_export"
        ),
        stale_v1_uniform_small_bounded_native_a,
    )
    expect_invalid(
        stale_v1_uniform_small_bounded_native_a,
        "direct-HIP bounded native-A reuse-B captures must use selected_kernel",
    )

    exact_wide_ck = as_exact_wide_capture(v4_ck_i64)
    validate_capture(exact_wide_ck)
    exact_wide_no_status = copy.deepcopy(exact_wide_ck)
    exact_wide_no_status["exact_wide_export_status_check"] = "elided_full_width_device_reconstruction"
    exact_wide_no_status["gpu_event_timings_us"]["exact_wide_export_status_memset"] = [
        0.0 for _ in range(exact_wide_no_status["repeats"])
    ]
    exact_wide_no_status["gpu_event_timings_us"]["exact_wide_export_status_d2h"] = [
        0.0 for _ in range(exact_wide_no_status["repeats"])
    ]
    exact_wide_no_status["gpu_event_timing_summary_us"]["exact_wide_export_status_memset"] = zero_summary()
    exact_wide_no_status["gpu_event_timing_summary_us"]["exact_wide_export_status_d2h"] = zero_summary()
    validate_capture(exact_wide_no_status)
    exact_chain_ck = as_residue_current_chain_capture(v4_ck_i64)
    validate_capture(exact_chain_ck)
    bounded_chain_ck = as_bounded_residue_current_chain_capture(v4_ck_i64)
    validate_capture(bounded_chain_ck)

    bad_exact_bound = copy.deepcopy(exact_wide_ck)
    bad_exact_bound["bound_kind"] = "global_max_abs"
    expect_invalid(bad_exact_bound, "exact-wide captures must use bound_kind=none and bound=0")

    bad_exact_epilogue = copy.deepcopy(exact_wide_ck)
    bad_exact_epilogue["epilogue_type"] = "crt_export"
    expect_invalid(bad_exact_epilogue, "exact_wide_signed_limb_export")

    bad_exact_limb_count = copy.deepcopy(exact_wide_ck)
    bad_exact_limb_count["exact_wide_limb_count"] = 33
    expect_invalid(bad_exact_limb_count, "exact_wide_limb_count in [1, 32]")

    missing_exact_limb_count = copy.deepcopy(exact_wide_ck)
    del missing_exact_limb_count["exact_wide_limb_count"]
    expect_invalid(missing_exact_limb_count, "exact_wide_limb_count in [1, 32]")

    bad_exact_backend_epilogue = copy.deepcopy(exact_wide_ck)
    bad_exact_backend_epilogue["backend_metadata"]["epilogue_mode"] = "ck_fused_i32_to_centered_residue_then_crt_export"
    expect_invalid(bad_exact_backend_epilogue, "ck_fused_i32_to_centered_residue_rns_output")

    bad_exact_status_check = copy.deepcopy(exact_wide_no_status)
    bad_exact_status_check["exact_wide_export_status_check"] = "required_for_range_check"
    expect_invalid(bad_exact_status_check, "exact_wide_export_status_check")

    bad_exact_status_elision_events = copy.deepcopy(exact_wide_no_status)
    bad_exact_status_elision_events["gpu_event_timings_us"]["exact_wide_export_status_d2h"][0] = 1.0
    expect_invalid(bad_exact_status_elision_events, "status-elided captures")

    bad_chain_export = copy.deepcopy(exact_chain_ck)
    bad_chain_export["raw_timings_us"]["crt_export"][0] = 1
    bad_chain_export["timing_summary_us"]["crt_export"]["avg"] = 1
    bad_chain_export["avg_crt_export_us"] = 1
    expect_invalid(bad_chain_export, "residue-current chain captures must report raw_timings_us.crt_export")

    bad_chain_gpu_events = copy.deepcopy(exact_chain_ck)
    bad_chain_gpu_events["timing_metadata"]["gpu_event_timing"] = True
    bad_chain_gpu_events["timing_metadata"]["gpu_event_timing_source"] = "hipEventElapsedTime"
    bad_chain_gpu_events["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
    )
    bad_chain_gpu_events["timing_metadata"]["gpu_event_phase_order"] = ["pack", "rns_gemm"]
    bad_chain_repeats = bad_chain_gpu_events["repeats"]
    bad_chain_gpu_events["gpu_event_timings_us"] = {
        "pack": [1.0 for _ in range(bad_chain_repeats)],
        "rns_gemm": [2.0 for _ in range(bad_chain_repeats)],
    }
    bad_chain_gpu_events["gpu_event_timing_summary_us"] = {
        "pack": {"avg": 1.0, "median": 1.0, "p95": 1.0},
        "rns_gemm": {"avg": 2.0, "median": 2.0, "p95": 2.0},
    }
    expect_invalid(bad_chain_gpu_events, "residue-current chain captures must not claim GPU event timings")

    bad_chain_mode = copy.deepcopy(exact_chain_ck)
    bad_chain_mode["residue_output_mode"] = "host_export"
    expect_invalid(bad_chain_mode, "residue_output_mode=residue_current_rns")

    bad_chain_shape = copy.deepcopy(exact_chain_ck)
    bad_chain_shape["n"] = 128
    expect_invalid(bad_chain_shape, "square m=n=k shapes")

    bad_bounded_chain_vector = copy.deepcopy(bounded_chain_ck)
    bad_bounded_chain_vector["backend_selected"] = "hip-vector-alu-int64"
    bad_bounded_chain_vector["selected_kernel"] = "hip_vector_alu_i64_exact_192b_v1"
    bad_bounded_chain_vector["backend_metadata"]["selected_kernel"] = "hip_vector_alu_i64_exact_192b_v1"
    expect_invalid(bad_bounded_chain_vector, "must not select hip-vector-alu-int64")

    bad_bounded_chain_bound_mode = copy.deepcopy(bounded_chain_ck)
    bad_bounded_chain_bound_mode["bound_mode"] = "per_tile"
    expect_invalid(bad_bounded_chain_bound_mode, "bounded residue-current chains must use bound_mode=global")

    bad_reused_pack = copy.deepcopy(reused_ck_i64)
    bad_reused_pack["raw_timings_us"]["pack"][0] = 1
    expect_invalid(bad_reused_pack, "zero-valued repeats")

    bad_reused_prepack = copy.deepcopy(reused_ck_i64)
    bad_reused_prepack["prepack_setup_us"] = None
    expect_invalid(bad_reused_prepack, "prepack_setup_us")

    bad_reused_mode = copy.deepcopy(reused_ck_i64)
    bad_reused_mode["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    expect_invalid(bad_reused_mode, "timing_metadata.pack_mode")

    bad_reused_operands = copy.deepcopy(reused_a_ck_i64)
    bad_reused_operands["prepack_reuse_operands"] = ["B"]
    expect_invalid(bad_reused_operands, "prepack_reuse_operands")

    bad_reused_metadata_operands = copy.deepcopy(reused_a_ck_i64)
    bad_reused_metadata_operands["timing_metadata"]["prepack_reuse_operands"] = ["A", "B"]
    expect_invalid(bad_reused_metadata_operands, "timing_metadata.prepack_reuse_operands")

    bad_reused_strategy = copy.deepcopy(reused_ck_i64)
    bad_reused_strategy["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
    bad_reused_strategy["timing_metadata"]["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
    expect_invalid(bad_reused_strategy, "pack_mode=prepacked_reuse_b")

    bad_reused_strategy_backend = copy.deepcopy(reused_ck_i64)
    bad_reused_strategy_backend["pack_mode"] = "prepacked_reuse_b"
    bad_reused_strategy_backend["prepack_reuse_operands"] = ["B"]
    bad_reused_strategy_backend["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
    bad_reused_strategy_backend["timing_metadata"]["pack_mode"] = "prepacked_reuse_b"
    bad_reused_strategy_backend["timing_metadata"]["prepack_reuse_operands"] = ["B"]
    bad_reused_strategy_backend["timing_metadata"]["prepack_reuse_strategy"] = "rocwmma_reusable_b_cache"
    expect_invalid(bad_reused_strategy_backend, "backend_selected=rocwmma")

    bad_reused_metadata_strategy = copy.deepcopy(reused_ck_i64)
    bad_reused_metadata_strategy["timing_metadata"]["prepack_reuse_strategy"] = "none"
    expect_invalid(bad_reused_metadata_strategy, "timing_metadata.prepack_reuse_strategy")

    bad_hipblaslt_full_reuse_stale_event = copy.deepcopy(reused_hipblaslt_i64)
    bad_hipblaslt_full_reuse_stale_event["gpu_event_timings_us"]["hipblaslt_pack_transpose_centered"] = [
        0.0
    ] * bad_hipblaslt_full_reuse_stale_event["repeats"]
    bad_hipblaslt_full_reuse_stale_event["gpu_event_timing_summary_us"][
        "hipblaslt_pack_transpose_centered"
    ] = zero_summary()
    expect_invalid(
        bad_hipblaslt_full_reuse_stale_event,
        "undeclared phase hipblaslt_pack_transpose_centered",
    )

    bad_oneshot_pack_timing = copy.deepcopy(direct_hip_oneshot_i64)
    bad_oneshot_pack_timing["raw_timings_us"]["pack"][0] = 1
    expect_invalid(bad_oneshot_pack_timing, "public one-shot captures must report raw_timings_us.pack")

    bad_oneshot_scope = copy.deepcopy(direct_hip_oneshot_i64)
    bad_oneshot_scope["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_default_stream_backend_operation_groups"
    )
    expect_invalid(bad_oneshot_scope, "direct_hip_oneshot_default_stream_operation_groups")

    bad_oneshot_event_phase = copy.deepcopy(direct_hip_oneshot_i64)
    bad_oneshot_event_phase["timing_metadata"]["gpu_event_phase_order"].remove("oneshot_native_input_h2d")
    del bad_oneshot_event_phase["gpu_event_timings_us"]["oneshot_native_input_h2d"]
    del bad_oneshot_event_phase["gpu_event_timing_summary_us"]["oneshot_native_input_h2d"]
    expect_invalid(bad_oneshot_event_phase, "direct-HIP one-shot GPU event phase set is incomplete")

    small_u64_oneshot = copy.deepcopy(direct_hip_oneshot_i64)
    small_u64_oneshot["semantics"] = "bounded_u64"
    small_u64_oneshot["bound_kind"] = "global_max_unsigned"
    small_u64_oneshot["backend_metadata"]["autotune_key"] = small_u64_oneshot[
        "backend_metadata"
    ]["autotune_key"].replace("semantics=bounded_i64", "semantics=bounded_u64")
    validate_capture(small_u64_oneshot)

    large_u64_oneshot = copy.deepcopy(small_u64_oneshot)
    large_u64_oneshot["m"] = 512
    large_u64_oneshot["n"] = 512
    large_u64_oneshot["k"] = 512
    u64_oneshot_kernel = "direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2"
    large_u64_oneshot["selected_kernel"] = u64_oneshot_kernel
    large_u64_oneshot["backend_metadata"]["selected_kernel"] = u64_oneshot_kernel
    apply_int32_accumulator_contract(large_u64_oneshot)
    large_u64_oneshot["backend_metadata"]["autotune_key"] = with_accumulator_key_fields(
        large_u64_oneshot[
        "backend_metadata"
    ]["autotune_key"].replace("m=64;n=128;k=64", "m=512;n=512;k=512").replace(
        "direct_hip_prefix9_native_input_grouped_rns_gemm_v1",
        u64_oneshot_kernel,
        ),
        large_u64_oneshot,
    )
    validate_capture(large_u64_oneshot)

    bad_oneshot_stale_kernel = copy.deepcopy(large_u64_oneshot)
    old_oneshot_kernel = "direct_hip_prefix9_native_input_grouped_rns_gemm_v1"
    bad_oneshot_stale_kernel["selected_kernel"] = old_oneshot_kernel
    bad_oneshot_stale_kernel["backend_metadata"]["selected_kernel"] = old_oneshot_kernel
    bad_oneshot_stale_kernel["backend_metadata"]["autotune_key"] = bad_oneshot_stale_kernel[
        "backend_metadata"
    ]["autotune_key"].replace(
        "direct_hip_prefix9_native_input_colpair_grouped_rns_gemm_v2",
        old_oneshot_kernel,
    )
    expect_invalid(bad_oneshot_stale_kernel, "direct-HIP one-shot bounded captures must use selected_kernel")

    bad_finite_oneshot_pack_timing = copy.deepcopy(direct_hip_finite_oneshot)
    bad_finite_oneshot_pack_timing["raw_timings_us"]["pack"][0] = 1
    expect_invalid(bad_finite_oneshot_pack_timing, "public one-shot captures must report raw_timings_us.pack")

    bad_finite_oneshot_stale_pack_event = copy.deepcopy(direct_hip_finite_oneshot)
    bad_finite_oneshot_stale_pack_event["timing_metadata"]["gpu_event_phase_order"].insert(1, "finite_pack_kernel")
    bad_finite_oneshot_stale_pack_event["gpu_event_timings_us"]["finite_pack_kernel"] = [1.0, 1.0]
    bad_finite_oneshot_stale_pack_event["gpu_event_timing_summary_us"]["finite_pack_kernel"] = {
        "avg": 1.0,
        "median": 1.0,
        "p95": 1.0,
    }
    expect_invalid(bad_finite_oneshot_stale_pack_event, "direct-HIP finite one-shot GPU event phase set contains undeclared phases")

    bad_repack_prepack = copy.deepcopy(v4_ck_i64)
    bad_repack_prepack["reuse_packed_inputs"] = False
    bad_repack_prepack["pack_mode"] = "per_repeat_repack"
    bad_repack_prepack["prepack_setup_us"] = 1
    bad_repack_prepack["avg_prepack_setup_us"] = 1.0
    bad_repack_prepack["timing_metadata"]["pack_mode"] = "per_repeat_repack"
    expect_invalid(bad_repack_prepack, "prepack_setup_us=null")

    bad_repack_strategy = copy.deepcopy(v4_ck_i64)
    bad_repack_strategy["prepack_reuse_strategy"] = "persistent_matrix_residency"
    bad_repack_strategy["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    expect_invalid(bad_repack_strategy, "prepack_reuse_strategy=none")

    bad_length = copy.deepcopy(bounded)
    bad_length["raw_timings_us"]["pack"].pop()
    expect_invalid(bad_length, "raw_timings_us.pack length")

    bad_summary = copy.deepcopy(bounded)
    bad_summary["gpu_event_timing_summary_us"]["crt_export"]["avg"] = 999.0
    expect_invalid(bad_summary, "gpu_event_timing_summary_us.crt_export.avg")

    bad_event_source = copy.deepcopy(bounded)
    bad_event_source["timing_metadata"]["gpu_event_timing_source"] = "std::chrono::steady_clock"
    expect_invalid(bad_event_source, "gpu_event_timing_source must be hipEventElapsedTime")

    bad_event_scope = copy.deepcopy(bounded)
    bad_event_scope["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_unknown_scope"
    expect_invalid(bad_event_scope, "known direct-HIP scope")

    bad_hipblaslt_scope = copy.deepcopy(v4_hipblaslt_i64)
    bad_hipblaslt_scope["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
    expect_invalid(bad_hipblaslt_scope, "known hipBLASLt scope")

    bad_ck_library = copy.deepcopy(v4_ck_i64)
    bad_ck_library["backend_metadata"]["accelerator_library"] = "HIP runtime"
    expect_invalid(bad_ck_library, "Composable Kernel")

    bad_ck_kernel = copy.deepcopy(v4_ck_adaptive_u64)
    bad_ck_kernel["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    bad_ck_kernel["backend_metadata"]["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    expect_invalid(bad_ck_kernel, "per-tile adaptive ck captures")

    bad_ck_events = copy.deepcopy(v4_ck_adaptive_u64)
    bad_ck_events["timing_metadata"]["gpu_event_timing_source_scope"] = "ck_default_stream"
    expect_invalid(bad_ck_events, "accelerator_backend_default_stream_deep_kernel_events")

    bad_rocwmma_library = copy.deepcopy(v4_rocwmma_i64)
    bad_rocwmma_library["backend_metadata"]["accelerator_library"] = "HIP runtime"
    expect_invalid(bad_rocwmma_library, "rocWMMA")

    bad_rocwmma_kernel = copy.deepcopy(v4_rocwmma_adaptive_u64)
    bad_rocwmma_kernel["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    bad_rocwmma_kernel["backend_metadata"]["selected_kernel"] = "direct_hip_tiled_rns_gemm_v1"
    expect_invalid(bad_rocwmma_kernel, "per-tile adaptive rocwmma captures")

    bad_rocwmma_events = copy.deepcopy(v4_rocwmma_adaptive_u64)
    bad_rocwmma_events["timing_metadata"]["gpu_event_timing_source_scope"] = "rocwmma_default_stream"
    expect_invalid(bad_rocwmma_events, "accelerator_backend_default_stream_deep_kernel_events")

    bad_hip_target = copy.deepcopy(v4_ck_i64)
    bad_hip_target["device"]["gcn_arch"] = "unknown"
    expect_invalid(bad_hip_target, "HIP backend captures must include non-placeholder device.gcn_arch")

    missing_target_key = copy.deepcopy(v4_ck_i64)
    missing_target_key["backend_metadata"]["autotune_key"] = missing_target_key["backend_metadata"][
        "autotune_key"
    ].replace(";target_id=gfx1100", "")
    expect_invalid(missing_target_key, "backend_metadata.autotune_key must include target_id=gfx1100")

    wrong_target_key = copy.deepcopy(v4_ck_i64)
    wrong_target_key["backend_metadata"]["autotune_key"] = wrong_target_key["backend_metadata"][
        "autotune_key"
    ].replace(";target_id=gfx1100;", ";target_id=gfx1101;")
    expect_invalid(wrong_target_key, "backend_metadata.autotune_key must include target_id=gfx1100")

    bad_hip_available = copy.deepcopy(v4_rocwmma_i64)
    bad_hip_available["device"]["hip_available"] = 0
    expect_invalid(bad_hip_available, "HIP backend captures must use device.hip_available=1")

    bad_vector_source = copy.deepcopy(v4_vector_i64)
    bad_vector_source["backend_metadata"]["source"] = "rns8_get_plan_backend_info"
    expect_invalid(bad_vector_source, "rns8_bench_vector_alu_baseline")

    bad_vector_accelerator = copy.deepcopy(v4_vector_u64)
    bad_vector_accelerator["backend_metadata"]["accelerator_backend"] = True
    expect_invalid(bad_vector_accelerator, "accelerator_backend")

    bad_vector_epilogue = copy.deepcopy(v4_vector_i64)
    bad_vector_epilogue["epilogue_type"] = "crt_export"
    expect_invalid(bad_vector_epilogue, "direct_int64_export")

    bad_vector_prereq = copy.deepcopy(v4_vector_u64)
    bad_vector_prereq["comparison_baseline"]["required_before_speedup_claim"] = ["same_contract_cpu_reference"]
    expect_invalid(bad_vector_prereq, "same_contract_direct_hip_correctness")

    bad_finite_ring_modulus = copy.deepcopy(v4_finite_ring_ck)
    bad_finite_ring_modulus["finite_modulus"] = 1
    expect_invalid(bad_finite_ring_modulus, "finite_ring_u8 finite_modulus")

    bad_finite_field_modulus = copy.deepcopy(v4_finite_field_rocwmma)
    bad_finite_field_modulus["finite_modulus"] = 255
    expect_invalid(bad_finite_field_modulus, "finite_field_u8 finite_modulus")

    bad_finite_prefix = copy.deepcopy(v4_finite_ring_ck)
    bad_finite_prefix["prefix"] = 9
    expect_invalid(bad_finite_prefix, "finite-u8 captures must use prefix=0")

    bad_finite_key = copy.deepcopy(v4_finite_ring_ck)
    bad_finite_key["backend_metadata"]["autotune_key"] = bad_finite_key["backend_metadata"]["autotune_key"].replace(
        ";finite_modulus=255",
        "",
    )
    expect_invalid(bad_finite_key, "finite-u8 backend_metadata.autotune_key must include finite_modulus")

    bad_ck_finite_kernel = copy.deepcopy(v4_finite_ring_ck)
    bad_ck_finite_kernel["finite_modulus"] = 256
    bad_ck_finite_kernel["backend_metadata"]["autotune_key"] = bad_ck_finite_kernel["backend_metadata"][
        "autotune_key"
    ].replace(
        "finite_modulus=255",
        "finite_modulus=256",
    )
    expect_invalid(bad_ck_finite_kernel, "CK finite-u8 modulus 256 captures")

    bad_rocwmma_finite_kernel = copy.deepcopy(v4_finite_field_rocwmma)
    bad_rocwmma_finite_kernel["semantics"] = "finite_ring_u8"
    bad_rocwmma_finite_kernel["finite_modulus"] = 256
    bad_rocwmma_finite_kernel["backend_metadata"]["autotune_key"] = bad_rocwmma_finite_kernel["backend_metadata"][
        "autotune_key"
    ].replace(
        "semantics=finite_field_u8",
        "semantics=finite_ring_u8",
    ).replace(
        "finite_modulus=251",
        "finite_modulus=256",
    )
    expect_invalid(bad_rocwmma_finite_kernel, "rocWMMA finite-u8 modulus 256 captures")

    bad_finite_epilogue = copy.deepcopy(v4_finite_field_rocwmma)
    bad_finite_epilogue["epilogue_type"] = "crt_export"
    expect_invalid(bad_finite_epilogue, "canonical_u8_export")

    direct_finite_specialized = as_direct_hip_finite_capture(
        v4_finite_ring_ck,
        255,
        "direct_hip_tiled_finite_u8_gemm_mod255_v1",
        "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide",
    )
    validate_capture(direct_finite_specialized)

    bad_direct_finite_specialized_isa = copy.deepcopy(direct_finite_specialized)
    bad_direct_finite_specialized_isa["backend_metadata"]["isa_evidence"] = (
        "rns8_hip_direct_reciprocal_isa_gate"
    )
    expect_invalid(
        bad_direct_finite_specialized_isa,
        "direct-HIP finite-u8 specialized captures",
    )

    bad_direct_finite_specialized_kernel = copy.deepcopy(direct_finite_specialized)
    bad_direct_finite_specialized_kernel["selected_kernel"] = (
        "direct_hip_tiled_finite_u8_gemm_v1"
    )
    bad_direct_finite_specialized_kernel["backend_metadata"]["selected_kernel"] = (
        "direct_hip_tiled_finite_u8_gemm_v1"
    )
    expect_invalid(
        bad_direct_finite_specialized_kernel,
        "direct-HIP finite-u8 modulus 255 captures",
    )

    direct_finite_generic = as_direct_hip_finite_capture(
        v4_finite_ring_ck,
        127,
        "direct_hip_tiled_finite_u8_gemm_v1",
        "rns8_hip_direct_reciprocal_isa_gate",
    )
    validate_capture(direct_finite_generic)

    bad_direct_finite_generic_isa = copy.deepcopy(direct_finite_generic)
    bad_direct_finite_generic_isa["backend_metadata"]["isa_evidence"] = (
        "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide"
    )
    expect_invalid(
        bad_direct_finite_generic_isa,
        "direct-HIP generic finite-u8 captures",
    )

    missing_event_phase_order = copy.deepcopy(bounded)
    del missing_event_phase_order["timing_metadata"]["gpu_event_phase_order"]
    expect_invalid(missing_event_phase_order, "gpu_event_phase_order must be an array of strings when events are available")

    undeclared_event_phase = copy.deepcopy(bounded)
    undeclared_event_phase["gpu_event_timings_us"]["old_event_scope_phase"] = [1.0, 1.0, 1.0]
    expect_invalid(undeclared_event_phase, "undeclared phase old_event_scope_phase")

    duplicate_event_phase_order = copy.deepcopy(v4_vector_i64)
    duplicate_event_phase_order["timing_metadata"]["gpu_event_phase_order"].append("crt_export")
    expect_invalid(duplicate_event_phase_order, "gpu_event_phase_order must not contain duplicates")

    incomplete_vector_events = copy.deepcopy(v4_vector_i64)
    incomplete_vector_events["timing_metadata"]["gpu_event_phase_order"].remove("vector_alu_status_d2h")
    del incomplete_vector_events["gpu_event_timings_us"]["vector_alu_status_d2h"]
    del incomplete_vector_events["gpu_event_timing_summary_us"]["vector_alu_status_d2h"]
    expect_invalid(incomplete_vector_events, "vector-ALU GPU event phase set is incomplete")

    stale_deep_scope = copy.deepcopy(v4_ck_adaptive_u64)
    stale_deep_scope["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export"
    )
    expect_invalid(stale_deep_scope, "deep accelerator GPU event labels require")

    undeclared_deep_phase = copy.deepcopy(v4_ck_adaptive_u64)
    undeclared_deep_phase["timing_metadata"]["gpu_event_phase_order"].insert(6, "ck_prefix_99_fake_kernel")
    undeclared_deep_phase["gpu_event_timings_us"]["ck_prefix_99_fake_kernel"] = [1.0, 1.0]
    undeclared_deep_phase["gpu_event_timing_summary_us"]["ck_prefix_99_fake_kernel"] = zero_summary()
    expect_invalid(undeclared_deep_phase, "deep accelerator GPU event phase set contains undeclared phases")

    zero_deep_phase = copy.deepcopy(v4_ck_adaptive_u64)
    zero_deep_count = zero_deep_phase["schedule_metadata"]["tile_count"]
    zero_deep_phase["schedule_metadata"].update(
        {
            "flags": 1,
            "zero_output_tile_count": zero_deep_count,
            "zero_output_tile_fraction": 1.0,
            "zero_output_selected_residue_planes": zero_deep_count
            * zero_deep_phase["schedule_metadata"]["max_selected_prefix"],
            "zero_output_skip_active": True,
        }
    )
    insert_at = zero_deep_phase["timing_metadata"]["gpu_event_phase_order"].index("ck_add_centered_kernel") + 1
    zero_deep_phase["timing_metadata"]["gpu_event_phase_order"].insert(insert_at, "ck_zero_output_tile_memset")
    repeats = zero_deep_phase["repeats"]
    zero_deep_phase["gpu_event_timings_us"]["ck_zero_output_tile_memset"] = [0.25] * repeats
    zero_deep_phase["gpu_event_timing_summary_us"]["ck_zero_output_tile_memset"] = summary([0.25] * repeats)
    validate_capture(zero_deep_phase)

    bad_schedule_tile = copy.deepcopy(bounded)
    bad_schedule_tile["tile_m"] = 96
    bad_schedule_tile["schedule_metadata"]["tile_m"] = 96
    expect_invalid(bad_schedule_tile, "tile_m must be a power of two")

    bad_schedule_prefix = copy.deepcopy(bounded)
    bad_schedule_prefix["schedule_metadata"]["max_selected_prefix"] = bad_schedule_prefix["prefix"]
    expect_invalid(bad_schedule_prefix, "adaptive_skip_active must match")

    bad_wrap_prefix = copy.deepcopy(wrap64)
    bad_wrap_prefix["prefix"] = 9
    expect_invalid(bad_wrap_prefix, "wrap64 captures must use prefix=0")

    bad_wrap_backend = copy.deepcopy(wrap64)
    bad_wrap_backend["backend_selected"] = "cpu-reference"
    expect_invalid(bad_wrap_backend, "rocWMMA candidate")

    bad_wrap64_hip_phase = copy.deepcopy(v4_wrap64_hip)
    bad_wrap64_hip_phase["gpu_event_timing_summary_us"]["wrap64_export_d2h"]["avg"] = 999.0
    expect_invalid(bad_wrap64_hip_phase, "gpu_event_timing_summary_us.wrap64_export_d2h.avg")

    bad_candidate_schedule_source = copy.deepcopy(v4_wrap64_rocwmma_candidate)
    bad_candidate_schedule_source["schedule_metadata"]["source"] = "rns8_get_plan_schedule_info"
    expect_invalid(bad_candidate_schedule_source, "rns8_bench_wrap64_rocwmma_candidate_static_schedule")

    bad_candidate_scope = copy.deepcopy(v4_wrap64_rocwmma_candidate)
    bad_candidate_scope["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export"
    )
    expect_invalid(bad_candidate_scope, "rocwmma_wrap64_byte_gemm36_candidate_default_stream_operation_groups")

    bad_candidate_correctness_flag = copy.deepcopy(v4_wrap64_rocwmma_candidate)
    bad_candidate_correctness_flag["backend_metadata"]["correctness_backend"] = True
    expect_invalid(bad_candidate_correctness_flag, "correctness_backend=False")

    bad_baseline_prereq = copy.deepcopy(v4_adaptive_u64)
    bad_baseline_prereq["comparison_baseline"]["required_before_speedup_claim"] = ["same_contract_cpu_reference"]
    expect_invalid(bad_baseline_prereq, "same_contract_direct_hip_vector_alu_int64")

    bad_speedup_claim = copy.deepcopy(v4_ck_i64)
    bad_speedup_claim["comparison_baseline"]["speedup_claimed"] = True
    expect_invalid(bad_speedup_claim, "speedup claims require a reviewed same-contract comparison baseline")

    legacy_reviewed_speedup = copy.deepcopy(v4_ck_i64)
    legacy_reviewed_speedup["comparison_baseline"]["status"] = "reviewed_same_contract_baseline"
    legacy_reviewed_speedup["comparison_baseline"]["speedup_claimed"] = True
    legacy_reviewed_speedup["comparison_baseline"]["selected_reference"] = "hip-direct"
    validate_capture(legacy_reviewed_speedup)

    bad_performance_promotion = copy.deepcopy(v4_rocwmma_i64)
    bad_performance_promotion["backend_metadata"]["performance_validated"] = True
    expect_invalid(
        bad_performance_promotion,
        "performance_validated captures require comparison_baseline.status=reviewed_release_same_contract_baseline",
    )

    bad_legacy_performance_promotion = copy.deepcopy(v4_rocwmma_i64)
    bad_legacy_performance_promotion["comparison_baseline"]["status"] = "reviewed_same_contract_baseline"
    bad_legacy_performance_promotion["comparison_baseline"]["selected_reference"] = "hip-direct"
    bad_legacy_performance_promotion["backend_metadata"]["performance_validated"] = True
    expect_invalid(
        bad_legacy_performance_promotion,
        "performance_validated captures require comparison_baseline.status=reviewed_release_same_contract_baseline",
    )

    bad_legacy_derived_tops = copy.deepcopy(v4_ck_i64)
    bad_legacy_derived_tops["comparison_baseline"]["status"] = "reviewed_same_contract_baseline"
    bad_legacy_derived_tops["comparison_baseline"]["selected_reference"] = "hip-direct"
    bad_legacy_derived_tops["derived_tops_equivalent"] = 123.0
    expect_invalid(
        bad_legacy_derived_tops,
        "derived_tops_equivalent requires a reviewed release same-contract comparison baseline",
    )

    release_performance_promotion = copy.deepcopy(v4_rocwmma_i64)
    release_performance_promotion["comparison_baseline"]["status"] = "reviewed_release_same_contract_baseline"
    release_performance_promotion["comparison_baseline"]["speedup_claimed"] = True
    release_performance_promotion["comparison_baseline"]["selected_reference"] = "hip-direct"
    release_performance_promotion["backend_metadata"]["performance_validated"] = True
    release_performance_promotion["derived_tops_equivalent"] = 123.0
    validate_capture(release_performance_promotion)

    release_performance_capture_without_speedup_claim = copy.deepcopy(v4_rocwmma_i64)
    release_performance_capture_without_speedup_claim["backend_requested"] = "auto"
    release_performance_capture_without_speedup_claim["comparison_baseline"]["status"] = (
        "reviewed_release_same_contract_baseline"
    )
    release_performance_capture_without_speedup_claim["comparison_baseline"]["speedup_claimed"] = False
    release_performance_capture_without_speedup_claim["comparison_baseline"]["selected_reference"] = None
    release_performance_capture_without_speedup_claim["backend_metadata"]["performance_validated"] = True
    validate_capture(release_performance_capture_without_speedup_claim)

    bad_current_version = copy.deepcopy(v4_adaptive_u64)
    bad_current_version["schema_version"] = 3
    expect_invalid(bad_current_version, "expected 4")

    missing_current_version = copy.deepcopy(v4_adaptive_u64)
    del missing_current_version["schema_version"]
    expect_invalid(missing_current_version, "missing required field schema_version")

    bad_schedule_summary = copy.deepcopy(v4_adaptive_u64)
    bad_schedule_summary["raw_timings_us"]["scheduling"] = [6]
    expect_invalid(bad_schedule_summary, "timing_summary_us.scheduling.avg")

    missing_tile_bound_scan = copy.deepcopy(v4_adaptive_u64)
    del missing_tile_bound_scan["raw_timings_us"]["tile_bound_scan"]
    expect_invalid(missing_tile_bound_scan, "raw_timings_us.tile_bound_scan must be an array")

    bad_tile_bound_scan_summary = copy.deepcopy(v4_adaptive_u64)
    bad_tile_bound_scan_summary["raw_timings_us"]["tile_bound_scan"] = [11]
    expect_invalid(bad_tile_bound_scan_summary, "timing_summary_us.tile_bound_scan.avg")

    bad_tile_bound_scan_availability = copy.deepcopy(v4_adaptive_u64)
    del bad_tile_bound_scan_availability["timing_metadata"]["phase_availability"]["tile_bound_scan"]
    expect_invalid(
        bad_tile_bound_scan_availability,
        "phase_availability.tile_bound_scan must be an object for per-tile captures",
    )

    bad_reduction_scope = copy.deepcopy(v4_wrap64_hip)
    bad_reduction_scope["timing_metadata"]["phase_availability"]["reduction"]["scope"] = "fused_into_rns_gemm"
    expect_invalid(bad_reduction_scope, "phase_availability.reduction.scope")

    bad_v4_bound = copy.deepcopy(v4_adaptive_u64)
    bad_v4_bound["bound"] = 1
    expect_invalid(bad_v4_bound, "per-tile adaptive captures must use bound=0")

    bad_v4_tile_count = copy.deepcopy(v4_adaptive_u64)
    bad_v4_tile_count["tile_bounds_u64"]["count"] = 3
    expect_invalid(bad_v4_tile_count, "tile_bounds_u64.count must match")

    zero_skip_schedule = copy.deepcopy(v4_adaptive_u64)
    zero_tile_count = zero_skip_schedule["schedule_metadata"]["tile_count"]
    zero_skip_schedule["schedule_metadata"].update(
        {
            "flags": 1,
            "zero_output_tile_count": 1,
            "zero_output_tile_fraction": 1.0 / zero_tile_count,
            "zero_output_selected_residue_planes": zero_skip_schedule["schedule_metadata"]["min_selected_prefix"],
            "zero_output_skip_active": True,
        }
    )
    zero_skip_kernel = "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3"
    stale_zero_skip_kernel = "direct_hip_tiled_active_prefix_rns_gemm_v2"
    zero_skip_schedule["selected_kernel"] = zero_skip_kernel
    zero_skip_schedule["backend_metadata"]["selected_kernel"] = zero_skip_kernel
    zero_skip_schedule["backend_metadata"]["workspace_required_bytes"] = 1040
    zero_skip_schedule["backend_metadata"]["autotune_key"] = zero_skip_schedule["backend_metadata"][
        "autotune_key"
    ].replace(stale_zero_skip_kernel, zero_skip_kernel)
    zero_insert_at = zero_skip_schedule["timing_metadata"]["gpu_event_phase_order"].index("rns_gemm_kernel_group") + 1
    zero_skip_schedule["timing_metadata"]["gpu_event_phase_order"].insert(
        zero_insert_at, "direct_hip_zero_output_tile_memset"
    )
    repeats = zero_skip_schedule["repeats"]
    zero_skip_schedule["gpu_event_timings_us"]["direct_hip_zero_output_tile_memset"] = [0.25] * repeats
    zero_skip_schedule["gpu_event_timing_summary_us"]["direct_hip_zero_output_tile_memset"] = summary([0.25] * repeats)
    zero_skip_schedule["gpu_event_timings_us"]["rns_gemm"] = [
        value + 0.25 for value in zero_skip_schedule["gpu_event_timings_us"]["rns_gemm_kernel_group"]
    ]
    zero_skip_schedule["gpu_event_timing_summary_us"]["rns_gemm"] = summary(
        zero_skip_schedule["gpu_event_timings_us"]["rns_gemm"]
    )
    validate_capture(zero_skip_schedule)

    all_zero_skip_schedule = copy.deepcopy(zero_skip_schedule)
    all_zero_planes = (
        all_zero_skip_schedule["schedule_metadata"]["tile_count"]
        * all_zero_skip_schedule["schedule_metadata"]["min_selected_prefix"]
    )
    all_zero_skip_schedule["schedule_metadata"].update(
        {
            "zero_output_tile_count": zero_tile_count,
            "zero_output_tile_fraction": 1.0,
            "zero_output_selected_residue_planes": all_zero_planes,
        }
    )
    all_zero_skip_schedule["backend_metadata"]["workspace_required_bytes"] = 0
    all_zero_skip_schedule["raw_timings_us"]["pack"] = [0] * repeats
    all_zero_skip_schedule["timing_summary_us"]["pack"] = zero_summary()
    all_zero_skip_schedule["avg_pack_us"] = 0.0
    for phase in ["pack_h2d", "pack_kernel", "pack"]:
        all_zero_skip_schedule["gpu_event_timings_us"][phase] = [0.0] * repeats
        all_zero_skip_schedule["gpu_event_timing_summary_us"][phase] = zero_summary()
    all_zero_skip_schedule["gpu_event_timings_us"]["rns_gemm_kernel_group"] = [0.0] * repeats
    all_zero_skip_schedule["gpu_event_timing_summary_us"]["rns_gemm_kernel_group"] = summary([0.0] * repeats)
    all_zero_skip_schedule["gpu_event_timings_us"]["rns_gemm"] = [0.25] * repeats
    all_zero_skip_schedule["gpu_event_timing_summary_us"]["rns_gemm"] = summary([0.25] * repeats)
    all_zero_skip_schedule["gpu_event_timings_us"]["crt_export_status_memset"] = [0.0] * repeats
    all_zero_skip_schedule["gpu_event_timing_summary_us"]["crt_export_status_memset"] = summary([0.0] * repeats)
    all_zero_skip_schedule["gpu_event_timings_us"]["crt_export_status_d2h"] = [0.0] * repeats
    all_zero_skip_schedule["gpu_event_timing_summary_us"]["crt_export_status_d2h"] = summary([0.0] * repeats)
    validate_capture(all_zero_skip_schedule)

    bad_all_zero_pack_timing = copy.deepcopy(all_zero_skip_schedule)
    bad_all_zero_pack_timing["raw_timings_us"]["pack"][0] = 1
    bad_all_zero_pack_timing["timing_summary_us"]["pack"] = summary(bad_all_zero_pack_timing["raw_timings_us"]["pack"])
    bad_all_zero_pack_timing["avg_pack_us"] = 1.0 / repeats
    expect_invalid(
        bad_all_zero_pack_timing,
        "all-zero direct-HIP adaptive captures must report raw_timings_us.pack",
    )

    bad_all_zero_pack_event = copy.deepcopy(all_zero_skip_schedule)
    bad_all_zero_pack_event["gpu_event_timings_us"]["pack_h2d"][0] = 1.0
    bad_all_zero_pack_event["gpu_event_timing_summary_us"]["pack_h2d"] = summary(
        bad_all_zero_pack_event["gpu_event_timings_us"]["pack_h2d"]
    )
    expect_invalid(
        bad_all_zero_pack_event,
        "all-zero direct-HIP adaptive captures must report gpu_event_timings_us.pack_h2d",
    )

    bad_zero_skip_stale_kernel = copy.deepcopy(zero_skip_schedule)
    bad_zero_skip_stale_kernel["selected_kernel"] = stale_zero_skip_kernel
    bad_zero_skip_stale_kernel["backend_metadata"]["selected_kernel"] = stale_zero_skip_kernel
    bad_zero_skip_stale_kernel["backend_metadata"]["autotune_key"] = bad_zero_skip_stale_kernel["backend_metadata"][
        "autotune_key"
    ].replace(zero_skip_kernel, stale_zero_skip_kernel)
    expect_invalid(bad_zero_skip_stale_kernel, "direct_hip_tiled_active_prefix_zero_skip_rns_gemm_v3")

    bad_zero_skip_unknown_flag = copy.deepcopy(zero_skip_schedule)
    bad_zero_skip_unknown_flag["schedule_metadata"]["flags"] = 4
    expect_invalid(bad_zero_skip_unknown_flag, "unknown tile schedule flags")

    bad_zero_skip_missing_flag = copy.deepcopy(zero_skip_schedule)
    del bad_zero_skip_missing_flag["schedule_metadata"]["flags"]
    expect_invalid(bad_zero_skip_missing_flag, "requires ZERO_OUTPUT schedule flag")

    bad_zero_skip_count = copy.deepcopy(zero_skip_schedule)
    bad_zero_skip_count["schedule_metadata"]["zero_output_tile_count"] = zero_tile_count + 1
    expect_invalid(bad_zero_skip_count, "zero_output_tile_count must be <= tile_count")

    bad_zero_skip_fraction = copy.deepcopy(zero_skip_schedule)
    bad_zero_skip_fraction["schedule_metadata"]["zero_output_tile_fraction"] = 1.0
    expect_invalid(bad_zero_skip_fraction, "zero_output_tile_fraction must match")

    bad_v4_skip_flag = copy.deepcopy(v4_adaptive_u64)
    bad_v4_skip_flag["schedule_metadata"]["adaptive_skip_active"] = False
    expect_invalid(bad_v4_skip_flag, "adaptive_skip_active must match")

    bad_v4_per_modulus = copy.deepcopy(v4_adaptive_u64)
    bad_v4_per_modulus["per_modulus_gemm_estimate_applicable"] = True
    expect_invalid(bad_v4_per_modulus, "fixed-prefix contract")

    bad_v4_toolchain = copy.deepcopy(v4_adaptive_u64)
    del bad_v4_toolchain["hip_toolchain"]
    expect_invalid(bad_v4_toolchain, "missing required field hip_toolchain")

    bad_v4_toolchain_enabled = copy.deepcopy(v4_adaptive_u64)
    bad_v4_toolchain_enabled["hip_toolchain"]["enabled"] = False
    expect_invalid(bad_v4_toolchain_enabled, "HIP backend captures must set hip_toolchain.enabled=true")

    bad_v4_scope = copy.deepcopy(v4_adaptive_u64)
    bad_v4_scope["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
    expect_invalid(bad_v4_scope, "bounded_adaptive")

    bad_v4_wrap64_kernel = copy.deepcopy(v4_wrap64_hip)
    bad_v4_wrap64_kernel["selected_kernel"] = "direct_hip_wrap64_comba_correctness_v1"
    expect_invalid(bad_v4_wrap64_kernel, "byte_gemm36")

    stale_v3_wrap64_kernel = copy.deepcopy(v4_wrap64_hip)
    stale_v3_wrap64_kernel["selected_kernel"] = "direct_hip_wrap64_byte_gemm36_tiled_2d_v3"
    stale_v3_wrap64_kernel["backend_metadata"]["selected_kernel"] = "direct_hip_wrap64_byte_gemm36_tiled_2d_v3"
    stale_v3_wrap64_kernel["backend_metadata"]["autotune_key"] = stale_v3_wrap64_kernel["backend_metadata"][
        "autotune_key"
    ].replace(
        "kernel=direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4",
        "kernel=direct_hip_wrap64_byte_gemm36_tiled_2d_v3",
    )
    expect_invalid(stale_v3_wrap64_kernel, "direct_hip_wrap64_byte_gemm36_u32acc_tiled_2d_v4")

    bad_event_nullability = copy.deepcopy(wrap64)
    bad_event_nullability["timing_metadata"]["gpu_event_timing"] = False
    bad_event_nullability["timing_metadata"]["gpu_event_timing_source"] = None
    bad_event_nullability["timing_metadata"]["gpu_event_timing_source_scope"] = None
    bad_event_nullability["gpu_event_timings_us"] = {"pack": [1.0, 2.0]}
    bad_event_nullability["gpu_event_timing_summary_us"] = None
    expect_invalid(bad_event_nullability, "gpu_event_timings_us must be null")

    bad_event_phase_order_nullability = copy.deepcopy(wrap64)
    bad_event_phase_order_nullability["timing_metadata"]["gpu_event_timing"] = False
    bad_event_phase_order_nullability["timing_metadata"]["gpu_event_timing_source"] = None
    bad_event_phase_order_nullability["timing_metadata"]["gpu_event_timing_source_scope"] = None
    bad_event_phase_order_nullability["gpu_event_timings_us"] = None
    bad_event_phase_order_nullability["gpu_event_timing_summary_us"] = None
    expect_invalid(bad_event_phase_order_nullability, "gpu_event_phase_order must be null")

    print("benchmark schema self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
