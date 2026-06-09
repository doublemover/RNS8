#!/usr/bin/env python3
"""Self-test streaming-overlap reporting."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from benchmark_schema import validate_capture
from test_benchmark_schema_support_metadata import add_target_variant_fields
from test_benchmark_sweep_support import bounded_capture, remove_gpu_events, set_phase, with_accumulator_key_fields

import streaming_overlap_report


STREAMING_PHASES = [
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


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"avg": sum(ordered) / len(ordered), "median": ordered[len(ordered) // 2], "p95": ordered[-1]}


def _mark_reuse_b(capture: dict) -> dict:
    reused = copy.deepcopy(capture)
    reused["reuse_packed_inputs"] = True
    reused["reuse_packed_a"] = False
    reused["reuse_packed_b"] = True
    reused["pack_mode"] = "prepacked_reuse_b"
    reused["prepack_reuse_operands"] = ["B"]
    reused["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["prepack_setup_us"] = 11
    reused["avg_prepack_setup_us"] = 11.0
    reused["timing_metadata"]["pack_mode"] = "prepacked_reuse_b"
    reused["timing_metadata"]["prepack_reuse_operands"] = ["B"]
    reused["timing_metadata"]["prepack_reuse_strategy"] = "persistent_matrix_residency"
    reused["timing_metadata"].setdefault("phase_availability", {})["prepack_setup"] = {
        "timed": True,
        "timing_key": "prepack_setup_us",
        "scope": "one_time_before_warmups",
        "reason": "one-time persistent B packing before warmups",
    }
    return reused


def _as_generic_direct(capture: dict) -> dict:
    direct = copy.deepcopy(capture)
    kernel = "direct_hip_tiled_active_prefix_rns_gemm_v2"
    epilogue = "fused_centered_residue_then_crt_export"
    direct["benchmark_execution_mode"] = "persistent_resident_matrices"
    direct["benchmark"] = "rns8_bounded_gemm_persistent_rns"
    direct["selected_kernel"] = kernel
    metadata = direct["backend_metadata"]
    metadata["selected_kernel"] = kernel
    metadata["epilogue_mode"] = epilogue
    metadata["workspace_mode"] = "resident_device_buffers"
    metadata["autotune_key"] = with_accumulator_key_fields(
        (
            f"backend=hip-direct;semantics={direct['semantics']};m={direct['m']};n={direct['n']};"
            f"k={direct['k']};bound={direct['bound']};prefix={direct['prefix']};"
            f"tile_m={direct['tile_m']};tile_n={direct['tile_n']};groups=1;"
            "adaptive_prefix=0;adaptive_skip=0;execution=persistent_resident_matrices;"
            f"kernel={kernel};epilogue={epilogue}"
        ),
        direct,
    )
    direct["timing_metadata"]["benchmark_execution_mode"] = "persistent_resident_matrices"
    direct["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_backend_hooks"
    direct["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
    direct["timing_metadata"]["gpu_event_phase_order"] = list(STREAMING_PHASES)
    repeats = int(direct["repeats"])
    direct["gpu_event_timings_us"] = {phase: [1.0] * repeats for phase in STREAMING_PHASES}
    direct["gpu_event_timing_summary_us"] = {phase: _summary([1.0] * repeats) for phase in STREAMING_PHASES}
    return direct


def _as_cpu(capture: dict) -> dict:
    cpu = copy.deepcopy(capture)
    remove_gpu_events(cpu)
    cpu["backend_requested"] = "cpu-reference"
    cpu["backend_selected"] = "cpu-reference"
    cpu["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
    cpu["target_id"] = "cpu"
    cpu["device"] = {
        "device_id": -1,
        "name": "CPU reference",
        "gcn_arch": "none",
        "hip_available": 0,
        "hip_runtime_version": 0,
        "hip_driver_version": 0,
        "global_mem_bytes": 0,
    }
    metadata = cpu["backend_metadata"]
    metadata["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
    metadata["accelerator_backend"] = False
    metadata["matrix_engine_backend"] = False
    metadata["accelerator_library"] = None
    metadata["epilogue_mode"] = "fused_centered_residue_then_crt_export"
    metadata["workspace_mode"] = "host_reference_workspace"
    metadata["workspace_required_bytes"] = 0
    metadata["isa_evidence"] = "not_applicable_cpu"
    metadata["autotune_key"] = with_accumulator_key_fields(
        (
            f"backend=cpu-reference;semantics={cpu['semantics']};m={cpu['m']};n={cpu['n']};"
            f"k={cpu['k']};bound={cpu['bound']};prefix={cpu['prefix']};"
            f"tile_m={cpu['tile_m']};tile_n={cpu['tile_n']};groups=1;"
            "adaptive_prefix=0;adaptive_skip=0;execution=persistent_resident_matrices;"
            "kernel=cpu_reference_scalar_rns_gemm_v1;epilogue=fused_centered_residue_then_crt_export"
        ),
        cpu,
    )
    return cpu


def _as_streaming_candidate(capture: dict, end_to_end_us: int) -> dict:
    candidate = _mark_reuse_b(_as_generic_direct(capture))
    candidate["benchmark"] = "rns8_streaming_overlap_resident_b_pipeline"
    candidate["benchmark_execution_mode"] = "benchmark_streaming_overlap_resident_b_pipeline"
    candidate["command_line"] = f"{candidate['command_line']} --reuse-packed-b --streaming-overlap"
    candidate["timing_metadata"]["benchmark_execution_mode"] = "benchmark_streaming_overlap_resident_b_pipeline"
    candidate["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_streaming_overlap_hooks"
    candidate["timing_metadata"]["gpu_event_timing_source_scope"] = (
        "direct_hip_streaming_overlap_multistream_operation_groups"
    )
    candidate["timing_metadata"]["gpu_event_timing_caveat"] = (
        "HIP event timings record benchmark-owned nonblocking pack, GEMM, and export streams"
    )
    candidate["timing_metadata"]["gpu_event_phase_order"] = list(STREAMING_PHASES)
    set_phase(candidate, end_to_end_us)
    repeats = int(candidate["repeats"])
    candidate["gpu_event_timings_us"] = {
        phase: ([2.0] * repeats if phase in {"pack", "rns_gemm", "crt_export"} else [1.0] * repeats)
        for phase in STREAMING_PHASES
    }
    candidate["gpu_event_timing_summary_us"] = {
        phase: _summary(values) for phase, values in candidate["gpu_event_timings_us"].items()
    }
    candidate["streaming_overlap"] = {
        "requested": True,
        "pipeline": "pack_next_gemm_current_export_previous",
        "buffering": "double_buffered_benchmark_only",
        "dependency_contract": "pack_before_gemm;gemm_before_export;status_before_host_read;final_sync_before_checksum",
        "transfer_policy": "compact_or_padded_output_policy_declared_by_output_policy",
        "stream_count": 3,
        "buffer_count": 2,
        "measured_repeat_count": repeats,
        "batch_wall_us": int(end_to_end_us * repeats),
        "per_repeat_pipeline_us": end_to_end_us,
        "explicit_dependency_events": True,
        "stage_event_scope": "direct_hip_streaming_overlap_multistream_operation_groups",
        "capture_status": "executed",
        "unsupported_reason": None,
        "promotion_eligible": False,
    }
    candidate["correctness"] = "ok"
    candidate["per_modulus_gemm_estimate_applicable"] = False
    candidate["avg_per_modulus_gemm_estimate_us"] = float(candidate["avg_rns_gemm_us"])
    add_target_variant_fields(candidate)
    return candidate


def _sync_per_modulus_estimate(capture: dict) -> None:
    if capture.get("per_modulus_gemm_estimate_applicable") is False:
        capture["avg_per_modulus_gemm_estimate_us"] = float(capture["avg_rns_gemm_us"])
        return
    prefix = capture.get("selected_prefix", capture.get("prefix"))
    if isinstance(prefix, int) and prefix > 0:
        capture["avg_per_modulus_gemm_estimate_us"] = float(capture["avg_rns_gemm_us"]) / float(prefix)


def _write(path: Path, capture: dict) -> Path:
    _sync_per_modulus_estimate(capture)
    validate_capture(capture, path)
    path.write_text(json.dumps(capture, indent=2), encoding="utf-8")
    return path


with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    base = _as_generic_direct(bounded_capture("hip-direct", 1100))
    serial = _mark_reuse_b(base)
    set_phase(serial, 1100)
    cpu = _as_cpu(base)
    set_phase(cpu, 2000)
    candidate = _as_streaming_candidate(base, 900)
    report = streaming_overlap_report.build_report(
        [
            _write(tmp / "cpu.json", cpu),
            _write(tmp / "serial-direct.json", serial),
            _write(tmp / "streaming.json", candidate),
        ]
    )
    row = report["rows"][0]
    assert row["decision"] == "promote locally"
    assert row["speedup_vs_direct_hip"] and row["speedup_vs_direct_hip"] > 1.2
    assert row["blockers"] == []


with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    base = _as_generic_direct(bounded_capture("hip-direct", 1100))
    serial = _mark_reuse_b(base)
    set_phase(serial, 1100)
    candidate = _as_streaming_candidate(base, 900)
    report = streaming_overlap_report.build_report(
        [
            _write(tmp / "serial-direct.json", serial),
            _write(tmp / "streaming.json", candidate),
        ]
    )
    row = report["rows"][0]
    assert row["decision"] == "keep experimental"
    assert "missing_cpu_baseline" in row["blockers"]


with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    base = _as_generic_direct(bounded_capture("hip-direct", 1100))
    serial = _mark_reuse_b(base)
    set_phase(serial, 1100)
    cpu = _as_cpu(base)
    set_phase(cpu, 2000)
    candidate = _as_streaming_candidate(base, 1200)
    report = streaming_overlap_report.build_report(
        [
            _write(tmp / "cpu.json", cpu),
            _write(tmp / "serial-direct.json", serial),
            _write(tmp / "streaming.json", candidate),
        ]
    )
    row = report["rows"][0]
    assert row["decision"] == "drop/deprioritize"
