#!/usr/bin/env python3
"""Self-test adaptive grouped scheduler reporting."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from benchmark_schema import load_capture, validate_capture

import adaptive_grouped_scheduler_report


FIXTURE = Path("tests") / "fixtures" / "benchmark_schema" / "v4_bounded_i64_adaptive_hip.json"


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"avg": sum(ordered) / len(ordered), "median": ordered[len(ordered) // 2], "p95": ordered[-1]}


def _resize_timings(capture: dict, end_to_end_us: int, repeats: int = 9) -> None:
    capture["warmups"] = 3
    capture["repeats"] = repeats
    phases = list(capture["raw_timings_us"])
    one_time_phases = {"planning", "scheduling", "tile_bound_scan", "matrix_alloc"}
    for phase in phases:
        value = end_to_end_us if phase == "end_to_end" else int(capture["timing_summary_us"][phase]["median"])
        count = 1 if phase in one_time_phases else repeats
        capture["raw_timings_us"][phase] = [value] * count
        capture["timing_summary_us"][phase] = _summary([float(value)] * count)
        capture[f"avg_{phase}_us"] = float(value)
    capture["avg_end_to_end_us"] = float(end_to_end_us)
    events = capture.get("gpu_event_timings_us")
    if isinstance(events, dict):
        for phase, values in list(events.items()):
            value = float(values[0]) if values else 0.0
            events[phase] = [value] * repeats
            capture["gpu_event_timing_summary_us"][phase] = _summary([value] * repeats)


def _add_adaptive_grouped_metadata(capture: dict, executed: bool = True) -> None:
    schedule = capture["schedule_metadata"]
    active_entries = schedule["tile_count"] * schedule["max_selected_prefix"]
    capture["adaptive_grouped_scheduler"] = {
        "requested": True,
        "strategy": "prefix_tile_zero_mask_grouped_descriptors",
        "descriptor_identity": "prefix=4;tile_m=64;tile_n=64;zero_tiles=0",
        "group_count": schedule["prefix_group_count"],
        "active_prefix_count": schedule["max_selected_prefix"],
        "active_tile_count": schedule["tile_count"],
        "active_entry_count": active_entries,
        "zero_tile_count": 0,
        "independent_launch_count_model": active_entries,
        "aggregate_launch_count_model": 1,
        "launch_reduction_ratio": float(active_entries),
        "event_scope": "aggregate_rns_gemm_kernel_group_per_measured_repeat" if executed else "not_available",
        "selected_prefix_histogram": "min=1;max=4",
        "capture_status": "executed" if executed else "metadata_only_unsupported_for_execution_path",
        "unsupported_reason": None if executed else "adaptive_grouped_scheduler_not_executed_by_current_path",
        "promotion_eligible": False,
    }


def _remove_gpu_events(capture: dict) -> None:
    metadata = capture["timing_metadata"]
    metadata["gpu_event_timing"] = False
    metadata["gpu_event_timing_status"] = "not_applicable"
    metadata["gpu_event_timing_reason"] = "cpu_reference_capture"
    metadata["gpu_event_timing_source"] = None
    metadata["gpu_event_timing_source_scope"] = None
    metadata["gpu_event_timing_caveat"] = None
    metadata["gpu_event_phase_order"] = None
    capture["gpu_event_timings_us"] = None
    capture["gpu_event_timing_summary_us"] = None


def _capture(backend: str, end_to_end_us: int, *, candidate: bool = False) -> dict:
    capture = copy.deepcopy(load_capture(FIXTURE))
    _resize_timings(capture, end_to_end_us)
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    capture["command_line"] = f"rns8-bench --backend {backend} --semantics bounded-i64"
    metadata = capture["backend_metadata"]
    metadata["backend"] = backend
    if backend == "cpu-reference":
        capture["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
        metadata["selected_kernel"] = capture["selected_kernel"]
        metadata["accelerator_backend"] = False
        metadata["matrix_engine_backend"] = False
        metadata["accelerator_library"] = None
        metadata["capability_status"] = "implemented_correctness_backend"
        metadata["workspace_mode"] = "host_reference_workspace"
        metadata["workspace_required_bytes"] = 0
        metadata["isa_evidence"] = "not_applicable_cpu"
        metadata["autotune_key"] = metadata["autotune_key"].replace("backend=hip-direct", "backend=cpu-reference")
        metadata["autotune_key"] = metadata["autotune_key"].replace("target_id=gfx1100", "target_id=cpu")
        metadata["autotune_key"] = metadata["autotune_key"].replace(
            "kernel=direct_hip_grouped_active_prefix_schedule_rns_gemm_v3",
            "kernel=cpu_reference_scalar_rns_gemm_v1",
        )
        capture["target_id"] = "cpu"
        capture["device"] = {
            "device_id": -1,
            "name": "CPU reference",
            "gcn_arch": "none",
            "hip_available": 0,
            "hip_runtime_version": 0,
            "hip_driver_version": 0,
            "global_mem_bytes": 0,
        }
        _remove_gpu_events(capture)
        capture["comparison_baseline"]["required_before_speedup_claim"] = [
            "same_contract_cpu_reference",
            "same_contract_direct_hip_correctness",
            "same_contract_direct_hip_vector_alu_int64",
        ]
    elif backend == "hip-direct":
        capture["selected_kernel"] = "direct_hip_grouped_active_prefix_schedule_rns_gemm_v3"
        metadata["selected_kernel"] = capture["selected_kernel"]
    if candidate:
        _add_adaptive_grouped_metadata(capture)
        capture["target_variant"] = {
            "target_id": "gfx1100",
            "target_arch": "gfx1100",
            "target_namespace": "gfx1100",
            "target_cache_key": "gfx1100:AMD Radeon RX 7900 XTX:7.1",
            "target_instance_id": "device0",
            "device_name": "AMD Radeon RX 7900 XTX",
            "device_index": 0,
            "visible_device_count": 1,
            "node_gpu_count": 1,
            "configured_amdgpu_targets": "gfx1100",
            "review_group_key": "bounded-i64-adaptive-grouped-scheduler-gfx1100",
        }
        capture["command_line"] += " --adaptive-grouped-scheduler"
    validate_capture(capture, Path(f"{backend}.json"))
    return capture


def _write(path: Path, capture: dict) -> Path:
    path.write_text(__import__("json").dumps(capture, indent=2), encoding="utf-8")
    return path


with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    cpu = _write(tmp / "cpu.json", _capture("cpu-reference", 2000))
    direct = _write(tmp / "direct.json", _capture("hip-direct", 1100))
    candidate = _write(tmp / "candidate.json", _capture("hip-direct", 900, candidate=True))
    report = adaptive_grouped_scheduler_report.build_report([cpu, direct, candidate])
    row = report["rows"][0]
    assert row["decision"] == "promote locally"
    assert row["speedup_vs_direct_hip"] and row["speedup_vs_direct_hip"] > 1.2
    assert row["launch_reduction_ratio"] == 16.0
    assert row["blockers"] == []

with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    direct = _write(tmp / "direct.json", _capture("hip-direct", 1100))
    candidate = _write(tmp / "candidate.json", _capture("hip-direct", 900, candidate=True))
    report = adaptive_grouped_scheduler_report.build_report([direct, candidate])
    row = report["rows"][0]
    assert row["decision"] == "keep experimental"
    assert "missing_cpu_baseline" in row["blockers"]
