#!/usr/bin/env python3
"""Self-test tile-shape report promotion gates."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import tile_shape_report
from benchmark_schema import validate_capture
from test_benchmark_schema_support import (
    add_helper_lane_fields,
    apply_int32_accumulator_contract,
    expect_valid,
    summary,
    with_accumulator_key_fields,
)


DIRECT_KERNEL = "direct_hip_prefix9_grouped_rns_gemm_v1"
CPU_KERNEL = "cpu_reference_scalar_rns_gemm_v1"
REPEATS = 9
ONE_TIME_PHASES = {"planning", "scheduling", "matrix_alloc"}
HOST_PHASES = ["planning", "scheduling", "matrix_alloc", "pack", "rns_gemm", "crt_export", "end_to_end"]
DIRECT_GPU_EVENT_PHASES = [
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


def write_capture(path: Path, capture: dict) -> None:
    validate_capture(capture, path)
    path.write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")


def set_host_timings(capture: dict, medians: dict[str, int]) -> None:
    capture["warmups"] = 3
    capture["repeats"] = REPEATS
    capture["raw_timings_us"] = {}
    capture["timing_summary_us"] = {}
    for phase in HOST_PHASES:
        count = 1 if phase in ONE_TIME_PHASES else REPEATS
        values = [int(medians.get(phase, 1))] * count
        capture["raw_timings_us"][phase] = values
        capture["timing_summary_us"][phase] = summary(values)
        capture[f"avg_{phase}_us"] = float(sum(values) / len(values))
    capture["schedule_query_us"] = int(capture["avg_scheduling_us"])
    capture["avg_per_modulus_gemm_estimate_us"] = capture["avg_rns_gemm_us"] / float(capture["prefix"])


def set_direct_gpu_events(capture: dict) -> None:
    metadata = capture["timing_metadata"]
    metadata["gpu_event_timing"] = True
    metadata["gpu_event_timing_reason"] = "captured_by_direct_hip_backend_hooks"
    metadata["gpu_event_timing_status"] = "available"
    metadata["gpu_event_timing_source"] = "hipEventElapsedTime"
    metadata["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
    metadata["gpu_event_timing_caveat"] = "HIP event timings record Direct-HIP operation groups for tests"
    metadata["gpu_event_phase_order"] = list(DIRECT_GPU_EVENT_PHASES)
    metadata["phase_availability"]["reduction"]["scope"] = "fused_into_rns_gemm"
    capture["gpu_event_timings_us"] = {phase: [1.0] * REPEATS for phase in DIRECT_GPU_EVENT_PHASES}
    capture["gpu_event_timing_summary_us"] = {
        phase: summary(values) for phase, values in capture["gpu_event_timings_us"].items()
    }


def set_no_gpu_events(capture: dict) -> None:
    metadata = capture["timing_metadata"]
    metadata["gpu_event_timing"] = False
    metadata["gpu_event_timing_reason"] = "cpu_reference_backend"
    metadata["gpu_event_timing_status"] = "unavailable"
    metadata["gpu_event_timing_source"] = None
    metadata["gpu_event_timing_source_scope"] = None
    metadata["gpu_event_timing_caveat"] = "CPU reference capture has no GPU events"
    metadata["gpu_event_phase_order"] = None
    metadata["phase_availability"]["reduction"]["scope"] = "fused_into_rns_gemm"
    capture["gpu_event_timings_us"] = None
    capture["gpu_event_timing_summary_us"] = None


def set_tile_variant(capture: dict, name: str, tile_m: int, tile_n: int, kernel: str) -> None:
    capture["tile_m"] = tile_m
    capture["tile_n"] = tile_n
    capture["schedule_metadata"]["tile_m"] = tile_m
    capture["schedule_metadata"]["tile_n"] = tile_n
    capture["tile_shape_variant"] = {
        "name": name,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": capture["k_block_size"],
        "k_block_policy": "auto",
        "split_k_mode": "single_gpu_no_split_k",
        "accumulator_safety_key": (
            f"k_block_size={capture['k_block_size']};k_block_cap=65536;safe_for_k_block=true"
        ),
        "selected_kernel_identity": kernel,
        "resource_report_key": (
            f"tile_m={tile_m};tile_n={tile_n};tile_k={capture['k_block_size']};kernel={kernel}"
        ),
        "shape_family_bucket": "tiny",
        "resource_report_required": "isa_or_counter_for_non_default_tile_shape",
        "stale_kernel_rejection": "selected_kernel_identity_must_match_capture",
    }


def make_base_capture() -> dict:
    return add_helper_lane_fields(expect_valid("v4_bounded_i64_hipblaslt.json"))


def make_direct_capture(tile_m: int, tile_n: int, variant: str, end_to_end_us: int) -> dict:
    capture = make_base_capture()
    capture["backend_requested"] = "hip-direct"
    capture["backend_selected"] = "hip-direct"
    capture["selected_kernel"] = DIRECT_KERNEL
    metadata = capture["backend_metadata"]
    metadata.update(
        {
            "source": "rns8_get_plan_backend_info",
            "selected_kernel": DIRECT_KERNEL,
            "accelerator_backend": False,
            "correctness_backend": True,
            "matrix_engine_backend": False,
            "compiled_kernel_available": True,
            "exact_differential_validated": True,
            "performance_validated": False,
            "accelerator_library": "HIP runtime",
            "accelerator_version": None,
            "capability_status": "implemented_correctness_backend",
            "epilogue_mode": "fused_centered_residue_then_crt_export",
            "workspace_mode": "resident_device_buffers",
            "workspace_required_bytes": 0,
            "isa_evidence": "rns8_hip_direct_reciprocal_isa_gate",
        }
    )
    apply_int32_accumulator_contract(capture)
    set_tile_variant(capture, variant, tile_m, tile_n, DIRECT_KERNEL)
    metadata["autotune_key"] = with_accumulator_key_fields(
        "backend=hip-direct;"
        f"semantics={capture['semantics']};m={capture['m']};n={capture['n']};k={capture['k']};"
        f"prefix={capture['prefix']};tile_m={capture['tile_m']};tile_n={capture['tile_n']};"
        "groups=1;adaptive_prefix=0;adaptive_skip=0;"
        f"kernel={DIRECT_KERNEL};epilogue=fused_centered_residue_then_crt_export",
        capture,
    )
    set_direct_gpu_events(capture)
    set_host_timings(capture, {"end_to_end": end_to_end_us})
    return capture


def make_cpu_capture(end_to_end_us: int) -> dict:
    capture = make_base_capture()
    capture["backend_requested"] = "cpu-reference"
    capture["backend_selected"] = "cpu-reference"
    capture["selected_kernel"] = CPU_KERNEL
    metadata = capture["backend_metadata"]
    metadata.update(
        {
            "source": "rns8_get_plan_backend_info",
            "selected_kernel": CPU_KERNEL,
            "accelerator_backend": False,
            "correctness_backend": True,
            "matrix_engine_backend": False,
            "compiled_kernel_available": True,
            "exact_differential_validated": True,
            "performance_validated": False,
            "accelerator_library": None,
            "accelerator_version": None,
            "capability_status": "implemented_correctness_backend",
            "epilogue_mode": "host_reference_reconstruction",
            "workspace_mode": "host_reference_workspace",
            "workspace_required_bytes": 0,
            "isa_evidence": "not_applicable_cpu",
        }
    )
    apply_int32_accumulator_contract(capture)
    set_tile_variant(capture, "cpu-reference-default", capture["tile_m"], capture["tile_n"], CPU_KERNEL)
    metadata["autotune_key"] = with_accumulator_key_fields(
        "backend=cpu-reference;"
        f"semantics={capture['semantics']};m={capture['m']};n={capture['n']};k={capture['k']};"
        f"prefix={capture['prefix']};tile_m={capture['tile_m']};tile_n={capture['tile_n']};"
        "groups=1;adaptive_prefix=0;adaptive_skip=0;"
        f"kernel={CPU_KERNEL};epilogue=host_reference_reconstruction",
        capture,
    )
    set_no_gpu_events(capture)
    set_host_timings(capture, {"end_to_end": end_to_end_us})
    return capture


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        cpu_path = tmp / "cpu.json"
        default_path = tmp / "direct-default.json"
        candidate_path = tmp / "direct-64x64.json"
        write_capture(cpu_path, make_cpu_capture(200))
        write_capture(default_path, make_direct_capture(128, 128, "direct-hip-default-128x128", 100))
        write_capture(candidate_path, make_direct_capture(64, 64, "direct-hip-bounded-fixture-64x64", 80))

        manifest_path = tmp / "resource-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "captures": [
                        {
                            "capture": candidate_path.name,
                            "resource_summary": {
                                "vgpr": 40,
                                "sgpr": 36,
                                "lds_instruction_mentions": 2,
                                "occupancy": 0.625,
                            },
                            "evidence_status": {
                                "profiler_counter_status": "missing",
                                "isa_resource_status": "present",
                                "missing_evidence": ["missing_profiler_counter_export"],
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        report = tile_shape_report.build_report(
            [cpu_path, default_path, candidate_path],
            tile_shape_report.load_resource_manifest(manifest_path),
        )
        assert report["schema"] == "rns8_tile_shape_report_v2"
        assert report["candidate_count"] == 1
        assert report["cpu_anchor_count"] == 1
        assert report["default_direct_hip_anchor_count"] == 1
        row = report["rows"][0]
        assert row["decision"] == "promote locally"
        assert row["promotion_eligible"] is True
        assert row["speedup_vs_default_direct_hip_tile"] == 1.25
        assert row["promotion_blockers"] == []
        assert row["resource_status"]["resource_evidence_present"] is True
        assert row["resource_status"]["complete_for_promotion"] is True

        no_resource = tile_shape_report.build_report([cpu_path, default_path, candidate_path])
        no_resource_row = no_resource["rows"][0]
        assert no_resource_row["decision"] == "keep experimental"
        assert "missing_counter_or_isa_resource_evidence" in no_resource_row["promotion_blockers"]

        losing_path = tmp / "direct-64x64-loser.json"
        write_capture(losing_path, make_direct_capture(64, 64, "direct-hip-bounded-fixture-64x64", 130))
        losing_report = tile_shape_report.build_report([cpu_path, default_path, losing_path])
        losing_row = losing_report["rows"][0]
        assert losing_row["decision"] == "drop/deprioritize"
        assert losing_row["speedup_vs_default_direct_hip_tile"] < 0.98
        assert "missing_occupancy_signal" in losing_row["promotion_blockers"]

        stale_identity = copy.deepcopy(make_direct_capture(64, 64, "direct-hip-bounded-fixture-64x64", 80))
        stale_identity["backend_metadata"]["autotune_key"] = stale_identity["backend_metadata"]["autotune_key"].replace(
            ";tile_m=64;", ";tile_m=128;"
        )
        stale_path = tmp / "direct-stale-identity.json"
        write_capture(stale_path, stale_identity)
        stale_report = tile_shape_report.build_report(
            [cpu_path, default_path, stale_path],
            tile_shape_report.load_resource_manifest(manifest_path),
        )
        stale_row = stale_report["rows"][0]
        assert stale_row["decision"] == "keep experimental"
        assert "autotune_key_missing_tile_identity" in stale_row["promotion_blockers"]

        output_paths = tile_shape_report.write_outputs(report, tmp / "out")
        assert Path(output_paths["json"]).exists()
        assert Path(output_paths["markdown"]).exists()

    print("tile shape report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
