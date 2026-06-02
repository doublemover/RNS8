#!/usr/bin/env python3
"""Self-test benchmark sweep review and promotion helpers."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import benchmark_sweep
from benchmark_schema import load_capture, validate_capture


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def set_phase(capture: dict, end_to_end: int) -> None:
    capture["raw_timings_us"]["end_to_end"] = [end_to_end - 10, end_to_end]
    capture["timing_summary_us"]["end_to_end"] = {
        "avg": float(end_to_end - 5),
        "median": float(end_to_end),
        "p95": float(end_to_end),
    }


def finite_capture(backend: str, end_to_end: int) -> dict:
    capture = copy.deepcopy(load_capture(FIXTURE_DIR / "v4_finite_ring_u8_ck.json"))
    capture["_path"] = f"{backend}.json"
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    metadata = capture["backend_metadata"]
    metadata["source"] = "rns8_get_plan_backend_info"
    if backend == "cpu-reference":
        capture["selected_kernel"] = "cpu_reference_finite_u8_gemm_v1"
        metadata["selected_kernel"] = "cpu_reference_finite_u8_gemm_v1"
        metadata["accelerator_backend"] = False
        metadata["matrix_engine_backend"] = False
        metadata["accelerator_library"] = None
        metadata["capability_status"] = "implemented_correctness_backend"
        metadata["epilogue_mode"] = "fused_centered_residue_then_canonical_u8_export"
        metadata["workspace_mode"] = "host_reference_workspace"
        metadata["workspace_required_bytes"] = 0
        metadata["isa_evidence"] = "not_applicable_cpu"
        metadata["autotune_key"] = metadata["autotune_key"].replace("backend=ck", "backend=cpu-reference")
        capture["device"] = {
            "device_id": -1,
            "name": "CPU reference",
            "gcn_arch": "none",
            "hip_available": 0,
            "hip_runtime_version": 0,
            "hip_driver_version": 0,
            "global_mem_bytes": 0,
        }
    elif backend == "hip-direct":
        capture["selected_kernel"] = "direct_hip_tiled_finite_u8_gemm_v1"
        metadata["selected_kernel"] = "direct_hip_tiled_finite_u8_gemm_v1"
        metadata["accelerator_backend"] = False
        metadata["matrix_engine_backend"] = False
        metadata["accelerator_library"] = "HIP runtime"
        metadata["capability_status"] = "implemented_correctness_backend"
        metadata["epilogue_mode"] = "fused_centered_residue_then_canonical_u8_export"
        metadata["workspace_mode"] = "resident_device_buffers"
        metadata["workspace_required_bytes"] = 0
        metadata["isa_evidence"] = "rns8_hip_direct_reciprocal_isa_gate"
        metadata["autotune_key"] = metadata["autotune_key"].replace("backend=ck", "backend=hip-direct")
    set_phase(capture, end_to_end)
    return capture


def main() -> int:
    parsed = benchmark_sweep.parse_case("rect:64,128,256")
    assert parsed.name == "rect"
    assert (parsed.m, parsed.n, parsed.k) == (64, 128, 256)
    adaptive = benchmark_sweep.parse_case("adaptive:65,65,64,64,64", adaptive=True)
    assert adaptive.bound_mode == "per-tile"
    assert adaptive.require_adaptive is True
    assert benchmark_sweep.backend_allowed_for("wrap-u64", parsed, "wrap64-byte-limb") is True
    assert benchmark_sweep.backend_allowed_for("bounded-i64", parsed, "wrap64-byte-limb") is False
    assert benchmark_sweep.backend_allowed_for("finite-u8-ring", parsed, "hip-vector-alu-int64") is False
    assert benchmark_sweep.backend_allowed_for("bounded-u64", adaptive, "hipblaslt") is False

    ck = finite_capture("ck", 190)
    direct = finite_capture("hip-direct", 300)
    cpu = finite_capture("cpu-reference", 500)
    report = benchmark_sweep.review_captures([ck, direct, cpu])
    assert report["group_count"] == 1
    assert len(report["promotable_autotune_entries"]) == 1
    assert report["promotable_autotune_entries"][0]["selected_backend"] == "ck"
    group = report["groups"][0]
    assert group["missing_required_baselines"] == []
    assert group["fastest_promotable"]["backend"] == "ck"

    blocked = benchmark_sweep.review_captures([ck])
    assert blocked["promotable_autotune_entries"] == []
    assert blocked["groups"][0]["missing_required_baselines"] == ["cpu-reference", "hip-direct"]

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "autotune.json"
        promoted = benchmark_sweep.write_promoted_cache_entries(report, [ck, direct, cpu], cache_path)
        assert promoted == 1
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cache["entries"][0]["performance_validated"] is True
        assert cache["entries"][0]["validation_status"] == "reviewed_same_contract_fastest_windows_gfx1100"

    validate_capture(load_capture(FIXTURE_DIR / "v4_finite_ring_u8_ck.json"))
    validate_capture(load_capture(FIXTURE_DIR / "v4_finite_field_u8_rocwmma.json"))
    print("benchmark sweep self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
