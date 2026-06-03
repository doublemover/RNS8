#!/usr/bin/env python3
"""Self-test result comparison contract and GPU compatibility boundaries."""

from __future__ import annotations

import copy
from pathlib import Path

import result_compare
from benchmark_schema import load_capture


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def cpu_reference_from(capture: dict) -> dict:
    cpu = copy.deepcopy(capture)
    cpu["backend_requested"] = "cpu-reference"
    cpu["backend_selected"] = "cpu-reference"
    cpu["selected_kernel"] = "cpu_reference_finite_u8_gemm_v1"
    metadata = cpu["backend_metadata"]
    metadata["selected_kernel"] = "cpu_reference_finite_u8_gemm_v1"
    metadata["accelerator_backend"] = False
    metadata["correctness_backend"] = True
    metadata["matrix_engine_backend"] = False
    metadata["compiled_kernel_available"] = True
    metadata["accelerator_library"] = None
    metadata["accelerator_version"] = None
    metadata["capability_status"] = "implemented_correctness_backend"
    metadata["workspace_mode"] = "host_reference_workspace"
    metadata["workspace_required_bytes"] = 0
    metadata["isa_evidence"] = "not_applicable_cpu"
    metadata["autotune_key"] = metadata["autotune_key"].replace("backend=ck", "backend=cpu-reference")
    cpu["device"] = {
        "device_id": -1,
        "name": "CPU reference",
        "gcn_arch": "none",
        "hip_available": 0,
        "hip_runtime_version": 0,
        "hip_driver_version": 0,
        "global_mem_bytes": 0,
    }
    cpu["hip_toolchain"] = {
        "enabled": False,
        "hip_root": None,
        "hipcc_path": None,
        "hipcc_version": None,
        "hip_sdk_or_rocm_version": None,
    }
    return cpu


def main() -> int:
    gpu = load_capture(FIXTURE_DIR / "v4_finite_ring_u8_ck.json")
    cpu = cpu_reference_from(gpu)

    cpu_vs_gpu = result_compare.compare(cpu, gpu, Path("cpu.json"), Path("gpu.json"))
    assert cpu_vs_gpu["matching_contract"] is True
    assert cpu_vs_gpu["gpu_compatibility_required"] is False
    assert cpu_vs_gpu["gpu_compatible"] is True
    assert cpu_vs_gpu["backend_evidence"]["backend_selected"]["match"] is False

    mismatched_gpu = copy.deepcopy(gpu)
    mismatched_gpu["device"]["gcn_arch"] = "gfx1200"
    gpu_vs_gpu = result_compare.compare(gpu, mismatched_gpu, Path("gpu-a.json"), Path("gpu-b.json"))
    assert gpu_vs_gpu["matching_contract"] is True
    assert gpu_vs_gpu["gpu_compatibility_required"] is True
    assert gpu_vs_gpu["gpu_compatible"] is False
    assert gpu_vs_gpu["gpu_compatibility"]["device.gcn_arch"]["match"] is False

    reused_gpu = copy.deepcopy(gpu)
    reused_gpu["reuse_packed_inputs"] = True
    reused_gpu["pack_mode"] = "prepacked_reuse"
    reused_gpu["timing_metadata"]["pack_mode"] = "prepacked_reuse"
    repack_vs_reuse = result_compare.compare(gpu, reused_gpu, Path("gpu-repack.json"), Path("gpu-reuse.json"))
    assert repack_vs_reuse["matching_contract"] is False
    assert repack_vs_reuse["contract"]["reuse_packed_inputs"]["match"] is False
    assert repack_vs_reuse["contract"]["pack_mode"]["match"] is False

    print("result compare self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
