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

    static_bound_gpu = copy.deepcopy(gpu)
    static_bound_gpu["bound_source"] = "static_profile"
    legacy_vs_static = result_compare.compare(gpu, static_bound_gpu, Path("legacy.json"), Path("static.json"))
    assert legacy_vs_static["matching_contract"] is True
    assert legacy_vs_static["contract"]["bound_source"]["match"] is True

    reused_gpu = copy.deepcopy(gpu)
    reused_gpu["reuse_packed_inputs"] = True
    reused_gpu["pack_mode"] = "prepacked_reuse"
    reused_gpu["prepack_reuse_operands"] = ["A", "B"]
    reused_gpu["timing_metadata"]["pack_mode"] = "prepacked_reuse"
    reused_gpu["timing_metadata"]["prepack_reuse_operands"] = ["A", "B"]
    repack_vs_reuse = result_compare.compare(gpu, reused_gpu, Path("gpu-repack.json"), Path("gpu-reuse.json"))
    assert repack_vs_reuse["matching_contract"] is False
    assert repack_vs_reuse["contract"]["reuse_packed_inputs"]["match"] is False
    assert repack_vs_reuse["contract"]["pack_mode"]["match"] is False
    assert repack_vs_reuse["contract"]["prepack_reuse_operands"]["match"] is False

    selector_default = copy.deepcopy(gpu)
    selector_default["export_variant"] = {
        "name": "default",
        "semantic_contract": "finite_ring_u8",
        "signedness": "unsigned",
        "output_layout": "finite_u8_host",
        "selector_status_policy": "required",
        "d2h_policy": "host_ld_padded",
        "final_output_mode": "final_host_output",
        "selector_key": "semantics=finite_ring_u8;d2h_policy=host_ld_padded",
    }
    selector_default["reconstruction_variant"] = {"name": "default_garner"}
    selector_candidate = copy.deepcopy(selector_default)
    selector_candidate["export_variant"]["name"] = "compact-d2h-export-candidate"
    selector_candidate["export_variant"]["d2h_policy"] = "compact_contiguous"
    selector_candidate["export_variant"]["selector_key"] = (
        selector_candidate["export_variant"]["selector_key"] + ";d2h_policy=compact_contiguous"
    )
    strict_selector_compare = result_compare.compare(
        selector_default,
        selector_candidate,
        Path("selector-default.json"),
        Path("selector-candidate.json"),
    )
    assert strict_selector_compare["matching_contract"] is False
    assert strict_selector_compare["contract"]["export_variant.d2h_policy"]["match"] is False
    assert strict_selector_compare["contract"]["export_variant.d2h_policy"]["ignored_for_contract"] is False

    review_selector_compare = result_compare.compare(
        selector_default,
        selector_candidate,
        Path("selector-default.json"),
        Path("selector-candidate.json"),
        allow_export_selector_diff=True,
    )
    assert review_selector_compare["matching_contract"] is True
    assert review_selector_compare["comparison_mode"] == "export_selector_review"
    assert review_selector_compare["contract"]["export_variant.d2h_policy"]["match"] is False
    assert review_selector_compare["contract"]["export_variant.d2h_policy"]["ignored_for_contract"] is True

    invalid_selector_candidate = copy.deepcopy(selector_candidate)
    invalid_selector_candidate["export_variant"]["output_layout"] = "host_scalar_i64"
    invalid_selector_compare = result_compare.compare(
        selector_default,
        invalid_selector_candidate,
        Path("selector-default.json"),
        Path("selector-invalid.json"),
        allow_export_selector_diff=True,
    )
    assert invalid_selector_compare["matching_contract"] is False
    assert invalid_selector_compare["contract"]["export_variant.output_layout"]["ignored_for_contract"] is False

    layout_a = copy.deepcopy(gpu)
    layout_b = copy.deepcopy(gpu)
    layout_a["timing_metadata"]["pack_layout"] = "resident_rns_residue_planes"
    layout_b["timing_metadata"]["pack_layout"] = "matrix_engine_transient_pack_layout"
    layout_compare = result_compare.compare(layout_a, layout_b, Path("layout-a.json"), Path("layout-b.json"))
    assert layout_compare["matching_contract"] is True
    assert layout_compare["backend_evidence"]["timing_metadata.pack_layout"]["match"] is False

    native_layout = copy.deepcopy(gpu)
    native_layout["packed_layout_version"] = "native_i64_rowmajor_v1"
    native_layout["requested_next_op"] = {
        "requested": "native-gemm",
        "resolved": "native-gemm",
        "source": "benchmark_default",
    }
    native_layout["plan_packing"] = {
        "source": "rns8_get_plan_packing_info",
        "backend": "hip-vector-alu-int64",
        "semantics": "finite_ring_u8",
        "input_domain_name": "native_i64_u64_current",
        "output_domain_name": "native_i64_u64_current",
        "next_op_hint": "native-gemm",
        "input_domain": 2,
        "output_domain": 2,
        "next_op_flags": 4,
        "uses_transient_pack_workspace": False,
        "uses_matrix_engine_pack_layout": False,
        "residue_group_width": 1,
        "input_channel_count": 1,
        "output_channel_count": 1,
    }
    native_compare = result_compare.compare(gpu, native_layout, Path("gpu.json"), Path("native.json"))
    assert native_compare["matching_contract"] is True
    assert native_compare["backend_evidence"]["packed_layout_version"]["match"] is False
    assert native_compare["backend_evidence"]["requested_next_op.resolved"]["match"] is False
    assert native_compare["backend_evidence"]["plan_packing.input_domain_name"]["match"] is False

    target_a = copy.deepcopy(gpu)
    target_b = copy.deepcopy(gpu)
    target_a["target_variant"] = {
        "target_id": "gfx1100",
        "target_namespace": "gfx1100",
        "review_group_key": "gfx1100/target=gfx1100/backend=ck",
    }
    target_b["target_variant"] = {
        "target_id": "gfx1100",
        "target_namespace": "gfx11xx",
        "review_group_key": "gfx11xx/target=gfx1100/backend=ck",
    }
    target_compare = result_compare.compare(target_a, target_b, Path("target-a.json"), Path("target-b.json"))
    assert target_compare["gpu_compatibility_required"] is True
    assert target_compare["gpu_compatible"] is False
    assert target_compare["gpu_compatibility"]["target_variant.target_namespace"]["match"] is False

    print("result compare self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
