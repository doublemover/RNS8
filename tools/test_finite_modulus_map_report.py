#!/usr/bin/env python3
"""Self-test finite modulus-family map report grouping and gates."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import finite_modulus_map_report
from test_benchmark_sweep_support import finite_capture


def write_capture(path: Path, capture: dict) -> None:
    path.write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")


def finite_map_capture(backend: str, end_to_end: int, *, shape: int = 64, modulus: int = 255) -> dict:
    capture = finite_capture(backend, end_to_end)
    old_shape = f"m={capture['m']};n={capture['n']};k={capture['k']}"
    new_shape = f"m={shape};n={shape};k={shape}"
    capture["m"] = shape
    capture["n"] = shape
    capture["k"] = shape
    capture["finite_modulus"] = modulus
    metadata = capture["backend_metadata"]
    metadata["autotune_key"] = metadata["autotune_key"].replace(old_shape, new_shape)
    metadata["autotune_key"] = metadata["autotune_key"].replace("finite_modulus=255", f"finite_modulus={modulus}")
    capture["command_line"] = (
        f"rns8-bench --backend {backend} --semantics finite-u8-ring --modulus {modulus} "
        f"--m {shape} --n {shape} --k {shape} --warmups 3 --repeats 9 --seed 13"
    )
    if backend == "hip-direct" and modulus == 255:
        kernel = "direct_hip_tiled_finite_u8_gemm_mod255_v1"
        metadata["selected_kernel"] = kernel
        capture["selected_kernel"] = kernel
        metadata["isa_evidence"] = "rns8_hip_direct_finite_specialized_reducer_isa_gate_no_divide"
        metadata["autotune_key"] = metadata["autotune_key"].replace(
            "kernel=direct_hip_tiled_finite_u8_gemm_v1",
            f"kernel={kernel}",
        )
        capture["timing_metadata"]["gpu_event_timing_source_scope"] = (
            "direct_hip_default_stream_backend_operation_groups"
        )
    return capture


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        cpu_path = tmp / "cpu.json"
        direct_path = tmp / "direct.json"
        ck_path = tmp / "ck.json"
        write_capture(cpu_path, finite_map_capture("cpu-reference", 300))
        write_capture(direct_path, finite_map_capture("hip-direct", 200))
        write_capture(ck_path, finite_map_capture("ck", 100))

        report = finite_modulus_map_report.build_report(
            [tmp],
            expected_shapes=[64],
            expected_ring_moduli=[255],
            expected_field_moduli=[],
            expected_backends=["cpu-reference", "hip-direct", "ck"],
        )
        assert report["schema"] == "rns8_finite_modulus_map_report_v1"
        assert report["summary"]["map_complete"] is True
        assert report["summary"]["ready_groups"] == 1
        group = report["groups"][0]
        assert group["map_group_ready"] is True
        assert group["key"]["modulus_class"] == "composite"
        assert group["key"]["modulus_role"] == "hot_composite"
        assert group["winner"]["backend"] == "ck"
        assert group["winner"]["speedup_vs_direct_hip"] == 2.0
        assert group["promotion_eligible"] is False
        assert group["promotion_blockers"] == ["non_promoting_modulus_map"]

        missing_backend = finite_modulus_map_report.build_report(
            [cpu_path, direct_path],
            expected_shapes=[64],
            expected_ring_moduli=[255],
            expected_field_moduli=[],
            expected_backends=["cpu-reference", "hip-direct", "ck"],
        )
        missing_group = missing_backend["groups"][0]
        assert missing_backend["summary"]["map_complete"] is False
        assert missing_group["missing_backends"] == ["ck"]

        not_release_capture = finite_map_capture("ck", 100)
        not_release_capture["warmups"] = 1
        not_release_path = tmp / "ck-not-release.json"
        write_capture(not_release_path, not_release_capture)
        not_release = finite_modulus_map_report.build_report(
            [cpu_path, direct_path, not_release_path],
            expected_shapes=[64],
            expected_ring_moduli=[255],
            expected_field_moduli=[],
            expected_backends=["cpu-reference", "hip-direct", "ck"],
        )
        assert not_release["summary"]["map_complete"] is False
        assert not_release["groups"][0]["release_not_ready"] == ["ck"]

        generic_prime = copy.deepcopy(group)
        assert finite_modulus_map_report._modulus_class(127) == "prime"
        assert finite_modulus_map_report._modulus_role("finite_ring_u8", 127) == "generic_non_hot_prime"
        assert finite_modulus_map_report._modulus_class(256) == "power_of_two"
        assert finite_modulus_map_report._modulus_role("finite_ring_u8", 256) == "hot_power_of_two"
        assert generic_prime["promotion_eligible"] is False

    print("finite modulus map report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
