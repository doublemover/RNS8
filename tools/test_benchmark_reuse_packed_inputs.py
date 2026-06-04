#!/usr/bin/env python3
"""Smoke-test rns8-bench repeated packed-input capture mode."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_benchmark_reuse_packed_inputs.py BENCH SCHEMA OUT_DIR")
    bench = Path(sys.argv[1])
    schema = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("reuse-packed", "--reuse-packed-inputs", "prepacked_reuse", ["A", "B"], True),
        ("reuse-packed-a", "--reuse-packed-a", "prepacked_reuse_a", ["A"], False),
        ("reuse-packed-b", "--reuse-packed-b", "prepacked_reuse_b", ["B"], False),
    ]

    for label, flag, expected_mode, expected_operands, expect_zero_pack in cases:
        capture_path = out_dir / f"bounded-i64-cpu-{label}.json"
        command = [
            str(bench),
            "--backend",
            "cpu",
            "--semantics",
            "bounded-i64",
            "--m",
            "8",
            "--n",
            "8",
            "--k",
            "8",
            "--warmups",
            "1",
            "--repeats",
            "2",
            "--seed",
            "17",
            flag,
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        capture_path.write_text(completed.stdout, encoding="utf-8")
        subprocess.run([sys.executable, str(schema), str(capture_path)], check=True)

        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        assert capture["reuse_packed_inputs"] is True
        assert capture["pack_mode"] == expected_mode
        assert capture["prepack_reuse_operands"] == expected_operands
        assert capture["prepack_reuse_strategy"] == "persistent_matrix_residency"
        assert capture["timing_metadata"]["pack_mode"] == expected_mode
        assert capture["timing_metadata"]["prepack_reuse_operands"] == expected_operands
        assert capture["timing_metadata"]["prepack_reuse_strategy"] == "persistent_matrix_residency"
        assert isinstance(capture["prepack_setup_us"], int)
        assert capture["prepack_setup_us"] >= 0
        assert capture["avg_prepack_setup_us"] == float(capture["prepack_setup_us"])
        assert len(capture["raw_timings_us"]["pack"]) == 2
        if expect_zero_pack:
            assert capture["raw_timings_us"]["pack"] == [0, 0]
            assert capture["avg_pack_us"] == 0.0
        else:
            assert capture["avg_pack_us"] >= 0.0
        assert capture["timing_metadata"]["phase_availability"]["prepack_setup"]["timed"] is True
        assert capture["timing_metadata"]["phase_availability"]["prepack_setup"]["timing_key"] == "prepack_setup_us"
        assert capture["gpu_event_timings_us"] is None

    capture_path = out_dir / "bounded-i64-rocwmma-reuse-packed-b.json"
    command = [
        str(bench),
        "--backend",
        "rocwmma",
        "--semantics",
        "bounded-i64",
        "--m",
        "16",
        "--n",
        "16",
        "--k",
        "16",
        "--warmups",
        "1",
        "--repeats",
        "1",
        "--seed",
        "23",
        "--reuse-packed-b",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode == 0:
        capture_path.write_text(completed.stdout, encoding="utf-8")
        subprocess.run([sys.executable, str(schema), str(capture_path)], check=True)
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        assert capture["backend_selected"] == "rocwmma"
        assert capture["pack_mode"] == "prepacked_reuse_b"
        assert capture["prepack_reuse_operands"] == ["B"]
        assert capture["prepack_reuse_strategy"] == "rocwmma_reusable_b_cache"
        assert capture["timing_metadata"]["prepack_reuse_strategy"] == "rocwmma_reusable_b_cache"
        assert capture["timing_metadata"]["gpu_event_timing"] is True
        assert "rns_gemm_prepacked_b_kernel_group" in capture["timing_metadata"]["gpu_event_phase_order"]
        assert "rns_gemm_prepacked_b_kernel_group" in capture["gpu_event_timings_us"]
    elif "unsupported backend" not in (completed.stderr + completed.stdout).lower():
        raise SystemExit(
            "rocwmma reuse-packed-b smoke failed unexpectedly\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    capture_path = out_dir / "bounded-i64-hipblaslt-reuse-packed-inputs.json"
    command = [
        str(bench),
        "--backend",
        "hipblaslt",
        "--semantics",
        "bounded-i64",
        "--m",
        "16",
        "--n",
        "16",
        "--k",
        "16",
        "--warmups",
        "1",
        "--repeats",
        "1",
        "--seed",
        "31",
        "--reuse-packed-inputs",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode == 0:
        capture_path.write_text(completed.stdout, encoding="utf-8")
        subprocess.run([sys.executable, str(schema), str(capture_path)], check=True)
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        assert capture["backend_selected"] == "hipblaslt"
        assert capture["pack_mode"] == "prepacked_reuse"
        assert capture["prepack_reuse_operands"] == ["A", "B"]
        assert capture["prepack_reuse_strategy"] == "persistent_matrix_residency"
        assert capture["timing_metadata"]["gpu_event_timing"] is True
        phase_order = capture["timing_metadata"]["gpu_event_phase_order"]
        assert "hipblaslt_pack_transpose_centered" not in phase_order
        assert "hipblaslt_int8_i32_matmul" in phase_order
        assert "hipblaslt_i32_to_residue_reduce" in phase_order
        assert "rns_gemm" in phase_order
        assert "hipblaslt_pack_transpose_centered" not in capture["gpu_event_timings_us"]
        assert capture["gpu_event_timings_us"]["pack"] == [0.0]
    elif "unsupported backend" not in (completed.stderr + completed.stdout).lower():
        raise SystemExit(
            "hipblaslt reuse-packed-inputs smoke failed unexpectedly\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    print("benchmark reuse-packed-inputs smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
