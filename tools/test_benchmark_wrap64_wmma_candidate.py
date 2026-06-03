#!/usr/bin/env python3
"""Smoke-test the internal rocWMMA wrap64 benchmark candidate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmark_schema import load_capture, validate_capture


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: test_benchmark_wrap64_wmma_candidate.py RNS8_BENCH BENCHMARK_SCHEMA OUT_DIR",
            file=sys.stderr,
        )
        return 2

    bench = Path(sys.argv[1])
    schema = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_path = out_dir / "wrap64-wmma-candidate-smoke.json"

    command = [
        str(bench),
        "--backend",
        "rocwmma-wrap64-candidate",
        "--semantics",
        "wrap-u64",
        "--m",
        "16",
        "--n",
        "16",
        "--k",
        "16",
        "--warmups",
        "1",
        "--repeats",
        "2",
        "--seed",
        "20260603",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    capture_path.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        (out_dir / "wrap64-wmma-candidate-smoke.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode

    validate_capture(load_capture(capture_path), capture_path)
    schema_result = subprocess.run(
        [sys.executable, str(schema), str(capture_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if schema_result.returncode != 0:
        print(schema_result.stdout, file=sys.stderr, end="")
        print(schema_result.stderr, file=sys.stderr, end="")
        return schema_result.returncode

    capture = load_capture(capture_path)
    assert capture["backend_requested"] == "rocwmma-wrap64-candidate"
    assert capture["backend_selected"] == "wmma"
    assert capture["selected_kernel"] == "rocwmma_wrap64_byte_gemm36_candidate_v0"
    assert capture["backend_metadata"]["correctness_backend"] is False
    assert capture["backend_metadata"]["performance_validated"] is False
    assert capture["timing_metadata"]["gpu_event_timing"] is True
    assert "wrap64_wmma_candidate_gemm36_kernel_group" in capture["gpu_event_timings_us"]
    print("benchmark wrap64 rocWMMA candidate smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
