#!/usr/bin/env python3
"""Smoke-test the internal rocWMMA wrap64 benchmark candidate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmark_schema import load_capture, validate_capture


def run_capture(
    bench: Path,
    schema: Path,
    out_dir: Path,
    *,
    label: str,
    backend: str,
    m: int,
    n: int,
    k: int,
    seed: int,
) -> dict:
    capture_path = out_dir / f"{label}.json"
    command = [
        str(bench),
        "--backend",
        backend,
        "--semantics",
        "wrap-u64",
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--warmups",
        "1",
        "--repeats",
        "2",
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    capture_path.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        (out_dir / f"{label}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        raise SystemExit(completed.returncode)

    capture = load_capture(capture_path)
    validate_capture(capture, capture_path)
    schema_result = subprocess.run(
        [sys.executable, str(schema), str(capture_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if schema_result.returncode != 0:
        print(schema_result.stdout, file=sys.stderr, end="")
        print(schema_result.stderr, file=sys.stderr, end="")
        raise SystemExit(schema_result.returncode)
    return capture


def assert_candidate_capture(capture: dict, *, m: int, n: int, k: int) -> None:
    assert capture["backend_requested"] == "rocwmma-wrap64-candidate"
    assert capture["backend_selected"] == "wmma"
    assert capture["selected_kernel"] == "rocwmma_wrap64_byte_gemm36_candidate_v0"
    assert capture["m"] == m
    assert capture["n"] == n
    assert capture["k"] == k
    assert capture["backend_metadata"]["correctness_backend"] is False
    assert capture["backend_metadata"]["performance_validated"] is False
    assert capture["timing_metadata"]["gpu_event_timing"] is True
    assert "wrap64_wmma_candidate_gemm36_kernel_group" in capture["gpu_event_timings_us"]


def assert_wrap64_release_baselines(captures: list[dict]) -> None:
    cpu, hip, candidate = captures
    assert cpu["backend_requested"] == "wrap64-byte-limb"
    assert cpu["backend_selected"] == "wrap64-byte-limb"
    assert cpu["selected_kernel"] == "cpu_wrap64_byte_limb_reference_v1"
    assert hip["backend_requested"] == "hip-direct"
    assert hip["backend_selected"] == "hip-direct"
    assert hip["selected_kernel"] == "direct_hip_wrap64_byte_gemm36_tiled_2d_v3"
    assert candidate["backend_requested"] == "rocwmma-wrap64-candidate"


def assert_release_shape_checksums_match(captures: list[dict]) -> None:
    checksums = {capture["backend_requested"]: capture["checksum_u64"] for capture in captures}
    expected = checksums["wrap64-byte-limb"]
    for backend, checksum in checksums.items():
        assert checksum == expected, f"{backend} checksum {checksum} did not match CPU wrap64 {expected}"


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

    smoke = run_capture(
        bench,
        schema,
        out_dir,
        label="wrap64-wmma-candidate-smoke",
        backend="rocwmma-wrap64-candidate",
        m=16,
        n=16,
        k=16,
        seed=20260603,
    )
    assert_candidate_capture(smoke, m=16, n=16, k=16)

    release_shape = {"m": 64, "n": 64, "k": 64, "seed": 20260603}
    release_captures = [
        run_capture(
            bench,
            schema,
            out_dir,
            label="wrap64-release-shape-cpu",
            backend="wrap64-byte-limb",
            **release_shape,
        ),
        run_capture(
            bench,
            schema,
            out_dir,
            label="wrap64-release-shape-hip-direct",
            backend="hip-direct",
            **release_shape,
        ),
        run_capture(
            bench,
            schema,
            out_dir,
            label="wrap64-release-shape-wmma-candidate",
            backend="rocwmma-wrap64-candidate",
            **release_shape,
        ),
    ]
    assert_candidate_capture(release_captures[-1], m=64, n=64, k=64)
    assert_wrap64_release_baselines(release_captures)
    assert_release_shape_checksums_match(release_captures)

    print("benchmark wrap64 rocWMMA candidate smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
