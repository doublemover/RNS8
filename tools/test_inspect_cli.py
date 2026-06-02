#!/usr/bin/env python3
"""Self-test rns8-inspect hard-cut CLI diagnostics."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def run_command(exe: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(exe), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def expect_exit(result: subprocess.CompletedProcess[str], code: int, label: str) -> None:
    if result.returncode != code:
        raise AssertionError(
            f"{label}: expected exit {code}, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def expect_text(haystack: str, needle: str, label: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"{label}: expected {needle!r} in:\n{haystack}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inspect_exe", type=Path, help="path to rns8-inspect executable")
    args = parser.parse_args()
    inspect_exe = args.inspect_exe

    invalid = run_command(inspect_exe, "--backend", "not-a-backend")
    expect_exit(invalid, 2, "invalid backend")
    expect_text(invalid.stderr, "invalid backend string: not-a-backend", "invalid backend")
    expect_text(invalid.stderr, "unknown names are not routed to auto", "invalid backend")

    hipblaslt = run_command(inspect_exe, "--backend", "hipblaslt")
    if hipblaslt.returncode == 0:
        expect_text(hipblaslt.stdout, "capability_status: implemented_baseline_backend", "hipblaslt")
        expect_text(hipblaslt.stdout, "selected_kernel:   hipblaslt_int8_i32_scratch_reduce_baseline_v1", "hipblaslt")
        expect_text(hipblaslt.stdout, "exact_validated:   1", "hipblaslt")
        expect_text(hipblaslt.stdout, "perf_validated:    0", "hipblaslt")
    else:
        expect_exit(hipblaslt, 1, "hipblaslt")
        expect_text(hipblaslt.stderr, "unsupported backend", "hipblaslt")
        expect_text(hipblaslt.stderr, "requested accelerator is evidence-only", "hipblaslt")
        expect_text(hipblaslt.stderr, "real exact correctness backend", "hipblaslt")

    ck = run_command(inspect_exe, "--backend", "ck")
    if ck.returncode == 0:
        expect_text(ck.stdout, "capability_status: implemented_opt_in_ck_backend", "ck")
        expect_text(ck.stdout, "selected_kernel:   ck_wmma_cshuffle_i8_i32_centered_epilogue_v1", "ck")
        expect_text(ck.stdout, "exact_validated:   1", "ck")
        expect_text(ck.stdout, "perf_validated:    0", "ck")
        expect_text(ck.stdout, "isa_evidence:      ck_wmma_cshuffle_int8_matrix_isa_gate_no_int32_global_store", "ck")
    else:
        expect_exit(ck, 1, "ck")
        expect_text(ck.stderr, "unsupported backend", "ck")
        expect_text(ck.stderr, "requested accelerator is evidence-only", "ck")
        expect_text(ck.stderr, "real exact correctness backend", "ck")

    rocwmma = run_command(inspect_exe, "--backend", "rocwmma")
    expect_exit(rocwmma, 1, "rocwmma")
    expect_text(rocwmma.stderr, "unsupported backend", "rocwmma")
    expect_text(rocwmma.stderr, "requested accelerator is evidence-only", "rocwmma")
    expect_text(rocwmma.stderr, "real exact correctness backend", "rocwmma")

    cpu = run_command(inspect_exe, "--backend", "cpu-reference", "--json")
    expect_exit(cpu, 0, "cpu-reference json")
    expect_text(cpu.stdout, '"backend": "cpu-reference"', "cpu-reference json")
    expect_text(cpu.stdout, '"hip_available": 0', "cpu-reference json")

    with tempfile.TemporaryDirectory() as temp_dir:
        env = os.environ.copy()
        env["RNS8_AUTOTUNE_CACHE_PATH"] = str(Path(temp_dir) / "autotune.json")
        autotune = run_command(
            inspect_exe,
            "--backend",
            "cpu-reference",
            "--json",
            "--autotune-key",
            "unit-test-missing-autotune-key",
            env=env,
        )
        expect_exit(autotune, 0, "autotune json")
        expect_text(autotune.stdout, '"autotune_cache": {', "autotune json")
        expect_text(autotune.stdout, '"exact_hit": false', "autotune json")
        expect_text(autotune.stdout, "missing_cache_using_cpu_reference", "autotune json")

    print("rns8-inspect CLI self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
