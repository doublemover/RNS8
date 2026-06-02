#!/usr/bin/env python3
"""Self-test rns8-inspect hard-cut CLI diagnostics."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_command(exe: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(exe), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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

    for backend in ["hipblaslt", "ck", "rocwmma"]:
        accelerator = run_command(inspect_exe, "--backend", backend)
        expect_exit(accelerator, 1, backend)
        expect_text(accelerator.stderr, "unsupported backend", backend)
        expect_text(accelerator.stderr, "requested accelerator is evidence-only", backend)
        expect_text(accelerator.stderr, "real exact correctness backend", backend)

    cpu = run_command(inspect_exe, "--backend", "cpu-reference", "--json")
    expect_exit(cpu, 0, "cpu-reference json")
    expect_text(cpu.stdout, '"backend": "cpu-reference"', "cpu-reference json")
    expect_text(cpu.stdout, '"hip_available": 0', "cpu-reference json")

    print("rns8-inspect CLI self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
