#!/usr/bin/env python3
"""Smoke-test benchmark-only HIP vector-ALU baselines against CPU captures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_capture(bench: Path, output: Path, *args: str) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(bench), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{bench} {' '.join(args)} failed with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    output.write_text(completed.stdout, encoding="utf-8")
    return json.loads(completed.stdout)


def validate(schema: Path, capture: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(schema), str(capture)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"schema validation failed for {capture}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def assert_same_checksum(cpu: dict, vector: dict, semantics: str) -> None:
    for key in ["semantics", "m", "n", "k", "seed", "input_distribution"]:
        if cpu.get(key) != vector.get(key):
            raise SystemExit(f"{semantics}: contract key {key} differs: {cpu.get(key)!r} != {vector.get(key)!r}")
    if vector.get("backend_selected") != "hip-vector-alu-int64":
        raise SystemExit(f"{semantics}: vector capture selected {vector.get('backend_selected')!r}")
    if cpu.get("checksum_u64") != vector.get("checksum_u64"):
        raise SystemExit(
            f"{semantics}: checksum mismatch CPU={cpu.get('checksum_u64')} vector={vector.get('checksum_u64')}"
        )


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_vector_alu_baseline.py RNS8_BENCH BENCHMARK_SCHEMA OUT_DIR")
    bench = Path(sys.argv[1])
    schema = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])

    common = ["--m", "8", "--n", "8", "--k", "8", "--warmups", "1", "--repeats", "1"]
    cases = [
        ("bounded-i64", "13"),
        ("bounded-u64", "17"),
    ]
    for semantics, seed in cases:
        cpu_path = out_dir / f"{semantics}-cpu.json"
        vector_path = out_dir / f"{semantics}-vector.json"
        cpu = run_capture(
            bench,
            cpu_path,
            "--backend",
            "cpu",
            "--semantics",
            semantics,
            "--seed",
            seed,
            *common,
        )
        vector = run_capture(
            bench,
            vector_path,
            "--backend",
            "hip-vector-alu-int64-baseline",
            "--semantics",
            semantics,
            "--seed",
            seed,
            *common,
        )
        validate(schema, vector_path)
        assert_same_checksum(cpu, vector, semantics)

    print("vector-ALU baseline smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
