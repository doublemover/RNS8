#!/usr/bin/env python3
"""Smoke-test rns8-bench exact-wide signed and unsigned captures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_capture(bench: Path, schema: Path, out_dir: Path, semantics: str) -> dict:
    capture_path = out_dir / f"{semantics}-cpu.json"
    command = [
        str(bench),
        "--backend",
        "cpu",
        "--semantics",
        semantics,
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
        "31",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    capture_path.write_text(completed.stdout, encoding="utf-8")
    subprocess.run([sys.executable, str(schema), str(capture_path)], check=True)
    return json.loads(capture_path.read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_benchmark_exact_wide.py BENCH SCHEMA OUT_DIR")
    bench = Path(sys.argv[1])
    schema = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    signed = run_capture(bench, schema, out_dir, "exact-wide-signed")
    unsigned = run_capture(bench, schema, out_dir, "exact-wide-unsigned")

    assert signed["semantics"] == "exact_wide_signed"
    assert signed["bound_kind"] == "none"
    assert signed["bound"] == 0
    assert signed["prefix"] > 0
    assert signed["epilogue_type"] == "exact_wide_signed_limb_export"
    assert signed["comparison_baseline"]["required_before_speedup_claim"] == [
        "same_contract_cpu_reference",
        "same_contract_direct_hip_correctness",
    ]
    assert signed["gpu_event_timings_us"] is None

    assert unsigned["semantics"] == "exact_wide_unsigned"
    assert unsigned["bound_kind"] == "none"
    assert unsigned["bound"] == 0
    assert unsigned["prefix"] == signed["prefix"]
    assert unsigned["epilogue_type"] == "exact_wide_unsigned_limb_export"
    assert unsigned["finite_modulus"] is None
    assert unsigned["packed_layout_version"] is None
    print("benchmark exact-wide smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
