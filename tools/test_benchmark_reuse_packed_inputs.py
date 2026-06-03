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
    capture_path = out_dir / "bounded-i64-cpu-reuse-packed.json"

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
        "--reuse-packed-inputs",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    capture_path.write_text(completed.stdout, encoding="utf-8")
    subprocess.run([sys.executable, str(schema), str(capture_path)], check=True)

    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert capture["reuse_packed_inputs"] is True
    assert capture["pack_mode"] == "prepacked_reuse"
    assert capture["timing_metadata"]["pack_mode"] == "prepacked_reuse"
    assert isinstance(capture["prepack_setup_us"], int)
    assert capture["prepack_setup_us"] >= 0
    assert capture["avg_prepack_setup_us"] == float(capture["prepack_setup_us"])
    assert capture["raw_timings_us"]["pack"] == [0, 0]
    assert capture["avg_pack_us"] == 0.0
    assert capture["timing_metadata"]["phase_availability"]["prepack_setup"]["timed"] is True
    assert capture["timing_metadata"]["phase_availability"]["prepack_setup"]["timing_key"] == "prepack_setup_us"
    assert capture["gpu_event_timings_us"] is None
    print("benchmark reuse-packed-inputs smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
