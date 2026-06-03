#!/usr/bin/env python3
"""Smoke-test rns8-bench --backend auto fallback capture metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run_capture(bench: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RNS8_AUTOTUNE_CACHE_PATH"] = str(output.parent / "empty-autotune-cache.json")
    completed = subprocess.run(
        [
            str(bench),
            "--backend",
            "auto",
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
            "1",
            "--seed",
            "23",
        ],
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{bench} --backend auto failed with {completed.returncode}\n"
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


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_benchmark_auto_backend.py RNS8_BENCH BENCHMARK_SCHEMA OUT_DIR")
    bench = Path(sys.argv[1])
    schema = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    capture_path = out_dir / "bounded-i64-auto.json"

    capture = run_capture(bench, capture_path)
    validate(schema, capture_path)
    if capture.get("backend_requested") != "auto":
        raise SystemExit(f"expected backend_requested=auto, got {capture.get('backend_requested')!r}")
    if capture.get("backend_selected") != "hip-direct":
        raise SystemExit(f"expected AUTO fallback to select hip-direct, got {capture.get('backend_selected')!r}")
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    if metadata.get("performance_validated") is not False:
        raise SystemExit("empty-cache AUTO fallback must not report performance_validated=true")
    timing = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    if timing.get("gpu_event_timing") is not True:
        raise SystemExit("AUTO-selected hip-direct fallback must report selected-backend GPU event timing")
    if timing.get("gpu_event_timing_reason") != "captured_by_direct_hip_backend_hooks":
        raise SystemExit(
            "AUTO-selected hip-direct fallback reported unexpected GPU event timing reason "
            f"{timing.get('gpu_event_timing_reason')!r}"
        )
    print("benchmark AUTO backend smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
