#!/usr/bin/env python3
"""Smoke-test rns8-bench AUTO selection from the default cache path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import benchmark_sweep


def default_cache_path(env: dict[str, str], root: Path) -> Path:
    if os.name == "nt":
        local = root / "local-app-data"
        env["LOCALAPPDATA"] = str(local)
        env["USERPROFILE"] = str(root / "profile")
        return local / "rns8-gemm" / "autotune.json"
    xdg = root / "xdg-cache"
    env["XDG_CACHE_HOME"] = str(xdg)
    env["HOME"] = str(root / "home")
    return xdg / "rns8-gemm" / "autotune.json"


def safe_name(value: str) -> str:
    return value.replace("_", "-").replace("/", "-")


def run_capture(bench: Path, args: list[str], output: Path, env: dict[str, str]) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(bench), *args],
        check=False,
        env=env,
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


def write_reviewed_cache(path: Path, capture: dict) -> dict:
    entry = benchmark_sweep.cache_entry_from_capture(
        capture,
        "reviewed_release_same_contract_fastest_windows_gfx1100",
    )
    if not entry.get("key"):
        raise SystemExit("explicit backend capture did not report an autotune key")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "entries": [entry]}, indent=2, sort_keys=True) + "\n")
    return entry


def main() -> int:
    if len(sys.argv) not in {5, 6}:
        raise SystemExit(
            "usage: test_benchmark_auto_default_cache.py RNS8_BENCH BENCHMARK_SCHEMA OUT_DIR BACKEND [SEMANTICS]"
        )
    bench = Path(sys.argv[1])
    schema = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    requested_backend = sys.argv[4]
    semantics = sys.argv[5] if len(sys.argv) == 6 else "bounded-i64"

    common = [
        "--semantics",
        semantics,
        "--m",
        "64",
        "--n",
        "128",
        "--k",
        "64",
        "--warmups",
        "1",
        "--repeats",
        "1",
        "--seed",
        "41",
    ]
    if semantics == "finite-u8-ring":
        common.extend(["--modulus", "255"])
    elif semantics == "finite-u8-field":
        common.extend(["--modulus", "251"])

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        explicit_env = os.environ.copy()
        explicit_env["RNS8_AUTOTUNE_CACHE_PATH"] = str(root / "unused-explicit-cache.json")
        explicit_capture_path = out_dir / f"explicit-{requested_backend}-{safe_name(semantics)}.json"
        explicit = run_capture(
            bench,
            ["--backend", requested_backend, *common],
            explicit_capture_path,
            explicit_env,
        )
        validate(schema, explicit_capture_path)
        selected_backend = explicit.get("backend_selected")
        selected_kernel = explicit.get("selected_kernel")
        metadata = explicit.get("backend_metadata") if isinstance(explicit.get("backend_metadata"), dict) else {}
        explicit_key = metadata.get("autotune_key")

        auto_env = os.environ.copy()
        auto_env.pop("RNS8_AUTOTUNE_CACHE_PATH", None)
        cache_path = default_cache_path(auto_env, root)
        entry = write_reviewed_cache(cache_path, explicit)

        auto_capture_path = out_dir / f"default-cache-auto-{requested_backend}-{safe_name(semantics)}.json"
        auto = run_capture(bench, ["--backend", "auto", *common], auto_capture_path, auto_env)
        validate(schema, auto_capture_path)

    if auto.get("backend_requested") != "auto":
        raise SystemExit(f"expected backend_requested=auto, got {auto.get('backend_requested')!r}")
    if auto.get("backend_selected") != selected_backend:
        raise SystemExit(f"expected AUTO to select {selected_backend!r}, got {auto.get('backend_selected')!r}")
    if auto.get("selected_kernel") != selected_kernel:
        raise SystemExit(f"expected AUTO kernel {selected_kernel!r}, got {auto.get('selected_kernel')!r}")

    auto_metadata = auto.get("backend_metadata") if isinstance(auto.get("backend_metadata"), dict) else {}
    if auto_metadata.get("performance_validated") is not True:
        raise SystemExit("default-cache AUTO hit must report performance_validated=true")
    if auto_metadata.get("autotune_key") != explicit_key or entry.get("key") != explicit_key:
        raise SystemExit("AUTO selected a different autotune key than the reviewed default-cache entry")

    timing = auto.get("timing_metadata") if isinstance(auto.get("timing_metadata"), dict) else {}
    if timing.get("gpu_event_timing") is not True:
        raise SystemExit("default-cache AUTO hit must report selected-backend GPU event timing")
    scope = timing.get("gpu_event_timing_source_scope")
    if selected_backend == "hipblaslt":
        expected_scope = "hipblaslt_baseline_default_stream_backend_operation_groups"
    elif semantics in {"finite-u8-ring", "finite-u8-field"}:
        expected_scope = "accelerator_backend_default_stream_operation_groups_with_direct_hip_pack_export"
    else:
        expected_scope = "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
    if scope != expected_scope:
        raise SystemExit(f"expected GPU event source scope {expected_scope!r}, got {scope!r}")

    if auto.get("semantics") != explicit.get("semantics"):
        raise SystemExit(f"expected AUTO semantics {explicit.get('semantics')!r}, got {auto.get('semantics')!r}")

    print(f"benchmark AUTO default-cache smoke ({selected_backend}, {auto.get('semantics')}): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
