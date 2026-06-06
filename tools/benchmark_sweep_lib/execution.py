from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture

from .config import SweepCommand, WRAP64_ROCWMMA_CANDIDATE_BACKEND

def cli_backend(backend: str) -> str:
    if backend == "hip-vector-alu-int64":
        return "hip-vector-alu-int64-runtime"
    return backend


def normalize_semantics(value: str) -> str:
    aliases = {
        "bounded_i64": "bounded-i64",
        "bounded_u64": "bounded-u64",
        "exact_wide_signed": "exact-wide-signed",
        "exact-wide-i64": "exact-wide-signed",
        "exact_wide_unsigned": "exact-wide-unsigned",
        "exact-wide-u64": "exact-wide-unsigned",
        "wrap_u64_mod_2_64": "wrap-u64",
        "finite-ring-u8": "finite-u8-ring",
        "finite_ring_u8": "finite-u8-ring",
        "finite-field-u8": "finite-u8-field",
        "finite_field_u8": "finite-u8-field",
    }
    return aliases.get(value, value)


def parse_backend_bench(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--bench-for must use BACKEND=PATH, got {value!r}")
        backend, path = value.split("=", 1)
        if not backend or not path:
            raise SystemExit(f"--bench-for must use BACKEND=PATH, got {value!r}")
        result[backend] = Path(path)
    return result


def autotune_cache_path() -> Path:
    override = os.environ.get("RNS8_AUTOTUNE_CACHE_PATH")
    if override:
        return Path(override)
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("USERPROFILE")
        if root:
            suffix = ["rns8-gemm", "autotune.json"]
            if root == os.environ.get("USERPROFILE"):
                suffix = ["AppData", "Local", *suffix]
            return Path(root).joinpath(*suffix)
    root = os.environ.get("XDG_CACHE_HOME") or os.environ.get("HOME")
    if root:
        return Path(root) / ".cache" / "rns8-gemm" / "autotune.json"
    return Path("rns8-gemm") / "autotune.json"


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def run_command(
    command: list[str],
    output: Path,
    timeout_seconds: float | None = None,
    env_overrides: dict[str, str] | None = None,
) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if env_overrides:
        run_env.update(env_overrides)
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            env=run_env,
        )
    except subprocess.TimeoutExpired as exc:
        failure = {
            "command": command,
            "environment": env_overrides or {},
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "stdout": _timeout_text(exc.stdout),
            "stderr": _timeout_text(exc.stderr),
        }
        output.with_suffix(".failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return False
    if completed.returncode != 0:
        failure = {
            "command": command,
            "environment": env_overrides or {},
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        output.with_suffix(".failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return False
    output.write_text(completed.stdout, encoding="utf-8")
    return True


def validate_paths(paths: list[Path]) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    for path in paths:
        try:
            data = load_capture(path)
            validate_capture(data, path)
        except BenchmarkSchemaError as exc:
            raise SystemExit(str(exc)) from exc
        data["_path"] = str(path)
        captures.append(data)
    return captures


def existing_capture_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = load_capture(path)
        validate_capture(data, path)
    except (BenchmarkSchemaError, OSError, json.JSONDecodeError):
        return False
    return True


def execute_sweep_entries(
    entries: list[SweepCommand],
    args: argparse.Namespace,
    capture_paths: list[Path],
) -> dict[str, int]:
    stats = {
        "planned_captures": len(entries),
        "skipped_existing_captures": 0,
        "new_captures_attempted": 0,
        "new_captures_completed": 0,
        "deferred_captures": 0,
    }
    max_new = getattr(args, "max_new_captures", None)
    timeout_seconds = getattr(args, "capture_timeout_seconds", None)
    for entry in entries:
        if getattr(args, "skip_existing", False) and existing_capture_valid(entry.output):
            capture_paths.append(entry.output)
            stats["skipped_existing_captures"] += 1
            continue
        if max_new is not None and stats["new_captures_attempted"] >= max_new:
            stats["deferred_captures"] += 1
            continue
        stats["new_captures_attempted"] += 1
        if run_command(entry.command, entry.output, timeout_seconds=timeout_seconds, env_overrides=entry.env):
            capture_paths.append(entry.output)
            stats["new_captures_completed"] += 1
    return stats


