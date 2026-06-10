from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
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


def _command_option(command: list[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _cpu_reference_dedupe_key(entry: SweepCommand) -> tuple[Any, ...] | None:
    backend = _command_option(entry.command, "--backend")
    if backend == "cpu-reference":
        backend = "cpu"
    if backend not in {"cpu", "wrap64-byte-limb"}:
        return None
    env_items = tuple(sorted((entry.env or {}).items()))
    return (tuple(entry.command), env_items)


def _copy_capture(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def run_command(
    command: list[str],
    output: Path,
    timeout_seconds: float | None = None,
    env_overrides: dict[str, str] | None = None,
    progress: bool = False,
) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if env_overrides:
        run_env.update(env_overrides)
    started = time.perf_counter()
    if progress:
        sys.stderr.write(f"[benchmark_sweep] start {output}: {' '.join(command)}\n")
        sys.stderr.flush()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def read_stream(stream: Any, chunks: list[str], echo: bool) -> None:
        try:
            for text in stream:
                chunks.append(text)
                if echo:
                    sys.stderr.write(text)
                    sys.stderr.flush()
        finally:
            stream.close()

    try:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=run_env,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_chunks, False), daemon=True)
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_chunks, progress), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        try:
            returncode = process.wait(timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait()
            timed_out = True
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        failure = {
            "command": command,
            "environment": env_overrides or {},
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "duration_seconds": duration,
            "stdout": _timeout_text(exc.stdout),
            "stderr": _timeout_text(exc.stderr),
        }
        output.with_suffix(".failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        if progress:
            sys.stderr.write(f"[benchmark_sweep] timeout {output}: {duration:.3f}s\n")
            sys.stderr.flush()
        return False
    duration = time.perf_counter() - started
    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)
    if timed_out:
        failure = {
            "command": command,
            "environment": env_overrides or {},
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "duration_seconds": duration,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }
        output.with_suffix(".failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        if progress:
            sys.stderr.write(f"[benchmark_sweep] timeout {output}: {duration:.3f}s\n")
            sys.stderr.flush()
        return False
    if returncode != 0:
        failure = {
            "command": command,
            "environment": env_overrides or {},
            "returncode": returncode,
            "timed_out": False,
            "duration_seconds": duration,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }
        output.with_suffix(".failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        if progress:
            sys.stderr.write(f"[benchmark_sweep] fail {output}: {duration:.3f}s returncode={returncode}\n")
            sys.stderr.flush()
        return False
    output.write_text(stdout_text, encoding="utf-8")
    if progress:
        sys.stderr.write(f"[benchmark_sweep] done {output}: {duration:.3f}s\n")
        sys.stderr.flush()
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


def annotate_scenario_metadata(path: Path, scenario: dict[str, Any] | None) -> None:
    if scenario is None:
        return
    data = load_capture(path)
    data["scenario_metadata"] = scenario
    validate_capture(data, path)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


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
        "deduped_cpu_captures": 0,
        "deferred_captures": 0,
    }
    max_new = getattr(args, "max_new_captures", None)
    timeout_seconds = getattr(args, "capture_timeout_seconds", None)
    progress = bool(getattr(args, "progress", False))
    total_entries = len(entries)
    start_time = time.perf_counter()
    completed_cpu_captures: dict[tuple[Any, ...], Path] = {}
    entry_index = 0
    for entry in entries:
        entry_index += 1
        dedupe_key = _cpu_reference_dedupe_key(entry)
        if getattr(args, "skip_existing", False) and existing_capture_valid(entry.output):
            annotate_scenario_metadata(entry.output, entry.scenario)
            capture_paths.append(entry.output)
            stats["skipped_existing_captures"] += 1
            if dedupe_key is not None:
                completed_cpu_captures[dedupe_key] = entry.output
            continue
        if dedupe_key is not None and dedupe_key in completed_cpu_captures:
            _copy_capture(completed_cpu_captures[dedupe_key], entry.output)
            annotate_scenario_metadata(entry.output, entry.scenario)
            capture_paths.append(entry.output)
            stats["deduped_cpu_captures"] += 1
            continue
        if max_new is not None and stats["new_captures_attempted"] >= max_new:
            stats["deferred_captures"] += 1
            continue
        stats["new_captures_attempted"] += 1
        if run_command(
            entry.command,
            entry.output,
            timeout_seconds=timeout_seconds,
            env_overrides=entry.env,
            progress=progress,
        ):
            annotate_scenario_metadata(entry.output, entry.scenario)
            capture_paths.append(entry.output)
            stats["new_captures_completed"] += 1
            if dedupe_key is not None:
                completed_cpu_captures[dedupe_key] = entry.output
        if progress and entry_index % 10 == 0:
            elapsed = time.perf_counter() - start_time
            rate = entry_index / elapsed if elapsed > 0 else 0
            remaining = total_entries - entry_index
            eta = remaining / rate if rate > 0 else 0
            sys.stderr.write(f"[benchmark_sweep] progress: {entry_index}/{total_entries} ({entry_index*100//total_entries}%) elapsed={elapsed:.0f}s eta={eta:.0f}s remaining={remaining}\n")
            sys.stderr.flush()
    return stats


