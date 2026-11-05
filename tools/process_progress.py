"""Bounded owned-process capture with progress, output and artifact budgets."""

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Mapping, Sequence


def capture_succeeded(result: dict) -> bool:
    return (
        type(result.get("returncode")) is int
        and result["returncode"] == 0
        and result.get("timed_out") is False
        and result.get("limit_exceeded") is None
        and result.get("capture_error") is None
        and (result.get("interrupted") is None or result.get("interrupted") is False)
    )


def _artifact_bytes(roots: Sequence[Path], stop_after: int) -> int:
    total = entries_seen = 0
    pending = list(roots)
    deadline = time.monotonic() + 1
    while pending:
        if time.monotonic() >= deadline:
            raise ValueError("owned artifact inventory exceeds its scan time budget")
        path = pending.pop()
        try:
            info = path.lstat()
            if path.is_symlink() or path.is_junction():
                raise ValueError("owned artifact tree contains a link or junction")
            if path.is_dir():
                with os.scandir(path) as entries:
                    for entry in entries:
                        entries_seen += 1
                        if entries_seen > 100000:
                            raise ValueError(
                                "owned artifact tree exceeds its entry budget"
                            )
                        pending.append(Path(entry.path))
            elif path.is_file():
                total += info.st_size
                if total > stop_after:
                    return total
            else:
                raise ValueError("owned artifact tree contains a nonregular file")
        except FileNotFoundError:
            continue
    return total


def run_capture(
    command: list[str],
    *,
    cwd: Path,
    label: str,
    timeout_seconds: float = 1800,
    heartbeat_seconds: float = 10,
    env: Mapping[str, str] | None = None,
    max_output_bytes: int = 64 * 1024**2,
    artifact_roots: Sequence[Path] = (),
    max_artifact_bytes: int | None = None,
    return_on_interrupt: bool = False,
) -> dict:
    if type(return_on_interrupt) is not bool:
        raise ValueError("return_on_interrupt must be an explicit boolean")
    if (
        not math.isfinite(timeout_seconds)
        or not math.isfinite(heartbeat_seconds)
        or timeout_seconds <= 0
        or heartbeat_seconds <= 0
    ):
        raise ValueError(
            "timeouts and heartbeat intervals must be finite positive numbers"
        )
    if type(max_output_bytes) is not int or max_output_bytes < 1:
        raise ValueError("positive stdout/stderr capture budget required")
    if isinstance(artifact_roots, (str, bytes)):
        raise ValueError("artifact roots must be an explicit sequence of paths")
    if (
        (
            max_artifact_bytes is not None
            and (type(max_artifact_bytes) is not int or max_artifact_bytes < 1)
        )
        or artifact_roots
        and max_artifact_bytes is None
    ):
        raise ValueError("owned artifact trees require a positive byte budget")
    roots = tuple((cwd / path).absolute() for path in artifact_roots)
    artifact_budget = max_artifact_bytes or 0
    start = time.monotonic()
    print(f"[{label}] start", file=sys.stderr, flush=True)
    wake = threading.Event()
    exceeded = threading.Event()
    read_failed = threading.Event()
    lock = threading.Lock()
    buffers = [bytearray(), bytearray()]
    read_errors = []
    stored = 0

    def drain(pipe, index):
        nonlocal stored
        try:
            while True:
                chunk = pipe.read(64 * 1024)
                if not chunk:
                    break
                with lock:
                    count = min(len(chunk), max_output_bytes - stored)
                    buffers[index].extend(chunk[:count])
                    stored += count
                    if count != len(chunk):
                        exceeded.set()
                wake.set()
        except BaseException as error:
            read_failed.set()
            with lock:
                read_errors.append(type(error).__name__ + ": " + str(error))
        finally:
            wake.set()

    readers = []

    def stop_process_tree():
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()

    timed_out = interrupted = False
    limit = capture_error = None
    artifact_bytes = 0
    next_heartbeat, next_artifact_check = start + heartbeat_seconds, start
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except OSError as exc:
        return dict(
            command=command,
            returncode=None,
            timed_out=False,
            process_id=None,
            elapsed_seconds=time.monotonic() - start,
            stdout="",
            stderr=str(exc),
            limit_exceeded=None,
            capture_error=None,
            captured_output_bytes=0,
        )
    try:
        for index, pipe in enumerate((process.stdout, process.stderr)):
            reader = threading.Thread(
                target=drain,
                args=(pipe, index),
                daemon=True,
                name="rns8-capture-" + str(index),
            )
            readers.append(reader)
            reader.start()
        while True:
            now = time.monotonic()
            if exceeded.is_set():
                limit = "stdout_stderr_bytes"
                stop_process_tree()
                break
            with lock:
                errors = tuple(read_errors)
            if read_failed.is_set():
                capture_error = "stdout/stderr read failed: " + "; ".join(errors)
                stop_process_tree()
                break
            if roots and now >= next_artifact_check:
                artifact_bytes = _artifact_bytes(roots, artifact_budget)
                next_artifact_check = now + 0.25
                if artifact_bytes > artifact_budget:
                    limit = "owned_artifact_bytes"
                    stop_process_tree()
                    break
            if process.poll() is not None and not any(
                reader.is_alive() for reader in readers
            ):
                break
            if now - start >= timeout_seconds:
                timed_out = True
                stop_process_tree()
                break
            if now >= next_heartbeat:
                print(
                    f"[{label}] running elapsed={now - start:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                next_heartbeat = now + heartbeat_seconds
            wake.wait(min(0.1, timeout_seconds - (now - start)))
            wake.clear()
    except Exception as error:
        capture_error = "owned process capture failed: " + str(error)
        stop_process_tree()
    except KeyboardInterrupt:
        stop_process_tree()
        if not return_on_interrupt:
            raise
        interrupted = True
        capture_error = "owned process capture interrupted"
    except BaseException:
        stop_process_tree()
        raise
    finally:
        # A detached descendant retaining a pipe is incomplete evidence.
        # Never wait forever or report it as a successful process capture.
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            capture_error = "owned process did not terminate during bounded cleanup"
        for index, pipe in enumerate((process.stdout, process.stderr)):
            reader = readers[index] if index < len(readers) else None
            if reader is not None and reader.ident is not None:
                reader.join(timeout=2)
            if reader is not None and reader.is_alive():
                capture_error = (
                    "owned process output did not close during bounded cleanup"
                )
            else:
                pipe.close()
    if exceeded.is_set():
        limit = "stdout_stderr_bytes"
    if roots and limit is None:
        try:
            artifact_bytes = _artifact_bytes(roots, artifact_budget)
            if artifact_bytes > artifact_budget:
                limit = "owned_artifact_bytes"
        except (OSError, ValueError) as error:
            capture_error = "owned artifact inventory incomplete: " + str(error)
    with lock:
        if read_failed.is_set():
            capture_error = "stdout/stderr read failed: " + "; ".join(read_errors)
        outputs = []
        for value in buffers:
            try:
                decoded = value.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                capture_error = "owned process output is not valid UTF-8"
                decoded = value.decode("utf-8", errors="replace")
            outputs.append(decoded.replace("\r\n", "\n").replace("\r", "\n"))
        captured = stored
    elapsed = time.monotonic() - start
    state = (
        limit
        or capture_error
        or ("TIMEOUT" if timed_out else f"exit={process.returncode}")
    )
    print(f"[{label}] {state} elapsed={elapsed:.1f}s", file=sys.stderr, flush=True)
    return dict(
        command=command,
        returncode=process.returncode,
        timed_out=timed_out,
        interrupted=interrupted,
        process_id=process.pid,
        elapsed_seconds=elapsed,
        stdout=outputs[0],
        stderr=outputs[1],
        limit_exceeded=limit,
        capture_error=capture_error,
        captured_output_bytes=captured,
        observed_artifact_bytes=artifact_bytes,
        max_output_bytes=max_output_bytes,
        max_artifact_bytes=max_artifact_bytes,
    )
