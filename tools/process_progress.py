"""Bounded subprocess capture with honest, flushed progress and heartbeats."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping


def run_capture(
    command: list[str], *, cwd: Path, label: str,
    timeout_seconds: float = 1800, heartbeat_seconds: float = 10,
    env: Mapping[str, str] | None = None,
) -> dict:
    if timeout_seconds <= 0 or heartbeat_seconds <= 0:
        raise ValueError("timeouts and heartbeat intervals must be positive")
    start = time.monotonic()
    print(f"[{label}] start", file=sys.stderr, flush=True)
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=os.name != "nt",
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    except OSError as exc:
        return {"command": command, "returncode": None, "timed_out": False,
                "elapsed_seconds": time.monotonic() - start, "stdout": "", "stderr": str(exc)}
    def stop_process_tree():
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()

    timed_out = False
    try:
        while True:
            remaining = timeout_seconds - (time.monotonic() - start)
            if remaining <= 0:
                timed_out = True
                stop_process_tree()
                stdout, stderr = process.communicate()
                break
            try:
                stdout, stderr = process.communicate(timeout=min(heartbeat_seconds, remaining))
                break
            except subprocess.TimeoutExpired:
                print(f"[{label}] running elapsed={time.monotonic() - start:.1f}s",
                      file=sys.stderr, flush=True)
    except BaseException:
        stop_process_tree()
        process.communicate()
        raise
    elapsed = time.monotonic() - start
    state = "TIMEOUT" if timed_out else f"exit={process.returncode}"
    print(f"[{label}] {state} elapsed={elapsed:.1f}s", file=sys.stderr, flush=True)
    return {"command": command, "returncode": process.returncode, "timed_out": timed_out,
            "elapsed_seconds": elapsed, "stdout": stdout, "stderr": stderr}
