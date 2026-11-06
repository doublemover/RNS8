"""Own one verifier child and retain scoped terminal resource observations.

Only the explicit proof driver invokes this worker. Child stdout/stderr pass
through unchanged; resource metadata goes to a fresh, separate JSON document.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import platform
import subprocess
import struct
import sys
import time

SCHEMA = "rns8_scalar_proof_child_resources_v1"


@contextmanager
def windows_observer(pid: int):
    # Query-only access; never enable debug privileges or alter process limits.
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.K32GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    )
    kernel.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())

    def finish():
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not kernel.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)

    try:
        yield finish
    finally:
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def run_owned(command: list[str]) -> dict:
    if sys.version_info[:2] < (3, 15) or struct.calcsize("P") != 8:
        raise ValueError("64-bit Python 3.15+ is required for native child resource widths")
    start = time.monotonic()
    system = platform.system()
    result = dict(
        schema=SCHEMA,
        command=command,
        os=system,
        process_id=None,
        returncode=None,
        worker_python_executable=str(Path(sys.executable).resolve(strict=True)),
        worker_python_version=platform.python_version(),
        elapsed_seconds=None,
        elapsed_scope="owned_child_spawn_to_terminal_resource_observation",
        status="not_observed",
        raw_peak=None,
        peak_memory_bytes=None,
        resource_error=None,
    )
    if system == "Linux":
        result.update(
            source="linux_wait4_ru_maxrss",
            units="KiB",
            scope="waited_child_and_accounted_descendants_maxrss_not_concurrent_tree_total",
        )
        # Exactly one child is reaped, so unrelated workers cannot contaminate
        # a process-global RUSAGE_CHILDREN high-water mark.
        pid = os.posix_spawn(command[0], command, dict(os.environ))
        result["process_id"] = pid
        waited, status, usage = os.wait4(pid, 0)
        if waited != pid:
            raise ValueError("resource wait returned a different child")
        result["returncode"] = os.waitstatus_to_exitcode(status)
        result["raw_peak"] = int(usage.ru_maxrss)
        if result["raw_peak"] > 0:
            result.update(status="observed", peak_memory_bytes=result["raw_peak"] * 1024)
        else:
            result["resource_error"] = "kernel returned no positive child maxrss observation"
    elif system == "Windows":
        result.update(
            source="windows_K32GetProcessMemoryInfo_PeakWorkingSetSize",
            units="bytes",
            scope="single_owned_child_peak_working_set_excludes_descendants",
        )
        with subprocess.Popen(command) as child, ExitStack() as resources:
            result["process_id"] = child.pid
            try:
                finish = resources.enter_context(windows_observer(child.pid))
            except OSError as error:
                # A child that exits before OpenProcess may be unavailable.
                finish = None
                result["resource_error"] = str(error)
            result["returncode"] = child.wait()
            if finish is not None:
                try:
                    result["raw_peak"] = finish()
                    if result["raw_peak"] > 0:
                        result.update(status="observed", peak_memory_bytes=result["raw_peak"])
                    else:
                        result["resource_error"] = "kernel returned no positive peak working set"
                except OSError as error:
                    result["resource_error"] = str(error)
    else:
        raise ValueError("scalar child resource observation supports Windows/Linux only")
    result["elapsed_seconds"] = time.monotonic() - start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (
        not command
        or len(command) > 128
        or any(not item or "\0" in item for item in command)
        or not Path(command[0]).is_absolute()
        or args.result.exists()
        or not args.result.parent.is_dir()
    ):
        parser.error("an absolute child command and fresh owned result file are required")
    try:
        result = run_owned(command)
        with args.result.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        code = result["returncode"]
        return code if type(code) is int and 0 <= code <= 125 else 125
    except (ValueError, OSError) as error:
        print("scalar proof child/resource acquisition failed: " + str(error), file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
