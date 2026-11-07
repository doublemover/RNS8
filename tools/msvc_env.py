#!/usr/bin/env python3
"""Helpers for running commands inside a Visual Studio developer environment."""

from __future__ import annotations

import os
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

Command = Sequence[str] | str


def _run_text(args: Sequence[str], timeout: int = 20) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def find_visual_studio_installation() -> Path | None:
    vsinstall = os.environ.get("VSINSTALLDIR")
    if vsinstall:
        path = Path(vsinstall)
        if path.exists():
            return path

    vswhere_paths = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"),
        Path(r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"),
    ]
    for vswhere in vswhere_paths:
        if not vswhere.exists():
            continue
        code, output = _run_text(
            [
                str(vswhere),
                "-latest",
                "-version",
                "[18.0,19.0)",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ]
        )
        if code == 0 and output:
            path = Path(output.splitlines()[0].strip())
            if path.exists():
                return path

    for path in [
        Path(r"C:\Program Files\Microsoft Visual Studio\18\Community"),
        Path(r"C:\Program Files\Microsoft Visual Studio\18\BuildTools"),
        Path(r"C:\Program Files\Microsoft Visual Studio\18\Professional"),
        Path(r"C:\Program Files\Microsoft Visual Studio\18\Enterprise"),
    ]:
        if path.exists():
            return path
    return None


def find_vsdevcmd() -> Path | None:
    installation = find_visual_studio_installation()
    if not installation:
        return None
    path = installation / "Common7" / "Tools" / "VsDevCmd.bat"
    return path if path.exists() else None


def has_developer_environment() -> bool:
    return bool(
        os.environ.get("VCINSTALLDIR")
        and os.environ.get("INCLUDE")
        and os.environ.get("LIB")
        and shutil.which("cl")
    )


def command_in_msvc_environment(
    args: Sequence[str],
    *,
    arch: str = "x64",
    host_arch: str = "x64",
) -> tuple[Command, bool]:
    command = [str(arg) for arg in args]
    if has_developer_environment():
        return command, False

    vsdevcmd = find_vsdevcmd()
    if not vsdevcmd:
        raise RuntimeError("Visual Studio developer command script not found")

    if not all(re.fullmatch(r"[A-Za-z0-9_]+", value) for value in (arch, host_arch)):
        raise RuntimeError("invalid Visual Studio target or host architecture")
    # Only the environment setup uses CMD. The relay launches the user's argv
    # directly so regex pipes, percent signs and quotes cannot become shell code.
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--vsdevcmd",
        str(vsdevcmd),
        "--arch",
        arch,
        "--host-arch",
        host_arch,
        "--",
        *command,
    ], True


def _developer_environment(script: Path, arch: str, host_arch: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment["RNS8_VSDEVCMD_SCRIPT"] = str(script)
    setup = (
        'cmd.exe /d /u /v:off /s /c "call "%RNS8_VSDEVCMD_SCRIPT%" '
        f'-no_logo -arch={arch} -host_arch={host_arch} >nul && set"'
    )
    completed = subprocess.run(
        setup,
        shell=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Visual Studio environment setup failed ({completed.returncode})")
    result = {}
    for line in completed.stdout.decode("utf-16-le").splitlines():
        name, separator, value = line.partition("=")
        if separator and name:
            result[name.upper()] = value
    # A failed or malformed capture must never execute a command with a partial
    # developer environment. Hidden CMD drive variables have no subprocess use.
    if not all(result.get(name) for name in ("VCINSTALLDIR", "INCLUDE", "LIB", "PATH")):
        raise RuntimeError("Visual Studio environment setup returned incomplete variables")
    result.pop("RNS8_VSDEVCMD_SCRIPT", None)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vsdevcmd", type=Path, required=True)
    parser.add_argument("--arch", default="x64")
    parser.add_argument("--host-arch", default="x64")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    command = options.command[1:] if options.command[:1] == ["--"] else options.command
    if not command or not all(
        re.fullmatch(r"[A-Za-z0-9_]+", value) for value in (options.arch, options.host_arch)
    ):
        parser.error("a command and valid target/host architectures are required")
    try:
        environment = _developer_environment(options.vsdevcmd, options.arch, options.host_arch)
        return subprocess.run(command, env=environment, check=False, shell=False).returncode
    except (OSError, RuntimeError, subprocess.TimeoutExpired, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def run_in_msvc_environment(
    args: Sequence[str],
    *,
    arch: str = "x64",
    host_arch: str = "x64",
    cwd: str | Path | None = None,
    timeout: int | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    command, _ = command_in_msvc_environment(args, arch=arch, host_arch=host_arch)
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        timeout=timeout,
        check=False,
        shell=isinstance(command, str),
    )


if __name__ == "__main__":
    raise SystemExit(_main())
