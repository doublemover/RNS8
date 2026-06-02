#!/usr/bin/env python3
"""Helpers for running commands inside a Visual Studio developer environment."""

from __future__ import annotations

import os
import shutil
import subprocess
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
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"),
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

    quoted = subprocess.list2cmdline(command)
    wrapped = f'cmd /d /c "call "{vsdevcmd}" -arch={arch} -host_arch={host_arch} && {quoted}"'
    return wrapped, True


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
