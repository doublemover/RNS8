#!/usr/bin/env python3
"""Report RNS8 local development dependencies.

The script is intentionally read-only. It checks command discovery, Python
packages, vcpkg manifest dependencies, HIP device visibility, MSVC availability,
and optional Radeon Developer Tool Suite utilities.
"""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_COMMANDS = ["cmake", "ninja", "git", "python", "vcpkg", "hipcc", "hipInfo"]
PYTHON_PACKAGES = ["numpy", "pandas", "matplotlib", "pytest", "scipy"]
VCPKG_PACKAGES = [
    "benchmark",
    "boost-multiprecision",
    "catch2",
    "cli11",
    "flint",
    "fmt",
    "gmp",
    "nlohmann-json",
    "spdlog",
]
RADEON_TOOLS = [
    "rga",
    "RadeonGPUProfiler",
    "RadeonDeveloperPanel",
    "RadeonMemoryVisualizer",
    "RadeonDeveloperServiceCLI",
]


def run(args: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def user_tools_dirs() -> list[Path]:
    home = Path.home()
    return [Path(p) for p in glob.glob(str(home / "Tools" / "RadeonDeveloperToolSuite-*"))]


def find_command(name: str) -> str | None:
    candidates: list[Path] = []
    found = shutil.which(name)
    if found:
        return found

    if name == "vcpkg":
        root = os.environ.get("VCPKG_ROOT", r"C:\vcpkg")
        candidates.append(Path(root) / "vcpkg.exe")

    if name in {"hipcc", "hipInfo", "hipconfig"}:
        hip_root = os.environ.get("HIP_PATH", r"C:\Program Files\AMD\ROCm\7.1")
        candidates.extend(
            [
                Path(hip_root) / "bin" / f"{name}.exe",
                Path(hip_root) / "bin" / f"{name}.bat",
            ]
        )

    if name in RADEON_TOOLS:
        for tools_dir in user_tools_dirs():
            candidates.extend([tools_dir / f"{name}.exe", tools_dir / name])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def command_version(name: str, path: str) -> str:
    if name in {"RadeonDeveloperPanel", "RadeonGPUProfiler", "RadeonMemoryVisualizer"}:
        return "found"
    if name == "ninja":
        _, output = run([path, "--version"])
    elif name == "vcpkg":
        _, output = run([path, "version"])
    elif name == "hipInfo":
        _, output = run([path], timeout=30)
        return parse_hip_info(output)
    elif name == "RadeonDeveloperServiceCLI":
        _, output = run([path, "--help"])
    else:
        _, output = run([path, "--version"])

    first = output.splitlines()[0] if output else ""
    return first.strip()


def parse_hip_info(output: str) -> str:
    name = ""
    arch = ""
    mem = ""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Name:"):
            name = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("gcnArchName:"):
            arch = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("totalGlobalMem:"):
            mem = stripped.split(":", 1)[1].strip()
    parts = [part for part in [name, arch, mem] if part]
    return ", ".join(parts) if parts else "no HIP device details parsed"


def find_msvc() -> str | None:
    cl = shutil.which("cl")
    if cl:
        return cl

    vswhere_paths = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"),
        Path(r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"),
    ]
    for vswhere in vswhere_paths:
        if not vswhere.exists():
            continue
        code, output = run(
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
            return output.splitlines()[0].strip()
    return None


def python_packages() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in PYTHON_PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def installed_vcpkg_packages(vcpkg_path: str | None) -> dict[str, str]:
    if not vcpkg_path:
        return {}
    code, output = run([vcpkg_path, "list", "--classic"], timeout=30)
    if code != 0:
        return {}

    packages: dict[str, str] = {}
    for line in output.splitlines():
        match = re.match(r"^([^:\s]+):[^\s]+\s+([^\s]+)", line)
        if match:
            packages[match.group(1)] = match.group(2)
    return packages


def build_report() -> tuple[dict[str, object], bool]:
    commands = {}
    missing_required = False
    for command in REQUIRED_COMMANDS + ["hipconfig"] + RADEON_TOOLS:
        path = find_command(command)
        required = command in REQUIRED_COMMANDS
        if not path:
            commands[command] = {"ok": False, "required": required, "detail": "not found"}
            missing_required = missing_required or required
            continue
        commands[command] = {
            "ok": True,
            "required": required,
            "path": path,
            "detail": command_version(command, path),
        }

    packages = python_packages()
    for version in packages.values():
        if version is None:
            missing_required = True

    vcpkg_path = find_command("vcpkg")
    vcpkg_installed = installed_vcpkg_packages(vcpkg_path)
    vcpkg_report = {}
    for package in VCPKG_PACKAGES:
        version = vcpkg_installed.get(package)
        vcpkg_report[package] = version
        if version is None:
            missing_required = True

    msvc = find_msvc()
    if not msvc:
        missing_required = True

    report: dict[str, object] = {
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "commands": commands,
        "msvc": {"ok": bool(msvc), "detail": msvc or "not found"},
        "python_packages": packages,
        "vcpkg_packages": vcpkg_report,
    }
    return report, not missing_required


def print_human(report: dict[str, object]) -> None:
    print("RNS8 dependency check")
    print("======================")
    system = report["system"]
    assert isinstance(system, dict)
    print(f"platform: {system['platform']}")
    print(f"python:   {system['python']}")
    print()

    print("Commands")
    commands = report["commands"]
    assert isinstance(commands, dict)
    for name in sorted(commands):
        item = commands[name]
        assert isinstance(item, dict)
        status = "OK" if item["ok"] else "MISSING"
        req = "required" if item["required"] else "optional"
        path = item.get("path", "")
        detail = item.get("detail", "")
        print(f"[{status}] {name} ({req})")
        if path:
            print(f"  path:   {path}")
        if detail:
            print(f"  detail: {detail}")
    print()

    msvc = report["msvc"]
    assert isinstance(msvc, dict)
    print(f"MSVC: {'OK' if msvc['ok'] else 'MISSING'} - {msvc['detail']}")
    print()

    print("Python packages")
    py_packages = report["python_packages"]
    assert isinstance(py_packages, dict)
    for name in PYTHON_PACKAGES:
        version = py_packages[name]
        print(f"[{'OK' if version else 'MISSING'}] {name}: {version or 'not found'}")
    print()

    print("vcpkg packages")
    vcpkg_packages = report["vcpkg_packages"]
    assert isinstance(vcpkg_packages, dict)
    for name in VCPKG_PACKAGES:
        version = vcpkg_packages[name]
        print(f"[{'OK' if version else 'MISSING'}] {name}: {version or 'not installed'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    report, ok = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
