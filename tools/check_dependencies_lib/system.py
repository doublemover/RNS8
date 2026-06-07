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

from msvc_env import command_in_msvc_environment, find_visual_studio_installation

from .config import *

def run(args: list[str] | str, timeout: int = 20, env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
            shell=isinstance(args, str),
        )
        return completed.returncode, completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def compact_output(output: str, limit: int = 6000) -> str:
    if len(output) <= limit:
        return output
    return output[: limit // 2] + "\n...[truncated]...\n" + output[-limit // 2 :]


def user_tools_dirs() -> list[Path]:
    home = Path.home()
    return [Path(p) for p in glob.glob(str(home / "Tools" / "RadeonDeveloperToolSuite-*"))]


def vcpkg_required_for_host(host_system: str | None = None) -> bool:
    return (host_system or platform.system()) == "Windows"


def command_required_for_host(name: str, host_system: str | None = None) -> bool:
    host = host_system or platform.system()
    return (
        name in HOST_NEUTRAL_CORE_COMMANDS
        or (host == "Windows" and (name in WINDOWS_CORE_COMMANDS or name in WINDOWS_HIP_COMMANDS))
        or (host == "Linux" and name in LINUX_ROCM_COMMANDS)
    )


def command_names_for_host(host_system: str | None = None) -> list[str]:
    host = host_system or platform.system()
    names = (
        HOST_NEUTRAL_CORE_COMMANDS
        + WINDOWS_CORE_COMMANDS
        + WINDOWS_HIP_COMMANDS
        + LINUX_READINESS_COMMANDS
        + RADEON_TOOLS
    )
    if host != "Windows":
        names = [name for name in names if name not in WINDOWS_HIP_COMMANDS]
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def find_command(name: str) -> str | None:
    candidates: list[Path] = []
    found = shutil.which(name)
    if found:
        return found

    if name == "vcpkg" and platform.system() == "Windows":
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

    if name in set(LINUX_READINESS_COMMANDS):
        rocm_root = os.environ.get("ROCM_PATH", "/opt/rocm")
        candidates.extend(
            [
                Path(rocm_root) / "bin" / name,
                Path(rocm_root) / "bin" / f"{name}.exe",
                Path(rocm_root) / "rccl" / "bin" / name,
            ]
        )

    if name in RADEON_TOOLS:
        for tools_dir in user_tools_dirs():
            candidates.extend([tools_dir / f"{name}.exe", tools_dir / name])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def target_list(value: object) -> list[str]:
    if value is None:
        return []
    raw = str(value)
    targets = [part.strip() for part in re.split(r"[;,]", raw) if part.strip()]
    return targets


def manifest_vcpkg_packages() -> list[str]:
    manifest = load_json(repo_root() / "vcpkg.json")
    dependencies = manifest.get("dependencies", [])
    packages: list[str] = []
    if not isinstance(dependencies, list):
        return packages
    for dependency in dependencies:
        if isinstance(dependency, str):
            packages.append(dependency)
        elif isinstance(dependency, dict):
            name = dependency.get("name")
            if isinstance(name, str):
                packages.append(name)
    return packages


def cmake_presets_report() -> dict[str, object]:
    path = repo_root() / "CMakePresets.json"
    data = load_json(path)
    configure = data.get("configurePresets", [])
    build = data.get("buildPresets", [])
    test = data.get("testPresets", [])
    configure_presets = configure if isinstance(configure, list) else []
    build_presets = build if isinstance(build, list) else []
    test_presets = test if isinstance(test, list) else []

    configure_by_name = {
        item.get("name"): item for item in configure_presets if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    build_names = [item.get("name") for item in build_presets if isinstance(item, dict) and isinstance(item.get("name"), str)]
    test_names = [item.get("name") for item in test_presets if isinstance(item, dict) and isinstance(item.get("name"), str)]

    windows = configure_by_name.get("windows-msvc-hip-debug", {})
    linux = configure_by_name.get("linux-rocm-debug", {})
    windows_cache = windows.get("cacheVariables", {}) if isinstance(windows, dict) else {}
    linux_cache = linux.get("cacheVariables", {}) if isinstance(linux, dict) else {}
    if not isinstance(windows_cache, dict):
        windows_cache = {}
    if not isinstance(linux_cache, dict):
        linux_cache = {}

    windows_targets = target_list(windows_cache.get("RNS8_AMDGPU_TARGETS"))
    linux_active_targets = target_list(linux_cache.get("RNS8_AMDGPU_TARGETS"))
    linux_coverage_targets = target_list(linux_cache.get("RNS8_ROCM_COVERAGE_TARGETS"))
    if not linux_coverage_targets:
        linux_coverage_targets = linux_active_targets
    missing_linux_coverage = [target for target in LINUX_ROCM_COVERAGE_TARGETS if target not in linux_coverage_targets]
    windows_ok = bool(windows) and windows_cache.get("RNS8_ENABLE_HIP") == "ON" and "gfx1100" in windows_targets
    linux_represented = (
        bool(linux)
        and linux_cache.get("RNS8_ENABLE_HIP") == "ON"
        and bool(linux_active_targets)
        and not missing_linux_coverage
    )

    return {
        "ok": bool(data) and windows_ok and linux_represented,
        "path": str(path),
        "configure_presets": sorted(configure_by_name),
        "build_presets": sorted(name for name in build_names if isinstance(name, str)),
        "test_presets": sorted(name for name in test_names if isinstance(name, str)),
        "windows_hip_debug": {
            "ok": windows_ok,
            "toolchain": windows_cache.get("CMAKE_TOOLCHAIN_FILE"),
            "hip_root": windows_cache.get("RNS8_HIP_ROOT"),
            "amdgpu_targets": windows_cache.get("RNS8_AMDGPU_TARGETS"),
            "parsed_targets": windows_targets,
            "vcpkg_triplet": windows_cache.get("VCPKG_TARGET_TRIPLET"),
        },
        "linux_rocm_debug": {
            "represented": linux_represented,
            "toolchain": linux_cache.get("CMAKE_TOOLCHAIN_FILE"),
            "hip_root": linux_cache.get("RNS8_HIP_ROOT"),
            "amdgpu_targets": linux_cache.get("RNS8_AMDGPU_TARGETS"),
            "active_targets": linux_active_targets,
            "coverage_targets": linux_coverage_targets,
            "missing_coverage_targets": missing_linux_coverage,
            "vcpkg_triplet": linux_cache.get("VCPKG_TARGET_TRIPLET"),
            "detail": "source-level preset representation only; active compile targets are separate from RDNA/CDNA family coverage metadata",
        },
    }


