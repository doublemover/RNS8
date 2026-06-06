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
from .system import *

def hip_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("HIP_PATH", "ROCM_PATH"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    roots.append(Path(r"C:\Program Files\AMD\ROCm\7.1"))
    roots.append(Path("/opt/rocm"))
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def vcpkg_roots() -> list[Path]:
    roots: list[Path] = []
    if platform.system() != "Windows":
        return roots
    root = os.environ.get("VCPKG_ROOT", r"C:\vcpkg")
    triplets = ["x64-windows", "x64-linux"]
    env_triplet = os.environ.get("VCPKG_TARGET_TRIPLET")
    if env_triplet and env_triplet not in triplets:
        triplets.append(env_triplet)
    for triplet in triplets:
        roots.append(Path(root) / "installed" / triplet)
        roots.append(repo_root() / "vcpkg_installed" / triplet)
    return roots


def repo_local_accelerator_roots(name: str) -> list[Path]:
    root = repo_root()
    if name == "ck":
        roots = [
            root / "third_party" / "rocm" / "composable_kernel",
        ]
        if platform.system() == "Windows":
            roots.extend(
                [
                    root / "out" / "third_party" / "rocm" / "install" / "windows-gfx1100",
                    root / "out" / "third_party" / "rocm" / "build" / "windows-gfx1100" / "composable_kernel",
                ]
            )
        return roots
    if name == "rocwmma":
        roots = [
            root / "third_party" / "rocm" / "rocWMMA",
        ]
        if platform.system() == "Windows":
            roots.extend(
                [
                    root / "out" / "third_party" / "rocm" / "install" / "windows-gfx1100",
                    root / "out" / "third_party" / "rocm" / "build" / "windows-gfx1100" / "rocWMMA",
                ]
            )
        return roots
    return []


def git_text(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    command = ["git", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd or repo_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
            check=False,
        )
        return completed.returncode, completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def repo_local_dependency_report(name: str) -> dict[str, object]:
    spec = EXPECTED_ROCM_SUBMODULES.get(name)
    if not spec:
        return {"status": "not_repo_local_dependency", "ok": False}
    root = repo_root()
    relative_path = str(spec["path"])
    path = root / relative_path
    gitmodules = root / ".gitmodules"
    configured_url = ""
    configured_branch = ""
    if gitmodules.exists():
        code, configured_url = git_text(
            ["config", "--file", str(gitmodules), "--get", f"submodule.{relative_path}.url"]
        )
        if code != 0:
            configured_url = ""
        code, configured_branch = git_text(
            ["config", "--file", str(gitmodules), "--get", f"submodule.{relative_path}.branch"]
        )
        if code != 0:
            configured_branch = ""

    actual_sha = ""
    actual_branch = ""
    actual_url = ""
    if path.exists():
        code, actual_sha = git_text(["rev-parse", "HEAD"], cwd=path)
        if code != 0:
            actual_sha = ""
        code, actual_branch = git_text(["branch", "--show-current"], cwd=path)
        if code != 0:
            actual_branch = ""
        code, actual_url = git_text(["remote", "get-url", "origin"], cwd=path)
        if code != 0:
            actual_url = ""

    initialized = bool(actual_sha)
    expected_sha = str(spec["sha"])
    expected_url = str(spec["url"])
    expected_branch = str(spec["branch"])
    return {
        "status": "present" if initialized else "missing",
        "ok": initialized,
        "path": str(path),
        "relative_path": relative_path,
        "expected_url": expected_url,
        "configured_url": configured_url,
        "actual_url": actual_url,
        "url_matches": configured_url == expected_url and (not actual_url or actual_url == expected_url),
        "expected_branch": expected_branch,
        "configured_branch": configured_branch,
        "actual_branch": actual_branch,
        "branch_matches": configured_branch == expected_branch and (not actual_branch or actual_branch == expected_branch),
        "expected_sha": expected_sha,
        "actual_sha": actual_sha,
        "sha_matches": actual_sha == expected_sha,
        "readiness": (
            "initialized at expected pinned release SHA"
            if actual_sha == expected_sha
            else "run python tools/bootstrap_rocm_accelerators.py --init --probe --target gfx1100"
        ),
    }


def first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return str(path)
    return None


def find_under_roots(roots: list[Path], relatives: list[str]) -> str | None:
    candidates: list[Path] = []
    for root in roots:
        for relative in relatives:
            candidates.append(root / relative)
    return first_existing(candidates)


def command_version(name: str, path: str) -> str:
    if name in {"RadeonDeveloperPanel", "RadeonGPUProfiler", "RadeonMemoryVisualizer"}:
        return "found"
    if name == "ninja":
        _, output = run([path, "--version"])
    elif name == "vcpkg":
        _, output = run([path, "version"])
    elif name == "hipInfo":
        _, output = run([path], timeout=30)
        return parse_hip_info_summary(output)
    elif name in {"rocminfo", "rocm-smi", "amd-smi"}:
        _, output = run([path, "--version"], timeout=30)
        if not output and name == "rocminfo":
            _, output = run([path], timeout=30)
    elif name == "rocprofv3-avail":
        _, output = run([path, "--help"], timeout=30)
    elif name in {"rocprofv3", "rocm-bandwidth-test", *RCCL_TEST_COMMANDS}:
        _, output = run([path, "--version"], timeout=30)
        if not output:
            _, output = run([path, "--help"], timeout=30)
    elif name == "RadeonDeveloperServiceCLI":
        _, output = run([path, "--help"])
    else:
        _, output = run([path, "--version"])

    first = output.splitlines()[0] if output else ""
    return first.strip()


def _ldconfig_library_path(name: str) -> str | None:
    if platform.system() != "Linux":
        return None
    code, output = run(["ldconfig", "-p"], timeout=30)
    if code != 0:
        return None
    for line in output.splitlines():
        if name in line and "=>" in line:
            return line.split("=>", 1)[1].strip()
    return None


def rccl_discovery_report() -> dict[str, object]:
    roots = hip_roots()
    header = find_under_roots(roots, ["include/rccl/rccl.h", "include/rccl.h"])
    library = find_under_roots(roots, ["lib/librccl.so", "lib64/librccl.so"]) or _ldconfig_library_path("librccl.so")
    test_commands = {name: find_command(name) for name in RCCL_TEST_COMMANDS}
    tests_present = {name: path for name, path in test_commands.items() if path}
    return {
        "ok": bool(header and library),
        "required_for_single_device_smoke": False,
        "readiness_lane": "future_multi_gpu_platform",
        "header": header,
        "library": library,
        "rccl_tests_ready": bool(tests_present),
        "rccl_test_commands": test_commands,
        "detail": (
            "RCCL headers/libraries discovered; rccl-tests command present"
            if header and library and tests_present
            else "RCCL and rccl-tests are future multi-GPU platform readiness signals, not single-device smoke blockers"
        ),
    }


def parse_hip_info_details(output: str) -> dict[str, str]:
    name = ""
    arch = ""
    mem = ""
    hip_version = ""
    runtime_version = ""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Name:"):
            name = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("gcnArchName:"):
            arch = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("totalGlobalMem:"):
            mem = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("HIP version:"):
            hip_version = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Runtime version:"):
            runtime_version = stripped.split(":", 1)[1].strip()
    return {
        "gpu_name": name,
        "gcn_arch": arch,
        "total_global_mem": mem,
        "hip_version": hip_version,
        "runtime_version": runtime_version,
    }


def parse_hip_info_summary(output: str) -> str:
    details = parse_hip_info_details(output)
    parts = [
        details["gpu_name"],
        details["gcn_arch"],
        details["total_global_mem"],
        details["hip_version"],
        details["runtime_version"],
    ]
    parts = [part for part in parts if part]
    return ", ".join(parts) if parts else "no HIP device details parsed"


def hip_info_report(path: str | None) -> dict[str, object]:
    if not path:
        return {
            "ok": False,
            "detail": "hipInfo not found",
            "gpu_name": "",
            "gcn_arch": "",
            "target": "",
            "target_supported_by_spec": False,
        }
    code, output = run([path], timeout=30)
    details = parse_hip_info_details(output)
    target = details["gcn_arch"].split(":", 1)[0].strip()
    return {
        "ok": code == 0 and bool(target),
        "detail": parse_hip_info_summary(output),
        "exit_code": code,
        "gpu_name": details["gpu_name"],
        "gcn_arch": details["gcn_arch"],
        "target": target,
        "target_supported_by_spec": target in SUPPORTED_TARGETS,
        "target_info": SUPPORTED_TARGETS.get(target),
        "hip_version": details["hip_version"],
        "runtime_version": details["runtime_version"],
    }


def find_msvc() -> str | None:
    installation = find_visual_studio_installation()
    if installation:
        return str(installation)
    return shutil.which("cl")


def msvc_probe_command(source: Path, binary: Path, include_root: str, library: str) -> list[str] | str | None:
    obj = binary.with_suffix(".obj")
    command = [
        "cl",
        "/nologo",
        "/EHsc",
        "/std:c++17",
        "/D__HIP_PLATFORM_AMD__=1",
        f"/I{include_root}",
        str(source),
        library,
        f"/Fo:{obj}",
        f"/Fe:{binary}",
    ]
    try:
        wrapped, _ = command_in_msvc_environment(command)
        return wrapped
    except RuntimeError:
        return None


def command_line_for_report(command: list[str] | str) -> str:
    if isinstance(command, str):
        return command
    return subprocess.list2cmdline(command)


def command_with_windows_developer_environment(command: list[str]) -> tuple[list[str] | str | None, str]:
    if platform.system() != "Windows":
        return command, command_line_for_report(command)
    try:
        wrapped, _ = command_in_msvc_environment(command)
        return wrapped, command_line_for_report(wrapped)
    except RuntimeError:
        return None, ""


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


