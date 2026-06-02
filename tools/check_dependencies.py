#!/usr/bin/env python3
"""Report RNS8 local development dependencies.

The script is intentionally read-only. It checks command discovery, Python
packages, vcpkg manifest dependencies, HIP device visibility, MSVC availability,
optional accelerator components and opt-in accelerator compile/run probes,
repository tools, and optional Radeon Developer Tool Suite utilities.
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


CORE_COMMANDS = ["cmake", "ninja", "git", "python", "vcpkg"]
WINDOWS_HIP_COMMANDS = ["hipcc", "hipInfo", "hipconfig"]
LINUX_ROCM_COMMANDS = ["hipcc", "hipconfig", "rocminfo"]
LINUX_SMI_COMMANDS = ["rocm-smi", "amd-smi"]
PYTHON_PACKAGES = ["numpy", "pandas", "matplotlib", "pytest", "scipy"]
CORE_VCPKG_PACKAGES = ["boost-multiprecision", "catch2"]
OPTIONAL_CPP_PACKAGES = ["gmp", "flint", "ntl", "fflas-ffpack", "linbox"]
RADEON_TOOLS = [
    "rga",
    "RadeonGPUProfiler",
    "RadeonDeveloperPanel",
    "RadeonMemoryVisualizer",
    "RadeonDeveloperServiceCLI",
]
SUPPORTED_TARGETS = {
    "gfx1030": {"tier": "W2", "family": "RDNA2", "role": "functional HIP regression"},
    "gfx1100": {"tier": "W0", "family": "RDNA3", "role": "local Windows bring-up and RDNA3 optimization"},
    "gfx1200": {"tier": "W1", "family": "RDNA4", "role": "current consumer matrix-core target"},
    "gfx1201": {"tier": "W1", "family": "RDNA4", "role": "current consumer matrix-core target"},
    "gfx90a": {"tier": "I2", "family": "CDNA2", "role": "supported CDNA2 cluster target"},
    "gfx942": {"tier": "I1", "family": "CDNA3", "role": "previous-generation Instinct production"},
    "gfx950": {"tier": "I0", "family": "CDNA4", "role": "current Instinct production"},
}
LINUX_ROCM_COVERAGE_TARGETS = tuple(SUPPORTED_TARGETS)
LINUX_RDNA_TARGETS = {"gfx1030", "gfx1100", "gfx1200", "gfx1201"}
LINUX_CDNA_TARGETS = {"gfx90a", "gfx942", "gfx950"}
ACCELERATOR_NAMES = ("hipblaslt", "ck", "rocwmma", "amdgpu_builtins")

ACCELERATOR_PROBE_SOURCES = {
    "hipblaslt": """#include <hipblaslt/hipblaslt.h>

int main() {
  hipblasLtHandle_t handle = nullptr;
  hipblasStatus_t status = hipblasLtCreate(&handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return 2;
  }
  int version = 0;
  (void)hipblasLtGetVersion(handle, &version);
  status = hipblasLtDestroy(handle);
  return status == HIPBLAS_STATUS_SUCCESS ? 0 : 3;
}
""",
    "ck": """#include <ck/ck.hpp>

int main() {
  return 0;
}
""",
    "rocwmma": """#include <rocwmma/rocwmma.hpp>

int main() {
  return 0;
}
""",
}


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


def compact_output(output: str, limit: int = 6000) -> str:
    if len(output) <= limit:
        return output
    return output[: limit // 2] + "\n...[truncated]...\n" + output[-limit // 2 :]


def user_tools_dirs() -> list[Path]:
    home = Path.home()
    return [Path(p) for p in glob.glob(str(home / "Tools" / "RadeonDeveloperToolSuite-*"))]


def command_names_for_host() -> list[str]:
    names = CORE_COMMANDS + WINDOWS_HIP_COMMANDS + LINUX_ROCM_COMMANDS + LINUX_SMI_COMMANDS + RADEON_TOOLS
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

    if name in {"rocminfo", "rocm-smi", "amd-smi"}:
        rocm_root = os.environ.get("ROCM_PATH", "/opt/rocm")
        candidates.extend(
            [
                Path(rocm_root) / "bin" / name,
                Path(rocm_root) / "bin" / f"{name}.exe",
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
    return Path(__file__).resolve().parents[1]


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
    root = os.environ.get("VCPKG_ROOT", r"C:\vcpkg")
    triplets = ["x64-windows", "x64-linux"]
    env_triplet = os.environ.get("VCPKG_TARGET_TRIPLET")
    if env_triplet and env_triplet not in triplets:
        triplets.append(env_triplet)
    for triplet in triplets:
        roots.append(Path(root) / "installed" / triplet)
        roots.append(repo_root() / "vcpkg_installed" / triplet)
    return roots


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
    elif name == "RadeonDeveloperServiceCLI":
        _, output = run([path, "--help"])
    else:
        _, output = run([path, "--version"])

    first = output.splitlines()[0] if output else ""
    return first.strip()


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


def accelerator_components() -> dict[str, dict[str, object]]:
    roots = hip_roots() + vcpkg_roots()
    modules = repo_root() / "cmake" / "modules"
    components = {
        "hipblaslt": {
            "header": find_under_roots(roots, ["include/hipblaslt/hipblaslt.h", "include/hipblaslt.h"]),
            "library": find_under_roots(
                roots,
                [
                    "lib/hipblaslt.lib",
                    "lib64/hipblaslt.lib",
                    "lib/libhipblaslt.dll.a",
                    "lib64/libhipblaslt.dll.a",
                    "lib/libhipblaslt.so",
                    "lib64/libhipblaslt.so",
                    "bin/libhipblaslt.dll",
                ],
            ),
            "tool": find_under_roots(roots, ["bin/hipblaslt-bench.exe", "bin/hipblaslt-bench"]),
            "cmake_module": first_existing([modules / "FindRNS8HIPBLASLT.cmake"]),
            "probe": "header/library/tool discovery only; no compile, link, or device capability test",
            "backend_stage": "B3/B4",
            "experiment": "E005",
            "capability": "hipBLASLt INT8 per-modulus and grouped/batched GEMM",
        },
        "ck": {
            "header": find_under_roots(roots, ["include/ck/ck.hpp", "include/ck.hpp"]),
            "library": None,
            "tool": None,
            "cmake_module": first_existing([modules / "FindRNS8CK.cmake"]),
            "probe": "header discovery only; no compile, link, or device capability test",
            "backend_stage": "B5/B6",
            "experiment": "E006",
            "capability": "CK grouped GEMM and custom epilogues",
        },
        "rocwmma": {
            "header": find_under_roots(roots, ["include/rocwmma/rocwmma.hpp"]),
            "library": None,
            "tool": None,
            "cmake_module": first_existing([modules / "FindRNS8ROCWMMA.cmake"]),
            "probe": "header discovery only; no compile, link, or device capability test",
            "backend_stage": "B7",
            "experiment": "E007",
            "capability": "rocWMMA target-specific hot kernels",
        },
        "amdgpu_builtins": {
            "header": None,
            "library": None,
            "tool": None,
            "cmake_module": None,
            "probe": "no shallow discovery probe; requires real target-specific kernels and exact CPU differential validation",
            "backend_stage": "B7",
            "experiment": "E008",
            "capability": "AMDGPU builtin target-specific hot kernels",
            "not_ready_detail": (
                "AMDGPU builtin path has no discovery-only readiness; backend remains disabled until "
                "target-specific kernels, exact CPU differentials, and ISA evidence exist"
            ),
        },
    }
    return {
        name: {
            "ok": bool(item["header"] or item["library"] or item["tool"]),
            "required": False,
            "header": item["header"],
            "library": item["library"],
            "tool": item["tool"],
            "cmake_module": item["cmake_module"],
            "probe": item["probe"],
            "backend_stage": item["backend_stage"],
            "experiment": item["experiment"],
            "capability": item["capability"],
            "readiness": (
                "candidate evidence only; backend remains disabled until compiled capability and exact correctness probes pass"
                if bool(item["header"] or item["library"] or item["tool"])
                else item.get(
                    "not_ready_detail",
                    "not discovered; optional on Windows and required on Linux only where officially supported",
                )
            ),
        }
        for name, item in components.items()
    }


def include_root_for_header(header: str | None) -> str | None:
    if not header:
        return None
    path = Path(header)
    parent = path.parent
    if parent.name in {"hipblaslt", "ck", "rocwmma"}:
        return str(parent.parent)
    return str(parent)


def probe_binary_path(probe_dir: Path, name: str) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    return probe_dir / f"{name}_probe{suffix}"


def not_run_probe(
    name: str,
    probe_dir: Path,
    status: str,
    detail: str,
    requested: bool,
) -> dict[str, object]:
    return {
        "probe_requested": requested,
        "requested": requested,
        "status": status,
        "ok": False,
        "compiled_probe_ok": False,
        "runtime_probe_ok": False,
        "device_capability_ok": False,
        "backend_enablement": "disabled",
        "readiness_effect": "none",
        "source": str(probe_dir / f"{name}_probe.cpp"),
        "binary": str(probe_binary_path(probe_dir, name)),
        "command": [],
        "exit_code": None,
        "run_command": [],
        "run_exit_code": None,
        "run_output": "",
        "output": "",
        "detail": detail,
    }


def accelerator_compile_probes(
    accelerators: dict[str, dict[str, object]],
    requested: bool,
    probe_dir: Path | None,
    offload_target: str | None,
) -> dict[str, object]:
    root = probe_dir or (repo_root() / "temp" / "accelerator-probes")
    items: dict[str, dict[str, object]] = {}
    if not requested:
        for name in ACCELERATOR_NAMES:
            items[name] = not_run_probe(
                name,
                root,
                "NOT_REQUESTED",
                "compile/link/run probe not requested; pass --accelerator-probes to run evidence probes under temp/",
                False,
            )
        return {
            "requested": False,
            "probe_root": str(root),
            "policy": "not run by default; probes never enable correctness backends or affect host readiness",
            "items": items,
        }

    hipcc = find_command("hipcc")
    root.mkdir(parents=True, exist_ok=True)
    for name in ACCELERATOR_NAMES:
        component = accelerators.get(name, {})
        if name == "amdgpu_builtins":
            items[name] = not_run_probe(
                name,
                root,
                "NOT_RUN_NO_CORRECTNESS_KERNEL",
                "AMDGPU builtin probes are disabled until a real target-specific exact kernel exists",
                True,
            )
            continue
        header = component.get("header") if isinstance(component, dict) else None
        library = component.get("library") if isinstance(component, dict) else None
        if not isinstance(header, str) or not header:
            items[name] = not_run_probe(name, root, "NOT_RUN_MISSING_HEADER", "component header not discovered", True)
            continue
        if name == "hipblaslt" and (not isinstance(library, str) or not library):
            items[name] = not_run_probe(name, root, "NOT_RUN_MISSING_LIBRARY", "hipBLASLt library not discovered", True)
            continue
        if not hipcc:
            items[name] = not_run_probe(name, root, "NOT_RUN_MISSING_COMPILER", "hipcc not discovered", True)
            continue

        source = root / f"{name}_probe.cpp"
        binary = probe_binary_path(root, name)
        source.write_text(ACCELERATOR_PROBE_SOURCES[name], encoding="utf-8")
        include_root = include_root_for_header(header)
        command = [hipcc, "-std=c++17"]
        if offload_target:
            command.append(f"--offload-arch={offload_target}")
        command.append(str(source))
        if include_root:
            command.extend(["-I", include_root])
        if name == "hipblaslt" and isinstance(library, str):
            command.extend(["-L", str(Path(library).parent), "-lhipblaslt"])
        command.extend(["-o", str(binary)])
        code, output = run(command, timeout=60)
        run_command: list[str] = []
        run_code: int | None = None
        run_output = ""
        if code == 0:
            run_command = [str(binary)]
            run_code, run_output = run(run_command, timeout=30)
        compiled = code == 0
        runtime_ok = run_code == 0
        if not compiled:
            status = "COMPILE_LINK_FAIL"
        elif runtime_ok:
            status = "COMPILE_LINK_PASS_RUN_PASS"
        else:
            status = "COMPILE_LINK_PASS_RUN_FAIL"
        items[name] = {
            "probe_requested": True,
            "requested": True,
            "status": status,
            "ok": compiled and runtime_ok,
            "compiled_probe_ok": compiled,
            "runtime_probe_ok": runtime_ok,
            "device_capability_ok": False,
            "backend_enablement": "disabled",
            "readiness_effect": "none",
            "source": str(source),
            "binary": str(binary),
            "command": command,
            "exit_code": code,
            "run_command": run_command,
            "run_exit_code": run_code,
            "run_output": compact_output(run_output),
            "output": compact_output(output),
            "detail": (
                "tiny compile/link/run probe passed; still not a correctness backend or device capability validation"
                if compiled and runtime_ok
                else "tiny compile/link probe passed but runtime probe failed; this is optional accelerator evidence only"
                if compiled
                else "tiny compile/link probe failed; this is optional accelerator evidence only"
            ),
        }
    return {
        "requested": True,
        "probe_root": str(root),
        "policy": "compile/link/run evidence only; probes never enable correctness backends or affect host readiness",
        "items": items,
    }


def optional_cpp_references(vcpkg_installed: dict[str, str]) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for package in OPTIONAL_CPP_PACKAGES:
        report[package] = {
            "ok": package in vcpkg_installed,
            "required": False,
            "version": vcpkg_installed.get(package),
        }
    return report


def project_tools() -> dict[str, dict[str, object]]:
    root = repo_root()
    tools = {
        "rns8-inspect": [root / "tools" / "rns8_inspect.cpp", root / "build" / "windows-msvc-hip-debug" / "rns8-inspect.exe"],
        "rns8-verify": [root / "tools" / "rns8_verify.cpp", root / "build" / "windows-msvc-hip-debug" / "rns8-verify.exe"],
        "rns8-bench": [root / "benchmarks" / "rns8_bench.cpp", root / "build" / "windows-msvc-hip-debug" / "rns8-bench.exe"],
        "result_compare": [root / "tools" / "result_compare.py"],
    }
    return {
        name: {
            "ok": bool(first_existing(paths)),
            "required": True,
            "detail": first_existing(paths) or "not found",
        }
        for name, paths in tools.items()
    }


def cmake_hip_language_report() -> dict[str, object]:
    return {
        "ok": False,
        "required": False,
        "detail": "not probed by the read-only dependency checker; RNS8 Windows builds use explicit hipcc integration instead of enable_language(HIP)",
    }


def command_ok(commands: dict[str, object], name: str) -> bool:
    item = commands.get(name)
    return isinstance(item, dict) and bool(item.get("ok"))


def packages_ok(packages: dict[str, str | None], names: list[str]) -> bool:
    return all(packages.get(name) is not None for name in names)


def vcpkg_ok(vcpkg_report: dict[str, str | None], names: list[str]) -> bool:
    return all(vcpkg_report.get(name) is not None for name in names)


def vcpkg_package_role(name: str) -> str:
    if name in CORE_VCPKG_PACKAGES:
        return "host-required"
    if name in OPTIONAL_CPP_PACKAGES:
        return "optional-reference"
    return "manifest-tracked"


def status_label(ok: bool, applicable: bool = True) -> str:
    if not applicable:
        return "NOT_APPLICABLE"
    return "PASS" if ok else "FAIL"


def accelerator_gate(
    name: str,
    accelerators: dict[str, dict[str, object]],
    host_is_windows: bool,
    probes: dict[str, object],
) -> dict[str, object]:
    item = accelerators[name]
    found = bool(item["ok"])
    probe_items = probes.get("items") if isinstance(probes, dict) else {}
    probe = probe_items.get(name) if isinstance(probe_items, dict) else None
    probe_status = probe.get("status") if isinstance(probe, dict) else "NOT_REQUESTED"
    if probe_status == "COMPILE_LINK_PASS_RUN_PASS":
        status = "PROBE_PASS"
    elif probe_status == "COMPILE_LINK_PASS_RUN_FAIL":
        status = "PROBE_RUNTIME_FAIL"
    elif probe_status == "COMPILE_LINK_FAIL":
        status = "PROBE_COMPILE_FAIL"
    else:
        status = "CANDIDATE" if found else "NOT_READY"
    if name == "amdgpu_builtins":
        status = "NOT_READY"
        detail = item["readiness"]
    elif probe_status == "COMPILE_LINK_PASS_RUN_PASS":
        detail = "optional probe evidence exists, but this checker still does not enable the backend or prove exactness"
    elif found:
        detail = "component discovered, but this checker does not enable the backend without a compiled capability probe"
    else:
        detail = "component not discovered; optional on Windows and required on Linux only where the target officially supports it"
    return {
        "status": status,
        "ok": False,
        "required_for_host_readiness": False,
        "backend_stage": item["backend_stage"],
        "evidence": [
            f"probe: {item.get('probe')}",
            f"cmake module: {item.get('cmake_module') or 'not found'}",
            f"header: {item.get('header') or 'not found'}",
            f"library: {item.get('library') or 'not found'}",
            f"tool: {item.get('tool') or 'not found'}",
            f"compile/link probe: {probe_status}",
        ],
        "detail": detail,
        "windows_policy": "optional feature-detected accelerator" if host_is_windows else "not the active host policy",
    }


def readiness_report(report: dict[str, object]) -> dict[str, object]:
    commands = report["commands"]
    py_packages = report["python_packages"]
    vcpkg_packages = report["vcpkg_packages"]
    cmake_presets = report["cmake_presets"]
    accelerators = report["accelerator_components"]
    accelerator_probes = report["accelerator_compile_probes"]
    hip_info = report["hip_info"]
    msvc = report["msvc"]
    assert isinstance(commands, dict)
    assert isinstance(py_packages, dict)
    assert isinstance(vcpkg_packages, dict)
    assert isinstance(cmake_presets, dict)
    assert isinstance(accelerators, dict)
    assert isinstance(accelerator_probes, dict)
    assert isinstance(hip_info, dict)
    assert isinstance(msvc, dict)

    host_system = platform.system()
    host_is_windows = host_system == "Windows"
    host_is_linux = host_system == "Linux"
    windows = cmake_presets["windows_hip_debug"]
    linux = cmake_presets["linux_rocm_debug"]
    assert isinstance(windows, dict)
    assert isinstance(linux, dict)

    core_host_ok = (
        all(command_ok(commands, name) for name in CORE_COMMANDS)
        and packages_ok(py_packages, PYTHON_PACKAGES)
        and vcpkg_ok(vcpkg_packages, CORE_VCPKG_PACKAGES)
    )
    windows_hip_ok = (
        host_is_windows
        and all(command_ok(commands, name) for name in WINDOWS_HIP_COMMANDS)
        and bool(msvc["ok"])
        and bool(windows["ok"])
        and bool(hip_info["ok"])
    )
    linux_smi_ok = any(command_ok(commands, name) for name in LINUX_SMI_COMMANDS)
    linux_rocm_ok = (
        host_is_linux
        and all(command_ok(commands, name) for name in LINUX_ROCM_COMMANDS)
        and linux_smi_ok
        and bool(linux["represented"])
    )
    gpu_arch_ok = bool(hip_info["ok"]) and bool(hip_info["target_supported_by_spec"])

    gates: dict[str, dict[str, object]] = {
        "E001_cpu_compiler_and_boost_reference": {
            "status": status_label(core_host_ok),
            "ok": core_host_ok,
            "required_for_host_readiness": True,
            "evidence": [
                "commands: " + ", ".join(f"{name}={'OK' if command_ok(commands, name) else 'MISSING'}" for name in CORE_COMMANDS),
                "python packages: "
                + ", ".join(f"{name}={'OK' if py_packages.get(name) else 'MISSING'}" for name in PYTHON_PACKAGES),
                "vcpkg core: "
                + ", ".join(
                    f"{name}={'OK' if vcpkg_packages.get(name) else 'MISSING'}"
                    for name in CORE_VCPKG_PACKAGES
                ),
            ],
            "detail": "Phase 0 host/reference readiness; optional differential libraries are reported separately",
        },
        "E002_windows_hip_sdk_detection": {
            "status": status_label(windows_hip_ok, host_is_windows),
            "ok": windows_hip_ok,
            "required_for_host_readiness": host_is_windows,
            "evidence": [
                "commands: "
                + ", ".join(f"{name}={'OK' if command_ok(commands, name) else 'MISSING'}" for name in WINDOWS_HIP_COMMANDS),
                f"MSVC={'OK' if msvc['ok'] else 'MISSING'}",
                f"windows preset={'OK' if windows['ok'] else 'MISSING'}",
                f"HIP device={hip_info.get('detail') or 'not parsed'}",
            ],
            "detail": "Windows GPU path gate; CMake HIP language is intentionally not required",
        },
        "E003_linux_rocm_detection": {
            "status": status_label(linux_rocm_ok, host_is_linux),
            "ok": linux_rocm_ok,
            "required_for_host_readiness": host_is_linux,
            "evidence": [
                "commands: "
                + ", ".join(f"{name}={'OK' if command_ok(commands, name) else 'MISSING'}" for name in LINUX_ROCM_COMMANDS),
                "smi: " + ", ".join(f"{name}={'OK' if command_ok(commands, name) else 'MISSING'}" for name in LINUX_SMI_COMMANDS),
                f"linux preset represented={'OK' if linux['represented'] else 'MISSING'}",
            ],
            "detail": "Linux ROCm production/profiling gate; not required to pass on a Windows host",
        },
        "E004_gpu_architecture_detection": {
            "status": status_label(gpu_arch_ok, bool(hip_info["ok"])),
            "ok": gpu_arch_ok,
            "required_for_host_readiness": host_is_windows or host_is_linux,
            "evidence": [
                f"gpu={hip_info.get('gpu_name') or 'not parsed'}",
                f"target={hip_info.get('target') or 'not parsed'}",
                f"spec target={'OK' if hip_info.get('target_supported_by_spec') else 'MISSING'}",
            ],
            "detail": "Target-specific backend selection is allowed only after architecture detection",
        },
        "E005_hipblaslt_int8_capability": accelerator_gate(
            "hipblaslt", accelerators, host_is_windows, accelerator_probes
        ),
        "E006_ck_capability": accelerator_gate("ck", accelerators, host_is_windows, accelerator_probes),
        "E007_rocwmma_capability": accelerator_gate("rocwmma", accelerators, host_is_windows, accelerator_probes),
        "E008_amdgpu_builtin_capability": accelerator_gate(
            "amdgpu_builtins", accelerators, host_is_windows, accelerator_probes
        ),
    }

    target = hip_info.get("target")
    platform_gates = {
        "E070_windows_rdna3_direct_hip": {
            "status": status_label(windows_hip_ok and target == "gfx1100", host_is_windows),
            "ok": windows_hip_ok and target == "gfx1100",
            "required_for_host_readiness": host_is_windows,
            "detail": "local Windows bring-up gate for RX 7900-class gfx1100 targets",
        },
        "E071_windows_rdna4_direct_hip": {
            "status": status_label(windows_hip_ok and target in {"gfx1200", "gfx1201"}, host_is_windows and target in {"gfx1200", "gfx1201"}),
            "ok": windows_hip_ok and target in {"gfx1200", "gfx1201"},
            "required_for_host_readiness": False,
            "detail": "current Radeon gate; evaluated only on matching Windows RDNA4 hardware",
        },
        "E072_linux_rdna2_rdna3_rdna4_rocm": {
            "status": status_label(
                linux_rocm_ok and target in LINUX_RDNA_TARGETS,
                host_is_linux and target in LINUX_RDNA_TARGETS,
            ),
            "ok": linux_rocm_ok and target in LINUX_RDNA_TARGETS,
            "required_for_host_readiness": False,
            "detail": "Radeon Linux gate for documented RDNA2/RDNA3/RDNA4 targets; not required on Windows",
        },
        "E073_E074_E075_linux_instinct_rocm": {
            "status": status_label(linux_rocm_ok and target in LINUX_CDNA_TARGETS, host_is_linux and target in LINUX_CDNA_TARGETS),
            "ok": linux_rocm_ok and target in LINUX_CDNA_TARGETS,
            "required_for_host_readiness": False,
            "detail": "Instinct validation gates cover documented CDNA2/CDNA3/CDNA4 targets and require supported Linux ROCm hardware",
        },
    }

    required_gates = [gate for gate in gates.values() if gate["required_for_host_readiness"]]
    return {
        "host": host_system,
        "host_readiness_ok": all(bool(gate["ok"]) for gate in required_gates),
        "policy": "optional accelerators are never promoted to enabled backends by this checker",
        "gates": gates,
        "platform_gates": platform_gates,
    }


def build_report(run_accelerator_probes: bool = False, accelerator_probe_dir: Path | None = None) -> tuple[dict[str, object], bool]:
    commands = {}
    missing_required = False
    host_system = platform.system()
    vcpkg_packages = manifest_vcpkg_packages()
    if not vcpkg_packages:
        missing_required = True
    cmake_presets = cmake_presets_report()
    if not cmake_presets["ok"]:
        missing_required = True
    for command in command_names_for_host():
        path = find_command(command)
        required = command in CORE_COMMANDS or (
            host_system == "Windows" and command in WINDOWS_HIP_COMMANDS
        ) or (host_system == "Linux" and command in LINUX_ROCM_COMMANDS)
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
    for package in vcpkg_packages:
        version = vcpkg_installed.get(package)
        vcpkg_report[package] = version
        if package in CORE_VCPKG_PACKAGES and version is None:
            missing_required = True

    msvc = find_msvc()
    if host_system == "Windows" and not msvc:
        missing_required = True

    hip_info = hip_info_report(find_command("hipInfo"))
    offload_target_value = hip_info.get("target") if isinstance(hip_info.get("target"), str) else None
    accelerators = accelerator_components()
    report: dict[str, object] = {
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "commands": commands,
        "msvc": {"ok": bool(msvc), "detail": msvc or "not found"},
        "python_packages": packages,
        "cmake_presets": cmake_presets,
        "vcpkg_manifest": {
            "path": str(repo_root() / "vcpkg.json"),
            "packages": vcpkg_packages,
        },
        "vcpkg_package_roles": {package: vcpkg_package_role(package) for package in vcpkg_packages},
        "vcpkg_packages": vcpkg_report,
        "hip_info": hip_info,
        "optional_cpp_references": optional_cpp_references(vcpkg_installed),
        "accelerator_components": accelerators,
        "accelerator_compile_probes": accelerator_compile_probes(
            accelerators, run_accelerator_probes, accelerator_probe_dir, offload_target_value
        ),
        "cmake_hip_language": cmake_hip_language_report(),
        "project_tools": project_tools(),
    }
    report["readiness"] = readiness_report(report)
    readiness = report["readiness"]
    assert isinstance(readiness, dict)
    return report, not missing_required and bool(readiness["host_readiness_ok"])


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

    print("CMake presets")
    cmake_presets = report["cmake_presets"]
    assert isinstance(cmake_presets, dict)
    print(f"[{'OK' if cmake_presets['ok'] else 'MISSING'}] {cmake_presets['path']}")
    print(f"  configure: {', '.join(cmake_presets['configure_presets'])}")
    print(f"  build:     {', '.join(cmake_presets['build_presets'])}")
    print(f"  test:      {', '.join(cmake_presets['test_presets'])}")
    windows = cmake_presets["windows_hip_debug"]
    linux = cmake_presets["linux_rocm_debug"]
    assert isinstance(windows, dict)
    assert isinstance(linux, dict)
    print(f"  windows HIP debug: {'OK' if windows['ok'] else 'MISSING'}")
    print(f"    toolchain: {windows.get('toolchain')}")
    print(f"    HIP root:  {windows.get('hip_root')}")
    print(f"    targets:   {windows.get('amdgpu_targets')}")
    print(f"    triplet:   {windows.get('vcpkg_triplet')}")
    print(f"  linux ROCm debug represented: {'OK' if linux['represented'] else 'MISSING'}")
    print(f"    toolchain: {linux.get('toolchain')}")
    print(f"    HIP root:  {linux.get('hip_root')}")
    print(f"    active targets:   {linux.get('amdgpu_targets')}")
    coverage_targets = linux.get("coverage_targets") or []
    assert isinstance(coverage_targets, list)
    print(f"    coverage targets: {', '.join(coverage_targets) if coverage_targets else 'not represented'}")
    missing_coverage = linux.get("missing_coverage_targets") or []
    assert isinstance(missing_coverage, list)
    print(f"    missing coverage: {', '.join(missing_coverage) if missing_coverage else 'none'}")
    print(f"    triplet:   {linux.get('vcpkg_triplet')}")
    print(f"    detail:    {linux.get('detail')}")
    print()

    print("vcpkg packages")
    manifest = report["vcpkg_manifest"]
    assert isinstance(manifest, dict)
    manifest_packages = manifest["packages"]
    assert isinstance(manifest_packages, list)
    print(f"manifest: {manifest['path']}")
    vcpkg_packages = report["vcpkg_packages"]
    assert isinstance(vcpkg_packages, dict)
    for name in manifest_packages:
        version = vcpkg_packages[name]
        role = vcpkg_package_role(name)
        print(f"[{'OK' if version else 'MISSING'}] {name} ({role}): {version or 'not installed'}")
    print()

    print("Optional CPU reference packages")
    optional_refs = report["optional_cpp_references"]
    assert isinstance(optional_refs, dict)
    for name in OPTIONAL_CPP_PACKAGES:
        item = optional_refs[name]
        assert isinstance(item, dict)
        print(f"[{'OK' if item['ok'] else 'MISSING'}] {name}: {item.get('version') or 'not installed'}")
    print()

    print("Accelerator components")
    accelerators = report["accelerator_components"]
    assert isinstance(accelerators, dict)
    for name in sorted(accelerators):
        item = accelerators[name]
        assert isinstance(item, dict)
        status = "OK" if item["ok"] else "MISSING"
        print(f"[{status}] {name} ({item['backend_stage']}, {item['experiment']}, optional)")
        print(f"  probe: {item['probe']}")
        print(f"  cmake module: {item.get('cmake_module') or 'not found'}")
        if item.get("header"):
            print(f"  header: {item['header']}")
        if item.get("library"):
            print(f"  library: {item['library']}")
        if item.get("tool"):
            print(f"  tool: {item['tool']}")
        print(f"  readiness: {item['readiness']}")
    print()

    print("Accelerator compile/run probes")
    accelerator_probes = report["accelerator_compile_probes"]
    assert isinstance(accelerator_probes, dict)
    print(f"requested: {accelerator_probes['requested']}")
    print(f"probe root: {accelerator_probes['probe_root']}")
    print(f"policy: {accelerator_probes['policy']}")
    probe_items = accelerator_probes["items"]
    assert isinstance(probe_items, dict)
    for name in sorted(probe_items):
        item = probe_items[name]
        assert isinstance(item, dict)
        print(f"[{item['status']}] {name}")
        print(f"  compiled: {item.get('compiled_probe_ok', False)}")
        print(f"  runtime:  {item.get('runtime_probe_ok', False)}")
        print(f"  backend enablement: {item.get('backend_enablement')}")
        print(f"  source: {item.get('source')}")
        print(f"  binary: {item.get('binary')}")
        if item.get("output"):
            print(f"  compile output: {item['output']}")
        if item.get("run_output"):
            print(f"  runtime output: {item['run_output']}")
        print(f"  detail: {item['detail']}")
    print()

    cmake_hip = report["cmake_hip_language"]
    assert isinstance(cmake_hip, dict)
    print(f"CMake HIP language: {'OK' if cmake_hip['ok'] else 'NOT REQUIRED'} - {cmake_hip['detail']}")
    print()

    hip_info = report["hip_info"]
    assert isinstance(hip_info, dict)
    print("HIP device")
    print(f"[{'OK' if hip_info['ok'] else 'MISSING'}] {hip_info['detail']}")
    print(f"  target: {hip_info.get('target') or 'not parsed'}")
    print(f"  spec target: {'OK' if hip_info.get('target_supported_by_spec') else 'MISSING'}")
    if hip_info.get("target_info"):
        target_info = hip_info["target_info"]
        assert isinstance(target_info, dict)
        print(f"  tier: {target_info['tier']} {target_info['family']} - {target_info['role']}")
    print()

    readiness = report["readiness"]
    assert isinstance(readiness, dict)
    print("Readiness gates")
    print(f"host readiness: {'OK' if readiness['host_readiness_ok'] else 'NOT READY'}")
    print(f"policy: {readiness['policy']}")
    gates = readiness["gates"]
    assert isinstance(gates, dict)
    for name in sorted(gates):
        gate = gates[name]
        assert isinstance(gate, dict)
        req = "host-required" if gate["required_for_host_readiness"] else "not host-required"
        print(f"[{gate['status']}] {name} ({req})")
        print(f"  detail: {gate['detail']}")
    platform_gates = readiness["platform_gates"]
    assert isinstance(platform_gates, dict)
    for name in sorted(platform_gates):
        gate = platform_gates[name]
        assert isinstance(gate, dict)
        req = "host-required" if gate["required_for_host_readiness"] else "not host-required"
        print(f"[{gate['status']}] {name} ({req})")
        print(f"  detail: {gate['detail']}")
    print()

    print("Project tools")
    tools = report["project_tools"]
    assert isinstance(tools, dict)
    for name in sorted(tools):
        item = tools[name]
        assert isinstance(item, dict)
        print(f"[{'OK' if item['ok'] else 'MISSING'}] {name}: {item['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--accelerator-probes",
        "--probe-accelerators",
        action="store_true",
        help="run optional tiny compile/run probes for discovered accelerator components under temp/",
    )
    parser.add_argument(
        "--accelerator-probe-dir",
        type=Path,
        default=None,
        help="directory for optional accelerator probe source and binaries; defaults to temp/accelerator-probes",
    )
    args = parser.parse_args()

    report, ok = build_report(args.accelerator_probes, args.accelerator_probe_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
