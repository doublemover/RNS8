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

from .accelerators import *
from .config import *
from .discovery import *
from .readiness import *
from .system import *

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
    for command in command_names_for_host(host_system):
        path = find_command(command)
        required = command_required_for_host(command, host_system)
        if not path:
            commands[command] = {"ok": False, "required": required, "detail": "not found"}
            missing_required = missing_required or required
            continue
        detail = command_version(command, path)
        version_ok, detail = command_version_ok(command, detail)
        commands[command] = {
            "ok": version_ok,
            "required": required,
            "path": path,
            "detail": detail,
        }
        missing_required = missing_required or (required and not version_ok)

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
        if host_system == "Windows" and package in CORE_VCPKG_PACKAGES and version is None:
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
        "rccl": rccl_discovery_report(),
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
    report["hard_cut_self_checks"] = hard_cut_self_checks(report)
    hard_cut_checks = report["hard_cut_self_checks"]
    assert isinstance(hard_cut_checks, dict)
    return report, not missing_required and bool(readiness["host_readiness_ok"]) and bool(hard_cut_checks["ok"])


