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
from .system import *

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
        return "windows-host-required"
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
    primitive_probe_status = (
        probe.get("primitive_probe_status") if isinstance(probe, dict) else "NOT_REQUESTED"
    )
    if primitive_probe_status == "OBJECT_COMPILE_PASS":
        status = "PRIMITIVE_PROBE_PASS"
    elif primitive_probe_status == "OBJECT_COMPILE_FAIL":
        status = "PRIMITIVE_PROBE_FAIL"
    elif probe_status == "COMPILE_LINK_PASS_RUN_PASS":
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
    elif name == "hipblaslt" and probe_status == "COMPILE_LINK_PASS_RUN_PASS":
        detail = (
            "optional probe evidence exists; hipBLASLt has an opt-in baseline backend, "
            "but this checker still does not enable the backend or prove exactness"
        )
    elif name == "hipblaslt" and found:
        detail = (
            "component discovered; hipBLASLt has an opt-in baseline backend, but discovery "
            "is not correctness validation and this checker does not enable the backend"
        )
    elif name == "ck" and primitive_probe_status == "OBJECT_COMPILE_PASS":
        detail = (
            "optional int8 matrix-engine primitive compile evidence exists; CK has an opt-in fused backend, "
            "but this checker still does not enable the backend or prove exactness"
        )
    elif name == "ck" and found:
        detail = (
            "component discovered; CK has an opt-in fused backend, but discovery is not correctness validation "
            "and this checker does not enable the backend"
        )
    elif name == "rocwmma" and primitive_probe_status == "OBJECT_COMPILE_PASS":
        detail = (
            "optional int8 matrix-engine primitive compile evidence exists; rocWMMA has an opt-in fused backend, "
            "but this checker still does not enable the backend or prove exactness"
        )
    elif name == "rocwmma" and found:
        detail = (
            "component discovered; rocWMMA has an opt-in fused backend, but discovery is not correctness validation "
            "and this checker does not enable the backend"
        )
    elif primitive_probe_status == "OBJECT_COMPILE_PASS":
        detail = (
            "optional int8 matrix-engine primitive compile evidence exists, but this checker still "
            "does not enable the backend or prove exactness"
        )
    elif probe_status == "COMPILE_LINK_PASS_RUN_PASS":
        detail = "optional probe evidence exists, but this checker still does not enable the backend or prove exactness"
    elif found:
        detail = "component discovered, but discovery is not correctness validation and this checker does not enable the backend"
    else:
        detail = "component not discovered; optional on Windows and required on Linux only where the target officially supports it"
    return {
        "status": status,
        "ok": False,
        "required_for_host_readiness": False,
        "backend_stage": item["backend_stage"],
        "evidence_class": CANDIDATE_ACCELERATOR_EVIDENCE_CLASS,
        "backend_enablement": (
            "requires_explicit_RNS8_ENABLE_HIPBLASLT_build"
            if name == "hipblaslt"
            else "requires_explicit_RNS8_ENABLE_CK_build"
            if name == "ck"
            else "requires_explicit_RNS8_ENABLE_ROCWMMA_build"
            if name == "rocwmma"
            else "disabled"
        ),
        "correctness_backend": (
            "implemented_opt_in_baseline_not_validated_by_dependency_report"
            if name == "hipblaslt"
            else "implemented_opt_in_fused_not_validated_by_dependency_report"
            if name in {"ck", "rocwmma"}
            else "not_implemented"
        ),
        "validated_correctness_backend": False,
        "candidate_evidence_is_correctness_validation": False,
        "evidence": [
            f"probe: {item.get('probe')}",
            f"cmake module: {item.get('cmake_module') or 'not found'}",
            f"cmake config: {item.get('cmake_config') or 'not found'}",
            f"cmake target: {item.get('cmake_target') or 'not found'}",
            f"header: {item.get('header') or 'not found'}",
            f"library: {item.get('library') or 'not found'}",
            f"library format: {item.get('library_format') or 'not found'}",
            f"msvc import library: {item.get('msvc_import_library') or 'not found'}",
            f"import archive: {item.get('import_archive') or 'not found'}",
            f"runtime library: {item.get('runtime_library') or 'not found'}",
            f"tool: {item.get('tool') or 'not found'}",
            f"compile/link probe: {probe_status}",
            f"int8 primitive object probe: {primitive_probe_status}",
        ],
        "detail": detail,
        "windows_policy": "optional feature-detected accelerator" if host_is_windows else "not the active host policy",
    }


def accelerator_enablement_policy(
    accelerators: dict[str, dict[str, object]],
    probes: dict[str, object],
) -> dict[str, object]:
    probe_items = probes.get("items") if isinstance(probes, dict) else {}
    flags: dict[str, dict[str, object]] = {}
    for name in ACCELERATOR_NAMES:
        component = accelerators.get(name, {})
        probe = probe_items.get(name) if isinstance(probe_items, dict) else None
        is_hipblaslt = name == "hipblaslt"
        is_ck = name == "ck"
        is_rocwmma = name == "rocwmma"
        flags[name] = {
            "enable_flag": ACCELERATOR_ENABLE_FLAGS[name],
            "enable_policy": (
                "explicit_opt_in_baseline_backend_with_exact_differentials"
                if is_hipblaslt
                else "explicit_opt_in_fused_backend_with_exact_differentials"
                if is_ck or is_rocwmma
                else ACCELERATOR_ENABLE_POLICY
            ),
            "backend_enablement": (
                "requires_explicit_RNS8_ENABLE_HIPBLASLT_build"
                if is_hipblaslt
                else "requires_explicit_RNS8_ENABLE_CK_build"
                if is_ck
                else "requires_explicit_RNS8_ENABLE_ROCWMMA_build"
                if is_rocwmma
                else "disabled"
            ),
            "correctness_backend": (
                "implemented_opt_in_baseline_not_validated_by_dependency_report"
                if is_hipblaslt
                else "implemented_opt_in_fused_not_validated_by_dependency_report"
                if is_ck or is_rocwmma
                else "not_implemented"
            ),
            "validated_correctness_backend": False,
            "can_enable_correctness_backend": (
                (is_hipblaslt or is_ck or is_rocwmma) and bool(component.get("ok"))
                if isinstance(component, dict)
                else False
            ),
            "feature_detection": "evidence_only",
            "evidence_class": CANDIDATE_ACCELERATOR_EVIDENCE_CLASS,
            "candidate_evidence_is_correctness_validation": False,
            "component_discovered": bool(component.get("ok")) if isinstance(component, dict) else False,
            "probe_status": probe.get("status") if isinstance(probe, dict) else "NOT_REQUESTED",
            "primitive_probe_status": (
                probe.get("primitive_probe_status") if isinstance(probe, dict) else "NOT_REQUESTED"
            ),
            "primitive_probe_ok": bool(probe.get("primitive_probe_ok")) if isinstance(probe, dict) else False,
            "readiness_effect": "none",
        }
    return {
        "backend_enablement": "probe_only_dependency_report",
        "correctness_backends_enabled": False,
        "validated_correctness_backend_count": 0,
        "enable_flags_fail_fast": "amdgpu_builtins",
        "evidence_class": CANDIDATE_ACCELERATOR_EVIDENCE_CLASS,
        "candidate_evidence_is_correctness_validation": False,
        "policy": (
            "hipBLASLt, CK, and rocWMMA have explicit opt-in backends validated by separate build/test presets; "
            "AMDGPU builtin flags fail fast until real exact correctness backends exist; "
            "discovery and probes are evidence only"
        ),
        "flags": flags,
    }


def correctness_backend_validation_status(
    host_system: str,
    hip_info: dict[str, object],
    accelerators: dict[str, dict[str, object]],
) -> dict[str, object]:
    target = hip_info.get("target")
    windows_gfx1100_visible = host_system == "Windows" and target == "gfx1100"
    accelerator_summary: dict[str, dict[str, object]] = {}
    for name in ACCELERATOR_NAMES:
        item = accelerators.get(name, {})
        is_hipblaslt = name == "hipblaslt"
        is_ck = name == "ck"
        is_rocwmma = name == "rocwmma"
        accelerator_summary[name] = {
            "evidence_class": CANDIDATE_ACCELERATOR_EVIDENCE_CLASS,
            "component_discovered": bool(item.get("ok")) if isinstance(item, dict) else False,
            "enable_flag": ACCELERATOR_ENABLE_FLAGS[name],
            "backend_enablement": (
                "requires_explicit_RNS8_ENABLE_HIPBLASLT_build"
                if is_hipblaslt
                else "requires_explicit_RNS8_ENABLE_CK_build"
                if is_ck
                else "requires_explicit_RNS8_ENABLE_ROCWMMA_build"
                if is_rocwmma
                else "disabled"
            ),
            "correctness_backend": (
                "implemented_opt_in_baseline_not_validated_by_dependency_report"
                if is_hipblaslt
                else "implemented_opt_in_fused_not_validated_by_dependency_report"
                if is_ck or is_rocwmma
                else "not_implemented"
            ),
            "validated_correctness_backend": False,
            "candidate_evidence_is_correctness_validation": False,
        }
    return {
        "checker_scope": CHECKER_VALIDATION_SCOPE,
        "validated_by_this_report": [],
        "validated_correctness_backend_count": 0,
        "candidate_accelerator_evidence_is_correctness_validation": False,
        "implemented_correctness_backends": {
            "cpu_reference": {
                "backend_enablement": "implemented",
                "evidence_class": "implemented_correctness_backend",
                "validated_by_this_report": False,
                "validation_source": "build and test runs, not dependency discovery",
            },
            "direct_hip": {
                "backend_enablement": "implemented_when_built_with_RNS8_ENABLE_HIP",
                "evidence_class": "implemented_correctness_backend",
                "host_target_evidence": "windows_gfx1100_visible" if windows_gfx1100_visible else "not_current_host_evidence",
                "validated_by_this_report": False,
                "validation_source": "Windows/Linux HIP build, ctest, and smoke runs, not dependency discovery",
            },
            "wrap64_byte_limb": {
                "backend_enablement": "implemented",
                "evidence_class": "implemented_correctness_backend",
                "validated_by_this_report": False,
                "validation_source": "unit/differential tests, not dependency discovery",
            },
            "hipblaslt": {
                "backend_enablement": "implemented_when_built_with_RNS8_ENABLE_HIPBLASLT",
                "evidence_class": "implemented_correctness_backend",
                "host_target_evidence": "windows_gfx1100_visible" if windows_gfx1100_visible else "not_current_host_evidence",
                "validated_by_this_report": False,
                "validation_source": "windows-hipblaslt-debug build and exact CPU/direct-HIP differentials, not dependency discovery",
            },
            "ck": {
                "backend_enablement": "implemented_when_built_with_RNS8_ENABLE_CK",
                "evidence_class": "implemented_correctness_backend",
                "host_target_evidence": "windows_gfx1100_visible" if windows_gfx1100_visible else "not_current_host_evidence",
                "validated_by_this_report": False,
                "validation_source": "windows-ck-debug build, exact CPU/direct-HIP differentials, CK ISA gate, and benchmark schema fixtures, not dependency discovery",
            },
            "rocwmma": {
                "backend_enablement": "implemented_when_built_with_RNS8_ENABLE_ROCWMMA",
                "evidence_class": "implemented_correctness_backend",
                "host_target_evidence": "windows_gfx1100_visible" if windows_gfx1100_visible else "not_current_host_evidence",
                "validated_by_this_report": False,
                "validation_source": "windows-rocwmma-debug build, exact CPU/direct-HIP differentials, rocWMMA ISA gate, and benchmark schema fixtures, not dependency discovery",
            },
        },
        "candidate_accelerators": accelerator_summary,
        "policy": (
            "This dependency report can expose prerequisites and candidate accelerator evidence, but only "
            "separate correctness runs can validate CPU, direct-HIP, wrap64, or future accelerator backends."
        ),
    }


def exact_wide_platform_validation_status(host_system: str, hip_info: dict[str, object]) -> dict[str, object]:
    target = hip_info.get("target")
    windows_gfx1100_evidence = host_system == "Windows" and target == "gfx1100"
    return {
        "host": host_system,
        "current_host_target": target or "not parsed",
        "windows_gfx1100_evidence_by_this_host": windows_gfx1100_evidence,
        "windows_gfx1100_scope": (
            "local direct-HIP bring-up evidence only"
            if windows_gfx1100_evidence
            else "not current host evidence"
        ),
        "linux_rocm_validated_by_this_host": False,
        "instinct_validated_by_this_host": False,
        "windows_evidence_validates_linux_rocm": False,
        "windows_evidence_validates_instinct": False,
        "linux_rocm_validation_evidence": "not_validated_by_this_host",
        "instinct_validation_evidence": "not_validated_by_this_host",
        "linux_validation_status": "unvalidated_until_real_linux_rocm_host",
        "instinct_validation_status": "unvalidated_until_real_linux_rocm_cdna_host",
        "validation_boundary": "Windows evidence is host-local and cannot be promoted to Linux ROCm or Instinct validation",
        "policy": (
            "exact-wide Windows evidence does not validate Linux ROCm, Radeon Linux, or Instinct CDNA; "
            "those gates require a real supported Linux ROCm host and exact CPU differential runs"
        ),
        "validation_claims": {
            "windows_gfx1100": "evidence_only" if windows_gfx1100_evidence else "not_current_host_evidence",
            "linux_rocm": "unvalidated_by_this_host",
            "instinct_cdna": "unvalidated_by_this_host",
        },
        "required_linux_gates": [
            "E003_linux_rocm_detection",
            "E072_linux_rdna2_rdna3_rdna4_rocm",
            "E073_E074_E075_linux_instinct_rocm",
        ],
    }


def readiness_report(report: dict[str, object]) -> dict[str, object]:
    commands = report["commands"]
    py_packages = report["python_packages"]
    vcpkg_packages = report["vcpkg_packages"]
    cmake_presets = report["cmake_presets"]
    accelerators = report["accelerator_components"]
    accelerator_probes = report["accelerator_compile_probes"]
    rccl = report.get("rccl", {})
    hip_info = report["hip_info"]
    msvc = report["msvc"]
    assert isinstance(commands, dict)
    assert isinstance(py_packages, dict)
    assert isinstance(vcpkg_packages, dict)
    assert isinstance(cmake_presets, dict)
    assert isinstance(accelerators, dict)
    assert isinstance(accelerator_probes, dict)
    assert isinstance(rccl, dict)
    assert isinstance(hip_info, dict)
    assert isinstance(msvc, dict)

    host_system = platform.system()
    host_is_windows = host_system == "Windows"
    host_is_linux = host_system == "Linux"
    windows = cmake_presets["windows_hip_debug"]
    linux = cmake_presets["linux_rocm_debug"]
    assert isinstance(windows, dict)
    assert isinstance(linux, dict)

    host_core_commands = HOST_NEUTRAL_CORE_COMMANDS + (WINDOWS_CORE_COMMANDS if host_is_windows else [])
    core_host_ok = (
        all(command_ok(commands, name) for name in host_core_commands)
        and (not host_is_windows or vcpkg_ok(vcpkg_packages, CORE_VCPKG_PACKAGES))
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
    linux_profiler_ok = host_is_linux and all(command_ok(commands, name) for name in LINUX_PROFILER_COMMANDS)
    linux_topology_present = any(command_ok(commands, name) for name in LINUX_TOPOLOGY_COMMANDS)
    rccl_ready = bool(rccl.get("ok"))
    rccl_tests_ready = bool(rccl.get("rccl_tests_ready"))
    gpu_arch_ok = bool(hip_info["ok"]) and bool(hip_info["target_supported_by_spec"])

    gates: dict[str, dict[str, object]] = {
        "E001_cpu_compiler_and_boost_reference": {
            "status": status_label(core_host_ok),
            "ok": core_host_ok,
            "required_for_host_readiness": True,
            "evidence": [
                "commands: " + ", ".join(f"{name}={'OK' if command_ok(commands, name) else 'MISSING'}" for name in host_core_commands),
                "python packages: "
                + ", ".join(f"{name}={'OK' if py_packages.get(name) else 'MISSING'}" for name in PYTHON_PACKAGES),
                (
                    "Windows vcpkg core: "
                    + ", ".join(
                        f"{name}={'OK' if vcpkg_packages.get(name) else 'MISSING'}"
                        for name in CORE_VCPKG_PACKAGES
                    )
                    if host_is_windows
                    else "Windows vcpkg core: not required on Linux/native-package hosts"
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
        "E009_linux_profiler_counter_tooling": {
            "status": status_label(linux_profiler_ok, host_is_linux),
            "ok": linux_profiler_ok,
            "required_for_host_readiness": False,
            "required_for_counter_resource_audit": host_is_linux,
            "evidence": [
                "commands: "
                + ", ".join(f"{name}={'OK' if command_ok(commands, name) else 'MISSING'}" for name in LINUX_PROFILER_COMMANDS)
            ],
            "detail": "rocprofv3 and rocprofv3-avail are required for counter/resource audit readiness, not for basic build-only smoke",
        },
        "E010_linux_topology_capture_tooling": {
            "status": status_label(linux_topology_present, host_is_linux),
            "ok": linux_topology_present,
            "required_for_host_readiness": False,
            "evidence": [
                "commands: "
                + ", ".join(f"{name}={'OK' if command_ok(commands, name) else 'MISSING'}" for name in LINUX_TOPOLOGY_COMMANDS)
            ],
            "detail": "numactl/lstopo improve topology evidence for CDNA reports; missing tools do not block a basic single-device smoke",
        },
        "E011_linux_rccl_multigpu_platform": {
            "status": status_label(rccl_ready and rccl_tests_ready, host_is_linux),
            "ok": rccl_ready and rccl_tests_ready,
            "required_for_host_readiness": False,
            "required_for_multi_gpu_platform": host_is_linux,
            "evidence": [
                f"RCCL={'OK' if rccl_ready else 'MISSING'}",
                f"rccl-tests={'OK' if rccl_tests_ready else 'MISSING'}",
                f"header={rccl.get('header') or 'not found'}",
                f"library={rccl.get('library') or 'not found'}",
            ],
            "detail": "RCCL and rccl-tests are future multi-GPU platform readiness signals; current multi-GPU smoke uses independent per-GPU shards",
        },
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
        "checker_validation_scope": CHECKER_VALIDATION_SCOPE,
        "policy": "optional accelerators are never promoted to enabled backends by this checker",
        "gates": gates,
        "platform_gates": platform_gates,
        "correctness_backend_validation": correctness_backend_validation_status(host_system, hip_info, accelerators),
        "accelerator_enablement": accelerator_enablement_policy(accelerators, accelerator_probes),
        "exact_wide_platform_validation": exact_wide_platform_validation_status(host_system, hip_info),
    }


def hard_cut_self_checks(report: dict[str, object]) -> dict[str, object]:
    readiness = report.get("readiness", {})
    assert isinstance(readiness, dict)
    accelerator_enablement = readiness.get("accelerator_enablement", {})
    exact_wide_platform = readiness.get("exact_wide_platform_validation", {})
    correctness_validation = readiness.get("correctness_backend_validation", {})
    assert isinstance(accelerator_enablement, dict)
    assert isinstance(exact_wide_platform, dict)
    assert isinstance(correctness_validation, dict)
    flags = accelerator_enablement.get("flags", {})
    candidate_accelerators = correctness_validation.get("candidate_accelerators", {})
    assert isinstance(flags, dict)
    assert isinstance(candidate_accelerators, dict)
    expected_flags_present = all(name in flags for name in ACCELERATOR_NAMES)
    expected_candidate_records_present = all(name in candidate_accelerators for name in ACCELERATOR_NAMES)
    accelerator_flags_policy_clean = True
    for name, item in flags.items():
        if not isinstance(item, dict):
            accelerator_flags_policy_clean = False
            break
        expected_backend = (
            "requires_explicit_RNS8_ENABLE_HIPBLASLT_build"
            if name == "hipblaslt"
            else "requires_explicit_RNS8_ENABLE_CK_build"
            if name == "ck"
            else "requires_explicit_RNS8_ENABLE_ROCWMMA_build"
            if name == "rocwmma"
            else "disabled"
        )
        expected_correctness = (
            "implemented_opt_in_baseline_not_validated_by_dependency_report"
            if name == "hipblaslt"
            else "implemented_opt_in_fused_not_validated_by_dependency_report"
            if name in {"ck", "rocwmma"}
            else "not_implemented"
        )
        if (
            item.get("backend_enablement") != expected_backend
            or item.get("correctness_backend") != expected_correctness
            or item.get("validated_correctness_backend") is not False
            or item.get("candidate_evidence_is_correctness_validation") is not False
        ):
            accelerator_flags_policy_clean = False
            break
    checks = {
        "accelerator_records_complete": {
            "ok": expected_flags_present and expected_candidate_records_present,
            "detail": "all expected accelerator enablement and candidate-evidence records are present",
        },
        "accelerator_discovery_does_not_enable_backends": {
            "ok": accelerator_enablement.get("backend_enablement") == "probe_only_dependency_report"
            and accelerator_enablement.get("correctness_backends_enabled") is False
            and expected_flags_present
            and accelerator_flags_policy_clean,
            "detail": "dependency discovery stays evidence-only; accelerators require explicit build/test enablement",
        },
        "no_validated_accelerator_correctness_backends": {
            "ok": accelerator_enablement.get("validated_correctness_backend_count") == 0
            and correctness_validation.get("candidate_accelerator_evidence_is_correctness_validation") is False,
            "detail": "candidate accelerator evidence is not reported as correctness validation",
        },
        "dependency_checker_does_not_claim_correctness_validation": {
            "ok": correctness_validation.get("validated_correctness_backend_count") == 0
            and correctness_validation.get("validated_by_this_report") == [],
            "detail": "dependency discovery does not claim to validate implemented correctness backends",
        },
        "windows_evidence_does_not_validate_linux_or_instinct": {
            "ok": exact_wide_platform.get("windows_evidence_validates_linux_rocm") is False
            and exact_wide_platform.get("windows_evidence_validates_instinct") is False
            and exact_wide_platform.get("linux_rocm_validated_by_this_host") is False
            and exact_wide_platform.get("instinct_validated_by_this_host") is False,
            "detail": "Windows host evidence remains separated from Linux ROCm and Instinct validation",
        },
    }
    return {
        "ok": all(bool(item["ok"]) for item in checks.values()),
        "scope": "internal JSON/report consistency only; no external command or correctness verification",
        "checker_validation_scope": CHECKER_VALIDATION_SCOPE,
        "checks": checks,
    }


