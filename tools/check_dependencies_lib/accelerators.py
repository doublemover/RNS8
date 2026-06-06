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
from .discovery import *
from .system import *

def accelerator_components() -> dict[str, dict[str, object]]:
    roots = hip_roots() + vcpkg_roots()
    modules = repo_root() / "cmake" / "modules"
    ck_dependency = repo_local_dependency_report("ck")
    rocwmma_dependency = repo_local_dependency_report("rocwmma")
    hipblaslt_header = find_under_roots(roots, ["include/hipblaslt/hipblaslt.h", "include/hipblaslt.h"])
    hipblaslt_cmake_config = find_under_roots(
        roots,
        [
            "lib/cmake/hipblaslt/hipblaslt-config.cmake",
            "lib64/cmake/hipblaslt/hipblaslt-config.cmake",
        ],
    )
    hipblaslt_msvc_import_library = find_under_roots(roots, ["lib/hipblaslt.lib", "lib64/hipblaslt.lib"])
    hipblaslt_import_archive = find_under_roots(
        roots,
        ["lib/libhipblaslt.dll.a", "lib64/libhipblaslt.dll.a"],
    )
    hipblaslt_shared_library = find_under_roots(roots, ["lib/libhipblaslt.so", "lib64/libhipblaslt.so"])
    hipblaslt_runtime_library = find_under_roots(roots, ["bin/libhipblaslt.dll", "bin/hipblaslt.dll"])
    hipblaslt_library = (
        hipblaslt_msvc_import_library
        or hipblaslt_import_archive
        or hipblaslt_shared_library
        or hipblaslt_runtime_library
    )
    hipblaslt_library_format = (
        "msvc_import_library"
        if hipblaslt_msvc_import_library
        else "gnu_import_archive"
        if hipblaslt_import_archive
        else "shared_library"
        if hipblaslt_shared_library
        else "runtime_dll"
        if hipblaslt_runtime_library
        else "not_found"
    )
    components = {
        "hipblaslt": {
            "header": hipblaslt_header,
            "library": hipblaslt_library,
            "library_format": hipblaslt_library_format,
            "cmake_config": hipblaslt_cmake_config,
            "cmake_target": "roc::hipblaslt" if hipblaslt_cmake_config else None,
            "msvc_import_library": hipblaslt_msvc_import_library,
            "import_archive": hipblaslt_import_archive,
            "runtime_library": hipblaslt_runtime_library,
            "tool": find_under_roots(roots, ["bin/hipblaslt-bench.exe", "bin/hipblaslt-bench"]),
            "cmake_module": first_existing([modules / "FindRNS8HIPBLASLT.cmake"]),
            "probe": "header/library/tool/CMake-target discovery only; no correctness backend or device capability test",
            "backend_stage": "B3/B4",
            "experiment": "E005",
            "capability": "hipBLASLt INT8 per-modulus and grouped/batched GEMM",
            "link_guidance": (
                "Prefer AMD's roc::hipblaslt CMake target. On Windows HIP SDK 7.1, the official "
                "import archive is libhipblaslt.dll.a and no separate MSVC hipblaslt.lib is required."
            ),
        },
        "ck": {
            "header": find_under_roots(
                repo_local_accelerator_roots("ck") + roots,
                ["include/ck/ck.hpp", "include/ck.hpp"],
            ),
            "library": None,
            "tool": None,
            "cmake_module": first_existing([modules / "FindRNS8CK.cmake"]),
            "probe": (
                "repo-local header discovery plus optional hipcc compile/run probe and object-only "
                "int8 DeviceGemmWmma_CShuffle primitive probe for gfx1100; no backend correctness validation"
            ),
            "backend_stage": "B5/B6",
            "experiment": "E006",
            "capability": "CK grouped GEMM and custom epilogues",
            "repo_local_dependency": ck_dependency,
        },
        "rocwmma": {
            "header": find_under_roots(
                repo_local_accelerator_roots("rocwmma") + roots,
                ["library/include/rocwmma/rocwmma.hpp", "include/rocwmma/rocwmma.hpp"],
            ),
            "library": None,
            "tool": None,
            "cmake_module": first_existing([modules / "FindRNS8ROCWMMA.cmake"]),
            "probe": (
                "repo-local header discovery plus optional hipcc compile/run probe and object-only "
                "int8 fragment mma_sync primitive probe for gfx1100; no backend correctness validation"
            ),
            "backend_stage": "B7",
            "experiment": "E007",
            "capability": "rocWMMA target-specific hot kernels",
            "repo_local_dependency": rocwmma_dependency,
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
            "enable_flag": ACCELERATOR_ENABLE_FLAGS[name],
            "enable_policy": (
                "explicit_opt_in_baseline_backend_with_exact_differentials"
                if name == "hipblaslt"
                else "explicit_opt_in_fused_backend_with_exact_differentials"
                if name in {"ck", "rocwmma"}
                else ACCELERATOR_ENABLE_POLICY
            ),
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
            "can_enable_correctness_backend": name in {"hipblaslt", "ck", "rocwmma"}
            and bool(item["header"] or item["library"] or item["tool"]),
            "feature_detection": "evidence_only",
            "evidence_class": CANDIDATE_ACCELERATOR_EVIDENCE_CLASS,
            "candidate_evidence_is_correctness_validation": False,
            "header": item["header"],
            "library": item["library"],
            "library_format": item.get("library_format"),
            "cmake_config": item.get("cmake_config"),
            "cmake_target": item.get("cmake_target"),
            "msvc_import_library": item.get("msvc_import_library"),
            "import_archive": item.get("import_archive"),
            "runtime_library": item.get("runtime_library"),
            "tool": item["tool"],
            "cmake_module": item["cmake_module"],
            "probe": item["probe"],
            "backend_stage": item["backend_stage"],
            "experiment": item["experiment"],
            "capability": item["capability"],
            "link_guidance": item.get("link_guidance"),
            "repo_local_dependency": item.get("repo_local_dependency"),
            "readiness": (
                (
                    "candidate evidence only; hipBLASLt backend validation requires the explicit "
                    "windows-hipblaslt-debug build/test preset"
                    if name == "hipblaslt"
                    else (
                        "candidate evidence only; CK backend validation requires the explicit "
                        "windows-ck-debug build/test preset"
                        if name == "ck"
                        else (
                            "candidate evidence only; rocWMMA backend validation requires the explicit "
                            "windows-rocwmma-debug build/test preset"
                            if name == "rocwmma"
                            else "candidate evidence only; backend remains disabled until a real exact correctness "
                            "backend has target capability checks and exact CPU differentials"
                        )
                    )
                )
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


def write_ck_generated_headers(root: Path, commit_id: str = "unknown") -> Path:
    include_dir = root / "generated" / "ck" / "include"
    ck_dir = include_dir / "ck"
    ck_dir.mkdir(parents=True, exist_ok=True)
    (ck_dir / "config.h").write_text(
        "\n".join(
            [
                "#ifndef CK_CONFIG_H_IN",
                "#define CK_CONFIG_H_IN",
                "#define CK_ENABLE_INT8 ON",
                "#define CK_ENABLE_FP8 ON",
                "#define CK_ENABLE_BF8 ON",
                "#define CK_ENABLE_FP16 ON",
                "#define CK_ENABLE_BF16 ON",
                "#define CK_ENABLE_FP32 ON",
                "#define CK_ENABLE_FP64 ON",
                "#define CK_ENABLE_DL_KERNELS ON",
                "#define CK_ENABLE_DPP_KERNELS ON",
                "#define CK_USE_WMMA ON",
                "#endif",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (ck_dir / "version.h").write_text(
        "\n".join(
            [
                "#ifndef CK_VERSION_H_",
                "#define CK_VERSION_H_",
                "#define CK_VERSION 1.1.0",
                "#define CK_VERSION_MAJOR 1",
                "#define CK_VERSION_MINOR 1",
                "#define CK_VERSION_PATCH 0",
                f"#define CK_COMMIT_ID {commit_id or 'unknown'}",
                "#endif",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return include_dir


def include_roots_for_component(name: str, component: dict[str, object], probe_dir: Path) -> list[str]:
    header = component.get("header")
    include_root = include_root_for_header(header if isinstance(header, str) else None)
    roots: list[str] = []
    if name == "ck":
        dependency = component.get("repo_local_dependency")
        commit_id = ""
        if isinstance(dependency, dict):
            commit_value = dependency.get("actual_sha") or dependency.get("expected_sha")
            commit_id = str(commit_value) if commit_value else ""
        roots.append(str(write_ck_generated_headers(probe_dir, commit_id)))
    if include_root:
        roots.append(include_root)
    return roots


def probe_binary_path(probe_dir: Path, name: str) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    return probe_dir / f"{name}_probe{suffix}"


def primitive_probe_source_path(probe_dir: Path, name: str) -> Path:
    return probe_dir / f"{name}_primitive_probe.cpp"


def primitive_probe_object_path(probe_dir: Path, name: str) -> Path:
    suffix = ".obj" if platform.system() == "Windows" else ".o"
    return probe_dir / f"{name}_primitive_probe{suffix}"


def primitive_probe_not_run_fields(name: str, probe_dir: Path, status: str, detail: str, requested: bool) -> dict[str, object]:
    applicable = name in ACCELERATOR_PRIMITIVE_PROBE_SOURCES
    primitive_status = status if requested and applicable else ("NOT_REQUESTED" if applicable else "NOT_APPLICABLE")
    primitive_detail = detail if applicable else "no matrix-engine primitive dependency probe is defined for this accelerator"
    return {
        "primitive_probe_requested": requested and applicable,
        "primitive_probe_status": primitive_status,
        "primitive_probe_ok": False,
        "primitive_source": str(primitive_probe_source_path(probe_dir, name)) if applicable else "",
        "primitive_object": str(primitive_probe_object_path(probe_dir, name)) if applicable else "",
        "primitive_command": [],
        "primitive_wrapped_command": "",
        "primitive_exit_code": None,
        "primitive_output": "",
        "primitive_detail": primitive_detail,
    }


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
        "backend_enablement": (
            "requires_explicit_RNS8_ENABLE_HIPBLASLT_build"
            if name == "hipblaslt"
            else "requires_explicit_RNS8_ENABLE_CK_build"
            if name == "ck"
            else "requires_explicit_RNS8_ENABLE_ROCWMMA_build"
            if name == "rocwmma"
            else "disabled"
        ),
        "enable_flag": ACCELERATOR_ENABLE_FLAGS[name],
        "enable_policy": (
            "explicit_opt_in_baseline_backend_with_exact_differentials"
            if name == "hipblaslt"
            else "explicit_opt_in_fused_backend_with_exact_differentials"
            if name in {"ck", "rocwmma"}
            else ACCELERATOR_ENABLE_POLICY
        ),
        "validated_correctness_backend": False,
        "can_enable_correctness_backend": name in {"hipblaslt", "ck", "rocwmma"} and requested,
        "evidence_class": CANDIDATE_ACCELERATOR_EVIDENCE_CLASS,
        "candidate_evidence_is_correctness_validation": False,
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
        **primitive_probe_not_run_fields(name, probe_dir, status, detail, requested),
    }


def compile_accelerator_primitive_probe(
    name: str,
    component: dict[str, object],
    root: Path,
    offload_target: str | None,
    hipcc: str | None,
) -> dict[str, object]:
    if name not in ACCELERATOR_PRIMITIVE_PROBE_SOURCES:
        return primitive_probe_not_run_fields(
            name,
            root,
            "NOT_APPLICABLE",
            "no matrix-engine primitive dependency probe is defined for this accelerator",
            False,
        )
    header = component.get("header")
    if not isinstance(header, str) or not header:
        return primitive_probe_not_run_fields(
            name,
            root,
            "NOT_RUN_MISSING_HEADER",
            "component header not discovered",
            True,
        )
    if not hipcc:
        return primitive_probe_not_run_fields(
            name,
            root,
            "NOT_RUN_MISSING_COMPILER",
            "hipcc not discovered",
            True,
        )

    source = primitive_probe_source_path(root, name)
    obj = primitive_probe_object_path(root, name)
    source.write_text(ACCELERATOR_PRIMITIVE_PROBE_SOURCES[name], encoding="utf-8")
    command = [hipcc, "-std=c++17", "-O2"]
    if offload_target:
        command.append(f"--offload-arch={offload_target}")
    command.extend(["-c", str(source)])
    for include_root in include_roots_for_component(name, component, root):
        command.extend(["-I", include_root])
    command.extend(["-o", str(obj)])
    actual_command, wrapped_command = command_with_windows_developer_environment(command)
    if actual_command is None:
        return primitive_probe_not_run_fields(
            name,
            root,
            "NOT_RUN_MISSING_COMPILER",
            "MSVC developer environment could not be discovered automatically",
            True,
        )
    code, output = run(actual_command, timeout=180)
    ok = code == 0
    return {
        "primitive_probe_requested": True,
        "primitive_probe_status": "OBJECT_COMPILE_PASS" if ok else "OBJECT_COMPILE_FAIL",
        "primitive_probe_ok": ok,
        "primitive_source": str(source),
        "primitive_object": str(obj),
        "primitive_command": command,
        "primitive_wrapped_command": wrapped_command,
        "primitive_exit_code": code,
        "primitive_output": compact_output(output),
        "primitive_detail": (
            "object-only int8 matrix-engine primitive probe compiled for the requested target; "
            "this is dependency evidence only, not runtime execution or correctness validation"
            if ok
            else "object-only int8 matrix-engine primitive probe failed to compile; backend remains disabled"
        ),
    }


def accelerator_compile_probes(
    accelerators: dict[str, dict[str, object]],
    requested: bool,
    probe_dir: Path | None,
    offload_target: str | None,
) -> dict[str, object]:
    root = probe_dir or (repo_root() / "temp" / "accelerator-deps" / "check-dependencies")
    items: dict[str, dict[str, object]] = {}
    if not requested:
        for name in ACCELERATOR_NAMES:
            items[name] = not_run_probe(
                name,
                root,
                "NOT_REQUESTED",
                "compile/link/run probe not requested; pass --accelerator-probes to run evidence probes under temp/accelerator-deps/",
                False,
            )
        return {
            "requested": False,
            "probe_root": str(root),
            "policy": "not run by default; probes never enable correctness backends or affect host readiness",
            "evidence_class": CANDIDATE_ACCELERATOR_EVIDENCE_CLASS,
            "candidate_evidence_is_correctness_validation": False,
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
        runtime_library = component.get("runtime_library") if isinstance(component, dict) else None
        if not isinstance(header, str) or not header:
            items[name] = not_run_probe(name, root, "NOT_RUN_MISSING_HEADER", "component header not discovered", True)
            continue
        if name == "hipblaslt" and (not isinstance(library, str) or not library):
            items[name] = not_run_probe(name, root, "NOT_RUN_MISSING_LIBRARY", "hipBLASLt library not discovered", True)
            continue

        source = root / f"{name}_probe.cpp"
        binary = probe_binary_path(root, name)
        source.write_text(ACCELERATOR_PROBE_SOURCES[name], encoding="utf-8")
        include_roots = include_roots_for_component(name, component, root)
        if name == "hipblaslt" and platform.system() == "Windows" and isinstance(library, str):
            command = msvc_probe_command(source, binary, include_roots[0] if include_roots else "", library)
            if not command:
                items[name] = not_run_probe(
                    name,
                    root,
                    "NOT_RUN_MISSING_COMPILER",
                    "MSVC developer environment could not be discovered automatically",
                    True,
                )
                continue
        else:
            if not hipcc:
                items[name] = not_run_probe(name, root, "NOT_RUN_MISSING_COMPILER", "hipcc not discovered", True)
                continue
            command = [hipcc, "-std=c++17"]
            if offload_target:
                command.append(f"--offload-arch={offload_target}")
            command.append(str(source))
            for include_root in include_roots:
                command.extend(["-I", include_root])
            if name == "hipblaslt" and isinstance(library, str):
                command.extend(["-L", str(Path(library).parent), "-lhipblaslt"])
            command.extend(["-o", str(binary)])
        actual_command: list[str] | str | None = command
        wrapped_command = command_line_for_report(command)
        if not isinstance(command, str):
            actual_command, wrapped_command = command_with_windows_developer_environment(command)
        if actual_command is None:
            items[name] = not_run_probe(
                name,
                root,
                "NOT_RUN_MISSING_COMPILER",
                "MSVC developer environment could not be discovered automatically",
                True,
            )
            continue
        code, output = run(actual_command, timeout=60)
        run_command: list[str] = []
        run_code: int | None = None
        run_output = ""
        if code == 0:
            run_command = [str(binary)]
            run_env = None
            if isinstance(runtime_library, str) and runtime_library:
                run_env = os.environ.copy()
                run_env["PATH"] = str(Path(runtime_library).parent) + os.pathsep + run_env.get("PATH", "")
            run_code, run_output = run(run_command, timeout=30, env=run_env)
        compiled = code == 0
        runtime_ok = run_code == 0
        if not compiled:
            status = "COMPILE_LINK_FAIL"
        elif runtime_ok:
            status = "COMPILE_LINK_PASS_RUN_PASS"
        else:
            status = "COMPILE_LINK_PASS_RUN_FAIL"
        primitive_probe = compile_accelerator_primitive_probe(name, component, root, offload_target, hipcc)
        items[name] = {
            "probe_requested": True,
            "requested": True,
            "status": status,
            "ok": compiled and runtime_ok,
            "compiled_probe_ok": compiled,
            "runtime_probe_ok": runtime_ok,
            "device_capability_ok": False,
            "backend_enablement": (
                "requires_explicit_RNS8_ENABLE_HIPBLASLT_build"
                if name == "hipblaslt"
                else "requires_explicit_RNS8_ENABLE_CK_build"
                if name == "ck"
                else "requires_explicit_RNS8_ENABLE_ROCWMMA_build"
                if name == "rocwmma"
                else "disabled"
            ),
            "enable_flag": ACCELERATOR_ENABLE_FLAGS[name],
            "enable_policy": (
                "explicit_opt_in_baseline_backend_with_exact_differentials"
                if name == "hipblaslt"
                else "explicit_opt_in_fused_backend_with_exact_differentials"
                if name in {"ck", "rocwmma"}
                else ACCELERATOR_ENABLE_POLICY
            ),
            "validated_correctness_backend": False,
            "can_enable_correctness_backend": name in {"hipblaslt", "ck", "rocwmma"} and compiled and runtime_ok,
            "evidence_class": CANDIDATE_ACCELERATOR_EVIDENCE_CLASS,
            "candidate_evidence_is_correctness_validation": False,
            "readiness_effect": "none",
            "source": str(source),
            "binary": str(binary),
            "command": command,
            "wrapped_command": wrapped_command,
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
            **primitive_probe,
        }
    return {
        "requested": True,
        "probe_root": str(root),
        "policy": "compile/link/run evidence only; probes never enable correctness backends or affect host readiness",
        "evidence_class": CANDIDATE_ACCELERATOR_EVIDENCE_CLASS,
        "candidate_evidence_is_correctness_validation": False,
        "items": items,
    }


