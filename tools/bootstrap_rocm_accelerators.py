#!/usr/bin/env python3
"""Initialize and probe repo-local ROCm accelerator dependencies."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Sequence

import check_dependencies_lib.accelerators as dependency_accelerators
import check_dependencies_lib.config as dependency_config
import check_dependencies_lib.discovery as dependency_discovery
import check_dependencies_lib.system as dependency_system
import rocm_sdk
from msvc_env import command_in_msvc_environment, find_visual_studio_installation
from check_dependencies_lib.resolution import Request, resolve, selection_context
from check_dependencies_lib.resolution import ACTIVE_RESOLUTION, compiler_environment


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 120,
    use_msvc: bool = False,
) -> dict[str, object]:
    original = [str(part) for part in command]
    actual: Sequence[str] | str = original
    environment = None
    active = ACTIVE_RESOLUTION.get()
    selected = active.get("hip", {}).get("selected") if active else None
    if (
        selected
        and original
        and Path(original[0]).resolve() == Path(selected["paths"]["compiler"]).resolve()
    ):
        environment = {**os.environ, **compiler_environment(active)}
    try:
        if use_msvc and platform.system() == "Windows":
            actual, _ = command_in_msvc_environment(actual)
    except RuntimeError as exc:
        return {
            "command": list(command),
            "wrapped_command": "",
            "exit_code": 1,
            "ok": False,
            "output": str(exc),
        }

    try:
        completed = subprocess.run(
            actual,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
            shell=isinstance(actual, str),
            env=environment,
        )
        return {
            "command": list(command),
            "wrapped_command": actual
            if isinstance(actual, str)
            else subprocess.list2cmdline(actual),
            "exit_code": completed.returncode,
            "ok": completed.returncode == 0,
            "output": dependency_system.compact_output(completed.stdout),
            "selected_hip_environment": compiler_environment(active)
            if environment is not None
            else {},
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "wrapped_command": actual
            if isinstance(actual, str)
            else subprocess.list2cmdline(actual),
            "exit_code": 1,
            "ok": False,
            "output": str(exc),
        }


def _checkout_problem(checkout: Path) -> str:
    status, changes = dependency_discovery.git_text(
        ["status", "--porcelain=v1", "--untracked-files=normal"], cwd=checkout
    )
    if status != 0:
        return "Cannot inspect dependency worktree; preserving checkout"
    if changes:
        return "Dependency contains user changes/untracked files; preserving checkout"
    return ""


def initialize_submodules(root: Path) -> dict[str, object]:
    items: dict[str, object] = {}
    for name, spec in dependency_config.EXPECTED_ROCM_SUBMODULES.items():
        path = str(spec["path"])
        checkout = root / path
        before = dependency_discovery.repo_local_dependency_report(name)
        blocked = ""
        if (
            checkout.is_symlink()
            or checkout.is_junction()
            or not checkout.resolve().is_relative_to(root.resolve())
        ):
            blocked = "Dependency checkout is a link or escapes the repository"
        elif not before.get("url_matches"):
            blocked = "Dependency URL differs from the reviewed source pin"
        elif (checkout / ".git").exists():
            blocked = _checkout_problem(checkout)
            if not blocked and before.get("sha_matches"):
                items[name] = {
                    "command": None,
                    "dependency": before,
                    "ok": True,
                    "action": "reused_clean_reviewed_pin",
                }
                continue
        elif checkout.exists() and (not checkout.is_dir() or any(checkout.iterdir())):
            blocked = "Nonempty foreign dependency destination; preserving files"
        if blocked:
            items[name] = {
                "command": None,
                "dependency": before,
                "ok": False,
                "action": "preserved_not_initialized",
                "detail": blocked,
            }
            continue
        result = run_command(
            ["git", "submodule", "update", "--init", "--checkout", "--", path],
            cwd=root,
            timeout=600,
        )
        report = dependency_discovery.repo_local_dependency_report(name)
        problem = _checkout_problem(checkout) if result["ok"] else "Initialization command failed"
        installed = (
            bool(result["ok"])
            and bool(report.get("sha_matches"))
            and bool(report.get("url_matches"))
            and not problem
        )
        items[name] = {
            "command": result,
            "dependency": report,
            "ok": installed,
            "worktree_clean": not problem if result["ok"] else None,
            "detail": problem,
            "action": "initialized_reviewed_pin"
            if installed
            else "initialization_failed_preserved",
        }
    return {
        "ok": all(bool(item["ok"]) for item in items.values() if isinstance(item, dict)),
        "items": items,
    }


def toolchain_report(target: str) -> dict[str, object]:
    host = platform.system()
    resolution = resolve(
        Request(
            host,
            overrides={"RNS8_AMDGPU_TARGETS": target},
            source_root=str(dependency_system.repo_root()),
            require_hip=True,
            require_vcpkg=host == "Windows",
        )
    )
    with selection_context(resolution):
        commands = {
            name: dependency_system.find_command(name)
            for name in ["git", "cmake", "ninja", "python", "hipcc", "hipInfo", "hipconfig"]
        }
    msvc_install = find_visual_studio_installation() if host == "Windows" else None
    msvc_probe = {"ok": False, "status": "not_applicable_native_linux", "command": []}
    native_compiler_probe = {"ok": False, "status": "not_run_missing_host_compiler", "command": []}
    with selection_context(resolution):
        if host == "Windows":
            msvc_probe = run_command(
                ["where", "cl"], cwd=dependency_system.repo_root(), timeout=60, use_msvc=True
            )
        else:
            compiler = resolution["tools"]["host_cxx"].get("path")
            if host == "Linux" and compiler:
                native_compiler_probe = run_command(
                    [compiler, "--version"], cwd=dependency_system.repo_root(), timeout=30
                )
    hip_info = (
        dependency_discovery.hip_info_report(commands.get("hipInfo"))
        if host == "Windows"
        else {"ok": False, "status": "not_run_native_linux_inventory", "target": None}
    )
    hip_target = hip_info.get("target") if isinstance(hip_info, dict) else ""
    return {
        "ok": not resolution["errors"]
        and all(commands[name] for name in ["git", "cmake", "ninja", "python", "hipcc"])
        and (
            (bool(msvc_install) and bool(msvc_probe["ok"]))
            if host == "Windows"
            else host == "Linux" and bool(native_compiler_probe["ok"])
        ),
        "commands": commands,
        "dependency_resolution": resolution,
        "msvc_install": str(msvc_install) if msvc_install else "",
        "msvc_probe": msvc_probe,
        "native_compiler_probe": native_compiler_probe,
        "host_platform": host,
        "host_environment": resolution.get("host_environment", {"kind": "unknown"}),
        "hip_info": hip_info,
        "requested_target": target,
        "target_matches_hip_info": hip_target == target if hip_target else False,
        "target_match_is_toolchain_gate": False,
        "validation_scope": "Native tool discovery/version probes only; a different or absent local GPU does not prevent target compilation and is not runtime qualification",
    }


def compile_dependency_probe(name: str, target: str, probe_root: Path) -> dict[str, object]:
    probe_root = probe_root.resolve()
    accelerators = dependency_accelerators.accelerator_components()
    component = accelerators.get(name)
    if not isinstance(component, dict):
        return {
            "ok": False,
            "status": "missing_component_record",
            "name": name,
            **dependency_accelerators.primitive_probe_not_run_fields(
                name,
                probe_root,
                "NOT_RUN_MISSING_COMPONENT",
                "component record not discovered",
                True,
            ),
        }
    header = component.get("header")
    if not isinstance(header, str) or not header:
        return {
            "ok": False,
            "status": "missing_header",
            "name": name,
            "component": component,
            **dependency_accelerators.primitive_probe_not_run_fields(
                name, probe_root, "NOT_RUN_MISSING_HEADER", "component header not discovered", True
            ),
        }
    hipcc = dependency_system.find_command("hipcc")
    if not hipcc:
        return {
            "ok": False,
            "status": "missing_hipcc",
            "name": name,
            "component": component,
            **dependency_accelerators.primitive_probe_not_run_fields(
                name, probe_root, "NOT_RUN_MISSING_COMPILER", "hipcc not discovered", True
            ),
        }

    probe_root.mkdir(parents=True, exist_ok=True)
    source = probe_root / f"{name}_probe.cpp"
    binary = probe_root / f"{name}_probe{'.exe' if platform.system() == 'Windows' else ''}"
    source.write_text(dependency_config.ACCELERATOR_PROBE_SOURCES[name], encoding="utf-8")
    command = [hipcc, "-std=c++17", f"--offload-arch={target}", str(source)]
    for include_root in dependency_accelerators.include_roots_for_component(
        name, component, probe_root
    ):
        command.extend(["-I", include_root])
    command.extend(["-o", str(binary)])
    result = run_command(command, cwd=dependency_system.repo_root(), timeout=120, use_msvc=True)
    primitive = dependency_accelerators.compile_accelerator_primitive_probe(
        name, component, probe_root, target, hipcc
    )
    header_ok = bool(result["ok"])
    primitive_ok = bool(primitive.get("primitive_probe_ok"))
    if header_ok and primitive_ok:
        status = "compile_probe_pass_primitive_probe_pass"
    elif header_ok:
        status = "compile_probe_pass_primitive_probe_fail"
    else:
        status = "compile_probe_fail"
    return {
        "ok": header_ok and primitive_ok,
        "status": status,
        "name": name,
        "source": str(source),
        "binary": str(binary),
        "component": component,
        "compile": result,
        **primitive,
    }


def probe_dependencies(target: str, probe_root: Path) -> dict[str, object]:
    items = {
        "ck": compile_dependency_probe("ck", target, probe_root),
        "rocwmma": compile_dependency_probe("rocwmma", target, probe_root),
    }
    return {
        "ok": all(bool(item["ok"]) for item in items.values()),
        "target": target,
        "probe_root": str(probe_root),
        "items": items,
        "policy": (
            "dependency compile/run probes plus object-only int8 primitive compile probes; "
            "backend runtime enablement remains disabled until exact kernels pass target validation"
        ),
    }


def dry_run_bootstrap_report(
    root: Path, target: str, probe_root: Path, init: bool, probe: bool
) -> dict[str, object]:
    submodule_items: dict[str, object] = {}
    for name, spec in dependency_config.EXPECTED_ROCM_SUBMODULES.items():
        path = str(spec["path"])
        dependency = dependency_discovery.repo_local_dependency_report(name)
        submodule_items[name] = {
            "ok": True,
            "dependency": dependency,
            "planned_command": ["git", "submodule", "update", "--init", "--checkout", "--", path]
            if init
            else [],
            "planned_write": "submodule checkout/update" if init else "",
        }

    probe_items: dict[str, object] = {}
    if probe:
        for name in ("ck", "rocwmma"):
            suffix = ".exe" if platform.system() == "Windows" else ""
            probe_items[name] = {
                "ok": True,
                "status": "dry_run_probe_planned",
                "name": name,
                "source": str(probe_root / f"{name}_probe.cpp"),
                "binary": str(probe_root / f"{name}_probe{suffix}"),
                "primitive_probe_status": "DRY_RUN_PLANNED",
                "primitive_probe_ok": None,
                "planned_write": "probe source, probe binary, and primitive probe artifacts",
            }

    record_path = probe_root / "bootstrap_rocm_accelerators.json"
    return {
        "repo_root": str(root),
        "target": target,
        "artifact_root": str(probe_root),
        "record_path": str(record_path),
        "record_written": False,
        "dry_run": True,
        "policy": "Pinned repository submodules only; no system packages, driver changes or global environment edits",
        "planned_actions": {
            "submodule_update": bool(init),
            "compile_probes": bool(probe),
            "write_report": False,
            "note": "dry-run reports planned actions without creating directories, updating submodules, writing probe files, compiling, or writing the JSON record",
        },
        "submodules": {
            "ok": True,
            "dry_run": True,
            "items": submodule_items,
            "note": "submodule update planned only" if init else "submodule update not requested",
        },
        "toolchain": {
            "ok": True,
            "dry_run": True,
            "requested": False,
            "requested_target": target,
            "note": "toolchain probes not executed in dry-run mode",
        },
        "probes": {
            "ok": True,
            "dry_run": True,
            "requested": bool(probe),
            "target": target,
            "probe_root": str(probe_root),
            "items": probe_items,
            "note": "compile probes planned only" if probe else "compile probes not requested",
        },
    }


def print_human(report: dict[str, object]) -> None:
    print("RNS8 ROCm accelerator dependency bootstrap")
    print("==========================================")
    if report.get("dry_run"):
        print("mode: dry-run")
        planned = report.get("planned_actions")
        if isinstance(planned, dict):
            print(f"planned submodule update: {planned.get('submodule_update')}")
            print(f"planned compile probes: {planned.get('compile_probes')}")
            print(f"planned report write: {planned.get('write_report')}")
    if report.get("submodules"):
        submodules = report["submodules"]
        assert isinstance(submodules, dict)
        print(f"submodules: {'OK' if submodules['ok'] else 'FAIL'}")
        items = submodules["items"]
        assert isinstance(items, dict)
        for name, item in items.items():
            assert isinstance(item, dict)
            dependency = item["dependency"]
            assert isinstance(dependency, dict)
            print(
                f"  {name}: {'OK' if item['ok'] else 'FAIL'} {dependency.get('actual_sha') or 'not initialized'}"
            )
    toolchain = report.get("toolchain")
    if isinstance(toolchain, dict):
        print(f"toolchain: {'OK' if toolchain['ok'] else 'FAIL'}")
        print(f"  target: {toolchain.get('requested_target')}")
        print(f"  MSVC: {toolchain.get('msvc_install') or 'not found'}")
        commands = toolchain.get("commands")
        if isinstance(commands, dict):
            for name in sorted(commands):
                print(f"  {name}: {commands[name] or 'not found'}")
    probes = report.get("probes")
    if isinstance(probes, dict):
        print(f"probes: {'OK' if probes['ok'] else 'FAIL'}")
        items = probes.get("items")
        if isinstance(items, dict):
            for name, item in items.items():
                assert isinstance(item, dict)
                print(f"  {name}: {item.get('status')}")
                print(
                    "    primitive: "
                    f"{item.get('primitive_probe_status')} ok={item.get('primitive_probe_ok')}"
                )
                compile_result = item.get("compile")
                if isinstance(compile_result, dict) and compile_result.get("output"):
                    print(f"    output: {compile_result['output']}")
                primitive_output = item.get("primitive_output")
                if primitive_output:
                    print(f"    primitive output: {primitive_output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init", action="store_true", help="initialize/update CK and rocWMMA submodules"
    )
    parser.add_argument(
        "--probe", action="store_true", help="compile repo-local CK and rocWMMA dependency probes"
    )
    parser.add_argument(
        "--target",
        default=None,
        help="AMDGPU offload target for probes (default gfx1100); also selects an SDK target",
    )
    rocm_sdk.add_arguments(parser)
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=None,
        help="probe artifact directory; defaults to temp/accelerator-deps/bootstrap",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned submodule/probe/report actions without writing, compiling, or updating submodules",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    if args.sdk_auto or args.sdk_install or args.sdk_root or args.sdk_build or args.sdk_test:
        if not (args.sdk_auto or args.sdk_install or args.sdk_root):
            parser.error("--sdk-build/--sdk-test requires --sdk-auto, --sdk-install or --sdk-root")
        if args.init or args.probe:
            parser.error(
                "SDK provisioning/build and repo-local submodule probes are separate explicit invocations"
            )
        try:
            report = rocm_sdk.run(args)
        except (
            OSError,
            ValueError,
            tarfile.TarError,
            zipfile.BadZipFile,
            subprocess.SubprocessError,
        ) as error:
            print(json.dumps({"schema": "rns8_rocm_sdk_init_v1", "error": str(error)}))
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    args.target = args.target or "gfx1100"
    try:
        dependency_accelerators._target_base(args.target)
    except ValueError as error:
        parser.error(str(error))

    root = dependency_system.repo_root()
    probe_root = (args.probe_root or (root / "temp" / "accelerator-deps" / "bootstrap")).resolve()
    if args.dry_run:
        report = dry_run_bootstrap_report(root, args.target, probe_root, args.init, args.probe)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print_human(report)
            print(
                f"record: {probe_root / 'bootstrap_rocm_accelerators.json'} (not written; dry-run)"
            )
        return 0

    report: dict[str, object] = {
        "repo_root": str(root),
        "target": args.target,
        "artifact_root": str(probe_root),
        "policy": "Pinned repository submodules only; no system packages, driver changes or global environment edits",
    }
    ok = True

    if args.init:
        submodules = initialize_submodules(root)
        report["submodules"] = submodules
        ok = ok and bool(submodules["ok"])
    else:
        report["submodules"] = {
            "ok": all(
                bool(dependency_discovery.repo_local_dependency_report(name).get("sha_matches"))
                for name in ("ck", "rocwmma")
            ),
            "items": {
                name: {
                    "dependency": dependency_discovery.repo_local_dependency_report(name),
                    "ok": dependency_discovery.repo_local_dependency_report(name).get(
                        "sha_matches"
                    ),
                }
                for name in ("ck", "rocwmma")
            },
            "note": "submodule update not requested; pass --init to initialize missing dependencies",
        }

    toolchain = toolchain_report(args.target)
    report["toolchain"] = toolchain
    ok = ok and bool(toolchain["ok"])

    if args.probe and toolchain["ok"]:
        with selection_context(toolchain["dependency_resolution"]):
            probes = probe_dependencies(args.target, probe_root)
        report["probes"] = probes
        ok = ok and bool(probes["ok"])
    elif args.probe:
        report["probes"] = {
            "ok": False,
            "requested": True,
            "status": "not_run_invalid_toolchain",
            "note": "Repair the shared toolchain selection before compilation; no fallback SDK was probed",
        }
    else:
        report["probes"] = {
            "ok": True,
            "requested": False,
            "note": "compile probes not requested; pass --probe",
        }

    output_path = probe_root / "bootstrap_rocm_accelerators.json"
    report["record_path"] = str(output_path)
    probe_root.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
        print(f"record: {output_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
