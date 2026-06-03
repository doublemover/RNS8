#!/usr/bin/env python3
"""Initialize and probe repo-local ROCm accelerator dependencies."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Sequence

import check_dependencies as deps
from msvc_env import command_in_msvc_environment, find_visual_studio_installation


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 120,
    use_msvc: bool = False,
) -> dict[str, object]:
    actual: Sequence[str] | str = [str(part) for part in command]
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
        )
        return {
            "command": list(command),
            "wrapped_command": actual if isinstance(actual, str) else subprocess.list2cmdline(actual),
            "exit_code": completed.returncode,
            "ok": completed.returncode == 0,
            "output": deps.compact_output(completed.stdout),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "wrapped_command": actual if isinstance(actual, str) else subprocess.list2cmdline(actual),
            "exit_code": 1,
            "ok": False,
            "output": str(exc),
        }


def initialize_submodules(root: Path) -> dict[str, object]:
    items: dict[str, object] = {}
    for name, spec in deps.EXPECTED_ROCM_SUBMODULES.items():
        path = str(spec["path"])
        result = run_command(["git", "submodule", "update", "--init", "--checkout", "--", path], cwd=root, timeout=600)
        report = deps.repo_local_dependency_report(name)
        items[name] = {
            "command": result,
            "dependency": report,
            "ok": bool(result["ok"]) and bool(report.get("sha_matches")),
        }
    return {
        "ok": all(bool(item["ok"]) for item in items.values() if isinstance(item, dict)),
        "items": items,
    }


def toolchain_report(target: str) -> dict[str, object]:
    commands = {
        name: deps.find_command(name)
        for name in ["git", "cmake", "ninja", "python", "hipcc", "hipInfo", "hipconfig"]
    }
    msvc_install = find_visual_studio_installation()
    msvc_probe = run_command(["where", "cl"], cwd=deps.repo_root(), timeout=60, use_msvc=True)
    hip_info = deps.hip_info_report(commands.get("hipInfo"))
    hip_target = hip_info.get("target") if isinstance(hip_info, dict) else ""
    return {
        "ok": all(commands[name] for name in ["git", "cmake", "ninja", "python", "hipcc"])
        and bool(msvc_install)
        and bool(msvc_probe["ok"])
        and (not hip_target or hip_target == target),
        "commands": commands,
        "msvc_install": str(msvc_install) if msvc_install else "",
        "msvc_probe": msvc_probe,
        "hip_info": hip_info,
        "requested_target": target,
        "target_matches_hip_info": hip_target == target if hip_target else False,
    }


def compile_dependency_probe(name: str, target: str, probe_root: Path) -> dict[str, object]:
    accelerators = deps.accelerator_components()
    component = accelerators.get(name)
    if not isinstance(component, dict):
        return {
            "ok": False,
            "status": "missing_component_record",
            "name": name,
            **deps.primitive_probe_not_run_fields(
                name, probe_root, "NOT_RUN_MISSING_COMPONENT", "component record not discovered", True
            ),
        }
    header = component.get("header")
    if not isinstance(header, str) or not header:
        return {
            "ok": False,
            "status": "missing_header",
            "name": name,
            "component": component,
            **deps.primitive_probe_not_run_fields(
                name, probe_root, "NOT_RUN_MISSING_HEADER", "component header not discovered", True
            ),
        }
    hipcc = deps.find_command("hipcc")
    if not hipcc:
        return {
            "ok": False,
            "status": "missing_hipcc",
            "name": name,
            "component": component,
            **deps.primitive_probe_not_run_fields(
                name, probe_root, "NOT_RUN_MISSING_COMPILER", "hipcc not discovered", True
            ),
        }

    probe_root.mkdir(parents=True, exist_ok=True)
    source = probe_root / f"{name}_probe.cpp"
    binary = probe_root / f"{name}_probe{'.exe' if platform.system() == 'Windows' else ''}"
    source.write_text(deps.ACCELERATOR_PROBE_SOURCES[name], encoding="utf-8")
    command = [hipcc, "-std=c++17", f"--offload-arch={target}", str(source)]
    for include_root in deps.include_roots_for_component(name, component, probe_root):
        command.extend(["-I", include_root])
    command.extend(["-o", str(binary)])
    result = run_command(command, cwd=deps.repo_root(), timeout=120, use_msvc=True)
    primitive = deps.compile_accelerator_primitive_probe(name, component, probe_root, target, hipcc)
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
            "backend enable flags remain fail-fast"
        ),
    }


def dry_run_bootstrap_report(root: Path, target: str, probe_root: Path, init: bool, probe: bool) -> dict[str, object]:
    submodule_items: dict[str, object] = {}
    for name, spec in deps.EXPECTED_ROCM_SUBMODULES.items():
        path = str(spec["path"])
        dependency = deps.repo_local_dependency_report(name)
        submodule_items[name] = {
            "ok": True,
            "dependency": dependency,
            "planned_command": ["git", "submodule", "update", "--init", "--checkout", "--", path] if init else [],
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
        "policy": "repo-local dependencies only; no source clones or installs under C:\\",
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
            print(f"  {name}: {'OK' if item['ok'] else 'FAIL'} {dependency.get('actual_sha') or 'not initialized'}")
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
    parser.add_argument("--init", action="store_true", help="initialize/update CK and rocWMMA submodules")
    parser.add_argument("--probe", action="store_true", help="compile repo-local CK and rocWMMA dependency probes")
    parser.add_argument("--target", default="gfx1100", help="AMDGPU offload target for probes")
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

    root = deps.repo_root()
    probe_root = args.probe_root or (root / "temp" / "accelerator-deps" / "bootstrap")
    if args.dry_run:
        report = dry_run_bootstrap_report(root, args.target, probe_root, args.init, args.probe)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print_human(report)
            print(f"record: {probe_root / 'bootstrap_rocm_accelerators.json'} (not written; dry-run)")
        return 0

    report: dict[str, object] = {
        "repo_root": str(root),
        "target": args.target,
        "artifact_root": str(probe_root),
        "policy": "repo-local dependencies only; no source clones or installs under C:\\",
    }
    ok = True

    if args.init:
        submodules = initialize_submodules(root)
        report["submodules"] = submodules
        ok = ok and bool(submodules["ok"])
    else:
        report["submodules"] = {
            "ok": all(bool(deps.repo_local_dependency_report(name).get("sha_matches")) for name in ("ck", "rocwmma")),
            "items": {
                name: {"dependency": deps.repo_local_dependency_report(name), "ok": deps.repo_local_dependency_report(name).get("sha_matches")}
                for name in ("ck", "rocwmma")
            },
            "note": "submodule update not requested; pass --init to initialize missing dependencies",
        }

    toolchain = toolchain_report(args.target)
    report["toolchain"] = toolchain
    ok = ok and bool(toolchain["ok"])

    if args.probe:
        probes = probe_dependencies(args.target, probe_root)
        report["probes"] = probes
        ok = ok and bool(probes["ok"])
    else:
        report["probes"] = {"ok": True, "requested": False, "note": "compile probes not requested; pass --probe"}

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
