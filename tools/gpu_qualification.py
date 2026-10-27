#!/usr/bin/env python3
"""Plan or run bounded GPU qualification. Default is plan-only; never promotes a cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import math
import re
import subprocess
import shutil
import signal
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from process_progress import run_capture
from session_artifacts import load_inventory, inspect_cache, stage_cache
from check_dependencies_lib.resolution import Request, REQUEST_KEYS, resolve
from session_environment import build_environment, execution_blockers, isolated_environment, snapshot

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("hip-direct", "amdgpu-builtins", "hipblaslt", "ck", "rocwmma")
FLAGS = {"amdgpu-builtins": "AMDGPU_BUILTINS", "hipblaslt": "HIPBLASLT", "ck": "CK", "rocwmma": "ROCWMMA"}
TARGETS = {"mi300x": ("gfx942", "Linux", "linux-cdna-release"),
           "7900xtx": ("gfx1100", "Windows", "windows-msvc-hip-release")}


def kernel_inventory(root: Path = ROOT) -> list[dict]:
    kernels = []
    for path in sorted((root / "src").rglob("*")):
        if path.suffix not in {".hip", ".cuh", ".inc"}:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"__global__\s+(?:__launch_bounds__\([^\n]*?\)\s+)?void\s+(\w+)\s*\(", text):
            kernels.append({"name": match[1], "source": path.relative_to(root).as_posix(),
                            "line": text.count("\n", 0, match.start()) + 1,
                            "evidence": "source_only_not_hardware_qualified"})
    return kernels


def command_plan(target: str, backends: list[str], out: Path, jobs: int, devices: list[int],
                 *, bindings: dict | None = None, session_kind: str = "first-pass",
                 timeout: float = 1800, preset_override: str | None = None, all_enabled: bool = False) -> list[dict]:
    arch, system, preset = TARGETS[target]
    if session_kind != "prepare" and (not devices or len(set(devices)) != len(devices) or any(type(device) is not int or device < 0 for device in devices)):
        raise ValueError("devices must be unique explicit nonnegative runtime ordinals")
    extension = ".exe" if system == "Windows" else ""
    python = sys.executable if platform.system() == system else "python" if system == "Windows" else "python3"
    wrapper = [python, "tools/windows_dev.py"] if system == "Windows" else []
    commands = []

    def add(name, argv, **checks):
        commands.append({"name": name, "argv": argv, "depends_on": [commands[-1]["name"]] if commands else [],
                         "timeout_seconds": timeout, **checks})

    if session_kind == "environment":
        return commands

    if session_kind == "prepare":
        if system != "Linux" or devices or backends or all_enabled:
            raise ValueError("CPU preparation requires Linux, no devices and no backend lanes")
        build = out / "build" / "cpu-preparation"
        bound = [f"-D{key}={value}" for key, value in sorted((bindings or {}).items())]
        options = [f"-DRNS8_ENABLE_{flag}=OFF" for flag in FLAGS.values()]
        add("prepare-configure", ["cmake", "--preset", preset_override or "linux-cpu-release", "-B", str(build),
            *bound, "-DRNS8_ENABLE_HIP=OFF", "-DRNS8_PROBE_ACCELERATORS=OFF", *options,
            "-DRNS8_BUILD_TESTS=ON", "-DRNS8_BUILD_EXAMPLES=ON", "-DRNS8_BUILD_TOOLS=ON",
            "-DRNS8_BUILD_BENCHMARKS=ON", "-DRNS8_WORKLOAD_GPU_TESTS=OFF",
            "-DVCPKG_MANIFEST_INSTALL=OFF", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"], build_dir=str(build))
        add("prepare-build", ["cmake", "--build", str(build), "--parallel", str(jobs)], build_dir=str(build))
        junit = out / "cpu-preparation.xml"
        add("prepare-cpu-tests", ["ctest", "--test-dir", str(build), "--output-on-failure", "--parallel", "1",
            "--timeout", "600", "--no-tests=error", "--output-junit", str(ROOT / junit)],
            build_dir=str(build), preparation_junit=str(junit))
        return commands

    lanes = [(backend, [backend]) for backend in backends]
    if all_enabled:
        lanes.append(("all-enabled", list(BACKENDS)))
    for lane, enabled_backends in lanes:
        # Every isolated or combined feature lane has its own build identity.
        build = out / "build" / lane
        binary = lambda name: str(build / (name + extension))
        options = [f"-DRNS8_ENABLE_{flag}={'ON' if name in enabled_backends else 'OFF'}" for name, flag in FLAGS.items()]
        bound = [f"-D{key}={value}" for key, value in sorted((bindings or {}).items())]
        add(f"{lane}-configure", wrapper + ["cmake", "--preset", preset_override or preset, "-B", str(build),
            *bound, f"-DRNS8_AMDGPU_TARGETS={arch}", "-DRNS8_ENABLE_HIP=ON", "-DVCPKG_MANIFEST_INSTALL=OFF",
            "-DRNS8_PROBE_ACCELERATORS=OFF", *options], device=None, build_dir=str(build))
        add(f"{lane}-build", wrapper + ["cmake", "--build", str(build), "--parallel", str(jobs)], build_dir=str(build))
        for backend_index, backend in enumerate(enabled_backends):
            prefix = lane + "-" + backend if len(enabled_backends) > 1 else backend
            # Parent runtime ordinals and child-visible zero remain distinct.
            for device in devices:
                add(f"{prefix}-device{device}-parent-inspect", [binary("rns8-inspect"), "--backend", backend,
                    "--device", str(device), "--json"], inspect_target=arch, inspect_backend=backend,
                    inspect_device=device, parent_inspect=True, device=None, selected_device=device)
                child = device_command_plan(backend, device, arch, binary, python, wrapper, build,
                    out, session_kind, stage_prefix=prefix, full_tests=backend_index == 0)
                for name, argv, checks in child:
                    add(name, argv, device=device, selected_device=device, **checks)
    return commands


def device_command_plan(backend, device, arch, binary, python, wrapper, build, out, session_kind, *, stage_prefix=None, full_tests=True):
    commands = []
    stem = f"{stage_prefix or backend}-device{device}"
    def add(name, argv, **checks):
        commands.append((stem + "-" + name, argv, checks))
    add("inspect", [binary("rns8-inspect"), "--backend", backend, "--device", "0", "--json"],
        inspect_target=arch, inspect_backend=backend, inspect_device=0,
        match_parent=stem + "-parent-inspect")
    # Unlike an all-skipped CTest run, this fails if there is no HIP device.
    add("smoke", [binary("rns8-verify"), "--hip-smoke"])
    if session_kind != "smoke" and full_tests:
        junit = out / f"{stem}-qualification.xml"
        add("regressions", [binary("rns8_tests"), "[qualification]", "--reporter", "junit", "--out", str(junit)],
            junit=str(junit), require_test="qualification pack covers dispatch thresholds and plane tails")
        add("ctest", wrapper + ["ctest", "--test-dir", str(build), "--output-on-failure",
            "--parallel", "1", "--timeout", "600", "--no-tests=error"])
    # Small captures are smoke evidence only, with bounded runtime per command.
    cases = [("bounded-i64", []), ("bounded-u64", []),
                 ("exact-wide-signed", ["--exact-wide-limbs", "3"]),
                 ("exact-wide-unsigned", ["--exact-wide-limbs", "3"]),
                 ("finite-u8-ring", ["--modulus", "255"]),
                 ("finite-u8-field", ["--modulus", "251"])]
    if session_kind == "smoke":
        cases = cases[:1]
    elif backend == "hip-direct":
        cases += [("wrap-u64", []), ("finite-u8-ring", ["--modulus", "256"])]
    for index, (semantics, extra) in enumerate(cases):
            name = f"{backend}-{index}-{semantics}"
            capture = out / f"{stem}-{name}.json"
            add(name, [binary("rns8-bench"), "--backend", backend, "--semantics", semantics,
                "--m", "32", "--n", "32", "--k", "64", "--warmups", "1", "--repeats", "3",
                "--seed", "20261006", "--cpu-threads", "1", "--device", "0", "--progress", *extra], capture=str(capture))
            add(name + "-schema", [python, "tools/benchmark_schema.py", str(capture)])
    if backend == "hip-direct" and session_kind != "smoke":
            # Existing graph capture, not the removed dead global graph helpers.
            name = "hip-direct-graph"
            capture = out / f"{stem}-{name}.json"
            add(name, [binary("rns8-bench"), "--backend", "hip-direct", "--semantics", "bounded-i64",
                "--m", "32", "--n", "32", "--k", "64", "--hip-graph-replay",
                "--warmups", "1", "--repeats", "3", "--seed", "20261006", "--device", "0", "--progress"], capture=str(capture))
            add(name + "-schema", [python, "tools/benchmark_schema.py", str(capture)])
    return commands


def validate_result(item: dict, result: dict, parent_info: dict | None = None) -> list[str]:
    errors = []
    if result["returncode"] != 0 or result["timed_out"]:
        return ["command failed or timed out"]
    if "inspect_target" in item:
        try:
            info = json.loads(result["stdout"])
            if info.get("gcn_arch", "").split(":")[0] != item["inspect_target"]:
                errors.append("runtime GPU target does not match the manifest")
            if info.get("backend") != item["inspect_backend"]:
                errors.append("runtime backend does not match the manifest")
            # rns8_device_info uses uint32_t and rns8-inspect emits numeric 0/1.
            # Accept the earlier Boolean fixtures, but not arbitrary truthy values.
            available = info.get("hip_available")
            if not (available is True or (type(available) is int and available == 1)) or type(info.get("global_mem_bytes")) is not int or info["global_mem_bytes"] <= 0:
                errors.append("runtime inspection did not establish an available GPU with positive memory")
            if info.get("device_id") != item["inspect_device"] or type(info.get("device_id")) is not int:
                errors.append("runtime ordinal does not match the explicitly selected inspection ordinal")
            if not isinstance(info.get("name"), str) or not info["name"].strip():
                errors.append("runtime GPU name is missing")
            if "match_parent" in item:
                if parent_info is None:
                    errors.append("parent runtime inspection is missing")
                elif any(info.get(key) != parent_info.get(key) for key in
                         ("name", "gcn_arch", "global_mem_bytes", "hip_runtime_version", "hip_driver_version", "backend")):
                    errors.append("isolated logical device does not match selected parent runtime evidence")
        except (ValueError, AttributeError, TypeError):
            errors.append("invalid device inspection JSON")
    if "junit" in item:
        try:
            root = ET.parse(ROOT / item["junit"]).getroot()
            cases = list(root.iter("testcase"))
            if not any(case.get("name") == item["require_test"] for case in cases):
                errors.append("required hardware regression missing (possibly a CPU-only build)")
            if any(list(case.iter("skipped")) or list(case.iter("failure")) or list(case.iter("error")) for case in cases):
                errors.append("hardware qualification tests failed or skipped")
        except (OSError, ET.ParseError):
            errors.append("missing or invalid qualification JUnit report")
    if "preparation_junit" in item:
        try:
            root = ET.parse(ROOT / item["preparation_junit"]).getroot()
            cases = list(root.iter("testcase"))
            if not cases or not any(not list(case.iter("skipped")) for case in cases):
                errors.append("CPU preparation report has no executed tests")
            if any(list(case.iter("failure")) or list(case.iter("error")) for case in cases):
                errors.append("CPU preparation tests failed")
        except (OSError, ET.ParseError):
            errors.append("missing or invalid CPU preparation JUnit report")
    return errors


def device_list(value: str) -> list[int]:
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise argparse.ArgumentTypeError("--devices requires comma-separated nonnegative runtime ordinals; no all/default selection")
    devices = [int(part) for part in value.split(",")]
    if len(set(devices)) != len(devices) or len(devices) > 64 or any(device > 65535 for device in devices):
        raise argparse.ArgumentTypeError("device list contains duplicates or exceeds the bounded 64-device/65535-ordinal limit")
    return devices


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_receipt(exclude: Path | None = None) -> dict:
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, check=True, timeout=30).stdout
    # Include untracked first-party source; git diff alone misses newly authored
    # files. Ignored build/temp evidence and external gitlink contents stay out.
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    return {"git_revision": git("rev-parse", "HEAD").decode().strip(),
            "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
            "untracked_files": {name: sha256(ROOT / name) for name in untracked if name and (ROOT / name).is_file()
                                and (exclude is None or not (ROOT / name).resolve().is_relative_to(exclude.resolve()))},
            "runner_sha256": sha256(Path(__file__)), "environment_helper_sha256": sha256(ROOT / "tools/session_environment.py"),
            "artifact_helper_sha256": sha256(ROOT / "tools/session_artifacts.py"),
            "dependency_policy_sha256": sha256(ROOT / "metadata/dependencies.json")}


def dependency_request(target: str, backends: list[str], definitions: list[str], *, preparation: bool = False) -> Request:
    overrides = {"RNS8_AMDGPU_TARGETS": TARGETS[target][0]}
    for definition in definitions:
        if "=" not in definition:
            raise ValueError("--set expects a supported dependency KEY=VALUE")
        key, value = definition.split("=", 1)
        if key not in REQUEST_KEYS or key in overrides or not value or re.search(r"[\x00-\x1f]|https?://", value, re.I):
            raise ValueError("unsupported, duplicate or unsafe dependency override: " + key)
        overrides[key] = value
    features = {"RNS8_ENABLE_HIP": not preparation, **{f"RNS8_ENABLE_{flag}": name in backends for name, flag in FLAGS.items()}}
    return Request(TARGETS[target][1], overrides, features, str(ROOT), require_hip=not preparation,
                   require_vcpkg=TARGETS[target][1] == "Windows")


def readiness_blockers(args, resolution: dict, environment: dict) -> list[str]:
    blockers = list(resolution["errors"])
    if getattr(args, "session_kind", None) != "prepare":
        blockers += execution_blockers(environment)
    if platform.system() != TARGETS[args.target][1]:
        blockers.append("execution requires target operating system " + TARGETS[args.target][1])
    for name in ("cmake", "ninja", "host_cxx"):
        if not resolution["tools"][name]["path"]:
            blockers.append("selected build prerequisite is missing: " + name)
    return blockers


def artifact_records(item: dict, out: Path) -> dict[str, str]:
    paths = [out / (item["name"] + suffix) for suffix in (".stdout.log", ".stderr.log")]
    paths += [ROOT / item[key] for key in ("capture", "junit", "preparation_junit") if key in item]
    executable = ROOT / item["argv"][0]
    if executable.is_file():
        paths.append(executable)
    if item.get("build_dir"):
        cache = ROOT / item["build_dir"] / "CMakeCache.txt"
        if cache.is_file():
            paths.append(cache)
    return {str(path.resolve()): sha256(path) for path in paths if path.is_file()}


def resume_results(out: Path, manifest: dict) -> list[dict]:
    report = json.loads((out / "results.json").read_text(encoding="utf-8"))
    if report.get("manifest_identity_sha256") != manifest["identity_sha256"]:
        raise ValueError("resume identity differs: source, dependencies, device selection, environment or plan changed")
    elapsed = report.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid elapsed time in resume checkpoint")
    completed = report.get("results", [])
    if not isinstance(completed, list) or len(completed) > len(manifest["commands"]):
        raise ValueError("invalid resume checkpoint")
    # A failed/incomplete command is rerun; its predecessors must still match.
    if completed and completed[-1].get("validation_errors"):
        completed = completed[:-1]
    for row, item in zip(completed, manifest["commands"]):
        if row.get("name") != item["name"] or row.get("validation_errors") or not row.get("artifacts"):
            raise ValueError("resume checkpoint is not a successful ordered prefix")
        for name, expected in row["artifacts"].items():
            if not Path(name).is_file() or sha256(Path(name)) != expected:
                raise ValueError("resume artifact missing or changed: " + name)
    return completed


def execute_plan(manifest: dict, out: Path, environment: dict, *, resume: bool = False,
                 runner=run_capture, clock=time.monotonic) -> int:
    results = resume_results(out, manifest) if resume else []
    inspections = {row["name"]: json.loads(row["stdout"]) for row in results if row.get("parent_inspect")}
    started = clock()
    # Include interrupted/failed work and checkpoint overhead in the same
    # session budget; removing a failed suffix must not reset its consumed time.
    prior_elapsed = json.loads((out / "results.json").read_text(encoding="utf-8"))["elapsed_seconds"] if resume else 0
    total = manifest["budgets"]["total_seconds"]
    reserve = manifest["budgets"]["export_reserve_seconds"]

    def checkpoint(status):
        write_json(out / "results.json", {"schema": "rns8_gpu_session_results_v2", "status": status,
            "promotion_allowed": False, "manifest_identity_sha256": manifest["identity_sha256"],
            "physical_identity_verified": False, "hardware_qualified": False,
            "observed_parent_runtime_devices": [{"stage": name, **info, "pci_bdf": None, "physical_uuid": None}
                                                for name, info in inspections.items()],
            "completed": len(results), "planned": len(manifest["commands"]),
            "elapsed_seconds": prior_elapsed + clock() - started, "results": results})

    checkpoint("in_progress")
    try:
        for index, item in enumerate(manifest["commands"][len(results):], len(results) + 1):
            remaining = total - prior_elapsed - (clock() - started) - reserve
            if remaining <= 0:
                checkpoint("budget_exhausted")
                return 1
            if shutil.disk_usage(out).free < manifest["budgets"]["minimum_free_bytes"]:
                checkpoint("disk_budget_exhausted")
                return 1
            child = isolated_environment(environment, item["device"]) if item.get("device") is not None else dict(environment)
            if item.get("device") is not None:
                child.update(RNS8_MULTI_GPU_MODE="embarrassingly_parallel_shards",
                             RNS8_RANK=str(manifest["devices"].index(item["device"])), RNS8_WORLD_SIZE=str(len(manifest["devices"])))
            result = runner(item["argv"], cwd=ROOT, label=f"{index}/{len(manifest['commands'])} {item['name']}",
                            timeout_seconds=min(item["timeout_seconds"], remaining), env=child)
            for stream in ("stdout", "stderr"):
                (out / f"{item['name']}.{stream}.log").write_text(result[stream], encoding="utf-8")
            errors = validate_result(item, result, inspections.get(item.get("match_parent")))
            if "capture" in item and not errors:
                (ROOT / item["capture"]).write_text(result["stdout"], encoding="utf-8")
            if item.get("parent_inspect") and not errors:
                inspections[item["name"]] = json.loads(result["stdout"])
            results.append({"name": item["name"], **result, "parent_inspect": item.get("parent_inspect", False),
                            "selected_runtime_ordinal": item.get("selected_device"), "physical_uuid": None, "pci_bdf": None,
                            "validation_errors": errors, "artifacts": artifact_records(item, out)})
            checkpoint("blocked" if errors else "in_progress")
            if errors:
                print(f"STOP: {item['name']}: {'; '.join(errors)}", file=sys.stderr, flush=True)
                return 1
    except KeyboardInterrupt:
        # run_capture has already terminated its owned process group.
        checkpoint("interrupted")
        return 130
    checkpoint("cpu_preparation_passed_hardware_pending" if manifest.get("session_kind") == "prepare" and results else
               "smoke_passed_release_qualification_pending" if results else "environment_receipt_only_not_hardware_qualified")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--devices", type=device_list,
                        help="explicit unmasked parent HIP runtime ordinals; never inferred physical ids")
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--accelerators", action="store_true", help="explicitly select all optional backend lanes")
    parser.add_argument("--all-enabled", action="store_true", help="add a combined optional-accelerator build and backend smoke lane after isolated lanes")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="execute the selected preparation or already available GPU session")
    parser.add_argument("--dry-run", action="store_true", help="explicit plan-only alias (the default)")
    parser.add_argument("--session-kind", choices=("prepare", "environment", "smoke", "first-pass", "multigpu"), default="first-pass")
    parser.add_argument("--preset", help="explicit existing configure preset; target and feature flags remain explicit")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="shared discovery dependency override")
    parser.add_argument("--resume", action="store_true", help="resume only matching successful checkpoints and unchanged artifacts")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--total-seconds", type=float, default=7200, help="whole session wall-clock cap including resumed command time")
    parser.add_argument("--export-reserve-seconds", type=float, default=30)
    parser.add_argument("--min-free-gib", type=float, default=2)
    parser.add_argument("--artifact-inventory", type=Path, help="explicit frozen local artifact inventory")
    parser.add_argument("--artifact-cache", type=Path, help="cache-only artifact source; no network fallback")
    parser.add_argument("--stage-artifacts", action="store_true", help="copy verified local artifacts into out-dir/artifacts, without running the session")
    args = parser.parse_args(argv)
    if bool(args.artifact_inventory) != bool(args.artifact_cache):
        parser.error("--artifact-inventory and --artifact-cache must be supplied together")
    if args.stage_artifacts and (not args.artifact_inventory or args.execute or args.resume or args.dry_run):
        parser.error("--stage-artifacts requires an inventory/cache and is separate from execute/resume/dry-run")
    artifacts = artifact_receipt = None
    if args.artifact_inventory:
        try:
            artifacts = load_inventory(args.artifact_inventory)
            artifact_receipt = inspect_cache(artifacts, args.artifact_cache)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    preparation = args.session_kind == "prepare"
    if preparation:
        if args.devices is not None or args.backend or args.accelerators or args.all_enabled or TARGETS[args.target][1] != "Linux":
            parser.error("prepare requires a Linux target and no --devices, --backend or --accelerators")
        args.devices = []
    elif args.devices is None:
        parser.error("GPU session planning requires explicit --devices")
    if args.jobs < 1 or args.jobs > 256 or any(not math.isfinite(value) or value <= 0 for value in
            (args.timeout_seconds, args.total_seconds, args.export_reserve_seconds, args.min_free_gib)):
        parser.error("jobs must be 1..256 and time/disk budgets must be finite positive numbers")
    if args.total_seconds <= args.export_reserve_seconds:
        parser.error("total session budget must exceed the export reserve")
    if args.execute and args.dry_run or args.resume and not args.execute:
        parser.error("--execute conflicts with --dry-run; --resume requires --execute")
    if args.accelerators and args.backend:
        parser.error("choose --accelerators or explicit --backend lanes, not both")
    if args.session_kind == "smoke" and len(args.devices) != 1:
        parser.error("smoke requires exactly one explicit device; use multigpu for independent device lanes")
    out = (args.out_dir or ROOT / "temp" / "qualification" / args.target).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists() and not args.resume:
        parser.error("results.json already exists; choose a fresh --out-dir or explicit matching --resume")
    backends = [] if preparation else list(dict.fromkeys(args.backend or BACKENDS))
    command_out = Path(os.path.relpath(out, ROOT))
    try:
        request = dependency_request(args.target, list(BACKENDS) if args.all_enabled else backends, args.set, preparation=preparation)
    except ValueError as error:
        parser.error(str(error))
    environment = build_environment(os.environ)
    # Use the same write-free selection as doctor/CMake. Cross-host planning
    # does not claim the planning machine's executables are target prerequisites.
    # Supported dependency environment values participate in selection without
    # forwarding arbitrary authentication or cloud variables to child commands.
    from session_environment import safe_value
    discovery_env = {key: value for key, value in os.environ.items()
                     if (key in REQUEST_KEYS or key in environment) and safe_value(value) == value} if platform.system() == request.host else {}
    resolution = resolve(request, discovery_env)
    environment.update({key: value for key, value in discovery_env.items() if key in REQUEST_KEYS})
    blockers = readiness_blockers(args, resolution, os.environ)
    if artifact_receipt and not artifact_receipt["complete"]:
        blockers += ["artifact " + row["id"] + ": " + row["status"] for row in artifact_receipt["artifacts"] if row["status"] != "verified"]
    from session_environment import BUILD_ENV_KEYS, ALLOWLIST
    blockers += [key + " contains an unsafe environment value (redacted); select a local prerequisite path explicitly"
                 for key, value in os.environ.items() if key in {*REQUEST_KEYS, *BUILD_ENV_KEYS, *ALLOWLIST}
                 and safe_value(value) != value]
    commands = command_plan(args.target, backends, command_out, args.jobs, args.devices,
                            bindings={**request.overrides, **resolution["cmake_bindings"]}, session_kind=args.session_kind,
                            timeout=args.timeout_seconds, preset_override=args.preset, all_enabled=args.all_enabled)
    for item in commands:
        for index, value in enumerate(item["argv"]):
            if value == "cmake" and resolution["tools"]["cmake"]["path"]:
                item["argv"][index] = resolution["tools"]["cmake"]["path"]
            elif value == "ctest" and resolution["tools"]["cmake"]["path"]:
                item["argv"][index] = str(Path(resolution["tools"]["cmake"]["path"]).with_name("ctest.exe" if request.host == "Windows" else "ctest"))
    manifest = {"schema_version": 2, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "target": args.target, "requested_architecture": TARGETS[args.target][0], "observed_architecture": None,
                "source": source_receipt(exclude=out), "backends": backends, "all_enabled_lane": args.all_enabled, "devices": args.devices,
                "device_namespace": "none_cpu_preparation" if preparation else "unmasked_parent_HIP_runtime_ordinal", "physical_identity_verified": False,
                "session_kind": args.session_kind, "dependency_resolution_sha256": resolution["identity_sha256"],
                "artifact_inventory": artifacts, "artifact_cache_receipt": artifact_receipt,
                "selected_tool_files": {name: {"path": item["path"], "sha256": sha256(Path(item["path"]))}
                                        for name, item in resolution["tools"].items()
                                        if item.get("path") and Path(item["path"]).is_file()},
                "execution_blockers": blockers, "environment_allowlist": snapshot(os.environ),
                "child_build_environment_sha256": identity({**environment, **resolution["compiler_environment"]}),
                "budgets": {"total_seconds": args.total_seconds, "export_reserve_seconds": args.export_reserve_seconds,
                            "minimum_free_bytes": int(args.min_free_gib * 1024**3), "jobs": args.jobs},
                "status": "planned_not_executed", "promotion_allowed": False,
                "preparation_only": preparation, "hardware_qualified": False,
                "command_working_directory": "repository_root",
                "environment": {"RNS8_AUTOTUNE_CACHE": str(command_out / "isolated-empty-cache.json")},
                "commands": commands, "kernels": kernel_inventory(),
                "remaining_gates": ["HIP compile and hardware execution for every selected backend",
                    "full correctness and backend-specific ISA checks", "large/padded/K-boundary and sparse target coverage",
                    "release benchmarks, timing variance, profiler/counter evidence and explicit cache review",
                    "physical UUID/BDF correlation, NUMA affinity, profiler/RCCL inventory and remote Linux rehearsal"]}
    manifest["identity_sha256"] = identity({key: value for key, value in manifest.items() if key not in {"generated_utc", "status"}})
    if args.resume:
        try:
            resume_results(out, manifest)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    write_json(out / "dependency-resolution.json", resolution)
    write_json(out / "environment-allowlist.json", {"schema": "rns8_session_environment_v1", "values": snapshot(os.environ)})
    write_json(out / "manifest.json", manifest)
    print(f"{len(commands)} commands, {len(manifest['kernels'])} source kernel definitions: {out / 'manifest.json'}", flush=True)
    if artifact_receipt is not None:
        write_json(out / "artifact-cache.json", artifact_receipt)
    if args.stage_artifacts:
        previous_handler = signal.getsignal(signal.SIGTERM)
        def interrupted_staging(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted_staging)
        try:
            staged = stage_cache(artifacts, args.artifact_cache, out / "artifacts",
                                 reserve_bytes=manifest["budgets"]["minimum_free_bytes"])
        except KeyboardInterrupt:
            write_json(out / "artifact-staging.json", {"status": "interrupted", "hardware_qualified": False})
            return 130
        except (OSError, ValueError) as error:
            write_json(out / "artifact-staging.json", {"status": "incomplete", "error": str(error), "hardware_qualified": False})
            print("Artifact staging blocked: " + str(error), file=sys.stderr)
            return 1
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        write_json(out / "artifact-staging.json", {"status": "staged_runtime_qualification_pending", **staged})
        print("Verified local artifacts staged; no installation or session execution.", flush=True)
        return 0
    if not args.execute:
        print("Plan only. No build, GPU work, benchmark, or cache promotion was executed.", flush=True)
        invalid = any(item.get("status") in {"invalid", "ambiguous"} for item in
                      (resolution["hip"], resolution["vcpkg"], *resolution["optional_components"].values()))
        invalid = invalid or any("Linux/UNIX builds" in error or error.startswith("explicit ") for error in resolution["errors"])
        if invalid:
            print("Invalid/ambiguous explicit selection: " + "; ".join(resolution["errors"]), file=sys.stderr)
        return 2 if invalid else 0
    if blockers:
        print("Execution blocked: " + "; ".join(blockers), file=sys.stderr)
        return 1
    if shutil.disk_usage(out).free < manifest["budgets"]["minimum_free_bytes"]:
        print("Execution blocked: insufficient free output/build space", file=sys.stderr)
        return 1
    cache = ROOT / manifest["environment"]["RNS8_AUTOTUNE_CACHE"]
    if cache.exists():
        parser.error("isolated cache path exists; choose a fresh output directory")
    environment.update(resolution["compiler_environment"])
    environment.update(manifest["environment"])
    # Translate termination into the same owned-process cleanup/checkpoint path
    # as Ctrl+C. Never kill unrelated host processes.
    previous_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute_plan(manifest, out, environment, resume=args.resume)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
