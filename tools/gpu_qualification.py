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
import uuid
from datetime import datetime, timezone
from pathlib import Path

from process_progress import run_capture, capture_succeeded
from session_artifacts import load_inventory, inspect_cache, stage_cache, export_session
from session_platform_contract import load_lock as load_platform_lock
from session_platform_contract import evaluate as evaluate_platform_lock
from session_platform_contract import observe as observe_platform
from session_budget import plan_budget, replay_budget, stage_fits
from source_gemm_contract import (
    CONTRACT as SOURCE_GEMM_CONTRACT,
    load_contract as load_source_gemm_contract,
    admission as source_gemm_admission,
)
from check_dependencies_lib.resolution import Request, REQUEST_KEYS, resolve
from session_environment import (
    build_environment,
    execution_blockers,
    isolated_environment,
    snapshot,
    runtime_mapping_snapshot,
)
from compiled_binary_identity import validate_document, bind_document
from session_resource_contract import (
    evaluate as evaluate_host_resources,
    resource_policy as host_resource_policy,
)
from json_evidence import parse_evidence_json, same_evidence_value
from benchmark_request import request_identity, attach_capture_identity
from device_observation import (
    IdentityUnavailable,
    IdentityMismatch,
    inspection_observation,
    observation_relation,
    capture_relation,
    distinct_execution_devices,
    inspection_relation,
    binding_identifiers,
)

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("hip-direct", "amdgpu-builtins", "hipblaslt", "ck", "rocwmma")
FLAGS = {
    "amdgpu-builtins": "AMDGPU_BUILTINS",
    "hipblaslt": "HIPBLASLT",
    "ck": "CK",
    "rocwmma": "ROCWMMA",
}
TARGETS = {
    "mi300x": ("gfx942", "Linux", "linux-cdna-release"),
    "7900xtx": ("gfx1100", "Windows", "windows-msvc-hip-release"),
}


def kernel_inventory(root: Path = ROOT) -> list[dict]:
    kernels = []
    for path in sorted((root / "src").rglob("*")):
        if path.suffix not in {".hip", ".cuh", ".inc"}:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(
            r"__global__\s+(?:__launch_bounds__\([^\n]*?\)\s+)?void\s+(\w+)\s*\(", text
        ):
            kernels.append(
                {
                    "name": match[1],
                    "source": path.relative_to(root).as_posix(),
                    "line": text.count("\n", 0, match.start()) + 1,
                    "evidence": "source_only_not_hardware_qualified",
                }
            )
    return kernels


def command_plan(
    target: str,
    backends: list[str],
    out: Path,
    jobs: int,
    devices: list[int],
    *,
    bindings: dict | None = None,
    session_kind: str = "first-pass",
    timeout: float = 1800,
    preset_override: str | None = None,
    all_enabled: bool = False,
    resource_policy: dict | None = None,
    platform_lock: dict | None = None,
) -> list[dict]:
    arch, system, preset = TARGETS[target]
    if session_kind != "prepare" and (
        not devices
        or len(set(devices)) != len(devices)
        or any(type(device) is not int or device < 0 for device in devices)
    ):
        raise ValueError("devices must be unique explicit nonnegative runtime ordinals")
    extension = ".exe" if system == "Windows" else ""
    python = (
        sys.executable
        if platform.system() == system
        else "python"
        if system == "Windows"
        else "python3"
    )
    wrapper = [python, "tools/windows_dev.py"] if system == "Windows" else []
    commands = []

    def add(name, argv, **checks):
        commands.append(
            {
                "name": name,
                "argv": argv,
                "depends_on": [commands[-1]["name"]] if commands else [],
                "timeout_seconds": timeout,
                **checks,
            }
        )

    if system == "Linux":
        policy = dict(resource_policy or {})
        policy.setdefault("minimum_free_bytes", 2 * 1024**3)
        policy.setdefault("minimum_memory_bytes", 0)
        policy.setdefault("peak_job_memory_bytes", 0)
        policy.setdefault("memory_reserve_bytes", 0)
        policy.update(
            jobs=jobs,
            gpu_required=session_kind != "prepare",
            architecture=None if session_kind == "prepare" else arch,
        )
        policy = host_resource_policy(**policy)
        resource_args = [python, "tools/session_resource_contract.py", "--output-root", str(out)]
        for field in (
            "jobs",
            "minimum_free_bytes",
            "minimum_memory_bytes",
            "peak_job_memory_bytes",
            "memory_reserve_bytes",
        ):
            resource_args += ["--" + field.replace("_", "-"), str(policy[field])]
        if policy["architecture"] is not None:
            resource_args += ["--architecture", policy["architecture"]]
        add("linux-resource-access-preflight", resource_args, host_resources=policy)
        if platform_lock is not None:
            add(
                "linux-platform-lock-preflight",
                [python, "tools/session_platform_contract.py", "--lock", platform_lock["path"]],
                platform_lock=platform_lock,
            )
    if session_kind == "environment":
        return commands

    if session_kind == "prepare":
        if system != "Linux" or devices or backends or all_enabled:
            raise ValueError("CPU preparation requires Linux, no devices and no backend lanes")
        build = out / "build" / "cpu-preparation"
        bound = [f"-D{key}={value}" for key, value in sorted((bindings or {}).items())]
        options = [f"-DRNS8_ENABLE_{flag}=OFF" for flag in FLAGS.values()]
        add(
            "prepare-configure",
            [
                "cmake",
                "--preset",
                preset_override or "linux-cpu-release",
                "-B",
                str(build),
                *bound,
                "-DRNS8_ENABLE_HIP=OFF",
                "-DRNS8_PROBE_ACCELERATORS=OFF",
                *options,
                "-DRNS8_BUILD_TESTS=ON",
                "-DRNS8_BUILD_EXAMPLES=ON",
                "-DRNS8_BUILD_TOOLS=ON",
                "-DRNS8_BUILD_BENCHMARKS=ON",
                "-DRNS8_WORKLOAD_GPU_TESTS=OFF",
                "-DVCPKG_MANIFEST_INSTALL=OFF",
                "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            ],
            build_dir=str(build),
        )
        add(
            "prepare-build",
            ["cmake", "--build", str(build), "--parallel", str(jobs)],
            build_dir=str(build),
        )
        add(
            "prepare-manifest-admission",
            [
                python,
                "tools/build_manifest.py",
                "--check-build",
                str(build),
                "--backend",
                "cpu-reference",
            ],
            build_dir=str(build),
        )
        junit = out / "cpu-preparation.xml"
        add(
            "prepare-cpu-tests",
            [
                "ctest",
                "--test-dir",
                str(build),
                "--output-on-failure",
                "--parallel",
                "1",
                "--timeout",
                "600",
                "--no-tests=error",
                "--output-junit",
                str(ROOT / junit),
            ],
            build_dir=str(build),
            preparation_junit=str(junit),
        )
        return commands

    lanes = [(backend, [backend]) for backend in backends]
    if all_enabled:
        lanes.append(("all-enabled", list(BACKENDS)))
    for lane, enabled_backends in lanes:
        # Every isolated or combined feature lane has its own build identity.
        build = out / "build" / lane

        def binary(name, *, build=build):
            return str(build / (name + extension))

        options = [
            f"-DRNS8_ENABLE_{flag}={'ON' if name in enabled_backends else 'OFF'}"
            for name, flag in FLAGS.items()
        ]
        bound = [f"-D{key}={value}" for key, value in sorted((bindings or {}).items())]
        add(
            f"{lane}-configure",
            wrapper
            + [
                "cmake",
                "--preset",
                preset_override or preset,
                "-B",
                str(build),
                *bound,
                f"-DRNS8_AMDGPU_TARGETS={arch}",
                "-DRNS8_ENABLE_HIP=ON",
                "-DVCPKG_MANIFEST_INSTALL=OFF",
                "-DRNS8_PROBE_ACCELERATORS=OFF",
                *options,
            ],
            device=None,
            build_dir=str(build),
        )
        add(
            f"{lane}-build",
            wrapper + ["cmake", "--build", str(build), "--parallel", str(jobs)],
            build_dir=str(build),
        )
        manifest_check = [python, "tools/build_manifest.py", "--check-build", str(build)]
        for backend in enabled_backends:
            manifest_check.extend(["--backend", backend])
        add(f"{lane}-manifest-admission", manifest_check, device=None, build_dir=str(build))
        for tool in ("rns8-inspect", "rns8-verify", "rns8_tests", "rns8-bench"):
            name = f"{lane}-identity-{tool}"
            add(
                name,
                [binary(tool), "--build-identity"],
                device=None,
                build_dir=str(build),
                compiled_identity=True,
                identity_backends=enabled_backends,
                identity_output=str(out / (name + ".identity.json")),
            )
        for backend_index, backend in enumerate(enabled_backends):
            prefix = lane + "-" + backend if len(enabled_backends) > 1 else backend
            # Parent runtime ordinals and child-visible zero remain distinct.
            for device in devices:
                add(
                    f"{prefix}-device{device}-parent-inspect",
                    [
                        binary("rns8-inspect"),
                        "--backend",
                        backend,
                        "--device",
                        str(device),
                        "--json",
                    ],
                    inspect_target=arch,
                    inspect_backend=backend,
                    inspect_device=device,
                    parent_inspect=True,
                    device=None,
                    selected_device=device,
                    require_physical_identity=len(devices) > 1,
                )
                child = device_command_plan(
                    backend,
                    device,
                    arch,
                    binary,
                    python,
                    wrapper,
                    build,
                    out,
                    session_kind,
                    stage_prefix=prefix,
                    full_tests=backend_index == 0,
                )
                for name, argv, checks in child:
                    checks["require_physical_identity"] = len(devices) > 1
                    if "capture" in checks:
                        checks.update(
                            expected_target=arch,
                            expected_backend=backend,
                            case_id=prefix + ":" + checks["case_id"],
                        )
                    add(name, argv, device=device, selected_device=device, **checks)
    return commands


def session_case_inventory(backend, arch, session_kind):
    contract = load_source_gemm_contract()
    cases = [
        ("bounded-i64", []),
        ("bounded-u64", []),
        ("exact-wide-signed", ["--exact-wide-limbs", "3"]),
        ("exact-wide-unsigned", ["--exact-wide-limbs", "3"]),
        ("finite-u8-ring", ["--modulus", "255"]),
        ("finite-u8-field", ["--modulus", "251"]),
    ]
    if session_kind == "smoke":
        cases = cases[:1]
    elif backend == "hip-direct":
        cases += [("wrap-u64", []), ("finite-u8-ring", ["--modulus", "256"])]
    result = []
    for index, (semantics, extra) in enumerate(cases):
        modulus = int(extra[extra.index("--modulus") + 1]) if "--modulus" in extra else 0
        result.append(
            dict(
                index=index,
                semantics=semantics,
                arguments=extra,
                admission=source_gemm_admission(
                    contract, backend, semantics, modulus=modulus, architecture=arch
                ),
            )
        )
    return result


def device_command_plan(
    backend,
    device,
    arch,
    binary,
    python,
    wrapper,
    build,
    out,
    session_kind,
    *,
    stage_prefix=None,
    full_tests=True,
):
    commands = []
    stem = f"{stage_prefix or backend}-device{device}"

    def add(name, argv, **checks):
        commands.append((stem + "-" + name, argv, checks))

    add(
        "inspect",
        [binary("rns8-inspect"), "--backend", backend, "--device", "0", "--json"],
        inspect_target=arch,
        inspect_backend=backend,
        inspect_device=0,
        match_parent=stem + "-parent-inspect",
    )
    # Unlike an all-skipped CTest run, this fails if there is no HIP device.
    add("smoke", [binary("rns8-verify"), "--hip-smoke"])
    if session_kind != "smoke" and full_tests:
        junit = out / f"{stem}-qualification.xml"
        add(
            "regressions",
            [binary("rns8_tests"), "[qualification]", "--reporter", "junit", "--out", str(junit)],
            junit=str(junit),
            require_test="qualification pack covers dispatch thresholds and plane tails",
        )
        ctest_junit = out / f"{stem}-ctest.xml"
        add(
            "ctest",
            wrapper
            + [
                "ctest",
                "--test-dir",
                str(build),
                "--output-on-failure",
                "--parallel",
                "1",
                "--timeout",
                "600",
                "--no-tests=error",
                "--output-junit",
                str(ctest_junit),
            ],
            ctest_junit=str(ctest_junit),
        )
    # Small captures are smoke evidence only, with bounded runtime per command.
    for case in session_case_inventory(backend, arch, session_kind):
        if not case["admission"]["source_eligible"]:
            continue
        index, semantics, extra = case["index"], case["semantics"], case["arguments"]
        name = f"{backend}-{index}-{semantics}"
        capture = out / f"{stem}-{name}.json"
        add(
            name,
            [
                binary("rns8-bench"),
                "--backend",
                backend,
                "--semantics",
                semantics,
                "--m",
                "32",
                "--n",
                "32",
                "--k",
                "64",
                "--warmups",
                "1",
                "--repeats",
                "3",
                "--seed",
                "20261006",
                "--cpu-threads",
                "1",
                "--device",
                "0",
                "--verify-native-output",
                "--progress",
                *extra,
            ],
            capture=str(capture),
            case_id=name,
            source_operation=case["admission"],
            observation_parent_stage=stem + "-parent-inspect",
            observation_child_stage=stem + "-inspect",
        )
        add(name + "-schema", [python, "tools/benchmark_schema.py", str(capture)])
    if backend == "hip-direct" and session_kind != "smoke":
        # Existing graph capture, not the removed dead global graph helpers.
        name = "hip-direct-graph"
        capture = out / f"{stem}-{name}.json"
        add(
            name,
            [
                binary("rns8-bench"),
                "--backend",
                "hip-direct",
                "--semantics",
                "bounded-i64",
                "--m",
                "32",
                "--n",
                "32",
                "--k",
                "64",
                "--hip-graph-replay",
                "--warmups",
                "1",
                "--repeats",
                "3",
                "--seed",
                "20261006",
                "--device",
                "0",
                "--progress",
            ],
            capture=str(capture),
            case_id=name,
            observation_parent_stage=stem + "-parent-inspect",
            observation_child_stage=stem + "-inspect",
            source_operation=source_gemm_admission(
                load_source_gemm_contract(), "hip-direct", "bounded-i64", architecture=arch
            ),
        )
        add(name + "-schema", [python, "tools/benchmark_schema.py", str(capture)])
    return commands


def validate_source_case(item):
    value = item.get("source_operation")
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("typed source operation admission required")
    expected = source_gemm_admission(
        load_source_gemm_contract(),
        value["backend"],
        value["semantics"],
        modulus=value["modulus"],
        architecture=value["requested_architecture"],
    )
    if not same_evidence_value(value, expected) or not value["source_eligible"]:
        raise ValueError("session positive case is unsupported or source inventory changed")
    argv = item["argv"]
    for flag, operand in (("--backend", value["backend"]), ("--semantics", value["semantics"])):
        if (
            argv.count(flag) != 1
            or argv.index(flag) + 1 >= len(argv)
            or argv[argv.index(flag) + 1] != operand
        ):
            raise ValueError("session argv differs from source operation admission")
    if value["modulus"]:
        if (
            argv.count("--modulus") != 1
            or argv.index("--modulus") + 1 >= len(argv)
            or argv[argv.index("--modulus") + 1] != str(value["modulus"])
        ):
            raise ValueError("session finite modulus differs from source admission")
    elif "--modulus" in argv:
        raise ValueError("nonfinite session case carries a finite modulus")


def validate_result(item: dict, result: dict, parent_info: dict | None = None) -> list[str]:
    errors = []
    if "platform_lock" in item:
        try:
            value = parse_evidence_json(result["stdout"])
            expected = evaluate_platform_lock(item["platform_lock"], value["observed"])
            if type(value.get("process_id")) is not int or value["process_id"] != result.get(
                "process_id"
            ):
                raise ValueError("platform preflight process mismatch")
            expected["process_id"] = value["process_id"]
            if load_platform_lock(Path(item["platform_lock"]["path"])) != item["platform_lock"]:
                raise ValueError("platform lock input changed")
            if not same_evidence_value(value, expected) or expected["blockers"]:
                raise ValueError("platform lock observations missing, mismatched or inconsistent")
        except (OSError, ValueError, KeyError, TypeError):
            errors.append("native Linux platform lock preflight failed")
    try:
        validate_source_case(item)
    except (KeyError, ValueError, TypeError):
        errors.append("session source-operation admission is inconsistent or unsupported")
    if "host_resources" in item:
        try:
            value = parse_evidence_json(result["stdout"])
            if (
                not isinstance(value, dict)
                or type(value.get("process_id")) is not int
                or type(result.get("process_id")) is not int
                or value["process_id"] != result.get("process_id")
            ):
                raise ValueError("preflight process mismatch")
            expected = evaluate_host_resources(value["observed"], **item["host_resources"])
            expected["process_id"] = value["process_id"]
            if not same_evidence_value(value, expected) or value["blockers"]:
                raise ValueError("resource receipt differs from observed values/policy")
        except (KeyError, ValueError, TypeError):
            errors.append(
                "native Linux resource/access preflight is missing, blocked or inconsistent"
            )
    if not capture_succeeded(result):
        return errors + ["command failed, timed out or exceeded its capture budget"]
    if item.get("compiled_identity"):
        try:
            value = parse_evidence_json(result["stdout"])
            if (
                isinstance(value, dict)
                and (value.get("process_id") is not None or result.get("process_id") is not None)
                and (
                    type(value.get("process_id")) is not int
                    or value["process_id"] != result.get("process_id")
                )
            ):
                raise ValueError("compiled query JSON belongs to another process")
            for backend in item["identity_backends"]:
                validate_document(value, backend)
        except (ValueError, TypeError, KeyError):
            errors.append("compiled executable/library identity is missing or incompatible")
    if "inspect_target" in item:
        try:
            info = parse_evidence_json(result["stdout"])
            if info.get("gcn_arch", "").split(":")[0] != item["inspect_target"]:
                errors.append("runtime GPU target does not match the manifest")
            if info.get("backend") != item["inspect_backend"]:
                errors.append("runtime backend does not match the manifest")
            # rns8_device_info uses uint32_t and rns8-inspect emits numeric 0/1.
            # Accept the earlier Boolean fixtures, but not arbitrary truthy values.
            available = info.get("hip_available")
            if (
                not (available is True or (type(available) is int and available == 1))
                or type(info.get("global_mem_bytes")) is not int
                or info["global_mem_bytes"] <= 0
            ):
                errors.append(
                    "runtime inspection did not establish an available GPU with positive memory"
                )
            if (
                info.get("device_id") != item["inspect_device"]
                or type(info.get("device_id")) is not int
            ):
                errors.append(
                    "runtime ordinal does not match the explicitly selected inspection ordinal"
                )
            if not isinstance(info.get("name"), str) or not info["name"].strip():
                errors.append("runtime GPU name is missing")
            if (
                info.get("process_id") is not None
                or result.get("process_id") is not None
                or item.get("require_physical_identity")
            ):
                if (
                    type(info.get("process_id")) is not int
                    or info["process_id"] <= 0
                    or info["process_id"] != result.get("process_id")
                ):
                    errors.append("runtime inspection JSON does not belong to the launched process")
            try:
                observed = inspection_observation(info)
                if "match_parent" in item:
                    relation = observation_relation(inspection_observation(parent_info), observed)
                    if relation.state != "verified":
                        errors.append(
                            "physical runtime device mapping is "
                            + relation.state
                            + ": "
                            + ",".join(relation.reasons)
                        )
            except IdentityUnavailable:
                if item.get("require_physical_identity"):
                    errors.append("runtime physical device identity is unavailable")
            except IdentityMismatch as error:
                errors.append("runtime physical device identity is contradictory: " + str(error))
            if "match_parent" in item:
                if parent_info is None:
                    errors.append("parent runtime inspection is missing")
                elif any(
                    info.get(key) != parent_info.get(key)
                    for key in (
                        "name",
                        "gcn_arch",
                        "global_mem_bytes",
                        "hip_runtime_version",
                        "hip_driver_version",
                        "backend",
                    )
                ):
                    errors.append(
                        "isolated logical device does not match selected parent runtime evidence"
                    )
        except (ValueError, AttributeError, TypeError):
            errors.append("invalid device inspection JSON")
    if "junit" in item:
        try:
            root = ET.parse(ROOT / item["junit"]).getroot()
            cases = list(root.iter("testcase"))
            if not any(case.get("name") == item["require_test"] for case in cases):
                errors.append("required hardware regression missing (possibly a CPU-only build)")
            if any(
                list(case.iter("skipped")) or list(case.iter("failure")) or list(case.iter("error"))
                for case in cases
            ):
                errors.append("hardware qualification tests failed or skipped")
        except (OSError, ET.ParseError):
            errors.append("missing or invalid qualification JUnit report")
    if "preparation_junit" in item or "ctest_junit" in item:
        test_scope = "CPU preparation" if "preparation_junit" in item else "CTest"
        try:
            root = ET.parse(ROOT / item.get("preparation_junit", item.get("ctest_junit"))).getroot()
            cases = list(root.iter("testcase"))
            if not cases or not any(not list(case.iter("skipped")) for case in cases):
                errors.append(test_scope + " report has no executed tests")
            if any(list(case.iter("failure")) or list(case.iter("error")) for case in cases):
                errors.append(test_scope + " tests failed")
        except (OSError, ET.ParseError):
            errors.append("missing or invalid " + test_scope + " JUnit report")
    return errors


def device_list(value: str) -> list[int]:
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise argparse.ArgumentTypeError(
            "--devices requires comma-separated nonnegative runtime ordinals; no all/default selection"
        )
    devices = [int(part) for part in value.split(",")]
    if (
        len(set(devices)) != len(devices)
        or len(devices) > 64
        or any(device > 65535 for device in devices)
    ):
        raise argparse.ArgumentTypeError(
            "device list contains duplicates or exceeds the bounded 64-device/65535-ordinal limit"
        )
    return devices


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_receipt(exclude: Path | None = None) -> dict:
    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, check=True, timeout=30
        ).stdout

    # Include untracked first-party source; git diff alone misses newly authored
    # files. Ignored build/temp evidence and external gitlink contents stay out.
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    return {
        "git_revision": git("rev-parse", "HEAD").decode().strip(),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
        "untracked_files": {
            name: sha256(ROOT / name)
            for name in untracked
            if name
            and (ROOT / name).is_file()
            and (exclude is None or not (ROOT / name).resolve().is_relative_to(exclude.resolve()))
        },
        "runner_sha256": sha256(Path(__file__)),
        "environment_helper_sha256": sha256(ROOT / "tools/session_environment.py"),
        "artifact_helper_sha256": sha256(ROOT / "tools/session_artifacts.py"),
        "platform_helper_sha256": sha256(ROOT / "tools/session_platform_contract.py"),
        "dependency_policy_sha256": sha256(ROOT / "metadata/dependencies.json"),
    }


def dependency_request(
    target: str, backends: list[str], definitions: list[str], *, preparation: bool = False
) -> Request:
    overrides = {"RNS8_AMDGPU_TARGETS": TARGETS[target][0]}
    for definition in definitions:
        if "=" not in definition:
            raise ValueError("--set expects a supported dependency KEY=VALUE")
        key, value = definition.split("=", 1)
        if (
            key not in REQUEST_KEYS
            or key in overrides
            or not value
            or re.search(r"[\x00-\x1f]|https?://", value, re.I)
        ):
            raise ValueError("unsupported, duplicate or unsafe dependency override: " + key)
        overrides[key] = value
    features = {
        "RNS8_ENABLE_HIP": not preparation,
        **{f"RNS8_ENABLE_{flag}": name in backends for name, flag in FLAGS.items()},
    }
    return Request(
        TARGETS[target][1],
        overrides,
        features,
        str(ROOT),
        require_hip=not preparation,
        require_vcpkg=TARGETS[target][1] == "Windows",
    )


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
    paths += [
        ROOT / item[key]
        for key in ("capture", "junit", "preparation_junit", "ctest_junit", "identity_output")
        if key in item
    ]
    additional = item.get("evidence_outputs", [])
    if not isinstance(additional, list) or any(not isinstance(path, str) for path in additional):
        raise ValueError("session evidence output inventory must contain explicit paths")
    paths += [ROOT / path for path in additional]
    owned_paths = []
    for directory in (
        *item.get("owned_artifact_directories", []),
        *item.get("monitor_artifact_directories", []),
    ):
        owned_paths += owned_artifact_files(Path(directory), out)
    if (
        owned_paths
        and sum(path.stat().st_size for path in owned_paths) > item["max_artifact_bytes"]
    ):
        raise ValueError(
            "owned artifact bytes exceed the acquisition budget; raw files retained without hashing"
        )
    paths += owned_paths
    executable = ROOT / item["argv"][0]
    if executable.is_file():
        paths.append(executable)
    if item.get("build_dir"):
        cache = ROOT / item["build_dir"] / "CMakeCache.txt"
        if cache.is_file():
            paths.append(cache)
    return {str(path.resolve()): sha256(path) for path in paths if path.is_file()}


def owned_artifact_files(directory: Path, out: Path) -> list[Path]:
    """Inventory only this session's bounded, link-free acquisition subtree."""
    directory = directory.absolute()
    if (
        not directory.is_relative_to(out.resolve())
        or directory == out.resolve()
        or directory.resolve() != directory
    ):
        raise ValueError("owned artifact directory must be a child of the session")
    pending, files, count = [directory], [], 0
    deadline = time.monotonic() + 5
    while pending:
        if time.monotonic() > deadline:
            raise ValueError("artifact inventory scan budget exhausted")
        path = pending.pop()
        if path.is_symlink() or path.is_junction():
            raise ValueError("artifact inventory contains a link or junction")
        if path.is_dir():
            with os.scandir(path) as entries:
                for entry in entries:
                    count += 1
                    if count > 100000:
                        raise ValueError("artifact inventory entry budget exhausted")
                    pending.append(Path(entry.path))
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError("artifact inventory contains an absent or nonregular file")
    return sorted(files)


def resume_results(out: Path, manifest: dict) -> list[dict]:
    report = json.loads((out / "results.json").read_text(encoding="utf-8"))
    if report.get("manifest_identity_sha256") != manifest["identity_sha256"]:
        raise ValueError(
            "resume identity differs: source, dependencies, device selection, environment or plan changed"
        )
    if not isinstance(report.get("run_id"), str) or not report["run_id"]:
        raise ValueError("legacy checkpoint has no explicit qualification run identity")
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
        if (
            row.get("name") != item["name"]
            or row.get("validation_errors")
            or not row.get("artifacts")
        ):
            raise ValueError("resume checkpoint is not a successful ordered prefix")
        for name, expected in row["artifacts"].items():
            if not Path(name).is_file() or sha256(Path(name)) != expected:
                raise ValueError("resume artifact missing or changed: " + name)
    return completed


def execute_plan(
    manifest: dict,
    out: Path,
    environment: dict,
    *,
    resume: bool = False,
    runner=run_capture,
    clock=time.monotonic,
) -> int:
    results = resume_results(out, manifest) if resume else []
    inspections = {
        row["name"]: parse_evidence_json(row["stdout"])
        for row in results
        if row.get("parent_inspect") or row.get("runtime_inspection")
    }
    started = clock()
    # Include interrupted/failed work and checkpoint overhead in the same
    # session budget; removing a failed suffix must not reset its consumed time.
    prior_elapsed = (
        json.loads((out / "results.json").read_text(encoding="utf-8"))["elapsed_seconds"]
        if resume
        else 0
    )
    run_id = (
        json.loads((out / "results.json").read_text(encoding="utf-8"))["run_id"]
        if resume
        else str(uuid.uuid4())
    )
    total = manifest["budgets"]["total_seconds"]
    reserve = manifest["budgets"]["export_reserve_seconds"]

    def checkpoint(status, error=None):
        observations = [
            row["device_binding"] if isinstance(row.get("device_binding"), dict) else {}
            for row in results
            if row.get("runtime_inspection") or row.get("capture_observation")
        ]
        expected = sum(
            "inspect_target" in item or "capture" in item for item in manifest["commands"]
        )
        physical_verified = (
            expected > 0
            and len(results) == len(manifest["commands"])
            and len(observations) == expected
            and all(item.get("state") in ("observed", "verified") for item in observations)
            and not any(row.get("validation_errors") for row in results)
        )
        write_json(
            out / "results.json",
            {
                "schema": "rns8_gpu_session_results_v2",
                "status": status,
                "promotion_allowed": False,
                "manifest_identity_sha256": manifest["identity_sha256"],
                "run_id": run_id,
                "physical_identity_verified": physical_verified,
                "hardware_qualified": False,
                "observed_parent_runtime_devices": [
                    {
                        "stage": row["name"],
                        **inspections[row["name"]],
                        "pci_bdf": binding_identifiers(row.get("device_binding"))["pci_bdf"],
                        "physical_uuid": binding_identifiers(row.get("device_binding"))["uuid"],
                    }
                    for row in results
                    if row.get("parent_inspect") and row["name"] in inspections
                ],
                "completed": len(results),
                "planned": len(manifest["commands"]),
                "runner_error": error,
                "elapsed_seconds": prior_elapsed + clock() - started,
                "results": results,
            },
        )

    checkpoint("in_progress")
    try:
        locked = manifest.get("platform_lock")
        if locked is not None:
            if load_platform_lock(Path(locked["path"])) != locked:
                raise ValueError("frozen platform lock changed")
            if (
                resume
                and evaluate_platform_lock(locked, observe_platform(locked["lock"]))["blockers"]
            ):
                raise ValueError("native platform or locked file changed before resume")
        forecast = {}
        if manifest.get("session_budget") is not None:
            budget = replay_budget(manifest["session_budget"], manifest["commands"])
            if (
                budget["blockers"]
                or type(total) not in (int, float)
                or type(reserve) not in (int, float)
                or total != budget["effective_total_seconds"]
                or reserve != budget["reserve_seconds"]
            ):
                raise ValueError(
                    "effective session budget is blocked or differs from frozen forecast/quote"
                )
            if budget["forecast"] is not None:
                forecast = {row["name"]: row for row in budget["forecast"]["stages"]}
        for item in manifest["commands"]:
            validate_source_case(item)
        for index, item in enumerate(manifest["commands"][len(results) :], len(results) + 1):
            remaining = total - prior_elapsed - (clock() - started) - reserve
            if remaining <= 0:
                checkpoint("budget_exhausted")
                return 1
            if not stage_fits(forecast.get(item["name"]), remaining):
                checkpoint("stage_forecast_does_not_fit", item["name"])
                return 1
            if shutil.disk_usage(out).free < manifest["budgets"]["minimum_free_bytes"]:
                checkpoint("disk_budget_exhausted")
                return 1
            child = (
                isolated_environment(environment, item["device"])
                if item.get("device") is not None
                else dict(environment)
            )
            if item.get("device") is not None:
                child.update(
                    RNS8_MULTI_GPU_MODE="embarrassingly_parallel_shards",
                    RNS8_RANK=str(manifest["devices"].index(item["device"])),
                    RNS8_WORLD_SIZE=str(len(manifest["devices"])),
                )
            identity_before = None
            capture_request = None
            fresh_outputs = []
            directories = item.get("owned_artifact_directories", [])
            if (
                not isinstance(directories, list)
                or len(set(directories)) != len(directories)
                or any(not isinstance(name, str) for name in directories)
            ):
                raise ValueError("owned artifact directories need unique explicit paths")
            for name in directories:
                directory = Path(name).absolute()
                if (
                    not directory.is_relative_to(out.resolve())
                    or directory == out.resolve()
                    or directory.resolve() != directory
                    or directory.exists()
                ):
                    raise ValueError(
                        "acquisition requires a fresh session-owned artifact directory"
                    )
                directory.mkdir(parents=True, exist_ok=False)
            monitors = item.get("monitor_artifact_directories", [])
            if (
                not isinstance(monitors, list)
                or any(not isinstance(name, str) for name in monitors)
                or len(set(monitors)) != len(monitors)
            ):
                raise ValueError("artifact monitor roots need unique explicit paths")
            earlier_directories = (
                {
                    name
                    for command in manifest["commands"][: index - 1]
                    for name in command.get("owned_artifact_directories", [])
                }
                if monitors
                else set()
            )
            for name in monitors:
                directory = Path(name).absolute()
                if (
                    name not in earlier_directories
                    or name in directories
                    or not directory.is_relative_to(out.resolve())
                    or directory == out.resolve()
                    or directory.resolve() != directory
                    or not directory.is_dir()
                    or directory.is_symlink()
                    or directory.is_junction()
                ):
                    raise ValueError(
                        "artifact monitoring requires an earlier session-owned directory"
                    )
            outputs = item.get("evidence_outputs", [])
            if (
                not isinstance(outputs, list)
                or any(not isinstance(name, str) for name in outputs)
                or len(set(outputs)) != len(outputs)
            ):
                raise ValueError("native evidence outputs need a unique explicit path inventory")
            for name in outputs:
                path = (ROOT / name).resolve()
                if path.exists():
                    raise ValueError(
                        "native evidence output already exists before acquisition: " + str(path)
                    )
                fresh_outputs.append(path)
            if item.get("compiled_identity") or "capture" in item:
                executable = (ROOT / item["argv"][0]).resolve(strict=True)
                identity_before = sha256(executable)
                if "capture" in item:
                    capture_request = request_identity(item["argv"], environ=child)
            capture_options = {}
            if directories or monitors:
                if (
                    type(item.get("max_artifact_bytes")) is not int
                    or item["max_artifact_bytes"] < 1
                ):
                    raise ValueError("owned/monitored artifacts require a positive byte budget")
                capture_options.update(
                    artifact_roots=[Path(name) for name in (*directories, *monitors)],
                    max_artifact_bytes=item["max_artifact_bytes"],
                )
            if "max_output_bytes" in item:
                capture_options["max_output_bytes"] = item["max_output_bytes"]
            result = runner(
                item["argv"],
                cwd=ROOT,
                label=f"{index}/{len(manifest['commands'])} {item['name']}",
                timeout_seconds=min(item["timeout_seconds"], remaining),
                env=child,
                **capture_options,
            )
            for stream in ("stdout", "stderr"):
                (out / f"{item['name']}.{stream}.log").write_text(result[stream], encoding="utf-8")
            errors = validate_result(item, result, inspections.get(item.get("match_parent")))
            binary_identity = None
            device_binding = None
            if "inspect_target" in item and not errors:
                try:
                    inspected = parse_evidence_json(result["stdout"])
                    device_binding = inspection_relation(
                        inspected, inspections.get(item.get("match_parent"))
                    )
                    if item.get("parent_inspect"):
                        current = inspection_observation(inspected)
                        for earlier in results:
                            if not earlier.get("parent_inspect") or earlier.get(
                                "selected_runtime_ordinal"
                            ) == item.get("selected_device"):
                                continue
                            previous = inspection_observation(inspections[earlier["name"]])
                            if not distinct_execution_devices(current, previous):
                                errors.append(
                                    "selected parent ordinals have duplicate or ambiguous observed execution-device identities"
                                )
                                break
                except IdentityUnavailable as error:
                    device_binding = dict(state="unavailable", reasons=[str(error)])
                except IdentityMismatch as error:
                    device_binding = dict(state="mismatch", reasons=[str(error)])
                    errors.append("runtime device observations are contradictory")
            if item.get("compiled_identity") and not errors:
                try:
                    identity_output = ROOT / item["identity_output"]
                    write_json(identity_output, parse_evidence_json(result["stdout"]))
                    binary_identity = bind_document(
                        executable, identity_output, before_sha256=identity_before
                    )
                except (OSError, ValueError, TypeError):
                    errors.append("compiled binary changed or identity query could not be bound")
            if "capture" in item and not errors:
                try:
                    from benchmark_schema import validate_capture

                    capture = parse_evidence_json(result["stdout"])
                    if not isinstance(capture, dict):
                        raise ValueError("benchmark output is not an object")
                    if (
                        type(capture.get("process_id")) is not int
                        or capture["process_id"] <= 0
                        or capture["process_id"] != result.get("process_id")
                    ):
                        raise ValueError("benchmark JSON does not belong to the launched process")
                    if capture_request["executable_sha256"] != identity_before:
                        raise ValueError("benchmark executable changed before launch")
                    attach_capture_identity(
                        capture, capture_request, executable, execution_id=str(uuid.uuid4())
                    )
                    if capture["artifact_capture"]["unchanged_during_run"] is not True:
                        raise ValueError("benchmark executable changed during capture")
                    validate_capture(capture)
                    device = capture.get("device") or {}
                    if str(device.get("gcn_arch", "")).split(":")[0] != item.get(
                        "expected_target"
                    ) or capture.get("backend_requested") != item.get("expected_backend"):
                        raise ValueError(
                            "benchmark runtime target/backend differs from session request"
                        )
                    try:
                        device_binding = capture_relation(
                            capture,
                            inspections.get(item.get("observation_parent_stage")),
                            inspections.get(item.get("observation_child_stage")),
                        )
                        if device_binding["state"] != "verified":
                            raise IdentityMismatch(
                                "capture differs from selected parent/child observations"
                            )
                    except IdentityUnavailable as error:
                        device_binding = dict(state="unavailable", reasons=[str(error)])
                        if item.get("require_physical_identity"):
                            raise ValueError(
                                "capture runtime device association is unavailable"
                            ) from error
                    write_json(ROOT / item["capture"], capture)
                except (OSError, ValueError, TypeError, KeyError):
                    errors.append(
                        "benchmark capture has invalid schema, artifact, process, target, device or request association"
                    )
            if "inspect_target" in item and not errors:
                inspections[item["name"]] = parse_evidence_json(result["stdout"])
            try:
                artifacts = artifact_records(item, out)
            except (OSError, ValueError) as error:
                errors.append("artifact inventory could not be completed: " + str(error))
                # Preserve the failed stage and bounded logs even when native
                # files exceed budget or cannot be safely inventoried.
                artifacts = artifact_records({"name": item["name"], "argv": item["argv"]}, out)
            results.append(
                {
                    "name": item["name"],
                    **result,
                    "parent_inspect": item.get("parent_inspect", False),
                    "selected_runtime_ordinal": item.get("selected_device"),
                    "physical_uuid": binding_identifiers(device_binding)["uuid"],
                    "pci_bdf": binding_identifiers(device_binding)["pci_bdf"],
                    "runtime_inspection": "inspect_target" in item,
                    "capture_observation": "capture" in item,
                    "device_binding": device_binding,
                    "execution_environment": runtime_mapping_snapshot(child),
                    "compiled_binary_identity": binary_identity,
                    "fresh_evidence_outputs": {
                        str(path): artifacts[str(path)]
                        for path in fresh_outputs
                        if str(path) in artifacts
                    },
                    "validation_errors": errors,
                    "artifacts": artifacts,
                }
            )
            checkpoint("blocked" if errors else "in_progress")
            if errors:
                print(f"STOP: {item['name']}: {'; '.join(errors)}", file=sys.stderr, flush=True)
                return 1
        if manifest.get("session_budget") is not None:
            replay_budget(manifest["session_budget"], manifest["commands"])
    except (OSError, ValueError, TypeError) as error:
        checkpoint("blocked", str(error))
        print(f"STOP: session evidence could not be bound: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        # run_capture has already terminated its owned process group.
        checkpoint("interrupted")
        return 130
    checkpoint(
        "cpu_preparation_passed_hardware_pending"
        if manifest.get("session_kind") == "prepare" and results
        else "smoke_passed_release_qualification_pending"
        if results
        else "environment_receipt_only_not_hardware_qualified"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument(
        "--platform-lock",
        type=Path,
        help="operator-reviewed exact Linux tuple/file lock; required for live Linux GPU execution",
    )
    parser.add_argument(
        "--export-session",
        action="store_true",
        help="export stopped session evidence only; no discovery or execution",
    )
    parser.add_argument(
        "--export-dir", type=Path, help="explicit private local/mounted evidence destination"
    )
    parser.add_argument("--export-maximum-bytes", type=int, default=8 * 1024**3)
    parser.add_argument(
        "--devices",
        type=device_list,
        help="explicit unmasked parent HIP runtime ordinals; never inferred physical ids",
    )
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument(
        "--accelerators", action="store_true", help="explicitly select all optional backend lanes"
    )
    parser.add_argument(
        "--all-enabled",
        action="store_true",
        help="add a combined optional-accelerator build and backend smoke lane after isolated lanes",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the selected preparation or already available GPU session",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="explicit plan-only alias (the default)"
    )
    parser.add_argument(
        "--session-kind",
        choices=("prepare", "environment", "smoke", "first-pass", "multigpu"),
        default="first-pass",
    )
    parser.add_argument(
        "--preset",
        help="explicit existing configure preset; target and feature flags remain explicit",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="shared discovery dependency override",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only matching successful checkpoints and unchanged artifacts",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument(
        "--total-seconds",
        type=float,
        default=7200,
        help="whole session wall-clock cap including resumed command time",
    )
    parser.add_argument("--export-reserve-seconds", type=float, default=30)
    parser.add_argument(
        "--stage-forecast",
        type=Path,
        help="Complete operator duration ranges for exact command names",
    )
    parser.add_argument(
        "--price-per-hour", help="Operator-supplied decimal quote; no price lookup or provisioning"
    )
    parser.add_argument("--maximum-cost", help="Decimal ceiling under the supplied linear quote")
    parser.add_argument("--currency", help="Explicit three-letter quoted currency")
    parser.add_argument(
        "--price-scope", choices=("allocation", "per-selected-device"), default="allocation"
    )
    parser.add_argument(
        "--already-incurred-cost", help="Operator-supplied quoted cost already incurred"
    )
    parser.add_argument("--min-free-gib", type=float, default=2)
    parser.add_argument(
        "--minimum-host-memory-gib",
        type=float,
        default=0,
        help="Explicit measured workload/build headroom floor; zero leaves the model unavailable",
    )
    parser.add_argument(
        "--peak-build-job-memory-gib",
        type=float,
        default=0,
        help="Measured peak per build job; combined with jobs and reserve",
    )
    parser.add_argument("--host-memory-reserve-gib", type=float, default=0)
    parser.add_argument(
        "--artifact-inventory", type=Path, help="explicit frozen local artifact inventory"
    )
    parser.add_argument(
        "--artifact-cache", type=Path, help="cache-only artifact source; no network fallback"
    )
    parser.add_argument(
        "--stage-artifacts",
        action="store_true",
        help="copy verified local artifacts into out-dir/artifacts, without running the session",
    )
    args = parser.parse_args(argv)
    if args.export_session:
        if (
            args.out_dir is None
            or args.export_dir is None
            or args.execute
            or args.resume
            or args.dry_run
            or args.stage_artifacts
            or args.artifact_inventory
            or args.artifact_cache
            or args.devices is not None
            or args.backend
            or args.accelerators
            or args.all_enabled
            or args.platform_lock
            or args.set
            or args.stage_forecast
            or args.price_per_hour is not None
        ):
            parser.error(
                "export requires --out-dir/--export-dir and cannot run a session, discovery or staging"
            )
        try:
            receipt = export_session(
                args.out_dir, args.export_dir, maximum_bytes=args.export_maximum_bytes
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(receipt, sort_keys=True, indent=2))
        return 0
    if args.export_dir is not None:
        parser.error("--export-dir requires the separate --export-session mode")
    locked_platform = None
    if args.platform_lock:
        if TARGETS[args.target][1] != "Linux":
            parser.error("--platform-lock currently requires the native Linux target")
        try:
            locked_platform = load_platform_lock(args.platform_lock)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if bool(args.artifact_inventory) != bool(args.artifact_cache):
        parser.error("--artifact-inventory and --artifact-cache must be supplied together")
    if args.stage_artifacts and (
        not args.artifact_inventory or args.execute or args.resume or args.dry_run
    ):
        parser.error(
            "--stage-artifacts requires an inventory/cache and is separate from execute/resume/dry-run"
        )
    artifacts = artifact_receipt = None
    if args.artifact_inventory:
        try:
            artifacts = load_inventory(args.artifact_inventory)
            artifact_receipt = inspect_cache(artifacts, args.artifact_cache)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    preparation = args.session_kind == "prepare"
    if preparation:
        if (
            args.devices is not None
            or args.backend
            or args.accelerators
            or args.all_enabled
            or TARGETS[args.target][1] != "Linux"
        ):
            parser.error(
                "prepare requires a Linux target and no --devices, --backend or --accelerators"
            )
        args.devices = []
    elif args.devices is None:
        parser.error("GPU session planning requires explicit --devices")
    if (
        args.jobs < 1
        or args.jobs > 256
        or any(
            not math.isfinite(value) or value <= 0
            for value in (
                args.timeout_seconds,
                args.total_seconds,
                args.export_reserve_seconds,
                args.min_free_gib,
            )
        )
    ):
        parser.error("jobs must be 1..256 and time/disk budgets must be finite positive numbers")
    if args.total_seconds <= args.export_reserve_seconds:
        parser.error("total session budget must exceed the export reserve")
    if (
        any(
            not math.isfinite(value) or not 0 <= value < 2**34
            for value in (
                args.minimum_host_memory_gib,
                args.peak_build_job_memory_gib,
                args.host_memory_reserve_gib,
            )
        )
        or args.min_free_gib >= 2**34
    ):
        parser.error("resource GiB budgets must be finite nonnegative values inside uint64 bytes")
    if int(args.min_free_gib * 1024**3) < 1:
        parser.error("minimum free-space budget must be at least one byte")
    if TARGETS[args.target][1] != "Linux" and any(
        (args.minimum_host_memory_gib, args.peak_build_job_memory_gib, args.host_memory_reserve_gib)
    ):
        parser.error("host memory model options currently require the native Linux session target")
    if args.execute and args.dry_run or args.resume and not args.execute:
        parser.error("--execute conflicts with --dry-run; --resume requires --execute")
    if args.accelerators and args.backend:
        parser.error("choose --accelerators or explicit --backend lanes, not both")
    if args.session_kind == "smoke" and len(args.devices) != 1:
        parser.error(
            "smoke requires exactly one explicit device; use multigpu for independent device lanes"
        )
    out = (args.out_dir or ROOT / "temp" / "qualification" / args.target).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists() and not args.resume:
        parser.error(
            "results.json already exists; choose a fresh --out-dir or explicit matching --resume"
        )
    backends = [] if preparation else list(dict.fromkeys(args.backend or BACKENDS))
    command_out = Path(os.path.relpath(out, ROOT))
    try:
        request = dependency_request(
            args.target,
            list(BACKENDS) if args.all_enabled else backends,
            args.set,
            preparation=preparation,
        )
    except ValueError as error:
        parser.error(str(error))
    environment = build_environment(os.environ)
    # Use the same write-free selection as doctor/CMake. Cross-host planning
    # does not claim the planning machine's executables are target prerequisites.
    # Supported dependency environment values participate in selection without
    # forwarding arbitrary authentication or cloud variables to child commands.
    from session_environment import safe_value

    discovery_env = (
        {
            key: value
            for key, value in os.environ.items()
            if (key in REQUEST_KEYS or key in environment) and safe_value(value) == value
        }
        if platform.system() == request.host
        else {}
    )
    resolution = resolve(request, discovery_env)
    environment.update({key: value for key, value in discovery_env.items() if key in REQUEST_KEYS})
    blockers = readiness_blockers(args, resolution, os.environ)
    if TARGETS[args.target][1] == "Linux" and not preparation and locked_platform is None:
        blockers.append(
            "live Linux GPU execution requires --platform-lock from an operator-reviewed tuple"
        )
    if artifact_receipt and not artifact_receipt["complete"]:
        blockers += [
            "artifact " + row["id"] + ": " + row["status"]
            for row in artifact_receipt["artifacts"]
            if row["status"] != "verified"
        ]
    from session_environment import BUILD_ENV_KEYS, ALLOWLIST

    blockers += [
        key
        + " contains an unsafe environment value (redacted); select a local prerequisite path explicitly"
        for key, value in os.environ.items()
        if key in {*REQUEST_KEYS, *BUILD_ENV_KEYS, *ALLOWLIST} and safe_value(value) != value
    ]
    commands = command_plan(
        args.target,
        backends,
        command_out,
        args.jobs,
        args.devices,
        bindings={**request.overrides, **resolution["cmake_bindings"]},
        session_kind=args.session_kind,
        timeout=args.timeout_seconds,
        preset_override=args.preset,
        all_enabled=args.all_enabled,
        resource_policy=dict(
            minimum_free_bytes=int(args.min_free_gib * 1024**3),
            minimum_memory_bytes=int(args.minimum_host_memory_gib * 1024**3),
            peak_job_memory_bytes=int(args.peak_build_job_memory_gib * 1024**3),
            memory_reserve_bytes=int(args.host_memory_reserve_gib * 1024**3),
        ),
        platform_lock=locked_platform,
    )
    if locked_platform is not None and args.session_kind != "environment":
        commands.append(
            dict(
                name="linux-platform-lock-closeout",
                argv=[
                    commands[0]["argv"][0],
                    "tools/session_platform_contract.py",
                    "--lock",
                    locked_platform["path"],
                ],
                depends_on=[commands[-1]["name"]],
                timeout_seconds=args.timeout_seconds,
                platform_lock=locked_platform,
            )
        )
    for item in commands:
        for index, value in enumerate(item["argv"]):
            if value == "cmake" and resolution["tools"]["cmake"]["path"]:
                item["argv"][index] = resolution["tools"]["cmake"]["path"]
            elif value == "ctest" and resolution["tools"]["cmake"]["path"]:
                item["argv"][index] = str(
                    Path(resolution["tools"]["cmake"]["path"]).with_name(
                        "ctest.exe" if request.host == "Windows" else "ctest"
                    )
                )
    try:
        session_budget = plan_budget(
            commands,
            total_seconds=args.total_seconds,
            reserve_seconds=args.export_reserve_seconds,
            forecast_path=args.stage_forecast,
            price_per_hour=args.price_per_hour,
            maximum_cost=args.maximum_cost,
            currency=args.currency,
            price_scope=args.price_scope,
            selected_devices=len(args.devices),
            already_incurred_cost=args.already_incurred_cost,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    blockers += session_budget["blockers"]
    manifest = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "requested_architecture": TARGETS[args.target][0],
        "observed_architecture": None,
        "source": source_receipt(exclude=out),
        "backends": backends,
        "source_operation_contract": dict(
            path=str(SOURCE_GEMM_CONTRACT),
            sha256=load_source_gemm_contract().source_sha256,
            version=1,
        ),
        "inapplicable_source_cases": [
            dict(
                backend=backend,
                expected_status="RNS8_UNSUPPORTED_BACKEND",
                negative_check_owner="existing_native_semantic_admission_tests",
                **case,
            )
            for backend in dict.fromkeys([*backends, *(BACKENDS if args.all_enabled else [])])
            for case in session_case_inventory(backend, TARGETS[args.target][0], args.session_kind)
            if not case["admission"]["source_eligible"]
        ],
        "all_enabled_lane": args.all_enabled,
        "devices": args.devices,
        "device_namespace": "none_cpu_preparation"
        if preparation
        else "unmasked_parent_HIP_runtime_ordinal",
        "physical_identity_verified": False,
        "session_kind": args.session_kind,
        "dependency_resolution_sha256": resolution["identity_sha256"],
        "platform_lock": locked_platform,
        "session_budget": session_budget,
        "artifact_inventory": artifacts,
        "artifact_cache_receipt": artifact_receipt,
        "selected_tool_files": {
            name: {"path": item["path"], "sha256": sha256(Path(item["path"]))}
            for name, item in resolution["tools"].items()
            if item.get("path") and Path(item["path"]).is_file()
        },
        "execution_blockers": blockers,
        "environment_allowlist": snapshot(os.environ),
        "child_build_environment_sha256": identity(
            {**environment, **resolution["compiler_environment"]}
        ),
        "budgets": {
            "total_seconds": session_budget["effective_total_seconds"],
            "export_reserve_seconds": args.export_reserve_seconds,
            "minimum_free_bytes": int(args.min_free_gib * 1024**3),
            "jobs": args.jobs,
        },
        "status": "planned_not_executed",
        "promotion_allowed": False,
        "preparation_only": preparation,
        "hardware_qualified": False,
        "command_working_directory": str(ROOT),
        "environment": {"RNS8_AUTOTUNE_CACHE": str(command_out / "isolated-empty-cache.json")},
        "commands": commands,
        "kernels": kernel_inventory(),
        "remaining_gates": [
            "HIP compile and hardware execution for every selected backend",
            "full correctness and backend-specific ISA checks",
            "large/padded/K-boundary and sparse target coverage",
            "release benchmarks, timing variance, profiler/counter evidence and explicit cache review",
            "physical UUID/BDF correlation, NUMA affinity, profiler/RCCL inventory and remote Linux rehearsal",
        ],
    }
    manifest["identity_sha256"] = identity(
        {key: value for key, value in manifest.items() if key not in {"generated_utc", "status"}}
    )
    if args.resume:
        try:
            resume_results(out, manifest)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    write_json(out / "dependency-resolution.json", resolution)
    write_json(
        out / "environment-allowlist.json",
        {"schema": "rns8_session_environment_v1", "values": snapshot(os.environ)},
    )
    write_json(out / "manifest.json", manifest)
    print(
        f"{len(commands)} commands, {len(manifest['kernels'])} source kernel definitions: {out / 'manifest.json'}",
        flush=True,
    )
    if artifact_receipt is not None:
        write_json(out / "artifact-cache.json", artifact_receipt)
    if args.stage_artifacts:
        previous_handler = signal.getsignal(signal.SIGTERM)

        def interrupted_staging(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted_staging)
        try:
            staged = stage_cache(
                artifacts,
                args.artifact_cache,
                out / "artifacts",
                reserve_bytes=manifest["budgets"]["minimum_free_bytes"],
            )
        except KeyboardInterrupt:
            write_json(
                out / "artifact-staging.json",
                {"status": "interrupted", "hardware_qualified": False},
            )
            return 130
        except (OSError, ValueError) as error:
            write_json(
                out / "artifact-staging.json",
                {"status": "incomplete", "error": str(error), "hardware_qualified": False},
            )
            print("Artifact staging blocked: " + str(error), file=sys.stderr)
            return 1
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        write_json(
            out / "artifact-staging.json",
            {"status": "staged_runtime_qualification_pending", **staged},
        )
        print("Verified local artifacts staged; no installation or session execution.", flush=True)
        return 0
    if not args.execute:
        print(
            "Plan only. No build, GPU work, benchmark, or cache promotion was executed.", flush=True
        )
        invalid = any(
            item.get("status") in {"invalid", "ambiguous"}
            for item in (
                resolution["hip"],
                resolution["vcpkg"],
                *resolution["optional_components"].values(),
            )
        )
        invalid = invalid or any(
            "Linux/UNIX builds" in error or error.startswith("explicit ")
            for error in resolution["errors"]
        )
        if invalid:
            print(
                "Invalid/ambiguous explicit selection: " + "; ".join(resolution["errors"]),
                file=sys.stderr,
            )
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
