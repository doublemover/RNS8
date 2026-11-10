"""Plan or explicitly build bounded native C++ production mutation copies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil

from analysis_inputs import cache_values, target_artifacts
from build_manifest import configuration_options, source_content_digest
from compiled_binary_identity import canonical_digest
from execution_inventory import bounded_read
from host_instrumentation_session import (
    Session,
    check_bookends,
    invocation_path,
    owned_output_path,
    reference,
    valid_total_budget,
    write_json,
)
from json_evidence import parse_evidence_json, same_evidence_value
from native_mutation_contract import assertion_result, change, copy_sources, tracked_inputs

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "metadata/native-production-mutations.json"
OFF = (
    "RNS8_ENABLE_HIP",
    "RNS8_ENABLE_HIPBLASLT",
    "RNS8_ENABLE_CK",
    "RNS8_ENABLE_ROCWMMA",
    "RNS8_ENABLE_AMDGPU_BUILTINS",
    "RNS8_ENABLE_CPU_AMX",
    "RNS8_ENABLE_CPU_I8MM",
    "RNS8_ENABLE_CPU_PARALLEL",
    "RNS8_ENABLE_HOST_COVERAGE",
    "RNS8_ENABLE_WINDOWS_CLANG_ASAN",
    "RNS8_BUILD_SHARED",
    "RNS8_BUILD_TOOLS",
    "RNS8_BUILD_BENCHMARKS",
    "RNS8_BUILD_EXAMPLES",
    "RNS8_BUILD_FUZZERS",
    "RNS8_BUILD_SANITIZER_CONTROLS",
    "RNS8_PROBE_ACCELERATORS",
    "RNS8_ENABLE_GMP",
    "RNS8_ENABLE_FLINT",
    "RNS8_ENABLE_DISTRIBUTED_MPI",
    "RNS8_ENABLE_RCCL",
    "RNS8_ENABLE_ROCSHMEM",
    "RNS8_ENABLE_ROCTX",
    "RNS8_ENABLE_ETW_RANGES",
)


def registry():
    value = parse_evidence_json(bounded_read(REGISTRY)[0])
    if (
        not isinstance(value, dict)
        or value.get("schema") != "rns8_native_production_mutations_v1"
        or not isinstance(value.get("seeds"), list)
        or not 1 <= len(value["seeds"]) <= 16
    ):
        raise ValueError("bounded native production mutation registry required")
    seen = set()
    for seed in value["seeds"]:
        if (
            not isinstance(seed, dict)
            or set(seed)
            != {"id", "source", "source_sha256", "anchor", "before", "after", "tests", "mandatory"}
            or seed.get("mandatory") is not True
            or any(
                type(seed.get(key)) is not str or not seed[key] or len(seed[key]) > 4096
                for key in ("id", "source", "source_sha256", "anchor", "before", "after")
            )
            or not re.fullmatch(r"[a-z][a-z0-9-]+", seed["id"])
            or seed["id"] in seen
            or not re.fullmatch(r"src/[A-Za-z0-9_/]+\.(?:cpp|hpp|inc)", seed["source"])
            or not re.fullmatch(r"[0-9a-f]{64}", seed["source_sha256"])
        ):
            raise ValueError("invalid native production mutation identity")
        seen.add(seed["id"])
        tests = seed.get("tests")
        if not isinstance(tests, list) or not tests or len(tests) > 16:
            raise ValueError("explicit native assertion cases required")
        for test in tests:
            if (
                not isinstance(test, dict)
                or set(test) != {"name", "source", "assertion_marker"}
                or any(
                    type(item) is not str
                    or not item
                    or len(item) > 4096
                    or "\n" in item
                    or "\r" in item
                    for item in test.values()
                )
                or not re.fullmatch(
                    r"tests/(?:unit|support)/[A-Za-z0-9_]+\.(?:cpp|hpp|inc)", test["source"]
                )
            ):
                raise ValueError("native required test/source/assertion mapping invalid")
        change((ROOT / seed["source"]).read_bytes(), seed)
    return value["seeds"]


def configure_command(args, source, build):
    command = [
        str(args.cmake),
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_CXX_COMPILER=" + str(args.compiler),
        "-DCMAKE_MAKE_PROGRAM=" + str(args.ninja),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=",
        "-DCMAKE_C_COMPILER_LAUNCHER=",
        "-DPython3_EXECUTABLE=" + str(args.python),
        "-DBUILD_TESTING=ON",
        "-DRNS8_BUILD_TESTS=ON",
        "-DRNS8_HOST_SANITIZER=none",
        "-DRNS8_WARNINGS_AS_ERRORS=OFF",
    ]
    command.extend("-D" + key + "=OFF" for key in OFF)
    if args.toolchain:
        command.extend(
            ["-DCMAKE_TOOLCHAIN_FILE=" + str(args.toolchain), "-DVCPKG_MANIFEST_INSTALL=OFF"]
        )
    if args.vcpkg_installed:
        command.append("-DVCPKG_INSTALLED_DIR=" + str(args.vcpkg_installed))
    return command


def copied_identity(capture, value, expected, source_digest):
    if (
        not isinstance(expected, dict)
        or not isinstance(value, dict)
        or value.get("schema") != "rns8_compiled_binary_identity_v1"
        or type(value.get("process_id")) is not int
        or value["process_id"] != capture.get("process_id")
        or not same_evidence_value(value.get("build_manifest"), expected)
        or not same_evidence_value(value.get("library_build_manifest"), expected)
        or expected.get("source_digest") != source_digest
        or expected.get("source_commit") != "unknown"
        or expected.get("source_dirty") is not True
        or type(value.get("library_abi_declaration_sha256")) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", value["library_abi_declaration_sha256"])
    ):
        raise ValueError("copied native executable/library/source identity differs")
    return dict(
        document=value,
        source_digest=source_digest,
        provenance_qualified=False,
        promotion_eligible=False,
        scope="copied_native_cpu_correctness_identity_not_target_qualification",
    )


def compiled_inputs(session, args, source, build, seed):
    database = build / "compile_commands.json"
    rows = parse_evidence_json(bounded_read(database)[0])
    if not isinstance(rows, list) or not rows:
        raise ValueError("native mutation lacks real configured compilation inventory")
    if seed:
        target = (source / seed["source"]).resolve()
        if target.suffix == ".cpp":
            if not any(
                isinstance(row, dict) and Path(row.get("file", "")).resolve() == target
                for row in rows
            ):
                raise ValueError("mutated native translation unit was not selected for compilation")
        else:
            result = session.run(
                seed["id"] + "-actual-header-dependencies",
                [str(args.ninja), "-C", str(build), "-t", "deps"],
            )
            dependencies = {
                (build / Path(line.strip().replace("\\", "/"))).resolve()
                for line in result["stdout"].splitlines()
                if line.startswith("    ") and line.strip()
            }
            if target not in dependencies:
                raise ValueError(
                    "mutated native header/fragment absent from actual compiler dependencies"
                )
    return reference(database)


def variant(session, args, inputs, *, name, tests, seed=None, compiler_flags=None):
    directory = session.output / name
    directory.mkdir()
    source = directory / "source"
    copy_sources(inputs, source)
    changed = None
    if seed:
        path = source / seed["source"]
        original = next(row for row in inputs["files"] if row["relative_path"] == seed["source"])
        before = {key: original[key] for key in ("path", "sha256", "bytes")}
        path.write_bytes(change(path.read_bytes(), seed))
        changed = dict(before=before, after=reference(path), operation=seed)
    digest = source_content_digest(source)
    build = directory / "build"
    row = dict(
        source_root=str(source),
        source_digest=digest,
        source_snapshot_identity=inputs["identity_sha256"],
        promotion_eligible=False,
    )
    row["source_files"] = [
        dict(
            path=str(source / item["relative_path"]),
            sha256=changed["after"]["sha256"]
            if changed and item["relative_path"] == seed["source"]
            else item["sha256"],
        )
        for item in inputs["files"]
    ]
    if changed:
        row["mutation"] = changed
    configure = configure_command(args, source, build)
    if compiler_flags is not None:
        configure.append("-DCMAKE_CXX_FLAGS=" + compiler_flags)
        row["compiler_transform_flags"] = compiler_flags
    session.run(name + "-configure", configure)
    cache = cache_values(build / "CMakeCache.txt")
    if (
        any(cache.get(key, "OFF").upper() not in {"OFF", "FALSE", "NO", "0", ""} for key in OFF)
        or cache.get("RNS8_HOST_SANITIZER") != "none"
    ):
        raise ValueError(
            "native mutation build enabled an excluded hardware/instrumentation feature"
        )
    built = session.run(
        name + "-build",
        [
            str(args.cmake),
            "--build",
            str(build),
            "--target",
            "rns8_tests",
            "--parallel",
            str(args.jobs),
            "--verbose",
        ],
        required=False,
    )
    if not session.stages[-1]["succeeded"]:
        row["classification"] = (
            "timeout"
            if built.get("timed_out")
            else "infrastructure_error"
            if built.get("capture_error") or built.get("limit_exceeded")
            else "compile_invalid"
        )
        return row, None
    row["compilation_inventory"] = compiled_inputs(session, args, source, build, seed)
    document = build / "rns8-analysis-targets-Debug.json"
    targets = target_artifacts(parse_evidence_json(bounded_read(document)[0]), "Debug")
    if any(
        not path.is_relative_to(build.resolve())
        for name, (path, _) in targets.items()
        if name in {"rns8_tests", "rns8_static"}
    ):
        raise ValueError("native test/library artifact escaped its owned build")
    artifacts = {
        name: reference(path)
        for name, (path, _) in targets.items()
        if name in {"rns8_tests", "rns8_static"}
    }
    if set(artifacts) != {"rns8_tests", "rns8_static"}:
        raise ValueError("actual native test/static-library artifacts missing")
    row["artifacts"] = artifacts
    manifest_path = build / "generated/Debug/rns8_build_manifest.json"
    manifest = parse_evidence_json(bounded_read(manifest_path)[0])
    options = configuration_options(manifest["configuration"], strict=True)
    if (
        options.get("hip") != "OFF"
        or options.get("host_sanitizer") != "none"
        or options.get("host_coverage") != "OFF"
    ):
        raise ValueError("compiled native mutation identity differs from CPU-only scope")
    capture, identity = session.json(
        name + "-compiled-identity", [artifacts["rns8_tests"]["path"], "--build-identity"]
    )
    row["compiled_identity"] = copied_identity(capture, identity, manifest, digest)
    row["configuration_domain"] = {
        key: options.get(key)
        for key in (
            "compiler",
            "compiler_path",
            "flags",
            "config",
            "hip",
            "host_sanitizer",
            "host_coverage",
            "openmp",
        )
    }
    row["manifest"] = reference(manifest_path)
    inventory_ref, inventory = session.tests(
        name + "-cases", build, args.ctest, required=tests, selection_names=tests
    )
    row["test_inventory"] = inventory_ref
    if source_content_digest(source) != digest:
        raise ValueError("native source snapshot changed during build/test")
    check_bookends([row, inventory])
    if seed:
        source_xml = bounded_read(Path(inventory["inputs"]["junit"]["path"]))[0]
        row["classification"] = assertion_result(seed, inventory, source_xml)
    else:
        row["classification"] = (
            "baseline_passed" if inventory["status"] == "passed" else "baseline_failed"
        )
    return row, inventory


def execute(args, plan, seeds):
    output = Path(plan["output"])
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("GIT_", "COVERAGE_", "ASAN_", "UBSAN_", "TSAN_", "LSAN_", "LLVM_PROFILE_")
        )
    }
    if any(key.startswith("MULL_") for key in environment) or any(
        environment.get(key)
        for key in (
            "LD_PRELOAD",
            "LD_AUDIT",
            "CCC_OVERRIDE_OPTIONS",
            "CLANG_CONFIG_FILE_USER_DIR",
            "CLANG_CONFIG_FILE_SYSTEM_DIR",
            "CL",
            "_CL_",
            "MULL_CONFIG",
        )
    ):
        raise ValueError("native mutation build rejects hidden compiler/runtime injection")
    for key in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"):
        environment.pop(key, None)
    environment.update(
        PYTHONOPTIMIZE="", PYTHONDONTWRITEBYTECODE="1", CTEST_NO_TESTS_ACTION="error"
    )
    output.mkdir(parents=True)
    session = Session(
        output,
        environment,
        args.timeout_seconds,
        args.artifact_mib * 1024**2,
        total_seconds=getattr(args, "total_seconds", None),
    )
    result = dict(
        schema="rns8_native_production_mutation_session_v1",
        status="blocked",
        plan=plan,
        blockers=[],
        mutants=[],
        scope="native_cpu_arithmetic_publication_and_mock_host_ownership_not_gpu_faults",
        production_latency=False,
        promotion_eligible=False,
        mull="not_run",
    )
    try:
        result["tools"] = {
            key: reference(getattr(args, key))
            for key in ("cmake", "ctest", "compiler", "ninja", "python")
        }
        result["registry"] = reference(REGISTRY)
        if args.mull_runner is not None:
            from mull_pilot import inspect_installation

            result["mull_admission"] = inspect_installation(session, args)
        if args.toolchain:
            result["toolchain"] = reference(args.toolchain)
        versions = {
            key: session.run(key + "-version", [str(getattr(args, key)), "--version"])["stdout"]
            for key in ("cmake", "ctest", "ninja", "python")
        }
        if versions["cmake"].splitlines()[0].replace("cmake ", "") != versions[
            "ctest"
        ].splitlines()[0].replace("ctest ", ""):
            raise ValueError("matching native CMake/CTest required")
        result["tool_versions"] = versions
        git = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(ROOT)]
        git_root = session.run("source-worktree-root", [*git, "rev-parse", "--show-toplevel"])[
            "stdout"
        ].strip()
        if Path(git_root).resolve() != ROOT:
            raise ValueError("actual source worktree root required")
        status = session.run(
            "source-clean-state",
            [*git, "status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=all"],
        )["stdout"]
        if status:
            raise ValueError("native source snapshots require all development committed first")
        result["source_revision"] = session.run("source-revision", [*git, "rev-parse", "HEAD"])[
            "stdout"
        ].strip()
        indexed = session.run("source-index", [*git, "ls-files", "--stage", "-z"])
        inputs = tracked_inputs(ROOT, indexed["stdout"], max_bytes=args.source_mib * 1024**2)
        result["source_inputs"] = write_json(output / "source-inputs.json", inputs)
        tests = sorted({test["name"] for seed in seeds for test in seed["tests"]})
        baseline, _ = variant(session, args, inputs, name="baseline", tests=tests)
        result["baseline"] = baseline
        if baseline["classification"] != "baseline_passed":
            result["blockers"].append("native_baseline_inconclusive:" + baseline["classification"])
        else:
            for seed in seeds:
                try:
                    row, _ = variant(session, args, inputs, name=seed["id"], tests=tests, seed=seed)
                    if row["classification"] == "killed_by_assertion":
                        original = baseline["compiled_identity"]["document"]["build_manifest"]
                        candidate = row["compiled_identity"]["document"]["build_manifest"]
                        if original.get("dependency_digest") != candidate.get(
                            "dependency_digest"
                        ) or not same_evidence_value(
                            baseline["configuration_domain"], row["configuration_domain"]
                        ):
                            row["classification"] = "infrastructure_error"
                            row["error"] = (
                                "native baseline/mutant dependency or compiler/configuration domain differs"
                            )
                except InterruptedError:
                    raise
                except (OSError, ValueError, TypeError, KeyError) as error:
                    row = dict(classification="infrastructure_error", error=str(error))
                row["id"] = seed["id"]
                result["mutants"].append(row)
                if row["classification"] != "killed_by_assertion":
                    result["blockers"].append(
                        "mandatory_native_mutation_unproven:"
                        + seed["id"]
                        + ":"
                        + row["classification"]
                    )
        if args.mull_runner is not None and not result["blockers"]:
            from mull_pilot import collect

            result["mull"] = collect(session, args, inputs, tests, result["mull_admission"])
        check_bookends([result, inputs, session.stages])
        result["status"] = "failed" if result["blockers"] else "passed"
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["blockers"].append(str(error))
    result["stages"] = session.stages
    result["counts"] = {
        key: sum(row["classification"] == key for row in result["mutants"])
        for key in (
            "killed_by_assertion",
            "survived",
            "not_covered",
            "compile_invalid",
            "timeout",
            "infrastructure_error",
            "unintended_assertion_failure",
        )
    }
    result["identity_sha256"] = canonical_digest(result)
    write_json(output / "report.json", result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                report=str(output / "report.json"),
                blockers=result["blockers"],
            )
        )
    )
    return 0 if result["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cmake", "ctest", "compiler", "python"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--ninja", type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--vcpkg-installed", type=Path)
    parser.add_argument(
        "--mull-runner", type=Path, help="optional installed native Mull0.37 runner"
    )
    parser.add_argument(
        "--mull-plugin", type=Path, help="exact matching installed native LLVM plugin"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="explicit absolute scratch root; default: repository temp",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--seed",
        action="append",
        help="select a registered production mutation; repeat to run a focused sensitivity lane",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--total-seconds", type=int, help="total owned child-process budget")
    parser.add_argument("--artifact-mib", type=int, default=32768)
    parser.add_argument("--source-mib", type=int, default=512)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.ninja is None:
            found = shutil.which("ninja")
            if not found:
                raise ValueError("native Ninja required; supply --ninja")
            args.ninja = Path(found)
        if (args.mull_runner is None) != (args.mull_plugin is None):
            raise ValueError("optional Mull pilot requires both --mull-runner and --mull-plugin")
        if args.mull_runner is not None and (
            platform.system() != "Linux" or "microsoft" in platform.release().lower()
        ):
            raise ValueError("optional Mull pilot is native Linux only; no Windows or HIP mutation")
        for key in (
            "cmake",
            "ctest",
            "compiler",
            "python",
            "ninja",
            "toolchain",
            "mull_runner",
            "mull_plugin",
        ):
            value = getattr(args, key)
            if value is not None:
                setattr(args, key, invocation_path(value))
        if platform.system() == "Windows":
            if args.toolchain is None:
                raise ValueError("Windows native mutation build requires explicit vcpkg toolchain")
            if args.vcpkg_installed is None:
                args.vcpkg_installed = ROOT / "vcpkg_installed"
        if args.vcpkg_installed:
            args.vcpkg_installed = args.vcpkg_installed.resolve(strict=True)
            if not args.vcpkg_installed.is_dir():
                raise ValueError("actual vcpkg installed directory required")
        artifact_root = args.artifact_root or ROOT / "temp"
        output = owned_output_path(args.out, artifact_root)
        if (
            not 1 <= args.jobs <= 64
            or not 1 <= args.timeout_seconds <= 14400
            or not 512 <= args.artifact_mib <= 65536
            or not 32 <= args.source_mib <= 2048
            or not valid_total_budget(args.total_seconds)
        ):
            raise ValueError("native mutation budgets exceed supported bounds")
        seeds = registry()
        if args.seed:
            selected = set(args.seed)
            available = {seed["id"] for seed in seeds}
            if len(selected) != len(args.seed) or not selected <= available:
                raise ValueError("mutation selection must name distinct registered seeds")
            seeds = [seed for seed in seeds if seed["id"] in selected]
        plan = dict(
            schema="rns8_native_production_mutation_plan_v1",
            artifact_root=str(artifact_root.resolve()),
            total_seconds=args.total_seconds,
            output=str(output),
            seed_ids=[seed["id"] for seed in seeds],
            selected_registry_subset=bool(args.seed),
            baseline_builds=1 + int(args.mull_runner is not None),
            mutant_builds=len(seeds),
            build_target="rns8_tests",
            gpu_enabled=False,
            shared_libraries=False,
            source_mib=args.source_mib,
            artifact_mib=args.artifact_mib,
            runtime_validation_executed=False,
            production_latency=False,
            configure_example=configure_command(
                args, output / "baseline/source", output / "baseline/build"
            ),
            optional_mull=dict(
                requested=args.mull_runner is not None,
                scope="two_native_cpu_production_files_diagnostic_only",
                runtime_validation_executed=False,
            ),
        )
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        if platform.system() not in {"Windows", "Linux"}:
            raise ValueError("native mutation lane supports Windows/Linux CPU only")
        return execute(args, plan, seeds)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("native production mutations: " + str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
