"""Plan/observe native GPU-ASan; optionally select bounded ordinary correctness."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys

from analysis_inputs import cache_values, load_inputs, target_artifacts
from build_manifest import source_content_digest
from check_dependencies_lib.resolution import header_version
from compiled_binary_identity import canonical_digest
from device_asan_contract import (
    RELEASE,
    TARGETS,
    CORRECTNESS_CASES,
    agent_observation,
    compiler_flags,
    dynamic_dependencies,
    host_prerequisites,
    hip_component_version,
    support,
)
from execution_inventory import bounded_read
from host_instrumentation_contract import native_argv, object_path
from host_instrumentation_session import (
    ROOT,
    Session,
    check_bookends,
    invocation_path,
    reference,
    source_inventory,
    version_number,
    write_json,
)
from isa_common import extract_code_object
from json_evidence import parse_evidence_json
from session_environment import build_environment

TOOLS = {
    "compiler": ("bin/amdclang++", "lib/llvm/bin/clang++", "llvm/bin/clang++", "bin/clang++"),
    "hipcc": ("bin/hipcc",),
    "rocminfo": ("bin/rocminfo",),
    **{
        name.replace("-", "_"): tuple(
            prefix + "/" + name for prefix in ("lib/llvm/bin", "llvm/bin", "bin")
        )
        for name in ("llvm-nm", "llvm-objcopy", "llvm-readelf", "clang-offload-bundler")
    },
}


def tool(root: Path, supplied: Path | None, names: tuple[str, ...]) -> Path:
    selected = supplied or next((root / name for name in names if (root / name).is_file()), None)
    if selected is None:
        raise ValueError("selected SDK tool unavailable: " + names[0])
    selected = invocation_path(selected)
    if not selected.resolve().is_relative_to(root):
        raise ValueError("device-ASan tools must belong to the explicit SDK prefix")
    return selected


def runtime(root: Path, pattern: str, explicit: Path | None) -> Path:
    if explicit:
        paths = {explicit.resolve(strict=True)}
    else:
        directories = [root / part for part in ("lib/asan", "lib64/asan", "lib", "lib64")]
        paths = {
            path.resolve()
            for directory in directories
            for path in directory.glob(pattern)
            if path.is_file()
        }
    if len(paths) != 1:
        raise ValueError("one exact SDK runtime required; supply its explicit path: " + pattern)
    selected = paths.pop()
    if not selected.is_file() or not selected.is_relative_to(root):
        raise ValueError("runtime must be a regular file inside the selected SDK")
    return selected


def kernel_configuration(release: str) -> tuple[str, dict]:
    # Only the running kernel's native configuration, never a caller's boolean.
    for path in (Path("/proc/config.gz"), Path("/boot") / ("config-" + release)):
        if path.is_file():
            before = reference(path)
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rb") as handle:
                data = handle.read(4 * 1024**2 + 1)
            if len(data) > 4 * 1024**2 or before != reference(path):
                raise ValueError("native kernel configuration changed or exceeds budget")
            return data.decode("utf-8"), before
    raise ValueError("running kernel HMM build configuration unavailable")


def inspect_build(session: Session, args, build: Path, runtimes: dict[str, Path]) -> dict:
    inputs = load_inputs(build)
    cache = cache_values(build / "CMakeCache.txt")
    if Path(cache["RNS8_HIP_LIBRARY"]).resolve(strict=True) != runtimes["hip"]:
        raise ValueError("configured HIP runtime differs from inspected instrumented runtime")
    if (
        cache.get("RNS8_ENABLE_DEVICE_ASAN", "").upper() != "ON"
        or cache.get("RNS8_DEVICE_ASAN_ROCM_VERSION") != args.release
        or cache.get("RNS8_AMDGPU_TARGETS") != args.target
    ):
        raise ValueError("configured build differs from reviewed device-ASan tuple")
    rows, device_sources = [], set()
    for index, context in enumerate(inputs.contexts):
        record = context.native_record()
        argv = native_argv(record, windows=False)
        expected = args.hipcc if context.kind == "hip" else args.compiler
        if Path(argv[0]).resolve() != expected.resolve():
            raise ValueError("actual first-party compiler differs from selected AMD SDK")
        flags = compiler_flags(argv, target=args.target if context.kind == "hip" else None)
        artifact = object_path(record, argv, build, windows=False)
        row = dict(
            source=reference(context.source),
            object=reference(artifact),
            context_sha256=context.identity,
            flags=flags,
        )
        symbols = session.run(
            "object-symbols-" + str(index), [str(args.llvm_nm), "--format=posix", str(artifact)]
        )["stdout"]
        row["host_asan_symbol_observed"] = bool(re.search(r"\b__asan_", symbols))
        if context.kind == "hip":
            device_sources.add(context.source.name)
            scratch = session.output / ("device-object-" + str(index))
            scratch.mkdir()
            sequence = 0

            def invoke(command, index=index):
                nonlocal sequence
                sequence += 1
                return session.run("device-extract-" + str(index) + "-" + str(sequence), command)[
                    "stdout"
                ]

            code = extract_code_object(
                str(args.llvm_objcopy),
                str(args.clang_offload_bundler),
                artifact,
                args.target,
                scratch,
                "kernel.fatbin",
                run=invoke,
            )
            native = session.run(
                "device-symbols-" + str(index), [str(args.llvm_nm), "--format=posix", str(code)]
            )["stdout"]
            if not re.search(r"\b__asan_(?:report_(?:load|store)|(?:load|store))\w*", native):
                raise ValueError(
                    "actual extracted device object lacks observed ASan access/report sites"
                )
            notes = session.run(
                "device-elf-" + str(index),
                [str(args.llvm_readelf), "--file-header", "--notes", str(code)],
            )["stdout"]
            if not re.search(r"AMDGPU|AMD GPU", notes):
                raise ValueError("extracted device artifact is not observed AMDGPU ELF")
            row["device_artifacts"] = [
                reference(path) for path in sorted(scratch.iterdir()) if path.is_file()
            ]
            row["device_asan_symbol_observed"] = True
            row["scope"] = (
                "configured_flags_and_device_symbol_sites_not_every_access_or_runtime_sensitivity"
            )
        rows.append(row)
    if not {"hip_direct_kernels.hip", "wrap64_hip_kernels.hip", "vector_alu_kernels.hip"}.issubset(
        device_sources
    ):
        raise ValueError("actual first-party core HIP object denominator incomplete")
    targets = target_artifacts(
        parse_evidence_json(
            bounded_read(build / ("rns8-analysis-targets-" + inputs.configuration + ".json"))[0]
        ),
        inputs.configuration,
    )
    if any(not path.is_relative_to(build) for path, _ in targets.values()):
        raise ValueError("device-ASan target artifact escaped fresh build")
    artifacts = {name: reference(path) for name, (path, _) in targets.items()}
    if not {"rns8_tests", "rns8_static"}.issubset(artifacts):
        raise ValueError("actual test executable/static library missing")
    needed = session.run(
        "application-dynamic-dependencies",
        [str(args.llvm_readelf), "--dynamic", artifacts["rns8_tests"]["path"]],
    )["stdout"]
    manifest_path = build / "generated" / inputs.configuration / "rns8_build_manifest.json"
    manifest = parse_evidence_json(bounded_read(manifest_path)[0])
    if not isinstance(manifest, dict) or manifest.get(
        "source_content_digest"
    ) != source_content_digest(ROOT):
        raise ValueError("device-ASan compiled source manifest differs from current source")
    result = dict(
        objects=rows,
        artifacts=artifacts,
        manifest=reference(manifest_path),
        configuration_inputs=[
            dict(path=path, sha256=digest) for path, digest in inputs.artifact_hashes
        ],
        generated_inputs=[
            reference(path)
            for path in sorted(
                {path for context in inputs.contexts for path in context.generated_inputs}
            )
        ],
        direct_application_dependencies=dynamic_dependencies(needed),
    )
    check_bookends(result)
    return result


def ordinary_correctness(session: Session, args, report: dict, output: Path) -> None:
    if any(
        os.environ.get(key)
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
    ):
        raise ValueError(
            "ordinary device-ASan requires no conflicting inherited HIP/CUDA ordinal masks"
        )
    report["ctest"] = reference(args.ctest)
    ctest_version = session.run("ctest-version", [str(args.ctest), "--version"])["stdout"]
    if version_number(ctest_version) != version_number(report["cmake_version"]):
        raise ValueError("selected CMake/CTest complete versions differ")
    report["ctest_version"] = ctest_version
    contract = dict(
        target=args.target,
        agent_uuid=report["agent"]["uuid"],
        runtimes=report["runtimes"],
        minimum_free_bytes=args.minimum_free_mib * 1024**2,
    )
    report["correctness_contract"] = contract
    selected_environment = session.environment
    session.environment = dict(selected_environment)
    try:
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            session.environment.pop(key, None)
        session.environment["ROCR_VISIBLE_DEVICES"] = report["agent"]["uuid"]
        report["correctness_environment"] = {
            key: session.environment.get(key)
            for key in (
                "ROCR_VISIBLE_DEVICES",
                "HSA_XNACK",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
            )
        }
        executable = report["compiled_objects"]["artifacts"]["rns8_tests"]["path"]
        report["device_tests"] = "attempted_incomplete"
        ref, result = session.tests(
            "device-correctness",
            output / "build",
            args.ctest,
            required=CORRECTNESS_CASES,
            selection_names=list(CORRECTNESS_CASES),
            runtime_witness=dict(
                executables=[executable],
                lane="device-address",
                runtimes=list(report["runtimes"].values()),
                device_contract=contract,
            ),
        )
        report["correctness_inventory"] = ref
        report["correctness_module_files"] = result["application_runtime_module_files"]
        if result["status"] != "passed":
            report["device_tests"] = "failed"
            raise ValueError("required ordinary device-ASan correctness subset did not pass")
        report["device_tests"] = "passed_bounded_subset"
        report["dynamic_loader_resolution"] = "observed_in_each_selected_test_process"
        report["device_available_memory"] = "native_per_case_entry_admission_and_exit_observations"
        report["qualification"]["device_execution"] = "bounded_ordinary_correctness_passed"
        report["pending_qualification"] = [
            "separately authorized device detector sensitivity if required",
            "external-backend/consumer instrumentation and broader target acceptance",
            "continuous supported-host CI qualification",
        ]
    finally:
        session.environment = selected_environment


def observe(args, plan) -> int:
    output = Path(plan["output"])
    if output.exists():
        raise ValueError("fresh device-ASan preflight output required")
    output.mkdir(parents=True)
    report = dict(
        schema="rns8_device_asan_preflight_v1",
        status="blocked",
        plan=plan,
        qualification=support(platform.system(), args.release, args.target),
        device_tests="not_run",
        faults="not_run_not_authorized",
        dynamic_loader_resolution="not_run",
        promotion_eligible=False,
        production_latency=False,
        blockers=[],
    )
    session = Session(
        output,
        build_environment(dict(os.environ)),
        args.timeout_seconds,
        args.artifact_mib * 1024**2,
    )
    try:
        if report["qualification"][
            "official_support"
        ] != "supported" or platform.machine().lower() not in {"x86_64", "amd64"}:
            report["status"] = "unsupported"
            raise ValueError("native Linux x86-64 reviewed ROCm/target tuple required")
        if any(
            os.environ.get(key)
            for key in (
                "LD_PRELOAD",
                "LD_AUDIT",
                "CCC_OVERRIDE_OPTIONS",
                "CLANG_CONFIG_FILE_USER_DIR",
                "CLANG_CONFIG_FILE_SYSTEM_DIR",
                "CL",
                "_CL_",
            )
        ):
            raise ValueError("unreviewed native process/compiler injection")
        source = source_inventory()
        report["sources"] = source
        configuration, report["kernel_configuration"] = kernel_configuration(platform.release())
        report["host"] = host_prerequisites(platform.release(), configuration, dict(os.environ))
        if not Path("/sys/module/amdgpu").is_dir() or not os.access("/dev/kfd", os.R_OK | os.W_OK):
            raise ValueError("loaded AMDGPU driver and native KFD read/write access required")
        driver = [
            reference(path)
            for path in (Path("/sys/module/amdgpu/version"), Path("/sys/module/amdgpu/srcversion"))
            if path.is_file()
        ]
        report["driver_identity_files"] = driver
        if not driver:
            raise ValueError("native AMDGPU driver identity unavailable")
        meminfo = bounded_read(Path("/proc/meminfo"))[0]
        memory = re.findall(r"^MemAvailable:\s+(\d+) kB$", meminfo, re.M)
        if len(memory) != 1 or int(memory[0]) <= 0:
            raise ValueError("native available host memory observation missing")
        report["available_host_memory_bytes"] = int(memory[0]) * 1024
        version_path = args.rocm_root / ".info/version"
        version_text = bounded_read(version_path)[0].strip()
        if not re.fullmatch(re.escape(args.release) + r"(?:[-+][A-Za-z0-9._-]+)?", version_text):
            raise ValueError("actual SDK version file differs from reviewed ROCm release")
        report["sdk_release"] = reference(version_path)
        report["tools"] = {name: reference(getattr(args, name)) for name in TOOLS}
        versions = {
            name: session.run(name + "-version", [str(getattr(args, name)), "--version"])["stdout"]
            for name in TOOLS
            if name != "rocminfo"
        }
        if len({version_number(text) for name, text in versions.items() if name != "hipcc"}) != 1:
            raise ValueError("selected AMD LLVM compiler/artifact tools differ in complete version")
        include = args.rocm_root / "include"
        if not (include / "hip/hip_runtime_api.h").is_file():
            include = args.rocm_root / "hip/include"
        hip_header = include / "hip/hip_version.h"
        report["hip_component"] = dict(
            version=hip_component_version(versions["hipcc"], header_version(hip_header)),
            version_header=reference(hip_header),
        )
        report["versions"] = versions
        resource = Path(
            session.run("compiler-resource", [str(args.compiler), "--print-resource-dir"])[
                "stdout"
            ].strip()
        ).resolve(strict=True)
        if not resource.is_relative_to(args.rocm_root):
            raise ValueError("compiler resource directory outside selected SDK")
        asan = args.asan_runtime or next(
            iter(sorted((resource / "lib/linux").glob("libclang_rt.asan-x86_64.so"))), None
        )
        runtimes = {
            "hip": runtime(args.rocm_root, "libamdhip64.so*", args.hip_runtime),
            "hsa": runtime(args.rocm_root, "libhsa-runtime64.so*", args.hsa_runtime),
            "asan": runtime(args.rocm_root, "libclang_rt.asan-x86_64.so", asan),
        }
        report["runtimes"] = {name: reference(path) for name, path in runtimes.items()}
        for name, path in runtimes.items():
            symbols = session.run(
                name + "-runtime-symbols",
                [
                    str(args.llvm_nm),
                    "--dynamic",
                    *(["--undefined-only"] if name != "asan" else ["--defined-only"]),
                    str(path),
                ],
            )["stdout"]
            if not re.search(r"\b__asan_(?:init|report_\w+|load\w*|store\w*)\b", symbols):
                raise ValueError(
                    "selected runtime lacks observed native ASan instrumentation/interface: " + name
                )
        runtime_dirs = list(dict.fromkeys(str(path.parent) for path in runtimes.values()))
        report["selected_runtime_search_directories"] = runtime_dirs
        report["device_available_memory"] = "not_observed_until_runtime_session"
        report["pending_qualification"] = [
            "actual application/consumer dynamic-loader resolution",
            "bounded ordinary CPU-differential device correctness",
            "separately authorized detector sensitivity if required",
            "external backend instrumentation and actual runtime memory admission",
        ]
        session.environment.update(
            HIP_PATH=str(args.rocm_root),
            ROCM_PATH=str(args.rocm_root),
            HIP_CLANG_PATH=str(args.compiler.resolve().parent),
            HIP_PLATFORM="amd",
            HSA_XNACK="1",
            LD_LIBRARY_PATH=":".join(runtime_dirs),
            ASAN_OPTIONS="halt_on_error=1:detect_leaks=0",
        )
        native = session.run("rocminfo", [str(args.rocminfo)], timeout=60)
        report["agent"] = agent_observation(native["stdout"], args.target, args.agent_index)
        if args.compile:
            report["cmake"] = reference(args.cmake)
            report["ninja"] = reference(args.ninja)
            report["cmake_version"] = session.run("cmake-version", [str(args.cmake), "--version"])[
                "stdout"
            ]
            report["ninja_version"] = session.run("ninja-version", [str(args.ninja), "--version"])[
                "stdout"
            ]
            configure = [
                *plan["configure"],
                "-DRNS8_HIP_LIBRARY=" + str(runtimes["hip"]),
                "-DRNS8_DEVICE_ASAN_HSA_RUNTIME=" + str(runtimes["hsa"]),
            ]
            report["configured_runtime_command"] = configure
            session.run("configure", configure)
            built = session.run("build", plan["build_command"])
            if re.search(
                r"(?:warning|error):[^\n]*(?:(?:ignor|unsupported|unused)[^\n]*(?:sanitiz|asan)|(?:sanitiz|asan)[^\n]*(?:ignor|unsupported|unused))",
                built["stdout"] + built["stderr"],
                re.I,
            ):
                raise ValueError("compiler ignored required device instrumentation")
            report["compiled_objects"] = inspect_build(session, args, output / "build", runtimes)
        if args.correctness:
            ordinary_correctness(session, args, report, output)
        check_bookends([report, source, session.stages])
        report["status"] = (
            "bounded_correctness_complete_sensitivity_not_run"
            if args.correctness
            else "artifact_preflight_ready"
            if args.compile
            else "prerequisites_observed"
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        report["blockers"].append(str(error))
    report["stages"] = session.stages
    report["environment_sha256"] = canonical_digest(session.environment)
    report["identity_sha256"] = canonical_digest(report)
    write_json(output / "report.json", report)
    print(
        json.dumps(
            dict(
                status=report["status"],
                report=str(output / "report.json"),
                blockers=report["blockers"],
            )
        )
    )
    return (
        0
        if report["status"]
        in {
            "artifact_preflight_ready",
            "prerequisites_observed",
            "bounded_correctness_complete_sensitivity_not_run",
        }
        else 1
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm-root", type=Path, required=True)
    parser.add_argument("--release", default=RELEASE)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--agent-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    for name in (*TOOLS, "cmake", "ctest", "ninja", "hip_runtime", "hsa_runtime", "asan_runtime"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--artifact-mib", type=int, default=32768)
    parser.add_argument("--observe", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="With --observe, build/inspect fresh real first-party HIP objects",
    )
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="With --compile and --observe, run only the fixed ordinary exact/canary subset; no device faults",
    )
    parser.add_argument("--minimum-free-mib", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        output = args.out.resolve()
        args.rocm_root = args.rocm_root.resolve(strict=True)
        if (
            not args.rocm_root.is_dir()
            or output == ROOT / "temp"
            or not output.is_relative_to(ROOT / "temp")
            or output.exists()
        ):
            raise ValueError("existing SDK prefix and fresh ignored repository temp child required")
        if (
            not 0 <= args.agent_index <= 1023
            or not 1 <= args.jobs <= 64
            or not 1 <= args.timeout_seconds <= 14400
            or not 256 <= args.artifact_mib <= 65536
            or not 64 <= args.minimum_free_mib <= 1048576
        ):
            raise ValueError("device-ASan preflight budgets/agent index invalid")
        if args.correctness and not args.compile:
            raise ValueError("ordinary correctness requires the fresh inspected device-ASan build")
        for name, names in TOOLS.items():
            setattr(args, name, tool(args.rocm_root, getattr(args, name), names))
        if args.compile and args.ninja is None:
            native = shutil.which("ninja")
            if not native:
                raise ValueError("native Ninja unavailable; supply --ninja")
            args.ninja = Path(native)
        if args.correctness and args.ctest is None and args.cmake is not None:
            args.ctest = args.cmake.with_name("ctest")
        for name in ("hip_runtime", "hsa_runtime", "asan_runtime", "cmake", "ctest", "ninja"):
            if getattr(args, name) is not None:
                setattr(args, name, invocation_path(getattr(args, name)))
        if args.compile and args.cmake is None:
            raise ValueError("fresh device-ASan build requires explicit native CMake")
        if args.correctness and args.ctest is None:
            raise ValueError("bounded ordinary correctness requires selected CTest")
        configure = [
            str(args.cmake),
            "-S",
            str(ROOT),
            "-B",
            str(output / "build"),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DCMAKE_CXX_COMPILER=" + str(args.compiler),
            "-DCMAKE_MAKE_PROGRAM=" + str(args.ninja),
            "-DRNS8_HIP_ROOT=" + str(args.rocm_root),
            "-DRNS8_AMDGPU_TARGETS=" + args.target,
            "-DRNS8_ENABLE_HIP=ON",
            "-DRNS8_ENABLE_DEVICE_ASAN=ON",
            "-DRNS8_DEVICE_ASAN_ROCM_VERSION=" + args.release,
            "-DRNS8_HOST_SANITIZER=none",
            "-DRNS8_ENABLE_HOST_COVERAGE=OFF",
            "-DRNS8_BUILD_SANITIZER_CONTROLS=OFF",
            "-DRNS8_BUILD_FUZZERS=OFF",
            "-DRNS8_BUILD_SHARED=OFF",
            "-DRNS8_ENABLE_CPU_PARALLEL=OFF",
            "-DRNS8_BUILD_TESTS=ON",
            "-DRNS8_BUILD_TOOLS=OFF",
            "-DRNS8_BUILD_BENCHMARKS=OFF",
            "-DRNS8_BUILD_EXAMPLES=OFF",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=",
            "-DCMAKE_C_COMPILER_LAUNCHER=",
            "-DPython3_EXECUTABLE=" + sys.executable,
        ]
        configure.extend(
            "-DRNS8_ENABLE_" + flag + "=OFF"
            for flag in (
                "HIPBLASLT",
                "CK",
                "ROCWMMA",
                "AMDGPU_BUILTINS",
                "CPU_AMX",
                "CPU_I8MM",
                "DISTRIBUTED_MPI",
                "RCCL",
                "ROCSHMEM",
                "ROCTX",
                "GMP",
                "FLINT",
            )
        )
        plan = dict(
            schema="rns8_device_asan_preflight_plan_v1",
            output=str(output),
            qualification=support("Linux", args.release, args.target),
            configure=configure if args.compile else None,
            build_command=[
                str(args.cmake),
                "--build",
                str(output / "build"),
                "--parallel",
                str(args.jobs),
                "--verbose",
            ]
            if args.compile
            else None,
            prerequisites=[
                "actual native kernel HMM configuration",
                "existing HSA_XNACK=1",
                "loaded AMDGPU/KFD permission",
                "observed agent UUID/xnack+ ISA",
                "exact selected SDK/tools and instrumented HIP/HSA/shared ASan runtimes",
            ],
            device_tests="not_run",
            device_faults="not_run_not_authorized",
            runtime_resolution="not_run",
            correctness=dict(
                cases=list(CORRECTNESS_CASES),
                ctest=str(args.ctest),
                minimum_free_bytes=args.minimum_free_mib * 1024**2,
                launch_requested_only_with_observe=True,
            )
            if args.correctness
            else None,
            production_latency=False,
        )
        if not args.observe:
            print(json.dumps(plan, indent=2))
            return 0
        return observe(args, plan)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("device ASan preflight: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
