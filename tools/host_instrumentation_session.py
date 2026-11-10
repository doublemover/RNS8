"""Plan or explicitly execute a bounded, fresh native host sanitizer session."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import secrets
import sys
import math
import time

from analysis_inputs import cache_values, load_inputs, target_artifacts
from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest, validate_document
from compiler_context import compiler_environment, invocation_path
from execution_inventory import bounded_read, discovery, reconcile
from host_instrumentation_contract import (
    detector_result,
    detector_sites,
    native_argv,
    object_path,
    selected_flags,
    symbol_classes,
    valid_result,
)
from json_evidence import parse_evidence_json
from host_runtime_contract import application_witnesses, control_document, loaded_inventory
from process_progress import capture_succeeded, run_capture
from quality_inventory import ROOT, inventory

PRESETS = {
    "cpu-asan-ubsan-debug": ("Linux", "address-undefined"),
    "cpu-asan-ubsan-release": ("Linux", "address-undefined"),
    "cpu-tsan-release": ("Linux", "thread"),
    "windows-clang-asan-debug": ("Windows", "windows-address"),
}
HOST_APPLICATIONS = ("rns8_tests", "rns8_exact_cpu_allocation_faults")
MANDATORY = (
    "independent CPU owners overlap with exact results and unique matrix identities",
    "shared CPU handles obey their external serialization contract across host threads",
)
CONSUMERS = (
    "rns8-downstream-smoke",
    "rns8-downstream-continuation",
    "rns8-downstream-sparse-finite-ring",
    "rns8-downstream-sparse-bounded-i64",
    "rns8-downstream-stream-bounded-i64",
    "rns8-downstream-materialized-product-recipe",
    "rns8-downstream-checked-crt",
)
CONSUMER_SOURCES = dict(
    zip(
        CONSUMERS,
        (
            "main.cpp",
            "exact_wide_cpu_continuation.cpp",
            "sparse_finite_ring.cpp",
            "sparse_bounded_i64.cpp",
            "stream_bounded_i64.cpp",
            "materialized_product_recipe.cpp",
            "checked_crt_export.cpp",
        ),
    )
)
DEVICE_CONSUMER_CASES = (
    *CONSUMERS,
    "rns8-downstream-checked-crt-hip",
    "rns8-downstream-stream-bounded-i64-hip",
)


def write_json(path: Path, value: object):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return dict(path=str(path), sha256=file_digest(path))


def reference(path: Path) -> dict:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("regular artifact required: " + str(path))
    return dict(path=str(path), sha256=file_digest(path), bytes=path.stat().st_size)


def child_environment(environment: dict[str, str], symbolizer: Path, lane: str) -> dict[str, str]:
    if any(
        environment.get(key)
        for key in (
            "LD_PRELOAD",
            "LD_AUDIT",
            "DYLD_INSERT_LIBRARIES",
            "CCC_OVERRIDE_OPTIONS",
            "CLANG_CONFIG_FILE_USER_DIR",
            "CLANG_CONFIG_FILE_SYSTEM_DIR",
            "CL",
            "_CL_",
        )
    ):
        raise ValueError("unreviewed process/compiler injection in host lane")
    result = {
        key: value
        for key, value in environment.items()
        if not key.startswith(
            (
                "ASAN_",
                "UBSAN_",
                "TSAN_",
                "LSAN_",
                "LLVM_PROFILE_",
                "COVERAGE_",
                "OMP_",
                "KMP_",
                "GOMP_",
                "ARCHER_",
            )
        )
        and key
        not in {"CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "RNS8_HOST_RUNTIME_WITNESS_TOKEN"}
    }
    result.update(
        PYTHONOPTIMIZE="",
        CTEST_NO_TESTS_ACTION="error",
        ASAN_SYMBOLIZER_PATH=str(symbolizer),
        ASAN_OPTIONS="halt_on_error=1:abort_on_error=0:exitcode=86:symbolize=1:detect_leaks="
        + ("0" if lane == "windows-address" else "1"),
        UBSAN_OPTIONS="halt_on_error=1:exitcode=87:print_stacktrace=1:symbolize=1",
        TSAN_OPTIONS="halt_on_error=1:exitcode=88:external_symbolizer_path=" + str(symbolizer),
    )
    return result


def version_number(source: str) -> str:
    match = re.search(r"\bversion\s+(\d+\.\d+\.\d+)", source)
    if not match:
        raise ValueError("native LLVM tool semantic version missing")
    return match[1]


def runtime_inventory(session, compiler: Path, lane: str) -> dict:
    result = session.run("compiler-resource-directory", [str(compiler), "--print-resource-dir"])
    resource = Path(result["stdout"].strip()).resolve(strict=True)
    if not resource.is_dir():
        raise ValueError("compiler resource directory missing")
    required = {
        "address-undefined": ("asan", "ubsan"),
        "thread": ("tsan",),
        "windows-address": ("asan",),
    }[lane]
    paths = []
    for path in sorted((resource / "lib").rglob("*")):
        if (
            path.is_file()
            and path.suffix in {".a", ".so", ".lib", ".dll", ".dylib"}
            and any("clang_rt." + name in path.name for name in required)
        ):
            paths.append(path)
    if (
        not paths
        or len(paths) > 128
        or any(not any("clang_rt." + name in path.name for path in paths) for name in required)
    ):
        raise ValueError("selected compiler detector runtime inventory incomplete")
    files = [reference(path) for path in paths]
    if sum(row["bytes"] for row in files) > 2 * 1024**3:
        raise ValueError("selected compiler detector runtime inventory exceeds budget")
    return dict(
        resource_directory=str(resource),
        files=files,
        scope="available_matching_compiler_runtimes_not_complete_dynamic_loader_inventory",
    )


def valid_total_budget(seconds):
    return seconds is None or (
        type(seconds) in (int, float) and 0 < seconds <= 86400 and math.isfinite(seconds)
    )


def owned_output_path(path: Path, root: Path) -> Path:
    """Admit a fresh scratch child without creating directories during planning."""
    if not root.is_absolute():
        raise ValueError("artifact root must be an explicit absolute path")
    root, output = root.resolve(), path.resolve()
    if (
        root.parent == root
        or (root.is_relative_to(ROOT) and not root.is_relative_to(ROOT / "temp"))
        or (root.exists() and not root.is_dir())
        or output == root
        or not output.is_relative_to(root)
        or output.exists()
    ):
        raise ValueError("fresh scratch child under the declared artifact root required")
    return output


class Session:
    def __init__(
        self,
        output: Path,
        environment: dict,
        timeout: int,
        artifact_bytes: int,
        *,
        total_seconds=None,
    ):
        if not valid_total_budget(total_seconds):
            raise ValueError("host session needs a finite total budget within24hours")
        self.output, self.environment, self.timeout, self.artifact_bytes = (
            output,
            environment,
            timeout,
            artifact_bytes,
        )
        self.stages = []
        self.deadline = time.monotonic() + total_seconds if total_seconds is not None else None

    def run(
        self,
        name: str,
        command: list[str],
        *,
        cwd=ROOT,
        required=True,
        timeout=None,
        max_output_bytes=32 * 1024**2,
        environment=None,
    ) -> dict:
        child_timeout = timeout or self.timeout
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("host session total time budget exhausted before stage: " + name)
            child_timeout = min(child_timeout, remaining)
        result = run_capture(
            command,
            cwd=cwd,
            label="host-check:" + name,
            env=self.environment if environment is None else environment,
            timeout_seconds=child_timeout,
            max_output_bytes=max_output_bytes,
            artifact_roots=(self.output,),
            max_artifact_bytes=self.artifact_bytes,
            return_on_interrupt=True,
        )
        try:
            receipt = write_json(
                self.output / (str(len(self.stages)).zfill(4) + "-" + name + ".process.json"),
                result,
            )
        except OSError as error:
            if result.get("interrupted"):
                raise InterruptedError(
                    "host session interrupted; capture persistence also failed"
                ) from error
            raise
        self.stages.append(
            dict(
                name=name,
                receipt=receipt,
                process_id=result.get("process_id"),
                returncode=result.get("returncode"),
                succeeded=capture_succeeded(result),
                interrupted=result.get("interrupted"),
            )
        )
        if result.get("interrupted"):
            raise InterruptedError("host session interrupted; no automatic retry")
        if required and not capture_succeeded(result):
            raise ValueError("required stage failed: " + name)
        return result

    def json(self, name, command, *, cwd=ROOT):
        result = self.run(name, command, cwd=cwd)
        value = parse_evidence_json(result["stdout"])
        return result, value

    def tests(
        self,
        name,
        build,
        ctest,
        *,
        required,
        allowed=None,
        selection_names=None,
        selected_inventory_only=False,
        runtime_witness=None,
    ):
        command = [str(ctest), "--test-dir", str(build), "--no-tests=error"]
        selection_ref = None
        if selected_inventory_only and selection_names is None:
            raise ValueError("selected inventory needs explicit test names")
        if selection_names is not None:
            if (
                not isinstance(selection_names, list)
                or not selection_names
                or any(
                    type(item) is not str or not item or "\n" in item or "\r" in item
                    for item in selection_names
                )
                or len(set(selection_names)) != len(selection_names)
            ):
                raise ValueError("explicit nonempty test selection required")
            path = self.output / (name + "-selection.txt")
            with path.open("x", encoding="utf-8") as handle:
                handle.write("\n".join(selection_names) + "\n")
            selection_ref = reference(path)
            if selected_inventory_only:
                command.extend(["--tests-from-file", str(path)])
        _, full = self.json(name + "-discovery", [*command, "--show-only=json-v1"])
        if selection_names is not None:
            if not set(selection_names).issubset(discovery(full)):
                raise ValueError("explicit nonempty discovered test selection required")
            if not selected_inventory_only:
                command.extend(["--tests-from-file", str(path)])
        _, selected = self.json(name + "-selection", [*command, "--show-only=json-v1"])
        discovery(full)
        discovery(selected)
        if selection_names is not None and set(discovery(selected)) != set(selection_names):
            raise ValueError(
                "native discovery selection differs from requested names; refusing execution"
            )
        full_ref = write_json(self.output / (name + "-discovered.json"), full)
        selected_ref = write_json(self.output / (name + "-selected.json"), selected)
        executable_refs = [
            reference(Path(path))
            for path in sorted({test.command[0] for test in discovery(selected).values()})
        ]
        junit_path = self.output / (name + ".xml")
        environment = None
        token = None
        if runtime_witness is not None:
            token = secrets.token_hex(32)
            environment = {**self.environment, "RNS8_HOST_RUNTIME_WITNESS_TOKEN": token}
            command.extend(
                ["--test-output-size-passed", "1048576", "--test-output-size-failed", "1048576"]
            )
        result = self.run(
            name + "-execution",
            [
                *command,
                "--parallel",
                "1",
                "--timeout",
                "180",
                "--output-on-failure",
                "--output-junit",
                str(junit_path),
            ],
            required=False,
            environment=environment,
        )
        source, junit_ref = bounded_read(junit_path)
        value = reconcile(full, selected, source, required=required, allowed_skips=allowed)
        value["inventory_scope"] = (
            "selected_registered_tests_only" if selected_inventory_only else "all_registered_tests"
        )
        value["inputs"] = dict(discovered=full_ref, selected=selected_ref, junit=junit_ref)
        if selection_ref:
            check_bookends(selection_ref)
            value["inputs"]["selection_names"] = selection_ref
        check_bookends(executable_refs)
        value["executables"] = executable_refs
        value["ctest_process_succeeded"] = capture_succeeded(result)
        value["process"] = {
            key: result.get(key)
            for key in (
                "process_id",
                "returncode",
                "timed_out",
                "interrupted",
                "limit_exceeded",
                "capture_error",
            )
        }
        if not capture_succeeded(result):
            value["status"] = "failed"
            value["blockers"].append("ctest_process_did_not_complete_successfully")
        if runtime_witness is not None:
            bindings = application_witnesses(
                source, selected, result, token=token, **runtime_witness
            )
            bindings["inputs"] = dict(selected=selected_ref, junit=junit_ref)
            check_bookends(bindings)
            value["application_runtime_bindings"] = write_json(
                self.output / (name + "-runtime-bindings.json"), bindings
            )
            value["application_runtime_module_files"] = bindings["files"]
        value["identity_sha256"] = canonical_digest(value)
        return write_json(self.output / (name + "-inventory.json"), value), value


def configured_inventory(session: Session, build: Path, lane: str, nm: Path) -> dict:
    inputs = load_inputs(build)
    windows = lane == "windows-address"
    objects = []
    paths = set()
    for context in inputs.contexts:
        if context.kind != "host":
            raise ValueError("host instrumentation cannot include HIP device contexts")
        record = context.native_record()
        argv = native_argv(record, windows=windows)
        if Path(argv[0]).resolve() != inputs.compiler:
            raise ValueError(
                "instrumentation inventory requires direct configured compiler invocation"
            )
        flags = selected_flags(argv, lane)
        path = object_path(record, argv, build, windows=windows)
        if path in paths:
            raise ValueError("different compilation contexts alias one object")
        paths.add(path)
        before = reference(path)
        symbols = session.run(
            "object-symbols-" + str(len(objects)), [str(nm), "--format=posix", str(path)]
        )
        if reference(path) != before:
            raise ValueError("object changed during native symbol inspection")
        objects.append(
            dict(
                context_sha256=context.identity,
                source=reference(context.source),
                object=before,
                registered_targets=list(context.registered_targets),
                flags=flags,
                observed_symbol_classes=symbol_classes(symbols["stdout"], lane),
                scope="configured_flags_and_object_symbol_sites_not_complete_access_coverage",
            )
        )
    if not objects:
        raise ValueError("empty instrumented object inventory")
    target_path = build / ("rns8-analysis-targets-" + inputs.configuration + ".json")
    targets = target_artifacts(
        parse_evidence_json(bounded_read(target_path)[0]), inputs.configuration
    )
    if any(not path.is_relative_to(build.resolve()) for path, _ in targets.values()):
        raise ValueError("instrumented target artifact escaped its owned build")
    artifacts = {name: {**reference(path), "type": kind} for name, (path, kind) in targets.items()}
    required = ["rns8_static", *HOST_APPLICATIONS]
    if lane != "coverage":
        required.append("rns8-host-sanitizer-control")
    for name in required:
        if name not in artifacts:
            raise ValueError("required instrumented target missing: " + name)
    # PE executables need not retain a COFF symbol table. Inspect the actual
    # static archive there; executable identity and runtime controls are separate.
    for name in ("rns8_static",) if windows else ("rns8_static", *HOST_APPLICATIONS):
        result = session.run(
            "linked-symbols-" + name, [str(nm), "--format=posix", artifacts[name]["path"]]
        )
        classes = set(symbol_classes(result["stdout"], lane))
        artifacts[name]["observed_symbol_classes"] = sorted(classes)
        expected = {
            "address-undefined": {"address", "undefined"},
            "thread": {"thread"},
            "windows-address": {"address"},
            "coverage": {"profile"},
        }[lane]
        if not expected.issubset(classes):
            raise ValueError("actual target objects lack selected detector symbol classes: " + name)
    return dict(
        schema="rns8_host_instrumented_objects_v1",
        configuration=inputs.configuration,
        compiler=reference(inputs.compiler),
        configured_inputs=[
            dict(path=path, sha256=digest) for path, digest in inputs.artifact_hashes
        ],
        generated_inputs=[
            reference(path)
            for path in sorted(
                {path for context in inputs.contexts for path in context.generated_inputs}
            )
        ],
        objects=objects,
        artifacts=artifacts,
        excluded_records=list(inputs.excluded_records),
        scope="first_party_host_not_external_libraries_or_device_objects",
        production_latency=False,
    )


def check_bookends(value: object):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            if file_digest(Path(value["path"])) != value["sha256"]:
                raise ValueError("session artifact/source changed: " + value["path"])
        for child in value.values():
            check_bookends(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            check_bookends(child)


def source_inventory() -> list[dict]:
    source = [reference(ROOT / row.path) for row in inventory()]
    source.extend(
        reference(ROOT / path)
        for path in (
            "CMakeLists.txt",
            "CMakePresets.json",
            "metadata/first-party-inventory.json",
        )
    )
    source.extend(
        reference(path) for path in sorted((ROOT / "cmake").rglob("*.cmake*")) if path.is_file()
    )
    return source


def installed_consumers(
    session: Session,
    args,
    report: dict,
    *,
    build: Path,
    output: Path,
    compiler: Path,
    static_library: dict,
    runtime_witness: dict,
) -> dict:
    lane = runtime_witness["lane"]
    windows = lane == "windows-address"
    windows_asan = runtime_witness.get("windows_asan")
    device_contract = runtime_witness.get("device_contract")
    if lane not in {"windows-address", "address-undefined", "thread", "device-address"} or (
        lane == "device-address"
    ) != (device_contract is not None):
        raise ValueError("explicit supported installed-consumer runtime contract required")
    cache = cache_values(build / "CMakeCache.txt")
    session.run(
        "install",
        [str(args.cmake), "--install", str(build), "--prefix", str(output / "prefix")],
    )
    installed = list((output / "prefix").rglob(Path(static_library["path"]).name))
    if len(installed) != 1 or file_digest(installed[0]) != static_library["sha256"]:
        raise ValueError("installed library is not the exact instrumented static artifact")
    report["installed_library"] = reference(installed[0])
    packages = list((output / "prefix").rglob("RNS8Config.cmake"))
    if len(packages) != 1:
        raise ValueError("unique actual installed RNS8 package configuration required")
    report["installed_package_config"] = reference(packages[0])
    package_files = sorted(packages[0].parent.glob("*.cmake"))
    if not 1 <= len(package_files) <= 64:
        raise ValueError("installed package metadata inventory exceeds bounded scope")
    report["installed_package_files"] = [reference(path) for path in package_files]
    examples = list((output / "prefix").rglob("downstream-cmake/CMakeLists.txt"))
    if len(examples) != 1:
        raise ValueError("unique installed downstream example required")
    consumer_source = output / "consumer-source"
    shutil.copytree(examples[0].parent, consumer_source)
    copied_sources = sorted(path for path in consumer_source.rglob("*") if path.is_file())
    if (
        not 1 <= len(copied_sources) <= 64
        or sum(path.stat().st_size for path in copied_sources) > 16 * 1024**2
    ):
        raise ValueError("installed consumer source inventory exceeds bounded package scope")
    report["consumer_sources"] = [reference(path) for path in copied_sources]
    consumer = output / "consumer-build"
    command = [
        str(args.cmake),
        "-S",
        str(consumer_source),
        "-B",
        str(consumer),
        "-G",
        "Ninja",
        "-DCMAKE_PREFIX_PATH=" + str(output / "prefix"),
        "-DRNS8_DIR=" + str(packages[0].parent),
        "-DCMAKE_CXX_COMPILER=" + str(compiler),
        "-DCMAKE_BUILD_TYPE=" + cache["CMAKE_BUILD_TYPE"],
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
        "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
        "-DRNS8_DOWNSTREAM_RUNTIME_WITNESS_ROOT=" + str(ROOT),
        "-Dnlohmann_json_DIR=" + cache["nlohmann_json_DIR"],
    ]
    if lane == "windows-address":
        command.extend(
            [
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
                "-DCMAKE_CXX_FLAGS=/Zi /O1 /MT",
                "-DCMAKE_PROJECT_RNS8DownstreamSmoke_INCLUDE="
                + str(build / "windows-clang-asan-consumer.cmake"),
            ]
        )
    if device_contract is not None:
        command.extend(
            [
                "-DRNS8_DOWNSTREAM_GPU_TESTS=ON",
                "-DRNS8_DOWNSTREAM_DEVICE_ASAN_HSA_RUNTIME="
                + device_contract["runtimes"]["hsa"]["path"],
                "-DRNS8_HIP_ROOT=" + cache["RNS8_HIP_ROOT"],
            ]
        )
    session.run("consumer-configure", command)
    session.run(
        "consumer-build",
        [str(args.cmake), "--build", str(consumer), "--parallel", str(args.jobs), "--verbose"],
    )
    if lane == "windows-address":
        copied = consumer / "clang_rt.asan_dynamic-x86_64.dll"
        if copied.exists():
            if file_digest(copied) != windows_asan["sha256"]:
                raise ValueError("consumer ASan DLL differs from selected runtime")
        else:
            shutil.copyfile(windows_asan["path"], copied)
        report["consumer_asan_dll"] = reference(copied)
    compile_source, report["consumer_compile_commands"] = bounded_read(
        consumer / "compile_commands.json"
    )
    consumer_records = parse_evidence_json(compile_source)
    if not isinstance(consumer_records, list) or len(consumer_records) != len(CONSUMERS):
        raise ValueError("installed consumer compilation inventory incomplete")
    consumer_objects = []
    expected_sources = {
        (consumer_source / name).resolve(strict=True) for name in CONSUMER_SOURCES.values()
    }
    seen_sources, seen_objects = set(), set()
    for row in consumer_records:
        argv = native_argv(row, windows=windows)
        source = Path(row["file"]).resolve(strict=True)
        if (
            source not in expected_sources
            or source in seen_sources
            or Path(argv[0]).resolve(strict=True) != compiler.resolve(strict=True)
        ):
            raise ValueError(
                "installed consumer source/compiler inventory differs from selected package"
            )
        seen_sources.add(source)
        selected_flags(argv, lane)
        artifact = object_path(row, argv, consumer, windows=windows)
        if artifact in seen_objects:
            raise ValueError("consumer compile contexts alias one object")
        seen_objects.add(artifact)
        consumer_objects.append(reference(artifact))
    report["consumer_objects"] = consumer_objects
    consumer_ref, consumer_result = session.tests(
        "consumer",
        consumer,
        args.ctest,
        required=DEVICE_CONSUMER_CASES if device_contract is not None else CONSUMERS,
        selection_names=list(DEVICE_CONSUMER_CASES if device_contract is not None else CONSUMERS),
        runtime_witness=dict(
            executables=[consumer / (name + (".exe" if windows else "")) for name in CONSUMERS],
            **{
                key: value
                for key, value in runtime_witness.items()
                if key not in {"executables", "windows_asan"}
            },
            windows_asan=Path(report["consumer_asan_dll"]["path"]) if windows else None,
        ),
    )
    report["consumer_inventory"] = consumer_ref
    check_bookends([report, consumer_result])
    return consumer_result


def execute(args, plan):
    output = Path(plan["output"])
    if output.exists():
        raise ValueError("host session output must not exist")
    environment = child_environment(dict(os.environ), args.symbolizer, plan["lane"])
    if args.openmp_archer:
        environment.update(
            OMP_TOOL="enabled",
            OMP_TOOL_LIBRARIES=str(args.openmp_archer),
            OMP_NUM_THREADS="2",
            OMP_DYNAMIC="FALSE",
            OMP_THREAD_LIMIT="2",
            ARCHER_OPTIONS="verbose=1 enable=1 ignore_serial=0",
        )
    output.mkdir(parents=True)
    report = dict(
        schema="rns8_host_dynamic_check_session_v1",
        status="blocked",
        plan=plan,
        scope="native_host_only_not_hip_device_or_performance",
        production_latency=False,
        device_asan="unsupported" if plan["system"] == "Windows" else "not_run",
        openmp_race_qualification="not_run",
        blockers=[],
    )
    session = Session(
        output,
        environment,
        args.timeout_seconds,
        args.artifact_mib * 1024**2,
        total_seconds=getattr(args, "total_seconds", None),
    )
    try:
        source = source_inventory()
        report["source_inventory"] = source
        report["tools"] = {
            name: reference(getattr(args, name)) for name in ("cmake", "ctest", "nm", "symbolizer")
        }
        if args.toolchain:
            report["toolchain"] = reference(args.toolchain)
        revision = session.run("revision", ["git", "rev-parse", "HEAD"])["stdout"].strip()
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None:
            raise ValueError("invalid native source revision")
        report["source_revision"] = revision
        versions = {}
        for name in ("cmake", "ctest", "nm", "symbolizer"):
            versions[name] = session.run(
                name + "-version", [str(getattr(args, name)), "--version"]
            )["stdout"]
        if versions["cmake"].splitlines()[0].replace("cmake ", "") != versions[
            "ctest"
        ].splitlines()[0].replace("ctest ", ""):
            raise ValueError("CMake and CTest must be matching tools")
        session.run("configure", plan["configure"])
        build = Path(plan["build"])
        cache = cache_values(build / "CMakeCache.txt")
        false_values = {"OFF", "FALSE", "NO", "0"}
        true_values = {"ON", "TRUE", "YES", "1"}
        if cache.get("RNS8_ENABLE_HIP", "").upper() not in false_values or cache.get(
            "RNS8_ENABLE_CPU_PARALLEL", ""
        ).upper() not in (true_values if args.openmp_archer else false_values):
            raise ValueError("host lane CPU/OpenMP configuration differs from plan")
        compiler = invocation_path(Path(cache["CMAKE_CXX_COMPILER"]))
        report["compiler"] = reference(compiler)
        versions["compiler"] = session.run("compiler-version", [str(compiler), "--version"])[
            "stdout"
        ]
        if not (
            version_number(versions["compiler"])
            == version_number(versions["nm"])
            == version_number(versions["symbolizer"])
        ):
            raise ValueError("compiler, llvm-nm and llvm-symbolizer must match exactly")
        report["tool_versions"] = versions
        report["runtime_inventory"] = runtime_inventory(session, compiler, plan["lane"])
        result = session.run("build", plan["build_command"])
        if re.search(
            r"(?:warning|error):[^\n]*(?:unused|ignored|unsupported)[^\n]*(?:sanitize|coverage|profile)",
            result["stdout"] + result["stderr"],
            re.I,
        ):
            raise ValueError("compiler ignored required instrumentation")
        objects = configured_inventory(session, build, plan["lane"], args.nm)
        report["object_inventory"] = write_json(output / "objects.json", objects)
        binaries = objects["artifacts"]
        if plan["lane"] == "windows-address":
            runtime = (
                Path(report["runtime_inventory"]["resource_directory"])
                / "lib/windows/clang_rt.asan_dynamic-x86_64.dll"
            )
            copied = build / runtime.name
            if file_digest(runtime) != file_digest(copied):
                raise ValueError("build ASan DLL differs from selected compiler runtime")
            report["windows_asan_dll"] = reference(copied)
        identities = {}
        for name in ("rns8_tests", "rns8-host-sanitizer-control"):
            capture, value = session.json(
                name + "-identity", [binaries[name]["path"], "--build-identity"]
            )
            validate_document(value)
            if type(value.get("process_id")) is not int or value["process_id"] != capture.get(
                "process_id"
            ):
                raise ValueError("compiled identity process mismatch")
            identities[name] = value
        if (
            identities["rns8_tests"]["build_manifest"]
            != identities["rns8-host-sanitizer-control"]["build_manifest"]
        ):
            raise ValueError("application and detector build manifests differ")
        report["compiled_identities"] = identities
        control = binaries["rns8-host-sanitizer-control"]["path"]
        _, capability = session.json("detector-capabilities", [control, "--capabilities"])
        if (
            not isinstance(capability, dict)
            or capability.get("schema") != "rns8_host_detector_capabilities_v1"
            or capability.get("lane") != plan["lane"]
            or capability.get("scope") != "host_only_not_hip_device"
            or capability.get("openmp_compiled") is not bool(args.openmp_archer)
            or capability.get("address_feature") is not (plan["lane"] != "thread")
            or capability.get("thread_feature") is not (plan["lane"] == "thread")
        ):
            raise ValueError("compiled detector features differ from selected host lane")
        report["capability"] = capability
        openmp = None
        if args.openmp_archer:
            openmp = Path(cache["OpenMP_omp_LIBRARY"]).resolve(strict=True)
            report["openmp_runtime"] = reference(openmp)
            report["archer_runtime"] = reference(args.openmp_archer)
            symbols = session.run(
                "archer-exported-symbols",
                [str(args.nm), "--dynamic", "--defined-only", str(args.openmp_archer)],
            )["stdout"]
            if not re.search(r"\bompt_start_tool\b", symbols):
                raise ValueError("selected Archer library lacks native OMPT entrypoint")
        loader_capture = session.run("loaded-runtimes", [control, "--runtime-inventory"])
        report["loaded_runtimes"] = loaded_inventory(
            control_document(loader_capture, archer=bool(args.openmp_archer)),
            loader_capture,
            executable=Path(control),
            runtimes=report["runtime_inventory"]["files"],
            windows_asan=Path(report["windows_asan_dll"]["path"])
            if plan["system"] == "Windows"
            else None,
            openmp=openmp,
            archer=args.openmp_archer,
        )
        valid_capture = session.run("valid-control", [control, "valid"], required=False, timeout=60)
        try:
            valid_doc = parse_evidence_json(valid_capture["stdout"])
        except ValueError:
            valid_doc = None
        controls = [valid_result(valid_capture, valid_doc)]
        sites = detector_sites(ROOT / "tests/controls/host_sanitizer_control.cpp")
        report["detector_source_sites"] = sites
        modes = {
            "thread": ["race"],
            "address-undefined": ["heap-overread", "signed-overflow"],
            "windows-address": ["heap-overread"],
        }[plan["lane"]]
        for mode in modes:
            controls.append(
                detector_result(
                    mode,
                    session.run(mode, [control, mode], required=False, timeout=60),
                    source_line=sites[mode],
                )
            )
        report["detectors"] = controls
        if args.openmp_archer:
            valid = session.run(
                "openmp-valid", [control, "openmp-valid"], required=False, timeout=60
            )
            try:
                document = control_document(valid, archer=True)
            except ValueError:
                document = None
            openmp_controls = [
                valid_result(valid, document),
                detector_result(
                    "openmp-race",
                    session.run(
                        "openmp-race", [control, "openmp-race"], required=False, timeout=60
                    ),
                    source_line=sites["openmp-race"],
                ),
            ]
            controls.extend(openmp_controls)
            report["openmp_race_qualification"] = dict(
                status="passed"
                if all(row["status"] == "passed" for row in openmp_controls)
                else "failed",
                controls=openmp_controls,
                scope="selected_libomp_archer_controls_and_registered_cpu_tests_not_all_schedules",
                external_runtime_instrumentation="OMPT_annotations_not_full_access_instrumentation",
                runtime_suppressions=[],
            )
        if any(row["status"] != "passed" for row in controls):
            raise ValueError("host_detector_sensitivity_failed_before_application_tests")
        allowed = {}
        if args.allowed_skips:
            raw, ref = bounded_read(args.allowed_skips)
            allowed = parse_evidence_json(raw)
            report["skip_policy"] = ref
        app_ref, app = session.tests(
            "application",
            build,
            args.ctest,
            required=MANDATORY,
            allowed=allowed,
            runtime_witness=dict(
                executables=[binaries[name]["path"] for name in HOST_APPLICATIONS],
                lane=plan["lane"],
                runtimes=report["runtime_inventory"]["files"],
                windows_asan=Path(report["windows_asan_dll"]["path"])
                if plan["lane"] == "windows-address"
                else None,
                openmp=openmp,
                archer=args.openmp_archer,
            ),
        )
        report["application_inventory"] = app_ref
        consumer_result = installed_consumers(
            session,
            args,
            report,
            build=build,
            output=output,
            compiler=compiler,
            static_library=binaries["rns8_static"],
            runtime_witness=dict(
                lane=plan["lane"],
                runtimes=report["runtime_inventory"]["files"],
                windows_asan=report.get("windows_asan_dll"),
                openmp=openmp,
                archer=args.openmp_archer,
            ),
        )
        check_bookends([report, objects, app, consumer_result, session.stages])
        if any(row["status"] != "passed" for row in controls):
            report["blockers"].append("host_detector_sensitivity_failed")
        if app["status"] != "passed":
            report["blockers"].append("application_inventory_failed")
        if consumer_result["status"] != "passed":
            report["blockers"].append("installed_consumer_inventory_failed")
        report["status"] = "failed" if report["blockers"] else "passed"
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["blockers"].append(str(error))
    report["stages"] = session.stages
    report["environment"] = dict(
        compiler_environment_sha256=canonical_digest(compiler_environment(session.environment)),
        sanitizer_options={
            key: value
            for key, value in session.environment.items()
            if key in {"ASAN_OPTIONS", "UBSAN_OPTIONS", "TSAN_OPTIONS", "ASAN_SYMBOLIZER_PATH"}
        },
    )
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
    return 0 if report["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="explicit absolute scratch root; default: repository temp",
    )
    for name in ("cmake", "ctest", "nm", "symbolizer"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--allowed-skips", type=Path)
    parser.add_argument(
        "--openmp-archer",
        type=Path,
        help="Explicit LLVM Archer library; separate native Linux TSan/OpenMP qualification",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--total-seconds",
        type=int,
        help="total owned child-process budget; cleanup/publication remains bounded separately",
    )
    parser.add_argument("--artifact-mib", type=int, default=32768)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        system, lane = PRESETS[args.preset]
        if args.openmp_archer and (system != "Linux" or lane != "thread"):
            raise ValueError("Archer/OpenMP qualification requires the native Linux TSan lane")
        if (
            args.jobs not in range(1, 65)
            or not 1 <= args.timeout_seconds <= 14400
            or not 256 <= args.artifact_mib <= 65536
            or not valid_total_budget(args.total_seconds)
        ):
            raise ValueError("host lane budgets exceed supported bounds")
        artifact_root = args.artifact_root or ROOT / "temp"
        output = owned_output_path(args.out, artifact_root)
        for name in (
            "cmake",
            "ctest",
            "nm",
            "symbolizer",
            "compiler",
            "toolchain",
            "allowed_skips",
            "openmp_archer",
        ):
            path = getattr(args, name)
            if path is not None:
                path = invocation_path(path)
                setattr(args, name, path)
        build = output / "build"
        command = [
            str(args.cmake),
            "--preset",
            args.preset,
            "-S",
            str(ROOT),
            "-B",
            str(build),
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DRNS8_ENABLE_CPU_PARALLEL=" + ("ON" if args.openmp_archer else "OFF"),
            "-DRNS8_REQUIRE_OPENMP=" + ("ON" if args.openmp_archer else "OFF"),
            "-DCMAKE_CXX_COMPILER_LAUNCHER=",
            "-DCMAKE_C_COMPILER_LAUNCHER=",
            "-DRNS8_BUILD_SHARED=OFF",
            "-DRNS8_BUILD_FUZZERS=OFF",
        ]
        if args.toolchain:
            command.append("-DCMAKE_TOOLCHAIN_FILE=" + str(args.toolchain))
        if args.compiler:
            command.append("-DCMAKE_CXX_COMPILER=" + str(args.compiler))
        if system == "Windows" and not args.toolchain:
            raise ValueError("Windows lane requires explicit vcpkg toolchain")
        plan = dict(
            schema="rns8_host_dynamic_check_plan_v1",
            artifact_root=str(artifact_root.resolve()),
            total_seconds=args.total_seconds,
            preset=args.preset,
            system=system,
            lane=lane,
            openmp_archer=str(args.openmp_archer) if args.openmp_archer else None,
            output=str(output),
            build=str(build),
            configure=command,
            build_command=[
                str(args.cmake),
                "--build",
                str(build),
                "--parallel",
                str(args.jobs),
                "--verbose",
            ],
            mandatory_tests=list(MANDATORY),
            ordinary_test_parallelism=1,
            negative_controls_outside_pass_counts=True,
            runtime_validation_executed=False,
            production_latency=False,
        )
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        if platform.system() != system:
            raise ValueError("native host lane unsupported on this operating system")
        return execute(args, plan)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("host checks: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
