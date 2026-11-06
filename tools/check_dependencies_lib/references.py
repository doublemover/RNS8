"""Explicit optional reference compile/link evidence, using the production finder.

Ordinary dependency discovery stays write-free. This opt-in path configures
only a fresh tiny CMake project; it never installs packages or runs its binary.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import secrets
import struct
import sys

from compiler_context import compiler_environment
from json_evidence import parse_evidence_json
from process_progress import capture_succeeded, run_capture
from .probe_artifacts import snapshot as _snapshot, write as _write
from .probe_artifacts import write_process
from .resolution import ACTIVE_RESOLUTION
from .system import repo_root

COMPONENTS = {"gmp": ("gmp",), "flint": ("flint", "mpfr", "gmp")}


def _inputs(resolution: dict, names: tuple[str, ...]) -> dict:
    root = repo_root()
    paths = {
        "finder": root / "cmake/modules/FindRNS8ThirdParty.cmake",
        "platform_guards": root / "cmake/rns8_platform_guards.cmake",
        "probe_project": root / "tools/probes/exact-reference/CMakeLists.txt",
        "arithmetic_source": root / "tests/unit/test_optional_exact_libs.cpp",
        "cmake": Path(resolution["tools"]["cmake"]["path"]),
        "ninja": Path(resolution["tools"]["ninja"]["path"]),
    }
    for name in names:
        selected = resolution["optional_components"][name]["selected"]
        for kind in ("header", "library"):
            paths[name + ":" + kind] = Path(selected["paths"][kind])
        if name == "flint":
            paths["flint:fmpz_header"] = Path(selected["paths"]["include"]) / "flint/fmpz.h"
    toolchain = resolution.get("toolchain", {}).get("path")
    if toolchain:
        paths["toolchain"] = Path(toolchain)
    compiler = resolution["tools"].get("host_cxx", {}).get("path")
    if compiler:
        paths["requested_compiler"] = Path(compiler)
    return {name: _snapshot(path) for name, path in paths.items()}


def _command(resolution: dict, name: str, build: Path) -> list[str]:
    root = repo_root()
    bindings = resolution["cmake_bindings"]
    # Preserve the selected host compiler/sysroot/package context, not the HIP
    # compiler or optional GPU feature switches of the enclosing configuration.
    definitions = {
        key: value
        for key, value in resolution["request"]["overrides"].items()
        if key.startswith("CMAKE_")
    }
    definitions.update(
        (key, value) for key, value in bindings.items() if key.startswith(("VCPKG_", "CMAKE_"))
    )
    for dependency in COMPONENTS[name]:
        selected = resolution["optional_components"][dependency]["selected"]
        prefix = "RNS8_" + dependency.upper()
        definitions[prefix + "_ROOT"] = selected["canonical_root"]
        for suffix, kind in (("_INCLUDE_DIR", "include"), ("_LIBRARY", "library")):
            definitions[prefix + suffix] = selected["paths"][kind]
    prefixes = [definitions.get("CMAKE_PREFIX_PATH", "")]
    prefixes += [
        resolution["optional_components"][dep]["selected"]["canonical_root"]
        for dep in COMPONENTS[name]
    ]
    definitions["CMAKE_PREFIX_PATH"] = ";".join(value for value in prefixes if value)
    compiler = resolution["tools"].get("host_cxx", {}).get("path")
    if compiler:
        definitions["CMAKE_CXX_COMPILER"] = compiler
    definitions.update(
        CMAKE_BUILD_TYPE="Release",
        CMAKE_TRY_COMPILE_CONFIGURATION="Release",
        CMAKE_MAKE_PROGRAM=resolution["tools"]["ninja"]["path"],
        VCPKG_MANIFEST_MODE="OFF",
        VCPKG_MANIFEST_INSTALL="OFF",
        RNS8_REFERENCE_COMPONENT=name,
    )
    command = [
        resolution["tools"]["cmake"]["path"],
        "-S",
        str(root / "tools/probes/exact-reference"),
        "-B",
        str(build),
        "-G",
        "Ninja",
        *[
            "-D" + key + "=" + str(value).replace("\\", "/")
            for key, value in sorted(definitions.items())
        ],
    ]
    if resolution["request"]["host"] == "Windows":
        command = [sys.executable, str(root / "tools/windows_dev.py"), *command]
    return command


def reference_compile_probes(requested: bool = False, probe_dir: Path | None = None) -> dict:
    root = repo_root()
    parent = (probe_dir or root / "temp/exact-reference-deps").resolve()
    resolution = ACTIVE_RESOLUTION.get()
    result = {
        "schema": "rns8_exact_reference_probes_v1",
        "requested": requested,
        "probe_root": str(parent),
        "policy": "optional compile/link evidence only; no installation, runtime execution, feature enablement or host readiness credit",
        "runtime_validation": "not_run",
        "items": {},
    }
    if requested and not parent.is_relative_to((root / "temp").resolve()):
        raise ValueError("reference probe artifacts must stay under the repository temp directory")
    for name, dependencies in COMPONENTS.items():
        row = {"status": "NOT_REQUESTED", "ok": False, "runtime_validation": "not_run"}
        result["items"][name] = row
        if not requested:
            continue
        if not resolution or resolution.get("errors"):
            row.update(
                status="NOT_RUN_INVALID_SELECTION",
                detail="Repair shared dependency resolution first",
            )
            continue
        if (
            platform.system() not in {"Windows", "Linux"}
            or resolution["request"]["host"] != platform.system()
            or resolution["request"]["overrides"].get("CMAKE_SYSTEM_NAME")
            not in (None, "", platform.system())
        ):
            row.update(
                status="NOT_RUN_UNSUPPORTED_HOST",
                detail="Reference doctor probes require a native host; use the selected cross build directly",
            )
            continue
        missing = [
            dep
            for dep in dependencies
            if not resolution["optional_components"].get(dep, {}).get("selected")
        ]
        missing += [
            tool for tool in ("cmake", "ninja") if not resolution["tools"].get(tool, {}).get("path")
        ]
        if missing:
            row.update(
                status="NOT_RUN_MISSING_INPUT",
                missing=missing,
                detail="Supply native dependency roots and CMake/Ninja; inventory versions are not usable files",
            )
            continue
        directory = parent / (name + "-" + secrets.token_hex(12))
        directory.mkdir(parents=True, exist_ok=False)
        build = directory / "build"
        try:
            before = _inputs(resolution, dependencies)
            command = _command(resolution, name, build)
            environment = dict(os.environ)
            # Default to native compiler search, never an inherited CXX=hipcc.
            environment.pop("CC", None)
            environment.pop("CXX", None)
            row.update(
                command=command,
                directory=str(directory),
                input_bookends=before,
                launcher_compiler_environment=compiler_environment(environment),
                selection_identity_sha256=resolution["identity_sha256"],
                host_environment=resolution.get("host_environment", {"kind": "unknown"}),
            )
            process = run_capture(
                command,
                cwd=root,
                label="reference-probe:" + name,
                env=environment,
                timeout_seconds=180,
                max_output_bytes=8 * 1024**2,
                artifact_roots=(directory,),
                max_artifact_bytes=256 * 1024**2,
                return_on_interrupt=True,
            )
            row["process_receipt"] = write_process(directory / "process.json", process)
            row["status"] = "COMPILE_LINK_FAIL"
            if capture_succeeded(process):
                native_path = build / "reference-probe.json"
                if native_path.stat().st_size > 65536:
                    raise ValueError("native reference probe receipt exceeds its byte budget")
                native = parse_evidence_json(native_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(native, dict)
                    or native.get("schema") != "rns8_exact_reference_link_probe_v1"
                    or native.get("component") != name
                    or native.get("compile_link") != "passed"
                    or native.get("runtime_validation") != "not_run"
                    or native.get("system") != platform.system()
                    or type(native.get("pointer_bytes")) is not int
                    or native["pointer_bytes"] != struct.calcsize("P")
                    or not all(
                        type(native.get(key)) is str and native[key]
                        for key in ("compiler", "compiler_id", "compiler_version")
                    )
                ):
                    raise ValueError("native compiler/link receipt is missing or inconsistent")
                actual_compiler = _snapshot(Path(native["compiler"]))
                requested_compiler = before.get("requested_compiler")
                if requested_compiler and requested_compiler != actual_compiler:
                    raise ValueError("native compiler differs from the selected host compiler")
                binary = build / (
                    "rns8-exact-reference-probe"
                    + (".exe" if platform.system() == "Windows" else "")
                )
                row.update(
                    native_receipt=_snapshot(native_path),
                    native_compiler=actual_compiler,
                    native=native,
                    binary=_snapshot(binary),
                )
                if before != _inputs(resolution, dependencies):
                    raise ValueError("selected probe inputs changed during compilation")
                row.update(status="COMPILE_LINK_PASS", ok=True)
            log = build / "exact-reference-link-probe.log"
            if log.is_file():
                row["link_log"] = _snapshot(log)
        except InterruptedError:
            raise
        except (OSError, ValueError) as error:
            row.update(status="PROBE_ERROR", ok=False, detail=str(error))
        _write(directory / "probe.json", row)
    return result
