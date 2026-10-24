#!/usr/bin/env python3
"""Build installed workload sources using public exported targets and run them.

Invoked by coordinated CTest only; this test intentionally configures/builds a
consumer. External Boost/json/compiler overrides are copied explicitly from the
parent cache. No RNS8 private or source include path is forwarded.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROGRAMS = ("integer-cross-product", "gf251-transform", "resident-chain", "complex-product", "exactness-walkthrough")


def cache_values(build: Path) -> dict[str, str]:
    result = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line or ":" not in line.split("=", 1)[0]: continue
        key, value = line.split("=", 1); result[key.split(":", 1)[0]] = value
    return result


def run(command: list[str], timeout: int = 180) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
    print(result.stdout, end=""); print(result.stderr, end="", file=sys.stderr)
    if result.returncode: raise RuntimeError(f"consumer command failed: {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--backend", choices=("cpu-reference", "hip-direct"), required=True)
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    build = args.build_dir.resolve(strict=True)
    values = cache_values(build)
    root = build / ("workload-installed-" + args.backend + ("-shared" if args.shared else "-static"))
    # Only the exact test-owned child may be replaced, never the parent build.
    if root.resolve().parent != build: raise RuntimeError("invalid consumer scratch path")
    if root.exists(): shutil.rmtree(root)
    stage = root / "stage"; consumer = root / "consumer"
    config = ["--config", args.config] if args.config else []
    cmake = values.get("CMAKE_COMMAND", "cmake")
    run([cmake, "--install", str(build), "--prefix", str(stage), *config])
    sources = list(stage.glob("**/RNS8/examples/workloads/CMakeLists.txt"))
    if len(sources) != 1: raise RuntimeError("installed public workload source directory missing/ambiguous")
    source = sources[0].parent
    for path in source.glob("*.?pp"):
        text = path.read_text(encoding="utf-8")
        if "api_internal" in text or "src/core" in text: raise RuntimeError("installed workload uses private source API")
    command = [cmake, "-S", str(source), "-B", str(consumer), "-G", values["CMAKE_GENERATOR"],
               "-DCMAKE_PREFIX_PATH=" + str(stage), "-DBUILD_TESTING=OFF", "-DRNS8_WORKLOAD_GPU_TESTS=OFF",
               "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF", "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
               "-DVCPKG_MANIFEST_INSTALL=OFF",
               "-DRNS8_WORKLOAD_USE_SHARED=" + ("ON" if args.shared else "OFF")]
    for key in ("CMAKE_CXX_COMPILER", "CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS", "CMAKE_EXE_LINKER_FLAGS",
                "CMAKE_CXX_FLAGS_DEBUG", "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_CXX_FLAGS_RELWITHDEBINFO", "CMAKE_CXX_FLAGS_MINSIZEREL",
                "CMAKE_MSVC_RUNTIME_LIBRARY", "CMAKE_TOOLCHAIN_FILE", "VCPKG_TARGET_TRIPLET", "VCPKG_HOST_TRIPLET",
                "VCPKG_INSTALLED_DIR",
                "RNS8_GMP_INCLUDE_DIR", "RNS8_GMP_LIBRARY", "RNS8_FLINT_INCLUDE_DIR", "RNS8_FLINT_LIBRARY",
                "RNS8_WORKLOAD_FLINT_EXTRA_LIBRARIES",
                "RNS8_BOOST_MULTIPRECISION_INCLUDE_DIR", "nlohmann_json_DIR"):
        if values.get(key): command.append("-D" + key + "=" + values[key])
    for library in ("GMP","FLINT"):
        enabled=values.get("RNS8_WORKLOAD_ENABLE_"+library,values.get("RNS8_ENABLE_"+library,"OFF"))
        command.append("-DRNS8_WORKLOAD_ENABLE_"+library+"="+enabled)
    run(command)
    run([cmake, "--build", str(consumer), "--parallel", "2", *config], timeout=300)
    for name in PROGRAMS:
        filename = "rns8-workload-" + name + (".exe" if os.name == "nt" else "")
        candidates = list(consumer.glob("**/" + filename))
        if len(candidates) != 1: raise RuntimeError("consumer executable missing/ambiguous: " + filename)
        command = [str(candidates[0]), "--backend", args.backend, "--device",
                   "-1" if args.backend == "cpu-reference" else "0", "--size", "tiny", "--repeats", "1", "--warmups", "0", "--json"]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=90)
        report = json.loads(result.stdout)
        if result.returncode == 77 and report.get("status") == "unsupported":
            print(report); return 77
        if result.returncode or report.get("status") != "passed":
            raise RuntimeError(name + " installed execution failed: " + result.stdout + result.stderr)
        if report.get("backend_requested") != args.backend: raise RuntimeError("installed backend identity mismatch")
        print(name + ": installed " + args.backend + " PASS " + report["checksum_hex"])
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr); raise SystemExit(1)
