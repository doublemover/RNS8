"""Authored CPU-only configure controls; no library/device build or analysis run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AnalysisCmake(unittest.TestCase):
    def test_compile_receipt_wrapper_preserves_configured_child_argv_without_running_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = r"""
cmake_minimum_required(VERSION 3.22)
project(authored_compile_receipt LANGUAGES NONE)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_OUTPUT_EXTENSION ".obj")
set(PROJECT_SOURCE_DIR "@ROOT@")
include("@ROOT@/cmake/rns8_platform_guards.cmake")
include("@ROOT@/cmake/rns8_analysis.cmake")
set(Python3_EXECUTABLE "@PYTHON@")
set(RNS8_DISCOVERY_IDENTITY_SHA256 "authored_fixture_not_an_SDK_selection")
set(RNS8_HIP_HIPCC "@PYTHON@")
set(RNS8_HIP_CLANG "@PYTHON@")
set(RNS8_HIP_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/authored_include")
set(RNS8_HIP_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/authored_runtime")
set(RNS8_AMDGPU_TARGETS "gfx1100")
include("@ROOT@/cmake/modules/FindRNS8HIP.cmake")
set(RNS8_MANIFEST_HEADER "${CMAKE_CURRENT_BINARY_DIR}/manifest.hpp")
file(WRITE "${RNS8_MANIFEST_HEADER}" "// authored, no compiled SDK claim\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/native.hip" "// authored source\n")
add_custom_target(rns8_build_manifest)
rns8_compile_hip_source(native "${CMAKE_CURRENT_BINARY_DIR}/native.hip")
add_custom_target(authored DEPENDS "${native}")
rns8_write_analysis_metadata()
""".replace("@ROOT@", ROOT.as_posix()).replace("@PYTHON@", Path(sys.executable).as_posix())
            (root / "CMakeLists.txt").write_text(source, encoding="utf-8")
            for enabled in (False, True):
                build = root / ("recorded" if enabled else "ordinary")
                configured = subprocess.run(
                    [
                        "cmake",
                        "-S",
                        str(root),
                        "-B",
                        str(build),
                        "-G",
                        "Ninja",
                        "-DRNS8_CAPTURE_HIP_COMPILE_COST=" + ("ON" if enabled else "OFF"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
                commands = subprocess.run(
                    ["ninja", "-C", str(build), "-t", "commands", "authored"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(commands.returncode, 0, commands.stdout + commands.stderr)
                self.assertEqual("compile_receipt.py" in commands.stdout, enabled)
                context = build / "hip/native.obj.compile-context-Release.json"
                self.assertEqual(context.is_file(), enabled)
                if enabled:
                    recorded = json.loads(context.read_text(encoding="utf-8"))
                    inventory = json.loads(
                        (build / "rns8-hip-compile-commands-Release.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(recorded["arguments"], inventory["commands"][0]["arguments"])
                    self.assertEqual(recorded["targets"], ["gfx1100"])
                    self.assertEqual(recorded["file"], str(build / "native.hip").replace("\\", "/"))
                    self.assertIn(
                        str(build / "manifest.hpp").replace("\\", "/"), recorded["inputs"]
                    )
                self.assertFalse((build / "hip/native.obj").exists())

    def test_private_warning_scope_and_exact_custom_hip_argv_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src/owned.cpp").write_text("int authored() { return 0; }\n")
            upstream = root / "third_party/vendor"
            upstream.mkdir(parents=True)
            (upstream / "CMakeLists.txt").write_text(
                "add_library(upstream STATIC ../../src/owned.cpp)\n"
            )
            module = (ROOT / "cmake/rns8_analysis.cmake").as_posix()
            source = r"""
cmake_minimum_required(VERSION 3.22)
project(authored_analysis_contract LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Debug)
include("@MODULE@")
add_library(owned STATIC src/owned.cpp)
add_executable(owned_tool src/owned.cpp)
add_library(interface_consumer INTERFACE)
target_link_libraries(interface_consumer INTERFACE owned)
add_subdirectory(third_party/vendor)
rns8_private_warnings_in_directory("${PROJECT_SOURCE_DIR}")
foreach(name IN ITEMS owned owned_tool)
  get_target_property(policy ${name} RNS8_PRIVATE_WARNING_POLICY)
  get_target_property(private ${name} COMPILE_OPTIONS)
  get_target_property(public ${name} INTERFACE_COMPILE_OPTIONS)
  if(NOT policy STREQUAL "first-party-v1" OR NOT private OR public)
    message(FATAL_ERROR "private first-party warning contract failed")
  endif()
endforeach()
get_target_property(vendor upstream COMPILE_OPTIONS)
get_target_property(consumer interface_consumer INTERFACE_COMPILE_OPTIONS)
if(vendor OR consumer)
  message(FATAL_ERROR "warning policy leaked to vendor/consumer")
endif()
set(RNS8_AMDGPU_TARGETS "gfx1100;gfx942")
set(args "compiler with spaces" "-Ipath with spaces/λ" "-DSEMICOLON=a\;b" "-DQUOTED=\"bytes\"" "-c" "input.hip")
set(environment "HIP_PATH=SDK with spaces" "EXPLICIT=a\;b")
rns8_record_hip_compile("input.hip" "output.obj" "${args}" "${environment}" "generated/$<CONFIG>/manifest.hpp;overlay.hpp")
rns8_write_analysis_metadata()
""".replace("@MODULE@", module)
            (root / "CMakeLists.txt").write_text(source, encoding="utf-8")
            wrapper = (
                [sys.executable, str(ROOT / "tools/windows_dev.py")] if os.name == "nt" else []
            )
            result = subprocess.run(
                [*wrapper, "cmake", "-S", str(root), "-B", str(root / "build"), "-G", "Ninja"],
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            document = json.loads((root / "build/rns8-hip-compile-commands.json").read_text())
            self.assertEqual(
                document["scope"], "configured_commands_not_executed_compiler_receipts"
            )
            self.assertEqual(len(document["commands"]), 1)
            command = document["commands"][0]
            self.assertEqual(
                command["arguments"],
                [
                    "compiler with spaces",
                    "-Ipath with spaces/λ",
                    "-DSEMICOLON=a;b",
                    '-DQUOTED="bytes"',
                    "-c",
                    "input.hip",
                ],
            )
            self.assertEqual(command["environment"], ["HIP_PATH=SDK with spaces", "EXPLICIT=a;b"])
            self.assertEqual(command["targets"], ["gfx1100", "gfx942"])
            self.assertEqual(
                command["generated_inputs"], ["generated/Debug/manifest.hpp", "overlay.hpp"]
            )
            self.assertEqual(command["configuration"], "Debug")
            self.assertEqual(
                document,
                json.loads((root / "build/rns8-hip-compile-commands-Debug.json").read_text()),
            )


if __name__ == "__main__":
    unittest.main()
