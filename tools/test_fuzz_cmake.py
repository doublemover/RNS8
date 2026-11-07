"""Authored build/object control for isolated production-library fuzz feedback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FuzzInstrumentation(unittest.TestCase):
    def test_library_object_has_feedback_and_ordinary_library_does_not(self):
        name = "clang-cl.exe" if os.name == "nt" else "clang++"
        selected = (
            Path(os.environ["LLVM_ROOT"]) / "bin" / name
            if os.environ.get("LLVM_ROOT")
            else shutil.which(name)
        )
        if not selected or not Path(selected).is_file():
            self.skipTest("native LLVM compiler unavailable for fuzz instrumentation control")
        compiler = Path(selected).resolve()
        nm = compiler.parent / ("llvm-nm.exe" if os.name == "nt" else "llvm-nm")
        if not nm.is_file():
            self.skipTest("matching llvm-nm unavailable for real object instrumentation inspection")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "branch.cpp").write_text(
                "int branch(int x) { return x == 19 ? x + 3 : x - 7; }\n"
            )
            text = r"""
cmake_minimum_required(VERSION 3.22)
project(authored_fuzz_instrumentation LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
  set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
endif()
set(RNS8_ENABLE_HIP OFF)
set(RNS8_ENABLE_WINDOWS_CLANG_ASAN ON)
set(RNS8_SOURCES branch.cpp)
function(rns8_configure_library name)
  target_compile_features(${name} PRIVATE cxx_std_17)
endfunction()
include("@ROOT@/cmake/rns8_analysis.cmake")
include("@ROOT@/cmake/rns8_fuzz.cmake")
rns8_add_fuzz_library()
add_library(ordinary STATIC branch.cpp)
get_target_property(public rns8_fuzz_library INTERFACE_COMPILE_OPTIONS)
if(public)
  message(FATAL_ERROR "fuzz instrumentation leaked through the target interface")
endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/archives.json"
  CONTENT "[\"$<TARGET_FILE:rns8_fuzz_library>\",\"$<TARGET_FILE:ordinary>\"]")
""".replace("@ROOT@", ROOT.as_posix())
            (root / "CMakeLists.txt").write_text(text)
            build = root / "build"
            wrapper = (
                [sys.executable, str(ROOT / "tools/windows_dev.py")] if os.name == "nt" else []
            )

            def command(argv):
                value = subprocess.run(
                    [*wrapper, *argv], capture_output=True, text=True, timeout=90
                )
                self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
                return value.stdout

            command(
                [
                    "cmake",
                    "-S",
                    str(root),
                    "-B",
                    str(build),
                    "-G",
                    "Ninja",
                    "-DCMAKE_CXX_COMPILER=" + str(compiler),
                ]
            )
            commands = json.loads((build / "compile_commands.json").read_text())
            self.assertEqual(len(commands), 2)
            fuzz = next(row for row in commands if "rns8_fuzz_library.dir" in row["command"])
            ordinary = next(row for row in commands if "ordinary.dir" in row["command"])
            self.assertIn("-fsanitize=fuzzer-no-link", fuzz["command"])
            self.assertNotIn("-fsanitize=fuzzer", ordinary["command"])
            command(["cmake", "--build", str(build), "--target", "rns8_fuzz_library", "ordinary"])
            archives = json.loads((build / "archives.json").read_text())
            instrumented = command([str(nm), archives[0]])
            plain = command([str(nm), archives[1]])
            self.assertIn("__sancov", instrumented)
            self.assertNotIn("__sancov", plain)
            receipt = json.loads((build / "rns8-fuzz-instrumentation-Debug.json").read_text())
            self.assertEqual(receipt["scope"], "configured_not_executed")
            self.assertEqual(receipt["sources"], [(root / "branch.cpp").as_posix()])
            self.assertFalse(receipt["gpu_instrumented"])


if __name__ == "__main__":
    unittest.main()
