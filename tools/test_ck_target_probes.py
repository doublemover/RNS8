"""Configure command-recorder regressions; no compiler, SDK or GPU execution."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CkTargetProbes(unittest.TestCase):
    def test_every_target_and_late_failure_are_retained(self):
        cmake = shutil.which(
            "cmake", path=str(Path(sys.executable).parent / "Scripts")
        ) or shutil.which("cmake")
        self.assertIsNotNone(cmake)
        parent = ROOT / "temp/ck-target-probe-controls"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            base = Path(folder)
            prefix = base / "CK prefix with spaces"
            for header, contents in (
                ("ck.hpp", "// inert header"),
                ("version.h", "#define CK_VERSION_MAJOR 0"),
                (
                    "tensor_operation/gpu/warp/wmma_gemm.hpp",
                    "bool neg_a = true,\nbool neg_b = true,",
                ),
                (
                    "tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp",
                    "// inert header",
                ),
            ):
                path = prefix / "include/ck" / header
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            compiler = base / "inert hipcc.exe"
            compiler.write_text("never execute this file", encoding="utf-8")
            source = base / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.22)\nproject(InertCkProbe LANGUAGES NONE)\n"
                f'include("{ROOT.as_posix()}/cmake/rns8_platform_guards.cmake")\n'
                "set(CMAKE_DISABLE_FIND_PACKAGE_Git TRUE)\n"
                f'set(RNS8_CK_INCLUDE_DIR "{prefix.as_posix()}/include")\n'
                f'set(RNS8_CK_ROOT "{prefix.as_posix()}")\n'
                f'set(RNS8_CK_HIPCC "{compiler.as_posix()}")\n'
                # Override only the compiler-launch boundary. This fixture
                # writes command records and never supplies native evidence.
                "function(execute_process)\n"
                '  cmake_parse_arguments(call "" "RESULT_VARIABLE;OUTPUT_VARIABLE;ERROR_VARIABLE;TIMEOUT" "COMMAND" ${ARGN})\n'
                '  file(APPEND "${CMAKE_BINARY_DIR}/commands.txt" "${call_COMMAND}\\n")\n'
                "  set(result 0)\n"
                '  if(FAIL_TARGET AND "--offload-arch=${FAIL_TARGET}" IN_LIST call_COMMAND)\n'
                "    set(result 1)\n"
                "  endif()\n"
                "  set(${call_RESULT_VARIABLE} ${result} PARENT_SCOPE)\n"
                '  set(${call_OUTPUT_VARIABLE} "inert command recorder only" PARENT_SCOPE)\n'
                '  set(${call_ERROR_VARIABLE} "" PARENT_SCOPE)\n'
                "endfunction()\n"
                f'include("{ROOT.as_posix()}/cmake/modules/FindRNS8CK.cmake")\n'
                'file(WRITE "${CMAKE_BINARY_DIR}/status.txt" "${RNS8CK_FOUND}|${RNS8_CK_COMPILE_PROBE_STATUS}|${RNS8_CK_PRIMITIVE_PROBE_STATUS}\\n${RNS8_CK_TARGET_PROBE_RESULTS}\\n${RNS8_CK_USE_XDL}|${RNS8_CK_SHAPE_ALIGNMENT}")\n',
                encoding="utf-8",
            )
            for targets, xdl, alignment in (
                ("gfx1100;gfx1102:xnack-", 0, 64),
                ("gfx90a;gfx942:sramecc+:xnack-", 1, 128),
            ):
                for fail in (False, True):
                    with self.subTest(targets=targets, fail=fail):
                        build = base / ("build-" + str(xdl) + "-" + str(fail))
                        last = targets.split(";")[-1]
                        result = subprocess.run(
                            [
                                cmake,
                                "-S",
                                str(source),
                                "-B",
                                str(build),
                                "-DRNS8_AMDGPU_TARGETS=" + targets,
                                "-DFAIL_TARGET=" + (last if fail else ""),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        commands = (build / "commands.txt").read_text(encoding="utf-8").splitlines()
                        self.assertEqual(len(commands), 4)
                        for target in targets.split(";"):
                            self.assertEqual(
                                sum(
                                    "--offload-arch=" + target in row.split(";") for row in commands
                                ),
                                2,
                            )
                        primitive = [row for row in commands if "-c" in row.split(";")]
                        self.assertEqual(len(primitive), 2)
                        outputs = set()
                        for row in primitive:
                            argv = row.split(";")
                            self.assertIn("-DRNS8_CK_PRIMITIVE_USE_XDL=" + str(xdl), argv)
                            self.assertTrue(
                                any(arg.endswith("tools/probes/ck_primitive.hip") for arg in argv)
                            )
                            outputs.add(argv[argv.index("-o") + 1])
                        self.assertEqual(len(outputs), 2)
                        status = (build / "status.txt").read_text(encoding="utf-8")
                        self.assertTrue(
                            status.startswith(
                                "FALSE|compile_probe_fail|primitive_object_probe_fail"
                                if fail
                                else "TRUE|compile_probe_pass|primitive_object_probe_pass"
                            ),
                            status,
                        )
                        self.assertTrue(status.endswith(f"{xdl}|{alignment}"))
                        for target in targets.split(";"):
                            self.assertIn(target + "|", status)
                        self.assertEqual(
                            len(
                                list((build / "accelerator-dependency-probes").rglob("compile.log"))
                            ),
                            2,
                        )
                        self.assertEqual(
                            len(
                                list(
                                    (build / "accelerator-dependency-probes").rglob("primitive.log")
                                )
                            ),
                            2,
                        )


if __name__ == "__main__":
    unittest.main()
