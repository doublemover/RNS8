"""Inert process-recorder controls for optional exact-reference doctor probes."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import platform
import struct
import tempfile
import unittest
from unittest.mock import patch

from check_dependencies_lib import references as p
from check_dependencies_lib import probe_artifacts
from check_dependencies_lib.readiness import optional_cpp_references
from check_dependencies_lib.resolution import selection_context

ROOT = Path(__file__).resolve().parents[1]


class ReferenceProbes(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "temp/reference-probe-controls"
        scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for relative in (
            "cmake/modules/FindRNS8ThirdParty.cmake",
            "cmake/rns8_platform_guards.cmake",
            "tools/probes/exact-reference/CMakeLists.txt",
            "tests/unit/test_optional_exact_libs.cpp",
            "cmake.exe",
            "ninja.exe",
            "compiler.exe",
        ):
            self.put(relative)
        self.resolution = {
            "errors": [],
            "identity_sha256": "a" * 64,
            "request": {
                "host": platform.system(),
                "overrides": {
                    "CMAKE_PREFIX_PATH": "/chosen prefix",
                    "CMAKE_CXX_COMPILER": str(self.root / "compiler.exe"),
                },
            },
            "tools": {
                name: {"path": str(self.root / filename)}
                for name, filename in (
                    ("cmake", "cmake.exe"),
                    ("ninja", "ninja.exe"),
                    ("host_cxx", "compiler.exe"),
                )
            },
            "cmake_bindings": {},
            "optional_components": {},
            "toolchain": {},
        }
        for name, header in (("gmp", "gmp.h"), ("mpfr", "mpfr.h"), ("flint", "flint/flint.h")):
            prefix = self.root / (name + " prefix")
            header_path = self.put(prefix / "include" / header)
            library = self.put(prefix / "lib" / (name + ".lib"))
            self.resolution["optional_components"][name] = {
                "status": "discovered",
                "selected": {
                    "canonical_root": str(prefix),
                    "paths": {
                        "include": str(prefix / "include"),
                        "header": str(header_path),
                        "library": str(library),
                    },
                },
            }
        self.put(self.root / "flint prefix/include/flint/fmpz.h")
        root_patch = patch.object(p, "repo_root", return_value=self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def put(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"inert fixture bytes; never a native artifact or qualification")
        return path

    def capture(self, command, **kwargs):
        build = Path(command[command.index("-B") + 1])
        build.mkdir(parents=True)
        component = next(
            arg.split("=", 1)[1] for arg in command if arg.startswith("-DRNS8_REFERENCE_COMPONENT=")
        )
        native = {
            "schema": "rns8_exact_reference_link_probe_v1",
            "component": component,
            "compile_link": "passed",
            "runtime_validation": "not_run",
            "compiler": str(self.root / "compiler.exe"),
            "compiler_id": "inert control",
            "compiler_version": "0.0.0",
            "system": platform.system(),
            "processor": "fixture",
            "pointer_bytes": struct.calcsize("P"),
        }
        (build / "reference-probe.json").write_text(json.dumps(native), encoding="utf-8")
        suffix = ".exe" if platform.system() == "Windows" else ""
        self.put(build / ("rns8-exact-reference-probe" + suffix))
        self.put(build / "exact-reference-link-probe.log")
        return {
            "command": command,
            "process_id": 123,
            "returncode": 0,
            "timed_out": False,
            "limit_exceeded": None,
            "capture_error": None,
            "interrupted": False,
            "stdout": "inert fixture only",
            "stderr": "",
        }

    def run_probe(self, capture=None):
        with (
            selection_context(self.resolution),
            patch.object(p, "run_capture", side_effect=capture or self.capture) as execute,
        ):
            result = p.reference_compile_probes(True)
        return result, execute

    def test_default_never_writes_or_executes(self):
        with (
            patch.object(Path, "mkdir", side_effect=AssertionError("write")),
            patch.object(p, "run_capture", side_effect=AssertionError("process")),
        ):
            result = p.reference_compile_probes(False, self.root / "outside temp")
        self.assertFalse(result["requested"])
        self.assertTrue(all(row["status"] == "NOT_REQUESTED" for row in result["items"].values()))

    def test_explicit_probe_rejects_non_temp_destination(self):
        with self.assertRaisesRegex(ValueError, "temp directory"):
            p.reference_compile_probes(True, self.root / "tracked")

    def test_missing_and_cross_inputs_never_launch(self):
        original = deepcopy(self.resolution)
        for scenario in (
            "selection",
            "host",
            "cross",
            "missing gmp",
            "missing mpfr",
            "missing cmake",
        ):
            with self.subTest(scenario=scenario):
                self.resolution = deepcopy(original)
                if scenario == "selection":
                    self.resolution["errors"] = ["invalid explicit root"]
                elif scenario == "host":
                    self.resolution["request"]["host"] = "OtherHost"
                elif scenario == "cross":
                    self.resolution["request"]["overrides"]["CMAKE_SYSTEM_NAME"] = "OtherTarget"
                elif scenario.startswith("missing "):
                    name = scenario.split()[1]
                    if name == "cmake":
                        self.resolution["tools"][name]["path"] = None
                    else:
                        self.resolution["optional_components"][name]["selected"] = None
                result, execute = self.run_probe()
                self.assertFalse(result["items"]["flint"]["ok"])
                self.assertEqual(execute.call_count, 1 if scenario == "missing mpfr" else 0)

    def test_same_finder_and_native_receipt_without_runtime_credit(self):
        result, execute = self.run_probe()
        self.assertEqual(execute.call_count, 2)
        directories = set()
        for name, row in result["items"].items():
            self.assertEqual(row["status"], "COMPILE_LINK_PASS")
            self.assertEqual(row["runtime_validation"], "not_run")
            self.assertEqual(row["selection_identity_sha256"], "a" * 64)
            self.assertTrue(Path(row["process_receipt"]["path"]).is_file())
            self.assertTrue(Path(row["native_receipt"]["path"]).is_file())
            self.assertTrue(Path(row["link_log"]["path"]).is_file())
            directories.add(row["directory"])
            command = row["command"]
            self.assertIn("-DVCPKG_MANIFEST_INSTALL=OFF", command)
            self.assertIn("-DVCPKG_MANIFEST_MODE=OFF", command)
            self.assertFalse(any(arg in ("--build", "--install", "--test") for arg in command))
            self.assertTrue(any(arg.startswith("-DRNS8_GMP_LIBRARY=") for arg in command))
            if name == "flint":
                self.assertTrue(any(arg.startswith("-DRNS8_MPFR_LIBRARY=") for arg in command))
        self.assertEqual(len(directories), 2)
        with selection_context(self.resolution):
            refs = optional_cpp_references({"gmp": "untrusted inventory"}, result)
        self.assertEqual(refs["gmp"]["compatibility"], "compile_link_compatible_runtime_not_run")
        self.assertEqual(refs["gmp"]["package_inventory_version"], "untrusted inventory")
        for call in execute.call_args_list:
            self.assertEqual(call.kwargs["timeout_seconds"], 180)
            self.assertEqual(call.kwargs["max_artifact_bytes"], 256 * 1024**2)
            self.assertNotIn("CXX", call.kwargs["env"])

    def test_failure_and_native_receipt_substitutions_do_not_pass(self):
        for mutation in (
            "exit",
            "timeout",
            "budget",
            "compiler",
            "component",
            "pointer",
            "runtime",
            "source",
            "nonobject",
            "missing binary",
            "empty binary",
        ):
            with self.subTest(mutation=mutation):

                def capture(command, mutation=mutation, **kwargs):
                    result = self.capture(command, **kwargs)
                    build = Path(command[command.index("-B") + 1])
                    path = build / "reference-probe.json"
                    native = json.loads(path.read_text(encoding="utf-8"))
                    if mutation == "exit":
                        result["returncode"] = 1
                    elif mutation == "timeout":
                        result["timed_out"] = True
                    elif mutation == "budget":
                        result["limit_exceeded"] = "owned_artifact_bytes"
                    elif mutation == "compiler":
                        native["compiler"] = str(self.put("other compiler.exe"))
                    elif mutation == "component":
                        native["component"] = "other"
                    elif mutation == "pointer":
                        native["pointer_bytes"] = True
                    elif mutation == "runtime":
                        native["runtime_validation"] = "passed"
                    elif mutation == "source":
                        with (self.root / "tests/unit/test_optional_exact_libs.cpp").open(
                            "ab"
                        ) as stream:
                            stream.write(b"changed source")
                    elif mutation == "nonobject":
                        native = []
                    else:
                        suffix = ".exe" if platform.system() == "Windows" else ""
                        binary = build / ("rns8-exact-reference-probe" + suffix)
                        if mutation == "empty binary":
                            binary.write_bytes(b"")
                        else:
                            binary.unlink()
                    path.write_text(json.dumps(native), encoding="utf-8")
                    return result

                result, _ = self.run_probe(capture)
                self.assertTrue(all(not row["ok"] for row in result["items"].values()))

    def test_interruption_stops_before_second_component(self):
        def capture(command, **kwargs):
            result = self.capture(command, **kwargs)
            result["interrupted"] = True
            return result

        with (
            selection_context(self.resolution),
            patch.object(p, "run_capture", side_effect=capture) as execute,
        ):
            with self.assertRaises(InterruptedError):
                p.reference_compile_probes(True)
        self.assertEqual(execute.call_count, 1)

    def test_windows_command_retains_spaces_and_all_transitive_choices(self):
        self.resolution["request"]["host"] = "Windows"
        command = p._command(self.resolution, "flint", self.root / "build with spaces")
        self.assertEqual(command[1], str(self.root / "tools/windows_dev.py"))
        for name in ("GMP", "MPFR", "FLINT"):
            expected = self.resolution["optional_components"][name.lower()]["selected"]
            self.assertIn(
                "-DRNS8_" + name + "_ROOT=" + expected["canonical_root"].replace("\\", "/"), command
            )
        self.assertEqual(command[command.index("-B") + 1], str(self.root / "build with spaces"))

    def test_interruption_stops_even_when_its_receipt_cannot_be_written(self):
        def capture(command, **kwargs):
            result = self.capture(command, **kwargs)
            result["interrupted"] = True
            return result

        with (
            selection_context(self.resolution),
            patch.object(p, "run_capture", side_effect=capture) as execute,
            patch.object(probe_artifacts, "write", side_effect=OSError("inert disk failure")),
        ):
            with self.assertRaises(InterruptedError):
                p.reference_compile_probes(True)
        self.assertEqual(execute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
