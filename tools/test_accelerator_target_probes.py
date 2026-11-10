"""Inert target/probe-boundary controls; no compiler, SDK or GPU execution."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check_dependencies_lib import accelerators as p
from check_dependencies_lib import probe_artifacts
from check_dependencies_lib.resolution import selection_context

ROOT = Path(__file__).resolve().parents[1]


class AcceleratorTargets(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "temp/accelerator-target-probe-controls"
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.header = self.put("component header.hpp")
        self.compiler = self.put("compiler with spaces/hipcc.exe")
        self.put("src/backend_amdgpu_builtins/amdgpu_builtins_kernels.hip")
        self.put("src/backend_common/matrix_target.hpp")
        self.component = {"header": str(self.header)}
        for attribute, value in (
            ("repo_root", self.root),
            ("include_roots_for_component", [str(self.root / "include with spaces")]),
        ):
            item = patch.object(p, attribute, return_value=value)
            item.start()
            self.addCleanup(item.stop)
        item = patch.object(
            p,
            "command_with_windows_developer_environment",
            side_effect=lambda argv: (argv, "inert direct command"),
        )
        item.start()
        self.addCleanup(item.stop)

    def put(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"inert fixture; never executable code or native evidence")
        return path

    def capture(self, command, **kwargs):
        self.put(Path(command[command.index("-o") + 1]))
        return {
            "command": command,
            "process_id": 123,
            "returncode": 0,
            "stdout": "inert command recorder",
            "stderr": "",
            "timed_out": False,
            "interrupted": False,
            "capture_error": None,
            "limit_exceeded": None,
        }

    def probe(self, targets, name="ck", capture=None):
        with patch.object(p, "run_capture", side_effect=capture or self.capture) as execute:
            result = p.compile_accelerator_target_probes(
                name, self.component, self.root / "probes", targets, str(self.compiler)
            )
        return result, execute

    def test_explicit_targets_override_observed_device_and_defaults(self):
        for overrides, expected in (
            ({}, ("gfx1100",)),
            (
                {"RNS8_AMDGPU_TARGETS": "gfx942:xnack-;gfx950;gfx942:xnack-"},
                ("gfx942:xnack-", "gfx950"),
            ),
        ):
            with selection_context(
                {"request": {"overrides": overrides}, "effective_identity": {"targets": "gfx1100"}}
            ):
                self.assertEqual(p.requested_probe_targets("gfx1200"), expected)
        with selection_context(None):
            self.assertEqual(p.requested_probe_targets("gfx942"), ("gfx942",))
            with self.assertRaises(ValueError):
                p.requested_probe_targets(None)
        for invalid in (
            "",
            "gfx1100;",
            "gfx1100:xnack+:xnack-",
            "gfx1100:bad+",
            "gfx1100;--output=bad",
            ";".join("gfx" + str(n) for n in range(17)),
        ):
            with (
                self.subTest(targets=invalid),
                selection_context({"request": {"overrides": {"RNS8_AMDGPU_TARGETS": invalid}}}),
            ):
                with self.assertRaises(ValueError):
                    p.requested_probe_targets("gfx1100")

    def test_default_and_invalid_selection_do_not_write_or_run(self):
        with (
            patch.object(Path, "mkdir", side_effect=AssertionError("write")),
            patch.object(p, "run_capture", side_effect=AssertionError("process")),
        ):
            result = p.accelerator_compile_probes({}, False, self.root / "unwritten", None)
            self.assertFalse(result["requested"])
            with selection_context({"request": {"overrides": {"RNS8_AMDGPU_TARGETS": ""}}}):
                result = p.accelerator_compile_probes({}, True, self.root / "unwritten", "gfx1100")
            self.assertTrue(
                all(row["status"] == "NOT_RUN_INVALID_TARGETS" for row in result["items"].values())
            )

    def test_each_target_has_its_own_object_process_and_lowering(self):
        for targets, xdl in ((("gfx1100", "gfx1201"), 0), (("gfx90a", "gfx942:xnack-"), 1)):
            result, execute = self.probe(targets)
            self.assertTrue(result["primitive_probe_ok"])
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(result["primitive_targets"], list(targets))
            objects = set()
            for target, row in zip(targets, result["primitive_target_probes"]):
                self.assertEqual(row["target"], target)
                self.assertIn("--offload-arch=" + target, row["primitive_command"])
                self.assertIn("-DRNS8_CK_PRIMITIVE_USE_XDL=" + str(xdl), row["primitive_command"])
                self.assertTrue(Path(row["primitive_process_receipt"]["path"]).is_file())
                objects.add(row["primitive_object"])
            self.assertEqual(len(objects), 2)
            self.assertEqual(result["primitive_object"], "")
            for call in execute.call_args_list:
                self.assertEqual(call.kwargs["timeout_seconds"], 180)
                self.assertEqual(call.kwargs["max_output_bytes"], 8 * 1024**2)

    def test_late_failure_mixed_package_and_missing_objects_cannot_pass(self):
        for outcome in (
            "late failure",
            "mixed",
            "missing object",
            "empty object",
            "changed header",
            "timeout",
            "output budget",
        ):
            with self.subTest(outcome=outcome):

                def capture(command, outcome=outcome, **kwargs):
                    result = self.capture(command, **kwargs)
                    target = next(arg for arg in command if arg.startswith("--offload-arch="))
                    obj = Path(command[command.index("-o") + 1])
                    if outcome == "late failure" and target.endswith("gfx1102"):
                        result["returncode"] = 1
                    elif outcome == "missing object":
                        obj.unlink()
                    elif outcome == "empty object":
                        obj.write_bytes(b"")
                    elif outcome == "changed header":
                        with self.header.open("ab") as stream:
                            stream.write(b"changed")
                    elif outcome == "timeout":
                        result["timed_out"] = True
                    elif outcome == "output budget":
                        result["limit_exceeded"] = "stdout_stderr_bytes"
                    return result

                targets = ("gfx1100", "gfx942") if outcome == "mixed" else ("gfx1100", "gfx1102")
                result, _ = self.probe(targets, capture=capture)
                self.assertFalse(result["primitive_probe_ok"])
                self.assertEqual(len(result["primitive_target_probes"]), 2)
                if outcome == "mixed":
                    self.assertEqual(
                        result["primitive_probe_status"], "INCOMPATIBLE_HOST_TARGET_SET"
                    )
                if outcome == "late failure":
                    self.assertTrue(result["primitive_target_probes"][0]["primitive_probe_ok"])

    def test_builtins_compile_actual_body_without_empty_target_credit(self):
        result, execute = self.probe(("gfx1100", "gfx942"), name="amdgpu_builtins")
        self.assertTrue(result["primitive_probe_ok"])
        self.assertEqual(execute.call_count, 2)
        for row in result["primitive_target_probes"]:
            text = Path(row["primitive_source"]).read_text(encoding="utf-8")
            self.assertIn("backend_amdgpu_builtins/amdgpu_builtins_kernels.hip", text)
            self.assertIn("#error", text)
            self.assertIn("production_builtin_source", row["primitive_input_bookends"])
        result, execute = self.probe(("gfx90a", "gfx950"), name="amdgpu_builtins")
        self.assertFalse(result["primitive_probe_ok"])
        self.assertEqual(execute.call_count, 0)
        self.assertEqual(result["primitive_probe_status"], "NOT_RUN_UNSUPPORTED_TARGET")

    def test_wrapped_launch_preserves_selected_sdk_environment(self):
        active = {
            "request": {"host": "Windows"},
            "hip": {
                "selected": {
                    "canonical_root": str(self.root / "selected SDK"),
                    "paths": {
                        "include": str(self.root / "selected SDK/include"),
                        "clang": str(self.root / "selected SDK/llvm/bin/clang.exe"),
                        "device_libraries": str(self.root / "selected SDK/bitcode"),
                    },
                }
            },
        }
        with (
            selection_context(active),
            patch.object(
                p,
                "command_with_windows_developer_environment",
                return_value=("inert wrapped shell", "wrapped"),
            ),
            patch.dict(
                p.os.environ,
                {"HIP_PATH": "stale", "HIP_CLANG_PATH": "stale", "UNRELATED": "retained"},
            ),
        ):
            result, execute = self.probe(("gfx1100",), name="amdgpu_builtins")
        self.assertTrue(result["primitive_probe_ok"])
        command = execute.call_args.args[0]
        self.assertEqual(command[1], str(self.root / "tools/windows_dev.py"))
        environment = execute.call_args.kwargs["env"]
        for key, value in p.selected_hip_environment(active).items():
            self.assertEqual(environment[key], value)
        self.assertEqual(environment["UNRELATED"], "retained")

    def test_interrupt_does_not_run_second_target(self):
        def capture(command, **kwargs):
            result = self.capture(command, **kwargs)
            result["interrupted"] = True
            return result

        with patch.object(p, "run_capture", side_effect=capture) as execute:
            with self.assertRaises(InterruptedError):
                p.compile_accelerator_target_probes(
                    "ck",
                    self.component,
                    self.root / "probes",
                    ("gfx1100", "gfx1102"),
                    str(self.compiler),
                )
        self.assertEqual(execute.call_count, 1)

    def test_missing_compiler_is_distinct_from_unsupported_target(self):
        with patch.object(p, "run_capture", side_effect=AssertionError("process")):
            result = p.compile_accelerator_target_probes(
                "ck", self.component, self.root / "probes", ("gfx90a", "gfx942"), None
            )
        self.assertFalse(result["primitive_probe_ok"])
        self.assertEqual(result["primitive_probe_status"], "NOT_RUN_MISSING_COMPILER")
        self.assertFalse((self.root / "probes").exists())

    def test_interruption_survives_a_failed_receipt_write(self):
        def capture(command, **kwargs):
            result = self.capture(command, **kwargs)
            result["interrupted"] = True
            return result

        with (
            patch.object(p, "run_capture", side_effect=capture) as execute,
            patch.object(probe_artifacts, "write", side_effect=OSError("inert disk failure")),
        ):
            with self.assertRaises(InterruptedError):
                p.compile_accelerator_target_probes(
                    "ck",
                    self.component,
                    self.root / "probes",
                    ("gfx1100", "gfx1102"),
                    str(self.compiler),
                )
        self.assertEqual(execute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
