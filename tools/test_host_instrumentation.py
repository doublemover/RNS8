"""Authored detector/object/owned-session controls, using fixture processes only."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import host_instrumentation_session as session
from host_instrumentation_contract import (
    detector_result,
    native_argv,
    object_path,
    selected_flags,
    symbol_classes,
    valid_result,
)
from test_execution_inventory import inventory, report


def capture(stdout="", stderr="", returncode=0):
    return dict(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        process_id=4321,
        timed_out=False,
        interrupted=False,
        limit_exceeded=None,
        capture_error=None,
    )


class DetectorControls(unittest.TestCase):
    def test_only_intended_bound_nonzero_diagnostic_supplies_sensitivity(self):
        rows = (
            (
                "heap-overread",
                86,
                "==4321==ERROR: AddressSanitizer: heap-buffer-overflow\n host_sanitizer_control.cpp:28",
            ),
            (
                "signed-overflow",
                87,
                "host_sanitizer_control.cpp:35 runtime error: signed integer overflow",
            ),
            (
                "race",
                88,
                "WARNING: ThreadSanitizer: data race (pid=4321)\n host_sanitizer_control.cpp:50",
            ),
            (
                "openmp-race",
                88,
                "WARNING: ThreadSanitizer: data race (pid=4321)\n host_sanitizer_control.cpp:80",
            ),
        )
        for mode, rc, diagnostic in rows:
            result = capture(stderr=diagnostic, returncode=rc)
            with self.subTest(mode=mode):
                value = detector_result(mode, result)
                self.assertEqual(value["status"], "passed")
                self.assertEqual(value["application_pass_count"], 0)
                for key, replacement in (
                    ("returncode", 0),
                    ("returncode", 1),
                    ("returncode", -11),
                    ("stderr", "generic process crash"),
                    ("process_id", None),
                    ("timed_out", True),
                    ("capture_error", "pipe failed"),
                    ("limit_exceeded", "output_bytes"),
                ):
                    changed = {**result, key: replacement}
                    self.assertNotEqual(detector_result(mode, changed)["status"], "passed")
                self.assertEqual(
                    detector_result(mode, {**result, "returncode": 77})["status"], "unsupported"
                )
        wrong = capture(stderr=rows[0][2].replace("4321", "9999"), returncode=86)
        self.assertEqual(
            detector_result("heap-overread", wrong)["reason"], "asan_diagnostic_process_mismatch"
        )
        wrong = capture(stderr=rows[2][2].replace("4321", "9999"), returncode=88)
        self.assertEqual(
            detector_result("race", wrong)["reason"], "tsan_diagnostic_process_mismatch"
        )

    def test_synchronized_control_needs_clean_exit_and_exact_typed_values(self):
        doc = dict(schema="rns8_host_detector_valid_v1", value=128, expected=128)
        self.assertEqual(valid_result(capture(), doc)["status"], "passed")
        for changed in (
            None,
            {**doc, "value": 127},
            {**doc, "value": 128.0},
            {**doc, "expected": True},
        ):
            self.assertEqual(valid_result(capture(), changed)["status"], "failed")
        self.assertEqual(valid_result(capture(returncode=1), doc)["status"], "failed")


class ObjectContracts(unittest.TestCase):
    def test_native_argv_preserves_real_command_and_rejects_hidden_expansions(self):
        args = ["/opt/clang++", "-I/some path", "-o", "a.o", "-c", "source.cpp"]
        self.assertEqual(native_argv(dict(arguments=args), windows=False), args)
        self.assertEqual(
            native_argv(
                dict(command="/opt/clang++ '-I/some path' -o a.o -c source.cpp"), windows=False
            ),
            args,
        )
        for row in (
            dict(arguments=args, command="shadow"),
            dict(command="clang++ @flags.rsp"),
            dict(command="clang++ -o a.o && replace a.o"),
            dict(arguments=[]),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                native_argv(row, windows=False)

    def test_object_output_is_single_bound_and_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory).resolve()
            row = dict(directory=str(build), output="out.o")
            self.assertEqual(
                object_path(row, ["clang++", "-o", "out.o"], build, windows=False), build / "out.o"
            )
            for args in (
                ["clang++"],
                ["clang++", "-o", "out.o", "-o", "other.o"],
                ["clang++", "-o", "../escape.o"],
            ):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    object_path(row, args, build, windows=False)
            with self.assertRaises(ValueError):
                object_path(
                    {**row, "output": "other.o"}, ["clang++", "-o", "out.o"], build, windows=False
                )
            self.assertEqual(
                object_path(
                    dict(directory=str(build)), ["clang-cl", "/Foout.obj"], build, windows=True
                ),
                build / "out.obj",
            )

    def test_selected_effective_flags_do_not_accept_disabled_or_mixed_objects(self):
        fixtures = {
            "address-undefined": ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"],
            "thread": ["-fsanitize=thread"],
            "windows-address": ["/fsanitize=address"],
            "coverage": ["-fprofile-instr-generate", "-fcoverage-mapping"],
        }
        for lane, args in fixtures.items():
            with self.subTest(lane=lane):
                self.assertTrue(selected_flags(args, lane)["configured_flags_verified"])
                for bad in (
                    [],
                    [*args, "-fno-sanitize=all"],
                    [*args, "-fsanitize-ignorelist=ignore.txt"],
                    [*args, "--config=hidden.cfg"],
                    [*args, "-fsanitize=memory"],
                ):
                    with self.assertRaises(ValueError):
                        selected_flags(bad, lane)
        with self.assertRaisesRegex(ValueError, "nonrecovering"):
            selected_flags(["-fsanitize=address,undefined"], "address-undefined")
        self.assertEqual(
            symbol_classes(
                "__asan_report_load4 U\n__ubsan_handle_add_overflow_abort U\n", "address-undefined"
            ),
            ["address", "undefined"],
        )
        self.assertEqual(symbol_classes("plain_symbol T 0", "thread"), [])
        self.assertEqual(symbol_classes("__llvm_prf_cnts D 0", "coverage"), ["profile"])


class SessionControls(unittest.TestCase):
    def test_interrupted_stage_preserves_capture_and_stops_before_following_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            owned = session.Session(output, {}, 60, 1024**2)
            result = {**capture(), "interrupted": True, "returncode": -1}
            with mock.patch.object(session, "run_capture", return_value=result) as run:
                with self.assertRaises(InterruptedError):
                    owned.run("interrupted", ["never executed fixture"])
                self.assertEqual(run.call_count, 1)
            self.assertTrue(owned.stages[0]["interrupted"])

    def test_ignored_native_filter_cannot_execute_unselected_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            native = inventory("mandatory", "unexpected")
            owned = session.Session(output, {}, 60, 1024**2)
            with mock.patch.object(
                session, "run_capture", return_value=capture(json.dumps(native))
            ) as run:
                with self.assertRaisesRegex(ValueError, "refusing execution"):
                    owned.tests(
                        "scope",
                        output,
                        Path("/ctest"),
                        required=["mandatory"],
                        selection_names=["mandatory"],
                    )
                self.assertEqual(run.call_count, 2)

    def test_environment_cannot_disable_checks_or_inject_runtime(self):
        env = session.child_environment(
            {
                "ASAN_OPTIONS": "halt_on_error=0",
                "PYTHONOPTIMIZE": "2",
                "COVERAGE_PROCESS_START": "secret",
                "PATH": "/safe",
            },
            Path("/llvm/symbolizer"),
            "address-undefined",
        )
        self.assertEqual(env["PYTHONOPTIMIZE"], "")
        self.assertNotIn("COVERAGE_PROCESS_START", env)
        self.assertIn("halt_on_error=1", env["ASAN_OPTIONS"])
        for key in ("LD_PRELOAD", "CCC_OVERRIDE_OPTIONS", "CL"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                session.child_environment({key: "injected"}, Path("/llvm/symbolizer"), "thread")

    def test_plan_never_configures_runs_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "temp").mkdir()
            tool = root / "tool"
            tool.write_text("fixture", encoding="utf-8")
            out = root / "temp" / "planned"
            args = ["--preset", "cpu-tsan-release", "--out", str(out)]
            for name in ("cmake", "ctest", "nm", "symbolizer"):
                args.extend(["--" + name, str(tool)])
            with (
                mock.patch.object(session, "ROOT", root),
                mock.patch.object(session, "run_capture") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(session.main(args), 0)
                run.assert_not_called()
            value = json.loads(stdout.getvalue())
            self.assertFalse(out.exists())
            self.assertFalse(value["runtime_validation_executed"])
            self.assertIn("-DRNS8_BUILD_FUZZERS=OFF", value["configure"])

    def test_missing_and_changed_execution_evidence_never_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            executable = output / "test-binary"
            executable.write_text("unchanged", encoding="utf-8")
            native = inventory("mandatory")
            native["tests"][0]["command"] = [str(executable)]
            owned = session.Session(output, {}, 60, 1024**2)
            calls = []

            def fake(command, **kwargs):
                calls.append(command)
                if "--show-only=json-v1" in command:
                    return capture(json.dumps(native))
                (output / "app.xml").write_text(report(("mandatory", "run", "")), encoding="utf-8")
                return capture()

            with mock.patch.object(session, "run_capture", side_effect=fake):
                ref, value = owned.tests("app", output, Path("/ctest"), required=["mandatory"])
            self.assertEqual(value["status"], "passed")
            self.assertEqual(len(calls), 3)
            self.assertTrue(all("--no-tests=error" in call for call in calls))
            session.check_bookends(ref)
            executable.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                session.check_bookends(value)

    def test_owned_stage_failure_preserves_first_capture_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            owned = session.Session(output, {}, 60, 1024**2)
            failure = capture(stderr="first failure", returncode=3)
            with mock.patch.object(
                session, "run_capture", return_value=copy.deepcopy(failure)
            ) as run:
                with self.assertRaisesRegex(ValueError, "required stage failed"):
                    owned.run("compile", ["/compiler"])
                self.assertEqual(run.call_count, 1)
            retained = json.loads(Path(owned.stages[0]["receipt"]["path"]).read_text())
            self.assertEqual(retained["stderr"], "first failure")
            self.assertFalse(owned.stages[0]["succeeded"])


if __name__ == "__main__":
    unittest.main()
