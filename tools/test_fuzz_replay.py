"""Inert native-child controls for the explicit arithmetic replay session."""

from __future__ import annotations
from dataclasses import asdict
import contextlib
import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fuzz_replay as p
from fuzz_cases import Case, encode


class ReplayControls(unittest.TestCase):
    def test_device_tokens_do_not_silently_normalize_malformed_input(self):
        for token in ("", "+0", " 0", "0 ", "0x0", "0junk", "-2", "2147483648", "٠"):
            with self.assertRaises(argparse.ArgumentTypeError):
                p.device_number(token)
        for token in ("-1", "0", "7", "2147483647"):
            self.assertEqual(p.device_number(token), int(token))

    def setUp(self):
        root = p.ROOT / "temp"
        root.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.exe = self.root / "inert.exe"
        self.exe.write_bytes(b"inert native child; never executed")
        self.out = self.root / "results"
        self.case = Case(0, 1, 1, 1, (2,), (3,))
        self.paths = []
        for i in range(2):
            path = self.root / (str(i) + ".bin")
            path.write_bytes(encode(Case(0, 1, 1, 1, (2 + i,), (3,))))
            self.paths.append(path)
        self.identity = dict(compiled="synthetic; validator boundary mocked")

    def command(self, execute=True):
        argv = [
            "--exe",
            str(self.exe),
            "--backend",
            "hip-direct",
            "--device",
            "0",
            "--out",
            str(self.out),
        ]
        for path in self.paths:
            argv.extend(["--case", str(path)])
        return argv + (["--execute"] if execute else [])

    def native(self, command, state="passed"):
        path = Path(command[command.index("--case") + 1])
        raw, case = p.read_case(path)
        work = asdict(case)
        work.pop("a")
        work.pop("b")
        value = dict(
            schema="rns8_arithmetic_replay_v1",
            run_id=command[-1],
            process_id=123,
            case_sha256=p.hashlib.sha256(raw).hexdigest(),
            case_bytes=len(raw),
            requested_backend="hip-direct",
            requested_backend_value=2,
            requested_device=0,
            promotion_eligible=False,
            production_latency=False,
            work=work,
            state=state,
            exact_reference_passed=True,
            cleanup="checked_public_destruction_passed",
            failure=None,
            gemm_calls=2,
            full_output_comparisons=2,
            plan=dict(backend=2, kernel="synthetic kernel; not executed"),
            device=dict(hip_available=1, ordinal=0, target="gfx1100"),
        )
        if state != "passed":
            value.update(
                exact_reference_passed=False,
                cleanup="process_exit_without_further_library_calls",
                failure=dict(
                    kind="status",
                    phase="create plan" if state == "unsupported" else "GEMM",
                    actual_status=4 if state == "unsupported" else 8,
                    expected_status=0,
                ),
                gemm_calls=0,
                full_output_comparisons=0,
            )
        return value

    def process(self, command, native, code=0):
        return dict(
            command=command,
            process_id=123,
            returncode=code,
            stdout=json.dumps(native),
            stderr="",
            timed_out=False,
            interrupted=False,
            limit_exceeded=None,
            capture_error=None,
        )

    def collect(self, state="passed", mutate=None):
        calls = []

        def child(command, **kwargs):
            calls.append(command)
            if command[-1] == "--build-identity":
                return self.process(command, dict(synthetic_identity=True))
            native = self.native(command, state)
            result = self.process(
                command, native, 0 if state == "passed" else 77 if state == "unsupported" else 1
            )
            if mutate:
                mutate(command, result, native)
            return result

        with (
            patch.object(p, "run_capture", side_effect=child),
            patch.object(p, "validate_document", return_value=self.identity),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = p.main(self.command())
        return result, calls, json.loads((self.out / "results.json").read_text())

    def test_plan_never_opens_executable_or_launches_or_creates_output(self):
        self.exe.unlink()
        with (
            patch.object(p, "run_capture", side_effect=AssertionError("native process")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(p.main(self.command(False)), 0)
        self.assertFalse(self.out.exists())

    def test_pass_counts_and_all_unsupported_are_distinct(self):
        code, calls, result = self.collect()
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["counts"], {"passed": 2})
        self.assertFalse(result["hardware_qualification"])
        self.assertEqual(len(list(self.out.glob("*.bin"))), 2)
        self.out = self.root / "unsupported"
        code, calls, result = self.collect("unsupported")
        self.assertEqual(code, 77)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["counts"], {"unsupported": 2})

    def test_first_native_failure_stops_and_retains_original_bytes(self):
        code, calls, result = self.collect("failed")
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["counts"], {"failed": 1})
        self.assertEqual(result["cases"][0]["failure_signature"], ["status", "GEMM", "8", "0"])
        (retained,) = self.out.glob("*.bin")
        self.assertEqual(retained.read_bytes(), self.paths[0].read_bytes())

    def test_process_and_dispatch_substitutions_never_pass(self):
        for field, value in (
            ("process_id", 124),
            ("gemm_calls", True),
            ("full_output_comparisons", 1),
            ("requested_device", 1),
            ("case_sha256", "0" * 64),
            ("plan", dict(backend=1, kernel="cpu")),
            ("exact_reference_passed", False),
            ("cleanup", "ignored"),
        ):
            self.out = self.root / field

            def mutate(command, result, native, field=field, value=value):
                native[field] = value
                result["stdout"] = json.dumps(native)

            code, calls, result = self.collect(mutate=mutate)
            self.assertEqual(code, 1, field)
            self.assertEqual(len(calls), 2, field)
            self.assertFalse(result["accepted"], field)

    def test_earlier_retained_case_drift_is_detected(self):
        def mutate(command, result, native):
            if len(list(self.out.glob("*.bin"))) == 2:
                first = next(self.out.glob("*.bin"))
                first.write_bytes(encode(Case(0, 1, 1, 1, (7,), (8,))))

        code, calls, result = self.collect(mutate=mutate)
        self.assertEqual(code, 1)
        self.assertIn("input changed", result["error"])

    def test_timeout_and_interrupt_are_infrastructure_without_retry(self):
        for flag in ("timed_out", "interrupted"):
            self.out = self.root / flag

            def mutate(command, result, native, flag=flag):
                result[flag] = True

            code, calls, result = self.collect(mutate=mutate)
            self.assertEqual(code, 130 if flag == "interrupted" else 1)
            self.assertEqual(len(calls), 2)
            self.assertFalse(result["accepted"])

    def test_interrupt_survives_failed_process_receipt_write(self):
        original = p.write

        def write(path, value):
            if path.name.endswith("-process.json") and value.get("interrupted"):
                raise OSError("inert output failure")
            return original(path, value)

        def mutate(command, result, native):
            result["interrupted"] = True

        with patch.object(p, "write", side_effect=write):
            code, calls, result = self.collect(mutate=mutate)
        self.assertEqual(code, 130)
        self.assertEqual(len(calls), 2)
        self.assertTrue(result["interrupted"])


if __name__ == "__main__":
    unittest.main()
