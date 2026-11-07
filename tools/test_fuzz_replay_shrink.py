"""Inert session-boundary controls for separately requested native minimization."""

from __future__ import annotations
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_request import file_digest
from fuzz_cases import Case, encode
import fuzz_replay_shrink as p


class NativeMinimizerControls(unittest.TestCase):
    def setUp(self):
        directory = p.replay.ROOT / "temp"
        directory.mkdir(exist_ok=True)
        scratch = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.case = self.root / "original.bin"
        self.case.write_bytes(encode(Case(0, 2, 1, 2, (0, 0, 0, 73), (1, 1))))
        self.exe = self.root / "inert.exe"
        self.exe.write_bytes(b"inert; never native execution")
        self.out = self.root / "minimized"
        self.signature = ["value_or_padding", "export", "0", "0"]
        self.calls = []

    def command(self, execute=True):
        return [
            "--exe",
            str(self.exe),
            "--case",
            str(self.case),
            "--backend",
            "cpu-reference",
            "--device",
            "-1",
            "--out",
            str(self.out),
            "--signature",
            *self.signature,
            "--replays",
            "32",
        ] + (["--execute"] if execute else [])

    def native(self, args, plan, inputs):
        self.calls.append(args)
        path, raw, case = inputs[0]
        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(args.artifact_root, self.out.resolve())
        args.out.mkdir()
        failed = 73 in case.a
        result = dict(
            executable_sha256=file_digest(args.exe),
            input_receipts=[dict(path=str(path), sha256=file_digest(path), bytes=len(raw))],
            cases=[
                dict(
                    state="failed" if failed else "passed",
                    failure_signature=self.signature if failed else [],
                )
            ],
            error="native arithmetic failure" if failed else None,
            interrupted=False,
        )
        p.replay.write(args.out / "results.json", result)
        return 1 if failed else 0

    def test_plan_never_starts_native_session_or_creates_output(self):
        with (
            patch.object(p.replay, "execute", side_effect=AssertionError("native")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(p.main(self.command(False)), 0)
        self.assertFalse(self.out.exists())

    def test_only_arithmetic_failures_can_request_replays(self):
        for signature in (
            ("status", "GEMM", "8", "0"),
            ("status", "destroy matrix", "1", "0"),
            ("backend_identity", "plan identity", "0", "0"),
            ("status", "GEMM", "13", "0"),
        ):
            self.assertFalse(p.admissible_failure(signature))
        self.assertTrue(p.admissible_failure(tuple(self.signature)))

    def test_native_binding_is_reused_for_every_admitted_reduction(self):
        with (
            patch.object(p.replay, "execute", side_effect=self.native),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(p.main(self.command()), 0)
        raw, case = p.replay.read_case(self.out / "minimized.bin")
        self.assertEqual((case.m, case.n, case.k), (1, 1, 1))
        self.assertEqual(case.a, (73,))
        self.assertLessEqual(len(self.calls), 32)
        self.assertEqual((self.out / "original.bin").read_bytes(), self.case.read_bytes())

    def test_different_original_failure_stops_before_reductions(self):
        def native(args, plan, inputs):
            code = self.native(args, plan, inputs)
            path = args.out / "results.json"
            result = p.parse_evidence_json(path.read_text())
            result["cases"][0]["failure_signature"] = ["status", "GEMM", "5", "0"]
            path.write_text(json.dumps(result))
            return code

        with (
            patch.object(p.replay, "execute", side_effect=native),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(p.main(self.command()), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.out / "minimized.bin").exists())


if __name__ == "__main__":
    unittest.main()
