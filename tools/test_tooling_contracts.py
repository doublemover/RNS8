#!/usr/bin/env python3
"""No-GPU regressions for setup, capture resume and honest suite accounting."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import autotune_paths
import build_accelerators
import golden_regression_suite as golden
import gpu_isa_report as isa
import gpu_qualification as qualification
from benchmark_request import request_identity, request_mismatch
from benchmark_sweep_lib.config import SweepCommand
from benchmark_sweep_lib.execution import execute_sweep_entries

ROOT = Path(__file__).resolve().parents[1]


class ToolingContracts(unittest.TestCase):
    def test_cache_path_table(self):
        for platform, env, expected in [
            ("posix", {"XDG_CACHE_HOME": "/work/cache", "HOME": "/home/u"}, "/work/cache/rns8-gemm/autotune.json"),
            ("posix", {"HOME": "/home/u"}, "/home/u/.cache/rns8-gemm/autotune.json"),
            ("nt", {"LOCALAPPDATA": "C:/local", "USERPROFILE": "C:/user"}, "C:/local/rns8-gemm/autotune.json"),
            ("nt", {"USERPROFILE": "C:/user"}, "C:/user/AppData/Local/rns8-gemm/autotune.json"),
            ("nt", {}, "rns8-gemm/autotune.json"),
            ("posix", {"RNS8_AUTOTUNE_CACHE_PATH": "/explicit", "XDG_CACHE_HOME": "/unused"}, "/explicit"),
        ]:
            with self.subTest(platform=platform, env=env):
                self.assertEqual(autotune_paths.default_autotune_cache_path(env, platform), Path(expected))
        import install_autotune_cache
        from benchmark_sweep_lib import execution
        with patch.dict(autotune_paths.os.environ, {"XDG_CACHE_HOME": "/work/cache"}, clear=True):
            self.assertEqual(execution.autotune_cache_path(), install_autotune_cache.default_autotune_cache_path())

    def test_configure_only_all_selected_presets_and_full_build(self):
        for options, expected, build_count in [
            (["--release", "--configure-only", "--backend", "ck"], 1, 0),
            (["--release", "--configure-only"], 4, 0),
            (["--debug", "--release", "--configure-only"], 8, 0),
            (["--release"], 4, 4),
        ]:
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")
            with self.subTest(options=options), patch.object(sys, "argv", ["build_accelerators", *options]), \
                    patch.object(build_accelerators, "command_in_msvc_environment", side_effect=lambda command: (command, False)), \
                    patch.object(build_accelerators.subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(build_accelerators.main(), 0)
                self.assertEqual(sum("--build" not in c for c in calls), expected)
                self.assertEqual(sum("--build" in c for c in calls), build_count)
                self.assertIn(f"configured={expected}, built={build_count}, failed=0", output.getvalue())

    def test_numeric_inspector_format_and_strict_admission(self):
        item = dict(inspect_target="gfx942", inspect_backend="hip-direct", inspect_device=0)
        info = dict(gcn_arch="gfx942:sramecc+", backend="hip-direct", device_id=0,
                    name="fixture only", global_mem_bytes=1024, hip_available=1)
        for available, accepted in [(1, True), (True, True), (0, False), (False, False), (2, False), ("1", False), (1.0, False)]:
            result = dict(returncode=0, timed_out=False, stdout=json.dumps(dict(info, hip_available=available)))
            with self.subTest(available=available):
                self.assertEqual(qualification.validate_result(item, result) == [], accepted)
        producer = (ROOT / "tools/rns8_inspect_selector_print.inc").read_text()
        self.assertIn('info.hip_available <<', producer)

    def test_explicit_object_inventory_without_build_tree(self):
        self.assertEqual(isa.expected_backends_from_cmake_cache(None, "all"), set())
        self.assertEqual(isa.expected_backends_from_cmake_cache(None, "direct-hip"), {"direct-hip"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CMakeCache.txt").write_text("RNS8_ENABLE_HIP:BOOL=ON\n")
            self.assertTrue(isa.expected_backends_from_cmake_cache(root, "all"))

    def test_retired_cases_do_not_execute_or_pass(self):
        retired = [c for c in golden.DEFAULT_CASES if c.retired_reason is not None]
        self.assertEqual({c.name for c in retired}, {"streaming_overlap_report", "bounded_i64_1024_review", "cache_promotion_closeout"})
        with patch.object(golden, "run_capture", side_effect=AssertionError("retired case executed")):
            results = [golden.run_case(c) for c in retired]
        with tempfile.TemporaryDirectory() as directory:
            golden.write_report(results, Path(directory))
            report = json.loads((Path(directory) / "golden-regression-report.json").read_text())
            self.assertEqual(report["summary"]["passed"], 0)
            self.assertEqual(report["summary"]["tested"], 0)
            self.assertEqual(report["summary"]["attempted"], 0)
            self.assertEqual(report["summary"]["failed"], 0)
            self.assertEqual(report["summary"]["retired"], 3)
        for case in golden.DEFAULT_CASES:
            if case.retired_reason is None:
                source = (ROOT / case.command[-1]).read_text()
                self.assertNotEqual(source.strip(), '#!/usr/bin/env python3\nprint("stub: PASS")')

    def test_resume_binds_sampling_environment_and_executable(self):
        fixture = json.loads((ROOT / "tests/fixtures/benchmark_schema/v4_finite_ring_u8_ck.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "benchmark"
            executable.write_bytes(b"fixture artifact, never executed")
            command = [str(executable), "--seed", "13", "--warmups", "1", "--repeats", "2"]
            identity = request_identity(command)
            fixture["request_identity"] = identity
            fixture["artifact_capture"] = dict(executable_sha256=identity["executable_sha256"], unchanged_during_run=True)
            output = root / "capture.json"
            output.write_text(json.dumps(fixture))
            original = output.read_bytes()
            entry = SweepCommand("name", command, output, scenario={"later_label": True})
            paths = []
            stats = execute_sweep_entries([entry], argparse.Namespace(skip_existing=True, max_new_captures=0), paths)
            self.assertEqual(stats["matched_existing_captures"], 1)
            self.assertEqual(paths, [output])
            self.assertEqual(output.read_bytes(), original)
            for option in ("--seed", "--warmups", "--repeats"):
                changed = list(command); changed[changed.index(option) + 1] = "17"
                stats = execute_sweep_entries([SweepCommand("name", changed, output)], argparse.Namespace(skip_existing=True, max_new_captures=0), [])
                self.assertEqual(stats["stale_existing_captures"], 1)
                self.assertEqual(stats["deferred_captures"], 1)
                self.assertEqual(stats["matched_existing_captures"], 0)
            self.assertIsNotNone(request_mismatch(fixture, request_identity(command, {"RNS8_EXACT_PROOF_POLICY": "eager"})))
            executable.write_bytes(b"rebuilt fixture artifact")
            self.assertIsNotNone(request_mismatch(fixture, request_identity(command)))
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
