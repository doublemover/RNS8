"""CPU-only authored CLI/plan integration: no benchmark executable required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import ci_benchmark_smoke as smoke
from benchmark_sweep_lib.cli import parse_args
from benchmark_sweep_lib.commands import sweep_command_entries


class SmokeWorkflowContract(unittest.TestCase):
    def test_real_parser_and_command_matrix(self):
        args = parse_args(smoke.sweep_arguments(Path("missing-bench.exe"), Path("temp/smoke")))
        entries = sweep_command_entries(args)
        self.assertEqual(len(entries), 6)
        pairs = {(e.command[e.command.index("--backend")+1], e.command[e.command.index("--semantics")+1]) for e in entries}
        self.assertEqual(pairs, {(b,s) for b in ("cpu","hip-direct") for s in ("bounded-i64","bounded-u64","finite-u8-field")})
        self.assertEqual(len({str(e.output) for e in entries}), 6)
        self.assertFalse(args.write_autotune_cache)
        self.assertEqual(args.review_mode, "smoke")
        self.assertFalse(args.skip_existing)
        self.assertEqual(args.capture_timeout_seconds, 180)

    def test_plan_never_launches_or_requires_executable(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(smoke, "execute_sweep_entries") as execute:
            self.assertEqual(smoke.main(["--bench", "missing-bench.exe", "--out-root", directory, "--plan-only"]), 0)
            execute.assert_not_called()
            plan = json.loads((Path(directory)/"smoke-plan.json").read_text())
            self.assertEqual(len(plan["commands"]), 6)
            self.assertEqual(plan["policy"], smoke.POLICY)

    def test_incomplete_collection_cannot_report_success(self):
        args = parse_args(smoke.sweep_arguments(Path("missing"), Path("temp/smoke")))
        entries = sweep_command_entries(args)
        with patch.object(smoke, "validate_paths", return_value=[]):
            result = smoke.collection_result(entries, [], {"new_captures_completed":0})
        self.assertFalse(result["complete"])
        self.assertFalse(result["performance_regression_checked"])
        self.assertEqual(len(result["missing_captures"]), 6)

    def test_workflow_uses_shared_entrypoint_and_no_fake_baseline(self):
        text = (Path(__file__).resolve().parents[1]/".github/workflows/perf-regression.yml").read_text()
        self.assertIn("Non-promoting Benchmark Smoke", text)
        self.assertIn("tools/ci_benchmark_smoke.py", text)
        self.assertIn("windows-msvc-hip-release", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertNotIn("result_compare.py", text)
        self.assertNotIn("--max-regression", text)
        self.assertNotIn("--scenario bounded-", text)


if __name__ == "__main__":
    unittest.main()
