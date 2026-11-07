"""Owned offline symbol runner controls; all native processes are replaced."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import windows_cpu_symbols as symbols
from json_evidence import load_evidence_json


class SymbolRunner(unittest.TestCase):
    def test_review_uses_remaining_deadline_and_failure_prevents_PDB_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "symbols"
            plan = dict(
                output_directory=str(out),
                acquisition_plan=dict(path="capture/plan.json"),
                budgets=dict(
                    total_seconds=120,
                    max_artifact_bytes=128 * 1024**2,
                    minimum_free_bytes=256 * 1024**2,
                    stage_seconds=15,
                ),
            )
            observed = []

            def runner(command, **options):
                observed.append((command, options["timeout_seconds"]))
                raise ValueError("controlled incomplete retained-file decode")

            def review(path, *, runner):
                runner(["bound-etl-helper", "--input", "retained.etl"], timeout_seconds=120)
                self.fail("incomplete native review continued")

            with (
                patch.object(symbols.platform, "system", return_value="Windows"),
                patch.object(symbols, "rebuild_symbols", return_value=plan),
                patch.object(symbols, "review_capture", side_effect=review),
            ):
                result = symbols.execute_symbols(plan, runner=runner, clock=iter([0, 113]).__next__)
            self.assertEqual(result, 1)
            self.assertEqual(observed, [(["bound-etl-helper", "--input", "retained.etl"], 7)])
            failure = load_evidence_json(out / "failure.json")
            self.assertIn("incomplete retained-file", failure["error"])
            self.assertFalse(failure["promotion_allowed"])
            self.assertFalse(failure["stages"])

    def test_plan_only_CLI_never_calls_any_native_runner(self):
        from contextlib import redirect_stdout
        import io

        with (
            patch.object(symbols, "plan_symbols", return_value={"qualification": "plan_only"}),
            patch.object(symbols, "execute_symbols") as execute,
            patch.object(symbols, "run_capture") as native,
            patch.object(symbols, "review_capture") as review,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                symbols.main(
                    [
                        "--acquisition-plan",
                        "capture/plan.json",
                        "--symbolizer",
                        "helper.exe",
                        "--dbghelp",
                        "dbghelp.dll",
                        "--module",
                        "app.exe",
                        "app.pdb",
                        "app.exe",
                        "--out",
                        "temp/symbols",
                    ]
                ),
                0,
            )
        execute.assert_not_called()
        native.assert_not_called()
        review.assert_not_called()


if __name__ == "__main__":
    unittest.main()
