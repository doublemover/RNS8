"""Authored Git/CI selection controls; no workflow, test batch or Git child runs."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ci_change_scope as scope


class SourceScope(unittest.TestCase):
    def test_docs_skip_native_checks_but_unknown_source_and_deleted_code_do_not(self):
        for paths in (
            [],
            ["README.md", "CONTRIBUTING.md"],
            ["docs/guide.md", "docs/documentation-inventory.json"],
            ["src/backend_cpu_reference/README.md"],
        ):
            self.assertFalse(scope.requires_source_checks(paths))
        for path in (
            "src/core/api.cpp",
            "tools/reader.py",
            "metadata/policy.json",
            "CMakeLists.txt",
            "docs/new-code.cpp",
            ".github/workflows/ci.yml",
            "tests/fixtures/data.md",
            "new-unknown-file",
        ):
            self.assertTrue(scope.requires_source_checks([path]))
        self.assertTrue(scope.requires_source_checks(["src/deleted.cpp", "docs/renamed.md"]))
        for paths in (False, [None], ["../outside"], ["/absolute"], ["bad\0path"], ["a\\b"]):
            with self.assertRaises(ValueError):
                scope.requires_source_checks(paths)

    def test_real_pr_tree_command_preserves_deletions_and_requires_complete_native_inventory(self):
        for stdout, code in (
            ("docs/guide.md\0", 0),
            ("src/deleted.cpp\0docs/renamed.md\0", 0),
            ("truncated", 2),
        ):
            with self.subTest(stdout=stdout), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "github-output"
                capture = dict(
                    stdout=stdout,
                    returncode=0,
                    timed_out=False,
                    interrupted=False,
                    limit_exceeded=None,
                    capture_error=None,
                )
                with (
                    patch.object(scope, "run_capture", return_value=capture) as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(
                        scope.main(
                            [
                                "--event",
                                "pull_request",
                                "--base-sha",
                                "a" * 40,
                                "--github-output",
                                str(output),
                            ]
                        ),
                        code,
                    )
                command = run.call_args.args[0]
                for flag in ("--no-renames", "--no-ext-diff", "--no-textconv", "-z"):
                    self.assertIn(flag, command)
                if code:
                    self.assertFalse(output.exists())
                else:
                    self.assertEqual(
                        output.read_text(),
                        "source_required="
                        + str(stdout.startswith("src/")).lower()
                        + "\n"
                        + "analysis_changed_paths="
                        + scope.analysis_changed_paths(stdout[:-1].split("\0"))
                        + "\n",
                    )

    def test_explicit_and_periodic_selection_need_no_git_child(self):
        for event in ("workflow_dispatch", "schedule"):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "github-output"
                with (
                    patch.object(scope, "run_capture") as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(
                        scope.main(["--event", event, "--github-output", str(output)]), 0
                    )
                    run.assert_not_called()
                self.assertEqual(
                    output.read_text(), "source_required=true\nanalysis_changed_paths=null\n"
                )

    def test_json_transport_cannot_inject_outputs_or_truncate_change_selection(self):
        paths = ['src/quotes"\nline.cpp', "src/λ.cpp"]
        encoded = scope.analysis_changed_paths(paths)
        self.assertNotIn("\n", encoded)
        self.assertEqual(json.loads(encoded), paths)
        self.assertEqual(scope.analysis_changed_paths(["src/large.cpp"] * 1025), "null")
        self.assertEqual(scope.analysis_changed_paths(["src/" + "x" * 20000]), "null")


if __name__ == "__main__":
    unittest.main()
