"""Authored gate controls; fake checker bytes are never launched."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from quality_inventory import ROOT
from typecheck import checked_scope, run_checks


class TypecheckGate(unittest.TestCase):
    def test_scope_is_nonempty_strict_and_excludes_only_named_negative_inputs(self):
        scope = checked_scope()
        self.assertEqual(len(scope["strict_modules"]), 5)
        self.assertEqual(set(scope["negative_inputs"].values()), {"arg-type", "typeddict-item", "name-defined"})
        self.assertIn("tools/execution_observation.py", scope["strict_modules"])

    def fake_runner(self, command, **kwargs):
        label = kwargs["label"]
        self.assertNotIn("MYPYPATH", kwargs["env"])
        output, returned = "", 0
        if label == "mypy-version":
            output = "mypy 2.4.1 (authored checker output, not executed)"
        elif label.startswith("mypy-negative-"):
            source = command[-1]
            code = checked_scope()["negative_inputs"][source]
            output, returned = source + ":1: error: authored rejection [" + code + "]", 1
        return dict(command=command, process_id=123, returncode=returned, timed_out=False,
                    stdout=output, stderr="", elapsed_seconds=0.01)

    def checker(self, directory):
        path = Path(directory) / ("mypy.exe" if os.name == "nt" else "mypy")
        path.write_bytes(b"AUTHORED CHECKER NEVER EXECUTED")
        return path.parent

    def test_gate_requires_expected_negative_diagnostics_and_reports_remaining_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_checks(tool_bin=self.checker(temporary), self_test=True, runner=self.fake_runner)
        self.assertEqual(len(result["negative_controls"]), 3)
        self.assertEqual(result["explicit_strict_module_count"], 5)
        self.assertGreater(result["outside_explicit_strict_scope_count"], 0)
        self.assertTrue(result["remaining_migration"])
        self.assertIn("not_all_python", result["scope"])

    def test_wrong_checker_clean_negative_or_wrong_error_cannot_pass(self):
        for mutation in ("version", "no_error", "wrong_error", "truncated"):
            def runner(command, *, _mutation=mutation, **kwargs):
                value = self.fake_runner(command, **kwargs)
                if _mutation == "version" and kwargs["label"] == "mypy-version":
                    value["stdout"] = "mypy 0.0.0"
                elif kwargs["label"].startswith("mypy-negative-"):
                    if _mutation == "no_error":
                        value.update(returncode=0, stdout="")
                    elif _mutation == "wrong_error":
                        value["stdout"] = "unrelated.py:1: error: wrong input [name-defined]"
                    elif _mutation == "truncated":
                        value["limit_exceeded"] = "output"
                return value
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    run_checks(tool_bin=self.checker(temporary), self_test=True, runner=runner)

    def test_blanket_import_skip_or_empty_scope_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("mypy.ini", "metadata/python-typecheck-scope.json", "metadata/first-party-inventory.json"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, path)
            path = root / "mypy.ini"
            text = path.read_text().replace("[mypy]\n", "[mypy]\nignore_missing_imports = True\n")
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "blanket import"):
                checked_scope(root)
            shutil.copyfile(ROOT / "mypy.ini", path)
            scope_path = root / "metadata/python-typecheck-scope.json"
            scope = json.loads(scope_path.read_text())
            scope["strict_modules"] = []
            scope_path.write_text(json.dumps(scope))
            with self.assertRaisesRegex(ValueError, "nonempty"):
                checked_scope(root)


if __name__ == "__main__":
    unittest.main()
