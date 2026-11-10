"""Authored branch/process/source controls; no coverage or project subprocess runs."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from host_instrumentation_session import invocation_path, reference
from python_coverage_contract import (
    branch_report,
    invariant_observations,
    marker_lines,
    process_control,
    raw_data,
)
import python_coverage_session as session


def captured(pid):
    return dict(
        returncode=0,
        timed_out=False,
        interrupted=False,
        limit_exceeded=None,
        capture_error=None,
        process_id=pid,
    )


def native_report(path):
    return dict(
        meta=dict(format=3, version="7.17.2", branch_coverage=True),
        files={
            str(path): dict(
                summary=dict(
                    covered_lines=3,
                    num_statements=4,
                    missing_lines=1,
                    excluded_lines=0,
                    num_branches=2,
                    covered_branches=1,
                    missing_branches=1,
                ),
                executed_lines=[1, 2, 3],
                missing_lines=[4],
                excluded_lines=[],
                executed_branches=[[2, 3]],
                missing_branches=[[2, 4]],
            )
        },
    )


class PythonCoverageControls(unittest.TestCase):
    def test_native_branch_report_retains_missing_outcomes_and_explicit_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "tools/production.py"
            path.parent.mkdir()
            path.write_text(
                "def production(value):\n    if value:\n        return 1\n    return 0\n",
                encoding="utf-8",
            )
            sources = [reference(path)]
            native = native_report(path)
            value = branch_report(native, root=root, sources=sources, version="7.17.2")
            self.assertEqual(
                value["totals"],
                dict(statements=4, covered_statements=3, branches=2, covered_branches=1),
            )
            self.assertEqual(value["files"][0]["missing_branches"], [[2, 4]])
            self.assertIsNone(value["reviewed_percentage_threshold"])
            for edit in (
                lambda d: d["meta"].update(branch_coverage=False),
                lambda d: d["meta"].update(version="other"),
                lambda d: d["files"].clear(),
                lambda d: d["files"][str(path)]["summary"].update(covered_lines=True),
                lambda d: d["files"][str(path)]["summary"].update(covered_branches=2),
                lambda d: d["files"][str(path)].update(missing_lines=[]),
                lambda d: d["files"][str(path)].update(missing_branches=[[2, 3]]),
            ):
                changed = copy.deepcopy(native)
                edit(changed)
                with self.assertRaises(ValueError):
                    branch_report(changed, root=root, sources=sources, version="7.17.2")
            path.write_text("changed source", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source changed"):
                branch_report(native, root=root, sources=sources, version="7.17.2")

    def test_branch_observation_requires_body_under_the_actual_required_test_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "production.py"
            path.write_text(
                "def gate(value):\n    if value:\n        return 1\n    return 0\n",
                encoding="utf-8",
            )
            invariant = dict(
                name="gate",
                source="production.py",
                function="gate",
                test_id="test_gate.Case.test_reject",
            )
            native = native_report(path)
            row = native["files"][str(path)]
            row["functions"] = {
                "gate": dict(
                    start_line=1,
                    executed_lines=[1, 2, 3],
                    summary=dict(covered_branches=1),
                    missing_branches=[[2, 4]],
                )
            }
            row["contexts"] = {"2": ["lane|test_gate.Case.test_reject"]}
            self.assertEqual(
                invariant_observations(native, [invariant], root=root)[0]["observed_body_lines"],
                [2],
            )
            for changed in (
                {"1": ["lane|test_gate.Case.test_reject"]},
                {"2": ["lane|test_other.Case.test_reject"]},
                {},
            ):
                row["contexts"] = changed
                with self.assertRaisesRegex(ValueError, "did not observe"):
                    invariant_observations(native, [invariant], root=root)

    def test_actual_parent_child_body_sites_and_profile_hashes_are_mandatory(self):
        parent = captured(4321)
        child_capture = captured(4322)
        document = dict(
            schema="rns8_python_coverage_process_control_v1",
            process_id=4321,
            value=23,
            child=dict(role="child", process_id=4322, value=37),
            child_process=child_capture,
        )
        profiles = [
            dict(path="parent", sha256="a" * 64, process_id=4321),
            dict(path="child", sha256="b" * 64, process_id=4322),
        ]
        inspected = dict(
            schema="rns8_python_coverage_data_validation_v1",
            files=[
                dict(
                    **row,
                    has_arcs=True,
                    marker_lines_executed=[10 if row["process_id"] == 4321 else 20],
                )
                for row in profiles
            ],
        )
        markers = dict(parent_path=10, child_path=20)
        self.assertEqual(
            process_control(parent, document, profiles, inspected, markers)["status"], "passed"
        )
        for change in ("missing_child", "untraced_body", "different_hash", "nonbranch"):
            mutated = copy.deepcopy(inspected)
            if change == "missing_child":
                mutated["files"].pop()
            elif change == "untraced_body":
                mutated["files"][1]["marker_lines_executed"] = [1, 2]
            elif change == "different_hash":
                mutated["files"][1]["sha256"] = "c" * 64
            else:
                mutated["files"][1]["has_arcs"] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                process_control(parent, document, profiles, mutated, markers)

    def test_native_profile_filenames_empty_data_and_real_source_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(ValueError):
                raw_data(root)
            profile = root / ".coverage.host.4321.Xfixture"
            profile.write_bytes(b"fixture native data never read by coverage")
            self.assertEqual(raw_data(root)[0]["process_id"], 4321)
            profile.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                raw_data(root)
            profile.rename(root / "foreign-data")
            with self.assertRaisesRegex(ValueError, "filenames"):
                raw_data(root)
        markers = marker_lines(
            Path(__file__).resolve().parents[1] / "tests/controls/python_coverage_process.py"
        )
        self.assertGreater(markers["child_path"], markers["parent_path"])

    def test_invocation_path_preserves_driver_and_virtualenv_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            logical = root / "python"
            target = root / "base-python"
            logical.write_bytes(b"logical fixture")
            target.write_bytes(b"target fixture")
            with mock.patch.object(Path, "resolve", return_value=target):
                self.assertEqual(invocation_path(logical), logical)
                self.assertEqual(reference(logical)["path"], str(target))

    def test_plan_is_inert_and_does_not_import_or_execute_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "temp").mkdir()
            (root / "metadata").mkdir()
            interpreter = root / "python"
            interpreter.write_bytes(b"never executable fixture")
            deps = root / "metadata/dependencies.json"
            deps.write_text(
                json.dumps(dict(components=[dict(name="python", selected=dict(version="3.15.2"))]))
            )
            policy = root / "policy.json"
            policy.write_text("{}")
            selected = dict(test_modules=["test_gate"], production_modules=["tools/gate.py"])
            lock = dict(coverage=dict(wheels=[dict(version="7.17.2")]))
            with (
                mock.patch.object(session, "ROOT", root),
                mock.patch.object(session, "POLICY", policy),
                mock.patch.object(session, "policy", return_value=selected),
                mock.patch.object(session, "load_quality_lock", return_value=lock),
                mock.patch.object(session.Session, "run") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(
                    session.main(
                        ["--python", str(interpreter), "--out", str(root / "temp/planned")]
                    ),
                    0,
                )
                run.assert_not_called()
            plan = json.loads(stdout.getvalue())
            self.assertFalse(plan["runtime_validation_executed"])
            self.assertFalse((root / "temp/planned").exists())


if __name__ == "__main__":
    unittest.main()
