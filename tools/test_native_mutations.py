"""Authored native source/receipt controls; never configure, compile or run cases."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from xml.sax.saxutils import escape

from analysis_inputs import target_artifacts
from build_manifest import source_content_digest
from execution_inventory import reconcile
from native_mutation_contract import assertion_result, change, copy_sources, tracked_inputs
import native_mutation_session as session
from test_execution_inventory import inventory, report


def indexed(*names):
    return "".join("100644 " + "e" * 40 + " 0\t" + name + "\0" for name in names)


class NativeMutationControls(unittest.TestCase):
    def test_tracked_copy_excludes_gitlinks_preserves_sources_and_rejects_unsafe_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "original"
            (root / "tools").mkdir(parents=True)
            (root / "CMakeLists.txt").write_text("fixture never configured")
            (root / "tools/build_manifest.py").write_text("fixture never imported")
            records = (
                indexed("CMakeLists.txt", "tools/build_manifest.py")
                + "160000 "
                + "f" * 40
                + " 0\tthird_party/external\0"
            )
            value = tracked_inputs(root, records, max_bytes=1024)
            self.assertEqual(len(value["files"]), 2)
            self.assertEqual(len(value["excluded"]), 1)
            destination = root.parent / "snapshot"
            copy_sources(value, destination)
            self.assertEqual(
                (destination / "CMakeLists.txt").read_bytes(),
                (root / "CMakeLists.txt").read_bytes(),
            )
            self.assertFalse((destination / "third_party/external").exists())
            with self.assertRaises(ValueError):
                copy_sources(value, destination)
            for altered in (
                records[:-1],
                records + indexed("CMakeLists.txt"),
                indexed("../escape"),
                records.replace("100644", "120000", 1),
                records.replace(" 0\t", " 1\t", 1),
            ):
                with self.assertRaises(ValueError):
                    tracked_inputs(root, altered, max_bytes=1024)
            with self.assertRaises(ValueError):
                tracked_inputs(root, records, max_bytes=1)

    def test_cpp_operation_changes_only_intended_bytes_and_preserves_crlf(self):
        source = b"bool gate() {\r\n  if (value)\r\n    return false;\r\n  return true;\r\n}\r\n"
        seed = dict(
            source_sha256=hashlib.sha256(source).hexdigest(),
            anchor="bool gate() {",
            before="if (value)\n    return false;",
            after="if (!value)\n    return false;",
        )
        mutated = change(source, seed)
        self.assertEqual(mutated, source.replace(b"if (value)", b"if (!value)"))
        self.assertEqual(source.count(b"\r\n"), mutated.count(b"\r\n"))
        for changed in (
            {**seed, "source_sha256": "a" * 64},
            {**seed, "anchor": "absent"},
            {**seed, "before": "missing"},
        ):
            with self.assertRaises(ValueError):
                change(source, changed)

    def test_shared_source_digest_retains_legacy_content_format_and_untracked_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "CMakeLists.txt").write_bytes(b"project(fixture)\n")
            (root / "src/a.cpp").write_bytes(b"int a=1;\n")
            expected = hashlib.sha256(
                b"CMakeLists.txt\0project(fixture)\n\0src/a.cpp\0int a=1;\n\0"
            ).hexdigest()
            self.assertEqual(source_content_digest(root), expected)
            (root / "src/ignored.pyc").write_bytes(b"ignored")
            self.assertEqual(source_content_digest(root), expected)
            (root / "src/untracked.inc").write_bytes(b"new input")
            self.assertNotEqual(source_content_digest(root), expected)
            nested = root / "__pycache__/project"
            (nested / "src").mkdir(parents=True)
            (nested / "CMakeLists.txt").write_bytes(b"project(fixture)\n")
            (nested / "src/a.cpp").write_bytes(b"int a=1;\n")
            self.assertEqual(source_content_digest(nested), expected)
            (nested / "src/a.cpp").write_bytes(b"changed source")
            self.assertNotEqual(source_content_digest(nested), expected)

    def test_only_normal_required_catch_assertion_counts_as_native_mutation_kill(self):
        seed = dict(
            tests=[
                dict(
                    name="required native case",
                    source="tests/unit/test_gate.cpp",
                    assertion_marker="value == expected",
                )
            ]
        )
        full = inventory("required native case")
        failed = report(("required native case", "fail", "Failed")).replace(
            "<system-out/>",
            "<system-out>"
            + escape("test_gate.cpp:17 FAILED: CHECK(value == expected)")
            + "</system-out>",
        )
        value = reconcile(full, full, failed, required=["required native case"])
        value.update(
            ctest_process_succeeded=False,
            process=dict(
                process_id=4321,
                returncode=8,
                timed_out=False,
                interrupted=False,
                limit_exceeded=None,
                capture_error=None,
            ),
        )
        self.assertEqual(assertion_result(seed, value, failed), "killed_by_assertion")
        for altered in (
            failed.replace("value == expected", "other assertion"),
            failed.replace("test_gate.cpp", "unrelated.cpp"),
        ):
            self.assertEqual(assertion_result(seed, value, altered), "unintended_assertion_failure")
        self.assertEqual(
            assertion_result(seed, value, failed.replace('message="Failed"', 'message="SegFault"')),
            "infrastructure_error",
        )
        timed = copy.deepcopy(value)
        timed["process"]["timed_out"] = True
        self.assertEqual(assertion_result(seed, timed, failed), "timeout")
        passed = report(("required native case", "run", ""))
        positive = reconcile(full, full, passed, required=["required native case"])
        positive.update(ctest_process_succeeded=True, process={**value["process"], "returncode": 0})
        self.assertEqual(assertion_result(seed, positive, passed), "survived")

    def test_target_artifacts_reject_duplicates_wrong_configuration_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "native-test"
            row = dict(
                name="rns8_tests", type="EXECUTABLE", configuration="Debug", artifact=str(path)
            )
            value = dict(schema="rns8_first_party_analysis_targets_v1", targets=[row])
            self.assertEqual(target_artifacts(value, "Debug")["rns8_tests"], (path, "EXECUTABLE"))
            for changed in (
                {**value, "targets": [row, row]},
                {**value, "targets": [{**row, "configuration": "Release"}]},
                {**value, "targets": [{**row, "artifact": "relative"}]},
                {**value, "targets": [{**row, "type": []}]},
            ):
                with self.assertRaises(ValueError):
                    target_artifacts(changed, "Debug")

    def test_native_plan_is_inert_and_disables_all_gpu_transport_instrumentation_lanes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "temp").mkdir()
            executable = root / "tool"
            executable.write_bytes(b"never executable fixture")
            args = ["--out", str(root / "temp/planned")]
            for key in ("cmake", "ctest", "compiler", "ninja", "python"):
                args.extend(["--" + key, str(executable)])
            seeds = [dict(id="source-fixture")]
            with (
                mock.patch.object(session, "ROOT", root),
                mock.patch.object(session, "registry", return_value=seeds),
                mock.patch.object(session.platform, "system", return_value="Linux"),
                mock.patch.object(session.Session, "run") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(session.main(args), 0)
                run.assert_not_called()
            value = json.loads(stdout.getvalue())
            self.assertFalse((root / "temp/planned").exists())
            self.assertFalse(value["runtime_validation_executed"])
            self.assertEqual(value["baseline_builds"], 1)
            self.assertIn("-DRNS8_ENABLE_HIP=OFF", value["configure_example"])
            self.assertIn("-DRNS8_ENABLE_DISTRIBUTED_MPI=OFF", value["configure_example"])
            with (
                mock.patch.object(session, "ROOT", root),
                mock.patch.object(session, "registry", return_value=seeds),
                mock.patch.object(session.platform, "system", return_value="Linux"),
                mock.patch.object(session.platform, "release", return_value="native-fixture"),
                mock.patch.object(session.Session, "run") as run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.assertEqual(
                    session.main(
                        [*args, "--mull-runner", str(executable), "--mull-plugin", str(executable)]
                    ),
                    0,
                )
                run.assert_not_called()
            self.assertTrue(json.loads(stdout.getvalue())["optional_mull"]["requested"])
            self.assertEqual(json.loads(stdout.getvalue())["baseline_builds"], 2)
            self.assertFalse((root / "temp/planned").exists())


if __name__ == "__main__":
    unittest.main()
