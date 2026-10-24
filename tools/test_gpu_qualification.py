#!/usr/bin/env python3
"""No-GPU fixtures for explicit session selection, bounded execution and receipts."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gpu_qualification as q
import session_environment as env
from process_progress import run_capture


def inspection(device=0, **changes):
    return {"gcn_arch": "gfx942:sramecc+:xnack-", "backend": "hip-direct", "device_id": device,
            "name": "fixture device", "hip_available": True, "global_mem_bytes": 1024,
            "hip_runtime_version": 70100000, "hip_driver_version": 70100000, **changes}


def result(stdout="", **changes):
    return {"returncode": 0, "timed_out": False, "stdout": stdout, "stderr": "", "elapsed_seconds": 0.01, **changes}


class QualificationTests(unittest.TestCase):
    def test_cpu_preparation_has_no_device_or_network_stage(self):
        plan = q.command_plan("mi300x", [], Path("temp/preparation"), 3, [], session_kind="prepare")
        self.assertEqual([item["name"] for item in plan], ["prepare-configure", "prepare-build", "prepare-cpu-tests"])
        self.assertIn("-DRNS8_ENABLE_HIP=OFF", plan[0]["argv"])
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", plan[0]["argv"])
        self.assertIn("-DVCPKG_MANIFEST_INSTALL=OFF", plan[0]["argv"])
        self.assertIn("linux-cpu-release", plan[0]["argv"])
        for item in plan:
            self.assertNotIn("device", item)
            self.assertNotIn("inspect_target", item)
            self.assertNotIn("capture", item)
        request = q.dependency_request("mi300x", [], [], preparation=True)
        self.assertFalse(request.require_hip)
        self.assertFalse(any(request.features.values()))
        for backends, devices in [(["hip-direct"], []), ([], [0])]:
            with self.assertRaises(ValueError):
                q.command_plan("mi300x", backends, Path("temp/preparation"), 3, devices, session_kind="prepare")
        for extra in (["--devices", "0"], ["--backend", "hip-direct"], ["--accelerators"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                q.main(["--target", "mi300x", "--session-kind", "prepare", *extra])
            self.assertEqual(error.exception.code, 2)

    def test_cpu_preparation_completion_never_claims_gpu_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = self.manifest(out)
            manifest["session_kind"] = "prepare"
            self.assertEqual(q.execute_plan(manifest, out, {}, runner=lambda *a, **kw: result()), 0)
            receipt = json.loads((out / "results.json").read_text())
            self.assertEqual(receipt["status"], "cpu_preparation_passed_hardware_pending")
            self.assertFalse(receipt["hardware_qualified"])
            self.assertEqual(receipt["observed_parent_runtime_devices"], [])

    def test_cpu_preparation_report_requires_actual_passed_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "cpu.xml"
            item = {"preparation_junit": str(report)}
            for xml, success in [
                ('<testsuite/>', False),
                ('<testsuite><testcase name="only"><skipped/></testcase></testsuite>', False),
                ('<testsuite><testcase name="failed"><failure/></testcase></testsuite>', False),
                ('<testsuite><testcase name="passed"/><testcase name="gpu"><skipped/></testcase></testsuite>', True),
            ]:
                report.write_text(xml, encoding="utf-8")
                self.assertEqual(not q.validate_result(item, result()), success)
            report.unlink()
            self.assertTrue(q.validate_result(item, result()))

    def test_explicit_device_parser(self):
        self.assertEqual(q.device_list("4,7,0"), [4, 7, 0])
        for value in ("", "all", "0,", "-1", "0,0", "00,0", "0 1", "0, 1", "65536", "0;1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                q.device_list(value)
        with self.assertRaises(argparse.ArgumentTypeError):
            q.device_list(",".join(map(str, range(65))))

    def test_no_implicit_device_or_architecture(self):
        for argv in ([], ["--target", "mi300x"], ["--devices", "0"],
                     ["--target", "mi300x", "--devices", "0", "--total-seconds", "nan"],
                     ["--target", "mi300x", "--devices", "0", "--execute", "--dry-run"],
                     ["--target", "mi300x", "--devices", "0", "--resume"],
                     ["--target", "mi300x", "--devices", "0", "--skip-build"],
                     ["--target", "mi300x", "--devices", "0", "--rank-scenarios", "unbounded"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                q.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_all_lanes_have_ordered_device_checks_and_no_install(self):
        with tempfile.TemporaryDirectory(prefix="session space ") as directory:
            out = Path(directory)
            for target in q.TARGETS:
                plan = q.command_plan(target, list(q.BACKENDS), out, 2, [4, 7], timeout=123)
                self.assertEqual(len({item["name"] for item in plan}), len(plan))
                self.assertEqual(sum(item.get("parent_inspect", False) for item in plan), 2 * len(q.BACKENDS))
                self.assertEqual(sum("match_parent" in item for item in plan), 2 * len(q.BACKENDS))
                for index, item in enumerate(plan):
                    self.assertEqual(item["depends_on"], [plan[index - 1]["name"]] if index else [])
                    self.assertEqual(item["timeout_seconds"], 123)
                    self.assertTrue(all(type(arg) is str for arg in item["argv"]))
                    self.assertFalse(any(word in item["argv"] for word in ("pip", "install", "clone", "--write-autotune-cache")))
                    if "match_parent" in item:
                        self.assertEqual(item["inspect_device"], 0)
                        self.assertIn(item["selected_device"], [4, 7])
                    if item["name"].endswith("-configure"):
                        self.assertIn("-DVCPKG_MANIFEST_INSTALL=OFF", item["argv"])
                self.assertTrue(any("hip-direct-graph" in item["name"] for item in plan))
                self.assertEqual(q.command_plan(target, ["hip-direct"], out, 1, [7], session_kind="environment"), [])
            smoke = q.command_plan("mi300x", ["hip-direct"], out, 1, [7], session_kind="smoke")
            self.assertFalse(any("junit" in item for item in smoke))
            self.assertEqual(sum("capture" in item for item in smoke), 1)
            with self.assertRaises(ValueError):
                q.command_plan("mi300x", ["hip-direct"], out, 1, [])

    def test_shared_resolution_override_validation(self):
        request = q.dependency_request("mi300x", ["hip-direct"], ["RNS8_HIP_ROOT=/sdk with spaces", "CMAKE_PREFIX_PATH=/native"])
        self.assertEqual(request.overrides["RNS8_HIP_ROOT"], "/sdk with spaces")
        self.assertTrue(request.require_hip)
        self.assertFalse(request.require_vcpkg)
        self.assertFalse(request.features["RNS8_ENABLE_CK"])
        for definitions in (["unknown=/x"], ["RNS8_HIP_ROOT"], ["RNS8_HIP_ROOT="],
                            ["RNS8_AMDGPU_TARGETS=gfx1100"], ["RNS8_HIP_ROOT=/a", "RNS8_HIP_ROOT=/b"],
                            ["RNS8_HIP_ROOT=https://user:password@example/a"], ["RNS8_HIP_ROOT=/a\ncommand"]):
            with self.subTest(definitions=definitions), self.assertRaises(ValueError):
                q.dependency_request("mi300x", ["hip-direct"], definitions)

    def test_plan_receipt_uses_shared_selection_without_execution(self):
        selection = {"errors": ["hip: no complete installation discovered"], "hip": {"status": "missing"},
                     "vcpkg": {"status": "unsupported"}, "optional_components": {}, "cmake_bindings": {},
                     "identity_sha256": "selected-fixture", "compiler_environment": {},
                     "tools": {name: {"path": None} for name in ("cmake", "ninja", "host_cxx")}}
        with tempfile.TemporaryDirectory(prefix="plan space ") as directory:
            out = Path(directory)
            with patch.object(q, "resolve", return_value=selection) as shared, \
                 patch.object(q, "source_receipt", return_value={"git_revision": "fixture"}), \
                 patch.object(q, "kernel_inventory", return_value=[]), \
                 patch.object(q.platform, "system", return_value="Linux"), \
                 patch.object(q, "execute_plan") as forbidden, \
                 patch.dict(os.environ, {"PATH": "/usr/bin", "AWS_SECRET_ACCESS_KEY": "not-for-receipts"}, clear=True), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(q.main(["--target", "mi300x", "--devices", "4,7", "--backend", "hip-direct",
                                         "--out-dir", directory, "--dry-run"]), 0)
                forbidden.assert_not_called()
                self.assertNotIn("AWS_SECRET_ACCESS_KEY", shared.call_args.args[1])
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["devices"], [4, 7])
            self.assertEqual(manifest["requested_architecture"], "gfx942")
            self.assertIsNone(manifest["observed_architecture"])
            self.assertEqual(manifest["dependency_resolution_sha256"], "selected-fixture")
            self.assertFalse(manifest["promotion_allowed"])
            self.assertTrue(manifest["execution_blockers"])
            self.assertNotIn("not-for-receipts", (out / "manifest.json").read_text())
            self.assertEqual(set(path.name for path in out.iterdir()),
                             {"manifest.json", "dependency-resolution.json", "environment-allowlist.json"})

    def test_environment_allowlist_and_isolation(self):
        inherited = {"PATH": "/usr/bin", "HOME": "/home/person", "HIP_PATH": "/sdk with spaces",
                     "HIP_VISIBLE_DEVICES": "4,5", "ROCR_VISIBLE_DEVICES": "2", "AWS_SECRET_ACCESS_KEY": "secret",
                     "GITHUB_TOKEN": "secret", "LD_LIBRARY_PATH": "https://user:password@example/a?token=secret"}
        receipt = env.snapshot(inherited)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", receipt)
        self.assertNotIn("secret", json.dumps(receipt))
        self.assertIn("redacted", receipt["LD_LIBRARY_PATH"])
        forwarded = env.build_environment(inherited)
        self.assertNotIn("GITHUB_TOKEN", forwarded)
        self.assertNotIn("LD_LIBRARY_PATH", forwarded)
        self.assertTrue(env.execution_blockers(inherited))
        for key in (*env.VISIBILITY_KEYS, *env.OVERRIDE_KEYS):
            self.assertTrue(env.execution_blockers({key: ""}))
        self.assertFalse(env.execution_blockers({"HIP_PATH": "/sdk"}))
        child = env.isolated_environment(forwarded, 7)
        self.assertEqual(child["HIP_VISIBLE_DEVICES"], "7")
        self.assertNotIn("ROCR_VISIBLE_DEVICES", child)
        self.assertEqual(forwarded["ROCR_VISIBLE_DEVICES"], "2")
        with self.assertRaises(ValueError):
            env.isolated_environment({}, -1)

    def test_positive_runtime_identification(self):
        item = {"inspect_target": "gfx942", "inspect_backend": "hip-direct", "inspect_device": 4}
        self.assertFalse(q.validate_result(item, result(json.dumps(inspection(4)))))
        for changes in ({"gcn_arch": "gfx1100"}, {"backend": "cpu-reference"}, {"device_id": 0},
                        {"device_id": True}, {"hip_available": False}, {"global_mem_bytes": 0},
                        {"global_mem_bytes": None}, {"name": ""}):
            with self.subTest(changes=changes):
                self.assertTrue(q.validate_result(item, result(json.dumps(inspection(4, **changes)))))
        for text in ("{}", "[]", "null", "invalid"):
            self.assertTrue(q.validate_result(item, result(text)))
        child = {**item, "inspect_device": 0, "match_parent": "parent"}
        self.assertTrue(q.validate_result(child, result(json.dumps(inspection())), None))
        self.assertFalse(q.validate_result(child, result(json.dumps(inspection())), inspection(4)))
        self.assertTrue(q.validate_result(child, result(json.dumps(inspection(name="different"))), inspection(4)))

    def test_junit_is_mandatory_and_skips_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "junit.xml"
            item = {"junit": str(xml), "require_test": "required"}
            self.assertTrue(q.validate_result(item, result()))
            for body, valid in (("<testcase name='required'/>", True), ("<testcase name='wrong'/>", False),
                                ("<testcase name='required'><skipped/></testcase>", False),
                                ("<testcase name='required'><failure/></testcase>", False)):
                xml.write_text("<testsuites><testsuite>" + body + "</testsuite></testsuites>", encoding="utf-8")
                self.assertEqual(not q.validate_result(item, result()), valid)
            self.assertTrue(q.validate_result({}, result(timed_out=True)))
            self.assertTrue(q.validate_result({}, result(returncode=None)))

    def manifest(self, out, commands=None, **budgets):
        return {"identity_sha256": "fixture-identity", "devices": [7],
                "budgets": {"total_seconds": 100, "export_reserve_seconds": 1, "minimum_free_bytes": 1, **budgets},
                "commands": commands if commands is not None else [
                    {"name": "first", "argv": [str(out / "missing-tool")], "timeout_seconds": 10, "device": None}]}

    def test_receipt_resume_and_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = self.manifest(out)
            calls = []
            def runner(argv, **kwargs):
                calls.append(kwargs)
                return result("fixture")
            self.assertEqual(q.execute_plan(manifest, out, {}, runner=runner), 0)
            self.assertEqual(len(q.resume_results(out, manifest)), 1)
            self.assertEqual(q.execute_plan(manifest, out, {}, resume=True, runner=runner), 0)
            self.assertEqual(len(calls), 1)
            report = json.loads((out / "results.json").read_text())
            self.assertFalse(report["promotion_allowed"])
            self.assertEqual(report["status"], "smoke_passed_release_qualification_pending")
            self.assertIsNone(report["results"][0]["pci_bdf"])
            with self.assertRaises(ValueError):
                q.resume_results(out, {**manifest, "identity_sha256": "changed"})
            (out / "first.stdout.log").write_text("changed")
            with self.assertRaises(ValueError):
                q.resume_results(out, manifest)

    def test_stop_failure_timeout_interrupt_and_budget(self):
        for outcome, status, code in ((result(returncode=1), "blocked", 1), (result(timed_out=True), "blocked", 1),
                                      (KeyboardInterrupt(), "interrupted", 130)):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                def runner(*args, **kwargs):
                    if isinstance(outcome, BaseException):
                        raise outcome
                    return outcome
                self.assertEqual(q.execute_plan(self.manifest(out), out, {}, runner=runner), code)
                self.assertEqual(json.loads((out / "results.json").read_text())["status"], status)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            ticks = iter([0, 1, 2, 3])
            with patch.object(q, "run_capture") as forbidden:
                self.assertEqual(q.execute_plan(self.manifest(out, total_seconds=2), out, {},
                                                runner=forbidden, clock=lambda: next(ticks)), 1)
                forbidden.assert_not_called()
            self.assertEqual(json.loads((out / "results.json").read_text())["status"], "budget_exhausted")

    def test_owned_process_timeout_without_hardware(self):
        good = run_capture([sys.executable, "-c", "print('ok')"], cwd=q.ROOT, label="fixture", timeout_seconds=10)
        self.assertEqual(good["stdout"], "ok\n")
        timeout = run_capture([sys.executable, "-c", "import time; time.sleep(5)"], cwd=q.ROOT,
                              label="timeout-fixture", timeout_seconds=0.1, heartbeat_seconds=0.02)
        self.assertTrue(timeout["timed_out"])

    def test_kernel_inventory_remains_source_only(self):
        kernels = q.kernel_inventory()
        self.assertTrue(kernels)
        self.assertTrue(all(item["evidence"] == "source_only_not_hardware_qualified" for item in kernels))


if __name__ == "__main__":
    unittest.main()
