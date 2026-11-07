"""Authored WPR ownership/cleanup protocol; no Windows tool or workload runs."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_request import file_digest
import host_profile_windows as host
import check_dependencies_lib.system as dependencies
import check_dependencies_lib.discovery as discovery
from test_device_observation import capture as device_capture
from test_pe_symbol_identity import executable, symbols
from test_target_qualification_support import QualificationFixture, write
from test_wpa_sample_reader import sample_fields
from wpa_sample_reader import HEADERS


class WindowsHostProfile(unittest.TestCase):
    def setUp(self):
        (host.ROOT / "temp").mkdir(exist_ok=True)
        scratch = tempfile.TemporaryDirectory(dir=host.ROOT / "temp")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.fixture = fixture = QualificationFixture(self.root / "fixture")
        self.binary = fixture.binaries["rns8-bench"]
        self.binary.write_bytes(executable())
        fixture.capture["request_identity"]["executable_sha256"] = file_digest(self.binary)
        fixture.capture["request_identity"]["identity_sha256"] = host.canonical_digest(
            {
                key: value
                for key, value in fixture.capture["request_identity"].items()
                if key != "identity_sha256"
            }
        )
        fixture.capture["artifact_capture"]["executable_sha256"] = file_digest(self.binary)
        observed = device_capture()
        fixture.capture["device"].update(observed["device"])
        fixture.capture["device_identity_witness"] = observed["device_identity_witness"]
        write(fixture.capture_path, fixture.capture)
        self.pdb, self.wpr, self.exporter = (
            self.root / name for name in ("app.pdb", "wpr.exe", "wpaexporter.exe")
        )
        self.pdb.write_bytes(symbols())
        for path in (self.wpr, self.exporter):
            path.write_bytes(executable())
        self.out = self.root / "session"
        self.calls, self.fail = [], None

    def plan(self):
        return host.plan_capture(
            self.fixture.capture_path, self.wpr, self.exporter, self.pdb, self.out
        )

    def runner(self, command, **policy):
        name = policy["label"].split(":")[-1]
        self.calls.append((name, command, policy))
        result = dict(
            command=command,
            process_id=333,
            returncode=0,
            timed_out=False,
            interrupted=False,
            limit_exceeded=None,
            capture_error=None,
            stdout="authored output",
            stderr="",
        )
        if name == "profiles":
            result["stdout"] = "Rns8Cpu"
        elif name == "identity":
            value = json.loads(
                Path(self.fixture.binary_receipts["rns8-bench"]["identity_document"]).read_text()
            )
            value["process_id"] = 333
            result["stdout"] = json.dumps(value)
        elif name == "status":
            result.update(returncode=1, stderr="Error code: 0xc5583000")
        elif name == "workload":
            value = copy.deepcopy(self.fixture.capture)
            for key in ("request_identity", "artifact_capture", "execution_id"):
                value.pop(key, None)
            value.update(
                process_id=333,
                execution_environment_witness=dict(
                    schema="rns8_execution_environment_witness_v1",
                    state="stable",
                    arguments=command[1:],
                    before=self.fixture.capture["request_identity"]["environment"],
                    after=self.fixture.capture["request_identity"]["environment"],
                ),
            )
            write(self.out / "raw/benchmark.json", value)
        elif name == "stop":
            (self.out / "raw/profile.etl").write_bytes(b"synthetic ETL; native decoding is mocked")
        elif name == "export":
            with (self.out / "export/native.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(HEADERS)
                writer.writerow(sample_fields(333, image=self.binary.name))
        if name == self.fail:
            result["returncode"] = 7
        return result

    def execute(self, plan):
        with patch.object(host.platform, "system", return_value="Windows"):
            return host.execute_capture(
                plan, authorize_system_recording=True, runner=self.runner, environment={}
            )

    def test_plan_never_runs_and_every_owner_command_keeps_instance_last(self):
        with patch.object(host, "run_capture", side_effect=AssertionError("must not run")):
            plan = self.plan()
        self.assertFalse(self.out.exists())
        for name in ("status", "start", "stop", "cancel"):
            self.assertEqual(plan["commands"][name][-2:], ["-instancename", plan["instance"]])
        self.assertFalse(plan["promotion_allowed"])
        self.assertNotIn("srv*", plan["symbol_search_environment"]["_NT_SYMBOL_PATH"])
        with (
            patch.object(host.platform, "system", return_value="Windows"),
            self.assertRaisesRegex(ValueError, "authorize"),
        ):
            host.execute_capture(plan)
        self.assertFalse(self.out.exists())

    def test_actual_synthetic_producer_export_binding_preserves_partial_scope(self):
        self.assertEqual(self.execute(self.plan()), 0)
        result = json.loads((self.out / "host-profile.json").read_text())
        self.assertEqual(result["qualification"], "native_capture_export_bound_attribution_partial")
        self.assertEqual(result["recording_cleanup"], "stop_completed")
        self.assertFalse(result["promotion_allowed"])
        self.assertEqual([name for name, _, _ in self.calls][-1], "export")

    def test_failed_workload_stops_only_owned_instance_and_never_exports(self):
        self.fail = "workload"
        self.assertEqual(self.execute(self.plan()), 1)
        self.assertEqual([name for name, _, _ in self.calls][-1], "stop")
        result = json.loads((self.out / "host-profile.json").read_text())
        self.assertEqual(result["first_failed_stage"], "workload")
        self.assertTrue((self.out / "raw/profile.etl").is_file())
        self.assertFalse((self.out / "profiled.json").exists())

    def test_failed_stop_uses_owned_cancel_without_exhausted_artifact_monitor(self):
        self.fail = "stop"
        self.assertEqual(self.execute(self.plan()), 1)
        name, command, policy = self.calls[-1]
        self.assertEqual(name, "cancel")
        self.assertEqual(command[-2], "-instancename")
        self.assertEqual(policy["artifact_roots"], ())
        self.assertLessEqual(policy["timeout_seconds"], 15)

    def test_absence_requires_complete_native_positive_exit_hresult(self):
        value = dict(
            returncode=1,
            timed_out=False,
            interrupted=False,
            limit_exceeded=None,
            capture_error=None,
            stdout="",
            stderr="Error code: 0xc5583000",
        )
        self.assertTrue(host.instance_absent(value))
        for change in (
            dict(returncode=0),
            dict(returncode=-1),
            dict(timed_out=True),
            dict(interrupted=True),
            dict(capture_error="failed"),
            dict(stderr="access denied"),
        ):
            self.assertFalse(host.instance_absent(dict(value, **change)))

    def test_dependency_discovery_finds_WPT_without_running_a_tool(self):
        tool = self.root / "Windows Kits/10/Windows Performance Toolkit/wpaexporter.exe"
        tool.parent.mkdir(parents=True)
        tool.write_bytes(executable())
        with (
            patch.object(dependencies, "selected_command", return_value=(False, None)),
            patch.object(dependencies.shutil, "which", return_value=None),
            patch.object(dependencies.platform, "system", return_value="Windows"),
            patch.dict(dependencies.os.environ, {"ProgramFiles(x86)": str(self.root)}),
            patch.object(discovery, "run", side_effect=AssertionError("must not run")),
        ):
            self.assertEqual(dependencies.find_command("wpaexporter"), str(tool))
            self.assertIn("found", discovery.command_version("wpaexporter", str(tool)))
            self.assertFalse(dependencies.command_required_for_host("wpaexporter", "Windows"))


if __name__ == "__main__":
    unittest.main()
