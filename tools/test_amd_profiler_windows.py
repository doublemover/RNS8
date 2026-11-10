"""Authored RDP ownership/custody controls; no native process or GPU work runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import amd_profiler_windows as rdp
from amd_profiler_capabilities import probe_plan, preflight
from test_device_observation import capture as device_capture
from test_target_qualification_support import QualificationFixture, write
from test_benchmark_schema_support_metadata import add_target_variant_fields


class WindowsRdpCapture(unittest.TestCase):
    def setUp(self):
        (rdp.ROOT / "temp").mkdir(exist_ok=True)
        scratch = tempfile.TemporaryDirectory(dir=rdp.ROOT / "temp")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.fixture = fixture = QualificationFixture(
            self.root / "fixture", target="gfx1100", system="Windows"
        )
        observed = device_capture()
        fixture.capture["device"].update(observed["device"])
        fixture.capture["device"]["gcn_arch"] = "gfx1100"
        fixture.capture["device_identity_witness"] = observed["device_identity_witness"]
        add_target_variant_fields(fixture.capture, "gfx1100")
        write(fixture.capture_path, fixture.capture)
        self.tool = self.root / "RadeonDeveloperPanelCLI.exe"
        self.service = self.root / "RadeonDeveloperServiceCLI.exe"
        self.tool.write_bytes(b"AUTHORED CONTROLLER NEVER EXECUTED")
        self.service.write_bytes(b"AUTHORED SERVICE NEVER EXECUTED")
        plan = probe_plan(
            "Windows", "gfx1100", resolver=lambda name: str(self.root / (name + ".exe"))
        )

        def probe(command, **options):
            output = (
                "RadeonDeveloperPanelCLI 1.0.0\n"
                if options["label"] == "version"
                else "RadeonDeveloperPanelCLI - GPU Capture Tool\nUsage:\nRadeonDeveloperPanelCLI [OPTION...]\n"
                + " ".join((*rdp.FLAGS, "--rgp-counter-collection", "--rgp-instruction-tracing"))
            )
            return dict(
                command=command,
                process_id=99,
                returncode=1,
                timed_out=False,
                elapsed_seconds=0.01,
                stdout=output,
                stderr="",
            )

        with patch("amd_profiler_capabilities.platform.system", return_value="Windows"):
            preflight(
                plan, self.root / "preflight", environment={}, runner=probe, minimum_free_bytes=1
            )
        self.capabilities = self.root / "preflight/capabilities.json"
        self.out = self.root / "session"
        self.work_finished = threading.Event()
        self.fail_workload, self.foreign_pid, self.empty_trace = False, False, False
        self.calls = []

    def plan(self, **options):
        return rdp.plan_capture(
            self.fixture.capture_path, self.capabilities, self.out, minimum_free_bytes=1, **options
        )

    def runner(self, command, **options):
        name = options["label"].split(":")[-1]
        self.calls.append(name)
        pid = {"identity": 111, "controller": 112, "workload": 222}[name]
        result = dict(
            command=command,
            process_id=pid,
            returncode=0,
            timed_out=False,
            elapsed_seconds=0.01,
            stdout="",
            stderr="",
        )
        if name == "identity":
            value = json.loads(
                Path(self.fixture.binary_receipts["rns8-bench"]["identity_document"]).read_text()
            )
            value["process_id"] = pid
            result["stdout"] = json.dumps(value)
        elif name == "controller":
            options["on_start"](pid)
            while not self.work_finished.wait(0.01):
                if options["cancel_event"].is_set():
                    return dict(result, returncode=1, interrupted=True)
            trace = self.out / "raw/capture.rgp"
            trace.write_bytes(b"" if self.empty_trace else b"AUTHORED OPAQUE RGP NOT GPU EVIDENCE")
            binary = self.fixture.binaries["rns8-bench"]
            result["stdout"] = (
                f"Connecting to process: {binary} (PID: {223 if self.foreign_pid else 222})\nTrace saved to: {trace}\nSize: {trace.stat().st_size} bytes\nDone!\n"
            )
        else:
            if self.fail_workload:
                return dict(result, returncode=1)
            value = copy.deepcopy(self.fixture.capture)
            for key in (
                "request_identity",
                "artifact_capture",
                "execution_id",
                "external_instrumentation",
            ):
                value.pop(key, None)
            value.update(
                process_id=pid,
                execution_environment_witness=dict(
                    schema="rns8_execution_environment_witness_v1",
                    state="stable",
                    arguments=command[1:],
                    before=self.fixture.capture["request_identity"]["environment"],
                    after=self.fixture.capture["request_identity"]["environment"],
                ),
            )
            write(self.out / "raw/benchmark.json", value)
            self.work_finished.set()
        return result

    def execute(self, plan):
        with patch("amd_profiler_windows.platform.system", return_value="Windows"):
            return rdp.execute_capture(
                plan, authorize_driver_profile=True, environment={}, runner=self.runner
            )

    def test_plan_never_runs_service_or_changes_clock_or_interprets_dispatch_indices(self):
        plan = self.plan(first_dispatch=8, dispatch_count=3, counters=True, instruction_trace=True)
        self.assertFalse(self.out.exists())
        self.assertEqual(set(plan["commands"]), {"identity", "controller", "workload"})
        command = plan["commands"]["controller"]
        self.assertIn("dispatch:8:3", command)
        self.assertEqual(command[command.index("--remote-host") + 1], "127.0.0.1")
        self.assertNotIn(str(self.service), command)
        self.assertNotIn("--clock-mode", command)
        self.assertFalse(plan["performance_promotion_allowed"])

    def test_actual_synthetic_chain_reopens_owned_bytes_without_machine_evidence_credit(self):
        plan = self.plan()
        capabilities = json.loads(self.capabilities.read_text())
        self.assertEqual(capabilities["tools"]["RadeonDeveloperPanelCLI"]["version"], "1.0.0")
        self.assertTrue(all(row["completed"] for row in capabilities["observations"]))
        self.assertTrue(
            all(row["result"]["returncode"] == 1 for row in capabilities["observations"])
        )
        self.assertEqual(self.execute(plan), 0)
        summary = rdp.review_capture(self.out / "plan.json")
        self.assertEqual(summary["capture_state"], "raw_saved_unqualified")
        self.assertEqual(summary["process_id"], 222)
        self.assertEqual(self.calls, ["identity", "controller", "workload"])
        self.assertFalse(summary["performance_promotion_allowed"])
        self.assertEqual(summary["selected_dispatch_identity"], "unverified")
        self.assertIn("unavailable", summary["counter_values"])
        trace = self.out / "raw/capture.rgp"
        trace.write_bytes(b"CHANGED OPAQUE DATA")
        with self.assertRaises(ValueError):
            rdp.review_capture(self.out / "plan.json")

    def test_wrong_connected_pid_empty_trace_and_failed_workload_remain_partial(self):
        for setting in ("foreign_pid", "empty_trace", "fail_workload"):
            with self.subTest(setting=setting):
                self.out = self.root / setting
                self.work_finished.clear()
                setattr(self, setting, True)
                self.assertEqual(self.execute(self.plan()), 1)
                summary = json.loads((self.out / "capture-summary.json").read_text())
                self.assertEqual(summary["capture_state"], "failed_or_partial")
                self.assertTrue(summary["raw_artifacts_preserved"])
                self.assertTrue((self.out / "stages.json").is_file())
                setattr(self, setting, False)

    def test_mutated_plan_and_absent_explicit_driver_authorization_do_not_launch(self):
        plan = self.plan()
        with patch("amd_profiler_windows.platform.system", return_value="Windows"):
            with self.assertRaises(ValueError):
                rdp.execute_capture(
                    plan, runner=lambda *args, **kwargs: self.fail("must not launch")
                )
        plan["commands"]["controller"] += ["--clock-mode", "stable"]
        with self.assertRaisesRegex(ValueError, "plan changed"):
            self.execute(plan)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.calls)

    def test_malformed_budgets_and_dispatch_selectors_fail_before_output(self):
        for options in (
            dict(first_dispatch=True),
            dict(first_dispatch=0),
            dict(dispatch_count=0),
            dict(dispatch_count=1025),
            dict(counters=1),
            dict(instruction_trace=1),
            dict(timeout_seconds=10**1000),
            dict(total_seconds=150),
            dict(max_artifact_bytes=True),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.plan(**options)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
