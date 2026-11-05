#!/usr/bin/env python3
"""Host-session policy controls with synthetic ELF bytes; no tool is executed."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import host_profile_session as session
import host_profile_native as native
from test_device_observation import capture as device_capture
from test_host_profile_native import sample as native_sample
from test_target_qualification_support import QualificationFixture, write


def elf(machine=62):
    return b"\x7fELF\x02\x01" + bytes(12) + machine.to_bytes(2, "little")


class HostSession(unittest.TestCase):
    def setUp(self):
        root = session.ROOT / "temp/host-profile-session-controls"
        root.mkdir(parents=True, exist_ok=True)
        scratch = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.binary, self.perf, self.python = (
            self.root / name for name in ("benchmark", "perf", "python")
        )
        for path in (self.binary, self.perf, self.python):
            path.write_bytes(elf())
        self.capture = self.root / "production.json"
        self.capture.write_text(
            json.dumps(
                dict(
                    backend_selected="hip-direct",
                    request_identity=dict(
                        arguments=["--device", "0"], environment={"HIP_VISIBLE_DEVICES": "0"}
                    ),
                )
            )
        )
        self.out = self.root / "session"

    def plan(self, **kwargs):
        # Numerical/artifact admission is exercised by the shared production
        # contract corpus. This fixture isolates command and resource policy.
        with patch.object(session, "production_profile_input", return_value=self.binary):
            return session.plan_capture(self.capture, self.perf, self.python, self.out, **kwargs)

    def test_plan_is_read_only_and_samples_only_the_owned_workload(self):
        with patch.object(session, "execute_plan", side_effect=AssertionError("execution")):
            plan = self.plan()
        self.assertFalse(self.out.exists())
        self.assertEqual(
            [row["name"] for row in plan["commands"]],
            [
                "perf-version",
                "native-module",
                "benchmark-identity",
                "cpu-sampling",
                "native-decode",
            ],
        )
        record = plan["commands"][3]
        command = record["argv"]
        self.assertEqual(command[command.index("--event") + 1], "cpu-clock:u")
        self.assertEqual(command[command.index("--clockid") + 1], "CLOCK_MONOTONIC")
        self.assertEqual(command[command.index("--max-size") + 1], "64M")
        self.assertIn("--no-buildid-cache", command)
        self.assertNotIn("-a", command)
        self.assertNotIn("--pid", command)
        self.assertEqual(command[command.index("--") + 1], str(self.binary))
        self.assertEqual(
            plan["commands"][4]["monitor_artifact_directories"], [str(self.out / "native")]
        )
        self.assertEqual(plan["profiler_environment"]["DEBUGINFOD_URLS"], "")
        self.assertFalse(plan["promotion_allowed"])

    def test_script_launchers_mixed_abis_and_invalid_budgets_fail_before_output(self):
        self.perf.write_bytes(b"#!/bin/sh\nexit 0\n")
        with self.assertRaisesRegex(ValueError, "ELF64"):
            self.plan()
        self.perf.write_bytes(elf(183))
        with self.assertRaisesRegex(ValueError, "tuple"):
            self.plan()
        self.perf.write_bytes(elf())
        for policy in (
            dict(timeout_seconds=True),
            dict(total_seconds=20),
            dict(max_artifact_bytes=0),
            dict(max_output_bytes=True),
            dict(minimum_free_bytes=0),
        ):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.plan(**policy)
        self.assertFalse(self.out.exists())

    def test_mutated_command_and_non_linux_execution_do_not_start_work(self):
        original = self.plan()
        changed = copy.deepcopy(original)
        changed["commands"][3]["argv"][3] = "cycles:u"
        with (
            patch.object(session.platform, "system", return_value="Linux"),
            patch.object(session, "production_profile_input", return_value=self.binary),
            patch.object(session, "execute_plan", side_effect=AssertionError("workload")),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            session.execute_capture(changed)
        with (
            patch.object(session.platform, "system", return_value="Windows"),
            self.assertRaisesRegex(ValueError, "Linux"),
        ):
            session.execute_capture(original)
        self.assertFalse(self.out.exists())

    def test_failed_owned_session_preserves_manifest_and_sanitizes_observer_environment(self):
        plan = self.plan()

        def failed(manifest, out, environment, **options):
            self.assertEqual(manifest, plan)
            self.assertEqual(out, self.out)
            self.assertEqual(environment["HIP_VISIBLE_DEVICES"], "0")
            self.assertEqual(environment["PERF_CONFIG"], "/dev/null")
            self.assertEqual(environment["DEBUGINFOD_URLS"], "")
            self.assertNotIn("PERF_EXEC_PATH", environment)
            self.assertNotIn("PYTHONPATH", environment)
            return 1

        with (
            patch.object(session.platform, "system", return_value="Linux"),
            patch.object(session, "production_profile_input", return_value=self.binary),
            patch.object(session, "execute_plan", side_effect=failed),
            patch.object(session, "finalize", side_effect=AssertionError("finalize")),
        ):
            self.assertEqual(
                session.execute_capture(
                    plan,
                    environment={
                        "PATH": "/fixture/bin",
                        "HIP_VISIBLE_DEVICES": "99",
                        "PERF_EXEC_PATH": "/unbound/helpers",
                        "PYTHONPATH": "/unbound/modules",
                    },
                ),
                1,
            )
        self.assertTrue((self.out / "manifest.json").is_file())
        self.assertFalse((self.out / "profiled.json").exists())


class HostSessionBinding(unittest.TestCase):
    def setUp(self):
        root = session.ROOT / "temp/host-profile-session-controls"
        root.mkdir(parents=True, exist_ok=True)
        scratch = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.fixture = fixture = QualificationFixture(self.root / "fixture")
        benchmark = fixture.binaries["rns8-bench"]
        benchmark.write_bytes(elf() + b"synthetic benchmark never executed")
        capture = fixture.capture
        capture["request_identity"]["executable_sha256"] = session.file_digest(benchmark)
        capture["request_identity"]["identity_sha256"] = session.canonical_digest(
            {k: v for k, v in capture["request_identity"].items() if k != "identity_sha256"}
        )
        capture["artifact_capture"]["executable_sha256"] = session.file_digest(benchmark)
        observed = device_capture()
        capture["device"].update(observed["device"])
        capture["device_identity_witness"] = observed["device_identity_witness"]
        write(fixture.capture_path, capture)
        self.perf, self.python, self.module = (
            self.root / name for name in ("perf", "python", "perf.synthetic.so")
        )
        for path in (self.perf, self.python, self.module):
            path.write_bytes(elf() + b"synthetic tool never executed")
        self.out = self.root / "session"
        self.sample_change = lambda value: None

    def runner(self, command, **options):
        name = options["label"].split()[-1]
        value, pid = None, 111
        if name == "benchmark-identity":
            value = json.loads(
                Path(self.fixture.binary_receipts["rns8-bench"]["identity_document"]).read_text()
            )
            value["process_id"] = pid
        elif name == "native-module":
            value = dict(
                schema="rns8_perf_native_capabilities_v1",
                state="available",
                producer_contract=native.PRODUCER_CONTRACT,
                python_version=(session.ROOT / ".python-version").read_text().strip(),
                native_module=session.reference(self.module),
            )
        elif name == "cpu-sampling":
            pid = 444  # The owned profiler launcher, not its benchmark child.
            value = copy.deepcopy(self.fixture.capture)
            for key in ("request_identity", "artifact_capture", "execution_id"):
                value.pop(key, None)
            environment = dict(self.fixture.capture["request_identity"]["environment"])
            environment["PERF_CONFIG"] = "/dev/null"
            value.update(
                process_id=222,
                execution_environment_witness=dict(
                    schema="rns8_execution_environment_witness_v1",
                    state="stable",
                    arguments=command[command.index("--") + 2 :],
                    before=environment,
                    after=environment,
                ),
            )
            write(self.out / "native/benchmark.json", value)
            (self.out / "native/perf.data").write_bytes(b"authored recording never acquired")
            value = None
        elif name == "native-decode":

            def replay(data, *, sample):
                def process_events():
                    sample(native_sample(222))

                return SimpleNamespace(process_events=process_events, e_machine=62, is_64_bit=True)

            module = SimpleNamespace(
                __file__=str(self.module), data=lambda path: path, session=replay
            )
            report = native.decode(self.out / "native/perf.data", 222, perf_module=module)
            report["profiled_capture"] = session.reference(self.out / "native/benchmark.json")
            self.sample_change(report)
            native.seal(report, native.MAX_DECODED_BYTES)
            write(self.out / "native/samples.json", report)
        return dict(
            command=command,
            process_id=pid,
            returncode=0,
            timed_out=False,
            elapsed_seconds=0.01,
            stdout=json.dumps(value) if value is not None else "authored tool output\n",
            stderr="",
        )

    def execute(self):
        plan = session.plan_capture(
            self.fixture.capture_path, self.perf, self.python, self.out, minimum_free_bytes=1
        )
        with patch.object(session.platform, "system", return_value="Linux"):
            return session.execute_capture(plan, environment={}, runner=self.runner)

    def test_authored_stages_bind_actual_child_pid_and_preserve_native_capture(self):
        self.assertEqual(self.execute(), 0)
        receipt = json.loads((self.out / "host-profile.json").read_text())
        self.assertEqual((receipt["benchmark_pid"], receipt["profiler_launcher_pid"]), (222, 444))
        self.assertEqual(receipt["selected_operation_range"], "unbound")
        self.assertFalse(receipt["production_latency"])
        raw = json.loads((self.out / "native/benchmark.json").read_text())
        bound = json.loads((self.out / "profiled.json").read_text())
        self.assertNotIn("external_instrumentation", raw)
        self.assertEqual(bound["external_instrumentation"]["kind"], "host_profiler")
        self.assertEqual(bound["request_identity"]["environment"]["PERF_CONFIG"], "/dev/null")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            session.finalize(self.out / "manifest.json")

    def test_resealed_profiler_pid_substitution_cannot_become_benchmark_attribution(self):
        self.sample_change = lambda value: value.update(pid=444)
        self.assertEqual(self.execute(), 1)
        receipt = json.loads((self.out / "host-profile.json").read_text())
        self.assertEqual(receipt["qualification"], "binding_failed")
        self.assertTrue((self.out / "native/perf.data").is_file())
        self.assertFalse((self.out / "profiled.json").exists())


if __name__ == "__main__":
    unittest.main()
