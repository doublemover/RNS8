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
from test_perf_recording_contract import recording_bytes
from test_target_qualification_support import QualificationFixture, write


def elf(machine=62):
    return b"\x7fELF\x02\x01" + bytes(12) + machine.to_bytes(2, "little")


class RecordingFailure(unittest.TestCase):
    @staticmethod
    def process(stderr="", **changes):
        return dict(
            dict(
                returncode=1,
                timed_out=False,
                interrupted=False,
                limit_exceeded=None,
                capture_error=None,
                stdout="",
                stderr=stderr,
            ),
            **changes,
        )

    def test_only_completed_perf_specific_denials_are_blocked_permissions(self):
        for stderr, reason in (
            (
                "No permission to enable cpu-clock:u event.\n",
                "reported_perf_event_permission_denial",
            ),
            (
                "Access to performance monitoring and observability operations is limited.\n",
                "reported_perf_event_permission_denial",
            ),
            ("Permission error mapping pages.\n", "reported_perf_mapping_permission_denial"),
            (
                "Error:\r\n\tNo permission to enable cpu-clock:u event.\r\n",
                "reported_perf_event_permission_denial",
            ),
        ):
            with self.subTest(reason=reason, stderr=stderr):
                observed = session.recording_failure(self.process(stderr))
                self.assertEqual(observed["state"], "blocked")
                self.assertEqual(observed["reason"], reason)
                self.assertIn("not_observed_syscall_errno", observed["attribution"])
                for change, expected in (
                    (dict(returncode=0), "recording_process_exit_zero_not_profile_qualification"),
                    (dict(returncode=None), "recording_process_did_not_start"),
                    (dict(returncode=-15), "recording_process_signaled"),
                    (dict(timed_out=True), "recording_timeout"),
                    (dict(interrupted=True), "recording_interrupted"),
                    (dict(interrupted=1), "recording_process_outcome_incomplete"),
                    (dict(limit_exceeded="output"), "recording_capture_or_budget_failure"),
                    (dict(capture_error="reader failed"), "recording_capture_or_budget_failure"),
                ):
                    self.assertEqual(
                        session.recording_failure(self.process(stderr, **change))["reason"],
                        expected,
                    )

    def test_generic_permissions_and_policy_warnings_are_not_event_denials(self):
        for stderr in (
            "Permission denied opening perf.data",
            "perf_event_paranoid is 4",
            "workload: operation not permitted",
            "unknown option --clockid",
        ):
            with self.subTest(stderr=stderr):
                observed = session.recording_failure(self.process(stderr))
                self.assertEqual(observed["state"], "failed")
                self.assertEqual(observed["reason"], "recording_failure_unclassified")

    def test_failed_checkpoint_binding_and_first_failure_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = [
                dict(name="perf-version", argv=["/authored/perf", "--version"]),
                dict(name="cpu-sampling", argv=["/authored/perf", "record"]),
            ]
            plan = dict(identity_sha256="a" * 64, commands=commands)
            write(root / "manifest.json", plan)
            rows = [
                dict(
                    self.process(returncode=0),
                    name=commands[0]["name"],
                    command=commands[0]["argv"],
                ),
                dict(
                    self.process("No permission to enable cpu-clock:u event."),
                    name=commands[1]["name"],
                    command=commands[1]["argv"],
                ),
            ]
            results = dict(
                schema="rns8_gpu_session_results_v2",
                status="blocked",
                run_id="authored-run",
                manifest_identity_sha256=plan["identity_sha256"],
                results=rows,
            )
            write(root / "results.json", results)
            observed = session.acquisition_failure(plan, root)
            self.assertEqual(observed["stage"], "cpu-sampling")
            self.assertEqual(observed["reason"], "reported_perf_event_permission_denial")
            self.assertFalse(observed["acquisition_verified"])
            self.assertFalse(observed["promotion_allowed"])
            self.assertTrue(observed["raw_artifacts_preserved"])
            self.assertIn("sha256", observed["session_results"])
            read = session.bounded_read

            def changed_after_read(path):
                value = read(path)
                if path.name == "results.json":
                    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
                return value

            with patch.object(session, "bounded_read", changed_after_read):
                changed = session.acquisition_failure(plan, root)
            self.assertEqual(
                changed["reason"], "acquisition_failure_evidence_unavailable_or_changed"
            )
            self.assertFalse(changed["acquisition_verified"])
            rows[0].update(returncode=2, stderr="earlier version failure")
            write(root / "results.json", results)
            self.assertEqual(
                session.acquisition_failure(plan, root)["reason"],
                "acquisition_failure_evidence_unavailable_or_changed",
            )
            results["results"] = rows[:1]
            write(root / "results.json", results)
            self.assertEqual(session.acquisition_failure(plan, root)["stage"], "perf-version")
            rows[0]["command"] = ["/another/perf", "--version"]
            write(root / "results.json", results)
            self.assertEqual(
                session.acquisition_failure(plan, root)["reason"],
                "acquisition_failure_evidence_unavailable_or_changed",
            )


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
        for option in ("--sample-cpu", "--sample-identifier", "--timestamp", "--period"):
            self.assertIn(option, command)
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
            dict(scheduling=1),
            dict(timeout_seconds=True),
            dict(total_seconds=20),
            dict(max_artifact_bytes=0),
            dict(max_output_bytes=True),
            dict(minimum_free_bytes=0),
        ):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.plan(**policy)

    def test_scheduling_is_explicit_process_scoped_and_receipt_reconstructed(self):
        self.assertNotIn("--switch-events", self.plan()["commands"][3]["argv"])
        plan = self.plan(scheduling=True)
        self.assertIn("--switch-events", plan["commands"][3]["argv"])
        self.assertNotIn("-a", plan["commands"][3]["argv"])
        with patch.object(session, "production_profile_input", return_value=self.binary):
            self.assertEqual(plan, session.rebuild(plan))
        self.assertFalse(self.out.exists())

    def test_range_selection_is_opt_in_and_rebuild_preserves_its_environment(self):
        self.assertFalse(self.plan()["operation_ranges"])
        value = json.loads(self.capture.read_text())
        value["build_manifest"] = dict(configuration="roctx=OFF\n")
        write(self.capture, value)
        with self.assertRaisesRegex(ValueError, "compiled with ROCTx"):
            self.plan(operation_ranges=True)
        value["build_manifest"]["configuration"] = "roctx=ON\n"
        write(self.capture, value)
        plan = self.plan(operation_ranges=True)
        for key in ("expected_native_environment", "profiler_environment"):
            self.assertEqual(plan[key]["RNS8_PROFILE_RANGES"], "1")
        with patch.object(session, "production_profile_input", return_value=self.binary):
            self.assertEqual(session.rebuild(plan), plan)
        with self.assertRaises(ValueError):
            self.plan(operation_ranges=1)
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
        failure = json.loads((self.out / "host-profile.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["qualification"], "failed_acquisition")
        self.assertFalse(failure["acquisition_verified"])
        self.assertFalse(failure["promotion_allowed"])


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
            (self.out / "native/perf.data").write_bytes(recording_bytes(samples=[dict(pid=222)]))
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
        self.assertFalse(receipt["leaf_module_custody"]["complete"])
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

    def test_resealed_module_match_cannot_replace_actual_native_file_binding(self):
        def change(value):
            value["leaf_module_custody"]["complete"] = True
            value["leaf_module_custody"]["matched_build_id_samples"] = 1
            value["leaf_module_custody"]["modules"][0]["state"] = "matched"

        self.sample_change = change
        self.assertEqual(self.execute(), 1)
        receipt = json.loads((self.out / "host-profile.json").read_text())
        self.assertEqual(receipt["qualification"], "binding_failed")
        self.assertFalse((self.out / "profiled.json").exists())

    def test_resealed_recorded_frame_attribution_requires_native_mapping_inputs(self):
        def change(value):
            value["recorded_frame_custody"]["selected_frame_observations"] += 1

        self.sample_change = change
        self.assertEqual(self.execute(), 1)
        receipt = json.loads((self.out / "host-profile.json").read_text())
        self.assertEqual(receipt["qualification"], "binding_failed")
        self.assertFalse((self.out / "profiled.json").exists())

    def test_resealed_current_source_counts_cannot_replace_native_source_custody(self):
        def change(value):
            value["current_source_custody"]["selected_frame_observations"] += 1

        self.sample_change = change
        self.assertEqual(self.execute(), 1)
        self.assertFalse((self.out / "profiled.json").exists())


if __name__ == "__main__":
    unittest.main()
