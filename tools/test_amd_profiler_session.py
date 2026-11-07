"""Authored collector controls; dummy tools and benchmark bytes are never executed."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from amd_profiler_capabilities import preflight, probe_plan
from amd_profiler_session import (
    ROOT,
    TRACE_FLAGS,
    execute_capture,
    finalize,
    plan_capture,
    selected_dispatches,
    timeline_environment,
    _write_diagnostic,
)
from test_amd_rocpd_timeline import timeline_database
from test_host_profile_ranges import snapshot, scope
from benchmark_request import file_digest
from compiled_binary_identity import bind_document
from amd_profiler_evidence import validate_dispatch_receipt
from target_qualification import EvidenceStore
from test_device_observation import capture as device_capture
from test_target_qualification_support import QualificationFixture, write


class ProfilerSession(unittest.TestCase):
    def setUp(self):
        (ROOT / "temp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "temp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = fixture = QualificationFixture(self.root / "fixture")
        observed = device_capture()
        fixture.capture["device"].update(observed["device"])
        fixture.capture["device_identity_witness"] = observed["device_identity_witness"]
        write(fixture.capture_path, fixture.capture)
        fixture.make_profiler()
        self.sdk = self.root / "sdk"
        (self.sdk / "bin").mkdir(parents=True)
        for name in ("rocprofv3", "rocprofv3-avail", "rocpd", "rocprof-compute"):
            (self.sdk / "bin" / name).write_bytes(b"AUTHORED NEVER EXECUTED " + name.encode())
        for name in (
            "librocprofiler-sdk.so",
            "librocprofiler-sdk-roctx.so",
            "rocprofiler-sdk/librocprofiler-sdk-tool.so",
        ):
            path = self.sdk / "lib" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"AUTHORED NEVER LOADED " + name.encode())
        self.inspector = self.root / "rns8-inspect"
        self.inspector.write_bytes(b"AUTHORED INSPECTOR NEVER EXECUTED")
        plan = probe_plan(
            "Linux",
            "gfx942",
            agent=7,
            resolver=lambda name: str(self.sdk / "bin" / name),
        )

        def probe(command, **kwargs):
            output = {
                "version": "rocprofv3 version 1.4.1\n",
                "help": " ".join((*TRACE_FLAGS, "--output-format")),
                "agents": "GPU:7\n name:gfx942\n logical_node_id:9\n",
            }.get(kwargs["label"], "authored")
            return dict(
                command=command,
                process_id=99,
                returncode=0,
                timed_out=False,
                elapsed_seconds=0.01,
                stdout=output,
                stderr="",
            )

        with patch("amd_profiler_capabilities.platform.system", return_value="Linux"):
            preflight(
                plan,
                self.root / "preflight",
                environment={},
                runner=probe,
                minimum_free_bytes=1,
            )
        self.capabilities = self.root / "preflight/capabilities.json"
        self.out = self.root / "session"

    def plan(self, **kwargs):
        return plan_capture(
            self.fixture.capture_path,
            self.capabilities,
            self.fixture.isa_path,
            self.inspector,
            self.out,
            minimum_free_bytes=1,
            **kwargs,
        )

    def runner(self, command, **kwargs):
        name = kwargs["label"].split()[-1]
        fixture = self.fixture
        pid = {
            "benchmark-identity": 101,
            "inspector-identity": 102,
            "device-inspection": 103,
            "timeline": 222,
        }[name]
        if name.endswith("identity"):
            value = json.loads(
                Path(fixture.binary_receipts["rns8-bench"]["identity_document"]).read_text()
            )
            value["process_id"] = pid
        elif name == "device-inspection":
            value = copy.deepcopy(fixture.capture["device"])
            value.update(
                process_id=pid,
                backend="hip-direct",
                runtime_device_identity=fixture.capture["device_identity_witness"]["before"],
            )
        else:
            self.assertEqual(kwargs["artifact_roots"], [self.out / "native"])
            self.assertGreater(kwargs["max_artifact_bytes"], 0)
            value = copy.deepcopy(fixture.capture)
            for key in (
                "request_identity",
                "execution_id",
                "artifact_capture",
                "external_instrumentation",
            ):
                value.pop(key, None)
            arguments = command[command.index("--") + 2 :]
            env = timeline_environment(
                fixture.capture["request_identity"]["environment"],
                self.sdk / "bin/rocprofv3",
                self.out / "native",
            )
            if kwargs["env"].get("RNS8_PROFILE_RANGES") == "1":
                env["RNS8_PROFILE_RANGES"] = "1"
                value["profiler_ranges"] = snapshot([scope()])
                value["profiler_ranges"]["marker_label_contract"] = "rns8_phase_slot_v1"
                value["profiler_ranges"]["records"][0]["process_id"] = 222
            value.update(
                process_id=pid,
                execution_environment_witness=dict(
                    schema="rns8_execution_environment_witness_v1",
                    state="stable",
                    arguments=arguments,
                    before=env,
                    after=env,
                ),
            )
            write(self.out / "native/benchmark.json", value)
            timeline_database(fixture).replace(self.out / "native/native.db")
            value = None
        return dict(
            command=command,
            process_id=pid,
            returncode=0,
            timed_out=False,
            elapsed_seconds=0.01,
            stdout=json.dumps(value) if value else "profiler tool output\n",
            stderr="",
        )

    def test_plan_is_only_data_with_explicit_roles_and_limits(self):
        plan = self.plan()
        self.assertFalse(self.out.exists())
        self.assertEqual(len(plan["commands"]), 4)
        command = plan["commands"][-1]
        self.assertEqual(command["argv"].count("--"), 1)
        self.assertEqual(command["owned_artifact_directories"], [str(self.out / "native")])
        self.assertEqual(plan["profiler_agent_selector"], 7)
        self.assertFalse(plan["promotion_allowed"])
        self.assertIn("not_unmasked_parent", plan["device_namespace"])

    def test_authored_acquisition_reopens_raw_stages_and_binds_dispatches(self):
        plan = self.plan()
        with patch("amd_profiler_session.platform.system", return_value="Linux"):
            self.assertEqual(execute_capture(plan, environment={}, runner=self.runner), 0)
        summary = json.loads((self.out / "profile-summary.json").read_text())
        self.assertTrue(summary["facts"]["qualified_selected_dispatches"])
        self.assertEqual(summary["facts"]["acquisition"]["native_process_id"], 222)
        self.assertFalse(summary["facts"]["complete_run_coverage"])
        self.assertTrue((self.out / "native-timeline.json").is_file())
        self.assertEqual(
            json.loads((self.out / "operation-join.json").read_text())["state"], "not_requested"
        )
        raw = json.loads((self.out / "native/benchmark.json").read_text())
        self.assertNotIn("request_identity", raw)
        self.assertNotIn("external_instrumentation", raw)
        with self.assertRaisesRegex(ValueError, "already exist"):
            finalize(self.out / "manifest.json")

    def configure_roctx(self, setting):
        fixture = self.fixture
        config = fixture.capture["build_manifest"]["configuration"].splitlines()
        config = [line for line in config if "=" in line and not line.startswith("roctx=")]
        fixture.capture["build_manifest"]["configuration"] = "\n".join(
            [*config, f"roctx={setting}", "fixture=authored"]
        )
        write(fixture.capture_path, fixture.capture)
        for name, binary in fixture.binaries.items():
            identity = Path(fixture.binary_receipts[name]["identity_document"])
            value = json.loads(identity.read_text())
            value["build_manifest"] = value["library_build_manifest"] = fixture.capture[
                "build_manifest"
            ]
            write(identity, value)
            fixture.binary_receipts[name] = bind_document(
                binary, identity, before_sha256=file_digest(binary)
            )
        fixture.make_profiler()

    def test_operation_matching_replays_actual_native_sidecars_and_rejects_changed_derivation(self):
        fixture = self.fixture
        self.configure_roctx("ON")
        plan = self.plan(operation_ranges=True)
        self.assertEqual(plan["expected_native_environment"]["RNS8_PROFILE_RANGES"], "1")
        with patch("amd_profiler_session.platform.system", return_value="Linux"):
            self.assertEqual(execute_capture(plan, environment={}, runner=self.runner), 0)
        operation_path = self.out / "operation-join.json"
        operation = json.loads(operation_path.read_text())
        self.assertEqual(len(operation["selected_dispatches"]), 1)
        self.assertFalse(operation["complete_run_coverage"])
        operation["selected_dispatches"][0]["operation_id"] = 999
        write(operation_path, operation)
        receipt_path = self.out / "dispatch-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["operation_join"]["sha256"] = file_digest(operation_path)
        write(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "operation join differs"):
            validate_dispatch_receipt(
                dict(path=str(receipt_path), sha256=file_digest(receipt_path)),
                capture=fixture.capture,
                store=EvidenceStore(),
                parent=ROOT,
            )

    def test_mutated_plan_is_rejected_before_creating_output_or_calling_runner(self):
        plan = self.plan()
        plan["commands"][-1]["argv"].insert(1, "--pmc")
        with patch("amd_profiler_session.platform.system", return_value="Linux"):
            with self.assertRaisesRegex(ValueError, "plan changed"):
                execute_capture(
                    plan,
                    environment={},
                    runner=lambda *a, **k: self.fail("must not launch"),
                )
        self.assertFalse(self.out.exists())

    def test_operation_ranges_require_actual_compiled_feature_before_any_output(self):
        self.configure_roctx("OFF")
        with self.assertRaisesRegex(ValueError, "compiled with ROCTx"):
            self.plan(operation_ranges=True)
        self.assertFalse(self.out.exists())

    def test_derived_publication_checks_owned_total_before_creating_the_document(self):
        plan = self.plan(max_artifact_bytes=1)
        self.out.mkdir()
        (self.out / "retained.log").write_bytes(b"already retained source evidence")
        target = self.out / "derived.json"
        with self.assertRaisesRegex(ValueError, "total artifact budget"):
            _write_diagnostic(target, dict(state="authored"), plan)
        self.assertFalse(target.exists())

    def test_selection_uses_native_symbols_and_agents_not_temporal_neighbours(self):
        native = self.fixture.native["rocprofiler-sdk-tool"][0]
        self.assertEqual(
            selected_dispatches(native, {"synthetic_selected_symbol"}, "gfx942"),
            [dict(agent_id=3, queue_id=5, dispatch_id=11)],
        )
        for symbols, target in (
            ({"renamed_display_symbol"}, "gfx942"),
            ({"synthetic_selected_symbol"}, "gfx1100"),
        ):
            with self.assertRaises(ValueError):
                selected_dispatches(native, symbols, target)

    def test_unknown_libraries_bad_budgets_and_existing_outputs_are_rejected(self):
        for options in (
            dict(timeout_seconds=0),
            dict(total_seconds=330),
            dict(max_artifact_bytes=0),
            dict(max_output_bytes=True),
            dict(timeout_seconds=10**1000),
            dict(operation_ranges=1),
        ):
            with self.assertRaises(ValueError):
                self.plan(**options)
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError, "previously absent"):
            self.plan()
        self.out.rmdir()
        (self.sdk / "lib/librocprofiler-sdk.so").unlink()
        with self.assertRaisesRegex(ValueError, "library is absent"):
            self.plan()

    def test_instrumented_library_build_is_not_a_production_profile_baseline(self):
        self.fixture.capture["build_manifest"]["configuration"] += "\nhost_sanitizer=thread\n"
        write(self.fixture.capture_path, self.fixture.capture)
        with self.assertRaisesRegex(ValueError, "instrumented build"):
            self.plan()


if __name__ == "__main__":
    unittest.main()
