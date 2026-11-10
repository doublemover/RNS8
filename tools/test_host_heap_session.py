"""Authored direct-child allocation protocol controls; no native tool is launched."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_request import file_digest
import host_heap_session as host
from test_device_observation import capture as device_capture
from test_elf_module_identity import image
from test_heaptrack_native import raw_fixture
from test_target_qualification_support import QualificationFixture, write


class HeapPlatform(unittest.TestCase):
    def test_foreign_platform_refuses_before_any_plan_or_launch(self):
        with (
            patch.object(host.platform, "system", return_value="Windows"),
            patch.object(host, "Session") as child,
            self.assertRaisesRegex(ValueError, "requires native Linux"),
        ):
            host.execute_capture({})
        child.assert_not_called()


@unittest.skipUnless(os.name == "posix", "native Linux path/LD_PRELOAD contract fixture")
class HeapSession(unittest.TestCase):
    def setUp(self):
        (host.ROOT / "temp").mkdir(exist_ok=True)
        scratch = tempfile.TemporaryDirectory(dir=host.ROOT / "temp")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.fixture = fixture = QualificationFixture(self.root / "fixture")
        self.binary = fixture.binaries["rns8-bench"]
        self.binary.write_bytes(image(executable=True))
        self.preload = self.root / "libheaptrack_preload.so"
        self.preload.write_bytes(image())
        capture = fixture.capture
        capture["request_identity"]["executable_sha256"] = file_digest(self.binary)
        capture["request_identity"]["identity_sha256"] = host.canonical_digest(
            {
                key: value
                for key, value in capture["request_identity"].items()
                if key != "identity_sha256"
            }
        )
        capture["artifact_capture"]["executable_sha256"] = file_digest(self.binary)
        observed = device_capture()
        capture["device"].update(observed["device"])
        capture["device_identity_witness"] = observed["device_identity_witness"]
        write(fixture.capture_path, capture)
        self.out = self.root / "session"
        self.fail = self.wrong_pid = self.mutate = False
        self.calls = []

    def plan(self, **kwargs):
        return host.plan_capture(
            self.fixture.capture_path, self.preload, self.out, minimum_free_bytes=1, **kwargs
        )

    def runner(self, command, **options):
        name = options["label"].split(":")[-1]
        self.calls.append((name, command, options))
        value = None
        if name == "compiled-identity":
            value = json.loads(
                Path(self.fixture.binary_receipts["rns8-bench"]["identity_document"]).read_text()
            )
            value["process_id"] = 123
            self.assertNotIn("LD_PRELOAD", options["env"])
        elif name == "allocation-workload":
            self.assertEqual(options["env"]["LD_PRELOAD"], str(self.preload))
            self.assertEqual(
                options["env"]["DUMP_HEAPTRACK_OUTPUT"], str(self.out / "heaptrack.raw")
            )
            value = copy.deepcopy(self.fixture.capture)
            for key in ("request_identity", "artifact_capture", "execution_id"):
                value.pop(key, None)
            environment = dict(self.fixture.capture["request_identity"]["environment"])
            value.update(
                process_id=999 if self.wrong_pid else 123,
                execution_environment_witness=dict(
                    schema="rns8_execution_environment_witness_v1",
                    state="stable",
                    arguments=command[1:],
                    before=environment,
                    after=environment,
                ),
            )
            write(self.out / "benchmark.json", value)
            (self.out / "heaptrack.raw").write_text(
                raw_fixture("+ 20 1 aa\n- aa\n", executable=str(self.binary))
            )
            if self.mutate:
                self.preload.write_bytes(image(extra=b"changed after launch"))
            value = None
        return dict(
            command=command,
            process_id=123,
            returncode=1 if self.fail else 0,
            timed_out=False,
            interrupted=False,
            limit_exceeded=None,
            capture_error=None,
            stdout=json.dumps(value) if value is not None else "authored tool output\n",
            stderr="",
        )

    def execute(self):
        plan = self.plan()
        with (
            patch.object(host.platform, "system", return_value="Linux"),
            patch("host_instrumentation_session.run_capture", side_effect=self.runner),
        ):
            return host.execute_capture(
                plan, environment={"HEAPTRACK_UNWIND_DEPTH": "999", "LD_PRELOAD": "/unowned.so"}
            )

    def test_plan_is_read_only_and_retains_explicit_native_library_without_launch(self):
        with patch.object(host, "Session") as child:
            plan = self.plan()
        child.assert_not_called()
        self.assertFalse(self.out.exists())
        self.assertEqual(plan["launch_only_environment"]["LD_PRELOAD"], str(self.preload))
        self.assertEqual(plan["qualification"], "plan_only")
        self.assertEqual(plan["identity_query"], [str(self.binary), "--build-identity"])
        for seconds in (None, True, 0, 30, float("nan"), float("inf"), 10**1000):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.plan(total_seconds=seconds)

    def test_actual_raw_decoder_and_shared_capture_binding_preserve_allocation_diagnostic_role(
        self,
    ):
        self.assertEqual(self.execute(), 0)
        result = json.loads((self.out / "host-heap.json").read_text())
        allocations = json.loads((self.out / "allocations.json").read_text())
        self.assertEqual(result["benchmark_pid"], 123)
        self.assertEqual(allocations["summary"]["requested_bytes"], 32)
        self.assertEqual(allocations["execution_id"], result["execution_id"])
        stacks = allocations["recorded_stacks"]
        self.assertEqual(stacks["roots"][0]["requested_bytes"], 32)
        self.assertEqual(stacks["frames"][0]["allocation_module_generation"], 1)
        self.assertEqual(stacks["frames"][0]["recorded_module"]["state"], "unique_recorded_module")
        self.assertFalse(stacks["loaded_module_bytes_verified"])
        self.assertFalse(result["production_latency"])
        bound = json.loads((self.out / "profiled.json").read_text())
        self.assertEqual(bound["external_instrumentation"]["mode"], "host_heap_interception")
        self.assertNotIn(
            "external_instrumentation", json.loads((self.out / "benchmark.json").read_text())
        )
        self.assertNotIn("HEAPTRACK_UNWIND_DEPTH", self.calls[-1][2]["env"])

    def test_failed_process_preserves_receipt_and_never_publishes_allocation_evidence(self):
        self.fail = True
        self.assertEqual(self.execute(), 1)
        self.assertFalse((self.out / "allocations.json").exists())
        result = json.loads((self.out / "host-heap.json").read_text())
        self.assertEqual(result["qualification"], "incomplete")
        self.assertFalse(result["stages"][0]["succeeded"])

    def test_wrong_native_pid_and_changed_preload_cannot_bind_success(self):
        for mutation in ("wrong_pid", "mutate"):
            with self.subTest(mutation=mutation):
                self.out = self.root / mutation
                setattr(self, mutation, True)
                self.assertEqual(self.execute(), 1)
                self.assertFalse((self.out / "allocations.json").exists())
                self.assertTrue((self.out / "host-heap.json").exists())
                setattr(self, mutation, False)


if __name__ == "__main__":
    unittest.main()
