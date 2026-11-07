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
from test_etl_process_contract import native_metadata
from test_etl_image_contract import image
from test_etl_range_contract import windows_snapshot
from test_etl_cpu_contract import cpu_metadata
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
        (self.binary.parent / "rns8-etl-inspect.exe").write_bytes(executable())
        self.out = self.root / "session"
        self.calls, self.fail = [], None
        self.ranges = False

    def plan(self, **options):
        return host.plan_capture(
            self.fixture.capture_path, self.wpr, self.exporter, self.pdb, self.out, **options
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
        elif name == "inspector_identity":
            result["stdout"] = json.dumps(
                dict(
                    schema="rns8_etl_inspector_identity_v1",
                    process_id=333,
                    source_sha256=host.etl_source_identity(),
                    pointer_bytes=8,
                )
            )
        elif name == "inspect_etl":
            self.assertEqual(command[-2:], ["--pid", "333"])
            metadata = native_metadata(
                parent_pid=host.os.getpid(),
                decoder_pid=333,
                image=self.binary.name,
                source_sha256=host.etl_source_identity(),
                file_bytes=(self.out / "raw/profile.etl").stat().st_size,
            )
            metadata.update(
                observed_image_events=2,
                image_events=[
                    image(image_path_utf8=str(self.binary)),
                    image(2, 800, image_path_utf8=str(self.binary)),
                ],
            )
            if self.ranges:
                markers = cpu_metadata()
                for key in (
                    "observed_events",
                    "observed_marker_events",
                    "undecoded_marker_events",
                    "marker_events",
                    "observed_thread_events",
                    "undecoded_thread_events",
                    "thread_events",
                    "sample_walk_events",
                    "observed_sample_events",
                    "undecoded_sample_events",
                    "excluded_sample_events",
                    "sample_events",
                ):
                    metadata[key] = markers[key]
            result["stdout"] = json.dumps(metadata)
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
            if self.ranges:
                value["profiler_ranges"] = windows_snapshot()
                for key in ("before", "after"):
                    value["execution_environment_witness"][key] = dict(
                        value["execution_environment_witness"][key], RNS8_PROFILE_RANGES="1"
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
        self.ranges = plan["operation_ranges"]
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
        self.assertTrue(result["selected_process_identity"]["process_identity_complete"])
        images = json.loads(Path(result["native_image_catalog"]["path"]).read_text())
        self.assertEqual(images["selected_image_events"], 2)
        self.assertEqual(images["generations"][0]["observed_lifetime_100ns"], 590)
        self.assertFalse(images["loaded_PDB_identity_verified"])
        self.assertEqual([name for name, _, _ in self.calls][-1], "export")

    def review(self, *, change=None):
        def decoder(command, **policy):
            self.assertEqual(
                command, self.plan_for_review["commands"]["inspect_etl"] + ["--pid", "333"]
            )
            self.assertEqual(policy["label"], "windows-host-review:inspect_etl")
            value = json.loads(
                json.loads((self.out / "inspect_etl.process.json").read_text())["stdout"]
            )
            value["decoder_pid"] = 777
            if change:
                change(value)
            return dict(
                command=command,
                process_id=777,
                returncode=0,
                timed_out=False,
                stdout=json.dumps(value),
                stderr="",
            )

        with patch.object(host.platform, "system", return_value="Windows"):
            return host.review_capture(self.out / "plan.json", runner=decoder)

    def test_read_only_review_replays_custody_from_a_different_process(self):
        self.plan_for_review = self.plan()
        self.assertEqual(self.execute(self.plan_for_review), 0)
        before = {str(path): file_digest(path) for path in self.out.rglob("*") if path.is_file()}
        with patch.object(host.os, "getpid", return_value=999):
            result = self.review()
        after = {str(path): file_digest(path) for path in self.out.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result["decoder_process_id"], 777)
        self.assertFalse(result["recording_started"])
        self.assertFalse(result["workload_started"])
        self.assertFalse(result["outputs_rewritten"])
        self.assertFalse(result["promotion_allowed"])

    def test_review_rejects_resealed_stage_summary_derivation_and_fresh_decode_changes(self):
        self.plan_for_review = self.plan()
        self.assertEqual(self.execute(self.plan_for_review), 0)
        originals = {path: path.read_bytes() for path in self.out.rglob("*") if path.is_file()}
        for mode in (
            "stage_command",
            "stage_status",
            "owner",
            "owner_null",
            "owner_missing",
            "derived",
            "fresh",
            "tool",
        ):
            with self.subTest(mode=mode):
                report_path = self.out / "host-profile.json"
                report = json.loads(report_path.read_text())
                if mode.startswith("stage_"):
                    path = self.out / "workload.process.json"
                    stage = json.loads(path.read_text())
                    if mode == "stage_command":
                        stage["command"] += ["--foreign"]
                    else:
                        stage["returncode"] = 1
                    write(path, stage)
                    next(row for row in report["stages"] if row["name"] == "workload")[
                        "receipt"
                    ] = host.reference(path)
                elif mode == "owner":
                    report["recording_owner_pid"] += 1
                elif mode == "owner_null":
                    report["recording_owner_pid"] = None
                elif mode == "owner_missing":
                    report.pop("recording_owner_pid")
                elif mode == "derived":
                    path = self.out / "samples.json"
                    samples = json.loads(path.read_text())
                    samples["samples"][0]["function"] = "forged_source"
                    write(path, samples)
                    report["samples"] = host.reference(path)
                elif mode == "tool":
                    tool = Path(self.plan_for_review["tools"]["inspector"]["path"])
                    previous_tool = tool.read_bytes()
                    tool.write_bytes(previous_tool + b"changed")
                write(report_path, host.seal(report))
                try:
                    with self.assertRaises(ValueError):
                        self.review(
                            change=(lambda value: value.update(observed_events=11))
                            if mode == "fresh"
                            else None
                        )
                finally:
                    for path, raw in originals.items():
                        path.write_bytes(raw)
                    if mode == "tool":
                        tool.write_bytes(previous_tool)

    def test_native_marker_path_is_opt_in_and_requires_same_compiled_provider(self):
        self.assertFalse(self.plan()["operation_ranges"])
        with self.assertRaisesRegex(ValueError, "compiled with ETW"):
            self.plan(operation_ranges=True)
        value = copy.deepcopy(self.fixture.capture)
        value["build_manifest"]["configuration"] = (
            value["build_manifest"]["configuration"].replace(
                "synthetic fixture", "fixture=authored"
            )
            + "\netw_ranges=ON\n"
        )
        write(self.fixture.capture_path, value)
        with patch.object(host, "production_profile_input", return_value=self.binary):
            plan = self.plan(operation_ranges=True)
            self.assertEqual(plan["expected_native_environment"]["RNS8_PROFILE_RANGES"], "1")
            self.assertEqual(plan, host.rebuild(plan))
        self.assertFalse(self.out.exists())

    def test_actual_synthetic_marker_publication_uses_native_windows_and_raw_snapshot(self):
        manifest = self.fixture.capture["build_manifest"]
        manifest["configuration"] = (
            manifest["configuration"].replace("synthetic fixture", "fixture=authored")
            + "\netw_ranges=ON\n"
        )
        write(self.fixture.capture_path, self.fixture.capture)
        identity_path = Path(self.fixture.binary_receipts["rns8-bench"]["identity_document"])
        identity = json.loads(identity_path.read_text())
        identity.update(build_manifest=manifest, library_build_manifest=manifest)
        write(identity_path, identity)
        self.assertEqual(self.execute(self.plan(operation_ranges=True)), 0)
        result = json.loads((self.out / "host-profile.json").read_text())
        ranges = json.loads(Path(result["native_operation_ranges"]["path"]).read_text())
        self.assertEqual(result["measured_operation_range"], "native_measured_windows_bound")
        self.assertEqual(
            ranges["phase_attribution"]["threads"][0]["exclusive_phase_wall_ns"]["proof"], 10_000
        )
        self.assertEqual(ranges["wpa_relative_clock_join"], "unbound")
        cpu = json.loads(Path(result["native_cpu_observations"]["path"]).read_text())
        self.assertEqual(cpu["selected_operation"]["sample_count"], 2)
        self.assertEqual(
            [row["phase"] for row in cpu["selected_operation"]["samples"]], ["proof", "public_call"]
        )
        self.assertIsNone(cpu["CPU_duration_ns"])
        self.assertFalse(result["promotion_allowed"])

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
