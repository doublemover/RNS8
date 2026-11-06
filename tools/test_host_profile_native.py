#!/usr/bin/env python3
"""Synthetic native-perf API controls; these are not measured profiles."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import host_profile_native as native
from test_execution_observation import capture as execution_capture
from test_perf_recording_contract import recording_bytes
from test_elf_module_identity import image as elf_image

ROOT = Path(__file__).resolve().parents[1]


def sample(pid=41):
    return SimpleNamespace(
        sample_pid=pid,
        sample_tid=42,
        sample_time=123456789,
        sample_cpu=3,
        sample_period=1000000,
        sample_ip=4099,
        machine_pid=0,
        evsel="evsel(cpu-clock:u)",
        symbol="exact_product",
        sym_start=4096,
        sym_end=8192,
        sym_offset=3,
        map_start=4096,
        map_end=8192,
        map_pgoff=0,
        dso="librns8.so",
        dso_long_name="/fixture/librns8.so",
        dso_bid="1234abcd",
        srccode=lambda: ("/fixture/source.cpp", 17, "return value;"),
        callchain=[SimpleNamespace(ip=4099, symbol="exact_product", dso="librns8.so")],
    )


class NativePerf(unittest.TestCase):
    def test_native_leaf_build_id_binds_actual_ELF_without_claiming_source_contents(self):
        library = self.root / "library.so"
        library.write_bytes(elf_image())
        self.rows = [sample()]
        self.rows[0].dso_long_name = str(library)
        self.rows[0].dso_bid = "0123abcd"
        report = self.decode()
        custody = report["leaf_module_custody"]
        self.assertTrue(custody["complete"])
        self.assertEqual(custody["matched_leaf_source_location_samples"], 1)
        self.assertFalse(custody["loaded_image_history_verified"])
        self.assertEqual(custody["modules"][0]["file"]["path"], str(library.resolve()))
        self.rows[0].dso_bid = "different"
        with self.assertRaises(ValueError):
            self.decode()
        self.rows[0].dso_bid = ""
        self.assertFalse(self.decode()["leaf_module_custody"]["complete"])

    def setUp(self):
        root = ROOT / "temp/host-profile-native-controls"
        root.mkdir(parents=True, exist_ok=True)
        scratch = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.data = self.root / "perf.data"
        self.module = self.root / "perf.synthetic.so"
        self.module.write_bytes(b"synthetic native module fixture")
        self.rows = [sample(), sample(99)]
        self.sessions = 0
        self.write_recording()

    def write_recording(self):
        rows = []
        for sample in self.rows:
            row = {}
            for name, attr, default, bits in (
                ("pid", "sample_pid", 41, 32),
                ("tid", "sample_tid", 42, 32),
                ("time_ns", "sample_time", 123456789, 64),
                ("period", "sample_period", 1000000, 64),
                ("ip", "sample_ip", 4099, 64),
                ("cpu", "sample_cpu", 3, 32),
            ):
                value = getattr(sample, attr)
                valid = type(value) is int and 0 <= value < 1 << bits
                if name in {"pid", "tid", "period"} and not value:
                    valid = False
                row[name] = value if valid else default
            rows.append(row)
        self.data.write_bytes(recording_bytes(samples=rows))

    def binding(self):
        def session(data, *, sample):
            self.sessions += 1

            def process_events():
                for row in self.rows:
                    sample(row)

            return SimpleNamespace(process_events=process_events, e_machine=62, is_64_bit=True)

        return SimpleNamespace(
            __file__=str(self.module),
            data=lambda path: path,
            session=session,
            sample_event=type("SyntheticSampleType", (), dict.fromkeys(native.SAMPLE_ATTRIBUTES)),
        )

    def decode(self, **kwargs):
        self.write_recording()
        return native.decode(self.data, 41, perf_module=self.binding(), **kwargs)

    def test_process_symbols_period_and_recorded_loss_remain_separate(self):
        report = self.decode()
        self.assertEqual(report["observed_samples"], 2)
        self.assertFalse(report["leaf_module_custody"]["complete"])
        self.assertEqual(report["leaf_module_custody"]["matched_build_id_samples"], 0)
        self.assertEqual(report["excluded_other_process_samples"], 1)
        self.assertEqual(report["selected_process_samples"], 1)
        row = report["samples"][0]
        self.assertEqual((row["pid"], row["tid"], row["time_ns"]), (41, 42, 123456789))
        self.assertEqual(row["period_unit"], "cpu_clock_nanoseconds")
        self.assertEqual(row["source"]["line"], 17)
        self.assertEqual(row["dso_build_id"], "1234abcd")
        self.assertEqual(row["callchain"][0]["symbol"], "exact_product")
        self.assertFalse(report["production_latency"])
        self.assertFalse(report["acquisition_verified"])
        self.assertFalse(report["promotion_allowed"])
        self.assertEqual(report["loss_status"], "no_known_loss_observed_complete_record_walk")
        self.assertFalse(
            report["recording_contract"]["loss"]["complete_loss_free_acquisition_proven"]
        )
        self.assertEqual(report["clock"]["attribute_clock_id"], 1)

    def test_valid_looking_decoded_fields_cannot_disagree_with_raw_samples(self):
        original = copy.deepcopy(self.rows)
        for field in (
            "sample_pid",
            "sample_tid",
            "sample_ip",
            "sample_time",
            "sample_period",
            "sample_cpu",
        ):
            self.rows = copy.deepcopy(original)
            setattr(self.rows[0], field, getattr(self.rows[0], field) + 1)
            with self.subTest(field=field), self.assertRaises(ValueError):
                native.decode(self.data, 41, perf_module=self.binding())
        self.rows = original

    def test_unresolved_symbols_are_missing_without_invented_attribution(self):
        row = self.rows[0]
        row.symbol = row.sym_start = row.sym_end = row.sym_offset = None
        row.dso_bid = row.callchain = None
        row.srccode = lambda: None
        report = self.decode()
        self.assertEqual(report["symbolized_samples"], 0)
        self.assertEqual(report["source_attributed_samples"], 0)
        self.assertIsNone(report["samples"][0]["symbol"])
        self.assertEqual(report["samples"][0]["callchain"], [])

    def test_thread_identity_and_partial_source_locations_are_not_collapsed(self):
        first, second = sample(), sample()
        second.sample_tid = 43
        second.sample_time += 1
        second.srccode = lambda: (None, 0, None)
        self.rows = [first, second]
        report = self.decode()
        self.assertEqual([row["tid"] for row in report["samples"]], [42, 43])
        self.assertEqual(report["source_attributed_samples"], 1)
        self.assertEqual(report["scope"], "selected_process_total_not_selected_operation")

    def test_native_fields_and_callchain_are_bounded_before_publication(self):
        original = self.rows
        for field, value in (
            ("sample_pid", True),
            ("sample_period", 0),
            ("sample_time", -1),
            ("sample_cpu", 2**64),
            ("machine_pid", 9),
            ("evsel", "evsel(cycles:u)"),
            ("symbol", "x" * 4097),
            ("sym_start", True),
            ("map_start", 4100),
            ("map_end", 4099),
            ("map_pgoff", 1 << 64),
            ("callchain", [SimpleNamespace(ip=1, symbol=None, dso=None)] * 129),
        ):
            self.rows = copy.deepcopy(original)
            setattr(self.rows[0], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.decode()
        self.rows = original

    def test_observed_and_decoded_budgets_include_excluded_samples(self):
        for limits in (
            dict(max_samples=1),
            dict(max_decoded_bytes=1),
            dict(max_input_bytes=1),
            dict(max_samples=True),
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.decode(**limits)
        self.rows = [sample(99)]
        with self.assertRaisesRegex(ValueError, "no selected-process"):
            self.decode()

    def test_input_or_native_module_mutation_rejects_the_receipt(self):
        for path in (self.data, self.module):
            original = path.read_bytes()
            row = sample()
            row.srccode = lambda path=path: path.write_bytes(b"changed") and None
            self.rows = [row]
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, "changed"):
                self.decode()
            path.write_bytes(original)

    def test_missing_native_interface_and_existing_output_are_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, "standalone"):
            native.decode(self.data, 41, perf_module=SimpleNamespace(data=lambda path: path))
        output = self.root / "profile.json"
        output.write_text("preserved")
        with (
            patch.object(native, "decode", side_effect=AssertionError("decode")),
            patch("builtins.print"),
        ):
            self.assertEqual(
                native.main(["--input", str(self.data), "--pid", "41", "--output", str(output)]), 1
            )
        self.assertEqual(output.read_text(), "preserved")
        output = self.root / "new-profile.json"
        with patch.object(native, "decode", return_value=dict(schema=native.SCHEMA)):
            self.assertEqual(
                native.main(["--input", str(self.data), "--pid", "41", "--output", str(output)]), 0
            )
        self.assertEqual(json.loads(output.read_text())["schema"], native.SCHEMA)

    def test_capability_query_does_not_decode_and_keeps_unavailable_distinct(self):
        selected = (ROOT / ".python-version").read_text().strip()
        with (
            patch.object(native.platform, "system", return_value="Linux"),
            patch.object(native.platform, "python_version", return_value=selected),
            patch.object(native.importlib, "import_module", return_value=self.binding()),
        ):
            self.assertEqual(native.capabilities()["state"], "available")
            self.assertEqual(self.sessions, 0)
        missing = self.binding()
        missing.sample_event = type("MissingSampleFields", (), {})
        with (
            patch.object(native.platform, "system", return_value="Linux"),
            patch.object(native.platform, "python_version", return_value=selected),
            patch.object(native.importlib, "import_module", return_value=missing),
        ):
            value = native.capabilities()
            self.assertEqual(value["state"], "unsupported")
            self.assertIn("map_start", value["missing_sample_attributes"])
            self.assertEqual(self.sessions, 0)
        with (
            patch.object(native.platform, "system", return_value="Linux"),
            patch.object(native.platform, "python_version", return_value=selected),
            patch.object(
                native.importlib, "import_module", side_effect=ImportError("binding missing")
            ),
        ):
            self.assertEqual(native.capabilities()["state"], "unavailable")
        with (
            patch.object(native.platform, "system", return_value="Windows"),
            patch.object(native.importlib, "import_module", side_effect=AssertionError("import")),
        ):
            self.assertEqual(native.capabilities()["state"], "unsupported")

    def test_native_benchmark_pid_binding_is_explicit_and_content_bound(self):
        capture = execution_capture()
        capture["process_id"] = 41
        path = self.root / "benchmark.json"
        path.write_text(json.dumps(capture))
        output = self.root / "selected.json"
        with patch.object(native, "decode", return_value=dict(schema=native.SCHEMA)) as decode:
            self.assertEqual(
                native.main(
                    ["--input", str(self.data), "--capture", str(path), "--output", str(output)]
                ),
                0,
            )
        self.assertEqual(decode.call_args.args[1], 41)
        report = json.loads(output.read_text())
        self.assertEqual(report["profiled_capture"]["sha256"], native.file_digest(path))
        capture["process_id"] = True
        path.write_text(json.dumps(capture))
        with (
            patch.object(native, "decode", side_effect=AssertionError("decode")),
            patch("builtins.print"),
        ):
            self.assertEqual(
                native.main(
                    [
                        "--input",
                        str(self.data),
                        "--capture",
                        str(path),
                        "--output",
                        str(self.root / "invalid.json"),
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
