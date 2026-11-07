"""Authored native CPU/thread observations; no ETL, profiler or decoder runs."""

from __future__ import annotations

import hashlib
import unittest
from unittest.mock import patch

from etl_cpu_contract import cpu_catalog
from etl_process_contract import process_contract
from etl_range_contract import operation_ranges
import etl_inspector_identity as inspector
from test_etl_process_contract import native_metadata
from test_etl_range_contract import marker_metadata, windows_snapshot


def thread(opcode, stamp, ordinal, *, tid=42):
    return dict(
        pid=333,
        tid=tid,
        opcode=opcode,
        time_filetime_100ns=stamp,
        record_index=ordinal,
        pointer_bytes=8,
        event_version=3,
    )


def sample(ordinal, stamp, *, header_pid=2**32 - 1, tid=42, flags=0):
    return dict(
        tid=tid,
        native_header_pid=header_pid,
        native_header_tid=2**32 - 1,
        time_filetime_100ns=stamp,
        instruction_pointer=0x180001234,
        pointer_bytes=8,
        packed_count_flags_rank=1 + (flags << 16),
        count=1,
        sample_flags=flags,
        rank=0,
        processor=3,
        processor_index_mode=True,
        record_index=ordinal,
        wire_contract="windows_kernel_SampledProfile_v2",
    )


def cpu_metadata():
    value = marker_metadata()
    value.update(
        observed_events=20,
        sample_walk_events=20,
        observed_thread_events=2,
        thread_events=[thread(1, 250, 7), thread(2, 850, 8)],
        observed_sample_events=2,
        sample_events=[sample(9, 450), sample(10, 650)],
    )
    return value


def read(metadata, *, ranges=False):
    process = process_contract(
        metadata,
        pid=333,
        parent_pid=444,
        executable="app.exe",
        decoder_pid=555,
        source_sha256="a" * 64,
        file_bytes=4096,
    )
    snapshot = windows_snapshot() if ranges else None
    windows = operation_ranges(metadata, process, snapshot) if ranges else None
    return cpu_catalog(metadata, process, snapshot=snapshot, native_ranges=windows)


class NativeCpu(unittest.TestCase):
    def test_native_points_bind_threads_and_marker_phases_without_WPA_clock_alignment(self):
        metadata = cpu_metadata()
        metadata["sample_events"].append(sample(11, 250))
        metadata["observed_sample_events"] = 3
        value = read(metadata, ranges=True)
        self.assertEqual(len(value["samples"]), 3)
        self.assertEqual({row["thread_generation_id"] for row in value["samples"]}, {7})
        self.assertEqual(value["selected_operation"]["sample_count"], 2)
        self.assertEqual(
            [row["phase"] for row in value["selected_operation"]["samples"]],
            ["proof", "public_call"],
        )
        self.assertEqual(value["selected_operation"]["outside_or_unbound_thread_samples"], 1)
        self.assertIsNone(value["CPU_duration_ns"])
        self.assertIsNone(value["weighted_sampling_exposure_ns"])
        self.assertFalse(value["loaded_symbols_verified"])
        self.assertFalse(value["promotion_allowed"])

    def test_late_records_and_known_PID_reused_thread_generations_are_distinct(self):
        value = cpu_metadata()
        value["thread_events"] = [
            thread(2, 850, 12),
            thread(1, 600, 11),
            thread(2, 500, 8),
            thread(1, 250, 7),
        ]
        value["observed_thread_events"] = 4
        value["sample_events"] = [sample(9, 450, header_pid=333), sample(10, 650, header_pid=333)]
        report = read(value)
        self.assertEqual([row["thread_generation_id"] for row in report["samples"]], [7, 11])
        value["sample_events"][0]["native_header_pid"] = 2**32 - 1
        with self.assertRaises(ValueError):
            read(value)

    def test_missing_lifetimes_interrupt_samples_and_undecoded_threads_remain_unresolved(self):
        for mode in (
            "missing",
            "interrupt",
            "unknown_flags",
            "header_tid",
            "undecoded",
            "loss",
            "rundown",
        ):
            value = cpu_metadata()
            if mode == "missing":
                value["thread_events"].pop()
                for row in value["sample_events"]:
                    row["native_header_pid"] = 333
            elif mode in {"interrupt", "unknown_flags"}:
                value["sample_events"] = [
                    sample(9, 450, flags=1 if mode == "interrupt" else 4),
                    sample(10, 650, flags=1 if mode == "interrupt" else 4),
                ]
            elif mode == "header_tid":
                for row in value["sample_events"]:
                    row["native_header_tid"] = 43
            elif mode == "undecoded":
                value["observed_thread_events"] = 3
                value["undecoded_thread_events"] = 1
            elif mode == "loss":
                value["header"]["events_lost"] = 1
            else:
                value["thread_events"] = [thread(3, 250, 7), thread(4, 850, 8)]
                for row in value["sample_events"]:
                    row["native_header_pid"] = 333
            with self.subTest(mode=mode):
                report = read(value)
                self.assertEqual(report["samples"], [])
                self.assertTrue(report["unresolved"])
                self.assertFalse(report["complete_selected_sample_coverage_proven"])

    def test_raw_width_packed_fields_process_record_counts_and_namespaces_reject_drift(self):
        for mode in ("width", "packed", "pid", "ordinal", "walk", "count", "mode", "namespace"):
            value = cpu_metadata()
            if mode == "width":
                value["sample_events"][0]["pointer_bytes"] = 4
            elif mode == "packed":
                value["sample_events"][0]["count"] = 2
            elif mode == "pid":
                value["sample_events"][0]["native_header_pid"] = 334
            elif mode == "ordinal":
                value["sample_events"][1]["record_index"] = 9
            elif mode == "walk":
                value["sample_walk_events"] = 19
            elif mode == "count":
                value["observed_sample_events"] = True
            elif mode == "mode":
                value["sample_events"][0]["processor_index_mode"] = 1
            else:
                value["sample_record_namespace"] = "first_native_event_walk"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                read(value)

    def test_empty_native_projection_does_not_invent_zero_CPU_duration(self):
        value = read(native_metadata())
        self.assertEqual(value["state"], "no_resolved_native_CPU_points")
        self.assertIsNone(value["CPU_duration_ns"])
        self.assertIsNone(value["selected_operation"])

    def test_compiled_reader_identity_includes_exact_payload_helper_contents(self):
        with patch.object(inspector, "file_digest", side_effect=["a" * 64, "b" * 64]):
            expected = hashlib.sha256(("a" * 64 + "\n" + "b" * 64 + "\n").encode()).hexdigest()
            self.assertEqual(inspector.source_identity(), expected)
        with patch.object(inspector, "file_digest", side_effect=["a" * 64, "c" * 64]):
            self.assertNotEqual(inspector.source_identity(), expected)


if __name__ == "__main__":
    unittest.main()
