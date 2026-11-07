"""Authored native heap generation controls; no ETL or Windows API runs."""

from __future__ import annotations

import unittest

from etl_cpu_contract import cpu_catalog
from etl_process_contract import process_contract
from etl_heap_contract import heap_catalog
from etl_range_contract import operation_ranges
from test_etl_cpu_contract import cpu_metadata
from test_etl_range_contract import windows_snapshot


def event(opcode, stamp, ordinal, words, *, tail=1):
    return dict(
        wire_contract="windows_kernel_HeapTrace_v0",
        opcode=opcode,
        time_filetime_100ns=stamp,
        raw_time_qpc=stamp * 10,
        record_index=ordinal,
        pointer_bytes=8,
        native_header_pid=333,
        native_header_tid=42,
        payload_words=words,
        tail=None if opcode == 35 else tail,
    )


def heap_metadata():
    value = cpu_metadata()
    value.update(
        observed_events=40,
        sample_walk_events=40,
        observed_heap_events=7,
        raw_walk_events=40,
        raw_source_point_count=9,
        heap_events=[
            event(32, 300, 11, [0x80]),
            event(33, 350, 12, [0x80, 64, 0x1000]),
            event(34, 450, 13, [0x80, 0x2000, 0x1000, 96, 64]),
            event(36, 550, 14, [0x80, 0x2000]),
            event(33, 600, 15, [0x80, 32, 0x1000]),
            event(35, 650, 16, [0x80]),
            event(36, 700, 17, [0x80, 0x3333]),
        ],
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
    native = operation_ranges(metadata, process, snapshot) if ranges else None
    cpu = cpu_catalog(metadata, process, snapshot=snapshot, native_ranges=native)
    return heap_catalog(metadata, process, cpu, snapshot=snapshot, native_ranges=native)


class NativeHeap(unittest.TestCase):
    def test_realloc_pointer_reuse_destroy_and_phase_counts_preserve_domains(self):
        report = read(heap_metadata(), ranges=True)
        self.assertEqual([r["generation_id"] for r in report["allocations"]], [12, 13, 15])
        self.assertEqual([r["heap_generation_id"] for r in report["allocations"]], [11, 11, 11])
        self.assertEqual(
            [r["end_reason"] for r in report["allocations"]],
            ["reallocation", "free", "heap_destroy"],
        )
        self.assertEqual(
            report["summary"]["conditional_tracked_live_boundary_high_water_bytes"], 96
        )
        self.assertEqual(report["summary"]["conditional_tracked_retained_bytes"], 0)
        self.assertEqual(report["summary"]["native_reported_allocated_bytes"], 192)
        self.assertEqual(report["summary"]["within_selected_operation_generation_count"], 3)
        self.assertEqual(report["allocations"][1]["allocation_phase"]["phase"], "proof")
        self.assertEqual(report["allocations"][0]["observed_event_lifetime_ns"], 10000)
        self.assertEqual(report["unresolved"][0]["pointer"], 0x3333)
        self.assertFalse(report["complete_heap_history_proven"])
        self.assertFalse(report["zero_allocation_behavior_proven"])
        self.assertIsNone(report["exact_process_peak_bytes"])
        self.assertIsNone(report["device_or_pinned_bytes"])

    def test_late_records_and_multiheap_peak_use_combined_native_timestamp_boundaries(self):
        value = heap_metadata()
        value["heap_events"] += [
            event(33, 400, 18, [0x90, 80, 0x1000]),
            event(36, 500, 19, [0x90, 0x1000]),
        ]
        value["observed_heap_events"] = len(value["heap_events"])
        value["heap_events"].reverse()
        report = read(value)
        self.assertEqual(
            report["summary"]["conditional_tracked_live_boundary_high_water_bytes"], 176
        )
        self.assertEqual(report["summary"]["conditional_tracked_retained_bytes"], 0)
        self.assertIsNone(report["allocations"][-1]["heap_generation_id"])

    def test_heap_reuse_after_destroy_is_a_new_observed_generation(self):
        value = heap_metadata()
        value["heap_events"] += [event(32, 710, 18, [0x80]), event(33, 720, 19, [0x80, 24, 0x1000])]
        value["observed_heap_events"] = len(value["heap_events"])
        report = read(value)
        self.assertEqual(report["allocations"][-1]["heap_generation_id"], 18)
        self.assertEqual(report["summary"]["conditional_tracked_retained_bytes"], 24)

    def test_missing_or_ambiguous_free_and_failed_realloc_do_not_invent_lifetimes(self):
        for kind in ("replace", "same_time", "failed_realloc", "old_size"):
            value = heap_metadata()
            if kind == "replace":
                value["heap_events"].insert(2, event(33, 400, 18, [0x80, 20, 0x1000]))
            elif kind == "same_time":
                value["heap_events"][2]["time_filetime_100ns"] = 350
            elif kind == "failed_realloc":
                value["heap_events"][2]["payload_words"][1] = 0
            else:
                value["heap_events"][2]["payload_words"][4] = 63
            value["observed_heap_events"] = len(value["heap_events"])
            report = read(value)
            reasons = {r["reason"] for r in report["unresolved"]}
            with self.subTest(kind=kind):
                expected = {
                    "replace": "replaced",
                    "same_time": "same_time",
                    "failed_realloc": "null_reallocation",
                    "old_size": "old_native_size",
                }[kind]
                self.assertTrue(any(expected in reason for reason in reasons))
                self.assertFalse(report["complete_heap_history_proven"])

    def test_lost_unsupported_and_empty_capture_never_establish_zero_allocations(self):
        for kind in ("empty", "undecoded", "unsupported", "loss"):
            value = heap_metadata()
            if kind == "empty":
                value.update(heap_events=[], observed_heap_events=0)
            elif kind == "undecoded":
                value["undecoded_heap_events"] = 1
                value["observed_heap_events"] += 1
            elif kind == "unsupported":
                value["unsupported_heap_events"] = 1
                value["observed_heap_events"] += 1
            else:
                value["header"]["events_lost"] = 1
            report = read(value)
            self.assertFalse(report["zero_allocation_behavior_proven"])
            self.assertFalse(report["complete_heap_history_proven"])
            if kind in {"empty", "loss"}:
                self.assertFalse(report["allocations"])

    def test_raw_accounting_PID_width_payload_and_identity_substitution_are_rejected(self):
        for kind in (
            "pid",
            "duplicate",
            "sample_alias",
            "word_bool",
            "width_bool",
            "size_overflow",
            "wrong_words",
            "tail_bool",
            "filter",
            "namespace",
            "count",
        ):
            value = heap_metadata()
            row = value["heap_events"][0]
            if kind == "pid":
                row["native_header_pid"] = 999
            elif kind == "duplicate":
                value["heap_events"][1]["record_index"] = row["record_index"]
            elif kind == "sample_alias":
                row["record_index"] = 9
            elif kind == "word_bool":
                row["payload_words"][0] = True
            elif kind == "width_bool":
                row["pointer_bytes"] = True
            elif kind == "size_overflow":
                row.update(pointer_bytes=4, payload_words=[2**32])
            elif kind == "wrong_words":
                row["payload_words"].append(1)
            elif kind == "tail_bool":
                row["tail"] = True
            elif kind == "filter":
                value["heap_filter"] = "matching_TID"
            elif kind == "namespace":
                value["heap_record_namespace"] = "first_native_event_walk"
            else:
                value["observed_heap_events"] += 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                read(value)


if __name__ == "__main__":
    unittest.main()
