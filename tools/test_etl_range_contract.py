"""Authored actual private ETW wire observations; no native ETL or tool runs."""

from __future__ import annotations

import unittest

from etl_range_contract import operation_ranges
from etl_process_contract import process_contract
from test_etl_process_contract import native_metadata
from test_host_profile_ranges import scope, snapshot


def windows_snapshot():
    value = snapshot([scope(thread=42), scope(1, 0, start=120, end=180, phase=2, thread=42)])
    value.update(
        marker_label_contract="rns8_etw_slot_v1",
        native_write_failures=0,
        host_clock=dict(
            schema="rns8_windows_host_range_clock_v1",
            name="QueryPerformanceCounter",
            unit="nanoseconds",
            frequency_hz=10_000_000,
            timestamp_scope="brackets_around_native_ETW_calls",
        ),
    )
    for row in value["records"]:
        row["process_id"] = 333
    return value


def marker(ordinal, slot, event, time, *, tid=42):
    return dict(
        record_index=ordinal,
        slot=slot,
        event_id=event,
        time_filetime_100ns=time,
        pid=333,
        tid=tid,
        wire_contract="rns8_etw_slot_v1",
    )


def marker_metadata():
    value = native_metadata()
    value.update(
        observed_marker_events=4,
        marker_events=[
            marker(3, 0, 1, 300),
            marker(4, 1, 1, 400),
            marker(5, 1, 2, 500),
            marker(6, 0, 2, 700),
        ],
    )
    return value


def bind(metadata, ranges):
    process = process_contract(
        metadata,
        pid=333,
        parent_pid=444,
        executable="app.exe",
        decoder_pid=555,
        source_sha256="a" * 64,
        file_bytes=4096,
    )
    return operation_ranges(metadata, process, ranges)


class NativeRanges(unittest.TestCase):
    def test_exact_native_windows_and_phases_do_not_align_wpa_or_gpu(self):
        value = bind(marker_metadata(), windows_snapshot())
        self.assertEqual(value["state"], "native_measured_windows_bound")
        self.assertEqual(value["measured_root_ids"], [0])
        self.assertEqual(value["markers"][0]["start_ns"], 10_000)
        self.assertEqual(value["markers"][0]["end_ns"], 50_000)
        self.assertEqual(
            value["phase_attribution"]["threads"][0]["exclusive_phase_wall_ns"],
            {"proof": 10_000, "public_call": 30_000},
        )
        self.assertEqual(value["wpa_relative_clock_join"], "unbound")
        self.assertEqual(value["host_QPC_to_ETL_clock_join"], "unbound")
        self.assertEqual(value["GPU_clock_join"], "unbound")
        self.assertFalse(value["promotion_allowed"])

    def test_missing_duplicate_pending_and_undecoded_markers_stay_partial(self):
        for mode in ("missing", "duplicate", "pending", "undecoded"):
            metadata, ranges = marker_metadata(), windows_snapshot()
            if mode == "missing":
                metadata["marker_events"].pop()
            elif mode == "duplicate":
                metadata["marker_events"].append(marker(7, 0, 2, 710))
                metadata["observed_marker_events"] += 1
            elif mode == "pending":
                ranges["records"][1].update(
                    stop_state=0, stop_thread_id=None, stop_before_ns=None, stop_after_ns=None
                )
            else:
                metadata["observed_marker_events"] += 1
                metadata["undecoded_marker_events"] = 1
            with self.subTest(mode=mode):
                value = bind(metadata, ranges)
                self.assertEqual(value["state"], "partial_native_marker_association")
                self.assertIsNone(value["phase_attribution"])

    def test_forged_slot_process_thread_clock_counts_and_native_parent_rejected(self):
        for mode in (
            "slot",
            "pid",
            "thread",
            "clock",
            "count",
            "write_failure",
            "parent",
            "record",
        ):
            metadata, ranges = marker_metadata(), windows_snapshot()
            if mode == "slot":
                metadata["marker_events"][0]["slot"] = 3
            elif mode == "pid":
                metadata["marker_events"][0]["pid"] = True
            elif mode == "thread":
                metadata["marker_events"][0]["tid"] = 43
            elif mode == "clock":
                ranges["host_clock"]["frequency_hz"] = True
            elif mode == "count":
                metadata["observed_marker_events"] = True
            elif mode == "write_failure":
                ranges["native_write_failures"] = 1
            elif mode == "parent":
                metadata["marker_events"][1]["time_filetime_100ns"] = 250
            else:
                metadata["marker_events"][1]["record_index"] = 3
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                bind(metadata, ranges)

    def test_cross_thread_async_lifetime_is_retained_without_host_exclusive_credit(self):
        metadata, ranges = marker_metadata(), windows_snapshot()
        ranges["records"][1].update(stop_state=2, stop_thread_id=43)
        metadata["marker_events"][2]["tid"] = 43
        value = bind(metadata, ranges)
        self.assertEqual(value["state"], "native_measured_windows_bound")
        self.assertEqual(len(value["phase_attribution"]["excluded_scopes"]), 1)


if __name__ == "__main__":
    unittest.main()
