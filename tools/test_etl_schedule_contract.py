"""Authored CSwitch/ReadyThread controls; no native trace or profiler runs."""

from __future__ import annotations

import unittest

from etl_schedule_contract import schedule_catalog
from etl_process_contract import process_contract
from etl_cpu_contract import cpu_catalog
from etl_range_contract import operation_ranges
from test_etl_cpu_contract import cpu_metadata, thread
from test_etl_range_contract import windows_snapshot


def switched(stamp, ordinal, *, old=42, new=7, state=5):
    return dict(
        record_index=ordinal,
        time_filetime_100ns=stamp,
        native_header_pid=999,
        native_header_tid=999,
        processor_index_mode=True,
        processor=3,
        wire_contract="windows_kernel_CSwitch_v2",
        old_tid=old,
        new_tid=new,
        control_bytes=[8, 8, 0, 0, 29, 0, state, 0],
        new_wait_time=19,
        reserved=0,
    )


def readied(stamp=550, ordinal=12, *, flags=0):
    return dict(
        record_index=ordinal,
        time_filetime_100ns=stamp,
        native_header_pid=777,
        native_header_tid=999,
        processor_index_mode=True,
        processor=4,
        wire_contract="windows_kernel_ReadyThread_v2",
        tid=42,
        control_bytes=[1, 2, flags, 0],
    )


def schedule_metadata():
    value = cpu_metadata()
    value.update(
        observed_events=30,
        sample_walk_events=30,
        observed_schedule_events=3,
        schedule_events=[switched(350, 11), readied(), switched(750, 13, old=7, new=42)],
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
    return schedule_catalog(metadata, process, cpu, snapshot=snapshot, native_ranges=native)


class NativeScheduling(unittest.TestCase):
    def test_wait_ready_and_in_separate_conditional_thread_time_and_exclusive_phases(self):
        report = read(schedule_metadata(), ranges=True)
        self.assertEqual(report["conditional_off_cpu_thread_ns"], 40000)
        self.assertEqual(report["conditional_waiting_thread_ns"], 20000)
        self.assertEqual(report["conditional_runnable_thread_ns"], 20000)
        self.assertEqual(report["intervals"][0]["old_wait_reason"], 29)
        selection = report["selected_operation_by_state"]
        self.assertEqual(selection["off_cpu"]["conditional_thread_ns"], 35000)
        self.assertEqual(
            selection["off_cpu"]["exclusive_phase_thread_ns"], dict(proof=10000, public_call=25000)
        )
        self.assertEqual(selection["waiting"]["conditional_thread_ns"], 20000)
        self.assertEqual(selection["runnable"]["conditional_thread_ns"], 15000)
        self.assertFalse(report["wait_object_or_lock_owner_verified"])
        self.assertFalse(report["waker_or_GPU_dependency_verified"])
        self.assertIsNone(report["active_spin_time_ns"])

    def test_late_records_keep_event_identity_but_do_not_change_time_order(self):
        value = schedule_metadata()
        value["schedule_events"].reverse()
        report = read(value)
        self.assertEqual(report["conditional_off_cpu_thread_ns"], 40000)
        self.assertEqual(report["intervals"][0]["ready_records"], [12])

    def test_ready_or_standby_at_out_is_runnable_without_a_wait_cause(self):
        for state in (1, 3):
            value = schedule_metadata()
            value["schedule_events"] = [
                switched(350, 11, state=state),
                switched(750, 13, old=7, new=42),
            ]
            value["observed_schedule_events"] = 2
            report = read(value)
            self.assertEqual(report["conditional_waiting_thread_ns"], 0)
            self.assertEqual(report["conditional_runnable_thread_ns"], 40000)

    def test_missing_swapped_multiple_unknown_and_reserved_ready_states_stay_unclassified(self):
        for kind in (
            "missing",
            "swapped",
            "multiple",
            "unknown",
            "reserved",
            "unknown_wait_mode",
            "reserved_switch",
        ):
            value = schedule_metadata()
            if kind == "missing":
                value["schedule_events"].pop(1)
            elif kind == "swapped":
                value["schedule_events"][1]["control_bytes"][2] = 2
            elif kind == "multiple":
                value["schedule_events"].append(readied(600, 14))
            elif kind == "unknown":
                value["schedule_events"][1]["control_bytes"][0] = 3
            elif kind == "reserved":
                value["schedule_events"][1]["control_bytes"][3] = 1
            elif kind == "unknown_wait_mode":
                value["schedule_events"][0]["control_bytes"][5] = 255
            else:
                value["schedule_events"][0]["reserved"] = 1
            value["observed_schedule_events"] = len(value["schedule_events"])
            with self.subTest(kind=kind):
                report = read(value)
                self.assertEqual(report["conditional_off_cpu_thread_ns"], 40000)
                self.assertEqual(report["conditional_waiting_thread_ns"], 0)
                self.assertEqual(report["conditional_runnable_thread_ns"], 0)

    def test_ambiguous_missing_and_lost_boundaries_cannot_gain_duration_credit(self):
        for kind in (
            "same_time",
            "no_in",
            "no_out",
            "undecoded_schedule",
            "undecoded_thread",
            "loss",
            "terminated",
        ):
            value = schedule_metadata()
            if kind == "same_time":
                value["schedule_events"][1]["time_filetime_100ns"] = 350
            elif kind == "no_in":
                value["schedule_events"].pop()
            elif kind == "no_out":
                value["schedule_events"].pop(0)
            elif kind == "undecoded_schedule":
                value["undecoded_schedule_events"] = 1
            elif kind == "undecoded_thread":
                value["undecoded_thread_events"] = 1
                value["observed_thread_events"] += 1
            elif kind == "loss":
                value["header"]["events_lost"] = 1
            else:
                value["schedule_events"][0]["control_bytes"][6] = 4
            value["observed_schedule_events"] = (
                len(value["schedule_events"]) + value["undecoded_schedule_events"]
            )
            with self.subTest(kind=kind):
                report = read(value)
                self.assertFalse(report["intervals"])
                self.assertTrue(report["unresolved"])

    def test_reused_or_foreign_TID_and_typed_raw_accounting_drift_are_rejected(self):
        for kind in (
            "reuse",
            "foreign",
            "duplicate",
            "sample_alias",
            "control_bool",
            "new_wait_bool",
            "processor",
            "namespace",
            "count",
            "filter",
        ):
            value = schedule_metadata()
            row = value["schedule_events"][0]
            if kind == "reuse":
                value["thread_events"] = [
                    thread(1, 250, 7),
                    thread(2, 500, 8),
                    thread(1, 600, 14),
                    thread(2, 850, 15),
                ]
                value["observed_thread_events"] = 4
                for sample in value["sample_events"]:
                    sample["native_header_pid"] = 333
            elif kind == "foreign":
                row["old_tid"] = 999
            elif kind == "duplicate":
                value["schedule_events"][1]["record_index"] = row["record_index"]
            elif kind == "sample_alias":
                row["record_index"] = 9
            elif kind == "control_bool":
                row["control_bytes"][0] = True
            elif kind == "new_wait_bool":
                row["new_wait_time"] = True
            elif kind == "processor":
                row.update(processor=300, processor_index_mode=False)
            elif kind == "namespace":
                value["schedule_record_namespace"] = "first_native_event_walk"
            elif kind == "count":
                value["observed_schedule_events"] += 1
            else:
                value["schedule_filter"] = "matching_TID"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                read(value)


if __name__ == "__main__":
    unittest.main()
