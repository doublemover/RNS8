"""Authored native original-event stack controls; no ETL decoder executes."""

from __future__ import annotations

import copy
import unittest

from etl_process_contract import process_contract
from etl_cpu_contract import cpu_catalog
from etl_heap_contract import heap_catalog
from etl_schedule_contract import schedule_catalog
from etl_stack_contract import stack_catalog
from test_etl_cpu_contract import cpu_metadata
from test_etl_heap_contract import heap_metadata
from test_etl_schedule_contract import schedule_metadata


def fragment(ordinal=11, qpc=4500, *, frames=None, tid=42):
    return dict(
        record_index=ordinal,
        wire_contract="windows_kernel_StackWalk_prefix_versions_0_to_2",
        event_version=2,
        event_time_qpc=qpc,
        pid=333,
        tid=tid,
        pointer_bytes=8,
        frames=[0x180001234, 0x180001400] if frames is None else frames,
    )


def stack_metadata():
    metadata = cpu_metadata()
    metadata.update(
        observed_stack_events=2,
        stack_events=[fragment(), fragment(12, frames=[0x180001600])],
        stack_frames=3,
    )
    return metadata


def read(metadata, *, heap=False, schedule=False):
    process = process_contract(
        metadata,
        pid=333,
        parent_pid=444,
        executable="app.exe",
        decoder_pid=555,
        source_sha256="a" * 64,
        file_bytes=4096,
    )
    cpu = cpu_catalog(metadata, process)
    return stack_catalog(
        metadata,
        process,
        cpu,
        heap=heap_catalog(metadata, process, cpu) if heap else None,
        schedule=schedule_catalog(metadata, process, cpu) if schedule else None,
    )


class NativeStacks(unittest.TestCase):
    def test_multiple_native_fragments_bind_one_original_sample_without_concatenation(self):
        report = read(stack_metadata())
        target = next(r for r in report["targets"] if r["native_point_record"] == 9)
        self.assertEqual(target["stack_records"], [11, 12])
        self.assertEqual(target["thread_generation_id"], 7)
        self.assertEqual(target["time_filetime_100ns"], 450)
        self.assertEqual(target["raw_time_qpc"], 4500)
        self.assertEqual(report["targets_without_observed_stack"], 1)
        self.assertEqual(len(report["fragments"]), 2)
        self.assertFalse(report["complete_unwound_or_inlined_callchain_proven"])
        self.assertFalse(report["loaded_symbol_identity_verified"])

    def test_FILETIME_or_arrival_order_cannot_replace_original_raw_QPC_identity(self):
        metadata = stack_metadata()
        metadata["stack_events"].reverse()
        metadata["stack_events"][0]["event_time_qpc"] = 450
        report = read(metadata)
        target = next(r for r in report["targets"] if r["native_point_record"] == 9)
        self.assertEqual(target["stack_records"], [11])
        self.assertIn("raw_QPC", report["unresolved"][0]["reason"])

    def test_heap_CPU_timestamp_collision_is_unresolved_while_distinct_heap_events_bind(self):
        metadata = heap_metadata()
        metadata.update(
            observed_stack_events=2,
            stack_events=[fragment(20), fragment(21, qpc=6000)],
            stack_frames=4,
        )
        report = read(metadata, heap=True)
        self.assertEqual(sum(bool(r["stack_records"]) for r in report["targets"]), 1)
        target = next(r for r in report["targets"] if r["stack_records"])
        self.assertEqual(target["source_kind"], "heap_event")
        self.assertEqual(target["native_point_record"], 15)
        self.assertIn("not_unique", report["unresolved"][0]["reason"])

    def test_scheduler_stack_uses_original_selected_thread_instead_of_native_header_actor(self):
        metadata = schedule_metadata()
        metadata.update(
            observed_stack_events=1, stack_events=[fragment(20, qpc=3500)], stack_frames=2
        )
        report = read(metadata, schedule=True)
        target = next(r for r in report["targets"] if r["stack_records"])
        self.assertEqual(target["source_kind"], "schedule_out")
        self.assertEqual(target["tid"], 42)

    def test_unknown_source_formats_loss_empty_frames_and_missing_threads_never_gain_stack_credit(
        self,
    ):
        for kind in ("undecoded_source", "loss", "empty", "foreign_thread"):
            metadata = stack_metadata()
            if kind == "undecoded_source":
                metadata["undecoded_sample_events"] = 1
                metadata["observed_sample_events"] += 1
            elif kind == "loss":
                metadata["header"]["events_lost"] = 1
            elif kind == "empty":
                for row in metadata["stack_events"]:
                    row["frames"] = []
                metadata["stack_frames"] = 0
            else:
                for row in metadata["stack_events"]:
                    row["tid"] = 99
            report = read(metadata)
            self.assertFalse(any(r["stack_records"] for r in report["targets"]))
            self.assertTrue(report["unresolved"])

    def test_non_QPC_recording_is_explicitly_unbound_without_made_up_conversion(self):
        metadata = cpu_metadata()
        metadata["header"]["clock_type"] = 2
        metadata.update(
            raw_binding_state="unsupported_non_QPC_clock",
            raw_walk_events=0,
            raw_source_point_count=0,
        )
        for row in metadata["sample_events"]:
            del row["raw_time_qpc"]
        report = read(metadata)
        self.assertEqual(report["state"], "no_resolved_native_stack_association")
        self.assertFalse(report["fragments"])
        self.assertFalse(report["targets"])

    def test_raw_walk_identity_accounting_payload_pointer_domains_and_source_binding_are_strict(
        self,
    ):
        for key, value in (
            ("raw_walk_events", 19),
            ("raw_source_point_count", 1),
            ("raw_binding_state", "ordinal_only"),
            ("raw_record_namespace", "second_native_event_walk"),
            ("observed_stack_events", 3),
            ("stack_frames", 4),
        ):
            metadata = stack_metadata()
            metadata[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                read(metadata)
        for key, value in (
            ("record_index", 9),
            ("pid", 999),
            ("pid", True),
            ("event_version", 3),
            ("pointer_bytes", True),
            ("frames", [True]),
            ("event_time_qpc", True),
            ("frames", [2**64]),
        ):
            metadata = stack_metadata()
            metadata["stack_events"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                read(metadata)
        metadata = stack_metadata()
        metadata["sample_events"][0]["raw_time_qpc"] = True
        with self.assertRaises(ValueError):
            read(metadata)
        metadata = stack_metadata()
        metadata["stack_events"].append(copy.deepcopy(metadata["stack_events"][0]))
        metadata["observed_stack_events"] += 1
        metadata["stack_frames"] += 2
        with self.assertRaises(ValueError):
            read(metadata)


if __name__ == "__main__":
    unittest.main()
