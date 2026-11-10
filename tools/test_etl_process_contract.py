"""Authored native ETL JSON controls; these never claim a real Windows trace."""

from __future__ import annotations

import copy
import unittest

from etl_process_contract import process_contract


def native_metadata(
    *,
    pid=333,
    parent_pid=444,
    decoder_pid=555,
    image="app.exe",
    source_sha256="a" * 64,
    file_bytes=4096,
):
    return dict(
        schema="rns8_native_etl_process_metadata_v1",
        reader_source_sha256=source_sha256,
        decoder_pid=decoder_pid,
        selected_pid=pid,
        native_process_trace_status=0,
        complete_event_walk=True,
        file_bytes=file_bytes,
        observed_events=10,
        observed_process_events=2,
        undecoded_process_events=0,
        observed_image_events=0,
        undecoded_image_events=0,
        image_events=[],
        observed_marker_events=0,
        undecoded_marker_events=0,
        marker_events=[],
        observed_thread_events=0,
        undecoded_thread_events=0,
        thread_events=[],
        sample_walk_events=10,
        observed_sample_events=0,
        undecoded_sample_events=0,
        excluded_sample_events=0,
        sample_events=[],
        sample_filter="known_selected_header_PID_or_unique_selected_thread_Start_End_for_unknown_header_PID",
        sample_record_namespace="second_native_event_walk",
        complete_selected_sample_coverage_proven=False,
        observed_schedule_events=0,
        undecoded_schedule_events=0,
        excluded_schedule_events=0,
        schedule_events=[],
        schedule_filter="unique_selected_thread_Start_End_window",
        schedule_record_namespace="second_native_event_walk",
        observed_heap_events=0,
        undecoded_heap_events=0,
        excluded_heap_events=0,
        unsupported_heap_events=0,
        heap_events=[],
        heap_filter="known_selected_header_PID_or_unique_selected_thread_Start_End_for_unknown_header_PID",
        heap_record_namespace="second_native_event_walk",
        raw_binding_state="no_selected_native_points",
        raw_walk_events=0,
        raw_source_point_count=0,
        raw_record_namespace="third_native_event_walk",
        observed_stack_events=0,
        undecoded_stack_events=0,
        excluded_stack_events=0,
        unsupported_stack_events=0,
        stack_events=[],
        stack_frames=0,
        event_timestamp_domain="native_ProcessTrace_converted_FILETIME_100ns",
        complete_loss_free_acquisition_proven=False,
        image_path_or_loaded_PDB_verified=False,
        production_latency=False,
        promotion_allowed=False,
        header=dict(
            pointer_bytes=8,
            clock_type=1,
            perf_frequency=10000000,
            start_filetime_100ns=100,
            end_filetime_100ns=1000,
            events_lost=0,
            buffers_lost=0,
            buffers_written=2,
        ),
        process_events=[
            dict(
                pid=pid,
                parent_pid=parent_pid,
                process_key=7,
                opcode=opcode,
                time_filetime_100ns=timestamp,
                image_basename_ascii=image,
                event_version=4,
                pointer_bytes=8,
            )
            for opcode, timestamp in ((1, 200), (2, 900))
        ],
    )


class NativeProcessCustody(unittest.TestCase):
    def read(self, metadata):
        return process_contract(
            metadata,
            pid=333,
            parent_pid=444,
            executable="app.exe",
            decoder_pid=555,
            source_sha256="a" * 64,
            file_bytes=4096,
        )

    def test_complete_native_start_end_key_parent_image_and_header_loss_scope(self):
        value = self.read(native_metadata())
        self.assertTrue(value["process_identity_complete"])
        self.assertEqual(value["process_key"], 7)
        self.assertEqual(value["actual_start_filetime_100ns"], 200)
        self.assertFalse(value["complete_loss_free_acquisition_proven"])
        self.assertFalse(value["promotion_allowed"])

    def test_lost_unknown_incomplete_rundown_and_PID_reuse_never_gain_complete_identity(self):
        original = native_metadata()
        variants = []
        for field in ("events_lost", "buffers_lost"):
            value = copy.deepcopy(original)
            value["header"][field] = 1
            variants.append(value)
        value = copy.deepcopy(original)
        value["undecoded_process_events"] = 1
        variants.append(value)
        value = copy.deepcopy(original)
        value["process_events"][0]["opcode"] = 3
        variants.append(value)
        value = copy.deepcopy(original)
        value["process_events"][1]["process_key"] = 8
        variants.append(value)
        value = copy.deepcopy(original)
        value["process_events"][0]["image_basename_ascii"] = None
        variants.append(value)
        for value in variants:
            report = self.read(value)
            self.assertFalse(report["process_identity_complete"])
            self.assertTrue(report["blockers"])

    def test_substituted_identity_false_fields_and_out_of_window_records_are_rejected(self):
        for field, value in (
            ("decoder_pid", 999),
            ("selected_pid", 334),
            ("reader_source_sha256", "b" * 64),
            ("native_process_trace_status", True),
            ("file_bytes", 3),
            ("complete_event_walk", 1),
            ("production_latency", True),
        ):
            metadata = native_metadata()
            metadata[field] = value
            with self.assertRaises(ValueError):
                self.read(metadata)
        for field, value in (
            ("pid", 999),
            ("parent_pid", 999),
            ("image_basename_ascii", "other.exe"),
            ("time_filetime_100ns", 99),
            ("process_key", True),
        ):
            metadata = native_metadata()
            metadata["process_events"][0][field] = value
            with self.assertRaises(ValueError):
                self.read(metadata)


if __name__ == "__main__":
    unittest.main()
