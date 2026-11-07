"""Authored real-table→marker/SDK identity joins, not collected GPU evidence."""

from __future__ import annotations

from contextlib import closing

import copy
from pathlib import Path
import sqlite3
import tempfile
import unittest

from amd_profiler_ranges import join
from amd_rocpd import read_process
from test_amd_rocpd_timeline import timeline_database
from test_host_profile_ranges import snapshot, scope
from test_target_qualification_support import QualificationFixture


def markers():
    value = snapshot([scope(), scope(123, 0, start=110, end=140, phase=2)])
    value["marker_label_contract"] = "rns8_phase_slot_v1"
    for row in value["records"]:
        row["process_id"] = 222
    return value


class NativeMarkerJoin(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.path = timeline_database(QualificationFixture(Path(scratch.name)))
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript("""
              INSERT INTO rocpd_string VALUES(8,'authored','rns8.proof#slot=1');
              INSERT INTO rocpd_event(id,guid,stack_id,parent_stack_id,correlation_id,category_id,extdata)
                VALUES(202,'authored',1202,0,0,3,'{"message":"rns8.proof#slot=1"}');
              INSERT INTO rocpd_region VALUES(202,'authored',1,1,42,6,15,8,202);
            """)

    def timeline(self):
        return read_process(self.path, 222, include_timeline=True)["timeline"]

    def test_literal_slot_and_exact_internal_correlation_bind_overlapping_API_device_intervals(
        self,
    ):
        result = join(self.timeline(), markers())
        (selected,) = result["selected_dispatches"]
        self.assertEqual((selected["operation_id"], selected["iteration_id"]), (1, 0))
        self.assertEqual((selected["host_range_id"], selected["native_marker_id"]), (0, 201))
        self.assertEqual(
            (selected["native_api_id"], selected["sdk_internal_correlation"]), (71, 71)
        )
        self.assertEqual((selected["api_end_ns"], selected["start_ns"]), (12, 10))
        self.assertFalse(result["API_completion_to_kernel_start_dependency_inferred"])
        self.assertEqual(result["marker_slot_count"], 2)
        self.assertFalse(result["complete_run_coverage"])
        self.assertFalse(result["production_latency"])
        self.assertEqual(result["selected_device_intervals"]["interval_union_ns"], 10)
        self.assertEqual(result["measured_native_windows"]["interval_union_ns"], 85)
        phases = result["phase_attribution"]
        self.assertEqual(phases["records"][0]["launch_phase"], "proof")
        self.assertEqual(phases["records"][0]["launch_phase_range_id"], 123)
        self.assertEqual(
            phases["threads"][0]["exclusive_phase_wall_ns"], dict(proof=9, public_call=76)
        )
        self.assertEqual(phases["threads"][0]["observed_scope_union_ns"], 85)
        self.assertFalse(phases["thread_totals_add_to_process_wall"])

    def test_overlapping_siblings_remain_ambiguous_instead_of_deepest_name_wins(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript("""
              INSERT INTO rocpd_string VALUES(10,'authored','rns8.dispatch_enqueue#slot=2');
              INSERT INTO rocpd_event(id,guid,stack_id,parent_stack_id,correlation_id,category_id,extdata)
                VALUES(210,'authored',1210,0,0,3,'{"message":"rns8.dispatch_enqueue#slot=2"}');
              INSERT INTO rocpd_region VALUES(210,'authored',1,1,42,7,13,10,210);
            """)
        value = markers()
        value["records"].append(dict(scope(321, 0, start=115, end=135, phase=3), process_id=222))
        value["claimed"] += 1
        report = join(self.timeline(), value)["phase_attribution"]
        self.assertIsNone(report["records"][0]["launch_phase"])
        self.assertEqual(report["threads"][0]["ambiguous_wall_ns"], 6)
        self.assertEqual(
            report["threads"][0]["exclusive_phase_wall_ns"], dict(proof=3, public_call=76)
        )
        self.assertEqual(
            sum(report["threads"][0]["exclusive_phase_wall_ns"].values())
            + report["threads"][0]["ambiguous_wall_ns"],
            85,
        )

    def test_copies_allocations_graphs_and_events_use_their_exact_API_correlations(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            for identity, internal, first, last in (
                (111, 72, 18, 22),
                (112, 73, 5, 7),
                (113, 74, 79, 82),
                (114, 75, 6, 8),
                (115, 76, 25, 28),
            ):
                db.execute(
                    "INSERT INTO rocpd_event(id,guid,stack_id,parent_stack_id,correlation_id,category_id,extdata) VALUES(?,'authored',?,0,777,4,'{}')",
                    (identity, internal),
                )
                db.execute(
                    "INSERT INTO rocpd_region VALUES(?,'authored',1,1,42,?,?,2,?)",
                    (identity, first, last, identity),
                )
        report = join(self.timeline(), markers())
        self.assertEqual(len(report["selected_related_intervals"]), 5)
        self.assertFalse(report["unresolved_related_intervals"])
        self.assertFalse(report["outside_measured_related_intervals"])
        self.assertEqual(
            {row["kind"] for row in report["selected_related_intervals"]},
            {"memory_copy", "memory_lifecycle", "hip_event", "graph_launch"},
        )
        freed = next(
            row
            for row in report["selected_related_intervals"]
            if row["kind"] == "memory_lifecycle" and row["native_record"]["type"] == "FREE"
        )
        self.assertEqual(freed["native_record"]["size"], 0)
        self.assertEqual(freed["native_api_id"], 113)
        self.assertEqual(len(report["phase_attribution"]["records"]), 6)
        self.assertFalse(report["complete_run_coverage"])

    def test_foreign_phase_contract_is_excluded_and_parent_receives_only_observed_wall(self):
        value = markers()
        value["records"][1]["operation_id"] = 999
        report = join(self.timeline(), value)["phase_attribution"]
        self.assertIsNone(report["records"][0]["launch_phase"])
        self.assertEqual(report["threads"][0]["exclusive_phase_wall_ns"], dict(public_call=76))
        self.assertEqual(report["threads"][0]["ambiguous_wall_ns"], 9)
        self.assertEqual(report["excluded_scopes"][0]["host_range_id"], 123)
        self.assertIn("contract_differs", report["excluded_scopes"][0]["reason"])

    def test_old_labels_or_missing_duplicate_conflicting_slot_metadata_do_not_join(self):
        native = self.timeline()
        for change in (
            lambda value: value.pop("marker_label_contract"),
            lambda value: value.update(dropped=1),
            lambda value: value.update(unpublished=1),
            lambda value: value["records"][0].update(process_id=223),
            lambda value: value["records"][0].update(
                stop_state=0, stop_thread_id=None, stop_before_ns=None, stop_after_ns=None
            ),
            lambda value: value["records"][1].update(phase=1),
            lambda value: value["records"][1].update(start_thread_id=99),
            lambda value: value["records"][1].update(parent_range_id=123),
        ):
            value = markers()
            change(value)
            with self.assertRaises(ValueError):
                join(native, value)
        for change in (
            "UPDATE rocpd_string SET string='rns8.proof' WHERE id=8",
            "UPDATE rocpd_string SET string='rns8.proof#slot=0' WHERE id=8",
            "UPDATE rocpd_event SET extdata='{}' WHERE id=202",
            "DELETE FROM rocpd_region WHERE id=202",
            "UPDATE rocpd_region SET end=95 WHERE id=202",
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                altered = Path(directory) / "altered.db"
                altered.write_bytes(self.path.read_bytes())
                with closing(sqlite3.connect(altered)) as db, db:
                    db.execute(change)
                with self.assertRaises(ValueError):
                    join(read_process(altered, 222, include_timeline=True)["timeline"], markers())

    def test_zero_unknown_ambiguous_and_outliving_launches_remain_unresolved(self):
        native = self.timeline()
        for value in (None, 0, 999):
            changed = copy.deepcopy(native)
            next(row for row in changed["events"] if row["id"] == 17)["stack_id"] = value
            result = join(changed, markers())
            self.assertFalse(result["selected_dispatches"])
            self.assertEqual(len(result["unresolved_dispatches"]), 1)
        changed = copy.deepcopy(native)
        api = next(
            row for row in changed["records"] if row["kind"] == "host_region" and row["id"] == 71
        )
        changed["records"].append(dict(api, id=72))
        self.assertEqual(
            join(changed, markers())["unresolved_dispatches"][0]["reason"],
            "missing_or_ambiguous_exact_HIP_API_correlation",
        )
        changed = copy.deepcopy(native)
        next(row for row in changed["records"] if row["kind"] == "kernel_dispatch")["end_ns"] = 95
        self.assertEqual(
            join(changed, markers())["unresolved_dispatches"][0]["reason"],
            "launch_or_device_work_escapes_completed_operation",
        )
        changed = copy.deepcopy(native)
        api = next(
            row for row in changed["records"] if row["kind"] == "host_region" and row["id"] == 71
        )
        api.update(start_ns=1, end_ns=2)
        self.assertEqual(len(join(changed, markers())["outside_measured_dispatches"]), 1)

    def test_overlapping_callers_keep_native_threads_and_operation_ids_separate(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript("""
              INSERT INTO rocpd_info_thread VALUES(43,'authored',1,1,43);
              INSERT INTO rocpd_info_queue VALUES(6,'authored',1,1);
              INSERT INTO rocpd_info_stream VALUES(10,'authored',1,1);
              INSERT INTO rocpd_string VALUES(9,'authored','rns8.public_call#slot=2');
              INSERT INTO rocpd_event(id,guid,stack_id,parent_stack_id,correlation_id,category_id,extdata)
                VALUES(203,'authored',203,0,0,3,'{"message":"rns8.public_call#slot=2"}'),
                (204,'authored',81,0,777,4,'{}'),(205,'authored',81,81,777,4,'{}');
              INSERT INTO rocpd_region VALUES(203,'authored',1,1,43,5,90,9,203),
                (81,'authored',1,1,43,8,12,2,204);
              INSERT INTO rocpd_kernel_dispatch SELECT 2,guid,nid,pid,agent_id,kernel_id,
                dispatch_id,6,10,11,21,205,grid_size_x,grid_size_y,grid_size_z,
                workgroup_size_x,workgroup_size_y,workgroup_size_z,graph_exec_id,
                graph_node_id,group_segment_size,private_segment_size,43,region_name_id
                FROM rocpd_kernel_dispatch WHERE id=1;
            """)
        value = markers()
        other = scope(456, start=100, end=300, thread=43, operation=2)
        other["process_id"] = 222
        value["records"].append(other)
        value["claimed"] += 1
        result = join(self.timeline(), value)
        self.assertEqual({row["operation_id"] for row in result["selected_dispatches"]}, {1, 2})
        self.assertFalse(result["unresolved_dispatches"])
        self.assertEqual(result["selected_device_intervals"]["service_ns"], 20)
        self.assertEqual(result["selected_device_intervals"]["interval_union_ns"], 11)


if __name__ == "__main__":
    unittest.main()
