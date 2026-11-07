"""Authored actual3.0.4 interval-table controls; no database fixture has been run."""

from __future__ import annotations

from contextlib import closing

from pathlib import Path
import sqlite3
import tempfile
import unittest

from amd_rocpd import read_process
from test_amd_rocpd import database
from test_target_qualification_support import QualificationFixture


def timeline_database(fixture):
    path = database(fixture)
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
          ALTER TABLE rocpd_event ADD COLUMN category_id INTEGER;
          ALTER TABLE rocpd_event ADD COLUMN extdata JSONB DEFAULT '{}';
          ALTER TABLE rocpd_kernel_dispatch ADD COLUMN tid INTEGER;
          ALTER TABLE rocpd_kernel_dispatch ADD COLUMN region_name_id INTEGER;
          UPDATE rocpd_kernel_dispatch SET tid=42;
          CREATE TABLE rocpd_string(id INTEGER,guid TEXT,string TEXT);
          INSERT INTO rocpd_string VALUES(1,'authored','rns8.public_call#slot=0'),
            (2,'authored','hipLaunchKernel'),(3,'authored','MARKER_CORE_RANGE_API'),
            (4,'authored','HIP_RUNTIME_API'),(5,'authored','MEMORY_COPY'),
            (6,'authored','MEMORY_ALLOCATION'),(7,'authored','HIP_EVENT_RECORD');
          UPDATE rocpd_event SET category_id=4;
          INSERT INTO rocpd_event(id,guid,stack_id,parent_stack_id,correlation_id,category_id,extdata)
            VALUES(100,'authored',201,0,0,3,'{"message":"rns8.public_call#slot=0"}'),
            (101,'authored',71,0,777,4,'{}'),(102,'authored',72,71,777,5,'{}'),
            (103,'authored',73,0,0,6,'{"future_field":7}'),
            (104,'authored',74,0,0,6,'{}'),(105,'authored',75,0,0,4,'{}'),
            (106,'authored',76,0,0,4,'{}');
          CREATE TABLE rocpd_info_thread(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,tid INTEGER);
          INSERT INTO rocpd_info_thread VALUES(42,'authored',1,1,42);
          CREATE TABLE rocpd_region(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,tid INTEGER,start INTEGER,end INTEGER,name_id INTEGER,event_id INTEGER);
          INSERT INTO rocpd_region VALUES(201,'authored',1,1,42,5,90,1,100),
            (71,'authored',1,1,42,8,12,2,101);
          CREATE TABLE rocpd_memory_copy(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,tid INTEGER,start INTEGER,end INTEGER,name_id INTEGER,dst_agent_id INTEGER,dst_address INTEGER,src_agent_id INTEGER,src_address INTEGER,size INTEGER,queue_id INTEGER,stream_id INTEGER,graph_exec_id INTEGER,graph_node_id INTEGER,region_name_id INTEGER,event_id INTEGER);
          INSERT INTO rocpd_memory_copy VALUES(1,'authored',1,1,42,20,25,5,3,256,NULL,512,16,5,9,42,12,1,102);
          CREATE TABLE rocpd_memory_allocate(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,tid INTEGER,agent_id INTEGER,type TEXT,level TEXT,start INTEGER,end INTEGER,address INTEGER,size INTEGER,queue_id INTEGER,stream_id INTEGER,event_id INTEGER,extdata JSONB);
          INSERT INTO rocpd_memory_allocate VALUES(1,'authored',1,1,42,3,'ALLOC','REAL',6,7,4096,64,5,9,103,'{}'),
            (2,'authored',1,1,42,3,'FREE','REAL',80,81,4096,0,5,9,104,'{}');
          CREATE TABLE rocpd_hip_event(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,tid INTEGER,start INTEGER,end INTEGER,name_id INTEGER,agent_id INTEGER,queue_id INTEGER,stream_id INTEGER,hip_event_handle INTEGER,source_queue_id INTEGER,event_id INTEGER);
          INSERT INTO rocpd_hip_event VALUES(1,'authored',1,1,42,26,27,7,3,5,9,65536,5,106);
          CREATE TABLE rocpd_graph_launch(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,tid INTEGER,agent_id INTEGER,queue_id INTEGER,start INTEGER,end INTEGER,graph_exec_id INTEGER,kernel_dispatch_count INTEGER,event_id INTEGER);
          INSERT INTO rocpd_graph_launch VALUES(1,'authored',1,1,42,3,5,7,26,42,1,105);
        """)
    return path


class NativeTimeline(unittest.TestCase):
    def test_real_tables_preserve_typed_process_thread_event_and_memory_domains(self):
        with tempfile.TemporaryDirectory() as directory:
            path = timeline_database(QualificationFixture(Path(directory)))
            result = read_process(path, 222, include_timeline=True)["timeline"]
            self.assertEqual(result["process_id"], 222)
            self.assertEqual(
                result["record_counts"],
                dict(
                    host_region=2,
                    memory_copy=1,
                    memory_lifecycle=2,
                    hip_event=1,
                    graph_launch=1,
                    kernel_dispatch=1,
                ),
            )
            records = result["records"]
            kernel = next(row for row in records if row["kind"] == "kernel_dispatch")
            self.assertEqual((kernel["thread_id"], kernel["native_event_id"]), (42, 17))
            events = {row["id"]: row for row in result["events"]}
            self.assertEqual(events[17]["stack_id"], events[101]["stack_id"])
            self.assertEqual(events[103]["extdata"], dict(future_field=7))
            freed = next(row for row in records if row.get("type") == "FREE")
            self.assertEqual(freed["size"], 0)  # Raw zero is not guessed freed payload.
            copy = next(row for row in records if row["kind"] == "memory_copy")
            self.assertIsNone(copy["src_agent_id"])
            self.assertEqual(copy["size"], 16)
            self.assertFalse(result["complete_run_coverage"])
            self.assertFalse(result["production_latency"])
            self.assertIn("unknown", result["loss_status"])

    def test_missing_timeline_is_explicit_opt_in_and_never_an_empty_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = database(QualificationFixture(Path(directory)))
            self.assertNotIn("timeline", read_process(path, 222))
            with self.assertRaises(ValueError):
                read_process(path, 222, include_timeline=True)

    def test_foreign_orphan_reversed_malformed_and_untyped_values_are_rejected(self):
        for sql in (
            "UPDATE rocpd_info_thread SET guid='foreign'",
            "UPDATE rocpd_region SET event_id=999",
            "UPDATE rocpd_region SET start=0",
            "UPDATE rocpd_memory_copy SET end=101",
            "UPDATE rocpd_memory_copy SET src_agent_id=999",
            "UPDATE rocpd_memory_copy SET queue_id=5.5",
            "UPDATE rocpd_memory_allocate SET type='GUESSED'",
            "UPDATE rocpd_memory_allocate SET level='PINNED'",
            "UPDATE rocpd_event SET extdata='{invalid'",
            "UPDATE rocpd_event SET category_id=999",
            "UPDATE rocpd_hip_event SET source_queue_id=999",
        ):
            with self.subTest(sql=sql), tempfile.TemporaryDirectory() as directory:
                path = timeline_database(QualificationFixture(Path(directory)))
                with closing(sqlite3.connect(path)) as db, db:
                    db.execute(sql)
                with self.assertRaises(ValueError):
                    read_process(path, 222, include_timeline=True)

    def test_combined_row_and_parameter_budgets_cover_the_entire_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = timeline_database(QualificationFixture(Path(directory)))
            for changes in (
                dict(maximum_rows=10),
                dict(maximum_rows=True),
                dict(maximum_rows=1_000_001),
                dict(timeout_seconds=True),
                dict(include_timeline=1),
            ):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    read_process(path, 222, **changes)


if __name__ == "__main__":
    unittest.main()
