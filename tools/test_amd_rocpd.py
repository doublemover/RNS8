"""Authored native-schema databases, never collected profiling evidence."""

from __future__ import annotations

from contextlib import closing

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from amd_rocpd import read_process
from amd_profiler_evidence import validate_dispatch_receipt
from target_qualification import EvidenceStore
from test_target_qualification_support import QualificationFixture, reference, write
from amd_profiler_review import selected_interval_summary, review
from gpu_counter_report import report_for_capture


def database(fixture: QualificationFixture) -> Path:
    """Minimal projections of actual 3.0.4 columns; authored values throughout."""
    path = fixture.root / "authored-native.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
          CREATE TABLE rocpd_metadata(tag TEXT,value TEXT);
          INSERT INTO rocpd_metadata VALUES('schema_version','3.0.4');
          CREATE TABLE rocpd_info_process(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,init INTEGER,fini INTEGER);
          INSERT INTO rocpd_info_process VALUES(1,'authored',1,222,1,100);
          CREATE TABLE rocpd_info_agent(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,name TEXT,type TEXT,uuid INTEGER,absolute_index INTEGER,logical_index INTEGER,type_index INTEGER,extdata JSONB DEFAULT '{}');
          INSERT INTO rocpd_info_agent VALUES(3,'authored',1,1,'gfx942','GPU',888,4,7,0,'{}');
          CREATE TABLE rocpd_info_code_object(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,agent_id INTEGER,uri TEXT,storage_type TEXT);
          CREATE TABLE rocpd_info_kernel_symbol(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,code_object_id INTEGER,kernel_name TEXT);
          INSERT INTO rocpd_info_kernel_symbol VALUES(13,'authored',1,1,7,'synthetic_selected_symbol');
          CREATE TABLE rocpd_info_queue(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER);
          INSERT INTO rocpd_info_queue VALUES(5,'authored',1,1);
          CREATE TABLE rocpd_info_stream(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER);
          INSERT INTO rocpd_info_stream VALUES(9,'authored',1,1);
          CREATE TABLE rocpd_event(id INTEGER,guid TEXT,stack_id INTEGER,parent_stack_id INTEGER,correlation_id INTEGER);
          INSERT INTO rocpd_event VALUES(17,'authored',71,71,777);
          CREATE TABLE rocpd_kernel_dispatch(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,
            agent_id INTEGER,kernel_id INTEGER,dispatch_id INTEGER,queue_id INTEGER,stream_id INTEGER,
            start INTEGER,end INTEGER,event_id INTEGER,grid_size_x INTEGER,grid_size_y INTEGER,
            grid_size_z INTEGER,workgroup_size_x INTEGER,workgroup_size_y INTEGER,workgroup_size_z INTEGER,
            graph_exec_id INTEGER,graph_node_id INTEGER,group_segment_size INTEGER,private_segment_size INTEGER);
          INSERT INTO rocpd_kernel_dispatch VALUES(1,'authored',1,1,3,13,11,5,9,10,20,17,64,1,1,16,1,1,42,12,1024,0);
          CREATE TABLE rocpd_info_pmc(id INTEGER,guid TEXT,nid INTEGER,pid INTEGER,agent_id INTEGER,
            name TEXT,symbol TEXT,description TEXT,units TEXT,block TEXT,expression TEXT,
            value_type TEXT,is_derived INTEGER,instance_id INTEGER);
          INSERT INTO rocpd_info_pmc VALUES(19,'authored',1,1,3,'AUTHORED_REQUESTS','AUTHORED_REQUESTS',
            'Authored definition only','requests','TCC',NULL,'ACCUM',0,2);
          CREATE TABLE rocpd_pmc_event(id INTEGER,guid TEXT,event_id INTEGER,sample_id INTEGER,
            pmc_id INTEGER,value REAL,xcc INTEGER,shader_engine INTEGER,instance INTEGER);
          INSERT INTO rocpd_pmc_event VALUES(1,'authored',17,NULL,19,0.0,1,2,3);
        """)
        uri = fixture.native["rocprofiler-sdk-tool"][0]["code_objects"][0]["uri"]
        agent = dict(
            fixture.native["rocprofiler-sdk-tool"][0]["agents"][0],
            id=dict(handle=333),
            device_id=888,
        )
        db.execute("UPDATE rocpd_info_agent SET extdata=? WHERE id=3", (json.dumps(agent),))
        db.execute(
            "INSERT INTO rocpd_info_code_object VALUES(7,'authored',1,1,3,?,'FILE')",
            (uri,),
        )
    return path


class RocpdEvidence(unittest.TestCase):
    def test_native_process_dispatch_and_units_retain_namespaces_and_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = QualificationFixture(Path(temporary))
            path = database(fixture)
            value = read_process(path, 222)
            self.assertEqual(value["native"]["agents"][0]["id"]["handle"], 3)
            self.assertEqual(
                value["native"]["agents"][0]["native_agent_metadata"]["id"]["handle"], 333
            )
            self.assertIn("database_ids", value["identity_namespace"])
            dispatch = value["native"]["buffer_records"]["kernel_dispatch"][0]
            self.assertEqual(dispatch["correlation_id"], dict(internal=71, external=777))
            self.assertEqual(dispatch["native_rocpd"]["parent_stack_id"], 71)
            metric = value["metrics"][0]
            self.assertEqual(metric["unit"], "requests")
            self.assertEqual(metric["value"], 0.0)
            self.assertIsNone(metric["missingness"])
            self.assertEqual(
                (metric["xcc"], metric["shader_engine"], metric["instance"]), (1, 2, 3)
            )
            self.assertEqual(metric["definition_instance"], 2)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("UPDATE rocpd_event SET correlation_id=0,parent_stack_id=NULL")
            actual_zero = read_process(path, 222)["native"]["buffer_records"]["kernel_dispatch"][0]
            self.assertEqual(actual_zero["correlation_id"], dict(internal=71, external=0))
            self.assertIsNone(actual_zero["native_rocpd"]["parent_stack_id"])
            self.assertFalse(value["complete_run_coverage"])
            self.assertTrue(value["loss_status"].startswith("unknown"))
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("UPDATE rocpd_info_pmc SET units=''")
            missing = read_process(path, 222)["metrics"][0]
            self.assertEqual(missing["value"], 0.0)
            self.assertEqual(missing["missingness"], "vendor_unit_unavailable")

    def test_rocpd_reuses_actual_selected_dispatch_and_artifact_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = QualificationFixture(Path(temporary))
            path = database(fixture)
            receipt = copy.deepcopy(fixture.dispatch)
            receipt["adapter"] = "rocprofiler-sdk-rocpd-3.0.4-kernel-dispatch-v1"
            receipt["native_trace"] = reference(path)
            options = receipt["acquisition_command"]
            options[options.index("--output-format") + 1] = "rocpd"
            write(fixture.dispatch_path, receipt)
            store = EvidenceStore()
            value = validate_dispatch_receipt(
                reference(fixture.dispatch_path),
                capture=fixture.capture,
                store=store,
                parent=fixture.root,
            )
            store.finish()
            self.assertTrue(value["qualified_selected_dispatches"])
            self.assertEqual(
                value["physical_device_associations"][0]["pci_bdf"],
                fixture.capture["device_identity_witness"]["before"]["pci_bdf"],
            )
            self.assertEqual(value["native_schema_version"], "3.0.4")
            self.assertEqual(value["observations"][0]["native_relationships"]["graph_exec_id"], 42)
            self.assertEqual(value["native_metrics"][0]["raw_name"], "AUTHORED_REQUESTS")
            self.assertFalse(value["performance_promotion_allowed"])
            report = report_for_capture(fixture.capture_path, [], [], 3, [fixture.dispatch_path])
            self.assertEqual(
                report["native_profiler_evidence"][0]["native_metrics"][0]["value"], 0.0
            )
            self.assertEqual(
                report["native_profiler_review"]["passes"][0]["intervals"]["service_ns"],
                10,
            )

    def test_rocpd_agent_device_id_column_and_PCI_metadata_cannot_be_substituted(self):
        for mode in ("device_id", "domain", "boolean", "unknown_alias"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = QualificationFixture(Path(temporary))
                path = database(fixture)
                with closing(sqlite3.connect(path)) as db, db:
                    if mode == "device_id":
                        db.execute("UPDATE rocpd_info_agent SET uuid=999")
                    else:
                        metadata = json.loads(
                            db.execute("SELECT extdata FROM rocpd_info_agent").fetchone()[0]
                        )
                        if mode == "domain":
                            metadata["domain"] += 1
                        elif mode == "boolean":
                            metadata["location_id"] = False
                        else:
                            metadata["native_agent_metadata"] = dict(metadata)
                            metadata.pop("location_id")
                        db.execute("UPDATE rocpd_info_agent SET extdata=?", (json.dumps(metadata),))
                receipt = copy.deepcopy(fixture.dispatch)
                receipt["adapter"] = "rocprofiler-sdk-rocpd-3.0.4-kernel-dispatch-v1"
                receipt["native_trace"] = reference(path)
                command = receipt["acquisition_command"]
                command[command.index("--output-format") + 1] = "rocpd"
                write(fixture.dispatch_path, receipt)
                with self.assertRaises(ValueError):
                    validate_dispatch_receipt(
                        reference(fixture.dispatch_path),
                        capture=fixture.capture,
                        store=EvidenceStore(),
                        parent=fixture.root,
                    )

    def test_overlap_union_and_passes_are_not_summed_as_production_time(self):
        observations = [
            dict(start_ns=0, end_ns=100),
            dict(start_ns=10, end_ns=20),
            dict(start_ns=200, end_ns=300),
        ]
        summary = selected_interval_summary(observations)
        self.assertEqual(summary["service_ns"], 210)
        self.assertEqual(summary["interval_union_ns"], 200)
        self.assertEqual(summary["uncovered_span_ns"], 100)
        self.assertEqual(summary["overlap_service_ns"], 10)
        result = review(
            [
                dict(
                    observations=observations,
                    receipt_sha256="a" * 64,
                    native_metrics=[],
                    loss_status="unknown",
                    complete_run_coverage=False,
                )
            ]
        )
        self.assertIn("unselected kernels", result["passes"][0]["hypotheses"][0]["contradiction"])
        self.assertFalse(result["performance_promotion_allowed"])

    def test_unknown_schema_cross_process_orphan_and_duplicate_identities_fail(self):
        changes = (
            "UPDATE rocpd_metadata SET value='99.0.0'",
            "INSERT INTO rocpd_info_process SELECT * FROM rocpd_info_process",
            "UPDATE rocpd_info_code_object SET agent_id=8",
            "UPDATE rocpd_info_queue SET guid='other'",
            "UPDATE rocpd_kernel_dispatch SET event_id=999",
            "UPDATE rocpd_event SET stack_id=NULL",
            "UPDATE rocpd_event SET correlation_id=-1",
            "UPDATE rocpd_event SET parent_stack_id=-1",
            "INSERT INTO rocpd_kernel_dispatch SELECT 2,guid,nid,pid,agent_id,kernel_id,dispatch_id,queue_id,stream_id,start,end,event_id,grid_size_x,grid_size_y,grid_size_z,workgroup_size_x,workgroup_size_y,workgroup_size_z,graph_exec_id,graph_node_id,group_segment_size,private_segment_size FROM rocpd_kernel_dispatch",
            "UPDATE rocpd_kernel_dispatch SET start=0",
            "UPDATE rocpd_kernel_dispatch SET workgroup_size_x=0",
            "UPDATE rocpd_info_pmc SET agent_id=8",
        )
        for change in changes:
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = QualificationFixture(Path(temporary))
                path = database(fixture)
                with closing(sqlite3.connect(path)) as db, db:
                    db.execute(change)
                with self.assertRaises(ValueError):
                    read_process(path, 222)

    def test_live_snapshot_and_bounded_read_fail_without_writing_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = QualificationFixture(Path(temporary))
            path = database(fixture)
            before = path.read_bytes()
            sidecar = Path(str(path) + "-wal")
            sidecar.write_bytes(b"authored unfinished WAL")
            with self.assertRaises(ValueError):
                read_process(path, 222)
            sidecar.unlink()
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                read_process(path, 333)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("INSERT INTO rocpd_event VALUES(18,'authored',72,72,778)")
            with self.assertRaises(ValueError):
                read_process(path, 222, maximum_rows=1)


if __name__ == "__main__":
    unittest.main()
