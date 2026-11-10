"""Authored actual-table generation/interval bounds; no allocation trace was run."""

from __future__ import annotations

import copy
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from amd_rocpd import read_process
from interval_observations import weighted_peak
from native_allocation_lifetimes import summarize
from test_amd_rocpd_timeline import timeline_database
from test_target_qualification_support import QualificationFixture


class AllocationLifetimes(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.path = timeline_database(QualificationFixture(Path(scratch.name)))

    def sql(self, source):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript(source)

    def timeline(self):
        return read_process(self.path, 222, include_timeline=True)["timeline"]

    def test_actual_free_zero_recovers_size_and_reuse_gets_a_distinct_generation(self):
        self.sql("""
          INSERT INTO rocpd_memory_allocate VALUES(3,'authored',1,1,42,3,'ALLOC','REAL',85,86,4096,32,5,9,103,'{}'),
            (4,'authored',1,1,42,3,'FREE','REAL',88,89,4096,0,5,9,104,'{}');
        """)
        result = summarize(self.timeline())
        self.assertEqual((result["allocation_calls"], result["free_calls"]), (2, 2))
        self.assertEqual([row["generation_id"] for row in result["generations"]], [1, 3])
        self.assertEqual([row["resolved_pointer_ordinal"] for row in result["generations"]], [1, 2])
        (domain,) = result["domains"]
        self.assertEqual(domain["requested_payload_bytes"], 96)
        self.assertEqual(domain["matched_free_bytes"], 96)
        self.assertEqual(domain["observed_retained_bytes"], 0)
        self.assertEqual(domain["resolved_payload_peak_minimum_bytes"], 64)
        self.assertEqual(domain["resolved_payload_peak_maximum_bytes"], 64)
        self.assertIsNone(domain["whole_domain_peak_bytes"])
        self.assertFalse(result["complete_generation_history_proven"])

    def test_effect_uncertainty_bounds_peak_instead_of_treating_API_start_as_effect(self):
        self.sql("""
          UPDATE rocpd_memory_allocate SET start=20,end=30 WHERE id=2;
          INSERT INTO rocpd_memory_allocate VALUES(3,'authored',1,1,42,3,'ALLOC','REAL',25,35,8192,32,5,9,103,'{}'),
            (4,'authored',1,1,42,3,'FREE','REAL',40,45,8192,0,5,9,104,'{}');
        """)
        result = summarize(self.timeline())
        (domain,) = result["domains"]
        self.assertEqual(domain["resolved_payload_peak_minimum_bytes"], 64)
        self.assertEqual(domain["resolved_payload_peak_maximum_bytes"], 96)
        self.assertEqual(
            result["generations"][0]["lifetime_bounds_ns"], dict(minimum=13, maximum=24)
        )

    def test_agent_real_virtual_scratch_and_unknown_pinned_state_remain_separate(self):
        self.sql("""
          INSERT INTO rocpd_info_agent VALUES(4,'authored',1,1,'cpu','CPU',NULL,0,0,0,'{}');
          INSERT INTO rocpd_memory_allocate VALUES(3,'authored',1,1,42,4,'ALLOC','REAL',30,31,4096,16,NULL,NULL,103,'{}'),
            (4,'authored',1,1,42,3,'ALLOC','VIRTUAL',32,33,4096,1024,5,9,103,'{}'),
            (5,'authored',1,1,42,3,'ALLOC','SCRATCH',34,35,4096,8,5,9,103,'{}');
        """)
        result = summarize(self.timeline())
        self.assertEqual(
            {(row["agent_id"], row["level"]) for row in result["domains"]},
            {(3, "REAL"), (4, "REAL"), (3, "VIRTUAL"), (3, "SCRATCH")},
        )
        self.assertEqual(result["requested_payload_bytes"], 1112)
        self.assertFalse(result["device_and_pinned_domains_combined"])
        self.assertFalse(result["sampled_RSS_included"])
        self.assertTrue(
            all(
                row["pinned_host_classification"].startswith("unresolved")
                for row in result["domains"]
            )
        )

    def test_overlapping_pointer_effects_and_unknown_free_do_not_invent_generations(self):
        self.sql("UPDATE rocpd_memory_allocate SET start=6,end=8 WHERE id=2;")
        result = summarize(self.timeline())
        self.assertFalse(result["generations"])
        self.assertEqual(len(result["unresolved_records"]), 2)
        self.assertEqual(result["domains"][0]["requested_payload_bytes"], 64)
        self.assertIsNone(result["domains"][0]["whole_domain_peak_bytes"])
        self.sql("UPDATE rocpd_memory_allocate SET address=8192,start=80,end=81 WHERE id=2;")
        result = summarize(self.timeline())
        self.assertEqual(
            result["unresolved_records"][0]["reason"], "free_without_observed_live_generation"
        )
        self.assertEqual(result["domains"][0]["observed_retained_bytes"], 64)

    def test_realloc_without_old_new_relation_and_null_agent_remain_unresolved(self):
        self.sql("""
          INSERT INTO rocpd_memory_allocate VALUES(3,'authored',1,1,42,3,'REALLOC','REAL',30,31,4096,128,5,9,103,'{}'),
            (4,'authored',1,1,42,NULL,'ALLOC','REAL',32,33,8192,16,NULL,NULL,103,'{}');
        """)
        result = summarize(self.timeline())
        self.assertEqual(result["realloc_or_reclaim_calls"], 1)
        self.assertEqual(result["unattributed_requested_payload_bytes"], 16)
        self.assertIsNone(result["generations"][0]["lifetime_bounds_ns"])
        self.assertEqual(result["domains"][0]["matched_free_bytes"], 0)
        self.assertFalse(result["production_latency"])
        value = copy.deepcopy(self.timeline())
        next(row for row in value["records"] if row["kind"] == "memory_lifecycle")["address"] = True
        with self.assertRaises(ValueError):
            summarize(value)

    def test_weighted_endpoint_policy_distinguishes_touching_and_instantaneous_effects(self):
        rows = [(1, 2, 64), (2, 3, 32)]
        self.assertEqual(weighted_peak(rows), 64)
        self.assertEqual(weighted_peak(rows, include_endpoints=True), 96)
        self.assertEqual(weighted_peak([(1, 1, 64)]), 0)
        self.assertEqual(weighted_peak([(1, 1, 64)], include_endpoints=True), 64)
        for rows in ([(2, 1, 1)], [(1, 2, True)], [(1, 2, -1)], [(1, 2)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                weighted_peak(rows)


if __name__ == "__main__":
    unittest.main()
