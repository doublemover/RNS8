"""Authored schedule custody/phase controls; no recorder or workload launched."""

from __future__ import annotations

import copy
import unittest

from host_schedule_attribution import summarize
from perf_recording_contract import SCHEMA
from test_host_profile_ranges import scope, snapshot


def switch(ordinal, time, direction, *, tid=42, pid=41, preempted=False):
    return dict(
        record_index=ordinal,
        pid=pid,
        tid=tid,
        time_ns=time,
        clock_id=1,
        direction=direction,
        preempted=preempted,
    )


def recording(rows, boundaries=None):
    return dict(
        schema=SCHEMA,
        complete_record_walk=True,
        clock=dict(attribute_clock_id=1),
        context_switches=rows,
        task_lifetime_boundaries=boundaries or [],
        blockers=[],
    )


class Scheduling(unittest.TestCase):
    def test_off_cpu_phase_intersections_do_not_invent_blocked_or_gpu_cost(self):
        rows = [
            switch(1, 110, "out", preempted=True),
            switch(2, 190, "in"),
            switch(3, 200, "out", tid=43),
            switch(4, 250, "in", tid=43),
            switch(5, 100, "out", pid=99),
            switch(6, 280, "in", pid=99),
        ]
        report = summarize(
            recording(list(reversed(rows))),
            pid=41,
            snapshot=snapshot([scope(), scope(1, 0, start=120, end=180, phase=2)]),
        )
        self.assertEqual(report["conditional_off_cpu_thread_ns"], 130)
        self.assertEqual(
            report["selected_operation"]["exclusive_phase_off_cpu_thread_ns"],
            {"proof": 58, "public_call": 22},
        )
        self.assertEqual(report["observed_switch_records"], 4)
        self.assertEqual(report["intervals"][0]["state"], "preempted_TASK_RUNNING_at_out")
        self.assertIsNone(report["exact_blocked_time_ns"])
        self.assertIsNone(report["wakeup_to_run_delay_ns"])
        self.assertFalse(report["complete_scheduler_coverage_proven"])
        self.assertFalse(report["gpu_dependency_edges_verified"])

    def test_same_time_and_lifetime_boundaries_do_not_bridge_generations(self):
        rows = [
            switch(1, 110, "out"),
            switch(2, 150, "in"),
            switch(3, 150, "out"),
            switch(4, 200, "in"),
            switch(5, 220, "out"),
            switch(7, 260, "in"),
        ]
        boundary = dict(pid=41, tid=42, record_index=6, clock_id=1, time_ns=240, kind="exit")
        report = summarize(recording(rows, [boundary]), pid=41)
        self.assertEqual(report["intervals"], [])
        self.assertIn("same_time_order_ambiguous", {row["reason"] for row in report["unresolved"]})
        self.assertIn(
            "task_lifetime_boundary_before_in", {row["reason"] for row in report["unresolved"]}
        )
        boundary.update(tid=44, kind="exec")
        self.assertIn(
            "process_exec_before_in",
            {row["reason"] for row in summarize(recording(rows, [boundary]), pid=41)["unresolved"]},
        )

    def test_foreign_phase_masks_parent_and_loss_remains_partial(self):
        data = recording([switch(1, 110, "out"), switch(2, 190, "in")])
        data["blockers"] = ["recorded_loss"]
        report = summarize(
            data,
            pid=41,
            snapshot=snapshot([scope(), scope(1, 0, start=120, end=180, phase=2, operation=99)]),
        )
        self.assertEqual(report["selected_operation"]["conditional_off_cpu_thread_ns"], 22)
        self.assertEqual(report["recording_blockers"], ["recorded_loss"])
        self.assertFalse(report["promotion_allowed"])

    def test_missing_pairs_and_malformed_identity_remain_unresolved_or_rejected(self):
        data = recording([switch(1, 110, "in"), switch(2, 120, "out"), switch(3, 130, "out")])
        report = summarize(data, pid=41)
        self.assertEqual(len(report["unresolved"]), 3)
        for change in (
            dict(pid=True),
            dict(time_ns=-1),
            dict(clock_id=True),
            dict(preempted=1),
            dict(direction="sleep"),
        ):
            altered = copy.deepcopy(data)
            altered["context_switches"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize(altered, pid=41)

    def test_unobserved_child_stop_withholds_parent_phase_credit(self):
        child = scope(1, 0, start=120, end=180, phase=2)
        child.update(stop_state=0, stop_thread_id=None, stop_before_ns=None, stop_after_ns=None)
        report = summarize(
            recording([switch(1, 110, "out"), switch(2, 190, "in")]),
            pid=41,
            snapshot=snapshot([scope(), child]),
        )
        self.assertEqual(report["selected_operation"]["intervals"], [])
        self.assertEqual(report["selected_operation"]["phase_unbound_threads"], [42])


if __name__ == "__main__":
    unittest.main()
