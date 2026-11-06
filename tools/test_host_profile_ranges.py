#!/usr/bin/env python3
"""Authored same-clock host-range controls; no profiler or native code runs."""

from __future__ import annotations

import copy
import unittest

from host_profile_native import SCHEMA
from host_profile_ranges import join
from interval_observations import interval_summary


def scope(identity=0, parent=None, *, start=100, end=300, phase=10, thread=42, operation=1):
    return dict(
        range_id=identity,
        parent_range_id=parent,
        operation_id=operation,
        iteration_id=0,
        operation_namespace=1,
        role=3,
        phase=phase,
        stop_state=1,
        process_id=41,
        start_thread_id=thread,
        stop_thread_id=thread,
        start_before_ns=start,
        start_after_ns=start + 1,
        stop_before_ns=end - 1,
        stop_after_ns=end,
    )


def snapshot(rows=None):
    rows = [scope()] if rows is None else rows
    return dict(
        schema="rns8_profiler_ranges_v1",
        provider_status="active",
        capacity=4096,
        claimed=len(rows),
        dropped=0,
        unpublished=0,
        records=rows,
        host_clock=dict(
            schema="rns8_linux_host_range_clock_v1",
            clock_id=1,
            name="CLOCK_MONOTONIC",
            unit="nanoseconds",
            timestamp_scope="brackets_around_native_ROCTx_calls",
        ),
    )


def sample(time, *, thread=42):
    return dict(pid=41, tid=thread, time_ns=time, period=10, ip=4099, cpu=3)


def native(rows=None):
    return dict(
        schema=SCHEMA,
        pid=41,
        scope="selected_process_total_not_selected_operation",
        clock=dict(attribute_clock_id=1),
        samples=[sample(150)] if rows is None else rows,
    )


class HostRanges(unittest.TestCase):
    def test_nested_scopes_attribute_exclusively_without_summing_parent_time(self):
        ranges = snapshot(
            [
                scope(),
                scope(1, 0, start=120, end=200, phase=2),
                scope(2, 0, start=220, end=250, phase=6),
            ]
        )
        report = join(native([sample(110), sample(150), sample(230), sample(270)]), ranges)
        self.assertEqual(report["selected_samples"], 4)
        self.assertEqual(
            report["exclusive_sampled_phases"],
            dict(
                proof=dict(samples=1, sampled_cpu_exposure_ns=10),
                completion=dict(samples=1, sampled_cpu_exposure_ns=10),
                public_call=dict(samples=2, sampled_cpu_exposure_ns=20),
            ),
        )
        self.assertEqual(report["measured_host_windows"]["interval_union_ns"], 198)
        self.assertEqual(report["measured_host_windows"]["overlap_service_ns"], 0)
        self.assertFalse(report["production_latency"])
        self.assertFalse(report["source_content_verified"])

    def test_observer_brackets_are_ambiguous_even_inside_parent_body(self):
        child = scope(1, 0, start=120, end=200, phase=2)
        child.update(start_after_ns=130, stop_before_ns=190)
        report = join(
            native([sample(t) for t in (99, 100, 101, 125, 150, 195, 299, 300)]),
            snapshot([scope(), child]),
        )
        self.assertEqual(report["selected_samples"], 2)
        self.assertEqual(report["observer_boundary_ambiguous_samples"], 4)
        self.assertEqual(report["outside_or_unbound_thread_samples"], 2)

    def test_independently_bound_threads_preserve_overlap_and_unbound_samples(self):
        ranges = snapshot([scope(), scope(1, thread=43, operation=2)])
        report = join(native([sample(150), sample(150, thread=43), sample(150, thread=44)]), ranges)
        self.assertEqual(report["selected_operation_count"], 2)
        self.assertEqual(report["selected_samples"], 2)
        self.assertEqual(report["outside_or_unbound_thread_samples"], 1)
        self.assertEqual(report["measured_host_windows"]["service_ns"], 396)
        self.assertEqual(report["measured_host_windows"]["interval_union_ns"], 198)
        self.assertEqual(report["measured_host_windows"]["overlap_service_ns"], 198)

    def test_async_completion_does_not_become_CPU_service_or_bind_worker(self):
        child = scope(1, 0, start=120, end=200, phase=6)
        child.update(stop_state=2, stop_thread_id=43)
        report = join(native([sample(150), sample(150, thread=43)]), snapshot([scope(), child]))
        self.assertEqual(set(report["exclusive_sampled_phases"]), {"public_call"})
        self.assertEqual(report["outside_or_unbound_thread_samples"], 1)
        self.assertFalse(report["worker_scope_inferred"])

    def test_invalid_identity_clock_and_snapshot_contracts_fail_closed(self):
        changes = [
            lambda n, s: n["clock"].update(attribute_clock_id=True),
            lambda n, s: s["host_clock"].update(clock_id=0),
            lambda n, s: s.update(dropped=False),
            lambda n, s: s.update(unpublished=1),
            lambda n, s: s.update(claimed=2),
            lambda n, s: s["records"][0].update(process_id=42),
            lambda n, s: s["records"][0].update(phase=11),
            lambda n, s: s["records"][0].update(role=True),
            lambda n, s: s["records"][0].update(parent_range_id=7),
            lambda n, s: s["records"][0].update(parent_range_id=0),
            lambda n, s: s["records"][0].update(start_after_ns=None),
            lambda n, s: s["records"][0].update(stop_before_ns=99),
            lambda n, s: s["records"][0].update(stop_state=0),
            lambda n, s: s["records"][0].update(stop_thread_id=43),
            lambda n, s: s["records"][0].update(extra=1),
            lambda n, s: n["samples"][0].update(pid=42),
            lambda n, s: n["samples"][0].update(period=True),
        ]
        for change in changes:
            with self.subTest(change=change):
                n, s = native(), snapshot()
                change(n, s)
                with self.assertRaises(ValueError):
                    join(n, s)

    def test_duplicate_nested_and_temporally_invalid_ancestry_is_rejected(self):
        invalid = [
            [scope(), scope()],
            [scope(), scope(1, 0, start=120, end=200, operation=2)],
            [
                scope(),
                scope(1, 0, start=120, end=200, phase=2),
                scope(2, 0, start=150, end=220, phase=3),
            ],
            [scope(), scope(1, 0, start=301, end=400, phase=2)],
            [scope(), scope(1, 0, start=100, end=200, phase=2)],
            [scope(), dict(scope(1, 0, start=120, end=200, phase=2), role=4)],
        ]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                join(native(), snapshot(rows))

    def test_reordered_inputs_have_the_same_witness_and_empty_exposure_is_inconclusive(self):
        n, s = (
            native([sample(110), sample(150)]),
            snapshot([scope(), scope(1, 0, start=120, end=200, phase=2)]),
        )
        expected = join(n, s)
        n, s = copy.deepcopy(n), copy.deepcopy(s)
        n["samples"].reverse()
        s["records"].reverse()
        self.assertEqual(join(n, s), expected)
        with self.assertRaisesRegex(ValueError, "sampling exposure"):
            join(native([sample(99)]), snapshot())

    def test_interval_union_uses_true_maximum_and_retains_gaps(self):
        observed = interval_summary([(0, 100), (10, 20), (110, 120)])
        self.assertEqual(
            observed,
            dict(
                count=3,
                service_ns=120,
                interval_union_ns=110,
                span_ns=120,
                uncovered_span_ns=10,
                overlap_service_ns=10,
            ),
        )
        for intervals in ([1], [(True, 3)], [(4, 3)], [(0, float("inf"))]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                interval_summary(intervals)


if __name__ == "__main__":
    unittest.main()
