"""Authored search/regression-scope controls; no native tool or workload runs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from native_variant_search import regression_scope, summarize

A, B, C = (letter * 64 for letter in "abc")
FAMILY = dict(
    schema="rns8_generated_native_wmma_v1", variants=[dict(id=value) for value in (A, B, C)]
)
COHORTS = {name: {A, B, C} for name in ("square", "skinny")}


def observations():
    rows = []
    for cohort in COHORTS:
        for variant, latency in ((A, 10), (B, 20), (C, 25 if cohort == "square" else 5)):
            for repeat in range(5):
                identity = f"{cohort}-{variant}-{repeat}"
                rows.append(
                    dict(
                        variant_id=variant,
                        cohort=cohort,
                        run_id=identity,
                        latency_ns=latency + repeat / 100,
                        payload_bytes=100,
                        prepare_ns=5,
                        capture=dict(
                            path=identity + ".json",
                            sha256=hashlib.sha256(identity.encode()).hexdigest(),
                        ),
                    )
                )
    return rows


def report(rows=None):
    return summarize(
        FAMILY, observations() if rows is None else rows, required_cohorts=COHORTS, baseline_ids={A}
    )


class Search(unittest.TestCase):
    def test_all_cohort_dominance_preserves_explicit_baseline_and_held_out_diversity(self):
        value = report()
        decisions = {row["variant_id"]: row for row in value["decisions"]}
        self.assertEqual(decisions[A]["state"], "retain_explicit_baseline")
        self.assertEqual(decisions[B]["state"], "conditional_retirement_candidate")
        self.assertEqual(decisions[B]["dominating_variants"], [A])
        self.assertEqual(decisions[C]["state"], "retain_no_complete_dominance")
        self.assertFalse(value["automatic_retirement"])
        self.assertFalse(value["promotion_allowed"])
        self.assertEqual(value["coverage_gaps"], [])

    def test_missing_cost_or_held_out_observations_cannot_become_zero_or_retirement(self):
        for missing in ("prepare_ns", "payload_bytes", "held_out"):
            rows = observations()
            if missing == "held_out":
                rows = [
                    row
                    for row in rows
                    if not (row["cohort"] == "skinny" and row["variant_id"] == B)
                ]
            else:
                for row in rows:
                    if row["variant_id"] == B:
                        row[missing] = None
            value = report(rows)
            decision = next(row for row in value["decisions"] if row["variant_id"] == B)
            self.assertEqual(decision["state"], "retain_no_complete_dominance")
            self.assertTrue(value["coverage_gaps"])
            if missing != "held_out":
                self.assertTrue(value["partial_objective_points"])

    def test_process_spread_and_cost_intervals_prevent_false_dominance(self):
        for metric, value in (("latency_ns", 4), ("payload_bytes", 50), ("prepare_ns", 1)):
            rows = observations()
            next(row for row in rows if row["variant_id"] == B)[metric] = value
            decision = next(row for row in report(rows)["decisions"] if row["variant_id"] == B)
            self.assertEqual(decision["state"], "retain_no_complete_dominance")

    def test_duplicate_processes_and_capture_copies_fail(self):
        for field in ("run_id", "capture"):
            rows = observations()
            rows[1][field] = deepcopy(rows[0][field])
            with self.subTest(field=field), self.assertRaises(ValueError):
                report(rows)

    def test_invalid_objectives_identity_and_schema_fail(self):
        for field, value in (
            ("latency_ns", True),
            ("latency_ns", float("nan")),
            ("latency_ns", 1 << 4096),
            ("payload_bytes", 1.0),
            ("payload_bytes", -1),
            ("prepare_ns", float("inf")),
            ("cohort", []),
            ("variant_id", "unknown"),
        ):
            rows = observations()
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                report(rows)
        with self.assertRaises(ValueError):
            summarize(FAMILY, [], required_cohorts=COHORTS, baseline_ids=set())
        with self.assertRaises(ValueError):
            summarize(dict(FAMILY, schema="future"), [], required_cohorts=COHORTS, baseline_ids={A})


class KnownRegressions(unittest.TestCase):
    def setUp(self):
        self.corpus = json.loads(
            (
                Path(__file__).resolve().parents[1] / "metadata/native-known-regressions.json"
            ).read_text()
        )

    def scope(self, **changes):
        policy = dict(
            component="composable_kernel",
            target="gfx1250",
            version="1.3.0",
            features={"mx_operands", "double_buffered_lds"},
        )
        policy.update(changes)
        return regression_scope(self.corpus, **policy)

    def test_reported_fixed_release_never_inherits_execution_or_qualification(self):
        value = self.scope()
        row = next(row for row in value["records"] if row["id"] == "mx-double-buffer-lds-order")
        self.assertEqual(row["state"], "reported_fix_release_requires_actual_regression")
        self.assertTrue(
            all(not row["executed"] and not row["qualified"] for row in value["records"])
        )
        row = next(
            row
            for row in self.scope(version="1.2.0")["records"]
            if row["id"] == "mx-double-buffer-lds-order"
        )
        self.assertEqual(row["state"], "affected_version_range_unknown_requires_actual_regression")

    def test_different_component_path_or_unreported_target_does_not_claim_safety(self):
        for changes, expected in (
            (
                dict(component="rns8_direct_hip"),
                "different_component_no_inherited_vendor_qualification",
            ),
            (
                dict(features={"direct_iu8_wmma"}),
                "different_path_no_inherited_vendor_qualification",
            ),
            (dict(target="gfx1100"), "target_not_reported_not_proven_unaffected"),
        ):
            value = self.scope(**changes)
            row = next(row for row in value["records"] if row["id"] == "mx-double-buffer-lds-order")
            self.assertEqual(row["state"], expected)

    def test_unknown_affected_range_or_duplicate_records_fail_closed(self):
        for change in ("range", "duplicate", "inherit"):
            if change == "range":
                self.corpus["affected_release_range"] = "all older versions"
            elif change == "duplicate":
                self.corpus["records"].append(deepcopy(self.corpus["records"][0]))
            else:
                self.corpus["qualification_inherited"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.scope()
            self.setUp()


if __name__ == "__main__":
    unittest.main()
