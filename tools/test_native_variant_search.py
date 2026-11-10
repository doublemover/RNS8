"""Authored search/regression-scope controls; no native tool or workload runs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from native_variant_search import (
    collection_schedule,
    combination_coverage,
    regression_scope,
    summarize,
)
from check_generated_native_wmma import load_bound_inspection
from native_wmma_lowering import fingerprint

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
    def test_combination_coverage_never_inherits_other_pitches_or_component_qualification(self):
        manifest = dict(
            variants=[dict(id=A, kernel_symbol="kernel_a", compatibility=dict(group=1))]
        )
        case = dict(m=17, n=19, k=257, prefix=3)
        configuration = dict(case, lda=257, ldb=19, ldc=19)
        logical = dict(
            source_current=True,
            path="logical.json",
            sha256="a" * 64,
            inventory=dict(
                variants=[
                    dict(
                        id=A,
                        configurations=[
                            dict(
                                configuration=configuration,
                                logical_verdict="proven",
                                memory_verdict="proven",
                            )
                        ],
                    )
                ]
            ),
        )
        rows = [dict(variant_id=A, cohort="tail")]
        value = combination_coverage(manifest, rows, dict(tail=case), logical=logical)
        variant = value["variants"][0]
        self.assertEqual(variant["runtime_microcases"][0]["owned_invocations"], 1)
        self.assertTrue(variant["runtime_microcases"][0]["logical_memory"])
        self.assertIn("exact_executable_symbol_inspection_not_recorded", variant["coverage_gaps"])
        self.assertFalse(variant["qualified"])
        self.assertFalse(value["component_qualification_inherited"])
        changed = deepcopy(logical)
        changed["inventory"]["variants"][0]["configurations"][0]["configuration"]["lda"] += 1
        value = combination_coverage(manifest, rows, dict(tail=case), logical=changed)
        self.assertIsNone(value["variants"][0]["runtime_microcases"][0]["logical_memory"])
        self.assertIn(
            "source_bound_logical_configuration_not_recorded", value["variants"][0]["coverage_gaps"]
        )

    def test_inspection_join_binds_exact_executable_family_checker_and_negative_verdicts(self):
        manifest = dict(
            source_sha256="b" * 64,
            header_sha256="c" * 64,
            variants=[dict(id=A, kernel_symbol="kernel_a", spec=dict(target="gfx1100"))],
        )
        report = dict(
            schema="rns8_generated_native_isa_v2",
            target="gfx1100",
            generated_manifest_sha256=fingerprint(manifest),
            generated_source_sha256=manifest["source_sha256"],
            generated_header_sha256=manifest["header_sha256"],
            checker_sha256=hashlib.sha256(
                Path(__file__).with_name("check_generated_native_wmma.py").read_bytes()
            ).hexdigest(),
            artifact=dict(host_object_sha256="e" * 64),
            symbols=[dict(id=A, symbol="kernel_a")],
            inspection_passed=True,
            inspection_failures=[],
            device_differential_executed=False,
            performance_qualified=False,
            promotion_eligible=False,
        )
        scratch = Path(__file__).resolve().parents[1] / "temp"
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "inspection.json"

            def load(value):
                path.write_text(json.dumps(value), encoding="utf-8")
                return load_bound_inspection(path, manifest, "e" * 64)

            self.assertTrue(load(report)["executable_bound"])
            failed = dict(
                report, inspection_passed=False, inspection_failures=["missing_matrix_instruction"]
            )
            self.assertFalse(load(failed)["report"]["inspection_passed"])
            for field, value in (
                ("checker_sha256", "0" * 64),
                ("generated_manifest_sha256", "0" * 64),
                ("generated_header_sha256", "0" * 64),
                ("target", "gfx942"),
                ("artifact", dict(host_object_sha256="f" * 64)),
                ("symbols", [dict(id=A, symbol="sibling")]),
                ("symbols", []),
                ("inspection_passed", 1),
                ("inspection_failures", ["failure"]),
                ("promotion_eligible", True),
            ):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load(dict(report, **{field: value}))

    def test_coarse_to_fine_preserves_all_controls_and_held_out_repeats(self):
        cases = [
            dict(candidate=name, m=m, n=n, k=257)
            for name in ("generated_wmma", "wmma", "pack_gemm")
            for m, n in ((256, 256), (1, 33), (17, 19))
        ]
        jobs = collection_schedule(cases, 5, 3135)
        self.assertEqual(jobs, collection_schedule(cases, 5, 3135))
        self.assertEqual(len(jobs), 45)
        self.assertEqual(len({(index, repeat) for index, repeat, _ in jobs}), 45)
        self.assertTrue(
            all(
                repeat == 0 and stage == "coarse_single_invocation" for _, repeat, stage in jobs[:9]
            )
        )
        self.assertTrue(
            all(repeat > 0 and stage == "fine_independent_repeats" for _, repeat, stage in jobs[9:])
        )
        work = [
            cases[index]["m"] * cases[index]["n"] * cases[index]["k"] for index, _, _ in jobs[:9]
        ]
        self.assertEqual(work, sorted(work))
        self.assertEqual(len(collection_schedule(cases, 1, 1)), 9)
        ordinary = [dict(candidate="wmma")]
        self.assertTrue(
            all(stage == "ordinary" for _, _, stage in collection_schedule(ordinary, 5, 1))
        )
        for runs in (True, 0, 101, 1.0):
            with self.assertRaises(ValueError):
                collection_schedule(cases, runs, 1)

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
