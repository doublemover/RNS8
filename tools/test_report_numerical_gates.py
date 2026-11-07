"""Synthetic negative controls for every standalone numerical report adapter."""

from __future__ import annotations

import copy
import unittest

import adaptive_grouped_scheduler_report as adaptive
import cpu_small_shape_selector_report as selector
import fhe_workload_report as fhe
import incremental_result_cache_report as incremental
import layout_search_report as layout
import tile_shape_report as tile
import vector_to_rns_chain_report as vector
from reference_verdict import CPUReferenceIndex
from report_comparison import ReportCohorts
from test_benchmark_sweep_support import finite_capture
from test_layout_search_report import with_common_layout
from test_review_reference_support import add_synthetic_full_reference


class ReportNumericalGatesTest(unittest.TestCase):
    def fixtures(self):
        candidate = add_synthetic_full_reference(finite_capture("hip-direct", 100))
        cpu = add_synthetic_full_reference(finite_capture("cpu-reference", 500))
        direct = copy.deepcopy(candidate)
        return candidate, cpu, direct

    def consumers(self, candidate, cpu, direct):
        references = CPUReferenceIndex([cpu])
        cohorts = ReportCohorts([candidate, cpu, direct])
        yield adaptive._row(candidate, {}, cohorts), "blockers"
        yield selector.row_for_capture(candidate, references), "blockers"
        yield fhe.row_for_capture(candidate, references), "blockers"
        yield incremental.row_for_capture(candidate, cohorts), "blockers"
        yield (
            tile.row_for_candidate(candidate, direct, tile.SelectedStaticResources(), cohorts),
            "promotion_blockers",
        )
        yield vector.classify_pair(candidate, direct, cohorts), "blockers"

    def test_all_row_adapters_preserve_mismatch_and_missing_coverage(self):
        for failure in ("mismatch", "missing"):
            candidate, cpu, direct = self.fixtures()
            if failure == "mismatch":
                candidate["checksum_u64"] ^= 1
                expected = "checksum_mismatch_vs_reference"
            else:
                candidate.pop("native_reference_verification")
                expected = "full_output_reference_pending"
            for row, field in self.consumers(candidate, cpu, direct):
                with self.subTest(failure=failure, fields=sorted(row)):
                    self.assertIn(expected, row[field])
                    self.assertFalse(row["numerical_reference"]["passed"])

    def test_conflicting_cpu_processes_are_not_hidden_by_timing_selection(self):
        candidate, cpu, _ = self.fixtures()
        conflicting = copy.deepcopy(cpu)
        conflicting["checksum_u64"] ^= 1
        index = CPUReferenceIndex([cpu, conflicting])
        verdict = index.verdict(candidate)
        self.assertFalse(verdict.passed)
        self.assertIn("reference_checksum_conflict", verdict.blockers)

    def test_cpu_name_without_matching_work_does_not_complete_comparison(self):
        candidate, cpu, _ = self.fixtures()
        cpu["seed"] += 1
        verdict = CPUReferenceIndex([cpu]).verdict(candidate)
        self.assertFalse(verdict.passed)
        self.assertIn("cpu_reference_work_coverage_missing", verdict.blockers)

    def test_pending_policy_remains_distinct_from_a_completed_numerical_receipt(self):
        candidate, cpu, _ = self.fixtures()
        candidate["verification_amortization"] = {
            "enabled": True,
            "final_exact_comparison_required": True,
            "final_exact_comparison_status": "reference_required",
            "promotion_eligible": False,
        }
        candidate["incremental_result_cache"] = dict(candidate["verification_amortization"])
        for row in (
            fhe.row_for_capture(candidate, CPUReferenceIndex([cpu])),
            incremental.row_for_capture(candidate, ReportCohorts([candidate, cpu])),
        ):
            self.assertTrue(row["numerical_reference"]["passed"])
            self.assertIn("final_exact_comparison_pending", row["blockers"])
            self.assertEqual(row["status"], "blocked")

    def test_vector_missing_checksums_do_not_compare_equal(self):
        candidate, _, direct = self.fixtures()
        candidate.pop("checksum_u64")
        direct.pop("checksum_u64")
        row = vector.classify_pair(candidate, direct, ReportCohorts([candidate, direct]))
        self.assertFalse(row["checksum_match"])
        self.assertIn("final_checksum_mismatch_or_missing", row["blockers"])

    def test_layout_pair_mismatch_blocks_local_promotion(self):
        candidate, _, direct = self.fixtures()
        baseline = with_common_layout(direct, name="baseline", role="baseline")
        subject = with_common_layout(
            candidate, name="candidate", role="candidate", baseline="baseline"
        )
        subject["checksum_u64"] ^= 1
        (row,) = layout.candidate_rows([baseline, subject])
        self.assertFalse(row["promotion_eligible"])
        self.assertFalse(row["numerical_reference"]["passed"])
        self.assertIn("checksum_mismatch_vs_reference", row["promotion_blockers"])


if __name__ == "__main__":
    unittest.main()
