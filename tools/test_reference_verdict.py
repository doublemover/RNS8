"""Synthetic metadata contracts; no executable, compiler or device is invoked."""

import copy
import unittest

from reference_verdict import checksum_u64, numerical_observation, reference_verdict
from json_evidence import parse_evidence_json


def capture(checksum=7, *, complete=True, seed=3135):
    result = dict(
        m=2,
        n=3,
        k=5,
        semantics="bounded_i64",
        seed=seed,
        checksum_u64=checksum,
        backend_selected="hip-direct",
    )
    if complete:
        result["native_reference_verification"] = dict(
            method="boost_cpp_int_full_integer_reference",
            cache_key="a" * 64,
            payload_sha256="",
            checked_cells=6,
            elapsed_us=1.0,
            inside_production_interval=False,
        )
    return result


def cohort(records):
    view = copy.deepcopy(records[0])
    view["_observation_aggregate"] = dict(
        schema="rns8_process_observations_v1",
        numerical_observations=[numerical_observation(c) for c in records],
        submitted_observations=[
            dict(capture_sha256=numerical_observation(c)["capture_sha256"]) for c in records
        ],
    )
    return view


class ReferenceVerdictTest(unittest.TestCase):
    def test_evidence_json_rejects_duplicate_keys_nonfinite_constants_and_exponent_overflow(self):
        for text in (
            '{"checksum_u64":1,"checksum_u64":2}',
            '{"nested":[{"x":1,"x":2}]}',
            '{"x":NaN}',
            '{"x":Infinity}',
            '{"x":-Infinity}',
            '{"x":1e999}',
            '{"x":-1e999}',
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_evidence_json(text)
        self.assertEqual(
            parse_evidence_json('{"checksum_u64":0,"value":1.25}'), dict(checksum_u64=0, value=1.25)
        )

    def test_full_comparison_checksum_agreement_and_promotion_are_distinct(self):
        candidate, cpu = capture(), capture()
        cpu["backend_selected"] = "cpu-reference"
        verdict = reference_verdict(candidate, [cpu])
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.checksum_comparison, "matched")
        self.assertTrue(verdict.reference_coverage_present)
        self.assertEqual(verdict.full_output_comparison, "passed")
        self.assertTrue(
            reference_verdict(candidate).passed
        )  # independent full Boost postflight is itself evidence
        missing = capture(complete=False)
        verdict = reference_verdict(missing, [dict(missing, backend_selected="cpu-reference")])
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.checksum_comparison, "matched")
        self.assertIn("full_output_reference_pending", verdict.blockers)
        self.assertNotIn("promotion_eligible", verdict.to_json())

    def test_explicit_mismatch_and_conflicting_references_always_fail(self):
        self.assertFalse(reference_verdict(capture(8), [capture(7)]).passed)
        self.assertEqual(
            reference_verdict(capture(8), [capture(7)]).full_output_comparison, "mismatch"
        )
        verdict = reference_verdict(capture(7), [capture(7), capture(8)])
        self.assertFalse(verdict.passed)
        self.assertIn("reference_checksum_conflict", verdict.blockers)
        self.assertIn("checksum_mismatch_vs_reference", verdict.blockers)

    def test_zero_checksum_is_valid_but_alias_missing_boolean_and_overflow_are_not(self):
        for value in (0, 2**64 - 1):
            self.assertEqual(checksum_u64(capture(value)), value)
            self.assertTrue(reference_verdict(capture(value), [capture(value)]).passed)
        for value in (None, True, False, -1, 2**64, "7", 7.0):
            with self.subTest(value=value):
                candidate = capture(value)
                self.assertIsNone(checksum_u64(candidate))
                self.assertFalse(reference_verdict(candidate, [candidate]).passed)
        alias = capture()
        alias["checksum"] = alias.pop("checksum_u64")
        self.assertFalse(reference_verdict(alias, [alias]).passed)

    def test_partial_unbound_unknown_or_timed_reference_claims_remain_pending(self):
        for key, value in [
            ("checked_cells", 5),
            ("checked_cells", True),
            ("method", "cpu_name_only"),
            ("inside_production_interval", True),
            ("elapsed_us", float("nan")),
            ("cache_key", ""),
            ("cache_key", "g" * 64),
            ("payload_sha256", None),
        ]:
            with self.subTest(key=key):
                candidate = capture()
                candidate["native_reference_verification"][key] = value
                self.assertFalse(reference_verdict(candidate).passed)
        cached = capture()
        cached["native_reference_verification"]["method"] = "cached_full_integer_reference"
        self.assertFalse(reference_verdict(cached).passed)
        cached["native_reference_verification"]["payload_sha256"] = "b" * 64
        self.assertTrue(reference_verdict(cached).passed)

    def test_work_and_actual_input_keys_bind_comparisons_but_physical_padding_does_not(self):
        candidate = capture()
        same = dict(capture(), tile_m=64, tile_n=32, output_logical_ld=19, output_ld_padding=16)
        self.assertTrue(reference_verdict(candidate, [same]).passed)
        for key, value in [
            ("seed", 123),
            ("m", 3),
            ("semantics", "bounded_u64"),
            ("input_profile", "full-width"),
        ]:
            changed = capture()
            changed[key] = value
            verdict = reference_verdict(candidate, [changed])
            self.assertFalse(verdict.passed)
            self.assertIn("reference_work_contract_missing", verdict.blockers)
        changed = capture()
        changed["native_reference_verification"]["cache_key"] = "c" * 64
        self.assertFalse(reference_verdict(candidate, [changed]).passed)

    def test_each_independent_process_needs_consistent_content_and_coverage(self):
        first, second = capture(), capture()
        first["execution_id"] = "one"
        second["execution_id"] = "two"
        self.assertTrue(reference_verdict(cohort([first, second]), [first]).passed)
        second["checksum_u64"] = 8
        verdict = reference_verdict(cohort([first, second]), [first])
        self.assertFalse(verdict.passed)
        self.assertIn("candidate_checksum_conflict", verdict.blockers)
        second = capture(complete=False)
        second["execution_id"] = "two"
        self.assertFalse(reference_verdict(cohort([first, second]), [first]).passed)
        forged = cohort([first])
        forged["_observation_aggregate"]["numerical_observations"][0]["checksum_u64"] = 9
        self.assertFalse(reference_verdict(forged).passed)
        absent = cohort([first])
        absent["_observation_aggregate"].pop("numerical_observations")
        self.assertFalse(reference_verdict(absent).passed)
        malformed = cohort([first])
        malformed["_observation_aggregate"]["submitted_observations"][0]["capture_sha256"] = {}
        self.assertFalse(reference_verdict(malformed).passed)


if __name__ == "__main__":
    unittest.main()
