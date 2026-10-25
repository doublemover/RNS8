"""Synthetic evidence plumbing only; these fixtures are not GPU calibration."""
from dataclasses import replace
from fractions import Fraction
import unittest

from calibration_manifest import (Calibration, Identity, Quantity, Run, held_out_report,
                                  planner_rates, repeated_b_break_even, summarize, validate)


def fixture(label="training"):
    identity = Identity("synthetic-target", "synthetic-device", "runtime", "compiler", "a"*64,
                        "b"*64, "c"*64, "fixed", "fixed", "opcode", "kernel", "d"*64,
                        "hot", "canonical_three_limbs", "hip_event", "none")
    return Calibration(identity, "reconstruction", "elements", tuple(
        Run(f"{label}-{index}", (99.0+index, 100.0+index), 0.1, 64, 3, True, True,
            (Quantity("actual_hbm_bytes", None, "bytes", "unavailable"),
             Quantity("unique_source_bytes", 128, "bytes", "modeled"))) for index in range(5)))


class CalibrationTests(unittest.TestCase):
    def test_interval_and_rate(self):
        sample = fixture()
        report = summarize(sample)
        self.assertTrue(report["rate_export_eligible"])
        self.assertFalse(report["promotion_eligible"])
        self.assertLess(report["interval"]["lower_ns"], 101.5)
        self.assertGreater(report["interval"]["upper_ns"], 101.5)
        rates = planner_rates((sample,), {sample.work_kind: sample.identity})
        self.assertLess(rates[sample.work_kind].lower_units_per_ns, Fraction(64, 102))
        self.assertIsNone(report["raw"]["runs"][0]["counters"][0]["value"])

    def test_bad_observations(self):
        sample = fixture()
        run = sample.runs[0]
        for bad in (replace(run, elapsed_ns=(0.0,)), replace(run, elapsed_ns=(0.01,)),
                    replace(run, elapsed_ns=(float("nan"),)), replace(run, resolution_ns=True),
                    replace(run, work_units=True), replace(run, counters=(Quantity("missing", 0, "bytes", "unavailable"),))):
            with self.assertRaises(ValueError):
                validate(replace(sample, runs=(bad,)+sample.runs[1:]))
        with self.assertRaises(ValueError):
            validate(replace(sample, runs=(run, run)))

    def test_admission_is_not_raw_validation(self):
        sample = fixture()
        for rejected in (replace(sample, runs=sample.runs[:4]),
                         replace(sample, runs=(replace(sample.runs[0], elapsed_ns=(None,)),)+sample.runs[1:]),
                         replace(sample, runs=(replace(sample.runs[0], exact_reference_passed=False),)+sample.runs[1:]),
                         replace(sample, runs=(replace(sample.runs[0], isa_verified=False),)+sample.runs[1:]),
                         replace(sample, identity=replace(sample.identity, profiling_mode="kernel_replay"))):
            validate(rejected)
            self.assertFalse(summarize(rejected)["rate_export_eligible"])

    def test_public_pipeline_requires_explicit_uninstrumented_identity(self):
        sample = replace(fixture(), work_kind="public_pipeline")
        self.assertFalse(summarize(sample)["rate_export_eligible"])
        production = replace(sample, identity=replace(sample.identity, host_instrumentation_state="hooks_disabled"))
        self.assertTrue(summarize(production)["rate_export_eligible"])
        diagnostic = replace(sample, identity=replace(sample.identity, host_instrumentation_state="diagnostic_gpu_events"))
        self.assertFalse(summarize(diagnostic)["rate_export_eligible"])

    def test_retirement_and_environment(self):
        sample = fixture()
        for key in ("artifact_sha256", "isa_sha256", "configuration_sha256", "dataset_sha256"):
            changed = replace(sample.identity, **{key: "e"*64})
            with self.assertRaises(ValueError):
                planner_rates((sample,), {sample.work_kind: changed})
        other = replace(sample, work_kind="conversion", identity=replace(sample.identity, device="different"))
        with self.assertRaises(ValueError):
            planner_rates((sample, other), {sample.work_kind: sample.identity, other.work_kind: other.identity})

    def test_held_out_and_amortization(self):
        sample = fixture("held-out")
        report = held_out_report([("case", 99.0, 104.0, sample)], {"train-1"}, 0.1)
        self.assertTrue(report["held_out_passed"])
        self.assertFalse(held_out_report([("case", 50.0, 51.0, sample)], set(), 0.1)["held_out_passed"])
        with self.assertRaises(ValueError):
            held_out_report([("case", 99.0, 104.0, sample)], {"held-out-0"}, 0.1)
        self.assertEqual(repeated_b_break_even(Fraction(100), Fraction(10), Fraction(20)), 11)
        self.assertIsNone(repeated_b_break_even(Fraction(100), Fraction(20), Fraction(20)))


if __name__ == "__main__":
    unittest.main()
