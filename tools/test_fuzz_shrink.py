"""Authored semantic minimizer controls with synthetic replay callbacks only."""

from dataclasses import replace
import unittest

from fuzz_cases import Case, admitted, decode, encode
from fuzz_shrink import Verdict, minimize

FAILURE = Verdict("failed", ("wrong_value", "export"))


class SemanticShrink(unittest.TestCase):
    def test_history_dimensions_and_values_shrink_with_same_failure_and_wire_contract(self):
        original = Case(0, 2, 2, 3, (0, 0, 0, 0, 73, 0), (1,) * 6, operations=7)
        observed = []

        def replay(case):
            self.assertEqual(decode(encode(case)), case)
            self.assertTrue(admitted(case))
            self.assertEqual((case.semantic, case.prefix, case.modulus), (0, 20, 0))
            observed.append(case)
            return FAILURE if case.operations & 4 and 73 in case.a else Verdict("passed")

        result = minimize(original, replay, max_replays=256)
        self.assertEqual(result.status, "local_fixed_point")
        self.assertEqual(result.original, original)
        self.assertEqual(result.minimized.operations, 4)
        self.assertEqual((result.minimized.m, result.minimized.n, result.minimized.k), (1, 1, 1))
        self.assertEqual(result.minimized.a, (73,))
        self.assertEqual(result.minimized.b, (0,))
        self.assertEqual(len(observed), len(result.attempts))
        summary = result.summary()
        self.assertGreater(summary["reduction_count"], 0)
        self.assertNotEqual(summary["original_sha256"], summary["minimized_sha256"])
        self.assertFalse(summary["production_latency"])

    def test_changed_failure_or_unsupported_is_never_a_reproduction(self):
        original = Case(0, 1, 1, 1, (73,), (1,), operations=0)
        for disposition in (
            Verdict("unsupported"),
            Verdict("rejected"),
            Verdict("failed", ("wrong_status", "create")),
        ):
            calls = []

            def replay(case, disposition=disposition, calls=calls):
                calls.append(case)
                return FAILURE if case == original else disposition

            result = minimize(original, replay, max_replays=32)
            self.assertEqual(result.minimized, original)
            self.assertTrue(all(not row["accepted"] for row in result.attempts[1:]))
            self.assertEqual(result.summary()["reduction_count"], 0)
            self.assertGreater(len(calls), 1)

    def test_only_admitted_originals_and_explicit_replay_failures_are_eligible(self):
        original = Case(4, 1, 1, 1, (128,), (0,), prefix=1)
        calls = []

        def replay(case):
            calls.append(case)
            return FAILURE

        with self.assertRaises(ValueError):
            minimize(original, replay)
        self.assertEqual(calls, [])
        admitted_case = replace(original, a=(127,))
        for disposition in (Verdict("passed"), Verdict("unsupported"), Verdict("rejected")):
            with self.assertRaises(ValueError):
                minimize(admitted_case, lambda case, result=disposition: result)
        with self.assertRaises(TypeError):
            minimize(admitted_case, lambda case: True)
        with self.assertRaisesRegex(RuntimeError, "adapter failed"):

            def broken(case):
                raise RuntimeError("adapter failed")

            minimize(admitted_case, broken)

    def test_replay_and_deadline_budgets_retain_original_reproducer(self):
        original = Case(0, 2, 2, 2, (1,) * 4, (1,) * 4)
        calls = []

        def replay(case):
            calls.append(case)
            return FAILURE

        result = minimize(original, replay, max_replays=1)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.minimized, original)
        self.assertEqual(len(calls), 1)
        tick = [0.0]

        def clock():
            tick[0] += 0.5
            return tick[0]

        result = minimize(original, replay, seconds=0.75, clock=clock)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(len(result.attempts), 1)
        for budget in (True, 0, 4097):
            with self.assertRaises(ValueError):
                minimize(original, replay, max_replays=budget)
        for seconds in (True, 0, float("nan"), float("inf"), 3601):
            with self.assertRaises(ValueError):
                minimize(original, replay, seconds=seconds)

    def test_failure_signature_is_bounded_and_cannot_be_attached_to_a_skip(self):
        for kind, signature in (
            ("unknown", ()),
            ("passed", ("x",)),
            ("failed", ()),
            ("failed", ["x"]),
            ("failed", ("x" * 257,)),
        ):
            with self.assertRaises(ValueError):
                Verdict(kind, signature)


if __name__ == "__main__":
    unittest.main()
