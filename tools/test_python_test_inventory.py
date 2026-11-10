"""Authored native unittest journaling controls; intentional cases stay local."""

from __future__ import annotations

import io
import unittest

from python_test_inventory import Journal, cases


def run_fixture(test_type):
    selected = list(cases(unittest.TestLoader().loadTestsFromTestCase(test_type)))
    names = [test.id() for test in selected]

    def factory(stream, descriptions, verbosity, **kwargs):
        return Journal(stream, descriptions, verbosity, selected=names, **kwargs)

    result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=factory).run(
        unittest.TestSuite(selected)
    )
    return result.document(names, names, [])


class NativeJournal(unittest.TestCase):
    def test_optional_native_observer_tracks_case_scope_and_always_closes_it(self):
        class Positive(unittest.TestCase):
            def test_case(self):
                self.assertTrue(True)

        events = []
        selected = list(cases(unittest.TestLoader().loadTestsFromTestCase(Positive)))
        names = [test.id() for test in selected]

        def factory(stream, descriptions, verbosity, **kwargs):
            return Journal(
                stream, descriptions, verbosity, selected=names, observer=events.append, **kwargs
            )

        unittest.TextTestRunner(stream=io.StringIO(), resultclass=factory).run(
            unittest.TestSuite(selected)
        )
        self.assertEqual(events, [names[0], None])

    def test_passing_cases_have_exact_discovery_execution_and_pass_denominators(self):
        class Positive(unittest.TestCase):
            def test_first(self):
                self.assertEqual(2 + 2, 4)

            def test_second(self):
                for value in (1, 2):
                    with self.subTest(value=value):
                        self.assertGreater(value, 0)

        value = run_fixture(Positive)
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["counts"]["executed"], 2)
        self.assertEqual(value["counts"]["passed"], 2)
        self.assertEqual(value["discovered"], value["selected"])
        self.assertEqual(value["selected"], value["executed"])
        self.assertFalse(value["production_latency"])

    def test_failure_error_and_subtest_failure_do_not_become_passing_parent_cases(self):
        class Negative(unittest.TestCase):
            def test_assertion(self):
                self.assertEqual(1, 2)

            def test_exception(self):
                raise ValueError("native fixture error")

            def test_subtest(self):
                with self.subTest(value=3):
                    self.assertEqual(3, 4)

        value = run_fixture(Negative)
        self.assertEqual(value["status"], "failed")
        self.assertEqual(value["counts"]["failed"], 2)
        self.assertEqual(value["counts"]["error"], 1)
        self.assertEqual(value["counts"]["passed"], 0)
        self.assertEqual(value["counts"]["executed"], 3)

    def test_skip_expected_failure_and_unexpected_success_are_separate_unaccepted_states(self):
        class NonPassing(unittest.TestCase):
            @unittest.skip("review still required")
            def test_skip(self):
                pass

            @unittest.expectedFailure
            def test_expected(self):
                self.fail("expected fixture failure")

            @unittest.expectedFailure
            def test_unexpected(self):
                pass

        value = run_fixture(NonPassing)
        self.assertEqual(value["status"], "failed")
        for state in ("skipped", "expected_failure", "unexpected_success"):
            self.assertEqual(value["counts"][state], 1)
        self.assertEqual(value["counts"]["started"], 3)
        self.assertEqual(value["counts"]["executed"], 2)
        self.assertEqual(value["counts"]["passed"], 0)

    def test_native_class_fixture_error_keeps_unexecuted_cases_visible(self):
        class BrokenClass(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise RuntimeError("setup could not start")

            def test_never_started(self):
                pass

        value = run_fixture(BrokenClass)
        self.assertEqual(value["status"], "failed")
        self.assertEqual(value["counts"]["not_run"], 1)
        self.assertEqual(value["counts"]["executed"], 0)
        self.assertEqual(len(value["fixture_failures"]), 1)


if __name__ == "__main__":
    unittest.main()
