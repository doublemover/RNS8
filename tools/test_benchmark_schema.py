"""Aggregate independently runnable schema contract tests."""
from __future__ import annotations

import unittest
import test_benchmark_schema_case_vector_grouped
import test_benchmark_schema_case_reuse_chain
import test_benchmark_schema_case_oneshot_accelerators
import test_benchmark_schema_case_events_schedule


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite([tests])
    suite.addTests(loader.loadTestsFromModule(test_benchmark_schema_case_vector_grouped))
    suite.addTests(loader.loadTestsFromModule(test_benchmark_schema_case_reuse_chain))
    suite.addTests(loader.loadTestsFromModule(test_benchmark_schema_case_oneshot_accelerators))
    suite.addTests(loader.loadTestsFromModule(test_benchmark_schema_case_events_schedule))
    if suite.countTestCases() != 4:
        raise RuntimeError("aggregate contract test inventory changed or became empty")
    return suite


def main() -> int:
    if not __debug__:
        raise RuntimeError("contract assertions require non-optimized Python")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(__name__))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
