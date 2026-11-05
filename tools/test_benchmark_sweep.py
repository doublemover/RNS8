"""Aggregate independently runnable sweep contract tests."""
from __future__ import annotations

import unittest
import test_benchmark_sweep_case_scenarios
import test_benchmark_sweep_case_command_matrix
import test_benchmark_sweep_case_large_resume
import test_benchmark_sweep_case_chain_adaptive
import test_benchmark_sweep_case_cli_review
import test_benchmark_sweep_case_review_metadata
import test_benchmark_sweep_case_cache_wrap_exact

import inspect
import typing
from benchmark_sweep_lib import commands, reports


class RuntimeAnnotations(unittest.TestCase):
    def test_public_helpers_support_runtime_type_hints(self) -> None:
        for module in (commands, reports):
            for _, function in inspect.getmembers(module, inspect.isfunction):
                if function.__module__ == module.__name__:
                    typing.get_type_hints(function)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite([tests])
    suite.addTests(loader.loadTestsFromModule(test_benchmark_sweep_case_scenarios))
    suite.addTests(loader.loadTestsFromModule(test_benchmark_sweep_case_command_matrix))
    suite.addTests(loader.loadTestsFromModule(test_benchmark_sweep_case_large_resume))
    suite.addTests(loader.loadTestsFromModule(test_benchmark_sweep_case_chain_adaptive))
    suite.addTests(loader.loadTestsFromModule(test_benchmark_sweep_case_cli_review))
    suite.addTests(loader.loadTestsFromModule(test_benchmark_sweep_case_review_metadata))
    suite.addTests(loader.loadTestsFromModule(test_benchmark_sweep_case_cache_wrap_exact))
    if suite.countTestCases() != 8:
        raise RuntimeError("aggregate contract test inventory changed or became empty")
    return suite


def main() -> int:
    if not __debug__:
        raise RuntimeError("contract assertions require non-optimized Python")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(__name__))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
