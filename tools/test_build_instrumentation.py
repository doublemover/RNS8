"""Source-authored production admission controls; no sanitizer or benchmark runs."""

from __future__ import annotations

import unittest

from build_instrumentation import instrumentation_reasons
from call_observation_contract import validate_completed_call_samples
from test_benchmark_sweep_support import bounded_capture
from timing_eligibility import phase_blockers


class InstrumentationAdmission(unittest.TestCase):
    def test_declared_and_actual_flags_never_supply_production_latency(self):
        for configuration in (
            "host_sanitizer=address-undefined",
            "host_sanitizer=thread",
            "host_sanitizer=unrecognized",
            "host_coverage=ON",
            "windows_clang_asan=ON",
            "device_asan=ON",
            "device_asan=unknown",
            "host_coverage=unknown",
            "flags=-O1 -fsanitize=address,undefined -g",
            "flags='/fsanitize=address' /O1",
            'flags="-fprofile-instr-generate" -fcoverage-mapping',
            "cxx_flags=--coverage",
            "exe_linker_flags=-fsanitize=thread",
            "flags=-fprofile-arcs -ftest-coverage",
            "flags=-fsanitize-coverage=trace-pc-guard",
            "flags=-g -fpass-plugin=/installed/mull-ir-frontend-23",
            "flags='-fpass-plugin=/installed prefix/mull-ir-frontend-23' -g",
            "host_sanitizer=none\nhost_sanitizer=thread",
        ):
            with self.subTest(configuration=configuration):
                capture = bounded_capture("hip-direct", 100)
                capture["build_manifest"] = dict(configuration=configuration)
                self.assertTrue(instrumentation_reasons(capture["build_manifest"]))
                self.assertEqual(
                    phase_blockers(capture), ["instrumented_build_time_not_production_evidence"]
                )
                with self.assertRaisesRegex(ValueError, "instrumented build"):
                    validate_completed_call_samples(capture, 3)

    def test_clean_selectors_and_dependency_paths_are_not_instrumentation_flags(self):
        for configuration in (
            "host_sanitizer=none\nhost_coverage=OFF\nwindows_clang_asan=OFF\nflags=-O3 -g",
            "flags=/O2 /Zi\n",
            "synthetic fixture",
            "dependency_file.ASAN=/libraries/compiler-rt/asan.so",
            "dependency_tree.COVERAGE=/inputs/coverage/project",
            "device_asan=OFF",
        ):
            self.assertEqual(instrumentation_reasons(dict(configuration=configuration)), [])
        self.assertEqual(instrumentation_reasons(None), [])
        self.assertEqual(
            instrumentation_reasons(dict(configuration=False)),
            ["build_instrumentation_configuration_unreadable"],
        )


if __name__ == "__main__":
    unittest.main()
