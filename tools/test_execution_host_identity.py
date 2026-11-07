"""Authored host/runtime-device cohort identities; no native host probe runs."""

from __future__ import annotations

from copy import deepcopy
import unittest

from execution_host_identity import (
    hardware_cohort_identity,
    hardware_identity_blockers,
    hardware_pair_blockers,
    host_identity,
)
from observation_cohorts import aggregate_captures, comparison_environment, evidence
from perf_variance_report import execution_cohort
from test_benchmark_sweep_support import bounded_capture
from test_capture_identity_support import (
    synthetic_execution_host_identity,
    synthetic_independent_runs,
)
from timing_eligibility import pair_blockers, phase_blockers


class ExecutionHostIdentities(unittest.TestCase):
    def test_explicit_cpu_gpu_policy_preserves_host_build_clock_and_processor_scope(self):
        from report_comparison import ComparisonPolicy

        cpu, gpu = bounded_capture("cpu-reference", 100), bounded_capture("hip-direct", 100)
        policy = ComparisonPolicy("explicit_same_host_cpu_gpu", cpu_gpu_comparison=True)
        ordinary = ComparisonPolicy("ordinary")
        self.assertTrue(policy.processor_pair(gpu, cpu))
        self.assertFalse(policy.processor_pair(gpu, gpu))
        self.assertFalse(policy.processor_pair(cpu, cpu))
        self.assertFalse(ordinary.processor_pair(gpu, cpu))
        self.assertEqual(policy.environment(cpu), policy.environment(gpu))
        self.assertEqual(
            policy.environment(cpu), comparison_environment(cpu, cpu_gpu_comparison=True)
        )
        with self.assertRaises(ValueError):
            comparison_environment(cpu, cpu_anchor=True, cpu_gpu_comparison=True)
        self.assertNotEqual(ordinary.environment(cpu), ordinary.environment(gpu))
        for change in ("host", "build", "clock", "thread_policy", "compiler"):
            changed = deepcopy(cpu)
            if change == "host":
                changed["execution_host_identity"]["os_instance_sha256"] = "b" * 64
            elif change == "build":
                changed["build_manifest"]["source_digest"] = "b" * 64
            elif change == "clock":
                changed["timing_metadata"]["clock_period_denominator"] = 1000000
            elif change == "thread_policy":
                changed["cpu_parallel"] = {"requested_threads": 999}
            else:
                changed["compiler"] = {"name": "other"}
            self.assertNotEqual(policy.environment(changed), policy.environment(gpu), change)
            if change not in ("host", "build"):
                self.assertEqual(
                    comparison_environment(changed, cpu_anchor=True),
                    comparison_environment(gpu, cpu_anchor=True),
                    change,
                )

    def test_native_envelope_and_observed_scope(self):
        capture = bounded_capture("hip-direct", 100)
        self.assertEqual(hardware_identity_blockers(capture), [])
        self.assertEqual(host_identity(capture).scope, "observed_os_instance")
        linux = deepcopy(capture)
        linux["execution_host_identity"] = synthetic_execution_host_identity(platform="linux")
        dbus = deepcopy(linux)
        dbus["execution_host_identity"]["source"] = "/var/lib/dbus/machine-id"
        self.assertEqual(host_identity(linux), host_identity(dbus))
        self.assertNotEqual(host_identity(capture), host_identity(linux))
        # Derived digest values have no reserved bit patterns.
        for identity in ("0" * 64, "f" * 64):
            capture["execution_host_identity"]["os_instance_sha256"] = identity
            self.assertEqual(host_identity(capture).os_instance_sha256, identity)

    def test_missing_failed_malformed_and_unknown_material_never_becomes_observed(self):
        base = bounded_capture("hip-direct", 100)
        for key, value in (
            ("schema", "unknown"),
            ("scope", "physical_board_attestation"),
            ("status", "unavailable"),
            ("platform", "unknown"),
            ("source", "environment_guess"),
            ("native_status", False),
            ("native_status", 5),
            ("os_instance_sha256", "0" * 63),
            ("os_instance_sha256", "f" * 65),
            ("os_instance_sha256", "A" * 64),
            ("os_instance_sha256", None),
            ("extra", True),
        ):
            capture = deepcopy(base)
            capture["execution_host_identity"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    host_identity(capture)
                self.assertTrue(hardware_identity_blockers(capture))
                self.assertEqual(
                    hardware_cohort_identity(capture)["host_os_instance"]["state"], "unqualified"
                )
        missing = deepcopy(base)
        missing.pop("execution_host_identity")
        self.assertIn("execution_host_identity_missing", phase_blockers(missing))

    def test_host_and_runtime_device_change_partition_process_cohorts(self):
        base = bounded_capture("hip-direct", 100)
        other_host = deepcopy(base)
        other_host["execution_host_identity"]["os_instance_sha256"] = "b" * 64
        other_device = deepcopy(base)
        for side in ("before", "after"):
            other_device["device_identity_witness"][side]["uuid"] = (
                "0000000000000000000000000000000b"
            )
        self.assertNotEqual(execution_cohort(base), execution_cohort(other_host))
        self.assertNotEqual(execution_cohort(base), execution_cohort(other_device))
        self.assertNotEqual(comparison_environment(base), comparison_environment(other_host))
        self.assertNotEqual(comparison_environment(base), comparison_environment(other_device))
        observations = aggregate_captures(
            synthetic_independent_runs([base])
            + synthetic_independent_runs([other_host])
            + synthetic_independent_runs([other_device])
        )
        self.assertEqual(len(observations), 3)
        self.assertTrue(all(evidence(view)["independent_runs"] == 2 for view in observations))
        self.assertIn("execution_host_identity_mismatch", pair_blockers(base, other_host))
        self.assertTrue(hardware_pair_blockers(base, other_device))

    def test_cpu_anchor_requires_the_same_observed_host_but_no_gpu_ordinal(self):
        cpu, gpu = bounded_capture("cpu-reference", 100), bounded_capture("hip-direct", 100)
        self.assertEqual(hardware_pair_blockers(cpu, gpu), [])
        self.assertEqual(
            comparison_environment(cpu, cpu_anchor=True),
            comparison_environment(gpu, cpu_anchor=True),
        )
        gpu["execution_host_identity"]["os_instance_sha256"] = "b" * 64
        self.assertEqual(hardware_pair_blockers(cpu, gpu), ["execution_host_identity_mismatch"])
        self.assertNotEqual(
            comparison_environment(cpu, cpu_anchor=True),
            comparison_environment(gpu, cpu_anchor=True),
        )

    def test_device_witness_must_remain_stable_and_available(self):
        base = bounded_capture("hip-direct", 100)
        for change in ("missing", "changed", "unknown"):
            capture = deepcopy(base)
            if change == "missing":
                capture.pop("device_identity_witness")
            elif change == "changed":
                capture["device_identity_witness"]["after"]["pci_bdf"] = "0000:08:00.0"
            else:
                capture["device_identity_witness"]["unrecognized"] = True
            self.assertTrue(hardware_identity_blockers(capture), change)
            self.assertEqual(
                hardware_cohort_identity(capture)["node_local_runtime_device"]["state"],
                "unqualified",
            )


if __name__ == "__main__":
    unittest.main()
