"""Authored independent-run and profiler partition regressions; no GPU use."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import uuid

from observation_cohorts import aggregate_captures, evidence, phase_median, compound_median
from test_benchmark_sweep_support import finite_capture, bounded_capture
from test_capture_identity_support import synthetic_independent_runs
from test_fhe_workload_report import proxy_capture
from test_finite_modulus_map_report import finite_map_capture
import finite_modulus_map_report
import fhe_workload_report
import gpu_counter_report


def runs(capture, count=2):
    return synthetic_independent_runs([capture], count)


def write_all(root, captures):
    paths = []
    for index, capture in enumerate(captures):
        path = root / f"capture-{index}.json"
        path.write_text(json.dumps(capture)); paths.append(path)
    return paths


class ProcessObservations(unittest.TestCase):
    def test_duplicate_and_cohort_boundaries(self):
        a, b = runs(finite_capture("hip-direct", 100))
        result = aggregate_captures([a, b, deepcopy(a)])
        self.assertEqual(len(result), 1)
        self.assertTrue(evidence(result[0])["qualified"])
        self.assertEqual(evidence(result[0])["independent_runs"], 2)
        self.assertEqual(evidence(result[0])["submitted_records"], 3)
        self.assertEqual(evidence(result[0])["unique_records"], 2)
        self.assertFalse(evidence(aggregate_captures([a, deepcopy(a)])[0])["qualified"])
        for dimension in ("artifact", "device", "policy", "seed"):
            different = deepcopy(b)
            if dimension == "artifact": different["artifact_capture"]["executable_sha256"] = "f" * 64
            elif dimension == "device": different["device"]["name"] += " different device"
            elif dimension == "policy": different["exact_proof_policy"] = "scalar_first_v1"
            else: different["seed"] += 1
            cohorts = aggregate_captures([a, different])
            self.assertEqual(len(cohorts), 2)
            self.assertTrue(all(not evidence(c)["qualified"] for c in cohorts))
        legacy = deepcopy(a); legacy.pop("artifact_capture"); legacy.pop("execution_id")
        self.assertIn("independent_execution_provenance_unavailable", evidence(aggregate_captures([legacy])[0])["blockers"])

    def test_order_invariance_and_conflicting_reused_id(self):
        a, b = runs(finite_capture("hip-direct", 100))
        b["_path"] = "second.json"; a["_path"] = "first.json"
        first = aggregate_captures([a, b]); second = aggregate_captures([b, a])
        self.assertEqual(first, second)
        changed = deepcopy(a)
        changed["timing_summary_us"]["end_to_end"]["median"] = 200
        conflict = aggregate_captures([a, changed, b])[0]
        self.assertIn("conflicting_reused_execution_id", evidence(conflict)["blockers"])
        self.assertEqual(len(evidence(conflict)["submitted_observations"]), 3)

    def test_setup_inclusive_estimator_sums_each_process_first(self):
        observations = []
        for end, setup in ((100, 50), (120, 10), (140, 0)):
            capture = bounded_capture("hip-direct", end)
            capture["timing_summary_us"]["global_bound_scan"] = {"avg":setup,"median":setup,"p95":setup}
            observations.extend(runs(capture, 1))
        combined = aggregate_captures(observations)[0]
        self.assertEqual(phase_median(combined), 120)
        self.assertEqual(compound_median(combined, ["end_to_end", "global_bound_scan"]), 140)

    def test_finite_map_aggregates_before_selecting_and_self_ratio_is_one(self):
        captures = runs(finite_map_capture("cpu-reference", 2000))
        captures += runs(finite_map_capture("hip-direct", 900), 3)
        captures += runs(finite_map_capture("ck", 1500))
        with tempfile.TemporaryDirectory() as directory:
            paths = write_all(Path(directory), captures)
            options = dict(expected_shapes=[captures[0]["m"]], expected_ring_moduli=[255],
                           expected_field_moduli=[], expected_backends=["cpu-reference","hip-direct","ck"])
            forward = finite_modulus_map_report.build_report(paths, **options)
            reverse = finite_modulus_map_report.build_report(list(reversed(paths)), **options)
            self.assertEqual(forward, reverse)
            winner = forward["groups"][0]["winner"]
            self.assertEqual(winner["backend"], "hip-direct")
            self.assertEqual(winner["speedup_vs_direct_hip"], 1.0)
            self.assertEqual(winner["measurement_cohort"]["independent_runs"], 3)

    def test_fhe_mixed_shape_cannot_supply_another_shapes_cpu_anchor(self):
        cpu = proxy_capture("cpu-reference", "key_switch_digit_aggregation", 1000)
        gpu = proxy_capture("hip-direct", "key_switch_digit_aggregation", 400)
        old = gpu["m"]; gpu["m"] = old // 2
        def rewrite(value):
            if isinstance(value, dict): return {key: rewrite(item) for key,item in value.items()}
            if isinstance(value, list): return [rewrite(item) for item in value]
            if isinstance(value, str): return value.replace(f"m={old};", f"m={gpu['m']};").replace(f"--m {old} ", f"--m {gpu['m']} ")
            return value
        gpu = rewrite(gpu)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_all(Path(directory), runs(cpu) + runs(gpu))
            report = fhe_workload_report.build_report(paths)
            self.assertEqual(report["promotable_workload_profile_count"], 0)
            self.assertEqual(len(report["groups"]), 2)
            self.assertTrue(all(group["speedup_vs_cpu"] is None for group in report["groups"]))
            gpu_group = next(group for group in report["groups"] if "hip-direct" in group["backends"])
            self.assertIn("cpu_reference_baseline_missing", gpu_group["blockers"])


class CounterObservations(unittest.TestCase):
    def test_uneven_partition_missing_metrics_and_kernel_identity(self):
        rows = [{"KernelName":"gemm", "VGPR":64, "occupancy":0.75} for _ in range(9)]
        rows.append({"KernelName":"export", "VGPR":128, "occupancy":0.25})
        rows[0].pop("occupancy")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def save(name, records):
                path = root/name; path.write_text(json.dumps(records)); return gpu_counter_report.load_counter_file(path, 20)
            whole = save("whole.json", rows)
            split = [save("first.json", rows[:9]), save("second.json", rows[9:])]
            self.assertEqual(gpu_counter_report._merged_counter_metrics([whole]), gpu_counter_report._merged_counter_metrics(split))
            self.assertEqual(gpu_counter_report._merged_counter_metrics(split), gpu_counter_report._merged_counter_metrics(list(reversed(split))))
            self.assertEqual(gpu_counter_report._merged_counter_metrics(split)["VGPR"], 70.4)
            self.assertEqual(whole["numeric_metric_counts"]["occupancy"], 9)
            self.assertEqual(gpu_counter_report._merged_counter_groups([whole]), gpu_counter_report._merged_counter_groups(split))
            groups = gpu_counter_report._merged_counter_groups(split)
            self.assertEqual({g["identity"]["kernel"] for g in groups}, {"gemm","export"})
            self.assertEqual(next(g for g in groups if g["identity"]["kernel"]=="gemm")["numeric_metric_average"]["VGPR"], 64)
            units = save("units.json", [{"KernelName":"gemm","occupancy":0.5,"Unit":"fraction"},
                                       {"KernelName":"gemm","occupancy":50,"Unit":"percent"}])
            metrics = gpu_counter_report._merged_counter_metrics([units])
            self.assertEqual(metrics["occupancy [fraction]"], .5)
            self.assertEqual(metrics["occupancy [percent]"], 50)


if __name__ == "__main__":
    unittest.main()
