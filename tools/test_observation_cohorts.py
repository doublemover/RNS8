"""Authored independent-run and profiler partition regressions; no GPU use."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from observation_cohorts import (
    aggregate_captures,
    evidence,
    phase_median,
    compound_median,
    amortized_prepack_measurement,
)
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
        path.write_text(json.dumps(capture))
        paths.append(path)
    return paths


class ProcessObservations(unittest.TestCase):
    def test_native_once_setup_scalar_needs_matching_alias_and_explicit_scope(self):
        base = bounded_capture("hip-direct", 100)
        base["prepack_setup_us"] = base["avg_prepack_setup_us"] = 123
        base["timing_metadata"].setdefault("phase_availability", {})["prepack_setup"] = dict(
            timed=True, timing_key="prepack_setup_us", scope="one_time_before_warmups"
        )
        self.assertNotIn("prepack_setup", base["raw_timings_us"])
        self.assertEqual(phase_median(base, "prepack_setup"), 123)
        for field, value in (
            ("avg_prepack_setup_us", 124),
            ("prepack_setup_us", None),
            ("prepack_setup_us", True),
        ):
            invalid = deepcopy(base)
            invalid[field] = value
            invalid["timing_summary_us"]["prepack_setup"] = dict(avg=123, median=123, p95=123)
            self.assertIsNone(phase_median(invalid, "prepack_setup"))
        for field, value in (
            ("scope", "inside_every_call"),
            ("timing_key", "other"),
            ("timed", False),
        ):
            invalid = deepcopy(base)
            invalid["timing_metadata"]["phase_availability"]["prepack_setup"][field] = value
            self.assertIsNone(phase_median(invalid, "prepack_setup"))

    def test_prepack_setup_amortizes_each_process_over_its_own_completed_repeats(self):
        from itertools import permutations
        from report_comparison import ComparisonPolicy
        from test_capture_identity_support import synthetic_host_measurements

        base = bounded_capture("hip-direct", 100)
        base["reuse_packed_inputs"] = True
        base["prepack_setup_us"] = base["avg_prepack_setup_us"] = 900
        base["timing_metadata"].setdefault("phase_availability", {})["prepack_setup"] = dict(
            timed=True, timing_key="prepack_setup_us", scope="one_time_before_warmups"
        )
        captures = runs(base, 3)
        for record, repeats in zip(captures, (9, 18, 27)):
            original_repeats = record["repeats"]
            record["repeats"] = repeats
            setup = repeats * 100
            record["prepack_setup_us"] = record["avg_prepack_setup_us"] = setup
            for phase, values in record["raw_timings_us"].items():
                if len(values) == original_repeats:
                    record["raw_timings_us"][phase] = [values[0]] * repeats
            synthetic_host_measurements(record)
        inclusive = ComparisonPolicy("amortized_preparation", amortize_prepack_setup=True)
        for order in permutations(captures):
            views = aggregate_captures(list(order))
            self.assertEqual(len(views), 1)
            view = views[0]
            measured = amortized_prepack_measurement(view)
            self.assertTrue(
                all("prepack_setup" not in record["raw_timings_us"] for record in order)
            )
            self.assertEqual(measured["median_us"], 200)
            self.assertEqual(measured["process_relative_range"], 0)
            self.assertEqual(measured["blockers"], [])
            self.assertEqual(
                sorted(item["repeat_count"] for item in measured["observations"]), [9, 18, 27]
            )
            self.assertEqual(inclusive.measurement(view)["median_us"], 200)
            self.assertEqual(ComparisonPolicy("steady").measurement(view)["median_us"], 100)
        view = aggregate_captures([*captures, deepcopy(captures[0])])[0]
        self.assertEqual(len(amortized_prepack_measurement(view)["observations"]), 3)
        noisy = deepcopy(view)
        noisy["_observation_aggregate"]["observations"][0]["phase_medians_us"]["prepack_setup"] = (
            100000
        )
        self.assertIn(
            "setup_inclusive_process_spread_exceeds_ten_percent",
            amortized_prepack_measurement(noisy)["blockers"],
        )
        for key, value in (("measured_repeat_count", 0), ("measured_repeat_count", True)):
            malformed = deepcopy(view)
            malformed["_observation_aggregate"]["observations"][0][key] = value
            self.assertIsNone(amortized_prepack_measurement(malformed)["median_us"])
        missing = deepcopy(view)
        missing["_observation_aggregate"]["observations"][0]["phase_medians_us"].pop(
            "prepack_setup"
        )
        self.assertIn(
            "prepack_setup_process_measurement_invalid",
            amortized_prepack_measurement(missing)["blockers"],
        )
        overflow = deepcopy(view)
        observation = overflow["_observation_aggregate"]["observations"][0]
        observation["measured_repeat_count"] = 1
        observation["phase_medians_us"].update(end_to_end=1.5e308, prepack_setup=1.5e308)
        self.assertIsNone(amortized_prepack_measurement(overflow)["median_us"])

    def test_cache_dirty_composition_splits_implementations_but_process_handles_do_not(self):
        from test_incremental_result_cache_report import public_cache_capture

        observations = []
        for extent in (16, 32):
            base = public_cache_capture("hip-direct", 800, "public_result_cache_candidate", True)
            base["incremental_result_cache"]["dirty_regions"][0]["row_extent"] = extent
            records = runs(base)
            for ordinal, record in enumerate(records):
                cache = record["incremental_result_cache"]
                cache["a_matrix_instance_id"] = 101 + ordinal
                cache["b_matrix_instance_id"] = 201 + ordinal
                cache["result_cache_key_fingerprint"] = 301 + ordinal
            observations.extend(records)
        cohorts = aggregate_captures(observations)
        self.assertEqual(len(cohorts), 2)
        self.assertTrue(all(evidence(view)["independent_runs"] == 2 for view in cohorts))

    def test_grouped_task_composition_and_execution_strategy_are_not_merged(self):
        from test_benchmark_schema_support_grouped import as_grouped_dispatch_capture

        base = as_grouped_dispatch_capture(bounded_capture("hip-direct", 100))
        device = deepcopy(base)
        device["grouped_dispatch"]["execution_strategy"] = "device_grouped_pack_gemm_host_exports"
        device["grouped_dispatch"]["task_descriptor_contract"]["device_descriptor_policy"] = (
            "device_pointer_tables_and_compact_slabs"
        )
        different_work = deepcopy(base)
        different_work["grouped_dispatch"]["task_descriptor_contract"]["buckets"] = [
            {"shape_key": "different ordered task composition", "task_count": 8}
        ]
        observations = aggregate_captures(runs(base) + runs(device) + runs(different_work))
        self.assertEqual(len(observations), 3)
        self.assertTrue(all(evidence(view)["independent_runs"] == 2 for view in observations))

    def test_hipblaslt_operand_policies_remain_separate_implementations(self):
        captures = []
        for policy in ("none", "a", "b", "both"):
            capture = bounded_capture("hipblaslt", 100)
            fields = (
                "a_retained_builds",
                "b_retained_builds",
                "a_retained_hits",
                "b_retained_hits",
                "a_retained_planes_packed",
                "b_retained_planes_packed",
                "a_transient_planes_packed",
                "b_transient_planes_packed",
            )
            capture["hipblaslt_operand_prepack"] = {
                "abi_version": 1,
                "policy": policy,
                "scope": "calling_thread_successful_pack_submissions_and_cache_matches_not_bytes_or_time",
                "after_warmups": dict.fromkeys(fields, 0),
                "after_measured_repeats": dict.fromkeys(fields, 0),
            }
            captures.extend(runs(capture))
        cohorts = aggregate_captures(captures)
        self.assertEqual(len(cohorts), 4)
        self.assertEqual(
            {c["hipblaslt_operand_prepack"]["policy"] for c in cohorts}, {"none", "a", "b", "both"}
        )
        self.assertTrue(all(evidence(c)["independent_runs"] == 2 for c in cohorts))

    def test_measurement_identity_binds_estimator_not_path_aliases(self):
        from observation_cohorts import measurement_identity

        a, b = runs(finite_capture("hip-direct", 100))
        original = aggregate_captures([a, b])[0]
        changed = deepcopy([a, b])
        for index, capture in enumerate(changed):
            capture["_path"] = f"renamed-{index}.json"
        renamed = aggregate_captures(list(reversed(changed)) + [deepcopy(changed[0])])[0]
        self.assertEqual(measurement_identity(original), measurement_identity(renamed))
        altered = deepcopy(original)
        altered["_observation_aggregate"]["phase_medians_us"]["end_to_end"] = 1
        self.assertNotEqual(measurement_identity(original), measurement_identity(altered))

    def test_in_memory_private_verdicts_cannot_supply_measurements(self):
        from report_comparison import ReportCohorts

        a, b = runs(finite_capture("hip-direct", 100))
        fake = dict(
            schema="rns8_process_observations_v1",
            qualified=True,
            blockers=[],
            independent_runs=99,
            phase_medians_us=dict(end_to_end=1),
            numerical_observations=[],
        )
        for capture in (a, b):
            capture["_observation_aggregate"] = deepcopy(fake)
            capture["_other_private_verdict"] = True
        cohorts = ReportCohorts([a, b])
        view = cohorts.observations[0]
        self.assertEqual(phase_median(view), 100)
        self.assertEqual(evidence(view)["independent_runs"], 2)
        self.assertTrue(
            all("_other_private_verdict" not in raw for raw in cohorts.raw_observations(view))
        )
        self.assertTrue(
            all("_observation_aggregate" not in raw for raw in cohorts.raw_observations(view))
        )
        singleton = ReportCohorts([a, deepcopy(a)]).observations[0]
        self.assertFalse(evidence(singleton)["qualified"])
        self.assertEqual(evidence(singleton)["independent_runs"], 1)
        self.assertEqual(a["_observation_aggregate"], fake)

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
            if dimension == "artifact":
                different["artifact_capture"]["executable_sha256"] = "f" * 64
            elif dimension == "device":
                different["device"]["name"] += " different device"
            elif dimension == "policy":
                different["exact_proof_policy"] = "scalar_first_v1"
            else:
                different["seed"] += 1
            cohorts = aggregate_captures([a, different])
            self.assertEqual(len(cohorts), 2)
            self.assertTrue(all(not evidence(c)["qualified"] for c in cohorts))
        legacy = deepcopy(a)
        legacy.pop("artifact_capture")
        legacy.pop("execution_id")
        self.assertIn(
            "independent_execution_provenance_unavailable",
            evidence(aggregate_captures([legacy])[0])["blockers"],
        )

    def test_order_invariance_and_conflicting_reused_id(self):
        a, b = runs(finite_capture("hip-direct", 100))
        b["_path"] = "second.json"
        a["_path"] = "first.json"
        first = aggregate_captures([a, b])
        second = aggregate_captures([b, a])
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
            capture["timing_summary_us"]["global_bound_scan"] = {
                "avg": setup,
                "median": setup,
                "p95": setup,
            }
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
            options = dict(
                expected_shapes=[captures[0]["m"]],
                expected_ring_moduli=[255],
                expected_field_moduli=[],
                expected_backends=["cpu-reference", "hip-direct", "ck"],
            )
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
        old = gpu["m"]
        gpu["m"] = old // 2

        def rewrite(value):
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, str):
                return value.replace(f"m={old};", f"m={gpu['m']};").replace(
                    f"--m {old} ", f"--m {gpu['m']} "
                )
            return value

        gpu = rewrite(gpu)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_all(Path(directory), runs(cpu) + runs(gpu))
            report = fhe_workload_report.build_report(paths)
            self.assertEqual(report["promotable_workload_profile_count"], 0)
            self.assertEqual(len(report["groups"]), 2)
            self.assertTrue(all(group["speedup_vs_cpu"] is None for group in report["groups"]))
            gpu_group = next(
                group for group in report["groups"] if "hip-direct" in group["backends"]
            )
            self.assertIn("cpu_reference_baseline_missing", gpu_group["blockers"])

    def test_cpu_timing_partitions_preserve_build_and_timer_policy(self):
        from observation_cohorts import best_implementation, row_identity, partition_rows

        cpu = finite_map_capture("cpu-reference", 2000)
        gpu = finite_map_capture("hip-direct", 900)
        for change in ("matched", "compiler", "configuration", "clock", "clock_policy", "host"):
            changed = deepcopy(cpu)
            if change == "compiler":
                changed["compiler"]["version"] = "different compiler version"
            elif change == "configuration":
                changed["build_manifest"]["configuration"] += "\nother_policy=ON"
            elif change == "clock":
                changed["timing_metadata"]["clock_period_denominator"] = 1000000
            elif change == "clock_policy":
                changed["clock_policy"] = {"mode": "other"}
            elif change == "host":
                changed["execution_host_identity"]["os_instance_sha256"] = "b" * 64
            observations = aggregate_captures(runs(changed) + runs(gpu))
            rows = [dict(row_identity(c), backend=c["backend_selected"]) for c in observations]
            partitions = partition_rows(rows)
            device = next(c for c in observations if c["backend_selected"] == "hip-direct")
            cpus = [c for c in observations if c["backend_selected"] == "cpu-reference"]
            selected = best_implementation(cpus, candidate=device, cpu_gpu_comparison=True)
            self.assertEqual(selected is not None, change == "matched", change)
            device_rows = next(
                members
                for _, members in partitions
                if any(r["backend"] == "hip-direct" for r in members)
            )
            self.assertEqual(
                any(r["backend"] == "cpu-reference" for r in device_rows),
                change == "matched",
                change,
            )
            with tempfile.TemporaryDirectory() as directory:
                paths = write_all(Path(directory), runs(changed) + runs(gpu))
                report = finite_modulus_map_report.build_report(
                    paths,
                    expected_shapes=[gpu["m"]],
                    expected_ring_moduli=[255],
                    expected_field_moduli=[],
                    expected_backends=["cpu-reference", "hip-direct"],
                )
                device_group = next(
                    group for group in report["groups"] if "hip-direct" in group["present_backends"]
                )
                self.assertEqual(
                    device_group["winner"]["speedup_vs_cpu"] is not None,
                    change == "matched",
                    change,
                )


class CounterObservations(unittest.TestCase):
    def test_uneven_partition_missing_metrics_and_kernel_identity(self):
        rows = [{"KernelName": "gemm", "VGPR": 64, "occupancy": 0.75} for _ in range(9)]
        rows.append({"KernelName": "export", "VGPR": 128, "occupancy": 0.25})
        rows[0].pop("occupancy")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def save(name, records):
                path = root / name
                path.write_text(json.dumps(records))
                return gpu_counter_report.load_counter_file(path, 20)

            whole = save("whole.json", rows)
            split = [save("first.json", rows[:9]), save("second.json", rows[9:])]
            self.assertEqual(
                gpu_counter_report._merged_counter_metrics([whole]),
                gpu_counter_report._merged_counter_metrics(split),
            )
            self.assertEqual(
                gpu_counter_report._merged_counter_metrics(split),
                gpu_counter_report._merged_counter_metrics(list(reversed(split))),
            )
            self.assertEqual(gpu_counter_report._merged_counter_metrics(split)["VGPR"], 70.4)
            self.assertEqual(whole["numeric_metric_counts"]["occupancy"], 9)
            self.assertEqual(
                gpu_counter_report._merged_counter_groups([whole]),
                gpu_counter_report._merged_counter_groups(split),
            )
            groups = gpu_counter_report._merged_counter_groups(split)
            self.assertEqual({g["identity"]["kernel"] for g in groups}, {"gemm", "export"})
            self.assertEqual(
                next(g for g in groups if g["identity"]["kernel"] == "gemm")[
                    "numeric_metric_average"
                ]["VGPR"],
                64,
            )
            units = save(
                "units.json",
                [
                    {"KernelName": "gemm", "occupancy": 0.5, "Unit": "fraction"},
                    {"KernelName": "gemm", "occupancy": 50, "Unit": "percent"},
                ],
            )
            metrics = gpu_counter_report._merged_counter_metrics([units])
            self.assertEqual(metrics["occupancy [fraction]"], 0.5)
            self.assertEqual(metrics["occupancy [percent]"], 50)


if __name__ == "__main__":
    unittest.main()
