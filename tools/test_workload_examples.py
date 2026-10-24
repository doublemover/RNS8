#!/usr/bin/env python3
"""Workload recipes, malformed observation fixtures and optional compiled CLI tests."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import collect_workload_examples as collector

ROOT = Path(__file__).resolve().parents[1]
RECIPES = json.loads((ROOT / "examples/workloads/recipes.json").read_text(encoding="utf-8"))
BIN = None
PROGRAMS = ("integer-cross-product", "gf251-transform", "resident-chain", "exactness-walkthrough")


def fixture(workload="integer_cross_product"):
    field = workload == "gf251_linear_transform"; chain = workload == "resident_integer_chain"; gram = workload == "integer_gram"
    recipe = "tiny_gf251_v1" if field else "tiny_cross_chain_v1" if chain else "tiny_cross_v1_gram_transpose_v1" if gram else "tiny_cross_v1"
    values = RECIPES["fixtures"][recipe]
    a, b, t = list(values["a"]), list(values["b"]), list(values.get("t", []))
    input_hash = collector.checksum_bytes(collector.canonical_words(a + b + t))
    a[0] = (a[0] + 1) % 251 if field else a[0] + 1
    if gram: b[0] = a[0]
    changed_input = collector.checksum_bytes(collector.canonical_words(a + b + t))
    result = {"schema": collector.SCHEMA, "status": "passed", "run_id": "run1", "workload": workload,
              "m": 2, "n": 2, "k": 3, "bits": 8, "seed": 3135, "pattern": "seeded", "size": "tiny",
              "repeats": 2, "warmups": 0, "sustained_ms": 0, "sustained_order": [], "backend_requested": "cpu-reference", "device_requested": -1,
              "compiler": "fixture compiler", "os": "fixture", "cpu_comparator_threads": 1,
              "build_manifest": {"source": "fixture"}, "workload_source_identity": {"recipes.json": "a" * 64},
              "semantics": "finite_u8_field" if field else "exact_wide_signed" if chain else "bounded_i64",
              "output_contract": "fixture_contract", "dataset_recipe": recipe, "input_checksum_u64": input_hash,
              "changed_input_checksum_u64": changed_input, "checksum_u64": 123, "checksum_hex": "000000000000007b",
              "changed_checksum_u64": 456, "clock": {"source": "host_monotonic", "resolution_ns": 1, "censor_threshold_ns": 5},
              "promotion_eligible": False, "isa_verified": False}
    def sample(hash_value):
        return {"application_ns": 100, "verification_ns": 2, "phases_ns": {"application_ns": 90},
                "checksum_u64": hash_value, "exact_reference_passed": True}
    changed = sample(456); changed["source_version"] = 2
    result["variants"] = [{"name": "rns8_cpu-reference", "status": "passed", "median_steady_ns": 100,
        "calls_per_second": 10000000, "samples": [sample(123), sample(123)], "cold_call": sample(123),
        "changed_input": changed, "cold_application_total_ns": 200, "cold_total_scope": "explicit_scope",
        "metadata": {"device": {"backend_requested": "cpu-reference", "backend_selected": "cpu-reference",
            "device_id": -1, "device": "fixtureCPU", "target": "host"}, "plan": {"selected_kernel": "fixture"}}}]
    return result


class Reports(unittest.TestCase):
    def test_sustained_final_verification_and_bounded_duration(self):
        report = fixture()
        report.update(sustained_ms=250, sustained_order=["rns8_cpu-reference"])
        burst = {"calls": 100, "elapsed_ns": 250000000, "mean_call_ns": 2500000,
                 "target_ms": 250, "call_limit": 4096, "target_reached": True,
                 "checksum_u64": 123, "final_exact_reference_passed": True,
                 "verification_ns": 10, "phase_timing_enabled": False, "scope": "full application"}
        report["variants"][0]["sustained"] = burst
        collector.validate_report(report, "run1", "cpu-reference", -1)
        for key, value in (("calls", True), ("calls", 4097), ("mean_call_ns", 1),
                           ("elapsed_ns", float("nan")), ("target_reached", False),
                           ("target_ms", 1), ("call_limit", 1), ("checksum_u64", 456),
                           ("final_exact_reference_passed", False), ("phase_timing_enabled", True),
                           ("verification_ns", -1), ("scope", "")):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(report); bad["variants"][0]["sustained"][key] = value
                with self.assertRaises(ValueError): collector.validate_report(bad, "run1", "cpu-reference", -1)
        burst.update(calls=4096, elapsed_ns=4096000, mean_call_ns=1000, target_reached=False)
        collector.validate_report(report, "run1", "cpu-reference", -1)
        bad = copy.deepcopy(report); bad["sustained_order"] = []
        with self.assertRaises(ValueError): collector.validate_report(bad, "run1", "cpu-reference", -1)
        burst.update(calls=4095, elapsed_ns=4095000)
        with self.assertRaises(ValueError): collector.validate_report(report, "run1", "cpu-reference", -1)

    def test_exactness_witness_validation(self):
        report = fixture(); report.update(workload="exactness_walkthrough", dataset_recipe="exactness_five_witnesses_v1")
        values = [int(value) for value in RECIPES["fixtures"]["exactness_five_witnesses_v1"]["output_decimal"]]
        report["checksum_u64"] = collector.checksum_bytes(collector.canonical_words(values, 3))
        report["checksum_hex"] = f'{report["checksum_u64"]:016x}'
        report["witnesses"] = [{"name": str(index), "exact_decimal": str(value), "exact_reference_passed": True,
            "checksum_hex": f'{collector.checksum_bytes(collector.canonical_words([value], 3)):016x}',
            "selected_prefix": 20, "device": {"backend_selected": "cpu-reference", "device_id": -1}}
            for index, value in enumerate(values)]
        report["witnesses"][0].update(checked_i64={"status": "insufficient_range"}, binary64={"integer_difference_from_exact": "15"})
        report["witnesses"][-1].update(previous_output_preserved=True, out_of_range_status="range error")
        collector.validate_report(report, "run1", "cpu-reference", -1)
        for index, key, value in ((0, "exact_decimal", "1"), (4, "previous_output_preserved", False),
                                  (2, "checksum_hex", "0000000000000000"), (4, "out_of_range_status", "success")):
            with self.subTest(index=index, key=key):
                bad = copy.deepcopy(report); bad["witnesses"][index][key] = value
                with self.assertRaises(ValueError): collector.validate_report(bad, "run1", "cpu-reference", -1)

    def test_all_recipe_families(self):
        for family in ("integer_cross_product", "integer_gram", "gf251_linear_transform", "resident_integer_chain"):
            with self.subTest(family=family):
                report = fixture(family)
                identity = collector.validate_report(report, "run1", "cpu-reference", -1)
                self.assertRegex(identity["dataset_bytes"]["sha256"], "^[0-9a-f]{64}$")

    def test_hand_checked_fixtures(self):
        for name, row in RECIPES["fixtures"].items():
            if "a" not in row: continue
            m, n, k = row["shape"]
            product = [sum(row["a"][i*k+q] * row["b"][q*n+j] for q in range(k)) for i in range(m) for j in range(n)]
            if "t" in row: product = [sum(product[i*n+q] * row["t"][q*n+j] for q in range(n)) for i in range(m) for j in range(n)]
            if "gf251" in name: product = [value % 251 for value in product]
            self.assertEqual(product, row["output"], name)

    def test_malformed_observations(self):
        mutations = {
            "schema": lambda r: r.update(schema="rns8_benchmark_v4"),
            "failed": lambda r: r.update(status="failed"),
            "run_id": lambda r: r.update(run_id="different"),
            "backend": lambda r: r.update(backend_requested="auto"),
            "device": lambda r: r.update(device_requested=0),
            "promotion": lambda r: r.update(promotion_eligible=True),
            "isa": lambda r: r.update(isa_verified=True),
            "dimensions_bool": lambda r: r.update(m=True),
            "unknown_family": lambda r: r.update(workload="invented"),
            "source_missing": lambda r: r.update(workload_source_identity=None),
            "source_bad_hash": lambda r: r.update(workload_source_identity={"recipes.json": "bad"}),
            "recipe_missing": lambda r: r.update(dataset_recipe="invented"),
            "recipe_shape": lambda r: r.update(m=3),
            "input_checksum": lambda r: r.update(input_checksum_u64=0),
            "changed_input_checksum": lambda r: r.update(changed_input_checksum_u64=0),
            "manifest_missing": lambda r: r.update(build_manifest=None),
            "checksum_hex": lambda r: r.update(checksum_hex="bad"),
            "clock_nan": lambda r: r["clock"].update(resolution_ns=float("nan")),
            "empty_variants": lambda r: r.update(variants=[]),
            "duplicate_variant": lambda r: r["variants"].append(copy.deepcopy(r["variants"][0])),
            "effective_backend": lambda r: r["variants"][0]["metadata"]["device"].update(backend_selected="hip-direct"),
            "effective_device": lambda r: r["variants"][0]["metadata"]["device"].update(device_id=0),
            "repeats": lambda r: r.update(repeats=3),
            "exact_failure": lambda r: r["variants"][0]["samples"][0].update(exact_reference_passed=False),
            "checksum_failure": lambda r: r["variants"][0]["samples"][0].update(checksum_u64=0),
            "time_zero": lambda r: r["variants"][0]["samples"][0].update(application_ns=0),
            "time_inf": lambda r: r["variants"][0]["samples"][0].update(application_ns=float("inf")),
            "time_null": lambda r: r["variants"][0]["samples"][0].update(application_ns=None),
            "phase_missing": lambda r: r["variants"][0]["samples"][0].update(phases_ns={}),
            "phase_negative": lambda r: r["variants"][0]["samples"][0].update(phases_ns={"bad": -1}),
            "verification_negative": lambda r: r["variants"][0]["samples"][0].update(verification_ns=-1),
            "changed_generation": lambda r: r["variants"][0]["changed_input"].update(source_version=1),
            "cold_scope": lambda r: r["variants"][0].update(cold_total_scope=""),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                report = fixture(); mutate(report)
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    collector.validate_report(report, "run1", "cpu-reference", -1)

    def test_process_identity_and_failures_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / "executable"; executable.write_bytes(b"immutable")
            counter = 0
            def process(command, **kwargs):
                nonlocal counter
                report = fixture(); report["run_id"] = command[-1]
                if counter: report["compiler"] = "changed compiler"
                counter += 1
                return subprocess.CompletedProcess(command, 0, json.dumps(report), "diagnostic stderr")
            with patch.object(collector.subprocess, "run", side_effect=process):
                record = collector.collect(executable, ["--reference-only"], 2, 1)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(len(record["runs"]), 2)
            self.assertIn("identity changed", record["errors"][0]["reason"])
            self.assertEqual(record["runs"][1]["stderr"], "diagnostic stderr")
            for result in (subprocess.CompletedProcess([], 77, '{"status":"unsupported"}', "missingGPU"),
                           subprocess.CompletedProcess([], 0, "not JSON", ""),
                           subprocess.TimeoutExpired("executable", 1, output=b"partial")):
                with patch.object(collector.subprocess, "run", side_effect=result if isinstance(result, Exception) else None,
                                  return_value=result):
                    record = collector.collect(executable, ["--reference-only"], 1, 1)
                self.assertEqual(record["status"], "failed")
                self.assertTrue(record["runs"][0]["error"])

    def test_mutating_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / "executable"; executable.write_bytes(b"before")
            def process(command, **kwargs):
                executable.write_bytes(b"after")
                return subprocess.CompletedProcess(command, 0, "{}", "")
            with patch.object(collector.subprocess, "run", side_effect=process):
                record = collector.collect(executable, ["--reference-only"], 1, 1)
            self.assertIn("executable changed", record["errors"][0]["reason"])

    def test_adapter_preserves_admission_gaps_and_censoring(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "actual-isa.txt"; path.write_text("actual captured bytes", encoding="utf-8")
            report = fixture(); report["variants"][0]["samples"][0]["application_ns"] = 2
            identity = collector.validate_report(report, "run1", "cpu-reference", -1)
            record = {"status": "passed", "executable_sha256": "b" * 64, "comparison_identity": identity,
                      "runs": [{"run_id": "run1", "observation": report}]}
            result = collector.calibration_adapter(record, path, "rns8_cpu-reference", "explicit_label", "unlocked", "default")
            self.assertIn("isa_not_verified", result["admission_failures"])
            self.assertIn("censored_samples", result["admission_failures"])
            self.assertIn("fewer_than_five_independent_runs", result["admission_failures"])
            self.assertFalse(result["rate_export_eligible"])
            self.assertFalse(result["promotion_eligible"])
            self.assertIsNone(result["raw"]["runs"][0]["elapsed_ns"][0])

    def test_reserved_and_invalid_collection_controls(self):
        for arguments in (["--json"], ["--run-id", "x"], ["--backend", "auto"], ["--reference-only", "--backend", "ck"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError): collector.request_identity(arguments)


class Compiled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if BIN is None: raise unittest.SkipTest("compiled workload directory not supplied")
        missing = [name for name in PROGRAMS if not cls.exe(name).is_file()]
        if missing: raise RuntimeError("missing compiled workloads: " + ", ".join(missing))

    @staticmethod
    def exe(name): return BIN / ("rns8-workload-" + name + (".exe" if os.name == "nt" else ""))

    def invoke(self, name, arguments, success=True):
        result = subprocess.run([str(self.exe(name)), "--reference-only", "--json", *arguments], text=True,
                                capture_output=True, encoding="utf-8", timeout=45)
        self.assertEqual(result.returncode, 0 if success else 1, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "passed" if success else "failed")
        return report

    def test_all_tiny_modes_collect_independent_processes(self):
        for name in PROGRAMS:
            with self.subTest(program=name):
                result = collector.collect(self.exe(name), ["--reference-only", "--size", "tiny", "--repeats", "1", "--warmups", "0"], 2, 45)
                self.assertEqual(result["status"], "passed", result["errors"])
                self.assertEqual(len({row["run_id"] for row in result["runs"]}), 2)

    def test_gram_wide_patterns_and_generator_override(self):
        cases = [("integer-cross-product", ["--gram"]), ("integer-cross-product", ["--wide", "--bits", "40"]),
                 ("integer-cross-product", ["--seed", "0"]), ("integer-cross-product", ["--pattern", "zero"]),
                 ("gf251-transform", ["--seed", "1"]), ("gf251-transform", ["--pattern", "sparse"]),
                 ("resident-chain", ["--pattern", "zero"]), ("resident-chain", ["--bits", "12"]),
                 ("integer-cross-product", ["--m", "3", "--n", "1", "--k", "17"]),
                 ("gf251-transform", ["--m", "3", "--n", "5", "--k", "33"]),
                 ("resident-chain", ["--m", "3", "--n", "1", "--k", "17"])]
        for name, flags in cases:
            with self.subTest(program=name, flags=flags):
                result = collector.collect(self.exe(name), ["--reference-only", "--size", "tiny", "--repeats", "2", "--warmups", "0", *flags], 1, 45)
                self.assertEqual(result["status"], "passed", result["errors"])

    def test_sustained_application_modes(self):
        for name in PROGRAMS[:-1]:
            with self.subTest(program=name):
                result = collector.collect(self.exe(name), ["--reference-only", "--size", "tiny", "--repeats", "1",
                    "--warmups", "0", "--sustained-ms", "1"], 1, 45)
                self.assertEqual(result["status"], "passed", result["errors"])

    def test_invalid_cli_is_rejected_before_gpu_work(self):
        invalid = (["--m", "0"], ["--n", "513"], ["--k", "-1"], ["--repeats", "0"], ["--warmups", "21"], ["--sustained-ms", "10001"],
                   ["--seed", "18446744073709551616"], ["--bits", "64"], ["--memory-mib", "0"],
                   ["--m", "2junk"], ["--repeats", "1", "--repeats", "1"], ["--unknown", "1"],
                   ["--backend", "auto"], ["--backend", "hip-direct"], ["--device", "0"], ["--m"])
        for name in PROGRAMS:
            for flags in invalid:
                with self.subTest(program=name, flags=flags): self.invoke(name, flags, success=False)
        for name, flags in (("gf251-transform", ["--wide"]), ("resident-chain", ["--gram"]),
                            ("exactness-walkthrough", ["--bits", "8"]), ("exactness-walkthrough", ["--repeats", "2"]),
                            ("exactness-walkthrough", ["--sustained-ms", "1"]),
                            ("integer-cross-product", ["--size", "large", "--memory-mib", "1"]),
                            ("integer-cross-product", ["--size", "tiny", "--bits", "63"]),
                            ("resident-chain", ["--size", "tiny", "--bits", "63"])):
            with self.subTest(program=name, flags=flags): self.invoke(name, flags, success=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--bin-dir", type=Path)
    args, rest = parser.parse_known_args(); BIN = args.bin_dir.resolve() if args.bin_dir else None
    unittest.main(argv=[sys.argv[0], *rest])
