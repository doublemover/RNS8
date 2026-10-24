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
PROGRAMS = ("integer-cross-product", "gf251-transform", "resident-chain", "complex-product", "exactness-walkthrough")


def fixture(workload="integer_cross_product"):
    if workload == "split_complex_integer_product": return complex_fixture()
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


def complex_fixture():
    report = fixture()
    row = RECIPES["fixtures"]["tiny_split_complex_v1"]
    operands = [list(row[key]) for key in ("ar", "ai", "br", "bi")]
    before = collector.canonical_words(sum(operands, []))
    operands[0][0] += 1
    changed = collector.canonical_words(sum(operands, []))
    expected = collector.checksum_bytes(collector.canonical_words(row["real"] + row["imag"], 3))
    changed_expected = collector.checksum_bytes(collector.canonical_words(row["changed_real"] + row["changed_imag"], 3))
    report.update(workload="split_complex_integer_product", semantics="exact_wide_signed_split_complex",
                  output_limbs_per_component_cell=3,
                  output_contract="real_row_major_signed_three_u64_limbs_then_imag_same", dataset_recipe="tiny_split_complex_v1",
                  input_checksum_u64=collector.checksum_bytes(before), changed_input_checksum_u64=collector.checksum_bytes(changed),
                  checksum_u64=expected, checksum_hex=f"{expected:016x}", changed_checksum_u64=changed_expected,
                  abs_partial_sum_bound="207", bounds={"component_absolute_partial_sum": "207", "selected_prefix": 20})
    prototype = report["variants"][0]
    for sample in [prototype["cold_call"], *prototype["samples"]]: sample["checksum_u64"] = expected
    prototype["changed_input"]["checksum_u64"] = changed_expected
    report["variants"] = []
    for method in ("3m", "4m"):
        variant = copy.deepcopy(prototype); variant["name"] = "rns8_complex_" + method
        variant["metadata"].update(public_api="rns8_gemm_complex_exact_signed", method=method,
                                    real_product_count=int(method[0]), scratch_budget_bytes=4096)
        variant["metadata"]["plan"]["selected_prefix"] = 20
        report["variants"].append(variant)
    baseline = copy.deepcopy(prototype); baseline["name"] = "blocked_cpp_int_complex_4m"; baseline["metadata"] = {}
    report["variants"].insert(0, baseline)
    return report


def gram_batch_fixture(products=3,policy="once"):
    report=fixture("integer_gram");batch=collector.gram_batch_contract(products,policy)
    report.update(gram_batch=batch,output_limbs_per_cell=1,output_contract="row_major_signed_i64",abs_partial_sum_bound="243")
    report.update(useful_integer_products_per_call=12*products,gram_dense_equivalent_integer_products_per_call=12*products,
        gram_independent_integer_products_per_call=9*products)
    before=collector.checksum_bytes(collector.canonical_words([14,-22,-22,161]));after=collector.checksum_bytes(collector.canonical_words([17,-14,-14,161]))
    report.update(checksum_u64=before,checksum_hex=f"{before:016x}",changed_checksum_u64=after)
    prototype=report["variants"][0];report["variants"]=[]
    for name,implementation in (("blocked_cpp_int","cpu_native_comparator"),("rns8_prepared_dense_gram_control","public_prepared_dense"),("rns8_single_source_gram_scalar","retained_single_source"),("blocked_native_i64","cpu_native_comparator")):
        row=copy.deepcopy(prototype);row["name"]=name;rns=name.startswith("rns8_");single=implementation=="retained_single_source"
        metadata=dict(batch,implementation=implementation,public_api="rns8_gram_residues" if single else "rns8_execute_prepared" if rns else None,
            rns_exports_per_batch=(products if policy=="each" else 1) if rns else 0,native_serializations_per_batch=0 if rns else products,
            prepared_gram_api=False,job_list_allocated_per_call=single,capability_probed_per_call=False)
        row["metadata"]["gram_batch"]=metadata
        if single:row["metadata"].update(independent_integer_products_per_call=9*products,dense_equivalent_integer_products_per_call=12*products)
        row.update(median_gram_product_ns=100/products,gram_products_per_second=1e9*products/100,call_rate_unit="complete_R_product_application_batches_per_second")
        for sample in [row["cold_call"],*row["samples"],row["changed_input"]]:
            sample.update(checksum_u64=before,gram_products=products,rns_exports=metadata["rns_exports_per_batch"],verified_outputs_after_interval=1)
        row["changed_input"].update(checksum_u64=after)
        if rns:row["changed_input"]["phases_ns"]["changed_input_pack_and_prepared_refresh_ns"]=10
        report["variants"].append(row)
    return report


class GramBatchReports(unittest.TestCase):
    def test_matched_batch_export_policies_and_counts(self):
        for products in (1,3,32,256):
            for policy in ("each","once"):
                report=gram_batch_fixture(products,policy)
                identity=collector.validate_report(report,"run1","cpu-reference",-1,collector.gram_batch_contract(products,policy))
                self.assertEqual(identity["gram_batch"]["products_per_batch"],products)
        for controls in (["--gram-reuse","3"],["--gram","--gram-export","once"],["--gram","--gram-reuse","0"],
            ["--gram","--gram-reuse","1.0"],["--gram","--gram-reuse","3","--gram-export","sometimes"],
            ["--gram","--gram-reuse","3","--gram-reuse","3"]):
            with self.assertRaises(ValueError):collector.requested_gram_batch(controls)
        self.assertEqual(collector.requested_gram_batch(["--gram","--gram-reuse","3"]),collector.gram_batch_contract(3,"once"))

    def test_malformed_preparation_materialization_and_oracle(self):
        mutations=[lambda r:r["gram_batch"].update(products_per_batch=True),lambda r:r["gram_batch"].update(export_policy="late"),
            lambda r:r.update(gram_batch={}),lambda r:r.update(checksum_u64=123,checksum_hex="000000000000007b"),
            lambda r:r["variants"][0]["metadata"]["gram_batch"].update(native_serializations_per_batch=1),
            lambda r:r["variants"][1]["metadata"]["gram_batch"].update(public_api="rns8_gemm_rns"),
            lambda r:r["variants"][2]["metadata"]["gram_batch"].update(prepared_gram_api=True),
            lambda r:r.update(useful_integer_products_per_call=12),
            lambda r:r["variants"][2]["metadata"].update(dense_equivalent_integer_products_per_call=12),
            lambda r:r["variants"][1]["samples"][0].update(gram_products=1),
            lambda r:r["variants"][1]["samples"][0]["phases_ns"].update(pack_and_upload_ns=10),
            lambda r:r["variants"][1]["changed_input"]["phases_ns"].pop("changed_input_pack_and_prepared_refresh_ns"),
            lambda r:r["variants"][1].update(median_gram_product_ns=1)]
        for mutate in mutations:
            report=gram_batch_fixture();mutate(report)
            with self.assertRaises(ValueError):collector.validate_report(report,"run1","cpu-reference",-1)
        report=gram_batch_fixture()
        with self.assertRaises(ValueError):collector.validate_report(report,"run1","cpu-reference",-1,collector.gram_batch_contract(3,"each"))

    def test_sustained_batch_products_and_final_only_verification(self):
        report=gram_batch_fixture();report["sustained_ms"]=1
        report["sustained_order"]=[row["name"] for row in report["variants"]]
        for row in report["variants"]:
            row["sustained"]={"calls":16,"elapsed_ns":2000000,"mean_call_ns":125000,"target_ms":1,"call_limit":4096,
                "target_reached":True,"checksum_u64":report["checksum_u64"],"final_exact_reference_passed":True,"verification_ns":2,
                "phase_timing_enabled":False,"scope":"fixture final output only","gram_products":48,
                "rns_exports":16*row["metadata"]["gram_batch"]["rns_exports_per_batch"],"mean_gram_product_ns":2000000/48,"verified_outputs_after_interval":1}
        collector.validate_report(report,"run1","cpu-reference",-1)
        report["variants"][0]["sustained"]["gram_products"]=16
        with self.assertRaises(ValueError):collector.validate_report(report,"run1","cpu-reference",-1)

    def test_requested_cohort_mismatch_retains_process(self):
        with tempfile.TemporaryDirectory() as directory:
            executable=Path(directory)/"fake.exe";executable.write_bytes(b"fixture")
            def process(command,**kwargs):
                report=gram_batch_fixture(3,"once");report["run_id"]=command[-1]
                return subprocess.CompletedProcess(command,0,json.dumps(report),"")
            with patch.object(collector.subprocess,"run",side_effect=process):
                record=collector.collect(executable,["--reference-only","--gram","--gram-reuse","3","--gram-export","each"],1,10)
            self.assertEqual(record["status"],"failed");self.assertIn("requested Gram batch",record["errors"][0]["reason"])


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
        for family in ("integer_cross_product", "integer_gram", "gf251_linear_transform", "resident_integer_chain", "split_complex_integer_product"):
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

    def test_complex_hand_answer_and_malformed_method_contracts(self):
        row = RECIPES["fixtures"]["tiny_split_complex_v1"]
        m, n, k = row["shape"]
        def product(a, b): return [sum(a[i*k+q] * b[q*n+j] for q in range(k)) for i in range(m) for j in range(n)]
        rr, ii = product(row["ar"], row["br"]), product(row["ai"], row["bi"])
        ri, ir = product(row["ar"], row["bi"]), product(row["ai"], row["br"])
        self.assertEqual([a-b for a,b in zip(rr, ii)], row["real"])
        self.assertEqual([a+b for a,b in zip(ri, ir)], row["imag"])
        mutations = [lambda r: r.update(output_contract="interleaved"), lambda r: r.update(semantics="finite_ring"),
                     lambda r: r.update(abs_partial_sum_bound="0"), lambda r: r["bounds"].update(selected_prefix=9),
                     lambda r: r.update(checksum_u64=0, checksum_hex="0000000000000000"),
                     lambda r: r.update(changed_checksum_u64=0), lambda r: r["variants"].pop(),
                     lambda r: r["variants"].pop(0), lambda r: r["variants"][1]["metadata"].update(method="4m"),
                     lambda r: r["variants"][1]["metadata"].update(real_product_count=4),
                     lambda r: r["variants"][1]["metadata"].update(scratch_budget_bytes=True),
                     lambda r: r["variants"][1]["metadata"].update(public_api="private_kernel")]
        for mutate in mutations:
            bad = complex_fixture(); mutate(bad)
            with self.assertRaises(ValueError): collector.validate_report(bad, "run1", "cpu-reference", -1)

    def test_complex_independent_process_scratch_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "fixture"; executable.write_bytes(b"immutable")
            count = 0
            def process(command, **kwargs):
                nonlocal count
                report = complex_fixture(); report["run_id"] = command[-1]
                report["variants"][1]["metadata"]["scratch_budget_bytes"] += count
                count += 1
                return subprocess.CompletedProcess(command, 0, json.dumps(report), "")
            with patch.object(collector.subprocess, "run", side_effect=process):
                record = collector.collect(executable, ["--reference-only"], 2, 1)
            self.assertEqual(record["status"], "failed")
            self.assertIn("identity changed", record["errors"][0]["reason"])

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
        cases = [("integer-cross-product", ["--gram"]), ("integer-cross-product", ["--gram", "--gram-reuse", "3", "--gram-export", "once"]),
                 ("integer-cross-product", ["--gram", "--gram-reuse", "3", "--gram-export", "each"]), ("integer-cross-product", ["--wide", "--bits", "40"]),
                 ("integer-cross-product", ["--seed", "0"]), ("integer-cross-product", ["--pattern", "zero"]),
                 ("gf251-transform", ["--seed", "1"]), ("gf251-transform", ["--pattern", "sparse"]),
                 ("resident-chain", ["--pattern", "zero"]), ("resident-chain", ["--bits", "12"]),
                 ("integer-cross-product", ["--m", "3", "--n", "1", "--k", "17"]),
                 ("gf251-transform", ["--m", "3", "--n", "5", "--k", "33"]),
                 ("resident-chain", ["--m", "3", "--n", "1", "--k", "17"]),
                 ("complex-product", ["--bits", "63"]), ("complex-product", ["--pattern", "zero"]),
                 ("complex-product", ["--pattern", "sparse", "--seed", "0"]),
                 ("complex-product", ["--m", "3", "--n", "5", "--k", "17"])]
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
        invalid = (["--gram-reuse", "3"], ["--gram-export", "once"], ["--gram", "--gram-reuse", "0"], ["--gram", "--gram-reuse", "257"], ["--m", "0"], ["--n", "513"], ["--k", "-1"], ["--repeats", "0"], ["--warmups", "21"], ["--sustained-ms", "10001"],
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
        for name in PROGRAMS[1:]:
            self.invoke(name,["--gram","--gram-reuse","3","--gram-export","once"],success=False)

    def test_complex_specific_controls_and_unsupported_backend(self):
        for flags in (["--wide"], ["--gram"], ["--size", "large", "--memory-mib", "1"]):
            self.invoke("complex-product", flags, success=False)
        result = subprocess.run([str(self.exe("complex-product")), "--backend", "ck", "--device", "0", "--json"],
                                text=True, capture_output=True, encoding="utf-8", timeout=10)
        self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "unsupported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--bin-dir", type=Path)
    args, rest = parser.parse_known_args(); BIN = args.bin_dir.resolve() if args.bin_dir else None
    unittest.main(argv=[sys.argv[0], *rest])
