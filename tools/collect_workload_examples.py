#!/usr/bin/env python3
"""Collect independent public-workload processes; retain raw diagnostic evidence.

This is an application-scope adapter, not a schema-4 kernel benchmark validator.
An optional real ISA file permits serialization through calibration_manifest;
its presence alone never marks ISA as verified or enables promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

SCHEMA = "rns8_workload_observation_v1"
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
IDENTITY_KEYS = ("workload", "m", "n", "k", "bits", "seed", "pattern", "repeats", "warmups",
                 "size", "sustained_ms", "backend_requested", "device_requested", "os", "compiler", "cpu_comparator_threads",
                 "build_manifest", "workload_source_identity", "semantics", "output_contract", "dataset_recipe",
                 "input_checksum_u64", "checksum_u64", "changed_input_checksum_u64", "changed_checksum_u64",
                 "abs_partial_sum_bound", "bounds", "formula", "lifecycle", "work_definition", "logical_layout", "a_ld", "b_ld", "output_ld",
                 "declared_memory_budget_bytes", "aggregate_memory_envelope_bytes", "gram_batch", "gram_prepared_owner_cohort")
MASK = (1 << 64) - 1


def canonical_words(values: list[int], limbs: int = 1) -> bytes:
    return b"".join((value % (1 << (64 * limbs))).to_bytes(8 * limbs, "little") for value in values)


def checksum_bytes(data: bytes) -> int:
    result = 14695981039346656037
    for byte in data:
        result = ((result ^ byte) * 1099511628211) & MASK
    return result


def dataset_identity(report: dict) -> dict:
    """Reconstruct recipe bytes, not merely hash a claimed checksum/seed."""
    recipe = report["dataset_recipe"]
    if report["workload"] == "split_complex_integer_product":
        return complex_dataset_identity(report)
    if recipe == "exactness_five_witnesses_v1":
        values = [1 << 40, -3, 1 << 30, 5, 1 << 40, -(1 << 40), 1 << 30, 1 << 30,
                  0, 0, -(1 << 63), (1 << 63) - 1, -(1 << 63), (1 << 63) - 1, 1, 1,
                  -(1 << 63), -(1 << 63), 8, 1 << 30]
        return {"sha256": hashlib.sha256(canonical_words(values)).hexdigest(), "serialization": "ordered witness native inputs as little-endian signed64"}
    m, n, k = (report[key] for key in ("m", "n", "k"))
    demand(all(type(value) is int and 1 <= value <= 512 for value in (m, n, k)), "recipe shape outside bounds")
    chain = report["workload"] == "resident_integer_chain"
    field = report["workload"] == "gf251_linear_transform"
    gram = report["workload"] == "integer_gram"
    valid = {"splitmix64_mod251_v1", "tiny_gf251_v1"} if field else {
        "splitmix64_signed_magnitude_v1", "tiny_cross_v1", "tiny_cross_chain_v1",
        "splitmix64_signed_magnitude_v1_gram_transpose_v1", "tiny_cross_v1_gram_transpose_v1"}
    demand(recipe in valid, "unknown versioned data recipe")
    state = report["seed"]; bits = report["bits"]
    demand(type(state) is int and 0 <= state <= MASK and type(bits) is int and 1 <= bits <= 63, "invalid generator seed/bits")
    maximum = (1 << bits) - 1
    operands = []
    for count in (m * k, k * n, n * n if chain else 0):
        operand = []
        for index in range(count):
            state = (state + 0x9e3779b97f4a7c15) & MASK
            word = state; word = ((word ^ (word >> 30)) * 0xbf58476d1ce4e5b9) & MASK
            word = ((word ^ (word >> 27)) * 0x94d049bb133111eb) & MASK; word ^= word >> 31
            value = word % (251 if field else maximum * 2 + 1)
            if not field and value > maximum: value = -(value - maximum)
            if report["pattern"] == "zero" or (report["pattern"] == "sparse" and index % 8): value = 0
            operand.append(value)
        operands.append(operand)
    if recipe.startswith("tiny_"):
        demand((m, n, k) == (2, 2, 3), "tiny fixture has wrong shape")
        operands[0] = [250, 2, 3, 4, 5, 6] if field else [2, -3, 4, -5, 6, 7]
        operands[1] = [7, 8, 9, 10, 11, 12] if field else [1, 8, -2, 9, 3, -4]
        if chain: operands[2] = [2, 1, -1, 2]
    if gram:
        demand(m == n, "Gram recipe shape mismatch")
        operands[0] = [operands[1][q * n + row] for row in range(m) for q in range(k)]
    if gram and report.get("gram_batch"):
        original_gram_max=max(map(abs,operands[1]))
        changed_first=operands[1][0]+(-1 if operands[1][0]==(1 << 63)-1 else 1)
        maximum=max(max(map(abs,operands[1])),abs(changed_first))
        demand(report.get("abs_partial_sum_bound")==str(k*maximum*maximum),"Gram full-expression bound differs from replayed inputs")
    def check_gram_output(key):
        if not gram or not report.get("gram_batch"): return
        limbs=report.get("output_limbs_per_cell")
        wide=report.get("semantics")=="exact_wide_signed"
        demand(type(limbs) is int and limbs==(3 if wide else 1) and report.get("semantics") in {"exact_wide_signed","bounded_i64"}
            and report.get("output_contract")==("row_major_signed_three_u64_limbs_per_cell" if wide else "row_major_signed_i64"),"Gram batch native output semantics differ")
        values=[sum(operands[0][row*k+q]*operands[1][q*n+col] for q in range(k)) for row in range(m) for col in range(n)]
        demand(wide or all(-(1 << 63)<=value<(1 << 63) for value in values),"Gram bounded oracle exceeds native output")
        demand(report.get(key)==checksum_bytes(canonical_words(values,limbs)),"Gram complete output differs from independently replayed integer reference")
    check_gram_output("checksum_u64")
    data = canonical_words([value for operand in operands for value in operand])
    demand(checksum_bytes(data) == report.get("input_checksum_u64"), "input bytes differ from claimed recipe/checksum")
    operands[0][0] = (operands[0][0] + 1) % 251 if field else operands[0][0] + (-1 if operands[0][0] == (1 << 63) - 1 else 1)
    if gram: operands[1][0] = operands[0][0]
    check_gram_output("changed_checksum_u64")
    changed = canonical_words([value for operand in operands for value in operand])
    demand(checksum_bytes(changed) == report.get("changed_input_checksum_u64"), "changed input bytes differ from generation2")
    identity={"sha256": hashlib.sha256(data).hexdigest(), "changed_sha256": hashlib.sha256(changed).hexdigest(),
            "bytes": len(data), "serialization": "A then B then optionalT, row-major little-endian signed64"}
    if gram and report.get("gram_prepared_owner_cohort") and report.get("gram_batch"):
        identity["gram_max_magnitudes"]={"original":str(original_gram_max),"changed":str(max(map(abs,operands[1])))}
    return identity


def complex_dataset_identity(report: dict) -> dict:
    m, n, k = (report[key] for key in ("m", "n", "k"))
    demand(all(type(value) is int and 1 <= value <= 512 for value in (m, n, k)), "complex recipe shape outside bounds")
    recipe = report["dataset_recipe"]
    demand(recipe in {"tiny_split_complex_v1", "splitmix64_split_complex_signed_v1"}, "unknown split-complex recipe")
    state, bits = report["seed"], report["bits"]
    demand(type(state) is int and 0 <= state <= MASK and type(bits) is int and 1 <= bits <= 63, "invalid complex seed/bits")
    demand(report["pattern"] in {"seeded", "zero", "sparse"}, "invalid complex pattern")
    maximum = (1 << bits) - 1
    operands = []
    for count in (m * k, m * k, k * n, k * n):
        operand = []
        for index in range(count):
            state = (state + 0x9e3779b97f4a7c15) & MASK
            word = state; word = ((word ^ (word >> 30)) * 0xbf58476d1ce4e5b9) & MASK
            word = ((word ^ (word >> 27)) * 0x94d049bb133111eb) & MASK; word ^= word >> 31
            value = word % (maximum * 2 + 1)
            if value > maximum: value = -(value - maximum)
            if report["pattern"] == "zero" or report["pattern"] == "sparse" and index % 8: value = 0
            operand.append(value)
        operands.append(operand)
    if recipe == "tiny_split_complex_v1":
        demand((m, n, k) == (2, 2, 3) and report["pattern"] == "seeded" and report["seed"] == 3135
               and report["bits"] == 8 and report["size"] == "tiny", "tiny complex fixture controls differ")
        operands = [[2, -3, 4, -5, 6, 7], [1, 0, -1, 2, -2, 1], [1, 8, -2, 9, 3, -4], [0, 1, 2, 0, -1, 3]]
    before = canonical_words([value for operand in operands for value in operand])
    demand(checksum_bytes(before) == report.get("input_checksum_u64"), "complex input bytes differ from recipe")
    maxima = [max(map(abs, operand)) for operand in operands]
    first = operands[0][0]
    operands[0][0] += -1 if first == (1 << 63) - 1 else 1
    maxima[0] = max(maxima[0], abs(operands[0][0]))
    bound = k * max(maxima[0] * maxima[2] + maxima[1] * maxima[3], maxima[0] * maxima[3] + maxima[1] * maxima[2])
    demand(report.get("abs_partial_sum_bound") == str(bound), "complex complete-expression bound differs")
    demand(report.get("bounds", {}).get("component_absolute_partial_sum") == str(bound)
           and report["bounds"].get("selected_prefix") == 20, "complex admitted basis/bound metadata differs")
    changed = canonical_words([value for operand in operands for value in operand])
    demand(checksum_bytes(changed) == report.get("changed_input_checksum_u64"), "complex generation2 input differs")
    if recipe == "tiny_split_complex_v1":
        for key, values in (("checksum_u64", [19, -25, 9, -19, -12, 26, 14, 10]),
                            ("changed_checksum_u64", [20, -17, 9, -19, -12, 27, 14, 10])):
            demand(report.get(key) == checksum_bytes(canonical_words(values, 3)), "tiny complex full-pair checksum differs")
    return {"sha256": hashlib.sha256(before).hexdigest(), "changed_sha256": hashlib.sha256(changed).hexdigest(),
            "bytes": len(before), "serialization": "Ar,Ai,Br,Bi row-major little-endian signed64"}


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def demand(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_sample(sample: dict, expected: int) -> None:
    demand(type(sample) is dict, "sample must be an object")
    demand(sample.get("exact_reference_passed") is True, "sample exact comparison missing")
    demand(sample.get("checksum_u64") == expected, "sample checksum differs from fully checked reference")
    demand(positive(sample.get("application_ns")), "invalid application time")
    demand(type(sample.get("verification_ns")) in (int, float) and math.isfinite(sample["verification_ns"])
           and sample["verification_ns"] >= 0, "invalid verification time")
    demand(type(sample.get("phases_ns")) is dict and bool(sample["phases_ns"]), "missing phase scopes")
    demand(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
               for value in sample["phases_ns"].values()), "invalid phase time")



def gram_batch_contract(products: int, policy: str) -> dict:
    return {"version":1,"products_per_batch":products,"export_policy":policy,"reference_outputs_per_batch":1,
            "input_preparation":"pack_once_per_generation",
            "scope":"R identical products recomputed synchronously; complete final native host output"}


def exact_value(actual, expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return actual.keys()==expected.keys() and all(exact_value(actual[key],value) for key,value in expected.items())
    return actual==expected


def requested_gram_batch(arguments: list[str]) -> dict:
    controls={}
    for key in ("--gram-reuse","--gram-export"):
        if key in arguments:
            demand(arguments.count(key)==1,"duplicate Gram batch option")
            index=arguments.index(key);demand(index+1<len(arguments),"missing Gram batch value");controls[key]=arguments[index+1]
    if not controls:return {}
    demand("--gram" in arguments and "--gram-reuse" in controls,"Gram batch controls require --gram and --gram-reuse")
    text=controls["--gram-reuse"]
    demand(bool(re.fullmatch(r"[0-9]+",text)),"invalid Gram reuse integer")
    products=int(text);policy=controls.get("--gram-export","once")
    demand(1<=products<=256 and policy in {"each","once"},"invalid Gram batch controls")
    return gram_batch_contract(products,policy)


def validate_gram_batch(report: dict, variants: list[dict], requested=None,dataset=None):
    batch=report.get("gram_batch",{})
    prepared_cohort="gram_prepared_owner_cohort" in report
    if prepared_cohort:demand(exact_value(report["gram_prepared_owner_cohort"],1) and bool(batch),"invalid prepared Gram cohort version")
    if "gram_batch" in report:demand(type(batch) is dict and bool(batch),"Gram batch must be nonempty contract object")
    if requested is not None:demand(exact_value(batch,requested),"requested Gram batch differs from capture")
    if not batch:return
    demand(report["workload"]=="integer_gram" and report["m"]==report["n"],"Gram batch requires single-source Gram workload")
    products=batch.get("products_per_batch");policy=batch.get("export_policy")
    demand(type(products) is int and 1<=products<=256 and type(policy) is str and policy in {"each","once"},"invalid Gram batch contract")
    demand(exact_value(batch,gram_batch_contract(products,policy)),"invalid Gram preparation/materialization scope")
    demand(report["m"]*report["n"]*report["k"]*products<=134217728,"Gram batch work limit exceeded")
    dense_products=report["m"]*report["n"]*report["k"]*products
    independent_products=report["n"]*(report["n"]+1)//2*report["k"]*products
    demand(exact_value(report.get("useful_integer_products_per_call"),dense_products)
        and exact_value(report.get("gram_dense_equivalent_integer_products_per_call"),dense_products)
        and exact_value(report.get("gram_independent_integer_products_per_call"),independent_products),"Gram report mathematical work mismatch")
    passed=[row for row in variants if row.get("status")=="passed"]
    names={row["name"] for row in passed}
    demand({"rns8_prepared_dense_gram_control","rns8_single_source_gram_scalar","blocked_cpp_int"}<=names,
           "Gram batch requires prepared dense, single-source scalar and full exact blocked controls")
    if int(report["abs_partial_sum_bound"])<=(1 << 63)-1:demand("blocked_native_i64" in names,"admitted native64 blocked comparator missing")
    if prepared_cohort:
        demand("rns8_prepared_single_source_gram_scalar" in names,"prepared scalar Gram comparator missing")
        demand(("rns8_single_source_gram_wmma" in names)==("rns8_prepared_single_source_gram_wmma" in names),"ordinary/prepared WMMA controls must match")
        magnitudes=dataset["gram_max_magnitudes"]
        expanded=int(magnitudes["changed"])>int(magnitudes["original"])
    for row in passed:
        name=row["name"];rns=name.startswith("rns8_")
        allowed={"rns8_prepared_dense_gram_control","rns8_single_source_gram_scalar","rns8_single_source_gram_wmma"}
        if prepared_cohort:allowed|={"rns8_prepared_single_source_gram_scalar","rns8_prepared_single_source_gram_wmma"}
        demand(not rns or name in allowed,"unexpected Gram implementation")
        single=rns and name!="rns8_prepared_dense_gram_control";wmma=name.endswith("_wmma");prepared=name.startswith("rns8_prepared_single_source_gram_")
        expected=dict(batch,implementation="public_prepared_single_source" if prepared else "retained_single_source" if single else "public_prepared_dense" if rns else "cpu_native_comparator",
            public_api="rns8_execute_prepared_gram" if prepared else "rns8_gram_residues" if single else "rns8_execute_prepared" if rns else None,
            rns_exports_per_batch=(products if policy=="each" else 1) if rns else 0,
            native_serializations_per_batch=0 if rns else products,prepared_gram_api=prepared,
            job_list_allocated_per_call=single and not prepared,capability_probed_per_call=wmma and not prepared)
        if wmma:
            demand(report["backend_requested"]=="hip-direct" and row.get("metadata",{}).get("device",{}).get("target","").split(":")[0]
                in {"gfx1100","gfx1101","gfx1102"},"Gram WMMA requires explicit admitted direct-HIP target")
        if prepared:
            hip=report["backend_requested"]=="hip-direct"
            expected.update(generation_policy="refresh_or_explicit_recreate_on_expanded_bound",
                host_job_preparation="once_per_owner_during_setup_or_changed_input",
                device_job_allocation="once_per_owner_during_setup_or_changed_input" if hip else "not_applicable",
                device_job_initialization="first_execution_per_owner" if hip else "not_applicable",
                wmma_capability_probe="first_execution_per_owner" if wmma else "not_applicable")
        demand(exact_value(row.get("metadata",{}).get("gram_batch"),expected),"Gram variant preparation/export identity mismatch")
        if single:
            demand(exact_value(row["metadata"].get("independent_integer_products_per_call"),independent_products)
                and exact_value(row["metadata"].get("dense_equivalent_integer_products_per_call"),dense_products),"Gram variant mathematical work mismatch")
        if prepared:
            tiles=(report["n"]+15)//16
            demand(exact_value(row["metadata"].get("modeled_retained_gram_job_payload_bytes"),tiles*(tiles+1)//2*16+4),"prepared Gram job payload model differs")
            unchanged=dict(generation_action="retained_owner",packed_source_version=1,owner_first_execution=False,owner_builds=1,owner_refreshes=0)
            cold=dict(unchanged,owner_first_execution=True)
            changed=dict(generation_action="recreated_owner_for_larger_bound" if expanded else "refreshed_owner",packed_source_version=2,
                owner_first_execution=expanded,owner_builds=2 if expanded else 1,owner_refreshes=0 if expanded else 1)
            demand(exact_value(row["cold_call"].get("execution_evidence"),cold),"prepared Gram first-use evidence missing")
            demand(all(exact_value(sample.get("execution_evidence"),unchanged) for sample in row["samples"]),"unchanged prepared Gram owner was rebuilt or initialized")
            demand(exact_value(row["changed_input"].get("execution_evidence"),changed),"prepared Gram generation refresh/recreation differs from replayed envelope")
            if report["sustained_ms"]:demand(exact_value(row["sustained"].get("execution_evidence"),unchanged),"sustained prepared Gram lifecycle differs")
        demand(positive(row.get("median_gram_product_ns")) and math.isclose(row["median_gram_product_ns"]*products,row["median_steady_ns"],rel_tol=1e-12)
            and positive(row.get("gram_products_per_second")) and math.isclose(row["gram_products_per_second"]*row["median_steady_ns"],1e9*products,rel_tol=1e-12)
            and row.get("call_rate_unit")=="complete_R_product_application_batches_per_second","Gram batch rate unit mismatch")
        for sample in [row["cold_call"],*row["samples"],row["changed_input"]]:
            demand(exact_value(sample.get("gram_products"),products)
                and exact_value(sample.get("rns_exports"),expected["rns_exports_per_batch"])
                and exact_value(sample.get("verified_outputs_after_interval"),1),"Gram sample work/verification count mismatch")
        if rns:
            for sample in row["samples"]:
                demand(not {"pack_and_upload_ns","changed_input_pack_and_prepared_refresh_ns"}&sample["phases_ns"].keys(),"unchanged Gram samples must reuse packed input")
            demand("changed_input_pack_and_prepared_refresh_ns" in row["changed_input"]["phases_ns"],"changed Gram generation must repack/refresh")
        if report["sustained_ms"]:
            burst=row["sustained"];count=burst["calls"]
            demand(exact_value(burst.get("gram_products"),count*products)
                and exact_value(burst.get("rns_exports"),count*expected["rns_exports_per_batch"])
                and exact_value(burst.get("verified_outputs_after_interval"),1)
                and positive(burst.get("mean_gram_product_ns"))
                and math.isclose(burst["mean_gram_product_ns"]*count*products,burst["elapsed_ns"],rel_tol=1e-12),"Gram sustained work mismatch")


def validate_report(report: dict, run_id: str, backend: str, device: int, gram_request=None) -> dict:
    demand(type(report) is dict and report.get("schema") == SCHEMA, "unexpected workload schema")
    demand(report.get("status") == "passed", "workload did not pass")
    demand(report.get("workload") in {"integer_cross_product", "integer_gram", "gf251_linear_transform",
                                      "resident_integer_chain", "split_complex_integer_product", "exactness_walkthrough"}, "unknown workload family")
    demand(report.get("run_id") == run_id, "run identity mismatch")
    demand(report.get("backend_requested") == backend and report.get("device_requested") == device,
           "requested backend/device mismatch")
    demand(report.get("promotion_eligible") is False and report.get("isa_verified") is False,
           "raw application observation cannot self-qualify")
    for key in ("m", "n", "k", "repeats"):
        demand(type(report.get(key)) is int and report[key] > 0, f"invalid {key}")
    demand(type(report.get("warmups")) is int and report["warmups"] >= 0, "invalid warmups")
    sustained_ms = report.get("sustained_ms")
    demand(type(sustained_ms) is int and 0 <= sustained_ms <= 10000, "invalid sustained duration")
    source = report.get("workload_source_identity")
    demand(type(source) is dict and bool(source), "compiled example source identity missing")
    demand(all(type(key) is str and type(value) is str and DIGEST.fullmatch(value)
               for key, value in source.items()), "invalid compiled source hashes")
    demand("recipes.json" in source, "compiled recipe identity missing")
    demand(type(report.get("build_manifest")) is dict, "compiled library manifest missing")
    demand(type(report.get("checksum_u64")) is int and 0 <= report["checksum_u64"] < 1 << 64,
           "all-output checksum missing")
    demand(report.get("checksum_hex") == f'{report["checksum_u64"]:016x}', "checksum encoding mismatch")
    clock = report.get("clock", {})
    demand(clock.get("source") == "host_monotonic" and positive(clock.get("resolution_ns"))
           and positive(clock.get("censor_threshold_ns")), "invalid host clock calibration")
    identity = {key: report.get(key) for key in IDENTITY_KEYS}
    identity["dataset_bytes"] = dataset_identity(report)
    if report.get("workload") == "exactness_walkthrough":
        demand(sustained_ms == 0, "fixed exactness witnesses cannot supply sustained timing")
        rows = report.get("witnesses", [])
        demand(len(rows) == 5 and all(row.get("exact_reference_passed") is True for row in rows),
               "walkthrough witnesses missing")
        expected = [(1 << 70) - 15, 0, 0, -1, 1 << 129]
        demand([row.get("exact_decimal") for row in rows] == [str(value) for value in expected], "wrong exactness witness")
        demand(checksum_bytes(canonical_words(expected, 3)) == report["checksum_u64"], "walkthrough aggregate checksum differs")
        for row, value in zip(rows, expected):
            demand(row.get("checksum_hex") == f'{checksum_bytes(canonical_words([value], 3)):016x}', "witness checksum differs")
            actual = row.get("candidate", {}).get("device", row.get("device", {}))
            demand(actual.get("backend_selected") == backend and actual.get("device_id") == device,
                   "walkthrough effective backend/device differs")
        demand(rows[0].get("checked_i64", {}).get("status") == "insufficient_range", "missing overflow negative control")
        demand(rows[0].get("binary64", {}).get("integer_difference_from_exact") == "15", "missing binary64 negative control")
        demand(rows[-1].get("previous_output_preserved") is True and rows[-1].get("out_of_range_status") == "range error",
               "missing range rollback check")
        identity["witnesses"] = [{key: row.get(key) for key in ("name", "exact_decimal", "checksum_hex", "selected_prefix")}
                                 for row in rows]
        # These fixed witnesses deliberately provide no repeated benchmark cohort.
        return identity
    variants = report.get("variants")
    demand(type(variants) is list and bool(variants), "missing variants")
    demand(len({row.get("name") for row in variants}) == len(variants), "duplicate variant identity")
    admitted = []
    candidates = []
    for row in variants:
        demand(type(row.get("name")) is str and row["name"], "empty variant identity")
        if row.get("status") != "passed":
            demand(row.get("status") in {"unavailable", "unsupported", "not_run", "insufficient_range"} and bool(row.get("reason")),
                   "invalid omitted comparator status")
            admitted.append({"name": row["name"], "status": row["status"], "reason": row["reason"]})
            continue
        demand(positive(row.get("median_steady_ns")) and positive(row.get("calls_per_second")), "invalid steady summary")
        samples = row.get("samples", [])
        demand(len(samples) == report["repeats"], "repeat count mismatch")
        exact_sample(row["cold_call"], report["checksum_u64"])
        for sample in samples:
            exact_sample(sample, report["checksum_u64"])
        exact_sample(row["changed_input"], report["changed_checksum_u64"])
        if sustained_ms:
            burst = row.get("sustained", {})
            count = burst.get("calls")
            elapsed = burst.get("elapsed_ns")
            demand(type(count) is int and 1 <= count <= 4096 and positive(elapsed), "invalid sustained call count/time")
            demand(burst.get("target_ms") == sustained_ms and burst.get("call_limit") == 4096,
                   "sustained controls differ from requested bounds")
            demand(positive(burst.get("mean_call_ns")) and math.isclose(burst["mean_call_ns"], elapsed / count, rel_tol=1e-12),
                   "sustained mean differs from elapsed/count")
            reached = elapsed >= sustained_ms * 1e6
            demand(burst.get("target_reached") is reached and (reached or count == 4096), "sustained burst stopped early")
            demand(burst.get("final_exact_reference_passed") is True
                   and burst.get("checksum_u64") == report["checksum_u64"], "sustained final exact comparison missing")
            demand(burst.get("phase_timing_enabled") is False and bool(burst.get("scope")), "sustained timing scope missing")
            demand(type(burst.get("verification_ns")) in (int, float) and math.isfinite(burst["verification_ns"])
                   and burst["verification_ns"] >= 0, "invalid sustained verification time")
        else:
            demand("sustained" not in row, "unexpected unrequested sustained measurement")
        demand(row["changed_input"].get("source_version") == 2, "changed-input generation missing")
        demand(positive(row.get("cold_application_total_ns")) and bool(row.get("cold_total_scope")), "cold lifecycle scope missing")
        metadata = row.get("metadata", {})
        if row["name"].startswith("rns8_"):
            candidates.append(row)
            actual = metadata.get("device", {})
            demand(actual.get("backend_selected") == backend and actual.get("backend_requested") == backend
                   and actual.get("device_id") == device, "effective backend/device differs from request")
        admitted.append({"name": row["name"], "status": row["status"], "metadata": metadata,
                         "phase_scopes": sorted(samples[0]["phases_ns"]), "cold_total_scope": row["cold_total_scope"]})
    demand(bool(candidates), "requested RNS8 implementation missing")
    if report["workload"] == "split_complex_integer_product":
        demand(report.get("semantics") == "exact_wide_signed_split_complex"
               and report.get("output_contract") == "real_row_major_signed_three_u64_limbs_then_imag_same",
               "complex component semantics/serialization differs")
        demand(report.get("output_limbs_per_component_cell") == 3, "complex component export width differs")
        demand({row["name"] for row in candidates} == {"rns8_complex_3m", "rns8_complex_4m"}, "both public complex methods are required")
        demand(any(row["name"] == "blocked_cpp_int_complex_4m" and row.get("status") == "passed" for row in variants),
               "strongest implemented exact complex CPU comparator missing")
        for row in candidates:
            metadata = row["metadata"]
            method = "3m" if row["name"] == "rns8_complex_3m" else "4m"
            demand(metadata.get("public_api") == "rns8_gemm_complex_exact_signed" and metadata.get("method") == method
                   and metadata.get("real_product_count") == int(method[0]), "complex method identity differs")
            demand(metadata.get("plan", {}).get("selected_prefix") == 20, "complex fixed basis differs")
            demand(type(metadata.get("scratch_budget_bytes")) is int and metadata["scratch_budget_bytes"] > 0,
                   "complex transient scratch budget missing")
    expected_order = [row["name"] for row in variants if row.get("status") == "passed"] if sustained_ms else []
    order = report.get("sustained_order")
    demand(type(order) is list and len(order) == len(expected_order)
           and sorted(order) == sorted(expected_order), "sustained order must cover each measured variant exactly once")
    validate_gram_batch(report,variants,gram_request,identity["dataset_bytes"])
    identity["variants"] = admitted
    return identity


def request_identity(arguments: list[str]) -> tuple[str, int]:
    demand(not any(arg == "--json" or arg == "--run-id" or arg.startswith("--run-id=") for arg in arguments),
           "collector owns --json and --run-id")
    backend = "hip-direct"
    device = None
    for index, arg in enumerate(arguments):
        if arg in {"--backend", "--device"}:
            demand(index + 1 < len(arguments), "missing backend/device value")
            if arg == "--backend": backend = arguments[index + 1]
            else: device = int(arguments[index + 1])
    if "--reference-only" in arguments:
        demand(backend in {"hip-direct", "cpu-reference"} and ("--backend" not in arguments or backend == "cpu-reference"),
               "reference-only conflicts with requested backend")
        backend = "cpu-reference"
    demand(backend in {"hip-direct", "cpu-reference", "hip-vector-alu-int64", "hipblaslt", "ck", "rocwmma", "amdgpu-builtins"},
           "explicit supported backend required")
    return backend, (-1 if backend == "cpu-reference" else 0) if device is None else device


def collect(executable: Path, arguments: list[str], runs: int, timeout: float) -> dict:
    demand(1 <= runs <= 100 and positive(timeout), "invalid process count/timeout")
    executable = executable.resolve(strict=True)
    backend, device = request_identity(arguments)
    gram_request=requested_gram_batch(arguments)
    digest = sha(executable)
    tools = {Path(__file__).name: sha(Path(__file__))}
    record = {"schema": "rns8_workload_collection_v1", "executable": str(executable), "executable_sha256": digest,
              "collector_source_sha256": tools, "requested_backend": backend, "requested_device": device,
              "arguments": arguments, "runs": [], "errors": [], "promotion_eligible": False,
              "qualification_note": "application-scope host observations; no schema-4 benchmark pass, ISA verification or runtime route promotion"}
    record["host"] = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                      "logical_threads": os.cpu_count(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                      "OMP_DYNAMIC": os.environ.get("OMP_DYNAMIC")}
    baseline = None
    for _ in range(runs):
        run_id = str(uuid.uuid4())
        command = [str(executable), *arguments, "--json", "--run-id", run_id]
        row = {"run_id": run_id, "command": command, "started_unix_ns": time.time_ns()}
        try:
            demand(sha(executable) == digest, "executable changed before independent process")
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=timeout, check=False)
            row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            demand(sha(executable) == digest, "executable changed during independent process")
            demand(result.returncode == 0, f"process returned {result.returncode}; unsupported is retained without fallback")
            report = json.loads(result.stdout)
            identity = validate_report(report, run_id, backend, device, gram_request)
            row["observation"] = report
            if baseline is None: baseline = identity
            demand(identity == baseline, "workload/provenance/comparison identity changed across independent processes")
            row["comparison_identity_sha256"] = hashlib.sha256(encoded(identity)).hexdigest()
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            row["error"] = str(error)
            if isinstance(error, subprocess.TimeoutExpired):
                row["stdout"] = (error.stdout or b"").decode("utf-8", errors="replace") if isinstance(error.stdout, bytes) else error.stdout
                row["stderr"] = (error.stderr or b"").decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else error.stderr
            record["errors"].append({"run_id": run_id, "reason": str(error)})
        record["runs"].append(row)
    record["status"] = "failed" if record["errors"] else "passed"
    record["comparison_identity"] = baseline
    return record


def calibration_adapter(record: dict, isa_file: Path, variant: str, instruction: str,
                        clocks_policy: str, power_policy: str) -> dict:
    import calibration_manifest as cm
    demand(record["status"] == "passed", "failed collection cannot adapt")
    demand(all(value.strip() for value in (variant, instruction, clocks_policy, power_policy)), "explicit adapter identity required")
    demand(record["comparison_identity"]["workload"] != "exactness_walkthrough", "walkthrough is not a repeated timing workload")
    rows = [next((item for item in run["observation"]["variants"] if item["name"] == variant), None) for run in record["runs"]]
    demand(all(row and row["status"] == "passed" for row in rows), "requested adapter variant did not pass")
    demand(variant.startswith("rns8_"), "ISA adapter binds only the requested RNS8 implementation")
    device = rows[0]["metadata"]["device"]
    identity = record["comparison_identity"]
    configuration = {"application_identity": identity, "variant": variant, "scope": "steady synchronous complete application call"}
    isa_digest = sha(isa_file)
    bound = cm.Identity(target=device["target"], device=device["device"], runtime=json.dumps(device, sort_keys=True),
        compiler=identity["compiler"], artifact_sha256=record["executable_sha256"], isa_sha256=isa_digest,
        configuration_sha256=hashlib.sha256(encoded(configuration)).hexdigest(), clocks_policy=clocks_policy,
        power_policy=power_policy, instruction=instruction, kernel_symbol=json.dumps(rows[0]["metadata"], sort_keys=True),
        dataset_sha256=identity["dataset_bytes"]["sha256"], cache_state="hot", output_contract=identity["output_contract"],
        timing_source="host_monotonic", profiling_mode="none")
    raw_runs = []
    for process, row in zip(record["runs"], rows):
        report = process["observation"]; threshold = report["clock"]["censor_threshold_ns"]
        elapsed = tuple(sample["application_ns"] if sample["application_ns"] >= threshold else None for sample in row["samples"])
        raw_runs.append(cm.Run(process["run_id"], elapsed, threshold, 1, report["warmups"], True, False))
    result = cm.summarize(cm.Calibration(bound, identity["workload"] + "/" + variant + "/steady_application", "calls", tuple(raw_runs)))
    result["adapter_scope"] = configuration["scope"]
    result["isa_note"] = "actual file hash retained; kernel selection/instruction correspondence has not been independently verified"
    result["adapter_sources_sha256"] = {name: sha(Path(__file__).with_name(name)) for name in
        ("calibration_manifest.py", "analytical_planner.py", "arithmetic_certificate.py")}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--isa-file", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--instruction")
    parser.add_argument("--clocks-policy")
    parser.add_argument("--power-policy")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        if not args.isa_file:
            demand(not any((args.variant, args.instruction, args.clocks_policy, args.power_policy)), "adapter controls require --isa-file")
        result = collect(args.exe, arguments, args.runs, args.timeout)
        if args.isa_file:
            try:
                result["calibration"] = calibration_adapter(result, args.isa_file, args.variant or "", args.instruction or "",
                                                            args.clocks_policy or "", args.power_policy or "")
            except (OSError, ValueError, KeyError, TypeError) as error:
                result["calibration"] = None
                result["errors"].append({"run_id": "adapter", "reason": str(error)})
                result["status"] = "failed"
        else:
            result["calibration"] = None
            result["calibration_unavailable_reason"] = "no real ISA artifact supplied; no placeholder digest or fabricated verified kernel identity"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f'{result["status"]}: {len(result["runs"])} independent processes, {len(result["errors"])} errors; {args.output}')
        return 0 if result["status"] == "passed" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
