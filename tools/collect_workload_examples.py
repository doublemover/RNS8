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
                 "declared_memory_budget_bytes", "aggregate_memory_envelope_bytes", "gram_batch", "gram_prepared_owner_cohort", "gram_ablation_cohort",
                 "optional_reference_cohort", "optional_references")
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
        if not (gram and report.get("gram_batch")) and not report.get("optional_reference_cohort"): return
        limbs=report.get("output_limbs_per_cell")
        wide=report.get("semantics")=="exact_wide_signed"
        demand(type(limbs) is int and limbs==(3 if wide else 1) and report.get("semantics") in {"exact_wide_signed","bounded_i64"}
            and report.get("output_contract")==("row_major_signed_three_u64_limbs_per_cell" if wide else "row_major_signed_i64"),"Gram batch native output semantics differ")
        values=[sum(operands[0][row*k+q]*operands[1][q*n+col] for q in range(k)) for row in range(m) for col in range(n)]
        if chain:
            values=[sum(values[row*n+q]*operands[2][q*n+col] for q in range(n)) for row in range(m) for col in range(n)]
        demand(wide or all(-(1 << 63)<=value<(1 << 63) for value in values),"Gram bounded oracle exceeds native output")
        demand(report.get(key)==checksum_bytes(canonical_words(values,limbs)),"complete output differs from independently replayed integer reference")
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
    ablation_cohort="gram_ablation_cohort" in report
    if ablation_cohort:demand(exact_value(report["gram_ablation_cohort"],1) and prepared_cohort,"invalid Gram ablation cohort version")
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
    ablation_methods=["scalar"]+(["wmma"] if "rns8_single_source_gram_wmma" in names else [])
    ablation_names={f"rns8_prepared_single_source_gram_{kind}_{method}" for method in ablation_methods for kind in ("native","transpose","reconstruct_once","separate_mirror")}
    if ablation_cohort:demand(ablation_names<=names,"Gram native/transpose/reconstruct-once controls must all be present")
    for row in passed:
        name=row["name"];rns=name.startswith("rns8_")
        allowed={"rns8_prepared_dense_gram_control","rns8_single_source_gram_scalar","rns8_single_source_gram_wmma"}
        if prepared_cohort:allowed|={"rns8_prepared_single_source_gram_scalar","rns8_prepared_single_source_gram_wmma"}
        if ablation_cohort:allowed|=ablation_names
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
            if ablation_cohort:
                native="_gram_native_" in name;transpose="_gram_transpose_" in name;once="_reconstruct_once_" in name
                contract=row["metadata"].get("gram_ablations",{})
                prefix=row["metadata"]["plan"]["selected_prefix"]
                order=report["n"];limbs=report["output_limbs_per_cell"]
                adapter=report["k"]*order*prefix if transpose else 0
                host_export=order*order*limbs*8 if once else 0
                device_export=host_export+4 if hip and once else 0
                expected_contract=dict(version=1,input_policy=1 if native else 2 if transpose else 0,
                    reconstruction_policy="independent_once_then_mirror" if once else "dense_each_cell",
                    owned_payload_bytes=contract.get("owned_payload_bytes"),adapter_bytes=adapter,
                    host_export_bytes=host_export,device_export_bytes=device_export,
                    independent_cells=order*(order+1)//2,dense_cells=order*order,native_storage_retained=native,
                    adapter_generation_policy="first_execute_after_prepare_or_refresh",mirror_policy="separate_tiled" if "_separate_mirror_" in name else "producer_stores",output_validity="complete_dense_residues")
                demand(exact_value(contract,expected_contract),"Gram ablation identity or payload mismatch")
                jobs=tiles*(tiles+1)//2*8
                device=jobs+(4 if wmma else 0)+adapter if hip else 0
                if hip and once:device=(device+7)//8*8+device_export
                minimum=jobs+(adapter if not hip else 0)+host_export+(prefix if once and not hip else 0)+device
                demand(type(contract["owned_payload_bytes"]) is int and minimum<=contract["owned_payload_bytes"]<=report["declared_memory_budget_bytes"],
                       "Gram owned payload omits retained adapter/export/alignment")
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


def optional_reference_contract(library: str, chain: bool, limbs: int, reuse: int = 1) -> dict:
    return dict(version=1,library=library,arithmetic="blocked_mpz_addmul_tile32" if library=="gmp" else "fmpz_mat_mul_library_dispatch",
        application_boundary="native_signed64_host_inputs_to_complete_signed_host_words",input_conversion_per_call=True,
        retained_result_reuse=False,retained_allocations=True,intermediate_serialization=False,chain=chain,
        real_gemm_products_per_call=2 if chain else reuse,output_limbs_per_cell=limbs,threads=1,
        output_conversion="checked_twos_complement_u64_limbs",library_private_peak_bytes=None,actual_peak_measured=False)


def validate_optional_references(report: dict, variants: list[dict]) -> None:
    if "optional_reference_cohort" not in report:
        demand(not any(row["name"].startswith(("gmp_","flint_")) for row in variants),"optional comparator missing cohort contract")
        return
    demand(exact_value(report["optional_reference_cohort"],1),"invalid optional reference cohort")
    demand(report["workload"] in {"integer_cross_product","integer_gram","resident_integer_chain"},"optional exact adapter on unsupported workload")
    libraries=report.get("optional_references")
    demand(type(libraries) is dict and set(libraries)=={"version","gmp","flint"} and exact_value(libraries["version"],1),"invalid optional library identity")
    chain=report["workload"]=="resident_integer_chain";limbs=report.get("output_limbs_per_cell")
    demand({"workload_optional_references.hpp","optional_references.cmake"}<=report["workload_source_identity"].keys(),"optional adapter source/configuration identity missing")
    demand(type(limbs) is int and limbs in {1,3},"invalid optional comparator output width")
    demand(exact_value(report.get("cpu_comparator_threads"),1),"optional exact comparators require serial thread identity")
    def artifacts(value,library):
        headers={"gmp.h"} if library=="gmp" else {"flint/flint.h","flint/fmpz.h","flint/fmpz_mat.h"}
        demand(type(value) is dict and set(value)=={"headers_sha256","link_input_sha256","extra_link_inputs_sha256","runtime_binary_sha256","runtime_binary_verified"},"invalid optional artifact fields")
        demand(type(value["headers_sha256"]) is dict and set(value["headers_sha256"])==headers and
            all(type(digest) is str and DIGEST.fullmatch(digest) for digest in value["headers_sha256"].values()) and
            type(value["link_input_sha256"]) is str and DIGEST.fullmatch(value["link_input_sha256"]),"optional configured artifact hash missing")
        demand(value["runtime_binary_sha256"] is None and value["runtime_binary_verified"] is False,"configured link input is not loaded-runtime attestation")
        demand(type(value["extra_link_inputs_sha256"]) is list and all(type(digest) is str and DIGEST.fullmatch(digest)
            for digest in value["extra_link_inputs_sha256"]),"invalid explicit static link dependency hashes")
        demand(library=="flint" or not value["extra_link_inputs_sha256"],"unexpected GMP extra link dependencies")
    def version(header,runtime):
        demand(type(header) is str and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+",header) and header==runtime,"optional header/runtime version mismatch")
    names={row["name"]:row for row in variants}
    expected_names=set()
    reserved=0
    for library in ("gmp","flint"):
        item=libraries[library]
        demand(type(item) is dict and type(item.get("enabled")) is bool,"invalid optional enablement")
        name=("gmp_mpz_blocked" if library=="gmp" else "flint_fmpz_mat_mul")+("_chain" if chain else "")
        expected_names.add(name);demand(name in names,"optional enabled/unavailable row missing")
        row=names[name]
        if not item["enabled"]:
            demand(set(item)=={"enabled","status","reason"} and item["status"]=="unavailable" and
                type(item["reason"]) is str and bool(item["reason"]) and row.get("status")=="unavailable" and
                row.get("reason")==item["reason"],"disabled optional adapter must remain unavailable")
            continue
        fields={"enabled","status","header_version","runtime_version","artifacts"}
        if library=="flint":fields|={"gmp_header_version","gmp_runtime_version","gmp_artifacts"}
        demand(set(item)==fields and item["status"]=="available" and row.get("status")=="passed","enabled optional adapter did not execute")
        version(item["header_version"],item["runtime_version"]);artifacts(item["artifacts"],library)
        if library=="flint":
            version(item["gmp_header_version"],item["gmp_runtime_version"]);artifacts(item["gmp_artifacts"],"gmp")
            if libraries["gmp"]["enabled"]:
                demand(item["gmp_runtime_version"]==libraries["gmp"]["runtime_version"] and
                    exact_value(item["gmp_artifacts"],libraries["gmp"]["artifacts"]),"FLINT/GMP support dependency identity differs")
        reuse=report.get("gram_batch",{}).get("products_per_batch",1)
        expected=optional_reference_contract(library,chain,limbs,reuse)
        expected["identity"]=item
        m,n,k=(report[key] for key in ("m","n","k"))
        cells=m*k+k*n+m*n+(n*n+m*n if chain else 0)
        expected["modeled_memory_allowance_bytes"]=1024*cells+m*n*3*8+4096
        reserved+=expected["modeled_memory_allowance_bytes"]
        demand(exact_value(row.get("metadata",{}).get("optional_exact_reference"),expected),"optional comparator boundary/resource identity differs")
        for sample in [row["cold_call"],*row["samples"],row["changed_input"]]:
            demand(set(sample["phases_ns"])=={"optional_input_conversion_ns","optional_exact_gemm_ns","optional_output_serialization_ns"},"optional host-to-host conversion/arithmetic/serialization phases missing")
    demand({name for name in names if name.startswith(("gmp_","flint_"))}==expected_names,"unexpected optional comparator identity")
    demand(type(report.get("aggregate_memory_envelope_bytes")) is int and
        type(report.get("declared_memory_budget_bytes")) is int and
        reserved<=report["aggregate_memory_envelope_bytes"]<=report["declared_memory_budget_bytes"],"optional aggregate memory admission missing")


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
    validate_optional_references(report,variants)
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


def collect(executable: Path, arguments: list[str], runs: int, timeout: float, *, case_file=None, case_identity=None) -> dict:
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
    demand((case_file is None)==(case_identity is None),"case identity/file must be supplied together")
    if case_identity is not None:record["case_manifest"]=case_identity
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
            if case_file is not None:demand(sha(case_file)==case_identity["sha256"],"case manifest changed before independent process")
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=timeout, check=False)
            row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            demand(sha(executable) == digest, "executable changed during independent process")
            if case_file is not None:demand(sha(case_file)==case_identity["sha256"],"case manifest changed during independent process")
            demand(result.returncode == 0, f"process returned {result.returncode}; unsupported is retained without fallback")
            report = json.loads(result.stdout)
            if case_identity is not None:demand(exact_value(report.get("optional_reference_cohort"),1),"selected optional case requires explicit enabled/unavailable adapter cohort")
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


def load_optional_case(path: Path, case_id: str, executable: Path) -> tuple[list[str], dict]:
    raw=path.read_bytes()
    def object_pairs(pairs):
        result={}
        for key,value in pairs:
            demand(key not in result,"duplicate case manifest key");result[key]=value
        return result
    manifest=json.loads(raw,object_pairs_hook=object_pairs)
    demand(type(manifest) is dict and set(manifest)=={"schema","cases"} and manifest["schema"]=="rns8_optional_reference_cases_v1","invalid optional case manifest")
    cases=manifest["cases"];demand(type(cases) is list and 1<=len(cases)<=256,"invalid optional case count")
    names=set();selected=None
    for case in cases:
        demand(type(case) is dict and set(case)=={"id","program","arguments"},"invalid optional case fields")
        name=case["id"];demand(type(name) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",name) and name not in names,"invalid or duplicate optional case id")
        names.add(name)
        demand(type(case["program"]) is str and case["program"] in {"integer-cross-product","resident-chain"},"unsupported optional workload program")
        arguments=case["arguments"]
        demand(type(arguments) is list and all(type(arg) is str and arg and len(arg)<=128 for arg in arguments),"invalid optional workload arguments")
        demand("--reference-only" in arguments and request_identity(arguments)==("cpu-reference",-1),"optional cases use explicit CPU host control")
        # The executable owns full shape/range admission; reject ambiguous or
        # collector-owned CLI controls before constructing any process command.
        flags={"--reference-only","--wide","--gram"};values={"--size","--m","--n","--k","--bits","--seed","--pattern","--repeats","--warmups","--sustained-ms","--memory-mib","--baseline-max-products","--gram-reuse","--gram-export"}
        seen=set();index=0
        while index<len(arguments):
            key=arguments[index];demand(key in flags|values and key not in seen,"unknown or duplicate optional case option");seen.add(key);index+=1
            if key in values:
                demand(index<len(arguments) and not arguments[index].startswith("--"),"missing optional case value");index+=1
        requested_gram_batch(arguments)
        demand("--sustained-ms" in seen,"optional cases require explicit sustained scope")
        if name==case_id:selected=case
    demand(selected is not None,"optional case id not found")
    demand(executable.stem=="rns8-workload-"+selected["program"],"case program differs from selected executable")
    return selected["arguments"],dict(schema=manifest["schema"],case=selected,sha256=hashlib.sha256(raw).hexdigest())


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
    parser.add_argument("--case-file",type=Path,help="frozen optional exact-reference workload case manifest")
    parser.add_argument("--case",help="explicit case id from --case-file; replaces trailing workload arguments")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        demand(bool(args.case_file)==bool(args.case),"--case-file and --case must be supplied together")
        case_identity=None
        if args.case_file:
            demand(not arguments,"case manifest conflicts with trailing workload arguments")
            arguments,case_identity=load_optional_case(args.case_file,args.case,args.exe)
        if not args.isa_file:
            demand(not any((args.variant, args.instruction, args.clocks_policy, args.power_policy)), "adapter controls require --isa-file")
        result = collect(args.exe, arguments, args.runs, args.timeout,case_file=args.case_file,case_identity=case_identity)
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
