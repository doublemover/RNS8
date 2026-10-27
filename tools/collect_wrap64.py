#!/usr/bin/env python3
"""Collect strict wrap64 private controls with paired production/event timing."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import uuid
from build_manifest import manifest_provenance_complete
from collect_finite_verify import (BUDGET, MASK, checksum, demand, file_digest, integer, positive, same, strict_json, validate_provenance)
RDNA_CANDIDATES = ("signed_radix", "native_u8", "direct_v4", "rocwmma_corrected")
CDNA_CANDIDATES = ("signed_radix_mfma", "direct_v4", "native_u64")
CANDIDATES = (*RDNA_CANDIDATES, "signed_radix_mfma", "native_u64")
SCOPES = ("resident", "repeated_b", "roundtrip", "cold")
KEYS = {"device", "candidate", "scope", "m", "n", "k", "pattern", "reuse_count"}
EDGES = (0, 1, MASK, 0x8080808080808080, 0x7f807f807f807f80, 0xfefdfcfbfaf9f8f7, 0x0101010101010101, 1 << 63)
RECIPE = "splitmix64_u64_extrema_seed3135_v1"


def validate_case(case):
    demand(type(case) is dict and set(case) == KEYS, "case requires exactly documented controls")
    demand(integer(case["device"], 0, 63) and case["candidate"] in CANDIDATES and case["scope"] in SCOPES, "invalid explicit device/candidate/scope")
    demand(all(integer(case[key], 1, 131071 if key == "k" else 512) for key in ("m", "n", "k"))
           and case["m"]*case["n"]*case["k"] <= 134217728, "shape/reference work outside bounds")
    demand(case["candidate"] != "rocwmma_corrected" or case["k"] <= 32768, "existing rocWMMA control requires K<=32768")
    demand(case["pattern"] in ("seeded", "carry", "zero") and integer(case["reuse_count"], 1, 256), "invalid pattern/reuse")
    return dict(case)


def load_cases(path):
    doc = strict_json(Path(path).read_text(encoding="utf-8-sig"))
    demand(type(doc) is dict and set(doc) == {"schema", "cases"} and doc["schema"] == "rns8_wrap64_cases_v1", "invalid case manifest")
    demand(type(doc["cases"]) is list and 1 <= len(doc["cases"]) <= 64, "case count must be 1..64")
    cases = [validate_case(case) for case in doc["cases"]]
    demand(len(cases) == len({json.dumps(case, sort_keys=True) for case in cases}), "duplicate cases do not create independent observations")
    return cases


def dataset(case):
    state = 3135; result = []
    for count in (case["m"]*case["k"], case["k"]*case["n"]):
        values = bytearray()
        for index in range(count):
            state = (state + 0x9e3779b97f4a7c15) & MASK; word = state
            word = ((word ^ (word >> 30))*0xbf58476d1ce4e5b9) & MASK
            word = ((word ^ (word >> 27))*0x94d049bb133111eb) & MASK; word ^= word >> 31
            if case["pattern"] == "zero":
                word = 0
            elif case["pattern"] == "carry":
                word = EDGES[index % 8]
            elif index % 257 < 8:
                word = EDGES[index % 257]
            values.extend(word.to_bytes(8, "little"))
        result.append(values)
    return result


def contract(case, run_id, repeats, warmups, target=None):
    candidate = case["candidate"]; radix = candidate in ("signed_radix", "signed_radix_mfma", "native_u8")
    mfma_k = {"gfx90a":16,"gfx942":32,"gfx950":64}.get((target or "").split(":")[0]) if candidate=="signed_radix_mfma" else None
    if candidate=="signed_radix_mfma":demand(mfma_k is not None,"MFMA target must name compiled generation")
    selection = ("direct_hip_wrap64_byte_gemm36_" + ("u32acc" if case["k"] <= 4096 else "u64acc") + "_tiled_2d_v4") if candidate == "direct_v4" else {
        "native_u64":"wrap64_native_u64_vector", "signed_radix": "wrap64_signed_radix_wmma", "signed_radix_mfma":"wrap64_signed_radix_mfma", "native_u8": "wrap64_native_u8_wmma", "rocwmma_corrected": "rocwmma_wrap64_byte_gemm36_candidate_kernel"}[candidate]
    return {"schema": "rns8_wrap64_microcase_v1", "promotion_eligible": False, "public_ownership_measured": False, "run_id": run_id,
            **{key: case[key] for key in ("candidate", "scope", "m", "n", "k", "pattern", "reuse_count")},
            "device_requested": case["device"], "device_actual": case["device"], "context_backend": "hip-direct",
            "execution_backend": "rocwmma" if candidate == "rocwmma_corrected" else "hip-direct",
            "repeats": repeats, "warmups": warmups, "semantics": "wrap_u64_mod_2_64",
            "input_domain": "canonical_compact_byte_limbs_little_endian", "output_contract": "canonical_compact_low64_byte_limbs",
            "dataset_recipe": RECIPE, "dataset_seed": 3135,
            "reference_method": "boost_int256_full_unsigned_product_native_u64_full_audit_cpp_int_row_edges",
            "reference_buffers_released_before_setup": True,
            "setup_scope": "device_probe_common_output_and_audit_buffers_optional_resident_input_alloc_pack",
            "cold_boundary": "context_and_output_retained_input_and_transport_alloc_pack_export_free_per_batch",
            "kernel_selection": selection, "arithmetic_certificate_sha256": (file_digest(Path(__file__).parent/"data"/("wrap64_signed_mfma_"+target.split(":")[0]+".json" if mfma_k else "wrap64_signed_radix_certificate.json")) if candidate in ("signed_radix","signed_radix_mfma") else None),
            "compute_layout": "cdna_wave64_16x16_signed_radix_register_v1" if mfma_k else "tile16_canonical_bytes_recode_v1" if candidate=="signed_radix" else "canonical_bytes_control", "colpair_experiment_enabled": False, "mma_terms_per_fold": (mfma_k or 16) if radix else None,
            "mathematical_pair_i32_cap": (131071 if candidate != "native_u8" else 33025) if radix else None,
            "static_lds_bytes": (0 if mfma_k else 4096) if radix else None, "explicit_call_scratch_bytes": 4 if radix else 0, "rocwmma_max_k": 32768,
            "timer": "std_chrono_steady_clock_nanoseconds", "production_timing_hooks_enabled": False, "phase_clocks_inside_production_call": False,
            "memory_budget_bytes": BUDGET,
            "memory_exclusions": "allocator bookkeeping, driver/HIP events, context and fixed metadata; logical payload only, not RSS",
            "device_output_canary_bytes": 64, "device_output_canary_checked": True, "host_export_padding_columns": 3,
            "host_export_padding_checked": True, "verified_outputs_per_batch": 1}


def validate_capture(capture, case, run_id, repeats, warmups):
    demand(type(capture) is dict, "capture must be object")
    for key, value in contract(case, run_id, repeats, warmups, capture.get("target")).items():
        demand(key in capture and same(capture[key], value), f"metadata mismatch: {key}")
    demand(type(capture.get("device_name")) is str and capture["device_name"].strip(), "missing device identity")
    demand(type(capture.get("target")) is str and re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+:-]+)?", capture["target"]), "invalid actual target")
    if case["candidate"] in ("signed_radix", "native_u8"):
        demand(capture["target"].split(":")[0] in ("gfx1100", "gfx1101", "gfx1102"), "target disagrees with current native WMMA admission")
    for key in ("hip_runtime_version", "hip_driver_version", "setup_ns", "reference_and_dataset_ns", "clock_period_numerator", "clock_period_denominator"):
        demand(integer(capture.get(key), 1, (1 << 63)-1), f"invalid identity/timing: {key}")
    if case["scope"] in ("resident","repeated_b"):
        demand(positive(capture.get("resident_input_prepare_ns")) and capture["resident_input_prepare_ns"] <= capture["setup_ns"], "invalid resident preparation")
    else:
        demand("resident_input_prepare_ns" in capture and capture["resident_input_prepare_ns"] is None, "irrelevant resident preparation")
    for key in ("a_checksum", "b_checksum", "output_checksum"):
        demand(type(capture.get(key)) is str and re.fullmatch(r"[0-9a-f]{16}", capture[key]), "invalid full checksum")
    a, b = dataset(case)
    demand(capture["a_checksum"] == checksum(a) and capture["b_checksum"] == checksum(b), "input recipe/checksum mismatch")
    ab, mn = case["m"]*case["k"] + case["k"]*case["n"], case["m"]*case["n"]
    width = capture.get("reference_fixed_operand_object_bytes")
    demand(integer(width, 32, 128) and same(capture.get("reference_operand_storage_bytes"), ab*width), "invalid Boost reference memory")
    oracle = 8*ab+8*mn+ab*width+2048
    execution = 24*ab+64*mn+24*case["m"]+4096
    demand(integer(capture.get("logical_live_payload_envelope_bytes"), 1, BUDGET)
           and capture["logical_live_payload_envelope_bytes"] == max(oracle, execution), "invalid aggregate payload bound")
    samples = capture.get("samples")
    demand(type(samples) is list and len(samples) == repeats, "invalid sample count")
    phase_names = {"input_buffer_allocation", "input_pack_and_upload", "gemm_calls", "output_export", "input_and_transport_release"}
    for sample in samples:
        demand(type(sample) is dict and sample.get("exact_reference_passed") is True and sample.get("uninstrumented_exact_reference_passed") is True
               and positive(sample.get("uninstrumented_host_call_ns")) and "host_call_ns" in sample
               and (sample["host_call_ns"] is None or positive(sample["host_call_ns"])), "invalid paired timing/audit")
        phases = sample.get("instrumented_phases_ns")
        demand(type(phases) is dict and set(phases) == phase_names, "missing diagnostic phases")
        for key, value in phases.items():
            used = key == "gemm_calls" or (key in ("input_buffer_allocation", "input_and_transport_release") and case["scope"] == "cold") or (key in ("input_pack_and_upload", "output_export") and case["scope"] != "resident")
            demand(integer(value, 1 if used else 0, (1 << 63)-1 if used else 0), "invalid diagnostic phase scope/timing")
        events = sample.get("events")
        demand(type(events) is list, "missing diagnostic event list")
        for event in events:
            demand(type(event) is dict and type(event.get("label")) is str and event["label"] and "elapsed_ns" in event
                   and (event["elapsed_ns"] is None or positive(event["elapsed_ns"])), "invalid or uncensored HIP event")
    median = statistics.median(sample["uninstrumented_host_call_ns"] for sample in samples)
    demand(positive(capture.get("median_production_batch_ns")) and capture["median_production_batch_ns"] == median
           and positive(capture.get("gemm_calls_per_second")) and math.isclose(capture["gemm_calls_per_second"], 1e9*case["reuse_count"]/median, rel_tol=1e-12), "invalid production median/rate")
    s = capture.get("sustained")
    demand(type(s) is dict, "missing sustained measurement")
    calls, elapsed = s.get("calls"), s.get("elapsed_ns")
    demand(integer(calls, 16, 4096) and integer(elapsed, 1, (1 << 63)-1) and same(s.get("gemm_calls"), calls*case["reuse_count"])
           and same(s.get("minimum_calls"), 16) and same(s.get("maximum_calls"), 4096) and same(s.get("target_ns"), 250000000)
           and s.get("independent_audit_inside_interval") is False and s.get("final_exact_reference_passed") is True
           and positive(s.get("mean_batch_ns")) and positive(s.get("mean_gemm_ns"))
           and math.isclose(s["mean_batch_ns"], elapsed/calls, rel_tol=1e-12)
           and math.isclose(s["mean_gemm_ns"], elapsed/(calls*case["reuse_count"]), rel_tol=1e-12)
           and ((elapsed >= 250000000 and s.get("stop_reason") == "duration") or (elapsed < 250000000 and calls == 4096 and s.get("stop_reason") == "call_cap")), "invalid sustained work/timing/audit")
    validate_provenance(capture.get("build_manifest"), "rocwmma" if case["candidate"] == "rocwmma_corrected" else None)
    return {key: capture[key] for key in ("build_manifest", "device_actual", "device_name", "target", "hip_runtime_version", "hip_driver_version")}


def summarize(observations):
    groups = {}
    names = ("device", "m", "n", "k", "pattern", "scope", "reuse_count")
    for row in observations:
        key = tuple(row["case"][name] for name in names)
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in sorted(groups.items()):
        mfma=any(row["case"]["candidate"]=="signed_radix_mfma" for row in rows)
        expected = CDNA_CANDIDATES if mfma else RDNA_CANDIDATES if key[3] <= 32768 else RDNA_CANDIDATES[:-1]
        native_control=any(row["case"]["candidate"]=="native_u64" for row in rows)
        if native_control and not mfma:expected=(*expected,"native_u64")
        cohorts = {name: [row for row in rows if row["case"]["candidate"] == name] for name in expected}
        indices = [{row["independent_run"] for row in cohort} for cohort in cohorts.values()]
        complete = all(len(index)>=2 for index in indices) and not any(row["case"]["candidate"] not in expected for row in rows) and all(index == indices[0] for index in indices) and all(len(cohort) == len(index) for cohort, index in zip(cohorts.values(), indices))
        record = dict(zip(names, key), complete_matched_cohort=complete,
                      comparison_scope="cdna_signed_mfma_direct_v4_native_u64" if mfma else "rdna_native_u64_inclusive" if native_control else "four_way" if key[3] <= 32768 else "rocwmma_unavailable_k_cap_three_way", promotion_eligible=False)
        if complete:
            stats = {name: dict(processes=len(cohort), median_batch_ns=statistics.median(row["mean_batch_ns"] for row in cohort),
                                minimum_batch_ns=min(row["mean_batch_ns"] for row in cohort), maximum_batch_ns=max(row["mean_batch_ns"] for row in cohort)) for name, cohort in cohorts.items()}
            record["observed_process_statistics"] = stats
            record["median_direct_v4_over_candidate"] = {name: stats["direct_v4"]["median_batch_ns"]/values["median_batch_ns"] for name, values in stats.items() if name != "direct_v4"}
        result.append(record)
    return result


def collect(args):
    cases = load_cases(args.cases)
    demand(integer(args.runs, 1, 100) and integer(args.repeats, 1, 1000) and integer(args.warmups, 1, 1000) and integer(args.timeout, 1, 3600), "invalid collection bounds")
    demand(not os.environ.get("RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT", "").startswith("1"), "active colpair experiment conflicts with direct_v4 control")
    executable = Path(args.exe).resolve(strict=True); artifact = file_digest(executable)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    campaign = uuid.uuid4().hex; cases_hash = file_digest(args.cases); here = Path(__file__).resolve()
    hashes = {name: file_digest(here.with_name(name)) for name in (here.name, "collect_finite_verify.py", "build_manifest.py")}
    build = None; devices = {}; outputs = {}; observations = []
    for independent in range(args.runs):
        order=list(enumerate(cases));shift=independent%len(order);order=order[shift:]+order[:shift]
        if independent%2:order.reverse()
        for case_index, case in order:
            a, b = dataset(case); data_hash = hashlib.sha256(a+b).hexdigest()
            run_id = f"{campaign}-{case_index}-{independent}"
            command = [str(executable), str(case["device"]), case["candidate"], case["scope"], *(str(case[key]) for key in ("m", "n", "k", "pattern", "reuse_count")), str(args.repeats), str(args.warmups), run_id]
            prefix = output/run_id
            record = dict(schema="rns8_wrap64_collection_v1", campaign_id=campaign, run_id=run_id, case=case, independent_run=independent,
                          command=command, artifact_sha256=artifact, cases_sha256=cases_hash, dataset_sha256=data_hash, tool_sha256=hashes,
                          promotion_eligible=False, validated=False,
                          host=dict(platform=platform.platform(), python=sys.version, python_executable=sys.executable, python_executable_sha256=file_digest(sys.executable)),
                          environment={key: os.environ[key] for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "RNS8_WRAP64_HIP_COLPAIR_EXPERIMENT") if key in os.environ})
            def save():
                prefix.with_suffix(".record.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
            try:
                demand(file_digest(executable) == artifact and file_digest(args.cases) == cases_hash, "artifact/cases changed before launch")
                child = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=args.timeout, check=False)
                prefix.with_suffix(".stdout.json").write_text(child.stdout, encoding="utf-8")
                prefix.with_suffix(".stderr.txt").write_text(child.stderr, encoding="utf-8"); record["returncode"] = child.returncode
                demand(file_digest(executable) == artifact and file_digest(args.cases) == cases_hash, "artifact/cases changed during run")
                demand(child.returncode == 0, f"child returned {child.returncode}; unsupported is diagnostic, never a pass")
                capture = strict_json(child.stdout); identity = validate_capture(capture, case, run_id, args.repeats, args.warmups)
                capture["artifact_capture"] = dict(executable_sha256=artifact,executable_path=str(executable),unchanged_during_run=True,rns8_linkage="static" if identity["build_manifest"].get("linkage")=="static-rns8" else "unresolved-shared")
                capture["execution_id"]=run_id
                prefix.with_suffix(".capture.json").write_text(json.dumps(capture,indent=2)+"\n",encoding="utf-8")
                if build is None:
                    build = identity["build_manifest"]
                demand(identity["build_manifest"] == build, "build provenance changed within campaign")
                demand(case["device"] not in devices or devices[case["device"]] == identity, "actual device/runtime identity changed")
                devices[case["device"]] = identity
                key = tuple(case[key] for key in ("m", "n", "k", "pattern"))
                demand(key not in outputs or outputs[key] == capture["output_checksum"], "matched canonical output checksum changed")
                outputs[key] = capture["output_checksum"]
                record.update(validated=True, identity=identity, provenance_complete=manifest_provenance_complete(build, "rocwmma" if case["candidate"] == "rocwmma_corrected" else None))
                observations.append(dict(case=case, independent_run=independent, mean_batch_ns=capture["sustained"]["mean_batch_ns"]))
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                if isinstance(error, subprocess.TimeoutExpired):
                    prefix.with_suffix(".stdout.json").write_bytes(error.stdout or b"")
                    prefix.with_suffix(".stderr.txt").write_bytes(error.stderr or b"")
                record["error"] = str(error); save(); raise
            save()
    summary = dict(schema="rns8_wrap64_comparison_v1", campaign_id=campaign, artifact_sha256=artifact, build_manifest=build,
                   promotion_eligible=False, comparison=summarize(observations), interpretation="one sustained process mean per observation; descriptive ranges, no interpolation, confidence or promotion claim")
    (output/f"comparison-{campaign}.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "cases", "output"):
        parser.add_argument("--"+key, required=True)
    for key, default in (("runs", 5), ("repeats", 9), ("warmups", 3), ("timeout", 600)):
        parser.add_argument("--"+key, type=int, default=default)
    args = parser.parse_args(argv)
    try:
        print(collect(args)); return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
