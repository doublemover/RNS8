#!/usr/bin/env python3
"""Collect matched public block-sparse cold/warm/reuse controls; no promotion."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import random
import time
import statistics
import subprocess
import sys
import uuid
from build_manifest import manifest_provenance_complete
from collect_finite_verify import (BUDGET, MASK, demand, file_digest, integer,
                                   positive, same, strict_json, validate_provenance)

MODES = ("dense", "sparse_scalar", "sparse_wmma", "sparse_hybrid", "sparse_scalar_cold", "sparse_wmma_cold", "sparse_hybrid_cold")
KEYS = {"device", "mode", "m", "n", "k", "pattern", "reuse_count", "hybrid_threshold"}
RECIPE = "splitmix64_signed64_block_masks_seed3135_v1"


def validate_case(case):
    demand(type(case) is dict and set(case) == KEYS, "case requires exactly documented controls")
    demand(integer(case["device"], 0, 63) and case["mode"] in MODES, "invalid explicit HIP device/mode")
    demand(all(integer(case[key], 1, 65537 if key == "k" else 512) for key in ("m", "n", "k"))
           and case["m"]*case["n"]*case["k"] <= 134217728, "shape/reference work outside bounds")
    demand(case["pattern"] in ("dense", "mixed", "disjoint") and integer(case["reuse_count"], 1, 256)
           and integer(case["hybrid_threshold"], 0, (1 << 32)-1), "invalid pattern/reuse/threshold")
    demand("hybrid" in case["mode"] or case["hybrid_threshold"] == 0, "threshold applies only to hybrid")
    return dict(case)


def load_cases(path):
    doc = strict_json(Path(path).read_text(encoding="utf-8-sig"))
    demand(type(doc) is dict and set(doc) == {"schema", "cases"} and doc["schema"] == "rns8_block_sparse_cases_v1", "invalid case manifest")
    demand(type(doc["cases"]) is list and 1 <= len(doc["cases"]) <= 64, "case count must be 1..64")
    cases = [validate_case(case) for case in doc["cases"]]
    demand(len({json.dumps(case, sort_keys=True) for case in cases}) == len(cases), "duplicate cases are not independent observations")
    return cases


def active(case, lhs, tile, kb):
    if case["pattern"] == "dense":
        return True
    if case["pattern"] == "disjoint":
        return kb % 2 == (0 if lhs else 1)
    return True if tile % 3 == 0 else kb == 0 if tile % 3 == 2 else kb % 2 == (0 if lhs else 1)


def structure(case):
    rt, ct, kt = (case["m"]+15)//16, (case["n"]+15)//16, (case["k"]+63)//64
    ac = sum(active(case, True, r, k) for r in range(rt) for k in range(kt))
    bc = sum(active(case, False, col, k) for col in range(ct) for k in range(kt))
    intersections = [sum(active(case, True, r, k) and active(case, False, col, k) for k in range(kt))
                     for r in range(rt) for col in range(ct)]
    hybrid = "hybrid" in case["mode"]
    return dict(modeled_a_blocks=ac, modeled_b_blocks=bc, modeled_packed_payload_bytes=(ac+bc)*20*1024,
                modeled_directory_bytes=4*(rt+ct+2+ac+bc), modeled_total_block_intersections=sum(intersections),
                modeled_empty_output_tiles=intersections.count(0),
                modeled_hybrid_scalar_tiles=sum(value <= case["hybrid_threshold"] for value in intersections) if hybrid else None,
                modeled_hybrid_wmma_tiles=sum(value > case["hybrid_threshold"] for value in intersections) if hybrid else None)


def dataset(case):
    state = 3135
    operands = []
    for lhs, count in ((True, case["m"]*case["k"]), (False, case["k"]*case["n"])):
        values = bytearray()
        for index in range(count):
            state = (state + 0x9e3779b97f4a7c15) & MASK
            word = state
            word = ((word ^ (word >> 30)) * 0xbf58476d1ce4e5b9) & MASK
            word = ((word ^ (word >> 27)) * 0x94d049bb133111eb) & MASK
            word ^= word >> 31
            if index % 257 == 0:
                word = 1 << 63
            elif index % 257 == 1:
                word = (1 << 63)-1
            tile, kb = ((index//case["k"])//16, (index % case["k"])//64) if lhs else ((index % case["n"])//16, (index//case["n"])//64)
            word = (word or 1) if active(case, lhs, tile, kb) else 0
            values.extend(word.to_bytes(8, "little"))
        operands.append(values)
    return operands


def checksum(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte)*1099511628211) & MASK
    return f"{value:016x}"


def contract(case, run_id, repeats, warmups):
    cold = case["mode"].endswith("_cold")
    return {"schema": "rns8_block_sparse_microcase_v1", "promotion_eligible": False, "run_id": run_id,
            "backend_requested": "hip-direct", "backend_actual": "hip-direct", "device_requested": case["device"], "device_actual": case["device"],
            **{key: case[key] for key in ("mode", "m", "n", "k", "pattern", "reuse_count", "hybrid_threshold")},
            "prefix": 20, "semantics": "exact_wide_signed", "repeats": repeats, "warmups": warmups,
            "dataset_recipe": RECIPE, "dataset_seed": 3135, "source_versions": [1, 1], "content_generations": None,
            "content_generation_visibility": "opaque_handle_checked_by_public_owner_api",
            "reference_method": "boost_int256_full_product_with_cpp_int_row_edge_audits", "reference_buffers_released_before_setup": True,
            "setup_scope": "context_plan_workspace_matrices_native_and_RNS_pack_then_optional_warm_prepare",
            "timed_scope": "reuse_public_dense_GEMMs" if case["mode"] == "dense" else
                           "prepare_both_reuse_public_sparse_GEMMs_checked_destroy_both" if cold else "reuse_public_prepared_sparse_GEMMs",
            "timer": "std_chrono_steady_clock_nanoseconds", "actual_cache_allocation_bytes": None,
            "structure_origin": "deterministic_exact_zero_native_recipe_not_device_query", **structure(case),
            "memory_budget_bytes": BUDGET, "resident_payload_scope": "queried_after_input_packing_before_first_call",
            "allocation_query_scope": "A_B_C_requested_capacity_and_cached_staging_at_two_boundaries_not_peak_or_sparse_cache",
            "memory_exclusions": "allocator/vector spare capacity, context runtime and fixed metadata; logical payload only, not process RSS",
            "exact_full_output_and_padding_checked": True, "verified_outputs_per_batch": 1}


def validate_allocations(values, case, envelope):
    names = {"logical_primary_bytes", "host_payload_capacity_bytes", "device_payload_capacity_bytes", "host_current_bytes", "device_current_bytes",
             "device_upload_staging_bytes", "device_export_staging_bytes", "device_status_and_schedule_bytes", "device_total_capacity_bytes"}
    demand(type(values) is list and len(values) == 3, "missing A/B/C allocation query")
    total = 0
    for record, cells in zip(values, (case["m"]*case["k"], case["k"]*case["n"], case["m"]*case["n"])):
        demand(type(record) is dict and set(record) == names and all(integer(value, 0, BUDGET) for value in record.values()), "invalid allocation fields")
        demand(record["logical_primary_bytes"] == cells*20
               and record["host_current_bytes"] <= record["host_payload_capacity_bytes"]
               and record["device_current_bytes"] <= record["device_payload_capacity_bytes"]
               and record["device_total_capacity_bytes"] == sum(record[key] for key in ("device_payload_capacity_bytes", "device_upload_staging_bytes", "device_export_staging_bytes", "device_status_and_schedule_bytes")), "incoherent allocation accounting")
        total += record["host_payload_capacity_bytes"] + record["device_total_capacity_bytes"]
    demand(total <= envelope, "queried matrix capacities exceed conservative envelope")


def validate_capture(capture, case, run_id, repeats, warmups):
    demand(type(capture) is dict, "capture must be object")
    for key, expected in contract(case, run_id, repeats, warmups).items():
        demand(key in capture and same(capture[key], expected), f"metadata mismatch: {key}")
    for key in ("device_name", "selected_dense_kernel"):
        demand(type(capture.get(key)) is str and capture[key].strip(), f"missing identity: {key}")
    demand(type(capture.get("target")) is str and re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+:-]+)?", capture["target"]), "invalid target")
    if "wmma" in case["mode"] or "hybrid" in case["mode"]:
        demand(capture["target"].split(":")[0] in ("gfx1100", "gfx1101", "gfx1102"), "target disagrees with current sparse WMMA admission")
    for key in ("hip_runtime_version", "hip_driver_version"):
        demand(integer(capture.get(key), 1, (1 << 31)-1), "invalid runtime identity")
    for key in ("reference_and_dataset_ns", "native_setup_ns", "setup_ns", "clock_period_numerator", "clock_period_denominator"):
        demand(integer(capture.get(key), 1, (1 << 63)-1), f"invalid timing: {key}")
    demand(capture["setup_ns"] >= capture["native_setup_ns"], "incoherent setup timing")
    warm = case["mode"] != "dense" and not case["mode"].endswith("_cold")
    demand(positive(capture.get("warm_prepare_ns")) if warm else "warm_prepare_ns" in capture and capture["warm_prepare_ns"] is None, "invalid warm preparation scope")
    if warm:
        demand(capture["setup_ns"] >= capture["warm_prepare_ns"], "warm preparation exceeds setup")
    for key in ("a_checksum", "b_checksum", "output_checksum"):
        demand(type(capture.get(key)) is str and re.fullmatch(r"[0-9a-f]{16}", capture[key]), "invalid full checksum")
    a, b = dataset(case)
    demand(capture["a_checksum"] == checksum(a) and capture["b_checksum"] == checksum(b), "native input recipe/checksum mismatch")
    ids = capture.get("source_instance_ids")
    out = capture.get("output_instance_id")
    demand(type(ids) is list and len(ids) == 2 and all(integer(value, 1, MASK) for value in ids)
           and integer(out, 1, MASK) and len(set([*ids, out])) == 3, "invalid public matrix instance identity")
    ab, mn = case["m"]*case["k"] + case["k"]*case["n"], case["m"]*case["n"]
    width = capture.get("reference_fixed_operand_object_bytes")
    demand(integer(width, 32, 128) and same(capture.get("reference_operand_storage_bytes"), ab*width), "invalid fixed Boost reference memory")
    counts = structure(case)
    modeled = 80*ab + 112*mn + 2*(counts["modeled_packed_payload_bytes"]+counts["modeled_directory_bytes"]) + 8*max(case["m"]*case["k"], case["k"]*case["n"]) + counts["modeled_directory_bytes"]+4
    work = capture.get("plan_workspace_bytes")
    demand(integer(work, 0, BUDGET), "invalid plan workspace")
    envelope = max(8*ab+24*mn+ab*width+2048, modeled+work)
    demand(integer(capture.get("logical_live_payload_envelope_bytes"), 1, BUDGET)
           and capture["logical_live_payload_envelope_bytes"] == envelope
           and integer(capture.get("resident_payload_bytes"), 20*(ab+mn), 64*(ab+mn)), "invalid aggregate memory envelope")
    for key in ("matrix_allocations_at_setup", "matrix_allocations_after_final_audit"):
        validate_allocations(capture.get(key), case, modeled)
    samples = capture.get("samples")
    demand(type(samples) is list and len(samples) == repeats, "invalid sample count")
    cold = case["mode"].endswith("_cold")
    for sample in samples:
        demand(type(sample) is dict and set(sample) == {"wall_ns", "preparation_ns", "destruction_ns", "exact_reference_passed"}
               and integer(sample["wall_ns"], 1, (1 << 63)-1) and sample["exact_reference_passed"] is True, "invalid sample timing/audit")
        for key in ("preparation_ns", "destruction_ns"):
            demand(integer(sample[key], 1 if cold else 0, sample["wall_ns"] if cold else 0), "invalid cold preparation/destruction timing")
        demand(sample["preparation_ns"]+sample["destruction_ns"] <= sample["wall_ns"], "cold phases exceed whole call")
    median = statistics.median(sample["wall_ns"] for sample in samples)
    demand(positive(capture.get("median_wall_ns")) and capture["median_wall_ns"] == median
           and positive(capture.get("gemm_calls_per_second")) and math.isclose(capture["gemm_calls_per_second"], 1e9*case["reuse_count"]/median, rel_tol=1e-12), "invalid median/call rate")
    sustained = capture.get("sustained")
    demand(type(sustained) is dict, "missing sustained measurement")
    calls, elapsed = sustained.get("calls"), sustained.get("elapsed_ns")
    demand(integer(calls, 16, 4096) and integer(elapsed, 1, (1 << 63)-1)
           and same(sustained.get("gemm_calls"), calls*case["reuse_count"])
           and same(sustained.get("minimum_calls"), 16) and same(sustained.get("maximum_calls"), 4096)
           and same(sustained.get("target_ns"), 250000000) and sustained.get("independent_audit_inside_interval") is False
           and sustained.get("final_exact_reference_passed") is True
           and positive(sustained.get("mean_batch_ns")) and positive(sustained.get("mean_gemm_ns"))
           and math.isclose(sustained["mean_batch_ns"], elapsed/calls, rel_tol=1e-12)
           and math.isclose(sustained["mean_gemm_ns"], elapsed/(calls*case["reuse_count"]), rel_tol=1e-12)
           and ((elapsed >= 250000000 and sustained.get("stop_reason") == "duration") or
                (elapsed < 250000000 and calls == 4096 and sustained.get("stop_reason") == "call_cap")), "invalid sustained work/timing/audit")
    validate_provenance(capture.get("build_manifest"))
    return {key: capture[key] for key in ("build_manifest", "device_actual", "device_name", "target", "hip_runtime_version", "hip_driver_version")}


def summarize(observations):
    """Only complete, independently matched observed cold/warm/control cohorts.

    A range separation is descriptive, never a confidence interval. No linear
    extrapolation or untested reuse count becomes a break-even result.
    """
    groups = {}
    for observation in observations:
        case = observation["case"]
        key = tuple(case[key] for key in ("device", "m", "n", "k", "pattern"))
        groups.setdefault(key, []).append(observation)
    results = []
    for key, rows in sorted(groups.items()):
        candidates = sorted({(row["case"]["mode"].removesuffix("_cold"), row["case"]["hybrid_threshold"])
                             for row in rows if row["case"]["mode"] != "dense"})
        for mode, threshold in candidates:
            points = []; first = None
            for reuse in sorted({row["case"]["reuse_count"] for row in rows}):
                modes = {name: [row for row in rows if row["case"]["mode"] == name and row["case"]["reuse_count"] == reuse
                               and (name == "dense" or row["case"]["hybrid_threshold"] == threshold)]
                         for name in ("dense", mode, mode+"_cold")}
                indices = [{row["independent_run"] for row in values} for values in modes.values()]
                complete = all(indices) and all(values == indices[0] for values in indices) and all(len(values) == len(index) for values, index in zip(modes.values(), indices))
                point = {"reuse_count": reuse, "complete_matched_cohort": complete}
                if complete:
                    stats = {name: {"processes": len(values), "median_batch_ns": statistics.median(row["mean_batch_ns"] for row in values),
                                    "minimum_batch_ns": min(row["mean_batch_ns"] for row in values), "maximum_batch_ns": max(row["mean_batch_ns"] for row in values)} for name, values in modes.items()}
                    point["observed_process_statistics"] = stats
                    point["median_dense_over_warm"] = stats["dense"]["median_batch_ns"]/stats[mode]["median_batch_ns"]
                    point["median_dense_over_cold"] = stats["dense"]["median_batch_ns"]/stats[mode+"_cold"]["median_batch_ns"]
                    separated = len(indices[0]) >= 5 and stats[mode+"_cold"]["maximum_batch_ns"] < stats["dense"]["minimum_batch_ns"]
                    point["at_least_five_processes_and_cold_range_below_dense"] = separated
                    if separated and first is None:
                        first = reuse
                points.append(point)
            results.append(dict(zip(("device", "m", "n", "k", "pattern"), key), mode=mode, hybrid_threshold=threshold,
                                first_observed_range_separated_cold_win_reuse=first, tested_points=points,
                                interpretation="independent process sustained batch observations; descriptive ranges, no extrapolation or confidence claim"))
    return results


def collection_schedule(cases,runs,seed):
    """Bounded matched blocks with one process per method, rotated per repeat."""
    demand(integer(runs,1,100) and type(seed) is int,"invalid schedule controls")
    groups={}
    for index,case in enumerate(cases):
        key=tuple(case[name] for name in ('device','m','n','k','pattern','reuse_count'))
        groups.setdefault(key,[]).append(index)
    blocks=list(groups.values());rng=random.Random(seed)
    for block in blocks:rng.shuffle(block)
    result=[]
    for run in range(runs):
        order=list(range(len(blocks)));rng.shuffle(order)
        for group in order:
            block=blocks[group];shift=run%len(block);indices=block[shift:]+block[:shift]
            for position,index in enumerate(indices):
                result.append(dict(case_index=index,independent_run=run,matched_block=group,position_in_block=position,ordinal=len(result)))
    return result


def collect(args):
    cases = load_cases(args.cases)
    demand(integer(args.runs, 1, 100) and integer(args.repeats, 1, 1000) and integer(args.warmups, 1, 1000) and integer(args.timeout, 1, 3600), "invalid collection bounds")
    executable = Path(args.exe).resolve(strict=True); artifact = file_digest(executable)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    campaign = uuid.uuid4().hex; cases_hash = file_digest(args.cases)
    here = Path(__file__).resolve()
    hashes = {name: file_digest(here.with_name(name)) for name in (here.name, "collect_finite_verify.py", "build_manifest.py")}
    cohort = {}; outputs = {}; build = None; observations = []
    seed=getattr(args,'order_seed',5151);schedule=collection_schedule(cases,args.runs,seed)
    schedule_document=dict(schema='rns8_block_sparse_schedule_v1',order_seed=seed,cases_sha256=cases_hash,schedule=schedule)
    schedule_text=json.dumps(schedule_document,sort_keys=True,separators=(',',':'))+'\n'
    schedule_hash=hashlib.sha256(schedule_text.encode()).hexdigest();schedule_path=output/(campaign+'.schedule.json')
    with schedule_path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(schedule_text)
    for job in schedule:
            case_index=job['case_index'];independent=job['independent_run'];case=cases[case_index]
            a,b=dataset(case);dataset_hash=hashlib.sha256(a+b).hexdigest()
            run_id = f"{campaign}-{case_index}-{independent}"
            command = [str(executable), str(case["device"]), case["mode"], *(str(case[key]) for key in ("m", "n", "k", "pattern", "reuse_count", "hybrid_threshold")), str(args.repeats), str(args.warmups), run_id]
            prefix = output/run_id
            record = dict(schema="rns8_block_sparse_collection_v1", campaign_id=campaign, run_id=run_id, case=case,
                          independent_run=independent, command=command, order_seed=seed, schedule_sha256=schedule_hash, ordinal=job["ordinal"], matched_block=job["matched_block"], position_in_block=job["position_in_block"], started_unix_ns=time.time_ns(), artifact_sha256=artifact, cases_sha256=cases_hash,
                          dataset_sha256=dataset_hash, tool_sha256=hashes, promotion_eligible=False, validated=False,
                          host=dict(platform=platform.platform(), python=sys.version, python_executable=sys.executable, python_executable_sha256=file_digest(sys.executable)),
                          environment={key: os.environ[key] for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if key in os.environ})
            def save():
                prefix.with_suffix(".record.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
            try:
                demand(file_digest(schedule_path)==schedule_hash,"schedule changed before launch")
                demand(file_digest(executable) == artifact, "executable changed before launch")
                demand(file_digest(args.cases) == cases_hash, "case manifest changed before launch")
                child = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=args.timeout, check=False)
                prefix.with_suffix(".stdout.json").write_text(child.stdout, encoding="utf-8")
                prefix.with_suffix(".stderr.txt").write_text(child.stderr, encoding="utf-8")
                record["returncode"] = child.returncode
                demand(file_digest(executable) == artifact, "executable changed during run")
                demand(file_digest(args.cases) == cases_hash, "case manifest changed during run")
                demand(child.returncode == 0, f"child returned {child.returncode}; unsupported is diagnostic, not a pass")
                capture = strict_json(child.stdout); identity = validate_capture(capture, case, run_id, args.repeats, args.warmups)
                if build is None:
                    build = identity["build_manifest"]
                demand(identity["build_manifest"] == build, "build provenance changed within campaign")
                demand(case["device"] not in cohort or cohort[case["device"]] == identity, "device/runtime identity changed within campaign")
                cohort[case["device"]] = identity
                output_key = tuple(case[key] for key in ("m", "n", "k", "pattern"))
                demand(output_key not in outputs or outputs[output_key] == capture["output_checksum"], "matched output checksum changed")
                outputs[output_key] = capture["output_checksum"]
                record.update(validated=True, identity=identity, provenance_complete=manifest_provenance_complete(build))
                observations.append(dict(case=case, independent_run=independent, mean_batch_ns=capture["sustained"]["mean_batch_ns"]))
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                if isinstance(error, subprocess.TimeoutExpired):
                    prefix.with_suffix(".stdout.json").write_bytes(error.stdout or b"")
                    prefix.with_suffix(".stderr.txt").write_bytes(error.stderr or b"")
                record["error"] = str(error); save(); raise
            save()
    summary = dict(schema="rns8_block_sparse_reuse_comparison_v1", campaign_id=campaign, artifact_sha256=artifact,
                   build_manifest=build, order_seed=seed, schedule_sha256=schedule_hash, promotion_eligible=False, comparison=summarize(observations))
    (output/f"comparison-{campaign}.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("exe", "cases", "output"):
        parser.add_argument("--"+flag, required=True)
    for flag, default in (("runs", 5), ("repeats", 9), ("warmups", 3), ("timeout", 600)):
        parser.add_argument("--"+flag, type=int, default=default)
    parser.add_argument("--order-seed",type=int,default=5151)
    args = parser.parse_args(argv)
    try:
        print(collect(args)); return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
