#!/usr/bin/env python3
"""Collect public finite verifier microcases, preserving diagnostic raw captures.

This is a finite-ring replay experiment, not a production qualification adapter.
One fresh child process is one independent observation; within-child repeats are
not independent runs. No ISA, default-route or integer-exactness promotion.
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
import statistics
import subprocess
import sys
import uuid
from build_manifest import manifest_provenance_complete

SCHEMA = "rns8_finite_verify_microcase_v1"
CASE_KEYS = {"backend", "device", "mode", "m", "n", "k", "modulus", "trials", "challenge_seed"}
MODES = ("verify_only", "compute_verify", "full_reference")
MASK = (1 << 64) - 1
RECIPE = "splitmix64_finite_u8_seed3135_extrema_v1"
BUDGET = 256 * 1024 * 1024


def demand(value, why):
    if not value:
        raise ValueError(why)


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def same(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same(actual[key], value) for key, value in expected.items())
    return actual == expected


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            demand(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def validate_case(case):
    demand(type(case) is dict and set(case) == CASE_KEYS, "case requires exactly the documented controls")
    demand(case["backend"] in ("cpu-reference", "hip-direct"), "explicit backend required")
    demand(same(case["device"], -1) if case["backend"] == "cpu-reference" else integer(case["device"], 0, 63), "invalid explicit device")
    demand(case["mode"] in MODES, "invalid mode")
    demand(all(integer(case[key], 1, 65537 if key == "k" else 512) for key in ("m", "n", "k")), "shape outside bounds")
    demand(case["m"] * case["n"] * case["k"] <= 134217728, "reference work exceeds bound")
    demand(integer(case["modulus"], 2, 256) and integer(case["trials"], 0, 256)
           and integer(case["challenge_seed"], 0, MASK), "invalid modulus/trials/replay seed")
    demand(case["trials"] == 0 and case["challenge_seed"] == 0 if case["mode"] == "full_reference" else case["trials"] > 0,
           "irrelevant or missing reference/verifier controls")
    return dict(case)


def load_cases(path):
    document = strict_json(Path(path).read_text(encoding="utf-8-sig"))
    demand(type(document) is dict and set(document) == {"schema", "cases"}
           and document["schema"] == "rns8_finite_verify_cases_v1", "invalid case manifest")
    cases = document["cases"]
    demand(type(cases) is list and 1 <= len(cases) <= 64, "case count must be 1..64")
    cases = [validate_case(case) for case in cases]
    identities = [json.dumps(case, sort_keys=True, separators=(",", ":")) for case in cases]
    demand(len(identities) == len(set(identities)), "duplicate cases are not independent observations")
    return cases


def checksum(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & MASK
    return f"{value:016x}"


def dataset(case):
    state = 3135
    values = []
    for count in (case["m"] * case["k"], case["k"] * case["n"]):
        operand = bytearray(count)
        for index in range(count):
            state = (state + 0x9e3779b97f4a7c15) & MASK
            word = state
            word = ((word ^ (word >> 30)) * 0xbf58476d1ce4e5b9) & MASK
            word = ((word ^ (word >> 27)) * 0x94d049bb133111eb) & MASK
            word ^= word >> 31
            operand[index] = word % case["modulus"]
        operand[0] = case["modulus"] - 1
        if count > 1:
            operand[1] = 0
        values.append(operand)
    return values


def contract(case, run_id, repeats, warmups):
    return {"schema": SCHEMA, "promotion_eligible": False, "run_id": run_id, "mode": case["mode"],
            "backend_requested": case["backend"], "backend_actual": case["backend"],
            "device_requested": case["device"], "device_actual": case["device"],
            **{key: case[key] for key in ("m", "n", "k", "modulus", "trials", "challenge_seed")},
            "semantics": "finite_ring_u8", "dataset_recipe": RECIPE, "dataset_seed": 3135,
            "repeats": repeats, "warmups": warmups,
            "reference_method": "scalar_uint64_full_dot_then_modulus",
            "comparator_method": "blocked_uint64_32x32x64_full_dot_then_modulus",
            "timer": "std_chrono_steady_clock_nanoseconds",
            "timed_scope": {"full_reference": "allocate_export_A_B_C_blocked_reference_full_compare",
                            "compute_verify": "public_gemm_then_public_verify_all_wall_cost",
                            "verify_only": "public_verify_all_wall_cost"}[case["mode"]],
            "setup_scope": "context_plans_workspace_matrices_pack_A_B_gold_C",
            "memory_exclusions": "allocator bookkeeping, context/compiler runtime and fixed metadata; this is not process RSS",
            "memory_budget_bytes": BUDGET, "exact_full_output_and_padding_checked": True, "inputs_unchanged_checked": True}


def validate_provenance(manifest, backend=None):
    """Accept structural diagnostic evidence without upgrading its eligibility."""
    demand(type(manifest) is dict and type(manifest.get("source_dirty")) is bool, "missing build provenance")
    structural = dict(manifest, source_dirty=False)
    if structural.get("dependency_digest") == "unavailable":
        structural["dependency_digest"] = "0"*64
    configuration = structural.get("configuration")
    if isinstance(configuration, str):
        structural["configuration"] = "\n".join("garner_table=OFF" if line.partition("=")[0] == "garner_table" else line for line in configuration.splitlines())
    demand(manifest_provenance_complete(structural, backend), "invalid structural build provenance")


def validate_capture(capture, case, run_id, repeats, warmups):
    demand(type(capture) is dict, "capture must be an object")
    for key, value in contract(case, run_id, repeats, warmups).items():
        demand(key in capture and same(capture[key], value), f"metadata mismatch: {key}")
    for key in ("device_name", "target", "selected_compute_kernel", "memory_exclusions"):
        demand(type(capture.get(key)) is str and bool(capture[key].strip()), f"missing identity/scope: {key}")
    demand(capture["target"] == "cpu" if case["backend"] == "cpu-reference" else
           bool(re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+:-]+)?", capture["target"])), "invalid actual target")
    for key in ("hip_runtime_version", "hip_driver_version"):
        demand(integer(capture.get(key), 0 if case["backend"] == "cpu-reference" else 1, (1 << 31)-1), "invalid runtime identity")
    for key in ("setup_ns", "oracle_and_dataset_ns"):
        demand(positive(capture.get(key)), f"invalid timing: {key}")
    for key in ("clock_period_numerator", "clock_period_denominator"):
        demand(integer(capture.get(key), 1, (1 << 63)-1), f"invalid clock period: {key}")
    for key in ("a_checksum", "b_checksum", "output_checksum"):
        demand(type(capture.get(key)) is str and re.fullmatch(r"[0-9a-f]{16}", capture[key]), f"invalid checksum: {key}")
    a, b = dataset(case)
    demand(capture["a_checksum"] == checksum(a) and capture["b_checksum"] == checksum(b), "dataset recipe/checksum mismatch")
    cells = case["m"]*case["k"] + case["k"]*case["n"] + case["m"]*case["n"]
    scratch = (case["n"] if case["backend"] == "cpu-reference" else 2*case["n"]+8) + 4*case["k"]
    work = capture.get("plan_workspace_bytes")
    demand(integer(work, 0, BUDGET), "invalid plan workspace")
    envelope = 5*cells + 16*case["m"]*case["n"] + 3*case["m"] + scratch + work
    demand(integer(capture.get("logical_live_payload_envelope_bytes"), 1, BUDGET)
           and capture["logical_live_payload_envelope_bytes"] == envelope
           and integer(capture.get("resident_payload_bytes"), cells, 2*cells), "invalid aggregate live memory accounting")
    samples = capture.get("samples")
    demand(type(samples) is list and len(samples) == repeats, "invalid sample count")
    for sample in samples:
        demand(type(sample) is dict and set(sample) == {"wall_ns", "exact_reference_passed"}
               and integer(sample["wall_ns"], 1, (1 << 63)-1) and sample["exact_reference_passed"] is True,
               "invalid exact sample/timing")
    median = statistics.median(sample["wall_ns"] for sample in samples)
    demand(positive(capture.get("median_wall_ns")) and capture["median_wall_ns"] == median
           and positive(capture.get("calls_per_second"))
           and math.isclose(capture["calls_per_second"], 1e9/median, rel_tol=1e-12), "invalid median/rate")
    sustained = capture.get("sustained")
    demand(type(sustained) is dict, "missing sustained measurement")
    calls, elapsed = sustained.get("calls"), sustained.get("elapsed_ns")
    demand(integer(calls, 16, 4096) and integer(elapsed, 1, (1 << 63)-1)
           and same(sustained.get("minimum_calls"), 16) and same(sustained.get("maximum_calls"), 4096)
           and same(sustained.get("target_ns"), 250000000)
           and sustained.get("independent_audit_inside_interval") is False
           and sustained.get("final_exact_reference_passed") is True
           and positive(sustained.get("mean_wall_ns"))
           and math.isclose(sustained["mean_wall_ns"], elapsed/calls, rel_tol=1e-12)
           and ((elapsed >= 250000000 and sustained.get("stop_reason") == "duration")
                or (elapsed < 250000000 and calls == 4096 and sustained.get("stop_reason") == "call_cap")),
           "invalid sustained count/duration/audit")
    verification = capture.get("verification")
    if case["mode"] == "full_reference":
        demand("verification" in capture and verification is None, "reference must not claim probabilistic verification")
    else:
        demand(type(verification) is dict, "missing verification report")
        expected = {"passed": True, "sampler": 1, "sampler_name": "splitmix64_binary_v1", "replay_seed": case["challenge_seed"],
                    "trials": case["trials"], "workspace_bytes": scratch, "ideal_failure_bound_exponent": case["trials"],
                    "ideal_failure_probability": math.ldexp(1., -case["trials"]), "probability_model": "independent_uniform_binary_bits_only",
                    "fresh_entropy": False, "integer_equality_certified": False}
        for key, value in expected.items():
            # JSON represents the exact power-of-two probability as a float.
            demand(key in verification and same(verification[key], value), f"verifier metadata mismatch: {key}")
        for key in ("matrix_instance_ids", "generations"):
            values = verification.get(key)
            demand(type(values) is list and len(values) == 3 and all(integer(value, 1, MASK) for value in values), "invalid verifier lifetime bindings")
        demand(len(set(verification["matrix_instance_ids"])) == 3, "unexpected aliased benchmark operands")
    manifest = capture.get("build_manifest")
    validate_provenance(manifest)
    return {"build_manifest": manifest, **{key: capture[key] for key in ("backend_actual", "device_actual", "device_name", "target", "hip_runtime_version", "hip_driver_version")}}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(args):
    cases = load_cases(args.cases)
    demand(integer(args.runs, 1, 100) and integer(args.repeats, 1, 1000) and integer(args.warmups, 1, 1000)
           and integer(args.timeout, 1, 3600), "invalid collection bounds")
    executable = Path(args.exe).resolve(strict=True)
    artifact = file_digest(executable)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    campaign = uuid.uuid4().hex
    cases_hash = file_digest(args.cases)
    cohort = {}; outputs = {}; build = None
    collector_file = Path(__file__).resolve()
    tool_hashes = {path.name: file_digest(path) for path in (collector_file, collector_file.with_name("build_manifest.py"))}
    for case_index, case in enumerate(cases):
        a, b = dataset(case)
        dataset_hash = hashlib.sha256(a+b).hexdigest()
        for independent in range(args.runs):
            run_id = f"{campaign}-{case_index}-{independent}"
            command = [str(executable), case["backend"], str(case["device"]), case["mode"],
                       *(str(case[key]) for key in ("m", "n", "k", "modulus", "trials")),
                       str(args.repeats), str(args.warmups), str(case["challenge_seed"]), run_id]
            prefix = destination/run_id
            record = {"schema": "rns8_finite_verify_collection_v1", "campaign_id": campaign, "run_id": run_id,
                      "case": case, "independent_run": independent, "command": command, "artifact_sha256": artifact,
                      "tool_sha256": tool_hashes, "cases_sha256": cases_hash, "dataset_sha256": dataset_hash,
                      "host": {"platform": platform.platform(), "python": sys.version, "python_executable": sys.executable,
                               "python_executable_sha256": file_digest(sys.executable)},
                      "environment": {key: os.environ[key] for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS") if key in os.environ},
                      "promotion_eligible": False, "validated": False}
            def save():
                prefix.with_suffix(".record.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
            try:
                demand(file_digest(executable) == artifact, "executable changed before child launch")
                demand(file_digest(args.cases) == cases_hash, "case manifest changed before child launch")
                child = subprocess.run(command, text=True, encoding="utf-8", capture_output=True, timeout=args.timeout, check=False)
                prefix.with_suffix(".stdout.json").write_text(child.stdout, encoding="utf-8")
                prefix.with_suffix(".stderr.txt").write_text(child.stderr, encoding="utf-8")
                record["returncode"] = child.returncode
                demand(file_digest(executable) == artifact, "executable changed during child run")
                demand(file_digest(args.cases) == cases_hash, "case manifest changed during child run")
                demand(child.returncode == 0, f"child returned {child.returncode}; unsupported is diagnostic, not a passing observation")
                capture = strict_json(child.stdout)
                identity = validate_capture(capture, case, run_id, args.repeats, args.warmups)
                if build is None:
                    build = identity["build_manifest"]
                demand(identity["build_manifest"] == build, "build provenance changed within campaign")
                output_key = tuple(case[key] for key in ("m", "n", "k", "modulus"))
                demand(output_key not in outputs or outputs[output_key] == capture["output_checksum"],
                       "matched output checksum changed across processes/modes/backends")
                outputs[output_key] = capture["output_checksum"]
                key = (case["backend"], case["device"])
                demand(key not in cohort or cohort[key] == identity, "device/runtime/provenance identity changed within campaign")
                cohort[key] = identity
                record.update(validated=True, identity=identity, provenance_complete=manifest_provenance_complete(build))
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                if isinstance(error, subprocess.TimeoutExpired):
                    prefix.with_suffix(".stdout.json").write_bytes(error.stdout or b"")
                    prefix.with_suffix(".stderr.txt").write_bytes(error.stderr or b"")
                record["error"] = str(error)
                save()
                raise
            save()
    return campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        print(collect(args))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
