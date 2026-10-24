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
                 "declared_memory_budget_bytes", "aggregate_memory_envelope_bytes")
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
    data = canonical_words([value for operand in operands for value in operand])
    demand(checksum_bytes(data) == report.get("input_checksum_u64"), "input bytes differ from claimed recipe/checksum")
    operands[0][0] = (operands[0][0] + 1) % 251 if field else operands[0][0] + (-1 if operands[0][0] == (1 << 63) - 1 else 1)
    if gram: operands[1][0] = operands[0][0]
    changed = canonical_words([value for operand in operands for value in operand])
    demand(checksum_bytes(changed) == report.get("changed_input_checksum_u64"), "changed input bytes differ from generation2")
    return {"sha256": hashlib.sha256(data).hexdigest(), "changed_sha256": hashlib.sha256(changed).hexdigest(),
            "bytes": len(data), "serialization": "A then B then optionalT, row-major little-endian signed64"}


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


def validate_report(report: dict, run_id: str, backend: str, device: int) -> dict:
    demand(type(report) is dict and report.get("schema") == SCHEMA, "unexpected workload schema")
    demand(report.get("status") == "passed", "workload did not pass")
    demand(report.get("workload") in {"integer_cross_product", "integer_gram", "gf251_linear_transform",
                                      "resident_integer_chain", "exactness_walkthrough"}, "unknown workload family")
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
    expected_order = [row["name"] for row in variants if row.get("status") == "passed"] if sustained_ms else []
    order = report.get("sustained_order")
    demand(type(order) is list and len(order) == len(expected_order)
           and sorted(order) == sorted(expected_order), "sustained order must cover each measured variant exactly once")
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
            identity = validate_report(report, run_id, backend, device)
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
