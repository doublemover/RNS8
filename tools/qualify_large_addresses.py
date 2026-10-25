"""Run the opt-in physical address fixture in bounded independent processes.

No resume, installation, cache admission or GPU routing changes. A skipped test
is an unrun gate (exit 77), even when Catch itself returns success.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import os
import statistics
import subprocess
import time
import uuid

from build_manifest import manifest_provenance_complete

CASE = "budgeted finite controls read real device addresses beyond signed 32-bit range"
PREFIX = "RNS8_LARGE_ADDRESS_RECEIPT "


def strict_json(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def validate_receipt(data):
    if not isinstance(data, dict) or data.get("schema") != "rns8_large_address_qualification_v1":
        raise ValueError("missing physical large-address receipt")
    required = {"backend": "hip-direct", "semantic_contract": "finite_ring_mod253_centered_i8",
        "m": 32769, "n": 1, "k": 65536, "ldc": 3, "window_rows": 128, "prefix_count": 1,
        "a_allocation_bytes": 2147549184, "b_allocation_bytes": 65536, "c_allocation_bytes": 98307,
        "first_a_offset_above_int32": 2147483648, "sparse_existing_u64_control_modulus": 256,
        "sparse_existing_u64_high_offset": 2147483649, "compute_host_pointer_rebasing": False,
        "all_rows_and_padding_exact": True, "all_owned_device_bytes_retired": True,
        "device_allocation_calls": 3, "device_free_calls": 3, "matched_m": 32767,
        "matched_first_row": 32639, "matched_warmups": 2, "matched_repeats": 9,
        "matched_timing_hooks_enabled": False, "promotion_eligible": False,
        "timing_source": "host_monotonic_completed_backend_window"}
    for key, value in required.items():
        if type(data.get(key)) is not type(value) or data[key] != value:
            raise ValueError(f"incorrect physical fixture field: {key}")
    payload = sum(data[k] for k in ("a_allocation_bytes", "b_allocation_bytes", "c_allocation_bytes"))
    for key in ("device_budget_bytes", "available_device_bytes_before", "total_device_bytes", "reserve_bytes", "observed_requested_device_peak_delta"):
        if type(data.get(key)) is not int or data[key] < 0:
            raise ValueError(f"invalid byte observation: {key}")
    if (data["observed_requested_device_peak_delta"] != payload or data["device_budget_bytes"] < payload or
            data["available_device_bytes_before"] < payload + data["reserve_bytes"] or
            data["total_device_bytes"] < data["available_device_bytes_before"] or
            data["reserve_bytes"] != 512 * 1024 * 1024):
        raise ValueError("device budget, available memory and observed peak do not reconcile")
    if data.get("narrow_rejected_before_write") != [32768, 32769]:
        raise ValueError("both conservative narrow boundaries must preserve output")
    manifest = data.get("build_manifest")
    if not manifest_provenance_complete(manifest, "hip-direct") or data.get("compiled_source_commit") != manifest.get("source_commit"):
        raise ValueError("missing clean source-bound compiled manifest")
    options = dict(line.split("=", 1) for line in manifest["configuration"].splitlines() if "=" in line)
    if options.get("hip", "OFF").upper() not in ("ON", "TRUE", "1", "YES"):
        raise ValueError("physical qualification requires a HIP-enabled artifact")
    if not isinstance(data.get("target"), str) or not data["target"].startswith("gfx"):
        raise ValueError("unknown actual GPU target")
    if not isinstance(data.get("device"), str) or not data["device"].strip():
        raise ValueError("missing actual GPU name")
    checksum = data.get("centered_output_weighted_checksum")
    if not isinstance(checksum, str) or not checksum or str(int(checksum)) != checksum:
        raise ValueError("missing canonical exact checksum")
    for key in ("hip_runtime_version", "hip_driver_version"):
        if type(data.get(key)) is not int or data[key] <= 0:
            raise ValueError(f"missing actual {key}")
    for key in ("matched_i32_host_ns", "matched_u64_host_ns"):
        values = data.get(key)
        if not isinstance(values, list) or len(values) != 9 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f"invalid matched completed-call samples: {key}")
    tick = data.get("nominal_clock_tick_ns")
    if type(tick) not in (int, float) or not math.isfinite(tick) or tick <= 0:
        raise ValueError("missing nominal clock period")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-budget-bytes", type=int, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.runs <= 10 or not 0 < args.device_budget_bytes <= 2**64-1 or not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 3600:
        parser.error("need 1..10 processes, a positive u64 byte budget and a bounded positive timeout")
    binary = args.binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    pinned = digest(binary)
    result = dict(schema="rns8_large_address_collection_v1", binary=str(binary), executable_sha256=pinned,
        case=CASE, requested_processes=args.runs, records=[], campaign_resumed=False, promotion_eligible=False)
    def save():
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    def fail(record, error):
        record["error"] = str(error)
        result["error"] = str(error)
        result["status"] = "failed"
        save()
        return 1
    result["status"] = "pending"
    save()
    target_identity = None
    for index in range(args.runs):
        if digest(binary) != pinned:
            return fail({}, "pinned test binary changed between independent processes")
        run_id = str(uuid.uuid4())
        env = dict(os.environ, RNS8_LARGE_ADDRESS_BUDGET_BYTES=str(args.device_budget_bytes))
        command = [str(binary), CASE]
        log = args.output / f"run-{index:02d}.log"
        started = time.time()
        record = dict(run_id=run_id, command=command, overrides={"RNS8_LARGE_ADDRESS_BUDGET_BYTES": str(args.device_budget_bytes)}, log=log.name)
        result["records"].append(record)
        save()
        error = None
        with log.open("w", encoding="utf-8") as output:
            try:
                completed = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", timeout=args.timeout_seconds)
                record["exit_code"] = completed.returncode
            except (subprocess.TimeoutExpired, OSError) as caught:
                error = caught
        record.update(wall_seconds=time.time()-started, log_sha256=digest(log), executable_unchanged=digest(binary)==pinned)
        if error is not None:
            return fail(record, error)
        if completed.returncode or not record["executable_unchanged"]:
            return fail(record, "nonzero test exit or changed executable")
        if log.stat().st_size > 16*1024*1024:
            return fail(record, "unbounded qualification output")
        lines = [line[len(PREFIX):] for line in log.read_text(encoding="utf-8").splitlines() if line.startswith(PREFIX)]
        if not lines:
            result["status"]="unrun_or_skipped"
            save()
            return 77
        if len(lines) != 1:
            return fail(record, "ambiguous physical qualification receipt")
        try:
            report = strict_json(lines[0]);validate_receipt(report)
        except (ValueError, TypeError, KeyError) as caught:
            return fail(record, caught)
        identity = (report["target"], report["device"], report["hip_runtime_version"], report["hip_driver_version"],
            report["build_manifest"]["source_digest"], report["build_manifest"]["dependency_digest"],
            report["build_manifest"]["configuration"], report["centered_output_weighted_checksum"])
        if target_identity is not None and identity != target_identity:
            return fail(record, "mixed physical targets, builds or numerical witnesses")
        target_identity=identity;record["receipt"]=report
        save()
        print(f"large-address process {index+1}/{args.runs}: exact", flush=True)
    result["status"]="all_exact"
    result["matched_process_means_ns"]={key:[statistics.fmean(r["receipt"][key]) for r in result["records"]]
        for key in ("matched_i32_host_ns", "matched_u64_host_ns")}
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
