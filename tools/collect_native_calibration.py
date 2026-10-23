"""Collect independent native conversion/WMMA pipeline processes, without promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import uuid
from modulus_set_search import DEFAULT_MODULI


def validate_case(case):
    keys = ("candidate", "m", "n", "k", "prefix", "group", "unsigned_inputs")
    if not isinstance(case, dict) or set(case) != set(keys):
        raise ValueError("native case needs exactly candidate,m,n,k,prefix,group,unsigned_inputs")
    limits = {"m": (1, 1024), "n": (1, 1024), "k": (1, 65537), "prefix": (1, 20)}
    if any(type(case[key]) is not int or not low <= case[key] <= high for key, (low, high) in limits.items()):
        raise ValueError("invalid native calibration shape/prefix")
    native = case["candidate"] in ("wmma", "queue", "split_k")
    narrow = case["candidate"] in ("narrow_wmma", "narrow_scalar", "narrow_vector", "narrow_rns")
    if ((not native and not narrow and case["candidate"] not in ("pack_pair", "packed_gemm", "pack_gemm", "epilogue_direct", "epilogue_rocwmma", "epilogue_unfused"))
            or type(case["group"]) is not int or case["group"] not in (1, 2, 4)
            or (not native and case["group"] != 1) or type(case["unsigned_inputs"]) is not bool
            or case["m"] * case["n"] * case["k"] > 16777216):
        raise ValueError("invalid native calibration candidate/group/reference budget")
    if narrow and (case["prefix"] != 1 or case["k"] > 32768
                   or (case["candidate"] == "narrow_scalar" and case["unsigned_inputs"])):
        raise ValueError("narrow cases require prefix=1, K<=32768; scalar is signed only")
    return case


def epilogue_contract(case):
    if not case["candidate"].startswith("epilogue_"):
        return {}
    size = case["m"] * case["n"] * case["prefix"]
    return {"input_domain": "resident_centered_residues",
            "output_contract": "integer_alpha_ab_plus_beta_d_residues_no_integer_export",
            "expression": {"alpha": (1 << 63) - 1 if case["unsigned_inputs"] else -(1 << 63),
                           "beta": (1 << 63) - 1, "coefficients": "original_integers",
                           "d_dataset": "seed_3135_after_a_b"},
            "reserved_intermediate_bytes": size,
            "materialized_intermediate_bytes": size if case["candidate"] == "epilogue_unfused" else 0}


def narrow_contract(case):
    if not case["candidate"].startswith("narrow_"):
        return {}
    bound = case["k"] * (255 ** 2 if case["unsigned_inputs"] else 128 ** 2)
    prefix, product = 0, 1
    if case["candidate"] == "narrow_rns":
        while product <= bound * (1 if case["unsigned_inputs"] else 2):
            product *= DEFAULT_MODULI[prefix]
            prefix += 1
    return {"dataset": "byte_domain_mt19937_64_seed3135_forced_maxima_v1",
            "input_domain": "resident_native_64_byte_values",
            "output_contract": "bounded_exact_native_64",
            "abs_partial_sum_bound": bound,
            "selected_rns_prefix": prefix,
            "residue_storage_bytes": (case["m"] * case["k"] + case["k"] * case["n"] + case["m"] * case["n"]) * prefix,
            "native_input_storage_bytes": (case["m"] * case["k"] + case["k"] * case["n"]) * 8,
            "native_output_bytes": case["m"] * case["n"] * 8,
            "reserved_status_bytes": 4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of explicit case objects")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--order-seed", type=int, default=3135)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if (not isinstance(cases, list) or not 1 <= len(cases) <= 64 or not 1 <= args.runs <= 100
            or not 1 <= args.repeats <= 10000 or not 1 <= args.warmups <= 1000 or args.timeout <= 0):
        parser.error("invalid case/run/repeat/timeout domain")
    cases = [validate_case(c) for c in cases]
    exe = args.exe.resolve(strict=True)
    artifact = hashlib.sha256(exe.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    campaign = str(uuid.uuid4())
    jobs = [(c, independent) for independent in range(args.runs) for c in cases]
    random.Random(args.order_seed).shuffle(jobs)
    for ordinal, (case, independent) in enumerate(jobs):
        if hashlib.sha256(exe.read_bytes()).hexdigest() != artifact:
            raise RuntimeError("native executable changed during collection")
        run_id = str(uuid.uuid4())
        command = [str(exe), case["candidate"], *(str(case[k]) for k in ("m", "n", "k", "prefix", "group")),
                   str(int(case["unsigned_inputs"])), str(args.repeats), str(args.warmups), run_id]
        record = {"schema": "rns8_native_collection_v1", "campaign_id": campaign, "run_id": run_id,
                  "artifact_sha256": artifact, "command": command, "case": case,
                  "started_unix_ns": time.time_ns(), "ordinal": ordinal, "independent_repeat": independent,
                  "order_seed": args.order_seed, "promotion_eligible": False}
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
            record.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired:
            record.update(exit_code=None, reason="timeout", timeout_seconds=args.timeout)
        with (args.output / (run_id + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        if record["exit_code"] != 0:
            raise RuntimeError(f"native microcase failed; retained {run_id}")
        capture = json.loads(record["stdout"])
        if (capture.get("schema") != "rns8_native_microcase_v1" or capture.get("run_id") != run_id
                or any(capture.get(key) != value for key, value in case.items())
                or any(capture.get(key) != value for key, value in epilogue_contract(case).items())
                or any(capture.get(key) != value for key, value in narrow_contract(case).items())
                or capture.get("repeats") != args.repeats or capture.get("warmups") != args.warmups
                or len(capture.get("samples", [])) != args.repeats
                or any(sample.get("exact_reference_passed") is not True for sample in capture["samples"])):
            raise RuntimeError(f"native microcase contract/reference mismatch; retained {run_id}")
        if hashlib.sha256(exe.read_bytes()).hexdigest() != artifact:
            raise RuntimeError("native executable changed while executing; capture is ineligible")
    print(f"Collected {len(jobs)} independent native microcase records; qualification remains pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
