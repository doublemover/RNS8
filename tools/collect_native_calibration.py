"""Collect independent native conversion/WMMA pipeline processes, without promotion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
import uuid
from build_manifest import manifest_provenance_complete
from modulus_set_search import DEFAULT_MODULI


CANDIDATES = ("pack_pair", "packed_gemm", "pack_gemm", "wmma", "grid_prepared", "queue", "queue_prepared", "split_k",
              "adaptive_grid", "adaptive_compact",
              "complex_3m", "complex_4m",
              "epilogue_direct", "epilogue_rocwmma", "epilogue_unfused",
              "narrow_wmma", "narrow_scalar", "narrow_vector", "narrow_rns")
QUEUED = ("queue", "queue_prepared", "split_k")
ADAPTIVE = ("adaptive_grid", "adaptive_compact")


def strict_json(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def validate_case(case):
    """Return a new canonical case; only queue controls may be omitted."""
    keys = {"candidate", "m", "n", "k", "prefix", "group", "unsigned_inputs"}
    if not isinstance(case, dict) or not keys <= set(case) or set(case) - keys - {"workers", "bound_method"}:
        raise ValueError("native case needs candidate,m,n,k,prefix,group,unsigned_inputs; workers for queues/compact; adaptive requires bound_method")
    limits = {"m": (1, 1024), "n": (1, 1024), "k": (1, 65537), "prefix": (1, 20)}
    if any(type(case[key]) is not int or not low <= case[key] <= high for key, (low, high) in limits.items()):
        raise ValueError("invalid native calibration shape/prefix")
    native = case["candidate"] in ("wmma", "grid_prepared", *QUEUED, *ADAPTIVE)
    narrow = case["candidate"] in ("narrow_wmma", "narrow_scalar", "narrow_vector", "narrow_rns")
    if (case["candidate"] not in CANDIDATES
            or type(case["group"]) is not int or case["group"] not in (1, 2, 4)
            or (not native and case["group"] != 1) or type(case["unsigned_inputs"]) is not bool
            or case["m"] * case["n"] * case["k"] > 1073741824):
        raise ValueError("invalid native calibration candidate/group/reference budget")
    if narrow and (case["prefix"] != 1 or case["k"] > 32768
                   or (case["candidate"] == "narrow_scalar" and case["unsigned_inputs"])):
        raise ValueError("narrow cases require prefix=1, K<=32768; scalar is signed only")
    if case["candidate"].startswith("complex_") and (case["prefix"] != 20 or case["unsigned_inputs"]):
        raise ValueError("complex cases require signed inputs, prefix20 and group1")
    if case["candidate"] in ADAPTIVE:
        if case["prefix"] != 20 or case.get("bound_method") != "axis_l1":
            raise ValueError("adaptive cases require prefix20 and explicit bound_method=axis_l1")
    elif "bound_method" in case:
        raise ValueError("bound_method applies only to adaptive cases")
    normalized = dict(case)
    if case["candidate"] == "adaptive_compact":
        workers = case.get("workers")
        if type(workers) is not int or not 1 <= workers <= 65535:
            raise ValueError("adaptive_compact requires explicit integer workers1..65535")
    elif case["candidate"] in QUEUED:
        workers = case.get("workers", 32)
        if type(workers) is not int or not 1 <= workers <= 1024:
            raise ValueError("queue/queue_prepared/split_k workers must be an integer in 1..1024")
        normalized["workers"] = workers
    elif "workers" in case:
        raise ValueError("workers applies only to queue/queue_prepared/split_k/adaptive_compact candidates")
    return normalized


def scheduling_contract(case):
    queued = case["candidate"] in QUEUED
    splits = min(case["k"], 3) if case["candidate"] == "split_k" else 1
    descriptors = (((case["m"] + 15) // 16) * ((case["n"] + 15) // 16)
                   * ((case["prefix"] + case["group"] - 1) // case["group"]) * splits) if queued else 0
    compact = case["candidate"] == "adaptive_compact"
    requested = case["workers"] if compact else case.get("workers", 32) if queued else 0
    return {"splits": splits, "requested_workers": requested,
            "effective_workers": requested if compact else min(requested, descriptors), "queue_descriptors": descriptors,
            "queue_kernel_selection": (f"homogeneous_group_{case['group']}_"
                                       + ("unsigned" if case["unsigned_inputs"] else "signed"))
                                      if queued else "not_applicable"}


def adaptive_contract(case):
    if case["candidate"] not in ADAPTIVE:
        return {}
    compact = case["candidate"] == "adaptive_compact"
    return {"bound_method": "axis_l1", "adaptive_launch": "compact_workers" if compact else "dense_grid",
            "adaptive_dataset": "full_width_native64_mt19937_64_seed3135_forced_extrema_v1",
            "reference_method": "boost_int256_full_product_with_cpp_int_row_edge_audits_and_epilogue",
            "adaptive_dense_group_upper_bound": ((case["m"] + 15) // 16) * ((case["n"] + 15) // 16)
                * ((case["prefix"] + case["group"] - 1) // case["group"]),
            "adaptive_actual_compact_groups": None, "adaptive_compact_count_source": "device_offsets_not_read_back",
            "adaptive_worker_policy": "exact_requested_worker_launch_no_count_clamp" if compact else "dense_grid_control",
            "adaptive_kernel_selection": ("compact_group_" if compact else "grid_group_") + str(case["group"])
                + ("_unsigned_axis_l1" if case["unsigned_inputs"] else "_signed_axis_l1"),
            "adaptive_output_scope": "fresh bounds and per-tile prefix; exact suffix completes every capacity20 output plane"}


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


def positive_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def valid_sustained(value):
    if not isinstance(value, dict) or value.get("final_exact_reference_passed") is not True:
        return False
    calls, elapsed, mean = (value.get(k) for k in ("calls", "elapsed_ns", "mean_call_ns"))
    return (type(calls) is int and 16 <= calls <= 4096
            and all(positive_number(x) for x in (elapsed, mean))
            and (elapsed >= 250000000 or calls == 4096)
            and math.isclose(mean * calls, elapsed, rel_tol=1e-12)
            and type(value.get("scope")) is str and bool(value["scope"]))


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
            "reserved_status_bytes": 4,
            "reference": "exact_int64_all_outputs_with_boost_boundary_columns"}


def capture_contract(case, run_id, repeats, warmups):
    expected = {key: value for key, value in case.items() if key != "workers"}
    expected.update(schema="rns8_native_microcase_v2", run_id=run_id, repeats=repeats,
                    warmups=warmups, seed=3135, promotion_eligible=False, isa_verified=False,
                    actual_hbm_bytes=None, timing_resolution_ns=None)
    expected.update(scheduling_contract(case))
    expected.update(adaptive_contract(case))
    expected.update(narrow_contract(case))
    if case["candidate"].startswith("complex_"):
        m, n, k, prefix = (case[key] for key in ("m", "n", "k", "prefix"))
        three = case["candidate"] == "complex_3m"
        expected.update(dataset="split_complex_full_signed64_mt19937_3135_extrema_v1",
                        input_domain="resident_split_complex_centered_residues",
                        output_contract="paired_exact_signed_residues_no_integer_export",
                        real_products=3 if three else 4,
                        scratch_peak_bytes=((m*k+k*n)+5*m*n)*prefix if three else 6*m*n*prefix,
                        native_input_storage_bytes=2*(m*k+k*n)*8,
                        residue_input_storage_bytes=2*(m*k+k*n)*prefix,
                        residue_output_bytes=2*m*n*prefix, component_bound=str(k*(1 << 127)),
                        reference_method="boost_int256_four_products_all_outputs_with_cpp_int_boundary_columns",
                        input_preparation_scope="native upload, four input packs and common allocations; outside resident repeated calls")
        return expected
    if case["candidate"] in ("queue_prepared", "grid_prepared"):
        expected["queue_preparation_scope"] = "one host preparation including allocation, capability check and descriptor upload; excluded from steady calls"
    if not case["candidate"].startswith("narrow_"):
        epilogue = case["candidate"].startswith("epilogue_")
        expected.update(
            input_domain="resident_centered_residues" if epilogue or case["candidate"] == "packed_gemm" else "resident_native_64",
            output_contract="operand_residue_planes" if case["candidate"] == "pack_pair" else "product_residues_default_basis_no_integer_export",
            expression=None, reserved_intermediate_bytes=0, materialized_intermediate_bytes=0,
            native_input_storage_bytes=(case["m"] * case["k"] + case["k"] * case["n"]) * 8,
            residue_input_storage_bytes=(case["m"] * case["k"] + case["k"] * case["n"]
                                         + (case["m"] * case["n"] if epilogue else 0)) * case["prefix"],
            residue_output_bytes=case["m"] * case["n"] * case["prefix"])
        expected.update(epilogue_contract(case))
    return expected


def same_value(actual, expected):
    """JSON bools and floats cannot impersonate integer controls or bounds."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_value(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_value(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_capture(capture, case, run_id, repeats, warmups):
    if not isinstance(capture, dict):
        raise ValueError("capture must be a JSON object")
    for key, value in capture_contract(case, run_id, repeats, warmups).items():
        if key not in capture or not same_value(capture[key], value):
            raise ValueError(f"metadata mismatch: {key}")
    if not valid_sustained(capture.get("sustained")):
        raise ValueError("invalid sustained verification/timing")
    if case["candidate"].startswith("complex_") and not positive_number(capture.get("input_prepare_ns")):
        raise ValueError("invalid complex input preparation timing")
    if case["candidate"] in ("queue_prepared", "grid_prepared"):
        if not positive_number(capture.get("queue_prepare_ns")):
            raise ValueError("invalid prepared queue setup timing")
        size = capture.get("queue_descriptor_storage_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError("invalid prepared queue descriptor storage")
    samples = capture.get("samples")
    if not isinstance(samples, list) or len(samples) != repeats:
        raise ValueError("invalid sample count")
    for sample in samples:
        if (not isinstance(sample, dict) or sample.get("exact_reference_passed") is not True
                or sample.get("uninstrumented_exact_reference_passed") is not True
                or not positive_number(sample.get("host_call_ns"))
                or not positive_number(sample.get("uninstrumented_host_call_ns"))
                or not isinstance(sample.get("events"), list)):
            raise ValueError("invalid sample verification/timing")
        for event in sample["events"]:
            if (not isinstance(event, dict) or type(event.get("label")) is not str or not event["label"]
                    or "elapsed_ns" not in event
                    or (event["elapsed_ns"] is not None and not positive_number(event["elapsed_ns"]))):
                raise ValueError("invalid HIP event timing")
    for key in ("device", "target", "timing_scope", "cache_state"):
        if type(capture.get(key)) is not str or not capture[key]:
            raise ValueError(f"missing identity/scope: {key}")
    if type(capture.get("runtime_version")) is not int or capture["runtime_version"] <= 0:
        raise ValueError("invalid runtime identity")
    manifest = capture.get("build_manifest")
    # Diagnostic runs may use dirty sources or unavailable dependency identity.
    # Validate their shape without misrepresenting them as release-qualified.
    if not isinstance(manifest, dict) or type(manifest.get("source_dirty")) is not bool:
        raise ValueError("invalid build provenance")
    qualified_shape = dict(manifest, source_dirty=False)
    if manifest.get("dependency_digest") == "unavailable":
        qualified_shape["dependency_digest"] = "0" * 64
    # Experimental Garner is valid diagnostic provenance, but the original
    # manifest must still fail the separate release-provenance check below.
    configuration = manifest.get("configuration")
    if isinstance(configuration, str):
        lines = configuration.splitlines()
        qualified_shape["configuration"] = "\n".join(
            "garner_table=OFF" if line.partition("=")[0] == "garner_table" else line
            for line in lines)
    backend = "rocwmma" if case["candidate"] == "epilogue_rocwmma" else None
    if not manifest_provenance_complete(qualified_shape, backend):
        raise ValueError("invalid build provenance")
    return {key: capture[key] for key in ("build_manifest", "device", "target", "runtime_version")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True,
                        help="JSON cases; queue workers1..1024 default32; adaptive prefix20/bound_method=axis_l1; compact requires workers1..65535")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--order-seed", type=int, default=3135)
    args = parser.parse_args()
    try:
        cases = strict_json(args.cases.read_text(encoding="utf-8"))
        if (not isinstance(cases, list) or not 1 <= len(cases) <= 64 or not 1 <= args.runs <= 100
                or not 1 <= args.repeats <= 10000 or not 1 <= args.warmups <= 1000 or args.timeout <= 0):
            raise ValueError("invalid case/run/repeat/timeout domain")
        cases = [validate_case(c) for c in cases]
        if len({json.dumps(case, sort_keys=True) for case in cases}) != len(cases):
            raise ValueError("duplicate normalized cases; use --runs for independent processes")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    exe = args.exe.resolve(strict=True)
    artifact = hashlib.sha256(exe.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    collector_sources = [Path(__file__).resolve(), Path(__file__).with_name("build_manifest.py").resolve(),
                         Path(__file__).with_name("modulus_set_search.py").resolve()]
    collector_identity = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in collector_sources}
    campaign = str(uuid.uuid4())
    jobs = [(c, independent) for independent in range(args.runs) for c in cases]
    random.Random(args.order_seed).shuffle(jobs)
    campaign_identity = None
    for ordinal, (case, independent) in enumerate(jobs):
        if hashlib.sha256(exe.read_bytes()).hexdigest() != artifact:
            raise RuntimeError("native executable changed during collection")
        run_id = str(uuid.uuid4())
        command = [str(exe), case["candidate"], *(str(case[k]) for k in ("m", "n", "k", "prefix", "group")),
                   str(int(case["unsigned_inputs"])), str(args.repeats), str(args.warmups), run_id]
        if case["candidate"] in (*QUEUED, "adaptive_compact"):
            command.append(str(case["workers"]))
        if case["candidate"] in ADAPTIVE:
            command.append(case["bound_method"])
        record = {"schema": "rns8_native_collection_v2", "campaign_id": campaign, "run_id": run_id,
                  "artifact_sha256": artifact, "command": command, "case": case,
                  "collector_sha256": collector_identity["collect_native_calibration.py"],
                  "collector_sources_sha256": collector_identity,
                  "started_unix_ns": time.time_ns(), "ordinal": ordinal, "independent_repeat": independent,
                  "order_seed": args.order_seed, "promotion_eligible": False}
        failure = None
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
            record.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if hashlib.sha256(exe.read_bytes()).hexdigest() != artifact:
                raise ValueError("native executable changed while executing; capture is ineligible")
            if result.returncode != 0:
                raise ValueError("native microcase failed")
            capture = strict_json(result.stdout)
            identity = validate_capture(capture, case, run_id, args.repeats, args.warmups)
            if campaign_identity is not None and not same_value(identity, campaign_identity):
                raise ValueError("build/device/runtime identity changed during collection")
            campaign_identity = identity
            record["comparison_identity"] = {"artifact_sha256": artifact, **identity,
                                             "case": case, **scheduling_contract(case),
                                             **adaptive_contract(case),
                                             "input_domain": capture["input_domain"],
                                             "output_contract": capture["output_contract"],
                                             "seed": capture["seed"], "repeats": args.repeats,
                                             "warmups": args.warmups, "timing_scope": capture["timing_scope"],
                                             "cache_state": capture["cache_state"]}
            record["provenance_complete"] = manifest_provenance_complete(
                capture["build_manifest"], "rocwmma" if case["candidate"] == "epilogue_rocwmma" else None)
            record["validated"] = True
        except subprocess.TimeoutExpired as error:
            def timeout_text(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            record.update(exit_code=None, reason="timeout", timeout_seconds=args.timeout,
                          stdout=timeout_text(error.stdout), stderr=timeout_text(error.stderr))
            failure = "native microcase timeout"
        except (ValueError, OSError) as error:
            record.setdefault("exit_code", None)
            record.update(reason="contract/reference mismatch", validation_error=str(error))
            failure = f"native microcase contract/reference mismatch: {error}"
        record.setdefault("validated", False)
        with (args.output / (run_id + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        if failure:
            raise RuntimeError(f"{failure}; retained {run_id}")
    print(f"Collected {len(jobs)} independent native microcase records; qualification remains pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
