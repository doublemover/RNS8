"""Matched public narrow/RNS/UInt192 routes; no measured-cache promotion."""

from call_observation_contract import observation_boundary, validate_completed_call_samples
from native_diagnostic_contract import COPY_COUNTERS, validate_native_observation
import argparse
import json
from pathlib import Path
import math
import statistics
import uuid
from collect_int4_calibration import collection_plan as shared_plan
from json_evidence import parse_evidence_json as strict_json
from collect_native_calibration import DEFAULT_MODULI
from benchmark_sweep_lib.execution import run_command
from build_manifest import manifest_provenance_complete

MODES = (
    "scalar-staged",
    "scalar-fused",
    "scalar-split-k-fused",
    "scalar-split-k-resident-fused",
    "wmma-staged",
    "wmma-fused",
    "rns-public",
    "rns-parallel",
    "vector-width8",
    "vector-exact-width",
    "adaptive-fresh",
    "adaptive-cached",
    "adaptive-resident-fresh",
    "adaptive-resident-cached",
    "wmma-resident-fused",
    "scalar-resident-fused",
    "rns-resident-public",
)
RESIDENT_OUTPUT_MODES = {
    "scalar-split-k-resident-fused",
    "adaptive-resident-fresh",
    "adaptive-resident-cached",
    "wmma-resident-fused",
    "scalar-resident-fused",
    "rns-resident-public",
}


def validate_case(c):
    if (
        not isinstance(c, dict)
        or set(c)
        != {
            "schema",
            "m",
            "n",
            "k",
            "prefix",
            "unsigned_inputs",
            "mode",
            "residency",
            "repeats",
            "warmups",
            "seed",
        }
        or c["schema"] != "narrow_public_case_v1"
    ):
        raise ValueError("case ABI")
    for key, low, high in [
        ("m", 1, 512),
        ("n", 1, 512),
        ("k", 1, 131073),
        ("prefix", 0, 20),
        ("repeats", 3, 100),
        ("warmups", 1, 100),
        ("seed", 0, 2**32 - 1),
    ]:
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError(key)
    if (
        type(c["unsigned_inputs"]) is not bool
        or c["mode"] not in MODES
        or c["residency"] not in ("cold", "resident", "repack", "repack-a", "repack-b")
    ):
        raise ValueError("mode/domain")
    if (c["mode"].startswith("adaptive-") or c["mode"] in RESIDENT_OUTPUT_MODES) and c[
        "residency"
    ] != "resident":
        raise ValueError("adaptive summary comparison requires retained resident inputs")
    if (
        (c["mode"].startswith("scalar-") and c["unsigned_inputs"])
        or (c["mode"] == "rns-parallel" and c["n"] > 8)
        or (c["mode"] == "vector-exact-width" and not (2 <= c["n"] <= 8 and c["k"] >= 512))
    ):
        raise ValueError("concrete recipe")
    if c["m"] * c["n"] * c["k"] > 134217728:
        raise ValueError("complete oracle work")
    if c["prefix"] and math.prod(DEFAULT_MODULI[: c["prefix"]]) <= c["k"] * 49 * (
        1 if c["unsigned_inputs"] else 2
    ):
        raise ValueError("strict prefix capacity")
    return c


def selected_prefix(c):
    if c["prefix"]:
        return c["prefix"]
    product = 1
    for index, modulus in enumerate(DEFAULT_MODULI, 1):
        product *= modulus
        if product > c["k"] * 49 * (1 if c["unsigned_inputs"] else 2):
            return index
    raise ValueError("capacity")


def narrow_product_bytes(case):
    if case["mode"].startswith("scalar-split-k-"):
        return case["m"] * case["n"] * ((case["k"] + 4095) // 4096) * 8
    return 0 if case["mode"].endswith("-fused") else case["m"] * case["n"] * 8


def validate_capture(c, entry):
    observation_boundary(c)
    case = validate_case(entry["case"])
    mode = case["mode"]
    narrow = mode.startswith(("scalar-", "wmma-"))
    vector = mode.startswith("vector-")
    resident_output = mode in RESIDENT_OUTPUT_MODES
    scratch = (
        ((32 if mode.startswith("wmma-") else 16) + narrow_product_bytes(case)) if narrow else 0
    )
    fixed = dict(
        schema="rns8_public_narrow_microcase_v2",
        case=case,
        run_id=entry["run_id"],
        dataset="affine_native_small_v1",
        selected_prefix=selected_prefix(case),
        narrow_producer=mode if narrow else "not_applicable",
        narrow_input_admission_policy="generation_bound_host_pack_certificate"
        if narrow
        else "not_applicable",
        narrow_product_storage="not_applicable"
        if not narrow
        else "workspace_retained_split_k_partials"
        if mode.startswith("scalar-split-k-")
        else "fused_no_native_product"
        if mode.endswith("-fused")
        else "workspace_retained_native_product",
        known_narrow_scratch_bytes=scratch,
        output_boundary="complete_public_product_resident_rns"
        if resident_output
        else "complete_public_product_and_padded_three_limb_integer_adapter",
        vector_export_adapter="public_native64_export_then_host_sign_or_zero_extension"
        if vector
        else "not_applicable",
        host_instrumentation_state="hooks_disabled",
        exact_reference_passed=True,
        oracle_inside_production_interval=False,
        all_residue_planes_audited=not vector,
        residue_audit_scope="not_applicable_native_control"
        if vector
        else "post_timing_public_product",
        promotion_eligible=False,
        native_useful_products=case["m"] * case["n"] * case["k"],
    )
    for key, value in fixed.items():
        if type(c.get(key)) is not type(value) or c[key] != value:
            raise ValueError("capture " + key)
    if not manifest_provenance_complete(c.get("build_manifest")) or not c.get("execution_id"):
        raise ValueError("compiled/process provenance")
    artifact = c.get("artifact_capture") or {}
    if (
        artifact.get("unchanged_during_run") is not True
        or len(artifact.get("executable_sha256", "")) != 64
    ):
        raise ValueError("artifact identity")
    validate_completed_call_samples(c, case["repeats"], resident_output=resident_output)
    kernel = c.get("ordinary_plan_kernel", "")
    if mode == "rns-parallel" and kernel != (
        "direct_hip_selected_prefix_rns_gemv_n1_v1"
        if case["n"] == 1
        else "direct_hip_selected_prefix_rns_gemv_small_n_v1"
    ):
        raise ValueError("public parallel route identity")
    if mode in ("rns-public", "rns-resident-public") and kernel != (
        "direct_hip_persistent_small_rns_gemm_v1"
        if case["m"] * case["n"] <= 64
        else "direct_hip_selected_prefix_grouped_rns_gemm_v1"
    ):
        raise ValueError("public default route identity")
    if mode == "vector-exact-width" and f"_n{case['n']}_exact_width_" not in kernel:
        raise ValueError("exact-width vector route identity")
    validate_narrow_summary_capture(c, case)
    validate_narrow_native_observation(c, case)
    if mode.startswith("adaptive-"):
        validate_adaptive_summary_capture(c, case)
    if resident_output:
        for key, value in dict(
            timed_host_export=False,
            oracle_export_outside_production_interval=True,
            exact_proof_policy="scalar_first",
        ).items():
            if type(c.get(key)) is not type(value) or c[key] != value:
                raise ValueError("resident output identity " + key)
    return c


def validate_narrow_native_observation(c, case):
    info = c.get("separate_native_observation", False)
    if not case["mode"].startswith(("scalar-", "wmma-")):
        if info is not None:
            raise ValueError("non-narrow native observation")
        return
    validate_native_observation(
        info, scope="one_separate_complete_public_call", calls_key="public_calls", calls=1
    )
    counters = COPY_COUNTERS
    labels = info["event_labels"]
    wmma = case["mode"].startswith("wmma-")
    fused = case["mode"].endswith("-fused")
    product = (
        ("native_byte_wmma_fused_rns" if fused else "native_byte_wmma_gemm")
        if wmma
        else ("narrow_native_i64_fused_rns_kernel" if fused else "narrow_native_i64_gemm_kernel")
    )
    split_k = case["mode"].startswith("scalar-split-k-")
    if (
        (not split_k and labels.get(product) != 1)
        or (
            split_k
            and (
                labels.get("narrow_native_split_k_partials") != 1
                or labels.get("narrow_native_split_k_encode") != 1
                or labels.get(product, 0)
            )
        )
        or labels.get("narrow_native_encode_rns", 0) != int(not fused)
        or labels.get("native_wmma_prepare_capability", 0)
        != int(wmma and case["residency"] == "cold")
        or any(
            labels.get(key, 0)
            for key in (
                "narrow_native_domain_scan",
                "narrow_native_bound_scan",
                "native_byte_wmma_admission",
            )
        )
    ):
        raise ValueError("narrow actual native stages")
    if case["residency"] == "resident" and case["mode"] in RESIDENT_OUTPUT_MODES:
        if any(info[key] for key in counters):
            raise ValueError("resident narrow call allocated or copied data")


def validate_narrow_summary_capture(c, case):
    narrow = case["mode"].startswith(("scalar-", "wmma-"))
    cold = case["residency"] == "cold"
    scope = (
        "not_applicable"
        if not narrow
        else "cold_workspace_not_observed"
        if cold
        else "completed_retained_workspace_snapshots_outside_timing"
    )
    if c.get("narrow_summary_scope") != scope:
        raise ValueError("narrow summary scope")
    if not narrow or cold:
        if (
            c.get("narrow_summary_first", False) is not None
            or c.get("narrow_summary_final", False) is not None
        ):
            raise ValueError("unobserved narrow summary")
        return
    calls = 1 + case["warmups"] + case["repeats"] + sum(b["calls"] for b in c["sustained"])
    repack_a = case["residency"] in ("repack", "repack-a")
    repack_b = case["residency"] in ("repack", "repack-b")
    wmma = int(case["mode"].startswith("wmma-"))
    keys = {
        "retained_scan_bytes",
        "retained_product_bytes",
        "wmma_capability_observed",
        "wmma_capability_supported",
        "a_instance",
        "a_generation",
        "b_instance",
        "b_generation",
        "a_device_scans",
        "b_device_scans",
        "a_reuses",
        "b_reuses",
        "a_host_proofs",
        "b_host_proofs",
    }
    for name, count in (("narrow_summary_first", 1), ("narrow_summary_final", calls)):
        info = c.get(name)
        if (
            not isinstance(info, dict)
            or set(info) != keys
            or any(type(v) is not int or not 0 <= v <= (1 << 64) - 1 for v in info.values())
        ):
            raise ValueError("narrow summary record")
        expected = dict(
            retained_scan_bytes=0,
            retained_product_bytes=narrow_product_bytes(case),
            wmma_capability_observed=wmma,
            wmma_capability_supported=wmma,
            a_generation=1 + count if repack_a else 1,
            b_generation=1 + count if repack_b else 1,
            a_device_scans=0,
            b_device_scans=0,
            a_reuses=0 if repack_a else count - 1,
            b_reuses=0 if repack_b else count - 1,
            a_host_proofs=count if repack_a else 1,
            b_host_proofs=count if repack_b else 1,
        )
        if (
            not info["a_instance"]
            or not info["b_instance"]
            or info["a_instance"] == info["b_instance"]
        ):
            raise ValueError("narrow input instances")
        if any(info[key] != value for key, value in expected.items()):
            raise ValueError("narrow completed proof accounting")
    if any(
        c["narrow_summary_first"][key] != c["narrow_summary_final"][key]
        for key in ("a_instance", "b_instance")
    ):
        raise ValueError("narrow retained input identity")


def validate_adaptive_summary_capture(c, case):
    cached = case["mode"].endswith("-cached")
    expected = dict(
        adaptive_summary_policy="generation_bound" if cached else "fresh_scan",
        adaptive_method="axis_l1",
        adaptive_group=2,
        adaptive_workers=32,
        exact_proof_policy="scalar_first",
    )
    for key, value in expected.items():
        if type(c.get(key)) is not type(value) or c[key] != value:
            raise ValueError("adaptive identity " + key)
    calls = 1 + case["warmups"] + case["repeats"] + sum(b["calls"] for b in c["sustained"])
    size = ((case["m"] + 15) // 16 + (case["n"] + 15) // 16) * 24
    for name, count in [("adaptive_summary_first", 1), ("adaptive_summary_final", calls)]:
        info = c.get(name)
        keys = {
            "retained_bytes",
            "a_instance",
            "a_generation",
            "b_instance",
            "b_generation",
            "a_scans",
            "b_scans",
            "a_reuses",
            "b_reuses",
        }
        if (
            not isinstance(info, dict)
            or set(info) != keys
            or any(type(v) is not int or v < 0 for v in info.values())
        ):
            raise ValueError("adaptive summary receipt")
        if info["retained_bytes"] != (size if cached else 0):
            raise ValueError("adaptive retained bytes")
        for side in ("a", "b"):
            if info[side + "_scans"] != int(cached) or info[side + "_reuses"] != (
                count - 1 if cached else 0
            ):
                raise ValueError("adaptive scans/reuses")
            if cached and (not info[side + "_instance"] or not info[side + "_generation"]):
                raise ValueError("adaptive generation identity")
            if not cached and (info[side + "_instance"] or info[side + "_generation"]):
                raise ValueError("fresh route cannot report cached identity")
    if cached:
        for key in ("a_instance", "a_generation", "b_instance", "b_generation"):
            if c["adaptive_summary_first"][key] != c["adaptive_summary_final"][key]:
                raise ValueError("unchanged generation required")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--cases", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--seed", type=int, default=5353)
    p.add_argument("--timeout-seconds", type=float, default=600)
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    plan = shared_plan(
        a.binary, a.cases, a.output, a.runs, a.seed, str(uuid.uuid4()), validate_case
    )
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "plan.json").exists():
        raise ValueError("immutable collection exists")
    (a.output / "plan.json").write_text(
        json.dumps(
            dict(schema="narrow_public_collection_v1", observations=plan, promotion_eligible=False),
            indent=2,
        )
        + "\n"
    )
    if a.plan_only:
        return 0
    cohorts = {}
    for entry in plan:
        path = Path(entry["output"])
        if path.exists() or not run_command(entry["command"], path, a.timeout_seconds):
            raise RuntimeError("immutable capture exists or subprocess failed: " + str(path))
        c = validate_capture(strict_json(path.read_text()), entry)
        identity = dict(
            hot_loop_serialization=observation_boundary(c),
            case={k: v for k, v in entry["case"].items() if k != "mode"},
            target=c.get("target"),
            device=c.get("device"),
            runtime=c.get("runtime_version"),
            proof_policy=c.get("exact_proof_policy", "default_eager"),
            output_boundary=c["output_boundary"],
            build=c["build_manifest"],
            artifact={
                k: c["artifact_capture"].get(k)
                for k in ("executable_sha256", "rns8_linkage", "shared_library_sha256")
            },
            environment=(c.get("request_identity") or {}).get("environment"),
        )
        cohorts.setdefault(json.dumps(identity, sort_keys=True), []).append(
            dict(
                mode=entry["case"]["mode"],
                execution_id=c["execution_id"],
                independent_run=entry["independent_run"],
                path=str(path),
                median_us=statistics.median(s["total_us"] for s in c["samples"]),
                setup_us=c["setup_us"],
                scratch_bytes=c["known_narrow_scratch_bytes"],
            )
        )
    result = []
    for identity, rows in cohorts.items():
        if len({r["execution_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate process observation")
        medians = {}
        for mode in MODES:
            values = [r["median_us"] for r in rows if r["mode"] == mode]
            if len(values) >= 2:
                medians[mode] = statistics.median(values)
        result.append(
            dict(
                identity=json.loads(identity),
                observations=rows,
                independent_process_medians_us=medians,
                qualification="not_inferred",
                promotion_eligible=False,
            )
        )
    (a.output / "summary.json").write_text(
        json.dumps(
            dict(schema="narrow_public_summary_v1", cohorts=result, promotion_eligible=False),
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
