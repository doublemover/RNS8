"""Plan matched real grouped/single/RNS task batches; never promote source models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import uuid

from benchmark_request import (
    attach_capture_identity,
    file_digest,
    request_environment_identity,
    request_identity,
)
from build_manifest import manifest_provenance_complete
from call_observation_contract import observation_boundary, validate_completed_call_samples
from collect_int4_calibration import collection_plan as shared_plan
from collect_narrow_public_calibration import selected_prefix
from collect_native_calibration import DEFAULT_MODULI
from device_observation import capture_observation
from execution_observation import validate_execution_request
from json_evidence import parse_evidence_json, same_evidence_value
from native_diagnostic_contract import COPY_COUNTERS, validate_native_observation
from process_progress import capture_succeeded, run_capture

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_LIMIT = 16 * 1024**2

FIELDS = frozenset(
    (
        "schema",
        "m",
        "n",
        "k",
        "tasks",
        "prefix",
        "unsigned_inputs",
        "shared_b",
        "mode",
        "method",
        "residency",
        "output",
        "repeats",
        "warmups",
        "seed",
    )
)


def validate_case(case):
    if (
        not isinstance(case, dict)
        or set(case) != FIELDS
        or case["schema"] != "narrow_group_case_v1"
    ):
        raise ValueError("group case ABI")
    for key, low, high in (
        ("m", 1, 512),
        ("n", 1, 512),
        ("k", 1, 131073),
        ("tasks", 2, 64),
        ("prefix", 0, 20),
        ("repeats", 3, 100),
        ("warmups", 1, 100),
        ("seed", 0, 2**32 - 1),
    ):
        if type(case[key]) is not int or not low <= case[key] <= high:
            raise ValueError("group " + key)
    if (
        type(case["unsigned_inputs"]) is not bool
        or type(case["shared_b"]) is not bool
        or case["mode"] not in ("single", "group", "rns")
        or case["method"] not in ("scalar", "wmma", "split-k")
        or case["residency"] not in ("cold", "resident", "repack", "repack-a", "repack-b")
        or case["output"] not in ("resident", "limbs")
        or (case["unsigned_inputs"] and case["method"] != "wmma")
        or (case["residency"] == "cold" and case["output"] == "resident")
        or case["tasks"] * case["m"] * case["n"] * case["k"] > 134217728
    ):
        raise ValueError("group concrete work/domain")
    # This is the same conservative affine input envelope as the single owner.
    prefix = selected_prefix(case)
    if case["prefix"] and math.prod(DEFAULT_MODULI[:prefix]) <= case["k"] * 49 * (
        1 if case["unsigned_inputs"] else 2
    ):
        raise ValueError("group strict prefix capacity")
    return case


def partial_bytes(case):
    return (
        case["m"] * case["n"] * ((case["k"] + 4095) // 4096) * 8
        if case["method"] == "split-k"
        else 0
    )


def scratch_bytes(case):
    if case["mode"] == "rns":
        return 0
    task = partial_bytes(case) + (32 if case["method"] == "wmma" else 16)
    return case["tasks"] * (task + 32) if case["mode"] == "group" else task


def collection_plan(binary, cases, output, runs, seed, identity):
    plan = shared_plan(
        binary.resolve(strict=True),
        [path.resolve(strict=True) for path in cases],
        output.resolve(),
        runs,
        seed,
        identity,
        validate_case,
    )
    for row in plan:
        row["command"].append("--record-execution-environment")
    return plan


def production_environment(environment):
    if any(
        key.upper().startswith(("PERF_", "ROCP_", "ROCPROF", "RNS8_PROFILE_"))
        or key.upper() == "LD_PRELOAD"
        for key in environment
    ):
        raise ValueError("group production observation contains profiler injection")


def validate_acquisition(capture, entry):
    reference = capture.get("acquisition")
    expected = Path(entry["output"]).with_suffix(".process.json").resolve()
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "sha256"}
        or Path(reference["path"]).resolve() != expected
    ):
        raise ValueError("group raw process association missing")
    if not 1 <= expected.stat().st_size <= 128 * 1024**2:
        raise ValueError("group raw process receipt exceeds byte budget")
    raw_bytes = expected.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != reference["sha256"]:
        raise ValueError("group raw process receipt changed")
    raw = parse_evidence_json(raw_bytes.decode("utf-8"))
    if (
        not isinstance(raw, dict)
        or not capture_succeeded(raw)
        or raw.get("command") != entry["command"]
        or type(raw.get("process_id")) is not int
        or raw["process_id"] <= 0
        or capture.get("process_id") != raw["process_id"]
        or type(capture.get("process_id")) is not int
        or raw.get("max_output_bytes") != OUTPUT_LIMIT
    ):
        raise ValueError("group raw command/PID/status association")
    projection = {
        key: value
        for key, value in capture.items()
        if key not in {"acquisition", "request_identity", "artifact_capture", "execution_id"}
    }
    if not same_evidence_value(projection, parse_evidence_json(raw["stdout"])):
        raise ValueError("group capture changed relative to native stdout")
    if file_digest(expected) != reference["sha256"]:
        raise ValueError("group acquisition changed during review")


def acquire(entry, timeout_seconds, *, environment=None, runner=run_capture):
    child = dict(os.environ if environment is None else environment)
    command = entry["command"]
    source = Path(command[1])
    if file_digest(source) != entry["case_sha256"]:
        raise ValueError("group case changed before execution")
    request = request_identity(command, environ=child)
    production_environment(child)
    raw = runner(
        command,
        cwd=ROOT,
        label="narrow-group " + entry["run_id"],
        timeout_seconds=timeout_seconds,
        max_output_bytes=OUTPUT_LIMIT,
        env=child,
    )
    output = Path(entry["output"])
    process_path = output.with_suffix(".process.json")
    with process_path.open("x", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=2)
    if not capture_succeeded(raw):
        raise RuntimeError(
            "group process failed; bounded raw output retained: " + str(process_path)
        )
    if file_digest(source) != entry["case_sha256"]:
        raise ValueError("group case changed during execution")
    capture = parse_evidence_json(raw["stdout"])
    attach_capture_identity(capture, request, Path(command[0]), execution_id=str(uuid.uuid4()))
    capture["acquisition"] = dict(path=str(process_path), sha256=file_digest(process_path))
    validate_capture(capture, entry)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(capture, handle, indent=2)
    return capture


def validate_summaries(capture, case):
    if case["mode"] == "rns" or case["residency"] == "cold":
        if capture.get("summary_first") is not None or capture.get("summary_final") is not None:
            raise ValueError("unobserved group summaries must stay missing")
        return
    total_calls = (
        1 + case["warmups"] + case["repeats"] + sum(row["calls"] for row in capture["sustained"])
    )
    keys = {
        "a_instance",
        "b_instance",
        "a_generation",
        "b_generation",
        "a_host_proofs",
        "b_host_proofs",
        "a_reuses",
        "b_reuses",
        "a_device_scans",
        "b_device_scans",
        "scan_bytes",
        "partial_bytes",
        "directory_bytes",
        "directory_tasks",
    }
    for name, calls in (("summary_first", 1), ("summary_final", total_calls)):
        rows = capture.get(name)
        if not isinstance(rows, list) or len(rows) != case["tasks"]:
            raise ValueError("complete per-task group summary required")
        for i, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or set(row) != keys
                or any(type(value) is not int or not 0 <= value < 2**64 for value in row.values())
                or not row["a_instance"]
                or not row["b_instance"]
            ):
                raise ValueError("typed group summary")
            directory = case["mode"] == "group" and i == 0
            expected = dict(
                scan_bytes=0,
                partial_bytes=partial_bytes(case),
                directory_bytes=case["tasks"] * 32 if directory else 0,
                directory_tasks=case["tasks"] if directory else 0,
            )
            for side in ("a", "b"):
                repack = case["residency"] in ("repack", "repack-" + side)
                expected.update(
                    {
                        side + "_generation": 1 + calls if repack else 1,
                        side + "_host_proofs": calls if repack else 1,
                        side + "_reuses": 0 if repack else calls - 1,
                        side + "_device_scans": 0,
                    }
                )
            if any(row[key] != value for key, value in expected.items()):
                raise ValueError("group proof/reuse/resource accounting")
        if len({row["a_instance"] for row in rows}) != case["tasks"] or len(
            {row["b_instance"] for row in rows}
        ) != (1 if case["shared_b"] else case["tasks"]):
            raise ValueError("independent/shared group input identities")
    for first, last in zip(capture["summary_first"], capture["summary_final"]):
        if any(first[key] != last[key] for key in ("a_instance", "b_instance")):
            raise ValueError("group owner instance changed during reuse")


def validate_diagnostic(capture, case):
    info = capture.get("separate_native_observation")
    validate_native_observation(
        info,
        scope="one_separate_complete_task_batch",
        calls_key="public_product_calls",
        calls=1 if case["mode"] == "group" else case["tasks"],
    )
    labels = info["event_labels"]
    if case["mode"] != "rns":
        grouped = case["mode"] == "group"
        if case["method"] == "split-k":
            selected = (
                ("narrow_group_split_k_partials", "narrow_group_split_k_encode")
                if grouped
                else ("narrow_native_split_k_partials", "narrow_native_split_k_encode")
            )
        else:
            selected = (
                (
                    "narrow_group_byte_wmma_fused_rns"
                    if case["method"] == "wmma"
                    else "narrow_group_scalar_fused_rns",
                )
                if grouped
                else (
                    "native_byte_wmma_fused_rns"
                    if case["method"] == "wmma"
                    else "narrow_native_i64_fused_rns_kernel",
                )
            )
        if any(labels.get(key) != (1 if grouped else case["tasks"]) for key in selected):
            raise ValueError("actual grouped versus single native stages")
        if labels.get("native_wmma_prepare_capability", 0) != int(
            case["method"] == "wmma" and case["residency"] == "cold"
        ):
            raise ValueError("compiled capability is not a per-task probe")
        if any(
            labels.get(key, 0)
            for key in (
                "narrow_native_domain_scan",
                "narrow_native_bound_scan",
                "native_byte_wmma_admission",
                "narrow_native_encode_rns",
            )
        ):
            raise ValueError("host-pack certified group unexpectedly rescanned/staged")
        if (
            case["residency"] == "resident"
            and case["output"] == "resident"
            and any(info[key] for key in COPY_COUNTERS)
        ):
            raise ValueError("warm resident group allocated/copied data")


def validate_capture(capture, entry):
    if not isinstance(capture, dict):
        raise ValueError("group capture object required")
    case = validate_case(entry["case"])
    observation_boundary(capture)
    fixed = dict(
        schema="rns8_narrow_group_microcase_v1",
        hot_loop_serialization="typed_observation_no_phase_clocks_in_sustained_v2",
        case=case,
        run_id=entry["run_id"],
        dataset="affine_independent_native_tasks_v1",
        selected_prefix=selected_prefix(case),
        exact_proof_policy="scalar_first",
        host_instrumentation_state="hooks_disabled",
        task_directory_entry_bytes=32,
        output_boundary="complete_task_batch_resident_rns"
        if case["output"] == "resident"
        else "complete_task_batch_padded_three_limb_export",
        private_narrow_budget_bytes=scratch_bytes(case),
        native_useful_products=case["tasks"] * case["m"] * case["n"] * case["k"],
        full_output_reference_values=case["tasks"] * case["m"] * case["n"],
        exact_reference_passed=True,
        all_task_residue_planes_audited=True,
        oracle_inside_production_interval=False,
        promotion_eligible=False,
    )
    if any(
        type(capture.get(key)) is not type(value) or capture[key] != value
        for key, value in fixed.items()
    ):
        raise ValueError("group capture contract")
    validate_acquisition(capture, entry)
    if not manifest_provenance_complete(capture.get("build_manifest")) or not capture.get(
        "execution_id"
    ):
        raise ValueError("group compiled/process provenance")
    artifact = capture.get("artifact_capture") or {}
    if (
        artifact.get("unchanged_during_run") is not True
        or artifact.get("rns8_linkage") != "static"
        or len(artifact.get("executable_sha256", "")) != 64
    ):
        raise ValueError("group immutable static executable required")
    request = capture.get("request_identity")
    _, error = request_environment_identity(request)
    if error:
        raise ValueError(error)
    validate_execution_request(capture, request, required=True)
    production_environment(request["environment"])
    observed = capture_observation(capture)
    if (
        case["method"] == "wmma"
        and case["mode"] != "rns"
        and observed.architecture not in {"gfx1100", "gfx1101", "gfx1102"}
    ):
        raise ValueError("group byte WMMA requires the actual compiled gfx11 target family")
    validate_completed_call_samples(
        capture, case["repeats"], resident_output=case["output"] == "resident"
    )
    validate_summaries(capture, case)
    validate_diagnostic(capture, case)
    return capture


def cohort_identity(capture, case):
    return dict(
        case={key: value for key, value in case.items() if key != "mode"},
        device=capture_observation(capture).to_json(),
        build=capture["build_manifest"],
        output_boundary=capture["output_boundary"],
        proof_policy=capture["exact_proof_policy"],
        observation_boundary=observation_boundary(capture),
        artifact={
            key: capture["artifact_capture"].get(key)
            for key in ("executable_sha256", "rns8_linkage", "shared_library_sha256")
        },
        environment=capture["request_identity"]["environment"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=5353)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 3600:
        raise ValueError("bounded collection timeout required")
    args.output = args.output.resolve()
    if args.output == ROOT / "temp" or not args.output.is_relative_to(ROOT / "temp"):
        raise ValueError("group captures require an ignored child of repository temp/")
    plan = collection_plan(
        args.binary, args.cases, args.output, args.runs, args.seed, str(uuid.uuid4())
    )
    manifest = dict(
        schema="narrow_group_collection_v1", observations=plan, promotion_eligible=False
    )
    if args.plan_only:
        print(json.dumps(manifest, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "plan.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
    cohorts = {}
    for entry in plan:
        path = Path(entry["output"])
        if path.exists():
            raise ValueError("immutable group capture exists")
        capture = acquire(entry, args.timeout_seconds)
        identity = cohort_identity(capture, entry["case"])
        cohorts.setdefault(json.dumps(identity, sort_keys=True), []).append(
            dict(
                mode=entry["case"]["mode"],
                execution_id=capture["execution_id"],
                independent_run=entry["independent_run"],
                path=str(path),
                median_us=statistics.median(row["total_us"] for row in capture["samples"]),
                setup_us=capture["setup_us"],
            )
        )
    rows = []
    for identity, observations in cohorts.items():
        if len({row["execution_id"] for row in observations}) != len(observations):
            raise ValueError("group captures reused one process observation")
        medians = {
            mode: statistics.median(row["median_us"] for row in observations if row["mode"] == mode)
            for mode in ("single", "group", "rns")
            if sum(row["mode"] == mode for row in observations) >= 2
        }
        rows.append(
            dict(
                identity=json.loads(identity),
                observations=observations,
                independent_process_medians_us=medians,
                unqualified_baseline_over_group={
                    baseline: medians[baseline] / medians["group"]
                    for baseline in ("single", "rns")
                    if baseline in medians and "group" in medians
                },
                qualification="not_inferred",
                promotion_eligible=False,
            )
        )
    with (args.output / "summary.json").open("x", encoding="utf-8") as output:
        json.dump(
            dict(schema="narrow_group_summary_v1", cohorts=rows, promotion_eligible=False),
            output,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
