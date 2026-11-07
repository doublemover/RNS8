#!/usr/bin/env python3
"""Plan bounded process-scoped perf sampling; execution requires --execute."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

from benchmark_request import file_digest
from build_manifest import configuration_options
from compiled_binary_identity import canonical_digest
from check_dependencies_lib.system import find_command
from execution_environment_contract import relevant_environment_key
from execution_inventory import bounded_read
from elf_module_identity import module_inventory
from gpu_qualification import ROOT, execute_plan, write_json
from host_profile_native import PRODUCER_CONTRACT, SCHEMA as NATIVE_SCHEMA
from host_profile_ranges import join as join_host_ranges
from perf_recording_contract import read_recording, validate_decoded_samples
from perf_mapping_contract import frame_custody
from host_source_attribution import source_inventory, bind_selected_sites
from json_evidence import parse_evidence_json, same_evidence_value
from profiler_contract import production_profile_input, bind_host_profile_output
from profiler_artifacts import write_diagnostic
from process_progress import capture_succeeded
from target_qualification import EvidenceStore, session_stage
from timing_eligibility import capture_identity

SCHEMA = "rns8_host_perf_session_v1"
SOURCE_INPUTS = (
    ".python-version",
    "tools/host_profile_session.py",
    "tools/host_profile_native.py",
    "tools/host_profile_ranges.py",
    "tools/host_range_contract.py",
    "tools/interval_observations.py",
    "tools/profiler_contract.py",
    "tools/profiler_artifacts.py",
    "tools/build_instrumentation.py",
    "tools/execution_environment_contract.py",
    "tools/execution_observation.py",
    "tools/gpu_qualification.py",
    "tools/process_progress.py",
    "tools/target_qualification.py",
    "tools/benchmark_request.py",
    "tools/diagnostic_companion.py",
    "tools/companion_contract.py",
    "tools/json_evidence.py",
    "tools/execution_inventory.py",
    "tools/perf_recording_contract.py",
    "tools/perf_mapping_contract.py",
    "tools/host_source_attribution.py",
    "tools/elf_module_identity.py",
)

PERMISSION_DIAGNOSTIC_SOURCES = (
    "https://code.googlesource.com/linux/torvalds/linux/+/473f6c8f437b049f8ec015d57cd59bb983b1d85c/tools/perf/util/evsel.c",
    "https://code.googlesource.com/linux/torvalds/linux/+/4edcdefd4083ae04b1a5656f4be6cd83ae919ef4/tools/perf/builtin-record.c",
)
PERMISSION_PATTERNS = tuple(
    (
        reason,
        re.compile(r"^[ \t]*" + message + r"[ \t\r]*$", re.MULTILINE | re.IGNORECASE | re.ASCII),
    )
    for reason, message in (
        ("reported_perf_mapping_permission_denial", r"permission error mapping pages\."),
        (
            "reported_perf_event_permission_denial",
            r"access to performance monitoring and observability operations is limited\.",
        ),
        ("reported_perf_event_permission_denial", r"no permission to enable [^\r\n]+ event\."),
    )
)


def recording_failure(process: dict) -> dict:
    """Classify complete perf stderr reports, never infer kernel errno/policy."""
    if capture_succeeded(process):
        return dict(
            state="not_failed", reason="recording_process_exit_zero_not_profile_qualification"
        )
    if process.get("interrupted") is True:
        return dict(state="failed", reason="recording_interrupted")
    if process.get("interrupted") is not None and process.get("interrupted") is not False:
        return dict(state="failed", reason="recording_process_outcome_incomplete")
    if process.get("timed_out") is True:
        return dict(state="failed", reason="recording_timeout")
    if process.get("limit_exceeded") is not None or process.get("capture_error") is not None:
        return dict(state="failed", reason="recording_capture_or_budget_failure")
    code = process.get("returncode")
    if code is None:
        return dict(state="blocked", reason="recording_process_did_not_start")
    if type(code) is not int or code == 0 or process.get("timed_out") is not False:
        return dict(state="failed", reason="recording_process_outcome_incomplete")
    if code < 0:
        return dict(state="failed", reason="recording_process_signaled")
    stderr = process.get("stderr")
    if not isinstance(stderr, str):
        return dict(state="failed", reason="recording_stderr_unavailable")
    # Generic "permission denied" can refer to output files or the workload.
    # Only upstream perf-specific terminal diagnostics support these labels.
    # Scan in place instead of allocating one Python object per log line.
    reason = next(
        (reason for reason, pattern in PERMISSION_PATTERNS if pattern.search(stderr)), None
    )
    if reason is None:
        return dict(state="failed", reason="recording_failure_unclassified")
    return dict(
        state="blocked",
        reason=reason,
        diagnostic_sources=list(PERMISSION_DIAGNOSTIC_SOURCES),
        attribution="perf_reported_stderr_not_observed_syscall_errno_or_policy_proof",
    )


def acquisition_failure(plan: dict, out: Path) -> dict:
    """Retain the first failed checkpoint and all partial files; never retry."""
    base = dict(
        schema="rns8_host_perf_acquisition_failure_v1",
        qualification="failed_acquisition",
        state="failed",
        plan_identity_sha256=plan["identity_sha256"],
        raw_artifacts_preserved=True,
        acquisition_verified=False,
        production_latency=False,
        promotion_allowed=False,
    )
    try:
        manifest_text, manifest_ref = bounded_read(out / "manifest.json")
        base["manifest"] = manifest_ref
        if not same_evidence_value(parse_evidence_json(manifest_text), plan):
            raise ValueError("acquisition manifest changed")
        text, results_ref = bounded_read(out / "results.json")
        base["session_results"] = results_ref
        results = parse_evidence_json(text)
        rows = results.get("results") if isinstance(results, dict) else None
        if (
            not isinstance(results, dict)
            or results.get("schema") != "rns8_gpu_session_results_v2"
            or results.get("manifest_identity_sha256") != plan["identity_sha256"]
            or results.get("status") not in {"blocked", "failed", "interrupted"}
            or not isinstance(results.get("run_id"), str)
            or not results["run_id"]
            or not isinstance(rows, list)
            or not rows
            or len(rows) > len(plan["commands"])
        ):
            raise ValueError("failed acquisition checkpoint does not bind the plan")
        failed = None
        for expected, row in zip(plan["commands"], rows):
            if failed is not None:
                raise ValueError("acquisition checkpoint continues after its first failure")
            if not isinstance(row, dict) or row.get("name") != expected["name"]:
                raise ValueError("acquisition checkpoint order changed")
            if row.get("command") != expected["argv"]:
                raise ValueError("acquisition checkpoint command changed")
            if failed is None and (not capture_succeeded(row) or row.get("validation_errors")):
                failed = row
        if failed is None:
            raise ValueError("failed session has no failed checkpoint")
        classification = (
            recording_failure(failed)
            if failed["name"] == "cpu-sampling"
            else dict(state="failed", reason="pre_sampling_or_decode_stage_failed")
        )
        if classification["state"] == "not_failed":
            classification = dict(state="failed", reason="acquisition_stage_validation_failed")
        for ref in (manifest_ref, results_ref):
            if file_digest(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("acquisition checkpoint changed during classification")
        return dict(
            base,
            **classification,
            stage=failed["name"],
            process_returncode=failed.get("returncode"),
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        return dict(
            base,
            reason="acquisition_failure_evidence_unavailable_or_changed",
            error=str(error)[:4096],
        )


def reference(path: Path) -> dict:
    path = path.resolve(strict=True)
    return dict(path=str(path), sha256=file_digest(path))


def elf_machine(path: Path) -> int:
    with path.open("rb") as source:
        header = source.read(20)
    if len(header) != 20 or header[:6] != b"\x7fELF\x02\x01":
        raise ValueError(
            "Linux sampling requires explicit native ELF64 little-endian tools/binary, not launcher scripts"
        )
    return int.from_bytes(header[18:20], "little")


def plan_capture(
    production: Path,
    perf_tool: Path,
    python: Path,
    out: Path,
    *,
    timeout_seconds=120.0,
    total_seconds=240.0,
    max_artifact_bytes=256 * 1024**2,
    max_output_bytes=16 * 1024**2,
    minimum_free_bytes=1024**3,
    operation_ranges=False,
    _review_existing=False,
) -> dict:
    if type(operation_ranges) is not bool:
        raise ValueError("operation range selection must be explicit boolean policy")
    if (
        any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (timeout_seconds, total_seconds)
        )
        or total_seconds <= timeout_seconds + 30
    ):
        raise ValueError("host sampling needs positive finite budgets and an export/query reserve")
    if any(
        type(value) is not int or value < 1
        for value in (max_artifact_bytes, max_output_bytes, minimum_free_bytes)
    ):
        raise ValueError("host sampling needs explicit positive byte budgets")
    out = out.resolve()
    if out == (ROOT / "temp").resolve() or not out.is_relative_to((ROOT / "temp").resolve()):
        raise ValueError("host profiles require an ignored child of repository temp/")
    if out.exists() and not _review_existing:
        raise ValueError("host profile output must be absent")
    inputs = dict(
        production_capture=reference(production),
        perf_tool=reference(perf_tool),
        python=reference(python),
    )
    store = EvidenceStore()
    _, capture = store.json(inputs["production_capture"], ROOT)
    executable = production_profile_input(capture, store)
    if (
        operation_ranges
        and configuration_options(capture["build_manifest"]["configuration"], strict=True).get(
            "roctx"
        )
        != "ON"
    ):
        raise ValueError("operation ranges require the same production binary compiled with ROCTx")
    perf_tool, python = (store.reference(inputs[key], ROOT) for key in ("perf_tool", "python"))
    machine = elf_machine(executable)
    if machine not in (62, 183) or any(
        elf_machine(tool) != machine for tool in (perf_tool, python)
    ):
        raise ValueError("benchmark, perf and Python need one native x86-64/AArch64 ELF tuple")
    sources = [reference(ROOT / name) for name in SOURCE_INPUTS]
    arguments = list(capture["request_identity"]["arguments"])
    arguments += ["--json-output", str(out / "native/benchmark.json")]
    if "--record-execution-environment" not in arguments:
        arguments.append("--record-execution-environment")
    expected_environment = dict(capture["request_identity"]["environment"], PERF_CONFIG="/dev/null")
    profiler_environment = dict(
        PERF_CONFIG="/dev/null", DEBUGINFOD_URLS="", PYTHONDONTWRITEBYTECODE="1"
    )
    if operation_ranges:
        expected_environment["RNS8_PROFILE_RANGES"] = "1"
        profiler_environment["RNS8_PROFILE_RANGES"] = "1"
    commands = []

    def add(name, argv, **policy):
        commands.append(
            dict(
                name=name,
                argv=argv,
                depends_on=[commands[-1]["name"]] if commands else [],
                timeout_seconds=timeout_seconds,
                **policy,
            )
        )

    add("perf-version", [str(perf_tool), "--version"])
    add(
        "native-module", [str(python), str(ROOT / "tools/host_profile_native.py"), "--capabilities"]
    )
    add(
        "benchmark-identity",
        [str(executable), "--build-identity"],
        compiled_identity=True,
        identity_backends=[capture["backend_selected"]],
        identity_output=str(out / "benchmark-identity.json"),
    )
    add(
        "cpu-sampling",
        [
            str(perf_tool),
            "record",
            "--event",
            "cpu-clock:u",
            "--freq",
            "99",
            "--sample-cpu",
            "--sample-identifier",
            "--timestamp",
            "--period",
            "--call-graph",
            "dwarf,8192",
            "--clockid",
            "CLOCK_MONOTONIC",
            "--max-size",
            "64M",
            "--no-buildid-cache",
            "--buildid-mmap",
            "--output",
            str(out / "native/perf.data"),
            "--",
            str(executable),
            *arguments,
        ],
        owned_artifact_directories=[str(out / "native")],
        evidence_outputs=[str(out / "native/perf.data"), str(out / "native/benchmark.json")],
        max_artifact_bytes=max_artifact_bytes,
        max_output_bytes=max_output_bytes,
    )
    add(
        "native-decode",
        [
            str(python),
            str(ROOT / "tools/host_profile_native.py"),
            "--input",
            str(out / "native/perf.data"),
            "--capture",
            str(out / "native/benchmark.json"),
            "--output",
            str(out / "native/samples.json"),
        ],
        evidence_outputs=[str(out / "native/samples.json")],
        monitor_artifact_directories=[str(out / "native")],
        max_artifact_bytes=max_artifact_bytes,
        max_output_bytes=max_output_bytes,
    )
    plan = dict(
        schema=SCHEMA,
        schema_version=3,
        session_kind="host-perf-cpu-sampling",
        system="Linux",
        command_working_directory=str(ROOT),
        output_directory=str(out),
        **inputs,
        benchmark=reference(executable),
        source_inputs=sources,
        production_capture_identity=capture_identity(capture),
        inherited_environment=capture["request_identity"]["environment"],
        expected_native_environment=expected_environment,
        profiler_environment=profiler_environment,
        operation_ranges=operation_ranges,
        commands=commands,
        devices=[],
        budgets=dict(
            total_seconds=total_seconds,
            export_reserve_seconds=30,
            minimum_free_bytes=minimum_free_bytes,
        ),
        event="cpu-clock:u",
        sample_frequency_hz=99,
        record_clock="CLOCK_MONOTONIC",
        scope="owned_process_tree_sampling_selected_benchmark_pid_decoding",
        qualification="plan_only",
        promotion_allowed=False,
    )
    plan["identity_sha256"] = canonical_digest(plan)
    store.finish()
    return plan


def rebuild(plan: dict, *, review_existing=False) -> dict:
    return plan_capture(
        *(Path(plan[key]["path"]) for key in ("production_capture", "perf_tool", "python")),
        Path(plan["output_directory"]),
        timeout_seconds=plan["commands"][0]["timeout_seconds"],
        total_seconds=plan["budgets"]["total_seconds"],
        max_artifact_bytes=plan["commands"][3]["max_artifact_bytes"],
        max_output_bytes=plan["commands"][3]["max_output_bytes"],
        minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
        operation_ranges=plan["operation_ranges"],
        _review_existing=review_existing,
    )


def finalize(manifest_path: Path) -> dict:
    store = EvidenceStore()
    _, plan = store.json(reference(manifest_path), ROOT)
    out = manifest_path.resolve(strict=True).parent
    if plan.get("schema") != SCHEMA or plan.get("output_directory") != str(out):
        raise ValueError("wrong host profile manifest or output directory")
    if not same_evidence_value(plan, rebuild(plan, review_existing=True)):
        raise ValueError("host profile plan no longer matches the maintained policy/inputs")
    for artifact in (plan["perf_tool"], plan["python"], *plan["source_inputs"]):
        store.reference(artifact, ROOT)
    _, production = store.json(plan["production_capture"], ROOT)
    executable = production_profile_input(production, store)
    _, results = store.json(reference(out / "results.json"), ROOT)
    proof = dict(
        session_manifest=reference(manifest_path),
        session_results=reference(out / "results.json"),
        session_id=plan["identity_sha256"],
        run_id=results["run_id"],
        working_directory=str(ROOT),
        stage="cpu-sampling",
        binary_identity_stage="benchmark-identity",
    )
    item, row, _, identity_row, stages = session_stage(proof, ROOT, store)
    if len(stages) != len(plan["commands"]):
        raise ValueError("host sampling session is incomplete")
    binary = store.binary(identity_row["compiled_binary_identity"], production["backend_selected"])
    if binary["executable_sha256"] != plan["benchmark"]["sha256"] or binary[
        "build_manifest_sha256"
    ] != canonical_digest(production["build_manifest"]):
        raise ValueError("host profile compiled identity differs from production")
    for command, observed in stages.values():
        if (
            observed.get("command") != command["argv"]
            or observed.get("validation_errors")
            or not capture_succeeded(observed)
        ):
            raise ValueError("host profile stage differs from the launched command")
    cap = parse_evidence_json(stages["native-module"][1]["stdout"])
    if (
        not isinstance(cap, dict)
        or cap.get("schema") != "rns8_perf_native_capabilities_v1"
        or cap.get("state") != "available"
        or cap.get("producer_contract") != PRODUCER_CONTRACT
        or cap.get("python_version") != (ROOT / ".python-version").read_text().strip()
    ):
        raise ValueError("native perf module/interpreter capability is incompatible")
    store.reference(cap["native_module"], ROOT)
    raw_path, samples_path = out / "native/benchmark.json", out / "native/samples.json"
    _, raw = store.json(dict(path=str(raw_path), sha256=row["artifacts"][str(raw_path)]), ROOT)
    separator = item["argv"].index("--")
    workload = item["argv"][separator + 1 :]
    profiled = bind_host_profile_output(
        production,
        raw,
        workload=workload,
        environment=plan["expected_native_environment"],
        executable=executable,
        execution_id=results["run_id"],
    )
    decode_row = stages["native-decode"][1]
    _, samples = store.json(
        dict(path=str(samples_path), sha256=decode_row["artifacts"][str(samples_path)]), ROOT
    )
    data_path = out / "native/perf.data"
    recording = read_recording(data_path)
    if samples.get("identity_sha256") != canonical_digest(
        {key: value for key, value in samples.items() if key != "identity_sha256"}
    ):
        raise ValueError("native CPU sample document identity mismatch")
    if (
        samples.get("schema") != NATIVE_SCHEMA
        or samples.get("producer_contract") != PRODUCER_CONTRACT
        or samples.get("pid") != raw.get("process_id")
        or samples.get("input") != reference(data_path)
        or samples.get("recording_sha256") != file_digest(data_path)
        or samples.get("decoder_source_sha256")
        != file_digest(ROOT / "tools/host_profile_native.py")
        or samples.get("native_module") != cap.get("native_module")
        or samples.get("profiled_capture") != reference(raw_path)
        or type(samples.get("pid")) is not int
        or samples["pid"] <= 0
        or samples["pid"] == row["process_id"]
        or samples.get("machine") != elf_machine(executable)
        or samples.get("is_64_bit") is not True
        or samples.get("acquisition_verified") is not False
        or samples.get("scope") != "selected_process_total_not_selected_operation"
        or not same_evidence_value(samples.get("recording_contract"), recording)
        or samples.get("recording_contract_source_sha256")
        != file_digest(ROOT / "tools/perf_recording_contract.py")
        or not same_evidence_value(samples.get("clock"), recording["clock"])
        or recording["clock"]["attribute_clock_id"] != 1
        or samples.get("loss_status") != recording["loss"]["status"]
        or samples.get("leaf_module_custody_source_sha256")
        != file_digest(ROOT / "tools/elf_module_identity.py")
        or samples.get("recorded_frame_custody_source_sha256")
        != file_digest(ROOT / "tools/perf_mapping_contract.py")
        or samples.get("current_source_custody_source_sha256")
        != file_digest(ROOT / "tools/host_source_attribution.py")
        or samples.get("production_latency") is not False
        or samples.get("promotion_allowed") is not False
    ):
        raise ValueError(
            "native CPU samples do not bind the actual benchmark PID/data/module/capture"
        )
    validate_decoded_samples(
        recording,
        samples.get("samples"),
        pid=raw["process_id"],
        observed=samples.get("observed_samples"),
        excluded=samples.get("excluded_other_process_samples"),
    )
    modules = module_inventory(samples["samples"], machine=samples["machine"])
    if not same_evidence_value(samples.get("leaf_module_custody"), modules):
        raise ValueError("native leaf module identities changed or contradict the sample receipt")
    frames = frame_custody(samples["samples"], recording, machine=samples["machine"])
    if not same_evidence_value(samples.get("recorded_frame_custody"), frames):
        raise ValueError(
            "native frame attribution differs from recorded mappings/current ELF files"
        )
    sources = source_inventory(samples["samples"])
    if not same_evidence_value(samples.get("current_source_custody"), sources):
        raise ValueError("native source attribution differs from current first-party source files")
    for source in sources["files"]:
        store.reference(source, ROOT)
    for module in [*modules["modules"], *frames["current_modules"]["modules"]]:
        if module["file"] is not None:
            store.reference(module["file"], ROOT)
    if type(samples.get("selected_process_samples")) is not int or samples[
        "selected_process_samples"
    ] != len(samples["samples"]):
        raise ValueError("native selected sample count differs from its rows")
    selected_operation_range = (
        join_host_ranges(samples, raw.get("profiler_ranges"))
        if plan["operation_ranges"]
        else "unbound"
    )
    if plan["operation_ranges"]:
        selected_operation_range["selected_source_sites"] = bind_selected_sites(
            selected_operation_range["selected_source_sites"], sources
        )
    # perf forks the benchmark. The launched profiler's PID is intentionally not
    # substituted for the native application PID or used as a sample selector.
    profiled["external_instrumentation"] = dict(
        schema="rns8_external_instrumentation_v1",
        kind="host_profiler",
        tool="perf",
        mode="cpu_sampling",
    )
    bound = out / "profiled.json"
    if bound.exists():
        raise ValueError("host profile finalization cannot overwrite a bound capture")
    store.finish()
    write_diagnostic(
        bound,
        profiled,
        root=out,
        total_bytes=plan["commands"][3]["max_artifact_bytes"],
        minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
    )
    return dict(
        schema="rns8_host_perf_acquisition_v1",
        acquisition_session=proof,
        production_capture=plan["production_capture"],
        profiled_capture=reference(bound),
        native_samples=reference(samples_path),
        native_data=reference(data_path),
        benchmark_pid=raw["process_id"],
        profiler_launcher_pid=row["process_id"],
        clock=dict(recording["clock"], requested=plan["record_clock"]),
        recording_contract=recording,
        sampling_integrity="partial" if recording["blockers"] else "complete_known_record_walk",
        scope=samples["scope"],
        loss_status=samples["loss_status"],
        leaf_module_custody=modules,
        recorded_frame_custody=frames,
        current_source_custody=sources,
        selected_operation_range=selected_operation_range,
        production_latency=False,
        promotion_allowed=False,
    )


def execute_capture(plan: dict, *, environment=None, runner=None) -> int:
    if platform.system() != "Linux":
        raise ValueError("native perf acquisition requires Linux")
    if not same_evidence_value(plan, rebuild(plan)):
        raise ValueError("host sampling plan changed before admission")
    child = {
        key: value
        for key, value in dict(os.environ if environment is None else environment).items()
        if not relevant_environment_key(key)
        and not key.startswith("PERF_")
        and key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    child.update(plan["expected_native_environment"])
    child.update(plan["profiler_environment"])
    out = Path(plan["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", plan)
    options = {} if runner is None else dict(runner=runner)
    result = execute_plan(plan, out, child, **options)
    if result:
        write_json(out / "host-profile.json", acquisition_failure(plan, out))
        return result
    try:
        receipt = finalize(out / "manifest.json")
        write_diagnostic(
            out / "host-profile.json",
            receipt,
            root=out,
            total_bytes=plan["commands"][3]["max_artifact_bytes"],
            minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        write_json(
            out / "host-profile.json",
            dict(
                qualification="binding_failed",
                error=str(error),
                raw_artifacts_preserved=True,
                promotion_allowed=False,
            ),
        )
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument(
        "--perf",
        type=Path,
        help="Explicit native perf executable; defaults to dependency discovery",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--operation-ranges",
        action="store_true",
        help="Enable existing ROCTx host scopes and require same-clock measured-operation attribution",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--total-seconds", type=float, default=240)
    parser.add_argument("--max-artifact-mib", type=int, default=256)
    parser.add_argument("--max-output-mib", type=int, default=16)
    parser.add_argument("--min-free-mib", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        perf = args.perf
        if perf is None:
            located = find_command("perf")
            if not located:
                raise ValueError("native perf executable is unavailable; no profiler was started")
            perf = Path(located)
        plan = plan_capture(
            args.production,
            perf,
            args.python,
            args.out,
            timeout_seconds=args.timeout_seconds,
            total_seconds=args.total_seconds,
            max_artifact_bytes=args.max_artifact_mib * 1024**2,
            max_output_bytes=args.max_output_mib * 1024**2,
            minimum_free_bytes=args.min_free_mib * 1024**2,
            operation_ranges=args.operation_ranges,
        )
        if args.execute:
            return execute_capture(plan)
        print(json.dumps(plan, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("host profile session: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
