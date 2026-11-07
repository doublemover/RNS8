#!/usr/bin/env python3
"""Plan uniquely owned WPR/WPA sampling; recording needs two explicit CLI flags."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import uuid

from benchmark_schema import validate_capture
from build_manifest import configuration_options
from check_dependencies_lib.system import find_command
from compiled_binary_identity import bind_document, canonical_digest, validate_document
from execution_environment_contract import relevant_environment_key
from etl_process_contract import process_contract
from etl_image_contract import image_catalog
from etl_range_contract import operation_ranges as bind_operation_ranges
from etl_cpu_contract import cpu_catalog
from etl_schedule_contract import schedule_catalog
from etl_inspector_identity import source_identity as etl_source_identity
from gpu_qualification import owned_artifact_files
from host_instrumentation_session import reference, write_json
from json_evidence import parse_evidence_json, same_evidence_value
from pe_symbol_identity import symbol_pair, native_image
from process_progress import capture_succeeded, run_capture
from profiler_contract import production_profile_input, bind_host_profile_output
from profiler_artifacts import write_diagnostic
from target_qualification import EvidenceStore
from execution_inventory import bounded_read
from wpa_sample_reader import read_samples

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rns8_windows_host_profile_plan_v4"
MAX_ETL_METADATA_BYTES = 48 * 1024**2
SUCCESS_STAGES = (
    "wpr_help",
    "exporter_help",
    "profiles",
    "identity",
    "inspector_identity",
    "status",
    "start",
    "workload",
    "stop",
    "inspect_etl",
    "export",
)
SOURCE_INPUTS = (
    "tools/host_profile_windows.py",
    "tools/etl_process_contract.py",
    "tools/etl_image_contract.py",
    "tools/etl_range_contract.py",
    "tools/etl_cpu_contract.py",
    "tools/etl_schedule_contract.py",
    "tools/host_interval_phases.py",
    "tools/etl_inspector_identity.py",
    "src/diagnostics/etl_sample_payload.hpp",
    "tools/host_range_contract.py",
    "tools/native_phase_attribution.py",
    "tools/interval_observations.py",
    "tools/build_manifest.py",
    "tools/rns8_etl_inspect.cpp",
    "tools/pe_symbol_identity.py",
    "tools/wpa_sample_reader.py",
    "tools/profiler_contract.py",
    "tools/profiler_artifacts.py",
    "tools/process_progress.py",
    "tools/execution_observation.py",
    "tools/execution_environment_contract.py",
    "tools/compiled_binary_identity.py",
    "tools/host_instrumentation_session.py",
    "tools/gpu_qualification.py",
    "tools/benchmark_request.py",
    "tools/benchmark_schema.py",
    "tools/device_observation.py",
    "tools/diagnostic_companion.py",
    "tools/companion_contract.py",
    "tools/json_evidence.py",
    "tools/target_qualification.py",
    "tools/execution_inventory.py",
    "tools/reference_verdict.py",
    "tools/build_instrumentation.py",
    "metadata/host-profile-cpu.wprp",
    "metadata/host-profile-scheduling.wprp",
    "metadata/host-profile-cpu.wpaProfile",
)


def seal(value):
    value["identity_sha256"] = canonical_digest(
        {key: item for key, item in value.items() if key != "identity_sha256"}
    )
    return value


def plan_capture(
    production,
    wpr,
    exporter,
    pdb,
    out,
    *,
    instance=None,
    inspector=None,
    timeout_seconds=120,
    total_seconds=360,
    max_artifact_bytes=512 * 1024**2,
    minimum_free_bytes=1024**3,
    operation_ranges=False,
    scheduling=False,
    review_existing=False,
):
    if any(type(value) is not bool for value in (operation_ranges, scheduling)):
        raise ValueError("Windows operation range/scheduler policies must be explicit booleans")
    if (
        any(
            type(value) not in (int, float) or not 0 < value <= 1800 or not math.isfinite(value)
            for value in (timeout_seconds, total_seconds)
        )
        or total_seconds <= timeout_seconds + 60
    ):
        raise ValueError(
            "Windows sampling requires bounded time and a60-second cleanup/export reserve"
        )
    if (
        type(max_artifact_bytes) is not int
        or not 256 * 1024**2 <= max_artifact_bytes <= 2 * 1024**3
    ):
        raise ValueError("Windows sampling needs256MiB..2GiB artifact allowance")
    if type(minimum_free_bytes) is not int or minimum_free_bytes < max_artifact_bytes:
        raise ValueError("Windows sampling free-space budget must cover retained artifacts")
    out = Path(out).resolve()
    if (
        out == (ROOT / "temp").resolve()
        or not out.is_relative_to((ROOT / "temp").resolve())
        or (out.exists() and not review_existing)
    ):
        raise ValueError("Windows profiling requires a fresh ignored child of temp/")
    instance = "RNS8_" + uuid.uuid4().hex if instance is None else instance
    if type(instance) is not str or not re.fullmatch(r"RNS8_[0-9a-f]{32}", instance):
        raise ValueError("unique canonical owned WPR instance required")
    store = EvidenceStore()
    production_ref = reference(Path(production))
    _, baseline = store.json(production_ref, ROOT)
    executable = production_profile_input(baseline, store)
    if (
        operation_ranges
        and configuration_options(baseline["build_manifest"]["configuration"], strict=True).get(
            "etw_ranges"
        )
        != "ON"
    ):
        raise ValueError(
            "Windows operation ranges require the same production binary compiled with ETW ranges"
        )
    symbols = symbol_pair(executable, pdb)
    inspector = executable.parent / "rns8-etl-inspect.exe" if inspector is None else Path(inspector)
    if not inspector.is_file():
        raise ValueError(
            "native ETL inspector missing; build rns8-etl-inspect or supply --etl-inspector"
        )
    tools = dict(
        wpr=native_image(wpr), exporter=native_image(exporter), inspector=native_image(inspector)
    )
    if any(tool["machine"] != symbols["executable"]["machine"] for tool in tools.values()):
        raise ValueError("Windows profiler tools and benchmark require one native architecture")
    arguments = [
        *baseline["request_identity"]["arguments"],
        "--json-output",
        str(out / "raw/benchmark.json"),
    ]
    if "--record-execution-environment" not in arguments:
        arguments.append("--record-execution-environment")
    profiles = ROOT / (
        "metadata/host-profile-scheduling.wprp" if scheduling else "metadata/host-profile-cpu.wprp"
    )
    profile_name = "Rns8CpuSchedule" if scheduling else "Rns8Cpu"
    commands = dict(
        wpr_help=[tools["wpr"]["path"], "-help", "start"],
        exporter_help=[tools["exporter"]["path"], "-h"],
        profiles=[tools["wpr"]["path"], "-profiles", str(profiles)],
        identity=[str(executable), "--build-identity"],
        inspector_identity=[tools["inspector"]["path"], "--build-identity"],
        status=[tools["wpr"]["path"], "-status", "-instancename", instance],
        start=[
            tools["wpr"]["path"],
            "-start",
            str(profiles) + "!" + profile_name,
            "-filemode",
            "-recordtempto",
            str(out / "raw"),
            "-instancename",
            instance,
        ],
        workload=[str(executable), *arguments],
        stop=[
            tools["wpr"]["path"],
            "-stop",
            str(out / "raw/profile.etl"),
            "-instancename",
            instance,
        ],
        cancel=[tools["wpr"]["path"], "-cancel", "-instancename", instance],
        inspect_etl=[tools["inspector"]["path"], "--input", str(out / "raw/profile.etl")],
        export=[
            tools["exporter"]["path"],
            "-i",
            str(out / "raw/profile.etl"),
            "-profile",
            str(ROOT / "metadata/host-profile-cpu.wpaProfile"),
            "-delimiter",
            ",",
            "-outputfolder",
            str(out / "export"),
            "-symbols",
        ],
    )
    native_environment = dict(baseline["request_identity"]["environment"])
    if operation_ranges:
        native_environment["RNS8_PROFILE_RANGES"] = "1"
    result = seal(
        dict(
            schema=SCHEMA,
            system="Windows",
            instance=instance,
            output_directory=str(out),
            production_capture=production_ref,
            benchmark=reference(executable),
            symbols=symbols,
            tools=tools,
            source_inputs=[reference(ROOT / path) for path in SOURCE_INPUTS],
            expected_native_environment=native_environment,
            operation_ranges=operation_ranges,
            scheduling=scheduling,
            profile_name=profile_name,
            symbol_search_environment=dict(
                _NT_SYMBOL_PATH=str(Path(pdb).resolve(strict=True).parent),
                _NT_ALT_SYMBOL_PATH="",
                _NT_SYMCACHE_PATH=str(out / "symbols"),
            ),
            commands=commands,
            runtime_argument_bindings=dict(
                inspect_etl="append --pid and the exact completed workload process_id after owned stop"
            ),
            budgets=dict(
                timeout_seconds=timeout_seconds,
                total_seconds=total_seconds,
                cleanup_reserve_seconds=30,
                max_artifact_bytes=max_artifact_bytes,
                minimum_free_bytes=minimum_free_bytes,
            ),
            scope="system_ETW_window_selected_benchmark_PID_decode",
            requires_explicit_system_recording_authorization=True,
            qualification="plan_only_no_native_command_executed",
            promotion_allowed=False,
        )
    )
    store.finish()
    return result


def rebuild(plan, *, review_existing=False):
    return plan_capture(
        Path(plan["production_capture"]["path"]),
        Path(plan["tools"]["wpr"]["path"]),
        Path(plan["tools"]["exporter"]["path"]),
        Path(plan["symbols"]["symbols"]["path"]),
        Path(plan["output_directory"]),
        instance=plan["instance"],
        inspector=Path(plan["tools"]["inspector"]["path"]),
        review_existing=review_existing,
        operation_ranges=plan["operation_ranges"],
        scheduling=plan["scheduling"],
        **{
            key: plan["budgets"][key]
            for key in (
                "timeout_seconds",
                "total_seconds",
                "max_artifact_bytes",
                "minimum_free_bytes",
            )
        },
    )


def instance_absent(capture):
    # Native HRESULT is locale independent. Other failures (including access
    # denial and broken output) do not establish absence or authorize cleanup.
    return (
        capture_succeeded(dict(capture, returncode=0))
        and type(capture.get("returncode")) is int
        and capture["returncode"] > 0
        and re.search(
            r"(?i)(?<![0-9a-f])0xc5583000(?![0-9a-f])", capture["stdout"] + "\n" + capture["stderr"]
        )
        is not None
    )


def validate_preflight_stages(plan, stages):
    if not re.search(r"\b" + re.escape(plan["profile_name"]) + r"\b", stages["profiles"]["stdout"]):
        raise ValueError("native WPR did not advertise the selected custom profile")
    identity = parse_evidence_json(stages["identity"]["stdout"])
    store = EvidenceStore()
    production_ref, production = read_retained(Path(plan["production_capture"]["path"]), store)
    if not same_evidence_value(production_ref, plan["production_capture"]):
        raise ValueError("Windows production input changed during preflight")
    summary = validate_document(identity, production["backend_selected"])
    if (
        summary["build_manifest_sha256"] != canonical_digest(production["build_manifest"])
        or type(identity.get("process_id")) is not int
        or identity["process_id"] <= 0
        or identity.get("process_id") != stages["identity"]["process_id"]
    ):
        raise ValueError(
            "native Windows identity preflight differs from the production process/build"
        )
    inspector = parse_evidence_json(stages["inspector_identity"]["stdout"])
    if (
        type(inspector) is not dict
        or inspector.get("schema") != "rns8_etl_inspector_identity_v1"
        or type(inspector.get("process_id")) is not int
        or inspector["process_id"] != stages["inspector_identity"]["process_id"]
        or inspector.get("source_sha256") != etl_source_identity()
        or type(inspector.get("pointer_bytes")) is not int
        or inspector["pointer_bytes"] != 8
    ):
        raise ValueError("native ETL inspector was compiled from another source or ABI")
    store.finish()


def read_retained(path, store):
    text, ref = bounded_read(path)
    ref["bytes"] = len(text.encode("utf-8"))
    value = parse_evidence_json(text)
    if not isinstance(value, dict):
        raise ValueError("retained Windows document must be an object")
    store.reference(ref, ROOT)
    return ref, value


def completed_capture(plan, stages, *, owner_pid=None, review_existing=False):
    out, store = Path(plan["output_directory"]), EvidenceStore()
    owner_pid = os.getpid() if owner_pid is None and not review_existing else owner_pid
    if type(owner_pid) is not int or not 0 < owner_pid < 1 << 32:
        raise ValueError("actual recording owner PID required")
    if list(stages) != list(SUCCESS_STAGES):
        raise ValueError("complete ordered native Windows stages required")
    for name, process in stages.items():
        if (
            not isinstance(process, dict)
            or type(process.get("process_id")) is not int
            or not 0 < process["process_id"] < 1 << 32
        ):
            raise ValueError("actual Windows stage process identity required")
        expected = plan["commands"][name]
        if name == "inspect_etl":
            expected = [*expected, "--pid", str(stages["workload"]["process_id"])]
        if process.get("command") != expected or not (
            instance_absent(process) if name == "status" else capture_succeeded(process)
        ):
            raise ValueError("Windows stage command/outcome differs from owned plan: " + name)
    validate_preflight_stages(plan, stages)

    def publish(name, value):
        if review_existing:
            path = out / name
            _, retained = read_retained(path, store)
            if not same_evidence_value(retained, value):
                raise ValueError("Windows derived artifact differs from raw inputs: " + name)
            return reference(path)
        return write_diagnostic(
            out / name,
            value,
            root=out,
            total_bytes=plan["budgets"]["max_artifact_bytes"],
            minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
        )

    if not same_evidence_value(plan, rebuild(plan, review_existing=True)):
        raise ValueError("Windows profile source/tool/production/symbol inputs changed")
    _, production = store.json(plan["production_capture"], ROOT)
    _, raw = read_retained(out / "raw/benchmark.json", store)
    validate_capture(raw)
    process = stages["workload"]
    if type(raw.get("process_id")) is not int or raw["process_id"] != process["process_id"]:
        raise ValueError("Windows benchmark document belongs to another process")
    metadata_stage = stages["inspect_etl"]
    if metadata_stage["command"] != [
        *plan["commands"]["inspect_etl"],
        "--pid",
        str(raw["process_id"]),
    ]:
        raise ValueError("native ETL metadata command did not select the owned benchmark PID")
    native_metadata = parse_evidence_json(metadata_stage["stdout"])
    process_identity = process_contract(
        native_metadata,
        pid=raw["process_id"],
        parent_pid=owner_pid,
        executable=plan["benchmark"]["path"],
        decoder_pid=metadata_stage["process_id"],
        source_sha256=etl_source_identity(),
        file_bytes=(out / "raw/profile.etl").stat().st_size,
    )
    images = image_catalog(native_metadata, process_identity)
    ranges = (
        bind_operation_ranges(native_metadata, process_identity, raw.get("profiler_ranges"))
        if plan["operation_ranges"]
        else None
    )
    native_cpu = cpu_catalog(
        native_metadata, process_identity, snapshot=raw.get("profiler_ranges"), native_ranges=ranges
    )
    schedule = (
        schedule_catalog(
            native_metadata,
            process_identity,
            native_cpu,
            snapshot=raw.get("profiler_ranges"),
            native_ranges=ranges,
        )
        if plan["scheduling"]
        else None
    )
    profiled = bind_host_profile_output(
        production,
        raw,
        workload=plan["commands"]["workload"],
        environment=plan["expected_native_environment"],
        executable=Path(plan["benchmark"]["path"]),
        execution_id=str(uuid.UUID(hex=plan["instance"][5:])),
    )
    identity = validate_document(
        parse_evidence_json(stages["identity"]["stdout"]), production["backend_selected"]
    )
    if identity["build_manifest_sha256"] != canonical_digest(production["build_manifest"]):
        raise ValueError("Windows native build identity differs from the production manifest")
    native_id = parse_evidence_json(stages["identity"]["stdout"])
    if (
        type(native_id.get("process_id")) is not int
        or native_id["process_id"] != stages["identity"]["process_id"]
    ):
        raise ValueError("Windows identity JSON belongs to another process")
    identity_path = out / "identity.json"
    publish("identity.json", native_id)
    compiled = bind_document(
        Path(plan["benchmark"]["path"]),
        identity_path,
        before_sha256=plan["benchmark"]["sha256"],
        backend=production["backend_selected"],
    )
    exported = owned_artifact_files(out / "export", out)
    csv_files = [path for path in exported if path.suffix.lower() == ".csv"]
    if len(csv_files) != 1:
        raise ValueError("one native CPU-sample CSV is required from the reviewed WPA profile")
    samples = read_samples(
        csv_files[0], pid=raw["process_id"], executable=plan["benchmark"]["path"]
    )
    profiled["external_instrumentation"] = dict(
        schema="rns8_external_instrumentation_v1",
        kind="host_profiler",
        tool="WPR_WPA",
        mode="cpu_sampling",
    )
    publish("samples.json", samples)
    publish("native-process-metadata.json", native_metadata)
    publish("native-image-catalog.json", images)
    publish("native-cpu-observations.json", native_cpu)
    if schedule is not None:
        publish("native-schedule-observations.json", schedule)
    if ranges is not None:
        publish("native-operation-ranges.json", ranges)
    publish("profiled.json", profiled)
    store.finish()
    return dict(
        qualification="native_capture_export_bound_attribution_partial",
        compiled_binary_identity=compiled,
        native_etl=reference(out / "raw/profile.etl"),
        native_export=reference(csv_files[0]),
        samples=reference(out / "samples.json"),
        profiled_capture=reference(out / "profiled.json"),
        symbols=plan["symbols"],
        native_process_metadata=reference(out / "native-process-metadata.json"),
        native_image_catalog=reference(out / "native-image-catalog.json"),
        native_cpu_observations=reference(out / "native-cpu-observations.json"),
        native_schedule_observations=reference(out / "native-schedule-observations.json")
        if schedule is not None
        else None,
        selected_process_identity=process_identity,
        sample_loss=process_identity["header_loss"],
        clock_join="unbound",
        measured_operation_range="not_requested" if ranges is None else ranges["state"],
        native_operation_ranges=None
        if ranges is None
        else reference(out / "native-operation-ranges.json"),
        production_latency=False,
        promotion_allowed=False,
    )


def execute_capture(
    plan,
    *,
    authorize_system_recording=False,
    environment=None,
    runner=run_capture,
    clock=time.monotonic,
):
    if platform.system() != "Windows" or authorize_system_recording is not True:
        raise ValueError(
            "Windows recording needs explicit --execute --authorize-system-recording; no elevation is launched"
        )
    if not same_evidence_value(plan, rebuild(plan)):
        raise ValueError("Windows profile plan changed before execution")
    out = Path(plan["output_directory"])
    parent = next(path for path in out.parents if path.exists())
    if shutil.disk_usage(parent).free < plan["budgets"]["minimum_free_bytes"]:
        raise ValueError("insufficient free space for Windows profile")
    child = {
        key: value
        for key, value in dict(os.environ if environment is None else environment).items()
        if not relevant_environment_key(key)
        and key not in {"_NT_SYMBOL_PATH", "_NT_ALT_SYMBOL_PATH", "_NT_SYMCACHE_PATH"}
    }
    child.update(plan["expected_native_environment"])
    child.update(plan["symbol_search_environment"])
    out.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "export", "symbols"):
        (out / name).mkdir()
    write_json(out / "plan.json", plan)
    stages, failure, possibly_owned = {}, None, False
    deadline = clock() + plan["budgets"]["total_seconds"]

    def call(name, *, cleanup=False, required=True):
        remaining = (
            deadline
            - clock()
            - (0 if cleanup or name == "export" else plan["budgets"]["cleanup_reserve_seconds"])
        )
        if remaining <= 0:
            raise ValueError("Windows profiling total deadline exhausted before " + name)
        command = plan["commands"][name]
        if name == "inspect_etl":
            command = [*command, "--pid", str(stages["workload"]["process_id"])]
        result = runner(
            command,
            cwd=ROOT,
            env=child,
            label="windows-host-profile:" + name,
            timeout_seconds=min(15 if cleanup else plan["budgets"]["timeout_seconds"], remaining),
            max_output_bytes=MAX_ETL_METADATA_BYTES if name == "inspect_etl" else 4 * 1024**2,
            artifact_roots=() if cleanup else (out,),
            max_artifact_bytes=plan["budgets"]["max_artifact_bytes"],
            return_on_interrupt=True,
        )
        stages[name] = result
        write_json(out / (name + ".process.json"), result)
        if result.get("interrupted"):
            raise InterruptedError("owned Windows profiling child interrupted")
        if required and not capture_succeeded(result):
            raise ValueError("Windows profiling stage failed: " + name)
        return result

    try:
        for name in ("wpr_help", "exporter_help", "profiles", "identity", "inspector_identity"):
            call(name)
        validate_preflight_stages(plan, stages)
        if not instance_absent(call("status", required=False)):
            raise ValueError("native WPR did not prove the unique instance absent")
        try:
            possibly_owned = True  # Arm before start: partial startup also needs owned cleanup.
            call("start")
            call("workload")
        finally:
            if possibly_owned:
                try:
                    stop = call("stop", cleanup=True, required=False)
                    possibly_owned = not capture_succeeded(stop)
                finally:
                    if possibly_owned:
                        call("cancel", cleanup=True, required=False)
        if not capture_succeeded(stages.get("stop", {})):
            raise ValueError("owned WPR stop/trace export did not complete")
        etl_before = reference(out / "raw/profile.etl")
        call("inspect_etl")
        call("export")
        if not same_evidence_value(etl_before, reference(out / "raw/profile.etl")):
            raise ValueError("native ETL changed during metadata decoding/export")
        result = completed_capture(plan, stages)
    except (OSError, ValueError, KeyError, TypeError, InterruptedError, KeyboardInterrupt) as error:
        failure = str(error) or type(error).__name__
        result = dict(
            qualification="incomplete",
            error=failure,
            raw_artifacts_preserved=True,
            production_latency=False,
            promotion_allowed=False,
        )
    result.update(
        recording_owner_pid=os.getpid(),
        first_failed_stage=next(
            (
                name
                for name, row in stages.items()
                if not capture_succeeded(row) and not (name == "status" and instance_absent(row))
            ),
            None,
        ),
        plan=reference(out / "plan.json"),
        instance=plan["instance"],
        stages=[
            dict(name=name, receipt=reference(out / (name + ".process.json")))
            for name in stages
            if (out / (name + ".process.json")).is_file()
        ],
        recording_cleanup="stop_completed"
        if capture_succeeded(stages.get("stop", {}))
        else "cancel_completed"
        if capture_succeeded(stages.get("cancel", {}))
        else "not_started_or_cleanup_unconfirmed",
    )
    if failure:
        write_json(out / "host-profile.json", seal(result))
    else:
        try:
            write_diagnostic(
                out / "host-profile.json",
                seal(result),
                root=out,
                total_bytes=plan["budgets"]["max_artifact_bytes"],
                minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
            )
        except (OSError, ValueError, TypeError) as error:
            failure = str(error)
            result.update(qualification="incomplete", error=failure, raw_artifacts_preserved=True)
            write_json(out / "host-profile.json", seal(result))
    return 1 if failure else 0


def located(name):
    found = find_command(name)
    if found is None:
        raise ValueError("Windows profiler tool unavailable; select explicit --" + name)
    return Path(found)


def review_capture(plan_path, *, runner=run_capture):
    """Fresh file-only native decode and derived receipt replay; no recorder."""
    if platform.system() != "Windows":
        raise ValueError("fresh native ETL review requires the bound Windows inspector")
    plan_path = Path(plan_path).resolve(strict=True)
    plan_text, plan_ref = bounded_read(plan_path)
    plan_ref["bytes"] = len(plan_text.encode("utf-8"))
    plan = parse_evidence_json(plan_text)
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError("maintained Windows acquisition plan required")
    out = Path(plan["output_directory"])
    if plan_path != out / "plan.json" or not same_evidence_value(
        plan, rebuild(plan, review_existing=True)
    ):
        raise ValueError("Windows review plan location or source inputs changed")
    files = owned_artifact_files(out, out.parent)
    if sum(path.stat().st_size for path in files) > plan["budgets"]["max_artifact_bytes"]:
        raise ValueError("retained Windows capture exceeds its owned artifact budget")
    store = EvidenceStore()
    store.reference(plan_ref, ROOT)
    summary_ref, summary = read_retained(out / "host-profile.json", store)
    if (
        not isinstance(summary, dict)
        or summary.get("qualification") != "native_capture_export_bound_attribution_partial"
        or summary.get("identity_sha256") != seal(dict(summary))["identity_sha256"]
        or not same_evidence_value(summary.get("plan"), plan_ref)
        or summary.get("instance") != plan["instance"]
        or summary.get("first_failed_stage") is not None
        or summary.get("recording_cleanup") != "stop_completed"
    ):
        raise ValueError("successful owned Windows acquisition receipt required")
    references = summary.get("stages")
    if not isinstance(references, list) or [
        row.get("name") if isinstance(row, dict) else None for row in references
    ] != list(SUCCESS_STAGES):
        raise ValueError("Windows acquisition stage inventory changed")
    stages = {}
    for row in references:
        name = row["name"]
        expected = reference(out / (name + ".process.json"))
        if set(row) != {"name", "receipt"} or not same_evidence_value(row["receipt"], expected):
            raise ValueError("Windows acquisition stage receipt changed or moved")
        actual, stages[name] = read_retained(out / (name + ".process.json"), store)
        if not same_evidence_value(actual, expected):
            raise ValueError("Windows stage changed while opening its receipt")
    expected = completed_capture(
        plan, stages, owner_pid=summary.get("recording_owner_pid"), review_existing=True
    )
    wrapper_keys = {
        "recording_owner_pid",
        "first_failed_stage",
        "plan",
        "instance",
        "stages",
        "recording_cleanup",
        "identity_sha256",
    }
    if set(summary) != set(expected) | wrapper_keys or any(
        not same_evidence_value(summary.get(key), value) for key, value in expected.items()
    ):
        raise ValueError(
            "Windows acquisition summary differs from freshly reconstructed raw inputs"
        )
    store.reference(expected["native_export"], ROOT)
    etl = reference(out / "raw/profile.etl")
    store.reference(etl, ROOT)
    command = plan["commands"]["inspect_etl"] + ["--pid", str(stages["workload"]["process_id"])]
    # Decode only the retained finalized file. No WPR, exporter, workload,
    # service, symbol download or persistent configuration command is issued.
    decoded = runner(
        command,
        cwd=ROOT,
        env={"SystemRoot": os.environ.get("SystemRoot", r"C:\Windows")},
        label="windows-host-review:inspect_etl",
        timeout_seconds=min(120, plan["budgets"]["timeout_seconds"]),
        max_output_bytes=MAX_ETL_METADATA_BYTES,
        return_on_interrupt=True,
    )
    if (
        not capture_succeeded(decoded)
        or decoded.get("command") != command
        or type(decoded.get("process_id")) is not int
        or not 0 < decoded["process_id"] < 1 << 32
    ):
        raise ValueError("fresh native Windows file-only decoder did not complete")
    fresh = parse_evidence_json(decoded["stdout"])
    retained = parse_evidence_json(stages["inspect_etl"]["stdout"])
    if (
        not isinstance(fresh, dict)
        or type(fresh.get("decoder_pid")) is not int
        or fresh["decoder_pid"] != decoded["process_id"]
    ):
        raise ValueError("fresh ETL decoder result belongs to another process")
    # The decoder process itself changes on replay; every ETL-derived field,
    # compiled source identity and input size must remain identical.
    if not same_evidence_value(dict(fresh, decoder_pid=retained["decoder_pid"]), retained):
        raise ValueError("fresh native ETL decode differs from retained observations")
    if not same_evidence_value(plan, rebuild(plan, review_existing=True)):
        raise ValueError("Windows source/tool/production inputs changed during review")
    store.finish()
    return seal(
        dict(
            schema="rns8_windows_host_profile_review_v1",
            acquisition=summary_ref,
            plan=plan_ref,
            qualification="fresh_file_decode_and_derived_receipts_replayed_attribution_partial",
            native_etl=etl,
            native_decoder=plan["tools"]["inspector"],
            decoder_process_id=decoded["process_id"],
            measured_operation_range=expected["measured_operation_range"],
            recording_started=False,
            workload_started=False,
            outputs_rewritten=False,
            production_latency=False,
            promotion_allowed=False,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path)
    parser.add_argument("--pdb", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--review-plan",
        type=Path,
        help="Replay retained receipts and freshly decode the finalized ETL; no recording",
    )
    parser.add_argument("--wpr", type=Path)
    parser.add_argument("--wpaexporter", type=Path)
    parser.add_argument("--etl-inspector", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--scheduling",
        action="store_true",
        help="Opt into native CSwitch/ReadyThread observation; requires system-recording authorization",
    )
    parser.add_argument(
        "--operation-ranges",
        action="store_true",
        help="Enable compiled opt-in ETW markers and bind native measured windows",
    )
    parser.add_argument("--authorize-system-recording", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.review_plan:
            if any(
                (
                    args.production,
                    args.pdb,
                    args.out,
                    args.wpr,
                    args.wpaexporter,
                    args.etl_inspector,
                    args.execute,
                    args.operation_ranges,
                    args.scheduling,
                    args.authorize_system_recording,
                )
            ):
                raise ValueError("--review-plan cannot be combined with acquisition options")
            print(json.dumps(review_capture(args.review_plan), indent=2))
            return 0
        if not all((args.production, args.pdb, args.out)):
            raise ValueError("acquisition requires --production, --pdb and --out")
        plan = plan_capture(
            args.production,
            args.wpr or located("wpr"),
            args.wpaexporter or located("wpaexporter"),
            args.pdb,
            args.out,
            inspector=args.etl_inspector,
            operation_ranges=args.operation_ranges,
            scheduling=args.scheduling,
        )
        if args.execute:
            return execute_capture(plan, authorize_system_recording=args.authorize_system_recording)
        print(json.dumps(plan, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("Windows host profile: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
