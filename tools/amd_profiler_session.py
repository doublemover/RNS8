#!/usr/bin/env python3
"""Plan one bounded native AMD timeline capture; workload execution is opt-in."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import sys
import uuid

from amd_profiler_capabilities import validate_preflight
from amd_profiler_evidence import SCHEMA as DISPATCH_SCHEMA, validate_dispatch_receipt
from amd_rocpd import read_process
from amd_profiler_ranges import join as join_native_ranges
from amd_counter_pass import counter_names, summarize as summarize_counter_pass
from native_allocation_lifetimes import summarize as summarize_allocations
from build_manifest import configuration_options
from benchmark_request import (
    attach_capture_identity,
    file_digest,
    request_identity,
)
from compiled_binary_identity import canonical_digest
from device_observation import (
    capture_observation,
    inspection_observation,
    observation_relation,
)
from evidence_database_lib.isa import (
    isa_index_key,
    lookup_isa_resources,
    summarize_isa_report,
)
from execution_observation import execution_observation
from execution_environment_contract import relevant_environment_key
from evidence_records import ArtifactReference
from gpu_qualification import ROOT, execute_plan, owned_artifact_files, write_json
from json_evidence import parse_evidence_json, same_evidence_value
from process_progress import capture_succeeded
from target_qualification import EvidenceStore, session_stage
from profiler_contract import production_profile_input
from profiler_artifacts import write_diagnostic
from timing_eligibility import capture_identity

SCHEMA = "rns8_amd_profiler_session_v1"
SOURCE_INPUTS = (
    "tools/amd_profiler_session.py",
    "tools/amd_profiler_evidence.py",
    "tools/amd_profiler_device.py",
    "tools/amd_profiler_capabilities.py",
    "tools/amd_rocpd.py",
    "tools/amd_profiler_ranges.py",
    "tools/amd_counter_pass.py",
    "tools/native_phase_attribution.py",
    "tools/native_marker_clock.py",
    "tools/native_dependency_report.py",
    "tools/native_allocation_lifetimes.py",
    "tools/host_range_contract.py",
    "tools/interval_observations.py",
    "tools/profiler_contract.py",
    "tools/profiler_artifacts.py",
    "tools/execution_observation.py",
    "tools/execution_host_identity.py",
    "tools/execution_environment_contract.py",
    "tools/benchmark_request.py",
    "tools/build_manifest.py",
    "tools/compiled_binary_identity.py",
    "tools/device_observation.py",
    "tools/benchmark_schema.py",
    "tools/evidence_database_lib/isa.py",
    "tools/timing_eligibility.py",
    "tools/target_qualification.py",
    "tools/gpu_qualification.py",
    "tools/json_evidence.py",
    "tools/process_progress.py",
    "src/diagnostics/trace_ranges.cpp",
    "src/diagnostics/trace_label.hpp",
)
TRACE_FLAGS = (
    "--hip-trace",
    "--kernel-trace",
    "--memory-copy-trace",
    "--memory-allocation-trace",
    "--scratch-memory-trace",
    "--marker-trace",
)
TRACE_VARIABLES = (
    "HIP_COMPILER_API_TRACE",
    "HIP_RUNTIME_API_TRACE",
    "HIP_GRAPH_TRACE",
    "HIP_EVENT_TRACE",
    "KERNEL_TRACE",
    "MEMORY_COPY_TRACE",
    "MEMORY_ALLOCATION_TRACE",
    "SCRATCH_MEMORY_TRACE",
    "MARKER_API_TRACE",
)


def reference(path: Path) -> ArtifactReference:
    path = path.resolve(strict=True)
    return dict(path=str(path), sha256=file_digest(path))


def _inputs(plan: dict, store: EvidenceStore) -> tuple[dict, dict, Path, dict]:
    _, capture = store.json(plan["production_capture"], ROOT)
    executable = production_profile_input(capture, store)
    preflight = validate_preflight(plan["capabilities"], store=store, parent=ROOT)
    target = capture["device"]["gcn_arch"].split(":")[0]
    feature = preflight["features"]["timeline"]
    if (
        preflight["system"] != "Linux"
        or preflight["target"] != target
        or feature["official_support"] != "supported"
        or feature["advertised"] != "available"
        or feature["permission"] == "denied"
        or preflight["tools"]["rocprofv3"]["version_state"] != "supported"
    ):
        raise ValueError("no maintained, advertised Linux native timeline capability")
    isa_path, isa = store.json(plan["selected_static_isa"], ROOT)
    backend = capture["backend_selected"]
    resources = lookup_isa_resources(
        {isa_index_key(backend, target): [summarize_isa_report(isa_path, isa)]},
        backend,
        target,
        capture=capture,
    )
    if not (resources.get("isa_association") or {}).get("qualified"):
        raise ValueError(
            "selected static ISA is not associated with production artifact/target/kernel"
        )
    return capture, preflight, executable, isa


def timeline_libraries(tool: Path) -> tuple[Path, ...]:
    root = tool.resolve(strict=True).parent.parent
    return (
        root / "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so",
        root / "lib/librocprofiler-sdk.so",
        root / "lib/librocprofiler-sdk-roctx.so",
    )


def timeline_environment(
    baseline: dict[str, str], tool: Path, directory: Path, *, counters=()
) -> dict[str, str]:
    """Pinned 1.4.1 single-pass launcher contract, not a permissive prefix mask."""
    root = tool.resolve(strict=True).parent.parent
    libraries = timeline_libraries(tool)
    for library in libraries:
        if not library.is_file():
            raise ValueError("required profiler injection library is absent: " + str(library))
    result = dict(baseline)
    result.update(
        LD_PRELOAD=":".join(str(path) for path in libraries),
        ROCP_TOOL_LIBRARIES=str(libraries[0]),
        LD_LIBRARY_PATH=(baseline.get("LD_LIBRARY_PATH", "") + ":" + str(root / "lib")).strip(":"),
        ROCPROFILER_LIBRARY_CTOR="1",
        ROCPROF_OUTPUT_PATH=str(directory),
        ROCPROF_OUTPUT_FILE_NAME="rns8",
        ROCPROF_OUTPUT_FORMAT="rocpd",
    )
    result.update({"ROCPROF_" + name: "1" for name in TRACE_VARIABLES})
    names = counter_names(counters)
    if names:
        result.update(ROCPROF_COUNTER_COLLECTION="1", ROCPROF_COUNTERS="pmc: " + " ".join(names))
    return result


def plan_capture(
    production: Path,
    capabilities: Path,
    selected_isa: Path,
    inspector: Path,
    out: Path,
    *,
    timeout_seconds=300.0,
    total_seconds=600.0,
    max_artifact_bytes=2 * 1024**3,
    max_output_bytes=16 * 1024**2,
    minimum_free_bytes=5 * 1024**3,
    operation_ranges=False,
    counters=(),
    _review_existing=False,
) -> dict:
    for value in (timeout_seconds, total_seconds):
        if type(value) not in (int, float) or not 0 < value <= 86400 or not math.isfinite(value):
            raise ValueError("positive finite acquisition time budgets within24hours required")
    if total_seconds <= timeout_seconds + 60:
        raise ValueError("session budget must include query stages and a 60-second export reserve")
    if type(operation_ranges) is not bool:
        raise ValueError("native operation ranges require explicit Boolean policy")
    for value in (max_artifact_bytes, max_output_bytes, minimum_free_bytes):
        if type(value) is not int or value < 1:
            raise ValueError("positive byte budgets required")
    out = out.resolve()
    if not out.is_relative_to((ROOT / "temp").resolve()) or out == (ROOT / "temp").resolve():
        raise ValueError("profiler sessions must use an ignored child of repository temp/")
    if out.exists() and not _review_existing:
        raise ValueError("profiler session output must be previously absent")
    store = EvidenceStore()
    inputs = dict(
        production_capture=reference(production),
        capabilities=reference(capabilities),
        selected_static_isa=reference(selected_isa),
        inspector=reference(inspector),
    )
    capture, preflight, executable, _ = _inputs(inputs, store)
    names = counter_names(counters)
    if names:
        feature = preflight["features"]["counters"]
        if (
            list(names) != preflight["requested_counters"]
            or feature["official_support"] != "supported"
            or feature["advertised"] != "available"
            or feature["permission"] == "denied"
            or type(preflight["profiler_agent_selector"]) is not int
        ):
            raise ValueError(
                "counter group lacks its exact native preflight collectability observation"
            )
    if (
        operation_ranges
        and configuration_options(capture["build_manifest"]["configuration"], strict=True).get(
            "roctx"
        )
        != "ON"
    ):
        raise ValueError("native operation matching needs the same binary compiled with ROCTx")
    inspector = store.reference(inputs["inspector"], ROOT)
    tool = Path(preflight["tools"]["rocprofv3"]["path"]).resolve(strict=True)
    # A bounded explicit single execution must not inherit pass/replay controls.
    baseline = capture["request_identity"]["environment"]
    expected = timeline_environment(baseline, tool, out / "native", counters=names)
    if operation_ranges:
        expected["RNS8_PROFILE_RANGES"] = "1"
    library_receipts = [reference(path) for path in timeline_libraries(tool)]
    arguments = list(capture["request_identity"]["arguments"])
    arguments += ["--json-output", str(out / "native/benchmark.json")]
    if "--record-execution-environment" not in arguments:
        arguments.append("--record-execution-environment")
    command = [
        str(tool),
        *TRACE_FLAGS,
        *(["--pmc", *names] if names else []),
        "--output-format",
        "rocpd",
        "-d",
        str(out / "native"),
        "-o",
        "rns8",
        "--",
        str(executable),
        *arguments,
    ]
    _, raw_preflight = store.json(inputs["capabilities"], ROOT)
    help_rows = [row for row in raw_preflight["observations"] if row["name"] == "help"]
    if len(help_rows) != 1 or any(
        flag not in help_rows[0]["result"]["stdout"]
        for flag in (*TRACE_FLAGS, *(("--pmc",) if names else ()))
    ):
        raise ValueError("selected rocprofv3 help does not advertise the maintained timeline flags")
    version_rows = [row for row in raw_preflight["observations"] if row["name"] == "version"]
    if len(version_rows) != 1:
        raise ValueError("one retained tool version probe required")
    version_path = Path(inputs["capabilities"]["path"]).parent / "version.stdout.log"
    version_output = reference(version_path)
    if "1.4.1" not in version_path.read_text(encoding="utf-8"):
        # Do not rewrite or merge raw logs when a release prints its version to stderr.
        version_output = reference(version_path.with_name("version.stderr.log"))
    backend, target = (
        capture["backend_selected"],
        capture["device"]["gcn_arch"].split(":")[0],
    )
    commands = []

    def add(name, argv, **checks):
        commands.append(
            dict(
                name=name,
                argv=argv,
                depends_on=[commands[-1]["name"]] if commands else [],
                timeout_seconds=timeout_seconds,
                **checks,
            )
        )

    for name, binary in (
        ("benchmark-identity", executable),
        ("inspector-identity", inspector),
    ):
        add(
            name,
            [str(binary), "--build-identity"],
            compiled_identity=True,
            identity_backends=[backend],
            identity_output=str(out / (name + ".json")),
        )
    ordinal = capture_observation(capture).identity.runtime_ordinal
    add(
        "device-inspection",
        [str(inspector), "--backend", backend, "--device", str(ordinal), "--json"],
        inspect_target=target,
        inspect_backend=backend,
        inspect_device=ordinal,
        require_physical_identity=True,
    )
    add(
        "timeline",
        command,
        owned_artifact_directories=[str(out / "native")],
        evidence_outputs=[str(out / "native/benchmark.json")],
        max_artifact_bytes=max_artifact_bytes,
        max_output_bytes=max_output_bytes,
    )
    plan = dict(
        schema=SCHEMA,
        schema_version=4,
        session_kind="amd-profiler-timeline",
        system="Linux",
        command_working_directory=str(ROOT),
        output_directory=str(out),
        **inputs,
        source_inputs=[reference(ROOT / name) for name in SOURCE_INPUTS],
        operation_ranges=operation_ranges,
        requested_counters=list(names),
        counter_policy="one_explicit_group_one_application_execution_no_kernel_replay",
        profiler_tool=reference(tool),
        injection_libraries=library_receipts,
        tool_version_output=version_output,
        production_capture_identity=capture_identity(capture),
        commands=commands,
        devices=[],
        inherited_environment=copy.deepcopy(baseline),
        expected_native_environment=expected,
        device_namespace="same_admitted_visibility_as_production_not_unmasked_parent_ordinals",
        profiler_agent_selector=preflight["profiler_agent_selector"],
        budgets=dict(
            total_seconds=total_seconds,
            export_reserve_seconds=60,
            minimum_free_bytes=minimum_free_bytes,
        ),
        qualification="plan_only",
        promotion_allowed=False,
    )
    plan["identity_sha256"] = canonical_digest(plan)
    store.finish()
    return plan


def _rebuild(plan: dict, *, review_existing=False) -> dict:
    return plan_capture(
        *(
            Path(plan[key]["path"])
            for key in (
                "production_capture",
                "capabilities",
                "selected_static_isa",
                "inspector",
            )
        ),
        Path(plan["output_directory"]),
        timeout_seconds=plan["commands"][-1]["timeout_seconds"],
        total_seconds=plan["budgets"]["total_seconds"],
        max_artifact_bytes=plan["commands"][-1]["max_artifact_bytes"],
        max_output_bytes=plan["commands"][-1]["max_output_bytes"],
        minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
        operation_ranges=plan["operation_ranges"],
        counters=plan["requested_counters"],
        _review_existing=review_existing,
    )


def validate_acquisition(
    receipt: dict,
    capture: dict,
    store: EvidenceStore,
    parent: Path,
    *,
    native_projection: dict | None,
) -> dict:
    """Bind acquisition claims to the maintained plan and its actual raw stages."""
    proof = receipt["acquisition_session"]
    manifest_path, plan = store.json(proof["session_manifest"], parent)
    out = manifest_path.parent
    if not same_evidence_value(plan, _rebuild(plan, review_existing=True)):
        raise ValueError("retained profiler acquisition differs from maintained command policy")
    if plan["production_capture_identity"] != capture_identity(capture):
        raise ValueError("acquisition session belongs to another production capture")
    item, row, _, identity_row, stages = session_stage(proof, parent, store)
    if proof["stage"] != "timeline" or proof["binary_identity_stage"] != "benchmark-identity":
        raise ValueError("profiler receipt selects unrelated acquisition stages")
    if set(stages) != {command["name"] for command in plan["commands"]}:
        raise ValueError("acquisition stages are incomplete")
    documents = {
        name: _stage_stdout(command, observed, out, store)
        for name, (command, observed) in stages.items()
    }
    if receipt["acquisition_command"] != item["argv"]:
        raise ValueError("dispatch command differs from launched acquisition")
    for name in ("benchmark-identity", "inspector-identity"):
        binary = store.binary(
            stages[name][1]["compiled_binary_identity"], capture["backend_selected"]
        )
        if binary["build_manifest_sha256"] != canonical_digest(capture["build_manifest"]):
            raise ValueError("acquisition executable/library differs from production manifest")
        if documents[name].get("process_id") != stages[name][1]["process_id"]:
            raise ValueError("acquisition compiled query belongs to another process")
    binary = store.binary(identity_row["compiled_binary_identity"], capture["backend_selected"])
    if binary["executable_sha256"] != capture["artifact_capture"]["executable_sha256"]:
        raise ValueError("acquisition benchmark differs from production executable")
    live = inspection_observation(documents["device-inspection"])
    if (
        documents["device-inspection"].get("process_id")
        != stages["device-inspection"][1]["process_id"]
    ):
        raise ValueError("acquisition runtime inspection belongs to another process")
    if observation_relation(capture_observation(capture), live).state != "verified":
        raise ValueError("acquisition inspected a different production device")
    _, profiled = store.json(receipt["profiled_capture"], parent)
    observed = execution_observation(profiled, required=True)
    if (
        profiled.get("process_id") != row["process_id"]
        or observation_relation(capture_observation(profiled), live).state != "verified"
        or not same_evidence_value(observed.environment, plan["expected_native_environment"])
    ):
        raise ValueError("profiled process/device/environment differs from acquired observations")
    for key in ("native_trace", "native_benchmark_output"):
        path = store.reference(receipt[key], parent)
        if row["artifacts"].get(str(path)) != receipt[key]["sha256"]:
            raise ValueError("native artifact was not retained by the acquisition stage")
    for key in ("capabilities", "selected_static_isa", "tool_version_output"):
        if receipt[key] != plan[key]:
            raise ValueError("dispatch attachment differs from admitted acquisition input")
    if receipt["tool_executable"] != plan["profiler_tool"]:
        raise ValueError("dispatch tool differs from admitted acquisition tool")
    if native_projection is None or "timeline" not in native_projection:
        raise ValueError("acquisition needs the receiver's freshly decoded native timeline")
    _, retained_timeline = store.json(receipt.get("native_timeline"), parent)
    timeline = native_projection["timeline"]
    if not same_evidence_value(retained_timeline, timeline):
        raise ValueError("retained native timeline differs from the actual database")
    operation = (
        join_native_ranges(timeline, profiled.get("profiler_ranges"))
        if plan["operation_ranges"]
        else dict(state="not_requested", production_latency=False)
    )
    _, retained_operation = store.json(receipt.get("operation_join"), parent)
    if not same_evidence_value(retained_operation, operation):
        raise ValueError("retained marker operation join differs from actual native evidence")
    _, retained_allocations = store.json(receipt.get("native_allocations"), parent)
    if not same_evidence_value(retained_allocations, summarize_allocations(timeline)):
        raise ValueError("retained allocation generations differ from actual native evidence")
    _, retained_counters = store.json(receipt.get("counter_pass"), parent)
    if not same_evidence_value(
        retained_counters,
        summarize_counter_pass(
            native_projection, receipt["selected_dispatches"], plan["requested_counters"]
        ),
    ):
        raise ValueError("retained counter pass differs from actual native evidence")
    for item in [
        plan["inspector"],
        plan["profiler_tool"],
        *plan["injection_libraries"],
        *plan["source_inputs"],
    ]:
        store.reference(item, ROOT)
    return dict(
        session_id=proof["session_id"],
        run_id=proof["run_id"],
        native_process_id=row["process_id"],
        physical_device=live.to_json(),
        native_timeline=receipt["native_timeline"],
        operation_join=receipt["operation_join"],
        native_allocations=receipt["native_allocations"],
        counter_pass=receipt["counter_pass"],
        scope="bounded_selected_dispatch_acquisition_not_complete_profile_coverage",
    )


def _stage_stdout(item: dict, row: dict, out: Path, store: EvidenceStore) -> dict:
    if (
        not capture_succeeded(row)
        or row.get("validation_errors")
        or row.get("command") != item["argv"]
        or type(row.get("process_id")) is not int
        or row["process_id"] < 1
    ):
        raise ValueError("acquisition stage lacks successful actual process evidence")
    for stream in ("stdout", "stderr"):
        raw = out / (item["name"] + "." + stream + ".log")
        store.verify(raw, row["artifacts"][str(raw)])
        if raw.read_text(encoding="utf-8") != row[stream]:
            raise ValueError("stage summary differs from retained raw process output")
    return parse_evidence_json(row["stdout"]) if item["name"] != "timeline" else {}


def selected_dispatches(native: dict, symbols: set[str], target: str) -> list[dict]:
    """Select exact native symbols on the observed target; validation joins loads."""
    kernels = {row["kernel_id"]: row for row in native["kernel_symbols"]}
    agents = {row["id"]["handle"]: row for row in native["agents"]}
    result = []
    for record in native["buffer_records"]["kernel_dispatch"]:
        info = record["dispatch_info"]
        agent_id = info["agent_id"]["handle"]
        kernel = kernels.get(info["kernel_id"], {})
        if kernel.get("kernel_name") in symbols and agents.get(agent_id, {}).get("name") == target:
            result.append(
                dict(
                    agent_id=agent_id,
                    queue_id=info["queue_id"]["handle"],
                    dispatch_id=info["dispatch_id"],
                )
            )
    if not result:
        raise ValueError("native capture contains no selected static mangled symbol on the target")
    return result


def _write_diagnostic(path: Path, value: dict, plan: dict) -> dict:
    return write_diagnostic(
        path,
        value,
        root=Path(plan["output_directory"]),
        total_bytes=plan["commands"][-1]["max_artifact_bytes"],
        minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
    )


def finalize(manifest_path: Path) -> dict:
    """Reopen retained stage bytes and emit a dispatch receipt, never execute work."""
    out = manifest_path.resolve(strict=True).parent
    store = EvidenceStore()
    _, plan = store.json(reference(manifest_path), ROOT)
    capture, preflight, executable, isa = _inputs(plan, store)
    if plan.get("schema") != SCHEMA or plan.get("output_directory") != str(out):
        raise ValueError("wrong profiler session manifest or output directory")
    for item in [
        plan["profiler_tool"],
        plan["inspector"],
        *plan["injection_libraries"],
    ]:
        store.reference(item, ROOT)
    result_path = out / "results.json"
    _, results = store.json(reference(result_path), ROOT)
    proof = dict(
        session_manifest=reference(manifest_path),
        session_results=reference(result_path),
        session_id=plan["identity_sha256"],
        run_id=results["run_id"],
        working_directory=str(ROOT),
        stage="timeline",
        binary_identity_stage="benchmark-identity",
    )
    item, row, _, identity_row, stages = session_stage(proof, ROOT, store)
    if len(results["results"]) != len(plan["commands"]):
        raise ValueError("profiler acquisition session is incomplete")
    documents = {
        name: _stage_stdout(command, observed, out, store)
        for name, (command, observed) in stages.items()
    }
    for name in ("benchmark-identity", "inspector-identity"):
        binary = store.binary(
            stages[name][1]["compiled_binary_identity"], capture["backend_selected"]
        )
        if binary["build_manifest_sha256"] != canonical_digest(capture["build_manifest"]):
            raise ValueError(
                "live companion executable/library differs from production compiled manifest"
            )
        if documents[name].get("process_id") != stages[name][1]["process_id"]:
            raise ValueError("compiled query identity belongs to another process")
    binary = store.binary(identity_row["compiled_binary_identity"], capture["backend_selected"])
    if binary["executable_sha256"] != capture["artifact_capture"]["executable_sha256"]:
        raise ValueError("profiled binary differs from production")
    live = inspection_observation(documents["device-inspection"])
    if (
        documents["device-inspection"].get("process_id")
        != stages["device-inspection"][1]["process_id"]
    ):
        raise ValueError("live runtime inspection belongs to another process")
    if observation_relation(capture_observation(capture), live).state != "verified":
        raise ValueError("live device differs from production physical execution device")
    raw_path = out / "native/benchmark.json"
    raw_reference = dict(path=str(raw_path), sha256=row["artifacts"][str(raw_path)])
    _, raw = store.json(raw_reference, ROOT)
    observed = execution_observation(raw, required=True)
    if raw.get("process_id") != row["process_id"]:
        raise ValueError("single-pass rocprofv3 did not exec the observed benchmark process")
    if observation_relation(capture_observation(raw), live).state != "verified":
        raise ValueError("profiled benchmark device differs from live device inspection")
    if not same_evidence_value(observed.environment, plan["expected_native_environment"]):
        raise ValueError("native environment differs from pinned timeline launcher contract")
    separator = item["argv"].index("--")
    workload = item["argv"][separator + 1 :]
    if list(observed.arguments) != workload[1:]:
        raise ValueError("observed native arguments differ from launched workload")
    request = request_identity(workload, environ=observed.environment)
    # Preserve explicit non-profiler override provenance from the production request.
    request["overrides_sha256"] = capture["request_identity"]["overrides_sha256"]
    request["identity_sha256"] = canonical_digest(
        {k: v for k, v in request.items() if k != "identity_sha256"}
    )
    profiled = copy.deepcopy(raw)
    attach_capture_identity(profiled, request, executable, execution_id=str(uuid.uuid4()))
    profiled["external_instrumentation"] = dict(
        schema="rns8_external_instrumentation_v1",
        kind="amd_profiler",
        tool="rocprofv3",
        mode="kernel_trace",
    )
    outputs = owned_artifact_files(out / "native", out)
    if {str(path) for path in outputs} != {
        name for name in row["artifacts"] if Path(name).is_relative_to(out / "native")
    }:
        raise ValueError("native artifact inventory changed after acquisition")
    databases = [path for path in outputs if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}]
    if len(databases) != 1:
        raise ValueError("one unambiguous finalized native rocpd database required")
    native_path = databases[0]
    projection = read_process(native_path, raw["process_id"], include_timeline=True)
    native = projection["native"]
    selected = selected_dispatches(
        native, set(isa["capture_binding"]["symbols"]), live.architecture.split(":")[0]
    )
    bound_path, receipt_path = out / "profiled.json", out / "dispatch-receipt.json"
    timeline_path, operation_path = out / "native-timeline.json", out / "operation-join.json"
    allocation_path = out / "native-allocations.json"
    counter_path = out / "counter-pass.json"
    if any(
        path.exists()
        for path in (
            bound_path,
            receipt_path,
            timeline_path,
            operation_path,
            allocation_path,
            counter_path,
        )
    ):
        raise ValueError("profiler finalization outputs already exist")
    operation = (
        join_native_ranges(projection["timeline"], raw.get("profiler_ranges"))
        if plan["operation_ranges"]
        else dict(state="not_requested", production_latency=False)
    )
    timeline_ref = _write_diagnostic(timeline_path, projection["timeline"], plan)
    operation_ref = _write_diagnostic(operation_path, operation, plan)
    allocation_ref = _write_diagnostic(
        allocation_path, summarize_allocations(projection["timeline"]), plan
    )
    counter_ref = _write_diagnostic(
        counter_path, summarize_counter_pass(projection, selected, plan["requested_counters"]), plan
    )
    _write_diagnostic(bound_path, profiled, plan)
    injection = {
        key: value
        for key, value in observed.environment.items()
        if capture["request_identity"]["environment"].get(key) != value
    }
    receipt = dict(
        schema=DISPATCH_SCHEMA,
        adapter="rocprofiler-sdk-rocpd-3.0.4-kernel-dispatch-v1",
        production_capture_sha256=capture_identity(capture),
        profiled_capture=reference(bound_path),
        native_benchmark_output=raw_reference,
        native_trace=reference(native_path),
        native_timeline=timeline_ref,
        operation_join=operation_ref,
        native_allocations=allocation_ref,
        counter_pass=counter_ref,
        selected_static_isa=plan["selected_static_isa"],
        profiler_environment=injection,
        selected_dispatches=selected,
        tool_executable=plan["profiler_tool"],
        tool_version_output=plan["tool_version_output"],
        tool_version="1.4.1",
        acquisition_command=item["argv"],
        acquisition_session=proof,
        capabilities=plan["capabilities"],
        device_observation=dict(
            source="native_capture_and_live_inspector_same_admitted_visibility",
            production=capture_observation(capture).to_json(),
            live=live.to_json(),
        ),
        profiler_agent_selector=preflight["profiler_agent_selector"],
        complete_run_coverage=False,
        performance_promotion_allowed=False,
    )
    _write_diagnostic(receipt_path, receipt, plan)
    facts = validate_dispatch_receipt(
        reference(receipt_path), capture=capture, store=store, parent=ROOT
    )
    store.finish()
    return dict(
        receipt=reference(receipt_path),
        facts=facts,
        qualification="selected_dispatches_bound_not_performance_qualification",
    )


def execute_capture(plan: dict, *, environment=None, runner=None) -> int:
    if platform.system() != "Linux":
        raise ValueError("maintained native timeline acquisition requires Linux")
    out = Path(plan["output_directory"])
    # Rebuild policy and reopen inputs immediately before reserving output.
    expected = _rebuild(plan)
    if not same_evidence_value(plan, expected):
        raise ValueError("acquisition plan changed after admission")
    child = {
        key: value
        for key, value in dict(os.environ if environment is None else environment).items()
        if not relevant_environment_key(key)
    }
    child.update(plan["inherited_environment"])
    if plan["operation_ranges"]:
        child["RNS8_PROFILE_RANGES"] = "1"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", plan)
    options = {} if runner is None else dict(runner=runner)
    result = execute_plan(plan, out, child, **options)
    if result:
        return result
    try:
        facts = finalize(out / "manifest.json")
        _write_diagnostic(out / "profile-summary.json", facts, plan)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        write_json(
            out / "profile-summary.json",
            dict(
                qualification="binding_failed",
                error=str(error),
                promotion_allowed=False,
                raw_artifacts_preserved=True,
            ),
        )
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production", type=Path)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--selected-isa", type=Path, required=True)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--total-seconds", type=float, default=600)
    parser.add_argument("--max-artifact-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--max-output-bytes", type=int, default=16 * 1024**2)
    parser.add_argument("--minimum-free-bytes", type=int, default=5 * 1024**3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--operation-ranges", action="store_true")
    parser.add_argument("--counter", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        plan = plan_capture(
            args.production,
            args.capabilities,
            args.selected_isa,
            args.inspector,
            args.out_dir,
            timeout_seconds=args.timeout_seconds,
            total_seconds=args.total_seconds,
            max_artifact_bytes=args.max_artifact_bytes,
            max_output_bytes=args.max_output_bytes,
            minimum_free_bytes=args.minimum_free_bytes,
            operation_ranges=args.operation_ranges,
            counters=args.counter,
        )
        if args.execute:
            return execute_capture(plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
