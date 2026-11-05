"""Strict ROCprofiler-SDK dispatch joins; raw inventory is not performance proof.

Native JSON fields follow the ROCprofiler-SDK 1.4.1 kernel-dispatch schema:
https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
Only file-backed loaded code objects can currently qualify this adapter.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
from typing import Protocol
from urllib.parse import parse_qsl, unquote

from benchmark_schema import validate_capture
from build_manifest import manifest_provenance_complete
from benchmark_request import request_environment_identity
from compiled_binary_identity import canonical_digest
from execution_observation import validate_execution_request
from diagnostic_companion import common_identity
from evidence_database_lib.isa import (
    summarize_isa_report,
    lookup_isa_resources,
    isa_index_key,
)
from json_evidence import JsonValue, same_evidence_value
from reference_verdict import numerical_observation
from timing_eligibility import capture_identity

SCHEMA = "rns8_amd_profiler_dispatch_receipt_v1"
ADAPTERS = {
    "rocprofiler-sdk-json-kernel-dispatch-v1": "json",
    "rocprofiler-sdk-rocpd-3.0.4-kernel-dispatch-v1": "rocpd",
}
PROFILER_ENVIRONMENT = frozenset(
    (
        "ROCP_TOOL_LIBRARIES",
        "LD_PRELOAD",
        # Maintained SDK 1.4.1 rocprofv3 timeline launcher (therock-10.1).
        "LD_LIBRARY_PATH",
        "ROCPROFILER_LIBRARY_CTOR",
        "ROCPROF_OUTPUT_FILE_NAME",
        "ROCPROF_OUTPUT_PATH",
        "ROCPROF_OUTPUT_FORMAT",
        "ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC",
        "ROCPROF_HIP_COMPILER_API_TRACE",
        "ROCPROF_HIP_RUNTIME_API_TRACE",
        "ROCPROF_HIP_GRAPH_TRACE",
        "ROCPROF_HIP_EVENT_TRACE",
        "ROCPROF_MARKER_API_TRACE",
        "ROCPROF_KERNEL_TRACE",
        "ROCPROF_MEMORY_COPY_TRACE",
        "ROCPROF_MEMORY_ALLOCATION_TRACE",
        "ROCPROF_SCRATCH_MEMORY_TRACE",
        "RNS8_PROFILE_RANGES",
    )
)


class EvidenceReader(Protocol):
    def verify(self, path: Path, expected: str) -> Path: ...
    def reference(self, reference: JsonValue, parent: Path) -> Path: ...
    def json(self, reference: JsonValue, parent: Path) -> tuple[Path, dict]: ...


def _object(value: JsonValue) -> dict:
    if not isinstance(value, dict):
        raise ValueError("profiler evidence requires an object")
    return value


def _integer(value: JsonValue) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("native profiler identity/timestamp must be uint64")
    return value


def _index(values: JsonValue, key: str) -> dict[int, dict]:
    if not isinstance(values, list) or not values:
        raise ValueError("native profiler inventory is empty")
    result = {}
    for value in values:
        row = _object(value)
        identifier = _integer(row.get(key))
        if identifier in result:
            raise ValueError("duplicate process-local profiler identity")
        result[identifier] = row
    return result


def _c_integer(value: str) -> int:
    if re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)", value) is None:
        raise ValueError("invalid code-object URI integer")
    return _integer(
        int(
            value,
            16
            if value.lower().startswith("0x")
            else 8
            if value.startswith("0")
            else 10,
        )
    )


def code_object_digest(
    uri: str, executable: Path, executable_sha256: str, store: EvidenceReader
) -> str:
    # AMD's URI grammar permits '#' or '?' range specifiers and decimal,
    # hexadecimal or C octal literals. Memory URIs need a retained memory-load
    # association and are deliberately not guessed from a similarly named ELF.
    if not isinstance(uri, str) or not uri.startswith("file://"):
        raise ValueError("loaded code object has no supported file-backed association")
    encoded = uri[len("file://") :]
    match = re.fullmatch(r"([^?#]+)(?:[?#](.*))?", encoded)
    if match is None:
        raise ValueError("unsupported loaded code-object URI")
    filename = Path(unquote(match[1], encoding="utf-8", errors="strict")).resolve(
        strict=True
    )
    executable = executable.resolve(strict=True)
    if filename != executable:
        raise ValueError("loaded code object belongs to a different executable/library")
    store.verify(filename, executable_sha256)
    parameters = parse_qsl(match[2] or "", strict_parsing=True)
    if len({key for key, _ in parameters}) != len(parameters) or any(
        key not in {"offset", "size"} for key, _ in parameters
    ):
        raise ValueError("duplicate or unsupported code-object URI range")
    options = {key: _c_integer(value) for key, value in parameters}
    extent = filename.stat().st_size
    offset = options.get("offset", 0)
    count = options.get("size", extent - offset)
    if offset > extent or count <= 0 or count > extent - offset:
        raise ValueError("code-object URI extends beyond its source artifact")
    digest = hashlib.sha256()
    with filename.open("rb") as source:
        source.seek(offset)
        remaining = count
        while remaining:
            chunk = source.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("loaded code-object source was truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _profile_pair(
    production: dict, profiled: dict, declared_environment: JsonValue, *,
    native_output: Path | None = None,
) -> None:
    if not isinstance(declared_environment, dict) or any(
        key not in PROFILER_ENVIRONMENT for key in declared_environment
    ):
        raise ValueError("undeclared or unsupported profiler environment differences")
    identities = []
    for capture in (production, profiled):
        if not manifest_provenance_complete(
            capture.get("build_manifest"), capture.get("backend_selected")
        ):
            raise ValueError("profiler observation lacks complete compiled provenance")
        _, error = request_environment_identity(capture.get("request_identity"))
        if error:
            raise ValueError(error)
        identities.append(common_identity(capture))
    validate_execution_request(production, production["request_identity"])
    validate_execution_request(profiled, profiled["request_identity"], required=True)
    arguments = identities[1]["execution_request"]["arguments"]
    if native_output is not None:
        if arguments.count("--json-output") != 1 or "--json-output" in identities[0]["execution_request"]["arguments"]:
            raise ValueError("one declared native JSON output is required for this profiler relation")
        position = arguments.index("--json-output")
        if (position+1 >= len(arguments) or not Path(arguments[position+1]).is_absolute()
                or Path(arguments[position+1]).resolve(strict=True) != native_output):
            raise ValueError("profiled JSON output path differs from the retained raw native artifact")
        del arguments[position:position+2]
    elif "--json-output" in arguments:
        raise ValueError("profiled file output has no retained raw native artifact")
    flag = "--record-execution-environment"
    baseline_flags = identities[0]["execution_request"]["arguments"].count(flag)
    if baseline_flags == 0:
        arguments.remove(flag)
    elif baseline_flags != 1:
        raise ValueError("production environment observation flag is ambiguous")
    if production.get("external_instrumentation") is not None:
        raise ValueError("production capture is already externally instrumented")
    marker = profiled.get("external_instrumentation")
    if marker != dict(
        schema="rns8_external_instrumentation_v1",
        kind="amd_profiler",
        tool="rocprofv3",
        mode="kernel_trace",
    ):
        raise ValueError(
            "profiled capture lacks an explicit external instrumentation role"
        )
    identities[1].pop("external_instrumentation", None)
    p_env = identities[0]["execution_request"]["environment"]
    d_env = identities[1]["execution_request"]["environment"]
    actual = {key: value for key, value in d_env.items() if p_env.get(key) != value}
    if not same_evidence_value(actual, declared_environment):
        raise ValueError(
            "actual profiler injection differs from its declared environment"
        )
    replaced = {"LD_LIBRARY_PATH", "RNS8_PROFILE_RANGES"}
    if any(key in p_env and key not in replaced for key in declared_environment):
        raise ValueError("production environment already contains profiler injection")
    for key in declared_environment:
        if key in p_env:
            d_env[key] = p_env[key]
        else:
            d_env.pop(key, None)
    if not same_evidence_value(identities[0], identities[1]):
        raise ValueError(
            "profiled and production captures differ beyond declared instrumentation observations"
        )
    if (
        not isinstance(profiled.get("execution_id"), str)
        or not profiled["execution_id"]
        or profiled["execution_id"] == production.get("execution_id")
    ):
        raise ValueError("profiler needs its own process observation identity")
    facts = numerical_observation(profiled)
    if not facts.get("full_output_reference_covered") or facts.get(
        "checksum_u64"
    ) != production.get("checksum_u64"):
        raise ValueError(
            "profiled workload lacks matching completed exact-reference evidence"
        )


def validate_dispatch_receipt(
    reference: JsonValue, *, capture: dict, store: EvidenceReader, parent: Path
) -> dict:
    validate_capture(capture)
    receipt_path, receipt = store.json(reference, parent)
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("adapter") not in ADAPTERS
    ):
        raise ValueError(
            "unsupported native profiler adapter; raw RGP/unknown formats remain inventory-only"
        )
    if receipt.get("production_capture_sha256") != capture_identity(capture):
        raise ValueError("profiler receipt belongs to a different production capture")
    base = receipt_path.parent
    _, profiled = store.json(receipt.get("profiled_capture"), base)
    validate_capture(profiled)
    native_output = None
    if receipt.get("native_benchmark_output") is not None:
        native_output, native_capture = store.json(receipt["native_benchmark_output"], base)
        validate_capture(native_capture)
        added = {"request_identity", "artifact_capture", "execution_id", "external_instrumentation"}
        if (set(native_capture).intersection(added)
                or set(profiled) != set(native_capture) | added
                or not same_evidence_value(native_capture,
                    {key: value for key, value in profiled.items() if key not in added})):
            raise ValueError("bound profiler capture differs from its untouched native output")
    _profile_pair(capture, profiled, receipt.get("profiler_environment"), native_output=native_output)
    process_id = _integer(profiled.get("process_id"))
    if process_id == 0:
        raise ValueError("profiled benchmark must emit its actual OS process ID")
    native_extension = None
    if ADAPTERS[receipt["adapter"]] == "rocpd":
        from amd_rocpd import read_process
        native_path = store.reference(receipt.get("native_trace"), base)
        native_extension = read_process(native_path, process_id)
        native = {"rocprofiler-sdk-tool": [native_extension["native"]]}
    else:
        native_path, native = store.json(receipt.get("native_trace"), base)
    processes = native.get("rocprofiler-sdk-tool")
    if (
        not isinstance(processes, list)
        or not processes
        or any(not isinstance(item, dict) for item in processes)
    ):
        raise ValueError("native trace has no process inventory")
    matched = [
        item
        for item in processes
        if _integer(_object(item.get("metadata")).get("pid")) == process_id
    ]
    if len(matched) != 1:
        raise ValueError(
            "native trace cannot uniquely identify the profiled benchmark process"
        )
    process = matched[0]
    metadata = _object(process.get("metadata"))
    if _integer(metadata.get("fini_time")) <= _integer(metadata.get("init_time")):
        raise ValueError("native profiler process interval is incomplete")
    symbols = _index(process.get("kernel_symbols"), "kernel_id")
    objects = _index(process.get("code_objects"), "code_object_id")
    agents = {}
    agent_rows = process.get("agents")
    if not isinstance(agent_rows, list):
        raise ValueError("native trace has no agent inventory")
    for row in agent_rows:
        row = _object(row)
        identifier = _integer(_object(row.get("id")).get("handle"))
        if identifier in agents:
            raise ValueError("duplicate native profiler agent")
        agents[identifier] = row
    records = _object(process.get("buffer_records")).get("kernel_dispatch")
    if not isinstance(records, list) or not records:
        raise ValueError("native profiler has no executed dispatch records")
    dispatches = {}
    for record in records:
        record = _object(record)
        info = _object(record.get("dispatch_info"))
        for dimension in ("grid_size", "workgroup_size"):
            values = _object(info.get(dimension))
            if any(_integer(values.get(axis)) == 0 for axis in ("x", "y", "z")):
                raise ValueError("native dispatch launch geometry is missing")
        _integer(_object(record.get("correlation_id")).get("internal"))
        key = (
            _integer(_object(info.get("agent_id")).get("handle")),
            _integer(_object(info.get("queue_id")).get("handle")),
            _integer(info.get("dispatch_id")),
        )
        if key in dispatches:
            raise ValueError("ambiguous native process/agent/queue/dispatch identity")
        if _integer(record.get("end_timestamp")) <= _integer(
            record.get("start_timestamp")
        ):
            raise ValueError("native dispatch has no complete timestamp interval")
        dispatches[key] = record
    isa_path, isa = store.json(receipt.get("selected_static_isa"), base)
    target = str(_object(profiled.get("device")).get("gcn_arch", "")).split(":")[0]
    backend = profiled["backend_selected"]
    summary = summarize_isa_report(isa_path, isa)
    resources = lookup_isa_resources(
        {isa_index_key(backend, target): [summary]}, backend, target, capture=profiled
    )
    if not (resources.get("isa_association") or {}).get("qualified"):
        raise ValueError(
            "selected static ISA is not bound to the profiled artifact/kernel/target"
        )
    selected_symbols = set(_object(isa.get("capture_binding")).get("symbols", []))
    selected = receipt.get("selected_dispatches")
    if not isinstance(selected, list) or not selected:
        raise ValueError("profiler receipt has no explicit selected dispatch set")
    executable = Path(
        str(_object(profiled.get("artifact_capture")).get("executable_path"))
    )
    executable_sha = profiled["artifact_capture"]["executable_sha256"]
    seen, observations, object_hashes = set(), [], {}
    for selection in selected:
        selection = _object(selection)
        key = tuple(
            _integer(selection.get(name))
            for name in ("agent_id", "queue_id", "dispatch_id")
        )
        if key in seen or key not in dispatches:
            raise ValueError("selected native dispatch is absent or duplicated")
        seen.add(key)
        record = dispatches[key]
        info = _object(record["dispatch_info"])
        kernel = symbols.get(_integer(info.get("kernel_id")))
        if kernel is None or kernel.get("kernel_name") not in selected_symbols:
            raise ValueError(
                "executed native symbol is not in the selected mangled static symbol set"
            )
        agent = agents.get(key[0])
        if agent is None or agent.get("name") != target:
            raise ValueError(
                "executed native agent architecture does not match the capture"
            )
        code = objects.get(_integer(kernel.get("code_object_id")))
        if (
            code is None
            or not isinstance(code.get("agent_id"), dict)
            or _integer(code["agent_id"].get("handle")) != key[0]
        ):
            raise ValueError("loaded code object lacks its SDK agent_id or belongs to another agent")
        object_id = code["code_object_id"]
        if object_id not in object_hashes:
            object_hashes[object_id] = code_object_digest(
                code.get("uri"), executable, executable_sha, store
            )
        if object_hashes[object_id] != isa.get("code_object_sha256"):
            raise ValueError(
                "executed loaded code object differs from the selected static ISA artifact"
            )
        observations.append(
            dict(
                process_id=process_id,
                agent_id=key[0],
                queue_id=key[1],
                dispatch_id=key[2],
                correlation_id=record["correlation_id"]["internal"],
                kernel_id=info["kernel_id"],
                mangled_symbol=kernel["kernel_name"],
                code_object_id=object_id,
                code_object_sha256=object_hashes[object_id],
                start_ns=record["start_timestamp"],
                end_ns=record["end_timestamp"],
                grid=copy.deepcopy(info.get("grid_size")),
                workgroup=copy.deepcopy(info.get("workgroup_size")),
                native_relationships=copy.deepcopy(record.get("native_rocpd")),
            )
        )
    command = receipt.get("acquisition_command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) for item in command)
    ):
        raise ValueError("profiler acquisition argv is missing")
    tool = store.reference(receipt.get("tool_executable"), base)
    version_file = store.reference(receipt.get("tool_version_output"), base)
    version = receipt.get("tool_version")
    if (
        version != "1.4.1"
        or re.search(
            r"(?<![\w.])" + re.escape(version) + r"(?![\w.+-])",
            version_file.read_text(encoding="utf-8"),
        )
        is None
    ):
        raise ValueError(
            "profiler executable release is not bound to its raw version output"
        )
    if Path(command[0]).resolve(strict=True) != tool or command.count("--") != 1:
        raise ValueError(
            "profiler acquisition does not identify its tool/workload roles"
        )
    separator = command.index("--")
    options = command[1:separator]
    if options.count("--kernel-trace") != 1 or options.count("--output-format") != 1:
        raise ValueError("native acquisition does not select its declared kernel-trace output")
    formats = options[options.index("--output-format") + 1 :]
    if not formats or formats[0] != ADAPTERS[receipt["adapter"]]:
        raise ValueError("native acquisition output format differs from its adapter")
    workload = command[separator + 1 :]
    if (
        not workload
        or Path(workload[0]).resolve(strict=True) != executable.resolve(strict=True)
        or not same_evidence_value(
            workload[1:], profiled["request_identity"]["arguments"]
        )
    ):
        raise ValueError("profiled benchmark argv differs from the native acquisition")
    acquisition = None
    if receipt.get("acquisition_session") is not None:
        from amd_profiler_session import validate_acquisition
        acquisition = validate_acquisition(receipt, capture, store, base)
    return dict(
        schema=SCHEMA,
        acquisition=acquisition,
        qualified_selected_dispatches=True,
        observations=observations,
        native_trace_sha256=receipt["native_trace"]["sha256"],
        profiled_capture=str(
            (base / receipt["profiled_capture"]["path"]).resolve(strict=True)
        ),
        profiled_capture_sha256=receipt["profiled_capture"]["sha256"],
        acquisition_command=command,
        native_trace=str(native_path),
        receipt_sha256=canonical_digest(receipt),
        scope="selected_executed_dispatch_identity_not_complete_run_counters_or_production_latency",
        complete_run_coverage=False,
        performance_promotion_allowed=False,
        native_schema_version=native_extension["schema_version"] if native_extension else "rocprofiler-sdk-json-1.4.1",
        native_identity_namespace=native_extension["identity_namespace"] if native_extension else "rocprofiler-sdk-process-local-handles",
        native_metrics=([metric for metric in native_extension["metrics"] if
                         (metric["agent_id"], metric["queue_id"], metric["dispatch_id"]) in seen]
                        if native_extension else []),
        unresolved_metrics=native_extension["unresolved_metrics"] if native_extension else [],
        loss_status=native_extension["loss_status"] if native_extension else "unknown_not_established_by_dispatch_identity",
    )
