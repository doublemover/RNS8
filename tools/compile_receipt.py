"""Bounded receipts for the actual configured explicit HIP compiler process."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest
from compiler_context import compiler_environment
from json_evidence import parse_evidence_json, same_evidence_value
from process_progress import capture_succeeded, run_capture

SCHEMA = "rns8_native_compile_receipt_v1"
CONTEXT_SCHEMA = "rns8_native_compile_context_v1"
ROOT = Path(__file__).resolve().parents[1]
SCOPE = "one_owned_compiler_process_capture_host_and_device_including_startup_and_observation_excluding_generator_and_outer_bookkeeping"
MAX_JSON = 8 * 1024**2
MAX_INPUT_BYTES = 512 * 1024**2
MAX_OBJECT_BYTES = 2 * 1024**3
MAX_OUTPUT_BYTES = 16 * 1024**2
MIN_FREE_BYTES = 5 * 1024**3


def read_json(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON:
        raise ValueError("compile context/receipt exceeds its JSON byte budget")
    value = parse_evidence_json(raw.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError("compile context/receipt must be a JSON object")
    return value


def seal(value: dict) -> dict:
    return {**value, "identity_sha256": canonical_digest(value)}


def write_json(path: Path, value: dict) -> None:
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_JSON:
        raise ValueError("compile receipt exceeds its JSON byte budget")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("xb") as stream:
        stream.write(raw)
    temporary.replace(path)


def absolute(value: object) -> Path:
    if type(value) is not str or not value or not Path(value).is_absolute():
        raise ValueError("compile context paths must be explicit absolute paths")
    return Path(value).resolve()


def configured_plan(context_path: Path, *, environment=None, timeout_seconds=1800) -> dict:
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 3600
    ):
        raise ValueError("compile timeout must be finite and lie in (0,3600] seconds")
    context_path = context_path.resolve(strict=True)
    context = read_json(context_path)
    fields = {
        "schema",
        "source_root",
        "build_root",
        "directory",
        "file",
        "output",
        "configuration",
        "arguments",
        "inputs",
        "targets",
    }
    if set(context) != fields or context["schema"] != CONTEXT_SCHEMA:
        raise ValueError("unsupported configured compile context")
    source_root = absolute(context["source_root"])
    build = absolute(context["build_root"])
    if Path(context["output"]).is_symlink() or Path(context["output"]).is_junction():
        raise ValueError("configured output cannot be a link or junction")
    source = absolute(context["file"])
    output = absolute(context["output"])
    directory = absolute(context["directory"])
    if (
        source_root != ROOT
        or not build.is_dir()
        or not directory.is_dir()
        or not directory.is_relative_to(build)
        or not output.is_relative_to(build)
        or output in (source, build)
        or not context_path.is_relative_to(build)
        or not (source.is_relative_to(source_root) or source.is_relative_to(build))
    ):
        raise ValueError("compile context source/build/output ownership differs")
    argv = context["arguments"]
    targets = context["targets"]
    inputs = context["inputs"]
    for values in (argv, targets, inputs):
        if (
            type(values) is not list
            or not values
            or any(type(item) is not str or not item for item in values)
        ):
            raise ValueError("compile argv/target/input inventories must be nonempty string arrays")
    if (
        len(argv) > 8192
        or len(inputs) > 10000
        or len(targets) > 32
        or len(set(targets)) != len(targets)
        or type(context["configuration"]) is not str
        or not context["configuration"]
    ):
        raise ValueError("compile context inventory/configuration is invalid or exceeds its budget")
    compiler = absolute(argv[0])
    compiler.resolve(strict=True)
    for option, path in (("-c", source), ("-o", output)):
        if (
            argv.count(option) != 1
            or argv.index(option) + 1 >= len(argv)
            or absolute(argv[argv.index(option) + 1]) != path
        ):
            raise ValueError("compile source/output argv differs from its configured record")
    if any(argv.count("--offload-arch=" + target) != 1 for target in targets):
        raise ValueError("compile target inventory differs from actual compiler argv")
    tool_inputs = [
        ROOT / "tools" / name
        for name in (
            "compile_receipt.py",
            "compiler_context.py",
            "process_progress.py",
            "benchmark_request.py",
            "compiled_binary_identity.py",
            "json_evidence.py",
        )
    ]
    files = list(
        dict.fromkeys(
            [
                context_path,
                source,
                compiler,
                Path(sys.executable).resolve(),
                *tool_inputs,
                *(absolute(name) for name in inputs),
            ]
        )
    )
    total = 0
    hashes = {}
    for path in files:
        if not path.is_file():
            raise ValueError("compile input is missing: " + str(path))
        total += path.stat().st_size
        if total > MAX_INPUT_BYTES:
            raise ValueError("compile input hashing exceeds its byte budget")
        hashes[str(path)] = file_digest(path)
    child = dict(os.environ if environment is None else environment)
    return dict(
        schema=SCHEMA,
        status="planned",
        scope=SCOPE,
        source=str(source),
        output=str(output),
        source_sha256=hashes[str(source)],
        compiler=str(compiler),
        compiler_sha256=hashes[str(compiler)],
        context=str(context_path),
        configuration=context["configuration"],
        targets=targets,
        command=argv,
        directory=str(directory),
        build_root=str(build),
        input_sha256=hashes,
        input_scope="configured_CMake_prerequisites_receipt_tooling_direct_compiler_and_Python_executables_not_transitive_SDK_closure",
        compiler_environment=compiler_environment(child),
        budgets=dict(
            timeout_seconds=timeout_seconds,
            max_object_bytes=MAX_OBJECT_BYTES,
            max_output_bytes=MAX_OUTPUT_BYTES,
            minimum_free_bytes=MIN_FREE_BYTES,
        ),
        hardware_qualification=False,
        performance_qualified=False,
    )


def execute(context_path: Path, *, environment=None, timeout_seconds=1800) -> dict:
    start = time.monotonic()
    child = dict(os.environ if environment is None else environment)
    plan = configured_plan(context_path, environment=child, timeout_seconds=timeout_seconds)
    output = Path(plan["output"])
    build = Path(plan["build_root"])
    if shutil.disk_usage(build).free < MIN_FREE_BYTES:
        raise ValueError("insufficient free space for bounded native compilation")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Each attempt retains its own logs/receipt; only the small latest pointer is
    # replaced. Build output is generated and belongs to this configured rule.
    run_root = Path(str(output) + ".compile")
    if run_root.is_symlink() or run_root.is_junction():
        raise ValueError("compile receipt root cannot be a link")
    run_dir = run_root / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    if not run_dir.resolve().is_relative_to(build):
        raise ValueError("compile receipt directory escaped its build")
    receipt = run_dir / "receipt.json"
    latest = Path(str(output) + ".compile.json")
    running = seal({**plan, "status": "running", "fresh_output": True})
    write_json(receipt, running)
    write_json(
        latest,
        dict(schema="rns8_native_compile_pointer_v1", receipt=str(receipt), status="running"),
    )
    if output.exists():
        if not output.is_file() or output.is_symlink() or output.is_junction():
            raise ValueError("configured compile output is not a regular generated file")
        output.unlink()  # never accept a successful exit with a stale old object
    try:
        result = run_capture(
            plan["command"],
            cwd=Path(plan["directory"]),
            label="native-compile",
            env=child,
            timeout_seconds=timeout_seconds,
            max_output_bytes=MAX_OUTPUT_BYTES,
            artifact_roots=(output, run_dir),
            max_artifact_bytes=MAX_OBJECT_BYTES + MAX_OUTPUT_BYTES + MAX_JSON,
            return_on_interrupt=True,
        )
    except (OSError, ValueError) as error:
        result = dict(
            command=plan["command"],
            returncode=None,
            timed_out=False,
            interrupted=False,
            limit_exceeded=None,
            capture_error=str(error),
            elapsed_seconds=time.monotonic() - start,
            stdout="",
            stderr=type(error).__name__ + ": " + str(error),
        )
    streams = {}
    for stream in ("stdout", "stderr"):
        path = run_dir / (stream + ".log")
        with path.open("xb") as file:
            file.write(result.pop(stream).encode("utf-8"))
        streams[stream] = dict(path=str(path), sha256=file_digest(path), bytes=path.stat().st_size)
    input_error = None
    try:
        unchanged = all(
            Path(name).is_file() and file_digest(name) == digest
            for name, digest in plan["input_sha256"].items()
        )
    except OSError as error:
        unchanged = False
        input_error = str(error)
    unchanged &= compiler_environment(child) == plan["compiler_environment"]
    artifact = None
    if (
        output.is_file()
        and not output.is_symlink()
        and not output.is_junction()
        and 0 < output.stat().st_size <= MAX_OBJECT_BYTES
    ):
        artifact = dict(path=str(output), sha256=file_digest(output), bytes=output.stat().st_size)
    complete = capture_succeeded(result) and unchanged and artifact is not None
    record = seal(
        {
            **plan,
            "status": "complete" if complete else "failed",
            "fresh_output": True,
            "inputs_unchanged": unchanged,
            "input_error": input_error,
            "process": result,
            "streams": streams,
            "artifact": artifact,
            "failure_reasons": ([] if capture_succeeded(result) else ["compiler_capture_failed"])
            + ([] if unchanged else ["input_bookend_changed"])
            + ([] if artifact is not None else ["object_missing_or_size_budget"]),
            "driver_elapsed_seconds": time.monotonic() - start,
        }
    )
    write_json(receipt, record)
    write_json(
        latest,
        dict(
            schema="rns8_native_compile_pointer_v1",
            receipt=str(receipt),
            status=record["status"],
            sha256=file_digest(receipt),
        ),
    )
    return record


def load_compile_cost(path: Path, *, source: Path, host_object: Path, target: str) -> dict:
    path = path.resolve(strict=True)
    record = read_json(path)
    if record.get("schema") == "rns8_native_compile_pointer_v1":
        receipt = absolute(record.get("receipt"))
        if (
            receipt.stat().st_size > MAX_JSON
            or file_digest(receipt) != record.get("sha256")
            or record.get("status") != "complete"
        ):
            raise ValueError("native compile pointer is incomplete or changed")
        path = receipt
        record = read_json(path)
    unsigned = {key: value for key, value in record.items() if key != "identity_sha256"}
    if (
        type(record.get("process")) is not dict
        or type(record.get("compiler_environment")) is not dict
        or type(record.get("budgets")) is not dict
    ):
        raise ValueError("native compile process/environment/budget records are malformed")
    if (
        record.get("schema") != SCHEMA
        or record.get("identity_sha256") != canonical_digest(unsigned)
        or record.get("status") != "complete"
        or record.get("fresh_output") is not True
        or record.get("inputs_unchanged") is not True
        or record.get("scope") != SCOPE
        or not capture_succeeded(record.get("process", {}))
    ):
        raise ValueError("native compile receipt is incomplete or unsealed")
    if (
        absolute(record.get("source")) != source.resolve(strict=True)
        or absolute(record.get("output")) != host_object.resolve(strict=True)
        or record.get("targets") != [target]
        or record.get("source_sha256") != file_digest(source)
    ):
        raise ValueError("native compile receipt differs from generated source/object/target")
    if host_object.stat().st_size > MAX_OBJECT_BYTES:
        raise ValueError("native object exceeds its receipt byte budget")
    expected = configured_plan(
        absolute(record.get("context")),
        environment=record["compiler_environment"],
        timeout_seconds=record["budgets"]["timeout_seconds"],
    )
    for key, value in expected.items():
        if key != "status" and not same_evidence_value(record.get(key), value):
            raise ValueError("native compile context changed: " + key)
    if not same_evidence_value(record["process"].get("command"), record["command"]):
        raise ValueError("native compiler process argv differs from configured command")
    artifact = record.get("artifact")
    if (
        type(artifact) is not dict
        or type(artifact.get("bytes")) is not int
        or artifact.get("sha256") != file_digest(host_object)
        or artifact.get("bytes") != host_object.stat().st_size
        or absolute(artifact.get("path")) != host_object.resolve()
    ):
        raise ValueError("native compiled artifact differs from its receipt")
    streams = record.get("streams")
    if type(streams) is not dict or set(streams) != {"stdout", "stderr"}:
        raise ValueError("native compile logs are incomplete")
    for stream in streams.values():
        if type(stream) is not dict:
            raise ValueError("native compile log reference is malformed")
        log = absolute(stream.get("path"))
        if (
            type(stream.get("bytes")) is not int
            or not log.is_relative_to(path.parent)
            or log.stat().st_size > MAX_OUTPUT_BYTES
            or file_digest(log) != stream.get("sha256")
            or log.stat().st_size != stream.get("bytes")
        ):
            raise ValueError("native compile log changed")
    seconds = record["process"].get("elapsed_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("native compile elapsed time is absent/nonfinite")
    driver_seconds = record.get("driver_elapsed_seconds")
    if (
        type(driver_seconds) not in (int, float)
        or not math.isfinite(driver_seconds)
        or driver_seconds < seconds
    ):
        raise ValueError("native compile driver timing is absent/inconsistent")
    return dict(
        status="recorded",
        seconds=seconds,
        scope=SCOPE,
        receipt=str(path),
        receipt_sha256=file_digest(path),
        compiler_sha256=record["compiler_sha256"],
        command_sha256=canonical_digest(record["command"]),
        driver_elapsed_seconds=driver_seconds,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    result = (
        execute(args.context, timeout_seconds=args.timeout_seconds)
        if args.execute
        else seal(configured_plan(args.context, timeout_seconds=args.timeout_seconds))
    )
    if not args.execute:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(
            "Native compile receipt:",
            result["status"],
            "compiler seconds:",
            result["process"]["elapsed_seconds"],
        )
    return 0 if result["status"] in ("planned", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
