"""Plan bounded explicit-backend arithmetic replay; execute only with --execute."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from benchmark_request import file_digest
from compiled_binary_identity import validate_document
from fuzz_cases import MAX_BYTES, admitted, decode
from fuzz_shrink import Verdict
from json_evidence import parse_evidence_json, same_evidence_value
from process_progress import capture_succeeded, run_capture

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = (
    "cpu-reference",
    "hip-direct",
    "hipblaslt",
    "ck",
    "rocwmma",
    "wrap64-byte-limb",
    "hip-vector-alu-int64",
    "amdgpu-builtins",
    "cpu-amx",
)


def device_number(text):
    if not re.fullmatch(r"-?[0-9]{1,10}", text):
        raise argparse.ArgumentTypeError("complete decimal device ordinal required")
    value = int(text)
    if not -1 <= value < 2**31:
        raise argparse.ArgumentTypeError("device must be -1 or an int32 ordinal")
    return value


def raw_case(path):
    if not path.is_file():
        raise ValueError("regular arithmetic input required")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("arithmetic input exceeds byte budget")
    return raw


def read_case(path):
    raw = raw_case(path)
    case = decode(raw)
    if not admitted(case):
        raise ValueError("replay requires a range-admitted arithmetic case")
    return raw, case


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_result(native, process, case, raw, backend, device, run_id, identity):
    if (
        not isinstance(native, dict)
        or type(process.get("returncode")) is not int
        or process.get("timed_out") is not False
        or process.get("interrupted")
        or process.get("limit_exceeded") is not None
        or process.get("capture_error") is not None
    ):
        raise ValueError("native replay process did not complete within its contract")
    work = {
        key: getattr(case, key)
        for key in (
            "semantic",
            "m",
            "n",
            "k",
            "operations",
            "padding_a",
            "padding_b",
            "padding_c",
            "prefix",
            "modulus",
        )
    }
    expected = dict(
        schema="rns8_arithmetic_replay_v1",
        run_id=run_id,
        process_id=process["process_id"],
        case_sha256=hashlib.sha256(raw).hexdigest(),
        case_bytes=len(raw),
        requested_backend=backend,
        requested_backend_value=BACKENDS.index(backend) + 1,
        requested_device=device,
        promotion_eligible=False,
        production_latency=False,
        work=work,
    )
    if any(not same_evidence_value(native.get(key), value) for key, value in expected.items()):
        raise ValueError("native replay case/process/backend contract differs")
    bound = validate_document(dict(native, schema="rns8_compiled_binary_identity_v1"))
    if not same_evidence_value(bound, identity):
        raise ValueError("replay executable/library identities differ from the preflight")
    state = native.get("state")
    if state == "passed":
        gemms = 1 + bool(case.operations & 1) + bool(case.operations & 4)
        comparisons = 1 + bool(case.operations & 1) + 2 * bool(case.operations & 4)
        plan, observed = native.get("plan"), native.get("device")
        if (
            not capture_succeeded(process)
            or native.get("exact_reference_passed") is not True
            or native.get("failure") is not None
            or native.get("cleanup") != "checked_public_destruction_passed"
            or not same_evidence_value(native.get("gemm_calls"), gemms)
            or not same_evidence_value(native.get("full_output_comparisons"), comparisons)
            or not isinstance(plan, dict)
            or not same_evidence_value(plan.get("backend"), expected["requested_backend_value"])
            or not isinstance(plan.get("kernel"), str)
            or not plan["kernel"]
            or not isinstance(observed, dict)
            or (
                device >= 0
                and (
                    not same_evidence_value(observed.get("hip_available"), 1)
                    or not same_evidence_value(observed.get("ordinal"), device)
                    or not observed.get("target")
                )
            )
        ):
            raise ValueError("native replay passed without complete output/count/dispatch custody")
        return Verdict("passed")
    failure = native.get("failure")
    if (
        not isinstance(failure, dict)
        or type(failure.get("kind")) is not str
        or type(failure.get("phase")) is not str
        or not failure["kind"]
        or not failure["phase"]
        or len(failure["phase"]) > 128
        or failure["kind"] not in ("status", "no_write", "value_or_padding", "backend_identity")
        or any(
            type(failure.get(key)) is not int or not 0 <= failure[key] <= 13
            for key in ("actual_status", "expected_status")
        )
        or native.get("cleanup") != "process_exit_without_further_library_calls"
        or native.get("exact_reference_passed") is not False
    ):
        raise ValueError("native replay lacks a structured first failure")
    if state == "unsupported":
        if (
            process.get("returncode") != 77
            or failure["kind"] != "status"
            or failure["phase"] not in ("create context", "create plan")
            or type(failure.get("actual_status")) is not int
            or failure["actual_status"] not in (2, 3, 4)
            or not same_evidence_value(native.get("gemm_calls"), 0)
            or not same_evidence_value(native.get("full_output_comparisons"), 0)
        ):
            raise ValueError("unsupported replay is not a no-arithmetic admission result")
        return Verdict("unsupported")
    if state != "failed" or process.get("returncode") != 1:
        raise ValueError("unrecognized native replay outcome")
    return Verdict(
        "failed",
        (
            failure["kind"],
            failure["phase"],
            str(failure.get("actual_status")),
            str(failure.get("expected_status")),
        ),
    )


def execute(args, plan, inputs):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    exe = args.exe.resolve(strict=True)
    before = file_digest(exe)
    sources = {
        name: file_digest(ROOT / name)
        for name in (
            "tools/fuzz_replay.py",
            "tools/fuzz_cases.py",
            "tools/fuzz_shrink.py",
            "tools/compiled_binary_identity.py",
            "tools/json_evidence.py",
            "tools/process_progress.py",
        )
    }
    deadline = time.monotonic() + args.seconds
    records = []
    retained_cases = {}

    def run(command, label):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("arithmetic replay wall budget exhausted")
        return run_capture(
            command,
            cwd=ROOT,
            label=label,
            timeout_seconds=min(args.case_seconds, remaining),
            max_output_bytes=8 * 1024**2,
            artifact_roots=[out],
            max_artifact_bytes=256 * 1024**2,
            return_on_interrupt=True,
        )

    def bookend():
        if before != file_digest(exe) or any(
            file_digest(ROOT / name) != sha for name, sha in sources.items()
        ):
            raise ValueError("replay executable or adapter source changed")
        if any(raw_case(path) != raw for path, raw, _ in inputs) or any(
            raw_case(path) != raw for path, raw in retained_cases.items()
        ):
            raise ValueError("arithmetic input changed during replay")

    error = None
    interrupted = False
    try:
        process = run([str(exe), "--build-identity"], "arithmetic-replay-identity")
        interrupted = bool(process.get("interrupted"))
        write(out / "identity-process.json", process)
        if not capture_succeeded(process):
            interrupted = bool(process.get("interrupted"))
            raise ValueError("native compiled identity preflight failed")
        document = parse_evidence_json(process["stdout"])
        identity = validate_document(document)
        write(out / "identity.json", document)
        bookend()
        for ordinal, (path, raw, case) in enumerate(inputs):
            run_id = uuid.uuid4().hex
            row = dict(
                run_id=run_id, ordinal=ordinal, original=str(path), state="infrastructure_error"
            )
            records.append(row)
            bookend()
            retained = out / (run_id + ".bin")
            with retained.open("xb") as stream:
                stream.write(raw)
            retained_cases[retained] = raw
            command = [
                str(exe),
                "--case",
                str(retained),
                "--backend",
                args.backend,
                "--device",
                str(args.device),
                "--run-id",
                run_id,
            ]
            row["command"] = command
            process = run(command, "arithmetic-replay:" + str(ordinal))
            interrupted = bool(process.get("interrupted"))
            write(out / (run_id + "-process.json"), process)
            bookend()
            native = parse_evidence_json(process["stdout"])
            write(out / (run_id + "-native.json"), native)
            verdict = validate_result(
                native, process, case, raw, args.backend, args.device, run_id, identity
            )
            row.update(
                state=verdict.kind,
                failure_signature=list(verdict.signature),
                case_sha256=hashlib.sha256(raw).hexdigest(),
            )
            write(out / (run_id + "-verdict.json"), row)
            if verdict.kind == "failed":
                raise ValueError("native arithmetic failure; further cases stopped")
        bookend()
    except (OSError, ValueError, KeyError, TypeError) as failure:
        error = str(failure)
    except KeyboardInterrupt:
        interrupted, error = True, "user interruption"
    counts = dict(Counter(row["state"] for row in records))
    accepted = error is None and len(records) == len(inputs) and counts.get("passed", 0) > 0
    result = dict(
        schema="rns8_arithmetic_replay_session_v1",
        state="completed" if accepted else "incomplete",
        accepted=accepted,
        error=error,
        interrupted=interrupted,
        executable_sha256=before,
        adapter_sources_sha256=sources,
        plan=plan,
        cases=records,
        counts=counts,
        hardware_qualification=False,
        promotion_eligible=False,
        production_latency=False,
    )
    try:
        write(out / "results.json", result)
    except OSError:
        if not interrupted:
            raise
    print(json.dumps(result, indent=2))
    return 130 if interrupted else 0 if accepted else 77 if error is None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--case", dest="cases", type=Path, action="append", required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--device", type=device_number, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--case-seconds", type=int, default=60)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if (
            not 1 <= len(args.cases) <= 64
            or not 1 <= args.seconds <= 3600
            or not 1 <= args.case_seconds <= args.seconds
            or not -1 <= args.device < 2**31
            or (args.backend in ("cpu-reference", "cpu-amx") and args.device != -1)
            or (
                args.backend not in ("cpu-reference", "cpu-amx", "wrap64-byte-limb")
                and args.device < 0
            )
        ):
            raise ValueError("explicit bounded backend/device/session contract required")
        inputs = [(path.resolve(strict=True), *read_case(path)) for path in args.cases]
        digests = [hashlib.sha256(raw).hexdigest() for _, raw, _ in inputs]
        if len(set(digests)) != len(digests):
            raise ValueError("duplicate arithmetic inputs are not independent coverage")
        plan = dict(
            schema="rns8_arithmetic_replay_plan_v1",
            backend=args.backend,
            device=args.device,
            seconds=args.seconds,
            case_seconds=args.case_seconds,
            execute=args.execute,
            case_sha256=digests,
            gpu_work_executed=False,
            stop_on_first_failure=True,
            automatic_shrinking=False,
            production_latency=False,
        )
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        return execute(args, plan, inputs)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
