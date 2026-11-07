"""Explicit bounded semantic minimization using the retained native replay owner."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

from benchmark_request import file_digest
import fuzz_replay as replay
from fuzz_cases import encode
from fuzz_shrink import Verdict, minimize as shrink
from json_evidence import parse_evidence_json


def admissible_failure(signature):
    if not isinstance(signature, tuple) or len(signature) != 4:
        return False
    kind, phase, actual, expected = signature
    if not isinstance(phase, str) or not phase or len(phase) > 128 or phase.startswith("destroy "):
        return False
    if kind in ("value_or_padding", "no_write"):
        return actual == expected == "0"
    # Device/library failure, cancellation, timeout and identity/cleanup defects
    # stop the campaign; they are not arithmetic predicates safe to minimize.
    return kind == "status" and (
        (expected == "0" and actual in ("1", "5", "6", "7", "9"))
        or (expected == "1" and actual == "0")
    )


def minimize(args):
    original_raw, original = replay.read_case(args.case)
    expected_signature = tuple(args.signature)
    if not admissible_failure(expected_signature):
        raise ValueError("only explicit arithmetic/status/no-write failures can be minimized")
    if args.device >= 0 and not args.allow_gpu:
        raise ValueError("GPU minimization requires separate explicit --allow-gpu selection")
    out = args.out.resolve()
    plan = dict(
        schema="rns8_native_arithmetic_shrink_plan_v1",
        backend=args.backend,
        device=args.device,
        original_sha256=hashlib.sha256(original_raw).hexdigest(),
        failure_signature=list(expected_signature),
        max_replays=args.replays,
        seconds=args.seconds,
        case_seconds=args.case_seconds,
        execute=args.execute,
        gpu_opt_in=args.allow_gpu,
        production_latency=False,
    )
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    replay.write(out / "plan.json", plan)
    (out / "original.bin").write_bytes(original_raw)
    native_sessions = []
    native_inputs = []
    executable_sha256 = file_digest(args.exe.resolve(strict=True))
    deadline = time.monotonic() + args.seconds
    inputs = {
        name: file_digest(replay.ROOT / name)
        for name in ("tools/fuzz_replay_shrink.py", "tools/fuzz_replay.py", "tools/fuzz_shrink.py")
    }

    def attempt(case):
        if any(file_digest(replay.ROOT / name) != sha for name, sha in inputs.items()):
            raise ValueError("native minimizer source changed")
        if file_digest(args.exe) != executable_sha256 or replay.raw_case(args.case) != original_raw:
            raise ValueError("native minimizer executable or original changed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("native minimization wall budget exhausted")
        index = len(native_sessions)
        case_path = out / (str(index) + ".bin")
        raw = encode(case)
        case_path.write_bytes(raw)
        session = out / ("replay-" + str(index))
        options = SimpleNamespace(
            exe=args.exe,
            out=session,
            backend=args.backend,
            device=args.device,
            seconds=remaining,
            case_seconds=min(args.case_seconds, remaining),
            artifact_root=out,
            absolute_deadline=deadline,
        )
        single_plan = dict(
            schema="rns8_arithmetic_replay_plan_v1",
            backend=args.backend,
            device=args.device,
            seconds=remaining,
            case_seconds=options.case_seconds,
            execute=True,
            case_sha256=[hashlib.sha256(raw).hexdigest()],
            gpu_work_executed=False,
            stop_on_first_failure=True,
            automatic_shrinking=False,
            production_latency=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            code = replay.execute(options, single_plan, [(case_path, raw, case)])
        with (session / "results.json").open("rb") as stream:
            receipt = stream.read(8 * 1024**2 + 1)
        if len(receipt) > 8 * 1024**2:
            raise ValueError("native minimizer receipt byte budget exceeded")
        result = parse_evidence_json(receipt.decode("utf-8"))
        native_sessions.append(
            dict(
                path=str(session / "results.json"),
                sha256=file_digest(session / "results.json"),
                exit_code=code,
            )
        )
        if not isinstance(result, dict):
            raise ValueError("native minimizer receipt must be an object")
        if (
            result.get("executable_sha256") != executable_sha256
            or not isinstance(result.get("input_receipts"), list)
            or not result["input_receipts"]
        ):
            raise ValueError("native minimizer executable/input custody mismatch")
        native_inputs.extend(result["input_receipts"])
        if code == 130 or result.get("interrupted"):
            raise InterruptedError("native minimization interrupted; no further candidates")
        if not isinstance(result.get("cases"), list) or len(result["cases"]) != 1:
            raise ValueError("native minimization lacks exactly one retained result")
        row = result["cases"][0]
        if not isinstance(row, dict):
            raise ValueError("native minimizer case must be an object")
        state = row.get("state")
        if state == "failed":
            signature = tuple(row.get("failure_signature", []))
            if not admissible_failure(signature):
                raise ValueError("unsafe/non-arithmetic native failure; minimization stopped")
            return Verdict("failed", signature)
        if state in ("passed", "unsupported") and result.get("error") is None:
            return Verdict(state)
        raise ValueError("native minimization replay infrastructure failed")

    result = None
    error = None
    interrupted = False
    try:
        # The supplied signature is an expectation, not evidence. The first
        # fresh native replay must actually reproduce it before any reduction.
        def checked(case):
            verdict = attempt(case)
            if len(native_sessions) == 1 and (
                verdict.kind != "failed" or verdict.signature != expected_signature
            ):
                raise ValueError("original native failure signature was not reproduced")
            return verdict

        result = shrink(original, checked, max_replays=args.replays, seconds=args.seconds)
        (out / "minimized.bin").write_bytes(encode(result.minimized))
        if (
            replay.raw_case(args.case) != original_raw
            or replay.raw_case(out / "original.bin") != original_raw
        ):
            raise ValueError("original minimizer input changed")
        if any(file_digest(replay.ROOT / name) != sha for name, sha in inputs.items()):
            raise ValueError("minimizer source changed before publication")
        if any(file_digest(Path(row["path"])) != row["sha256"] for row in native_sessions):
            raise ValueError("native minimizer session receipt changed")
        if file_digest(args.exe) != executable_sha256 or any(
            not Path(row["path"]).resolve().is_relative_to(out)
            or Path(row["path"]).stat().st_size != row["bytes"]
            or file_digest(Path(row["path"])) != row["sha256"]
            for row in native_inputs
        ):
            raise ValueError("native minimizer executable or retained inputs changed")
    except (OSError, ValueError, TypeError, KeyError) as failure:
        interrupted = isinstance(failure, InterruptedError)
        error = str(failure)
    except KeyboardInterrupt:
        interrupted, error = True, "user interruption"
    summary = dict(
        schema="rns8_native_arithmetic_shrink_v1",
        complete=error is None,
        error=error,
        interrupted=interrupted,
        plan=plan,
        native_sessions=native_sessions,
        adapter_sources_sha256=inputs,
        executable_sha256=executable_sha256,
        input_receipts=native_inputs,
        shrink=result.summary() if result is not None else None,
        hardware_qualification=False,
        promotion_eligible=False,
        production_latency=False,
    )
    try:
        replay.write(out / "results.json", summary)
    except OSError:
        if not interrupted:
            raise
    print(json.dumps(summary, indent=2))
    return 130 if interrupted else 0 if error is None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--backend", choices=replay.BACKENDS, required=True)
    parser.add_argument("--device", type=replay.device_number, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--signature", nargs=4, required=True, metavar=("KIND", "PHASE", "ACTUAL", "EXPECTED")
    )
    parser.add_argument("--replays", type=int, default=32)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--case-seconds", type=int, default=60)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if (
            not 1 <= args.replays <= 128
            or not 1 <= args.seconds <= 3600
            or not 1 <= args.case_seconds <= args.seconds
            or (args.backend in ("cpu-reference", "cpu-amx") and args.device != -1)
            or (
                args.backend not in ("cpu-reference", "cpu-amx", "wrap64-byte-limb")
                and args.device < 0
            )
        ):
            raise ValueError("bounded explicit native minimization contract required")
        return minimize(args)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
