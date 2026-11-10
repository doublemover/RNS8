"""Plan a counterbalanced two-build experiment; collection requires --execute."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import uuid

from benchmark_references import DEFAULT_CACHE, DEFAULT_INDEX, prepare
from benchmark_request import file_digest
from benchmark_sweep_lib.execution import run_command
from diagnostic_companion import digest
from paired_experiment import (
    ARMS,
    RECEIPT_SCHEMA,
    build_plan,
    check_arm,
    file_reference,
    load_experiment,
    validate_plan,
)
from review_receipt import ReviewReceiptReader
from timing_eligibility import capture_identity


def write_new(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")


def collect(plan_path: Path, timeout: float, reference_index: Path | None = None) -> Path:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("finite positive process timeout required")
    reader = ReviewReceiptReader()
    _, plan = reader.document(file_reference(plan_path))
    # Replay plan validation before starting any subprocess.
    validate_plan(plan)
    receipt_path = plan_path.with_name("collection.json")
    if receipt_path.exists() or any(Path(job["output"]).exists() for job in plan["jobs"]):
        raise ValueError("refusing to overwrite an experiment collection or observation")
    for arm in ARMS:
        _, manifest = reader.document(plan["arms"][arm]["manifest"])
        if digest(manifest) != plan["arms"][arm]["identity"]["build_manifest_sha256"]:
            raise ValueError("declared build manifest differs from arm identity")
        if (
            file_digest(Path(plan["arms"][arm]["bench"]))
            != plan["arms"][arm]["identity"]["executable_sha256"]
        ):
            raise ValueError("declared arm executable changed before collection")
    reader.finish()
    command = plan["jobs"][0]["command"]
    if reference_index is not None and "--reference-cache-dir" in command:
        position = command.index("--reference-cache-dir")
        prepare(reference_index, Path(command[position + 1]))
    observations = []
    for job in plan["jobs"]:
        arm = plan["arms"][job["arm"]]
        if file_digest(Path(arm["bench"])) != arm["identity"]["executable_sha256"]:
            raise ValueError("declared arm executable changed before collection")
        output = Path(job["output"])
        if output.exists():
            raise ValueError("refusing to overwrite an experiment observation")
        started = time.perf_counter_ns()
        if not run_command(job["command"], output, timeout):
            raise RuntimeError("paired experiment process failed: " + str(output))
        finished = time.perf_counter_ns()
        reference = file_reference(output)
        capture = reader.capture(reference)
        check_arm(capture, plan, job["arm"])
        observations.append(
            dict(
                ordinal=job["ordinal"],
                run=job["run"],
                arm=job["arm"],
                mode=job["mode"],
                started_ns=started,
                finished_ns=finished,
                capture=reference,
                capture_sha256=capture_identity(capture),
                execution_id=capture.get("execution_id"),
            )
        )
    reader.finish()
    write_new(
        receipt_path,
        dict(
            schema=RECEIPT_SCHEMA,
            plan=file_reference(plan_path),
            observations=observations,
            completed=True,
        ),
    )
    # This establishes collection custody only; no timing or cache verdict is written.
    _, _, _, owner = load_experiment(plan_path, receipt_path)
    owner.finish()
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--definition", type=Path)
    source.add_argument("--collect-plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--reference-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-cache-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.collect_plan is not None:
        if not args.execute or args.output is not None or args.arguments:
            parser.error("--collect-plan requires --execute and accepts no output or new arguments")
        collect(
            args.collect_plan,
            args.timeout_seconds,
            args.reference_cache_index if args.reference_cache else None,
        )
        return 0
    if args.output is None:
        parser.error("--definition requires --output")
    reader = ReviewReceiptReader()
    _, definition = reader.document(file_reference(args.definition))
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    plan = build_plan(
        definition,
        args.definition.resolve().parent,
        args.output,
        args.runs,
        str(uuid.uuid4()),
        arguments,
        DEFAULT_CACHE if args.reference_cache else None,
    )
    reader.finish()
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    write_new(plan_path, plan)
    if args.execute:
        collect(
            plan_path,
            args.timeout_seconds,
            args.reference_cache_index if args.reference_cache else None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
