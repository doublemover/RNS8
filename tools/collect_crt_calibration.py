"""Collect separate CRT microcase processes without qualifying their results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import uuid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, nargs="+", default=[1, 257, 4096])
    parser.add_argument("--prefixes", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--candidates", nargs="+", choices=["garner", "direct", "balanced", "tile_balanced", "tile_convert_balanced", "tile_convert_garner"], default=["garner", "direct", "balanced"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--order-seed", type=int, default=3903)
    args = parser.parse_args()
    if (not 1 <= args.runs <= 100 or not 1 <= args.repeats <= 10000 or not 1 <= args.warmups <= 1000
            or args.timeout <= 0 or any(not 1 <= x <= 1048576 for x in args.cells)
            or any(not 1 <= p <= 20 for p in args.prefixes)):
        parser.error("invalid run, shape, prefix or timeout limits")
    exe = args.exe.resolve(strict=True)
    artifact = hashlib.sha256(exe.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [(candidate, cells, prefix, repeat) for repeat in range(args.runs)
             for candidate in args.candidates for cells in args.cells for prefix in args.prefixes]
    random.Random(args.order_seed).shuffle(cases)
    campaign = str(uuid.uuid4())
    for ordinal, (candidate, cells, prefix, repeat) in enumerate(cases):
        if hashlib.sha256(exe.read_bytes()).hexdigest() != artifact:
            raise RuntimeError("executable changed during collection")
        run_id = str(uuid.uuid4())
        command = [str(exe), candidate, str(cells), str(prefix), str(args.repeats), str(args.warmups), run_id]
        started = time.time_ns()
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired as error:
            failure = {"schema": "rns8_crt_collection_failure_v1", "run_id": run_id, "command": command,
                       "artifact_sha256": artifact, "reason": "timeout", "timeout_seconds": args.timeout}
            with (args.output / f"{run_id}.failure.json").open("x", encoding="utf-8") as stream:
                json.dump(failure, stream, indent=2)
            raise RuntimeError("CRT collection timed out; failure retained") from error
        record = {"schema": "rns8_crt_collection_v1", "campaign_id": campaign, "run_id": run_id,
                  "artifact_sha256": artifact, "command": command, "started_unix_ns": started,
                  "ordinal": ordinal, "independent_repeat": repeat, "order_seed": args.order_seed,
                  "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                  "promotion_eligible": False}
        with (args.output / f"{run_id}.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        if result.returncode:
            raise RuntimeError(f"CRT microcase failed; retained run {run_id}")
        case = json.loads(result.stdout)
        if (case.get("run_id") != run_id or case.get("candidate") != candidate or case.get("cells") != cells
                or case.get("prefix") != prefix or case.get("repeats") != args.repeats
                or len(case.get("samples", [])) != args.repeats
                or any(sample.get("exact_reference_passed") is not True for sample in case["samples"])):
            raise RuntimeError(f"CRT microcase metadata/reference mismatch; retained run {run_id}")
        tile_input = candidate.startswith("tile_")
        if (case.get("source_layout") != ("tile_major_8x16_v1" if tile_input else "modulus_major_v1")
                or case.get("conversion_inside_call") is not (tile_input and candidate != "tile_balanced")
                or case.get("conversion_scratch_bytes") != (cells * prefix if tile_input and candidate != "tile_balanced" else 0)):
            raise RuntimeError(f"CRT source layout/boundary mismatch; retained run {run_id}")
        if hashlib.sha256(exe.read_bytes()).hexdigest() != artifact:
            raise RuntimeError("executable changed while running; current capture is ineligible")
    print(f"Collected {len(cases)} independent process records in {args.output}; qualification remains pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
