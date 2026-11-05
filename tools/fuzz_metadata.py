#!/usr/bin/env python3
"""Bounded byte/property driver for the production benchmark JSON consumer."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from benchmark_schema import BenchmarkSchemaError, validate_capture, validation_errors
from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest
from gpu_qualification import write_json
from json_evidence import load_evidence_json, parse_evidence_json
from process_progress import capture_succeeded, run_capture
from quality_inventory import ROOT, inventory

MAX_BYTES, MAX_NODES, MAX_DEPTH = 1024**2, 8192, 64
FIXTURES = ROOT / "tests/fixtures/benchmark_schema"


def bounded_json(raw: bytes):
    if len(raw) > MAX_BYTES:
        raise ValueError("metadata byte budget")
    # Bound container nesting before invoking the parser, respecting quoted
    # delimiters/escapes. Malformed syntax remains the real parser's job.
    depth = 0
    quoted = escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > MAX_DEPTH:
                raise ValueError("metadata depth budget")
        elif byte in (93, 125):
            depth -= 1
    value = parse_evidence_json(raw.decode("utf-8", errors="strict"))
    pending, count = [value], 0
    while pending:
        node = pending.pop()
        count += 1
        if count > MAX_NODES:
            raise ValueError("metadata node budget")
        if isinstance(node, dict):
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return value


def exercise(raw: bytes) -> str:
    try:
        value = bounded_json(raw)
    except (ValueError, UnicodeError):
        return "parse_or_budget_rejected"
    if not isinstance(value, dict):
        return "root_rejected"
    before = canonical_digest(value)
    errors = validation_errors(value)
    if not isinstance(errors, list) or any(type(error) is not str for error in errors):
        raise AssertionError("production validator returned a malformed diagnostic inventory")
    if validation_errors(value) != errors or canonical_digest(value) != before:
        raise AssertionError("production validation is nondeterministic or mutated its input")
    try:
        accepted = validate_capture(value)
    except BenchmarkSchemaError:
        if canonical_digest(value) != before:
            raise AssertionError("production throwing validator mutated its rejected input")
        if not errors:
            raise AssertionError("production throwing and diagnostic APIs disagree")
        return "schema_rejected"
    if errors or not isinstance(accepted, dict):
        raise AssertionError("production validator accepted a rejected/malformed result")
    if canonical_digest(value) != before:
        raise AssertionError("production throwing validator mutated its input")
    return "accepted"


def paths(value):
    pending = deque([((), value)])
    while pending:
        path, node = pending.popleft()
        if path:
            yield path
        if isinstance(node, dict):
            pending.extend(((*path, key), item) for key, item in sorted(node.items()))
        elif isinstance(node, list):
            pending.extend(((*path, index), item) for index, item in enumerate(node))


def mutations(raw):
    base = bounded_json(raw)
    yield dict(kind="valid_control", path=[]), raw
    alternatives = (None, False, True, -1, 0, 1, 1 << 2048, 1.0, "", [], {}, {"authored": "wrong type"})
    for path in paths(base):
        for replacement in alternatives:
            value = copy.deepcopy(base)
            owner = value
            for key in path[:-1]:
                owner = owner[key]
            owner[path[-1]] = copy.deepcopy(replacement)
            encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            if len(encoded) <= MAX_BYTES:
                yield dict(kind="replace_value", path=list(path), replacement=replacement), encoded


def worker(plan):
    out = Path(plan["out"])
    counts, cases, started = {}, [], time.monotonic()
    result = dict(schema="rns8_metadata_property_results_v1", plan_identity_sha256=plan["identity_sha256"],
                  status="in_progress", counts=counts, cases=cases, hardware_qualification=False,
                  performance_promotion_allowed=False)
    try:
        limited = False
        for index, fixture in enumerate(plan["fixtures"]):
            raw = Path(fixture).read_bytes()
            if file_digest(fixture) != plan["source_artifacts"][fixture]:
                raise ValueError("metadata fixture changed")
            # Reserve a positive control and an equal mutation share for every
            # selected fixture; a long first fixture cannot consume the lane.
            quota = plan["case_limit"] // len(plan["fixtures"]) + (index < plan["case_limit"] % len(plan["fixtures"]))
            for ordinal, (mutation, candidate) in enumerate(mutations(raw)):
                if ordinal >= quota:
                    limited = True
                    break
                if time.monotonic()-started > plan["seconds"]:
                    raise TimeoutError("metadata property session time budget exhausted")
                # Always retain the exact current input before calling the code
                # under test. An unexpected exception is a failed run, never an
                # expected parser/type rejection.
                (out / "current-input.json").write_bytes(candidate)
                write_json(out / "current-mutation.json", dict(fixture=fixture, **mutation))
                verdict = exercise(candidate)
                if mutation["kind"] == "valid_control" and verdict != "accepted":
                    raise AssertionError("positive production metadata control did not validate")
                counts[verdict] = counts.get(verdict, 0)+1
                cases.append(dict(fixture=fixture, kind=mutation["kind"], path=mutation["path"],
                                  input_sha256=hashlib.sha256(candidate).hexdigest(), verdict=verdict))
        if not cases or counts.get("accepted", 0) == 0:
            raise AssertionError("metadata property session exercised no accepted control")
        for path, digest in plan["source_artifacts"].items():
            if file_digest(path) != digest:
                raise ValueError("metadata source/tool/fixture changed: "+path)
        if result["status"] == "in_progress":
            result["status"] = "bounded_case_limit_completed" if limited else "selected_mutations_completed"
    except Exception as error:
        # Reporting boundary only: retain the failure and return nonzero.
        result.update(status="failed", error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        if (out / "current-input.json").is_file():
            (out / "failure-input.json").write_bytes((out / "current-input.json").read_bytes())
    result["elapsed_seconds"] = time.monotonic()-started
    result["identity_sha256"] = canonical_digest(result)
    write_json(out / "worker-results.json", result)
    return 1 if result["status"] == "failed" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fixture", type=Path, action="append")
    parser.add_argument("--cases", type=int, default=2048)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker_plan:
            plan = json.loads(args.worker_plan.read_text(encoding="utf-8"))
            if canonical_digest({key:value for key,value in plan.items() if key!="identity_sha256"}) != plan["identity_sha256"]:
                raise ValueError("metadata worker plan changed")
            return worker(plan)
        if args.out is None or not 1 <= args.cases <= 32768 or not 0 < args.seconds <= 300:
            raise ValueError("fresh output and bounded case/time limits required")
        out = args.out.resolve()
        if out.exists() or out == ROOT / "temp" or not out.is_relative_to(ROOT / "temp"):
            raise ValueError("metadata fuzz output must be a fresh repository temp child")
        fixtures = sorted({path.resolve(strict=True) for path in (args.fixture or FIXTURES.glob("*.json"))})
        if not fixtures or len(fixtures)>min(128,args.cases) or any(path.stat().st_size>MAX_BYTES for path in fixtures):
            raise ValueError("bounded nonempty metadata fixture inventory required")
        sources = {str(ROOT / row.path):file_digest(ROOT / row.path) for row in inventory() if row.role=="python"}
        sources.update({str(path):file_digest(path) for path in fixtures})
        sources[str(Path(sys.executable).resolve())] = file_digest(sys.executable)
        plan = dict(schema="rns8_metadata_property_plan_v1", fixtures=[str(path) for path in fixtures], out=str(out),
                    case_limit=args.cases, seconds=args.seconds, source_artifacts=sources,
                    interpreter=sys.version, hardware_qualification=False)
        plan["identity_sha256"] = canonical_digest(plan)
        if not args.execute:
            print(json.dumps(plan, indent=2)); return 0
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "plan.json", plan)
        process = run_capture([sys.executable, str(Path(__file__).resolve()), "--worker-plan", str(out / "plan.json")],
            cwd=ROOT, label="metadata-property", timeout_seconds=args.seconds+10, max_output_bytes=1024**2,
            artifact_roots=[out], max_artifact_bytes=32*1024**2, return_on_interrupt=True)
        write_json(out / "process.json", process)
        report_path = out / "worker-results.json"
        verdict, report_reference, counts, controls = "process_failed", None, {}, set()
        if report_path.is_file() and report_path.stat().st_size <= 16*1024**2:
            report = load_evidence_json(report_path)
            if (isinstance(report,dict) and report.get("schema")=="rns8_metadata_property_results_v1"
                    and report.get("plan_identity_sha256")==plan["identity_sha256"]
                    and canonical_digest({key:value for key,value in report.items() if key!="identity_sha256"})==report.get("identity_sha256")
                    and type(report.get("counts")) is dict
                    and all(type(value) is int and value>=0 for value in report["counts"].values())
                    and sum(report["counts"].values())==len(report["cases"])):
                report_reference=dict(path=str(report_path),sha256=file_digest(report_path))
                counts=report["counts"]
                controls={row["fixture"] for row in report["cases"] if row["kind"]=="valid_control" and row["verdict"]=="accepted"}
                if capture_succeeded(process):
                    verdict=report["status"]
        accepted=verdict in {"selected_mutations_completed","bounded_case_limit_completed"} and controls==set(plan["fixtures"])
        if process.get("interrupted"):
            verdict="interrupted"
        result=dict(schema="rns8_metadata_property_session_v1",status=verdict,counts=counts,
            worker_result=report_reference,plan=dict(path=str(out/"plan.json"),sha256=file_digest(out/"plan.json")),
            process=dict(path=str(out/"process.json"),sha256=file_digest(out/"process.json")),
            accepted=accepted,hardware_qualification=False,performance_promotion_allowed=False)
        result["identity_sha256"]=canonical_digest(result)
        write_json(out/"results.json",result)
        print(json.dumps(result,indent=2))
        return 130 if process.get("interrupted") else 0 if accepted else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, str(error)+"\n")


if __name__ == "__main__":
    raise SystemExit(main())
