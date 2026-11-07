"""Plan or explicitly evaluate bounded source-bound Python production mutations."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re

from compiled_binary_identity import canonical_digest
from execution_inventory import bounded_read
from host_instrumentation_session import (
    Session,
    check_bookends,
    invocation_path,
    reference,
    source_inventory,
    write_json,
    valid_total_budget,
)
from json_evidence import parse_evidence_json
from quality_inventory import ROOT
from seeded_mutation_contract import classification, transform

REGISTRY = ROOT / "metadata/production-mutations.json"
WORKER = ROOT / "tools/seeded_mutation_worker.py"


def registry():
    value = parse_evidence_json(bounded_read(REGISTRY)[0])
    if (
        not isinstance(value, dict)
        or value.get("schema") != "rns8_python_production_mutations_v1"
        or not isinstance(value.get("seeds"), list)
        or not 1 <= len(value["seeds"]) <= 32
    ):
        raise ValueError("bounded explicit production mutation registry required")
    ids, operations = set(), set()
    required = {
        "id",
        "module",
        "source",
        "function",
        "source_sha256",
        "before",
        "after",
        "test_module",
        "required_test",
        "assertion_marker",
        "mandatory",
    }
    for seed in value["seeds"]:
        if (
            not isinstance(seed, dict)
            or set(seed) != required
            or seed.get("mandatory") is not True
            or any(
                type(item) is not str or not item or len(item) > 4096
                for key, item in seed.items()
                if key != "mandatory"
            )
        ):
            raise ValueError("invalid production mutation seed")
        if (
            not re.fullmatch(r"[a-z][a-z0-9-]+", seed["id"])
            or seed["id"] in ids
            or not re.fullmatch(r"[a-z][a-z0-9_]+", seed["module"])
            or seed["source"] != "tools/" + seed["module"] + ".py"
            or not seed["function"].isidentifier()
            or not re.fullmatch(r"[0-9a-f]{64}", seed["source_sha256"])
            or not re.fullmatch(r"test_[A-Za-z0-9_]+", seed["test_module"])
            or not seed["required_test"].startswith(seed["test_module"] + ".")
            or seed["before"].count("\n") != seed["after"].count("\n")
        ):
            raise ValueError("mutation operation/module/source/test identity invalid")
        ids.add(seed["id"])
        operation = (seed["source"], seed["function"], seed["before"], seed["after"])
        if operation in operations:
            raise ValueError("duplicate production mutation operation")
        operations.add(operation)
        if not (ROOT / seed["source"]).resolve(strict=True).is_relative_to(ROOT / "tools"):
            raise ValueError("mutation source escaped first-party tools")
    return value


def variant(session, args, seed, prepared, *, kind, baseline=None):
    directory = session.output / seed["id"] / kind
    directory.mkdir(parents=True)
    overlay = directory / (seed["module"] + ".py")
    with overlay.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(prepared[kind])
    before = reference(overlay)
    report = directory / "tests.json"
    observation = directory / "observation.json"
    capture = session.run(
        seed["id"] + "-" + kind,
        [
            str(args.python),
            str(WORKER),
            "--seed",
            seed["id"],
            "--variant",
            kind,
            "--module",
            seed["module"],
            "--overlay",
            str(overlay),
            "--sha256",
            before["sha256"],
            "--line",
            str(prepared["line"]),
            "--output",
            str(report),
            "--observation",
            str(observation),
            "--test-module",
            seed["test_module"],
        ],
        required=False,
    )
    value = None
    observed = None
    artifacts = dict(overlay=before)
    evidence_error = None
    try:
        if report.is_file():
            raw, artifacts["test_inventory"] = bounded_read(report)
            value = parse_evidence_json(raw)
        if observation.is_file():
            raw, artifacts["observation"] = bounded_read(observation)
            observed = parse_evidence_json(raw)
    except (OSError, ValueError, TypeError) as error:
        evidence_error = str(error)
    check_bookends(artifacts)
    status = classification(
        seed,
        capture,
        value,
        observed,
        variant_sha256=before["sha256"],
        line=prepared["line"],
        baseline=baseline,
    )
    if evidence_error:
        status = "infrastructure_error"
    return dict(
        classification=status,
        artifacts=artifacts,
        process_id=capture.get("process_id"),
        returncode=capture.get("returncode"),
        evidence_error=evidence_error,
    ), value


def execute(args, plan, seeds, prepared):
    output = Path(plan["output"])
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COVERAGE_", "ASAN_", "UBSAN_", "TSAN_", "LSAN_", "LLVM_PROFILE_"))
    }
    if any(environment.get(key) for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_AUDIT")):
        raise ValueError(
            "isolated production mutation rejects inherited interpreter/runtime injection"
        )
    environment.update(PYTHONOPTIMIZE="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    output.mkdir(parents=True)
    session = Session(
        output,
        environment,
        args.timeout_seconds,
        args.artifact_mib * 1024**2,
        total_seconds=getattr(args, "total_seconds", None),
    )
    result = dict(
        schema="rns8_production_mutation_session_v1",
        status="blocked",
        plan=plan,
        blockers=[],
        seeds=[],
        production_latency=False,
        scope="curated_real_python_production_gates_not_models_native_code_or_gpu_faults",
        native_cpp_mutation="not_run",
        mull="not_run",
    )
    baselines = {}
    try:
        result["source_inventory"] = source_inventory()
        result["registry"] = reference(REGISTRY)
        result["dependency_policy"] = reference(ROOT / "metadata/dependencies.json")
        result["interpreter"] = reference(args.python)
        for seed in seeds:
            row = dict(
                id=seed["id"],
                source=reference(ROOT / seed["source"]),
                required_test=seed["required_test"],
                mandatory=True,
            )
            result["seeds"].append(row)
            if "compile_invalid" in prepared[seed["id"]]:
                row["classification"] = "compile_invalid"
                row["preparation_error"] = prepared[seed["id"]]["compile_invalid"]
                continue
            baseline, inventory = variant(
                session, args, seed, prepared[seed["id"]], kind="baseline"
            )
            row["baseline"] = baseline
            if baseline["classification"] == "baseline_passed":
                baselines[seed["id"]] = inventory
            else:
                row["classification"] = baseline["classification"]
        # Complete all independently owned baselines before the mutant batch.
        # A failed baseline is inconclusive, never a successful mutation kill.
        for seed, row in zip(seeds, result["seeds"]):
            if seed["id"] not in baselines:
                continue
            mutant, _ = variant(
                session,
                args,
                seed,
                prepared[seed["id"]],
                kind="mutant",
                baseline=baselines[seed["id"]],
            )
            row["mutant"] = mutant
            row["classification"] = mutant["classification"]
        check_bookends([result, session.stages])
        for row in result["seeds"]:
            if row["classification"] != "killed_by_assertion":
                result["blockers"].append(
                    "mandatory_mutation_unproven:" + row["id"] + ":" + row["classification"]
                )
        result["status"] = "failed" if result["blockers"] else "passed"
    except SyntaxError as error:
        result["blockers"].append("compile_invalid:" + str(error))
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["blockers"].append(str(error))
    result["stages"] = session.stages
    result["counts"] = {
        name: sum(row.get("classification") == name for row in result["seeds"])
        for name in (
            "killed_by_assertion",
            "killed_by_sanitizer",
            "survived",
            "not_covered",
            "baseline_failed",
            "compile_invalid",
            "timeout",
            "infrastructure_error",
            "unintended_assertion_failure",
        )
    }
    result["attempted_mutants"] = sum("mutant" in row for row in result["seeds"])
    result["identity_sha256"] = canonical_digest(result)
    write_json(output / "report.json", result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                report=str(output / "report.json"),
                blockers=result["blockers"],
            )
        )
    )
    return 0 if result["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--total-seconds", type=int, help="total owned child-process budget")
    parser.add_argument("--artifact-mib", type=int, default=1024)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.python = invocation_path(args.python)
        output = args.out.resolve()
        if output == ROOT / "temp" or not output.is_relative_to(ROOT / "temp") or output.exists():
            raise ValueError("fresh ignored production mutation output required")
        if (
            not 1 <= args.timeout_seconds <= 1800
            or not 128 <= args.artifact_mib <= 4096
            or not valid_total_budget(args.total_seconds)
        ):
            raise ValueError("production mutation budgets exceed supported bounds")
        seeds = registry()["seeds"]
        prepared = {}
        for seed in seeds:
            try:
                prepared[seed["id"]] = transform((ROOT / seed["source"]).read_bytes(), seed)
            except SyntaxError as error:
                prepared[seed["id"]] = dict(compile_invalid=str(error))
        # Syntax only: preparing a plan never imports or executes a mutation.
        for row in prepared.values():
            if "mutant" in row:
                ast.parse(row["mutant"])
        plan = dict(
            schema="rns8_production_mutation_plan_v1",
            total_seconds=args.total_seconds,
            python=str(args.python),
            output=str(output),
            registry=reference(REGISTRY),
            seeds=[
                dict(
                    id=seed["id"],
                    source=seed["source"],
                    source_sha256=seed["source_sha256"],
                    required_test=seed["required_test"],
                    line=prepared[seed["id"]].get("line"),
                    preparation="compile_invalid"
                    if "compile_invalid" in prepared[seed["id"]]
                    else "syntax_admitted",
                )
                for seed in seeds
            ],
            original_checkout_mutated=False,
            all_baselines_before_mutants=True,
            runtime_validation_executed=False,
            production_latency=False,
        )
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        return execute(args, plan, seeds, prepared)
    except SyntaxError as error:
        print("production mutations: compile_invalid: " + str(error))
        return 2
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("production mutations: " + str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
