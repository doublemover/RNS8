#!/usr/bin/env python3
"""Plan direct, owned Linux heaptrack allocation diagnostics; execute explicitly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

from compiled_binary_identity import canonical_digest, validate_document
from elf_module_identity import identity as elf_identity
from execution_environment_contract import relevant_environment_key
from heaptrack_native import read_allocations
from host_instrumentation_session import Session, reference, valid_total_budget, write_json
from json_evidence import same_evidence_value
from profiler_contract import production_profile_input, bind_host_profile_output
from profiler_artifacts import write_diagnostic
from target_qualification import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rns8_host_heap_plan_v1"
SOURCE_INPUTS = (
    "tools/host_heap_session.py",
    "tools/heaptrack_native.py",
    "tools/host_instrumentation_session.py",
    "tools/profiler_contract.py",
    "tools/profiler_artifacts.py",
    "tools/elf_module_identity.py",
    "tools/execution_environment_contract.py",
    "tools/execution_observation.py",
    "tools/execution_inventory.py",
    "tools/compiled_binary_identity.py",
    "tools/benchmark_request.py",
    "tools/benchmark_schema.py",
    "tools/build_instrumentation.py",
    "tools/device_observation.py",
    "tools/diagnostic_companion.py",
    "tools/companion_contract.py",
    "tools/json_evidence.py",
    "tools/target_qualification.py",
    "tools/reference_verdict.py",
    "tools/process_progress.py",
    "src/tool_common/execution_environment_json.hpp",
)
PRODUCER = "https://github.com/KDE/heaptrack/blob/v1.5.0/src/track/heaptrack_preload.cpp"


def plan_capture(
    production: Path,
    preload: Path,
    out: Path,
    *,
    total_seconds=180,
    artifact_bytes=256 * 1024**2,
    minimum_free_bytes=1024**3,
    _review_existing=False,
) -> dict:
    if not valid_total_budget(total_seconds) or total_seconds is None or total_seconds <= 30:
        raise ValueError("heap profiling requires a finite total budget with export reserve")
    if any(
        type(value) is not int or not 1 <= value <= 2 * 1024**3
        for value in (artifact_bytes, minimum_free_bytes)
    ):
        raise ValueError("heap profiling requires bounded positive artifact/free-space budgets")
    out = out.resolve()
    scratch = (ROOT / "temp").resolve()
    if out == scratch or not out.is_relative_to(scratch) or (out.exists() and not _review_existing):
        raise ValueError("heap profiling needs an absent ignored child of repository temp/")
    store = EvidenceStore()
    production_ref = reference(production)
    _, capture = store.json(production_ref, ROOT)
    executable = production_profile_input(capture, store)
    preload = preload.resolve(strict=True)
    binary, library = elf_identity(executable), elf_identity(preload)
    if (
        binary["machine"] not in (62, 183)
        or binary["machine"] != library["machine"]
        or library["elf_type"] != 3
        or any(character in str(preload) for character in (":", " ", "\t", "\n", "\r"))
    ):
        raise ValueError("one native Linux ELF64 tuple and unambiguous LD_PRELOAD path required")
    arguments = [str(executable), *capture["request_identity"]["arguments"]]
    arguments += ["--json-output", str(out / "benchmark.json")]
    if "--record-execution-environment" not in arguments:
        arguments.append("--record-execution-environment")
    observed_environment = dict(capture["request_identity"]["environment"])
    if any(
        key.startswith("HEAPTRACK_") or key == "DUMP_HEAPTRACK_OUTPUT"
        for key in observed_environment
    ):
        raise ValueError("production baseline already contains heaptrack controls")
    plan = dict(
        schema=SCHEMA,
        system="Linux",
        production_capture=production_ref,
        benchmark=reference(executable),
        preload_library=reference(preload),
        native_binary=binary,
        native_preload=library,
        source_inputs=[reference(ROOT / name) for name in SOURCE_INPUTS],
        output_directory=str(out),
        workload=arguments,
        identity_query=[str(executable), "--build-identity"],
        expected_native_environment=observed_environment,
        launch_only_environment=dict(
            LD_PRELOAD=str(preload), DUMP_HEAPTRACK_OUTPUT=str(out / "heaptrack.raw")
        ),
        native_preload_environment_policy="upstream_unsets_both_injection_keys_before_main",
        producer_contract=PRODUCER,
        budgets=dict(
            total_seconds=total_seconds,
            export_reserve_seconds=30,
            artifact_bytes=artifact_bytes,
            minimum_free_bytes=minimum_free_bytes,
        ),
        qualification="plan_only",
        promotion_allowed=False,
    )
    plan["identity_sha256"] = canonical_digest(plan)
    store.finish()
    return plan


def rebuild(plan, *, review_existing=False):
    return plan_capture(
        Path(plan["production_capture"]["path"]),
        Path(plan["preload_library"]["path"]),
        Path(plan["output_directory"]),
        total_seconds=plan["budgets"]["total_seconds"],
        artifact_bytes=plan["budgets"]["artifact_bytes"],
        minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
        _review_existing=review_existing,
    )


def execute_capture(plan: dict, *, environment=None) -> int:
    if platform.system() != "Linux":
        raise ValueError("direct heaptrack preload collection requires native Linux")
    if not same_evidence_value(plan, rebuild(plan)):
        raise ValueError("heap profile inputs/options changed before execution")
    out = Path(plan["output_directory"])
    if shutil.disk_usage(out.parent).free < plan["budgets"]["minimum_free_bytes"]:
        raise ValueError("heap profiling free-space budget unavailable")
    child = {
        key: value
        for key, value in dict(os.environ if environment is None else environment).items()
        if not relevant_environment_key(key)
        and not key.startswith("HEAPTRACK_")
        and key not in {"DUMP_HEAPTRACK_OUTPUT", "PYTHONHOME", "PYTHONPATH", "DEBUGINFOD_URLS"}
    }
    child.update(plan["expected_native_environment"])
    child["DEBUGINFOD_URLS"] = ""
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", plan)
    budgets = plan["budgets"]
    session = Session(
        out,
        child,
        budgets["total_seconds"] - 30,
        budgets["artifact_bytes"],
        total_seconds=budgets["total_seconds"] - 30,
    )
    result = dict(
        schema="rns8_host_heap_acquisition_v1",
        execution_id=str(uuid.uuid4()),
        qualification="incomplete",
        production_latency=False,
        promotion_allowed=False,
    )
    try:
        identity_process, document = session.json("compiled-identity", plan["identity_query"])
        store = EvidenceStore()
        _, production = store.json(plan["production_capture"], ROOT)
        summary = validate_document(document, production["backend_selected"])
        if (
            type(document.get("process_id")) is not int
            or not 0 < document["process_id"] < 1 << 32
            or document["process_id"] != identity_process["process_id"]
            or summary["build_manifest_sha256"] != canonical_digest(production["build_manifest"])
        ):
            raise ValueError("heap diagnostic compiled identity differs from production")
        injected = dict(child, **plan["launch_only_environment"])
        process = session.run("allocation-workload", plan["workload"], environment=injected)
        raw_path = out / "benchmark.json"
        _, raw = store.json(reference(raw_path), ROOT)
        if (
            not isinstance(raw, dict)
            or type(raw.get("process_id")) is not int
            or raw["process_id"] != process["process_id"]
        ):
            raise ValueError("heap capture does not belong to the directly owned native child")
        profiled = bind_host_profile_output(
            production,
            raw,
            workload=plan["workload"],
            environment=plan["expected_native_environment"],
            executable=Path(plan["benchmark"]["path"]),
            execution_id=result["execution_id"],
        )
        allocations = read_allocations(
            out / "heaptrack.raw",
            pid=raw["process_id"],
            execution_id=result["execution_id"],
            executable=plan["benchmark"]["path"],
        )
        store.reference(allocations["input"], ROOT)
        profiled["external_instrumentation"] = dict(
            schema="rns8_external_instrumentation_v1",
            kind="host_profiler",
            tool="heaptrack",
            mode="host_heap_interception",
        )
        if not same_evidence_value(plan, rebuild(plan, review_existing=True)):
            raise ValueError("heap diagnostic source, library or production inputs changed")
        store.finish()
        for name, document in (("allocations.json", allocations), ("profiled.json", profiled)):
            write_diagnostic(
                out / name,
                document,
                root=out,
                total_bytes=plan["budgets"]["artifact_bytes"],
                minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
            )
        result.update(
            qualification="native_allocation_generation_capture_bound_attribution_partial",
            production_capture=plan["production_capture"],
            profiled_capture=reference(out / "profiled.json"),
            native_data=allocations["input"],
            native_allocations=reference(out / "allocations.json"),
            benchmark_pid=raw["process_id"],
            preload_library=plan["preload_library"],
            unresolved=[
                "operation_phase_clock_and_thread_identity",
                "loaded_symbol_attribution",
                "complete_allocator_interception",
                "realloc_pair_identity",
            ],
        )
    except (OSError, ValueError, TypeError, KeyError, InterruptedError, KeyboardInterrupt) as error:
        result.update(error=str(error) or type(error).__name__, raw_artifacts_preserved=True)
    finally:
        result["stages"] = session.stages
        if result["qualification"].startswith("native_allocation_"):
            try:
                write_diagnostic(
                    out / "host-heap.json",
                    result,
                    root=out,
                    total_bytes=plan["budgets"]["artifact_bytes"],
                    minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
                )
            except (OSError, ValueError, TypeError) as error:
                result.update(
                    qualification="incomplete", error=str(error), raw_artifacts_preserved=True
                )
                write_json(out / "host-heap.json", result)
        else:
            write_json(out / "host-heap.json", result)
    return 0 if result["qualification"].startswith("native_allocation_") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--preload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total-seconds", type=float, default=180)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = plan_capture(
            args.production, args.preload, args.out, total_seconds=args.total_seconds
        )
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        return execute_capture(plan)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("host heap profiling: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
