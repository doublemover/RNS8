#!/usr/bin/env python3
"""Plan/replay local PDB lookup for retained native Windows CPU observations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import sys
import time

from benchmark_request import file_digest
from etl_source_contract import (
    module_points,
    decode_lookup,
    frame_address_catalog,
    address_module_candidates,
    symbolized_report,
    source_observations,
)
from host_instrumentation_session import reference
from host_profile_windows import read_retained, review_capture, seal, rebuild
from host_source_attribution import source_inventory
from json_evidence import parse_evidence_json, same_evidence_value
from pe_symbol_identity import native_image, symbol_pair
from process_progress import run_capture, capture_succeeded
from profiler_artifacts import write_diagnostic
from target_qualification import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE_INPUTS = (
    "tools/windows_cpu_symbols.py",
    "tools/etl_source_contract.py",
    "tools/rns8_pdb_symbolize.cpp",
    "tools/host_source_attribution.py",
)


def plan_symbols(acquisition_plan, helper, dbghelp, modules, out, *, review_existing=False):
    out = Path(out).resolve()
    if (
        out == (ROOT / "temp").resolve()
        or not out.is_relative_to((ROOT / "temp").resolve())
        or out.exists()
        and not review_existing
    ):
        raise ValueError("local symbol lookup needs a fresh ignored child of temp/")
    store = EvidenceStore()
    acquisition_ref, acquisition = read_retained(Path(acquisition_plan), store)
    if not same_evidence_value(acquisition, rebuild(acquisition, review_existing=True)):
        raise ValueError("Windows acquisition source/tool/production inputs changed")
    parent = Path(acquisition["output_directory"])
    if Path(acquisition_ref["path"]) != parent / "plan.json":
        raise ValueError("Windows acquisition plan moved from its owned output directory")
    summary_ref, summary = read_retained(parent / "host-profile.json", store)
    if (
        summary.get("qualification") != "native_capture_export_bound_attribution_partial"
        or not same_evidence_value(summary.get("plan"), acquisition_ref)
        or summary.get("identity_sha256") != seal(dict(summary))["identity_sha256"]
    ):
        raise ValueError("completed retained Windows acquisition required")
    inputs = []
    documents = []
    for key, name in (
        ("native_cpu_observations", "native-cpu-observations.json"),
        ("native_image_catalog", "native-image-catalog.json"),
        ("native_stack_observations", "native-stack-observations.json"),
    ):
        ref, document = read_retained(parent / name, store)
        if not same_evidence_value(summary.get(key), ref):
            raise ValueError("native Windows observation changed after acquisition")
        inputs.append(ref)
        documents.append(document)
    cpu, images, stacks = documents
    if not isinstance(modules, list) or not 1 <= len(modules) <= 32:
        raise ValueError("one to32 explicit local image/PDB/recorded-path triples required")
    pairs = []
    for image, pdb, recorded in modules:
        pair = symbol_pair(image, pdb)
        pairs.append(dict(recorded_image_path=recorded, pair=pair))
    tools = dict(helper=native_image(helper), dbghelp=native_image(dbghelp))
    if (
        any(p["pair"]["executable"]["machine"] != tools["helper"]["machine"] for p in pairs)
        or tools["helper"]["machine"] != tools["dbghelp"]["machine"]
        or Path(tools["dbghelp"]["path"]).name.lower() != "dbghelp.dll"
    ):
        raise ValueError("symbol helper, DbgHelp and PE/PDB pairs need one native architecture")
    candidates = module_points(cpu, images, pairs)
    frame_addresses = frame_address_catalog(stacks, images)
    frame_candidates = address_module_candidates(
        frame_addresses["addresses"], cpu["selected_pid"], images, pairs
    )
    requested = defaultdict(set)
    for collection in (candidates["points"], frame_candidates["points"]):
        for point in collection:
            requested[point["module_id"]].add(point["rva"])
    if sum(map(len, requested.values())) > 32768:
        raise ValueError("local symbol lookup exceeds32768 unique module/RVA budget")
    batches = []
    for module_id, rvas in sorted(requested.items()):
        ordered = sorted(rvas)
        for offset in range(0, len(ordered), 512):
            pair = pairs[module_id]["pair"]
            request = dict(
                pdb=pair["symbols"]["path"],
                dbghelp=tools["dbghelp"]["path"],
                image_size=pair["executable"]["image_size"],
                rvas=ordered[offset : offset + 512],
            )
            batches.append(
                dict(
                    module_id=module_id,
                    request=request,
                    command=[
                        tools["helper"]["path"],
                        "--input",
                        str(out / f"request-{len(batches)}.json"),
                    ],
                )
            )
    result = seal(
        dict(
            schema="rns8_windows_native_source_plan_v2",
            acquisition_plan=acquisition_ref,
            acquisition=summary_ref,
            inputs=inputs,
            modules=pairs,
            tools=tools,
            source_inputs=[reference(ROOT / p) for p in SOURCE_INPUTS],
            output_directory=str(out),
            candidates=candidates,
            frame_addresses=frame_addresses,
            frame_candidates=frame_candidates,
            batches=batches,
            commands=dict(identity=[tools["helper"]["path"], "--build-identity"]),
            budgets=dict(
                total_seconds=120,
                stage_seconds=15,
                max_artifact_bytes=128 * 1024**2,
                minimum_free_bytes=256 * 1024**2,
            ),
            execution_order="fresh_retained_ETL_review_then_file_only_local_PDB_lookup",
            qualification="plan_only_no_native_command_executed",
            promotion_allowed=False,
        )
    )
    store.finish()
    return result


def rebuild_symbols(plan, *, review_existing=False):
    return plan_symbols(
        Path(plan["acquisition_plan"]["path"]),
        Path(plan["tools"]["helper"]["path"]),
        Path(plan["tools"]["dbghelp"]["path"]),
        [
            (
                p["pair"]["executable"]["path"],
                p["pair"]["symbols"]["path"],
                p["recorded_image_path"],
            )
            for p in plan["modules"]
        ],
        Path(plan["output_directory"]),
        review_existing=review_existing,
    )


def execute_symbols(plan, *, runner=run_capture, clock=time.monotonic):
    if platform.system() != "Windows":
        raise ValueError("native local PDB lookup requires Windows")
    if not same_evidence_value(plan, rebuild_symbols(plan)):
        raise ValueError("local symbol plan inputs changed")
    out = Path(plan["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    deadline = clock() + plan["budgets"]["total_seconds"]
    stages = []

    def bounded_runner(command, **options):
        remaining = deadline - clock()
        if remaining <= 0:
            raise ValueError("local symbol lookup deadline exhausted")
        options["timeout_seconds"] = min(options["timeout_seconds"], remaining)
        return runner(command, **options)

    def publish(name, value):
        return write_diagnostic(
            out / name,
            value,
            root=out,
            total_bytes=plan["budgets"]["max_artifact_bytes"],
            minimum_free_bytes=plan["budgets"]["minimum_free_bytes"],
        )

    def call(command, name):
        remaining = deadline - clock()
        if remaining <= 0:
            raise ValueError("local symbol lookup deadline exhausted")
        value = bounded_runner(
            command,
            cwd=ROOT,
            env={"SystemRoot": os.environ.get("SystemRoot", r"C:\Windows")},
            label="windows-local-symbols:" + name,
            timeout_seconds=min(remaining, plan["budgets"]["stage_seconds"]),
            max_output_bytes=8 * 1024**2,
            artifact_roots=(out,),
            max_artifact_bytes=plan["budgets"]["max_artifact_bytes"],
            return_on_interrupt=True,
        )
        stages.append(publish(name + ".process.json", value))
        if (
            not capture_succeeded(value)
            or value.get("command") != command
            or type(value.get("process_id")) is not int
            or not 0 < value["process_id"] < 2**32
        ):
            raise ValueError("owned local symbol stage failed: " + name)
        return value, parse_evidence_json(value["stdout"])

    def review_runner(command, **options):
        value = bounded_runner(command, **options)
        stages.append(publish("retained-file-decode.process.json", value))
        return value

    publish("plan.json", plan)
    try:
        # This starts no WPR, application, exporter, registry operation or service.
        review = review_capture(Path(plan["acquisition_plan"]["path"]), runner=review_runner)
        publish("acquisition-review.json", review)
        process, identity = call(plan["commands"]["identity"], "identity")
        source_hash = file_digest(ROOT / "tools/rns8_pdb_symbolize.cpp")
        if (
            identity.get("schema") != "rns8_pdb_symbolizer_identity_v1"
            or type(identity.get("process_id")) is not int
            or identity["process_id"] != process["process_id"]
            or type(identity.get("pointer_bytes")) is not int
            or identity["pointer_bytes"] != 8
            or identity.get("source_sha256") != source_hash
        ):
            raise ValueError("local PDB helper compiled source/ABI identity differs")
        lookups = {}
        for index, batch in enumerate(plan["batches"]):
            publish(f"request-{index}.json", batch["request"])
            process, document = call(batch["command"], f"lookup-{index}")
            values = decode_lookup(
                document,
                request=batch["request"],
                pair=plan["modules"][batch["module_id"]]["pair"],
                process_id=process["process_id"],
                source_sha256=source_hash,
            )
            for rva, value in values.items():
                key = batch["module_id"], rva
                if key in lookups:
                    raise ValueError("duplicate cross-batch native symbol observation")
                lookups[key] = value
        store = EvidenceStore()
        _, cpu = store.json(plan["inputs"][0], ROOT)
        report = symbolized_report(
            cpu, plan["candidates"], plan["frame_addresses"], plan["frame_candidates"], lookups
        )
        report["current_source_custody"] = source_inventory(source_observations(report))
        store.finish()
        if not same_evidence_value(plan, rebuild_symbols(plan, review_existing=True)):
            raise ValueError("local PDB/PE/helper/acquisition inputs changed during lookup")
        if clock() > deadline:
            raise ValueError("local symbol lookup deadline exhausted during source/input review")
        report.update(
            qualification="local_PDB_leaf_and_native_fragment_lookup_conditional_on_observed_image_identity",
            acquisition_review=reference(out / "acquisition-review.json"),
            plan=reference(out / "plan.json"),
            stages=stages,
            recording_started=False,
            workload_started=False,
        )
        publish("source-report.json", seal(report))
        return 0
    except (OSError, ValueError, KeyError, TypeError, InterruptedError, KeyboardInterrupt) as error:
        publish(
            "failure.json",
            dict(
                qualification="incomplete",
                error=str(error) or type(error).__name__,
                stages=stages,
                raw_artifacts_preserved=True,
                promotion_allowed=False,
            ),
        )
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-plan", type=Path, required=True)
    parser.add_argument("--symbolizer", type=Path, required=True)
    parser.add_argument("--dbghelp", type=Path, required=True)
    parser.add_argument(
        "--module",
        nargs=3,
        action="append",
        required=True,
        metavar=("IMAGE", "PDB", "RECORDED_ETL_PATH"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = plan_symbols(
            args.acquisition_plan, args.symbolizer, args.dbghelp, args.module, args.out
        )
        if args.execute:
            return execute_symbols(plan)
        print(json.dumps(plan, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
