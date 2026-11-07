#!/usr/bin/env python3
"""Plan an owned Windows RDP dispatch capture; raw RGP is manual-review evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import threading
import time
import uuid

from amd_profiler_capabilities import validate_preflight
from compiled_binary_identity import canonical_digest, bind_document
from execution_environment_contract import relevant_environment_key
from gpu_qualification import ROOT, owned_artifact_files, write_json
from host_instrumentation_session import reference
from json_evidence import parse_evidence_json, same_evidence_value
from process_progress import capture_succeeded, run_capture
from profiler_contract import production_profile_input, bind_host_profile_output
from target_qualification import EvidenceStore

SCHEMA = "rns8_windows_rdp_capture_v1"
SOURCE_INPUTS = (
    "tools/amd_profiler_windows.py",
    "tools/amd_profiler_capabilities.py",
    "tools/process_progress.py",
    "tools/profiler_contract.py",
    "tools/compiled_binary_identity.py",
    "tools/target_qualification.py",
    "tools/gpu_qualification.py",
    "tools/benchmark_request.py",
    "tools/benchmark_schema.py",
    "tools/device_observation.py",
    "tools/execution_observation.py",
    "tools/execution_environment_contract.py",
    "tools/diagnostic_companion.py",
    "tools/companion_contract.py",
    "tools/measurement_contract.py",
    "tools/reference_verdict.py",
    "tools/build_instrumentation.py",
    "tools/json_evidence.py",
    "tools/host_instrumentation_session.py",
)
NATIVE_POLICY = "https://github.com/GPUOpen-Tools/radeon_developer_panel/blob/fcc2ec5e8ed7f3432e59297a59fcec740434ee4b/source/cli/capture_cli.cpp"
FLAGS = (
    "--mode",
    "--process",
    "--output",
    "--remote-host",
    "--remote-port",
    "--rgp-auto-capture",
    "--rgp-capture-mode",
    "--rgp-render-op-count",
)


def plan_capture(
    production: Path,
    capabilities: Path,
    out: Path,
    *,
    first_dispatch=1,
    dispatch_count=1,
    counters=False,
    instruction_trace=False,
    timeout_seconds=120,
    total_seconds=240,
    max_artifact_bytes=1024**3,
    minimum_free_bytes=5 * 1024**3,
    _review_existing=False,
):
    if (
        type(first_dispatch) is not int
        or not 1 <= first_dispatch <= 10**9
        or type(dispatch_count) is not int
        or not 1 <= dispatch_count <= 1024
    ):
        raise ValueError("explicit bounded native dispatch indices required")
    if type(counters) is not bool or type(instruction_trace) is not bool:
        raise ValueError("counter/instruction tracing policy must be Boolean")
    if (
        any(
            type(value) not in (int, float) or not 0 < value <= 1800 or not math.isfinite(value)
            for value in (timeout_seconds, total_seconds)
        )
        or total_seconds <= timeout_seconds + 30
    ):
        raise ValueError("bounded time budgets with a 30-second cleanup reserve required")
    if any(
        type(value) is not int or value < 1 for value in (max_artifact_bytes, minimum_free_bytes)
    ):
        raise ValueError("positive artifact and free-space byte budgets required")
    out = out.resolve()
    if (
        out == (ROOT / "temp").resolve()
        or not out.is_relative_to((ROOT / "temp").resolve())
        or out.exists()
        and not _review_existing
    ):
        raise ValueError("a fresh ignored child of temp/ is required")
    store = EvidenceStore()
    inputs = dict(production_capture=reference(production), capabilities=reference(capabilities))
    _, capture = store.json(inputs["production_capture"], ROOT)
    executable = production_profile_input(capture, store)
    preflight = validate_preflight(inputs["capabilities"], store=store, parent=ROOT)
    tool = preflight["tools"].get("RadeonDeveloperPanelCLI", {})
    if (
        preflight["system"] != "Windows"
        or preflight["target"] != capture["device"]["gcn_arch"].split(":")[0]
        or preflight["features"]["timeline"]["official_support"] != "supported"
        or tool.get("version_state") != "supported"
    ):
        raise ValueError(
            "maintained Windows RDP version and official HIP target preflight required"
        )
    _, raw_preflight = store.json(inputs["capabilities"], ROOT)
    required = [
        *FLAGS,
        *(("--rgp-counter-collection",) if counters else ()),
        *(("--rgp-instruction-tracing",) if instruction_trace else ()),
    ]
    help_rows = [row for row in raw_preflight["observations"] if row["name"] == "help"]
    if len(help_rows) != 1 or any(
        flag not in help_rows[0]["result"]["stdout"] for flag in required
    ):
        raise ValueError("RDP profiling help lacks the requested maintained flags")
    workload = [
        str(executable),
        *capture["request_identity"]["arguments"],
        "--json-output",
        str(out / "raw/benchmark.json"),
    ]
    if "--record-execution-environment" not in workload:
        workload.append("--record-execution-environment")
    # Loopback explicitly selects an existing headless service. No service,
    # GUI, registry configuration or clocks command is launched by this lane.
    controller = [
        tool["path"],
        "--mode",
        "profiling",
        "--remote-host",
        "127.0.0.1",
        "--remote-port",
        "27300",
        "--process",
        str(executable),
        "--output",
        str(out / "raw/capture.rgp"),
        "--rgp-auto-capture",
        f"dispatch:{first_dispatch}:{dispatch_count}",
        "--rgp-capture-mode",
        "dispatch",
        "--rgp-render-op-count",
        str(dispatch_count),
    ]
    controller += ["--rgp-counter-collection"] if counters else []
    controller += ["--rgp-instruction-tracing"] if instruction_trace else []
    result = dict(
        schema=SCHEMA,
        **inputs,
        output_directory=str(out),
        executable=reference(executable),
        controller=reference(Path(tool["path"])),
        source_inputs=[reference(ROOT / name) for name in SOURCE_INPUTS],
        native_policy=NATIVE_POLICY,
        first_dispatch=first_dispatch,
        dispatch_count=dispatch_count,
        counters=counters,
        instruction_trace=instruction_trace,
        expected_native_environment=capture["request_identity"]["environment"],
        commands=dict(
            identity=[str(executable), "--build-identity"], controller=controller, workload=workload
        ),
        budgets=dict(
            timeout_seconds=timeout_seconds,
            total_seconds=total_seconds,
            cleanup_reserve_seconds=30,
            max_artifact_bytes=max_artifact_bytes,
            minimum_free_bytes=minimum_free_bytes,
        ),
        connection_policy="existing_loopback_service_only_no_service_start",
        dispatch_namespace="native_RDP_process_dispatch_indices_not_measured_operation_or_HIP_ordinal",
        qualification="plan_only",
        performance_promotion_allowed=False,
    )
    result["identity_sha256"] = canonical_digest(result)
    store.finish()
    return result


def rebuild(plan, *, review_existing=False):
    return plan_capture(
        Path(plan["production_capture"]["path"]),
        Path(plan["capabilities"]["path"]),
        Path(plan["output_directory"]),
        first_dispatch=plan["first_dispatch"],
        dispatch_count=plan["dispatch_count"],
        counters=plan["counters"],
        instruction_trace=plan["instruction_trace"],
        **{
            key: value for key, value in plan["budgets"].items() if key != "cleanup_reserve_seconds"
        },
        _review_existing=review_existing,
    )


def completed_capture(plan, stages, *, store):
    if not same_evidence_value(plan, rebuild(plan, review_existing=True)):
        raise ValueError("RDP plan/source inputs changed before review")
    if set(stages) != set(plan["commands"]):
        raise ValueError("incomplete RDP acquisition stages")
    out = Path(plan["output_directory"])
    for item in (plan["controller"], plan["executable"], *plan["source_inputs"]):
        store.reference(item, ROOT)
    for name, row in stages.items():
        if (
            not capture_succeeded(row)
            or row.get("command") != plan["commands"][name]
            or type(row.get("process_id")) is not int
            or row["process_id"] <= 0
        ):
            raise ValueError("unsuccessful or unrelated RDP stage: " + name)
        for stream in ("stdout", "stderr"):
            path = out / (name + "." + stream + ".log")
            if (
                store.verify(path, row["streams"][stream]["sha256"]).read_text(encoding="utf-8")
                != row[stream]
            ):
                raise ValueError("RDP stage summary differs from retained stream")
    identity_path = out / "identity.json"
    identity = parse_evidence_json(stages["identity"]["stdout"])
    if identity.get("process_id") != stages["identity"]["process_id"]:
        raise ValueError("compiled identity belongs to another query process")
    if not same_evidence_value(store.json(reference(identity_path), ROOT)[1], identity):
        raise ValueError("compiled identity document differs from native query")
    _, production = store.json(plan["production_capture"], ROOT)
    binary = store.binary(
        bind_document(
            Path(plan["executable"]["path"]),
            identity_path,
            before_sha256=plan["executable"]["sha256"],
        ),
        production["backend_selected"],
    )
    if binary["build_manifest_sha256"] != canonical_digest(production["build_manifest"]):
        raise ValueError("RDP benchmark compiled build differs from production")
    workload = stages["workload"]
    if workload["native_output"]["path"] != str(out / "raw/benchmark.json"):
        raise ValueError("RDP native workload output escaped its owned path")
    _, raw = store.json(workload["native_output"], ROOT)
    if raw.get("process_id") != workload["process_id"]:
        raise ValueError("profiled benchmark belongs to another process")
    profiled = bind_host_profile_output(
        production,
        raw,
        workload=plan["commands"]["workload"],
        environment=plan["expected_native_environment"],
        executable=Path(plan["executable"]["path"]),
        execution_id=stages["workload"]["execution_id"],
    )
    native_log = stages["controller"]["stdout"]
    connected = re.findall(
        r"^Connecting to process: (.+) \(PID: ([1-9][0-9]*)\)$", native_log, re.MULTILINE
    )
    if not connected or any(
        Path(path).resolve() != Path(plan["executable"]["path"]).resolve()
        or int(pid) != workload["process_id"]
        for path, pid in connected
    ):
        raise ValueError("RDP connection filter did not select only the owned benchmark PID/path")
    trace = out / "raw/capture.rgp"
    raw_files = owned_artifact_files(out / "raw", out)
    if set(raw_files) != {trace, out / "raw/benchmark.json"}:
        raise ValueError("RDP acquisition has unknown or missing raw outputs")
    native = stages["controller"]["native_trace"]
    store.reference(native, ROOT)
    size = trace.stat().st_size
    if (
        size < 1
        or native["path"] != str(trace)
        or re.findall(r"^Trace saved to: (.+)$", native_log, re.MULTILINE) != [str(trace)]
        or re.findall(r"^Size: ([0-9]+) bytes$", native_log, re.MULTILINE) != [str(size)]
    ):
        raise ValueError("RDP save report differs from its nonempty retained bytes")
    if (
        sum(path.stat().st_size for path in owned_artifact_files(out, out.parent))
        > plan["budgets"]["max_artifact_bytes"]
    ):
        raise ValueError("RDP session exceeds its total retained artifact budget")
    profiled["external_instrumentation"] = dict(
        schema="rns8_external_instrumentation_v1",
        kind="amd_profiler",
        tool="RadeonDeveloperPanelCLI",
        mode="native_dispatch_capture",
    )
    return dict(
        schema=SCHEMA,
        capture_state="raw_saved_unqualified",
        native_trace=native,
        profiled_capture=profiled,
        process_id=workload["process_id"],
        connected_processes=len(connected),
        controller_readiness="not_established_by_process_creation",
        raw_format="opaque_RGP_requires_supported_native_decoder_or_manual_review",
        device_in_raw_capture="unverified",
        selected_dispatch_identity="unverified",
        semantic_operation_range="unbound_native_dispatch_indices_only",
        counter_values="unavailable_without_supported_machine_export",
        instruction_trace="requested_not_decoded" if plan["instruction_trace"] else "not_requested",
        complete_run_coverage=False,
        performance_promotion_allowed=False,
    )


def review_capture(plan_path: Path):
    """Reopen raw stage streams and bytes; no controller or workload is started."""
    store = EvidenceStore()
    path, plan = store.json(reference(plan_path), ROOT)
    _, stages = store.json(reference(path.parent / "stages.json"), ROOT)
    summary = completed_capture(plan, stages, store=store)
    _, retained = store.json(reference(path.parent / "capture-summary.json"), ROOT)
    if not same_evidence_value(summary, retained):
        raise ValueError("retained RDP summary differs from reconstructed raw evidence")
    store.finish()
    return summary


def publish_summary(path, summary, plan):
    raw = (
        json.dumps(summary, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    out = Path(plan["output_directory"])
    retained = sum(item.stat().st_size for item in owned_artifact_files(out, out.parent))
    if len(raw) > 64 * 1024**2 or retained + len(raw) > plan["budgets"]["max_artifact_bytes"]:
        raise ValueError("derived RDP summary exceeds its retained artifact budget")
    if shutil.disk_usage(out).free < plan["budgets"]["minimum_free_bytes"] + len(raw):
        raise ValueError("derived RDP summary lacks its free-space reserve")
    with path.open("xb") as stream:
        stream.write(raw)


def execute_capture(
    plan,
    *,
    authorize_driver_profile=False,
    environment=None,
    runner=run_capture,
    clock=time.monotonic,
):
    if platform.system() != "Windows" or authorize_driver_profile is not True:
        raise ValueError(
            "Windows driver profiling requires --execute --authorize-driver-profile; no service is started"
        )
    if not same_evidence_value(plan, rebuild(plan)):
        raise ValueError("RDP acquisition plan changed before execution")
    out = Path(plan["output_directory"])
    parent = next(path for path in out.parents if path.exists())
    if shutil.disk_usage(parent).free < plan["budgets"]["minimum_free_bytes"]:
        raise ValueError("RDP capture lacks its free-space reserve")
    child = {
        key: value
        for key, value in dict(os.environ if environment is None else environment).items()
        if not relevant_environment_key(key)
    }
    child.update(plan["expected_native_environment"])
    (out / "raw").mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", plan)
    stages, errors = {}, []
    started, finished, cancel = threading.Event(), threading.Event(), threading.Event()
    deadline = (
        clock() + plan["budgets"]["total_seconds"] - plan["budgets"]["cleanup_reserve_seconds"]
    )

    def call(name, **options):
        remaining = deadline - clock()
        if remaining <= 0:
            raise ValueError("RDP session deadline exhausted")
        row = runner(
            plan["commands"][name],
            cwd=ROOT,
            env=child,
            label="windows-rdp:" + name,
            timeout_seconds=min(remaining, plan["budgets"]["timeout_seconds"]),
            max_output_bytes=4 * 1024**2,
            artifact_roots=(out,),
            max_artifact_bytes=plan["budgets"]["max_artifact_bytes"],
            return_on_interrupt=True,
            **options,
        )
        row["streams"] = {}
        for stream in ("stdout", "stderr"):
            path = out / (name + "." + stream + ".log")
            path.write_text(row[stream], encoding="utf-8", newline="\n")
            row["streams"][stream] = reference(path)
        stages[name] = row
        if not capture_succeeded(row):
            raise ValueError("RDP stage failed: " + name)
        return row

    def controller():
        try:
            call("controller", on_start=lambda pid: started.set(), cancel_event=cancel)
        except BaseException as error:
            errors.append(type(error).__name__ + ": " + str(error))
        finally:
            finished.set()

    thread = None
    success = False
    try:
        identity = call("identity")
        write_json(out / "identity.json", parse_evidence_json(identity["stdout"]))
        # Validate the actual loaded binary/library pair before profiling.
        store = EvidenceStore()
        _, baseline = store.json(plan["production_capture"], ROOT)
        binary = store.binary(
            bind_document(
                Path(plan["executable"]["path"]),
                out / "identity.json",
                before_sha256=plan["executable"]["sha256"],
            ),
            baseline["backend_selected"],
        )
        if (
            binary["build_manifest_sha256"] != canonical_digest(baseline["build_manifest"])
            or parse_evidence_json(identity["stdout"]).get("process_id") != identity["process_id"]
        ):
            raise ValueError("RDP compiled identity differs from production")
        store.finish()
        thread = threading.Thread(target=controller, name="rns8-owned-rdp-controller", daemon=False)
        thread.start()
        while not started.wait(0.05):
            if finished.is_set() or clock() >= deadline:
                raise ValueError("RDP controller process was not created")
        if finished.is_set():
            raise ValueError("RDP controller ended before the workload")
        workload = call("workload")
        workload["native_output"] = reference(out / "raw/benchmark.json")
        workload["execution_id"] = str(uuid.uuid4())
        if not finished.wait(max(0, deadline - clock())):
            raise ValueError("RDP capture did not finish before its deadline")
        if errors:
            raise ValueError("; ".join(errors))
        stages["controller"]["native_trace"] = reference(out / "raw/capture.rgp")
        write_json(out / "stages.json", stages)
        store = EvidenceStore()
        summary = completed_capture(plan, stages, store=store)
        store.finish()
        publish_summary(out / "capture-summary.json", summary, plan)
        success = True
    except (OSError, ValueError, TypeError, KeyError, InterruptedError, KeyboardInterrupt) as error:
        write_json(
            out / "capture-summary.json",
            dict(
                schema=SCHEMA,
                capture_state="failed_or_partial",
                error=str(error),
                raw_artifacts_preserved=True,
                performance_promotion_allowed=False,
            ),
        )
    finally:
        cancel.set()
        if thread is not None:
            thread.join(plan["budgets"]["cleanup_reserve_seconds"])
            if thread.is_alive():
                success = False
                write_json(
                    out / "capture-summary.json",
                    dict(
                        schema=SCHEMA,
                        capture_state="owned_cleanup_incomplete",
                        raw_artifacts_preserved=True,
                        performance_promotion_allowed=False,
                    ),
                )
        write_json(out / "stages.json", dict(stages))
    return 0 if success else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production", type=Path)
    parser.add_argument("--capabilities", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--first-dispatch", type=int, default=1)
    parser.add_argument("--dispatch-count", type=int, default=1)
    parser.add_argument("--counters", action="store_true")
    parser.add_argument("--instruction-trace", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-driver-profile", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = plan_capture(
            args.production,
            args.capabilities,
            args.out_dir,
            first_dispatch=args.first_dispatch,
            dispatch_count=args.dispatch_count,
            counters=args.counters,
            instruction_trace=args.instruction_trace,
        )
        if args.execute:
            return execute_capture(plan, authorize_driver_profile=args.authorize_driver_profile)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
