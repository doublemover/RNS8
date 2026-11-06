"""Plan, or explicitly execute, the optional source-bound scalar proof pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import struct

from compiled_binary_identity import canonical_digest
from compiler_context import compiler_environment
from host_instrumentation_session import (
    Session,
    check_bookends,
    invocation_path,
    reference,
    write_json,
)
from native_mutation_contract import change
from scalar_proof_contract import (
    FAMILIES,
    HARNESS,
    HEADERS,
    ROOT,
    SAFETY,
    SCHEMA,
    SOURCES,
    VERSION,
    complete_process,
    proof_result,
    property_inventory,
    registry,
    replay_result,
)


def plan(args) -> dict:
    policy = registry()
    tool_sources = sorted((ROOT / "tools").rglob("*.py"))
    if (
        not 1 <= len(tool_sources) <= 4096
        or sum(path.stat().st_size for path in tool_sources) > 64 * 1024**2
    ):
        raise ValueError("bounded Python runner/checker source inventory required")
    sources = [
        reference(ROOT / name) | dict(relative_path=name)
        for name in (
            HARNESS,
            *HEADERS,
            "metadata/scalar-proof-pilot.json",
            "tools/scalar_proof_contract.py",
            "tools/scalar_proof_session.py",
            "tools/native_mutation_contract.py",
            "tools/host_instrumentation_session.py",
            "tools/process_progress.py",
            "tools/json_evidence.py",
            "tools/compiler_context.py",
        )
    ]
    known = {row["path"] for row in sources}
    sources.extend(
        reference(path) | dict(relative_path=path.relative_to(ROOT).as_posix())
        for path in tool_sources
        if str(path.resolve()) not in known
    )
    for seed in policy["seeds"]:
        change((ROOT / seed["source"]).read_bytes(), seed)
    if args.out.exists() or not 1 <= args.timeout <= 600:
        raise ValueError("fresh output path and 1..600 second per-process budget required")
    tools = {
        name: reference(invocation_path(getattr(args, name))) for name in ("cbmc", "cmake", "cxx")
    }
    if args.toolchain:
        tools["toolchain"] = reference(args.toolchain)
    return dict(
        schema=SCHEMA,
        status="planned_not_run",
        sources=sources,
        tools=tools,
        tool_invocations={
            name: str(invocation_path(getattr(args, name))) for name in ("cbmc", "cmake", "cxx")
        },
        policy=policy,
        output=str(args.out.resolve()),
        models=args.abi,
        compiler_environment=compiler_environment(os.environ),
        timeout_seconds=args.timeout,
        output_limit_bytes=32 * 1024**2,
        artifact_limit_bytes=512 * 1024**2,
        frontend="pinned_cbmc_cpp11_frontend_must_accept_unmodified_cpp17_headers",
        source_language="c++17",
        preprocessing="same_pinned_frontend_retained_complete_output",
        safety=list(SAFETY),
        unsigned_wrap="intentional_modular_reducer_masks_not_globally_forbidden",
        assumptions=[],
        nonvacuity="total_input_domain_no_assumptions_plus_native_boundary_witnesses",
        trusted_computing_base=[
            "CBMC frontend/preprocessor/bitvector semantics and bundled SAT solver",
            "selected integer ABI and two's-complement signed conversions",
            "harness specification, property inventory and receipt readers",
            "ordinary compiler and Boost oracle for independent counterexample replay",
        ],
        excluded_claims=[
            "GPU instructions",
            "HIP synchronization/ownership",
            "compiler lowering equivalence",
            "whole-kernel exactness",
            "AUTO promotion",
            "new accumulator caps",
        ],
        documented_abi_sources=SOURCES,
        production_eligible=False,
    )


def snapshot(destination: Path, seed=None):
    destination.mkdir(parents=True)
    refs = []
    for name in (HARNESS, *HEADERS):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = (ROOT / name).read_bytes()
        target.write_bytes(change(source, seed) if seed and seed["source"] == name else source)
        refs.append(reference(target))
    return refs


def verify(
    session: Session, selected: dict, model: str, name: str, family: str, source: Path, seed=None
) -> dict:
    family_id = FAMILIES[family][0]
    base = [
        selected["tool_invocations"]["cbmc"],
        "--cpp11",
        "--" + model,
        "--function",
        "main",
        "-DRNS8_BV_CBMC=1",
        f"-DRNS8_BV_FAMILY={family_id}",
        f"-DRNS8_BV_LLP64={int(model == 'LLP64')}",
        "-I",
        str(source / "src"),
    ]
    row = dict(
        name=name,
        model=model,
        family=family,
        verdict="unknown",
        blocker=None,
        intended_negative_control=seed is not None,
        intended_properties=seed["intended_properties"] if seed else [],
        native_replay="not_run",
        source_inputs=[reference(source / item) for item in (HARNESS, *HEADERS)],
    )
    # A capability failure is retained verbatim. Never remove casts, constexpr,
    # namespaces or template dispatch to persuade an incompatible frontend.
    pre = session.run(
        name + "-" + model + "-preprocess",
        [*base, "--preprocess", str(source / HARNESS)],
        required=False,
    )
    if not complete_process(pre) or not pre.get("stdout"):
        row["blocker"] = "source_frontend_or_preprocess_failed_see_retained_process"
        return row
    preprocessed = session.output / (name + "-" + model + ".preprocessed.cpp")
    preprocessed.write_text(pre["stdout"], encoding="utf-8", newline="")
    row["preprocessed_source"] = reference(preprocessed)
    command = [*base, *SAFETY, "--json-ui", str(preprocessed)]
    discovered = session.run(
        name + "-" + model + "-properties", [*command, "--show-properties"], required=False
    )
    try:
        inventory = property_inventory(discovered, family)
    except ValueError as error:
        row["blocker"] = "source_frontend_or_property_inventory_failed: " + str(error)
        return row
    row["property_inventory"] = write_json(
        session.output / (name + "-" + model + ".properties.json"), inventory
    )
    result = session.run(
        name + "-" + model + "-proof", [*command, "--trace", "--all-properties"], required=False
    )
    row.update(
        proof_result(
            result, inventory, family, intended=seed["intended_properties"] if seed else ()
        )
    )
    row["elapsed_seconds"] = result.get("elapsed_seconds")
    row["resource_observations"] = dict(
        peak_memory_bytes=None, status="not_observed", raw_statistics="retained_process_stdout"
    )
    return row


def boundary_inputs():
    # Concrete nonempty-domain controls, not a replacement for symbolic coverage.
    return (
        [0, 0, 17, 0, 0, 0],
        [0, (1 << 64) - 1, 17, 255, 1, 1],
        [(1 << 64) - 1, 1, 17, 249, 0, 1],
        [(1 << 64) - 1, 2, 17, 253, 1, 2],
        [1 << 63, (1 << 64) - 1, 17, 254, 0, 2],
        [0xFFFFFFFF, 1, 17, 249, 1, 0],
        [125, 0, 17, 249, 2, 0],
        [127, 0, 17, 253, 3, 0],
        [128, 0, 17, 254, 4, 0],
    )


def replay(
    session: Session, selected: dict, variants: dict[str, tuple[Path, str]], rows: list[dict]
):
    host_model = "LLP64" if struct.calcsize("@l") == 4 else "LP64"
    source = session.output / "native-replay"
    source.mkdir()
    cmake = [
        "cmake_minimum_required(VERSION 3.25)",
        "project(rns8_scalar_proof_replay LANGUAGES CXX)",
        "find_package(Boost CONFIG REQUIRED)",
    ]
    host_rows = [row for row in rows if row["model"] == host_model]
    if not host_rows:
        raise ValueError("ordinary replay requires the current host ABI in the selected models")
    for row in host_rows:
        name, family = row["name"], row["family"]
        variant, _ = variants[name]
        # Only reviewed ASCII registry identifiers enter generated CMake syntax.
        relative = os.path.relpath(variant, source).replace("\\", "/")
        cmake.extend(
            [
                f'add_executable({name} "{relative}/{HARNESS}")',
                f'target_include_directories({name} PRIVATE "{relative}/src")',
                f"target_link_libraries({name} PRIVATE Boost::headers)",
                f"target_compile_features({name} PRIVATE cxx_std_17)",
                f"target_compile_definitions({name} PRIVATE RNS8_BV_CBMC=0 RNS8_BV_FAMILY={FAMILIES[family][0]} RNS8_BV_LLP64={int(host_model == 'LLP64')})",
            ]
        )
    cmake_file = source / "CMakeLists.txt"
    cmake_file.write_text("\n".join(cmake) + "\n", encoding="utf-8")
    build = session.output / "native-build"
    command = [
        selected["tool_invocations"]["cmake"],
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCMAKE_CXX_COMPILER=" + selected["tool_invocations"]["cxx"],
    ]
    if "toolchain" in selected["tools"]:
        command.append("-DCMAKE_TOOLCHAIN_FILE=" + selected["tools"]["toolchain"]["path"])
    session.run("native-replay-configure", command)
    session.run(
        "native-replay-build",
        [selected["tool_invocations"]["cmake"], "--build", str(build), "--parallel", "2"],
    )
    build_inputs = [
        reference(cmake_file),
        reference(build / "CMakeCache.txt"),
        reference(build / "compile_commands.json"),
    ]
    compiler_records = list((build / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    if len(compiler_records) != 1:
        raise ValueError("one native compiler identification record required")
    build_inputs.append(reference(compiler_records[0]))
    for row in host_rows:
        exe = build / (row["name"] + (".exe" if platform.system() == "Windows" else ""))
        row["native_executable"] = reference(exe)
        cases = (
            [item["inputs"] for item in row.get("counterexamples", [])]
            if row["intended_negative_control"]
            else boundary_inputs()
        )
        cases = [item for item in cases if item[4] < (2 if row["family"] == "checked" else 5)]
        if not cases:
            row["native_replay"] = "missing_counterexample"
            continue
        all_pass = True
        for index, fields in enumerate(cases):
            result = session.run(
                row["name"] + "-replay-" + str(index), [str(exe), *map(str, fields)], required=False
            )
            accepted = replay_result(result, row["family"], intended=row["intended_properties"])
            if row["intended_negative_control"]:
                baseline = build / (
                    "baseline_" + row["family"] + (".exe" if platform.system() == "Windows" else "")
                )
                baseline_result = session.run(
                    row["name"] + "-baseline-replay-" + str(index),
                    [str(baseline), *map(str, fields)],
                    required=False,
                )
                accepted = replay_result(baseline_result, row["family"]) and accepted
            all_pass = accepted and all_pass
        row["native_replay"] = "matched_exact_oracle" if all_pass else "failed_or_incomplete"
    return build_inputs


def execute(selected: dict) -> dict:
    output = Path(selected["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", selected)
    session = Session(
        output, dict(os.environ), selected["timeout_seconds"], selected["artifact_limit_bytes"]
    )
    report = dict(
        schema=SCHEMA,
        status="unknown",
        plan_sha256=canonical_digest(selected),
        rows=[],
        stages=session.stages,
        production_eligible=False,
        target_execution_qualified=False,
        compiler_lowering_qualified=False,
        blockers=[],
        replay_build_inputs=[],
    )
    try:
        check_bookends(selected)
        if compiler_environment(os.environ) != selected["compiler_environment"]:
            raise ValueError("compiler-affecting environment changed after planning")
        version = session.run(
            "cbmc-version", [selected["tool_invocations"]["cbmc"], "--version"], required=False
        )
        if not complete_process(version) or version.get("stdout", "").strip() != VERSION:
            raise ValueError("selected CBMC does not match the reviewed exact version")
        variants = {}
        for family in FAMILIES:
            name = "baseline_" + family
            source = output / name
            snapshot(source)
            variants[name] = (source, family)
            for model in selected["models"]:
                report["rows"].append(verify(session, selected, model, name, family, source))
        if any(row["verdict"] != "proven" for row in report["rows"]):
            raise ValueError(
                "source capability or baseline proof incomplete; negative controls not started"
            )
        for seed in selected["policy"]["seeds"]:
            source = output / seed["id"]
            snapshot(source, seed)
            variants[seed["id"]] = (source, seed["family"])
            for model in selected["models"]:
                report["rows"].append(
                    verify(session, selected, model, seed["id"], seed["family"], source, seed)
                )
        if any(
            row["verdict"] != ("refuted" if row["intended_negative_control"] else "proven")
            for row in report["rows"]
        ):
            raise ValueError(
                "negative controls did not all refute their intended arithmetic obligations"
            )
        report["replay_build_inputs"] = replay(session, selected, variants, report["rows"])
        host_model = "LLP64" if struct.calcsize("@l") == 4 else "LP64"
        if any(
            row["native_replay"] != "matched_exact_oracle"
            for row in report["rows"]
            if row["model"] == host_model
        ):
            raise ValueError("ordinary exact-oracle replay incomplete")
        report["status"] = "symbolic_properties_and_host_model_replay_complete"
        report["native_replay_model"] = host_model
        report["other_models_native_replay"] = "not_run_requires_matching_native_abi"
    except (ValueError, OSError, InterruptedError) as error:
        report["blockers"].append(str(error))
    try:
        check_bookends([selected, report])
    except (ValueError, OSError) as error:
        report["status"] = "unknown"
        report["blockers"].append(str(error))
    write_json(output / "scalar-proof.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cbmc", "cmake", "cxx", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--abi", choices=("LP64", "LLP64"), action="append")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.abi = list(dict.fromkeys(args.abi or ["LP64", "LLP64"]))
    selected = plan(args)
    result = execute(selected) if args.execute else selected
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if args.execute and result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
