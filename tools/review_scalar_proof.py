"""Reconstruct a retained scalar proof session without running any native tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from compiled_binary_identity import canonical_digest
from execution_inventory import bounded_read
from host_instrumentation_session import check_bookends, reference
from json_evidence import same_evidence_value
from native_mutation_contract import change
from process_progress import capture_succeeded
from review_receipt import ReviewReceiptReader
from scalar_proof_contract import (
    FAMILIES,
    HARNESS,
    HEADERS,
    ROOT,
    SAFETY,
    SCHEMA,
    VERSION,
    command_prefix,
    complete_process,
    proof_result,
    property_inventory,
    registry,
    replay_result,
    bind_resources,
    proof_command,
)
from scalar_proof_session import (
    boundary_inputs,
    replay_build_inputs,
    replay_cmake,
    replay_commands,
    source_inputs,
)


def reviewed_session(path: Path) -> dict:
    owner = ReviewReceiptReader()
    snapshots = []

    def document(ref):
        return owner.document({key: ref[key] for key in ("path", "sha256")})[1]

    report_ref = reference(path)
    report = document(report_ref)
    output = path.resolve(strict=True).parent
    selected = document(reference(output / "plan.json"))
    if (
        report.get("schema") != SCHEMA
        or selected.get("schema") != SCHEMA
        or report.get("plan_sha256") != canonical_digest(selected)
        or selected.get("status") != "planned_not_run"
        or selected.get("output") != str(output)
        or not same_evidence_value(selected.get("sources"), source_inputs())
        or not same_evidence_value(selected.get("policy"), registry())
        or not same_evidence_value(selected.get("safety"), list(SAFETY))
        or selected.get("source_language") != "c++17"
        or selected.get("assumptions") != []
        or selected.get("production_eligible") is not False
        or report.get("production_eligible") is not False
        or report.get("target_execution_qualified") is not False
        or report.get("compiler_lowering_qualified") is not False
    ):
        raise ValueError("scalar proof source/plan/claim contract differs from the current reader")
    models = selected.get("models")
    if (
        not isinstance(models, list)
        or not 1 <= len(models) <= 2
        or any(type(model) is not str or model not in {"LP64", "LLP64"} for model in models)
        or len(set(models)) != len(models)
    ):
        raise ValueError("explicit unique scalar proof ABI models required")
    tools, invocations = selected.get("tools"), selected.get("tool_invocations")
    if (
        not isinstance(tools, dict)
        or set(tools) not in ({"cbmc", "cmake", "cxx"}, {"cbmc", "cmake", "cxx", "toolchain"})
        or not isinstance(invocations, dict)
        or set(invocations) != {"cbmc", "cmake", "cxx"}
    ):
        raise ValueError("complete selected verifier/compiler/build tool identities required")
    for name, invocation in invocations.items():
        if type(invocation) is not str or not same_evidence_value(
            reference(Path(invocation)), tools[name]
        ):
            raise ValueError("tool invocation differs from its selected executable bytes")
    python_invocation, python_version = (
        selected.get("proof_python_invocation"),
        selected.get("proof_python_version"),
    )
    if (
        type(python_invocation) is not str
        or not same_evidence_value(reference(Path(python_invocation)), selected.get("proof_python"))
        or type(python_version) is not str
        or len(python_version) > 32
        or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", python_version)
        or tuple(map(int, python_version.split("."))) < (3, 15, 0)
    ):
        raise ValueError("selected proof worker Python identity/version differs")
    check_bookends(selected)
    stages, rows = report.get("stages"), report.get("rows")
    if (
        not isinstance(stages, list)
        or not 1 <= len(stages) <= 256
        or not isinstance(rows, list)
        or len(rows) > 38
    ):
        raise ValueError("bounded scalar process and result inventories required")
    cursor = 0

    def stage(name: str, command: list[str]):
        nonlocal cursor
        if cursor >= len(stages):
            raise ValueError("missing retained scalar process: " + name)
        summary = stages[cursor]
        cursor += 1
        if not isinstance(summary, dict) or summary.get("name") != name:
            raise ValueError("scalar process order differs from the maintained driver")
        process = document(summary["receipt"])
        if (
            not same_evidence_value(process.get("command"), command)
            or any(
                not same_evidence_value(summary.get(key), process.get(key))
                for key in ("process_id", "returncode", "interrupted")
            )
            or summary.get("succeeded") is not capture_succeeded(process)
        ):
            raise ValueError("scalar process command/identity/outcome binding differs")
        return process

    version = stage("cbmc-version", [invocations["cbmc"], "--version"])
    version_ok = (
        complete_process(version)
        and type(version.get("stdout")) is str
        and version["stdout"].strip() == VERSION
    )
    expected = [
        ("baseline_" + family, family, model, None) for family in FAMILIES for model in models
    ]
    expected += [
        (seed["id"], seed["family"], model, seed)
        for seed in selected["policy"]["seeds"]
        for model in models
    ]
    if len(rows) > len(expected) or not version_ok and rows:
        raise ValueError("scalar result rows do not match the executed version/case scope")
    reconstructed = []
    for row, (name, family, model, seed) in zip(rows, expected):
        if (
            not isinstance(row, dict)
            or row.get("name") != name
            or row.get("family") != family
            or row.get("model") != model
            or row.get("intended_negative_control") is not (seed is not None)
            or not same_evidence_value(
                row.get("intended_properties"), seed["intended_properties"] if seed else []
            )
        ):
            raise ValueError("scalar result case identity differs from declared execution order")
        if seed and any(
            item["verdict"] != "proven" for item in reconstructed if not item["negative"]
        ):
            raise ValueError("negative controls ran before complete passing baseline proofs")
        source = output / name
        source_refs = []
        for relative in (HARNESS, *HEADERS):
            target = source / relative
            if not target.resolve(strict=True).is_relative_to(output):
                raise ValueError("scalar source snapshot escapes its owned output")
            raw = (ROOT / relative).read_bytes()
            expected_bytes = change(raw, seed) if seed and seed["source"] == relative else raw
            actual_text, ref = bounded_read(target)
            if actual_text.encode("utf-8") != expected_bytes:
                raise ValueError(
                    "scalar source snapshot differs from its exact production/mutation source"
                )
            source_refs.append(reference(target))
            snapshots.append(ref)
        if not same_evidence_value(row.get("source_inputs"), source_refs):
            raise ValueError("scalar row source inventory differs")
        base = command_prefix(selected, model, family, source)
        pre = stage(
            name + "-" + model + "-preprocess", [*base, "--preprocess", str(source / HARNESS)]
        )
        facts = dict(verdict="unknown", blocker=None, properties=[], counterexamples=[])
        if not complete_process(pre) or not pre.get("stdout"):
            facts["blocker"] = "source_frontend_or_preprocess_failed_see_retained_process"
        else:
            preprocessed = output / (name + "-" + model + ".preprocessed.cpp")
            pre_text, pre_ref = bounded_read(preprocessed)
            snapshots.append(pre_ref)
            if pre_text != pre["stdout"] or not same_evidence_value(
                reference(preprocessed), row.get("preprocessed_source")
            ):
                raise ValueError(
                    "retained scalar preprocessor input differs from its actual process output"
                )
            command = [*base, *SAFETY, "--json-ui", str(preprocessed)]
            discovered = stage(name + "-" + model + "-properties", [*command, "--show-properties"])
            try:
                inventory = property_inventory(discovered, family)
            except ValueError as error:
                facts["blocker"] = "source_frontend_or_property_inventory_failed: " + str(error)
            else:
                if not same_evidence_value(document(row["property_inventory"]), inventory):
                    raise ValueError(
                        "retained scalar property inventory differs from native discovery"
                    )
                invocation, resource_path = proof_command(selected, name, model, command)
                result = stage(name + "-" + model + "-proof", invocation)
                facts = proof_result(
                    result, inventory, family, intended=seed["intended_properties"] if seed else ()
                )
                if not same_evidence_value(
                    row.get("elapsed_seconds"), result.get("elapsed_seconds")
                ):
                    raise ValueError("scalar proof elapsed observation differs from its process")
                resource = None
                if row.get("resource_receipt") is not None:
                    if Path(row["resource_receipt"]["path"]).resolve(
                        strict=True
                    ) != resource_path.resolve(strict=True):
                        raise ValueError(
                            "child resource receipt belongs to a different proof command"
                        )
                    resource = document(row["resource_receipt"])
                elif resource_path.exists():
                    raise ValueError("child resource receipt was omitted from the proof summary")
                observed = bind_resources(
                    facts,
                    resource,
                    result,
                    [*command, "--trace", "--all-properties"],
                    selected=selected,
                )
                facts["resource_observations"] = observed
        # Early capability failures do not publish property/result fields.
        for key, value in facts.items():
            actual = row.get(key, [] if key in {"properties", "counterexamples"} else None)
            if not same_evidence_value(actual, value):
                raise ValueError("derived scalar " + key + " differs from the retained raw results")
        reconstructed.append(
            dict(name=name, family=family, model=model, negative=seed is not None, **facts)
        )
    complete = (
        version_ok
        and len(rows) == len(expected)
        and all(
            row["verdict"] == ("refuted" if row["negative"] else "proven") for row in reconstructed
        )
    )
    if complete:
        host = selected.get("native_host")
        if (
            host not in (dict(os="Windows", abi="LLP64"), dict(os="Linux", abi="LP64"))
            or host["abi"] not in models
        ):
            raise ValueError("native replay has no supported selected host ABI")
        model = host["abi"]
        cmake_file = output / "native-replay/CMakeLists.txt"
        cmake_text, cmake_ref = bounded_read(cmake_file)
        snapshots.append(cmake_ref)
        if cmake_text != replay_cmake(output, rows, model):
            raise ValueError("native scalar replay project differs from its reviewed source recipe")
        configure, compile_command = replay_commands(selected, output)
        configured = stage("native-replay-configure", configure)
        compiled = (
            stage("native-replay-build", compile_command) if complete_process(configured) else None
        )
        complete = compiled is not None and complete_process(compiled)
        if complete:
            if not same_evidence_value(
                report.get("replay_build_inputs"), replay_build_inputs(output)
            ):
                raise ValueError("native replay compiler/project/build input inventory differs")
            for row, facts in zip(rows, reconstructed):
                if row["model"] != model:
                    if row.get("native_replay") != "not_run":
                        raise ValueError("non-native ABI claims ordinary replay")
                    continue
                suffix = ".exe" if host["os"] == "Windows" else ""
                exe = output / "native-build" / (row["name"] + suffix)
                if not same_evidence_value(reference(exe), row.get("native_executable")):
                    raise ValueError("native scalar replay executable changed")
                cases = (
                    [item["inputs"] for item in facts["counterexamples"]]
                    if facts["negative"]
                    else boundary_inputs()
                )
                cases = [
                    item for item in cases if item[4] < (2 if row["family"] == "checked" else 5)
                ]
                passed = bool(cases)
                for index, fields in enumerate(cases):
                    result = stage(
                        row["name"] + "-replay-" + str(index), [str(exe), *map(str, fields)]
                    )
                    accepted = replay_result(
                        result, row["family"], intended=row["intended_properties"]
                    )
                    if facts["negative"]:
                        baseline = output / "native-build" / ("baseline_" + row["family"] + suffix)
                        result = stage(
                            row["name"] + "-baseline-replay-" + str(index),
                            [str(baseline), *map(str, fields)],
                        )
                        accepted = replay_result(result, row["family"]) and accepted
                    passed = passed and accepted
                expected_replay = "matched_exact_oracle" if passed else "failed_or_incomplete"
                if row.get("native_replay") != expected_replay:
                    raise ValueError(
                        "native scalar replay summary differs from exact-oracle processes"
                    )
                complete = complete and passed
    if cursor != len(stages):
        raise ValueError("scalar session contains unbound or unexpected process stages")
    status = "symbolic_properties_and_host_model_replay_complete" if complete else "unknown"
    if report.get("status") != status:
        raise ValueError(
            "scalar session status differs from reconstructed acquisition/replay facts"
        )
    if complete and (
        report.get("blockers") != []
        or report.get("native_replay_model") != selected["native_host"]["abi"]
    ):
        raise ValueError("complete scalar session has inconsistent blockers or native ABI")
    check_bookends([selected, report, snapshots])
    owner.finish()
    return dict(
        schema=SCHEMA + "_review",
        status=status,
        rows=reconstructed,
        report=report_ref,
        production_eligible=False,
        target_execution_qualified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        result = reviewed_session(args.report)
    except (ValueError, OSError, KeyError, TypeError) as error:
        result = dict(
            schema=SCHEMA + "_review",
            status="unknown",
            blocker=str(error),
            production_eligible=False,
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "symbolic_properties_and_host_model_replay_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
