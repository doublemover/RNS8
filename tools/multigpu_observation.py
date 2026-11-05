"""Bind existing session visibility observations to actual benchmark captures."""

from __future__ import annotations

from pathlib import Path

from benchmark_schema import load_capture, validate_capture
from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest
from device_observation import capture_relation, inspection_observation
from json_evidence import load_evidence_json, parse_evidence_json, same_evidence_value
from reference_verdict import numerical_contract
from session_environment import RUNTIME_MAPPING_KEYS, VISIBILITY_KEYS
from process_progress import capture_succeeded
from target_qualification import (
    EvidenceStore,
    candidate_binding,
    object_value,
    text,
    session_stage,
)

SCHEMA = "rns8_multigpu_observation_binding_v1"


def _passed(result: dict) -> None:
    if not capture_succeeded(result) or result.get("validation_errors") != []:
        raise ValueError("observation stage did not complete successfully")


def _stdout(item: dict, result: dict, results_parent: Path, store: EvidenceStore) -> dict:
    _passed(result)
    output = text(result.get("stdout"))
    path = (results_parent / (text(item.get("name")) + ".stdout.log")).resolve(strict=True)
    store.verify(path, object_value(result.get("artifacts")).get(str(path)))
    if path.read_text(encoding="utf-8") != output:
        raise ValueError("session observation differs from its raw stdout artifact")
    document = object_value(parse_evidence_json(output))
    pid = result.get("process_id")
    if (
        type(pid) is not int
        or pid <= 0
        or document.get("process_id") != pid
        or type(document.get("process_id")) is not int
    ):
        raise ValueError("native observation does not belong to its launched process")
    return document


def _environment(result: dict, *, selected: int | None, rank: int, world: int) -> dict:
    environment = object_value(result.get("execution_environment"))
    if set(environment) != set(RUNTIME_MAPPING_KEYS):
        raise ValueError("observation has no complete versioned runtime mapping environment")
    for key in VISIBILITY_KEYS:
        expected = str(selected) if key == "HIP_VISIBLE_DEVICES" and selected is not None else None
        if environment[key] != expected:
            raise ValueError("observed visibility differs from its explicit parent/child namespace")
    if selected is not None:
        for key, expected in dict(
            RNS8_RANK=str(rank),
            RNS8_WORLD_SIZE=str(world),
            RNS8_MULTI_GPU_MODE="embarrassingly_parallel_shards",
        ).items():
            if environment[key] != expected:
                raise ValueError("observed rank/world/mode differs from the session allocation")
    return environment


def _query_receipt(
    name: str,
    stages: dict,
    directory: Path,
    cwd: Path,
    store: EvidenceStore,
    backend: str,
    manifest_sha: str,
) -> dict:
    pair = stages.get(name)
    if pair is None:
        raise ValueError("compiled query stage is absent")
    item, result = pair
    if item.get("compiled_identity") is not True:
        raise ValueError("observation binary was not queried through its compiled identity API")
    document = _stdout(item, result, directory, store)
    receipt = store.binary(result.get("compiled_binary_identity"), backend)
    arguments = item.get("argv")
    if (
        not isinstance(arguments, list)
        or len(arguments) != 2
        or arguments[1] != "--build-identity"
        or (cwd / text(arguments[0])).resolve(strict=True) != Path(receipt["executable"])
    ):
        raise ValueError("query executable/arguments differ from the compiled receipt")
    if receipt["build_manifest_sha256"] != manifest_sha:
        raise ValueError("observation binary comes from a different build/configuration")
    _, stored = store.json(
        dict(
            path=receipt["identity_document"],
            sha256=receipt["identity_document_sha256"],
        ),
        cwd,
    )
    if not same_evidence_value(stored, document):
        raise ValueError("compiled query document differs from its actual process output")
    return receipt


def _observe(
    capture_path: Path,
    capture: dict,
    binding: dict,
    proof: dict,
    parent: Path,
    store: EvidenceStore,
) -> dict:
    item, result, _, _, stages = session_stage(proof, parent, store)
    _passed(result)
    manifest_path, manifest = store.json(proof.get("session_manifest"), parent)
    results_path = store.reference(proof.get("session_results"), parent)
    cwd = Path(text(manifest.get("command_working_directory"))).resolve(strict=True)
    devices = manifest.get("devices")
    if (
        not isinstance(devices, list)
        or not devices
        or any(type(value) is not int or value < 0 for value in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("session has no explicit unique parent runtime ordinal allocation")
    selected = item.get("selected_device")
    if type(selected) is not int or selected not in devices or item.get("device") != selected:
        raise ValueError("capture device is not in the session's declared visibility allocation")
    rank, world = devices.index(selected), len(devices)
    if (cwd / text(item.get("capture"))).resolve(strict=True) != capture_path.resolve(strict=True):
        raise ValueError("session capture stage names a different raw file")
    if (
        object_value(result.get("artifacts")).get(str(capture_path.resolve()))
        != binding["capture_sha256"]
    ):
        raise ValueError("capture bytes are not an immutable output of that session stage")
    environment = _environment(result, selected=selected, rank=rank, world=world)
    request = object_value(capture.get("request_identity"))
    for key in RUNTIME_MAPPING_KEYS:
        if object_value(request.get("environment")).get(key) != environment[key]:
            raise ValueError("capture request and actual child mapping environments differ")
    emitted = object_value(capture.get("runtime_environment"))
    for key in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "RNS8_RANK",
        "RNS8_WORLD_SIZE",
        "RNS8_MULTI_GPU_MODE",
    ):
        if emitted.get(key) != environment[key]:
            raise ValueError("benchmark-observed environment contradicts its subprocess request")
    arguments = item.get("argv")
    if (
        not isinstance(arguments, list)
        or not arguments
        or not same_evidence_value(arguments[1:], request.get("arguments"))
    ):
        raise ValueError("capture command differs from its actual recorded request")
    identity = object_value(binding["identity"])
    if type(result.get("process_id")) is not int or result["process_id"] != capture.get(
        "process_id"
    ):
        raise ValueError("capture process identity differs from its actual launched benchmark")
    benchmark = _query_receipt(
        text(proof.get("binary_identity_stage")),
        stages,
        results_path.parent,
        cwd,
        store,
        capture["backend_requested"],
        identity["build_manifest_sha256"],
    )
    if (cwd / text(arguments[0])).resolve(strict=True) != Path(
        benchmark["executable"]
    ) or benchmark["executable_sha256"] != identity["executable_sha256"]:
        raise ValueError("capture executed an artifact other than the queried benchmark")
    parent_name, child_name = (
        text(item.get("observation_parent_stage")),
        text(item.get("observation_child_stage")),
    )
    if proof.get("parent_stage") != parent_name or proof.get("child_stage") != child_name:
        raise ValueError("capture mapping substituted another observation stage")
    if parent_name not in stages or child_name not in stages:
        raise ValueError("capture has no completed parent and isolated child inspection")
    positions = {name: index for index, name in enumerate(stages)}
    if not positions[parent_name] < positions[child_name] < positions[item["name"]]:
        raise ValueError("capture did not follow its parent and isolated-child inspections")
    if (
        positions[text(proof.get("binary_identity_stage"))] >= positions[item["name"]]
        or positions[text(proof.get("inspector_identity_stage"))] >= positions[parent_name]
    ):
        raise ValueError("compiled identity admission did not precede device observations")
    if manifest.get("requested_architecture") != item.get("expected_target"):
        raise ValueError("capture target labels contradict the session's requested architecture")
    parent_item, parent_result = stages[parent_name]
    child_item, child_result = stages[child_name]
    if (
        parent_item.get("parent_inspect") is not True
        or parent_item.get("inspect_device") != selected
        or parent_item.get("selected_device") != selected
        or parent_item.get("device") is not None
        or child_item.get("inspect_device") != 0
        or child_item.get("device") != selected
        or child_item.get("selected_device") != selected
        or child_item.get("match_parent") != parent_name
    ):
        raise ValueError(
            "parent/isolated inspection does not establish the selected ordinal mapping"
        )
    _environment(parent_result, selected=None, rank=rank, world=world)
    _environment(child_result, selected=selected, rank=rank, world=world)
    parent_info = _stdout(parent_item, parent_result, results_path.parent, store)
    child_info = _stdout(child_item, child_result, results_path.parent, store)
    inspector = _query_receipt(
        text(proof.get("inspector_identity_stage")),
        stages,
        results_path.parent,
        cwd,
        store,
        capture["backend_requested"],
        identity["build_manifest_sha256"],
    )
    for inspection_item, inspection_result, info in (
        (parent_item, parent_result, parent_info),
        (child_item, child_result, child_info),
    ):
        inspection_args = inspection_item.get("argv")
        if (
            not isinstance(inspection_args, list)
            or not inspection_args
            or (cwd / text(inspection_args[0])).resolve(strict=True)
            != Path(inspector["executable"])
            or inspection_args[1:]
            != [
                "--backend",
                item.get("expected_backend"),
                "--device",
                str(inspection_item.get("inspect_device")),
                "--json",
            ]
            or object_value(inspection_result.get("artifacts")).get(inspector["executable"])
            != inspector["executable_sha256"]
        ):
            raise ValueError("runtime inspection did not use the queried inspector artifact")
        observed = inspection_observation(info)
        if (
            observed.identity.runtime_ordinal != inspection_item.get("inspect_device")
            or info.get("backend") != item.get("expected_backend")
            or observed.architecture.split(":")[0] != item.get("expected_target")
        ):
            raise ValueError(
                "runtime inspection contradicts its requested backend/ordinal/architecture"
            )
    if capture.get("backend_requested") != item.get("expected_backend") or identity[
        "target_id"
    ] != item.get("expected_target"):
        raise ValueError("capture backend/target contradicts the requested session case")
    relation = capture_relation(capture, parent_info, child_info)
    if relation["state"] != "verified":
        raise ValueError(
            "runtime parent/child/capture observations do not identify the same execution device"
        )
    case_id = text(item.get("case_id"))
    work = dict(
        case_id=case_id,
        mathematical_contract=numerical_contract(capture),
        backend=capture["backend_selected"],
        selected_kernel=capture["selected_kernel"],
        build_manifest_sha256=identity["build_manifest_sha256"],
        executable_sha256=identity["executable_sha256"],
    )
    cases = sorted(
        {
            text(command.get("case_id"))
            for command in manifest["commands"]
            if isinstance(command, dict) and "capture" in command
        }
    )
    return dict(
        rank=rank,
        world_size=world,
        selected_parent_ordinal=selected,
        case_id=case_id,
        exact_case_sha256=canonical_digest(work),
        exact_work=work,
        observation=relation,
        requested_devices=devices,
        expected_cases=cases,
        execution_id=capture["execution_id"],
        capture_sha256=binding["capture_sha256"],
        session_id=manifest["identity_sha256"],
        run_id=proof["run_id"],
        source_session_manifest=str(manifest_path),
        scope="this_capture_on_this_node_runtime_visibility_mapping_only",
        promotion_eligible=False,
    )


def make_binding(
    capture_path: Path,
    manifest_path: Path,
    results_path: Path,
    stage_name: str,
    *,
    store: EvidenceStore | None = None,
) -> dict:
    from target_validation_report import capture_target

    owned = store is None
    store = store if store is not None else EvidenceStore()
    manifest = object_value(load_evidence_json(manifest_path))
    results = object_value(load_evidence_json(results_path))
    cwd = Path(text(manifest.get("command_working_directory"))).resolve(strict=True)
    commands = manifest.get("commands")
    if not isinstance(commands, list):
        raise ValueError("session command inventory is missing")
    selected = [
        item for item in commands if isinstance(item, dict) and item.get("name") == stage_name
    ]
    if len(selected) != 1 or "capture" not in selected[0]:
        raise ValueError("one actual capture stage required")
    item = selected[0]

    def query_for(executable: str) -> str:
        candidates = [
            command
            for command in commands
            if isinstance(command, dict)
            and command.get("compiled_identity") is True
            and isinstance(command.get("argv"), list)
            and command["argv"]
            and (cwd / text(command["argv"][0])).resolve(strict=True)
            == (cwd / executable).resolve(strict=True)
        ]
        if len(candidates) != 1:
            raise ValueError("one compiled identity query is required for each observed executable")
        return text(candidates[0].get("name"))

    parent_names = [
        command
        for command in commands
        if isinstance(command, dict) and command.get("name") == item.get("observation_parent_stage")
    ]
    if len(parent_names) != 1:
        raise ValueError("capture's parent inspection is absent")
    capture = load_capture(capture_path)
    validate_capture(capture, capture_path)
    binding = candidate_binding(capture_path, capture, capture_target(capture_path))

    def reference(path: Path) -> dict:
        return dict(path=str(path.resolve(strict=True)), sha256=file_digest(path))

    proof = dict(
        session_manifest=reference(manifest_path),
        session_results=reference(results_path),
        session_id=text(manifest.get("identity_sha256")),
        run_id=text(results.get("run_id")),
        working_directory=str(cwd),
        stage=stage_name,
        binary_identity_stage=query_for(text(item["argv"][0])),
        inspector_identity_stage=query_for(text(parent_names[0]["argv"][0])),
        parent_stage=item.get("observation_parent_stage"),
        child_stage=item.get("observation_child_stage"),
    )
    facts = _observe(capture_path, capture, binding, proof, manifest_path.parent, store)
    result = dict(schema=SCHEMA, candidate=binding, proof=proof, facts=facts)
    result["identity_sha256"] = canonical_digest(result)
    if owned:
        store.finish()
    return result


def validate_binding(
    value: dict, capture_path: Path, *, store: EvidenceStore | None = None
) -> dict:
    from target_validation_report import capture_target

    owned = store is None
    store = store if store is not None else EvidenceStore()
    if value.get("schema") != SCHEMA or value.get("identity_sha256") != canonical_digest(
        {key: item for key, item in value.items() if key != "identity_sha256"}
    ):
        raise ValueError("versioned sealed multi-GPU observation binding required")
    capture = load_capture(capture_path)
    validate_capture(capture, capture_path)
    binding = candidate_binding(capture_path, capture, capture_target(capture_path))
    if not same_evidence_value(value.get("candidate"), binding):
        raise ValueError("multi-GPU binding belongs to another capture/artifact/execution")
    proof = object_value(value.get("proof"))
    manifest_path = Path(text(object_value(proof.get("session_manifest")).get("path"))).resolve(
        strict=True
    )
    facts = _observe(capture_path, capture, binding, proof, manifest_path.parent, store)
    if not same_evidence_value(value.get("facts"), facts):
        raise ValueError("multi-GPU observation summary differs from its raw session evidence")
    if owned:
        store.finish()
    return facts


def emit_session_bindings(manifest_path: Path, results_path: Path, output: Path) -> dict:
    import json

    manifest = object_value(load_evidence_json(manifest_path))
    commands = manifest.get("commands")
    if not isinstance(commands, list):
        raise ValueError("session has no command inventory")
    cwd = Path(text(manifest.get("command_working_directory"))).resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    store = EvidenceStore()
    receipts, errors = [], []
    planned = [item for item in commands if isinstance(item, dict) and "capture" in item]
    for index, item in enumerate(planned):
        try:
            capture_path = (cwd / text(item.get("capture"))).resolve(strict=True)
            receipt = make_binding(
                capture_path,
                manifest_path,
                results_path,
                text(item.get("name")),
                store=store,
            )
            target = output / ("capture-" + str(index) + ".binding.json")
            target.write_text(
                json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            receipts.append(
                dict(
                    path=str(target.resolve()),
                    sha256=file_digest(target),
                    capture=str(capture_path),
                )
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(dict(stage=item.get("name"), reason=str(error)))
    try:
        store.finish()
    except (OSError, ValueError) as error:
        errors.append(dict(reason=str(error)))
    result = dict(
        schema="rns8_multigpu_observation_emission_v1",
        planned=len(planned),
        emitted=len(receipts),
        complete=bool(planned) and not errors and len(receipts) == len(planned),
        observation_receipts=receipts,
        errors=errors,
        source_session_manifest=str(manifest_path.resolve()),
        source_session_results=str(results_path.resolve()),
        GPU_work_performed=False,
        promotion_eligible=False,
    )
    (output / "observation-index.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-manifest", type=Path, required=True)
    parser.add_argument("--session-results", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="fresh ignored output directory"
    )
    args = parser.parse_args()
    try:
        result = emit_session_bindings(args.session_manifest, args.session_results, args.out_dir)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["complete"] else 2
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
