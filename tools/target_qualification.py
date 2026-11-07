"""Candidate-bound qualification over existing immutable session artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from benchmark_request import file_digest, request_environment_identity
from benchmark_schema import load_capture, validate_capture
from build_manifest import manifest_provenance_complete
from compiled_binary_identity import canonical_digest, validate_receipt
from json_evidence import (
    JsonValue,
    load_evidence_json,
    parse_evidence_json,
    same_evidence_value,
)
from measurement_contract import host_instrumentation
from review_contract import release_capture_satisfied
from timing_eligibility import capture_identity
from reference_verdict import numerical_contract
from process_progress import capture_succeeded
from execution_observation import validate_execution_request

SCHEMA = "rns8_target_candidate_qualification_v1"
VERDICT_SCHEMA = "rns8_target_candidate_verdict_v1"
PHASES = ("build", "ctest", "smoke", "release_capture", "profiler")


def object_value(value: JsonValue) -> dict:
    if not isinstance(value, dict):
        raise ValueError("qualification evidence must be an object")
    return value


def text(value: JsonValue) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("qualification identity text is missing")
    return value


def digest_text(value: JsonValue) -> str:
    result = text(value)
    if re.fullmatch("[0-9a-f]{64}", result) is None:
        raise ValueError("qualification requires a complete SHA256 digest")
    return result


def candidate_identity(capture: dict, row: dict) -> dict:
    if row.get("supported_target") is not True or row.get("host_os") not in (
        "windows",
        "linux",
    ):
        raise ValueError("candidate target/platform has no supported qualification policy")
    backend = text(capture.get("backend_selected"))
    manifest = capture.get("build_manifest")
    if not manifest_provenance_complete(manifest, backend):
        raise ValueError("candidate compiled manifest is incomplete")
    request = object_value(capture.get("request_identity"))
    validate_execution_request(capture, request)
    environment, error = request_environment_identity(request)
    if error:
        raise ValueError(error)
    artifact = object_value(capture.get("artifact_capture"))
    if artifact.get("rns8_linkage") != "static" or artifact.get("unchanged_during_run") is not True:
        raise ValueError("candidate needs stable static executable attribution")
    executable = digest_text(artifact.get("executable_sha256"))
    if request.get("executable_sha256") != executable:
        raise ValueError("candidate request and executable identities differ")
    device = object_value(capture.get("device"))
    observed_arch = text(device.get("gcn_arch")).split(":")[0]
    if row.get("target_id") != observed_arch:
        raise ValueError("candidate expected and runtime-observed architectures differ")
    if row.get("target_cache_key") != row.get("runtime_observed_cache_key"):
        raise ValueError("candidate cache labels contradict its runtime-observed device")
    runtime = device.get("hip_runtime_version")
    if type(runtime) is not int or runtime <= 0:
        raise ValueError("candidate runtime identity is missing")
    driver = device.get("hip_driver_version")
    if type(driver) is not int or driver <= 0:
        raise ValueError("candidate driver identity is missing")
    return dict(
        build_manifest_sha256=canonical_digest(manifest),
        executable_sha256=executable,
        backend_selected=backend,
        selected_kernel=text(capture.get("selected_kernel")),
        target_id=observed_arch,
        host_os=text(row.get("host_os")),
        target_cache_key=text(row.get("target_cache_key")),
        runtime_version=runtime,
        environment=environment,
        driver_version=driver,
        compiler=copy.deepcopy(capture.get("compiler")),
        hip_toolchain=copy.deepcopy(capture.get("hip_toolchain")),
        device=copy.deepcopy(device),
        mathematical_contract=numerical_contract(capture),
        execution_mode=capture.get("benchmark_execution_mode"),
        target_instance_id=text(row.get("target_instance_id")),
    )


def candidate_binding(path: Path, capture: dict, row: dict) -> dict:
    raw = path.read_bytes()
    snapshot = object_value(parse_evidence_json(raw.decode("utf-8")))
    validate_capture(snapshot, path)
    if capture_identity(snapshot) != capture_identity(capture):
        raise ValueError("candidate file changed or differs from its parsed observation")
    identity = candidate_identity(snapshot, row)
    return dict(
        capture=str(path.resolve(strict=True)),
        capture_sha256=hashlib.sha256(raw).hexdigest(),
        capture_content_sha256=capture_identity(capture),
        execution_id=text(capture.get("execution_id")),
        identity=identity,
        identity_sha256=canonical_digest(identity),
    )


@dataclass
class EvidenceStore:
    """Hash each raw file at entry and exit, rather than once per phase."""

    hashes: dict[Path, str] = field(default_factory=dict)
    documents: dict[Path, JsonValue] = field(default_factory=dict)
    binary_receipts: dict[str, dict] = field(default_factory=dict)

    def verify(self, path: Path, expected: str) -> Path:
        path = path.resolve(strict=True)
        digest_text(expected)
        previous = self.hashes.get(path)
        if previous is not None and previous != expected:
            raise ValueError("conflicting content identities for " + str(path))
        if previous is None:
            if file_digest(path) != expected:
                raise ValueError("qualification artifact changed: " + str(path))
            self.hashes[path] = expected
        return path

    def reference(self, reference: JsonValue, parent: Path) -> Path:
        value = object_value(reference)
        return self.verify(parent / text(value.get("path")), digest_text(value.get("sha256")))

    def json(self, reference: JsonValue, parent: Path) -> tuple[Path, dict]:
        value = object_value(reference)
        path = (parent / text(value.get("path"))).resolve(strict=True)
        expected = digest_text(value.get("sha256"))
        if path in self.hashes and self.hashes[path] != expected:
            raise ValueError("conflicting qualification JSON content identities")
        if path not in self.documents:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("qualification JSON changed while it was being read")
            self.hashes[path] = expected
            self.documents[path] = parse_evidence_json(raw.decode("utf-8"))
        return path, object_value(self.documents[path])

    def binary(self, receipt: JsonValue, backend: str) -> dict:
        key = backend + ":" + canonical_digest(receipt)
        if key not in self.binary_receipts:
            self.binary_receipts[key] = validate_receipt(
                receipt, backend=backend, verify_file=self.verify
            )
        return self.binary_receipts[key]

    def finish(self) -> None:
        for path, expected in self.hashes.items():
            if file_digest(path) != expected:
                raise ValueError("qualification artifact changed during review: " + str(path))


def session_stage(
    proof: dict, parent: Path, store: EvidenceStore
) -> tuple[dict, dict, dict, dict, dict]:
    _, manifest = store.json(proof.get("session_manifest"), parent)
    _, results = store.json(proof.get("session_results"), parent)
    identity = canonical_digest(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"generated_utc", "status", "identity_sha256"}
        }
    )
    if (
        manifest.get("identity_sha256") != identity
        or results.get("manifest_identity_sha256") != identity
    ):
        raise ValueError("phase session manifest/result identity mismatch")
    if proof.get("session_id") != identity or proof.get("run_id") != results.get("run_id"):
        raise ValueError("phase does not belong to the declared executed session/run")
    working_directory = Path(text(manifest.get("command_working_directory"))).resolve(strict=True)
    if working_directory != Path(text(proof.get("working_directory"))).resolve(strict=True):
        raise ValueError("phase working directory differs from the executed session")
    manifest_version = manifest.get("schema_version")
    profiler_session = manifest.get("schema") in {
        "rns8_amd_profiler_session_v1",
        "rns8_host_perf_session_v1",
    }
    if (
        type(manifest_version) is not int
        or manifest_version != (4 if profiler_session else 2)
        or results.get("schema") != "rns8_gpu_session_results_v2"
    ):
        raise ValueError("unsupported raw qualification session schema")
    commands, rows = manifest.get("commands"), results.get("results")
    if (
        not isinstance(commands, list)
        or not isinstance(rows, list)
        or not rows
        or len(rows) > len(commands)
    ):
        raise ValueError("phase session is not a nonempty ordered checkpoint")
    indexed = {}
    for item, row in zip(commands, rows):
        item, row = object_value(item), object_value(row)
        name = text(item.get("name"))
        if row.get("name") != name or name in indexed:
            raise ValueError("phase session has reordered or duplicate stages")
        artifacts = object_value(row.get("artifacts"))
        if not artifacts:
            raise ValueError("phase session stage has no immutable artifacts")
        for filename, digest in artifacts.items():
            store.verify(Path(filename), digest_text(digest))
        indexed[name] = (item, row)
    stage = indexed.get(text(proof.get("stage")))
    binary = indexed.get(text(proof.get("binary_identity_stage")))
    if stage is None or binary is None or binary[0].get("compiled_identity") is not True:
        raise ValueError("phase or compiled binary query stage is absent")
    if binary[0].get("argv", [])[-1:] != ["--build-identity"]:
        raise ValueError("phase binary association was not a compiled identity query")
    return stage[0], stage[1], binary[0], binary[1], indexed


def phase_observation(
    phase: str,
    proof: dict,
    parent: Path,
    candidate: dict,
    capture: dict,
    store: EvidenceStore,
) -> dict:
    item, result, identity_item, identity_result, stages = session_stage(proof, parent, store)
    identity = object_value(candidate["identity"])
    binary = store.binary(
        identity_result.get("compiled_binary_identity"), identity["backend_selected"]
    )
    if binary["build_manifest_sha256"] != identity["build_manifest_sha256"]:
        raise ValueError("phase binary was compiled for a different candidate build/configuration")
    arguments = item.get("argv")
    identity_arguments = identity_item.get("argv")
    if (
        not isinstance(arguments, list)
        or not arguments
        or any(not isinstance(arg, str) for arg in arguments)
        or not isinstance(identity_arguments, list)
        or not identity_arguments
    ):
        raise ValueError("phase command arguments are incomplete")
    binary_path = Path(binary["executable"]).resolve(strict=True)
    if Path(identity_arguments[0]).resolve(strict=True) != binary_path:
        raise ValueError("phase binary query does not name the bound executable")
    working_directory = Path(str(proof.get("working_directory"))).resolve(strict=True)

    def operand(flag: str) -> Path:
        if arguments.count(flag) != 1 or arguments.index(flag) + 1 >= len(arguments):
            raise ValueError("phase command omits or duplicates " + flag)
        return (working_directory / arguments[arguments.index(flag) + 1]).resolve(strict=True)

    if phase in ("build", "ctest"):
        build = (working_directory / text(identity_item.get("build_dir"))).resolve(strict=True)
        if not binary_path.is_relative_to(build):
            raise ValueError("tested binary does not belong to the selected build directory")
        if operand("--build" if phase == "build" else "--test-dir") != build:
            raise ValueError("phase targets a different build directory")
        if phase == "ctest" and binary_path.stem != "rns8_tests":
            raise ValueError("CTest phase lacks the compiled Catch runner identity")
        if phase == "build" and "--target" in arguments:
            raise ValueError("qualification build must build the complete configured project")
        if phase == "ctest":
            if "--no-tests=error" not in arguments:
                raise ValueError("CTest phase may silently accept an empty test selection")
            check = stages.get(text(proof.get("correctness_stage")))
            if check is None:
                raise ValueError("CTest phase lacks its required unskipped GPU correctness stage")
            check_item, check_result = check
            check_args = check_item.get("argv")
            required = "qualification pack covers dispatch thresholds and plane tails"
            if (
                not isinstance(check_args, list)
                or not check_args
                or (working_directory / check_args[0]).resolve(strict=True) != binary_path
                or check_item.get("require_test") != required
                or "[qualification]" not in check_args
            ):
                raise ValueError("correctness stage does not run the bound GPU qualification suite")
            for stage_item, stage_result, field_name, required_case in (
                (item, result, "ctest_junit", None),
                (check_item, check_result, "junit", required),
            ):
                xml_path = (working_directory / text(stage_item.get(field_name))).resolve(
                    strict=True
                )
                store.verify(
                    xml_path,
                    digest_text(object_value(stage_result.get("artifacts")).get(str(xml_path))),
                )
                xml = xml_path.read_text(encoding="utf-8")
                if "<!DOCTYPE" in xml or "<!ENTITY" in xml:
                    raise ValueError("test evidence must not contain external XML definitions")
                try:
                    root = ET.fromstring(xml)
                except ET.ParseError as error:
                    raise ValueError("test phase JUnit XML is malformed") from error
                cases = list(root.iter("testcase"))
                if (
                    not cases
                    or any(
                        case.find("failure") is not None or case.find("error") is not None
                        for case in cases
                    )
                    or not any(case.find("skipped") is None for case in cases)
                ):
                    raise ValueError("test phase has no executed passing test evidence")
                if required_case is not None and not any(
                    case.get("name") == required_case and case.find("skipped") is None
                    for case in cases
                ):
                    raise ValueError("required GPU qualification test is missing or skipped")
            if not capture_succeeded(check_result) or check_result.get("validation_errors") != []:
                raise ValueError("required GPU correctness stage failed")
    elif phase == "smoke":
        if (
            binary_path.stem != "rns8-verify"
            or (working_directory / arguments[0]).resolve(strict=True) != binary_path
            or "--hip-smoke" not in arguments
        ):
            raise ValueError("smoke phase does not execute the bound GPU verifier")
    elif phase == "release_capture":
        if binary["executable_sha256"] != identity["executable_sha256"]:
            raise ValueError("release phase executes a different benchmark artifact")
        path = (working_directory / text(item.get("capture"))).resolve(strict=True)
        if file_digest(path) != candidate["capture_sha256"]:
            raise ValueError("release phase does not contain this exact candidate capture")
        if (
            item.get("expected_backend") != identity["backend_selected"]
            or item.get("expected_target") != identity["target_id"]
        ):
            raise ValueError("release phase backend/target does not match the candidate")
        if not same_evidence_value(
            arguments[1:], object_value(capture["request_identity"])["arguments"]
        ):
            raise ValueError("release phase request differs from the capture")
        if (
            not release_capture_satisfied(capture)
            or host_instrumentation(capture) != "hooks_disabled"
        ):
            raise ValueError(
                "smoke/instrumented sampling is not a qualified production release capture"
            )
    elif phase == "profiler":
        # The capability-gated profiler adapter owns native trace parsing and
        # selected-dispatch attribution. Availability/static ISA is insufficient.
        from amd_profiler_evidence import validate_dispatch_receipt

        if identity["host_os"] != "linux":
            raise ValueError(
                "native Windows profiler dispatch adapter is not qualified by Linux ROCprofiler data"
            )
        profiler = validate_dispatch_receipt(
            proof.get("dispatch_receipt"), capture=capture, store=store, parent=parent
        )
        if not same_evidence_value(profiler["acquisition_command"], arguments):
            raise ValueError("profiler evidence was acquired by a different command")
        artifacts = object_value(result.get("artifacts"))
        fresh = object_value(result.get("fresh_evidence_outputs"))
        for name in ("native_trace", "profiled_capture"):
            if artifacts.get(profiler[name]) != profiler[name + "_sha256"]:
                raise ValueError(
                    "profiler session stage does not retain its native workload/trace outputs"
                )
            if fresh.get(profiler[name]) != profiler[name + "_sha256"]:
                raise ValueError(
                    "profiler outputs were not established as fresh for this acquisition"
                )
    else:
        raise ValueError("unknown qualification phase")
    for observed in (identity_result, result):
        if (
            type(observed.get("returncode")) is not int
            or type(observed.get("timed_out")) is not bool
        ):
            raise ValueError("phase process status is malformed")
        if not isinstance(observed.get("validation_errors"), list):
            raise ValueError("phase validation result is absent")
    passed = all(
        capture_succeeded(observed) and not observed["validation_errors"]
        for observed in (identity_result, result)
    )
    return dict(
        status="pass" if passed else "fail",
        stage=item["name"],
        binary_identity_stage=identity_item["name"],
        binary_identity=binary,
        evidence_sha256=canonical_digest(proof),
    )


def load_qualification(path: Path, expected: dict, capture: dict, store: EvidenceStore) -> dict:
    path = path.resolve(strict=True)
    raw_hash = file_digest(path)
    store.verify(path, raw_hash)
    value = object_value(load_evidence_json(path))
    if value.get("schema") != SCHEMA:
        raise ValueError("candidate-specific versioned qualification receipt required")
    payload = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != canonical_digest(payload):
        raise ValueError("qualification receipt seal mismatch")
    actual = object_value(value.get("candidate"))
    # Absolute capture paths are an observation location, not build identity.
    for key in (
        "capture_sha256",
        "capture_content_sha256",
        "execution_id",
        "identity",
        "identity_sha256",
    ):
        if not same_evidence_value(actual.get(key), expected.get(key)):
            raise ValueError("qualification receipt belongs to a different candidate: " + key)
    text(value.get("session_id"))
    text(value.get("run_id"))
    proofs = object_value(value.get("phases"))
    if set(proofs) - set(PHASES):
        raise ValueError("unknown qualification phase names")
    observations, blockers = {}, []
    for phase in PHASES:
        if phase not in proofs:
            observations[phase] = dict(status="missing")
            blockers.append(phase + "_missing_candidate_evidence")
            continue
        proof = object_value(proofs[phase])
        if proof.get("session_id") != value["session_id"] or proof.get("run_id") != value["run_id"]:
            raise ValueError("phase belongs to a different qualification session/run")
        if proof.get("candidate_identity_sha256") != expected["identity_sha256"]:
            raise ValueError("phase candidate association is missing or inconsistent")
        try:
            observations[phase] = phase_observation(
                phase, proof, path.parent, expected, capture, store
            )
        except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
            observations[phase] = dict(status="unqualified", reason=str(error))
        if observations[phase]["status"] != "pass":
            blockers.append(phase + "_not_qualified_for_candidate")
    supersedes = value.get("supersedes")
    if not isinstance(supersedes, list) or any(not isinstance(item, str) for item in supersedes):
        raise ValueError("qualification requires an explicit supersession list")
    for item in supersedes:
        digest_text(item)
    return dict(
        path=str(path),
        sha256=raw_hash,
        session_id=value["session_id"],
        run_id=value["run_id"],
        candidate=expected,
        phases=observations,
        eligible=not blockers,
        blockers=blockers,
        supersedes=supersedes,
    )


def candidate_verdict(
    binding: dict,
    capture: dict,
    receipt_paths: list[Path],
    *,
    store: EvidenceStore | None = None,
) -> dict:
    owned_store = store is None
    store = store if store is not None else EvidenceStore()
    store.verify(Path(text(binding.get("capture"))), digest_text(binding.get("capture_sha256")))
    if canonical_digest(binding.get("identity")) != binding.get(
        "identity_sha256"
    ) or capture_identity(capture) != binding.get("capture_content_sha256"):
        raise ValueError("candidate binding no longer matches its observation/identity")
    receipts, exclusions = {}, []
    blockers = []
    for path in receipt_paths:
        try:
            raw = object_value(load_evidence_json(path))
            associated = object_value(raw.get("candidate"))
            same_capture = associated.get("capture_sha256") == binding["capture_sha256"]
            same_candidate = associated.get("identity_sha256") == binding["identity_sha256"]
            if not same_capture and not same_candidate:
                exclusions.append(dict(path=str(path), reason="different_candidate_capture"))
                continue
            source_binding, source_capture = binding, capture
            if not same_capture:
                # A new process/capture must not hide a current failure for the
                # same artifact and work contract. Verify its original files;
                # an older passing capture alone cannot qualify this one.
                from target_validation_report import capture_target

                source_path = path.parent / text(associated.get("capture"))
                store.verify(source_path, digest_text(associated.get("capture_sha256")))
                source_capture = load_capture(source_path)
                validate_capture(source_capture, source_path)
                source_binding = candidate_binding(
                    source_path, source_capture, capture_target(source_path)
                )
                if source_binding["identity_sha256"] != binding["identity_sha256"]:
                    raise ValueError("another observation has inconsistent candidate identity")
            receipt = load_qualification(path, source_binding, source_capture, store)
            receipt["qualifies_requested_capture"] = same_capture
            receipts[receipt["sha256"]] = receipt
        except (OSError, ValueError, TypeError, KeyError) as error:
            exclusions.append(dict(path=str(path), reason=str(error)))
            blockers.append("malformed_or_stale_current_candidate_qualification")
    superseded: set[str] = set()
    for key, receipt in receipts.items():
        for previous in receipt["supersedes"]:
            if previous not in receipts or previous == key:
                blockers.append("unknown_or_self_superseded_qualification")
            else:
                superseded.add(previous)
    visiting, visited = set(), set()

    def acyclic(key: str) -> bool:
        if key in visiting:
            return False
        if key in visited:
            return True
        visiting.add(key)
        for parent in receipts[key]["supersedes"]:
            if parent in receipts and not acyclic(parent):
                return False
        visiting.remove(key)
        visited.add(key)
        return True

    if any(not acyclic(key) for key in receipts):
        blockers.append("qualification_supersession_cycle")
    # A supersession cycle leaves no authoritative current qualification.
    active = [receipt for key, receipt in receipts.items() if key not in superseded]
    if not active:
        blockers.append("no_current_candidate_qualification")
    if any(not receipt["eligible"] for receipt in active):
        blockers.append("current_candidate_phase_evidence_incomplete_or_conflicting")
    for receipt in active:
        blockers.extend(receipt["blockers"])
    if owned_store:
        try:
            store.finish()
        except (OSError, ValueError) as error:
            blockers.append("qualification_artifacts_changed_during_review")
            exclusions.append(dict(reason=str(error)))
    qualified = sorted(
        receipt["sha256"]
        for receipt in active
        if receipt["eligible"] and receipt["qualifies_requested_capture"]
    )
    if not qualified:
        blockers.append("no_current_capture_phase_association")
    return dict(
        schema=VERDICT_SCHEMA,
        candidate=binding,
        eligible=bool(qualified) and not blockers,
        input_receipt_paths=[str(path.resolve()) for path in receipt_paths],
        blockers=sorted(set(blockers)),
        active_receipts=sorted(receipt["sha256"] for receipt in active),
        superseded_receipts=sorted(superseded),
        receipts=list(receipts.values()),
        excluded_receipts=exclusions,
        scope="this_capture_artifact_backend_kernel_contract_session_only",
    )


def validate_bound_verdict(value: JsonValue, path: Path, capture: dict, row: dict) -> dict:
    verdict = object_value(value)
    if verdict.get("schema") != VERDICT_SCHEMA or verdict.get("eligible") is not True:
        raise ValueError("candidate qualification verdict is not ready")
    paths = verdict.get("input_receipt_paths")
    if not isinstance(paths, list) or not paths or any(not isinstance(item, str) for item in paths):
        raise ValueError("candidate verdict has no original qualification receipts")
    binding = candidate_binding(path, capture, row)
    recomputed = candidate_verdict(binding, capture, [Path(item) for item in paths])
    if recomputed.get("eligible") is not True or not same_evidence_value(recomputed, verdict):
        raise ValueError("candidate qualification verdict is stale or does not match raw evidence")
    return recomputed


def emit_qualification(
    *,
    capture_path: Path,
    manifest_path: Path,
    results_path: Path,
    selections: dict[str, tuple[str, str]],
    output: Path,
    correctness_stage: str | None = None,
    dispatch_receipt: Path | None = None,
    supersedes: tuple[Path, ...] = (),
) -> dict:
    """Adapt existing session artifacts; never execute or relabel a native run."""
    from target_validation_report import capture_target

    capture = load_capture(capture_path)
    validate_capture(capture, capture_path)
    binding = candidate_binding(capture_path, capture, capture_target(capture_path))
    manifest = object_value(load_evidence_json(manifest_path))
    results = object_value(load_evidence_json(results_path))
    session_id, run_id = (
        text(manifest.get("identity_sha256")),
        text(results.get("run_id")),
    )

    def reference(path: Path) -> dict:
        return dict(path=str(path.resolve(strict=True)), sha256=file_digest(path))

    phases = {}
    for phase, (stage, binary_stage) in selections.items():
        if phase not in PHASES:
            raise ValueError("unknown selected phase")
        proof = dict(
            session_id=session_id,
            run_id=run_id,
            candidate_identity_sha256=binding["identity_sha256"],
            working_directory=text(manifest.get("command_working_directory")),
            session_manifest=reference(manifest_path),
            session_results=reference(results_path),
            stage=text(stage),
            binary_identity_stage=text(binary_stage),
        )
        if phase == "ctest" and correctness_stage is not None:
            proof["correctness_stage"] = correctness_stage
        if phase == "profiler" and dispatch_receipt is not None:
            proof["dispatch_receipt"] = reference(dispatch_receipt)
        phases[phase] = proof
    receipt = dict(
        schema=SCHEMA,
        candidate=binding,
        session_id=session_id,
        run_id=run_id,
        phases=phases,
        supersedes=[file_digest(path) for path in supersedes],
    )
    receipt["identity_sha256"] = canonical_digest(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return candidate_verdict(binding, capture, [*supersedes, output])


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--session-manifest", type=Path, required=True)
    parser.add_argument("--session-results", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new ignored receipt file; never overwrite",
    )
    parser.add_argument(
        "--phase",
        action="append",
        default=[],
        metavar="PHASE=STAGE,BINARY_IDENTITY_STAGE",
    )
    parser.add_argument("--correctness-stage")
    parser.add_argument("--dispatch-receipt", type=Path)
    parser.add_argument("--supersedes", action="append", default=[], type=Path)
    args = parser.parse_args()
    try:
        selections = {}
        for item in args.phase:
            phase, separator, names = item.partition("=")
            values = names.split(",")
            if (
                not separator
                or phase not in PHASES
                or phase in selections
                or len(values) != 2
                or not all(values)
            ):
                raise ValueError("each unique phase needs STAGE,BINARY_IDENTITY_STAGE")
            selections[phase] = tuple(values)
        verdict = emit_qualification(
            capture_path=args.candidate,
            manifest_path=args.session_manifest,
            results_path=args.session_results,
            selections=selections,
            output=args.output,
            correctness_stage=args.correctness_stage,
            dispatch_receipt=args.dispatch_receipt,
            supersedes=tuple(args.supersedes),
        )
        print(json.dumps(verdict, sort_keys=True, allow_nan=False))
        return 0 if verdict["eligible"] else 2
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
