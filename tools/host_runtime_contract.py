"""Bind native loader snapshots to detector or application process observations."""

from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET

from benchmark_request import file_digest
from json_evidence import parse_evidence_json
from process_progress import capture_succeeded
from compiled_binary_identity import canonical_digest
from execution_inventory import discovery, junit

WITNESS_PREFIX = "RNS8_HOST_RUNTIME_WITNESS "
APPLICATION_SCOPE = "application_process_loader_bookend_not_transient_image_history"

ARCHER_STARTUP = (
    "Archer detected OpenMP application with TSan, supplying OpenMP synchronization semantics"
)


def control_document(capture: dict, *, archer: bool = False) -> dict:
    if not capture_succeeded(capture):
        raise ValueError("native runtime control capture incomplete")
    lines = [line.strip() for line in capture["stdout"].splitlines() if line.strip()]
    startup = lines.count(ARCHER_STARTUP)
    if startup != int(archer):
        raise ValueError("native Archer startup observation differs from selected lane")
    payload = [line for line in lines if line != ARCHER_STARTUP]
    if len(payload) != 1:
        raise ValueError("one native control document required")
    document = parse_evidence_json(payload[0])
    if not isinstance(document, dict):
        raise ValueError("native control document must be an object")
    return document


def loaded_inventory(
    document: dict,
    capture: dict,
    *,
    executable: Path,
    runtimes: list[dict],
    windows_asan: Path | None = None,
    openmp: Path | None = None,
    archer: Path | None = None,
    application_snapshot: bool = False,
    file_references: dict | None = None,
) -> dict:
    pid = capture.get("process_id")
    if (
        not capture_succeeded(capture)
        or type(pid) is not int
        or pid <= 0
        or document.get("schema") != "rns8_host_loaded_modules_v1"
        or type(document.get("process_id")) is not int
        or document["process_id"] != pid
        or document.get("scope")
        != (
            APPLICATION_SCOPE
            if application_snapshot
            else "control_process_loader_snapshot_not_all_application_processes"
        )
    ):
        raise ValueError("native loader snapshot or process identity invalid")
    modules = document.get("modules")
    virtual = document.get("virtual_images")
    owners = document.get("symbol_owners")
    workers = document.get("openmp_workers")
    if (
        not isinstance(modules, list)
        or not 1 <= len(modules) <= 256
        or not isinstance(virtual, list)
        or len(virtual) > 2
        or any(
            not isinstance(name, str) or name not in {"linux-vdso.so.1", "linux-gate.so.1"}
            for name in virtual
        )
        or len(set(virtual)) != len(virtual)
        or not isinstance(owners, dict)
        or set(owners)
        != (set() if windows_asan else {"sanitizer", *(["openmp"] if openmp else [])})
        or type(workers) is not int
        or workers != (2 if openmp and not application_snapshot else 0)
        or bool(openmp) != bool(archer)
    ):
        raise ValueError("native loader inventory/feature shape invalid")
    files, paths, total = [], set(), 0
    references = {} if file_references is None else file_references
    for raw in modules:
        if not isinstance(raw, str) or not raw or "\0" in raw or len(raw) > 32768:
            raise ValueError("native module path invalid")
        path = Path(raw)
        if not path.is_absolute():
            raise ValueError("native module path must be absolute")
        path = path.resolve(strict=True)
        if path in paths or not path.is_file():
            raise ValueError("native module repeated or not a regular file")
        size = path.stat().st_size
        total += size
        if not 0 < size <= 512 * 1024**2 or total > 2 * 1024**3:
            raise ValueError("native module file budget exceeded")
        paths.add(path)
        # A lane's bookend reader hashes a module once, then rechecks every file
        # at final publication. This is file custody, not loaded-memory hashing.
        if path not in references:
            references[path] = dict(path=str(path), sha256=file_digest(path), bytes=size)
        if references[path]["bytes"] != size:
            raise ValueError("native module changed during loader receipt collection")
        files.append(references[path])
    executable = executable.resolve(strict=True)
    if executable not in paths:
        raise ValueError("actual captured executable absent from loaded modules")
    if windows_asan:
        runtime = windows_asan.resolve(strict=True)
        if runtime not in paths or not any(
            references[runtime]["sha256"] == row["sha256"] for row in runtimes
        ):
            raise ValueError("actual Windows ASan DLL not the selected compiler runtime")
    else:
        bindings = {}
        for name, raw in owners.items():
            if not isinstance(raw, str) or not Path(raw).is_absolute():
                raise ValueError("native symbol owner path invalid")
            bindings[name] = Path(raw).resolve(strict=True)
            if bindings[name] not in paths:
                raise ValueError("native symbol owner absent from loaded modules")
        sanitizer = bindings["sanitizer"]
        if sanitizer != executable and not any(
            str(sanitizer) == row["path"] and references[sanitizer]["sha256"] == row["sha256"]
            for row in runtimes
        ):
            raise ValueError("native sanitizer symbol not bound to selected runtime or executable")
        if openmp:
            if bindings["openmp"] != openmp.resolve(strict=True) or not re.fullmatch(
                r"libomp\.so(?:\.\d+)*", openmp.name
            ):
                raise ValueError("actual OpenMP symbol not supplied by configured LLVM libomp")
            if not application_snapshot and archer.resolve(strict=True) not in paths:
                raise ValueError("selected Archer library absent from loaded modules")
    return dict(
        schema="rns8_host_loaded_runtime_binding_v1",
        process_id=pid,
        files=files,
        virtual_images=virtual,
        symbol_owners=owners,
        openmp_workers=workers,
        archer_annotation_adapter=bool(archer and archer.resolve(strict=True) in paths),
        external_runtime_accesses_instrumented="not_established_by_loader_snapshot",
        scope=document["scope"],
        production_latency=False,
    )


def application_witnesses(
    source: str,
    selected: dict,
    capture: dict,
    *,
    token: str,
    executables: list[Path | str],
    lane: str,
    runtimes: list[dict],
    windows_asan: Path | None = None,
    openmp: Path | None = None,
    archer: Path | None = None,
) -> dict:
    """Join native bookends to the exact CTest case; never substitute launcher PID."""
    if (
        type(token) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", token)
        or lane
        not in {
            "address-undefined",
            "thread",
            "windows-address",
        }
    ):
        raise ValueError("explicit native witness token and sanitizer lane required")
    parent = capture.get("process_id")
    if type(parent) is not int or parent <= 0 or not capture_succeeded(capture):
        raise ValueError("successful witnessed CTest parent process required")
    tests, outcomes = discovery(selected), junit(source)
    if tests.keys() != outcomes.keys():
        raise ValueError("witness JUnit and selected command inventory differ")
    expected = {Path(path).resolve(strict=True) for path in executables}
    if not expected:
        raise ValueError("explicit nonempty witnessed native executable inventory required")
    root = ET.fromstring(source)  # junit() already bounds and rejects DTD/entity input.
    elements = {element.get("name"): element for element in root.findall("testcase")}
    rows, excluded, files, seen, reached = [], [], {}, set(), set()
    for name, test in tests.items():
        executable = Path(test.command[0]).resolve(strict=True)
        element = elements[name]
        outputs = element.findall("system-out")
        lines = (
            (outputs[0].text or "").splitlines()
            if len(outputs) == 1 and not list(outputs[0])
            else []
        )
        raw = [line[len(WITNESS_PREFIX) :] for line in lines if line.startswith(WITNESS_PREFIX)]
        if executable not in expected:
            if raw:
                raise ValueError("unselected native process emitted runtime witness")
            excluded.append(dict(test=name, reason="not_in_explicit_witness_executable_inventory"))
            continue
        reached.add(executable)
        passing = outcomes[name]["state"] == "run"
        if not passing and not raw:
            # Approved skips retain their disposition, but never gain loader credit.
            excluded.append(dict(test=name, reason="not_executed_passing_case"))
            continue
        if (
            len(outputs) != 1
            or len(raw) != 2
            or any(len(line.encode("utf-8")) > 256 * 1024 for line in raw)
        ):
            raise ValueError("exact native enter/leave runtime witness bookends required")
        bookends = [parse_evidence_json(line) for line in raw]
        bindings = []
        for phase, value in zip(("enter", "leave"), bookends):
            fields = {
                "schema",
                "token",
                "phase",
                "process_id",
                "parent_process_id",
                "executable",
                "arguments",
                "elapsed_ns",
                "unwinding",
                "address_feature",
                "thread_feature",
                "loaded_modules",
            }
            if (
                type(value) is not dict
                or set(value) != fields
                or value["schema"] != "rns8_host_application_witness_v1"
                or value["token"] != token
                or value["phase"] != phase
                or type(value["process_id"]) is not int
                or value["process_id"] <= 0
                or value["process_id"] == parent
                or type(value["parent_process_id"]) is not int
                or value["parent_process_id"] != parent
                or type(value["executable"]) is not str
                or not Path(value["executable"]).is_absolute()
                or Path(value["executable"]).resolve(strict=True) != executable
                or type(value["arguments"]) is not list
                or value["arguments"] != list(test.command[1:])
                or any(type(arg) is not str for arg in value["arguments"])
                or type(value["elapsed_ns"]) is not int
                or not 0 <= value["elapsed_ns"] < 2**64
                or value["unwinding"] is not False
                or value["address_feature"] is not (lane != "thread")
                or value["thread_feature"] is not (lane == "thread")
                or type(value["loaded_modules"]) is not dict
            ):
                raise ValueError(
                    "native runtime witness command, feature or process custody invalid"
                )
            binding = loaded_inventory(
                value["loaded_modules"],
                {**capture, "process_id": value["process_id"]},
                executable=executable,
                runtimes=runtimes,
                windows_asan=windows_asan,
                openmp=openmp,
                archer=archer,
                application_snapshot=True,
                file_references=files,
            )
            bindings.append({key: value for key, value in binding.items() if key != "files"})
            bindings[-1]["modules"] = [row["path"] for row in binding["files"]]
        first, last = bookends
        pid = first["process_id"]
        # Sequential cases may legally reuse an OS PID. Command plus nonce and
        # per-case paired output establish custody; PID alone is not a run key.
        identity = (name, pid)
        if (
            identity in seen
            or last["process_id"] != pid
            or last["elapsed_ns"] < first["elapsed_ns"]
        ):
            raise ValueError("native runtime bookend ordering or identity invalid")
        seen.add(identity)
        if last["elapsed_ns"] > (outcomes[name]["time_seconds"] + 0.5) * 1e9:
            raise ValueError("native runtime witness exceeds observed case duration")
        row = dict(
            test=name,
            process_id=pid,
            parent_process_id=parent,
            command_sha256=canonical_digest(list(test.command)),
            enter=bindings[0],
            leave=bindings[1],
            elapsed_ns=last["elapsed_ns"],
            witness_sha256=canonical_digest(bookends),
        )
        if passing:
            rows.append(row)
        else:
            excluded.append(
                dict(
                    test=name,
                    reason="not_executed_passing_case",
                    observed_witness_sha256=row["witness_sha256"],
                )
            )
    if reached != expected or not rows:
        raise ValueError("native witness executable inventory missing or empty")
    return dict(
        schema="rns8_host_application_runtime_bindings_v1",
        scope=APPLICATION_SCOPE,
        ctest_process_id=parent,
        token=token,
        cases=rows,
        excluded_cases=excluded,
        files=sorted(files.values(), key=lambda row: row["path"]),
        file_identity_scope="post_execution_files_rechecked_at_publication_not_in_memory_image_attestation",
        production_latency=False,
    )
