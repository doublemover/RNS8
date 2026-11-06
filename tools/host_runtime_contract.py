"""Bind a native detector's loader snapshot without inferring process-wide coverage."""

from __future__ import annotations

from pathlib import Path
import re

from benchmark_request import file_digest
from json_evidence import parse_evidence_json
from process_progress import capture_succeeded

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
) -> dict:
    pid = capture.get("process_id")
    if (
        not capture_succeeded(capture)
        or type(pid) is not int
        or pid <= 0
        or document.get("schema") != "rns8_host_loaded_modules_v1"
        or type(document.get("process_id")) is not int
        or document["process_id"] != pid
        or document.get("scope") != "control_process_loader_snapshot_not_all_application_processes"
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
        or workers != (2 if openmp else 0)
        or bool(openmp) != bool(archer)
    ):
        raise ValueError("native loader inventory/feature shape invalid")
    files, paths, total = [], set(), 0
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
        files.append(dict(path=str(path), sha256=file_digest(path), bytes=size))
    executable = executable.resolve(strict=True)
    if executable not in paths:
        raise ValueError("actual control executable absent from loaded modules")
    if windows_asan:
        runtime = windows_asan.resolve(strict=True)
        if runtime not in paths or not any(
            file_digest(runtime) == row["sha256"] for row in runtimes
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
            str(sanitizer) == row["path"] and file_digest(sanitizer) == row["sha256"]
            for row in runtimes
        ):
            raise ValueError("native sanitizer symbol not bound to selected runtime or executable")
        if openmp:
            if bindings["openmp"] != openmp.resolve(strict=True) or not re.fullmatch(
                r"libomp\.so(?:\.\d+)*", openmp.name
            ):
                raise ValueError("actual OpenMP symbol not supplied by configured LLVM libomp")
            if archer.resolve(strict=True) not in paths:
                raise ValueError("selected Archer library absent from loaded modules")
    return dict(
        schema="rns8_host_loaded_runtime_binding_v1",
        process_id=pid,
        files=files,
        virtual_images=virtual,
        symbol_owners=owners,
        openmp_workers=workers,
        archer_annotation_adapter=bool(archer),
        external_runtime_accesses_instrumented="not_established_by_loader_snapshot",
        scope=document["scope"],
        production_latency=False,
    )
