"""Native source-copy and intended Catch assertion evidence; no compiler execution."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import xml.etree.ElementTree as ET

from compiled_binary_identity import canonical_digest
from execution_inventory import junit
from host_instrumentation_session import reference


def tracked_inputs(source: Path, records: str, *, max_bytes: int) -> dict:
    if not records.endswith("\0"):
        raise ValueError("NUL-terminated native source inventory required")
    entries = records[:-1].split("\0")
    if not 1 <= len(entries) <= 50000:
        raise ValueError("bounded nonempty native source inventory required")
    files, excluded, seen, total = [], [], set(), 0
    for entry in entries:
        fields, name = entry.split("\t", 1)
        mode, object_id, stage = fields.split()
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or "\n" in name
            or "\r" in name
            or name in seen
            or stage != "0"
            or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", object_id)
        ):
            raise ValueError("native source index contains unsafe/duplicate/unmerged entries")
        seen.add(name)
        if mode == "160000":
            excluded.append(
                dict(path=name, reason="external_gitlink_disabled_in_cpu_mutation_build")
            )
            continue
        if mode not in {"100644", "100755"}:
            raise ValueError("native source snapshots require regular tracked files")
        original = source / name
        if original.is_symlink() or not original.resolve(strict=True).is_relative_to(
            source.resolve()
        ):
            raise ValueError("native source input escapes its source root")
        row = reference(original)
        if row["bytes"] > 64 * 1024**2:
            raise ValueError("native source file exceeds snapshot byte budget")
        total += row["bytes"]
        if total > max_bytes:
            raise ValueError("native source snapshot exceeds aggregate byte budget")
        files.append(dict(**row, relative_path=name, mode=mode))
    if not files or not {"CMakeLists.txt", "tools/build_manifest.py"}.issubset(seen):
        raise ValueError("native source snapshot lacks project inputs")
    return dict(
        files=files,
        excluded=excluded,
        total_bytes=total,
        identity_sha256=canonical_digest(
            [[row["relative_path"], row["mode"], row["sha256"]] for row in files]
        ),
    )


def copy_sources(inputs: dict, destination: Path):
    if destination.exists():
        raise ValueError("native source snapshot must be fresh")
    destination.mkdir(parents=True)
    for row in inputs["files"]:
        target = destination / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("native snapshot path alias")
        shutil.copyfile(row["path"], target)
        if reference(target)["sha256"] != row["sha256"]:
            raise ValueError("source changed during native snapshot copy")
        if row["mode"] == "100755":
            target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def change(source: bytes, seed: dict) -> bytes:
    if hashlib.sha256(source).hexdigest() != seed["source_sha256"]:
        raise ValueError("native mutation source changed; exact seed review required")
    before, after, anchor = (seed[key].encode("utf-8") for key in ("before", "after", "anchor"))
    matches = [
        value for value in {before, before.replace(b"\n", b"\r\n")} if source.count(value) == 1
    ]
    if (
        len(matches) != 1
        or source.count(anchor) != 1
        or source.index(matches[0]) < source.index(anchor)
        or before == after
    ):
        raise ValueError("unique native production operation/anchor required")
    original = matches[0]
    if b"\r\n" in original:
        after = after.replace(b"\n", b"\r\n")
    return source.replace(original, after, 1)


def assertion_result(seed: dict, inventory: dict, report: str) -> str:
    process = inventory.get("process")
    if (
        not isinstance(process, dict)
        or type(process.get("process_id")) is not int
        or process["process_id"] <= 0
        or type(process.get("returncode")) is not int
        or process["returncode"] not in {0, 8}
        or process.get("timed_out") is not False
        or (process.get("interrupted") is not None and process.get("interrupted") is not False)
        or process.get("limit_exceeded") is not None
        or process.get("capture_error") is not None
    ):
        return (
            "timeout"
            if isinstance(process, dict) and process.get("timed_out") is True
            else "infrastructure_error"
        )
    outcomes = junit(report)
    if (
        inventory.get("missing")
        or inventory.get("unexpected")
        or inventory.get("skipped")
        or inventory.get("disabled")
        or inventory.get("not_run")
        or not outcomes
        or any(row["state"] not in {"run", "fail"} for row in outcomes.values())
    ):
        return "infrastructure_error"
    requested = {row["name"] for row in inventory["selected"]}
    if set(outcomes) != requested:
        return "infrastructure_error"
    if all(row["state"] == "run" for row in outcomes.values()):
        return (
            "survived"
            if inventory.get("ctest_process_succeeded") is True
            else "infrastructure_error"
        )
    if process["returncode"] != 8:
        return "infrastructure_error"
    if any(row["reason"] != "Failed" for row in outcomes.values() if row["state"] == "fail"):
        return "infrastructure_error"
    root = ET.fromstring(report)
    output = {
        case.get("name"): " ".join("".join(case.itertext()).split())
        for case in root.findall("testcase")
    }
    for test in seed["tests"]:
        row = outcomes.get(test["name"])
        if row is None or row["state"] != "fail":
            return "not_covered" if row is None else "unintended_assertion_failure"
        message = output[test["name"]]
        if (
            Path(test["source"]).name not in message
            or " ".join(test["assertion_marker"].split()) not in message
        ):
            return "unintended_assertion_failure"
    return "killed_by_assertion"
