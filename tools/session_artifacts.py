"""Local, integrity-checked artifact staging for the existing qualification runner.

No download, extraction, install, executable launch or binary reuse admission.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

KINDS = {"package_source", "compiler_cache", "binary_code_object", "runtime_library", "autotune_evidence"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate artifact inventory field: " + key)
        result[key] = value
    return result


def load_inventory(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != {"schema", "artifacts"} or value["schema"] != "rns8_session_artifacts_v1":
        raise ValueError("unsupported artifact inventory schema")
    rows = value["artifacts"]
    if not isinstance(rows, list) or not rows or len(rows) > 10000:
        raise ValueError("artifact inventory must contain 1..10000 entries")
    names, paths = set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "kind", "path", "bytes", "sha256", "origin_url", "authentication"}:
            raise ValueError("invalid artifact fields")
        if not isinstance(row["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", row["id"]) or row["id"] in names:
            raise ValueError("invalid or duplicate artifact id")
        names.add(row["id"])
        name = row["path"]
        if not isinstance(name, str) or not name or len(name) > 1024 or re.search(r"[\\:\x00-\x1f]", name):
            raise ValueError("artifact path must be a safe relative POSIX path")
        parts = name.split("/")
        if any(part in {"", ".", ".."} or part.endswith((" ", ".")) for part in parts) or PurePosixPath(name).is_absolute():
            raise ValueError("artifact path escapes or aliases its cache root")
        if any(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part) for part in parts):
            raise ValueError("artifact path uses a reserved device name")
        folded = name.casefold()
        if folded in paths:
            raise ValueError("duplicate or overlapping artifact path")
        paths.add(folded)
        if not isinstance(row["kind"], str) or row["kind"] not in KINDS or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= (1 << 63) - 1:
            raise ValueError("invalid artifact kind or byte count")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("artifact requires a lowercase SHA256 identity")
        url = row["origin_url"]
        if url is not None:
            if not isinstance(url, str) or re.search(r"[\x00-\x20]", url):
                raise ValueError("invalid artifact origin")
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("origin must be a credential-free HTTPS attribution URL")
        if not isinstance(row["authentication"], str) or row["authentication"] not in {"local_digest_only", "upstream_checksum_verified", "upstream_signature_verified"}:
            raise ValueError("invalid declared authentication provenance")
    for name in paths:
        parts = name.split("/")
        if any("/".join(parts[:index]) in paths for index in range(1, len(parts))):
            raise ValueError("duplicate or overlapping artifact path")
    return value


def contained_path(root: Path, name: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*PurePosixPath(name).parts)
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("artifact cache paths cannot traverse links: " + name)
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("artifact path escapes cache root: " + name)
    return candidate


def inspect_cache(inventory: dict, root: Path) -> dict:
    records = []
    for row in inventory["artifacts"]:
        record = {"id": row["id"], "path": row["path"], "kind": row["kind"], "expected_bytes": row["bytes"],
                  "expected_sha256": row["sha256"], "observed_bytes": None, "observed_sha256": None}
        try:
            path = contained_path(root, row["path"])
            if not path.is_file():
                record["status"] = "missing"
            else:
                record["observed_bytes"] = path.stat().st_size
                record["observed_sha256"] = digest(path)
                record["status"] = "verified" if (record["observed_bytes"] == row["bytes"] and
                    record["observed_sha256"] == row["sha256"]) else "mismatch"
        except (OSError, ValueError) as error:
            record.update(status="unreadable_or_unsafe", error=str(error))
        records.append(record)
    return {"schema": "rns8_session_artifact_receipt_v1", "cache_root": str(root.resolve()),
            "complete": all(row["status"] == "verified" for row in records), "artifacts": records,
            "network_used": False, "installed": False, "binary_reuse_qualified": False,
            "authentication_checked_by_this_tool": False, "hardware_qualified": False}


def _copy_payload(input_stream, output, expected_bytes: int) -> None:
    remaining = expected_bytes
    while remaining:
        chunk = input_stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("artifact changed during staging: source truncated")
        output.write(chunk)
        remaining -= len(chunk)
    if input_stream.read(1):
        raise ValueError("artifact changed during staging: source grew")


def stage_cache(inventory: dict, source: Path, destination: Path, *, reserve_bytes: int = 0) -> dict:
    # Complete missing/corrupt list precedes mkdir/copy. Caller records this same
    # inspection in its plan so an offline session can prepare every missing item.
    if destination.is_symlink() or (hasattr(destination, "is_junction") and destination.is_junction()):
        raise ValueError("artifact staging destination cannot be a link")
    source_receipt = inspect_cache(inventory, source)
    if not source_receipt["complete"]:
        raise ValueError("artifact cache incomplete: " + ", ".join(row["id"] + ":" + row["status"]
                         for row in source_receipt["artifacts"] if row["status"] != "verified"))
    pending = []
    for row in inventory["artifacts"]:
        target = contained_path(destination, row["path"])
        if target.exists():
            if not target.is_file() or target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                raise ValueError("existing staged artifact differs; choose a fresh output directory: " + row["path"])
        else:
            pending.append(row)
    ancestor = destination.resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < sum(row["bytes"] for row in pending) + reserve_bytes:
        raise ValueError("insufficient free space for staged artifacts and session reserve")
    copied = 0
    for row in pending:
        target = contained_path(destination, row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        source_path = contained_path(source, row["path"])
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".rns8-stage-", dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                with source_path.open("rb") as input_stream:
                    _copy_payload(input_stream, output, row["bytes"])
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != row["bytes"] or digest(temporary) != row["sha256"]:
                raise ValueError("artifact changed during staging: " + row["id"])
            # Exclusive publication never replaces a pre-existing user's file.
            # Hard linking the completed same-directory temporary is atomic.
            target = contained_path(destination, row["path"])
            os.link(temporary, target)
            copied += row["bytes"]
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    receipt = inspect_cache(inventory, destination)
    if not receipt["complete"]:
        raise ValueError("staged artifact readback failed")
    receipt.update(copied_bytes=copied, reused_artifacts=len(inventory["artifacts"]) - len(pending))
    return receipt
