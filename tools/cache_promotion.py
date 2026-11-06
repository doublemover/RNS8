"""Bind reviewed cache payloads to unchanged, content-addressed evidence files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmark_request import file_digest

SCHEMA = "rns8_cache_promotion_receipt_v1"
ROLES = {
    "candidate",
    "observation",
    "baseline",
    "review",
    "variance",
    "target",
    "diagnostic",
    "shape",
    "counter",
}
MAX_INPUTS = 4096
MAX_FILE_BYTES = 128 * 1024**2
MAX_TOTAL_BYTES = 2 * 1024**3


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def entry_digest(entry: dict) -> str:
    if not isinstance(entry, dict) or not isinstance(entry.get("key"), str) or not entry["key"]:
        raise ValueError("cache payload requires a nonempty key")
    # This field records publication time, not measured work or dispatch policy.
    # All other fields, including future additions, participate in admission.
    return digest({key: value for key, value in entry.items() if key != "updated_utc"})


def unique_entries(entries: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for entry in entries:
        identity = entry_digest(entry)
        key = entry["key"]
        if key in result and entry_digest(result[key]) != identity:
            raise ValueError("conflicting same-key cache source payloads: " + key)
        result.setdefault(key, entry)
    return result


def evidence_reference(role: str, path: Path) -> dict:
    if type(role) is not str or role not in ROLES:
        raise ValueError("unknown cache promotion evidence role")
    path = path.resolve(strict=True)
    if not path.is_file() or not 1 <= path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("nonempty bounded regular promotion evidence file required")
    size = path.stat().st_size
    sha256 = file_digest(path)
    if path.stat().st_size != size:
        raise ValueError("promotion evidence changed while hashing")
    return dict(role=role, path=str(path), bytes=size, sha256=sha256)


def receipt_from_references(entry: dict, inputs: list[dict]) -> dict:
    if not 1 <= len(inputs) <= MAX_INPUTS:
        raise ValueError("bounded nonempty promotion evidence inventory required")
    references = {}
    for reference in inputs:
        _validate_reference(reference)
        key = (reference["role"], reference["path"])
        if key in references and references[key] != reference:
            raise ValueError("conflicting promotion evidence references")
        references[key] = dict(reference)
    files = sorted(references.values(), key=lambda item: (item["role"], item["path"]))
    if (
        not {"candidate", "review", "variance"} <= {item["role"] for item in files}
        or sum(item["role"] == "candidate" for item in files) != 1
        or sum(item["bytes"] for item in files) > MAX_TOTAL_BYTES
    ):
        raise ValueError("candidate, review and variance evidence within budget required")
    result = dict(schema=SCHEMA, key=entry["key"], entry_sha256=entry_digest(entry), inputs=files)
    result["identity_sha256"] = digest(result)
    return result


def make_receipt(entry: dict, inputs: list[tuple[str, Path]]) -> dict:
    return receipt_from_references(entry, evidence_references(inputs))


def evidence_references(inputs: list[tuple[str, Path]]) -> list[dict]:
    if not 1 <= len(inputs) <= MAX_INPUTS:
        raise ValueError("bounded nonempty promotion evidence inventory required")
    cached = {}
    references = {}
    total = 0
    for role, path in inputs:
        if type(role) is not str or role not in ROLES:
            raise ValueError("unknown cache promotion evidence role")
        path = path.resolve(strict=True)
        if path not in cached:
            cached[path] = evidence_reference(role, path)
        reference = {**cached[path], "role": role}
        key = (role, str(path))
        if key not in references:
            total += reference["bytes"]
            if total > MAX_TOTAL_BYTES:
                raise ValueError("promotion evidence exceeds total byte budget")
            references[key] = reference
    return list(references.values())


def _validate_reference(reference: object) -> None:
    if not isinstance(reference, dict) or set(reference) != {"role", "path", "bytes", "sha256"}:
        raise ValueError("malformed promotion evidence reference")
    if (
        type(reference["role"]) is not str
        or reference["role"] not in ROLES
        or type(reference["path"]) is not str
        or not Path(reference["path"]).is_absolute()
        or type(reference["bytes"]) is not int
        or not 1 <= reference["bytes"] <= MAX_FILE_BYTES
        or type(reference["sha256"]) is not str
        or len(reference["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in reference["sha256"])
    ):
        raise ValueError("absolute bounded content-addressed promotion evidence required")


def validate_receipt(entry: dict, receipt: object) -> None:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "key",
        "entry_sha256",
        "inputs",
        "identity_sha256",
    }:
        raise ValueError("current content-bound cache promotion receipt required")
    if (
        receipt["schema"] != SCHEMA
        or receipt["key"] != entry.get("key")
        or receipt["entry_sha256"] != entry_digest(entry)
        or receipt["identity_sha256"]
        != digest({key: value for key, value in receipt.items() if key != "identity_sha256"})
    ):
        raise ValueError("cache promotion payload or receipt identity changed")
    inputs = receipt["inputs"]
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= MAX_INPUTS:
        raise ValueError("bounded promotion evidence references required")
    references = []
    cached = {}
    total = 0
    for reference in inputs:
        _validate_reference(reference)
        total += reference["bytes"]
        if total > MAX_TOTAL_BYTES:
            raise ValueError("promotion evidence exceeds total byte budget")
        path = Path(reference["path"])
        if path not in cached:
            cached[path] = evidence_reference(reference["role"], path)
        current = {**cached[path], "role": reference["role"]}
        if current != reference:
            raise ValueError("promotion evidence changed: " + reference["path"])
        references.append(current)
    # Reapply role, duplicate, ordering and candidate cardinality policy as well.
    if receipt_from_references(entry, references) != receipt:
        raise ValueError("noncanonical cache promotion evidence inventory")
