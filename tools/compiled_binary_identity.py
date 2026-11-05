"""Validate the existing compiled-manifest contract at executable/library boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from collections.abc import Callable

from build_manifest import manifest_provenance_complete
from json_evidence import JsonValue, parse_evidence_json, same_evidence_value
from benchmark_request import file_digest

SCHEMA = "rns8_compiled_binary_identity_v1"
RECEIPT_SCHEMA = "rns8_compiled_binary_identity_receipt_v1"


def canonical_digest(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_document(value: JsonValue, backend: str | None = None) -> dict:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("versioned compiled binary identity required")
    executable = value.get("build_manifest")
    library = value.get("library_build_manifest")
    if not manifest_provenance_complete(executable, backend) or not manifest_provenance_complete(
        library, backend
    ):
        raise ValueError("complete compiled executable and library manifests required")
    if not same_evidence_value(executable, library):
        raise ValueError(
            "executable and actual linked library were compiled with different manifests"
        )
    declaration = value.get("library_abi_declaration_sha256")
    if not isinstance(declaration, str) or re.fullmatch("[0-9a-f]{64}", declaration) is None:
        raise ValueError("actual linked library ABI declaration identity required")
    return dict(
        build_manifest_sha256=canonical_digest(executable),
        library_build_manifest_sha256=canonical_digest(library),
        library_abi_declaration_sha256=declaration,
    )


def bind_document(
    binary: Path, document_path: Path, *, before_sha256: str, backend: str | None = None
) -> dict:
    raw = document_path.read_bytes()
    document = parse_evidence_json(raw.decode("utf-8"))
    summary = validate_document(document, backend)
    after = file_digest(binary)
    if before_sha256 != after:
        raise ValueError("compiled binary changed during identity query")
    return dict(
        schema=RECEIPT_SCHEMA,
        executable=str(binary.resolve(strict=True)),
        executable_sha256=after,
        identity_document=str(document_path.resolve(strict=True)),
        identity_document_sha256=hashlib.sha256(raw).hexdigest(),
        unchanged_during_query=True,
        **summary,
    )


def validate_receipt(
    value: JsonValue,
    *,
    backend: str | None = None,
    verify_file: Callable[[Path, str], object] | None = None,
) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema") != RECEIPT_SCHEMA
        or value.get("unchanged_during_query") is not True
    ):
        raise ValueError("stable compiled binary identity receipt required")
    binary = Path(str(value.get("executable"))).resolve(strict=True)
    document = Path(str(value.get("identity_document"))).resolve(strict=True)
    if verify_file is None:
        if file_digest(binary) != value.get("executable_sha256") or file_digest(
            document
        ) != value.get("identity_document_sha256"):
            raise ValueError("compiled binary identity receipt files are stale")
    else:
        verify_file(binary, value.get("executable_sha256"))
        verify_file(document, value.get("identity_document_sha256"))
    raw = document.read_bytes()
    if hashlib.sha256(raw).hexdigest() != value.get("identity_document_sha256"):
        raise ValueError("compiled identity JSON changed while it was read")
    summary = validate_document(parse_evidence_json(raw.decode("utf-8")), backend)
    if any(not same_evidence_value(value.get(key), item) for key, item in summary.items()):
        raise ValueError("compiled binary identity summary does not match its raw document")
    return dict(value)
