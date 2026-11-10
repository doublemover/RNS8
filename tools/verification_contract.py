"""Shared diagnostic microcase bytes/provenance contracts; never promotion proof."""

from __future__ import annotations

from collections.abc import Iterable

from build_manifest import manifest_provenance_complete
from evidence_scalars import demand
from json_evidence import JsonValue

MASK = (1 << 64) - 1
BUDGET = 256 * 1024 * 1024


def checksum(data: Iterable[int]) -> str:
    """FNV-1a over the supplied byte order, preserving leading hexadecimal zeros."""
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & MASK
    return f"{value:016x}"


def validate_provenance(manifest: JsonValue, backend: str | None = None) -> None:
    """Structural diagnostic admission preserves its narrower eligibility role."""
    if type(manifest) is not dict or type(manifest.get("source_dirty")) is not bool:
        raise ValueError("missing build provenance")
    structural = dict(manifest, source_dirty=False)
    if structural.get("dependency_digest") == "unavailable":
        structural["dependency_digest"] = "0" * 64
    configuration = structural.get("configuration")
    if isinstance(configuration, str):
        structural["configuration"] = "\n".join(
            "garner_table=OFF" if line.partition("=")[0] == "garner_table" else line
            for line in configuration.splitlines()
        )
    demand(manifest_provenance_complete(structural, backend), "invalid structural build provenance")
