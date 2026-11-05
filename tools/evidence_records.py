"""Small concrete evidence projections; runtime admission stays with validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class Command:
    role: str
    argv: tuple[str, ...]


class ArtifactReference(TypedDict):
    path: str
    sha256: str
