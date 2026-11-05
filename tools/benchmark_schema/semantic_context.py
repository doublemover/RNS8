"""Required semantic dispatch fields; raw capture values retain explicit admission."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticContext:
    semantics: object
    prefix: object
    packed_layout: object
    schedule: object
    backend_metadata: object
    k_value: int
    bound_mode: object
    residue_chain_length: int
    residue_output_mode: str
    status_check: object
    prefix_policy: object
    metadata: object
