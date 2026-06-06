#!/usr/bin/env python3
"""Shared benchmark schema self-test fixture builders."""

from __future__ import annotations

from pathlib import Path

_SUPPORT_FRAGMENTS = [
    "test_benchmark_schema_support_core.py",
    "test_benchmark_schema_support_metadata.py",
    "test_benchmark_schema_support_direct_reuse.py",
    "test_benchmark_schema_support_chains_wrap64.py",
]


def _load_support_fragment(name: str) -> None:
    source = (Path(__file__).resolve().with_name(name)).read_text(encoding="utf-8")
    exec(compile(source, name, "exec"), globals(), globals())


for _fragment in _SUPPORT_FRAGMENTS:
    _load_support_fragment(_fragment)

del _fragment
