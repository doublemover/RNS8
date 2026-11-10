"""Compiled ETL reader identity, including its portable native payload helper."""

from __future__ import annotations

import hashlib
from pathlib import Path

from benchmark_request import file_digest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_INPUTS = ("tools/rns8_etl_inspect.cpp", "src/diagnostics/etl_sample_payload.hpp")


def source_identity() -> str:
    # This exact UTF-8 sequence is also hashed by CMake before compilation.
    hashes = "".join(file_digest(ROOT / path) + "\n" for path in SOURCE_INPUTS)
    return hashlib.sha256(hashes.encode("ascii")).hexdigest()
