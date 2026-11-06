"""Small file-custody primitives shared by explicit dependency probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def snapshot(path: Path) -> dict:
    path = path.resolve(strict=True)
    size = path.stat().st_size
    if not path.is_file() or size > 1024**3:
        raise ValueError("probe input must be a regular file of at most 1 GiB")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": size, "sha256": digest}


def write(path: Path, value: dict) -> dict:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return snapshot(path)
