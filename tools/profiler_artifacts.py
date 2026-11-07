"""Exclusive, bounded diagnostic JSON publication inside an owned session."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from gpu_qualification import owned_artifact_files

MAX_DOCUMENT_BYTES = 64 * 1024**2


def write_diagnostic(
    path: Path, value: dict, *, root: Path, total_bytes: int, minimum_free_bytes: int
) -> dict:
    if any(type(limit) is not int or limit < 1 for limit in (total_bytes, minimum_free_bytes)):
        raise ValueError("positive diagnostic publication budgets required")
    root, path = Path(root).absolute(), Path(path).absolute()
    if root.resolve(strict=True) != root or path.parent != root:
        raise ValueError("derived profiler document must belong directly to its owned session")
    if path.exists():
        raise ValueError("derived profiler document already exists")
    retained = sum(item.stat().st_size for item in owned_artifact_files(root, root.parent))
    remaining = total_bytes - retained
    available = shutil.disk_usage(root).free - minimum_free_bytes
    if remaining < 1:
        raise ValueError("derived profiler document exceeds total artifact budget")
    if available < 1:
        raise ValueError("derived profiler document lacks its free-space reserve")
    encoder = json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":"))
    digest, written = hashlib.sha256(), 0
    # Encode arrays/maps incrementally instead of materializing all their JSON.
    # Individual strings retain the decoder's separate value-size limits.
    # A failed partial file stays in the ignored session, without a receipt.
    with path.open("xb") as stream:

        def emit(raw):
            nonlocal written
            if len(raw) > MAX_DOCUMENT_BYTES - written:
                raise ValueError("derived profiler document exceeds64MiB")
            if len(raw) > remaining - written:
                raise ValueError("derived profiler document exceeds total artifact budget")
            if len(raw) > available - written:
                raise ValueError("derived profiler document lacks its free-space reserve")
            if stream.write(raw) != len(raw):
                raise OSError("derived profiler document write was incomplete")
            digest.update(raw)
            written += len(raw)

        for chunk in encoder.iterencode(value):
            emit(chunk.encode("utf-8"))
        emit(b"\n")
    return dict(path=str(path), sha256=digest.hexdigest())
