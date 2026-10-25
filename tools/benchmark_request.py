"""Versioned capture request identity, independent of a mutable checkout HEAD.

RNS8's build manifest specifies static RNS8 linkage. Hashing the executable
therefore binds host code and embedded HIP code; runtime selection environment
is recorded separately. These hashes are computed outside benchmark repeats.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_identity(command: list[str], overrides: dict[str, str] | None = None,
                     environ: dict[str, str] | None = None) -> dict:
    env = dict(os.environ if environ is None else environ)
    env.update(overrides or {})
    executable = Path(shutil.which(command[0], path=env.get("PATH")) or command[0]).resolve()
    relevant = {key: value for key, value in env.items()
                if key.startswith(("RNS8_", "OMP_", "HIP_", "ROCR_", "HSA_", "KMP_"))
                or key in {"CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD"}}
    # Include explicit overrides even if they are a new benchmark control. Store
    # only their digest: unrelated inherited credentials are never inspected.
    override_digest = hashlib.sha256(json.dumps(overrides or {}, sort_keys=True).encode()).hexdigest()
    payload = {"schema": "rns8_benchmark_request_v1", "arguments": command[1:],
               "executable_sha256": file_digest(executable), "environment": relevant,
               "overrides_sha256": override_digest}
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def request_mismatch(capture: dict, requested: dict) -> str | None:
    observed = capture.get("request_identity")
    if not isinstance(observed, dict) or observed.get("schema") != requested["schema"]:
        return "missing versioned execution request identity (historical capture remains reviewable)"
    for field in ("arguments", "executable_sha256", "environment", "overrides_sha256", "identity_sha256"):
        if observed.get(field) != requested[field]:
            return "execution request differs: " + field
    artifact = capture.get("artifact_capture", {})
    if artifact.get("executable_sha256") != requested["executable_sha256"]:
        return "captured executable does not match the request"
    if artifact.get("unchanged_during_run") is not True:
        return "executable stability during capture was not established"
    return None
