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
import re

REQUEST_SCHEMA = "rns8_benchmark_request_v1"
ENVIRONMENT_SCHEMA = "rns8_relevant_environment_v1"


def _canonical_environment(environment: dict[str, str]) -> dict[str, str]:
    if type(environment) is not dict or any(type(k) is not str or not k or type(v) is not str
                                           for k, v in environment.items()):
        raise ValueError("environment must map nonempty string names to string values")
    return dict(sorted(environment.items()))


def request_environment_identity(request: object) -> tuple[dict | None, str | None]:
    """Bind the actual emitted object, never an unbound claimed digest.

    An explicitly recorded empty environment is valid. Historical requests with
    missing/malformed evidence stay descriptive and cannot qualify reruns. The
    request seal binds recorded values and explicit override provenance; the
    cohort domain excludes command arguments/output paths from its identity.
    """
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
        return None, "execution_environment_schema_unqualified"
    if "environment" not in request:
        return None, "execution_environment_missing"
    try:
        environment = _canonical_environment(request["environment"])
    except ValueError:
        return None, "execution_environment_malformed"
    sha256 = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if (not isinstance(request.get("arguments"), list) or any(type(v) is not str for v in request["arguments"])
            or not all(sha256(request.get(k)) for k in ("executable_sha256", "overrides_sha256", "identity_sha256"))):
        return None, "execution_request_identity_unqualified"
    # Verify the emitted request seal, but never use its full digest as the
    # cohort key: independent output/report paths may legitimately differ.
    try:
        sealed = {k: v for k, v in request.items() if k != "identity_sha256"}
        expected = hashlib.sha256(json.dumps(sealed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError):
        return None, "execution_request_identity_unqualified"
    if request["identity_sha256"] != expected:
        return None, "execution_request_identity_mismatch"
    payload = {"schema": ENVIRONMENT_SCHEMA, "environment": environment,
               "overrides_sha256": request["overrides_sha256"]}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": ENVIRONMENT_SCHEMA, "sha256": digest}, None


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
    payload = {"schema": REQUEST_SCHEMA, "arguments": command[1:],
               "executable_sha256": file_digest(executable), "environment": _canonical_environment(relevant),
               "overrides_sha256": override_digest}
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def request_mismatch(capture: dict, requested: dict) -> str | None:
    observed = capture.get("request_identity")
    if not isinstance(observed, dict) or observed.get("schema") != requested["schema"]:
        return "missing versioned execution request identity (historical capture remains reviewable)"
    observed_environment, observed_error = request_environment_identity(observed)
    requested_environment, requested_error = request_environment_identity(requested)
    if observed_error or requested_error:
        return "execution request environment unqualified: " + (observed_error or requested_error)
    if observed_environment != requested_environment:
        return "execution request differs: environment"
    for field in ("arguments", "executable_sha256", "environment", "overrides_sha256", "identity_sha256"):
        if observed.get(field) != requested[field]:
            return "execution request differs: " + field
    artifact = capture.get("artifact_capture", {})
    if artifact.get("executable_sha256") != requested["executable_sha256"]:
        return "captured executable does not match the request"
    if artifact.get("unchanged_during_run") is not True:
        return "executable stability during capture was not established"
    return None
