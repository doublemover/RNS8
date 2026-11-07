"""Shared production admission for immutable GPU/host profiling sessions."""

from pathlib import Path
import copy

from benchmark_request import (
    request_environment_identity,
    request_identity,
    attach_capture_identity,
)
from benchmark_schema import validate_capture
from build_manifest import manifest_provenance_complete
from device_observation import capture_observation, observation_relation
from execution_observation import validate_execution_request, execution_observation
from compiled_binary_identity import canonical_digest
from diagnostic_companion import common_identity
from json_evidence import same_evidence_value
from measurement_contract import profile_ranges_instrumented
from reference_verdict import numerical_observation
from target_qualification import EvidenceStore
from build_instrumentation import instrumentation_reasons


def production_profile_input(value: dict, store: EvidenceStore) -> Path:
    validate_capture(value)
    if instrumentation_reasons(value.get("build_manifest")):
        raise ValueError("instrumented build cannot supply the production profile baseline")
    request = value.get("request_identity")
    _, error = request_environment_identity(request)
    if error:
        raise ValueError(error)
    validate_execution_request(value, request)
    timing = value.get("timing_metadata") or {}
    if (
        value.get("external_instrumentation") is not None
        or profile_ranges_instrumented(value)
        or timing.get("host_instrumentation_state") != "hooks_disabled"
        or timing.get("gpu_event_timing") is True
    ):
        raise ValueError("collection needs a hook-disabled production observation")
    if not numerical_observation(value).get("full_output_reference_covered"):
        raise ValueError("collection requires completed full-output numerical evidence")
    if not manifest_provenance_complete(value.get("build_manifest"), value.get("backend_selected")):
        raise ValueError("production compiled manifest is incomplete")
    artifact = value.get("artifact_capture") or {}
    if (
        artifact.get("rns8_linkage") != "static"
        or artifact.get("unchanged_during_run") is not True
        or artifact.get("executable_sha256") != request["executable_sha256"]
    ):
        raise ValueError("production has no stable static executable association")
    binary = store.verify(Path(artifact["executable_path"]), artifact["executable_sha256"])
    if any(
        key.startswith(("ROCP_", "ROCPROF", "PERF_", "MULL_")) or key == "LD_PRELOAD"
        for key in request["environment"]
    ):
        raise ValueError("production environment already contains profiler injection")
    if any(flag in request["arguments"] for flag in ("--json-output", "--build-identity")):
        raise ValueError("production arguments contain an acquisition-only output/query option")
    capture_observation(value)  # No physical-identity-free collection admission.
    return binary


def bind_host_profile_output(production, raw, *, workload, environment, executable, execution_id):
    """Share exact work/environment/device admission across native host profilers."""
    validate_capture(raw)
    if not numerical_observation(raw).get("full_output_reference_covered"):
        raise ValueError("profiled benchmark has no completed full-output numerical evidence")
    observed = execution_observation(raw, required=True)
    if list(observed.arguments) != workload[1:] or not same_evidence_value(
        observed.environment, environment
    ):
        raise ValueError("profiled native argv/environment differs from acquisition")
    if (
        observation_relation(capture_observation(raw), capture_observation(production)).state
        != "verified"
    ):
        raise ValueError("profiled benchmark used a different physical execution device")
    request = request_identity(workload, environ=observed.environment)
    request["overrides_sha256"] = production["request_identity"]["overrides_sha256"]
    request["identity_sha256"] = canonical_digest(
        {key: value for key, value in request.items() if key != "identity_sha256"}
    )
    profiled = copy.deepcopy(raw)
    attach_capture_identity(profiled, request, executable, execution_id=execution_id)
    projection = copy.deepcopy(profiled)
    projection["request_identity"] = copy.deepcopy(production["request_identity"])
    if not same_evidence_value(common_identity(production), common_identity(projection)):
        raise ValueError("profiled work/proof/output/reuse contract differs from production")
    return profiled
