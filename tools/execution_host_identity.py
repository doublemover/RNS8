"""Observed OS-instance and node-local runtime-device identities for timing cohorts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from device_observation import capture_observation, observation_relation
from metadata_registry_constants import HIP_RESIDENT_BACKENDS

SCHEMA = "rns8_execution_host_identity_v1"
GPU_BACKENDS = frozenset(HIP_RESIDENT_BACKENDS)
SOURCES = {
    "windows": {"native_registry_machine_guid_64_view"},
    "linux": {"/etc/machine-id", "/var/lib/dbus/machine-id"},
}


def is_gpu_capture(capture: dict) -> bool:
    backend = capture.get("backend_selected")
    return isinstance(backend, str) and backend in GPU_BACKENDS


@dataclass(frozen=True)
class ExecutionHostIdentity:
    platform: str
    os_instance_sha256: str
    scope: str = "observed_os_instance"

    def to_json(self) -> dict:
        return asdict(self)


def host_identity(capture: dict) -> ExecutionHostIdentity:
    value = capture.get("execution_host_identity")
    if not isinstance(value, dict):
        raise ValueError("execution_host_identity_missing")
    if value.get("schema") != SCHEMA or set(value) != {
        "schema",
        "scope",
        "status",
        "platform",
        "source",
        "native_status",
        "os_instance_sha256",
    }:
        raise ValueError("execution_host_identity_schema_or_fields_invalid")
    platform = value.get("platform")
    if not isinstance(platform, str) or platform not in SOURCES:
        raise ValueError("execution_host_platform_unsupported")
    if value.get("scope") != "observed_os_instance" or value.get("status") != "observed":
        raise ValueError("execution_host_identity_not_observed")
    if type(value.get("source")) is not str or value["source"] not in SOURCES[platform]:
        raise ValueError("execution_host_identity_source_unknown")
    if type(value.get("native_status")) is not int or value["native_status"] != 0:
        raise ValueError("execution_host_native_observation_failed")
    identity = value.get("os_instance_sha256")
    if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise ValueError("execution_host_derived_identity_invalid")
    return ExecutionHostIdentity(platform, identity)


def hardware_identity_blockers(capture: dict) -> list[str]:
    blockers = []
    try:
        host_identity(capture)
    except ValueError as error:
        blockers.append(str(error))
    if is_gpu_capture(capture):
        try:
            capture_observation(capture)
        except ValueError as error:
            blockers.append("runtime_device_identity:" + str(error))
    return blockers


def hardware_cohort_identity(capture: dict, *, cpu_anchor: bool = False) -> dict:
    try:
        host = dict(state="observed", identity=host_identity(capture).to_json())
    except ValueError as error:
        host = dict(
            state="unqualified", reason=str(error), raw=capture.get("execution_host_identity")
        )
    device = None
    if not cpu_anchor and is_gpu_capture(capture):
        try:
            device = dict(state="observed", identity=capture_observation(capture).to_json())
        except ValueError as error:
            device = dict(
                state="unqualified", reason=str(error), raw=capture.get("device_identity_witness")
            )
    return dict(host_os_instance=host, node_local_runtime_device=device)


def hardware_pair_blockers(baseline: dict, candidate: dict) -> list[str]:
    try:
        if host_identity(baseline) != host_identity(candidate):
            return ["execution_host_identity_mismatch"]
        if is_gpu_capture(baseline) and is_gpu_capture(candidate):
            relation = observation_relation(
                capture_observation(baseline), capture_observation(candidate)
            )
            if relation.state != "verified":
                return ["runtime_device_identity_pair:" + value for value in relation.reasons]
    except ValueError:
        # Each side's phase admission retains the concrete unavailable/error fact.
        return []
    return []
