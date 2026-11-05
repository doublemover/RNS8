"""Typed runtime observations; process ordinals are never physical identities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Literal

from json_evidence import JsonValue


class IdentityUnavailable(ValueError):
    """Observation did not establish a usable identity."""


class IdentityMismatch(ValueError):
    """Observed facts contradict their declared schema or another observation."""


@dataclass(frozen=True)
class RuntimeDeviceIdentity:
    runtime_ordinal: int
    pci_bdf: str | None
    uuid: str | None

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DeviceObservation:
    architecture: str
    name: str
    runtime_version: int
    driver_version: int
    memory_bytes: int
    identity: RuntimeDeviceIdentity

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DeviceRelation:
    state: Literal["verified", "mismatch", "unavailable"]
    reasons: tuple[str, ...]

    def to_json(self) -> dict:
        return dict(
            state=self.state,
            reasons=list(self.reasons),
            scope="node_local_runtime_execution_device_not_a_board_count",
        )


def _uint32(value: JsonValue) -> int:
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
        raise IdentityMismatch("runtime_identity_integer_malformed")
    return value


def runtime_identity(value: JsonValue) -> RuntimeDeviceIdentity:
    if (
        not isinstance(value, dict)
        or value.get("schema") != "rns8_runtime_device_identity_v1"
    ):
        raise IdentityUnavailable("versioned_runtime_identity_missing")
    expected = {
        "schema",
        "runtime_ordinal",
        "query_status_code",
        "valid_fields",
        "pci_bdf",
        "uuid",
        "native_pci_status",
        "native_uuid_status",
    }
    if set(value) != expected:
        raise IdentityMismatch("runtime_identity_fields_incomplete_or_unknown")
    status = _uint32(value["query_status_code"])
    flags = _uint32(value["valid_fields"])
    pci_status, uuid_status = (
        _uint32(value["native_pci_status"]),
        _uint32(value["native_uuid_status"]),
    )
    if flags & ~3:
        raise IdentityMismatch("runtime_identity_availability_mask_unknown")
    if status != 0:
        if flags or value["pci_bdf"] is not None or value["uuid"] is not None:
            raise IdentityMismatch("failed_runtime_identity_query_exposes_identifiers")
        raise IdentityUnavailable("runtime_identity_query_failed")
    ordinal = value["runtime_ordinal"]
    if type(ordinal) is not int or not 0 <= ordinal < 2**31:
        raise IdentityMismatch("runtime_identity_ordinal_malformed")
    bdf = value["pci_bdf"]
    if flags & 1:
        if (
            pci_status != 0
            or not isinstance(bdf, str)
            or re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]", bdf) is None
        ):
            raise IdentityMismatch("runtime_identity_pci_bdf_malformed")
    elif bdf is not None:
        raise IdentityMismatch("unavailable_runtime_pci_bdf_has_a_value")
    uuid = value["uuid"]
    if flags & 2:
        if (
            uuid_status != 0
            or not isinstance(uuid, str)
            or re.fullmatch("[0-9a-f]{32}", uuid) is None
            or uuid == "0" * 32
        ):
            raise IdentityMismatch("runtime_identity_uuid_malformed")
    elif uuid is not None:
        raise IdentityMismatch("unavailable_runtime_uuid_has_a_value")
    if not flags:
        raise IdentityUnavailable("runtime_identity_has_no_available_identifier")
    return RuntimeDeviceIdentity(ordinal, bdf, uuid)


def identity_relation(
    left: RuntimeDeviceIdentity, right: RuntimeDeviceIdentity
) -> DeviceRelation:
    shared = []
    for field in ("uuid", "pci_bdf"):
        first, second = getattr(left, field), getattr(right, field)
        if first is not None and second is not None:
            if first != second:
                return DeviceRelation("mismatch", ("observed_" + field + "_mismatch",))
            shared.append(field)
    if not shared:
        return DeviceRelation("unavailable", ("no_common_observed_device_identifier",))
    # Ordinals intentionally differ between an unmasked parent and its child.
    return DeviceRelation("verified", ())


def _observation(
    device: JsonValue, identity: RuntimeDeviceIdentity
) -> DeviceObservation:
    if not isinstance(device, dict):
        raise IdentityUnavailable("runtime_device_observation_missing")
    available = device.get("hip_available")
    if not (available is True or type(available) is int and available == 1):
        raise IdentityUnavailable("runtime_device_is_not_available")
    ordinal = device.get("device_id")
    if type(ordinal) is not int or ordinal != identity.runtime_ordinal:
        raise IdentityMismatch("runtime_device_ordinal_and_identity_query_disagree")
    architecture, name = device.get("gcn_arch"), device.get("name")
    if (
        not isinstance(architecture, str)
        or re.fullmatch(r"gfx[0-9a-f]+(?::[a-z0-9_+-]+)*", architecture) is None
        or not isinstance(name, str)
        or not name.strip()
    ):
        raise IdentityMismatch("runtime_device_architecture_or_name_malformed")
    runtime, driver, memory = (
        device.get("hip_runtime_version"),
        device.get("hip_driver_version"),
        device.get("global_mem_bytes"),
    )
    if any(type(item) is not int or item <= 0 for item in (runtime, driver, memory)):
        raise IdentityMismatch("runtime_device_runtime_or_driver_identity_missing")
    return DeviceObservation(architecture, name, runtime, driver, memory, identity)


def inspection_observation(value: JsonValue) -> DeviceObservation:
    if not isinstance(value, dict):
        raise IdentityUnavailable("runtime_inspection_missing")
    return _observation(value, runtime_identity(value.get("runtime_device_identity")))


def capture_observation(capture: JsonValue) -> DeviceObservation:
    if not isinstance(capture, dict):
        raise IdentityUnavailable("runtime_capture_missing")
    witness = capture.get("device_identity_witness")
    if (
        not isinstance(witness, dict)
        or witness.get("schema") != "rns8_device_identity_witness_v1"
    ):
        raise IdentityUnavailable("runtime_capture_identity_witness_missing")
    if set(witness) != {"schema", "binding_state", "before", "after"}:
        raise IdentityMismatch("runtime_capture_identity_witness_fields_unknown")
    identities = []
    for field in ("before", "after"):
        try:
            identities.append(runtime_identity(witness[field]))
        except IdentityUnavailable:
            identities.append(None)
    before, after = identities
    if before is None and after is None:
        if witness["binding_state"] != "unavailable":
            raise IdentityMismatch("unavailable_runtime_identity_claims_stability")
        raise IdentityUnavailable("runtime_capture_identity_unavailable")
    if before is None or after is None:
        raise IdentityMismatch("runtime_identity_availability_changed_during_capture")
    if witness["binding_state"] != "stable" or before != after:
        raise IdentityMismatch("runtime_device_identity_changed_during_capture")
    return _observation(capture.get("device"), before)


def observation_relation(
    left: DeviceObservation, right: DeviceObservation
) -> DeviceRelation:
    for field in (
        "architecture",
        "name",
        "runtime_version",
        "driver_version",
        "memory_bytes",
    ):
        if getattr(left, field) != getattr(right, field):
            return DeviceRelation(
                "mismatch", ("observed_device_" + field + "_mismatch",)
            )
    return identity_relation(left.identity, right.identity)


def distinct_execution_devices(
    left: DeviceObservation, right: DeviceObservation
) -> bool:
    """Distinct UUIDs can identify partitions sharing the same PCI function."""
    return distinct_runtime_identities(left.identity, right.identity)


def distinct_runtime_identities(
    left: RuntimeDeviceIdentity, right: RuntimeDeviceIdentity
) -> bool:
    if left.uuid is not None and right.uuid is not None:
        return left.uuid != right.uuid
    if left.pci_bdf is not None and right.pci_bdf is not None:
        return left.pci_bdf != right.pci_bdf
    return False


def capture_relation(capture: JsonValue, parent: JsonValue, child: JsonValue) -> dict:
    observed = capture_observation(capture)
    parent_observed, child_observed = (
        inspection_observation(parent),
        inspection_observation(child),
    )
    for expected in (parent_observed, child_observed):
        relation = observation_relation(expected, observed)
        if relation.state != "verified":
            return relation.to_json()
    return dict(
        state="verified",
        reasons=[],
        observed=observed.to_json(),
        parent=parent_observed.to_json(),
        child=child_observed.to_json(),
        scope="node_local_parent_visibility_child_capture_association",
    )


def inspection_relation(value: JsonValue, parent: JsonValue = None) -> dict:
    observed = inspection_observation(value)
    result = dict(
        state="observed",
        reasons=[],
        observed=observed.to_json(),
        scope="runtime_observation_only",
    )
    if parent is not None:
        parent_observed = inspection_observation(parent)
        result.update(observation_relation(parent_observed, observed).to_json())
        result["parent"] = parent_observed.to_json()
    return result


def binding_identifiers(value: JsonValue) -> dict:
    if not isinstance(value, dict):
        return dict(uuid=None, pci_bdf=None)
    observed = value.get("observed")
    identity = observed.get("identity") if isinstance(observed, dict) else None
    if not isinstance(identity, dict):
        return dict(uuid=None, pci_bdf=None)
    return dict(uuid=identity.get("uuid"), pci_bdf=identity.get("pci_bdf"))
