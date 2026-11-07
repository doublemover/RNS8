"""Synthetic native identity records; no runtime or device is queried."""

from __future__ import annotations

import copy
import unittest

from device_observation import (
    IdentityMismatch,
    IdentityUnavailable,
    runtime_identity,
    identity_relation,
    inspection_observation,
    observation_relation,
    capture_observation,
    capture_relation,
    distinct_execution_devices,
)
import gpu_qualification
from test_capture_identity_support import synthetic_runtime_device_identity as identity


def inspection(ordinal=0, *, device=4):
    return dict(
        device_id=ordinal,
        process_id=100 + ordinal,
        name="synthetic device",
        gcn_arch="gfx942",
        hip_available=1,
        hip_runtime_version=70100000,
        hip_driver_version=70100000,
        global_mem_bytes=1024,
        backend="hip-direct",
        runtime_device_identity=identity(ordinal, device=device),
    )


def capture(*, device=4):
    return dict(
        device={
            key: value
            for key, value in inspection(device=device).items()
            if key not in ("backend", "runtime_device_identity", "process_id")
        },
        device_identity_witness=dict(
            schema="rns8_device_identity_witness_v1",
            binding_state="stable",
            before=identity(device=device),
            after=identity(device=device),
        ),
    )


class DeviceObservations(unittest.TestCase):
    def test_parent_and_isolated_ordinals_are_distinct_namespaces(self):
        parent, child = inspection(4), inspection(0)
        self.assertEqual(
            observation_relation(
                inspection_observation(parent), inspection_observation(child)
            ).state,
            "verified",
        )
        self.assertEqual(capture_relation(capture(), parent, child)["state"], "verified")
        self.assertEqual(runtime_identity(parent["runtime_device_identity"]).runtime_ordinal, 4)
        self.assertEqual(capture_observation(capture()).identity.runtime_ordinal, 0)
        other = inspection(0, device=5)
        self.assertEqual(
            observation_relation(
                inspection_observation(parent), inspection_observation(other)
            ).state,
            "mismatch",
        )

    def test_partitions_need_distinct_uuid_when_sharing_pci_function(self):
        first = inspection(0)
        second = inspection(1)
        second["runtime_device_identity"]["uuid"] = "e" * 32
        left, right = inspection_observation(first), inspection_observation(second)
        self.assertTrue(distinct_execution_devices(left, right))
        self.assertEqual(observation_relation(left, right).state, "mismatch")
        second["runtime_device_identity"]["uuid"] = None
        second["runtime_device_identity"]["valid_fields"] = 1
        self.assertFalse(distinct_execution_devices(left, inspection_observation(second)))
        self.assertEqual(
            identity_relation(
                runtime_identity(identity(uuid=False)),
                runtime_identity(identity(pci=False)),
            ).state,
            "unavailable",
        )

    def test_missing_malformed_unknown_and_nil_identifiers_do_not_qualify(self):
        with self.assertRaises(IdentityUnavailable):
            runtime_identity(None)
        with self.assertRaises(IdentityUnavailable):
            runtime_identity(identity(uuid=False, pci=False))
        for field, value in (
            ("uuid", "0" * 32),
            ("uuid", "a-b"),
            ("pci_bdf", "0000:01:20.0"),
            ("valid_fields", 8),
            ("runtime_ordinal", True),
            ("query_status_code", False),
            ("native_uuid_status", 801),
        ):
            bad = identity()
            bad[field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(IdentityMismatch),
            ):
                runtime_identity(bad)

    def test_changed_or_lost_identity_cannot_claim_a_stable_witness(self):
        original = capture()
        for mode in ("changed", "lost", "ordinal", "claimed_state"):
            value = copy.deepcopy(original)
            witness = value["device_identity_witness"]
            if mode == "changed":
                witness["after"]["uuid"] = "f" * 32
            elif mode == "lost":
                witness["after"] = identity(uuid=False, pci=False)
                witness["binding_state"] = "unavailable"
            elif mode == "ordinal":
                witness["after"]["runtime_ordinal"] = 1
            else:
                witness["binding_state"] = "changed"
            with self.subTest(mode=mode), self.assertRaises(IdentityMismatch):
                capture_observation(value)

    def test_session_requires_observed_identity_for_multidevice_inspection(self):
        item = dict(
            inspect_target="gfx942",
            inspect_backend="hip-direct",
            inspect_device=0,
            match_parent="parent",
            require_physical_identity=True,
        )
        import json

        result = dict(
            returncode=0,
            timed_out=False,
            process_id=100,
            stdout=json.dumps(inspection(0)),
        )
        self.assertEqual(gpu_qualification.validate_result(item, result, inspection(4)), [])
        swapped = inspection(0, device=5)
        self.assertTrue(
            gpu_qualification.validate_result(
                item, {**result, "stdout": json.dumps(swapped)}, inspection(4)
            )
        )
        absent = inspection(0)
        absent.pop("runtime_device_identity")
        self.assertTrue(
            gpu_qualification.validate_result(
                item, {**result, "stdout": json.dumps(absent)}, inspection(4)
            )
        )


if __name__ == "__main__":
    unittest.main()
