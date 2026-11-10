"""Authored offline tuple custody controls; synthetic inputs are not a host pass."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import gpu_qualification as qualification
import session_platform_contract as contract


def lock_fixture():
    return dict(
        schema=contract.SCHEMA,
        target="gfx942",
        matrix_url="https://rocm.docs.amd.com/en/docs-10.1.0/compatibility/compatibility-matrix.html",
        reviewed_date_utc="2026-11-07",
        image_digest="sha256:" + "a" * 64,
        expected=dict(
            os_id="ubuntu",
            os_version="24.04",
            kernel="fixture-kernel",
            machine="x86_64",
            glibc="glibc fixture-version",
            driver_version=None,
            driver_srcversion="fixture-module",
        ),
        files=[
            dict(role=role, path="/fixture/" + role, bytes=1, sha256="b" * 64)
            for role in ("host_compiler", "stdlib", "hip_compiler", "hip_runtime", "device_library")
        ],
    )


class PlatformContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "platform with spaces.json"
        self.write(lock_fixture())

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")

    def test_exact_native_inputs_do_not_become_support_or_abi_qualification(self):
        reference = contract.load_lock(self.path)
        lock = reference["lock"]
        observed = dict(native=deepcopy(lock["expected"]), files=deepcopy(lock["files"]))
        receipt = contract.evaluate(reference, observed)
        self.assertTrue(receipt["reproducible_inputs_match"])
        self.assertFalse(receipt["supported_tuple_qualified"])
        self.assertFalse(receipt["loaded_firmware_qualified"])
        self.assertFalse(receipt["runtime_transitive_abi_qualified"])
        self.assertIn("operator_declared", receipt["image_identity_scope"])
        for field in contract.FIELDS:
            if lock["expected"][field] is None:
                continue
            changed = deepcopy(observed)
            changed["native"][field] = None
            self.assertFalse(
                contract.evaluate(reference, changed)["reproducible_inputs_match"], field
            )
        for change in ("missing", "hash", "role", "length", "error", "duplicate"):
            changed = deepcopy(observed)
            if change == "missing":
                changed["files"].pop()
            elif change == "duplicate":
                changed["files"][1] = deepcopy(changed["files"][0])
            elif change == "hash":
                changed["files"][0]["sha256"] = "c" * 64
            elif change == "role":
                changed["files"][0]["role"] = "firmware"
            elif change == "length":
                changed["files"][0]["bytes"] = True
            else:
                changed["files"][0]["error"] = "permission denied"
            self.assertTrue(contract.evaluate(reference, changed)["blockers"], change)

    def test_lock_reader_rejects_moving_unbounded_and_incomplete_inputs(self):
        for field, value in (
            (
                "matrix_url",
                "https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html",
            ),
            ("image_digest", "latest"),
            ("target", "gfx1100"),
            ("reviewed_date_utc", "2026-99-99"),
            ("files", []),
            ("expected", {}),
        ):
            changed = lock_fixture()
            changed[field] = value
            self.write(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.load_lock(self.path)
        for key, value in (("path", "/fixture/../escape"), ("bytes", True), ("sha256", "bad")):
            changed = lock_fixture()
            changed["files"][0][key] = value
            self.write(changed)
            with self.assertRaises(ValueError):
                contract.load_lock(self.path)
        self.path.write_text('{"schema":1,"schema":2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            contract.load_lock(self.path)
        self.path.write_bytes(b" " * (1024 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            contract.load_lock(self.path)

    def test_plan_and_child_receipt_bind_the_exact_lock_and_process(self):
        reference = contract.load_lock(self.path)
        plan = qualification.command_plan(
            "mi300x", ["hip-direct"], Path("temp/platform-fixture"), 1, [4], platform_lock=reference
        )
        item = next(row for row in plan if "platform_lock" in row)
        self.assertEqual(plan[0]["name"], "linux-resource-access-preflight")
        self.assertEqual(item["depends_on"], [plan[0]["name"]])
        observed = dict(
            native=deepcopy(reference["lock"]["expected"]),
            files=deepcopy(reference["lock"]["files"]),
        )
        receipt = contract.evaluate(reference, observed) | {"process_id": 123}
        result = dict(
            returncode=0,
            timed_out=False,
            stdout=json.dumps(receipt),
            stderr="",
            process_id=123,
            elapsed_seconds=0.01,
        )
        self.assertEqual(qualification.validate_result(item, result), [])
        result["process_id"] = 124
        self.assertIn(
            "native Linux platform lock preflight failed",
            qualification.validate_result(item, result),
        )
        result["process_id"] = 123
        self.path.write_text(self.path.read_text() + "\n", encoding="utf-8")
        self.assertIn(
            "native Linux platform lock preflight failed",
            qualification.validate_result(item, result),
        )


if __name__ == "__main__":
    unittest.main()
