"""Authored shared source-contract and qualification case controls; no device use."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

import source_gemm_contract as source
import gpu_qualification as qualification


class SourceContract(unittest.TestCase):
    def test_source_inventory_preserves_signed_unsigned_wrap_and_finite_domains(self):
        contract = source.load_contract()
        self.assertEqual(contract.version, 1)
        self.assertEqual(len(contract.backends), 10)
        self.assertFalse(
            source.admission(contract, "ck", "bounded-u64", architecture="gfx942")[
                "source_eligible"
            ]
        )
        for backend in ("hip-direct", "hipblaslt", "rocwmma", "amdgpu-builtins"):
            value = source.admission(contract, backend, "bounded-u64", architecture="gfx942")
            self.assertTrue(value["source_eligible"])
            self.assertIsNone(value["compiled_available"])
            self.assertIsNone(value["runtime_available"])
            self.assertFalse(value["hardware_qualified"])
        self.assertTrue(
            source.admission(contract, "hip-direct", "wrap-u64", architecture="gfx1100")[
                "source_eligible"
            ]
        )
        self.assertFalse(source.admission(contract, "cpu-reference", "wrap-u64")["source_eligible"])
        for modulus, eligible in ((0, False), (1, False), (2, True), (255, True), (256, True)):
            self.assertEqual(
                source.admission(
                    contract, "ck", "finite-u8-ring", modulus=modulus, architecture="gfx1100"
                )["source_eligible"],
                eligible,
            )
        self.assertFalse(
            source.admission(contract, "ck", "finite-u8-field", modulus=255, architecture="gfx942")[
                "source_eligible"
            ]
        )
        self.assertTrue(
            source.admission(contract, "ck", "finite-u8-field", modulus=251, architecture="gfx942")[
                "source_eligible"
            ]
        )

    def test_parser_rejects_unknown_dialect_partial_duplicate_and_changed_domains(self):
        raw = source.CONTRACT.read_bytes()
        for changed in (
            raw + b"invoke_other_code()",
            raw.replace(b"CONTRACT(1)", b"CONTRACT(2)"),
            raw.replace(b'"bounded-u64"', b'"bounded-i64"'),
            raw.replace(b'"ck"', b'"hipblaslt"'),
            raw.replace(b"2, 256", b"2, 255"),
            raw[:100],
            b"\xff",
        ):
            with self.subTest(changed=changed[:40]), self.assertRaises(ValueError):
                source.parse_contract(changed)
        contract = source.load_contract()
        for backend, semantic, kwargs in (
            ("unknown", "bounded-i64", {}),
            ("ck", "unknown", {"architecture": "gfx942"}),
            ("ck", "bounded-i64", {}),
            ("ck", "bounded-i64", {"architecture": "gfx942", "modulus": True}),
            ("ck", "bounded-i64", {"architecture": "gfx942", "modulus": 7}),
        ):
            with self.subTest(backend=backend, semantic=semantic), self.assertRaises(ValueError):
                source.admission(contract, backend, semantic, **kwargs)

    def test_ck_positive_plans_reach_later_supported_cases_in_both_target_and_lane_modes(self):
        for target, arch in (("mi300x", "gfx942"), ("7900xtx", "gfx1100")):
            for combined in (False, True):
                with self.subTest(target=target, combined=combined):
                    plan = qualification.command_plan(
                        target, ["ck"], Path("temp/source-plan"), 1, [0], all_enabled=combined
                    )
                    captures = [
                        row
                        for row in plan
                        if "source_operation" in row and row["source_operation"]["backend"] == "ck"
                    ]
                    semantics = {row["source_operation"]["semantics"] for row in captures}
                    self.assertEqual(
                        semantics,
                        {
                            "bounded-i64",
                            "exact-wide-signed",
                            "exact-wide-unsigned",
                            "finite-u8-ring",
                            "finite-u8-field",
                        },
                    )
                    for row in captures:
                        qualification.validate_source_case(row)
                        self.assertEqual(row["source_operation"]["requested_architecture"], arch)
                    omitted = [
                        row
                        for row in qualification.session_case_inventory("ck", arch, "first-pass")
                        if not row["admission"]["source_eligible"]
                    ]
                    self.assertEqual([row["semantics"] for row in omitted], ["bounded-u64"])
        smoke = qualification.session_case_inventory("ck", "gfx942", "smoke")
        self.assertEqual([row["semantics"] for row in smoke], ["bounded-i64"])

    def test_resealed_source_decision_and_changed_argv_cannot_enter_positive_campaign(self):
        plan = qualification.command_plan("mi300x", ["ck"], Path("temp/source-plan"), 1, [0])
        case = next(row for row in plan if "source_operation" in row)
        for mode in ("source", "semantic", "backend", "modulus", "fingerprint"):
            altered = copy.deepcopy(case)
            if mode == "source":
                altered["source_operation"]["semantics"] = "bounded-u64"
            elif mode == "fingerprint":
                altered["source_operation"]["source_sha256"] = "0" * 64
            elif mode == "modulus":
                altered["argv"] += ["--modulus", "251"]
            else:
                flag = "--semantics" if mode == "semantic" else "--backend"
                altered["argv"][altered["argv"].index(flag) + 1] = (
                    "bounded-u64" if mode == "semantic" else "hip-direct"
                )
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                qualification.validate_source_case(altered)


if __name__ == "__main__":
    unittest.main()
