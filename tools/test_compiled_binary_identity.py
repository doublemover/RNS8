"""Authored binary-identity and collector controls; never executes a binary."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_request import file_digest
from compiled_binary_identity import SCHEMA, validate_document, bind_document, validate_receipt
from test_capture_identity_support import synthetic_identity
import gpu_qualification as qualification


def document():
    capture = {}
    synthetic_identity(capture)
    return dict(schema=SCHEMA, build_manifest=capture["build_manifest"],
                library_build_manifest=copy.deepcopy(capture["build_manifest"]),
                library_abi_declaration_sha256="e"*64)


class CompiledBinaryIdentity(unittest.TestCase):
    def test_manifests_must_be_real_complete_and_equal(self):
        value = document()
        self.assertEqual(validate_document(value)["build_manifest_sha256"],
                         validate_document(value)["library_build_manifest_sha256"])
        for mode in ("missing", "empty", "different_library", "dirty", "bad_declaration", "wrong_schema"):
            broken = copy.deepcopy(value)
            if mode == "missing":
                broken.pop("build_manifest")
            elif mode == "empty":
                broken["library_build_manifest"] = {}
            elif mode == "different_library":
                broken["library_build_manifest"]["source_digest"] = "f"*64
            elif mode == "dirty":
                broken["build_manifest"]["source_dirty"] = True
            elif mode == "bad_declaration":
                broken["library_abi_declaration_sha256"] = "not-a-digest"
            else:
                broken["schema"] = "old"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_document(broken)

    def test_receipt_reloads_actual_files_and_rejects_rewritten_summary(self):
        with tempfile.TemporaryDirectory(prefix="identity space \u03bb ") as directory:
            root = Path(directory)
            binary = root / "authored binary"
            binary.write_bytes(b"fixture only; not executable")
            raw = root / "identity.json"
            raw.write_text(json.dumps(document()), encoding="utf-8")
            value = bind_document(binary, raw, before_sha256=file_digest(binary))
            self.assertEqual(validate_receipt(value), value)
            changed = copy.deepcopy(value)
            changed["library_build_manifest_sha256"] = "a"*64
            with self.assertRaises(ValueError):
                validate_receipt(changed)
            before = file_digest(binary)
            binary.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                bind_document(binary, raw, before_sha256=before)
            with self.assertRaises(ValueError):
                validate_receipt(value)
            binary.write_bytes(b"fixture only; not executable")
            raw.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_receipt(value)

    def test_session_identity_checks_happen_before_any_runtime_probe(self):
        for target in qualification.TARGETS:
            plan = qualification.command_plan(target, ["hip-direct"], Path("temp/fixture"), 2, [0])
            probes = [index for index, item in enumerate(plan) if "inspect_target" in item]
            identity = [index for index, item in enumerate(plan) if item.get("compiled_identity")]
            self.assertEqual(len(identity), 4)
            self.assertLess(max(identity), min(probes))
            for index in identity:
                item = plan[index]
                self.assertEqual(item["argv"][-1], "--build-identity")
                self.assertIsNone(item["device"])
                self.assertEqual(item["identity_backends"], ["hip-direct"])
            item = plan[identity[0]]
            passed = dict(returncode=0, timed_out=False, stdout=json.dumps(document()))
            self.assertEqual(qualification.validate_result(item, passed), [])
            self.assertTrue(qualification.validate_result(item, {**passed, "stdout": "{}"}))
            self.assertTrue(qualification.validate_result(item, {**passed, "returncode": 2}))


if __name__ == "__main__":
    unittest.main()
