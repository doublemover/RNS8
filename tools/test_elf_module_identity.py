"""Synthetic ELF custody controls; no compiler, loader or profiler execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import elf_module_identity as elf


def note(value):
    data = struct.pack("<III", 4, len(value), 3) + b"GNU\0" + value
    return data + bytes((-len(data)) % 4)


def image(value=b"\x01\x23\xab\xcd", *, machine=62, sections=False, extended=False, extra=b""):
    payload = note(value) + extra
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    if sections:
        header = struct.pack(
            "<16sHHIQQQIHHHHHH",
            ident,
            3,
            machine,
            1,
            0,
            0,
            64,
            0,
            64,
            0,
            0,
            64,
            0 if extended else 2,
            0,
        )
        null = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 2 if extended else 0, 0, 0, 0, 0)
        entry = struct.pack("<IIQQQQIIQQ", 0, 7, 0, 0, 192, len(payload), 0, 0, 4, 0)
        return header + null + entry + payload
    header = struct.pack("<16sHHIQQQIHHHHHH", ident, 3, machine, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    program = struct.pack("<IIQQQQQQ", 4, 0, 120, 0, 0, len(payload), len(payload), 4)
    return header + program + payload


class ElfCustodyControls(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.module = self.root / "inert.so"

    def test_native_program_section_and_extended_note_layouts_bind_content(self):
        for sections, extended in ((False, False), (True, False), (True, True)):
            for machine in (62, 183):
                raw = image(sections=sections, extended=extended, machine=machine)
                self.module.write_bytes(raw)
                value = elf.identity(self.module)
                self.assertEqual(value["gnu_build_id"], "0123abcd")
                self.assertEqual(value["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(value["machine"], machine)
                self.assertFalse(value["build_id_is_content_checksum"])

    def test_note_identity_is_not_a_content_checksum(self):
        self.module.write_bytes(image())
        first = elf.identity(self.module)
        self.module.write_bytes(image() + b"changed non-note payload")
        second = elf.identity(self.module)
        self.assertEqual(first["gnu_build_id"], second["gnu_build_id"])
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.module.write_bytes(image(extra=note(b"different")))
        with self.assertRaisesRegex(ValueError, "contradictory"):
            elf.identity(self.module)

    def test_truncated_unknown_and_overbudget_modules_never_become_matched(self):
        for raw in (
            image()[:63],
            image().replace(b"\x7fELF\x02", b"\x7fELF\x01", 1),
            image(machine=224),
            image()[:-1],
            image(b""),
        ):
            self.module.write_bytes(raw)
            with self.assertRaises(ValueError):
                elf.identity(self.module)
        self.module.write_bytes(image())
        with self.assertRaises(elf.ModuleBudgetError):
            elf.identity(self.module, max_bytes=64)
        raw = bytearray(image())
        struct.pack_into("<Q", raw, 96, 1 << 63)  # PT_NOTE file extent.
        self.module.write_bytes(raw)
        with self.assertRaises(ValueError):
            elf.identity(self.module)

    def test_leaf_module_states_keep_missing_mismatch_and_source_scope_separate(self):
        self.module.write_bytes(image())
        sample = dict(
            dso_path=str(self.module),
            dso_build_id="0123ABCD",
            symbol="function",
            source=dict(path="unverified/source.cpp", line=1),
        )
        value = elf.module_inventory([sample, sample], machine=62)
        self.assertTrue(value["complete"])
        self.assertEqual(value["matched_build_id_samples"], 2)
        self.assertEqual(len(value["modules"]), 1)
        self.assertFalse(value["loaded_image_history_verified"])
        changed = {**sample, "dso_build_id": "abcd"}
        missing = {**sample, "dso_path": None}
        value = elf.module_inventory([sample, changed, missing], machine=62)
        self.assertFalse(value["complete"])
        self.assertEqual(
            [row["state"] for row in value["modules"]], ["matched", "mismatch", "unresolved"]
        )
        self.assertEqual(value["module_file_byte_budget_charged"], self.module.stat().st_size)
        self.assertEqual(
            elf.module_inventory([sample], machine=183)["modules"][0]["state"], "mismatch"
        )
        for options in (dict(max_modules=1), dict(max_total_bytes=64), dict(max_module_bytes=64)):
            with self.assertRaises(ValueError):
                elf.module_inventory([sample, changed], machine=62, **options)

    def test_an_earlier_module_cannot_change_while_later_modules_are_inspected(self):
        self.module.write_bytes(image())
        other = self.root / "other.so"
        other.write_bytes(image())
        original = elf.identity

        def inspect(path, **options):
            value = original(path, **options)
            if path == other:
                self.module.write_bytes(image() + b"changed")
            return value

        samples = [
            dict(dso_path=str(path), dso_build_id="0123abcd") for path in (self.module, other)
        ]
        with patch.object(elf, "identity", side_effect=inspect):
            with self.assertRaises(elf.ModuleChangedError):
                elf.module_inventory(samples, machine=62)


if __name__ == "__main__":
    unittest.main()
