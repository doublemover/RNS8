"""Authored real-layout PE/MSF identity controls; no loader or symbolizer runs."""

from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest
import uuid
from unittest.mock import patch

import pe_symbol_identity as pe

GUID = uuid.UUID("e93dca29-156c-43bb-b53b-280622752b35")


def executable(*, guid=GUID, age=1, machine=0x8664):
    data = bytearray(1024)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 132, machine, 1, 0, 0, 0, 240, 2)
    struct.pack_into("<H", data, 152, 0x20B)
    struct.pack_into("<I", data, 128 + 8, 456)
    struct.pack_into("<I", data, 152 + 56, 8192)
    struct.pack_into("<I", data, 152 + 64, 123)
    struct.pack_into("<I", data, 152 + 108, 16)
    struct.pack_into("<II", data, 152 + 160, 4096, 28)
    struct.pack_into("<IIII", data, 392 + 8, 512, 4096, 512, 512)
    rsds = b"RSDS" + guid.bytes_le + struct.pack("<I", age) + b"authored.pdb\0"
    struct.pack_into("<IIHHIIII", data, 512, 0, 0, 0, 0, 2, len(rsds), 4096 + 28, 540)
    data[540 : 540 + len(rsds)] = rsds
    return data


def symbols(*, guid=GUID, age=1):
    data = bytearray(4096)
    data[:32] = pe.MSF_MAGIC
    struct.pack_into("<6I", data, 32, 512, 1, 8, 16, 0, 3)
    struct.pack_into("<I", data, 3 * 512, 4)
    struct.pack_into("<4I", data, 4 * 512, 2, 0, 28, 5)
    struct.pack_into("<III16s", data, 5 * 512, 20000404, 123, age, guid.bytes_le)
    return data


class SymbolIdentity(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.binary, self.pdb = self.root / "app.exe", self.root / "app.pdb"
        self.binary.write_bytes(executable())
        self.pdb.write_bytes(symbols())

    def test_exact_guid_age_and_full_file_identity_are_separate_from_symbol_resolution(self):
        value = pe.symbol_pair(self.binary, self.pdb)
        self.assertTrue(value["identity_match"])
        self.assertEqual(value["symbols"]["guid"], str(GUID))
        self.assertFalse(value["native_symbol_resolution_observed"])
        self.assertFalse(value["source_contents_verified"])
        self.assertEqual(value["executable"]["recorded_pdb_path"], "authored.pdb")
        self.assertEqual(value["executable"]["image_size"], 8192)
        self.assertEqual(value["executable"]["image_timestamp"], 456)
        self.assertEqual(value["executable"]["image_checksum"], 123)
        for guid, age in ((GUID, 2), (uuid.UUID(int=7), 1)):
            self.pdb.write_bytes(symbols(guid=guid, age=age))
            with self.assertRaisesRegex(ValueError, "GUID/age"):
                pe.symbol_pair(self.binary, self.pdb)

    def test_pe_extent_ambiguity_legacy_codeview_and_wrong_machine_are_rejected(self):
        for offset, value in (
            (540, b"NB10"),
            (512 + 24, struct.pack("<I", 541)),
            (152 + 160, struct.pack("<I", 99999)),
            (132, struct.pack("<H", 0x14C)),
        ):
            data = executable()
            data[offset : offset + len(value)] = value
            self.binary.write_bytes(data)
            with self.assertRaises(ValueError):
                pe.pe_identity(self.binary)
        self.binary.write_bytes(executable(machine=0xAA64))
        self.assertEqual(pe.pe_identity(self.binary)["machine"], 0xAA64)

    def test_pdb_block_aliases_extents_missing_info_and_wrong_version_fail_closed(self):
        for offset, value in (
            (3 * 512, 3),
            (4 * 512 + 12, 4),
            (4 * 512 + 12, 8),
            (4 * 512 + 8, 0xFFFFFFFF),
            (5 * 512, 1),
            (44, 17),
        ):
            data = symbols()
            struct.pack_into("<I", data, offset, value)
            self.pdb.write_bytes(data)
            with self.assertRaises(ValueError):
                pe.pdb_identity(self.pdb)
        self.pdb.write_bytes(symbols()[:-1])
        with self.assertRaisesRegex(ValueError, "file/directory"):
            pe.pdb_identity(self.pdb)

    def test_budgets_and_hash_bookends_are_mandatory(self):
        with self.assertRaises(ValueError):
            pe.pdb_identity(self.pdb, max_bytes=True)
        with patch.object(pe.Image, "hash", side_effect=["a" * 64, "b" * 64]):
            with self.assertRaisesRegex(ValueError, "content changed"):
                pe.pe_identity(self.binary)


if __name__ == "__main__":
    unittest.main()
