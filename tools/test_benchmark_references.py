"""Exact reference fixture identity, corruption, bounded restore and collector tests."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest

from benchmark_references import DEFAULT_INDEX, digest, pack, prepare, validate_reference
from benchmark_schema.public_timing import validate_native_reference
from collect_public_calibration import commands, reference_cache_supported


class ReferenceTests(unittest.TestCase):
    def fixture(self, root):
        key = "a" * 64
        payload = struct.pack("<4q", 41, -14, -17, 82)
        data = b"RNS8REF1" + key.encode() + digest(payload).encode() + struct.pack("<Q", 4) + payload
        raw, authored = root / "raw", root / "authored"
        raw.mkdir()
        (raw / (key + ".bin")).write_bytes(data)
        provenance = root / "provenance.json"
        provenance.write_text(json.dumps(dict(oracle="boost_cpp_int_full_integer_reference", cases=[dict(key=key)])))
        pack(raw, authored, provenance)
        return key, data, authored / "index.json"

    def test_complete_fixture_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key, data, index = self.fixture(root)
            report = prepare(index, root / "restored")
            self.assertEqual(report["fixtures"], 1)
            self.assertEqual((root / "restored" / (key + ".bin")).read_bytes(), data)
            compressed = index.parent / (key + ".bin.gz")
            compressed.write_bytes(compressed.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                prepare(index, root / "rejected")
            self.assertFalse((root / "rejected").exists())
            with self.assertRaises(ValueError):
                validate_reference(data[:-1], key)
            with self.assertRaises(ValueError):
                validate_reference(data[:144] + b"\0" * 32, key)

    def test_expansion_budget_and_unsafe_keys_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key, _, index = self.fixture(root)
            compressed = gzip.compress(b"x" * 100000, mtime=0)
            (index.parent / (key + ".bin.gz")).write_bytes(compressed)
            document = json.loads(index.read_text())
            document["fixtures"][0].update(compressed_bytes=len(compressed), compressed_sha256=digest(compressed))
            index.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                prepare(index, root / "rejected")
            document["fixtures"][0]["key"] = "../escape"
            index.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                prepare(index, root / "rejected")
            self.assertFalse((root / "rejected").exists())

    def test_repository_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            report = prepare(DEFAULT_INDEX, Path(directory))
            self.assertGreaterEqual(report["fixtures"], 3)

    def test_receipts_require_full_comparison_and_content_identity(self):
        capture = dict(m=2, n=2, native_reference_verification=dict(method="cached_full_integer_reference",
            cache_key="a" * 64, payload_sha256="b" * 64, checked_cells=4, elapsed_us=3,
            inside_production_interval=False))
        self.assertEqual(validate_native_reference(capture), [])
        for field, value in (("checked_cells", 1), ("cache_key", "bad"), ("inside_production_interval", True)):
            changed = deepcopy(capture)
            changed["native_reference_verification"][field] = value
            self.assertTrue(validate_native_reference(changed))

    def test_collector_reuses_without_populating_and_rejects_control_override(self):
        cases = [dict(name="ordinary", arguments=["--backend", "hip-direct", "--semantics", "bounded-i64"])]
        plan = commands(Path("bench"), cases, 2, 7, Path("out"), reference_directory=Path("cache"))
        self.assertTrue(all("--reference-cache-dir" in entry["command"] for entry in plan))
        self.assertTrue(all("--populate-reference-cache" not in entry["command"] for entry in plan))
        fresh = commands(Path("bench"), cases, 2, 7, Path("out"), verify_native_output=True)
        self.assertTrue(all("--verify-native-output" in entry["command"] for entry in fresh))
        for arguments in (["--semantics", "wrap-u64"], ["--host-api-batch-size", "2"], ["--force-native-to-rns-bridge"]):
            self.assertFalse(reference_cache_supported(arguments))
        with self.assertRaises(ValueError):
            commands(Path("bench"), [dict(name="override", arguments=["--reference-cache-dir", "elsewhere"])], 2, 7, Path("out"))


if __name__ == "__main__":
    unittest.main()
