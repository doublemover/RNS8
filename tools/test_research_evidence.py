"""Archive tampering, index consistency and safe offline restoration checks."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from research_evidence import ROOT, fetch, restore


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name="metadata/result.json", corrupt_member=False):
        payload = b'{"passed":true}\n'
        index = dict(schema="rns8_research_archive_index_v1", source_commit="a" * 40,
            files=[dict(path=name, bytes=len(payload), sha256="0" * 64 if corrupt_member else hashlib.sha256(payload).hexdigest())])
        archive = self.root / "evidence.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            stream.writestr(name, payload)
            stream.writestr("archive-index.json", json.dumps(index))
        data = archive.read_bytes()
        manifest = dict(schema="rns8_research_evidence_archive_v1", source_commit="a" * 40,
            index_member="archive-index.json", bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
            url="https://github.com/doublemover/RNS8/releases/download/test/evidence.zip")
        return archive, manifest, payload

    def test_offline_restore_exact_bytes(self):
        archive, manifest, payload = self.fixture()
        report = restore(archive, self.root / "restored", manifest)
        self.assertEqual(report["files"], 1)
        self.assertEqual((self.root / "restored/metadata/result.json").read_bytes(), payload)

    def test_archive_and_member_tampering_fail_before_writes(self):
        for corrupt_member in (False, True):
            archive, manifest, _ = self.fixture(corrupt_member=corrupt_member)
            if not corrupt_member:
                archive.write_bytes(archive.read_bytes().replace(b"PK", b"XX", 1))
            output = self.root / "restored"
            with self.assertRaises(ValueError):
                restore(archive, output, manifest)
            self.assertFalse(output.exists())

    def test_escaping_paths_and_source_mismatch_fail(self):
        archive, manifest, _ = self.fixture("../escape.json")
        with self.assertRaises(ValueError):
            restore(archive, self.root / "restored", manifest)
        archive, manifest, _ = self.fixture()
        manifest["source_commit"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "source/index"):
            restore(archive, self.root / "restored", manifest)

    def test_download_pinned_identity_and_oversize_response(self):
        archive, manifest, _ = self.fixture()
        data = archive.read_bytes()
        for body, valid in ((data, True), (data + b"extra", False), (b"bad", False)):
            response = io.BytesIO(body)
            response.geturl = lambda: "https://release-assets.githubusercontent.com/example"
            target = self.root / "downloaded.zip"
            target.unlink(missing_ok=True)
            with patch("research_evidence.urlopen", return_value=response):
                if valid:
                    fetch(manifest, target)
                    self.assertEqual(target.read_bytes(), data)
                else:
                    with self.assertRaises(ValueError):
                        fetch(manifest, target)
                    self.assertFalse(target.exists())

    def test_metadata_budget_and_archived_summary_references(self):
        manifest = json.loads((ROOT / "metadata/research-evidence-20261026.json").read_text())
        summaries = []
        for path in (ROOT / "metadata").glob("*.json"):
            self.assertLessEqual(path.stat().st_size, 512 * 1024, str(path))
            value = json.loads(path.read_text())
            if value.get("schema") != "rns8_reviewed_evidence_summary_v1":
                continue
            summaries.append(path)
            reference = value["full_record"]
            self.assertEqual(reference["archive_manifest"], "research-evidence-20261026.json")
            self.assertEqual(reference["member"], path.relative_to(ROOT).as_posix())
            self.assertRegex(reference["sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("original_schema", value)
        self.assertEqual(len(summaries), 21)
        self.assertEqual(manifest["source_commit"], "25f774642a5276285e10931d3f55251358a5d832")


if __name__ == "__main__":
    unittest.main()
