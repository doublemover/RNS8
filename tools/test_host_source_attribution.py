"""Authored current-source custody controls, separate from compiled-source proof."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import host_source_attribution as source


class CurrentSource(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name).resolve()
        self.file = self.root / "current.cpp"
        self.data = b"// header\nreturn value;\n// final unterminated"
        self.file.write_bytes(self.data)

    def row(self, *, text="return value;", line=2, path=None):
        return dict(
            source=dict(path=str(self.file) if path is None else path, line=line, text=text),
            callchain=[],
        )

    def test_leaf_and_callchain_line_observations_share_bounded_current_file_identity(self):
        row = self.row()
        row["callchain"] = [self.row(text="// final unterminated", line=3)]
        report = source.source_inventory([row, row], root=self.root)
        self.assertEqual(len(report["files"]), 1)
        self.assertEqual(report["files"][0]["sha256"], hashlib.sha256(self.data).hexdigest())
        self.assertEqual(report["current_source_bytes"], len(self.data))
        self.assertEqual(report["selected_frame_observations"], 4)
        self.assertTrue(
            all(
                row["state"] == "native_text_matches_current_source_line"
                for row in report["counts"]
            )
        )
        self.assertFalse(report["compiled_source_contents_verified"])

    def test_missing_mismatched_and_external_sources_remain_distinct(self):
        rows = [
            self.row(text="wrong"),
            self.row(text=None),
            self.row(line=99),
            self.row(path="relative.cpp"),
            dict(source=None, callchain=[]),
        ]
        report = source.source_inventory(rows, root=self.root)
        self.assertEqual(len(report["counts"]), 5)
        self.assertFalse(report["compiled_source_contents_verified"])
        other = self.root / "child"
        other.mkdir()
        report = source.source_inventory([self.row()], root=other)
        self.assertEqual(report["files"], [])
        self.assertEqual(report["counts"][0]["state"], "outside_first_party_source_root")

    def test_source_byte_file_and_bookend_limits_fail_closed(self):
        for name, limit in (("MAX_FILE_BYTES", 2), ("MAX_TOTAL_BYTES", 2), ("MAX_FILES", 0)):
            with patch.object(source, name, limit), self.assertRaisesRegex(ValueError, "budget"):
                source.source_inventory([self.row()], root=self.root)
        with patch.object(source, "file_digest", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "source changed"):
                source.source_inventory([self.row()], root=self.root)
        for bad in (self.row(line=True), self.row(text=7), dict(source=[], callchain=[])):
            with self.assertRaises(ValueError):
                source.source_inventory([bad], root=self.root)


if __name__ == "__main__":
    unittest.main()
