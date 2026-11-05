"""Authored LLVM records; source and replacements are never executed."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from analysis_diagnostics import diagnostic_document, read_diagnostics

HAS_DECODER = (importlib.util.find_spec("ruamel") is not None
               and importlib.util.find_spec("ruamel.yaml") is not None)


class AnalysisDiagnostics(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "first party λ.cpp"
        self.original = "// λ\r\nreturn *pointer;\r\n".encode("utf-8")
        self.source.write_bytes(self.original)
        self.document = dict(MainSourceFile=str(self.source), Diagnostics=[dict(
            DiagnosticName="clang-analyzer-core.NullDereference", Level="Warning",
            BuildDirectory=str(self.root), DiagnosticMessage=dict(Message="authored null diagnostic",
                FilePath=str(self.source), FileOffset=self.original.index(b"return"),
                Replacements=[dict(FilePath=str(self.source), Offset=0, Length=2, ReplacementText="never applied")]),
            Notes=[])])

    def read(self, value=None):
        return diagnostic_document(self.document if value is None else value,
                                   source=self.source, directory=self.root)

    def test_native_bytes_preserve_crlf_unicode_and_unapplied_fixes(self):
        facts = self.read()
        location = facts[0]["message"]["location"]
        self.assertEqual((location["line"], location["column_byte"]), (2, 1))
        self.assertEqual(location["byte_offset"], len("// λ\r\n".encode("utf-8")))
        self.assertEqual(facts[0]["message"]["replacements"][0]["text"], "never applied")
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertFalse(facts[0]["parse_failure"])
        self.assertEqual(len(facts[0]["identity_sha256"]), 64)

    def test_parse_errors_and_malformed_foreign_records_cannot_be_clean(self):
        changed = copy.deepcopy(self.document)
        changed["Diagnostics"][0].update(DiagnosticName="clang-diagnostic-error", Level="Error")
        self.assertTrue(self.read(changed)[0]["parse_failure"])
        for mutation in ("offset_bool", "offset_past_file", "unknown_level", "unknown_field", "foreign_source"):
            value = copy.deepcopy(self.document)
            row = value["Diagnostics"][0]
            if mutation == "offset_bool":
                row["DiagnosticMessage"]["FileOffset"] = True
            elif mutation == "offset_past_file":
                row["DiagnosticMessage"]["FileOffset"] = len(self.original) + 1
            elif mutation == "unknown_level":
                row["Level"] = "Note"
            elif mutation == "unknown_field":
                row["PretendSuccess"] = True
            else:
                value["MainSourceFile"] = str(self.root / "another.cpp")
            with self.assertRaises(ValueError):
                self.read(value)

    def test_missing_location_remains_unknown(self):
        value = copy.deepcopy(self.document)
        value["Diagnostics"][0]["DiagnosticMessage"].pop("FilePath")
        value["Diagnostics"][0]["DiagnosticMessage"].pop("FileOffset")
        location = self.read(value)[0]["message"]["location"]
        self.assertEqual(location["state"], "unlocated")
        self.assertIsNone(location["path"])

    @unittest.skipUnless(HAS_DECODER, "optional locked analyzer decoder not installed")
    def test_yaml_duplicates_aliases_unsafe_tags_and_depth_are_rejected(self):
        path = self.root / "diagnostics.yaml"
        path.write_text(json.dumps(self.document), encoding="utf-8")
        self.assertEqual(read_diagnostics(path, source=self.source, directory=self.root), self.read())
        for text in ("MainSourceFile: a\nMainSourceFile: b\nDiagnostics: []\n",
                     "MainSourceFile: &anchor x\nDiagnostics: [*anchor]\n",
                     "!!python/object/apply:os.system ['must never execute']",
                     "MainSourceFile: source.cpp\nDiagnostics: " + "[" * 66 + "]" * 66):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_diagnostics(path, source=self.source, directory=self.root)


if __name__ == "__main__":
    unittest.main()
