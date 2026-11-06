"""Authored disposition protocol controls; native receipt acceptance is mocked."""

from __future__ import annotations

import copy
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analysis_dispositions as gate
from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest


class Dispositions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "temp").mkdir()
        self.source = self.root / "src/value.cpp"
        self.source.write_bytes(
            b"// context\n" * 40 + b"int value() { return 0; }\n" + b"// trailing context\n" * 30
        )
        self.evidence = self.root / "tests/control.cpp"
        self.evidence.write_bytes(b"// synthetic source-invariant fixture, not qualification\n")
        self.config = self.root / ".clang-tidy"
        self.config.write_text("Checks: '-*,clang-analyzer-core.*'\n")
        self.policy = self.root / "policy.json"
        self.receipt = self.root / "temp/results.json"
        self.receipt.write_text("{}")
        self.full = self.root / "temp/pass.json"
        self.diag = dict(
            check="clang-analyzer-core.NullDereference",
            level="Warning",
            parse_failure=False,
            role="program_diagnostic",
            message=dict(
                message="synthetic finding",
                location=dict(
                    path=str(self.source), state="located", byte_offset=40 * len(b"// context\n")
                ),
            ),
        )
        self.detail = dict(
            context="a" * 64, stage="host-analyzer", target=None, diagnostics=[self.diag]
        )
        self.full.write_text(json.dumps(self.detail))
        self.result = dict(
            status="findings_require_triage",
            unattempted_passes=0,
            analyzed_passes=1,
            selected_passes=1,
            results=[
                dict(
                    identity="b" * 64,
                    receipt=dict(path=str(self.full), sha256=file_digest(self.full)),
                )
            ],
            source_artifacts={str(self.source): file_digest(self.source)},
            retained_artifacts={str(self.full): file_digest(self.full)},
        )
        self.root_patch = patch.object(gate, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.native = patch.object(gate, "verify_receipt", return_value=self.result)
        self.native.start()
        self.addCleanup(self.native.stop)
        self.write_policy([])

    def write_policy(self, entries):
        self.policy.write_text(
            json.dumps(
                dict(
                    schema="rns8_analysis_dispositions_v1",
                    owner_issue=99,
                    tool_version=gate.SUPPORTED_TOOL,
                    config_sha256=file_digest(self.config),
                    entries=entries,
                )
            )
        )

    def entry(self):
        signature = gate.signature_for(self.diag, self.detail)
        return dict(
            signature,
            id=canonical_digest(signature),
            classification="tool_false_positive",
            owner="fixture-owner",
            reason="Synthetic narrow source invariant; no real false positive is asserted.",
            upstream=None,
            revisit_before="2040-01-01",
            evidence=dict(path="tests/control.cpp", sha256=file_digest(self.evidence)),
        )

    def evaluate(self):
        return gate.evaluate(self.receipt, self.policy, today=date(2026, 11, 6))

    def test_empty_policy_exposes_new_findings_and_reviewed_entry_is_exact(self):
        report = self.evaluate()
        self.assertEqual(report["status"], "unclassified_findings")
        self.assertEqual(len(report["unclassified"]), 1)
        self.write_policy([self.entry()])
        report = self.evaluate()
        self.assertEqual(report["status"], "reviewed_findings_no_new")
        self.assertFalse(report["hardware_qualification"])
        self.assertFalse(report["performance_promotion_allowed"])
        self.result["status"] = "syntax_only_not_analyzer_coverage"
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_reviewer_expiry_evidence_check_and_policy_cannot_mask_errors(self):
        entry = self.entry()
        for field, value in (
            ("classification", "reachable_defect"),
            ("owner", ""),
            ("reason", ""),
            ("revisit_before", "2026-11-06"),
            ("level", "Error"),
            ("check", "*"),
            ("offset_in_anchor", True),
            ("path", "src/*.cpp"),
            ("anchor_hex", "00"),
            ("context", "*"),
        ):
            bad = dict(entry, **{field: value})
            signature = {key: bad[key] for key in gate.SIGNATURE_FIELDS}
            bad["id"] = canonical_digest(signature)
            self.write_policy([bad])
            with self.subTest(field=field), self.assertRaises((ValueError, OSError)):
                self.evaluate()
        self.write_policy([entry, entry])
        with self.assertRaises(ValueError):
            self.evaluate()
        self.write_policy([entry])
        self.evidence.write_bytes(b"changed invariant")
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_line_churn_outside_anchor_does_not_erase_or_widen_a_disposition(self):
        signature = gate.signature_for(self.diag, self.detail)
        raw = self.source.read_bytes()
        self.source.write_bytes(b"// unrelated new line\n" + raw)
        moved = copy.deepcopy(self.diag)
        moved["message"]["location"]["byte_offset"] += len(b"// unrelated new line\n")
        self.assertEqual(gate.signature_for(moved, self.detail), signature)
        moved["message"]["message"] = "another finding at the same location"
        self.assertNotEqual(gate.signature_for(moved, self.detail), signature)
        moved["message"]["message"] = self.diag["message"]["message"]
        for changed in (
            dict(self.detail, context="c" * 64),
            dict(self.detail, stage="hip-device-analyzer", target="gfx942"),
        ):
            self.assertNotEqual(gate.signature_for(moved, changed), signature)

    def test_disappeared_findings_and_changed_supporting_files_need_review(self):
        self.write_policy([self.entry()])
        self.detail["diagnostics"] = []
        self.full.write_text(json.dumps(self.detail))
        self.result["retained_artifacts"][str(self.full)] = file_digest(self.full)
        self.result["status"] = "selected_static_contexts_clean"
        report = self.evaluate()
        self.assertEqual(report["status"], "stale_dispositions")
        self.assertEqual(len(report["stale_dispositions"]), 1)
        self.config.write_text("changed configuration")
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_external_or_unlocated_diagnostic_never_gets_dependency_wide_waiver(self):
        for location in (
            dict(path=str(self.root.parent / "external.hpp"), state="located", byte_offset=0),
            dict(path=None, state="unlocated", byte_offset=0),
        ):
            changed = copy.deepcopy(self.diag)
            changed["message"]["location"] = location
            with self.assertRaises((ValueError, OSError)):
                gate.signature_for(changed, self.detail)


if __name__ == "__main__":
    unittest.main()
