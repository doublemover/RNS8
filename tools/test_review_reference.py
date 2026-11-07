"""Producer-to-consumer numerical review contracts using synthetic child records."""

import json
from pathlib import Path
import tempfile
import unittest

import error_detection_policy_report
import verification_amortization_report
from review_contract import REVIEW_SCHEMA_VERSION
from test_error_detection_policy_report import error_detecting_capture
from test_verification_amortization_report import amortized_capture
from test_review_reference_support import write_policy_review

ROOT = Path(__file__).resolve().parents[1]


class ReviewReferenceTest(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "temp"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="review-binding-", dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def case(self, name, factory, mutate=None):
        root = self.root / name
        root.mkdir()
        capture = factory()
        if mutate:
            mutate(capture)
        path = root / "candidate.json"
        path.write_text(json.dumps(capture), encoding="utf-8")
        write_policy_review(root / "review_report.json", path)
        return root, path, capture

    def variants(self):
        return [
            (
                "amortized",
                amortized_capture,
                verification_amortization_report.build_report,
                "verification_amortization",
            ),
            (
                "error",
                error_detecting_capture,
                error_detection_policy_report.build_report,
                "error_detection_policy",
            ),
        ]

    def test_actual_current_producer_preserves_nonpromotion_and_completed_comparison(self):
        for name, factory, build, policy in self.variants():
            with self.subTest(policy=policy):
                root, _, _ = self.case(name, factory)
                report = build([root])
                (row,) = report["rows"]
                self.assertTrue(row["policy_metadata_valid"])
                self.assertTrue(row["reference_coverage_present"])
                self.assertTrue(row["comparison_passed"])
                self.assertTrue(row["promotion_prohibited"])
                self.assertTrue(row["ready"])
                sidecar = json.loads((root / "review_report.json").read_text(encoding="utf-8"))
                self.assertEqual(sidecar["schema_version"], REVIEW_SCHEMA_VERSION)
                self.assertTrue(
                    any(
                        candidate["promotable"] is False
                        for group in sidecar["groups"]
                        for candidate in group["candidates"]
                    )
                )

    def test_actual_producer_mismatch_missing_coverage_and_pending_status_never_complete(self):
        for name, factory, build, policy in self.variants():
            for failure in ("mismatch", "missing", "pending"):
                with self.subTest(policy=policy, failure=failure):

                    def mutate(capture, *, failure=failure, policy=policy):
                        if failure == "mismatch":
                            capture["checksum_u64"] ^= 1
                        elif failure == "missing":
                            capture.pop("native_reference_verification", None)
                            capture[policy]["final_exact_comparison_status"] = (
                                "checksum_recorded_reference_required"
                            )
                        else:
                            capture[policy]["final_exact_comparison_status"] = "reference_required"

                    root, _, _ = self.case(name + "-" + failure, factory, mutate)
                    (row,) = build([root])["rows"]
                    self.assertTrue(row["policy_metadata_valid"])
                    self.assertFalse(row["comparison_passed"])
                    self.assertFalse(row["ready"])
                    if failure == "mismatch":
                        self.assertTrue(any("checksum" in item for item in row["blockers"]))
                    if failure == "missing":
                        self.assertFalse(row["reference_coverage_present"])
                    if failure == "pending":
                        self.assertIn("final_exact_comparison_pending", row["blockers"])

    def test_stale_candidate_reference_schema_and_conflicting_sidecars_fail_closed(self):
        for name, factory, build, policy in self.variants():
            for failure in ("candidate", "reference", "schema", "conflict"):
                with self.subTest(policy=policy, failure=failure):
                    root, path, capture = self.case(name + "-" + failure, factory)
                    inputs = [root]
                    if failure == "candidate":
                        capture["checksum_u64"] ^= 1
                        path.write_text(json.dumps(capture), encoding="utf-8")
                    elif failure == "reference":
                        cpu = root / "cpu-reference.json"
                        data = json.loads(cpu.read_text(encoding="utf-8"))
                        data["checksum_u64"] ^= 1
                        cpu.write_text(json.dumps(data), encoding="utf-8")
                    else:
                        report = json.loads(
                            (root / "review_report.json").read_text(encoding="utf-8")
                        )
                        if failure == "schema":
                            report["schema_version"] = 1
                            (root / "review_report.json").write_text(
                                json.dumps(report), encoding="utf-8"
                            )
                        else:
                            for group in report["groups"]:
                                for candidate in group["candidates"]:
                                    if candidate["capture"] == str(path):
                                        candidate["checksum_matches_reference"] = False
                            conflict = self.root / (name + "-conflicting")
                            conflict.mkdir()
                            (conflict / "review_report.json").write_text(
                                json.dumps(report), encoding="utf-8"
                            )
                            inputs.append(conflict)
                    (row,) = build(inputs)["rows"]
                    self.assertFalse(row["ready"])
                    self.assertFalse(row["comparison_passed"])
                    if failure == "candidate":
                        self.assertIn("review_capture_content_mismatch", row["blockers"])
                    if failure == "reference":
                        self.assertIn("review_cpu_reference_binding_missing", row["blockers"])
                    if failure == "schema":
                        self.assertIn("review_schema_version_mismatch", row["blockers"])
                    if failure == "conflict":
                        self.assertIn("review_sidecar_conflict", row["blockers"])


if __name__ == "__main__":
    unittest.main()
