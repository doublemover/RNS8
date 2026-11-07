"""Offline raw-loader→report→ledger→installer controls; no binary execution."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_request import file_digest
from cache_promotion import make_receipt
from compiled_binary_identity import canonical_digest
from target_qualification import (
    EvidenceStore,
    candidate_verdict,
    validate_bound_verdict,
    emit_qualification,
)
from target_validation_report import build_report, capture_target
from promotion_ledger import capture_entry, load_target_validation_candidates
from install_autotune_cache import (
    validate_promotion_ledger_gate,
    AutotuneCacheInstallError,
)
from test_target_qualification_support import QualificationFixture, write


class TargetQualification(unittest.TestCase):
    def test_stale_artifact_is_rejected_before_entering_the_verified_store(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"reviewed artifact")
            expected = file_digest(path)
            path.write_bytes(b"changed artifact")
            store = EvidenceStore()
            with self.assertRaisesRegex(ValueError, "qualification artifact changed:"):
                store.verify(path, expected)
            self.assertNotIn(path.resolve(), store.hashes)
            current = file_digest(path)
            self.assertEqual(store.verify(path, current), path.resolve())
            self.assertEqual(store.verify(path, current), path.resolve())
            with self.assertRaisesRegex(ValueError, "conflicting content identities"):
                store.verify(path, expected)
            path.write_bytes(b"changed again during review")
            with self.assertRaisesRegex(ValueError, "changed during review"):
                store.finish()

    def test_coherent_candidate_through_real_report_ledger_and_installer_boundary(self):
        for target in ("gfx942", "gfx1100"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = QualificationFixture(Path(directory), target=target)
                receipt = fixture.qualification()
                report = build_report([fixture.capture_path], [], [receipt])
                verdict = report["groups"][0]["candidate_qualifications"][0]
                self.assertTrue(verdict["eligible"], verdict)
                self.assertEqual(
                    {item["status"] for item in verdict["receipts"][0]["phases"].values()},
                    {"pass"},
                )
                validate_bound_verdict(verdict, fixture.capture_path, fixture.capture, fixture.row)
                report_path = write(fixture.root / "report.json", report)
                candidates = load_target_validation_candidates([report_path])
                ledger = capture_entry(
                    fixture.capture_path,
                    target_validation_entry=candidates[file_digest(fixture.capture_path)],
                    target_validation_report_supplied=True,
                )
                self.assertTrue(
                    ledger["target_validation_gate_ready"],
                    ledger["target_cache_blockers"],
                )
                # Other promotion gates are independent of this target test.
                ledger.update(promotion_blockers=[], performance_validated=True)
                entry = dict(
                    key=fixture.capture["backend_metadata"]["autotune_key"],
                    selected_backend="hip-direct",
                    selected_kernel=fixture.capture["selected_kernel"],
                    semantic_contract=fixture.capture["semantics"],
                    target_id=target,
                    shape={key: fixture.capture[key] for key in ("m", "n", "k")},
                )
                # This control isolates the target gate. Other gate contents
                # are authored placeholders, not numerical/performance evidence.
                review_control = write(
                    fixture.root / "review-control.json", dict(synthetic_fixture_only=True)
                )
                variance_control = write(
                    fixture.root / "variance-control.json", dict(synthetic_fixture_only=True)
                )
                ledger["cache_promotion_receipt"] = make_receipt(
                    entry,
                    [
                        ("candidate", fixture.capture_path),
                        ("target", report_path),
                        ("review", review_control),
                        ("variance", variance_control),
                    ],
                )
                ledger_path = write(fixture.root / "ledger.json", dict(entries=[ledger]))
                validate_promotion_ledger_gate(
                    [entry], [ledger_path], require_target_validation_gate=True
                )
                entry["selected_kernel"] = "different"
                with self.assertRaises(AutotuneCacheInstallError):
                    validate_promotion_ledger_gate(
                        [entry], [ledger_path], require_target_validation_gate=True
                    )

    def test_historical_pass_cannot_erase_current_failure_or_qualify_an_unrelated_build(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = QualificationFixture(root / "current")
            old = QualificationFixture(root / "old", seed=current.capture["seed"] + 1)
            failed = current.qualification("failed", failing="ctest")
            passed = old.qualification("historical")
            verdict = candidate_verdict(current.binding, current.capture, [passed, failed])
            self.assertFalse(verdict["eligible"])
            self.assertIn("ctest_not_qualified_for_candidate", verdict["blockers"])
            self.assertEqual(
                verdict["excluded_receipts"][0]["reason"], "different_candidate_capture"
            )
            unrelated_only = candidate_verdict(current.binding, current.capture, [passed])
            self.assertFalse(unrelated_only["eligible"])
            repaired = current.qualification("repaired", supersedes=[file_digest(failed)])
            repaired_verdict = candidate_verdict(
                current.binding, current.capture, [passed, failed, repaired]
            )
            self.assertTrue(repaired_verdict["eligible"], repaired_verdict)
            self.assertEqual(repaired_verdict["superseded_receipts"], [file_digest(failed)])
            # A historical failure for another exact candidate is descriptive.
            old_failure = old.qualification("old-failure", failing="smoke")
            self.assertTrue(
                candidate_verdict(
                    current.binding, current.capture, [old_failure, failed, repaired]
                )["eligible"]
            )

    def test_conflicting_current_sessions_require_explicit_supersession(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualificationFixture(Path(directory))
            passing = fixture.qualification("pass")
            failed = fixture.qualification("fail", failing="ctest")
            self.assertFalse(
                candidate_verdict(fixture.binding, fixture.capture, [passing, failed])["eligible"]
            )
            missing = fixture.qualification("partial", missing=("profiler",))
            self.assertFalse(
                candidate_verdict(fixture.binding, fixture.capture, [passing, missing])["eligible"]
            )
            unknown_parent = fixture.qualification("unknown-parent", supersedes=["0" * 64])
            self.assertFalse(
                candidate_verdict(fixture.binding, fixture.capture, [unknown_parent])["eligible"]
            )

    def test_a_new_process_does_not_hide_failure_for_the_same_artifact_and_work(self):
        with tempfile.TemporaryDirectory() as directory:
            first = QualificationFixture(Path(directory) / "first")
            later = QualificationFixture(Path(directory) / "later")
            self.assertEqual(first.binding["identity_sha256"], later.binding["identity_sha256"])
            self.assertNotEqual(first.binding["capture_sha256"], later.binding["capture_sha256"])
            failed = first.qualification("failed", failing="ctest")
            passing = later.qualification("passing")
            self.assertFalse(
                candidate_verdict(later.binding, later.capture, [failed, passing])["eligible"]
            )
            # A historical pass for the same build is also not, by itself, an
            # explicit phase association for a newly supplied capture.
            earlier_pass = first.qualification("earlier-pass")
            self.assertFalse(
                candidate_verdict(later.binding, later.capture, [earlier_pass])["eligible"]
            )
            repaired = later.qualification("requalified", supersedes=[file_digest(failed)])
            self.assertTrue(
                candidate_verdict(later.binding, later.capture, [failed, repaired])["eligible"]
            )

    def test_stale_result_and_forged_ready_snapshot_are_rejected_at_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualificationFixture(Path(directory))
            receipt = fixture.qualification()
            verdict = candidate_verdict(fixture.binding, fixture.capture, [receipt])
            self.assertTrue(verdict["eligible"], verdict)
            changed = copy.deepcopy(verdict)
            changed["candidate"]["identity"]["backend_selected"] = "rocwmma"
            with self.assertRaises(ValueError):
                validate_bound_verdict(changed, fixture.capture_path, fixture.capture, fixture.row)
            results_path = receipt.parent / "results.json"
            raw = json.loads(results_path.read_text(encoding="utf-8"))
            next(row for row in raw["results"] if row["name"] == "ctest")["returncode"] = 1
            write(results_path, raw)
            with self.assertRaises(ValueError):
                validate_bound_verdict(verdict, fixture.capture_path, fixture.capture, fixture.row)
            self.assertFalse(
                candidate_verdict(fixture.binding, fixture.capture, [receipt])["eligible"]
            )

    def test_phase_scope_session_and_malformed_current_receipt_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualificationFixture(Path(directory))
            passed = fixture.qualification("passed")
            original = json.loads(passed.read_text(encoding="utf-8"))
            for index, mutate in enumerate(("backend", "session", "candidate", "seal")):
                bad = copy.deepcopy(original)
                if mutate == "backend":
                    bad["candidate"]["identity"]["backend_selected"] = "rocwmma"
                elif mutate == "session":
                    bad["phases"]["ctest"]["run_id"] = "different-run"
                elif mutate == "candidate":
                    bad["phases"]["ctest"]["candidate_identity_sha256"] = "1" * 64
                else:
                    bad["identity_sha256"] = "2" * 64
                if mutate != "seal":
                    bad["identity_sha256"] = canonical_digest(
                        {key: value for key, value in bad.items() if key != "identity_sha256"}
                    )
                path = write(fixture.root / ("bad-" + str(index) + ".json"), bad)
                verdict = candidate_verdict(fixture.binding, fixture.capture, [passed, path])
                self.assertFalse(verdict["eligible"], mutate)
                self.assertIn(
                    "malformed_or_stale_current_candidate_qualification",
                    verdict["blockers"],
                )

    def test_legacy_platform_status_is_inventory_only(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualificationFixture(Path(directory))
            status = write(
                fixture.root / "old-status.json",
                dict(
                    host_os="linux",
                    target_id="gfx942",
                    rocm_version="7.1",
                    device_name=fixture.capture["device"]["name"],
                    hip_runtime_version=70100000,
                    evidence={
                        phase: "pass"
                        for phase in (
                            "build",
                            "ctest",
                            "smoke",
                            "release_capture",
                            "profiler",
                        )
                    },
                    cache_eligibility=dict(eligible=True, blockers=[]),
                ),
            )
            report = build_report([fixture.capture_path], [status])
            self.assertFalse(report["groups"][0]["cache_eligibility"]["eligible"])
            self.assertFalse(report["groups"][0]["candidate_qualifications"][0]["eligible"])
            legacy_entry = dict(cache_eligibility=dict(eligible=True, blockers=[]))
            ledger = capture_entry(
                fixture.capture_path,
                target_validation_entry=legacy_entry,
                target_validation_report_supplied=True,
            )
            self.assertFalse(ledger["target_validation_gate_ready"])
            self.assertEqual(
                capture_target(fixture.capture_path)["candidate_binding"],
                fixture.binding,
            )

    def test_emitter_adapts_existing_artifacts_and_refuses_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualificationFixture(Path(directory))
            original = fixture.qualification()
            selections = {
                phase: (
                    phase,
                    "identity-"
                    + (
                        "rns8_tests"
                        if phase in ("build", "ctest")
                        else "rns8-verify"
                        if phase == "smoke"
                        else "rns8-bench"
                    ),
                )
                for phase in ("build", "ctest", "smoke", "release_capture", "profiler")
            }
            options = dict(
                capture_path=fixture.capture_path,
                manifest_path=original.parent / "manifest.json",
                results_path=original.parent / "results.json",
                selections=selections,
                output=fixture.root / "emitted.json",
                correctness_stage="qualification",
                dispatch_receipt=fixture.dispatch_path,
            )
            verdict = emit_qualification(**options)
            self.assertTrue(verdict["eligible"], verdict)
            with self.assertRaises(FileExistsError):
                emit_qualification(**options)

    def test_junit_skip_and_stale_native_outputs_fail_even_with_ready_process_status(
        self,
    ):
        for mode in ("empty_ctest", "skipped_gpu_case", "stale_profiler_outputs"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                fixture = QualificationFixture(Path(directory))
                receipt_path = fixture.qualification()
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                results_path = receipt_path.parent / "results.json"
                results = json.loads(results_path.read_text(encoding="utf-8"))
                if mode == "stale_profiler_outputs":
                    next(row for row in results["results"] if row["name"] == "profiler")[
                        "fresh_evidence_outputs"
                    ] = {}
                else:
                    filename = "ctest.xml" if mode == "empty_ctest" else "qualification.xml"
                    path = receipt_path.parent / filename
                    body = (
                        "<testsuite/>"
                        if mode == "empty_ctest"
                        else (
                            '<testsuite><testcase name="qualification pack covers dispatch thresholds and plane tails">'
                            "<skipped/></testcase></testsuite>"
                        )
                    )
                    path.write_text(body, encoding="utf-8")
                    stage = "ctest" if mode == "empty_ctest" else "qualification"
                    next(row for row in results["results"] if row["name"] == stage)["artifacts"][
                        str(path)
                    ] = file_digest(path)
                write(results_path, results)
                for proof in receipt["phases"].values():
                    proof["session_results"]["sha256"] = file_digest(results_path)
                receipt["identity_sha256"] = canonical_digest(
                    {key: value for key, value in receipt.items() if key != "identity_sha256"}
                )
                write(receipt_path, receipt)
                self.assertFalse(
                    candidate_verdict(fixture.binding, fixture.capture, [receipt_path])["eligible"]
                )


if __name__ == "__main__":
    unittest.main()
