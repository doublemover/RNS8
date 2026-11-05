"""Exercise raw session-to-report identity boundaries using authored files only."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from compiled_binary_identity import canonical_digest
from multigpu_observation import emit_session_bindings, make_binding, validate_binding
from multigpu_shard_report import (
    build_report,
    report_has_critical_failures,
    write_outputs,
)
from test_multigpu_observation_support import ObservationFixture
from test_target_qualification_support import write, reference


class MultiGPUObservations(unittest.TestCase):
    def test_four_rank_two_case_receipts_use_actual_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ObservationFixture(Path(temporary))
            receipts = fixture.receipts()
            report = build_report([], observation_receipts=receipts)
            self.assertEqual(report["qualified_observation_count"], 8)
            self.assertEqual(report["missing_ranks"], [])
            self.assertEqual(report["missing_cases"], [])
            self.assertEqual(len(report["checksum_groups"]), 2)
            self.assertFalse(report_has_critical_failures(report))
            self.assertFalse(report["promotion_eligible"])
            for row in report["rows"]:
                self.assertEqual(
                    row["physical_device_id"], fixture.devices[row["rank"]]
                )
                self.assertEqual(row["observed_runtime_ordinal"], 0)
                self.assertEqual(row["target_arch"], "gfx942")
                self.assertEqual(
                    row["device_bdf"], f"0000:{fixture.devices[row['rank']]:02x}:00.0"
                )
                self.assertEqual(row["observation_binding_status"], "verified")
            emission = emit_session_bindings(
                fixture.manifest_path, fixture.results_path, fixture.root / "emitted"
            )
            self.assertTrue(emission["complete"])
            self.assertEqual(emission["emitted"], 8)
            # Escape every metadata cell, including names read from external JSON.
            report["rows"][0]["device_name"] = "name | extra\nrow"
            markdown = Path(
                write_outputs(report, fixture.root / "report")["markdown"]
            ).read_text()
            self.assertIn("name &#124; extra<br>row", markdown)

    def test_missing_receipt_retains_all_requested_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ObservationFixture(Path(temporary))
            report = build_report([], observation_receipts=fixture.receipts()[:-1])
            self.assertTrue(report_has_critical_failures(report))
            self.assertIn(dict(rank=3, case=fixture.cases[-1]), report["missing_cases"])

    def test_distinct_finite_modulus_cases_do_not_merge_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ObservationFixture(Path(temporary), finite=True)
            report = build_report([], observation_receipts=fixture.receipts())
            self.assertEqual(report["qualified_observation_count"], 8)
            self.assertEqual(
                {row["finite_modulus"] for row in report["rows"]}, {254, 255}
            )
            self.assertEqual(len(report["checksum_groups"]), 2)
            self.assertFalse(report_has_critical_failures(report))

    def test_checksum_and_contract_mismatch_are_separate_failures(self):
        for option, blocker in (
            ("changed_checksum", "checksum_mismatch"),
            ("changed_seed", "case_contract_mismatch"),
            ("duplicate_execution", "duplicate_capture_association"),
        ):
            with (
                self.subTest(option=option),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = ObservationFixture(Path(temporary), **{option: True})
                report = build_report([], observation_receipts=fixture.receipts())
                self.assertIn(blocker, report["promotion_blockers"])
                self.assertTrue(report_has_critical_failures(report))

    def test_actual_child_environment_cannot_be_replaced_by_request_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ObservationFixture(Path(temporary), swapped_device=True)
            stage, capture = fixture.captures[-2]
            with self.assertRaises(ValueError):
                make_binding(
                    capture, fixture.manifest_path, fixture.results_path, stage
                )

    def test_stale_raw_artifacts_and_resealed_summary_substitution_fail(self):
        for mode in ("raw", "facts", "pid", "target", "missing_identity"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = ObservationFixture(Path(temporary))
                stage, path = fixture.captures[0]
                binding = make_binding(
                    path, fixture.manifest_path, fixture.results_path, stage
                )
                value = copy.deepcopy(binding)
                if mode == "facts":
                    value["facts"]["rank"] = 3
                    value["identity_sha256"] = canonical_digest(
                        {
                            key: item
                            for key, item in value.items()
                            if key != "identity_sha256"
                        }
                    )
                elif mode == "raw":
                    path.write_text(path.read_text() + "\n")
                else:
                    # Coherently rehashing result JSON cannot replace a native
                    # process output: the raw stdout/capture artifacts remain bound.
                    rows = copy.deepcopy(fixture.rows)
                    parent = next(row for row in rows if row["name"] == "parent-4")
                    document = json.loads(parent["stdout"])
                    if mode == "pid":
                        document["process_id"] += 1
                    elif mode == "target":
                        document["gcn_arch"] = "gfx1100"
                    else:
                        document.pop("runtime_device_identity")
                    raw = write(fixture.root / "parent-4.stdout.log", document)
                    parent["stdout"] = raw.read_text(encoding="utf-8")
                    parent["artifacts"][str(raw.resolve())] = reference(raw)["sha256"]
                    write(fixture.results_path, dict(fixture.results, results=rows))
                    value["proof"]["session_results"] = reference(fixture.results_path)
                    value["identity_sha256"] = canonical_digest(
                        {
                            key: item
                            for key, item in value.items()
                            if key != "identity_sha256"
                        }
                    )
                with self.assertRaises(ValueError):
                    validate_binding(value, path)

    def test_renamed_capture_does_not_relabel_observed_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ObservationFixture(Path(temporary))
            receipts = fixture.receipts()
            intent = write(
                fixture.root / "intent.json",
                dict(
                    requested_devices=list(map(str, fixture.devices)),
                    physical_devices=[
                        dict(
                            physical_device_id=device,
                            target_arch="gfx1100",
                            device_name="wrong label",
                        )
                        for device in fixture.devices
                    ],
                ),
            )
            report = build_report([], env_summary=intent, observation_receipts=receipts)
            self.assertTrue(report_has_critical_failures(report))
            self.assertIn("observed_target_mismatch", report["promotion_blockers"])
            self.assertTrue(
                all(row["target_arch"] == "gfx942" for row in report["rows"])
            )


if __name__ == "__main__":
    unittest.main()
