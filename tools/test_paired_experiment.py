"""Authored two-source controls; no benchmark or compiler is executed here."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_request import file_digest
from diagnostic_companion import digest
from paired_experiment import (
    RECEIPT_SCHEMA,
    build_plan,
    file_reference,
    load_experiment,
    validate_plan,
)
from paired_experiment_report import build_report
from report_comparison import ComparisonPolicy, ReportCohorts
from test_diagnostic_companion import authored_capture_pair
from test_direct_hip_report_support import direct_resident_capture
from test_review_reference_support import add_synthetic_full_reference


def write(path: Path, value: dict):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def dataset(root: Path, candidate_us=80):
    captures, arms = {}, {}
    for arm, commit, duration in (("baseline", "c", 100), ("candidate", "d", candidate_us)):
        binary = root / (arm + ".exe")
        binary.write_bytes(("AUTHORED NONEXECUTABLE " + arm).encode())
        for run in range(2):
            lane = root / f"{arm}-{run}"
            lane.mkdir()
            fixture = direct_resident_capture("direct_hip_prefix9_grouped_rns_gemm_v1", duration)
            add_synthetic_full_reference(fixture)
            production, diagnostic, _ = authored_capture_pair(lane, fixture)
            for capture, mode in ((production, "production"), (diagnostic, "instrumented")):
                capture["git_commit"] = commit * 40
                capture["build_manifest"].update(
                    source_commit=commit * 40, source_digest=commit * 64
                )
                capture["artifact_capture"]["executable_sha256"] = file_digest(binary)
                capture["request_identity"]["executable_sha256"] = file_digest(binary)
                captures[(arm, run, mode)] = capture
            if run == 0:
                manifest = root / (arm + "-manifest.json")
                write(manifest, production["build_manifest"])
                arms[arm] = dict(
                    bench=str(binary),
                    manifest=file_reference(manifest),
                    backend_selected=production["backend_selected"],
                    selected_kernel=production["selected_kernel"],
                )
    plan = build_plan(
        dict(allowed_delta="Authored source-only implementation change control", arms=arms),
        root,
        root,
        2,
        "authored-control",
        [],
    )
    rows = []
    for job in plan["jobs"]:
        capture = captures[(job["arm"], job["run"], job["mode"])]
        request = capture["request_identity"]
        request["arguments"] = job["command"][1:]
        request["identity_sha256"] = digest(
            {key: value for key, value in request.items() if key != "identity_sha256"}
        )
        path = Path(job["output"])
        write(path, capture)
        rows.append(
            dict(
                ordinal=job["ordinal"],
                run=job["run"],
                arm=job["arm"],
                mode=job["mode"],
                started_ns=job["ordinal"] * 100 + 1,
                finished_ns=job["ordinal"] * 100 + 90,
                capture=file_reference(path),
                capture_sha256=digest(capture),
                execution_id=capture["execution_id"],
            )
        )
    plan_path, receipt_path = root / "plan.json", root / "collection.json"
    write(plan_path, plan)
    write(
        receipt_path,
        dict(
            schema=RECEIPT_SCHEMA, completed=True, plan=file_reference(plan_path), observations=rows
        ),
    )
    return plan_path, receipt_path


def rewrite_observations(receipt_path, change):
    receipt = json.loads(receipt_path.read_text())
    for row in receipt["observations"]:
        path = Path(row["capture"]["path"])
        capture = json.loads(path.read_text())
        change(capture, row)
        request = capture["request_identity"]
        request["identity_sha256"] = digest(
            {key: value for key, value in request.items() if key != "identity_sha256"}
        )
        write(path, capture)
        row.update(
            capture=file_reference(path),
            capture_sha256=digest(capture),
            execution_id=capture["execution_id"],
        )
    write(receipt_path, receipt)


class PairedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_two_sources_qualify_only_inside_declared_experiment(self):
        plan, receipt = dataset(self.root)
        report = build_report(plan, receipt)
        self.assertTrue(report["qualified"], report)
        self.assertTrue(report["beneficial"])
        self.assertAlmostEqual(report["comparison"]["speedup"], 1.25)
        self.assertFalse(report["promotion_eligible"])
        experiment, captures, bindings, owner = load_experiment(plan, receipt)
        cohorts = ReportCohorts(captures, bindings)
        before = next(view for view in cohorts.observations if experiment.arm(view) == "baseline")
        after = next(view for view in cohorts.observations if experiment.arm(view) == "candidate")
        ordinary = ComparisonPolicy("ordinary", require_cpu_anchor=False)
        fact = cohorts.compare(after, before, ordinary)
        self.assertFalse(fact["qualified"])
        self.assertIn("comparison_environment_mismatch", fact["blockers"])
        # Retained observation hashes cannot authorize a changed representative.
        changed = copy.deepcopy(after)
        changed["build_manifest"]["source_commit"] = "e" * 40
        self.assertIsNone(experiment.arm(changed))
        paired = ComparisonPolicy("paired", require_cpu_anchor=False, paired_experiment=experiment)
        self.assertIn(
            "paired_experiment_arm_membership_mismatch",
            cohorts.compare(changed, before, paired)["blockers"],
        )
        owner.finish()

    def test_small_improvement_does_not_become_beneficial(self):
        plan, receipt = dataset(self.root, candidate_us=99)
        report = build_report(plan, receipt)
        self.assertTrue(report["qualified"], report)
        self.assertFalse(report["beneficial"])

    def test_reordered_missing_copied_or_overlapping_processes_rejected(self):
        plan, receipt = dataset(self.root)
        original = json.loads(receipt.read_text())
        for mutation in ("order", "missing", "copy", "overlap", "bool_time"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(original)
                rows = bad["observations"]
                if mutation == "order":
                    rows[0], rows[1] = rows[1], rows[0]
                elif mutation == "missing":
                    rows.pop()
                elif mutation == "copy":
                    rows[-1].update(
                        {key: rows[0][key] for key in ("capture", "capture_sha256", "execution_id")}
                    )
                elif mutation == "overlap":
                    rows[1]["started_ns"] = rows[0]["started_ns"]
                else:
                    rows[0]["started_ns"] = True
                write(receipt, bad)
                with self.assertRaises(ValueError):
                    load_experiment(plan, receipt)

    def test_stale_content_and_rebound_foreign_artifacts_rejected(self):
        plan, receipt = dataset(self.root)
        first = json.loads(receipt.read_text())["observations"][0]
        path = Path(first["capture"]["path"])
        value = json.loads(path.read_text())
        value["checksum_u64"] += 1
        write(path, value)
        with self.assertRaises(ValueError):
            load_experiment(plan, receipt)
        rewrite_observations(
            receipt,
            lambda capture, row: (
                capture["artifact_capture"].update(executable_sha256="0" * 64)
                if row["arm"] == "candidate"
                else None
            ),
        )
        with self.assertRaises(ValueError):
            load_experiment(plan, receipt)

    def test_matched_companions_do_not_waive_environment_or_math(self):
        mutations = {
            "environment": lambda capture: capture["request_identity"]["environment"].update(
                HIP_VISIBLE_DEVICES="1"
            ),
            "compiler": lambda capture: capture["compiler"].update(
                version="9999 authored compiler"
            ),
            "host": lambda capture: capture["execution_host_identity"].update(
                os_instance_sha256="f" * 64
            ),
            "checksum": lambda capture: capture.update(checksum_u64=capture["checksum_u64"] + 1),
            "reference": lambda capture: capture.pop("native_reference_verification", None),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                root = self.root / name
                root.mkdir()
                plan, receipt = dataset(root)
                rewrite_observations(
                    receipt,
                    lambda capture, row, apply=mutation: (
                        apply(capture) if row["arm"] == "candidate" else None
                    ),
                )
                if name in ("environment", "compiler", "host"):
                    with self.assertRaisesRegex(ValueError, "changed host/device/toolchain"):
                        load_experiment(plan, receipt)
                else:
                    report = build_report(plan, receipt)
                    self.assertFalse(report["qualified"], report)
                    self.assertTrue(report["blockers"])

    def test_renamed_content_still_binds_and_unknown_capture_does_not_join(self):
        plan, receipt = dataset(self.root)
        value = json.loads(receipt.read_text())
        for index, row in enumerate(value["observations"]):
            source = Path(row["capture"]["path"])
            alias = self.root / f"unrelated-{100 - index}.json"
            alias.write_bytes(source.read_bytes())
            row["capture"] = file_reference(alias)
        write(receipt, value)
        self.assertTrue(build_report(plan, receipt)["qualified"])
        experiment, captures, _, _ = load_experiment(plan, receipt)
        foreign = copy.deepcopy(captures[0])
        foreign["checksum_u64"] += 1
        self.assertIsNone(experiment.arm(foreign))

    def test_unsealed_or_unbalanced_plan_rejected_before_collection(self):
        path, _ = dataset(self.root)
        original = json.loads(path.read_text())
        for runs in (1, 3, True, 102):
            bad = copy.deepcopy(original)
            bad["runs"] = runs
            bad["identity_sha256"] = digest(
                {key: value for key, value in bad.items() if key != "identity_sha256"}
            )
            with self.assertRaises(ValueError):
                validate_plan(bad)
        original["allowed_delta"] += " changed after sealing"
        with self.assertRaises(ValueError):
            validate_plan(original)

    def test_collection_preflights_both_binaries_before_starting_a_process(self):
        from collect_paired_experiment import collect

        plan_path, receipt_path = dataset(self.root)
        plan = json.loads(plan_path.read_text())
        # These are this control's own TemporaryDirectory files, never build outputs.
        receipt_path.unlink()
        for job in plan["jobs"]:
            Path(job["output"]).unlink()
        Path(plan["arms"]["candidate"]["bench"]).write_bytes(b"changed authored nonexecutable")
        with patch("collect_paired_experiment.run_command") as run:
            with self.assertRaisesRegex(ValueError, "executable changed"):
                collect(plan_path, 10)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
