"""Authored grouped calibration controls; no benchmark or subprocess is executed."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect_narrow_group_calibration as collector
from benchmark_request import file_digest
from test_capture_identity_support import synthetic_identity
from test_device_observation import capture as device_capture


def case():
    return dict(
        schema="narrow_group_case_v1",
        m=2,
        n=3,
        k=4097,
        tasks=3,
        prefix=0,
        unsigned_inputs=False,
        shared_b=True,
        mode="group",
        method="split-k",
        residency="resident",
        output="resident",
        repeats=3,
        warmups=1,
        seed=5353,
    )


class GroupCollection(unittest.TestCase):
    def setUp(self):
        root = collector.ROOT / "temp/narrow-group-collector-controls"
        root.mkdir(parents=True, exist_ok=True)
        scratch = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.binary = self.root / "synthetic-never-executed"
        self.binary.write_bytes(b"authored executable bytes never executed")
        self.input = self.root / "case.json"
        self.input.write_text(json.dumps(case()), encoding="utf-8")
        self.entry = dict(
            case=case(),
            run_id="authored-not-evidence",
            command=[
                str(self.binary),
                str(self.input),
                "authored-not-evidence",
                "--record-execution-environment",
            ],
            case_sha256=file_digest(self.input),
            output=str(self.root / "capture.json"),
        )

    def capture(self):
        phases = dict(prepare_us=1, execute_us=2, export_us=0, cleanup_us=0, total_us=3)
        value = dict(
            schema="rns8_narrow_group_microcase_v1",
            case=case(),
            run_id=self.entry["run_id"],
            process_id=123,
            dataset="affine_independent_native_tasks_v1",
            selected_prefix=3,
            exact_proof_policy="scalar_first",
            host_instrumentation_state="hooks_disabled",
            task_directory_entry_bytes=32,
            output_boundary="complete_task_batch_resident_rns",
            private_narrow_budget_bytes=432,
            native_useful_products=3 * 2 * 3 * 4097,
            full_output_reference_values=18,
            exact_reference_passed=True,
            all_task_residue_planes_audited=True,
            oracle_inside_production_interval=False,
            promotion_eligible=False,
            hot_loop_serialization="typed_observation_no_phase_clocks_in_sustained_v2",
            setup_us=1,
            first_call=phases,
            samples=[copy.deepcopy(phases) for _ in range(3)],
            sustained=[
                dict(calls=16, total_us=250000, per_call_us=15625, minimum_duration_reached=True)
                for _ in range(3)
            ],
        )
        synthetic_identity(value)
        value.update(device_capture())
        for name, calls in (("summary_first", 1), ("summary_final", 53)):
            value[name] = [
                dict(
                    a_instance=11 + i,
                    b_instance=21,
                    a_generation=1,
                    b_generation=1,
                    a_host_proofs=1,
                    b_host_proofs=1,
                    a_reuses=calls - 1,
                    b_reuses=calls - 1,
                    a_device_scans=0,
                    b_device_scans=0,
                    scan_bytes=0,
                    partial_bytes=96,
                    directory_bytes=96 if not i else 0,
                    directory_tasks=3 if not i else 0,
                )
                for i in range(3)
            ]
        value["separate_native_observation"] = dict(
            scope="one_separate_complete_task_batch",
            inside_timed_samples=False,
            inside_sustained_interval=False,
            public_product_calls=1,
            exact_reference_passed=True,
            native_event_receipts_complete=True,
            copy_scope="tracked_completed_synchronous_HIP_copy_wrappers",
            allocation_epoch=1,
            device_allocation_calls=0,
            device_free_calls=0,
            h2d_calls=0,
            h2d_bytes=0,
            d2h_calls=0,
            d2h_bytes=0,
            event_labels={"narrow_group_split_k_partials": 1, "narrow_group_split_k_encode": 1},
            event_microseconds={
                "narrow_group_split_k_partials": 1.5,
                "narrow_group_split_k_encode": 1.5,
            },
        )
        # Keep only producer fields: acquisition attaches its own request/artifact.
        for key in ("execution_id", "request_identity", "artifact_capture", "git_commit"):
            value.pop(key, None)
        value["execution_environment_witness"] = dict(
            schema="rns8_execution_environment_witness_v1",
            state="stable",
            arguments=self.entry["command"][1:],
            before={},
            after={},
        )
        return value

    def runner(self, command, **options):
        self.assertEqual(command, self.entry["command"])
        self.assertEqual(options["max_output_bytes"], collector.OUTPUT_LIMIT)
        return dict(
            command=command,
            process_id=123,
            returncode=0,
            timed_out=False,
            stdout=json.dumps(self.capture()),
            stderr="",
            max_output_bytes=collector.OUTPUT_LIMIT,
        )

    def acquire(self):
        return collector.acquire(self.entry, 5, environment={}, runner=self.runner)

    def reseal_native(self, capture):
        path = Path(capture["acquisition"]["path"])
        raw = json.loads(path.read_text())
        raw["stdout"] = json.dumps(
            {
                key: value
                for key, value in capture.items()
                if key
                not in {"acquisition", "request_identity", "execution_id", "artifact_capture"}
            }
        )
        path.write_text(json.dumps(raw), encoding="utf-8")
        capture["acquisition"]["sha256"] = file_digest(path)

    def test_complete_authored_acquisition_binds_all_task_proofs_and_actual_pid(self):
        capture = self.acquire()
        self.assertEqual(collector.validate_capture(capture, self.entry), capture)
        bad = copy.deepcopy(capture)
        bad["process_id"] = 999
        self.reseal_native(bad)
        with self.assertRaisesRegex(ValueError, "PID"):
            collector.validate_capture(bad, self.entry)

    def test_resealed_receipts_cannot_hide_missing_partials_task_oracle_or_wrong_group_stages(self):
        capture = self.acquire()
        for key, value in (
            ("full_output_reference_values", 6),
            ("private_narrow_budget_bytes", 16),
            ("all_task_residue_planes_audited", False),
            ("oracle_inside_production_interval", True),
        ):
            bad = copy.deepcopy(capture)
            bad[key] = value
            self.reseal_native(bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                collector.validate_capture(bad, self.entry)
        for mutate in (
            lambda value: value["summary_final"][2].update(partial_bytes=0),
            lambda value: value["summary_final"][1].update(a_instance=11),
            lambda value: value["summary_final"][2].update(b_generation=2),
            lambda value: value["separate_native_observation"]["event_labels"].update(
                narrow_group_split_k_partials=3
            ),
            lambda value: value["sustained"][0].update(minimum_duration_reached=False),
        ):
            bad = copy.deepcopy(capture)
            mutate(bad)
            self.reseal_native(bad)
            with self.assertRaises(ValueError):
                collector.validate_capture(bad, self.entry)

    def test_case_limits_injection_and_plan_only_do_not_launch_work(self):
        for values in (
            dict(tasks=True),
            dict(tasks=65),
            dict(unsigned_inputs=True),
            dict(residency="cold"),
            dict(prefix=1),
            dict(m=512, n=512, tasks=64),
        ):
            changed = case()
            changed.update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                collector.validate_case(changed)
        with self.assertRaisesRegex(ValueError, "profiler"):
            collector.acquire(
                self.entry,
                5,
                environment={"PERF_EXEC_PATH": "unbound"},
                runner=lambda *args, **kwargs: self.fail("must not launch"),
            )
        output = self.root / "plan-only"
        with (
            patch.object(collector, "acquire", side_effect=AssertionError("workload")),
            patch("builtins.print"),
        ):
            self.assertEqual(
                collector.main(
                    [
                        "--binary",
                        str(self.binary),
                        "--cases",
                        str(self.input),
                        "--output",
                        str(output),
                        "--plan-only",
                    ]
                ),
                0,
            )
        self.assertFalse(output.exists())

    def test_authored_catalog_keeps_seven_complete_matched_cohorts(self):
        from benchmark_cases import catalog_documents

        catalog = json.loads((collector.ROOT / "benchmarks/cases/narrow-group.json").read_text())
        groups = {}
        for _, raw in catalog_documents(catalog):
            value = collector.validate_case(json.loads(raw))
            identity = json.dumps(
                {key: field for key, field in value.items() if key != "mode"}, sort_keys=True
            )
            groups.setdefault(identity, set()).add(value["mode"])
        self.assertEqual(len(groups), 7)
        self.assertTrue(all(modes == {"single", "group", "rns"} for modes in groups.values()))


if __name__ == "__main__":
    unittest.main()
