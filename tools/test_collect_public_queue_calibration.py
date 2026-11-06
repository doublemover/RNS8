"""Authored validators/fixtures only; synthetic timings are never collected evidence."""

from benchmark_cases import case_root
import copy
import json
from pathlib import Path
import tempfile
import unittest
from collect_public_queue_calibration import (
    collection_plan,
    counts,
    validate_case,
    validate_capture,
)
from native_locality_contract import public_contract as locality_contract
from test_capture_identity_support import synthetic_identity
from native_queue_schedule_contract import IDENTITIES as SCHEDULE_IDENTITIES


class PublicQueueCollection(unittest.TestCase):
    def test_host_control_diagnostics_are_separate_strict_and_cover_real_children(self):
        from collect_public_queue_calibration import HOST_CONTROL_FIELDS, ALLOCATION_FIELDS

        c, e = self.fixture()
        e["case"]["host_diagnostics"] = True
        with self.assertRaises(ValueError):
            validate_capture(c, e)
        cold = dict.fromkeys(HOST_CONTROL_FIELDS, 0)
        warm = copy.deepcopy(cold)
        warm.update(
            stream_device_queries=4,
            stream_capture_queries=4,
            submission_proofs=8,
            event_records=8,
            event_waits=8,
            proof_reuses=8,
        )
        counts = dict.fromkeys(ALLOCATION_FIELDS, 0)
        c.update(
            host_control_diagnostics_compiled=True,
            hot_loop_serialization="typed_observations_json_after_intervals_v1",
            host_control_diagnostic=dict(
                schema="rns8_public_queue_host_control_v2",
                phase_time_contract="calling_thread_steady_ns_inclusive_scopes_nonadditive_v1",
                scope="calling_thread_one_complete_observation_after_timing",
                complete_groups=4,
                cold_setup=cold,
                warm_observation=warm,
                cold_setup_allocations=counts,
                warm_observation_allocations=copy.deepcopy(counts),
                output_digest="1234",
                exact_reference_passed=True,
                attempt_count_scope="prepared_submission_and_managed_runtime_handles",
                latency_evidence=False,
            ),
        )
        self.assertEqual(validate_capture(c, e), c)
        for key, value in [
            ("stream_device_queries", 8),
            ("event_waits", 1),
            ("event_records", True),
            ("proof_builds", 1),
            ("event_creates", 1),
            ("range_proof_host_ns", True),
            ("proof_metadata_host_ns", -1),
            ("proof_identity_host_ns", 2**64),
            ("graph_preflight_host_ns", 1.5),
        ]:
            bad = copy.deepcopy(c)
            bad["host_control_diagnostic"]["warm_observation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(bad, e)
        for key, value in [
            ("host_control_diagnostics_compiled", False),
            ("hot_loop_serialization", "json_per_call"),
        ]:
            bad = copy.deepcopy(c)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(bad, e)
        for key, value in [
            ("schema", "rns8_public_queue_host_control_v1"),
            ("phase_time_contract", "sum_as_production_latency"),
        ]:
            bad = copy.deepcopy(c)
            bad["host_control_diagnostic"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(bad, e)
        e["case"]["host_diagnostics"] = False
        with self.assertRaises(ValueError):
            validate_capture(c, e)

    def test_integrity_controls_require_generation_bound_consistent_receipts(self):
        c, e = self.fixture()
        e["case"]["integrity"] = True
        with self.assertRaises(ValueError):
            validate_capture(c, e)
        c["native_integrity_receipts"] = [
            dict(
                task=i,
                enabled=True,
                outcome=1,
                failure_mask=0,
                prefix=t["prefix"],
                basis_version=1,
                check_moduli=[257, 263],
                operation=i + 1,
                epoch=3,
                a_generation=1,
                b_generation=1,
            )
            for i, t in enumerate(e["case"]["tasks"])
        ]
        self.assertEqual(validate_capture(c, e), c)
        for key, value in [
            ("outcome", 2),
            ("failure_mask", 1),
            ("basis_version", True),
            ("epoch", 0),
            ("task", True),
        ]:
            bad = copy.deepcopy(c)
            bad["native_integrity_receipts"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(bad, e)

    def test_nondefault_dispatch_requires_the_actual_selected_body(self):
        from native_queue_schedule_contract import dispatch_hint

        c, e = self.fixture()
        case = e["case"]
        case["dispatch_policy"] = 1
        with self.assertRaises(ValueError):
            validate_capture(c, e)
        c["metadata"]["dispatch_hint"] = dispatch_hint(case)
        self.assertEqual(validate_capture(c, e), c)
        c["metadata"]["dispatch_hint"]["group"] = 2
        with self.assertRaises(ValueError):
            validate_capture(c, e)
        for bad in (True, -1, 3):
            case["dispatch_policy"] = bad
            with self.assertRaises(ValueError):
                validate_case(case)
        case["dispatch_policy"] = 1
        case["mode"] = "ordinary"
        with self.assertRaises(ValueError):
            validate_case(case)

    def test_ranked_public_capture_requires_actual_selected_schedule_identity(self):
        c, e = self.fixture()
        e["case"]["schedule"] = 3
        _, metadata, _, _ = counts(e["case"])
        c["metadata"]["device_bytes"] = metadata
        c["metadata"]["preparation_peak_device_bytes"] = metadata
        with self.assertRaises(ValueError):
            validate_capture(c, e)
        c["metadata"].update(schedule_mode=3, schedule_identity=SCHEDULE_IDENTITIES[3])
        self.assertEqual(validate_capture(c, e), c)
        c["metadata"]["schedule_mode"] = 2
        with self.assertRaises(ValueError):
            validate_capture(c, e)

    def case(self):
        return dict(
            schema="public_native_group_case_v1",
            mode="persistent",
            input_policy="resident",
            encoded_b=True,
            schedule=2,
            chunk=4,
            workers=8,
            reuse=4,
            warmups=3,
            repeats=9,
            seed=3838,
            padding=3,
            tasks=[
                dict(m=17, n=19, k=257, prefix=3, group=1, unsigned_inputs=False, b_slot=0),
                dict(m=2, n=19, k=257, prefix=3, group=4, unsigned_inputs=False, b_slot=0),
            ],
        )

    def fixture(self):
        case = self.case()
        work, metadata, encoded, output = counts(case)
        sample = dict(
            prepare_us=1,
            execute_us=20,
            export_us=10,
            host_loop_us=2,
            total_us=33,
            complete_groups=4,
        )
        capture = dict(
            schema="rns8_public_native_group_microcase_v1",
            case=case,
            run_id="synthetic",
            seed=3838,
            dataset="bounded_mt19937_64_shared_b_v1",
            host_instrumentation_state="hooks_disabled",
            output_boundary="each_public_group_and_all_padded_three_limb_exports",
            exact_reference_passed=True,
            oracle_inside_production_interval=False,
            oracle_scope="complete_final_output_after_each_observation_and_sustained_interval",
            per_task_device_completion_latency_measured=False,
            promotion_eligible=False,
            useful_tile_channel_groups_per_call=work,
            output_payload_bytes_per_call=output,
            encoded_b_retained_device_bytes=encoded,
            encoded_b_peak_replacement_device_bytes=encoded,
            setup_us=100,
            metadata=dict(
                tasks=2,
                work=work,
                device_bytes=metadata,
                host_bytes=500,
                preparation_peak_device_bytes=metadata,
                graph_ready=False,
            ),
            first_call=copy.deepcopy(sample),
            samples=[copy.deepcopy(sample) for _ in range(9)],
            sustained=[
                dict(
                    calls=16,
                    complete_groups=64,
                    duration_us=250000,
                    first_input_generation=1,
                    last_input_generation=1,
                    rebuilds=0,
                    exact_reference_passed=True,
                )
                for _ in range(3)
            ],
        )
        synthetic_identity(capture)
        return capture, dict(case=case, run_id="synthetic")

    def test_public_locality_preserves_shared_encoded_source_accounting(self):
        for order in ("ordinary", "output-neighbor", "modulus", "same-output"):
            capture, entry = self.fixture()
            case = entry["case"]
            case.update(schedule=0, locality_order=order)
            capture["target"] = "gfx1100"
            work, metadata, encoded, output = counts(case)
            capture["metadata"].update(
                device_bytes=metadata, preparation_peak_device_bytes=metadata
            )
            capture["locality"] = locality_contract(case, "gfx1100")
            validate_case(case)
            validate_capture(capture, entry)
            # Native A remains raw; one shared encoded B has no extra capability
            # word in consumed-byte traffic (it is retained allocation metadata).
            expected = (17 + 2) * 257 * 8 + 3 * 19 * 272
            self.assertEqual(capture["locality"]["unique_source_bytes"], expected)
            bad = copy.deepcopy(capture)
            bad["locality"]["physical_affinity_established"] = True
            with self.assertRaises(ValueError):
                validate_capture(bad, entry)
            case["schedule"] = 2
            with self.assertRaises(ValueError):
                validate_case(case)

    def test_generation_and_reuse_receipts_cannot_be_normalized_into_existence(self):
        capture, entry = self.fixture()
        self.assertEqual(validate_capture(capture, entry), capture)
        for field, value in [
            ("complete_groups", 16),
            ("last_input_generation", 65),
            ("rebuilds", 64),
            ("duration_us", 100),
            ("calls", True),
        ]:
            bad = copy.deepcopy(capture)
            bad["sustained"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_capture(bad, entry)
        for field, value in [
            ("complete_groups", 1),
            ("total_us", 31),
            ("execute_us", float("nan")),
        ]:
            bad = copy.deepcopy(capture)
            bad["samples"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_capture(bad, entry)

    def test_metadata_counts_deduplicate_encoded_B_but_not_task_work(self):
        case = self.case()
        work, metadata, encoded, output = counts(case)
        self.assertEqual(work, 14)
        self.assertEqual(metadata, 2 * 104 + 2 * 16 + 2 * 16 + 4)
        self.assertEqual(encoded, 3 * 19 * 17 * 16 + 4)
        self.assertEqual(output, (17 + 2) * 19 * 24)
        for schedule in (0, 1, 2):
            c = copy.deepcopy(case)
            c["schedule"] = schedule
            self.assertEqual(counts(c)[0], work)
        capture, entry = self.fixture()
        capture["metadata"]["device_bytes"] += 16
        with self.assertRaises(ValueError):
            validate_capture(capture, entry)
        capture, entry = self.fixture()
        capture["encoded_b_retained_device_bytes"] *= 2
        with self.assertRaises(ValueError):
            validate_capture(capture, entry)

    def test_mutated_b_requires_actual_rebuilds_and_replacement_peak(self):
        capture, entry = self.fixture()
        entry["case"]["input_policy"] = "mutated-b"
        capture["encoded_b_peak_replacement_device_bytes"] *= 2
        capture["metadata"]["preparation_peak_device_bytes"] *= 2
        for burst in capture["sustained"]:
            burst.update(last_input_generation=65, rebuilds=64)
        self.assertEqual(validate_capture(capture, entry), capture)
        capture["metadata"]["graph_ready"] = True
        with self.assertRaises(ValueError):
            validate_capture(capture, entry)

    def test_case_admission_and_committed_controls(self):
        for field, value in [
            ("workers", 0),
            ("encoded_b", 1),
            ("reuse", True),
            ("schedule", 3),
            ("input_policy", "unspecified"),
        ]:
            case = self.case()
            case[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_case(case)
        case = self.case()
        case["tasks"][1]["unsigned_inputs"] = True
        with self.assertRaises(ValueError):
            validate_case(case)
        case = self.case()
        case["tasks"][0]["prefix"] = case["tasks"][1]["prefix"] = 1
        with self.assertRaises(ValueError):
            validate_case(case)
        cases = [
            validate_case(json.loads(p.read_text()))
            for p in (case_root() / "public-native-queue").glob("*.json")
        ]
        self.assertEqual(
            {c["mode"] for c in cases}, {"persistent", "ordinary", "graph", "resident"}
        )
        self.assertEqual({c["schedule"] for c in cases}, {0, 1, 2})
        self.assertTrue(any(t["k"] == 65537 for c in cases for t in c["tasks"]))

    def test_plan_rotates_real_independent_processes_without_opening_binary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for mode in ("ordinary", "persistent", "graph"):
                case = self.case()
                case["mode"] = mode
                path = root / (mode + ".json")
                path.write_text(json.dumps(case))
                paths.append(path)
            plan = collection_plan(Path("not-executed"), paths, root / "out", 3, 0, "fixture")
            self.assertEqual(len(plan), 9)
            self.assertEqual(len({e["run_id"] for e in plan}), 9)
            self.assertNotEqual([e["case"] for e in plan[:3]], [e["case"] for e in plan[3:6]])


if __name__ == "__main__":
    unittest.main()
