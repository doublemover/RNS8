from benchmark_cases import case_root
import copy
import json
from pathlib import Path
import unittest
from collect_queue_calibration import MODES, commands, validate_case, validate_capture
from native_queue_schedule_contract import IDENTITIES, layout
from test_capture_identity_support import synthetic_identity

ROOT = Path(__file__).resolve().parents[1]


class QueueCollectionTests(unittest.TestCase):
    def test_dispatch_identity_is_derived_from_group_signedness_and_representation(self):
        from native_queue_schedule_contract import dispatch_hint, validate_dispatch_capture

        base = dict(
            tasks=[dict(group=2, unsigned=False), dict(group=2, unsigned=False)],
            encoded_b=True,
            dispatch_policy=1,
        )
        self.assertEqual(
            dispatch_hint(base), dict(policy=1, group=2, unsigned_inputs=0, encoded_b=1)
        )
        validate_dispatch_capture(base, dispatch_hint(base))
        for bad in (
            None,
            dict(policy=1, group=0, unsigned_inputs=0, encoded_b=0),
            dict(policy=1, group=2, unsigned_inputs=False, encoded_b=1),
        ):
            with self.assertRaises(ValueError):
                validate_dispatch_capture(base, bad)
        for policy in (0, 2):
            base["dispatch_policy"] = policy
            self.assertEqual(dispatch_hint(base)["group"], 0)
        base["dispatch_policy"] = 1
        base["tasks"][1]["unsigned"] = True
        self.assertEqual(dispatch_hint(base)["group"], 0)
        base["tasks"][1]["unsigned"] = False
        base["tasks"][1]["group"] = 4
        self.assertEqual(dispatch_hint(base)["group"], 0)
        for bad in (True, -1, 3):
            base["dispatch_policy"] = bad
            with self.assertRaises(ValueError):
                dispatch_hint(base)

    def test_ranked_capture_binds_decoder_bytes_without_rewriting_legacy_scan_captures(self):
        case = dict(
            schema="native_queue_case_v1",
            workers=1,
            chunk=1,
            reuse=1,
            repeats=3,
            warmups=1,
            metadata_budget_mib=1,
            encoded_b=False,
            mutate_b=False,
            tasks=[dict(m=1, n=1, k=1, prefix=1, group=1, unsigned=False)],
        )
        entry = dict(case=case, run_id="synthetic", mode="rank-select")
        capture = dict(
            schema="rns8_native_queue_microcase_v1",
            promotion_eligible=False,
            case=case,
            run_id="synthetic",
            schedule="rank-select",
            schedule_identity=IDENTITIES[3],
            host_instrumentation_state="hooks_disabled",
            exact_reference_passed=True,
            oracle_inside_production_interval=False,
            target="synthetic-not-hardware-evidence",
            samples=[dict(batch_ns=1, exact_reference_passed=True) for _ in range(3)],
            work_assignments_per_call=1,
            completion_diagnostic=dict(
                completed_work=[1],
                completion_ticks=[1],
                wall_clock_rate_khz=1000,
                completion_us=[1],
            ),
            decoder_layout=dict(abi_version=2, phase_count=1, rank_directory_bytes=0),
        )
        synthetic_identity(capture)
        self.assertEqual(validate_capture(capture, entry), capture)
        for field, value in [
            ("abi_version", 1),
            ("phase_count", True),
            ("rank_directory_bytes", 4),
        ]:
            bad = copy.deepcopy(capture)
            bad["decoder_layout"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_capture(bad, entry)
        legacy = copy.deepcopy(capture)
        legacy.pop("decoder_layout")
        legacy.update(schedule="fair", schedule_identity=IDENTITIES[2])
        self.assertEqual(validate_capture(legacy, dict(entry, mode="fair")), legacy)
        capture.pop("decoder_layout")
        with self.assertRaises(ValueError):
            validate_capture(capture, entry)

    def test_rank_directory_prices_actual_sparse_phases_and_equal_count_fast_path(self):
        tasks = [dict(m=1, n=1, prefix=1, group=1) for _ in range(4096)]
        self.assertEqual(
            layout(tasks, 3),
            dict(abi_version=2, phase_count=1, rank_directory_bytes=0, device_bytes=491540),
        )
        tasks[-1]["m"] = 257
        self.assertEqual(
            layout(tasks, 3),
            dict(abi_version=2, phase_count=2, rank_directory_bytes=1028, device_bytes=492584),
        )
        self.assertEqual(layout(tasks, 2)["device_bytes"] + 1028, layout(tasks, 3)["device_bytes"])
        self.assertEqual(layout(tasks, 0)["rank_directory_bytes"], 0)
        with self.assertRaises(ValueError):
            layout([], 3)

    def test_repository_controls_and_independent_order(self):
        cases = sorted((case_root() / "native-queue").glob("*.json"))
        self.assertGreaterEqual(len(cases), 7)
        for path in cases:
            validate_case(json.loads(path.read_text()))
        plan = commands(Path("binary"), cases, Path("output"), 4, 38, "collection")
        expected = {
            path: set(MODES)
            - ({"ordinary"} if json.loads(path.read_text()).get("dispatch_policy", 0) else set())
            for path in cases
        }
        self.assertEqual(len(plan), sum(map(len, expected.values())) * 4)
        self.assertEqual(len({e["run_id"] for e in plan}), len(plan))
        for path in cases:
            own = [e for e in plan if e["command"][1] == str(path)]
            for run in range(4):
                self.assertEqual({e["mode"] for e in own if e["run"] == run}, expected[path])
            self.assertGreater(
                len({tuple(e["mode"] for e in own if e["run"] == run) for run in range(4)}), 1
            )

    def test_invalid_abi_limits_and_bool_integer_aliases(self):
        good = json.loads((case_root() / "native-queue/many-tiny.json").read_text())
        for key, value in (
            ("schema", "v0"),
            ("workers", True),
            ("workers", 1025),
            ("chunk", 33),
            ("reuse", 0),
            ("tasks", []),
            ("mutate_b", 1),
        ):
            bad = copy.deepcopy(good)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_case(bad)
        for key, value in (("m", True), ("prefix", 21), ("group", 3), ("unsigned", 1)):
            bad = copy.deepcopy(good)
            bad["tasks"][0][key] = value
            with self.assertRaises(ValueError):
                validate_case(bad)
        bad = copy.deepcopy(good)
        bad["unexpected"] = 1
        with self.assertRaises(ValueError):
            validate_case(bad)

    def test_single_process_is_not_independent_cohort(self):
        with self.assertRaises(ValueError):
            commands(Path("b"), [], Path("o"), 1, 0, "id")


if __name__ == "__main__":
    unittest.main()
