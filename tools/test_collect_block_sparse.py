"""Synthetic public sparse capture/collector fixtures, authored without execution."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from collect_block_sparse import (BUDGET, MODES, active, checksum, collect, contract, dataset,
                                 load_cases, structure, summarize, validate_capture, validate_case)
from test_capture_identity_support import synthetic_identity
BASE = dict(device=0, mode="sparse_scalar", m=2, n=3, k=65, pattern="mixed", reuse_count=1, hybrid_threshold=0)


def case_for(mode="sparse_scalar", **changes):
    return validate_case({**BASE, "mode": mode, "hybrid_threshold": 1 if "hybrid" in mode else 0, **changes})


def fixture(case=None, run_id="fixture-run", repeats=2, warmups=1):
    case = case or case_for(); capture = contract(case, run_id, repeats, warmups); synthetic_identity(capture)
    a, b = dataset(case)
    av = [int.from_bytes(a[i:i+8], "little", signed=True) for i in range(0, len(a), 8)]
    bv = [int.from_bytes(b[i:i+8], "little", signed=True) for i in range(0, len(b), 8)]
    result = bytearray()
    for r in range(case["m"]):
        for col in range(case["n"]):
            value = sum(av[r*case["k"]+t]*bv[t*case["n"]+col] for t in range(case["k"]))
            result.extend((value % (1 << 192)).to_bytes(24, "little"))
    ab, mn = len(av)+len(bv), case["m"]*case["n"]; counts = structure(case)
    modeled = 80*ab+112*mn+2*(counts["modeled_packed_payload_bytes"]+counts["modeled_directory_bytes"])+8*max(len(av), len(bv))+counts["modeled_directory_bytes"]+4
    cold = case["mode"].endswith("_cold"); warm = case["mode"] != "dense" and not cold
    allocations = []
    for index, cells in enumerate((len(av), len(bv), mn)):
        payload = (28 if index < 2 else 20)*cells
        allocations.append(dict(logical_primary_bytes=20*cells, host_payload_capacity_bytes=0, device_payload_capacity_bytes=payload,
            host_current_bytes=0, device_current_bytes=payload, device_upload_staging_bytes=0,
            device_export_staging_bytes=0, device_status_and_schedule_bytes=0, device_total_capacity_bytes=payload))
    capture.update(device_name="synthetic GPU only", target="gfx1100", hip_runtime_version=70100000, hip_driver_version=70100000,
        selected_dense_kernel="synthetic dense kernel", a_checksum=checksum(a), b_checksum=checksum(b), output_checksum=checksum(result),
        source_instance_ids=[1, 2], output_instance_id=3, reference_fixed_operand_object_bytes=48, reference_operand_storage_bytes=48*ab,
        reference_and_dataset_ns=5000, native_setup_ns=700, setup_ns=1000, warm_prepare_ns=200 if warm else None,
        clock_period_numerator=1, clock_period_denominator=1000000000, plan_workspace_bytes=0,
        logical_live_payload_envelope_bytes=max(8*ab+24*mn+48*ab+2048, modeled), resident_payload_bytes=28*ab+20*mn,
        matrix_allocations_at_setup=deepcopy(allocations), matrix_allocations_after_final_audit=deepcopy(allocations),
        samples=[dict(wall_ns=1000*(i+1), preparation_ns=200 if cold else 0, destruction_ns=100 if cold else 0, exact_reference_passed=True) for i in range(repeats)],
        median_wall_ns=(repeats+1)*500., gemm_calls_per_second=1e9*case["reuse_count"]/((repeats+1)*500.),
        sustained=dict(calls=16, gemm_calls=16*case["reuse_count"], elapsed_ns=320000000, mean_batch_ns=20000000.,
            mean_gemm_ns=20000000./case["reuse_count"], minimum_calls=16, maximum_calls=4096, target_ns=250000000,
            stop_reason="duration", independent_audit_inside_interval=False, final_exact_reference_passed=True))
    return capture


class Captures(unittest.TestCase):
    def test_modes_patterns_reuse_and_thresholds(self):
        for mode in MODES:
            for pattern in ("dense", "mixed", "disjoint"):
                for reuse in (1, 8, 256):
                    case = case_for(mode, pattern=pattern, reuse_count=reuse)
                    validate_capture(fixture(case), case, "fixture-run", 2, 1)
        for threshold in (0, 1, (1 << 32)-1):
            case = case_for("sparse_hybrid", m=17, n=19, k=129, hybrid_threshold=threshold)
            validate_capture(fixture(case), case, "fixture-run", 2, 1)
        for k in (1, 65535, 65536, 65537):
            case = case_for(m=1, n=1, k=k)
            validate_capture(fixture(case), case, "fixture-run", 2, 1)

    def test_exact_block_recipe_structure_and_full_width_values(self):
        case = case_for("sparse_hybrid", m=17, n=19, k=129)
        counts = structure(case)
        self.assertEqual((counts["modeled_a_blocks"], counts["modeled_b_blocks"]), (5, 4))
        self.assertEqual((counts["modeled_total_block_intersections"], counts["modeled_empty_output_tiles"]), (6, 1))
        self.assertEqual((counts["modeled_hybrid_scalar_tiles"], counts["modeled_hybrid_wmma_tiles"]), (2, 2))
        self.assertEqual(counts["modeled_packed_payload_bytes"], 9*20*1024)
        self.assertEqual(counts["modeled_directory_bytes"], 60)
        self.assertEqual(structure({**case, "pattern": "disjoint"})["modeled_total_block_intersections"], 0)
        a, b = dataset(BASE)
        self.assertEqual(int.from_bytes(a[:8], "little", signed=True), -(1 << 63))
        self.assertEqual(int.from_bytes(b[:8], "little", signed=True), -(1 << 63))
        self.assertEqual(dataset(BASE), dataset(BASE))
        self.assertNotEqual(dataset(BASE), dataset({**BASE, "pattern": "disjoint"}))

    def test_invalid_controls_and_manifest_duplicates(self):
        for key, values in {"device": (-1, 64, True, 0.), "mode": (None, "auto", "wmma"), "m": (0, 513, True, 1.),
            "n": (0, 513), "k": (0, 65538), "pattern": (None, "sparse"), "reuse_count": (0, 257, False, 1.),
            "hybrid_threshold": (-1, 1 << 32, True, 1)}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_case({**BASE, key: value})
        with self.assertRaises(ValueError):
            validate_case({**BASE, "m": 512, "n": 512, "k": 513})
        with self.assertRaises(ValueError):
            validate_case({**BASE, "workers": 32})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cases.json"
            for cases in ([], [BASE, BASE], [dict(BASE, extra=True)]):
                path.write_text(json.dumps(dict(schema="rns8_block_sparse_cases_v1", cases=cases)))
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_all_contract_fields_are_typed_and_bound(self):
        case = case_for("sparse_hybrid")
        for key in contract(case, "fixture-run", 2, 1):
            capture = fixture(case); capture.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(capture, case, "fixture-run", 2, 1)
        for changes in (dict(source_versions=[True, True]), dict(content_generations=[1, 1]),
                        dict(actual_cache_allocation_bytes=1), dict(modeled_a_blocks=True)):
            capture = fixture(case); capture.update(changes)
            with self.assertRaises(ValueError):
                validate_capture(capture, case, "fixture-run", 2, 1)

    def test_malformed_resource_reference_timing_and_identity(self):
        mutations = [lambda c: c.update(target=""), lambda c: c.update(hip_runtime_version=0),
            lambda c: c.update(a_checksum="0"*16), lambda c: c.update(output_checksum="short"),
            lambda c: c.update(source_instance_ids=[1, 1]), lambda c: c.update(output_instance_id=1),
            lambda c: c.update(reference_operand_storage_bytes=0), lambda c: c.update(reference_fixed_operand_object_bytes=16),
            lambda c: c.update(logical_live_payload_envelope_bytes=BUDGET+1), lambda c: c.update(resident_payload_bytes=0),
            lambda c: c.update(warm_prepare_ns=None), lambda c: c.update(setup_ns=1),
            lambda c: c["matrix_allocations_at_setup"][0].update(device_total_capacity_bytes=0),
            lambda c: c["matrix_allocations_after_final_audit"][0].update(host_current_bytes=1),
            lambda c: c["samples"][0].update(preparation_ns=1), lambda c: c["samples"][0].update(exact_reference_passed=False),
            lambda c: c.update(median_wall_ns=2000), lambda c: c.update(gemm_calls_per_second=1),
            lambda c: c["sustained"].update(gemm_calls=17), lambda c: c["sustained"].update(calls=True),
            lambda c: c["sustained"].update(mean_gemm_ns=float("inf")),
            lambda c: c["sustained"].update(independent_audit_inside_interval=True),
            lambda c: c["build_manifest"].update(source_commit="invalid")]
        for mutation in mutations:
            capture = fixture(); mutation(capture)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_capture(capture, BASE, "fixture-run", 2, 1)
        cold = case_for("sparse_scalar_cold")
        capture = fixture(cold); capture["samples"][0].update(preparation_ns=900, destruction_ns=900)
        with self.assertRaises(ValueError):
            validate_capture(capture, cold, "fixture-run", 2, 1)
        accelerated = case_for("sparse_wmma"); capture = fixture(accelerated); capture["target"] = "gfx942"
        with self.assertRaises(ValueError):
            validate_capture(capture, accelerated, "fixture-run", 2, 1)

    def test_tracked_manifests_have_complete_controls(self):
        root = Path(__file__).resolve().parents[1]/"benchmarks"/"block-sparse"
        for name, count in (("smoke.json", 7), ("tail-reuse.json", 63), ("512-reuse.json", 42)):
            cases = load_cases(root/name); self.assertEqual(len(cases), count)
            for shape in {(case["pattern"], case["reuse_count"]) for case in cases}:
                self.assertEqual({case["mode"] for case in cases if (case["pattern"], case["reuse_count"]) == shape}, set(MODES))


class ObservedReuse(unittest.TestCase):
    def observations(self):
        result = []
        for reuse in (1, 8, 32):
            for mode in ("dense", "sparse_scalar", "sparse_scalar_cold"):
                for independent in range(5):
                    time = 1000*reuse if mode == "dense" else 500*reuse+(3000 if mode.endswith("_cold") else 0)
                    result.append(dict(case=case_for(mode, reuse_count=reuse), independent_run=independent, mean_batch_ns=time+independent))
        return result

    def test_observed_reuse_counts_without_extrapolation(self):
        summary = summarize(self.observations())[0]
        self.assertEqual(summary["first_observed_range_separated_cold_win_reuse"], 8)
        self.assertEqual([point["reuse_count"] for point in summary["tested_points"]], [1, 8, 32])
        self.assertTrue(all(point["complete_matched_cohort"] for point in summary["tested_points"]))
        fewer = [row for row in self.observations() if row["independent_run"] < 4]
        self.assertIsNone(summarize(fewer)[0]["first_observed_range_separated_cold_win_reuse"])
        missing = [row for row in self.observations() if row["case"]["mode"] != "sparse_scalar_cold"]
        self.assertTrue(all(not point["complete_matched_cohort"] for point in summarize(missing)[0]["tested_points"]))
        duplicated = self.observations(); duplicated.append(duplicated[0])
        self.assertFalse(summarize(duplicated)[0]["tested_points"][0]["complete_matched_cohort"])


class Collection(unittest.TestCase):
    def setUp(self):
        # Host metadata may itself invoke subprocess.run (uname -p on Linux).
        host = patch("collect_block_sparse.platform.platform", return_value="synthetic-platform")
        host.start()
        self.addCleanup(host.stop)

    def scenario(self, mutation=None, status=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); exe = root/"fixture"; exe.write_bytes(b"fixture only")
            cases = root/"cases.json"; cases.write_text(json.dumps(dict(schema="rns8_block_sparse_cases_v1", cases=[BASE])))
            args = argparse.Namespace(exe=str(exe), cases=str(cases), output=str(root/"output"), runs=2, repeats=2, warmups=1, timeout=10)
            count = 0
            def child(command, **kwargs):
                nonlocal count
                count += 1; capture = fixture(BASE, command[-1])
                if mutation:
                    mutation(capture, count, exe)
                return subprocess.CompletedProcess(command, status, json.dumps(capture), "synthetic stderr")
            with patch("collect_block_sparse.subprocess.run", side_effect=child):
                if mutation or status:
                    with self.assertRaises(ValueError):
                        collect(args)
                else:
                    collect(args)
            records = [json.loads(path.read_text()) for path in sorted((root/"output").glob("*.record.json"))]
            self.assertEqual(len(records), count); self.assertEqual(len({record["run_id"] for record in records}), count)
            self.assertEqual(records[-1]["validated"], not bool(mutation or status))
            for record in records:
                self.assertFalse(record["promotion_eligible"])
                self.assertEqual(set(record["tool_sha256"]), {"collect_block_sparse.py", "collect_finite_verify.py", "build_manifest.py"})
            self.assertEqual(len(list((root/"output").glob("comparison-*.json"))), 0 if mutation or status else 1)
            return records

    def test_collector_retains_bound_process_receipts(self):
        records = self.scenario(); self.assertTrue(all(record["provenance_complete"] for record in records))

    def test_metadata_provenance_output_runtime_and_artifact_mismatches(self):
        for mutation in (lambda c, i, exe: c.update(run_id="wrong"), lambda c, i, exe: c.update(reuse_count=2),
                         lambda c, i, exe: c.update(content_generation_visibility="host guessed"),
                         lambda c, i, exe: c.update(output_checksum="0"*16) if i == 2 else None,
                         lambda c, i, exe: c.update(device_name="changed") if i == 2 else None,
                         lambda c, i, exe: c["build_manifest"].update(source_commit="e"*40) if i == 2 else None,
                         lambda c, i, exe: exe.write_bytes(b"changed executable")):
            with self.subTest(mutation=mutation):
                self.scenario(mutation)
        self.scenario(status=77)

    def test_timeout_preserves_partial_evidence_without_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); exe = root/"fixture"; exe.write_bytes(b"fixture only")
            cases = root/"cases.json"; cases.write_text(json.dumps(dict(schema="rns8_block_sparse_cases_v1", cases=[BASE])))
            args = argparse.Namespace(exe=str(exe), cases=str(cases), output=str(root/"output"), runs=1, repeats=2, warmups=1, timeout=1)
            with patch("collect_block_sparse.subprocess.run", side_effect=subprocess.TimeoutExpired("fixture", 1, output=b"partial stdout", stderr=b"partial stderr")):
                with self.assertRaises(subprocess.TimeoutExpired):
                    collect(args)
            self.assertEqual(next((root/"output").glob("*.stdout.json")).read_bytes(), b"partial stdout")
            self.assertEqual(next((root/"output").glob("*.stderr.txt")).read_bytes(), b"partial stderr")
            self.assertFalse(json.loads(next((root/"output").glob("*.record.json")).read_text())["validated"])
            self.assertFalse(list((root/"output").glob("comparison-*.json")))


@unittest.skipUnless(os.environ.get("RNS8_BLOCK_SPARSE_EXE"), "set RNS8_BLOCK_SPARSE_EXE for pre-context CLI fixtures")
class CompiledCLI(unittest.TestCase):
    def test_help_and_malformed_arguments_never_start_gpu(self):
        exe = os.environ["RNS8_BLOCK_SPARSE_EXE"]
        self.assertEqual(subprocess.run([exe, "--help"], capture_output=True).returncode, 0)
        valid = ["63", "dense", "2", "3", "65", "mixed", "1", "0", "2", "1", "fixture"]
        invalid = [valid[:-1], valid+["extra"]]
        for index, value in ((0, "-1"), (0, "64"), (1, "auto"), (2, "0"), (3, "513"), (4, "65538"),
                             (5, "unknown"), (6, "0"), (6, "257"), (7, "1"), (8, "0"), (9, "0"), (10, "bad/id")):
            args = list(valid); args[index] = value; invalid.append(args)
        for args in invalid:
            with self.subTest(args=args):
                self.assertEqual(subprocess.run([exe, *args], capture_output=True, timeout=10).returncode, 1)


if __name__ == "__main__":
    unittest.main()
