"""Authored synthetic collector/CLI fixtures; never performance evidence."""
from benchmark_cases import case_root
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from collect_finite_verify import (BUDGET, MODES, checksum, collect, contract, dataset,
                                  load_cases, strict_json, validate_capture, validate_case)
from test_capture_identity_support import synthetic_identity

BASE = dict(backend="cpu-reference", device=-1, mode="verify_only", m=2, n=3, k=65,
            modulus=250, trials=4, challenge_seed=3135)


def case_for(mode="verify_only", backend="cpu-reference", **changes):
    case = {**BASE, "mode": mode, "backend": backend, "device": -1 if backend == "cpu-reference" else 0}
    if mode == "full_reference":
        case.update(trials=0, challenge_seed=0)
    return validate_case({**case, **changes})


def fixture(case=None, run_id="fixture-run", repeats=2, warmups=1):
    case = case or case_for()
    capture = contract(case, run_id, repeats, warmups)
    synthetic_identity(capture)
    a, b = dataset(case)
    output = [sum(a[r*case["k"]+t]*b[t*case["n"]+col] for t in range(case["k"])) % case["modulus"]
              for r in range(case["m"]) for col in range(case["n"])]
    cells = len(a) + len(b) + len(output)
    scratch = (case["n"] if case["backend"] == "cpu-reference" else 2*case["n"]+8) + 4*case["k"]
    capture.update(device_name="synthetic fixture only", target="cpu" if case["backend"] == "cpu-reference" else "gfx1100",
                   hip_runtime_version=0 if case["backend"] == "cpu-reference" else 70100000,
                   hip_driver_version=0 if case["backend"] == "cpu-reference" else 70100000,
                   selected_compute_kernel="synthetic kernel",
                   setup_ns=1000, oracle_and_dataset_ns=2000, clock_period_numerator=1, clock_period_denominator=1000000000,
                   a_checksum=checksum(a), b_checksum=checksum(b), output_checksum=checksum(output),
                   plan_workspace_bytes=0, resident_payload_bytes=cells,
                   logical_live_payload_envelope_bytes=5*cells+16*case["m"]*case["n"]+3*case["m"]+scratch,
                   samples=[dict(wall_ns=100+index*100, exact_reference_passed=True) for index in range(repeats)],
                   median_wall_ns=(repeats+1)*50., calls_per_second=1e9/((repeats+1)*50.),
                   sustained=dict(calls=16, elapsed_ns=320000000, mean_wall_ns=20000000., minimum_calls=16, maximum_calls=4096,
                                  target_ns=250000000, stop_reason="duration", independent_audit_inside_interval=False,
                                  final_exact_reference_passed=True), verification=None)
    if case["mode"] != "full_reference":
        capture["verification"] = dict(passed=True, sampler=1, sampler_name="splitmix64_binary_v1", replay_seed=case["challenge_seed"],
            trials=case["trials"], workspace_bytes=scratch, ideal_failure_bound_exponent=case["trials"],
            ideal_failure_probability=math.ldexp(1., -case["trials"]), probability_model="independent_uniform_binary_bits_only",
            fresh_entropy=False, integer_equality_certified=False, matrix_instance_ids=[1, 2, 3], generations=[1, 1, 1])
    return capture


class Cases(unittest.TestCase):
    def test_all_modes_backends_moduli_and_boundaries(self):
        for backend in ("cpu-reference", "hip-direct"):
            for mode in MODES:
                for q in (2, 4, 6, 251, 256):
                    case = case_for(mode, backend, modulus=q)
                    validate_capture(fixture(case), case, "fixture-run", 2, 1)
        for controls in (dict(m=1, n=1, k=65537), dict(m=512, n=512, k=512), dict(trials=256, challenge_seed=(1 << 64)-1)):
            validate_case({**BASE, **controls})
        for trials, seed in ((1, 0), (256, (1 << 64)-1)):
            case = case_for(trials=trials, challenge_seed=seed)
            validate_capture(fixture(case), case, "fixture-run", 2, 1)

    def test_malformed_cases_and_irrelevant_controls(self):
        for key, values in {"backend": (None, "auto", "cpu", True), "device": (True, 0, -2, "-1"),
            "mode": ("verify", "", None), "m": (0, 513, True, 1., "2"), "n": (0, 513, -1),
            "k": (0, 65538, True), "modulus": (1, 257, 251., True), "trials": (0, 257, False, 4.),
            "challenge_seed": (-1, 1 << 64, True, "3135")}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_case({**BASE, key: value})
        for case in ({**BASE, "surprise": 1}, {key: value for key, value in BASE.items() if key != "m"},
                     {**BASE, "m": 512, "n": 512, "k": 513},
                     {**case_for("full_reference"), "trials": 1}, {**case_for("full_reference"), "challenge_seed": 1},
                     {**case_for(backend="hip-direct"), "device": -1}):
            with self.assertRaises(ValueError):
                validate_case(case)

    def test_manifest_duplicates_unknown_keys_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cases.json"
            for doc in ({"schema": "wrong", "cases": [BASE]}, {"schema": "rns8_finite_verify_cases_v1", "cases": []},
                        {"schema": "rns8_finite_verify_cases_v1", "cases": [BASE, BASE]},
                        {"schema": "rns8_finite_verify_cases_v1", "cases": [BASE], "extra": 1}):
                path.write_text(json.dumps(doc), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_cases(path)
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                strict_json(text)

    def test_tracked_manifests_cover_matched_controls(self):
        root = case_root()/"finite-verification"
        for name, count in (("matched.json", 24), ("k-boundary.json", 6)):
            cases = load_cases(root/name)
            self.assertEqual(len(cases), count)
            for backend in ("cpu-reference", "hip-direct"):
                for modulus in {case["modulus"] for case in cases}:
                    self.assertEqual({case["mode"] for case in cases if case["backend"] == backend and case["modulus"] == modulus}, set(MODES))

    def test_deterministic_recipe_edges(self):
        for q in (2, 6, 251, 256):
            case = case_for(modulus=q)
            self.assertEqual(dataset(case), dataset(case))
            for values in dataset(case):
                self.assertEqual(values[:2], bytes([q-1, 0]))
                self.assertTrue(all(0 <= value < q for value in values))
        self.assertNotEqual(dataset(BASE), dataset({**BASE, "modulus": 251}))

    def test_every_contract_control_is_bound(self):
        case = case_for()
        for key, value in contract(case, "fixture-run", 2, 1).items():
            capture = fixture(case)
            capture[key] = None if value is not None else "mismatch"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(capture, case, "fixture-run", 2, 1)
        for key in ("m", "n", "trials", "device_actual", "repeats", "dataset_seed", "memory_budget_bytes"):
            capture = fixture(); capture[key] = float(capture[key])
            with self.assertRaises(ValueError):
                validate_capture(capture, BASE, "fixture-run", 2, 1)

    def test_malformed_samples_memory_identity_and_checksums(self):
        mutations = [lambda c: c.pop("build_manifest"), lambda c: c.update(target="gfx942"),
            lambda c: c.update(a_checksum="0"*16), lambda c: c.update(output_checksum="42"),
            lambda c: c.update(logical_live_payload_envelope_bytes=BUDGET+1),
            lambda c: c.update(resident_payload_bytes=0), lambda c: c.update(plan_workspace_bytes=True),
            lambda c: c.update(median_wall_ns=200), lambda c: c.update(calls_per_second=1),
            lambda c: c["samples"][0].update(wall_ns=0), lambda c: c["samples"][0].update(wall_ns=True),
            lambda c: c["samples"][0].update(exact_reference_passed=False), lambda c: c.update(samples=[]),
            lambda c: c["sustained"].update(calls=15), lambda c: c["sustained"].update(elapsed_ns=1),
            lambda c: c["sustained"].update(mean_wall_ns=float("inf")),
            lambda c: c["sustained"].update(independent_audit_inside_interval=True),
            lambda c: c["sustained"].update(final_exact_reference_passed=False),
            lambda c: c["build_manifest"].update(source_digest="not a digest")]
        for mutation in mutations:
            capture = fixture(); mutation(capture)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_capture(capture, BASE, "fixture-run", 2, 1)
        capped = fixture(); capped["sustained"].update(calls=4096, elapsed_ns=4096000, mean_wall_ns=1000., stop_reason="call_cap")
        validate_capture(capped, BASE, "fixture-run", 2, 1)

    def test_replay_probability_lifetime_and_diagnostic_provenance(self):
        for key, value in {"passed": False, "sampler": True, "replay_seed": 0, "trials": 3,
            "workspace_bytes": 1, "ideal_failure_bound_exponent": 0, "ideal_failure_probability": 0.,
            "fresh_entropy": True, "integer_equality_certified": True,
            "matrix_instance_ids": [1, 1, 2], "generations": [0, 1, 1], "probability_model": "field_q_power"}.items():
            capture = fixture(); capture["verification"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_capture(capture, BASE, "fixture-run", 2, 1)
        capture = fixture(case_for("full_reference")); capture["verification"] = {}
        with self.assertRaises(ValueError):
            validate_capture(capture, case_for("full_reference"), "fixture-run", 2, 1)
        for dirty, digest, garner in ((True, "d"*64, "OFF"), (False, "unavailable", "OFF"), (False, "d"*64, "ON")):
            capture = fixture(); capture["build_manifest"].update(source_dirty=dirty, dependency_digest=digest)
            capture["build_manifest"]["configuration"] += f"\ngarner_table={garner}"
            validate_capture(capture, BASE, "fixture-run", 2, 1)


class Collection(unittest.TestCase):
    def setUp(self):
        # Host metadata may itself invoke subprocess.run (uname -p on Linux).
        host = patch("collect_finite_verify.platform.platform", return_value="synthetic-platform")
        host.start()
        self.addCleanup(host.stop)

    def scenario(self, mutation=None, runs=2, status=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); executable = root/"synthetic-executable"; executable.write_bytes(b"fixture only")
            cases = root/"cases.json"; cases.write_text(json.dumps(dict(schema="rns8_finite_verify_cases_v1", cases=[BASE])))
            args = argparse.Namespace(exe=str(executable), cases=str(cases), output=str(root/"output"), runs=runs, repeats=2, warmups=1, timeout=10)
            count = 0
            def child(command, **kwargs):
                nonlocal count
                capture = fixture(BASE, command[-1]); count += 1
                if mutation:
                    mutation(capture, count, executable)
                return subprocess.CompletedProcess(command, status, json.dumps(capture), "synthetic stderr retained")
            with patch("collect_finite_verify.subprocess.run", side_effect=child):
                if mutation or status:
                    with self.assertRaises(ValueError):
                        collect(args)
                else:
                    collect(args)
            records = [json.loads(path.read_text()) for path in sorted((root/"output").glob("*.record.json"))]
            self.assertEqual(len(records), count)
            self.assertEqual(len({record["run_id"] for record in records}), count)
            self.assertEqual(records[-1]["validated"], not bool(mutation or status))
            for path in (root/"output").glob("*.record.json"):
                self.assertTrue(path.with_name(path.name.replace(".record.json", ".stdout.json")).exists())
            for record in records:
                self.assertFalse(record["promotion_eligible"])
                self.assertEqual(set(record["tool_sha256"]), {"collect_finite_verify.py", "build_manifest.py"})
                self.assertTrue(all(len(value) == 64 for value in record["tool_sha256"].values()))
            return records

    def test_collection_independent_processes_and_receipts(self):
        records = self.scenario()
        self.assertTrue(all(record["provenance_complete"] for record in records))

    def test_collection_metadata_build_runtime_and_artifact_mismatch(self):
        for mutation in (lambda c, i, exe: c.update(run_id="wrong"),
                         lambda c, i, exe: c.update(device_actual=0),
                         lambda c, i, exe: c.update(trials=3),
                         lambda c, i, exe: c.update(output_checksum="0"*16) if i == 2 else None,
                         lambda c, i, exe: c["build_manifest"].update(source_commit="e"*40) if i == 2 else None,
                         lambda c, i, exe: c.update(device_name="different") if i == 2 else None,
                         lambda c, i, exe: exe.write_bytes(b"changed executable")):
            with self.subTest(mutation=mutation):
                self.scenario(mutation)
        self.scenario(status=77)

    def test_timeout_preserves_partial_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); executable = root/"fixture"; executable.write_bytes(b"fixture only")
            cases = root/"cases.json"; cases.write_text(json.dumps(dict(schema="rns8_finite_verify_cases_v1", cases=[BASE])))
            args = argparse.Namespace(exe=str(executable), cases=str(cases), output=str(root/"output"), runs=1, repeats=2, warmups=1, timeout=1)
            with patch("collect_finite_verify.subprocess.run", side_effect=subprocess.TimeoutExpired("fixture", 1, output=b"partial stdout", stderr=b"partial stderr")):
                with self.assertRaises(subprocess.TimeoutExpired):
                    collect(args)
            self.assertEqual(next((root/"output").glob("*.stdout.json")).read_bytes(), b"partial stdout")
            self.assertEqual(next((root/"output").glob("*.stderr.txt")).read_bytes(), b"partial stderr")
            self.assertFalse(json.loads(next((root/"output").glob("*.record.json")).read_text())["validated"])


@unittest.skipUnless(os.environ.get("RNS8_FINITE_VERIFY_EXE"), "set RNS8_FINITE_VERIFY_EXE for compile-only CLI fixtures")
class CompiledCLI(unittest.TestCase):
    def test_help_and_pre_context_rejections(self):
        exe = os.environ["RNS8_FINITE_VERIFY_EXE"]
        self.assertEqual(subprocess.run([exe, "--help"], capture_output=True).returncode, 0)
        valid = ["hip-direct", "63", "verify_only", "2", "3", "65", "250", "4", "2", "1", "3135", "fixture"]
        invalid = [valid[:-1], valid+["extra"]]
        for index, value in ((0, "auto"), (1, "-1"), (2, "bad"), (3, "0"), (4, "513"), (5, "65538"),
                             (6, "257"), (7, "0"), (8, "0"), (9, "0"), (10, "18446744073709551616"), (11, "bad/id")):
            args = list(valid); args[index] = value; invalid.append(args)
        invalid.append(["cpu-reference", "0", *valid[2:]])
        invalid.append([*valid[:2], "full_reference", *valid[3:]])
        for args in invalid:
            with self.subTest(args=args):
                # Every command is malformed before context creation; no GPU job.
                self.assertEqual(subprocess.run([exe, *args], capture_output=True, timeout=10).returncode, 1)


if __name__ == "__main__":
    unittest.main()
