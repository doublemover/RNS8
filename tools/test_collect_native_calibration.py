"""Authored collector fixtures mock all children; optional CLI checks never run GPU work."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from collect_native_calibration import (
    CANDIDATES, QUEUED, capture_contract, epilogue_contract, main, narrow_contract,
    scheduling_contract, strict_json, valid_sustained, validate_capture, validate_case,
)
from test_capture_identity_support import synthetic_identity


BASE = dict(candidate="wmma", m=17, n=19, k=257, prefix=3, group=2, unsigned_inputs=False)


def case_for(candidate, **changes):
    case = {**BASE, "candidate": candidate}
    if candidate not in ("wmma", *QUEUED):
        case["group"] = 1
    if candidate.startswith("narrow_"):
        case["prefix"] = 1
    return validate_case({**case, **changes})


def capture_for(case, run_id="fixture-run", repeats=3, warmups=1):
    capture = capture_contract(case, run_id, repeats, warmups)
    if case["candidate"] == "queue_prepared":
        capture.update(queue_prepare_ns=2000.25, queue_descriptor_storage_bytes=1024)
    synthetic_identity(capture)
    capture.update(device="synthetic mocked GPU", target="gfx1100", runtime_version=70100000,
                   timing_scope="synthetic scoped timing", cache_state="synthetic repeated inputs",
                   sustained=dict(calls=16, elapsed_ns=320000000, mean_call_ns=20000000,
                                  final_exact_reference_passed=True, scope="synthetic final readback"),
                   samples=[dict(host_call_ns=1000.25, uninstrumented_host_call_ns=900.5,
                                 exact_reference_passed=True, uninstrumented_exact_reference_passed=True,
                                 events=[dict(label="synthetic event", elapsed_ns=500.25)]) for _ in range(repeats)])
    return capture


class CaseTest(unittest.TestCase):
    def test_prepared_queue_requires_setup_cost_and_storage(self):
        case = case_for("queue_prepared")
        for key in ("queue_prepare_ns", "queue_descriptor_storage_bytes"):
            for value in (None, False, 0, -1, "100", float("nan"), float("inf")):
                with self.subTest(key=key, value=value):
                    capture = capture_for(case)
                    capture[key] = value
                    with self.assertRaises(ValueError):
                        validate_capture(capture, case, "fixture-run", 3, 1)
            capture = capture_for(case)
            del capture[key]
            with self.assertRaises(ValueError):
                validate_capture(capture, case, "fixture-run", 3, 1)

    def test_all_modes_groups_and_signedness(self):
        for candidate in CANDIDATES:
            for unsigned in (False, True):
                if candidate == "narrow_scalar" and unsigned:
                    continue
                for group in ((1, 2, 4) if candidate in ("wmma", *QUEUED) else (1,)):
                    with self.subTest(candidate=candidate, unsigned=unsigned, group=group):
                        case = case_for(candidate, group=group, unsigned_inputs=unsigned)
                        self.assertEqual(validate_case(case), case)
                        validate_capture(capture_for(case), case, "fixture-run", 3, 1)

    def test_worker_defaults_are_normalized_without_mutation(self):
        for candidate in QUEUED:
            original = {**BASE, "candidate": candidate}
            self.assertEqual(validate_case(original)["workers"], 32)
            self.assertNotIn("workers", original)
            for workers in (1, 2, 32, 1024):
                self.assertEqual(case_for(candidate, workers=workers)["workers"], workers)
        for candidate in set(CANDIDATES) - set(QUEUED):
            with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, "only to queue/queue_prepared/split_k"):
                case_for(candidate, workers=32)

    def test_schedule_count_clamping_and_split_boundaries(self):
        for candidate in QUEUED:
            for group in (1, 2, 4):
                for unsigned in (False, True):
                    for k in (1, 2, 3, 65535, 65536, 65537):
                        for workers in (1, 32, 1024):
                            case = case_for(candidate, m=1, n=17, k=k, prefix=5,
                                            group=group, unsigned_inputs=unsigned, workers=workers)
                            contract = scheduling_contract(case)
                            splits = min(k, 3) if candidate == "split_k" else 1
                            count = 2 * ((5 + group - 1) // group) * splits
                            self.assertEqual(contract["queue_descriptors"], count)
                            self.assertEqual(contract["splits"], splits)
                            self.assertEqual(contract["requested_workers"], workers)
                            self.assertEqual(contract["effective_workers"], min(count, workers))
                            self.assertEqual(contract["queue_kernel_selection"],
                                             f"homogeneous_group_{group}_" + ("unsigned" if unsigned else "signed"))
        for candidate in set(CANDIDATES) - set(QUEUED):
            contract = scheduling_contract(case_for(candidate))
            self.assertEqual(contract, dict(splits=1, requested_workers=0, effective_workers=0,
                                           queue_descriptors=0, queue_kernel_selection="not_applicable"))

    def test_shape_boundaries(self):
        for key, high in (("m", 1024), ("n", 1024), ("k", 65537), ("prefix", 20)):
            for value in (1, high):
                validate_case({**BASE, "m": 1, "n": 1, key: value})
        validate_case({**BASE, "m": 1024, "n": 1024, "k": 1024})
        with self.assertRaises(ValueError):
            validate_case({**BASE, "m": 1024, "n": 1024, "k": 1025})

    def test_malformed_cases(self):
        malformed = [None, [], "case", {}, {**BASE, "extra": 1}, {k: v for k, v in BASE.items() if k != "k"}]
        for key in ("m", "n", "k", "prefix", "group"):
            for value in (None, True, False, 0, -1, 1.0, "1", [], {}, 10**100):
                malformed.append({**BASE, key: value})
        for value in (None, "unknown", [], {}, 1, False):
            malformed.append({**BASE, "candidate": value})
        for value in (None, 0, 1, "false", [], {}):
            malformed.append({**BASE, "unsigned_inputs": value})
        malformed += [{**BASE, "m": 1025}, {**BASE, "n": 1025}, {**BASE, "k": 65538},
                      {**BASE, "prefix": 21}, {**BASE, "group": 3}]
        for candidate in QUEUED:
            for value in (None, True, False, 0, -1, 1025, 1.0, "32", [], {}):
                malformed.append({**BASE, "candidate": candidate, "workers": value})
        for candidate in (c for c in CANDIDATES if c not in ("wmma", *QUEUED)):
            malformed.append({**case_for(candidate), "group": 2})
        for candidate in (c for c in CANDIDATES if c.startswith("narrow_")):
            malformed += [{**case_for(candidate), "prefix": 2}, {**case_for(candidate), "k": 32769}]
        malformed.append({**case_for("narrow_scalar"), "unsigned_inputs": True})
        for case in malformed:
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_case(case)

    def test_narrow_and_epilogue_contracts(self):
        for unsigned in (False, True):
            for candidate in (c for c in CANDIDATES if c.startswith("narrow_")):
                if candidate == "narrow_scalar" and unsigned:
                    continue
                case = case_for(candidate, unsigned_inputs=unsigned, k=32768)
                contract = narrow_contract(case)
                self.assertEqual(contract["abs_partial_sum_bound"], 32768 * (255 if unsigned else 128)**2)
                self.assertLessEqual(contract["abs_partial_sum_bound"], (1 << 31) - 1)
                self.assertEqual(contract["selected_rns_prefix"] > 0, candidate == "narrow_rns")
            for candidate in (c for c in CANDIDATES if c.startswith("epilogue_")):
                contract = epilogue_contract(case_for(candidate, unsigned_inputs=unsigned))
                self.assertEqual(contract["expression"]["alpha"], (1 << 63) - 1 if unsigned else -(1 << 63))
                self.assertEqual(contract["expression"]["beta"], (1 << 63) - 1)
                self.assertEqual(bool(contract["materialized_intermediate_bytes"]), candidate == "epilogue_unfused")


class CaptureTest(unittest.TestCase):
    def test_every_required_metadata_field_is_checked_for_all_modes(self):
        for candidate in CANDIDATES:
            case = case_for(candidate)
            capture = capture_for(case)
            for key in capture_contract(case, "fixture-run", 3, 1):
                with self.subTest(candidate=candidate, key=key):
                    missing = copy.deepcopy(capture)
                    del missing[key]
                    with self.assertRaises(ValueError):
                        validate_capture(missing, case, "fixture-run", 3, 1)
                    wrong = copy.deepcopy(capture)
                    wrong[key] = "wrong" if type(wrong[key]) is not str else ""
                    with self.assertRaises(ValueError):
                        validate_capture(wrong, case, "fixture-run", 3, 1)

    def test_integer_controls_cannot_be_bools_or_floats(self):
        case = case_for("queue", m=1, n=1, prefix=1, group=1, workers=1)
        for key in ("m", "n", "group", "prefix", "requested_workers", "effective_workers", "queue_descriptors", "splits"):
            for value in (True, 1.0):
                capture = capture_for(case)
                capture[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_capture(capture, case, "fixture-run", 3, 1)

    def test_malformed_capture_and_measurements(self):
        case = case_for("wmma")
        for bad in (None, [], True, "json", 1):
            with self.assertRaises(ValueError):
                validate_capture(bad, case, "fixture-run", 3, 1)
        changes = [dict(samples=None), dict(samples={}), dict(samples=[None]*3), dict(samples=[]),
                   dict(sustained=None), dict(runtime_version=True), dict(runtime_version=0),
                   dict(device=""), dict(target=None), dict(timing_scope=""), dict(cache_state=None)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_capture({**capture_for(case), **change}, case, "fixture-run", 3, 1)
        for key in ("host_call_ns", "uninstrumented_host_call_ns"):
            for bad in (None, 0, -1, True, "1", float("nan"), float("inf"), 10**1000):
                capture = capture_for(case)
                capture["samples"][0][key] = bad
                with self.subTest(key=key, bad=str(bad)[:20]), self.assertRaises(ValueError):
                    validate_capture(capture, case, "fixture-run", 3, 1)
        for bad in (None, {}, [None], [dict(label="", elapsed_ns=1)], [dict(label="x", elapsed_ns=-1)],
                    [dict(label="x", elapsed_ns=True)], [dict(label="x")]):
            capture = capture_for(case)
            capture["samples"][0]["events"] = bad
            with self.assertRaises(ValueError):
                validate_capture(capture, case, "fixture-run", 3, 1)
        capture = capture_for(case)
        capture["samples"][0]["events"][0]["elapsed_ns"] = None
        validate_capture(capture, case, "fixture-run", 3, 1)

    def test_sustained_boundaries(self):
        value = capture_for(case_for("wmma"))["sustained"]
        self.assertTrue(valid_sustained(value))
        self.assertTrue(valid_sustained({**value, "calls": 4096, "elapsed_ns": 4096, "mean_call_ns": 1}))
        for change in ({"calls": True}, {"calls": 15}, {"calls": 4097}, {"calls": 16.0},
                       {"elapsed_ns": float("nan")}, {"elapsed_ns": 10**1000}, {"mean_call_ns": 0},
                       {"mean_call_ns": True}, {"final_exact_reference_passed": False}, {"scope": ""},
                       {"elapsed_ns": 1600, "mean_call_ns": 100}):
            self.assertFalse(valid_sustained({**value, **change}))

    def test_provenance_is_structured_and_dirty_diagnostics_are_allowed(self):
        case = case_for("queue")
        capture = capture_for(case)
        for change in ({"version": 1}, {"source_dirty": 1}, {"source_commit": "unknown"},
                       {"source_digest": "x"}, {"external_dependencies": []},
                       {"dependency_digest": "bad"}, {"configuration": ""}):
            changed = copy.deepcopy(capture)
            changed["build_manifest"].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "provenance"):
                validate_capture(changed, case, "fixture-run", 3, 1)
        capture["build_manifest"].update(source_dirty=True, dependency_digest="unavailable")
        validate_capture(capture, case, "fixture-run", 3, 1)

    def test_garner_on_and_off_are_valid_diagnostics(self):
        case = case_for("queue")
        for enabled in ("ON", "OFF", "TRUE", "1"):
            capture = capture_for(case)
            capture["build_manifest"]["configuration"] += "\ngarner_table=" + enabled
            validate_capture(capture, case, "fixture-run", 3, 1)

    def test_strict_json_rejects_duplicates_and_nonfinite_numbers(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
            with self.assertRaises(ValueError):
                strict_json(text)


class CollectorTest(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "temp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exe = self.root / "mock.exe"
        self.exe.write_bytes(b"synthetic mocked child; never executed")
        self.cases = self.root / "cases.json"
        self.output = self.root / "records"

    def collect(self, cases, mutate=None, runs=1, child_action=None):
        self.cases.write_text(json.dumps(cases), encoding="utf-8")
        argv = ["collector", "--exe", str(self.exe), "--cases", str(self.cases), "--output", str(self.output),
                "--runs", str(runs), "--repeats", "3", "--warmups", "1"]
        calls = []
        def child(command, **kwargs):
            calls.append(command)
            case = validate_case(dict(candidate=command[1], m=int(command[2]), n=int(command[3]),
                                      k=int(command[4]), prefix=int(command[5]), group=int(command[6]),
                                      unsigned_inputs=bool(int(command[7])),
                                      **({"workers": int(command[11])} if len(command) == 12 else {})))
            capture = capture_for(case, command[10])
            if mutate:
                mutate(capture, len(calls))
            if child_action:
                return child_action(command, capture)
            return subprocess.CompletedProcess(command, 0, json.dumps(capture), "mocked stderr")
        with patch.object(sys, "argv", argv), patch("collect_native_calibration.subprocess.run", side_effect=child), contextlib.redirect_stdout(io.StringIO()):
            result = main()
        return result, calls

    def records(self):
        return [json.loads(p.read_text(encoding="utf-8")) for p in self.output.glob("*.json")]

    def test_all_modes_end_to_end_and_distinct_worker_identity(self):
        cases = [case_for(name) for name in CANDIDATES]
        cases += [case_for("queue", workers=1), case_for("queue", workers=1024),
                  case_for("split_k", m=1, n=1, k=2, group=4, prefix=1, workers=1024)]
        result, calls = self.collect(cases, runs=2)
        self.assertEqual(result, 0)
        records = self.records()
        self.assertEqual(len(records), 2 * len(cases))
        self.assertEqual(len({r["run_id"] for r in records}), len(records))
        self.assertEqual(len({r["campaign_id"] for r in records}), 1)
        self.assertEqual({r["independent_repeat"] for r in records}, {0, 1})
        for record in records:
            self.assertEqual(record["schema"], "rns8_native_collection_v2")
            self.assertTrue(record["validated"])
            self.assertTrue(record["provenance_complete"])
            self.assertFalse(record["promotion_eligible"])
            import hashlib
            for name, digest in record["collector_sources_sha256"].items():
                self.assertEqual(digest, hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
            self.assertEqual(record["collector_sha256"], record["collector_sources_sha256"]["collect_native_calibration.py"])
            self.assertEqual(record["command"][10], record["run_id"])
            self.assertIn(record["command"], calls)
            self.assertEqual(record["comparison_identity"]["case"], record["case"])
            self.assertEqual(record["comparison_identity"]["artifact_sha256"], record["artifact_sha256"])
        queue_identities = [r["comparison_identity"] for r in records if r["case"]["candidate"] == "queue"]
        self.assertEqual({r["requested_workers"] for r in queue_identities}, {1, 32, 1024})
        split = next(r for r in records if r["case"].get("workers") == 1024 and r["case"]["candidate"] == "split_k")
        self.assertEqual(split["comparison_identity"]["effective_workers"], 2)

    def test_omitted_workers_get_explicit_command_and_identity(self):
        result, calls = self.collect([{**BASE, "candidate": "queue"}])
        self.assertEqual(result, 0)
        self.assertEqual(calls[0][11], "32")
        self.assertEqual(self.records()[0]["case"]["workers"], 32)

    def test_worker_and_all_contract_mismatches_retain_raw_capture(self):
        for candidate in CANDIDATES:
            case = case_for(candidate)
            for key in capture_contract(case, "fixture-run", 3, 1):
                with self.subTest(candidate=candidate, key=key):
                    self.output = self.root / f"mismatch-{candidate}-{key}"
                    def mutate(capture, count):
                        capture[key] = None if capture[key] is not None else "wrong"
                    with self.assertRaisesRegex(RuntimeError, "contract/reference mismatch"):
                        self.collect([case], mutate=mutate)
                    record, = self.records()
                    self.assertFalse(record["validated"])
                    self.assertIn(key, record["validation_error"])
                    self.assertIn("stdout", record)
                    self.assertEqual(record["stderr"], "mocked stderr")

    def test_changed_build_device_or_runtime_is_retained_and_rejected(self):
        for key in ("device", "target", "runtime_version", "build_manifest"):
            self.output = self.root / ("changed-" + key)
            def mutate(capture, count):
                if count == 2:
                    if key == "build_manifest":
                        capture[key]["source_commit"] = "d" * 40
                    elif key == "runtime_version":
                        capture[key] += 1
                    else:
                        capture[key] += " changed"
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "identity changed"):
                self.collect([BASE], mutate=mutate, runs=2)
            self.assertEqual(len(self.records()), 2)
            self.assertEqual(sum(r["validated"] for r in self.records()), 1)

    def test_dirty_or_incomplete_provenance_stays_diagnostic(self):
        def mutate(capture, count):
            capture["build_manifest"].update(source_dirty=True, dependency_digest="unavailable")
        self.collect([BASE], mutate=mutate)
        record, = self.records()
        self.assertTrue(record["validated"])
        self.assertFalse(record["provenance_complete"])
        self.assertFalse(record["promotion_eligible"])

    def test_garner_diagnostics_preserve_release_ineligibility(self):
        for enabled, complete in (("ON", False), ("OFF", True)):
            self.output = self.root / ("garner-" + enabled)
            def mutate(capture, count):
                capture["build_manifest"]["configuration"] += "\ngarner_table=" + enabled
            self.collect([BASE], mutate=mutate)
            record, = self.records()
            self.assertTrue(record["validated"])
            self.assertEqual(record["provenance_complete"], complete)
            self.assertFalse(record["promotion_eligible"])
            self.assertIn("garner_table=" + enabled, record["comparison_identity"]["build_manifest"]["configuration"])

    def test_timeout_partial_output_failure_and_invalid_json(self):
        def timeout(command, capture):
            raise subprocess.TimeoutExpired(command, 1, output=b"partial stdout", stderr=b"partial stderr")
        actions = [timeout, lambda c, j: subprocess.CompletedProcess(c, 3, "failure output", "failure stderr"),
                   lambda c, j: subprocess.CompletedProcess(c, 0, "not json", ""),
                   lambda c, j: subprocess.CompletedProcess(c, 0, "[]", ""),
                   lambda c, j: subprocess.CompletedProcess(c, 0, '{"x":1,"x":2}', ""),
                   lambda c, j: subprocess.CompletedProcess(c, 0, '{"x":NaN}', "")]
        for index, action in enumerate(actions):
            self.output = self.root / f"failure-{index}"
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                self.collect([BASE], child_action=action)
            record, = self.records()
            self.assertFalse(record["validated"])
            self.assertNotIn("comparison_identity", record)
            if index == 0:
                self.assertEqual(record["reason"], "timeout")
                self.assertEqual(record["stdout"], "partial stdout")
                self.assertEqual(record["stderr"], "partial stderr")

    def test_executable_changes_after_child_are_rejected(self):
        def action(command, capture):
            self.exe.write_bytes(b"changed binary")
            return subprocess.CompletedProcess(command, 0, json.dumps(capture), "")
        with self.assertRaisesRegex(RuntimeError, "executable changed"):
            self.collect([BASE], child_action=action)
        self.assertFalse(self.records()[0]["validated"])

    def test_invalid_cases_fail_before_launch_or_output_creation(self):
        for index, cases in enumerate((None, {}, [], [BASE] * 65, [BASE, BASE],
                                              [{**BASE, "candidate": "queue"}, {**BASE, "candidate": "queue", "workers": 32}],
                                              [{**BASE, "workers": 32}], [{**BASE, "m": True}])):
            self.output = self.root / f"bad-input-{index}"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.collect(cases)
            self.assertFalse(self.output.exists())


@unittest.skipUnless(os.environ.get("RNS8_NATIVE_CALIBRATION_EXE"), "set RNS8_NATIVE_CALIBRATION_EXE for compiled CLI rejection/help checks")
class CompiledCliTest(unittest.TestCase):
    def invoke(self, arguments):
        return subprocess.run([os.environ["RNS8_NATIVE_CALIBRATION_EXE"], *arguments],
                              capture_output=True, text=True, check=False, timeout=30)

    def test_help_lists_all_modes_and_worker_contract(self):
        result = self.invoke(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in CANDIDATES:
            self.assertIn(name, result.stdout)
        self.assertIn("1..1024", result.stdout)
        self.assertIn("default 32", result.stdout)

    def test_malformed_and_irrelevant_workers_fail_before_device_work(self):
        for candidate in CANDIDATES:
            case = case_for(candidate)
            base = [candidate, *(str(case[k]) for k in ("m", "n", "k", "prefix", "group")), "0", "1", "1", "cli-reject"]
            workers = ("0", "1025", "-1", "1.0", "+1", "", "4294967296", "32x") if candidate in QUEUED else ("32",)
            for value in workers:
                with self.subTest(candidate=candidate, workers=value):
                    result = self.invoke([*base, value])
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("numeric argument" if candidate in QUEUED else "workers applies only", result.stderr)
        result = self.invoke([])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
