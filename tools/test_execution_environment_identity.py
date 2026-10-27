"""Real request-emitter integration; all timing values are synthetic, no GPU."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_request import request_identity, request_environment_identity, request_mismatch
from observation_cohorts import aggregate_captures, comparison_environment, evidence
from perf_variance_report import build_report, execution_cohort
from promotion_ledger import capture_entry
from test_perf_variance_report import capture_with_timings, write_capture, write_review_report


class EnvironmentIdentity(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.executable = self.root / "synthetic-artifact-never-executed.bin"
        self.executable.write_bytes(b"authored request protocol fixture; never executable evidence")

    def capture(self, name, environment=None, overrides=None):
        c = capture_with_timings([100.0] * 9)
        request = request_identity([str(self.executable), "--report", str(self.root / name)],
                                   overrides=overrides, environ=environment or {})
        c["request_identity"] = request
        c["artifact_capture"]["executable_sha256"] = request["executable_sha256"]
        return c

    def report(self, *captures):
        paths = [write_capture(self.root / f"capture-{i}.json", c) for i, c in enumerate(captures)]
        return build_report(paths, min_repeats=9, min_runs=2, require_multi_run=True), paths

    def test_real_emitter_same_environment_and_distinct_output_paths_qualify(self):
        a = self.capture("a.json", {"OMP_PROC_BIND": "close"})
        b = self.capture("b.json", {"OMP_PROC_BIND": "close"})
        self.assertNotEqual(a["request_identity"]["identity_sha256"], b["request_identity"]["identity_sha256"])
        self.assertEqual(execution_cohort(a), execution_cohort(b))
        report, _ = self.report(a, b)
        self.assertEqual(report["blocked_count"], 0)
        self.assertTrue(all(r["promotion_ready"] and r["distinct_run_count"] == 2 for r in report["entries"]))
        self.assertTrue(evidence(aggregate_captures([a, b])[0])["qualified"])
        # A recorded empty object is evidence; absent evidence is not empty.
        empty = [self.capture(name) for name in ("empty-a", "empty-b")]
        self.assertEqual(self.report(*empty)[0]["blocked_count"], 0)

    def test_real_emitter_runtime_controls_separate_all_consumers(self):
        for key, values in (("OMP_PROC_BIND", ("close", "spread")), ("OMP_NUM_THREADS", ("2", "3")),
                            ("HIP_VISIBLE_DEVICES", ("0", "1")), ("HSA_ENABLE_SDMA", ("0", "1")),
                            ("LD_LIBRARY_PATH", ("/fixture/a", "/fixture/b"))):
            with self.subTest(control=key):
                a, b = [self.capture(str(i), {key: value}) for i, value in enumerate(values)]
                self.assertEqual(request_mismatch(a, b["request_identity"]), "execution request differs: environment")
                self.assertNotEqual(execution_cohort(a), execution_cohort(b))
                self.assertNotEqual(comparison_environment(a), comparison_environment(b))
                self.assertNotEqual(comparison_environment(a, cpu_anchor=True), comparison_environment(b, cpu_anchor=True))
                report, _ = self.report(a, b)
                self.assertEqual(report["group_count"], 2)
                self.assertTrue(all(r["distinct_run_count"] == 1 and not r["promotion_ready"] for r in report["entries"]))
                views = aggregate_captures([a, b])
                self.assertEqual(len(views), 2)
                self.assertTrue(all(not evidence(v)["qualified"] and evidence(v)["independent_runs"] == 1 for v in views))

    def test_explicit_controls_outside_known_prefixes_remain_bound(self):
        a = self.capture("a", overrides={"CUSTOM_NEW_BENCHMARK_CONTROL": "one"})
        b = self.capture("b", overrides={"CUSTOM_NEW_BENCHMARK_CONTROL": "two"})
        self.assertEqual(a["request_identity"]["environment"], b["request_identity"]["environment"])
        self.assertNotEqual(execution_cohort(a), execution_cohort(b))
        self.assertEqual(self.report(a, b)[0]["blocked_count"], 2)

    def test_duplicate_execution_and_legacy_or_malformed_evidence_fail_closed(self):
        a = self.capture("a", {"OMP_PROC_BIND": "close"})
        report, _ = self.report(a, deepcopy(a))
        self.assertTrue(all(r["distinct_run_count"] == 1 and not r["promotion_ready"] for r in report["entries"]))
        mutations = (
            lambda r: r.pop("environment"),
            lambda r: r.update(environment=None),
            lambda r: r.update(environment=[]),
            lambda r: r.update(environment={"OMP_PROC_BIND": 1}),
            lambda r: r.update(environment={"OMP_PROC_BIND": True}),
            lambda r: r.update(environment={"": "value"}),
            lambda r: r.update(schema="unsupported_request_version"),
            lambda r: r.pop("overrides_sha256"),
            lambda r: r.update(identity_sha256="0" * 64),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                bad = deepcopy(a); mutate(bad["request_identity"])
                other = self.capture("other"); other["request_identity"] = deepcopy(bad["request_identity"])
                self.assertIsNone(execution_cohort(bad))
                report, _ = self.report(bad, other)
                self.assertTrue(all(not r["promotion_ready"] and r["distinct_run_count"] == 0 for r in report["entries"]))
                self.assertTrue(all(any("environment" in reason or "request_identity" in reason for reason in r["blockers"]) for r in report["entries"]))
                self.assertTrue(all(not evidence(v)["qualified"] for v in aggregate_captures([bad, other])))

    def test_tampering_and_unbound_claimed_digest_cannot_override_actual_object(self):
        a = self.capture("a", {"OMP_PROC_BIND": "close"})
        forged = deepcopy(a); forged["request_identity"]["environment"]["OMP_PROC_BIND"] = "spread"
        self.assertEqual(request_environment_identity(forged["request_identity"])[1], "execution_request_identity_mismatch")
        self.assertIsNone(execution_cohort(forged))
        # Even a sealed extra claimed digest is ignored: the recorded map wins.
        b = self.capture("b", {"OMP_PROC_BIND": "spread"})
        for c in (a, b):
            request = c["request_identity"]; request["environment_sha256"] = "f" * 64
            payload = {k: v for k, v in request.items() if k != "identity_sha256"}
            request["identity_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertNotEqual(execution_cohort(a), execution_cohort(b))
        self.assertEqual(self.report(a, b)[0]["blocked_count"], 2)

    def test_variance_report_blockers_and_stale_cohorts_reach_the_ledger(self):
        a = self.capture("a", {"OMP_PROC_BIND": "close"})
        b = self.capture("b", {"OMP_PROC_BIND": "spread"})
        report, paths = self.report(a, b)
        review_path = write_review_report(self.root / "review.json", paths[0], 1.2)
        reviewed = json.loads(review_path.read_text())["groups"][0]["candidates"][0]
        variance = report["entries"][0]
        entry = capture_entry(paths[0], reviewed, variance, variance_gate_required=True)
        self.assertIn("variance_blocker:multi_run_variance_missing", entry["promotion_blockers"])
        # Old keys that treated missing environment digests as equal are stale.
        stale = deepcopy(variance); stale["execution_cohort"] = "0" * 64
        stale["promotion_ready"] = True; stale["blockers"] = []
        rejected = capture_entry(paths[0], reviewed, stale, variance_gate_required=True)
        self.assertIn("variance_capture_provenance_mismatch", rejected["promotion_blockers"])
        legacy = deepcopy(a); legacy["request_identity"].pop("environment")
        legacy_report, legacy_paths = self.report(legacy)
        forged = deepcopy(legacy_report["entries"][0]); forged["promotion_ready"] = True; forged["blockers"] = []
        self.assertIsNone(forged["execution_cohort"])
        rejected = capture_entry(legacy_paths[0], reviewed, forged, variance_gate_required=True)
        self.assertIn("variance_capture_provenance_mismatch", rejected["promotion_blockers"])


if __name__ == "__main__":
    unittest.main()
