"""Authored cache publication controls; synthetic evidence, no hardware claims."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cache_promotion as contract
from benchmark_schema import load_capture, validate_capture
from benchmark_sweep_lib.review import (
    cache_entry_from_capture,
    review_captures,
    reviewed_release_status_for_target,
)
from diagnostic_companion import build_binding
from install_autotune_cache import AutotuneCacheInstallError, install_cache
from perf_variance_report import build_report
from promotion_ledger import build_ledger
from test_autotune_cache_install import entry, write_cache, write_ledger
from test_benchmark_sweep_support import finite_capture
from test_diagnostic_companion import authored_capture_pair


class PayloadControls(unittest.TestCase):
    def test_timestamp_only_projection_and_all_other_payload_fields(self):
        original = entry()
        updated = {**original, "updated_utc": "2030-01-01T00:00:00Z"}
        self.assertEqual(contract.entry_digest(original), contract.entry_digest(updated))
        for field, value in (
            ("measured_medians_us", {"end_to_end": 0.001}),
            ("setup_inclusive_median_end_to_end_us", 0.001),
            ("setup_estimator", "changed"),
            ("workspace_bytes", 1),
            ("candidate_artifact", "changed"),
            ("future_dispatch_policy", "changed"),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(
                    contract.entry_digest(original),
                    contract.entry_digest({**original, field: value}),
                )
        self.assertEqual(len(contract.unique_entries([original, updated])), 1)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            contract.unique_entries([original, {**original, "workspace_bytes": 1}])

    def test_each_evidence_file_is_hashed_once_and_receipt_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for role in ("candidate", "review", "variance"):
                path = root / (role + ".json")
                path.write_text(json.dumps({"synthetic_fixture": role}), encoding="utf-8")
                inputs.append((role, path))
            inputs += [("observation", inputs[0][1]), inputs[1]]
            with patch.object(contract, "file_digest", wraps=contract.file_digest) as hashed:
                receipt = contract.make_receipt(entry(), inputs)
                self.assertEqual(hashed.call_count, 3)
            with patch.object(contract, "file_digest", wraps=contract.file_digest) as hashed:
                contract.validate_receipt(entry(), receipt)
                self.assertEqual(hashed.call_count, 3)
            for field, value in (
                ("schema", "legacy"),
                ("entry_sha256", "0" * 64),
                ("identity_sha256", "0" * 64),
                ("inputs", []),
            ):
                malformed = {**receipt, field: value}
                with self.subTest(field=field), self.assertRaises(ValueError):
                    contract.validate_receipt(entry(), malformed)
            for role, path in inputs[:3]:
                previous = path.read_bytes()
                path.write_bytes(previous.replace(b"fixture", b"changed"))
                with self.subTest(role=role), self.assertRaisesRegex(ValueError, "changed"):
                    contract.validate_receipt(entry(), receipt)
                path.write_bytes(previous)
            missing = deepcopy(receipt)
            missing["inputs"] = missing["inputs"][:-1]
            missing["identity_sha256"] = contract.digest(
                {key: value for key, value in missing.items() if key != "identity_sha256"}
            )
            with self.assertRaises(ValueError):
                contract.validate_receipt(entry(), missing)


class PublicationControls(unittest.TestCase):
    def test_counter_rows_bind_their_own_capture_in_a_multi_entry_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captures, counters = [], []
            for index, backend in enumerate(("ck", "hip-direct")):
                path = root / (backend + ".json")
                path.write_text(
                    json.dumps(finite_capture(backend, 190 + index * 100)), encoding="utf-8"
                )
                captures.append(path)
                counters.append(dict(capture=str(path), vgpr=17 + index * 66))
            report = root / "counters.json"
            report.write_text(json.dumps(dict(entries=counters)), encoding="utf-8")
            ledger = build_ledger(captures, None, counter_reports=[report])
            self.assertEqual(
                [row["counter_resource_summary"]["vgpr"] for row in ledger["entries"]], [17, 83]
            )

    def test_real_review_variance_ledger_install_chain_binds_payload_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            productions, bindings, binding_paths = [], {}, []
            first = []
            for backend, cost in (("ck", 190), ("hip-direct", 300), ("cpu-reference", 900)):
                for run in range(2):
                    lane = root / f"{backend}-{run}"
                    lane.mkdir()
                    _, _, paths = authored_capture_pair(lane, finite_capture(backend, cost))
                    production = load_capture(paths[0])
                    validate_capture(production, paths[0])
                    productions.append(paths[0])
                    if run == 0:
                        first.append(production)
                    if backend != "cpu-reference":
                        binding = build_binding(*paths)
                        bindings[str(paths[0].resolve())] = binding
                        path = lane / "binding.json"
                        path.write_text(json.dumps(binding), encoding="utf-8")
                        binding_paths.append(path)
            review = review_captures(first, review_mode="release", diagnostic_bindings=bindings)
            self.assertEqual(len(review["promotable_autotune_entries"]), 1)
            review_path, variance_path = root / "review.json", root / "variance.json"
            review_path.write_text(json.dumps(review), encoding="utf-8")
            variance_path.write_text(
                json.dumps(build_report(productions, require_multi_run=True)), encoding="utf-8"
            )
            candidate = first[0]
            expected = cache_entry_from_capture(
                candidate, reviewed_release_status_for_target("gfx1100")
            )
            source, ledger_path, destination = (
                root / name for name in ("source.json", "ledger.json", "installed.json")
            )
            write_cache(source, [expected])

            def author():
                return build_ledger(
                    [Path(candidate["_path"])],
                    source,
                    [review_path],
                    [variance_path],
                    require_variance_gate=True,
                    diagnostic_binding_paths=binding_paths,
                )

            ledger = author()
            self.assertEqual(ledger["blocked_count"], 0, ledger["entries"][0]["promotion_blockers"])
            self.assertIsNotNone(ledger["entries"][0]["cache_promotion_receipt"])
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            install_cache([source], destination, promotion_ledgers=[ledger_path])
            before = destination.read_bytes()
            changed = {
                **expected,
                "measured_medians_us": {**expected["measured_medians_us"], "end_to_end": 1},
            }
            write_cache(source, [changed])
            with self.assertRaisesRegex(AutotuneCacheInstallError, "payload_or_evidence_changed"):
                install_cache([source], destination, promotion_ledgers=[ledger_path])
            self.assertEqual(destination.read_bytes(), before)
            rejected = author()
            self.assertIn(
                "cache_payload_differs_from_reviewed_capture",
                rejected["entries"][0]["promotion_blockers"],
            )
            self.assertIsNone(rejected["entries"][0]["cache_promotion_receipt"])
            write_cache(source, [expected])
            # An independent run behind the same variance report is also immutable.
            other = productions[1]
            saved = other.read_bytes()
            other.write_bytes(saved + b"\n")
            with self.assertRaisesRegex(AutotuneCacheInstallError, "payload_or_evidence_changed"):
                install_cache([source], destination, promotion_ledgers=[ledger_path])
            self.assertEqual(destination.read_bytes(), before)
            other.write_bytes(saved)
            refreshed = author()
            ledger_path.write_text(json.dumps(refreshed), encoding="utf-8")
            install_cache([source], destination, promotion_ledgers=[ledger_path])
            self.assertEqual(destination.read_bytes(), before)

    def test_stale_payload_and_evidence_never_replace_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, ledger, destination = (
                root / name for name in ("source.json", "ledger.json", "installed.json")
            )
            original = entry()
            write_cache(source, [original])
            write_ledger(ledger, [original], variance=True)
            install_cache([source], destination, promotion_ledgers=[ledger])
            before = destination.read_bytes()
            for field, value in (
                ("measured_medians_us", {**original["measured_medians_us"], "end_to_end": 0.001}),
                ("setup_inclusive_median_end_to_end_us", 0.001),
                ("workspace_bytes", 1),
            ):
                write_cache(source, [{**original, field: value}])
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(
                        AutotuneCacheInstallError, "payload_or_evidence_changed"
                    ),
                ):
                    install_cache([source], destination, promotion_ledgers=[ledger])
                self.assertEqual(destination.read_bytes(), before)
            write_cache(source, [{**original, "updated_utc": "2030-01-01T00:00:00Z"}])
            summary = install_cache([source], destination, promotion_ledgers=[ledger])
            self.assertEqual(summary["replacement_history"], [])
            self.assertEqual(destination.read_bytes(), before)
            evidence = root / "ledger-review.json"
            evidence.write_text("changed after review", encoding="utf-8")
            with self.assertRaisesRegex(AutotuneCacheInstallError, "payload_or_evidence_changed"):
                install_cache([source], destination, promotion_ledgers=[ledger])
            self.assertEqual(destination.read_bytes(), before)

    def test_ready_legacy_ledger_and_conflicting_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, ledger, destination = (
                root / name for name in ("source.json", "ledger.json", "installed.json")
            )
            original = entry()
            write_cache(source, [original])
            write_ledger(ledger, [original], variance=True)
            document = json.loads(ledger.read_text(encoding="utf-8"))
            document["entries"][0].pop("cache_promotion_receipt")
            ledger.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(AutotuneCacheInstallError, "content-bound"):
                install_cache([source], destination, promotion_ledgers=[ledger])
            self.assertFalse(destination.exists())
            other = root / "conflicting.json"
            write_cache(other, [{**original, "workspace_bytes": 1}])
            with self.assertRaisesRegex(AutotuneCacheInstallError, "conflicting same-key"):
                install_cache([source, other], destination, promotion_ledgers=[ledger])
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
