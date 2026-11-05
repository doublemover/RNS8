"""Authored native-format joins and rejection controls, without profiling."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import unittest

from amd_profiler_evidence import validate_dispatch_receipt, code_object_digest
from target_qualification import EvidenceStore
from test_target_qualification_support import QualificationFixture, reference, write
from timing_eligibility import phase_blockers
from diagnostic_companion import production_evidence


class NativeProfilerEvidence(unittest.TestCase):
    def test_requested_or_active_ranges_cannot_enter_production_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualificationFixture(Path(directory))
            for state in ("active", "requested_provider_unavailable", "unknown"):
                candidate = copy.deepcopy(fixture.capture)
                candidate["profiler_ranges"] = dict(schema="rns8_profiler_ranges_v1", provider_status=state)
                self.assertIn("profile_range_instrumentation_not_production_evidence", phase_blockers(candidate))
                self.assertIn("profile_range_instrumentation_not_production_evidence", production_evidence(candidate)["blockers"])
            for state in ("disabled_at_build", "disabled_at_runtime"):
                candidate = copy.deepcopy(fixture.capture)
                candidate["profiler_ranges"] = dict(schema="rns8_profiler_ranges_v1", provider_status=state)
                self.assertNotIn("profile_range_instrumentation_not_production_evidence", phase_blockers(candidate))

    def test_selected_native_dispatch_retains_identity_without_claiming_production_latency(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualificationFixture(Path(directory))
            store = EvidenceStore()
            result = validate_dispatch_receipt(
                reference(fixture.dispatch_path),
                capture=fixture.capture,
                store=store,
                parent=fixture.root,
            )
            store.finish()
            self.assertTrue(result["qualified_selected_dispatches"])
            self.assertFalse(result["complete_run_coverage"])
            self.assertFalse(result["performance_promotion_allowed"])
            observed = result["observations"][0]
            self.assertEqual(
                (
                    observed["process_id"],
                    observed["agent_id"],
                    observed["queue_id"],
                    observed["dispatch_id"],
                    observed["correlation_id"],
                ),
                (222, 3, 5, 11, 17),
            )
            self.assertEqual(observed["mangled_symbol"], "synthetic_selected_symbol")
            self.assertIn(
                "externally_profiled_host_time_not_production_evidence",
                phase_blockers(fixture.profiled),
            )
            self.assertIn(
                "externally_profiled_host_time_not_production_evidence",
                production_evidence(fixture.profiled)["blockers"],
            )

    def test_wrong_process_agent_symbol_geometry_and_duplicate_dispatch_cannot_join(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualificationFixture(Path(directory))
            original = copy.deepcopy(fixture.native)
            for mutate in (
                "process",
                "agent",
                "symbol",
                "code_object",
                "duplicate",
                "missing_geometry",
                "timestamp",
            ):
                native = copy.deepcopy(original)
                process = native["rocprofiler-sdk-tool"][0]
                dispatch = process["buffer_records"]["kernel_dispatch"][0]
                if mutate == "process":
                    process["metadata"]["pid"] = 333
                elif mutate == "agent":
                    process["agents"][0]["name"] = "gfx1100"
                elif mutate == "symbol":
                    process["kernel_symbols"][0]["kernel_name"] = "unrelated_helper"
                elif mutate == "code_object":
                    process["code_objects"][0]["uri"] = "memory://222#offset=1&size=4"
                elif mutate == "duplicate":
                    process["buffer_records"]["kernel_dispatch"].append(
                        copy.deepcopy(dispatch)
                    )
                elif mutate == "missing_geometry":
                    dispatch["dispatch_info"]["grid_size"]["x"] = 0
                else:
                    dispatch["end_timestamp"] = dispatch["start_timestamp"]
                write(fixture.native_path, native)
                receipt = copy.deepcopy(fixture.dispatch)
                receipt["native_trace"] = reference(fixture.native_path)
                write(fixture.dispatch_path, receipt)
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    validate_dispatch_receipt(
                        reference(fixture.dispatch_path),
                        capture=fixture.capture,
                        store=EvidenceStore(),
                        parent=fixture.root,
                    )

    def test_missing_or_changed_raw_artifacts_and_unknown_adapters_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualificationFixture(Path(directory))
            receipt = copy.deepcopy(fixture.dispatch)
            receipt["adapter"] = "raw-rgp-guessed-column-names"
            write(fixture.dispatch_path, receipt)
            with self.assertRaises(ValueError):
                validate_dispatch_receipt(
                    reference(fixture.dispatch_path),
                    capture=fixture.capture,
                    store=EvidenceStore(),
                    parent=fixture.root,
                )
            write(fixture.dispatch_path, fixture.dispatch)
            fixture.native_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_dispatch_receipt(
                    reference(fixture.dispatch_path),
                    capture=fixture.capture,
                    store=EvidenceStore(),
                    parent=fixture.root,
                )

    def test_code_object_uri_uses_actual_bytes_and_checks_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = QualificationFixture(Path(directory))
            executable = fixture.binaries["rns8-bench"]
            sha = fixture.capture["artifact_capture"]["executable_sha256"]
            prefix = "file://" + executable.as_posix()
            first = code_object_digest(
                prefix + "#offset=6&size=" + str(len(fixture.code)),
                executable,
                sha,
                EvidenceStore(),
            )
            second = code_object_digest(
                prefix + "?offset=0x6&size=" + oct(len(fixture.code))[2:].rjust(4, "0"),
                executable,
                sha,
                EvidenceStore(),
            )
            self.assertEqual(first, second)
            for suffix in (
                "#offset=-1&size=4",
                "#offset=0&offset=1&size=4",
                "#offset=999999&size=1",
                "#offset=0&size=0",
                "#offset=0&size=1&unknown=3",
            ):
                with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                    code_object_digest(
                        prefix + suffix, executable, sha, EvidenceStore()
                    )


if __name__ == "__main__":
    unittest.main()
