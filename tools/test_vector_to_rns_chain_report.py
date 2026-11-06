#!/usr/bin/env python3
"""Self-test vector/native-to-RNS chain reporting."""

from __future__ import annotations

from test_review_reference_support import add_synthetic_full_reference

import copy
import json
import tempfile
from pathlib import Path

import vector_to_rns_chain_report
from report_capture_inputs import load_report_captures
from test_benchmark_schema_support_core import as_vector_to_rns_chain_capture, expect_valid


def make_capture(*, host_repack_control: bool, median: float, checksum: int = 12345) -> dict:
    capture = as_vector_to_rns_chain_capture(
        expect_valid("v4_bounded_i64_adaptive_hip.json"),
        "native_i64_to_rns_kernel",
        "vector_alu_i64_kernel",
        host_repack_control=host_repack_control,
    )
    capture["warmups"] = 3
    capture["repeats"] = 9
    capture["checksum_u64"] = checksum
    for phase in ["pack", "rns_gemm", "crt_export", "end_to_end"]:
        capture["timing_summary_us"][phase]["median"] = (
            median if phase == "end_to_end" else median / 4.0
        )
    return add_synthetic_full_reference(capture)


def production_controls():
    from diagnostic_companion import build_binding
    from itertools import permutations
    from test_benchmark_sweep_support import set_phase
    from test_benchmark_schema_support_metadata import add_target_variant_fields
    from test_diagnostic_companion import authored_capture_pair

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def collect(label, candidate_times=(100, 100), control_times=(160, 160), change=None):
            captures, bindings = [], {}
            for control, durations in ((False, candidate_times), (True, control_times)):
                for run, duration in enumerate(durations):
                    fixture = make_capture(host_repack_control=control, median=duration)
                    set_phase(fixture, duration)
                    fixture["avg_per_modulus_gemm_estimate_us"] = fixture["avg_rns_gemm_us"]
                    if change is not None:
                        change(fixture, control)
                    fixture = add_synthetic_full_reference(fixture)
                    lane = root / f"{label}-{control}-{run}"
                    lane.mkdir()
                    _, _, paths = authored_capture_pair(lane, fixture)
                    captures.extend(load_report_captures([paths[0]]))
                    bindings[str(paths[0].resolve())] = build_binding(*paths)
            return captures, bindings

        captures, bindings = collect("matched")

        def build(items, receipts=bindings):
            return vector_to_rns_chain_report.build_report(items, diagnostic_bindings=receipts)

        def candidate_row(report):
            return next(row for row in report["rows"] if row.get("candidate_capture"))

        row = candidate_row(build(captures))
        assert row["disposition"] == "local promote", row["blockers"]
        assert row["speedup_vs_host_repack_control"] == 1.6
        assert row["comparison"]["reference_requirement"] == "embedded_full_cpu_output_reference"
        assert row["comparison"]["candidate_numerical_reference"]["passed"]
        assert row["comparison"]["candidate_measurement_cohort"]["independent_runs"] == 2
        assert row["comparison"]["baseline_measurement_cohort"]["independent_runs"] == 2
        assert len(row["comparison"]["candidate_production_observations"]) == 2
        assert row["candidate_conversion_event_us"] is None
        for order in permutations(captures):
            current = candidate_row(build(list(order)))
            assert current["disposition"] == "local promote"
            assert current["speedup_vs_host_repack_control"] == 1.6
        assert (
            candidate_row(build(captures + [copy.deepcopy(captures[0])]))["disposition"]
            == "local promote"
        )
        copy_path = root / "unbound-identical-copy.json"
        copy_path.write_text(Path(captures[0]["_path"]).read_text(), encoding="utf-8")
        alias = load_report_captures([copy_path])[0]
        # A filename alias is not another run and cannot erase the original's
        # intact binding merely by being the last submitted dictionary.
        for items in (captures + [alias], [alias] + captures):
            current = candidate_row(build(items))
            assert current["disposition"] == "local promote"
            assert current["comparison"]["candidate_measurement_cohort"]["independent_runs"] == 2
        singleton = candidate_row(build([captures[0], copy.deepcopy(captures[0]), *captures[2:]]))
        assert singleton["disposition"] == "keep experimental"
        assert singleton["speedup_vs_host_repack_control"] is None
        missing = dict(bindings)
        missing.pop(str(Path(captures[1]["_path"]).resolve()))
        current = candidate_row(build(captures, missing))
        assert current["disposition"] == "keep experimental"
        assert "matched_diagnostic_companion_missing" in current["blockers"]
        assert current["speedup_vs_host_repack_control"] is None
        no_reference = copy.deepcopy(captures)
        for fixture in no_reference[:2]:
            fixture.pop("native_reference_verification")
        assert candidate_row(build(no_reference))["speedup_vs_host_repack_control"] is None
        conflict = copy.deepcopy(captures)
        conflict[0]["checksum_u64"] += 1
        current = candidate_row(build(conflict))
        assert current["disposition"] == "keep experimental"
        assert "checksum_mismatch_vs_reference" in current["blockers"]
        assert current["speedup_vs_host_repack_control"] is None
        for label, candidate_times, control_times, expected in (
            ("slow", (200, 200), (160, 160), "drop/deprioritize"),
            ("noise", (100, 100), (102, 102), "keep experimental"),
            ("unstable", (100, 100), (160, 260), "keep experimental"),
        ):
            values, receipts = collect(label, candidate_times, control_times)
            current = candidate_row(build(values, receipts))
            assert current["disposition"] == expected, current["blockers"]
            if label == "unstable":
                assert current["speedup_vs_host_repack_control"] is None

        def different_device(fixture, control):
            if not control:
                fixture["device"]["name"] = "Other physical GPU"
                add_target_variant_fields(fixture)

        values, receipts = collect("device", change=different_device)
        current = candidate_row(build(values, receipts))
        assert "missing_host_export_repack_control" in current["blockers"]
        assert current.get("speedup_vs_host_repack_control") is None

        def different_seed(fixture, control):
            if not control:
                fixture["seed"] += 1

        values, receipts = collect("seed", change=different_seed)
        current = candidate_row(build(values, receipts))
        assert "missing_host_export_repack_control" in current["blockers"]
        assert current.get("speedup_vs_host_repack_control") is None

        def no_warmups(fixture, control):
            if not control:
                fixture["warmups"] = 0

        values, receipts = collect("warmups", change=no_warmups)
        current = candidate_row(build(values, receipts))
        assert current["disposition"] == "keep experimental"
        assert "not_release_review" in current["blockers"]


def main():
    candidate = make_capture(host_repack_control=False, median=80.0)
    control = make_capture(host_repack_control=True, median=120.0)
    report = vector_to_rns_chain_report.build_report([candidate, control])
    assert report["capture_count"] == 2
    assert report["group_count"] == 1
    row = report["rows"][0]
    assert row["disposition"] == "keep experimental"
    assert row["checksum_match"] is True
    assert row["speedup_vs_host_repack_control"] is None
    assert row["control_host_repack_us"] == 3.0
    missing = vector_to_rns_chain_report.build_report([candidate])
    assert "missing_host_export_repack_control" in missing["rows"][0]["blockers"]
    mismatch_control = copy.deepcopy(control)
    mismatch_control["checksum_u64"] = 777
    mismatch = vector_to_rns_chain_report.build_report([candidate, mismatch_control])
    assert "final_checksum_mismatch_or_missing" in mismatch["rows"][0]["blockers"]
    conflicting_candidate = copy.deepcopy(candidate)
    conflicting_candidate["checksum_u64"] ^= 1
    conflicting = vector_to_rns_chain_report.build_report(
        [conflicting_candidate, candidate, control]
    )
    assert "checksum_mismatch_vs_reference" in conflicting["rows"][0]["blockers"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "review_report.json").write_text(
            '{"schema_version": 3, "groups": []}\n', encoding="utf-8"
        )
        (root / "capture.json").write_text(
            json.dumps(expect_valid("v4_bounded_i64_adaptive_hip.json")), encoding="utf-8"
        )
        loaded = load_report_captures([root])
        assert len(loaded) == 1
        assert loaded[0]["benchmark"] == "rns8_bounded_gemm_persistent_rns"
    production_controls()
    print("vector-to-RNS chain report self-test: PASS")


if __name__ == "__main__":
    main()
