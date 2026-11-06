#!/usr/bin/env python3
"""Self-test exact-wide residue-current chain pairing."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from itertools import permutations
import json

import exact_wide_chain_report
from diagnostic_companion import build_binding
from report_capture_inputs import load_report_captures
from test_benchmark_schema_support_core import expect_valid
from test_benchmark_schema_support_chains_wrap64 import (
    as_residue_current_chain_capture,
    as_residue_chain_final_export_capture,
)
from test_benchmark_sweep_support import exact_wide_capture, set_phase
from test_review_reference_support import add_synthetic_full_reference


def phase_summary(value: float) -> dict:
    return {"avg": value, "median": value, "p95": value}


def capture(*, backend: str, mode: str, median_us: float) -> dict:
    residue_current = mode == "residue_current"
    return {
        "_path": f"{backend}-{mode}.json",
        "schema_version": 4,
        "benchmark_execution_mode": "residue_current_rns_chain"
        if residue_current
        else "residue_chain_final_host_export",
        "backend_requested": backend,
        "backend_selected": backend,
        "semantics": "exact_wide_signed",
        "m": 512,
        "n": 512,
        "k": 512,
        "prefix": 20,
        "selected_prefix": 20,
        "requested_max_prefix": 20,
        "contract_prefix_policy": "fixed_requested_residue_chain",
        "seed": 20260606,
        "input_distribution": "signed_uniform_-16_16",
        "exact_wide_limb_count": 4,
        "residue_chain_length": 3,
        "residue_chain_final_export": not residue_current,
        "residue_output_mode": "residue_current_rns" if residue_current else "host_export",
        "warmups": 3,
        "repeats": 9,
        "plan_packing": {
            "output_device_current": residue_current,
            "output_host_current": not residue_current,
        },
        "plan_lowering": {
            "rns_continuation_available": True,
        },
        "requested_next_op": {
            "resolved": "rns-gemm" if residue_current else "final-export",
        },
        "output_policy": {
            "per_repeat_logical_export": not residue_current,
            "final_checksum_export_after_repeats": residue_current,
            "status_handling": "structurally_elided",
        },
        "timing_metadata": {
            "gpu_event_timing": backend != "cpu-reference",
            "gpu_event_timing_status": "available"
            if backend != "cpu-reference"
            else "not_applicable",
            "gpu_event_timing_source": "hipEventElapsedTime"
            if backend != "cpu-reference"
            else None,
            "gpu_event_phase_order": ["pack", "rns_gemm"]
            if residue_current
            else ["pack", "rns_gemm", "crt_export"],
        },
        "timing_summary_us": {
            "end_to_end": phase_summary(median_us),
        },
    }


def first_row(report: dict) -> dict:
    return report["groups"][0]["rows"][0]


def production_controls():
    from test_diagnostic_companion import authored_capture_pair

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def collect(label, change=None, times=(100, 160, 1600)):
            captures, bindings = [], {}
            for mode, durations in zip(("resident", "final", "cpu"), times):
                for run in range(2):
                    duration = durations[run] if isinstance(durations, tuple) else durations
                    base = expect_valid("v4_bounded_i64_ck.json")
                    builder = (
                        as_residue_current_chain_capture
                        if mode == "resident"
                        else as_residue_chain_final_export_capture
                    )
                    fixture = builder(base)
                    if mode == "resident":
                        fixture["plan_packing"] = dict(
                            source="rns8_get_plan_packing_info",
                            backend="ck",
                            semantics="exact_wide_signed",
                            input_domain_name="resident_rns",
                            output_domain_name="resident_rns",
                            next_op_hint="rns-gemm",
                            uses_resident_matrix_inputs=True,
                            uses_transient_pack_workspace=False,
                            uses_matrix_engine_pack_layout=True,
                            reusable_prepack_cache_available=False,
                            production_prepack_cache_available=False,
                            output_host_current=False,
                            output_device_current=True,
                            source_versioned_inputs=True,
                            same_source_version_pack_elision_available=True,
                            flags=0,
                            next_op_flags=0,
                            a_pack_workspace_bytes=0,
                            b_pack_workspace_bytes=0,
                            accumulator_workspace_bytes=0,
                            library_workspace_bytes=0,
                            total_transient_workspace_bytes=0,
                        )
                        fixture["plan_lowering"] = dict(
                            source="rns8_get_plan_lowering_info",
                            operation="gemm",
                            semantic_contract="exact_wide_signed",
                            backend_family="ck",
                            input_domain="resident_rns",
                            output_domain="resident_rns",
                            desired_output="resident_rns",
                            schedule_strategy="fixed_prefix",
                            packing_strategy="resident_rns",
                            reuse_strategy="none",
                            conversion_strategy="deferred_export",
                            lowering_path="synthetic_fixture",
                            final_export_available=True,
                            rns_continuation_available=True,
                            native_continuation_available=False,
                            native_to_rns_available=False,
                            reusable_b_prepack_available=False,
                        )
                    if mode == "cpu":
                        cpu = exact_wide_capture("cpu-reference", duration)
                        for field in (
                            "backend_requested",
                            "backend_selected",
                            "selected_kernel",
                            "backend_metadata",
                            "device",
                            "target_variant",
                            "int32_accumulator_safety",
                        ):
                            if field in cpu:
                                fixture[field] = copy.deepcopy(cpu[field])
                    fixture["warmups"], fixture["repeats"] = 3, 9
                    set_phase(fixture, duration)
                    fixture["avg_per_modulus_gemm_estimate_us"] = fixture["avg_rns_gemm_us"] / 20
                    fixture["checksum_u64"] = 12345
                    if change is not None:
                        change(fixture, mode, run)
                    add_synthetic_full_reference(fixture)
                    lane = root / f"{label}-{mode}-{run}"
                    lane.mkdir()
                    _, _, paths = authored_capture_pair(lane, fixture)
                    captures.extend(load_report_captures([paths[0]]))
                    if mode != "cpu":
                        bindings[str(paths[0].resolve())] = build_binding(*paths)
            return captures, bindings

        captures, bindings = collect("matched")

        def build(values, receipts=bindings):
            return exact_wide_chain_report.build_report_from_captures(
                values, diagnostic_bindings=receipts
            )

        row = first_row(build(captures))
        assert row["decision"] == "paired_residue_current_chain_ready", row["blockers"]
        assert row["final_output_vs_residue_current_ratio"] == 1.6
        assert row["cpu_vs_final_output_speedup"] == 10
        assert row["promotion_eligible"] is False
        assert (
            row["boundary_ratio_kind"] == "different_completion_boundaries_not_host_output_speedup"
        )
        assert row["boundary_comparison"]["candidate_measurement_cohort"]["independent_runs"] == 2
        assert len(row["boundary_comparison"]["candidate_production_observations"]) == 2
        for order in permutations(captures[:4]):
            ordered = first_row(build([*order, *captures[4:]]))
            assert ordered["cpu_vs_final_output_speedup"] == 10
            assert ordered["final_output_vs_residue_current_ratio"] == 1.6
        duplicate = first_row(build([captures[0], copy.deepcopy(captures[0]), *captures[2:]]))
        assert duplicate["final_output_vs_residue_current_ratio"] is None
        alias_path = root / "renamed-copy.json"
        alias_path.write_text(Path(captures[0]["_path"]).read_text())
        alias = load_report_captures([alias_path])[0]
        for values in ([alias, *captures], [*captures, alias]):
            assert first_row(build(values))["final_output_vs_residue_current_ratio"] == 1.6
        missing = dict(bindings)
        missing.pop(str(Path(captures[1]["_path"]).resolve()))
        assert first_row(build(captures, missing))["final_output_vs_residue_current_ratio"] is None
        noisy, receipts = collect("unstable", times=((100, 100), (160, 260), (1600, 1600)))
        assert first_row(build(noisy, receipts))["final_output_vs_residue_current_ratio"] is None
        for label, mutation in (
            (
                "seed",
                lambda fixture, mode, run: fixture.update(seed=99) if mode == "final" else None,
            ),
            (
                "host",
                lambda fixture, mode, run: (
                    fixture.setdefault("execution_host_identity", {}).update(
                        os_instance_sha256="b" * 64
                    )
                    if mode == "final"
                    else None
                ),
            ),
            (
                "proof",
                lambda fixture, mode, run: (
                    fixture.update(checksum_u64=54321) if mode == "resident" else None
                ),
            ),
        ):
            if label == "host":
                # Native envelopes are supplied before an intentional host change.
                from test_capture_identity_support import synthetic_execution_host_identity

                def mutate(fixture, mode, run, mutation=mutation):
                    fixture["execution_host_identity"] = synthetic_execution_host_identity()
                    mutation(fixture, mode, run)
            else:
                mutate = mutation
            changed, receipts = collect(label, mutate)
            report = build(changed, receipts)
            assert all(
                item["final_output_vs_residue_current_ratio"] is None
                for group in report["groups"]
                for item in group["rows"]
            ), json.dumps(report["summary"])


def main() -> int:
    cpu_final = capture(backend="cpu-reference", mode="final_output", median_us=100000.0)
    gpu_final = capture(backend="hip-direct", mode="final_output", median_us=1000.0)
    gpu_residue = capture(backend="hip-direct", mode="residue_current", median_us=600.0)
    report = exact_wide_chain_report.build_report_from_captures([cpu_final, gpu_final, gpu_residue])
    row = first_row(report)
    assert report["summary"]["ready_pairs"] == 0
    assert report["summary"]["experimental"] == 1
    assert row["decision"] == "keep_experimental"
    assert row["final_output_vs_residue_current_ratio"] is None
    assert row["cpu_vs_final_output_speedup"] is None
    assert row["residue_lifetime"]["available"] is True

    missing_final = exact_wide_chain_report.build_report_from_captures([cpu_final, gpu_residue])
    missing_final_row = first_row(missing_final)
    assert missing_final_row["decision"] == "keep_experimental"
    assert "missing_same_backend_final_output_capture" in missing_final_row["blockers"]

    broken_lifetime = copy.deepcopy(gpu_residue)
    broken_lifetime["plan_packing"]["output_device_current"] = False
    broken = exact_wide_chain_report.build_report_from_captures(
        [cpu_final, gpu_final, broken_lifetime]
    )
    broken_row = first_row(broken)
    assert broken_row["decision"] == "keep_experimental"
    assert "incomplete_residue_current_lifetime_metadata" in broken_row["blockers"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        capture_path = root / "capture.json"
        report_path = root / "exact-wide-chain-report.json"
        rns_report_path = root / "rns-chain-report.json"
        review_path = root / "review_report.json"
        manifest_path = root / "scenario_manifest.json"
        for path in [capture_path, report_path, rns_report_path, review_path, manifest_path]:
            path.write_text("{}", encoding="utf-8")
        capture_path.write_text(
            json.dumps(expect_valid("v4_bounded_i64_ck.json")), encoding="utf-8"
        )
        loaded = load_report_captures([root])
        assert len(loaded) == 1
        assert Path(loaded[0]["_path"]) == capture_path

    production_controls()
    print("exact wide chain report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
