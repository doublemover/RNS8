#!/usr/bin/env python3
"""Self-test final-output RNS-chain reporting."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import json
from itertools import permutations

import rns_chain_report
from diagnostic_companion import build_binding
from report_capture_inputs import load_report_captures
from test_chain_report_support import exact_chain_capture
from test_benchmark_schema_support_chains_wrap64 import as_reused_pack_capture
from test_benchmark_schema_support_metadata import add_device_allocation_fields
from test_review_reference_support import add_synthetic_full_reference


def phase_summary(value: float) -> dict:
    return {"avg": value, "median": value, "p95": value}


def capture(
    *,
    backend: str,
    median_us: float,
    pack_mode: str = "per_repeat_repack",
    independent: bool = False,
    setup_us: float | None = None,
    repeats: int = 9,
    warmups: int = 3,
    with_source_identity: bool = True,
) -> dict:
    reuse = pack_mode != "per_repeat_repack"
    operands = {
        "prepacked_reuse": ["A", "B"],
        "prepacked_reuse_a": ["A"],
        "prepacked_reuse_b": ["B"],
    }.get(pack_mode, [])
    result = {
        "_path": f"{backend}-{pack_mode}-{median_us}.json",
        "schema_version": 4,
        "benchmark_execution_mode": (
            "residue_chain_independent_final_host_export"
            if independent
            else "residue_chain_final_host_export"
        ),
        "semantics": "exact_wide_signed",
        "bound_kind": "none",
        "bound_mode": "none",
        "bound": 0,
        "bound_source": "static_profile",
        "m": 128,
        "n": 128,
        "k": 128,
        "prefix": 20,
        "selected_prefix": 20,
        "requested_max_prefix": 20,
        "contract_prefix_policy": "fixed_requested_residue_chain",
        "residue_planes_requested": 20,
        "residue_planes_selected": 20,
        "residue_planes_skipped": 0,
        "residue_output_mode": "host_export",
        "residue_chain_length": 3,
        "residue_chain_final_export": True,
        "residue_chain_independent_final_export": independent,
        "tile_m": 128,
        "tile_n": 128,
        "k_block_size": 65536,
        "seed": 20260605,
        "input_distribution": "signed_uniform_-16_16",
        "backend_requested": backend,
        "backend_selected": backend,
        "pack_mode": pack_mode,
        "reuse_packed_inputs": reuse,
        "prepack_reuse_operands": operands,
        "prepack_reuse_strategy": "persistent_matrix_residency" if reuse else "none",
        "warmups": warmups,
        "repeats": repeats,
        "avg_prepack_setup_us": setup_us,
        "requested_next_op": {"resolved": "final-export"},
        "output_policy": {
            "destination_layout": "contiguous_row_major",
            "status_handling": "structurally_elided",
            "per_repeat_logical_export": True,
            "final_checksum_export_after_repeats": False,
        },
        "exact_output_contract": {
            "requested_final_output": "exact_wide_limb_host",
            "limb_count": 4,
            "status_policy": "structurally_elided",
        },
        "export_variant": {
            "name": "default",
            "semantic_contract": "exact_wide_signed",
            "signedness": "signed",
            "output_layout": "fixed_width_limbs",
            "selector_status_policy": "structurally_elided",
            "d2h_policy": "host_ld_padded",
            "final_output_mode": "final_host_output",
            "selector_key": f"semantics=exact_wide_signed;backend={backend};selected_kernel={backend}_export",
        },
        "timing_metadata": {
            "host_instrumentation_state": "diagnostic_gpu_events",
            "benchmark_execution_mode": (
                "residue_chain_independent_final_host_export"
                if independent
                else "residue_chain_final_host_export"
            ),
            "residue_chain_final_export": True,
            "residue_chain_independent_final_export": independent,
            "pack_mode": pack_mode,
            "prepack_reuse_operands": operands,
            "prepack_reuse_strategy": "persistent_matrix_residency" if reuse else "none",
            "gpu_event_timing": backend != "cpu-reference",
            "gpu_event_timing_status": "available"
            if backend != "cpu-reference"
            else "not_applicable",
            "gpu_event_timing_source": "hip_events" if backend != "cpu-reference" else None,
            "gpu_event_phase_order": ["pack", "rns_gemm", "crt_export"]
            if backend != "cpu-reference"
            else [],
        },
        "timing_summary_us": {
            "pack": phase_summary(100.0),
            "rns_gemm": phase_summary(200.0),
            "crt_export": phase_summary(100.0),
            "end_to_end": phase_summary(median_us),
        },
    }
    if reuse and with_source_identity:
        result["device_allocation"] = {
            "tracking_available": True,
            "source": "hip_direct_allocation_counters_snapshot",
            "setup_scope": "persistent_plan_workspace_prepacked_reuse",
            "source_version_inputs": "monotonic_source_version_per_repeat_when_packing_runs",
        }
    return result


def first_gpu_row(
    report: dict,
    *,
    backend: str = "hip-direct",
    pack_mode: str = "per_repeat_repack",
    chain_mode: str = "resident_final_output",
) -> dict:
    return next(
        row
        for group in report["groups"]
        for row in group["rows"]
        if row["backend"] == backend
        and row["pack_mode"] == pack_mode
        and row["chain_mode"] == chain_mode
    )


def production_controls():
    from test_diagnostic_companion import authored_capture_pair

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def collect(label, setup_us=900, change=None):
            captures, bindings = [], {}
            for mode, duration in (
                ("cpu", 5000),
                ("independent", 2500),
                ("final", 1000),
                ("reuse", 700),
            ):
                for run in range(2):
                    fixture = exact_chain_capture("final" if mode == "reuse" else mode, duration)
                    if mode == "reuse":
                        fixture = as_reused_pack_capture(fixture)
                        fixture["prepack_setup_us"] = fixture["avg_prepack_setup_us"] = setup_us
                        fixture["raw_timings_us"]["prepack_setup"] = [setup_us]
                        fixture["timing_summary_us"]["prepack_setup"] = phase_summary(setup_us)
                        add_device_allocation_fields(fixture)
                        fixture["device_allocation"]["setup_scope"] = (
                            "persistent_plan_workspace_prepacked_reuse"
                        )
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
            return rns_chain_report.build_report_from_captures(values, diagnostic_bindings=receipts)

        row = first_gpu_row(build(captures), backend="ck", pack_mode="prepacked_reuse")
        assert row["decision"] == "candidate_reuse_chain_win", row["blockers"]
        assert row["setup_inclusive_median_per_repeat_us"] == 800
        assert row["speedup_vs_same_backend_nonreuse"] == 1.25
        assert row["speedup_vs_cpu"] == 6.25
        assert row["speedup_vs_independent_export_repack"] == 3.125
        assert row["break_even_repeats_same_backend"] == 4
        assert row["promotion_eligible"] is False
        assert len(row["observation"]["measurement"]["observations"]) == 2
        assert row["comparisons"]["cpu"]["cpu_gpu_comparison"] is True
        for order in permutations(captures[4:]):
            current = first_gpu_row(
                build([*captures[:4], *order]), backend="ck", pack_mode="prepacked_reuse"
            )
            assert current["speedup_vs_same_backend_nonreuse"] == 1.25
        singleton = first_gpu_row(
            build([*captures[:6], captures[6], copy.deepcopy(captures[6])]),
            backend="ck",
            pack_mode="prepacked_reuse",
        )
        assert singleton["speedup_vs_same_backend_nonreuse"] is None
        missing = dict(bindings)
        missing.pop(str(Path(captures[7]["_path"]).resolve()))
        assert (
            first_gpu_row(build(captures, missing), backend="ck", pack_mode="prepacked_reuse")[
                "speedup_vs_cpu"
            ]
            is None
        )
        slow, receipts = collect("slow", setup_us=3600)
        assert (
            first_gpu_row(build(slow, receipts), backend="ck", pack_mode="prepacked_reuse")[
                "decision"
            ]
            == "deprioritize"
        )

        def variable_setup(fixture, mode, run):
            if mode == "reuse" and run == 1:
                fixture["prepack_setup_us"] = fixture["avg_prepack_setup_us"] = 100000
                fixture["raw_timings_us"]["prepack_setup"] = [100000]
                fixture["timing_summary_us"]["prepack_setup"] = phase_summary(100000)

        noisy, receipts = collect("unstable-setup", change=variable_setup)
        current = first_gpu_row(build(noisy, receipts), backend="ck", pack_mode="prepacked_reuse")
        assert current["speedup_vs_same_backend_nonreuse"] is None
        assert "setup_inclusive_process_spread_exceeds_ten_percent" in current["blockers"]

        def different_input(fixture, mode, run):
            if mode == "reuse":
                fixture["seed"] += 1

        changed, receipts = collect("different-seed", change=different_input)
        current = first_gpu_row(build(changed, receipts), backend="ck", pack_mode="prepacked_reuse")
        assert current["speedup_vs_same_backend_nonreuse"] is None

        def incomplete_source(fixture, mode, run):
            if mode == "reuse" and run == 1:
                fixture.pop("device_allocation")

        incomplete, receipts = collect("one-process-source", change=incomplete_source)
        for order in (incomplete, list(reversed(incomplete))):
            current = first_gpu_row(
                build(order, receipts), backend="ck", pack_mode="prepacked_reuse"
            )
            assert current["speedup_vs_same_backend_nonreuse"] is None
            assert "reuse_process_source_identity_incomplete" in current["blockers"]
            assert len(current["source_identity"]["observations"]) == 2


def main() -> int:
    cpu = capture(backend="cpu-reference", median_us=5000.0)
    independent = capture(backend="hip-direct", median_us=2500.0, independent=True)
    direct = capture(backend="hip-direct", median_us=1000.0)
    report = rns_chain_report.build_report_from_captures([cpu, independent, direct])
    row = first_gpu_row(report)
    assert report["summary"]["candidate_wins"] == 0
    assert report["summary"]["independent_export_repack_baselines"] == 0
    assert row["decision"] == "keep_experimental"
    assert row["speedup_vs_cpu"] is None
    assert row["speedup_vs_independent_export_repack"] is None

    reuse = capture(
        backend="hip-direct", median_us=700.0, pack_mode="prepacked_reuse_b", setup_us=900.0
    )
    reuse_report = rns_chain_report.build_report_from_captures([cpu, independent, direct, reuse])
    reuse_row = first_gpu_row(reuse_report, pack_mode="prepacked_reuse_b")
    assert reuse_row["decision"] == "keep_experimental"
    assert reuse_row["setup_inclusive_median_per_repeat_us"] is None
    assert reuse_row["speedup_vs_same_backend_nonreuse"] is None
    assert reuse_row["break_even_repeats_same_backend"] is None

    slow_reuse = copy.deepcopy(reuse)
    slow_reuse["_path"] = "slow-reuse.json"
    slow_reuse["avg_prepack_setup_us"] = 3600.0
    slow_report = rns_chain_report.build_report_from_captures(
        [cpu, independent, direct, slow_reuse]
    )
    slow_row = first_gpu_row(slow_report, pack_mode="prepacked_reuse_b")
    assert slow_row["decision"] == "keep_experimental"
    assert slow_row["speedup_vs_same_backend_nonreuse"] is None

    missing_independent_report = rns_chain_report.build_report_from_captures([cpu, direct])
    missing_independent_row = first_gpu_row(missing_independent_report)
    assert missing_independent_row["decision"] == "keep_experimental"
    assert (
        "missing_same_backend_independent_export_repack_baseline"
        in missing_independent_row["blockers"]
    )

    independent_row = first_gpu_row(report, chain_mode="independent_export_repack")
    assert independent_row["decision"] == "keep_experimental"

    missing_cpu_report = rns_chain_report.build_report_from_captures([independent, direct])
    missing_row = first_gpu_row(missing_cpu_report)
    assert missing_row["decision"] == "keep_experimental"
    assert "missing_cpu_final_output_baseline" in missing_row["blockers"]

    missing_identity = copy.deepcopy(reuse)
    missing_identity["_path"] = "missing-identity.json"
    del missing_identity["device_allocation"]
    identity_report = rns_chain_report.build_report_from_captures(
        [cpu, independent, direct, missing_identity]
    )
    identity_row = first_gpu_row(identity_report, pack_mode="prepacked_reuse_b")
    assert identity_row["decision"] == "keep_experimental"
    assert "missing_device_allocation_metadata" in identity_row["blockers"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report_path = tmp_path / "rns-chain-report.json"
        review_path = tmp_path / "review_report.json"
        manifest_path = tmp_path / "scenario_manifest.json"
        capture_path = tmp_path / "capture.json"
        report_path.write_text("{}", encoding="utf-8")
        review_path.write_text("{}", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")
        capture_path.write_text("{}", encoding="utf-8")
        capture_path.write_text(json.dumps(exact_chain_capture("final", 100)), encoding="utf-8")
        loaded = load_report_captures([tmp_path])
        assert len(loaded) == 1
        assert Path(loaded[0]["_path"]) == capture_path

    production_controls()

    print("rns chain report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
