#!/usr/bin/env python3
"""Self-test reuse contract comparison reporting."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from itertools import permutations

import reuse_contract_report
from diagnostic_companion import build_binding
from report_capture_inputs import load_report_captures
from test_benchmark_sweep_support import bounded_capture
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
    setup_us: float | None = None,
    repeats: int = 9,
    warmups: int = 3,
    with_source_identity: bool = True,
    with_reuse_contract: bool = True,
    with_stale_source_rejection: bool = True,
    prepack_reuse_strategy: str | None = None,
    runtime_prepack_cache: dict | None = None,
) -> dict:
    reuse = pack_mode != "per_repeat_repack"
    strategy = prepack_reuse_strategy or ("persistent_matrix_residency" if reuse else "none")
    operands = {
        "prepacked_reuse": ["A", "B"],
        "prepacked_reuse_a": ["A"],
        "prepacked_reuse_b": ["B"],
    }.get(pack_mode, [])
    result = {
        "_path": f"{backend}-{pack_mode}-{median_us}.json",
        "schema_version": 4,
        "semantics": "bounded_i64",
        "bound_kind": "signed",
        "bound_mode": "global",
        "bound": 16384,
        "m": 1024,
        "n": 1024,
        "k": 1024,
        "prefix": 9,
        "bound_source": "static_profile",
        "selected_prefix": 9,
        "requested_max_prefix": 9,
        "contract_prefix_policy": "fixed",
        "residue_planes_requested": 9,
        "residue_planes_selected": 9,
        "residue_planes_skipped": 0,
        "residue_output_mode": "host_export",
        "tile_m": 128,
        "tile_n": 128,
        "k_block_size": 65536,
        "seed": 7,
        "input_distribution": "signed_uniform_-16_16",
        "backend_requested": backend,
        "backend_selected": backend,
        "pack_mode": pack_mode,
        "reuse_packed_inputs": reuse,
        "prepack_reuse_operands": operands,
        "prepack_reuse_strategy": strategy,
        "resident_lifetime": {
            "enabled": reuse,
            "output_domain": "native_i64_u64_host",
        },
        "export_variant": {
            "name": "default",
            "semantic_contract": "bounded_i64",
            "signedness": "signed",
            "output_layout": "scalar_i64",
            "selector_status_policy": "range_checked_status_buffer",
            "d2h_policy": "host_ld_padded",
            "final_output_mode": "final_host_output",
            "selector_key": f"semantics=bounded_i64;backend={backend};target_id=gfx1100",
        },
        "warmups": warmups,
        "repeats": repeats,
        "avg_prepack_setup_us": setup_us,
        "timing_metadata": {
            "pack_mode": pack_mode,
            "prepack_reuse_operands": operands,
            "prepack_reuse_strategy": strategy,
            "gpu_event_timing": True,
            "gpu_event_timing_status": "available",
            "gpu_event_timing_source": "hip_events",
            "gpu_event_phase_order": ["pack", "rns_gemm", "crt_export"],
        },
        "timing_summary_us": {
            "pack": phase_summary(100.0),
            "rns_gemm": phase_summary(500.0),
            "crt_export": phase_summary(100.0),
            "end_to_end": phase_summary(median_us),
        },
    }
    setup_breakdown = None
    if reuse and setup_us is not None:
        setup_breakdown = {
            "pack_a": setup_us if pack_mode == "prepacked_reuse_a" else 0.0,
            "pack_b": setup_us if pack_mode == "prepacked_reuse_b" else 0.0,
            "runtime_cache": 0.0,
            "unclassified": 0.0,
        }
        if pack_mode == "prepacked_reuse":
            setup_breakdown["pack_a"] = setup_us / 2.0
            setup_breakdown["pack_b"] = setup_us / 2.0
        result["prepack_setup_breakdown_us"] = setup_breakdown
    if reuse and with_source_identity:
        result["device_allocation"] = {
            "tracking_available": True,
            "source": "hip_direct_allocation_counters_snapshot",
            "setup_scope": "persistent_plan_workspace_prepacked_reuse",
            "source_version_inputs": "monotonic_source_version_per_repeat_when_packing_runs",
        }
    if reuse and with_reuse_contract:
        result["reuse_contract"] = {
            "enabled": True,
            "operand_role": "A+B" if operands == ["A", "B"] else operands[0],
            "source_version_inputs": "monotonic_source_version_per_repeat_when_packing_runs",
            "setup_scope": "persistent_plan_workspace_prepacked_reuse",
            "setup_cost_us": setup_us,
            "setup_breakdown_us": setup_breakdown,
            "measured_repeat_count": repeats,
            "break_even_repeat_count": None,
            "output_domain": "native_i64_u64_host",
            "next_op": "final-export",
            "target_fingerprint": "gfx1100",
            "backend_fingerprint": backend,
            "kernel_fingerprint": f"{backend}_fixture_kernel",
            "workspace_fingerprint": "fixture_workspace",
            "promotion_eligible": False,
            "invalidation_reasons": ["source_version_changed"]
            if with_stale_source_rejection
            else ["descriptor_identity_changed"],
        }
        if runtime_prepack_cache is not None:
            result["reuse_contract"]["runtime_prepack_cache"] = runtime_prepack_cache
    return result


def runtime_b_cache(source_version: int = 1, production: bool = True) -> dict:
    cache = {
        "source": "rns8_get_prepack_cache_info",
        "backend": "rocwmma",
        "semantics": "bounded_i64",
        "operand_role": "B",
        "cache_key_valid": True,
        "reusable_prepack_cache_available": source_version > 0,
        "production_prepack_cache_available": production,
        "hip_device_id": 0,
        "matrix_rows": 1024,
        "matrix_cols": 1024,
        "k": 1024,
        "max_prefix": 9,
        "finite_modulus": 0,
        "source_version": source_version,
        "plan_fingerprint": 123,
        "cache_key_hash": 456,
        "device_bytes": 1024,
        "operand_pack_bytes": 1024,
        "matrix_layout_version": "rns_centered_residue_planes_v1",
        "operand_layout_version": "rns_i8_tile_swizzled_b_v1",
        "cache_scope": "runtime_production_b_prepack_cache"
        if production
        else "runtime_reusable_b_prepack_cache",
        "cache_key": "",
        "detail": "synthetic runtime cache fixture",
    }
    cache["cache_key"] = (
        "prepack-v2"
        ";backend=rocwmma"
        ";target_id=gfx1100"
        ";kernel=rocwmma_rns_gemm_v1"
        ";prepack_kernel=rocwmma_rns_i8_tile_swizzled_b_prepack_v1"
        ";semantics=bounded_i64"
        ";prefix_schedule_hash=999"
        ";tile_m=128"
        ";tile_n=128"
        ";operand_tile_m=16"
        ";operand_tile_n=16"
        ";k_block_size=64"
        ";k_block_cap=65536"
        ";operand=B"
        ";m=1024"
        ";n=1024"
        ";k=1024"
        f";source_version={source_version}"
        ";hip_device_id=0"
        ";matrix_rows=1024"
        ";matrix_cols=1024"
        ";prefix=9"
        ";finite_modulus=0"
        ";matrix_layout=rns_centered_residue_planes_v1"
        ";operand_layout=rns_i8_tile_swizzled_b_v1"
        ";plan_fingerprint=123"
        ";hash=456"
    )
    return cache


def production_controls():
    from test_diagnostic_companion import authored_capture_pair

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def collect(label, setup=900, cpu_time=5000, change=None, reuse_backend="ck"):
            captures, bindings = [], {}
            for lane, backend, duration in (
                ("same", reuse_backend, 1000),
                ("best", "hip-direct" if reuse_backend == "ck" else "ck", 900),
                ("cpu", "cpu-reference", cpu_time),
                ("reuse", reuse_backend, 700),
            ):
                for run in range(2):
                    fixture = bounded_capture(backend, duration)
                    if lane == "reuse":
                        fixture = as_reused_pack_capture(fixture)
                        add_device_allocation_fields(fixture)
                        fixture["device_allocation"]["setup_scope"] = (
                            "persistent_plan_workspace_prepacked_reuse"
                        )
                        fixture["prepack_setup_us"] = fixture["avg_prepack_setup_us"] = setup
                        breakdown = dict(
                            pack_a=setup / 2, pack_b=setup / 2, runtime_cache=0, unclassified=0
                        )
                        fixture["prepack_setup_breakdown_us"] = breakdown
                        fixture["reuse_contract"] = dict(
                            enabled=True,
                            operand_role="A+B",
                            source_version_inputs="monotonic_source_version_per_repeat_when_packing_runs",
                            setup_scope="one_time_before_warmups",
                            setup_cost_us=setup,
                            setup_breakdown_us=dict(breakdown),
                            setup_amortized_us=setup / 9,
                            repeat_median_end_to_end_us=700,
                            setup_inclusive_median_end_to_end_us=700 + setup / 9,
                            setup_inclusive_policy="one_time_setup_amortized_over_measured_repeats",
                            measured_repeat_count=9,
                            break_even_repeat_count=None,
                            output_domain="native_i64_u64_host",
                            next_op="final-export",
                            target_fingerprint=fixture["device"]["gcn_arch"],
                            backend_fingerprint=backend,
                            kernel_fingerprint=fixture["selected_kernel"],
                            workspace_fingerprint="synthetic_retained",
                            runtime_prepack_cache=None,
                            production_runtime_prepack_cache_available=False,
                            setup_inclusive_cache_promotion_candidate=False,
                            promotion_eligible=False,
                            invalidation_reasons=["source_version_changed"],
                        )
                    if change is not None:
                        change(fixture, lane, run)
                    add_synthetic_full_reference(fixture)
                    path = root / f"{label}-{lane}-{run}"
                    path.mkdir()
                    _, _, paths = authored_capture_pair(path, fixture)
                    captures.extend(load_report_captures([paths[0]]))
                    if lane != "cpu":
                        bindings[str(paths[0].resolve())] = build_binding(*paths)
            return captures, bindings

        captures, bindings = collect("matched")

        def build(values, receipts=bindings):
            return reuse_contract_report.compare_reuse_contracts(
                values, diagnostic_bindings=receipts
            )

        item = build(captures)["comparisons"][0]
        assert item["decision"] == "candidate_workload_win", item["blockers"]
        assert item["phases"]["end_to_end"]["setup_inclusive_speedup"] == 1.25
        assert item["speedup_vs_best_nonreuse_setup_inclusive"] == 1.125
        assert item["selector_eligibility"]["explicit_workload_selector_eligible"] is True
        assert len(item["process_contract_observations"]) == 2
        for order in permutations(captures[::2]):
            current = build([*order, *captures[1::2]])["comparisons"][0]
            assert current["speedup_vs_best_nonreuse_setup_inclusive"] == 1.125
        singleton = build([*captures[:6], captures[6], copy.deepcopy(captures[6])])["comparisons"][
            0
        ]
        assert singleton["speedup_vs_best_nonreuse_setup_inclusive"] is None
        missing = dict(bindings)
        missing.pop(str(Path(captures[7]["_path"]).resolve()))
        assert (
            build(captures, missing)["comparisons"][0]["speedup_vs_best_nonreuse_setup_inclusive"]
            is None
        )
        faster_cpu, receipts = collect("cpu-control", cpu_time=750)
        current = build(faster_cpu, receipts)["comparisons"][0]
        assert current["decision"] == "deprioritize"
        assert current["best_nonreuse_comparison"]["cpu_gpu_comparison"] is True
        slow, receipts = collect("setup-regression", setup=3600)
        assert build(slow, receipts)["comparisons"][0]["decision"] == "deprioritize"

        def incomplete_source(fixture, lane, run):
            if lane == "reuse" and run == 1:
                fixture.pop("device_allocation")

        incomplete, receipts = collect("one-missing-source", change=incomplete_source)
        for order in (incomplete, list(reversed(incomplete))):
            current = build(order, receipts)["comparisons"][0]
            assert current["speedup_vs_best_nonreuse_setup_inclusive"] is None
            assert "missing_device_allocation_metadata" in current["blockers"]
            assert current["selector_eligibility"]["explicit_workload_selector_eligible"] is False

        def default_promotion_claim(fixture, lane, run):
            if lane == "reuse" and run == 1:
                fixture["reuse_contract"]["promotion_eligible"] = True

        claimed, receipts = collect("one-default-promotion", change=default_promotion_claim)
        for order in (claimed, list(reversed(claimed))):
            selector = build(order, receipts)["comparisons"][0]["selector_eligibility"]
            assert selector["explicit_workload_selector_eligible"] is False
            assert "reuse_contract_marked_default_promotion_eligible" in selector["blockers"]

        with (root / "review_report.json").open("w") as stream:
            json.dump(dict(schema_version=3, groups=[]), stream)
        # The real loader, not a filename allow-list, owns directory admission.
        assert (
            reuse_contract_report.build_report([Path(captures[0]["_path"])])["capture_count"] == 1
        )

        import direct_hip_reuse_expansion_report

        direct, receipts = collect("direct-wrapper", reuse_backend="hip-direct")
        wrapped = direct_hip_reuse_expansion_report.build_report(
            [Path(record["_path"]) for record in direct], diagnostic_bindings=receipts
        )
        assert wrapped["summary"]["candidate_workload_wins"] == 1
        assert wrapped["comparisons"][0]["speedup_vs_best_nonreuse_setup_inclusive"] == 1.125
        assert len(wrapped["observations"]) == 4
        blocked = direct_hip_reuse_expansion_report.build_direct_hip_report_from_captures(direct)
        assert blocked["summary"]["candidate_workload_wins"] == 0


def main() -> int:
    same_backend = capture(backend="hipblaslt", median_us=1000.0)
    best_nonreuse = capture(backend="ck", median_us=900.0)
    winning_reuse = capture(
        backend="hipblaslt", median_us=700.0, pack_mode="prepacked_reuse_b", setup_us=900.0
    )
    report = reuse_contract_report.compare_reuse_contracts(
        [same_backend, best_nonreuse, winning_reuse]
    )
    assert report["summary"]["candidate_workload_wins"] == 0
    item = report["comparisons"][0]
    assert item["decision"] == "keep_experimental"
    assert item["break_even_repeats_same_backend"] is None
    assert item["phases"]["end_to_end"]["setup_inclusive_speedup"] is None
    assert item["speedup_vs_best_nonreuse_setup_inclusive"] is None
    assert item["prepack_setup_breakdown_us"] == {
        "pack_a": 0.0,
        "pack_b": 900.0,
        "runtime_cache": 0.0,
        "unclassified": 0.0,
    }
    assert item["prepack_setup_primary_phase"] == "pack_b"
    assert item["reuse_contract"]["setup_primary_phase"] == "pack_b"
    assert report["summary"]["prepack_setup_primary_phase_counts"] == {"pack_b": 1}
    assert item["stale_source_rejection"]["available"] is True
    assert item["same_workload_family"]["available"] is True
    assert item["selector_eligibility"]["explicit_workload_selector_eligible"] is False
    assert item["selector_eligibility"]["autotune_selector_eligible"] is False
    assert report["summary"]["explicit_workload_selector_ready"] == 0

    slow_setup = copy.deepcopy(winning_reuse)
    slow_setup["_path"] = "slow-setup.json"
    slow_setup["avg_prepack_setup_us"] = 3600.0
    slow_report = reuse_contract_report.compare_reuse_contracts(
        [same_backend, best_nonreuse, slow_setup]
    )
    slow_item = slow_report["comparisons"][0]
    assert slow_item["decision"] == "keep_experimental"
    assert slow_item["break_even_repeats_same_backend"] is None
    assert slow_item["phases"]["end_to_end"]["setup_inclusive_speedup"] is None

    missing_identity = copy.deepcopy(winning_reuse)
    missing_identity["_path"] = "missing-identity.json"
    del missing_identity["device_allocation"]
    identity_report = reuse_contract_report.compare_reuse_contracts(
        [same_backend, best_nonreuse, missing_identity]
    )
    identity_item = identity_report["comparisons"][0]
    assert identity_item["decision"] == "keep_experimental"
    assert "missing_device_allocation_metadata" in identity_item["blockers"]

    missing_stale_rejection = capture(
        backend="hipblaslt",
        median_us=700.0,
        pack_mode="prepacked_reuse_b",
        setup_us=900.0,
        with_stale_source_rejection=False,
    )
    stale_report = reuse_contract_report.compare_reuse_contracts(
        [same_backend, best_nonreuse, missing_stale_rejection]
    )
    stale_item = stale_report["comparisons"][0]
    assert stale_item["decision"] == "keep_experimental"
    assert "reuse_contract_missing_source_version_invalidation" in stale_item["blockers"]
    assert stale_item["selector_eligibility"]["explicit_workload_selector_eligible"] is False

    legacy_reuse = capture(
        backend="hipblaslt",
        median_us=700.0,
        pack_mode="prepacked_reuse_b",
        setup_us=900.0,
        with_reuse_contract=False,
    )
    legacy_report = reuse_contract_report.compare_reuse_contracts(
        [same_backend, best_nonreuse, legacy_reuse]
    )
    legacy_item = legacy_report["comparisons"][0]
    assert legacy_item["decision"] == "keep_experimental"
    assert legacy_item["stale_source_rejection"]["available"] is True
    assert legacy_item["selector_eligibility"]["explicit_workload_selector_eligible"] is False
    assert "missing_reuse_contract_metadata" in legacy_item["selector_eligibility"]["blockers"]

    rocwmma_same_backend = capture(backend="rocwmma", median_us=1000.0)
    rocwmma_best_nonreuse = capture(backend="hip-direct", median_us=900.0)
    rocwmma_runtime_reuse = capture(
        backend="rocwmma",
        median_us=700.0,
        pack_mode="prepacked_reuse_b",
        setup_us=900.0,
        with_source_identity=False,
        prepack_reuse_strategy="rocwmma_reusable_b_cache",
        runtime_prepack_cache=runtime_b_cache(source_version=77, production=True),
    )
    runtime_report = reuse_contract_report.compare_reuse_contracts(
        [rocwmma_same_backend, rocwmma_best_nonreuse, rocwmma_runtime_reuse]
    )
    runtime_item = runtime_report["comparisons"][0]
    assert runtime_item["decision"] == "keep_experimental"
    assert runtime_item["prepack_setup_primary_phase"] == "pack_b"
    assert runtime_item["prepack_setup_breakdown_us"]["pack_b"] == 900.0
    assert runtime_item["source_identity"]["reason"] == "runtime_prepack_cache_source_identity"
    assert runtime_item["runtime_prepack_cache"]["available"] is True
    assert runtime_item["runtime_prepack_cache"]["production_available"] is True
    assert runtime_item["selector_eligibility"]["runtime_prepack_cache_required"] is True
    assert runtime_item["selector_eligibility"]["explicit_workload_selector_eligible"] is False
    assert runtime_report["summary"]["runtime_prepack_cache_ready"] == 1
    assert runtime_report["summary"]["runtime_prepack_cache_production_ready"] == 1

    zero_source_runtime_reuse = copy.deepcopy(rocwmma_runtime_reuse)
    zero_source_runtime_reuse["_path"] = "zero-source-runtime-cache.json"
    zero_source_runtime_reuse["reuse_contract"]["runtime_prepack_cache"] = runtime_b_cache(
        source_version=0,
        production=True,
    )
    zero_source_report = reuse_contract_report.compare_reuse_contracts(
        [rocwmma_same_backend, rocwmma_best_nonreuse, zero_source_runtime_reuse]
    )
    zero_source_item = zero_source_report["comparisons"][0]
    assert zero_source_item["decision"] == "keep_experimental"
    assert (
        "runtime_prepack_cache_missing_reusable_source_version_cache_key_source"
        in zero_source_item["blockers"]
    )
    assert zero_source_item["selector_eligibility"]["explicit_workload_selector_eligible"] is False

    nonproduction_runtime_reuse = copy.deepcopy(rocwmma_runtime_reuse)
    nonproduction_runtime_reuse["_path"] = "nonproduction-runtime-cache.json"
    nonproduction_runtime_reuse["reuse_contract"]["runtime_prepack_cache"] = runtime_b_cache(
        source_version=77,
        production=False,
    )
    nonproduction_report = reuse_contract_report.compare_reuse_contracts(
        [rocwmma_same_backend, rocwmma_best_nonreuse, nonproduction_runtime_reuse]
    )
    nonproduction_item = nonproduction_report["comparisons"][0]
    assert nonproduction_item["decision"] == "keep_experimental"
    assert nonproduction_item["runtime_prepack_cache"]["available"] is True
    assert nonproduction_item["runtime_prepack_cache"]["production_available"] is False
    assert (
        "runtime_prepack_cache_not_production_available"
        in nonproduction_item["selector_eligibility"]["blockers"]
    )
    assert (
        nonproduction_item["selector_eligibility"]["explicit_workload_selector_eligible"] is False
    )

    missing_baseline_report = reuse_contract_report.compare_reuse_contracts(
        [best_nonreuse, winning_reuse]
    )
    missing_item = missing_baseline_report["comparisons"][0]
    assert missing_item["decision"] == "missing_baseline"
    assert "missing_same_backend_nonreuse_baseline" in missing_item["blockers"]

    production_controls()
    print("reuse contract report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
