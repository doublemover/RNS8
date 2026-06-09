#!/usr/bin/env python3
"""Self-test reuse contract comparison reporting."""

from __future__ import annotations

import copy

import reuse_contract_report


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


def main() -> int:
    same_backend = capture(backend="hipblaslt", median_us=1000.0)
    best_nonreuse = capture(backend="ck", median_us=900.0)
    winning_reuse = capture(backend="hipblaslt", median_us=700.0, pack_mode="prepacked_reuse_b", setup_us=900.0)
    report = reuse_contract_report.compare_reuse_contracts([same_backend, best_nonreuse, winning_reuse])
    assert report["summary"]["candidate_workload_wins"] == 1
    item = report["comparisons"][0]
    assert item["decision"] == "candidate_workload_win"
    assert item["break_even_repeats_same_backend"] == 4
    assert round(item["phases"]["end_to_end"]["setup_inclusive_speedup"], 4) == 1.25
    assert round(item["speedup_vs_best_nonreuse_setup_inclusive"], 4) == 1.125
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
    assert item["selector_eligibility"]["explicit_workload_selector_eligible"] is True
    assert item["selector_eligibility"]["autotune_selector_eligible"] is False
    assert report["summary"]["explicit_workload_selector_ready"] == 1

    slow_setup = copy.deepcopy(winning_reuse)
    slow_setup["_path"] = "slow-setup.json"
    slow_setup["avg_prepack_setup_us"] = 3600.0
    slow_report = reuse_contract_report.compare_reuse_contracts([same_backend, best_nonreuse, slow_setup])
    slow_item = slow_report["comparisons"][0]
    assert slow_item["decision"] == "deprioritize"
    assert slow_item["break_even_repeats_same_backend"] == 13
    assert "repeat_count_below_same_backend_break_even" in slow_item["blockers"]

    missing_identity = copy.deepcopy(winning_reuse)
    missing_identity["_path"] = "missing-identity.json"
    del missing_identity["device_allocation"]
    identity_report = reuse_contract_report.compare_reuse_contracts([same_backend, best_nonreuse, missing_identity])
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
    legacy_report = reuse_contract_report.compare_reuse_contracts([same_backend, best_nonreuse, legacy_reuse])
    legacy_item = legacy_report["comparisons"][0]
    assert legacy_item["decision"] == "candidate_workload_win"
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
    assert runtime_item["decision"] == "candidate_workload_win"
    assert runtime_item["prepack_setup_primary_phase"] == "pack_b"
    assert runtime_item["prepack_setup_breakdown_us"]["pack_b"] == 900.0
    assert runtime_item["source_identity"]["reason"] == "runtime_prepack_cache_source_identity"
    assert runtime_item["runtime_prepack_cache"]["available"] is True
    assert runtime_item["runtime_prepack_cache"]["production_available"] is True
    assert runtime_item["selector_eligibility"]["runtime_prepack_cache_required"] is True
    assert runtime_item["selector_eligibility"]["explicit_workload_selector_eligible"] is True
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
    assert "runtime_prepack_cache_missing_reusable_source_version_cache_key_source" in zero_source_item["blockers"]
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
    assert nonproduction_item["decision"] == "candidate_workload_win"
    assert nonproduction_item["runtime_prepack_cache"]["available"] is True
    assert nonproduction_item["runtime_prepack_cache"]["production_available"] is False
    assert "runtime_prepack_cache_not_production_available" in nonproduction_item["selector_eligibility"]["blockers"]
    assert nonproduction_item["selector_eligibility"]["explicit_workload_selector_eligible"] is False

    missing_baseline_report = reuse_contract_report.compare_reuse_contracts([best_nonreuse, winning_reuse])
    missing_item = missing_baseline_report["comparisons"][0]
    assert missing_item["decision"] == "missing_baseline"
    assert "missing_same_backend_nonreuse_baseline" in missing_item["blockers"]

    print("reuse contract report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
