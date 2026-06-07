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
        "prepack_reuse_strategy": "persistent_matrix_residency" if reuse else "none",
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
            "prepack_reuse_strategy": "persistent_matrix_residency" if reuse else "none",
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
    return result


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

    missing_baseline_report = reuse_contract_report.compare_reuse_contracts([best_nonreuse, winning_reuse])
    missing_item = missing_baseline_report["comparisons"][0]
    assert missing_item["decision"] == "missing_baseline"
    assert "missing_same_backend_nonreuse_baseline" in missing_item["blockers"]

    print("reuse contract report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
