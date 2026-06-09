#!/usr/bin/env python3
"""Self-test Direct-HIP reuse expansion reporting."""

from __future__ import annotations

import copy

import direct_hip_reuse_expansion_report
from test_reuse_contract_report import capture


def with_scenario_metadata(item: dict, *, profile: str, role: str) -> dict:
    result = copy.deepcopy(item)
    result["scenario_metadata"] = {
        "family": "direct-hip-reuse-expansion",
        "name": f"{profile}-{role}",
        "promotion_eligibility": "reuse_contract_evidence_only" if role != "nonreuse_baseline" else "baseline_only",
        "metadata": {
            "workflow_name": "direct_hip_reuse_expansion",
            "reuse_profile": profile,
            "reuse_contract_role": role,
        },
    }
    return result


def bounded_adaptive_capture(
    *,
    backend: str,
    median_us: float,
    pack_mode: str = "per_repeat_repack",
    semantics: str = "bounded_u64",
) -> dict:
    item = capture(
        backend=backend,
        median_us=median_us,
        pack_mode=pack_mode,
        setup_us=90.0 if pack_mode != "per_repeat_repack" else None,
        repeats=9,
    )
    if semantics not in {"bounded_i64", "bounded_u64"}:
        raise ValueError(f"unsupported bounded semantics: {semantics}")
    profile_semantics = "i64" if semantics == "bounded_i64" else "u64"
    item["_path"] = f"bounded-{profile_semantics}-adaptive-{backend}-{pack_mode}.json"
    item["semantics"] = semantics
    item["bound_kind"] = "global_max_abs" if semantics == "bounded_i64" else "global_max_unsigned"
    item["input_distribution"] = (
        "signed_adaptive_bands_-16_16" if semantics == "bounded_i64" else "unsigned_adaptive_bands_0_16"
    )
    role = {
        "prepacked_reuse_a": "stable_a_candidate",
        "prepacked_reuse_b": "stable_b_candidate",
    }.get(pack_mode, "nonreuse_baseline")
    return with_scenario_metadata(item, profile=f"adaptive_bounded_{profile_semantics}_colpair", role=role)


def finite_capture(*, backend: str, median_us: float, pack_mode: str = "per_repeat_repack") -> dict:
    item = capture(
        backend=backend,
        median_us=median_us,
        pack_mode=pack_mode,
        setup_us=180.0 if pack_mode != "per_repeat_repack" else None,
        repeats=9,
    )
    item["_path"] = f"finite-ring-{backend}-{pack_mode}.json"
    item["semantics"] = "finite_ring_u8"
    item["finite_modulus"] = 251
    item["input_distribution"] = "u8_uniform_mod_251"
    role = "stable_b_candidate" if pack_mode == "prepacked_reuse_b" else "nonreuse_baseline"
    return with_scenario_metadata(item, profile="finite_ring_native_a_reuse_b", role=role)


def main() -> int:
    captures = [
        bounded_adaptive_capture(backend="hip-direct", median_us=1000.0),
        bounded_adaptive_capture(backend="ck", median_us=980.0),
        bounded_adaptive_capture(backend="hip-direct", median_us=820.0, pack_mode="prepacked_reuse_a"),
        bounded_adaptive_capture(backend="hip-direct", median_us=760.0, pack_mode="prepacked_reuse_b"),
        bounded_adaptive_capture(backend="hip-direct", median_us=990.0, semantics="bounded_i64"),
        bounded_adaptive_capture(
            backend="hip-direct", median_us=830.0, pack_mode="prepacked_reuse_a", semantics="bounded_i64"
        ),
        bounded_adaptive_capture(
            backend="hip-direct", median_us=790.0, pack_mode="prepacked_reuse_b", semantics="bounded_i64"
        ),
        finite_capture(backend="hip-direct", median_us=1500.0),
        finite_capture(backend="hipblaslt", median_us=1450.0),
        finite_capture(backend="hip-direct", median_us=1100.0, pack_mode="prepacked_reuse_b"),
    ]
    report = direct_hip_reuse_expansion_report.build_direct_hip_report_from_captures(captures)
    summary = report["summary"]
    assert summary["direct_hip_reuse_comparisons"] == 5
    assert summary["rank20_scope_comparisons"] == 5
    assert summary["candidate_workload_wins"] == 5
    assert summary["explicit_workload_selector_ready"] == 5
    assert summary["profiles"] == {
        "adaptive_bounded_i64_colpair": {"candidate_workload_win": 2},
        "adaptive_bounded_u64_colpair": {"candidate_workload_win": 2},
        "finite_ring_native_a_reuse_b": {"candidate_workload_win": 1},
    }
    assert {item["scenario_family"] for item in report["comparisons"]} == {"direct-hip-reuse-expansion"}
    assert {item["scenario_reuse_role"] for item in report["comparisons"]} == {
        "stable_a_candidate",
        "stable_b_candidate",
    }

    missing = direct_hip_reuse_expansion_report.build_direct_hip_report_from_captures([captures[-1]])
    assert missing["summary"]["missing_baselines"] == 1
    assert missing["comparisons"][0]["decision"] == "missing_baseline"

    print("direct-HIP reuse expansion report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
