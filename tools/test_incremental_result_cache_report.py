#!/usr/bin/env python3
"""Self-test incremental result-cache research report gates."""

from __future__ import annotations

from test_review_reference_support import add_synthetic_full_reference

import copy
import json
import tempfile
from pathlib import Path

import incremental_result_cache_report
from benchmark_schema import validate_capture
from test_benchmark_schema_support_metadata import add_target_variant_fields
from test_benchmark_sweep_support import bounded_capture, exact_wide_capture, set_phase


def cache_capture(backend: str = "hip-direct") -> dict:
    capture = bounded_capture("cpu-reference" if backend == "cpu-reference" else backend, 1000)
    capture["scenario_metadata"] = {
        "family": "incremental-result-cache",
        "name": "bounded-i64-dirty-tile-recompute",
        "promotion_eligibility": "result_cache_research_only",
        "metadata": {
            "promotion_scope": "result_cache_research_only",
            "source_identity": "A_and_B_matrix_identity_required",
            "source_version": "monotonic_per_matrix_version",
            "dirty_region_shape": "single_output_tile",
            "result_lifetime": "caller_visible_reuse_window_required",
            "checksum_policy": "final_exact_cpu_comparison_required",
        },
    }
    capture["incremental_result_cache"] = {
        "enabled": True,
        "policy": "bounded_i64_dirty_tile_partial_recompute_research",
        "source_identity_policy": "caller_visible_source_identity_required",
        "source_version_policy": "monotonic_source_versions_required",
        "dirty_region_policy": "explicit_dirty_region_shape_required",
        "result_lifetime_policy": "caller_visible_result_lifetime_required",
        "checksum_policy": "final_exact_cpu_comparison_required",
        "partial_recompute_policy": "dirty_region_only_when_version_contract_matches",
        "final_exact_comparison_required": True,
        "final_exact_comparison_status": "checksum_recorded_reference_required",
        "public_contract_available": False,
        "default_gemm_unchanged": True,
        "runtime_routing_allowed": False,
        "cache_eligible": False,
        "promotion_eligible": False,
    }
    if backend != "cpu-reference":
        if backend == "hip-direct":
            capture["timing_metadata"]["gpu_event_timing_source_scope"] = (
                "direct_hip_default_stream_backend_operation_groups"
            )
            capture["timing_metadata"]["gpu_event_timing_reason"] = (
                "captured_by_direct_hip_backend_hooks"
            )
        add_target_variant_fields(capture)
    return add_synthetic_full_reference(capture)


def public_cache_capture(
    backend: str, end_to_end: int, role: str, public_candidate: bool = False
) -> dict:
    capture = cache_capture(backend)
    capture["scenario_metadata"]["promotion_eligibility"] = "result_cache_contract_candidate"
    capture["scenario_metadata"]["metadata"]["promotion_scope"] = "result_cache_contract_candidate"
    capture["incremental_result_cache"].update(
        {
            "policy": "bounded_i64_dirty_tile_public_contract_v1",
            "candidate_role": role,
            "source_identity_policy": "matrix_instance_id_exact_match",
            "source_version_policy": "nonzero_monotonic_source_versions_required",
            "dirty_region_policy": "caller_provided_output_rectangles_full_k_recompute",
            "result_lifetime_policy": "explicit_result_cache_handle_lifetime",
            "partial_recompute_policy": "restore_cached_output_then_recompute_dirty_rectangles_full_k",
        }
    )
    if public_candidate:
        capture["incremental_result_cache"].update(
            {
                "public_contract_available": True,
                "runtime_routing_allowed": True,
                "cache_eligible": True,
                "promotion_eligible": True,
                "a_matrix_instance_id": 101,
                "b_matrix_instance_id": 102,
                "a_source_version": 11,
                "b_source_version": 12,
                "result_cache_key_fingerprint": 0xABCDEF,
                "dirty_region_count": 1,
                "recomputed_region_count": 1,
                "copied_from_cache_bytes": 512 * 512 * 9,
                "cache_allocation_bytes": 512 * 512 * 9,
                "stale_rejection_covered": True,
                "dirty_regions": [
                    {
                        "row_offset": 0,
                        "col_offset": 0,
                        "row_extent": 32,
                        "col_extent": 32,
                    }
                ],
            }
        )
    set_phase(capture, end_to_end)
    return add_synthetic_full_reference(capture)


def write(path: Path, capture: dict) -> Path:
    prefix = int(capture.get("selected_prefix") or capture.get("prefix") or 1)
    if capture.get("per_modulus_gemm_estimate_applicable") is False:
        capture["avg_per_modulus_gemm_estimate_us"] = float(capture["avg_rns_gemm_us"])
    else:
        capture["avg_per_modulus_gemm_estimate_us"] = float(capture["avg_rns_gemm_us"]) / float(
            prefix
        )
    validate_capture(capture, path)
    path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def production_controls():
    from itertools import permutations
    from diagnostic_companion import build_binding
    from test_diagnostic_companion import authored_capture_pair

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        captures, bindings, blocks = [], {}, []
        cases = (
            ("cpu-reference", 5000, "cpu_reference_baseline", False),
            ("hip-direct", 1000, "same_backend_full_recompute_baseline", False),
            ("hip-direct", 800, "public_result_cache_candidate", True),
            ("ck", 1200, "comparison_candidate", False),
        )
        for backend, duration, role, public in cases:
            block = []
            for run in range(2):
                fixture = public_cache_capture(backend, duration, role, public)
                fixture["scenario_metadata"]["name"] = role
                fixture["scenario_metadata"]["metadata"]["result_cache_contract_group"] = (
                    "matched-cache-contract"
                )
                lane = root / f"{role}-{run}"
                lane.mkdir()
                _, _, paths = authored_capture_pair(lane, fixture)
                capture = incremental_result_cache_report.load_report_captures([paths[0]])[0]
                block.append(capture)
                if backend != "cpu-reference":
                    bindings[str(paths[0].resolve())] = build_binding(*paths)
            captures.extend(block)
            blocks.append(block)

        def build(items, receipts=bindings):
            return incremental_result_cache_report.build_report_from_captures(
                items, diagnostic_bindings=receipts
            )

        def public_group(report):
            return next(group for group in report["groups"] if group["public_contract_available"])

        report = build(captures)
        assert report["rank78_gate_complete"], report["promotion_blocker_counts"]
        assert report["promotable_result_cache_candidate_count"] == 1
        group = public_group(report)
        assert group["promotion_decision"] == "promote"
        assert group["details"]["best_speedup_vs_full_recompute"] == 1.25
        comparison = group["details"]["comparisons"][0]["comparison"]
        assert comparison["qualified"]
        assert comparison["matching_work"] and comparison["matching_environment"]
        assert comparison["candidate_measurement_cohort"]["independent_runs"] == 2
        assert comparison["baseline_measurement_cohort"]["independent_runs"] == 2
        assert comparison["candidate_numerical_reference"]["passed"]
        for order in permutations(blocks):
            permuted = build([item for block in order for item in block])
            assert public_group(permuted)["details"]["best_speedup_vs_full_recompute"] == 1.25
            assert permuted["rank78_gate_complete"]
        copied = build(captures + [copy.deepcopy(captures[5])])
        assert copied["rank78_gate_complete"]
        assert (
            public_group(copied)["details"]["comparisons"][0]["comparison"][
                "candidate_measurement_cohort"
            ]["independent_runs"]
            == 2
        )
        singleton = build(captures[:5] + [copy.deepcopy(captures[4])] + captures[6:])
        assert not singleton["rank78_gate_complete"]
        assert public_group(singleton)["details"]["best_speedup_vs_full_recompute"] is None
        missing = dict(bindings)
        missing.pop(str(Path(captures[5]["_path"]).resolve()))
        report = build(captures, missing)
        assert not report["rank78_gate_complete"]
        assert "matched_diagnostic_companion_missing" in public_group(report)["promotion_blockers"]
        assert public_group(report)["details"]["best_speedup_vs_full_recompute"] is None
        optional_missing = dict(bindings)
        optional_missing.pop(str(Path(captures[-1]["_path"]).resolve()))
        report = build(captures, optional_missing)
        assert report["rank78_gate_complete"]
        assert any(
            row["blockers"]
            for group in report["groups"]
            for row in group["rows"]
            if row["backend"] == "ck"
        )
        for field, value in (
            ("seed", captures[4]["seed"] + 1),
            ("m", captures[4]["m"] + 1),
            ("device", dict(captures[4]["device"], name="Other physical device")),
        ):
            changed = copy.deepcopy(captures)
            for item in changed[4:6]:
                item[field] = value
            report = build(changed)
            assert not report["rank78_gate_complete"], field
            assert public_group(report)["details"]["best_speedup_vs_full_recompute"] is None
        zero = copy.deepcopy(captures)
        for item in zero[4:6]:
            item["warmups"] = 0
        assert not build(zero)["rank78_gate_complete"]
        no_reference = copy.deepcopy(captures)
        for item in no_reference[4:6]:
            item.pop("native_reference_verification")
        assert not build(no_reference)["rank78_gate_complete"]
        # An unrelated device cannot provide a baseline or poison the original
        # qualified pair; keep it as a separate visible, excluded partition.
        unrelated = copy.deepcopy(captures[4:6])
        for item in unrelated:
            item["device"]["name"] = "Unrelated physical device"
        report = build(captures + unrelated)
        assert report["rank78_gate_complete"]
        assert sum(group["promotion_decision"] == "promote" for group in report["groups"]) == 1
        loaded = incremental_result_cache_report.build_report(
            [Path(item["_path"]) for item in captures], diagnostic_bindings=bindings
        )
        assert loaded["rank78_gate_complete"]

        # Re-author both production and diagnostic files after timing changes.
        # These are authored values, never a benchmark or performance claim.
        for label, candidate_times, full_times, expected_blocker in (
            ("weak", (950, 950), (1000, 1000), "speedup_below_1_10x"),
            ("noisy", (800, 800), (1000, 2000), "same_backend_full_recompute_baseline_missing"),
        ):
            changed, receipts = list(captures[:2] + captures[6:]), dict(bindings)
            for role, durations, public in (
                ("same_backend_full_recompute_baseline", full_times, False),
                ("public_result_cache_candidate", candidate_times, True),
            ):
                for run, duration in enumerate(durations):
                    fixture = public_cache_capture("hip-direct", duration, role, public)
                    fixture["scenario_metadata"]["name"] = role
                    fixture["scenario_metadata"]["metadata"]["result_cache_contract_group"] = (
                        "matched-cache-contract"
                    )
                    lane = root / f"{label}-{role}-{run}"
                    lane.mkdir()
                    _, _, paths = authored_capture_pair(lane, fixture)
                    changed.append(
                        incremental_result_cache_report.load_report_captures([paths[0]])[0]
                    )
                    receipts[str(paths[0].resolve())] = build_binding(*paths)
            report = build(changed, receipts)
            assert not report["rank78_gate_complete"]
            assert expected_blocker in public_group(report)["promotion_blockers"]


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cpu = write(root / "cpu.json", cache_capture("cpu-reference"))
        direct = write(root / "direct.json", cache_capture("hip-direct"))
        report = incremental_result_cache_report.build_report([cpu, direct])
        assert report["schema"] == "rns8_incremental_result_cache_report_v1"
        assert not report["rank75_gate_complete"]
        assert not report["rank78_gate_complete"]
        assert report["promotable_result_cache_candidate_count"] == 0
        assert all(
            group["promotion_decision"] == "blocked_public_incremental_cache_contract_required"
            for group in report["groups"]
        )
        single_public = public_cache_capture(
            "hip-direct", 800, "public_result_cache_candidate", True
        )
        public = write(root / "candidate.json", single_public)
        report = incremental_result_cache_report.build_report([cpu, direct, public])
        assert not report["rank78_gate_complete"]
        assert all(
            group["details"].get("best_speedup_vs_full_recompute") is None
            for group in report["groups"]
        )
        exact = exact_wide_capture("hip-direct", 1200)
        template = cache_capture("hip-direct")
        exact["scenario_metadata"] = copy.deepcopy(template["scenario_metadata"])
        exact["scenario_metadata"]["name"] = "exact-wide-partial-output-recompute"
        exact["incremental_result_cache"] = copy.deepcopy(template["incremental_result_cache"])
        exact["timing_metadata"]["gpu_event_timing_source_scope"] = (
            "direct_hip_default_stream_backend_operation_groups"
        )
        exact["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_backend_hooks"
        add_target_variant_fields(exact)
        path = write(root / "exact.json", add_synthetic_full_reference(exact))
        assert not incremental_result_cache_report.build_report([path])["rank75_gate_complete"]
        # Invalid policy flags must be rejected by the actual schema or reported
        # with their specific blocker, never swallowed as an arbitrary exception.
        from benchmark_schema import BenchmarkSchemaError

        for field, value, blocker in (
            ("dirty_region_policy", "none", "dirty_region_policy_missing"),
            ("promotion_eligible", True, "promotion_eligible_must_be_false"),
        ):
            fixture = cache_capture("hip-direct")
            fixture["incremental_result_cache"][field] = value
            try:
                validate_capture(fixture)
            except BenchmarkSchemaError:
                continue
            rows = incremental_result_cache_report.build_report_from_captures([fixture])
            assert any(
                blocker in row["blockers"] for group in rows["groups"] for row in group["rows"]
            )
    production_controls()
    print("incremental result-cache report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
