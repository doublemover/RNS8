#!/usr/bin/env python3
"""Self-test CPU small-shape selector report gates."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import cpu_small_shape_selector_report
from benchmark_schema import validate_capture
from test_benchmark_schema_support_metadata import add_target_variant_fields
from test_benchmark_sweep_support import bounded_capture, finite_capture, set_phase


def selector_capture(backend: str, median_us: int, *, name: str = "bounded-i64-32-cutoff") -> dict:
    capture = bounded_capture("cpu-reference" if backend == "cpu-reference" else backend, median_us)
    capture["scenario_metadata"] = {
        "family": "cpu-small-shape-selector",
        "name": name,
        "promotion_eligibility": "cpu_selector_threshold_evidence_only",
        "shape": {"m": capture["m"], "n": capture["n"], "k": capture["k"]},
        "output_domain": "native_i64_u64_host",
        "metadata": {"promotion_scope": "cpu_selector_threshold_evidence_only"},
    }
    capture["cpu_small_shape_selector"] = {
        "enabled": True,
        "policy": "bounded_i64_32_cpu_cutoff_review",
        "candidate_role": "cpu_baseline" if backend == "cpu-reference" else "comparison_candidate",
        "boundary_key": f"semantics={capture['semantics']};name={name};target={capture.get('target_id')}",
        "threshold_scope": "semantic_layout_output_target_family_explicit_review_only",
        "selector_explanation": "non_routing_auto_explanation_metadata_only",
        "cpu_reference_required": True,
        "release_review_required": True,
        "runtime_routing_allowed": False,
        "cache_eligible": False,
        "promotion_eligible": False,
    }
    if backend != "cpu-reference":
        if backend == "hip-direct":
            capture["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
            capture["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_backend_hooks"
        add_target_variant_fields(capture)
    set_phase(capture, median_us)
    return capture


def finite_selector_capture(backend: str, median_us: int) -> dict:
    capture = finite_capture("cpu-reference" if backend == "cpu-reference" else backend, median_us)
    capture["scenario_metadata"] = {
        "family": "cpu-small-shape-selector",
        "name": "finite-ring-u8-64-cutoff",
        "promotion_eligibility": "cpu_selector_threshold_evidence_only",
        "shape": {"m": capture["m"], "n": capture["n"], "k": capture["k"]},
        "modulus": 251,
        "output_domain": "finite_u8_canonical_host_export",
        "metadata": {"promotion_scope": "cpu_selector_threshold_evidence_only"},
    }
    capture["cpu_small_shape_selector"] = {
        "enabled": True,
        "policy": "finite_u8_64_cpu_cutoff_review",
        "candidate_role": "cpu_baseline" if backend == "cpu-reference" else "comparison_candidate",
        "boundary_key": f"semantics={capture['semantics']};modulus=251;target={capture.get('target_id')}",
        "threshold_scope": "semantic_modulus_layout_output_target_family_explicit_review_only",
        "selector_explanation": "non_routing_auto_explanation_metadata_only",
        "cpu_reference_required": True,
        "release_review_required": True,
        "runtime_routing_allowed": False,
        "cache_eligible": False,
        "promotion_eligible": False,
    }
    if backend != "cpu-reference":
        if backend == "hip-direct":
            capture["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
            capture["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_backend_hooks"
        add_target_variant_fields(capture)
    set_phase(capture, median_us)
    return capture


def write(path: Path, capture: dict) -> Path:
    prefix = int(capture.get("selected_prefix") or capture.get("prefix") or 1)
    if capture.get("per_modulus_gemm_estimate_applicable") is False:
        capture["avg_per_modulus_gemm_estimate_us"] = float(capture["avg_rns_gemm_us"])
    else:
        capture["avg_per_modulus_gemm_estimate_us"] = float(capture["avg_rns_gemm_us"]) / float(prefix)
    validate_capture(capture, path)
    path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        cpu = write(tmp / "cpu.json", selector_capture("cpu-reference", 900))
        direct = write(tmp / "direct.json", selector_capture("hip-direct", 1200))
        report = cpu_small_shape_selector_report.build_report([cpu, direct])
        assert report["schema"] == "rns8_cpu_small_shape_selector_report_v1"
        assert report["rank69_gate_complete"] is True, json.dumps(report["blocker_counts"], indent=2)
        assert report["groups"][0]["recommendation"] == "cpu_wins"
        assert report["groups"][0]["promotion_decision"] == "promote_cpu_selector_threshold"
        assert report["groups"][0]["promoted_backend"] == "cpu-reference"
        assert report["promotable_selector_threshold_count"] == 1

        gpu_win_dir = tmp / "gpu-win"
        gpu_win_dir.mkdir()
        write(gpu_win_dir / "cpu.json", selector_capture("cpu-reference", 1500))
        write(gpu_win_dir / "ck.json", selector_capture("ck", 700))
        gpu_report = cpu_small_shape_selector_report.build_report([gpu_win_dir])
        assert gpu_report["groups"][0]["recommendation"] == "gpu_wins"
        assert gpu_report["groups"][0]["promotion_decision"] == "promote_gpu_selector_threshold"
        assert gpu_report["groups"][0]["promoted_backend"] == "ck"
        assert gpu_report["promotable_selector_threshold_count"] == 1

        boundary_dir = tmp / "boundary"
        boundary_dir.mkdir()
        write(boundary_dir / "cpu.json", selector_capture("cpu-reference", 1000))
        write(boundary_dir / "direct.json", selector_capture("hip-direct", 1030))
        boundary_report = cpu_small_shape_selector_report.build_report([boundary_dir])
        assert boundary_report["groups"][0]["recommendation"] == "threshold_boundary"
        assert boundary_report["groups"][0]["promotion_decision"] == "keep_threshold_boundary"
        assert boundary_report["promotable_selector_threshold_count"] == 0

        no_cpu_dir = tmp / "no-cpu"
        no_cpu_dir.mkdir()
        write(no_cpu_dir / "direct.json", selector_capture("hip-direct", 900))
        no_cpu_report = cpu_small_shape_selector_report.build_report([no_cpu_dir])
        assert no_cpu_report["rank69_gate_complete"] is False
        assert "cpu_reference_baseline_missing" in no_cpu_report["groups"][0]["blockers"]

        bad = selector_capture("hip-direct", 900)
        bad["cpu_small_shape_selector"]["runtime_routing_allowed"] = True
        try:
            write(tmp / "bad-routing.json", bad)
        except Exception:
            bad["cpu_small_shape_selector"]["runtime_routing_allowed"] = False
            bad["cpu_small_shape_selector"]["cache_eligible"] = True
            try:
                write(tmp / "bad-cache.json", bad)
            except Exception:
                pass

        finite_dir = tmp / "finite-boundary"
        finite_dir.mkdir()
        write(finite_dir / "cpu.json", finite_selector_capture("cpu-reference", 800))
        write(finite_dir / "ck.json", finite_selector_capture("ck", 600))
        combined = cpu_small_shape_selector_report.build_report([tmp, finite_dir])
        names = {group["name"] for group in combined["groups"]}
        assert "bounded-i64-32-cutoff" in names
        assert "finite-ring-u8-64-cutoff" in names

    print("cpu small-shape selector report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
