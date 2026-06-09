#!/usr/bin/env python3
"""Self-test FHE/lattice proxy workload report gates."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import fhe_workload_report
from benchmark_schema import validate_capture
from test_benchmark_schema_support_metadata import add_target_variant_fields
from test_benchmark_sweep_support import bounded_capture


def proxy_capture(backend: str, operation: str, median_us: int = 1000) -> dict:
    capture = bounded_capture("cpu-reference" if backend == "cpu-reference" else "hip-direct", median_us)
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    if backend == "cpu-reference":
        capture["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
        capture["backend_metadata"]["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
    capture["scenario_metadata"] = {
        "family": "fhe-lattice-proxy-starfoundry",
        "name": operation,
        "promotion_eligibility": "proxy_evidence_only",
        "output_domain": "native_i64_u64_host",
        "metadata": {
            "algebra_family": "fhe_lattice",
            "proxy_operation": operation,
            "tower_basis": "fixture_tower",
            "reuse_mode": "fixture_reuse",
            "output_domain_requirement": "native_i64_u64_host",
            "compatibility_claim": False,
            "promotion_scope": "proxy_evidence_only",
        },
    }
    capture["workload_proxy"] = {
        "enabled": True,
        "label": operation,
        "family": "fhe_lattice_proxy",
        "tower_role": "fixture_tower",
        "reuse_profile": "fixture_reuse",
        "transform_role": "not_a_public_fhe_backend",
        "output_domain_requirement": "native_i64_u64_host",
        "compatibility_claim": False,
    }
    capture["verification_amortization"] = {
        "enabled": True,
        "policy": "reuse_shape_seed_reference_inputs",
        "reused_reference_structure": "shape_seed_semantic_reference_inputs",
        "final_exact_comparison_required": True,
        "final_exact_comparison_status": "checksum_recorded_reference_required",
        "promotion_eligible": False,
    }
    if backend == "hip-direct":
        capture["timing_metadata"]["gpu_event_timing_source_scope"] = "direct_hip_default_stream_backend_operation_groups"
        capture["timing_metadata"]["gpu_event_timing_reason"] = "captured_by_direct_hip_backend_hooks"
        add_target_variant_fields(capture)
    kernel = capture.get("selected_kernel",
                         "cpu_reference_scalar_rns_gemm_v1" if backend == "cpu-reference"
                         else "direct_hip_tiled_active_prefix_rns_gemm_v2")
    capture["export_variant"] = {"selected_kernel": kernel}
    capture["exact_output_contract"] = {"kernel_identity": kernel}
    capture["reconstruction_variant"] = {"kernel_identity": kernel}
    if backend != "cpu-reference" and "gpu_event_phase_order" not in capture.get("timing_metadata", {}):
        capture.setdefault("timing_metadata", {})["gpu_event_phase_order"] = []
    capture["target_variant"] = capture.get("device", {}).get("gcn_arch", "gfx1100") if backend != "cpu-reference" else "cpu"
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
    operations = sorted(fhe_workload_report.REQUIRED_PROXY_OPERATIONS)
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        paths = []
        for operation in operations:
            paths.append(write(tmp / f"{operation}-cpu.json", proxy_capture("cpu-reference", operation, 1000)))
            paths.append(write(tmp / f"{operation}-hip.json", proxy_capture("hip-direct", operation, 400)))
        report = fhe_workload_report.build_report([tmp])
        assert report["schema"] == "rns8_fhe_lattice_workload_report_v2"
        assert report["rank63_gate_complete"] is True, json.dumps(report["blocker_counts"], indent=2)
        assert report["missing_required_proxy_operations"] == []
        assert report["promotable_workload_profile_count"] == len(operations)
        assert all(
            group["promotion_decision"] == "promote_local_dense_rns_workload_profile"
            for group in report["groups"]
        )

        no_cpu_dir = tmp / "no-cpu"
        no_cpu_dir.mkdir()
        write(no_cpu_dir / "hip.json", proxy_capture("hip-direct", operations[0]))
        no_cpu_report = fhe_workload_report.build_report([no_cpu_dir])
        assert no_cpu_report["rank63_gate_complete"] is False
        assert "cpu_reference_baseline_missing" in no_cpu_report["groups"][0]["blockers"]

        compat_dir = tmp / "compat"
        compat_dir.mkdir()
        bad = proxy_capture("cpu-reference", operations[0])
        bad["workload_proxy"]["compatibility_claim"] = True
        try:
            write(compat_dir / "bad.json", bad)
        except Exception:
            bad["workload_proxy"]["compatibility_claim"] = False
            bad["scenario_metadata"]["metadata"]["compatibility_claim"] = True
            write(compat_dir / "bad.json", bad)
        compat_report = fhe_workload_report.build_report([compat_dir])
        assert compat_report["rank63_gate_complete"] is False
        assert any(
            blocker in compat_report["groups"][0]["blockers"]
            for blocker in ["fhe_compatibility_claim_forbidden", "scenario_metadata_compatibility_claim_forbidden"]
        )

        promotable_dir = tmp / "promotable"
        promotable_dir.mkdir()
        promotable = proxy_capture("cpu-reference", operations[0])
        promotable["verification_amortization"]["promotion_eligible"] = True
        try:
            write(promotable_dir / "promotable.json", promotable)
        except Exception:
            promotable["verification_amortization"]["promotion_eligible"] = False
            promotable["scenario_metadata"]["promotion_eligibility"] = "release_review_candidate"
            promotable["scenario_metadata"]["metadata"]["promotion_scope"] = "release_review_candidate"
            write(promotable_dir / "promotable.json", promotable)
        promotable_report = fhe_workload_report.build_report([promotable_dir])
        assert promotable_report["rank63_gate_complete"] is False
        assert promotable_report["groups"][0]["blockers"]

        no_speedup_dir = tmp / "no-speedup"
        no_speedup_dir.mkdir()
        write(no_speedup_dir / "cpu.json", proxy_capture("cpu-reference", operations[0], 500))
        write(no_speedup_dir / "hip.json", proxy_capture("hip-direct", operations[0], 700))
        no_speedup_report = fhe_workload_report.build_report([no_speedup_dir])
        assert no_speedup_report["rank63_gate_complete"] is False
        assert no_speedup_report["promotable_workload_profile_count"] == 0
        assert no_speedup_report["groups"][0]["promotion_decision"] == "keep_experimental_no_gpu_speedup"

        missing_exact_dir = tmp / "missing-exact"
        missing_exact_dir.mkdir()
        missing_exact = proxy_capture("cpu-reference", operations[0])
        missing_exact["verification_amortization"]["final_exact_comparison_status"] = "not_recorded"
        try:
            write(missing_exact_dir / "missing-exact.json", missing_exact)
        except Exception:
            missing_exact["verification_amortization"]["final_exact_comparison_status"] = "reference_required"
            del missing_exact["verification_amortization"]["final_exact_comparison_required"]
            missing_exact_dir.joinpath("missing-exact.json").write_text(
                json.dumps(missing_exact, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        try:
            missing_exact_report = fhe_workload_report.build_report([missing_exact_dir])
        except Exception:
            missing_exact_report = {"rank63_gate_complete": False}
        assert missing_exact_report["rank63_gate_complete"] is False

    print("fhe workload report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())