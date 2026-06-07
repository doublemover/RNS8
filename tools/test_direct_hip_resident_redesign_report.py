#!/usr/bin/env python3
"""Self-test Direct-HIP resident redesign reporting."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import direct_hip_resident_redesign_report as report


def capture(
    *,
    kernel: str,
    median_us: float,
    rns_gemm_us: float,
    crt_export_us: float = 300.0,
    semantics: str = "bounded_i64",
    path: str = "capture.json",
    checksum: int = 1234,
    redesign: dict | None = None,
    schema_status: str = "valid",
    events: bool = True,
) -> dict:
    return {
        "_path": path,
        "_schema_status": schema_status,
        "_schema_errors": ["schema error"] if schema_status != "valid" else [],
        "schema_version": 4,
        "benchmark": "rns8_bounded_gemm_persistent_rns",
        "benchmark_execution_mode": "persistent_resident_matrices",
        "backend_selected": "hip-direct",
        "backend_requested": "hip-direct",
        "selected_kernel": kernel,
        "backend_metadata": {"selected_kernel": kernel},
        "semantics": semantics,
        "m": 512,
        "n": 512,
        "k": 512,
        "prefix": 9,
        "selected_prefix": 9,
        "requested_max_prefix": 9,
        "contract_prefix_policy": "fixed_requested",
        "input_distribution": "signed_uniform_-16_16",
        "seed": 20260606,
        "warmups": 3,
        "repeats": 9,
        "checksum_u64": checksum,
        "timing_summary_us": {
            "end_to_end": {"median": median_us},
            "rns_gemm": {"median": rns_gemm_us},
            "crt_export": {"median": crt_export_us},
        },
        "timing_metadata": {
            "gpu_event_timing": events,
            "gpu_event_timing_status": "available" if events else "not_available",
            "gpu_event_phase_order": ["pack", "rns_gemm_kernel_group", "crt_export"] if events else None,
            "gpu_event_timing_source": "hipEventElapsedTime" if events else None,
        },
        "gpu_event_timings_us": {"rns_gemm_kernel_group": [rns_gemm_us] * 9} if events else None,
        "resident_redesign": redesign or {},
    }


def redesign_metadata(candidate: str = "layout_tile_export_prototype") -> dict:
    return {
        "candidate": candidate,
        "dimensions": [
            "data_layout",
            "tile_shape",
            "export_interaction",
            "schedule_upload",
            "workspace_reuse",
        ],
        "resource_summary": {
            "vgpr": 48,
            "sgpr": 64,
            "lds_bytes": 4096,
            "occupancy": 0.75,
            "bottleneck_class": "memory_pressure",
        },
    }


def main() -> int:
    baseline = capture(
        kernel=report.DEFAULT_KERNEL,
        median_us=2000.0,
        rns_gemm_us=1000.0,
        path="baseline.json",
    )
    colpair_loss = capture(
        kernel=report.REJECTED_COLPAIR_KERNEL,
        median_us=2500.0,
        rns_gemm_us=800.0,
        path="colpair-loss.json",
        redesign=redesign_metadata("selected_prefix_colpair_rejected_route"),
    )
    route_candidate = capture(
        kernel=report.GROUPED_ACTIVE_SCHEDULE_KERNEL,
        median_us=1800.0,
        rns_gemm_us=900.0,
        path="layout-prototype.json",
        redesign=redesign_metadata(),
    )
    result = report.build_direct_hip_resident_redesign_report([baseline, colpair_loss, route_candidate])
    summary = result["summary"]
    assert summary["candidate_count"] == 2
    assert summary["resident_default_baseline_count"] == 1
    assert summary["rank51_gate_complete"] is True
    assert summary["route_candidate_count"] == 1
    assert summary["deprioritized_count"] == 1
    assert summary["gemm_only_win_blocked_count"] == 1

    colpair_row = next(item for item in result["comparisons"] if item["candidate_kernel"] == report.REJECTED_COLPAIR_KERNEL)
    assert colpair_row["decision"] == "drop/deprioritize"
    assert colpair_row["speedup_vs_resident_default_rns_gemm"] == 1.25
    assert "gemm_only_win_end_to_end_loss" in colpair_row["blockers"]
    assert "end_to_end_not_faster" in colpair_row["blockers"]

    route_row = next(item for item in result["comparisons"] if item["candidate"] == "layout_tile_export_prototype")
    assert route_row["decision"] == "route_candidate"
    assert route_row["promotion_eligible"] is True
    assert route_row["blockers"] == []

    missing_baseline = report.build_direct_hip_resident_redesign_report([route_candidate])
    assert missing_baseline["summary"]["rank51_gate_complete"] is False
    assert "missing_resident_default_baseline" in missing_baseline["comparisons"][0]["blockers"]

    missing_resource = copy.deepcopy(route_candidate)
    missing_resource["resident_redesign"].pop("resource_summary")
    missing_resource_report = report.build_direct_hip_resident_redesign_report([baseline, missing_resource])
    assert missing_resource_report["summary"]["rank51_gate_complete"] is False
    assert "missing_resource_explanation" in missing_resource_report["comparisons"][0]["blockers"]

    mismatch = copy.deepcopy(route_candidate)
    mismatch["checksum_u64"] = 999
    mismatch_report = report.build_direct_hip_resident_redesign_report([baseline, mismatch])
    assert mismatch_report["summary"]["rank51_gate_complete"] is False
    assert "checksum_mismatch_with_resident_default" in mismatch_report["comparisons"][0]["blockers"]

    with tempfile.TemporaryDirectory() as tmp_name:
        outputs = report.write_outputs(result, Path(tmp_name))
        assert Path(outputs["json"]).exists()
        assert Path(outputs["markdown"]).exists()

    print("direct-HIP resident redesign report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
