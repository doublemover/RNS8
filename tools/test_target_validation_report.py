#!/usr/bin/env python3
"""Self-test target validation grouping and CDNA readiness fixtures."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import target_validation_report


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def main() -> int:
    capture = FIXTURE_DIR / "v4_bounded_i64_hipblaslt.json"
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        status_path = tmp / "targets.json"
        status_path.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "host_os": "linux",
                            "target_id": "gfx942",
                            "rocm_version": "7.1",
                            "gpu_name": "AMD Instinct MI300X",
                            "device_index": 0,
                            "visible_device_count": 1,
                            "node_gpu_count": 8,
                            "multi_gpu_mode": "single_device_smoke",
                            "rank": 0,
                            "world_size": 1,
                            "device_bdf": "0000:01:00.0",
                            "numa_node": 0,
                            "xgmi_peers": [1, 2, 3],
                            "rocprofv3_ready": True,
                            "rccl_ready": False,
                            "rccl_tests_ready": False,
                            "configured_amdgpu_targets": "gfx90a;gfx942;gfx950",
                            "accelerators": {"hipblaslt": "available", "ck": "available", "rocwmma": "available"},
                            "evidence": {
                                "build": "pass",
                                "ctest": "pass",
                                "smoke": "pass",
                                "release_capture": "pass",
                                "profiler": "pass",
                            },
                            "cache_eligibility": {"eligible": True, "blockers": []},
                        },
                        *[
                            {
                                "host_os": "linux",
                                "target_id": "gfx942",
                                "rocm_version": "7.1",
                                "gpu_name": "AMD Instinct MI300X",
                                "device_index": rank,
                                "visible_device_count": 1,
                                "node_gpu_count": 4,
                                "multi_gpu_mode": "embarrassingly_parallel_shards",
                                "rank": rank,
                                "world_size": 4,
                                "device_bdf": f"0000:0{rank + 1}:00.0",
                                "numa_node": rank % 2,
                                "rocprofv3_ready": True,
                                "rccl_ready": False,
                                "rccl_tests_ready": False,
                                "evidence": {"smoke": "pass"},
                                "cache_eligibility": {"eligible": False, "blockers": ["smoke_only"]},
                            }
                            for rank in range(4)
                        ],
                        *[
                            {
                                "host_os": "linux",
                                "target_id": "gfx950",
                                "rocm_version": "7.1",
                                "gpu_name": "AMD Instinct MI350X",
                                "device_index": rank,
                                "visible_device_count": 1,
                                "node_gpu_count": 8,
                                "multi_gpu_mode": "embarrassingly_parallel_shards",
                                "rank": rank,
                                "world_size": 8,
                                "rocprofv3_ready": rank != 7,
                                "rccl_ready": True,
                                "rccl_tests_ready": rank != 6,
                                "evidence": {
                                    "build": "pass",
                                    "ctest": "pass",
                                    "smoke": "pass",
                                    "release_capture": "missing",
                                    "profiler": "missing" if rank == 7 else "pass",
                                },
                                "cache_eligibility": {"eligible": False, "blockers": ["release_capture_missing"]},
                            }
                            for rank in range(8)
                        ],
                        {
                            "host_os": "linux",
                            "target_id": "gfx1200",
                            "rocm_version": "7.1",
                            "gpu_name": "Radeon RX 9070 XT",
                            "evidence": {"build": "pass", "ctest": "missing"},
                            "cache_eligibility": {"eligible": False, "blockers": ["release_capture_missing"]},
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        report = target_validation_report.build_report([capture], [status_path])
        assert report["schema_version"] == 2
        assert report["capture_count"] == 1
        assert report["status_record_count"] == 14
        groups = {group["target_validation_group"]: group for group in report["groups"]}
        cdna = groups["os=linux;target=gfx942;toolchain=7.1"]
        assert cdna["target_class"] == "cdna"
        assert cdna["validation_phases"]["profiler"] == "pass"
        assert cdna["cache_eligibility"]["eligible"] is True
        assert cdna["cross_target_promotion_allowed"] is False
        assert cdna["multi_gpu_modes"] == ["embarrassingly_parallel_shards", "single_device_smoke"]
        assert cdna["world_sizes"] == ["1", "4"]
        assert cdna["ranks"] == ["0", "1", "2", "3"]
        assert cdna["device_bdfs"] == [
            "0000:01:00.0",
            "0000:02:00.0",
            "0000:03:00.0",
            "0000:04:00.0",
        ]
        assert cdna["numa_nodes"] == ["0", "1"]
        assert cdna["rocprofv3_ready_values"] == ["true"]
        assert cdna["rccl_ready_values"] == ["false"]
        assert cdna["rccl_tests_ready_values"] == ["false"]
        cdna8 = groups["os=linux;target=gfx950;toolchain=7.1"]
        assert cdna8["world_sizes"] == ["8"]
        assert cdna8["ranks"] == [str(rank) for rank in range(8)]
        assert cdna8["cache_eligibility"]["eligible"] is False
        assert "release_capture_not_ready" in cdna8["cache_eligibility"]["blockers"]
        assert cdna8["rocprofv3_ready_values"] == ["false", "true"]
        assert cdna8["rccl_ready_values"] == ["true"]
        assert cdna8["rccl_tests_ready_values"] == ["false", "true"]
        rdna = groups["os=linux;target=gfx1200;toolchain=7.1"]
        assert rdna["target_class"] == "rdna"
        assert rdna["cache_eligibility"]["eligible"] is False
        assert "ctest_not_ready" in rdna["cache_eligibility"]["blockers"]
        windows = groups["os=windows;target=gfx1100;toolchain=7.1"]
        assert windows["cache_eligibility"]["eligible"] is False
        assert "build_not_ready" in windows["cache_eligibility"]["blockers"]
        outputs = target_validation_report.write_outputs(report, tmp / "out")
        assert Path(outputs["json"]).exists()
        assert Path(outputs["markdown"]).exists()

    print("target validation report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
