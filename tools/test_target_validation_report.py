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
        assert report["status_record_count"] == 2
        groups = {group["target_validation_group"]: group for group in report["groups"]}
        cdna = groups["os=linux;target=gfx942;toolchain=7.1"]
        assert cdna["target_class"] == "cdna"
        assert cdna["validation_phases"]["profiler"] == "pass"
        assert cdna["cache_eligibility"]["eligible"] is True
        assert cdna["cross_target_promotion_allowed"] is False
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
