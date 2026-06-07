#!/usr/bin/env python3
"""Self-test independent multi-GPU shard report aggregation."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import multigpu_shard_report


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "benchmark_schema" / "v4_bounded_i64_hipblaslt.json"


def _base_capture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _write_capture(path: Path, rank: int, checksum: int = 987654321) -> None:
    capture = copy.deepcopy(_base_capture())
    capture["checksum_u64"] = checksum
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")
    (path.parent / f"benchmark-schema-rank{rank}.log").write_text("schema validation: PASS\n", encoding="utf-8")


def _env_summary(path: Path) -> None:
    physical_devices = []
    for device in range(8):
        physical_devices.append(
            {
                "physical_device_id": device,
                "target_arch": "gfx942",
                "device_name": f"MI300X-{device}",
                "bdf": f"0000:{device + 1:02x}:00.0",
                "numa_node": device % 2,
                "visible": device in {4, 5, 6, 7},
                "visibility_index": device - 4 if device >= 4 else None,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "requested_devices": ["4", "5", "6", "7"],
                "visible_gpu_count": 4,
                "node_gpu_count": 8,
                "physical_devices": physical_devices,
                "rocprofv3_ready": True,
                "rccl_ready": False,
                "rccl_tests_ready": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _target_status(path: Path, devices: list[int]) -> None:
    records = []
    for rank, device in enumerate(devices):
        records.append(
            {
                "host_os": "linux",
                "target_id": "gfx942",
                "rocm_version": "7.1",
                "gpu_name": f"MI300X-{device}",
                "device_index": device,
                "physical_device_id": device,
                "device_bdf": f"0000:{device + 1:02x}:00.0",
                "numa_node": device % 2,
                "multi_gpu_mode": "embarrassingly_parallel_shards",
                "rank": rank,
                "world_size": len(devices),
                "visible_gpu_count": 1,
                "node_gpu_count": 8,
                "rocprofv3_ready": True,
                "rccl_ready": False,
                "rccl_tests_ready": False,
                "validation": {"smoke": "pass", "profiler": "pass"},
                "cache_eligibility": {"eligible": False, "blockers": ["smoke_only"]},
            }
        )
    path.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        env_path = tmp / "env" / "cdna-env-summary.json"
        env_path.parent.mkdir(parents=True)
        _env_summary(env_path)
        status_path = tmp / "target-status.json"
        _target_status(status_path, [4, 5, 6, 7])
        for rank, device in enumerate([4, 5, 6, 7]):
            _write_capture(
                tmp / "shards" / f"gpu{device}" / f"bounded-i64-hip-direct-smoke-rank{rank}.json",
                rank,
            )
        report = multigpu_shard_report.build_report(
            [],
            env_summary=env_path,
            target_status=[status_path],
            shards_dir=tmp / "shards",
        )
        assert report["schema"] == "rns8_multigpu_shard_report_v1"
        assert report["env_summary_status"] == "present"
        assert report["capture_count"] == 4
        assert report["missing_ranks"] == []
        assert report["failed_ranks"] == []
        rows = {row["rank"]: row for row in report["rows"]}
        assert rows[0]["physical_device_id"] == 4
        assert rows[0]["device_bdf"] == "0000:05:00.0"
        assert rows[3]["physical_device_id"] == 7
        assert rows[3]["device_bdf"] == "0000:08:00.0"
        assert multigpu_shard_report.report_has_critical_failures(report) is False
        outputs = multigpu_shard_report.write_outputs(report, tmp / "report")
        assert Path(outputs["json"]).exists()
        assert Path(outputs["markdown"]).exists()

        failed = tmp / "shards" / "gpu5" / "bounded-i64-hip-direct-smoke-rank1.json"
        failed.write_text('{"not": "schema valid"}\n', encoding="utf-8")
        (failed.parent / "benchmark-schema-rank1.log").write_text("ERROR schema validation failed\n", encoding="utf-8")
        missing = tmp / "shards" / "gpu6" / "bounded-i64-hip-direct-smoke-rank2.json"
        missing.unlink()
        failed_report = multigpu_shard_report.build_report(
            [],
            env_summary=env_path,
            target_status=[status_path],
            shards_dir=tmp / "shards",
        )
        assert failed_report["missing_ranks"] == [2]
        assert failed_report["failed_ranks"] == [1]
        assert "failed_shards" in failed_report["promotion_blockers"]
        assert "missing_shards" in failed_report["promotion_blockers"]
        assert multigpu_shard_report.report_has_critical_failures(failed_report) is True

        missing_env_report = multigpu_shard_report.build_report(
            [],
            env_summary=tmp / "env" / "missing-cdna-env-summary.json",
            target_status=[status_path],
            shards_dir=tmp / "shards",
        )
        assert missing_env_report["env_summary_status"] == "missing"
        assert "env_summary_missing" in missing_env_report["promotion_blockers"]
        assert multigpu_shard_report.report_has_critical_failures(missing_env_report) is True
        missing_env_rows = {row["rank"]: row for row in missing_env_report["rows"]}
        assert missing_env_rows[0]["physical_device_id"] == 4
        assert missing_env_rows[0]["device_bdf"] == "0000:05:00.0"

    print("multi-GPU shard report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
