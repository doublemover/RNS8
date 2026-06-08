#!/usr/bin/env python3
"""Dry-run regression tests for CDNA readiness shell scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cdna_env_summary
import multigpu_shard_report


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/cdna_common.sh",
    "scripts/cdna_env_probe.sh",
    "scripts/cdna_smoke.sh",
    "scripts/cdna_first_pass.sh",
    "scripts/cdna_accelerators.sh",
    "scripts/cdna_multigpu_smoke.sh",
]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _require_ok(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise AssertionError(f"{label} failed with {result.returncode}\n{result.stdout}")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_portable_artifact_tests() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        missing_log_dir = Path(tmp_name) / "missing-env-logs"
        summary = cdna_env_summary.build_summary(
            missing_log_dir,
            devices_option="0,1,2,3",
            dry_run=True,
            environment={
                "HIP_VISIBLE_DEVICES": None,
                "ROCR_VISIBLE_DEVICES": None,
                "GPU_DEVICE_ORDINAL": None,
                "ROCM_PATH": "/opt/rocm",
                "HIP_PATH": "/opt/rocm",
                "LD_LIBRARY_PATH": "/opt/rocm/lib",
            },
        )
        physical = {item["physical_device_id"]: item for item in summary["physical_devices"]}
        assert summary["dry_run"] is True
        assert summary["visible_gpu_count"] == 4
        assert summary["node_gpu_count"] == 4
        assert summary["raw_logs"] == []
        assert physical[0]["topology_source"] == "heuristic_index_order"
        assert physical[3]["visible"] is True
        assert physical[3]["visibility_index"] == 3

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        status_path = tmp / "target-status.json"
        records = []
        for rank, device in enumerate([4, 5, 6, 7]):
            records.append(
                {
                    "device_index": device,
                    "device_bdf": f"0000:{device + 1:02x}:00.0",
                    "gpu_name": f"MI300X-{device}",
                    "multi_gpu_mode": "embarrassingly_parallel_shards",
                    "rank": rank,
                    "target_id": "gfx942",
                    "world_size": 4,
                }
            )
            capture = tmp / "shards" / f"gpu{device}" / f"bounded-i64-hip-direct-smoke-rank{rank}.json"
            _write_json(capture, {"dry_run": True, "rank": rank, "world_size": 4, "device": str(device)})
            (capture.parent / f"benchmark-schema-rank{rank}.log").write_text(
                "dry-run schema validation\n",
                encoding="utf-8",
            )
        _write_json(status_path, {"records": records})
        report = multigpu_shard_report.build_report(
            [],
            env_summary=tmp / "env" / "missing-cdna-env-summary.json",
            target_status=[status_path],
            shards_dir=tmp / "shards",
        )
        assert report["env_summary_status"] == "missing"
        assert report["capture_count"] == 4
        assert report["missing_ranks"] == []
        assert "env_summary_missing" in report["promotion_blockers"]
        assert multigpu_shard_report.report_has_critical_failures(report) is True
        rows = {row["rank"]: row for row in report["rows"]}
        assert rows[0]["physical_device_id"] == 4
        assert rows[3]["device_bdf"] == "0000:08:00.0"
        assert all("target_status_missing" not in row["blockers"] for row in report["rows"])


def main() -> int:
    _run_portable_artifact_tests()
    if os.name == "nt":
        print("CDNA script Bash dry-run self-test: SKIP (Linux Bash test)")
        print("CDNA script portable artifact self-test: PASS")
        return 0
    if shutil.which("bash") is None:
        print("CDNA script Bash dry-run self-test: SKIP (bash unavailable)")
        print("CDNA script portable artifact self-test: PASS")
        return 0
    probe = _run(["bash", "--version"])
    if probe.returncode != 0:
        print("CDNA script Bash dry-run self-test: SKIP (bash unusable)")
        print("CDNA script portable artifact self-test: PASS")
        return 0

    _require_ok(_run(["bash", "-n", *SCRIPTS]), "bash syntax")

    out_root = Path("temp") / "cdna-script-regression"
    first_pass = out_root / "first-pass"
    accelerators = out_root / "accelerators"
    multi4 = out_root / "multigpu-4"
    multi8 = out_root / "multigpu-8"
    partial = out_root / "multigpu-4-5-6-7"

    _require_ok(
        _run(
            [
                "bash",
                "scripts/cdna_first_pass.sh",
                "--dry-run",
                "--devices",
                "0",
                "--out-dir",
                str(first_pass),
                "--rank-scenarios",
                "vector-to-rns-chain",
            ]
        ),
        "first-pass dry-run",
    )
    first_status = json.loads((REPO_ROOT / first_pass / "target-status.json").read_text(encoding="utf-8"))
    assert first_status["records"][0]["device_index"] == 0
    assert first_status["records"][0]["validation"]["dependency_check"] == "planned"
    assert first_status["records"][0]["dependency_check"]["status"] == "planned"
    assert "required_gate_failures" in first_status["records"][0]["dependency_check"]
    assert (REPO_ROOT / first_pass / "env" / "cdna-env-summary.json").exists()
    first_plan = (REPO_ROOT / first_pass / "command-plan.txt").read_text(encoding="utf-8")
    assert "rank_scenario_vector-to-rns-chain_lint" in first_plan
    assert "--lint-scenarios" in first_plan
    assert "--scenario vector-to-rns-chain" in first_plan
    assert (REPO_ROOT / first_pass / "rank-scenarios" / "vector-to-rns-chain" / "rank-scenario-lint-plan.log").exists()
    assert (REPO_ROOT / first_pass / "rank-scenarios" / "vector-to-rns-chain" / "rank-scenario-plan.log").exists()

    _require_ok(
        _run(
            [
                "bash",
                "scripts/cdna_accelerators.sh",
                "--dry-run",
                "--devices",
                "0",
                "--out-dir",
                str(accelerators),
            ]
        ),
        "accelerators dry-run",
    )
    accelerator_status = json.loads((REPO_ROOT / accelerators / "target-status.json").read_text(encoding="utf-8"))
    assert accelerator_status["records"][0]["preset"] == "linux-cdna-accelerators-release"
    accelerator_plan = (REPO_ROOT / accelerators / "command-plan.txt").read_text(encoding="utf-8")
    assert "linux-cdna-accelerators-release" in accelerator_plan
    assert "--accelerators" in accelerator_plan
    assert (REPO_ROOT / accelerators / "benchmark-schema.log").exists()
    assert (REPO_ROOT / accelerators / "target-validation" / "target-validation-report.md").exists()

    skip_rank = root / "first-pass-skip-rank"
    _require_ok(
        _run(
            [
                "bash",
                "scripts/cdna_first_pass.sh",
                "--dry-run",
                "--devices",
                "0",
                "--out-dir",
                str(skip_rank),
                "--rank-scenarios",
                "vector-to-rns-chain",
                "--skip-rank-scenarios",
            ]
        ),
        "first-pass dry-run skip rank scenarios",
    )
    skip_rank_plan = (REPO_ROOT / skip_rank / "command-plan.txt").read_text(encoding="utf-8")
    assert "rank_scenario_" not in skip_rank_plan
    assert not (REPO_ROOT / skip_rank / "rank-scenarios").exists()

    for devices, out_dir, expected_world in [
        ("0,1,2,3", multi4, 4),
        ("0,1,2,3,4,5,6,7", multi8, 8),
        ("4,5,6,7", partial, 4),
    ]:
        _require_ok(
            _run(
                [
                    "bash",
                    "scripts/cdna_multigpu_smoke.sh",
                    "--dry-run",
                    "--devices",
                    devices,
                    "--out-dir",
                    str(out_dir),
                ]
            ),
            f"multi-GPU dry-run {devices}",
        )
        status = json.loads((REPO_ROOT / out_dir / "target-status.json").read_text(encoding="utf-8"))
        records = status["records"]
        assert len(records) == expected_world
        assert all(record["world_size"] == expected_world for record in records)
        report = json.loads(
            (REPO_ROOT / out_dir / "multigpu-shard-report" / "multigpu-shard-report.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["capture_count"] == expected_world
        assert report["missing_ranks"] == []
        assert "multi_gpu_smoke_not_release_reviewed" in report["promotion_blockers"]

    partial_status = json.loads((REPO_ROOT / partial / "target-status.json").read_text(encoding="utf-8"))
    partial_records = partial_status["records"]
    assert [record["rank"] for record in partial_records] == [0, 1, 2, 3]
    assert [record["device_index"] for record in partial_records] == [4, 5, 6, 7]
    plan = (REPO_ROOT / partial / "command-plan.txt").read_text(encoding="utf-8")
    for device in ["4", "5", "6", "7"]:
        assert f"ROCR_VISIBLE_DEVICES={device}" in plan
        assert f"HIP_VISIBLE_DEVICES={device}" in plan
    summary = json.loads((REPO_ROOT / partial / "env" / "cdna-env-summary.json").read_text(encoding="utf-8"))
    physical = {item["physical_device_id"]: item for item in summary["physical_devices"]}
    assert set([4, 5, 6, 7]).issubset(set(physical))
    assert physical[4]["visible"] is True
    assert physical[4]["visibility_index"] == 0

    print("CDNA script dry-run self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
