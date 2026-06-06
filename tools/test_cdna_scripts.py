#!/usr/bin/env python3
"""Dry-run regression tests for CDNA readiness shell scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/cdna_common.sh",
    "scripts/cdna_env_probe.sh",
    "scripts/cdna_smoke.sh",
    "scripts/cdna_first_pass.sh",
    "scripts/cdna_multigpu_smoke.sh",
]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _require_ok(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise AssertionError(f"{label} failed with {result.returncode}\n{result.stdout}")


def main() -> int:
    if os.name == "nt":
        print("CDNA script self-test: SKIP (Linux Bash test)")
        return 0
    if shutil.which("bash") is None:
        print("CDNA script self-test: SKIP (bash unavailable)")
        return 0
    probe = _run(["bash", "--version"])
    if probe.returncode != 0:
        print("CDNA script self-test: SKIP (bash unusable)")
        return 0

    _require_ok(_run(["bash", "-n", *SCRIPTS]), "bash syntax")

    out_root = Path("temp") / "cdna-script-regression"
    first_pass = out_root / "first-pass"
    multi4 = out_root / "multigpu-4"
    multi8 = out_root / "multigpu-8"
    partial = out_root / "multigpu-4-5-6-7"

    _require_ok(
        _run(["bash", "scripts/cdna_first_pass.sh", "--dry-run", "--devices", "0", "--out-dir", str(first_pass)]),
        "first-pass dry-run",
    )
    first_status = json.loads((REPO_ROOT / first_pass / "target-status.json").read_text(encoding="utf-8"))
    assert first_status["records"][0]["device_index"] == 0
    assert (REPO_ROOT / first_pass / "env" / "cdna-env-summary.json").exists()

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
