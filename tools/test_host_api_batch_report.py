#!/usr/bin/env python3
"""Self-test host API batch comparison reporting."""

from __future__ import annotations

import copy
from pathlib import Path

import host_api_batch_report
from benchmark_schema import load_capture, validate_capture
from test_benchmark_schema import as_host_api_batch_capture


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def with_path(capture: dict, path: str) -> dict:
    capture = copy.deepcopy(capture)
    capture["_path"] = path
    return capture


def scaled_batch(capture: dict, scale: float) -> dict:
    batch = copy.deepcopy(capture)
    for phase in host_api_batch_report.PHASES:
        summary = batch.get("timing_summary_us", {}).get(phase)
        if isinstance(summary, dict):
            for key in ["avg", "median", "p95"]:
                if isinstance(summary.get(key), (int, float)):
                    summary[key] *= scale
    return batch


def mark_release_counts(capture: dict) -> None:
    capture["warmups"] = 3
    capture["repeats"] = 9


def main() -> int:
    baseline = load_capture(FIXTURE_DIR / "v4_finite_ring_u8_ck.json")
    validate_capture(baseline)
    mark_release_counts(baseline)
    baseline = with_path(baseline, "independent-ck.json")

    winning_batch = as_host_api_batch_capture(baseline)
    winning_batch["host_api_batch"]["total_measured_tasks"] = (
        winning_batch["host_api_batch"]["batch_size"] * winning_batch["repeats"]
    )
    winning_batch = with_path(winning_batch, "host-batch-ck-win.json")

    report = host_api_batch_report.compare_host_batches([baseline, winning_batch])
    assert report["summary"]["comparisons"] == 1
    assert report["summary"]["candidate_wins"] == 1
    item = report["comparisons"][0]
    assert item["decision"] == "candidate_win"
    assert item["host_api_batch_size"] == 4
    assert item["phases"]["end_to_end"]["per_task_speedup_vs_independent"] > 1.0

    losing_batch = scaled_batch(winning_batch, 8.0)
    losing_batch["_path"] = "host-batch-ck-loss.json"
    loss_report = host_api_batch_report.compare_host_batches([baseline, losing_batch])
    assert loss_report["summary"]["deprioritized"] == 1
    assert loss_report["comparisons"][0]["decision"] == "deprioritize"
    assert "host_batch_not_faster_than_same_backend_per_task" in loss_report["comparisons"][0]["blockers"]

    missing_baseline = copy.deepcopy(winning_batch)
    missing_baseline["backend_selected"] = "hipblaslt"
    missing_baseline["backend_requested"] = "hipblaslt"
    missing_baseline["_path"] = "host-batch-hipblaslt-missing.json"
    missing_report = host_api_batch_report.compare_host_batches([baseline, missing_baseline])
    assert missing_report["summary"]["missing_baselines"] == 1
    assert missing_report["comparisons"][0]["decision"] == "missing_baseline"

    print("host API batch report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
