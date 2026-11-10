#!/usr/bin/env python3
"""Self-test host API batch comparison reporting."""

from __future__ import annotations

import copy
import tempfile
from itertools import permutations
from pathlib import Path

import host_api_batch_report
from benchmark_schema import load_capture, validate_capture
from diagnostic_companion import build_binding
from test_benchmark_sweep_support import finite_capture
from test_diagnostic_companion import authored_capture_pair
from test_review_reference_support import add_synthetic_full_reference
from test_benchmark_schema_support_core import (
    as_host_api_batch_capture,
)


def with_path(capture: dict, path: str) -> dict:
    capture = copy.deepcopy(capture)
    capture["_path"] = path
    return capture


def main() -> int:
    baseline = finite_capture("ck", 400)
    validate_capture(baseline)
    baseline = with_path(baseline, "independent-ck.json")

    winning_batch = as_host_api_batch_capture(baseline)
    winning_batch["host_api_batch"]["total_measured_tasks"] = (
        winning_batch["host_api_batch"]["batch_size"] * winning_batch["repeats"]
    )
    winning_batch = with_path(winning_batch, "host-batch-ck-win.json")

    report = host_api_batch_report.compare_host_batches([baseline, winning_batch])
    assert report["summary"]["comparisons"] == 1
    assert report["summary"]["candidate_wins"] == 0
    item = report["comparisons"][0]
    assert item["decision"] in {"missing_baseline", "keep_experimental"}
    assert item["host_api_batch_size"] == 4
    assert item["phases"]["end_to_end"]["per_task_speedup_vs_independent"] is None
    assert not item["release_review_pair"]

    losing_batch = as_host_api_batch_capture(finite_capture("ck", 3200))
    losing_batch["_path"] = "host-batch-ck-loss.json"
    loss_report = host_api_batch_report.compare_host_batches([baseline, losing_batch])
    assert loss_report["summary"]["deprioritized"] == 0
    assert loss_report["comparisons"][0]["speedup_vs_best_independent"] is None

    missing_baseline = copy.deepcopy(winning_batch)
    missing_baseline["backend_selected"] = "hipblaslt"
    missing_baseline["backend_requested"] = "hipblaslt"
    missing_baseline["_path"] = "host-batch-hipblaslt-missing.json"
    missing_report = host_api_batch_report.compare_host_batches([baseline, missing_baseline])
    assert missing_report["summary"]["missing_baselines"] == 1
    assert missing_report["comparisons"][0]["decision"] == "missing_baseline"

    production_controls()
    print("host API batch report self-test: PASS")
    return 0


def production_controls():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        captures, bindings = [], {}
        for backend in ("cpu-reference", "ck"):
            for batch in (False, True):
                for run in range(2):
                    lane = root / f"{backend}-{batch}-{run}"
                    lane.mkdir()
                    fixture = finite_capture(backend, 1000 if batch else 400)
                    fixture.update(
                        benchmark_execution_mode="persistent_resident",
                        pack_mode="per_repeat_repack",
                        reuse_packed_inputs=False,
                        prepack_reuse_operands=[],
                        prepack_reuse_strategy="none",
                    )
                    if batch:
                        fixture = as_host_api_batch_capture(fixture)
                    fixture = add_synthetic_full_reference(fixture)
                    _, _, paths = authored_capture_pair(lane, fixture)
                    capture = load_capture(paths[0])
                    validate_capture(capture, paths[0])
                    captures.append(capture)
                    if backend == "ck":
                        bindings[str(paths[0].resolve())] = build_binding(*paths)

        def build(items, bindings=bindings):
            return host_api_batch_report.compare_host_batches(items, diagnostic_bindings=bindings)

        def gpu_row(value):
            return next(row for row in value["comparisons"] if row["backend"] == "ck")

        expected = gpu_row(build(captures))
        assert expected["decision"] == "candidate_win", expected["blockers"]
        assert expected["release_review_pair"]
        assert expected["measurement_cohort"]["independent_runs"] == 2
        assert expected["baseline_measurement_cohort"]["independent_runs"] == 2
        assert expected["normalization"]["tasks"] == 4
        assert expected["phases"]["end_to_end"]["batch_per_task_median_us"] == 250
        assert expected["speedup_vs_best_independent"] == 1.6
        assert expected["numerical_reference"]["passed"]
        blocks = [captures[offset : offset + 2] for offset in range(0, len(captures), 2)]
        for ordering in permutations(blocks):
            assert (
                gpu_row(build([item for block in ordering for item in reversed(block)])) == expected
            )
        single = gpu_row(build([captures[0], captures[2], captures[4], captures[6]]))
        assert single["speedup_vs_best_independent"] is None
        assert not single["release_review_pair"]
        duplicate = gpu_row(build(captures[:6] + [captures[6], copy.deepcopy(captures[6])]))
        assert duplicate["measurement_cohort"]["independent_runs"] == 1
        assert duplicate["speedup_vs_best_independent"] is None
        for field in ("device", "seed", "output_policy"):
            changed = copy.deepcopy(captures)
            for candidate in changed[6:]:
                if field == "device":
                    candidate["device"]["name"] += " different synthetic GPU"
                elif field == "seed":
                    candidate["seed"] += 1
                else:
                    candidate["output_policy"] = {
                        **(candidate.get("output_policy") or {}),
                        "completion_boundary": "compact_task_major_host_slab",
                    }
            row = gpu_row(build(changed))
            assert row["speedup_vs_best_independent"] is None
            assert not row["release_review_pair"]
            assert row["same_backend_independent_capture"] is None
        missing = dict(bindings)
        missing.pop(str(Path(captures[7]["_path"]).resolve()))
        row = gpu_row(build(captures, missing))
        assert "matched_diagnostic_companion_missing" in row["blockers"]
        assert row["speedup_vs_best_independent"] is None


if __name__ == "__main__":
    raise SystemExit(main())
