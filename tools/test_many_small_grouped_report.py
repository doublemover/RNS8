#!/usr/bin/env python3
"""Self-test many-small grouped-dispatch comparison reporting."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import many_small_grouped_report
from benchmark_schema import load_capture
from metadata_registry_constants import GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS
from test_benchmark_schema import as_grouped_dispatch_capture, as_host_api_batch_capture


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def with_path(capture: dict, path: str) -> dict:
    capture = copy.deepcopy(capture)
    capture["_path"] = path
    return capture


def mark_release(capture: dict) -> dict:
    capture = copy.deepcopy(capture)
    capture["warmups"] = 3
    capture["repeats"] = 9
    return capture


def set_backend(capture: dict, backend: str) -> dict:
    capture = copy.deepcopy(capture)
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    return capture


def set_e2e_median(capture: dict, aggregate_median_us: float) -> dict:
    capture = copy.deepcopy(capture)
    summary = capture.setdefault("timing_summary_us", {}).setdefault("end_to_end", {})
    summary["median"] = aggregate_median_us
    summary["avg"] = aggregate_median_us
    summary["p95"] = aggregate_median_us
    return capture


def set_checksum(capture: dict, checksum: int) -> dict:
    capture = copy.deepcopy(capture)
    capture["checksum_u64"] = checksum
    return capture


def align_current_contract(capture: dict) -> dict:
    capture = copy.deepcopy(capture)
    capture["bound_source"] = "static_profile"
    capture["output_policy"] = {
        "destination_layout": "contiguous_row_major",
        "logical_ld": capture["n"],
        "ld_padding": 0,
        "per_repeat_logical_export": True,
        "final_checksum_export_after_repeats": False,
        "status_handling": "required",
        "status_event_policy": "status_memset_and_status_d2h_labels_required_when_gpu_events_available",
    }
    return capture


def build_grouped_case(
    grouped_per_task_us: float,
    *,
    include_host_batch: bool = True,
    mismatch_host_batch_checksum: bool = False,
    mismatch_host_batch_task_count: bool = False,
) -> list[dict]:
    base = align_current_contract(mark_release(load_capture(FIXTURE_DIR / "v4_bounded_i64_ck.json")))
    best_independent = with_path(set_e2e_median(set_backend(base, "cpu-reference"), 80.0), "independent-cpu.json")
    same_backend = with_path(set_e2e_median(set_backend(base, "hip-direct"), 100.0), "independent-hip-direct.json")
    grouped_task_count = 32

    captures = [best_independent, same_backend]
    if include_host_batch:
        host_batch = as_host_api_batch_capture(same_backend)
        host_batch = set_backend(host_batch, "hip-direct")
        if mismatch_host_batch_task_count:
            host_batch["host_api_batch"]["batch_size"] = 16
        else:
            host_batch["host_api_batch"]["batch_size"] = grouped_task_count
        host_batch = set_e2e_median(host_batch, 70.0 * host_batch["host_api_batch"]["batch_size"])
        if mismatch_host_batch_checksum:
            host_batch = set_checksum(host_batch, int(host_batch["checksum_u64"]) + 1)
        captures.append(with_path(host_batch, "hostbatch-hip-direct.json"))

    grouped = as_grouped_dispatch_capture(same_backend)
    grouped = set_backend(grouped, "hip-direct")
    grouped["grouped_dispatch"]["task_count"] = grouped_task_count
    grouped["grouped_dispatch"]["task_descriptor_contract"]["task_count"] = grouped_task_count
    grouped = set_e2e_median(grouped, grouped_per_task_us * grouped["grouped_dispatch"]["task_count"])
    captures.append(with_path(grouped, "grouped-hip-direct.json"))
    return captures


def main() -> int:
    report = many_small_grouped_report.build_report_from_captures(build_grouped_case(60.0))
    row = next(row for group in report["groups"] for row in group["rows"] if row["mode"] == "grouped_dispatch")
    assert report["summary"]["grouped_dispatch_comparisons"] == 1
    assert report["summary"]["candidate_wins"] == 1
    assert row["decision"] == "candidate_win"
    assert round(row["speedup_vs_best_independent"], 4) == round(80.0 / 60.0, 4)
    assert round(row["speedup_vs_same_backend_host_batch"], 4) == round(70.0 / 60.0, 4)
    assert row["checksum_matches_same_backend_host_batch"] is True
    assert row["same_backend_host_batch_task_count_matches"] is True
    assert row["grouped_task_descriptor_valid"] is True
    assert row["grouped_task_descriptor_bucket_count"] == 1
    assert row["grouped_task_descriptor_device_policy"] == "host_resident_task_loop"

    bucketed_grouped_captures = build_grouped_case(60.0)
    bucketed_descriptor = bucketed_grouped_captures[-1]["grouped_dispatch"]["task_descriptor_contract"]
    bucketed_descriptor.update(
        {
            "descriptor_layout": "same_contract_bucketed_resident_task_triplets_v1",
            "bucket_policy": "same_contract_shape_buckets",
            "bucket_count": 2,
            "same_shape_required": False,
            "shape_key": "multiple_shape_buckets",
            "buckets": [
                {
                    "bucket_index": 0,
                    "task_offset": 0,
                    "task_count": 16,
                    "shape_key": "m=64;n=64;k=64;tile_m=128;tile_n=128;prefix=9",
                    "semantics": bucketed_grouped_captures[-1]["semantics"],
                    "output_domain": "native_i64_u64_host",
                },
                {
                    "bucket_index": 1,
                    "task_offset": 16,
                    "task_count": 16,
                    "shape_key": "m=128;n=128;k=128;tile_m=128;tile_n=128;prefix=9",
                    "semantics": bucketed_grouped_captures[-1]["semantics"],
                    "output_domain": "native_i64_u64_host",
                },
            ],
        }
    )
    bucketed_report = many_small_grouped_report.build_report_from_captures(bucketed_grouped_captures)
    bucketed_row = next(
        row for group in bucketed_report["groups"] for row in group["rows"] if row["mode"] == "grouped_dispatch"
    )
    assert bucketed_row["decision"] == "candidate_win"
    assert bucketed_row["grouped_task_descriptor_valid"] is True
    assert bucketed_row["grouped_task_descriptor_bucket_policy"] == "same_contract_shape_buckets"
    assert bucketed_row["grouped_task_descriptor_bucket_count"] == 2

    device_grouped_captures = build_grouped_case(60.0)
    device_grouped_captures[-1]["grouped_dispatch"][
        "execution_strategy"
    ] = GROUPED_DISPATCH_STRATEGY_DEVICE_GROUPED_PACK_GEMM_HOST_EXPORTS
    device_grouped_captures[-1]["grouped_dispatch"]["task_descriptor_contract"][
        "device_descriptor_policy"
    ] = "device_pointer_tables_and_compact_slabs"
    device_grouped_report = many_small_grouped_report.build_report_from_captures(device_grouped_captures)
    device_grouped_row = next(
        row
        for group in device_grouped_report["groups"]
        for row in group["rows"]
        if row["mode"] == "grouped_dispatch"
    )
    assert device_grouped_row["decision"] == "candidate_win"
    assert device_grouped_row["grouped_task_descriptor_valid"] is True
    assert device_grouped_row["grouped_task_descriptor_device_policy"] == "device_pointer_tables_and_compact_slabs"

    loss_report = many_small_grouped_report.build_report_from_captures(build_grouped_case(90.0))
    loss_row = next(
        row for group in loss_report["groups"] for row in group["rows"] if row["mode"] == "grouped_dispatch"
    )
    assert loss_report["summary"]["deprioritized"] == 1
    assert loss_row["decision"] == "deprioritize"
    assert "grouped_not_faster_than_best_independent_per_task" in loss_row["blockers"]

    missing_report = many_small_grouped_report.build_report_from_captures(
        build_grouped_case(60.0, include_host_batch=False)
    )
    missing_row = next(
        row for group in missing_report["groups"] for row in group["rows"] if row["mode"] == "grouped_dispatch"
    )
    assert missing_report["summary"]["experimental"] == 1
    assert missing_row["decision"] == "keep_experimental"
    assert "missing_same_backend_host_batch_baseline" in missing_row["blockers"]

    checksum_mismatch_report = many_small_grouped_report.build_report_from_captures(
        build_grouped_case(60.0, mismatch_host_batch_checksum=True)
    )
    checksum_mismatch_row = next(
        row
        for group in checksum_mismatch_report["groups"]
        for row in group["rows"]
        if row["mode"] == "grouped_dispatch"
    )
    assert checksum_mismatch_report["summary"]["experimental"] == 1
    assert checksum_mismatch_row["decision"] == "keep_experimental"
    assert "checksum_mismatch_same_backend_host_batch" in checksum_mismatch_row["blockers"]

    task_count_mismatch_report = many_small_grouped_report.build_report_from_captures(
        build_grouped_case(60.0, mismatch_host_batch_task_count=True)
    )
    task_count_mismatch_row = next(
        row
        for group in task_count_mismatch_report["groups"]
        for row in group["rows"]
        if row["mode"] == "grouped_dispatch"
    )
    assert task_count_mismatch_report["summary"]["experimental"] == 1
    assert task_count_mismatch_row["decision"] == "keep_experimental"
    assert "host_batch_task_count_mismatch" in task_count_mismatch_row["blockers"]

    invalid_descriptor_captures = build_grouped_case(60.0)
    invalid_descriptor_captures[-1]["grouped_dispatch"]["task_descriptor_contract"]["task_count"] += 1
    invalid_descriptor_report = many_small_grouped_report.build_report_from_captures(invalid_descriptor_captures)
    invalid_descriptor_row = next(
        row
        for group in invalid_descriptor_report["groups"]
        for row in group["rows"]
        if row["mode"] == "grouped_dispatch"
    )
    assert invalid_descriptor_report["summary"]["experimental"] == 1
    assert invalid_descriptor_row["decision"] == "keep_experimental"
    assert "invalid_grouped_task_descriptor_contract" in invalid_descriptor_row["blockers"]

    invalid_descriptor_policy_captures = build_grouped_case(60.0)
    invalid_descriptor_policy_captures[-1]["grouped_dispatch"]["task_descriptor_contract"][
        "device_descriptor_policy"
    ] = "device_pointer_tables_and_compact_slabs"
    invalid_descriptor_policy_report = many_small_grouped_report.build_report_from_captures(
        invalid_descriptor_policy_captures
    )
    invalid_descriptor_policy_row = next(
        row
        for group in invalid_descriptor_policy_report["groups"]
        for row in group["rows"]
        if row["mode"] == "grouped_dispatch"
    )
    assert invalid_descriptor_policy_report["summary"]["experimental"] == 1
    assert invalid_descriptor_policy_row["decision"] == "keep_experimental"
    assert "invalid_grouped_task_descriptor_contract" in invalid_descriptor_policy_row["blockers"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report_path = tmp_path / "many-small-grouped-report.json"
        review_path = tmp_path / "review_report.json"
        manifest_path = tmp_path / "scenario_manifest.json"
        capture_path = tmp_path / "capture.json"
        report_path.write_text("{}", encoding="utf-8")
        review_path.write_text("{}", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")
        capture_path.write_text("{}", encoding="utf-8")
        expanded = many_small_grouped_report.expand_inputs([tmp_path])
        assert capture_path in expanded
        assert report_path not in expanded
        assert review_path not in expanded
        assert manifest_path not in expanded

    print("many-small grouped report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
