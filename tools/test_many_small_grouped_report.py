#!/usr/bin/env python3
"""Self-test many-small grouped-dispatch comparison reporting."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import many_small_grouped_report
from benchmark_schema import load_capture
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


def build_grouped_case(grouped_per_task_us: float, *, include_host_batch: bool = True) -> list[dict]:
    base = align_current_contract(mark_release(load_capture(FIXTURE_DIR / "v4_bounded_i64_ck.json")))
    best_independent = with_path(set_e2e_median(set_backend(base, "cpu-reference"), 80.0), "independent-cpu.json")
    same_backend = with_path(set_e2e_median(set_backend(base, "hip-direct"), 100.0), "independent-hip-direct.json")

    captures = [best_independent, same_backend]
    if include_host_batch:
        host_batch = as_host_api_batch_capture(same_backend)
        host_batch = set_backend(host_batch, "hip-direct")
        host_batch = set_e2e_median(host_batch, 70.0 * host_batch["host_api_batch"]["batch_size"])
        captures.append(with_path(host_batch, "hostbatch-hip-direct.json"))

    grouped = as_grouped_dispatch_capture(same_backend)
    grouped = set_backend(grouped, "hip-direct")
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

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report_path = tmp_path / "many-small-grouped-report.json"
        capture_path = tmp_path / "capture.json"
        report_path.write_text("{}", encoding="utf-8")
        capture_path.write_text("{}", encoding="utf-8")
        expanded = many_small_grouped_report.expand_inputs([tmp_path])
        assert capture_path in expanded
        assert report_path not in expanded

    print("many-small grouped report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
