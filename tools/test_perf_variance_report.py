#!/usr/bin/env python3
"""Self-test performance variance gate reporting and ledger integration."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from benchmark_schema import load_capture, validate_capture
from benchmark_schema.core_shared import REPEATED_TIMING_PHASES, _average, _percentile

import perf_variance_report
import promotion_ledger


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "benchmark_schema"
PHASES = ("planning", "scheduling", "matrix_alloc", "pack", "rns_gemm", "crt_export", "end_to_end")


def _summary(values: list[float]) -> dict[str, float]:
    return {"avg": _average(values), "median": _percentile(values, 0.50), "p95": _percentile(values, 0.95)}


def capture_with_timings(values: list[float]) -> dict:
    capture = copy.deepcopy(load_capture(FIXTURE_DIR / "v4_bounded_i64_ck.json"))
    capture["warmups"] = 3
    capture["repeats"] = len(values)
    raw = capture.setdefault("raw_timings_us", {})
    summary = capture.setdefault("timing_summary_us", {})
    for phase in PHASES:
        if phase == "end_to_end":
            phase_values = [int(value) for value in values]
        elif phase not in REPEATED_TIMING_PHASES:
            phase_values = [int(raw[phase][0])]
        else:
            current = summary.get(phase, {})
            baseline = int(current.get("avg", 1)) if isinstance(current, dict) else 1
            phase_values = [baseline] * len(values)
        raw[phase] = phase_values
        summary[phase] = _summary(phase_values)
    for field, phase in (
        ("avg_pack_us", "pack"),
        ("avg_rns_gemm_us", "rns_gemm"),
        ("avg_crt_export_us", "crt_export"),
        ("avg_end_to_end_us", "end_to_end"),
    ):
        capture[field] = summary[phase]["avg"]
    capture["avg_planning_us"] = summary["planning"]["avg"]
    capture["avg_scheduling_us"] = summary["scheduling"]["avg"]
    capture["avg_matrix_alloc_us"] = summary["matrix_alloc"]["avg"]
    capture["schedule_query_us"] = summary["scheduling"]["avg"]
    prefix = capture.get("selected_prefix", capture.get("prefix"))
    if isinstance(prefix, int) and prefix > 0 and capture.get("per_modulus_gemm_estimate_applicable") is not False:
        capture["avg_per_modulus_gemm_estimate_us"] = summary["rns_gemm"]["avg"] / float(prefix)
    gpu_raw = capture.get("gpu_event_timings_us")
    gpu_summary = capture.get("gpu_event_timing_summary_us")
    if isinstance(gpu_raw, dict) and isinstance(gpu_summary, dict):
        for phase, existing in list(gpu_raw.items()):
            current = gpu_summary.get(phase, {})
            baseline = float(current.get("median", existing[-1] if existing else 1.0))
            phase_values = [baseline] * len(values)
            gpu_raw[phase] = phase_values
            gpu_summary[phase] = _summary(phase_values)
    validate_capture(capture)
    return capture


def write_capture(path: Path, capture: dict) -> Path:
    path.write_text(json.dumps(capture), encoding="utf-8")
    return path


def write_review_report(path: Path, capture_path: Path, speedup: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "candidates": [
                            {
                                "capture": str(capture_path),
                                "promotable": True,
                                "promotion_blockers": [],
                                "speedup_vs_direct_hip": speedup,
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        stable_a = write_capture(tmp / "stable-a.json", capture_with_timings([100.0, 101.0, 99.0]))
        stable_b = write_capture(tmp / "stable-b.json", capture_with_timings([102.0, 101.0, 103.0]))
        report = perf_variance_report.build_report(
            [stable_a, stable_b],
            min_repeats=3,
            min_runs=2,
            require_multi_run=True,
            max_within_capture_rel_spread=0.05,
            max_run_to_run_rel_spread=0.05,
            min_noise_speedup_margin=0.03,
        )
        assert report["blocked_count"] == 0
        assert report["capture_count"] == 2
        assert all(entry["required_speedup_margin"] >= 1.03 for entry in report["entries"])

        single_report = perf_variance_report.build_report(
            [stable_a],
            min_repeats=3,
            min_runs=2,
            require_multi_run=True,
        )
        assert single_report["blocked_count"] == 1
        assert "multi_run_variance_missing" in single_report["entries"][0]["blockers"]

        noisy = write_capture(tmp / "noisy.json", capture_with_timings([100.0, 101.0, 140.0]))
        noisy_report = perf_variance_report.build_report(
            [noisy],
            min_repeats=3,
            max_within_capture_rel_spread=0.10,
        )
        assert noisy_report["blocked_count"] == 1
        assert "high_within_capture_variance" in noisy_report["entries"][0]["blockers"]

        variance_path = tmp / "variance-report.json"
        variance_path.write_text(json.dumps(report), encoding="utf-8")
        cache_key = capture_with_timings([100.0, 101.0, 99.0])["backend_metadata"]["autotune_key"]
        cache_path = tmp / "cache.json"
        cache_path.write_text(json.dumps({"schema_version": 1, "entries": [{"key": cache_key}]}), encoding="utf-8")

        narrow_review = write_review_report(tmp / "narrow-review.json", stable_a, 1.02)
        narrow_ledger = promotion_ledger.build_ledger([stable_a], cache_path, [narrow_review], [variance_path])
        assert "speedup_inside_variance_margin" in narrow_ledger["entries"][0]["promotion_blockers"]

        clear_review = write_review_report(tmp / "clear-review.json", stable_a, 1.10)
        clear_ledger = promotion_ledger.build_ledger([stable_a], cache_path, [clear_review], [variance_path])
        assert clear_ledger["entries"][0]["promotion_blockers"] == []

        missing_variance_ledger = promotion_ledger.build_ledger(
            [stable_a],
            cache_path,
            [clear_review],
            [],
            require_variance_gate=True,
        )
        assert "missing_variance_gate_entry" in missing_variance_ledger["entries"][0]["promotion_blockers"]

    print("perf variance report self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
