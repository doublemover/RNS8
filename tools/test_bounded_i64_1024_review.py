#!/usr/bin/env python3
"""Self-test bounded-i64 1024 hipBLASLt disposition reporting."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
from pathlib import Path

import bounded_i64_1024_review
import gpu_counter_report
import target_validation_report
from benchmark_schema import load_capture, validate_capture
from promotion_ledger import path_key


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "benchmark_schema"


def tune_capture(capture: dict, *, median: float, backend: str, kernel: str, key_suffix: str) -> dict:
    capture = copy.deepcopy(capture)
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    capture["selected_kernel"] = kernel
    capture["m"] = 1024
    capture["n"] = 1024
    capture["k"] = 1024
    capture["k_block_size"] = 1024
    gpu_events = capture.get("gpu_event_timings_us") if isinstance(capture.get("gpu_event_timings_us"), dict) else {}
    event_lengths = [len(value) for value in gpu_events.values() if isinstance(value, list) and value]
    repeats = event_lengths[0] if event_lengths else int(capture.get("repeats", 2))
    capture["warmups"] = 1
    capture["repeats"] = repeats
    raw = capture["raw_timings_us"]
    summary = capture["timing_summary_us"]
    for phase in ("pack", "rns_gemm", "crt_export", "end_to_end"):
        value = int(median if phase == "end_to_end" else summary[phase]["avg"])
        raw[phase] = [value] * repeats
        summary[phase] = {"avg": float(value), "median": float(value), "p95": float(value)}
    capture["avg_pack_us"] = summary["pack"]["avg"]
    capture["avg_rns_gemm_us"] = summary["rns_gemm"]["avg"]
    capture["avg_crt_export_us"] = summary["crt_export"]["avg"]
    capture["avg_end_to_end_us"] = summary["end_to_end"]["avg"]
    metadata = capture["backend_metadata"]
    metadata["selected_kernel"] = kernel
    metadata["exact_differential_validated"] = True
    metadata["performance_validated"] = backend != "hip-direct"
    metadata["kernel_family"] = kernel
    safety = metadata.get("accumulator_safety")
    if isinstance(safety, dict):
        safety["k_block_size"] = 1024
    key_fields = {}
    for part in str(metadata["autotune_key"]).split(";"):
        if "=" in part:
            name, value = part.split("=", 1)
            key_fields[name] = value
    key_fields.update(
        {
            "backend": backend,
            "target_id": "gfx1100",
            "semantics": "bounded_i64",
            "m": "1024",
            "n": "1024",
            "k": "1024",
            "k_block_size": "1024",
            "kernel": kernel,
            "epilogue": metadata["epilogue_mode"],
            "case": key_suffix,
        }
    )
    metadata["autotune_key"] = ";".join(f"{name}={value}" for name, value in key_fields.items())
    baseline = capture["comparison_baseline"]
    baseline["status"] = "reviewed_release_same_contract_baseline"
    baseline["speedup_claimed"] = True
    baseline["selected_reference"] = "hip-direct"
    baseline.setdefault("required_before_speedup_claim", ["same_contract_cpu_reference"])
    baseline.setdefault("reason", "synthetic self-test reviewed same-contract baseline")
    validate_capture(capture)
    return capture


def write_capture(path: Path, capture: dict) -> Path:
    path.write_text(json.dumps(capture), encoding="utf-8")
    return path


def write_counter_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["VGPR", "SGPR", "TCP_TCC_WRITE_REQ", "WAIT_COUNT"])
        writer.writeheader()
        writer.writerow({"VGPR": "48", "SGPR": "64", "TCP_TCC_WRITE_REQ": "8", "WAIT_COUNT": "1"})
        writer.writerow({"VGPR": "48", "SGPR": "64", "TCP_TCC_WRITE_REQ": "10", "WAIT_COUNT": "3"})


def write_isa(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "object": "hipblaslt.obj",
                "backend": "hipblaslt",
                "target": "gfx1100",
                "reported_symbol_count": 1,
                "instruction_totals": {"wmma": 0, "mfma": 1, "global_store": 4, "lds_mentions": 2, "wait_instructions": 1},
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    hipblaslt_fixture = load_capture(FIXTURE_DIR / "v4_bounded_i64_hipblaslt.json")
    direct_fixture = load_capture(FIXTURE_DIR / "v4_bounded_i64_adaptive_hip.json")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        hip_path = write_capture(
            tmp / "hipblaslt-1024.json",
            tune_capture(
                hipblaslt_fixture,
                median=1500.0,
                backend="hipblaslt",
                kernel="hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2",
                key_suffix="hipblaslt",
            ),
        )
        direct_path = write_capture(
            tmp / "direct-1024.json",
            tune_capture(
                direct_fixture,
                median=2000.0,
                backend="hip-direct",
                kernel="direct_hip_tiled_active_prefix_rns_gemm_v2",
                key_suffix="direct",
            ),
        )

        target_status = tmp / "target-status.json"
        target_status.write_text(
            json.dumps(
                {
                    "host_os": "windows",
                    "target_id": "gfx1100",
                    "hip_sdk_or_rocm_version": "7.1",
                    "gpu_name": "AMD Radeon RX 7900 XTX",
                    "device_index": 0,
                    "hip_runtime_version": hipblaslt_fixture["device"]["hip_runtime_version"],
                    "evidence": {
                        "build": "pass",
                        "ctest": "pass",
                        "smoke": "pass",
                        "release_capture": "pass",
                        "profiler": "pass",
                    },
                    "cache_eligibility": {"eligible": True, "blockers": []},
                }
            ),
            encoding="utf-8",
        )
        target_report = target_validation_report.build_report([hip_path, direct_path], [target_status])
        target_report_path = tmp / "target-validation-report.json"
        target_report_path.write_text(json.dumps(target_report), encoding="utf-8")

        counter_csv = tmp / "counters.csv"
        isa = tmp / "isa.json"
        write_counter_csv(counter_csv)
        write_isa(isa)
        counter = gpu_counter_report.report_for_capture(hip_path, [counter_csv], [isa], 5)
        counter_batch = gpu_counter_report.build_batch_report([counter])
        counter_batch_path = tmp / "gpu-counter-batch-report.json"
        counter_batch_path.write_text(json.dumps(counter_batch), encoding="utf-8")

        hip_capture = load_capture(hip_path)
        key = hip_capture["backend_metadata"]["autotune_key"]
        variance = tmp / "variance.json"
        variance.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "capture": str(hip_path),
                            "capture_key": path_key(hip_path),
                            "promotion_ready": True,
                            "required_speedup_margin": 1.03,
                            "observed_max_relative_noise": 0.01,
                            "blockers": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        ledger = tmp / "promotion-ledger.json"
        ledger.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "autotune_key": key,
                            "promotion_blockers": [],
                            "performance_validated": True,
                            "installed_cache_entry": True,
                            "variance_gate_available": True,
                            "variance_gate_ready": True,
                            "target_validation_gate_available": True,
                            "target_validation_gate_ready": True,
                            "target_cache_eligible": True,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        report = bounded_i64_1024_review.build_report(
            [direct_path, hip_path],
            target_validation=target_report_path,
            counter_report=counter_batch_path,
            promotion_ledger=ledger,
            variance_reports=[variance],
        )
        candidate = report["groups"][0]["candidates"][0]
        assert candidate["disposition"] == "keep cache"
        assert candidate["speedup_vs_direct_hip"] > 1.03
        assert candidate["blockers"] == []
        outputs = bounded_i64_1024_review.write_outputs(report, tmp / "out")
        assert Path(outputs["json"]).exists()
        assert Path(outputs["markdown"]).exists()

    print("bounded-i64 1024 review self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
