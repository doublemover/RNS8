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
HIPBLASLT_KERNEL = "hipblaslt_int8_i32_scratch_reduce_specialized_251_255_256_v2"
PACK_OPERANDS = {
    "per_repeat_repack": [],
    "prepacked_reuse_a": ["A"],
    "prepacked_reuse_b": ["B"],
    "prepacked_reuse": ["A", "B"],
}


def fixture_for_backend(backend: str) -> dict:
    if backend == "hipblaslt":
        return load_capture(FIXTURE_DIR / "v4_bounded_i64_hipblaslt.json")
    if backend == "hip-direct":
        return load_capture(FIXTURE_DIR / "v4_bounded_i64_adaptive_hip.json")
    if backend == "hip-vector-alu-int64":
        return load_capture(FIXTURE_DIR / "v4_bounded_i64_vector_alu.json")
    if backend == "ck":
        return load_capture(FIXTURE_DIR / "v4_bounded_i64_ck.json")
    if backend == "rocwmma":
        return load_capture(FIXTURE_DIR / "v4_bounded_i64_rocwmma.json")
    if backend == "cpu-reference":
        return cpu_reference_fixture()
    raise AssertionError(f"unhandled backend {backend}")


def cpu_reference_fixture() -> dict:
    capture = copy.deepcopy(load_capture(FIXTURE_DIR / "v4_bounded_i64_ck.json"))
    metadata = capture["backend_metadata"]
    capture["backend_requested"] = "cpu-reference"
    capture["backend_selected"] = "cpu-reference"
    capture["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
    metadata["selected_kernel"] = "cpu_reference_scalar_rns_gemm_v1"
    metadata["accelerator_backend"] = False
    metadata["matrix_engine_backend"] = False
    metadata["accelerator_library"] = None
    metadata["accelerator_version"] = None
    metadata["capability_status"] = "implemented_correctness_backend"
    metadata["epilogue_mode"] = "fused_centered_residue_then_crt_export"
    metadata["workspace_mode"] = "host_reference_workspace"
    metadata["workspace_required_bytes"] = 0
    metadata["isa_evidence"] = "not_applicable_cpu_reference"
    timing = capture.get("timing_metadata") if isinstance(capture.get("timing_metadata"), dict) else {}
    timing["gpu_event_timing"] = False
    timing["gpu_event_timing_status"] = "not_requested"
    timing["gpu_event_timing_reason"] = "cpu_reference_capture"
    timing["gpu_event_phase_order"] = None
    timing["gpu_event_timing_source"] = None
    timing["gpu_event_timing_source_scope"] = None
    capture["gpu_event_timings_us"] = None
    capture["gpu_event_timing_summary_us"] = None
    return capture


def apply_pack_mode(capture: dict, pack_mode: str) -> None:
    operands = PACK_OPERANDS[pack_mode]
    reuse = pack_mode != "per_repeat_repack"
    capture["pack_mode"] = pack_mode
    capture["reuse_packed_inputs"] = reuse
    capture["reuse_packed_a"] = "A" in operands
    capture["reuse_packed_b"] = "B" in operands
    capture["prepack_reuse_operands"] = operands
    capture["prepack_reuse_strategy"] = "persistent_matrix_residency" if reuse else "none"
    capture["prepack_setup_us"] = 900 if reuse else None
    capture["avg_prepack_setup_us"] = 900.0 if reuse else None
    timing = capture.get("timing_metadata")
    if not isinstance(timing, dict):
        timing = {}
        capture["timing_metadata"] = timing
    timing["pack_mode"] = pack_mode
    timing["prepack_reuse_operands"] = operands
    timing["prepack_reuse_strategy"] = capture["prepack_reuse_strategy"]
    phase_availability = timing.setdefault("phase_availability", {})
    phase_availability["prepack_setup"] = {
        "timed": reuse,
        "timing_key": "prepack_setup_us" if reuse else None,
        "scope": "one_time_before_warmups" if reuse else "not_requested_per_repeat_repack",
        "reason": "synthetic rank-50 fixture prepack setup timing" if reuse else "per-repeat repack fixture",
    }
    if pack_mode == "prepacked_reuse":
        repeats = int(capture.get("repeats", 2))
        for phase in ["pack"]:
            capture["raw_timings_us"][phase] = [0] * repeats
            capture["timing_summary_us"][phase] = {"avg": 0.0, "median": 0.0, "p95": 0.0}
        capture["avg_pack_us"] = 0.0
        event_timings = capture.get("gpu_event_timings_us")
        event_summary = capture.get("gpu_event_timing_summary_us")
        if isinstance(event_timings, dict):
            for phase in ["pack_h2d", "pack_kernel", "finite_pack_h2d", "finite_pack_kernel", "pack"]:
                if phase in event_timings:
                    event_timings[phase] = [0.0] * repeats
                if isinstance(event_summary, dict) and phase in event_summary:
                    event_summary[phase] = {"avg": 0.0, "median": 0.0, "p95": 0.0}


def resize_gpu_event_timings(capture: dict) -> None:
    repeats = int(capture["repeats"])
    event_timings = capture.get("gpu_event_timings_us")
    if not isinstance(event_timings, dict):
        return
    event_summary = capture.get("gpu_event_timing_summary_us")
    if not isinstance(event_summary, dict):
        event_summary = {}
        capture["gpu_event_timing_summary_us"] = event_summary
    for phase, values in list(event_timings.items()):
        if not isinstance(values, list):
            continue
        if values:
            value = float(values[0])
        else:
            value = 0.0
        event_timings[phase] = [value] * repeats
        event_summary[phase] = {"avg": value, "median": value, "p95": value}


def tune_capture(capture: dict, *, median: float, backend: str, kernel: str, key_suffix: str, pack_mode: str = "per_repeat_repack") -> dict:
    capture = copy.deepcopy(capture)
    capture["backend_requested"] = backend
    capture["backend_selected"] = backend
    capture["selected_kernel"] = kernel
    capture["m"] = 1024
    capture["n"] = 1024
    capture["k"] = 1024
    capture["k_block_size"] = 1024
    device = capture.get("device")
    if isinstance(device, dict):
        device["hip_runtime_version"] = 70100000
        device["hip_driver_version"] = 70100000
    gpu_events = capture.get("gpu_event_timings_us") if isinstance(capture.get("gpu_event_timings_us"), dict) else {}
    event_lengths = [len(value) for value in gpu_events.values() if isinstance(value, list) and value]
    repeats = event_lengths[0] if event_lengths else int(capture.get("repeats", 2))
    capture["warmups"] = 3
    capture["repeats"] = max(repeats, 9)
    raw = capture["raw_timings_us"]
    summary = capture["timing_summary_us"]
    for phase in ("pack", "rns_gemm", "crt_export", "end_to_end"):
        value = int(median if phase == "end_to_end" else summary[phase]["avg"])
        raw[phase] = [value] * capture["repeats"]
        summary[phase] = {"avg": float(value), "median": float(value), "p95": float(value)}
    resize_gpu_event_timings(capture)
    capture["avg_pack_us"] = summary["pack"]["avg"]
    capture["avg_rns_gemm_us"] = summary["rns_gemm"]["avg"]
    capture["avg_crt_export_us"] = summary["crt_export"]["avg"]
    capture["avg_end_to_end_us"] = summary["end_to_end"]["avg"]
    prefix = capture.get("prefix")
    if isinstance(prefix, int) and prefix > 0:
        capture["avg_per_modulus_gemm_estimate_us"] = capture["avg_rns_gemm_us"] / prefix
    metadata = capture["backend_metadata"]
    metadata["selected_kernel"] = kernel
    metadata["exact_differential_validated"] = True
    metadata["performance_validated"] = backend not in {"cpu-reference", "hip-direct", "hip-vector-alu-int64"}
    metadata["kernel_family"] = kernel
    k_block_cap_by_backend = {
        "cpu-reference": 65536,
        "hip-direct": 65536,
        "hip-vector-alu-int64": 0,
        "hipblaslt": 65536,
        "ck": 32768,
        "rocwmma": 65536,
    }
    k_block_cap = k_block_cap_by_backend[backend]
    safety = metadata.get("accumulator_safety")
    if isinstance(safety, dict):
        safety["k_block_size"] = 1024
        safety["k_block_cap"] = k_block_cap
    key_fields = {}
    for part in str(metadata["autotune_key"]).split(";"):
        if "=" in part:
            name, value = part.split("=", 1)
            key_fields[name] = value
    key_fields.update(
        {
            "backend": backend,
            "target_id": "cpu" if backend == "cpu-reference" else "gfx1100",
            "semantics": "bounded_i64",
            "m": "1024",
            "n": "1024",
            "k": "1024",
            "k_block_size": "1024",
            "k_block_cap": str(k_block_cap),
            "kernel": kernel,
            "epilogue": metadata["epilogue_mode"],
            "case": key_suffix,
        }
    )
    metadata["autotune_key"] = ";".join(f"{name}={value}" for name, value in key_fields.items())
    baseline = capture["comparison_baseline"]
    baseline["status"] = "reviewed_release_same_contract_baseline"
    baseline["speedup_claimed"] = backend not in {"cpu-reference", "hip-direct", "hip-vector-alu-int64"}
    baseline["selected_reference"] = "hip-direct"
    baseline.setdefault("required_before_speedup_claim", ["same_contract_cpu_reference"])
    baseline.setdefault("reason", "synthetic self-test reviewed same-contract baseline")
    apply_pack_mode(capture, pack_mode)
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


def make_capture(tmp: Path, backend: str, median: float, suffix: str, *, pack_mode: str = "per_repeat_repack") -> Path:
    kernels = {
        "cpu-reference": "cpu_reference_scalar_rns_gemm_v1",
        "hip-direct": "direct_hip_tiled_active_prefix_rns_gemm_v2",
        "hip-vector-alu-int64": "hip_vector_alu_i64_exact_192b_v1",
        "hipblaslt": HIPBLASLT_KERNEL,
        "ck": "ck_wmma_cshuffle_i8_i32_mod251_255_256_centered_epilogue_v2",
        "rocwmma": "rocwmma_i8_i32_signed_mod251_255_256_hot_residue_v2",
    }
    stem = f"{backend}-{suffix}".replace(":", "-")
    return write_capture(
        tmp / f"{stem}.json",
        tune_capture(
            fixture_for_backend(backend),
            median=median,
            backend=backend,
            kernel=kernels[backend],
            key_suffix=suffix,
            pack_mode=pack_mode,
        ),
    )


def write_rank50_sidecars(tmp: Path, capture_paths: list[Path], cache_candidate: Path) -> tuple[Path, Path, Path, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    target_status = tmp / "target-status.json"
    hipblaslt_capture = load_capture(cache_candidate)
    target_status.write_text(
        json.dumps(
            {
                "host_os": "windows",
                "target_id": "gfx1100",
                "hip_sdk_or_rocm_version": "7.1",
                "gpu_name": "AMD Radeon RX 7900 XTX",
                "device_index": 0,
                "hip_runtime_version": hipblaslt_capture["device"]["hip_runtime_version"],
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
    target_report = target_validation_report.build_report(capture_paths, [target_status])
    target_report_path = tmp / "target-validation-report.json"
    target_report_path.write_text(json.dumps(target_report), encoding="utf-8")

    counter_csv = tmp / "counters.csv"
    isa = tmp / "isa.json"
    write_counter_csv(counter_csv)
    write_isa(isa)
    counter = gpu_counter_report.report_for_capture(cache_candidate, [counter_csv], [isa], 5)
    counter_batch = gpu_counter_report.build_batch_report([counter])
    counter_batch_path = tmp / "gpu-counter-batch-report.json"
    counter_batch_path.write_text(json.dumps(counter_batch), encoding="utf-8")

    key = hipblaslt_capture["backend_metadata"]["autotune_key"]
    variance = tmp / "variance.json"
    variance.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "capture": str(cache_candidate),
                        "capture_key": path_key(cache_candidate),
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
    return target_report_path, counter_batch_path, ledger, variance


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        cpu = make_capture(tmp, "cpu-reference", 120000.0, "cpu")
        direct = make_capture(tmp, "hip-direct", 2000.0, "direct")
        vector = make_capture(tmp, "hip-vector-alu-int64", 2100.0, "vector")
        ck = make_capture(tmp, "ck", 2200.0, "ck")
        rocwmma = make_capture(tmp, "rocwmma", 2300.0, "rocwmma")
        hip = make_capture(tmp, "hipblaslt", 1500.0, "hipblaslt")
        reuse_a = make_capture(tmp, "hipblaslt", 1750.0, "reuse-a", pack_mode="prepacked_reuse_a")
        reuse_b = make_capture(tmp, "hipblaslt", 1900.0, "reuse-b", pack_mode="prepacked_reuse_b")
        reuse_ab = make_capture(tmp, "hipblaslt", 1400.0, "reuse-ab", pack_mode="prepacked_reuse")
        capture_paths = [cpu, direct, vector, ck, rocwmma, hip, reuse_a, reuse_b, reuse_ab]
        target_report_path, counter_batch_path, ledger, variance = write_rank50_sidecars(tmp, capture_paths, hip)
        report = bounded_i64_1024_review.build_report(
            capture_paths,
            target_validation=target_report_path,
            counter_report=counter_batch_path,
            promotion_ledger=ledger,
            variance_reports=[variance],
        )
        assert report["rank50_gate_complete"] is True
        group = report["groups"][0]
        assert group["rank50_group_complete"] is True
        assert group["comparator_coverage"]["missing_required_comparator_backends"] == []
        assert group["hipblaslt_pack_coverage"]["missing_hipblaslt_pack_modes"] == []
        candidates = {candidate["pack_mode"]: candidate for candidate in group["candidates"]}
        assert candidates["per_repeat_repack"]["disposition"] == "keep cache"
        assert candidates["per_repeat_repack"]["speedup_vs_direct_hip"] > 1.03
        assert candidates["per_repeat_repack"]["speedup_vs_best_required_comparator"] > 1.03
        assert candidates["per_repeat_repack"]["blockers"] == []
        assert candidates["prepacked_reuse_a"]["disposition"] == "keep experimental"
        assert "prepacked_reuse_not_cache_promotable" in candidates["prepacked_reuse_a"]["blockers"]
        assert candidates["prepacked_reuse_b"]["disposition"] == "keep experimental"
        assert candidates["prepacked_reuse"]["disposition"] == "keep experimental"

        missing_vector_report = bounded_i64_1024_review.build_report(
            [path for path in capture_paths if path != vector],
            target_validation=target_report_path,
            counter_report=counter_batch_path,
            promotion_ledger=ledger,
            variance_reports=[variance],
        )
        assert missing_vector_report["rank50_gate_complete"] is False
        missing_candidate = next(
            candidate
            for candidate in missing_vector_report["groups"][0]["candidates"]
            if candidate.get("pack_mode") == "per_repeat_repack"
        )
        assert "missing_required_comparator:hip-vector-alu-int64" in missing_candidate["blockers"]

        missing_pack_report = bounded_i64_1024_review.build_report(
            [path for path in capture_paths if path != reuse_b],
            target_validation=target_report_path,
            counter_report=counter_batch_path,
            promotion_ledger=ledger,
            variance_reports=[variance],
        )
        assert missing_pack_report["rank50_gate_complete"] is False
        missing_pack_candidate = next(
            candidate
            for candidate in missing_pack_report["groups"][0]["candidates"]
            if candidate.get("pack_mode") == "per_repeat_repack"
        )
        assert "missing_hipblaslt_pack_mode:prepacked_reuse_b" in missing_pack_candidate["blockers"]

        slow_hip = make_capture(tmp, "hipblaslt", 2500.0, "hipblaslt-slow")
        slow_capture_paths = [cpu, direct, vector, ck, rocwmma, slow_hip, reuse_a, reuse_b, reuse_ab]
        slow_target_report_path, slow_counter_path, slow_ledger, slow_variance = write_rank50_sidecars(
            tmp / "slow",
            slow_capture_paths,
            slow_hip,
        )
        slow_report = bounded_i64_1024_review.build_report(
            slow_capture_paths,
            target_validation=slow_target_report_path,
            counter_report=slow_counter_path,
            promotion_ledger=slow_ledger,
            variance_reports=[slow_variance],
        )
        slow_candidate = next(
            candidate
            for candidate in slow_report["groups"][0]["candidates"]
            if candidate.get("pack_mode") == "per_repeat_repack"
        )
        assert slow_report["rank50_gate_complete"] is False
        assert slow_candidate["disposition"] == "drop/deprioritize"
        assert "not_faster_than_best_required_comparator" in slow_candidate["blockers"]

        unsupported_report = bounded_i64_1024_review.build_report(
            [cpu, direct, vector, ck, rocwmma],
            target_validation=target_report_path,
            counter_report=counter_batch_path,
            promotion_ledger=ledger,
            variance_reports=[variance],
        )
        assert unsupported_report["rank50_gate_complete"] is False
        assert unsupported_report["groups"][0]["candidates"][0]["disposition"] == "unsupported accelerator"

        outputs = bounded_i64_1024_review.write_outputs(report, tmp / "out")
        assert Path(outputs["json"]).exists()
        assert Path(outputs["markdown"]).exists()

    print("bounded-i64 1024 review self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
