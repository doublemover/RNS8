#!/usr/bin/env python3
"""Compare many-small independent, host-batch, and grouped-dispatch captures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture
from metadata_registry_constants import GROUPED_STRATEGY_DEVICE_DESCRIPTOR_POLICIES


DEFAULT_OUT_DIR = Path("temp") / "many-small-grouped-reports"
RELEASE_MIN_WARMUPS = 3
RELEASE_MIN_REPEATS = 9
REFERENCE_BACKENDS = {"cpu-reference", "wrap64-byte-limb"}
REPORT_OUTPUT_NAMES = {"many-small-grouped-report.json"}
SINGLE_GROUPED_DESCRIPTOR_LAYOUT = "same_shape_resident_task_triplets_v1"
BUCKETED_GROUPED_DESCRIPTOR_LAYOUT = "same_contract_bucketed_resident_task_triplets_v1"
SINGLE_GROUPED_BUCKET_POLICY = "single_same_shape_bucket"
BUCKETED_GROUPED_BUCKET_POLICY = "same_contract_shape_buckets"


def expand_inputs(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(candidate for candidate in sorted(path.rglob("*.json")) if candidate.name not in REPORT_OUTPUT_NAMES)
        else:
            expanded.append(path)
    return expanded


def load_validated_capture(path: Path) -> dict[str, Any]:
    try:
        capture = load_capture(path)
        validate_capture(capture, path)
    except BenchmarkSchemaError as exc:
        raise SystemExit(str(exc)) from exc
    capture["_path"] = str(path)
    return capture


def benchmark_execution_mode(capture: dict[str, Any]) -> str:
    value = capture.get("benchmark_execution_mode")
    if isinstance(value, str):
        return value
    metadata = capture.get("timing_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("benchmark_execution_mode"), str):
        return metadata["benchmark_execution_mode"]
    return "persistent_resident"


def is_grouped_dispatch(capture: dict[str, Any]) -> bool:
    grouped = capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    if grouped.get("requested") is True:
        return True
    return benchmark_execution_mode(capture) == "benchmark_grouped_dispatch_evidence"


def is_host_api_batch(capture: dict[str, Any]) -> bool:
    host_batch = capture.get("host_api_batch") if isinstance(capture.get("host_api_batch"), dict) else {}
    if host_batch.get("enabled") is True:
        return True
    return benchmark_execution_mode(capture) == "benchmark_host_api_batch"


def mode_for_capture(capture: dict[str, Any]) -> str:
    if is_grouped_dispatch(capture):
        return "grouped_dispatch"
    if is_host_api_batch(capture):
        return "host_api_batch"
    return "independent_call"


def backend_id(capture: dict[str, Any]) -> str:
    selected = capture.get("backend_selected")
    return str(selected) if selected is not None else str(capture.get("backend_requested"))


def task_count_for_capture(capture: dict[str, Any]) -> int:
    grouped = capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    if grouped.get("requested"):
        value = grouped.get("task_count")
        return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 1
    host_batch = capture.get("host_api_batch") if isinstance(capture.get("host_api_batch"), dict) else {}
    if host_batch.get("enabled"):
        value = host_batch.get("batch_size")
        return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 1
    return 1


def grouped_task_descriptor_contract(capture: dict[str, Any] | None) -> dict[str, Any]:
    if capture is None:
        return {}
    grouped = capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    contract = grouped.get("task_descriptor_contract") if isinstance(grouped, dict) else None
    return contract if isinstance(contract, dict) else {}


def expected_grouped_output_domain(capture: dict[str, Any]) -> str:
    semantics = capture.get("semantics")
    if semantics in {"finite_ring_u8", "finite_field_u8"}:
        return "finite_u8_host"
    if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
        return "exact_wide_limb_host"
    return "native_i64_u64_host"


def expected_grouped_shape_key(capture: dict[str, Any]) -> str:
    selected_prefix = capture.get("selected_prefix", capture.get("prefix"))
    return (
        f"m={capture.get('m')};n={capture.get('n')};k={capture.get('k')};"
        f"tile_m={capture.get('tile_m')};tile_n={capture.get('tile_n')};"
        f"prefix={selected_prefix}"
    )


def expected_grouped_device_descriptor_policy(capture: dict[str, Any]) -> str:
    grouped = capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    strategy = grouped.get("execution_strategy")
    return GROUPED_STRATEGY_DEVICE_DESCRIPTOR_POLICIES.get(strategy, "host_resident_task_loop")


def grouped_task_descriptor_valid(capture: dict[str, Any]) -> bool:
    contract = grouped_task_descriptor_contract(capture)
    common_valid = (
        contract.get("schema_version") == 1
        and contract.get("task_count") == task_count_for_capture(capture)
        and contract.get("semantics") == capture.get("semantics")
        and contract.get("output_domain") == expected_grouped_output_domain(capture)
        and contract.get("source_version_policy") == "per_task_monotonic_source_version_repack"
        and contract.get("workspace_policy") == "one_workspace_per_task_shared_plan"
        and contract.get("matrix_ownership_policy") == "benchmark_owns_all_task_triplets_until_capture_end"
        and contract.get("descriptor_reuse_policy") == "reuse_after_shape_workspace_source_validation"
        and contract.get("stride_policy") == "matrix_ld_matches_logical_shape_host_output_ld_explicit"
        and contract.get("output_currentness_policy")
        == "device_residue_current_after_grouped_gemm_host_output_after_export"
        and contract.get("lifetime_policy") == "task_matrices_and_workspaces_destroyed_after_capture"
        and contract.get("checksum_policy") == "combined_per_task_checksum_u64"
        and contract.get("status_policy") == "fail_fast_per_task_operation_status"
        and contract.get("device_descriptor_policy") == expected_grouped_device_descriptor_policy(capture)
        and contract.get("promotion_eligible") is False
    )
    if not common_valid:
        return False
    if contract.get("bucket_policy") == SINGLE_GROUPED_BUCKET_POLICY:
        return (
            contract.get("descriptor_layout") == SINGLE_GROUPED_DESCRIPTOR_LAYOUT
            and contract.get("bucket_count") == 1
            and contract.get("same_shape_required") is True
            and contract.get("shape_key") == expected_grouped_shape_key(capture)
        )
    if contract.get("bucket_policy") != BUCKETED_GROUPED_BUCKET_POLICY:
        return False
    if (
        contract.get("descriptor_layout") != BUCKETED_GROUPED_DESCRIPTOR_LAYOUT
        or not isinstance(contract.get("bucket_count"), int)
        or contract.get("bucket_count") < 2
        or contract.get("same_shape_required") is not False
    ):
        return False
    buckets = contract.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != contract.get("bucket_count"):
        return False
    offset = 0
    for index, bucket in enumerate(buckets):
        if not isinstance(bucket, dict):
            return False
        bucket_task_count = bucket.get("task_count")
        if (
            bucket.get("bucket_index") != index
            or bucket.get("task_offset") != offset
            or not isinstance(bucket_task_count, int)
            or bucket_task_count <= 1
            or not isinstance(bucket.get("shape_key"), str)
            or not bucket.get("shape_key")
            or bucket.get("semantics") != capture.get("semantics")
            or bucket.get("output_domain") != expected_grouped_output_domain(capture)
        ):
            return False
        offset += bucket_task_count
    return offset == task_count_for_capture(capture)


def timing_summary_value(capture: dict[str, Any], phase: str, statistic: str) -> float | None:
    summary = capture.get("timing_summary_us")
    if not isinstance(summary, dict):
        return None
    phase_summary = summary.get(phase)
    if not isinstance(phase_summary, dict):
        return None
    value = phase_summary.get(statistic)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def median_end_to_end_us(capture: dict[str, Any]) -> float | None:
    return timing_summary_value(capture, "end_to_end", "median")


def median_per_task_end_to_end_us(capture: dict[str, Any]) -> float | None:
    median = median_end_to_end_us(capture)
    task_count = task_count_for_capture(capture)
    return median / task_count if median is not None and task_count > 0 else None


def release_satisfied(capture: dict[str, Any]) -> bool:
    return (
        isinstance(capture.get("warmups"), int)
        and isinstance(capture.get("repeats"), int)
        and capture["warmups"] >= RELEASE_MIN_WARMUPS
        and capture["repeats"] >= RELEASE_MIN_REPEATS
    )


def gpu_backend(capture: dict[str, Any] | None) -> bool:
    return capture is not None and backend_id(capture) not in REFERENCE_BACKENDS


def gpu_events_available(capture: dict[str, Any] | None) -> bool | None:
    if capture is None or not gpu_backend(capture):
        return None
    metadata = capture.get("timing_metadata")
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("gpu_event_timing") is True
        and metadata.get("gpu_event_timing_status") == "available"
        and isinstance(metadata.get("gpu_event_phase_order"), list)
        and bool(metadata.get("gpu_event_timing_source"))
    )


def checksum_value(capture: dict[str, Any] | None) -> int | None:
    if capture is None:
        return None
    value = capture.get("checksum_u64")
    if value is None:
        value = capture.get("checksum")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def checksum_matches(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool | None:
    left_checksum = checksum_value(left)
    right_checksum = checksum_value(right)
    if left_checksum is None or right_checksum is None:
        return None
    return left_checksum == right_checksum


def _nested(capture: dict[str, Any], path: str) -> Any:
    value: Any = capture
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _exact_output_status_policy(capture: dict[str, Any]) -> str | None:
    status_policy = _nested(capture, "exact_output_contract.status_policy")
    if status_policy is not None:
        return str(status_policy)
    status_check = capture.get("exact_wide_export_status_check")
    if isinstance(status_check, str) and "elided" in status_check:
        return "structurally_elided"
    if status_check is not None:
        return "required"
    return None


def _normalized_contract_value(capture: dict[str, Any], field: str) -> Any:
    value = _nested(capture, field)
    semantics = capture.get("semantics")
    if field == "exact_output_contract.requested_final_output" and value is None:
        if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
            return "exact_wide_limb_host"
    if field == "exact_output_contract.limb_count" and value is None:
        if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
            return capture.get("exact_wide_limb_count")
    if field == "exact_output_contract.status_policy":
        return _exact_output_status_policy(capture)
    if field == "export_variant.name" and value is None:
        return "default"
    if field == "reconstruction_variant.name" and value is None:
        return "default_garner"
    if field == "modulus_set.name" and value is None:
        return "default"
    if field == "residue_count_policy.policy" and value is None:
        return capture.get("contract_prefix_policy")
    if field == "tile_shape_variant.name" and value is None:
        return "default"
    if field == "tile_shape_variant.tile_m" and value is None:
        return capture.get("tile_m")
    if field == "tile_shape_variant.tile_n" and value is None:
        return capture.get("tile_n")
    if field == "tile_shape_variant.tile_k" and value is None:
        return capture.get("k_block_size")
    if field == "workload_proxy.family" and value is None:
        return "not_requested"
    if field == "workload_proxy.label" and value is None:
        return "none"
    return value


def normalized_contract_key(capture: dict[str, Any]) -> str:
    fields = [
        "semantics",
        "bound_kind",
        "bound_mode",
        "bound",
        "bound_source",
        "m",
        "n",
        "k",
        "prefix",
        "selected_prefix",
        "requested_max_prefix",
        "contract_prefix_policy",
        "residue_output_mode",
        "tile_m",
        "tile_n",
        "k_block_size",
        "finite_modulus",
        "exact_output_contract.requested_final_output",
        "exact_output_contract.limb_count",
        "exact_output_contract.status_policy",
        "output_policy.destination_layout",
        "output_policy.status_handling",
        "export_variant.name",
        "reconstruction_variant.name",
        "modulus_set.name",
        "residue_count_policy.policy",
        "tile_shape_variant.name",
        "tile_shape_variant.tile_m",
        "tile_shape_variant.tile_n",
        "tile_shape_variant.tile_k",
        "workload_proxy.family",
        "workload_proxy.label",
        "seed",
        "input_distribution",
    ]
    return ";".join(f"{field}={_normalized_contract_value(capture, field)}" for field in fields)


def capture_summary(capture: dict[str, Any] | None) -> dict[str, Any] | None:
    if capture is None:
        return None
    grouped = capture.get("grouped_dispatch") if isinstance(capture.get("grouped_dispatch"), dict) else {}
    descriptor = grouped_task_descriptor_contract(capture)
    return {
        "path": capture.get("_path"),
        "mode": mode_for_capture(capture),
        "backend": backend_id(capture),
        "task_count": task_count_for_capture(capture),
        "median_end_to_end_us": median_end_to_end_us(capture),
        "median_per_task_end_to_end_us": median_per_task_end_to_end_us(capture),
        "release_review": release_satisfied(capture),
        "gpu_events_available": gpu_events_available(capture),
        "grouped_dispatch_execution_strategy": grouped.get("execution_strategy"),
        "grouped_dispatch_batched_export_enabled": grouped.get("batched_export_enabled"),
        "grouped_task_descriptor_layout": descriptor.get("descriptor_layout"),
        "grouped_task_descriptor_bucket_policy": descriptor.get("bucket_policy"),
        "grouped_task_descriptor_bucket_count": descriptor.get("bucket_count"),
        "grouped_task_descriptor_device_policy": descriptor.get("device_descriptor_policy"),
        "grouped_task_descriptor_valid": grouped_task_descriptor_valid(capture) if is_grouped_dispatch(capture) else None,
    }


def fastest(captures: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [capture for capture in captures if median_per_task_end_to_end_us(capture) is not None]
    if not valid:
        return None
    return min(valid, key=lambda capture: median_per_task_end_to_end_us(capture) or float("inf"))


def speedup(numerator_capture: dict[str, Any] | None, denominator_capture: dict[str, Any]) -> float | None:
    numerator = median_per_task_end_to_end_us(numerator_capture) if numerator_capture is not None else None
    denominator = median_per_task_end_to_end_us(denominator_capture)
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def decision_for(
    candidate: dict[str, Any],
    same_backend_independent: dict[str, Any] | None,
    best_independent: dict[str, Any] | None,
    same_backend_host_batch: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if best_independent is None:
        return "missing_baseline", ["missing_independent_contract_baseline"]
    if same_backend_independent is None:
        blockers.append("missing_same_backend_independent_baseline")
    if same_backend_host_batch is None:
        blockers.append("missing_same_backend_host_batch_baseline")
    elif task_count_for_capture(candidate) != task_count_for_capture(same_backend_host_batch):
        blockers.append("host_batch_task_count_mismatch")
    host_batch_checksum_match = checksum_matches(candidate, same_backend_host_batch)
    if same_backend_host_batch is not None and host_batch_checksum_match is None:
        blockers.append("missing_same_backend_host_batch_checksum")
    elif host_batch_checksum_match is False:
        blockers.append("checksum_mismatch_same_backend_host_batch")
    if not grouped_task_descriptor_valid(candidate):
        blockers.append("invalid_grouped_task_descriptor_contract")
    release_inputs = [candidate, best_independent, same_backend_independent, same_backend_host_batch]
    if any(capture is not None and not release_satisfied(capture) for capture in release_inputs):
        blockers.append("not_release_review")
    for label, capture in [
        ("grouped_dispatch", candidate),
        ("best_independent", best_independent),
        ("same_backend_independent", same_backend_independent),
        ("same_backend_host_batch", same_backend_host_batch),
    ]:
        events = gpu_events_available(capture)
        if events is False:
            blockers.append(f"missing_{label}_gpu_events")
    if speedup(best_independent, candidate) is None:
        blockers.append("missing_end_to_end_timing")
    if blockers:
        return "keep_experimental", blockers
    if speedup(best_independent, candidate) <= 1.0:
        return "deprioritize", ["grouped_not_faster_than_best_independent_per_task"]
    same_backend_speedup = speedup(same_backend_independent, candidate)
    if same_backend_speedup is not None and same_backend_speedup <= 1.0:
        return "deprioritize", ["grouped_not_faster_than_same_backend_independent_per_task"]
    host_batch_speedup = speedup(same_backend_host_batch, candidate)
    if host_batch_speedup is not None and host_batch_speedup <= 1.0:
        return "deprioritize", ["grouped_not_faster_than_same_backend_host_batch_per_task"]
    return "candidate_win", []


def row_for_capture(
    capture: dict[str, Any],
    same_backend_independent: dict[str, Any] | None,
    best_independent: dict[str, Any] | None,
    same_backend_host_batch: dict[str, Any] | None,
) -> dict[str, Any]:
    mode = mode_for_capture(capture)
    descriptor = grouped_task_descriptor_contract(capture)
    if mode == "grouped_dispatch":
        decision, blockers = decision_for(capture, same_backend_independent, best_independent, same_backend_host_batch)
    elif mode == "host_api_batch":
        decision, blockers = "host_batch_baseline", []
    else:
        decision, blockers = "independent_baseline", []
    return {
        "path": capture.get("_path"),
        "mode": mode,
        "backend": backend_id(capture),
        "semantics": capture.get("semantics"),
        "finite_modulus": capture.get("finite_modulus"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
        "task_count": task_count_for_capture(capture),
        "median_end_to_end_us": median_end_to_end_us(capture),
        "median_per_task_end_to_end_us": median_per_task_end_to_end_us(capture),
        "release_review": release_satisfied(capture),
        "gpu_events_available": gpu_events_available(capture),
        "grouped_dispatch_status": (
            capture.get("grouped_dispatch", {}).get("capture_status")
            if isinstance(capture.get("grouped_dispatch"), dict)
            else None
        ),
        "grouped_dispatch_execution_strategy": (
            capture.get("grouped_dispatch", {}).get("execution_strategy")
            if isinstance(capture.get("grouped_dispatch"), dict)
            else None
        ),
        "grouped_dispatch_batched_export_enabled": (
            capture.get("grouped_dispatch", {}).get("batched_export_enabled")
            if isinstance(capture.get("grouped_dispatch"), dict)
            else None
        ),
        "grouped_task_descriptor_layout": descriptor.get("descriptor_layout"),
        "grouped_task_descriptor_bucket_policy": descriptor.get("bucket_policy"),
        "grouped_task_descriptor_bucket_count": descriptor.get("bucket_count"),
        "grouped_task_descriptor_device_policy": descriptor.get("device_descriptor_policy"),
        "grouped_task_descriptor_valid": grouped_task_descriptor_valid(capture) if mode == "grouped_dispatch" else None,
        "same_backend_independent": capture_summary(same_backend_independent),
        "best_independent": capture_summary(best_independent),
        "same_backend_host_batch": capture_summary(same_backend_host_batch),
        "checksum": checksum_value(capture),
        "checksum_matches_same_backend_host_batch": checksum_matches(capture, same_backend_host_batch),
        "same_backend_host_batch_task_count_matches": (
            None
            if same_backend_host_batch is None
            else task_count_for_capture(capture) == task_count_for_capture(same_backend_host_batch)
        ),
        "speedup_vs_same_backend_independent": speedup(same_backend_independent, capture),
        "speedup_vs_best_independent": speedup(best_independent, capture),
        "speedup_vs_same_backend_host_batch": speedup(same_backend_host_batch, capture),
        "decision": decision,
        "blockers": blockers,
    }


def build_report_from_captures(captures: list[dict[str, Any]]) -> dict[str, Any]:
    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        by_contract[normalized_contract_key(capture)].append(capture)

    groups = []
    for key, items in sorted(by_contract.items(), key=lambda item: item[0]):
        independent = [capture for capture in items if mode_for_capture(capture) == "independent_call"]
        host_batches = [capture for capture in items if mode_for_capture(capture) == "host_api_batch"]
        grouped = [capture for capture in items if mode_for_capture(capture) == "grouped_dispatch"]
        best_independent = fastest(independent)
        best_host_batch = fastest(host_batches)
        best_grouped = fastest(grouped)
        rows = []
        for capture in sorted(items, key=lambda item: (mode_for_capture(item), backend_id(item), str(item.get("_path")))):
            backend = backend_id(capture)
            same_backend_independent = fastest([item for item in independent if backend_id(item) == backend])
            same_backend_host_batch = fastest([item for item in host_batches if backend_id(item) == backend])
            rows.append(row_for_capture(capture, same_backend_independent, best_independent, same_backend_host_batch))
        groups.append(
            {
                "contract_key": key,
                "summary": {
                    "independent_count": len(independent),
                    "host_batch_count": len(host_batches),
                    "grouped_dispatch_count": len(grouped),
                    "best_independent": capture_summary(best_independent),
                    "best_host_batch": capture_summary(best_host_batch),
                    "best_grouped_dispatch": capture_summary(best_grouped),
                },
                "rows": sorted(rows, key=lambda row: row.get("median_per_task_end_to_end_us") or float("inf")),
            }
        )

    comparisons = [row for group in groups for row in group["rows"] if row["mode"] == "grouped_dispatch"]
    return {
        "schema_version": 2,
        "policy": "many_small_grouped_evidence_only_no_public_resident_lifetime_api",
        "summary": {
            "groups": len(groups),
            "grouped_dispatch_comparisons": len(comparisons),
            "candidate_wins": sum(1 for row in comparisons if row["decision"] == "candidate_win"),
            "deprioritized": sum(1 for row in comparisons if row["decision"] == "deprioritize"),
            "experimental": sum(1 for row in comparisons if row["decision"] == "keep_experimental"),
            "missing_baselines": sum(1 for row in comparisons if row["decision"] == "missing_baseline"),
        },
        "groups": groups,
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    captures = [load_validated_capture(path) for path in expand_inputs(paths)]
    return build_report_from_captures(captures)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Many-Small Grouped Dispatch Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "| backend | semantics | shape | tasks | strategy | descriptor | grouped per-task us | best independent | same-backend host-batch us | host-batch task count match | host-batch checksum match | speedup vs best independent | speedup vs host-batch | decision | blockers |",
            "|---|---|---|---:|---|---|---:|---|---:|---|---|---:|---:|---|---|",
        ]
    )
    for group in report["groups"]:
        for row in group["rows"]:
            if row["mode"] != "grouped_dispatch":
                continue
            shape = row["shape"]
            best = row.get("best_independent") or {}
            host_batch = row.get("same_backend_host_batch") or {}
            best_text = (
                f"{best.get('backend')} {best.get('median_per_task_end_to_end_us')}"
                if best
                else "none"
            )
            blockers = ",".join(row.get("blockers") or []) or "none"
            lines.append(
                "| {backend} | {semantics} | {m}x{n}x{k} | {tasks} | {strategy} | {descriptor} | {per_task} | {best} | {host_batch} | {task_count_match} | {checksum_match} | {speed_best} | {speed_batch} | {decision} | {blockers} |".format(
                    backend=row.get("backend"),
                    semantics=row.get("semantics"),
                    m=shape.get("m"),
                    n=shape.get("n"),
                    k=shape.get("k"),
                    tasks=row.get("task_count"),
                    strategy=row.get("grouped_dispatch_execution_strategy"),
                    descriptor=row.get("grouped_task_descriptor_device_policy"),
                    per_task=row.get("median_per_task_end_to_end_us"),
                    best=best_text,
                    host_batch=host_batch.get("median_per_task_end_to_end_us"),
                    task_count_match=row.get("same_backend_host_batch_task_count_matches"),
                    checksum_match=row.get("checksum_matches_same_backend_host_batch"),
                    speed_best=(
                        None
                        if row.get("speedup_vs_best_independent") is None
                        else round(float(row["speedup_vs_best_independent"]), 4)
                    ),
                    speed_batch=(
                        None
                        if row.get("speedup_vs_same_backend_host_batch") is None
                        else round(float(row["speedup_vs_same_backend_host_batch"]), 4)
                    ),
                    decision=row.get("decision"),
                    blockers=blockers,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "many-small-grouped-report.json"
    md_path = out_dir / "many-small-grouped-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        action="append",
        help="capture file or directory; directories are searched recursively for JSON",
    )
    parser.add_argument("captures", type=Path, nargs="*")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = list(args.captures)
    if args.capture:
        paths.extend(args.capture)
    if not paths:
        raise SystemExit("at least one capture file or directory is required")
    report = build_report(paths)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.out_md)
    if not args.json and not args.out_json and not args.out_md:
        paths_written = write_report(report, args.out_dir)
        print(paths_written["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
