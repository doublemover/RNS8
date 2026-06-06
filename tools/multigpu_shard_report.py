#!/usr/bin/env python3
"""Aggregate independent CDNA multi-GPU shard smoke outputs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import BenchmarkSchemaError, load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "multigpu-shard-report"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _status_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        data = _load_json(path)
        raw_records = data.get("records")
        if isinstance(raw_records, list):
            records.extend(item for item in raw_records if isinstance(item, dict))
        elif data:
            records.append(data)
    return records


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip(), 10)
    return None


def _parse_rank_from_path(path: Path) -> int | None:
    match = re.search(r"rank([0-9]+)", path.stem)
    return int(match.group(1), 10) if match else None


def _parse_device_from_path(path: Path) -> int | str | None:
    for part in reversed(path.parts):
        match = re.fullmatch(r"gpu(.+)", part)
        if match:
            value = match.group(1)
            return int(value, 10) if re.fullmatch(r"[0-9]+", value) else value
    return None


def _median_us(capture: dict[str, Any], phase: str = "end_to_end") -> float | None:
    timing = capture.get("timing_summary_us")
    if not isinstance(timing, dict):
        return None
    phase_data = timing.get(phase)
    if isinstance(phase_data, dict) and isinstance(phase_data.get("median"), (int, float)):
        return float(phase_data["median"])
    if isinstance(phase_data, (int, float)):
        return float(phase_data)
    return None


def _schema_status(path: Path) -> tuple[str, str | None, dict[str, Any]]:
    data = _load_json(path)
    if data.get("dry_run") is True:
        return "dry_run", None, data
    try:
        if not data:
            data = load_capture(path)
        validate_capture(data, path)
    except (BenchmarkSchemaError, OSError, json.JSONDecodeError) as exc:
        return "fail", str(exc), data
    return "pass", None, data


def _schema_log_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="replace")
    if "dry-run" in text.lower():
        return "dry_run"
    lowered = text.lower()
    failure_markers = ["traceback", "error", "invalid", "failed", "exception"]
    return "fail" if any(marker in lowered for marker in failure_markers) else "pass"


def _record_key(record: dict[str, Any]) -> tuple[int | str | None, int | None]:
    return record.get("device_index"), _as_int(record.get("rank"))


def _target_lookup(records: list[dict[str, Any]]) -> tuple[dict[Any, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_device: dict[Any, dict[str, Any]] = {}
    by_rank: dict[int, dict[str, Any]] = {}
    for record in records:
        device, rank = _record_key(record)
        if device is not None:
            by_device[str(device)] = record
        if rank is not None:
            by_rank[rank] = record
    return by_device, by_rank


def _env_physical_lookup(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    devices = summary.get("physical_devices")
    if not isinstance(devices, list):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for item in devices:
        if not isinstance(item, dict):
            continue
        value = item.get("physical_device_id")
        if value is not None:
            lookup[str(value)] = item
    return lookup


def _row_for_capture(
    path: Path,
    status_by_device: dict[Any, dict[str, Any]],
    status_by_rank: dict[int, dict[str, Any]],
    physical_by_device: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    schema_status, schema_error, capture = _schema_status(path)
    path_device = _parse_device_from_path(path)
    path_rank = _parse_rank_from_path(path)
    capture_device = capture.get("device") if capture.get("dry_run") is True else None
    if isinstance(capture_device, dict):
        capture_device = capture_device.get("device_id")
    rank = _as_int(capture.get("rank")) if capture.get("dry_run") is True else path_rank
    world_size = _as_int(capture.get("world_size")) if capture.get("dry_run") is True else None
    physical_device_id = path_device if path_device is not None else capture_device
    record = None
    if physical_device_id is not None:
        record = status_by_device.get(str(physical_device_id))
    if record is None and rank is not None:
        record = status_by_rank.get(rank)
    record = record or {}
    physical = physical_by_device.get(str(physical_device_id)) if physical_device_id is not None else None
    physical = physical or {}
    if world_size is None:
        world_size = _as_int(record.get("world_size"))
    target_arch = record.get("target_id") or physical.get("target_arch")
    device_name = record.get("gpu_name") or physical.get("device_name")
    device_bdf = record.get("device_bdf") or physical.get("bdf")
    numa_node = record.get("numa_node") if record.get("numa_node") is not None else physical.get("numa_node")
    schema_log = path.with_name(f"benchmark-schema-rank{rank}.log") if rank is not None else path.with_suffix(".schema.log")
    row = {
        "capture": str(path),
        "rank": rank,
        "world_size": world_size,
        "physical_device_id": physical_device_id,
        "target_arch": target_arch,
        "device_name": device_name,
        "device_bdf": device_bdf,
        "numa_node": numa_node,
        "multi_gpu_mode": record.get("multi_gpu_mode") or capture.get("multi_gpu_mode") or "embarrassingly_parallel_shards",
        "schema_status": schema_status,
        "schema_error": schema_error,
        "schema_log_status": _schema_log_status(schema_log),
        "target_validation_status": "pass" if record else "missing",
        "median_end_to_end_us": _median_us(capture),
        "checksum_u64": capture.get("checksum_u64"),
        "backend_selected": capture.get("backend_selected"),
        "semantics": capture.get("semantics"),
        "shape": {
            "m": capture.get("m"),
            "n": capture.get("n"),
            "k": capture.get("k"),
        },
        "rocprofv3_ready": record.get("rocprofv3_ready"),
        "rccl_ready": record.get("rccl_ready"),
        "rccl_tests_ready": record.get("rccl_tests_ready"),
    }
    blockers: list[str] = []
    if schema_status not in {"pass", "dry_run"}:
        blockers.append("schema_failed")
    if row["schema_log_status"] == "fail":
        blockers.append("schema_log_failed")
    if not record:
        blockers.append("target_status_missing")
    if row["checksum_u64"] is None and schema_status == "pass":
        blockers.append("checksum_missing")
    if record.get("rocprofv3_ready") is False:
        blockers.append("rocprofv3_missing")
    if record.get("rccl_ready") is False:
        blockers.append("rccl_missing_future_platform_gate")
    if record.get("rccl_tests_ready") is False:
        blockers.append("rccl_tests_missing_future_platform_gate")
    if schema_status == "dry_run":
        blockers.append("dry_run_command_plan_only")
    row["blockers"] = sorted(set(blockers))
    return row


def _discover_captures(shards_dir: Path | None, explicit: list[Path]) -> list[Path]:
    paths = list(explicit)
    if shards_dir is not None and shards_dir.exists():
        paths.extend(
            path
            for path in sorted(shards_dir.glob("gpu*/*.json"))
            if not path.name.startswith("target-status")
        )
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def _missing_ranks(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[int]:
    expected = {_as_int(record.get("rank")) for record in records}
    expected.discard(None)
    observed = {_as_int(row.get("rank")) for row in rows}
    observed.discard(None)
    if not expected:
        world_sizes = {_as_int(row.get("world_size")) for row in rows}
        world_sizes.discard(None)
        if len(world_sizes) == 1:
            expected = set(range(next(iter(world_sizes))))
    return sorted(int(rank) for rank in expected if rank not in observed)


def _checksum_groups(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    groups: dict[str, set[Any]] = defaultdict(set)
    for row in rows:
        checksum = row.get("checksum_u64")
        if checksum is None:
            continue
        key = (
            f"semantics={row.get('semantics')};backend={row.get('backend_selected')};"
            f"shape={row.get('shape', {}).get('m')}x{row.get('shape', {}).get('n')}x{row.get('shape', {}).get('k')}"
        )
        groups[key].add(checksum)
    return {key: sorted(values) for key, values in sorted(groups.items())}


def _timing_outliers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [float(row["median_end_to_end_us"]) for row in rows if isinstance(row.get("median_end_to_end_us"), (int, float))]
    if len(valid) < 3:
        return []
    median = statistics.median(valid)
    threshold = median * 1.25
    return [
        {
            "rank": row.get("rank"),
            "physical_device_id": row.get("physical_device_id"),
            "median_end_to_end_us": row.get("median_end_to_end_us"),
            "threshold_us": threshold,
        }
        for row in rows
        if isinstance(row.get("median_end_to_end_us"), (int, float)) and float(row["median_end_to_end_us"]) > threshold
    ]


def build_report(
    captures: list[Path],
    env_summary: Path | None = None,
    target_status: list[Path] | None = None,
    shards_dir: Path | None = None,
) -> dict[str, Any]:
    summary = _load_json(env_summary) if env_summary is not None else {}
    records = _status_records(target_status or [])
    status_by_device, status_by_rank = _target_lookup(records)
    physical_by_device = _env_physical_lookup(summary)
    capture_paths = _discover_captures(shards_dir, captures)
    rows = [
        _row_for_capture(path, status_by_device, status_by_rank, physical_by_device)
        for path in capture_paths
    ]
    checksum_groups = _checksum_groups(rows)
    checksum_mismatch_groups = {key: values for key, values in checksum_groups.items() if len(values) > 1}
    missing_ranks = _missing_ranks(rows, records)
    failed_ranks = sorted(
        {
            int(row["rank"])
            for row in rows
            if row.get("rank") is not None
            and (row.get("schema_status") == "fail" or row.get("schema_log_status") == "fail")
        }
    )
    timing_outliers = _timing_outliers(rows)
    blockers = set()
    if missing_ranks:
        blockers.add("missing_shards")
    if failed_ranks:
        blockers.add("failed_shards")
    if checksum_mismatch_groups:
        blockers.add("checksum_mismatch")
    if timing_outliers:
        blockers.add("timing_outliers")
    for row in rows:
        blockers.update(row.get("blockers", []))
    if rows:
        blockers.add("multi_gpu_smoke_not_release_reviewed")
    return {
        "schema": "rns8_multigpu_shard_report_v1",
        "capture_count": len(capture_paths),
        "status_record_count": len(records),
        "env_summary": str(env_summary) if env_summary else None,
        "target_status": [str(path) for path in target_status or []],
        "world_sizes": sorted({str(row.get("world_size")) for row in rows if row.get("world_size") is not None}),
        "observed_ranks": sorted({int(row["rank"]) for row in rows if row.get("rank") is not None}),
        "missing_ranks": missing_ranks,
        "failed_ranks": failed_ranks,
        "checksum_groups": checksum_groups,
        "checksum_mismatch_groups": checksum_mismatch_groups,
        "timing_outliers": timing_outliers,
        "rocprofv3_ready_values": sorted({str(row.get("rocprofv3_ready")).lower() for row in rows}),
        "rccl_ready_values": sorted({str(row.get("rccl_ready")).lower() for row in rows}),
        "rccl_tests_ready_values": sorted({str(row.get("rccl_tests_ready")).lower() for row in rows}),
        "promotion_eligible": False,
        "promotion_blockers": sorted(blockers),
        "rows": rows,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "multigpu-shard-report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = out_dir / "multigpu-shard-report.md"
    lines = [
        "# RNS8 Multi-GPU Shard Report",
        "",
        f"- Captures: {report['capture_count']}",
        f"- Status records: {report['status_record_count']}",
        f"- Promotion eligible: {report['promotion_eligible']}",
        f"- Blockers: {', '.join(report['promotion_blockers']) if report['promotion_blockers'] else 'none'}",
        "",
        "| Rank | Physical Device | Target | Device Name | BDF | NUMA | Schema | Median us | Checksum | Blockers |",
        "|---:|---:|---|---|---|---:|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        blockers = ", ".join(row.get("blockers", []))
        lines.append(
            "| {rank} | {physical} | {target} | {name} | {bdf} | {numa} | {schema} | {median} | {checksum} | {blockers} |".format(
                rank=row.get("rank", ""),
                physical=row.get("physical_device_id", ""),
                target=row.get("target_arch") or "",
                name=row.get("device_name") or "",
                bdf=row.get("device_bdf") or "",
                numa=row.get("numa_node") if row.get("numa_node") is not None else "",
                schema=row.get("schema_status") or "",
                median=row.get("median_end_to_end_us") if row.get("median_end_to_end_us") is not None else "",
                checksum=row.get("checksum_u64") if row.get("checksum_u64") is not None else "",
                blockers=blockers,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path)
    parser.add_argument("--env-summary", type=Path)
    parser.add_argument("--target-status", action="append", type=Path, default=[])
    parser.add_argument("--shards-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.captures,
        env_summary=args.env_summary,
        target_status=args.target_status,
        shards_dir=args.shards_dir,
    )
    outputs = write_outputs(report, args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(outputs["markdown"])
    return 0 if not report["failed_ranks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
