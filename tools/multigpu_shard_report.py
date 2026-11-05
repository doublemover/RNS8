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
from json_evidence import load_evidence_json
from compiled_binary_identity import canonical_digest
from benchmark_request import file_digest
from reference_verdict import numerical_contract, checksum_u64
from target_qualification import EvidenceStore
from device_observation import capture_observation, IdentityUnavailable, IdentityMismatch, RuntimeDeviceIdentity, distinct_runtime_identities
from multigpu_observation import validate_binding as validate_observation_binding


DEFAULT_OUT_DIR = Path("temp") / "multigpu-shard-report"
CRITICAL_EXIT_BLOCKERS = {
    "launch_inventory_unknown",
    "launch_inventory_invalid",
    "world_size_mismatch",
    "rank_device_mismatch",
    "duplicate_rank_status",
    "duplicate_rank_case",
    "missing_rank_status",
    "missing_cases",
    "checksum_mismatch",
    "env_summary_invalid_json",
    "env_summary_missing",
    "env_summary_unreadable",
    "failed_shards",
    "missing_shards",
    "schema_failed",
    "schema_log_failed",
    "checksum_missing",
    "target_status_missing",
    "capture_observation_unbound",
    "capture_observation_invalid",
    "observed_target_mismatch",
    "observed_device_mapping_mismatch",
    "duplicate_capture_association",
    "ambiguous_execution_device_mapping",
    "observation_receipt_invalid",
    "observation_sessions_ambiguous",
    "exact_case_identity_missing",
    "case_contract_mismatch",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = load_evidence_json(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_json_with_status(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        return {}, "not_provided"
    if not path.exists():
        return {}, "missing"
    try:
        value = load_evidence_json(path)
    except ValueError:
        return {}, "invalid_json"
    except OSError:
        return {}, "unreadable"
    return (value, "present") if isinstance(value, dict) else ({}, "invalid_json")


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
    observation: dict | None = None,
    observation_error: str | None = None,
) -> dict[str, Any]:
    schema_status, schema_error, capture = _schema_status(path)
    dry_run = capture.get("dry_run") is True
    path_device, path_rank = _parse_device_from_path(path), _parse_rank_from_path(path)
    device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
    runtime = capture.get("runtime_environment") if isinstance(capture.get("runtime_environment"), dict) else {}
    rank = observation["rank"] if observation else _as_int(capture.get("rank")) if dry_run else _as_int(runtime.get("RNS8_RANK"))
    if rank is None:
        rank = path_rank
    selector = observation["selected_parent_ordinal"] if observation else path_device
    record = status_by_device.get(str(selector)) if selector is not None else None
    record = record or (status_by_rank.get(rank) if rank is not None else None) or {}
    physical = physical_by_device.get(str(selector), {}) if selector is not None else {}
    world = observation["world_size"] if observation else _as_int(capture.get("world_size")) if dry_run else _as_int(runtime.get("RNS8_WORLD_SIZE"))
    if world is None:
        world = _as_int(record.get("world_size"))
    if selector is None:
        selector = record.get("device_index")
    expected_target = record.get("target_id") or physical.get("target_arch")
    expected_bdf = record.get("device_bdf") or physical.get("bdf")
    target = expected_target if dry_run else device.get("gcn_arch")
    name = record.get("gpu_name") or physical.get("device_name") if dry_run else device.get("name")
    observed = None
    observation_issue = observation_error
    if not dry_run and schema_status == "pass":
        try:
            observed = capture_observation(capture)
        except (IdentityUnavailable, IdentityMismatch) as error:
            observation_issue = observation_issue or str(error)
    bdf = observed.identity.pci_bdf if observed else None
    uuid = observed.identity.uuid if observed else None
    case = observation["case_id"] if observation else re.sub(r"-rank[0-9]+$", "", path.stem)
    exact_case = observation.get("exact_case_sha256") if observation else None
    exact_work = observation.get("exact_work") if observation else None
    if exact_case is None and not dry_run and schema_status == "pass":
        try:
            exact_work = dict(case_id=case, mathematical_contract=numerical_contract(capture),
                backend=capture.get("backend_selected"), selected_kernel=capture.get("selected_kernel"),
                build_manifest_sha256=canonical_digest(capture.get("build_manifest")))
            exact_case = canonical_digest(exact_work)
        except (ValueError, TypeError, KeyError):
            pass
    schema_log = path.with_name(f"benchmark-schema-rank{rank}.log") if rank is not None else path.with_suffix(".schema.log")
    blockers = []
    if schema_status not in {"pass", "dry_run"}:
        blockers.append("schema_failed")
    if not dry_run:
        if observation is None:
            blockers.append("capture_observation_invalid" if observation_error else "capture_observation_unbound")
        if exact_case is None:
            blockers.append("exact_case_identity_missing")
        for intended in (record, physical):
            intended_target = intended.get("target_id") or intended.get("target_arch")
            if intended_target and target and str(intended_target).split(":")[0] != str(target).split(":")[0]:
                blockers.append("observed_target_mismatch")
            intended_bdf = intended.get("device_bdf") or intended.get("bdf")
            if intended_bdf and bdf and str(intended_bdf).lower() != bdf:
                blockers.append("observed_device_mapping_mismatch")
            expected_uuid = intended.get("hip_uuid")
            if expected_uuid and uuid and expected_uuid != uuid:
                blockers.append("observed_device_mapping_mismatch")
        if observation and ((path_rank is not None and path_rank != rank) or
                            (path_device is not None and str(path_device) != str(selector))):
            blockers.append("observed_device_mapping_mismatch")
        if observation and record and _as_int(record.get("rank")) != rank:
            blockers.append("observed_device_mapping_mismatch")
        if observation and record and _as_int(record.get("world_size")) != world:
            blockers.append("world_size_mismatch")
    else:
        blockers.append("dry_run_command_plan_only")
    log_status = _schema_log_status(schema_log)
    if log_status == "fail":
        blockers.append("schema_log_failed")
    if not record:
        blockers.append("target_status_missing")
    elif (record.get("validation") or {}).get("smoke") not in {"pass", "planned"}:
        blockers.append("failed_shards")
    checksum = checksum_u64(capture)
    if checksum is None and schema_status == "pass":
        blockers.append("checksum_missing")
    if record.get("rocprofv3_ready") is False:
        blockers.append("rocprofv3_missing")
    if record.get("rccl_ready") is False:
        blockers.append("rccl_missing_future_platform_gate")
    if record.get("rccl_tests_ready") is False:
        blockers.append("rccl_tests_missing_future_platform_gate")
    binding_status = "verified" if observation else "planned" if dry_run else "unbound"
    if observation and set(blockers).intersection(CRITICAL_EXIT_BLOCKERS):
        binding_status = "contradictory"
    target_status = "planned" if dry_run and record else "pass" if binding_status == "verified" and record else "unqualified"
    return dict(capture=str(path), rank=rank, world_size=world, physical_device_id=selector,
        device_id_namespace="unmasked_parent_HIP_runtime_ordinal_selector",
        observed_runtime_ordinal=device.get("device_id") if not dry_run else None,
        target_arch=target, device_name=name, device_bdf=bdf, hip_uuid=uuid,
        expected_target_arch=expected_target, expected_device_name=record.get("gpu_name") or physical.get("device_name"),
        expected_device_bdf=expected_bdf, numa_node=record.get("numa_node", physical.get("numa_node")),
        multi_gpu_mode=record.get("multi_gpu_mode") or runtime.get("RNS8_MULTI_GPU_MODE") or "embarrassingly_parallel_shards",
        schema_status=schema_status, schema_error=schema_error, schema_log_status=log_status,
        target_validation_status=target_status,
        observation_binding_status=binding_status,
        observation_binding=observation, observation_error=observation_issue,
        median_end_to_end_us=_median_us(capture), checksum_u64=checksum,
        backend_selected=capture.get("backend_selected"), semantics=capture.get("semantics"),
        shape={key: capture.get(key) for key in ("m", "n", "k")}, seed=capture.get("seed"),
        finite_modulus=capture.get("finite_modulus"), case_id=case, exact_case_sha256=exact_case, exact_work=exact_work,
        execution_id=capture.get("execution_id"), rocprofv3_ready=record.get("rocprofv3_ready"),
        rccl_ready=record.get("rccl_ready"), rccl_tests_ready=record.get("rccl_tests_ready"),
        blockers=sorted(set(blockers)))


def _discover_captures(shards_dir: Path | None, explicit: list[Path]) -> list[Path]:
    paths = list(explicit)
    if shards_dir is not None and shards_dir.exists():
        paths.extend(
            path
            for path in sorted(shards_dir.glob("gpu*/*.json"))
            if not path.name.startswith("target-status") and not path.name.endswith(".binding.json")
        )
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def _launch_coverage(summary, status_paths, rows, records):
    """Expected ranks come from launcher intent, never surviving output ranks."""
    blockers = set()
    inventories = []
    requested = summary.get("requested_devices")
    if isinstance(requested, list) and requested:
        inventories.append([str(device) for device in requested])
    expected_cases = set()
    declared_worlds = []
    if "world_size" in summary:
        declared_worlds.append(_as_int(summary["world_size"]))
    for path in status_paths:
        inventory = _load_json(path).get("launch_inventory")
        if isinstance(inventory, dict):
            declared_worlds.append(_as_int(inventory.get("world_size")))
            devices = inventory.get("requested_devices")
            if isinstance(devices, list) and devices:
                inventories.append([str(device) for device in devices])
            cases = inventory.get("cases")
            if isinstance(cases, list):
                expected_cases.update(str(case) for case in cases)
    if not inventories:
        return [], [], [], {"launch_inventory_unknown"}
    devices = inventories[0]
    if len(set(devices)) != len(devices) or any(value != devices for value in inventories):
        blockers.add("launch_inventory_invalid")
    expected = set(range(len(devices)))
    worlds = declared_worlds + [_as_int(item.get("world_size")) for item in records + rows]
    if any(world != len(devices) for world in worlds):
        blockers.add("world_size_mismatch")
    status_ranks = set()
    for record in records:
        rank = _as_int(record.get("rank"))
        if rank in status_ranks:
            blockers.add("duplicate_rank_status")
        status_ranks.add(rank)
        if rank not in expected or str(record.get("device_index")) != devices[rank]:
            blockers.add("rank_device_mismatch")
    if expected - status_ranks:
        blockers.add("missing_rank_status")
    observed = set()
    rank_cases = set()
    for row in rows:
        rank = _as_int(row.get("rank"))
        observed.add(rank)
        if rank not in expected or str(row.get("physical_device_id")) != devices[rank]:
            blockers.add("rank_device_mismatch")
        case = row.get("case_id") or re.sub(r"-rank[0-9]+$", "", Path(row["capture"]).stem)
        key = (rank, case)
        if key in rank_cases:
            blockers.add("duplicate_rank_case")
        rank_cases.add(key)
    if not expected_cases:
        expected_cases = {case for _, case in rank_cases}
    missing_cases = [{"rank": rank, "case": case} for rank in sorted(expected)
                     for case in sorted(expected_cases) if (rank, case) not in rank_cases]
    if missing_cases:
        blockers.add("missing_cases")
    return devices, sorted(expected - observed), missing_cases, blockers


def _checksum_groups(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    groups: dict[str, set[Any]] = defaultdict(set)
    for row in rows:
        checksum = row.get("checksum_u64")
        if checksum is None:
            continue
        exact = row.get("exact_case_sha256")
        if not exact:
            continue
        key = "case=" + str(row.get("case_id")) + ";exact_work=" + exact
        groups[key].add(checksum)
    return {key: sorted(values) for key, values in sorted(groups.items())}


def _case_timing_outliers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _timing_outliers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        if row.get("exact_case_sha256"):
            groups[row["exact_case_sha256"]].append(row)
    result = []
    for exact, group in groups.items():
        result.extend({**item, "exact_case_sha256": exact, "case_id": group[0]["case_id"]}
                      for item in _case_timing_outliers(group))
    return result


def _load_observations(paths: list[Path], store: EvidenceStore) -> tuple[dict, dict, list]:
    observations, per_capture_errors, errors = {}, {}, []
    for path in dict.fromkeys(path.resolve() for path in paths):
        capture_path = None
        try:
            _, value = store.json(dict(path=str(path), sha256=file_digest(path)), path.parent)
            candidate = value.get("candidate")
            if not isinstance(candidate, dict) or not isinstance(candidate.get("capture"), str):
                raise ValueError("observation receipt has no raw capture reference")
            capture_path = Path(candidate["capture"]).resolve(strict=True)
            facts = validate_observation_binding(value, capture_path, store=store)
            if capture_path in per_capture_errors:
                raise ValueError("another supplied association for this capture is already invalid")
            previous = observations.get(capture_path)
            if previous is not None and canonical_digest(previous) != canonical_digest(facts):
                raise ValueError("conflicting observation associations for one capture")
            observations[capture_path] = facts
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(dict(path=str(path), reason=str(error)))
            if capture_path is not None:
                per_capture_errors[capture_path] = str(error)
                observations.pop(capture_path, None)
    return observations, per_capture_errors, errors


def _observed_launch(observations: dict) -> tuple[dict, list, list]:
    if not observations:
        return {}, [], []
    facts = list(observations.values())
    first = facts[0]
    contexts = {(item["session_id"], item["run_id"], tuple(item["requested_devices"]),
                 tuple(item["expected_cases"])) for item in facts}
    blockers = [] if len(contexts) == 1 else ["observation_sessions_ambiguous"]
    devices, cases = first["requested_devices"], set(first["expected_cases"])
    records = []
    for rank, device in enumerate(devices):
        found = [item for item in facts if item["rank"] == rank]
        if not found:
            continue
        actual = found[0]["observation"]["observed"]
        records.append(dict(rank=rank, world_size=len(devices), device_index=device,
            target_id=actual["architecture"], gpu_name=actual["name"],
            device_bdf=actual["identity"]["pci_bdf"], hip_uuid=actual["identity"]["uuid"],
            multi_gpu_mode="embarrassingly_parallel_shards",
            validation=dict(smoke="pass" if cases <= {item["case_id"] for item in found} else "missing"),
            source="verified_runtime_observation_receipts"))
    return dict(requested_devices=devices, world_size=len(devices), cases=sorted(cases)), records, blockers


def _observation_conflicts(rows: list[dict]) -> set[str]:
    blockers, executions, identities, cases = set(), {}, {}, {}
    for row in rows:
        if row["schema_status"] == "dry_run":
            continue
        execution = row.get("execution_id")
        if isinstance(execution, str) and execution:
            if execution in executions:
                blockers.add("duplicate_capture_association")
            executions[execution] = row["capture"]
        facts = row.get("observation_binding")
        if not facts or row.get("observation_binding_status") != "verified":
            continue
        # A case label cannot silently change work on another rank. Different
        # seeds/moduli are separate cases, not separate checksum cohorts of one.
        case, exact = facts["case_id"], facts["exact_case_sha256"]
        if case in cases and cases[case] != exact:
            blockers.add("case_contract_mismatch")
        cases[case] = exact
        selected = facts["selected_parent_ordinal"]
        identity = RuntimeDeviceIdentity(**facts["observation"]["observed"]["identity"])
        previous = identities.get(selected)
        if previous is not None and not (
                previous.uuid == identity.uuid and previous.pci_bdf == identity.pci_bdf):
            blockers.add("ambiguous_execution_device_mapping")
        identities[selected] = identity
    values = list(identities.items())
    for index, (selector, identity) in enumerate(values):
        for other_selector, other in values[index+1:]:
            if selector != other_selector and not distinct_runtime_identities(identity, other):
                blockers.add("ambiguous_execution_device_mapping")
    return blockers


def build_report(
    captures: list[Path],
    env_summary: Path | None = None,
    target_status: list[Path] | None = None,
    shards_dir: Path | None = None,
    observation_receipts: list[Path] | None = None,
) -> dict[str, Any]:
    store = EvidenceStore()
    observations, observation_errors, receipt_errors = _load_observations(observation_receipts or [], store)
    observed_intent, observed_records, observation_blockers = _observed_launch(observations)
    summary, env_summary_status = _load_json_with_status(env_summary)
    records = _status_records(target_status or [])
    if observed_intent and not summary:
        summary = dict(observed_intent, source="verified_session_launch_intent")
    elif observed_intent and summary.get("requested_devices") is not None:
        if (not isinstance(summary["requested_devices"], list) or
                list(map(str, summary["requested_devices"])) != list(map(str, observed_intent["requested_devices"]))):
            observation_blockers.append("launch_inventory_invalid")
    if not records:
        records = observed_records
    status_by_device, status_by_rank = _target_lookup(records)
    physical_by_device = _env_physical_lookup(summary)
    capture_paths = _discover_captures(shards_dir, [*captures, *observations, *observation_errors])
    rows = [
        _row_for_capture(path, status_by_device, status_by_rank, physical_by_device,
                         observations.get(path.resolve()), observation_errors.get(path.resolve()))
        for path in capture_paths
    ]
    try:
        store.finish()
    except (OSError, ValueError) as error:
        receipt_errors.append(dict(reason=str(error)))
        for row in rows:
            if row["schema_status"] != "dry_run":
                row["blockers"] = sorted(set(row["blockers"] + ["capture_observation_invalid"]))
                row["observation_binding_status"] = "invalid"
                row["target_validation_status"] = "unqualified"
    checksum_groups = _checksum_groups(rows)
    checksum_mismatch_groups = {key: values for key, values in checksum_groups.items() if len(values) > 1}
    expected_devices, missing_ranks, missing_cases, coverage_blockers = _launch_coverage(
        summary, target_status or [], rows, records)
    if observed_intent:
        actual_cases = {(row["rank"], row["case_id"]) for row in rows}
        for rank in range(len(observed_intent["requested_devices"])):
            for case in observed_intent["cases"]:
                if (rank, case) not in actual_cases and dict(rank=rank, case=case) not in missing_cases:
                    missing_cases.append(dict(rank=rank, case=case))
                    coverage_blockers.add("missing_cases")
    failed_ranks = sorted(
        {
            int(row["rank"])
            for row in rows
            if row.get("rank") is not None
            and (row.get("schema_status") == "fail" or row.get("schema_log_status") == "fail")
        }
    )
    timing_outliers = _timing_outliers(rows)
    conflicts = _observation_conflicts(rows)
    blockers = set(coverage_blockers) | set(observation_blockers) | conflicts
    if conflicts:
        for row in rows:
            if row["schema_status"] != "dry_run" and row["observation_binding_status"] == "verified":
                row["observation_binding_status"] = "contradictory"
                row["target_validation_status"] = "unqualified"
                row["blockers"] = sorted(set(row["blockers"]) | conflicts)
    if receipt_errors:
        blockers.add("observation_receipt_invalid")
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
    if env_summary_status in {"missing", "invalid_json", "unreadable"}:
        blockers.add(f"env_summary_{env_summary_status}")
    if rows:
        blockers.add("multi_gpu_smoke_not_release_reviewed")
    return {
        "schema": "rns8_multigpu_shard_report_v2",
        "observation_receipt_errors": receipt_errors,
        "observation_receipts": [str(path) for path in observation_receipts or []],
        "qualified_observation_count": sum(row["observation_binding_status"] == "verified" for row in rows),
        "unqualified_ranks": sorted({row["rank"] for row in rows if row["rank"] is not None and
                                    row["schema_status"] != "dry_run" and row["observation_binding_status"] != "verified"}),
        "capture_count": len(capture_paths),
        "status_record_count": len(records),
        "env_summary": str(env_summary) if env_summary else None,
        "target_status": [str(path) for path in target_status or []],
        "env_summary_status": env_summary_status,
        "world_sizes": sorted({str(row.get("world_size")) for row in rows if row.get("world_size") is not None}),
        "observed_ranks": sorted({int(row["rank"]) for row in rows if row.get("rank") is not None}),
        "expected_devices": expected_devices,
        "execution_status": "not_executed" if rows and all(row["schema_status"] == "dry_run" for row in rows) else "executed_or_incomplete",
        "missing_cases": missing_cases,
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
        "Parent ordinals are selectors in the launcher's unmasked HIP namespace; observed identity comes from bound runtime records.",
        "",
        "| Rank | Parent ordinal | Runtime ordinal | Case | Observed target | Device name | BDF | Binding | Schema | Median us | Checksum | Blockers |",
        "|---:|---:|---:|---|---|---|---|---|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        blockers = ", ".join(row.get("blockers", []))
        def cell(value: Any) -> str:
            return str(value if value is not None else "").replace("&", "&amp;").replace(
                "<", "&lt;").replace(">", "&gt;").replace("|", "&#124;").replace(
                "\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
        lines.append(
            "| {rank} | {physical} | {logical} | {case} | {target} | {name} | {bdf} | {binding} | {schema} | {median} | {checksum} | {blockers} |".format(
                rank=cell(row.get("rank")),
                physical=cell(row.get("physical_device_id")),
                logical=cell(row.get("observed_runtime_ordinal")),
                case=cell(row.get("case_id")),
                binding=cell(row.get("observation_binding_status")),
                target=cell(row.get("target_arch")),
                name=cell(row.get("device_name")),
                bdf=cell(row.get("device_bdf")),
                schema=cell(row.get("schema_status")),
                median=cell(row.get("median_end_to_end_us")),
                checksum=cell(row.get("checksum_u64")),
                blockers=cell(blockers),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def report_has_critical_failures(report: dict[str, Any]) -> bool:
    blockers = {str(item) for item in report.get("promotion_blockers") or []}
    return bool(blockers.intersection(CRITICAL_EXIT_BLOCKERS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path)
    parser.add_argument("--env-summary", type=Path)
    parser.add_argument("--target-status", action="append", type=Path, default=[])
    parser.add_argument("--shards-dir", type=Path)
    parser.add_argument("--observation-receipt", action="append", type=Path, default=[])
    parser.add_argument("--observation-dir", type=Path, help="directory emitted by multigpu_observation.py")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.captures,
        env_summary=args.env_summary,
        target_status=args.target_status,
        shards_dir=args.shards_dir,
        observation_receipts=[*args.observation_receipt,
            *(sorted(args.observation_dir.glob("*.binding.json")) if args.observation_dir else [])],
    )
    outputs = write_outputs(report, args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(outputs["markdown"])
    return 1 if report_has_critical_failures(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
