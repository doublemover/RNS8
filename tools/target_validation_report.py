#!/usr/bin/env python3
"""Summarize target validation without cross-target inference."""

from __future__ import annotations

import argparse
import json
import platform
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture
from check_dependencies_lib.config import LINUX_CDNA_TARGETS, LINUX_RDNA_TARGETS, SUPPORTED_TARGETS


DEFAULT_OUT_DIR = Path("temp") / "target-validation-reports"
READY_STATES = {"pass", "passed", "ok", "available", "captured", "complete", "success", True}
NOT_READY_STATES = {"fail", "failed", "missing", "unavailable", "not_run", "not-requested", "unknown", False, None}
VALIDATION_PHASES = ("build", "ctest", "smoke", "release_capture", "profiler")
TARGET_POLICY = "target_validation_is_per_os_gpu_toolchain_group_only"
NO_CROSS_TARGET_POLICY = "windows_rdna_cdna_evidence_groups_are_not_interchangeable"


def _clean_string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _target_id(value: Any) -> str | None:
    text = _clean_string(value)
    if not text:
        return None
    match = re.search(r"gfx[0-9a-fA-F]+", text)
    return match.group(0).lower() if match else text.lower()


def target_family(target_id: str | None) -> dict[str, Any]:
    target = _target_id(target_id)
    metadata = SUPPORTED_TARGETS.get(target or "", {})
    family = metadata.get("family")
    if target in LINUX_CDNA_TARGETS:
        target_class = "cdna"
    elif target in LINUX_RDNA_TARGETS or (family and str(family).startswith("RDNA")):
        target_class = "rdna"
    elif target:
        target_class = "unknown_gpu"
    else:
        target_class = "unknown"
    return {
        "target_id": target,
        "target_family": family or target_class.upper(),
        "target_class": target_class,
        "target_tier": metadata.get("tier"),
        "target_role": metadata.get("role"),
        "supported_target": target in SUPPORTED_TARGETS,
    }


def infer_host_os(capture: dict[str, Any] | None = None, record: dict[str, Any] | None = None) -> str:
    if record is not None:
        for key in ("host_os", "os", "platform"):
            value = _clean_string(record.get(key))
            if value:
                lowered = value.lower()
                if "windows" in lowered or lowered in {"win32", "win64", "nt"}:
                    return "windows"
                if "linux" in lowered or "ubuntu" in lowered or "debian" in lowered:
                    return "linux"
                return lowered
    if capture is not None:
        compiler = capture.get("compiler") if isinstance(capture.get("compiler"), dict) else {}
        compiler_id = str(compiler.get("id") or "").lower()
        toolchain = capture.get("hip_toolchain") if isinstance(capture.get("hip_toolchain"), dict) else {}
        hip_root = str(toolchain.get("hip_root") or toolchain.get("hipcc_path") or "")
        if compiler_id in {"msvc", "clang-cl"} or re.match(r"^[A-Za-z]:[\\/]", hip_root) or hip_root.endswith(".exe"):
            return "windows"
        if hip_root.startswith("/") or compiler_id in {"gcc", "clang"}:
            return "linux"
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system == "linux":
        return "linux"
    return system or "unknown"


def evidence_group_key(host_os: str, target_id: str | None, version: str | None) -> str:
    return f"os={host_os};target={target_id or 'unknown'};toolchain={version or 'unknown'}"


def target_cache_key(target_id: str | None, device_name: Any, version: str | None, hip_runtime_version: Any) -> str:
    return (
        f"arch={target_id or 'unknown'};"
        f"device_name={_clean_string(device_name) or 'unknown'};"
        f"hip_sdk_or_rocm={version or 'unknown'};"
        f"hip_runtime={hip_runtime_version if hip_runtime_version is not None else 'unknown'}"
    )


def target_instance_id(cache_key: str, device_index: Any, visibility: Any = None) -> str:
    visibility_text = _clean_string(visibility) or "runtime-default-visible-devices"
    index_text = str(device_index) if device_index is not None else "unknown"
    return f"{cache_key};device_index={index_text};visibility={visibility_text}"


def _status_ready(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in READY_STATES
    return value in READY_STATES


def _status_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    if value in READY_STATES:
        return "pass"
    if value in NOT_READY_STATES:
        return "missing"
    return str(value).strip().lower() if value is not None else "missing"


def _phase_status(record: dict[str, Any], phase: str) -> str:
    evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
    validation = record.get("validation") if isinstance(record.get("validation"), dict) else {}
    for container in (evidence, validation, record):
        if phase in container:
            return _status_text(container.get(phase))
        status_key = f"{phase}_status"
        if status_key in container:
            return _status_text(container.get(status_key))
    return "missing"


def _phase_ready(status: str) -> bool:
    return status in {"pass", "passed", "ok", "available", "captured", "complete", "success"}


def _accelerators_from_capture(capture: dict[str, Any]) -> dict[str, str]:
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    backend = capture.get("backend_selected")
    accelerators: dict[str, str] = {}
    if isinstance(backend, str) and backend in {"hipblaslt", "ck", "rocwmma"}:
        status = "available" if metadata.get("compiled_kernel_available") is True else "capture_present"
        accelerators[backend] = status
    library = metadata.get("accelerator_library")
    if isinstance(library, str) and library:
        accelerators.setdefault(library.lower(), "capture_present")
    return accelerators


def _accelerators_from_status(record: dict[str, Any]) -> dict[str, str]:
    value = record.get("accelerators") or record.get("accelerator_status")
    if not isinstance(value, dict):
        return {}
    return {str(key).lower(): _status_text(item) for key, item in value.items()}


def _cache_eligibility_from_capture(capture: dict[str, Any], target: dict[str, Any], host_os: str) -> dict[str, Any]:
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    blockers: list[str] = []
    if not target.get("target_id"):
        blockers.append("missing_target_id")
    if host_os == "unknown":
        blockers.append("missing_host_os")
    if metadata.get("performance_validated") is not True:
        blockers.append("performance_not_validated")
    if capture.get("comparison_baseline", {}).get("status") != "reviewed_release_same_contract_baseline":
        blockers.append("missing_reviewed_release_baseline")
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "scope": "matching_os_target_toolchain_group_only",
    }


def capture_target(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
    target = capture.get("target_variant") if isinstance(capture.get("target_variant"), dict) else {}
    toolchain = capture.get("hip_toolchain") if isinstance(capture.get("hip_toolchain"), dict) else {}
    metadata = capture.get("backend_metadata") if isinstance(capture.get("backend_metadata"), dict) else {}
    target_id = _target_id(target.get("target_id") or device.get("gcn_arch"))
    family = target_family(target_id)
    host_os = infer_host_os(capture=capture)
    version = _clean_string(toolchain.get("hip_sdk_or_rocm_version") or metadata.get("accelerator_version"))
    group_key = evidence_group_key(host_os, target_id, version)
    device_index = target.get("device_index", device.get("device_index", device.get("device_id")))
    cache_key = _clean_string(target.get("target_cache_key")) or target_cache_key(
        target_id,
        target.get("device_name") or device.get("device_name") or device.get("name"),
        version,
        device.get("hip_runtime_version"),
    )
    instance_id = _clean_string(target.get("target_instance_id")) or target_instance_id(
        cache_key,
        device_index,
        target.get("review_group_key"),
    )
    return {
        "source_kind": "capture",
        "path": str(path),
        "backend_selected": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "semantics": capture.get("semantics"),
        "target_id": target_id,
        "target_namespace": target.get("target_namespace"),
        "review_group_key": target.get("review_group_key"),
        "target_validation_group": group_key,
        "host_os": host_os,
        **family,
        "device_index": device_index,
        "device_name": device.get("name"),
        "visible_device_count": target.get("visible_device_count", device.get("visible_device_count")),
        "node_gpu_count": target.get("node_gpu_count", device.get("node_gpu_count")),
        "target_cache_key": cache_key,
        "target_instance_id": instance_id,
        "global_mem_bytes": device.get("global_mem_bytes"),
        "hip_runtime_version": device.get("hip_runtime_version"),
        "hip_driver_version": device.get("hip_driver_version"),
        "hip_sdk_or_rocm_version": version,
        "accelerator_library": metadata.get("accelerator_library"),
        "accelerator_version": metadata.get("accelerator_version"),
        "configured_amdgpu_targets": capture.get("configured_amdgpu_targets"),
        "accelerator_status": _accelerators_from_capture(capture),
        "validation_phases": {
            "build": "missing",
            "ctest": "missing",
            "smoke": "missing",
            "release_capture": "pass",
            "profiler": "missing",
        },
        "cache_eligibility": _cache_eligibility_from_capture(capture, family, host_os),
        "cross_target_promotion_allowed": False,
        "validated_scope": "this_os_target_toolchain_group_only",
    }


def _status_records_from_json(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("records", "targets", "groups", "entries"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [raw]
    raise RuntimeError(f"unsupported target status JSON shape: {path}")


def status_target(path: Path, record: dict[str, Any], index: int) -> dict[str, Any]:
    target_id = _target_id(record.get("target_id") or record.get("gcn_arch") or record.get("target"))
    family = target_family(target_id)
    host_os = infer_host_os(record=record)
    toolchain = record.get("hip_toolchain") if isinstance(record.get("hip_toolchain"), dict) else {}
    version = _clean_string(
        record.get("hip_sdk_or_rocm_version")
        or record.get("rocm_version")
        or toolchain.get("hip_sdk_or_rocm_version")
        or toolchain.get("rocm_version")
    )
    phases = {phase: _phase_status(record, phase) for phase in VALIDATION_PHASES}
    cache = record.get("cache_eligibility") if isinstance(record.get("cache_eligibility"), dict) else {}
    blockers = [str(item) for item in cache.get("blockers", []) if isinstance(item, str)]
    device_index = record.get("device_index")
    cache_key = _clean_string(record.get("target_cache_key")) or target_cache_key(
        target_id,
        record.get("device_name") or record.get("gpu_name"),
        version,
        record.get("hip_runtime_version"),
    )
    if cache.get("eligible") is not True:
        blockers.append("status_record_not_cache_eligible")
    return {
        "source_kind": "target_status",
        "path": str(path),
        "record_index": index,
        "target_id": target_id,
        "target_validation_group": evidence_group_key(host_os, target_id, version),
        "host_os": host_os,
        **family,
        "device_index": device_index,
        "device_name": record.get("device_name") or record.get("gpu_name"),
        "visible_device_count": record.get("visible_device_count"),
        "node_gpu_count": record.get("node_gpu_count"),
        "target_cache_key": cache_key,
        "target_instance_id": _clean_string(record.get("target_instance_id"))
        or target_instance_id(cache_key, device_index, record.get("visibility")),
        "global_mem_bytes": record.get("global_mem_bytes") or record.get("memory_bytes"),
        "hip_runtime_version": record.get("hip_runtime_version"),
        "hip_driver_version": record.get("hip_driver_version") or record.get("driver_version"),
        "hip_sdk_or_rocm_version": version,
        "configured_amdgpu_targets": record.get("configured_amdgpu_targets") or record.get("amdgpu_targets"),
        "clock_power_caveat": record.get("clock_power_caveat"),
        "accelerator_status": _accelerators_from_status(record),
        "validation_phases": phases,
        "cache_eligibility": {
            "eligible": cache.get("eligible") is True and not blockers,
            "blockers": sorted(set(blockers)),
            "scope": "matching_os_target_toolchain_group_only",
        },
        "cross_target_promotion_allowed": False,
        "validated_scope": "this_os_target_toolchain_group_only",
    }


def load_status_targets(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for index, record in enumerate(_status_records_from_json(path)):
            records.append(status_target(path, record, index))
    return records


def _merge_phase_status(rows: list[dict[str, Any]], phase: str) -> str:
    statuses = [str((row.get("validation_phases") or {}).get(phase) or "missing") for row in rows]
    if any(_phase_ready(status) for status in statuses):
        return "pass"
    if any(status not in {"missing", "not_run", "not-requested", "unknown"} for status in statuses):
        return sorted(statuses)[0]
    return "missing"


def _merge_accelerators(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        statuses = row.get("accelerator_status")
        if not isinstance(statuses, dict):
            continue
        for name, status in statuses.items():
            merged[str(name).lower()].add(str(status))
    return {name: sorted(values) for name, values in sorted(merged.items())}


def _group_cache_eligibility(rows: list[dict[str, Any]], phases: dict[str, str]) -> dict[str, Any]:
    blockers: list[str] = []
    if not any((row.get("cache_eligibility") or {}).get("eligible") is True for row in rows):
        blockers.append("no_cache_eligible_capture_or_status")
    for phase in VALIDATION_PHASES:
        if not _phase_ready(phases.get(phase, "missing")):
            blockers.append(f"{phase}_not_ready")
    target_ids = {row.get("target_id") for row in rows if row.get("target_id")}
    target_cache_keys = {row.get("target_cache_key") for row in rows if row.get("target_cache_key")}
    host_oses = {row.get("host_os") for row in rows if row.get("host_os")}
    if len(target_ids) != 1:
        blockers.append("ambiguous_target_group")
    if len(target_cache_keys) > 1:
        blockers.append("ambiguous_target_cache_key")
    if len(host_oses) != 1:
        blockers.append("ambiguous_host_os_group")
    return {
        "eligible": not blockers,
        "blockers": sorted(set(blockers)),
        "scope": "matching_os_target_toolchain_group_only",
        "cross_target_promotion_allowed": False,
    }


def build_report(captures: list[Path], status_paths: list[Path] | None = None) -> dict[str, Any]:
    rows = [capture_target(path) for path in captures]
    rows.extend(load_status_targets(status_paths or []))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("target_validation_group") or row.get("target_id") or "unknown")].append(row)

    report_groups = []
    for key, value in sorted(groups.items()):
        phases = {phase: _merge_phase_status(value, phase) for phase in VALIDATION_PHASES}
        representative = value[0] if value else {}
        report_groups.append(
            {
                "target_validation_group": key,
                "host_os": representative.get("host_os"),
                "target_id": representative.get("target_id"),
                "target_family": representative.get("target_family"),
                "target_class": representative.get("target_class"),
                "target_tier": representative.get("target_tier"),
                "supported_target": representative.get("supported_target"),
                "hip_sdk_or_rocm_versions": sorted(
                    {
                        str(row.get("hip_sdk_or_rocm_version"))
                        for row in value
                        if row.get("hip_sdk_or_rocm_version") is not None
                    }
                ),
                "device_names": sorted({str(row.get("device_name")) for row in value if row.get("device_name")}),
                "device_indices": sorted({str(row.get("device_index")) for row in value if row.get("device_index") is not None}),
                "visible_device_counts": sorted(
                    {str(row.get("visible_device_count")) for row in value if row.get("visible_device_count") is not None}
                ),
                "node_gpu_counts": sorted(
                    {str(row.get("node_gpu_count")) for row in value if row.get("node_gpu_count") is not None}
                ),
                "target_cache_keys": sorted(
                    {str(row.get("target_cache_key")) for row in value if row.get("target_cache_key")}
                ),
                "target_instance_ids": sorted(
                    {str(row.get("target_instance_id")) for row in value if row.get("target_instance_id")}
                ),
                "configured_amdgpu_targets": sorted(
                    {str(row.get("configured_amdgpu_targets")) for row in value if row.get("configured_amdgpu_targets")}
                ),
                "validation_phases": phases,
                "accelerator_status": _merge_accelerators(value),
                "cache_eligibility": _group_cache_eligibility(value, phases),
                "capture_count": sum(1 for row in value if row.get("source_kind") == "capture"),
                "status_record_count": sum(1 for row in value if row.get("source_kind") == "target_status"),
                "cross_target_promotion_allowed": False,
                "rows": value,
            }
        )
    return {
        "schema_version": 2,
        "policy": TARGET_POLICY,
        "cross_target_policy": NO_CROSS_TARGET_POLICY,
        "capture_count": len(captures),
        "status_record_count": len(rows) - len(captures),
        "group_count": len(report_groups),
        "groups": report_groups,
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "target-validation-report.json"
    md_path = out_dir / "target-validation-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RNS8 Target Validation Report",
        "",
        f"- Policy: `{report['policy']}`",
        f"- Cross-target policy: `{report['cross_target_policy']}`",
        f"- Capture rows: `{report['capture_count']}`",
        f"- Status rows: `{report['status_record_count']}`",
        "",
        "| group | OS | target | family | build | ctest | smoke | release | profiler | cache eligible | blockers |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group in report["groups"]:
        phases = group["validation_phases"]
        eligibility = group["cache_eligibility"]
        lines.append(
            "| `{group}` | `{os}` | `{target}` | `{family}` | `{build}` | `{ctest}` | `{smoke}` | `{release}` | `{profiler}` | `{eligible}` | `{blockers}` |".format(
                group=group["target_validation_group"],
                os=group.get("host_os"),
                target=group.get("target_id"),
                family=group.get("target_family"),
                build=phases.get("build"),
                ctest=phases.get("ctest"),
                smoke=phases.get("smoke"),
                release=phases.get("release_capture"),
                profiler=phases.get("profiler"),
                eligible=eligibility.get("eligible"),
                blockers=", ".join(eligibility.get("blockers") or []),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="*", help="schema-v4 benchmark captures")
    parser.add_argument(
        "--target-status",
        type=Path,
        action="append",
        default=[],
        help="JSON target validation status records for build/CTest/smoke/profiler readiness",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.captures, args.target_status)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
