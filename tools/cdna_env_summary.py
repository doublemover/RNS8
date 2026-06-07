#!/usr/bin/env python3
"""Normalize CDNA first-pass environment probe logs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


RUNTIME_ENV_KEYS = [
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "ROCM_PATH",
    "HIP_PATH",
    "LD_LIBRARY_PATH",
]
BDF_RE = re.compile(r"\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b")


def read_log(log_dir: Path, name: str) -> str:
    try:
        return (log_dir / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def split_visible(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_env_log(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("dry-run:"):
            continue
        key, value = line.split("=", 1)
        if key in RUNTIME_ENV_KEYS:
            values[key] = value
    return values


def captured_tool_ready(*logs: str, fallback_commands: list[str] | None = None) -> bool:
    present_logs = [log.strip() for log in logs if log.strip()]
    if not present_logs:
        return all(shutil.which(command) is not None for command in fallback_commands or [])
    for text in present_logs:
        lowered = text.lower()
        if lowered.startswith("dry-run:") or "command not found:" in lowered:
            return False
    return len(present_logs) == len(logs)


def parse_device_list(values: list[str]) -> list[int]:
    return [int(value, 10) for value in values if re.fullmatch(r"[0-9]+", value)]


def value_for_device(values: list[Any], device_index: int) -> Any:
    if 0 <= device_index < len(values):
        return values[device_index]
    if len(values) == 1:
        return values[0]
    return None


def unique_values(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def first_clean_version(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"n/a", "none", "unknown"}:
        return None
    return cleaned


def rocm_version_from_logs(hipconfig: str, version_files: str, package_versions: str) -> str | None:
    return first_clean_version(
        first_match(hipconfig, [r"ROCm Version:\s*([^\n\r]+)"])
        or first_match(version_files, [r"^[^=\n\r]*[/\\]\.info[/\\]version(?:-[A-Za-z0-9_-]+)?=([^\n\r]+)"])
        or first_match(package_versions, [r"^rocm-core\s+([^\s]+)"])
    )


def hip_version_from_logs(hipconfig: str, hipcc: str, package_versions: str) -> str | None:
    return first_clean_version(
        first_match(
            hipconfig + "\n" + hipcc,
            [r"HIP version:\s*([^\n\r]+)", r"HIP_VERSION\s*:\s*([^\n\r]+)", r"HIP version\s+([^\n\r]+)"],
        )
        or first_match(package_versions, [r"^hipcc\s+([^\s]+)", r"^hip-runtime-amd\s+([^\s]+)"])
    )


def smi_device_index(line: str) -> int | None:
    for pattern in [
        r"\bGPU\s*\[\s*([0-9]+)\s*\]",
        r"^\s*\[\s*([0-9]+)\s*\]",
        r"^\s*GPU\s+([0-9]+)\b",
    ]:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return int(match.group(1), 10)
    return None


def parse_indexed_smi_records(smi_text: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    name_patterns = [
        r"Card series:\s*([^\n\r|]+)",
        r"Product Name:\s*([^\n\r|]+)",
        r"Marketing Name:\s*([^\n\r|]+)",
        r"Device Name:\s*([^\n\r|]+)",
        r"ASIC:\s*([^\n\r|]+)",
    ]
    for line in smi_text.splitlines():
        device_index = smi_device_index(line)
        if device_index is None:
            continue
        record = records.setdefault(device_index, {"physical_device_id": device_index})
        if match := BDF_RE.search(line):
            record["bdf"] = match.group(0)
        for pattern in name_patterns:
            if match := re.search(pattern, line, flags=re.IGNORECASE):
                value = match.group(1).strip()
                if value and value.lower() not in {"n/a", "none", "unknown"}:
                    record["device_name"] = value
                break
        if match := re.search(r"\bNUMA(?:\s+Node)?\s*[:=]\s*([0-9]+)\b", line, flags=re.IGNORECASE):
            record["numa_node"] = int(match.group(1), 10)
        if match := re.search(r"\bgfx[0-9a-fA-F]+\b", line):
            record["target_arch"] = match.group(0)
    return {key: value for key, value in records.items() if len(value) > 1}


def unindexed_device_names(smi_text: str) -> list[str]:
    names: list[str] = []
    for pattern in [
        r"Card series:\s*([^\n\r]+)",
        r"Product Name:\s*([^\n\r]+)",
        r"Marketing Name:\s*([^\n\r]+)",
        r"Device Name:\s*([^\n\r]+)",
    ]:
        names.extend(re.findall(pattern, smi_text, flags=re.IGNORECASE))
    return unique_values(names)


def build_summary(
    log_dir: Path,
    *,
    devices_option: str = "",
    dry_run: bool = False,
    environment: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    environment = environment or dict(os.environ)
    rocminfo = read_log(log_dir, "rocminfo")
    hipconfig = read_log(log_dir, "hipconfig_full")
    hipcc = read_log(log_dir, "hipcc_version")
    rocm_version_files = read_log(log_dir, "rocm_version_files")
    rocm_package_versions = read_log(log_dir, "rocm_package_versions")
    smi_text = "\n".join(
        [
            read_log(log_dir, "rocm_smi_showallinfo"),
            read_log(log_dir, "rocm_smi_showbus"),
            read_log(log_dir, "rocm_smi_showid"),
            read_log(log_dir, "amd_smi_static"),
        ]
    )
    numactl = read_log(log_dir, "numactl_hardware")
    rccl = read_log(log_dir, "rccl_discovery")
    env_log = read_log(log_dir, "env")
    rocprofv3_version = read_log(log_dir, "rocprofv3_version")
    rocprofv3_agents = read_log(log_dir, "rocprofv3_avail_agents")
    rocprofv3_pmcs = read_log(log_dir, "rocprofv3_avail_pmcs")
    captured_environment = parse_env_log(env_log)
    runtime_environment = {key: captured_environment.get(key, environment.get(key)) for key in RUNTIME_ENV_KEYS}

    gfx_targets = sorted(set(re.findall(r"\bgfx[0-9a-fA-F]+\b", rocminfo)))
    device_names = unindexed_device_names(smi_text)
    bdfs = sorted(set(BDF_RE.findall(smi_text)))
    indexed_records = parse_indexed_smi_records(smi_text)

    numa_nodes: list[int] = []
    if match := re.search(r"available:\s*([0-9]+)\s+nodes?", numactl, flags=re.IGNORECASE):
        numa_nodes = list(range(int(match.group(1))))

    visible_values = split_visible(devices_option)
    if not visible_values:
        for key in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"]:
            visible_values = split_visible(runtime_environment.get(key))
            if visible_values:
                break

    requested_device_ids = parse_device_list(visible_values)
    if indexed_records:
        node_gpu_count = max(max(indexed_records) + 1, len(indexed_records), len(requested_device_ids) or 0)
    else:
        node_gpu_count = len(device_names) or len(bdfs) or len(gfx_targets) or len(visible_values) or None
    visible_gpu_count = len(visible_values) if visible_values else node_gpu_count

    inventory_ids = set(range(node_gpu_count or 0))
    inventory_ids.update(indexed_records)
    inventory_ids.update(requested_device_ids)
    physical_devices: list[dict[str, Any]] = []
    for device_index in sorted(inventory_ids):
        record = indexed_records.get(device_index, {})
        source = "per_device_smi" if record else "heuristic_index_order"
        physical_devices.append(
            {
                "physical_device_id": device_index,
                "target_arch": record.get("target_arch") or value_for_device(gfx_targets, device_index),
                "device_name": record.get("device_name") or value_for_device(device_names, device_index),
                "bdf": record.get("bdf") or value_for_device(bdfs, device_index),
                "numa_node": record.get("numa_node")
                if record.get("numa_node") is not None
                else (value_for_device(numa_nodes, device_index) if numa_nodes else None),
                "visible": device_index in requested_device_ids if requested_device_ids else None,
                "visibility_index": requested_device_ids.index(device_index) if device_index in requested_device_ids else None,
                "topology_source": source,
            }
        )

    rccl_header = first_match(rccl, [r"^(/.*rccl(?:/rccl)?\.h)$"])
    rccl_library = first_match(rccl, [r"^(/.*librccl\.so)$"])
    rccl_tests = {
        name: path
        for name, path in re.findall(r"^(all_[a-z_]+_perf|broadcast_perf|reduce_scatter_perf)=(.+)$", rccl, flags=re.MULTILINE)
        if path != "not-found"
    }
    rocm_version = rocm_version_from_logs(hipconfig, rocm_version_files, rocm_package_versions)
    hip_version = hip_version_from_logs(hipconfig, hipcc, rocm_package_versions)

    return {
        "schema_version": 1,
        "dry_run": dry_run,
        "runtime_environment": runtime_environment,
        "requested_devices": split_visible(devices_option),
        "rocminfo_gfx_targets": gfx_targets,
        "smi_device_names": device_names,
        "rocm_version": rocm_version,
        "hip_version": hip_version,
        "hip_sdk_or_rocm_version": hip_version or rocm_version,
        "visible_gpu_count": visible_gpu_count,
        "node_gpu_count": node_gpu_count,
        "physical_device_mapping_source": "per_device_smi" if indexed_records else "heuristic_index_order",
        "physical_devices": physical_devices,
        "numa_nodes": numa_nodes,
        "pci_bdf_ids": bdfs,
        "rocprofv3_ready": captured_tool_ready(
            rocprofv3_version,
            rocprofv3_agents,
            rocprofv3_pmcs,
            fallback_commands=["rocprofv3", "rocprofv3-avail"],
        ),
        "rccl_ready": bool(rccl_header and rccl_library),
        "rccl_tests_ready": bool(rccl_tests),
        "rccl_tests": rccl_tests,
        "raw_logs": sorted(str(path.name) for path in log_dir.glob("*.log")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--devices", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.log_dir, devices_option=args.devices, dry_run=args.dry_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
