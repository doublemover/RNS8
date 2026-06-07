#!/usr/bin/env python3
"""Self-test CDNA environment summary normalization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cdna_env_summary


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _indexed_logs(root: Path, count: int) -> None:
    smi_lines: list[str] = []
    for device in range(count):
        bus = (count - device) * 0x10
        smi_lines.extend(
            [
                f"GPU[{device}] : Card series: AMD Instinct MI300X-{device}",
                f"GPU[{device}] : PCI Bus: 0000:{bus:02x}:00.0",
                f"GPU[{device}] : NUMA Node: {device % 2}",
            ]
        )
    _write(root / "rocm_smi_showallinfo.log", "\n".join(smi_lines) + "\n")
    _write(root / "rocminfo.log", "Name: gfx942\n" * count)
    _write(root / "hipconfig_full.log", "ROCm Version: 7.1\nHIP version: 7.1.0\n")


def _summary(root: Path, devices: str):
    return cdna_env_summary.build_summary(
        root,
        devices_option=devices,
        dry_run=True,
        environment={
            "HIP_VISIBLE_DEVICES": None,
            "ROCR_VISIBLE_DEVICES": None,
            "GPU_DEVICE_ORDINAL": None,
            "ROCM_PATH": "/opt/rocm",
            "HIP_PATH": "/opt/rocm",
            "LD_LIBRARY_PATH": "/opt/rocm/lib",
        },
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        _indexed_logs(root, 1)
        summary = _summary(root, "0")
        physical = {item["physical_device_id"]: item for item in summary["physical_devices"]}
        assert summary["physical_device_mapping_source"] == "per_device_smi"
        assert summary["visible_gpu_count"] == 1
        assert physical[0]["bdf"] == "0000:10:00.0"
        assert physical[0]["topology_source"] == "per_device_smi"

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        _indexed_logs(root, 4)
        summary = _summary(root, "0,1,2,3")
        physical = {item["physical_device_id"]: item for item in summary["physical_devices"]}
        assert summary["node_gpu_count"] == 4
        assert physical[0]["bdf"] == "0000:40:00.0"
        assert physical[3]["bdf"] == "0000:10:00.0"
        assert physical[3]["visibility_index"] == 3

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        _indexed_logs(root, 8)
        summary = _summary(root, "4,5,6,7")
        physical = {item["physical_device_id"]: item for item in summary["physical_devices"]}
        assert summary["node_gpu_count"] == 8
        assert summary["visible_gpu_count"] == 4
        assert physical[4]["visible"] is True
        assert physical[4]["visibility_index"] == 0
        assert physical[4]["bdf"] == "0000:40:00.0"
        assert physical[7]["bdf"] == "0000:10:00.0"

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        _write(root / "rocm_smi_showallinfo.log", "Card series: AMD Instinct MI300X\n0000:01:00.0\n")
        _write(root / "rocminfo.log", "Name: gfx942\n")
        summary = _summary(root, "4,5,6,7")
        physical = {item["physical_device_id"]: item for item in summary["physical_devices"]}
        assert summary["physical_device_mapping_source"] == "heuristic_index_order"
        assert physical[4]["topology_source"] == "heuristic_index_order"
        assert physical[4]["visible"] is True
        assert physical[4]["visibility_index"] == 0

    print("CDNA env summary self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
