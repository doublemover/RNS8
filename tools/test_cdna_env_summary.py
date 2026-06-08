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
        missing_root = Path(tmp_name) / "missing-logs"
        summary = _summary(missing_root, "0,1,2,3")
        physical = {item["physical_device_id"]: item for item in summary["physical_devices"]}
        assert summary["dry_run"] is True
        assert summary["visible_gpu_count"] == 4
        assert summary["node_gpu_count"] == 4
        assert summary["raw_logs"] == []
        assert physical[0]["topology_source"] == "heuristic_index_order"
        assert physical[3]["visible"] is True
        assert physical[3]["visibility_index"] == 3

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        _write(root / "env.log", "ROCR_VISIBLE_DEVICES=4,5\nROCM_PATH=/captured/rocm\n")
        summary = cdna_env_summary.build_summary(
            root,
            devices_option="",
            dry_run=False,
            environment={
                "HIP_VISIBLE_DEVICES": None,
                "ROCR_VISIBLE_DEVICES": "0",
                "GPU_DEVICE_ORDINAL": None,
                "ROCM_PATH": "/process/rocm",
                "HIP_PATH": None,
                "LD_LIBRARY_PATH": None,
            },
        )
        assert summary["runtime_environment"]["ROCR_VISIBLE_DEVICES"] == "4,5"
        assert summary["runtime_environment"]["ROCM_PATH"] == "/captured/rocm"
        assert summary["visible_gpu_count"] == 2

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        _write(root / "rocprofv3_version.log", "rocprofv3 version 1.0\n")
        _write(root / "rocprofv3_avail_agents.log", "Agent 0 gfx942\n")
        _write(root / "rocprofv3_avail_pmcs.log", "SQ_INSTS_VALU\n")
        assert _summary(root, "0")["rocprofv3_ready"] is True
        _write(root / "rocprofv3_avail_pmcs.log", "command not found: rocprofv3-avail\n")
        assert _summary(root, "0")["rocprofv3_ready"] is False

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        _write(root / "amd_matrix_instruction_report.log", "AMD matrix instruction report: PASS\n")
        _write(root / "amd-matrix-instructions" / "amd-matrix-instruction-report.json", "{}\n")
        _write(root / "amd-matrix-instructions" / "amd-matrix-instruction-report.md", "# report\n")
        _write(root / "amd-matrix-instructions" / "layouts" / "gfx1200" / "v_swmmac_i32_16x16x32_iu8" / "compression-wave32-register-layout.csv", "matrix,0\n")
        summary = _summary(root, "0")
        assert summary["amd_matrix_instruction_calculator_ready"] is True
        assert summary["amd_matrix_instruction_reports"] == [
            "amd-matrix-instructions/amd-matrix-instruction-report.json",
            "amd-matrix-instructions/amd-matrix-instruction-report.md",
            "amd-matrix-instructions/layouts/gfx1200/v_swmmac_i32_16x16x32_iu8/compression-wave32-register-layout.csv",
        ]

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        _indexed_logs(root, 1)
        summary = _summary(root, "0")
        physical = {item["physical_device_id"]: item for item in summary["physical_devices"]}
        assert summary["rocm_version"] == "7.1"
        assert summary["hip_version"] == "7.1.0"
        assert summary["hip_sdk_or_rocm_version"] == "7.1.0"
        assert summary["physical_device_mapping_source"] == "per_device_smi"
        assert summary["visible_gpu_count"] == 1
        assert physical[0]["bdf"] == "0000:10:00.0"
        assert physical[0]["topology_source"] == "per_device_smi"

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        _write(root / "hipconfig_full.log", "HIP_PATH=/opt/rocm\n")
        _write(root / "hipcc_version.log", "HIP version: 7.2.26015-fc0010cf6a\n")
        _write(root / "rocm_version_files.log", "/opt/rocm/.info/version=7.2.0\n")
        summary = _summary(root, "0")
        assert summary["rocm_version"] == "7.2.0"
        assert summary["hip_version"] == "7.2.26015-fc0010cf6a"
        assert summary["hip_sdk_or_rocm_version"] == "7.2.26015-fc0010cf6a"

    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        _write(root / "hipconfig_full.log", "HIP_PATH=/opt/rocm\n")
        _write(root / "rocm_package_versions.log", "rocm-core 7.2.0.70200-1\nhipcc 7.2.26015-fc0010cf6a\n")
        summary = _summary(root, "0")
        assert summary["rocm_version"] == "7.2.0.70200-1"
        assert summary["hip_version"] == "7.2.26015-fc0010cf6a"
        assert summary["hip_sdk_or_rocm_version"] == "7.2.26015-fc0010cf6a"

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
