#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cdna_common.sh
source "${SCRIPT_DIR}/cdna_common.sh"

CDNA_OUT_DIR="temp/cdna-first-pass/$(cdna_timestamp)/env"
cdna_parse_common_args "$@"
cd "${CDNA_REPO_ROOT}"

mkdir -p "${CDNA_OUT_DIR}"
: >"$(cdna_plan_file)"

cdna_repo_capture env env || true
cdna_repo_capture uname uname -a || true
cdna_repo_capture lscpu lscpu || true
cdna_repo_capture numactl_hardware numactl --hardware || true
cdna_repo_capture lstopo lstopo --of console || true
cdna_repo_capture hipconfig_full hipconfig --full || true
cdna_repo_capture hipcc_version hipcc --version || true
cdna_repo_capture rocminfo rocminfo || true
if command -v rocm-smi >/dev/null 2>&1 || [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  cdna_repo_capture rocm_smi_showallinfo rocm-smi --showallinfo || true
  cdna_repo_capture rocm_smi_showbus rocm-smi --showbus || true
  cdna_repo_capture rocm_smi_showid rocm-smi --showid || true
elif command -v amd-smi >/dev/null 2>&1; then
  cdna_repo_capture amd_smi_static amd-smi static -a || true
else
  printf 'command not found: rocm-smi or amd-smi\n' >"${CDNA_OUT_DIR}/smi.log"
fi
cdna_repo_capture rocprofv3_version rocprofv3 --version || true
cdna_repo_capture rocprofv3_avail_agents rocprofv3-avail list --agent || true
cdna_repo_capture rocprofv3_avail_pmcs rocprofv3-avail list --pmc || true
cdna_repo_capture cmake_presets cmake --list-presets || true
cdna_repo_capture check_dependencies_json python tools/check_dependencies.py --json || true
cdna_repo_capture rccl_discovery bash -lc 'set -u; for p in /opt/rocm/include/rccl/rccl.h /opt/rocm/include/rccl.h /opt/rocm/lib/librccl.so /opt/rocm/lib64/librccl.so; do if [[ -e "$p" ]]; then echo "$p"; fi; done; for t in all_reduce_perf all_gather_perf broadcast_perf reduce_scatter_perf; do if command -v "$t" >/dev/null 2>&1; then echo "$t=$(command -v "$t")"; else echo "$t=not-found"; fi; done' || true

CDNA_SUMMARY_PATH="${CDNA_OUT_DIR}/cdna-env-summary.json" \
CDNA_PROBE_OUT_DIR="${CDNA_OUT_DIR}" \
CDNA_DEVICES_OPTION="${CDNA_DEVICES}" \
CDNA_DRY_RUN_VALUE="${CDNA_DRY_RUN}" \
python - <<'PY'
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

out_dir = Path(os.environ["CDNA_PROBE_OUT_DIR"])
summary_path = Path(os.environ["CDNA_SUMMARY_PATH"])
devices_option = os.environ.get("CDNA_DEVICES_OPTION", "")


def read(name: str) -> str:
    path = out_dir / f"{name}.log"
    try:
        return path.read_text(encoding="utf-8", errors="replace")
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


rocminfo = read("rocminfo")
hipconfig = read("hipconfig_full")
hipcc = read("hipcc_version")
smi_text = "\n".join(
    [
        read("rocm_smi_showallinfo"),
        read("rocm_smi_showbus"),
        read("rocm_smi_showid"),
        read("amd_smi_static"),
    ]
)
numactl = read("numactl_hardware")
rccl = read("rccl_discovery")

gfx_targets = sorted(set(re.findall(r"\bgfx[0-9a-fA-F]+\b", rocminfo)))
device_names = []
for pattern in [
    r"Card series:\s*([^\n\r]+)",
    r"Product Name:\s*([^\n\r]+)",
    r"Marketing Name:\s*([^\n\r]+)",
    r"Device Name:\s*([^\n\r]+)",
]:
    for value in re.findall(pattern, smi_text, flags=re.IGNORECASE):
        cleaned = value.strip()
        if cleaned and cleaned not in device_names:
            device_names.append(cleaned)

bdfs = sorted(set(re.findall(r"\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b", smi_text)))
numa_nodes = []
match = re.search(r"available:\s*([0-9]+)\s+nodes?", numactl, flags=re.IGNORECASE)
if match:
    numa_nodes = list(range(int(match.group(1))))

visible_values = split_visible(devices_option)
if not visible_values:
    for key in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"]:
        visible_values = split_visible(os.environ.get(key))
        if visible_values:
            break

node_gpu_count = len(device_names) or len(bdfs) or len(gfx_targets) or len(visible_values) or None
visible_gpu_count = len(visible_values) if visible_values else node_gpu_count
rocm_version = first_match(
    hipconfig,
    [
        r"ROCm Version:\s*([^\n\r]+)",
        r"ROCm Path:\s*([^\n\r]+)",
    ],
)
hip_version = first_match(
    hipconfig + "\n" + hipcc,
    [
        r"HIP version:\s*([^\n\r]+)",
        r"HIP_VERSION\s*:\s*([^\n\r]+)",
        r"HIP version\s+([^\n\r]+)",
    ],
)

rccl_header = first_match(rccl, [r"^(/.*rccl(?:/rccl)?\.h)$"])
rccl_library = first_match(rccl, [r"^(/.*librccl\.so)$"])
rccl_tests = {
    name: path
    for name, path in re.findall(r"^(all_[a-z_]+_perf|broadcast_perf|reduce_scatter_perf)=(.+)$", rccl, flags=re.MULTILINE)
    if path != "not-found"
}

summary = {
    "schema_version": 1,
    "dry_run": os.environ.get("CDNA_DRY_RUN_VALUE") == "1",
    "runtime_environment": {
        key: os.environ.get(key)
        for key in [
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "ROCM_PATH",
            "HIP_PATH",
            "LD_LIBRARY_PATH",
        ]
    },
    "requested_devices": split_visible(devices_option),
    "rocminfo_gfx_targets": gfx_targets,
    "smi_device_names": device_names,
    "rocm_version": rocm_version,
    "hip_version": hip_version,
    "visible_gpu_count": visible_gpu_count,
    "node_gpu_count": node_gpu_count,
    "numa_nodes": numa_nodes,
    "pci_bdf_ids": bdfs,
    "rocprofv3_ready": shutil.which("rocprofv3") is not None and shutil.which("rocprofv3-avail") is not None,
    "rccl_ready": bool(rccl_header and rccl_library),
    "rccl_tests_ready": bool(rccl_tests),
    "rccl_tests": rccl_tests,
    "raw_logs": sorted(str(path.name) for path in out_dir.glob("*.log")),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(summary_path)
PY

echo "CDNA env probe summary: ${CDNA_OUT_DIR}/cdna-env-summary.json"
