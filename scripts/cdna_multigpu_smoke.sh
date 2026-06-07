#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cdna_common.sh
source "${SCRIPT_DIR}/cdna_common.sh"

CDNA_OUT_DIR="temp/cdna-multigpu-smoke/$(cdna_timestamp)"
cdna_parse_common_args "$@"
cd "${CDNA_REPO_ROOT}"

mkdir -p "${CDNA_OUT_DIR}/shards"
: >"$(cdna_plan_file)"

PRESET="$(cdna_default_preset)"
PYTHON_BIN="$(cdna_python_bin)"
cdna_resolve_cmake_tools "${PYTHON_BIN}"
DEVICES="$(cdna_discover_devices)"
DEVICE_LIST=()
cdna_split_devices "${DEVICES}" DEVICE_LIST
if [[ "${#DEVICE_LIST[@]}" -eq 0 ]]; then
  DEVICE_LIST=(0)
fi
WORLD_SIZE="${#DEVICE_LIST[@]}"
ENV_DIR="${CDNA_OUT_DIR}/env"
STATUS_JSON="${CDNA_OUT_DIR}/target-status.json"
TARGET_REPORT_DIR="${CDNA_OUT_DIR}/target-validation"
SHARD_REPORT_DIR="${CDNA_OUT_DIR}/multigpu-shard-report"
BENCH_BIN="$(cdna_binary_path "${PRESET}" "rns8-bench")"
BUILD_STATUS="not_run"
CTEST_STATUS="not_run"

ENV_PROBE_ARGS=(--out-dir "${ENV_DIR}" --devices "${DEVICES}")
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  ENV_PROBE_ARGS+=(--dry-run)
fi
if [[ "${CDNA_ACCELERATORS}" -eq 1 ]]; then
  ENV_PROBE_ARGS+=(--accelerators)
fi
cdna_repo_run_artifact_command env_probe "${SCRIPT_DIR}/cdna_env_probe.sh" "${ENV_PROBE_ARGS[@]}"

if [[ "${CDNA_SKIP_BUILD}" -ne 0 ]]; then
  BUILD_STATUS="skipped"
  CTEST_STATUS="skipped"
elif [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  cdna_repo_run configure "${CDNA_CMAKE_BIN}" --preset "${PRESET}"
  cdna_repo_run build "${CDNA_CMAKE_BIN}" --build --preset "${PRESET}"
  cdna_repo_run ctest "${CDNA_CTEST_BIN}" --preset "${PRESET}" --output-on-failure
  BUILD_STATUS="planned"
  CTEST_STATUS="planned"
else
  cdna_repo_run configure "${CDNA_CMAKE_BIN}" --preset "${PRESET}"
  cdna_repo_run build "${CDNA_CMAKE_BIN}" --build --preset "${PRESET}"
  cdna_repo_run ctest "${CDNA_CTEST_BIN}" --preset "${PRESET}" --output-on-failure
  BUILD_STATUS="pass"
  CTEST_STATUS="pass"
fi

cdna_default_bench_command "${BENCH_BIN}"

PIDS=()
CAPTURES=()
for rank in "${!DEVICE_LIST[@]}"; do
  device="${DEVICE_LIST[$rank]}"
  shard_dir="${CDNA_OUT_DIR}/shards/gpu${device}"
  mkdir -p "${shard_dir}"
  capture="${shard_dir}/bounded-i64-hip-direct-smoke-rank${rank}.json"
  schema_log="${shard_dir}/benchmark-schema-rank${rank}.log"
  CAPTURES+=("${capture}")
  cdna_note_command "shard_${rank}_capture" env \
    ROCR_VISIBLE_DEVICES="${device}" \
    HIP_VISIBLE_DEVICES="${device}" \
    RNS8_MULTI_GPU_MODE=embarrassingly_parallel_shards \
    RNS8_RANK="${rank}" \
    RNS8_WORLD_SIZE="${WORLD_SIZE}" \
    "${CDNA_DEFAULT_BENCH_CMD[@]}"
  if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
    printf '{"dry_run": true, "rank": %s, "world_size": %s, "device": "%s"}\n' "${rank}" "${WORLD_SIZE}" "${device}" >"${capture}"
    printf 'dry-run schema validation for %s\n' "${capture}" >"${schema_log}"
  else
    (
      cd "${CDNA_REPO_ROOT}"
      env \
        ROCR_VISIBLE_DEVICES="${device}" \
        HIP_VISIBLE_DEVICES="${device}" \
        RNS8_MULTI_GPU_MODE=embarrassingly_parallel_shards \
        RNS8_RANK="${rank}" \
        RNS8_WORLD_SIZE="${WORLD_SIZE}" \
        "${CDNA_DEFAULT_BENCH_CMD[@]}" >"${capture}"
      "${PYTHON_BIN}" tools/benchmark_schema.py "${capture}" >"${schema_log}" 2>&1
    ) &
    PIDS+=("$!")
  fi
done

SHARD_FAILURES=0
if [[ "${CDNA_DRY_RUN}" -eq 0 ]]; then
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      SHARD_FAILURES=1
    fi
  done
fi

CDNA_ENV_SUMMARY="${ENV_DIR}/cdna-env-summary.json" \
CDNA_STATUS_JSON="${STATUS_JSON}" \
CDNA_DEVICES_VALUE="${DEVICES}" \
CDNA_WORLD_SIZE="${WORLD_SIZE}" \
CDNA_DRY_RUN_VALUE="${CDNA_DRY_RUN}" \
CDNA_BUILD_STATUS="${BUILD_STATUS}" \
CDNA_CTEST_STATUS="${CTEST_STATUS}" \
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

summary_path = Path(os.environ["CDNA_ENV_SUMMARY"])
status_path = Path(os.environ["CDNA_STATUS_JSON"])
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except OSError:
    summary = {}
except json.JSONDecodeError:
    summary = {}

devices = [part.strip() for part in os.environ["CDNA_DEVICES_VALUE"].split(",") if part.strip()]
world_size = int(os.environ["CDNA_WORLD_SIZE"])
targets = summary.get("rocminfo_gfx_targets") or []
names = summary.get("smi_device_names") or []
bdfs = summary.get("pci_bdf_ids") or []
numa_nodes = summary.get("numa_nodes") or []
physical_devices = {
    str(item.get("physical_device_id")): item
    for item in summary.get("physical_devices", [])
    if isinstance(item, dict) and item.get("physical_device_id") is not None
}


def by_rank_or_first(values, rank):
    if rank < len(values):
        return values[rank]
    return values[0] if values else None


records = []
for rank, device in enumerate(devices):
    physical = physical_devices.get(str(device), {})
    physical_device_id = int(device) if device.isdigit() else device
    records.append(
        {
            "host_os": "linux",
            "target_id": physical.get("target_arch") or by_rank_or_first(targets, rank),
            "rocm_version": summary.get("rocm_version"),
            "hip_sdk_or_rocm_version": summary.get("rocm_version"),
            "hip_runtime_version": summary.get("hip_version"),
            "gpu_name": physical.get("device_name") or by_rank_or_first(names, rank),
            "device_index": physical_device_id,
            "physical_device_id": physical_device_id,
            "visible_device_count": 1,
            "visible_gpu_count": 1,
            "node_gpu_count": summary.get("node_gpu_count") or world_size,
            "multi_gpu_mode": "embarrassingly_parallel_shards",
            "rank": rank,
            "world_size": world_size,
            "device_bdf": physical.get("bdf") or by_rank_or_first(bdfs, rank),
            "numa_node": physical.get("numa_node")
            if physical.get("numa_node") is not None
            else (numa_nodes[rank % len(numa_nodes)] if numa_nodes else None),
            "rocprofv3_ready": summary.get("rocprofv3_ready"),
            "rccl_ready": summary.get("rccl_ready"),
            "rccl_tests_ready": summary.get("rccl_tests_ready"),
            "configured_amdgpu_targets": "gfx90a;gfx942;gfx950",
            "validation": {
                "build": os.environ["CDNA_BUILD_STATUS"],
                "ctest": os.environ["CDNA_CTEST_STATUS"],
                "smoke": "planned" if os.environ["CDNA_DRY_RUN_VALUE"] == "1" else "pass",
                "release_capture": "planned" if os.environ["CDNA_DRY_RUN_VALUE"] == "1" else "pass",
                "profiler": "pass" if summary.get("rocprofv3_ready") else "missing",
            },
            "cache_eligibility": {
                "eligible": False,
                "blockers": ["multi_gpu_smoke_not_release_reviewed"],
            },
        }
    )
status_path.write_text(json.dumps({"records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(status_path)
PY

if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  mkdir -p "${TARGET_REPORT_DIR}"
  printf 'dry-run target validation for %s\n' "${STATUS_JSON}" >"${TARGET_REPORT_DIR}/target-validation-report.md"
else
  cdna_note_command target_validation "${PYTHON_BIN}" tools/target_validation_report.py --target-status "${STATUS_JSON}" --out-dir "${TARGET_REPORT_DIR}" "${CAPTURES[@]}"
  (cd "${CDNA_REPO_ROOT}" && "${PYTHON_BIN}" tools/target_validation_report.py --target-status "${STATUS_JSON}" --out-dir "${TARGET_REPORT_DIR}" "${CAPTURES[@]}") || SHARD_FAILURES=1
fi

cdna_note_command multigpu_shard_report "${PYTHON_BIN}" tools/multigpu_shard_report.py \
  --env-summary "${ENV_DIR}/cdna-env-summary.json" \
  --target-status "${STATUS_JSON}" \
  --shards-dir "${CDNA_OUT_DIR}/shards" \
  --out-dir "${SHARD_REPORT_DIR}"
if ! (cd "${CDNA_REPO_ROOT}" && "${PYTHON_BIN}" tools/multigpu_shard_report.py \
  --env-summary "${ENV_DIR}/cdna-env-summary.json" \
  --target-status "${STATUS_JSON}" \
  --shards-dir "${CDNA_OUT_DIR}/shards" \
  --out-dir "${SHARD_REPORT_DIR}"); then
  SHARD_FAILURES=1
fi

echo "CDNA multi-GPU smoke output: ${CDNA_OUT_DIR}"
if [[ "${SHARD_FAILURES}" -ne 0 ]]; then
  exit 1
fi
