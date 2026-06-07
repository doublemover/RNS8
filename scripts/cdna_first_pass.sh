#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cdna_common.sh
source "${SCRIPT_DIR}/cdna_common.sh"

CDNA_OUT_DIR="temp/cdna-first-pass/$(cdna_timestamp)"
cdna_parse_common_args "$@"
cd "${CDNA_REPO_ROOT}"

mkdir -p "${CDNA_OUT_DIR}"
: >"$(cdna_plan_file)"

PRESET="$(cdna_default_preset)"
PYTHON_BIN="$(cdna_python_bin)"
cdna_resolve_cmake_tools "${PYTHON_BIN}"
cdna_resolve_ninja
DEVICES="$(cdna_discover_devices)"
DEVICE="$(cdna_first_device "${DEVICES}")"
CMAKE_CONFIGURE_CMD=("${CDNA_CMAKE_BIN}" --preset "${PRESET}")
if [[ -n "${CDNA_NINJA_BIN}" ]]; then
  CMAKE_CONFIGURE_CMD+=("-DCMAKE_MAKE_PROGRAM=${CDNA_NINJA_BIN}")
fi
ENV_DIR="${CDNA_OUT_DIR}/env"
DEPS_JSON="${CDNA_OUT_DIR}/check-dependencies.json"
CAPTURE="${CDNA_OUT_DIR}/bounded-i64-hip-direct-smoke.json"
SCHEMA_LOG="${CDNA_OUT_DIR}/benchmark-schema.log"
STATUS_JSON="${CDNA_OUT_DIR}/target-status.json"
TARGET_REPORT_DIR="${CDNA_OUT_DIR}/target-validation"
VERIFY_BIN="$(cdna_binary_path "${PRESET}" "rns8-verify")"
BENCH_BIN="$(cdna_binary_path "${PRESET}" "rns8-bench")"

ENV_PROBE_ARGS=(--out-dir "${ENV_DIR}" --devices "${DEVICE}")
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  ENV_PROBE_ARGS+=(--dry-run)
fi
if [[ "${CDNA_ACCELERATORS}" -eq 1 ]]; then
  ENV_PROBE_ARGS+=(--accelerators)
fi
cdna_repo_run_artifact_command env_probe "${SCRIPT_DIR}/cdna_env_probe.sh" "${ENV_PROBE_ARGS[@]}"

cdna_note_command check_dependencies_json "${PYTHON_BIN}" tools/check_dependencies.py --json --accelerator-probes --accelerator-probe-dir "${CDNA_OUT_DIR}/dependency-probes"
DEPS_STATUS="not_run"
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  printf '{"dry_run": true, "accelerator_probes": "planned"}\n' >"${DEPS_JSON}"
  DEPS_STATUS="planned"
else
  if (cd "${CDNA_REPO_ROOT}" && "${PYTHON_BIN}" tools/check_dependencies.py --json --accelerator-probes --accelerator-probe-dir "${CDNA_OUT_DIR}/dependency-probes") >"${DEPS_JSON}"; then
    DEPS_STATUS="pass"
  else
    DEPS_STATUS="$("${PYTHON_BIN}" - "${DEPS_JSON}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("fail")
    raise SystemExit(0)

readiness = report.get("readiness")
if not isinstance(readiness, dict):
    print("fail")
    raise SystemExit(0)
gates = readiness.get("gates")
if not isinstance(gates, dict):
    print("fail")
    raise SystemExit(0)
required_failures = []
for name, gate in gates.items():
    if not isinstance(gate, dict):
        continue
    if gate.get("required_for_host_readiness") is True and gate.get("ok") is not True:
        required_failures.append(str(name))
print("fail" if required_failures else "pass")
PY
)"
    if [[ "${DEPS_STATUS}" == "pass" ]]; then
      printf 'warning: dependency checker returned nonzero for advisory gates only; host-required gates passed. See %s\n' "${DEPS_JSON}" >&2
    else
      printf 'warning: dependency checker reported host-required not-ready status; continuing with build/smoke. See %s\n' "${DEPS_JSON}" >&2
    fi
  fi
fi

cdna_resolve_catch2

BUILD_STATUS="not_run"
CTEST_STATUS="not_run"
SMOKE_STATUS="not_run"
CAPTURE_STATUS="not_run"
if [[ "${CDNA_SKIP_BUILD}" -ne 0 ]]; then
  BUILD_STATUS="skipped"
  CTEST_STATUS="skipped"
elif [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  cdna_repo_run configure "${CMAKE_CONFIGURE_CMD[@]}"
  cdna_repo_run build "${CDNA_CMAKE_BIN}" --build --preset "${PRESET}"
  cdna_repo_run ctest env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_CTEST_BIN}" --preset "${PRESET}" --output-on-failure
  BUILD_STATUS="planned"
  CTEST_STATUS="planned"
else
  cdna_repo_run configure "${CMAKE_CONFIGURE_CMD[@]}"
  cdna_repo_run build "${CDNA_CMAKE_BIN}" --build --preset "${PRESET}"
  BUILD_STATUS="pass"
  cdna_repo_run ctest env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_CTEST_BIN}" --preset "${PRESET}" --output-on-failure
  CTEST_STATUS="pass"
fi

cdna_repo_run hip_smoke env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${VERIFY_BIN}" --hip-smoke
SMOKE_STATUS=$([[ "${CDNA_DRY_RUN}" -eq 1 ]] && printf '%s' "planned" || printf '%s' "pass")

cdna_default_bench_command "${BENCH_BIN}"
cdna_note_command rns8_bench_capture env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_DEFAULT_BENCH_CMD[@]}"
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  printf '{"dry_run": true, "planned_capture": "%s"}\n' "${CAPTURE}" >"${CAPTURE}"
  printf 'dry-run schema validation for %s\n' "${CAPTURE}" >"${SCHEMA_LOG}"
  CAPTURE_STATUS="planned"
else
  (cd "${CDNA_REPO_ROOT}" && env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_DEFAULT_BENCH_CMD[@]}") >"${CAPTURE}"
  (cd "${CDNA_REPO_ROOT}" && "${PYTHON_BIN}" tools/benchmark_schema.py "${CAPTURE}") >"${SCHEMA_LOG}" 2>&1
  CAPTURE_STATUS="pass"
fi

CDNA_ENV_SUMMARY="${ENV_DIR}/cdna-env-summary.json" \
CDNA_STATUS_JSON="${STATUS_JSON}" \
CDNA_CAPTURE_JSON="${CAPTURE}" \
CDNA_DEVICE="${DEVICE}" \
CDNA_PRESET_VALUE="${PRESET}" \
CDNA_BUILD_STATUS="${BUILD_STATUS}" \
CDNA_CTEST_STATUS="${CTEST_STATUS}" \
CDNA_SMOKE_STATUS="${SMOKE_STATUS}" \
CDNA_CAPTURE_STATUS="${CAPTURE_STATUS}" \
CDNA_DEPS_STATUS="${DEPS_STATUS}" \
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

summary_path = Path(os.environ["CDNA_ENV_SUMMARY"])
status_path = Path(os.environ["CDNA_STATUS_JSON"])
capture_path = Path(os.environ["CDNA_CAPTURE_JSON"])
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except OSError:
    summary = {}
except json.JSONDecodeError:
    summary = {}
try:
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
except OSError:
    capture = {}
except json.JSONDecodeError:
    capture = {}

targets = summary.get("rocminfo_gfx_targets") or []
names = summary.get("smi_device_names") or []
bdfs = summary.get("pci_bdf_ids") or []
numa_nodes = summary.get("numa_nodes") or []
capture_device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
capture_target = capture.get("target_variant") if isinstance(capture.get("target_variant"), dict) else {}
capture_toolchain = capture.get("hip_toolchain") if isinstance(capture.get("hip_toolchain"), dict) else {}
device = os.environ["CDNA_DEVICE"]
physical_devices = {
    str(item.get("physical_device_id")): item
    for item in summary.get("physical_devices", [])
    if isinstance(item, dict) and item.get("physical_device_id") is not None
}
physical = physical_devices.get(str(device), {})
physical_device_id = int(device) if device.isdigit() else device
toolchain_version = (
    capture_toolchain.get("hip_sdk_or_rocm_version")
    or summary.get("hip_sdk_or_rocm_version")
    or summary.get("hip_version")
    or summary.get("rocm_version")
)
record = {
    "host_os": "linux",
    "target_id": capture_target.get("target_id") or physical.get("target_arch") or (targets[0] if targets else None),
    "rocm_version": summary.get("rocm_version"),
    "hip_sdk_or_rocm_version": toolchain_version,
    "hip_runtime_version": capture_device.get("hip_runtime_version") or summary.get("hip_version"),
    "hip_driver_version": capture_device.get("hip_driver_version"),
    "gpu_name": capture_device.get("name") or physical.get("device_name") or (names[0] if names else None),
    "device_index": physical_device_id,
    "physical_device_id": physical_device_id,
    "visible_device_count": 1,
    "visible_gpu_count": 1,
    "node_gpu_count": summary.get("node_gpu_count"),
    "multi_gpu_mode": "single_device_smoke",
    "rank": 0,
    "world_size": 1,
    "device_bdf": physical.get("bdf") or (bdfs[0] if bdfs else None),
    "numa_node": physical.get("numa_node") if physical.get("numa_node") is not None else (numa_nodes[0] if numa_nodes else None),
    "target_cache_key": capture_target.get("target_cache_key") or capture_device.get("target_cache_key"),
    "target_instance_id": capture_target.get("target_instance_id") or capture_device.get("target_instance_id"),
    "global_mem_bytes": capture_device.get("global_mem_bytes"),
    "rocprofv3_ready": summary.get("rocprofv3_ready"),
    "rccl_ready": summary.get("rccl_ready"),
    "rccl_tests_ready": summary.get("rccl_tests_ready"),
    "configured_amdgpu_targets": "gfx90a;gfx942;gfx950",
    "validation": {
        "dependency_check": os.environ["CDNA_DEPS_STATUS"],
        "build": os.environ["CDNA_BUILD_STATUS"],
        "ctest": os.environ["CDNA_CTEST_STATUS"],
        "smoke": os.environ["CDNA_SMOKE_STATUS"],
        "release_capture": os.environ["CDNA_CAPTURE_STATUS"],
        "profiler": "pass" if summary.get("rocprofv3_ready") else "missing",
    },
    "cache_eligibility": {
        "eligible": False,
        "blockers": ["first_pass_smoke_not_release_reviewed"],
    },
    "preset": os.environ["CDNA_PRESET_VALUE"],
}
status_path.write_text(json.dumps({"records": [record]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(status_path)
PY

cdna_note_command target_validation "${PYTHON_BIN}" tools/target_validation_report.py --target-status "${STATUS_JSON}" --out-dir "${TARGET_REPORT_DIR}" "${CAPTURE}"
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  mkdir -p "${TARGET_REPORT_DIR}"
  printf 'dry-run target validation for %s\n' "${STATUS_JSON}" >"${TARGET_REPORT_DIR}/target-validation-report.md"
else
  (cd "${CDNA_REPO_ROOT}" && "${PYTHON_BIN}" tools/target_validation_report.py --target-status "${STATUS_JSON}" --out-dir "${TARGET_REPORT_DIR}" "${CAPTURE}")
fi

if [[ -n "${CDNA_RANK_SCENARIOS}" ]]; then
  RANK_SCENARIO_ROOT="${CDNA_OUT_DIR}/rank-scenarios"
  cdna_split_devices "${CDNA_RANK_SCENARIOS}" RANK_SCENARIO_LIST
  mkdir -p "${RANK_SCENARIO_ROOT}"
  for scenario in "${RANK_SCENARIO_LIST[@]}"; do
    scenario_out="${RANK_SCENARIO_ROOT}/${scenario}"
    cdna_note_command "rank_scenario_${scenario}" env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" \
      "${PYTHON_BIN}" tools/benchmark_sweep.py \
      --bench "${BENCH_BIN}" \
      --out-root "${scenario_out}" \
      --scenario "${scenario}" \
      --review-mode release \
      --warmups 3 \
      --repeats 9 \
      --seed 20260606 \
      --skip-existing
    if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
      mkdir -p "${scenario_out}"
      printf 'dry-run rank scenario %s\n' "${scenario}" >"${scenario_out}/rank-scenario-plan.log"
    else
      (cd "${CDNA_REPO_ROOT}" && env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" \
        "${PYTHON_BIN}" tools/benchmark_sweep.py \
          --bench "${BENCH_BIN}" \
          --out-root "${scenario_out}" \
          --scenario "${scenario}" \
          --review-mode release \
          --warmups 3 \
          --repeats 9 \
          --seed 20260606 \
          --skip-existing)
    fi
  done
fi

echo "CDNA first-pass output: ${CDNA_OUT_DIR}"
