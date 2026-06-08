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
DEVICES="$(cdna_discover_devices)"
DEVICE="$(cdna_first_device "${DEVICES}")"
ENV_DIR="${CDNA_OUT_DIR}/env"
DEPS_JSON="${CDNA_OUT_DIR}/check-dependencies.json"
CAPTURE="${CDNA_OUT_DIR}/bounded-i64-hip-direct-smoke.json"
SCHEMA_LOG="${CDNA_OUT_DIR}/benchmark-schema.log"
STATUS_JSON="${CDNA_OUT_DIR}/target-status.json"
TARGET_REPORT_DIR="${CDNA_OUT_DIR}/target-validation"
ISA_REPORT_DIR="${CDNA_OUT_DIR}/isa-reports"
VERIFY_BIN="$(cdna_binary_path "${PRESET}" "rns8-verify")"
BENCH_BIN="$(cdna_binary_path "${PRESET}" "rns8-bench")"
CONFIGURED_AMDGPU_TARGETS=""
CDNA_REVIEW_ISA_ARGV=()
DEPS_STATUS="not_run"
DEPS_EXIT_CODE=0
BUILD_STATUS="not_run"
CTEST_STATUS="not_run"
SMOKE_STATUS="not_run"
CAPTURE_STATUS="not_run"
CDNA_CURRENT_PHASE="initialization"

cdna_first_pass_failure_report() {
  local exit_code=$?
  local failed_command="${BASH_COMMAND:-unknown}"
  local failed_line="${BASH_LINENO[0]:-${LINENO}}"
  local python_for_report="${PYTHON_BIN:-python3}"
  trap - ERR
  set +e
  mkdir -p "${CDNA_OUT_DIR}" "${TARGET_REPORT_DIR}"
  if [[ ! -f "${SCHEMA_LOG}" ]]; then
    {
      printf 'schema validation: NOT_RUN\n'
      printf 'script failed before benchmark schema validation\n'
      printf 'phase: %s\n' "${CDNA_CURRENT_PHASE}"
      printf 'command: %s\n' "${failed_command}"
      printf 'exit_code: %s\n' "${exit_code}"
    } >"${SCHEMA_LOG}"
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
  CDNA_DEPS_JSON="${DEPS_JSON}" \
  CDNA_DEPS_STATUS="${DEPS_STATUS}" \
  CDNA_DEPS_EXIT_CODE="${DEPS_EXIT_CODE}" \
  CDNA_CONFIGURED_AMDGPU_TARGETS="${CONFIGURED_AMDGPU_TARGETS}" \
  CDNA_FAILURE_PHASE="${CDNA_CURRENT_PHASE}" \
  CDNA_FAILURE_COMMAND="${failed_command}" \
  CDNA_FAILURE_LINE="${failed_line}" \
  CDNA_FAILURE_EXIT_CODE="${exit_code}" \
  "${python_for_report}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


def load_json(path: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def normalize_status(value: str) -> str:
    text = (value or "not_run").strip().lower()
    if text == "running":
        return "failed"
    return text or "not_run"


summary = load_json(os.environ["CDNA_ENV_SUMMARY"])
capture = load_json(os.environ["CDNA_CAPTURE_JSON"])
deps = load_json(os.environ["CDNA_DEPS_JSON"])
phase = os.environ["CDNA_FAILURE_PHASE"]

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
dependency_readiness = deps.get("readiness") if isinstance(deps, dict) else {}
if not isinstance(dependency_readiness, dict):
    dependency_readiness = {}
dependency_gates = dependency_readiness.get("gates")
dependency_failures = []
if isinstance(dependency_gates, dict):
    for name, gate in dependency_gates.items():
        if (
            isinstance(gate, dict)
            and gate.get("required_for_host_readiness") is True
            and gate.get("ok") is not True
        ):
            dependency_failures.append(str(name))
toolchain_version = (
    capture_toolchain.get("hip_sdk_or_rocm_version")
    or summary.get("hip_sdk_or_rocm_version")
    or summary.get("hip_version")
    or summary.get("rocm_version")
)
validation = {
    "dependency_check": normalize_status(os.environ["CDNA_DEPS_STATUS"]),
    "build": normalize_status(os.environ["CDNA_BUILD_STATUS"]),
    "ctest": normalize_status(os.environ["CDNA_CTEST_STATUS"]),
    "smoke": normalize_status(os.environ["CDNA_SMOKE_STATUS"]),
    "release_capture": normalize_status(os.environ["CDNA_CAPTURE_STATUS"]),
    "profiler": "pass" if summary.get("rocprofv3_ready") else "missing",
}
if phase in {"cmake_resolve", "ninja_resolve", "catch2_resolve", "configure", "build"}:
    if validation["build"] in {"not_run", "running"}:
        validation["build"] = "failed"
if phase == "ctest":
    validation["ctest"] = "failed"
if phase == "hip_smoke":
    validation["smoke"] = "failed"
if phase in {"benchmark", "schema"}:
    validation["release_capture"] = "failed"

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
    "configured_amdgpu_targets": os.environ.get("CDNA_CONFIGURED_AMDGPU_TARGETS") or capture.get("configured_amdgpu_targets"),
    "dependency_check": {
        "status": validation["dependency_check"],
        "checker_exit_code": int(os.environ["CDNA_DEPS_EXIT_CODE"]),
        "host_readiness_ok": dependency_readiness.get("host_readiness_ok"),
        "required_gate_failures": dependency_failures,
    },
    "validation": validation,
    "failure": {
        "phase": phase,
        "command": os.environ["CDNA_FAILURE_COMMAND"],
        "line": int(os.environ["CDNA_FAILURE_LINE"]),
        "exit_code": int(os.environ["CDNA_FAILURE_EXIT_CODE"]),
    },
    "cache_eligibility": {
        "eligible": False,
        "blockers": ["first_pass_script_failed", f"failed_phase:{phase}"],
    },
    "preset": os.environ["CDNA_PRESET_VALUE"],
}
status_path = Path(os.environ["CDNA_STATUS_JSON"])
status_path.parent.mkdir(parents=True, exist_ok=True)
status_path.write_text(json.dumps({"records": [record]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(status_path)
PY
  if [[ -f "${STATUS_JSON}" ]]; then
    (cd "${CDNA_REPO_ROOT}" && "${python_for_report}" tools/target_validation_report.py --target-status "${STATUS_JSON}" --out-dir "${TARGET_REPORT_DIR}") >/dev/null 2>&1
  fi
  if [[ ! -f "${TARGET_REPORT_DIR}/target-validation-report.md" ]]; then
    printf 'target validation: NOT_RUN\nscript failed during %s\n' "${CDNA_CURRENT_PHASE}" >"${TARGET_REPORT_DIR}/target-validation-report.md"
  fi
  printf 'CDNA first-pass failed during %s; output: %s\n' "${CDNA_CURRENT_PHASE}" "${CDNA_OUT_DIR}" >&2
  exit "${exit_code}"
}

trap cdna_first_pass_failure_report ERR

CDNA_CURRENT_PHASE="cmake_resolve"
cdna_resolve_cmake_tools "${PYTHON_BIN}"
CDNA_CURRENT_PHASE="ninja_resolve"
cdna_resolve_ninja

ENV_PROBE_ARGS=(--out-dir "${ENV_DIR}" --devices "${DEVICE}")
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  ENV_PROBE_ARGS+=(--dry-run)
fi
if [[ "${CDNA_ACCELERATORS}" -eq 1 ]]; then
  ENV_PROBE_ARGS+=(--accelerators)
fi
CDNA_CURRENT_PHASE="env_probe"
cdna_repo_run_artifact_command env_probe "${SCRIPT_DIR}/cdna_env_probe.sh" "${ENV_PROBE_ARGS[@]}"

CONFIGURED_AMDGPU_TARGETS="$(cdna_active_target_from_summary "${PYTHON_BIN}" "${ENV_DIR}/cdna-env-summary.json" "${DEVICE}" 2>/dev/null || true)"
if [[ -z "${CONFIGURED_AMDGPU_TARGETS}" ]]; then
  CONFIGURED_AMDGPU_TARGETS="gfx942"
fi

CMAKE_CONFIGURE_CMD=("${CDNA_CMAKE_BIN}" --preset "${PRESET}" "-DRNS8_AMDGPU_TARGETS=${CONFIGURED_AMDGPU_TARGETS}")
if [[ -n "${CDNA_NINJA_BIN}" ]]; then
  CMAKE_CONFIGURE_CMD+=("-DCMAKE_MAKE_PROGRAM=${CDNA_NINJA_BIN}")
fi

cdna_note_command check_dependencies_json "${PYTHON_BIN}" tools/check_dependencies.py --json --accelerator-probes --accelerator-probe-dir "${CDNA_OUT_DIR}/dependency-probes"
CDNA_CURRENT_PHASE="check_dependencies"
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  printf '{"dry_run": true, "accelerator_probes": "planned"}\n' >"${DEPS_JSON}"
  DEPS_STATUS="planned"
else
  if (cd "${CDNA_REPO_ROOT}" && "${PYTHON_BIN}" tools/check_dependencies.py --json --accelerator-probes --accelerator-probe-dir "${CDNA_OUT_DIR}/dependency-probes") >"${DEPS_JSON}"; then
    DEPS_STATUS="pass"
  else
    DEPS_EXIT_CODE=$?
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
if readiness.get("host_readiness_ok") is True:
    print("pass")
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

CDNA_CURRENT_PHASE="catch2_resolve"
cdna_resolve_catch2

if [[ "${CDNA_SKIP_BUILD}" -ne 0 ]]; then
  BUILD_STATUS="skipped"
  CTEST_STATUS="skipped"
elif [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  CDNA_CURRENT_PHASE="configure"
  cdna_repo_run configure "${CMAKE_CONFIGURE_CMD[@]}"
  CDNA_CURRENT_PHASE="build"
  cdna_repo_run build "${CDNA_CMAKE_BIN}" --build --preset "${PRESET}"
  CDNA_CURRENT_PHASE="ctest"
  cdna_repo_run ctest env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_CTEST_BIN}" --preset "${PRESET}" --output-on-failure
  BUILD_STATUS="planned"
  CTEST_STATUS="planned"
else
  CDNA_CURRENT_PHASE="configure"
  BUILD_STATUS="running"
  cdna_repo_run configure "${CMAKE_CONFIGURE_CMD[@]}"
  CDNA_CURRENT_PHASE="build"
  cdna_repo_run build "${CDNA_CMAKE_BIN}" --build --preset "${PRESET}"
  BUILD_STATUS="pass"
  CTEST_STATUS="running"
  CDNA_CURRENT_PHASE="ctest"
  cdna_repo_run ctest env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_CTEST_BIN}" --preset "${PRESET}" --output-on-failure
  CTEST_STATUS="pass"
fi

if [[ "${CDNA_ACCELERATORS}" -eq 1 ]]; then
  CDNA_CURRENT_PHASE="gpu_isa_report"
  ISA_TARGET="${CONFIGURED_AMDGPU_TARGETS%%;*}"
  ISA_TARGET="${ISA_TARGET%%,*}"
  ISA_REPORT_CMD=(
    "${PYTHON_BIN}"
    tools/gpu_isa_report.py
    --build-tree "$(cdna_build_dir_for_preset "${PRESET}")"
    --backend all
    --target "${ISA_TARGET}"
    --scratch-root "${CDNA_OUT_DIR}/isa-scratch"
    --out-dir "${ISA_REPORT_DIR}"
  )
  if [[ -x /opt/rocm/bin/hipcc ]]; then
    ISA_REPORT_CMD+=(--hipcc /opt/rocm/bin/hipcc)
  fi
  MATRIX_REPORT_JSON="${ENV_DIR}/amd-matrix-instructions/amd-matrix-instruction-report.json"
  if [[ -f "${MATRIX_REPORT_JSON}" ]]; then
    ISA_REPORT_CMD+=(--matrix-instruction-report "${MATRIX_REPORT_JSON}")
  fi
  cdna_repo_run gpu_isa_report "${ISA_REPORT_CMD[@]}"
  CDNA_REVIEW_ISA_ARGV=(--isa-report "${ISA_REPORT_DIR}")
fi

CDNA_CURRENT_PHASE="hip_smoke"
SMOKE_STATUS="running"
cdna_repo_run hip_smoke env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${VERIFY_BIN}" --hip-smoke
SMOKE_STATUS=$([[ "${CDNA_DRY_RUN}" -eq 1 ]] && printf '%s' "planned" || printf '%s' "pass")

cdna_default_bench_command "${BENCH_BIN}"
cdna_note_command rns8_bench_capture env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_DEFAULT_BENCH_CMD[@]}"
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  cdna_progress rns8_bench_capture planned env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_DEFAULT_BENCH_CMD[@]}"
  printf '{"dry_run": true, "planned_capture": "%s"}\n' "${CAPTURE}" >"${CAPTURE}"
  cdna_progress benchmark_schema planned "${PYTHON_BIN}" tools/benchmark_schema.py "${CAPTURE}"
  printf 'dry-run schema validation for %s\n' "${CAPTURE}" >"${SCHEMA_LOG}"
  CAPTURE_STATUS="planned"
else
  CDNA_CURRENT_PHASE="benchmark"
  CAPTURE_STATUS="running"
  start_seconds="$(date +%s)"
  cdna_progress rns8_bench_capture start env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_DEFAULT_BENCH_CMD[@]}"
  (cd "${CDNA_REPO_ROOT}" && env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_DEFAULT_BENCH_CMD[@]}") >"${CAPTURE}"
  stop_seconds="$(date +%s)"
  cdna_progress rns8_bench_capture "done $((stop_seconds - start_seconds))s"
  CDNA_CURRENT_PHASE="schema"
  start_seconds="$(date +%s)"
  cdna_progress benchmark_schema start "${PYTHON_BIN}" tools/benchmark_schema.py "${CAPTURE}"
  (cd "${CDNA_REPO_ROOT}" && "${PYTHON_BIN}" tools/benchmark_schema.py "${CAPTURE}") >"${SCHEMA_LOG}" 2>&1
  stop_seconds="$(date +%s)"
  cdna_progress benchmark_schema "done $((stop_seconds - start_seconds))s"
  CAPTURE_STATUS="pass"
fi

CDNA_CURRENT_PHASE="write_status"
CDNA_ENV_SUMMARY="${ENV_DIR}/cdna-env-summary.json" \
CDNA_STATUS_JSON="${STATUS_JSON}" \
CDNA_CAPTURE_JSON="${CAPTURE}" \
CDNA_DEVICE="${DEVICE}" \
CDNA_PRESET_VALUE="${PRESET}" \
CDNA_BUILD_STATUS="${BUILD_STATUS}" \
CDNA_CTEST_STATUS="${CTEST_STATUS}" \
CDNA_SMOKE_STATUS="${SMOKE_STATUS}" \
CDNA_CAPTURE_STATUS="${CAPTURE_STATUS}" \
CDNA_DEPS_JSON="${DEPS_JSON}" \
CDNA_DEPS_STATUS="${DEPS_STATUS}" \
CDNA_DEPS_EXIT_CODE="${DEPS_EXIT_CODE}" \
CDNA_CONFIGURED_AMDGPU_TARGETS="${CONFIGURED_AMDGPU_TARGETS}" \
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
try:
    deps = json.loads(Path(os.environ["CDNA_DEPS_JSON"]).read_text(encoding="utf-8"))
except OSError:
    deps = {}
except json.JSONDecodeError:
    deps = {}

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
dependency_failures = []
dependency_readiness = deps.get("readiness") if isinstance(deps, dict) else {}
if not isinstance(dependency_readiness, dict):
    dependency_readiness = {}
dependency_gates = dependency_readiness.get("gates")
if isinstance(dependency_gates, dict):
    for name, gate in dependency_gates.items():
        if (
            isinstance(gate, dict)
            and gate.get("required_for_host_readiness") is True
            and gate.get("ok") is not True
        ):
            dependency_failures.append(str(name))
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
    "configured_amdgpu_targets": os.environ.get("CDNA_CONFIGURED_AMDGPU_TARGETS") or capture.get("configured_amdgpu_targets"),
    "dependency_check": {
        "status": os.environ["CDNA_DEPS_STATUS"],
        "checker_exit_code": int(os.environ["CDNA_DEPS_EXIT_CODE"]),
        "host_readiness_ok": dependency_readiness.get("host_readiness_ok"),
        "required_gate_failures": dependency_failures,
    },
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
CDNA_CURRENT_PHASE="target_validation"
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  mkdir -p "${TARGET_REPORT_DIR}"
  printf 'dry-run target validation for %s\n' "${STATUS_JSON}" >"${TARGET_REPORT_DIR}/target-validation-report.md"
else
  (cd "${CDNA_REPO_ROOT}" && "${PYTHON_BIN}" tools/target_validation_report.py --target-status "${STATUS_JSON}" --out-dir "${TARGET_REPORT_DIR}" "${CAPTURE}")
fi

if [[ "${CDNA_SKIP_RANK_SCENARIOS}" -eq 0 && -n "${CDNA_RANK_SCENARIOS}" ]]; then
  RANK_SCENARIO_ROOT="${CDNA_OUT_DIR}/rank-scenarios"
  cdna_split_devices "${CDNA_RANK_SCENARIOS}" RANK_SCENARIO_LIST
  mkdir -p "${RANK_SCENARIO_ROOT}"
  for scenario in "${RANK_SCENARIO_LIST[@]}"; do
    scenario_out="${RANK_SCENARIO_ROOT}/${scenario}"
    CDNA_CURRENT_PHASE="rank_scenario_${scenario}"
    cdna_note_command "rank_scenario_${scenario}_lint" \
      "${PYTHON_BIN}" tools/benchmark_sweep.py \
      --lint-scenarios \
      --out-root "${scenario_out}" \
      --scenario "${scenario}" \
      --review-mode release \
      --warmups 3 \
      --repeats 9 \
      --seed 20260606 \
      "${CDNA_SWEEP_ARGV[@]}"
    if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
      mkdir -p "${scenario_out}"
      printf 'dry-run rank scenario lint %s\n' "${scenario}" >"${scenario_out}/rank-scenario-lint-plan.log"
    else
      (cd "${CDNA_REPO_ROOT}" && \
        "${PYTHON_BIN}" tools/benchmark_sweep.py \
          --lint-scenarios \
          --out-root "${scenario_out}" \
          --scenario "${scenario}" \
          --review-mode release \
          --warmups 3 \
          --repeats 9 \
          --seed 20260606 \
          "${CDNA_SWEEP_ARGV[@]}")
    fi
    cdna_note_command "rank_scenario_${scenario}" env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" \
      "${PYTHON_BIN}" tools/benchmark_sweep.py \
      --bench "${BENCH_BIN}" \
      --out-root "${scenario_out}" \
      --scenario "${scenario}" \
      --review-mode release \
      --warmups 3 \
      --repeats 9 \
      --seed 20260606 \
      "${CDNA_REVIEW_ISA_ARGV[@]}" \
      "${CDNA_SWEEP_ARGV[@]}" \
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
          "${CDNA_REVIEW_ISA_ARGV[@]}" \
          "${CDNA_SWEEP_ARGV[@]}" \
          --skip-existing)
    fi
  done
fi

echo "CDNA first-pass output: ${CDNA_OUT_DIR}"
