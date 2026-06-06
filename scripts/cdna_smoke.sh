#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cdna_common.sh
source "${SCRIPT_DIR}/cdna_common.sh"

CDNA_OUT_DIR="temp/cdna-smoke/$(cdna_timestamp)"
cdna_parse_common_args "$@"
cd "${CDNA_REPO_ROOT}"

mkdir -p "${CDNA_OUT_DIR}"
: >"$(cdna_plan_file)"

PRESET="$(cdna_default_preset)"
DEVICES="$(cdna_discover_devices)"
DEVICE="$(cdna_first_device "${DEVICES}")"
ENV_DIR="${CDNA_OUT_DIR}/env"
CAPTURE="${CDNA_OUT_DIR}/bounded-i64-hip-direct-smoke.json"
SCHEMA_LOG="${CDNA_OUT_DIR}/benchmark-schema.log"
VERIFY_BIN="$(cdna_binary_path "${PRESET}" "rns8-verify")"
BENCH_BIN="$(cdna_binary_path "${PRESET}" "rns8-bench")"

ENV_PROBE_ARGS=(--out-dir "${ENV_DIR}" --devices "${DEVICE}")
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  ENV_PROBE_ARGS+=(--dry-run)
fi
if [[ "${CDNA_ACCELERATORS}" -eq 1 ]]; then
  ENV_PROBE_ARGS+=(--accelerators)
fi
cdna_repo_run env_probe "${SCRIPT_DIR}/cdna_env_probe.sh" "${ENV_PROBE_ARGS[@]}"

if [[ "${CDNA_SKIP_BUILD}" -eq 0 ]]; then
  cdna_repo_run configure cmake --preset "${PRESET}"
  cdna_repo_run build cmake --build --preset "${PRESET}"
  cdna_repo_run ctest ctest --preset "${PRESET}" --output-on-failure
fi

cdna_repo_run hip_smoke env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${VERIFY_BIN}" --hip-smoke

cdna_default_bench_command "${BENCH_BIN}"
cdna_note_command rns8_bench_capture env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_DEFAULT_BENCH_CMD[@]}"
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  printf '{"dry_run": true, "planned_capture": "%s"}\n' "${CAPTURE}" >"${CAPTURE}"
  printf 'dry-run schema validation for %s\n' "${CAPTURE}" >"${SCHEMA_LOG}"
else
  (cd "${CDNA_REPO_ROOT}" && env ROCR_VISIBLE_DEVICES="${DEVICE}" HIP_VISIBLE_DEVICES="${DEVICE}" "${CDNA_DEFAULT_BENCH_CMD[@]}") >"${CAPTURE}"
  (cd "${CDNA_REPO_ROOT}" && python tools/benchmark_schema.py "${CAPTURE}") >"${SCHEMA_LOG}" 2>&1
fi

echo "CDNA smoke output: ${CDNA_OUT_DIR}"
