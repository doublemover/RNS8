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

summary_args=(
  python tools/cdna_env_summary.py
  --log-dir "${CDNA_OUT_DIR}"
  --devices "${CDNA_DEVICES}"
  --out "${CDNA_OUT_DIR}/cdna-env-summary.json"
)
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  summary_args+=(--dry-run)
fi
"${summary_args[@]}"

echo "CDNA env probe summary: ${CDNA_OUT_DIR}/cdna-env-summary.json"
