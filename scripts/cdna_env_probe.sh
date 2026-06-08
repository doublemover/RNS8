#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cdna_common.sh
source "${SCRIPT_DIR}/cdna_common.sh"

CDNA_OUT_DIR="temp/cdna-first-pass/$(cdna_timestamp)/env"
cdna_parse_common_args "$@"
cd "${CDNA_REPO_ROOT}"

PYTHON_BIN="$(cdna_python_bin)"

mkdir -p "${CDNA_OUT_DIR}"
: >"$(cdna_plan_file)"
cdna_resolve_cmake_tools "${PYTHON_BIN}"

cdna_repo_capture env env || true
cdna_repo_capture uname uname -a || true
cdna_repo_capture lscpu lscpu || true
cdna_repo_capture numactl_hardware numactl --hardware || true
cdna_repo_capture lstopo lstopo --of console || true
cdna_repo_capture hipconfig_full hipconfig --full || true
cdna_repo_capture hipcc_version hipcc --version || true
cdna_repo_capture rocm_version_files bash -lc 'set -u; seen=""; for root in "${ROCM_PATH:-}" "${HIP_PATH:-}" /opt/rocm; do [[ -n "$root" && -d "$root" ]] || continue; case ":$seen:" in *":$root:"*) continue ;; esac; seen="${seen:+$seen:}$root"; for name in version version-dev version-utils; do path="$root/.info/$name"; if [[ -r "$path" ]]; then printf "%s=" "$path"; tr -d "\r\n" <"$path"; printf "\n"; fi; done; done' || true
cdna_repo_capture rocm_package_versions bash -lc 'set -u; if command -v dpkg-query >/dev/null 2>&1; then dpkg-query -W -f="\${Package} \${Version}\n" rocm-core hip-runtime-amd hipcc rocminfo rocm-smi-lib rocm-dev 2>/dev/null || true; elif command -v rpm >/dev/null 2>&1; then rpm -q --qf "%{NAME} %{VERSION}-%{RELEASE}\n" rocm-core hip-runtime-amd hipcc rocminfo rocm-smi-lib rocm-dev 2>/dev/null || true; fi' || true
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
cdna_repo_capture cmake_presets "${CDNA_CMAKE_BIN}" --list-presets || true
cdna_repo_capture check_dependencies_json "${PYTHON_BIN}" tools/check_dependencies.py --json || true
cdna_repo_capture rccl_discovery bash -lc 'set -u; for p in /opt/rocm/include/rccl/rccl.h /opt/rocm/include/rccl.h /opt/rocm/lib/librccl.so /opt/rocm/lib64/librccl.so; do if [[ -e "$p" ]]; then echo "$p"; fi; done; for t in all_reduce_perf all_gather_perf broadcast_perf reduce_scatter_perf; do if command -v "$t" >/dev/null 2>&1; then echo "$t=$(command -v "$t")"; else echo "$t=not-found"; fi; done' || true
if [[ -f "${CDNA_REPO_ROOT}/temp/amd_matrix_instruction_calculator/matrix_calculator.py" ]]; then
  cdna_repo_capture amd_matrix_instruction_report \
    "${PYTHON_BIN}" tools/amd_matrix_instruction_report.py \
    --calculator temp/amd_matrix_instruction_calculator/matrix_calculator.py \
    --architectures gfx942,gfx1100 \
    --out-dir "${CDNA_OUT_DIR}/amd-matrix-instructions" \
    --markdown || true
else
  printf 'AMD matrix instruction calculator not found at temp/amd_matrix_instruction_calculator/matrix_calculator.py\n' >"${CDNA_OUT_DIR}/amd_matrix_instruction_report.log"
fi

summary_args=(
  "${PYTHON_BIN}" tools/cdna_env_summary.py
  --log-dir "${CDNA_OUT_DIR}"
  --devices "${CDNA_DEVICES}"
  --out "${CDNA_OUT_DIR}/cdna-env-summary.json"
)
if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
  summary_args+=(--dry-run)
fi
"${summary_args[@]}"

echo "CDNA env probe summary: ${CDNA_OUT_DIR}/cdna-env-summary.json"
