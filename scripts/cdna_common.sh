#!/usr/bin/env bash

set -uo pipefail

CDNA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDNA_REPO_ROOT="$(cd "${CDNA_SCRIPT_DIR}/.." && pwd)"

CDNA_OUT_DIR=""
CDNA_PRESET=""
CDNA_DEVICES=""
CDNA_BENCH_ARGS=""
CDNA_RANK_SCENARIOS=""
CDNA_SKIP_BUILD=0
CDNA_DRY_RUN=0
CDNA_ACCELERATORS=0
CDNA_BENCH_ARGV=()

cdna_timestamp() {
  date -u +%Y%m%dT%H%M%SZ
}

cdna_usage_common() {
  cat <<'EOF'
Common options:
  --out-dir DIR       output directory
  --preset NAME      CMake configure/build/test preset
  --devices LIST     comma-separated physical GPU ids, for example 0,1,2,3
  --bench-args ARGS  extra rns8-bench arguments appended to the smoke command
  --rank-scenarios LIST
                      optional comma-separated benchmark_sweep scenario groups
                      to run after the first smoke, for example
                      wrap64-carry,k-block-tile-variants,layout-search,
                      finite-distributions,vector-to-rns-chain
  --accelerators     use linux-cdna-accelerators-release unless --preset is set
  --skip-build       skip configure/build/CTest steps
  --dry-run          print and record planned commands without executing them
EOF
}

cdna_parse_common_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --out-dir)
        CDNA_OUT_DIR="${2:-}"
        shift 2
        ;;
      --preset)
        CDNA_PRESET="${2:-}"
        shift 2
        ;;
      --devices)
        CDNA_DEVICES="${2:-}"
        shift 2
        ;;
      --bench-args)
        CDNA_BENCH_ARGS="${2:-}"
        shift 2
        ;;
      --rank-scenarios)
        CDNA_RANK_SCENARIOS="${2:-}"
        shift 2
        ;;
      --accelerators)
        CDNA_ACCELERATORS=1
        shift
        ;;
      --skip-build)
        CDNA_SKIP_BUILD=1
        shift
        ;;
      --dry-run)
        CDNA_DRY_RUN=1
        shift
        ;;
      -h|--help)
        cdna_usage_common
        exit 0
        ;;
      *)
        echo "unknown option: $1" >&2
        cdna_usage_common >&2
        exit 2
        ;;
    esac
  done
  if [[ -n "${CDNA_BENCH_ARGS}" ]]; then
    # shellcheck disable=SC2206
    CDNA_BENCH_ARGV=(${CDNA_BENCH_ARGS})
  fi
}

cdna_default_preset() {
  if [[ -n "${CDNA_PRESET}" ]]; then
    printf '%s\n' "${CDNA_PRESET}"
  elif [[ "${CDNA_ACCELERATORS}" -eq 1 ]]; then
    printf '%s\n' "linux-cdna-accelerators-release"
  else
    printf '%s\n' "linux-cdna-release"
  fi
}

cdna_python_bin() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "${PYTHON}"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    printf '%s\n' "python3"
  fi
}

cdna_command_line() {
  printf '%q ' "$@"
}

cdna_plan_file() {
  printf '%s\n' "${CDNA_OUT_DIR}/command-plan.txt"
}

cdna_note_command() {
  local label="$1"
  shift
  mkdir -p "${CDNA_OUT_DIR}"
  {
    printf '[%s] ' "${label}"
    cdna_command_line "$@"
    printf '\n'
  } >>"$(cdna_plan_file)"
}

cdna_run() {
  local label="$1"
  shift
  cdna_note_command "${label}" "$@"
  if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

cdna_run_capture() {
  local label="$1"
  shift
  local log_path="${CDNA_OUT_DIR}/${label}.log"
  cdna_note_command "${label}" "$@"
  if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
    printf 'dry-run: ' >"${log_path}"
    cdna_command_line "$@" >>"${log_path}"
    printf '\n' >>"${log_path}"
    return 0
  fi
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'command not found: %s\n' "$1" >"${log_path}"
    return 127
  fi
  "$@" >"${log_path}" 2>&1
}

cdna_repo_run() {
  (cd "${CDNA_REPO_ROOT}" && cdna_run "$@")
}

cdna_repo_run_artifact_command() {
  local label="$1"
  shift
  cdna_note_command "${label}" "$@"
  (cd "${CDNA_REPO_ROOT}" && "$@")
}

cdna_repo_capture() {
  (cd "${CDNA_REPO_ROOT}" && cdna_run_capture "$@")
}

cdna_split_devices() {
  local list="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<<"${list}"
  local cleaned=()
  local item
  for item in "${out_ref[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      cleaned+=("${item}")
    fi
  done
  out_ref=("${cleaned[@]}")
}

cdna_discover_devices() {
  if [[ -n "${CDNA_DEVICES}" ]]; then
    printf '%s\n' "${CDNA_DEVICES}"
    return 0
  fi
  if command -v rocm-smi >/dev/null 2>&1; then
    local ids
    ids="$(rocm-smi --showid 2>/dev/null | sed -n 's/^GPU\[\([0-9][0-9]*\)\].*/\1/p' | paste -sd, -)"
    if [[ -n "${ids}" ]]; then
      printf '%s\n' "${ids}"
      return 0
    fi
  fi
  if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "${HIP_VISIBLE_DEVICES}"
    return 0
  fi
  if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "${ROCR_VISIBLE_DEVICES}"
    return 0
  fi
  printf '%s\n' "0"
}

cdna_first_device() {
  local devices="$1"
  local parsed=()
  cdna_split_devices "${devices}" parsed
  printf '%s\n' "${parsed[0]:-0}"
}

cdna_build_dir_for_preset() {
  printf '%s/build/%s\n' "${CDNA_REPO_ROOT}" "$1"
}

cdna_binary_path() {
  local preset="$1"
  local binary="$2"
  printf '%s/%s\n' "$(cdna_build_dir_for_preset "${preset}")" "${binary}"
}

cdna_default_bench_command() {
  local binary="$1"
  shift
  CDNA_DEFAULT_BENCH_CMD=(
    "${binary}"
    --backend hip-direct
    --semantics bounded-i64
    --m 64
    --n 64
    --k 64
    --warmups 1
    --repeats 1
    --seed 1
    --device 0
  )
  if [[ "${#CDNA_BENCH_ARGV[@]}" -gt 0 ]]; then
    CDNA_DEFAULT_BENCH_CMD+=("${CDNA_BENCH_ARGV[@]}")
  fi
}
