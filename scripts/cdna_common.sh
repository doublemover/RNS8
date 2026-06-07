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
CDNA_CMAKE_MIN_VERSION="3.28.0"
CDNA_CATCH2_VERSION="${CDNA_CATCH2_VERSION:-v3.5.4}"
CDNA_CMAKE_BIN=""
CDNA_CTEST_BIN=""
CDNA_NINJA_BIN=""

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

cdna_cmake_version() {
  local cmake_bin="$1"
  "${cmake_bin}" --version 2>/dev/null | sed -n '1s/^cmake version //p'
}

cdna_version_ge() {
  local python_bin="$1"
  local actual="$2"
  local minimum="$3"
  "${python_bin}" - "${actual}" "${minimum}" <<'PY'
from __future__ import annotations

import re
import sys


def parse(value: str) -> tuple[int, int, int]:
    match = re.match(r"^([0-9]+)(?:\.([0-9]+))?(?:\.([0-9]+))?", value)
    if not match:
        return (0, 0, 0)
    return tuple(int(part or "0") for part in match.groups())


raise SystemExit(0 if parse(sys.argv[1]) >= parse(sys.argv[2]) else 1)
PY
}

cdna_accept_cmake_pair() {
  local python_bin="$1"
  local cmake_bin="$2"
  local cmake_version
  cmake_version="$(cdna_cmake_version "${cmake_bin}")"
  if [[ -z "${cmake_version}" ]]; then
    return 1
  fi
  if ! cdna_version_ge "${python_bin}" "${cmake_version}" "${CDNA_CMAKE_MIN_VERSION}"; then
    return 1
  fi

  local cmake_dir
  cmake_dir="$(cd "$(dirname "${cmake_bin}")" && pwd)"
  CDNA_CMAKE_BIN="${cmake_bin}"
  if [[ -n "${CDNA_CTEST:-}" ]]; then
    CDNA_CTEST_BIN="${CDNA_CTEST}"
  elif [[ -x "${cmake_dir}/ctest" ]]; then
    CDNA_CTEST_BIN="${cmake_dir}/ctest"
  elif command -v ctest >/dev/null 2>&1; then
    CDNA_CTEST_BIN="$(command -v ctest)"
  else
    CDNA_CTEST_BIN="ctest"
  fi
  return 0
}

cdna_pip_target_cmake_bin() {
  local python_bin="$1"
  local target_dir="$2"
  [[ -d "${target_dir}" ]] || return 1
  "${python_bin}" - "${target_dir}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
try:
    import cmake
except Exception:
    raise SystemExit(1)

print(Path(cmake.CMAKE_BIN_DIR) / "cmake")
PY
}

cdna_prepend_path_dir() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 0
  case ":${PATH}:" in
    *":${dir}:"*) ;;
    *) export PATH="${dir}:${PATH}" ;;
  esac
}

cdna_resolve_cmake_tools() {
  local python_bin="$1"
  local tool_root="${CDNA_REPO_ROOT}/temp/cdna-tools/cmake-${CDNA_CMAKE_MIN_VERSION}"
  local target_site="${tool_root}/python-packages"
  local candidate
  local target_candidate
  local candidates=()

  if [[ -n "${CDNA_CMAKE:-}" ]]; then
    candidates+=("${CDNA_CMAKE}")
  fi
  if command -v cmake >/dev/null 2>&1; then
    candidates+=("$(command -v cmake)")
  fi
  candidates+=("${tool_root}/bin/cmake")
  target_candidate="$(cdna_pip_target_cmake_bin "${python_bin}" "${target_site}" 2>/dev/null || true)"
  if [[ -n "${target_candidate}" ]]; then
    candidates+=("${target_candidate}")
  fi
  cdna_prepend_path_dir "${target_site}/bin"

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]] && cdna_accept_cmake_pair "${python_bin}" "${candidate}"; then
      cdna_prepend_path_dir "$(dirname "${CDNA_CMAKE_BIN}")"
      return 0
    fi
  done

  if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
    CDNA_CMAKE_BIN="${candidates[0]:-cmake}"
    CDNA_CTEST_BIN="${CDNA_CTEST:-ctest}"
    return 0
  fi

  mkdir -p "$(dirname "${tool_root}")"
  cdna_note_command bootstrap_cmake_venv "${python_bin}" -m venv "${tool_root}"
  if "${python_bin}" -m venv "${tool_root}"; then
    local venv_python="${tool_root}/bin/python"
    cdna_note_command bootstrap_cmake_pip "${venv_python}" -m pip install --upgrade "pip" "cmake>=${CDNA_CMAKE_MIN_VERSION},<4" ninja
    "${venv_python}" -m pip install --upgrade "pip" "cmake>=${CDNA_CMAKE_MIN_VERSION},<4" ninja
    if cdna_accept_cmake_pair "${python_bin}" "${tool_root}/bin/cmake"; then
      cdna_prepend_path_dir "$(dirname "${CDNA_CMAKE_BIN}")"
      return 0
    fi
  else
    printf 'warning: python venv bootstrap failed; falling back to pip --target under temp/cdna-tools.\n' >&2
  fi

  mkdir -p "${target_site}"
  cdna_note_command bootstrap_cmake_pip_target "${python_bin}" -m pip install --upgrade --target "${target_site}" "cmake>=${CDNA_CMAKE_MIN_VERSION},<4" ninja
  "${python_bin}" -m pip install --upgrade --target "${target_site}" "cmake>=${CDNA_CMAKE_MIN_VERSION},<4" ninja
  cdna_prepend_path_dir "${target_site}/bin"

  target_candidate="$(cdna_pip_target_cmake_bin "${python_bin}" "${target_site}")"
  if ! cdna_accept_cmake_pair "${python_bin}" "${target_candidate}"; then
    {
      printf 'error: CMake %s or newer is required, but no usable cmake was found.\n' "${CDNA_CMAKE_MIN_VERSION}"
      printf '       System cmake: %s\n' "$(command -v cmake || printf 'not found')"
      printf '       Venv cmake: %s\n' "${tool_root}/bin/cmake"
      printf '       Pip-target cmake: %s\n' "${target_candidate}"
    } >&2
    return 1
  fi

  cdna_prepend_path_dir "$(dirname "${CDNA_CMAKE_BIN}")"
}

cdna_resolve_ninja() {
  if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
    CDNA_NINJA_BIN="${CDNA_NINJA:-ninja}"
    return 0
  fi

  local candidate
  local candidates=()
  if [[ -n "${CDNA_NINJA:-}" ]]; then
    candidates+=("${CDNA_NINJA}")
  fi
  if command -v ninja >/dev/null 2>&1; then
    candidates+=("$(command -v ninja)")
  fi
  if command -v ninja-build >/dev/null 2>&1; then
    candidates+=("$(command -v ninja-build)")
  fi

  for candidate in "${candidates[@]}"; do
    local executable=""
    if [[ -x "${candidate}" ]]; then
      executable="${candidate}"
    elif command -v "${candidate}" >/dev/null 2>&1; then
      executable="$(command -v "${candidate}")"
    fi
    if [[ -n "${executable}" ]] && "${executable}" --version >/dev/null 2>&1; then
      CDNA_NINJA_BIN="${executable}"
      return 0
    fi
  done

  {
    printf 'error: Ninja is required by the Linux CDNA CMake presets, but no usable ninja executable was found.\n'
    printf '       Install ninja-build, or set CDNA_NINJA to the ninja executable.\n'
  } >&2
  return 1
}

cdna_prepend_cmake_prefix() {
  local prefix="$1"
  if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    case ":${CMAKE_PREFIX_PATH}:" in
      *":${prefix}:"*) ;;
      *) export CMAKE_PREFIX_PATH="${prefix}:${CMAKE_PREFIX_PATH}" ;;
    esac
  else
    export CMAKE_PREFIX_PATH="${prefix}"
  fi
}

cdna_probe_catch2_v3() {
  local probe_root="${CDNA_REPO_ROOT}/temp/cdna-tools/probes/catch2-v3"
  local probe_src="${probe_root}/src"
  local probe_build="${probe_root}/build-$$-${RANDOM}"
  local probe_log="${probe_root}/probe.log"
  mkdir -p "${probe_src}" "${probe_build}"
  cat >"${probe_src}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.28)
project(rns8_catch2_probe LANGUAGES CXX)
find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)
EOF
  "${CDNA_CMAKE_BIN}" -S "${probe_src}" -B "${probe_build}" >"${probe_log}" 2>&1
}

cdna_resolve_catch2() {
  local install_root="${CDNA_REPO_ROOT}/temp/cdna-tools/catch2-${CDNA_CATCH2_VERSION}/install"
  local source_root="${CDNA_REPO_ROOT}/temp/cdna-tools/catch2-${CDNA_CATCH2_VERSION}/src"
  local build_root="${CDNA_REPO_ROOT}/temp/cdna-tools/catch2-${CDNA_CATCH2_VERSION}/build"

  if [[ -d "${install_root}" ]]; then
    cdna_prepend_cmake_prefix "${install_root}"
  fi
  if cdna_probe_catch2_v3; then
    return 0
  fi
  if [[ "${CDNA_DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  mkdir -p "$(dirname "${source_root}")"
  if [[ ! -d "${source_root}/.git" ]]; then
    cdna_note_command bootstrap_catch2_clone git clone --depth 1 --branch "${CDNA_CATCH2_VERSION}" https://github.com/catchorg/Catch2.git "${source_root}"
    git clone --depth 1 --branch "${CDNA_CATCH2_VERSION}" https://github.com/catchorg/Catch2.git "${source_root}"
  fi

  cdna_note_command bootstrap_catch2_configure "${CDNA_CMAKE_BIN}" -S "${source_root}" -B "${build_root}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${install_root}" -DCATCH_INSTALL_DOCS=OFF -DCATCH_INSTALL_EXTRAS=ON -DBUILD_TESTING=OFF
  "${CDNA_CMAKE_BIN}" -S "${source_root}" -B "${build_root}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${install_root}" -DCATCH_INSTALL_DOCS=OFF -DCATCH_INSTALL_EXTRAS=ON -DBUILD_TESTING=OFF

  cdna_note_command bootstrap_catch2_install "${CDNA_CMAKE_BIN}" --build "${build_root}" --target install
  "${CDNA_CMAKE_BIN}" --build "${build_root}" --target install

  cdna_prepend_cmake_prefix "${install_root}"
  if ! cdna_probe_catch2_v3; then
    printf 'error: Catch2 3 CMake package is still unavailable after bootstrap. See temp/cdna-tools/probes/catch2-v3/probe.log\n' >&2
    return 1
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
