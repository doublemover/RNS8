#!/usr/bin/env bash
# Existing CDNA entrypoints share the bounded Python qualification runner.
# No discovery command, package install or hardware probe runs in this shell.
set -euo pipefail

CDNA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDNA_REPO_ROOT="$(cd "${CDNA_SCRIPT_DIR}/.." && pwd)"

cdna_session() {
  local kind="$1"
  shift
  local python_bin="${PYTHON:-}"
  if [[ -z "${python_bin}" ]]; then
    python_bin="$(command -v python3 || command -v python || true)"
  fi
  if [[ -z "${python_bin}" ]]; then
    echo "A readable Python interpreter is required; select it with PYTHON=/path/to/python." >&2
    return 2
  fi
  local default_backend=(--backend hip-direct)
  local option
  for option in "$@"; do
    if [[ "${option}" == --backend || "${option}" == --backend=* || "${option}" == --accelerators ]]; then
      default_backend=()
    fi
  done
  cd "${CDNA_REPO_ROOT}"
  exec "${python_bin}" tools/gpu_qualification.py --session-kind "${kind}" "${default_backend[@]}" "$@"
}
