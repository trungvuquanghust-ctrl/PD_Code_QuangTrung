#!/usr/bin/env bash

# Shared path/runtime resolution for TIM_2026 launchers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  TIM_PYTHON="${PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  TIM_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  TIM_PYTHON="python3"
fi

require_directory() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Error: ${label} directory does not exist: ${path}" >&2
    exit 2
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Error: ${label} file does not exist: ${path}" >&2
    exit 2
  fi
}
