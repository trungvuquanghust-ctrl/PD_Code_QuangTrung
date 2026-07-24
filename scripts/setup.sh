#!/usr/bin/env bash

# Create a local virtual environment from the canonical requirements file.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${1:-${PROJECT_ROOT}/.venv}"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-python3}"

echo "Project: ${PROJECT_ROOT}"
echo "Venv   : ${VENV_PATH}"
"${BOOTSTRAP_PYTHON}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "Environment ready. Activate with:"
echo "  source \"${VENV_PATH}/bin/activate\""
