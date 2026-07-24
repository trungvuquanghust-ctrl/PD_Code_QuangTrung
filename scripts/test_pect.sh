#!/usr/bin/env bash

# Usage:
#   bash scripts/test_pect.sh DATASET_PATH WEIGHTS [SHOT] [GPU] [PECT_ARGS...]
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/test_pect.sh DATASET_PATH WEIGHTS [SHOT] [GPU] [PECT_ARGS...]" >&2
  exit 2
fi

DATASET_PATH="$1"
WEIGHTS="$2"
shift 2
SHOT="${1:-1}"
[[ $# -gt 0 ]] && shift
GPU="${1:-0}"
[[ $# -gt 0 ]] && shift

DATASET_NAME="${DATASET_NAME:-knee_aug_split}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FINAL_TEST_SEED="${FINAL_TEST_SEED:-200042}"

require_directory "${DATASET_PATH}" "dataset"
require_file "${WEIGHTS}" "checkpoint"

command=(
  "${TIM_PYTHON}" "${PROJECT_ROOT}/pect.py"
  --mode test
  --weights "${WEIGHTS}"
  --dataset-path "${DATASET_PATH}"
  --dataset-name "${DATASET_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --shot "${SHOT}"
  --gpu "${GPU}"
  --num-workers "${NUM_WORKERS}"
  --final-test-seed "${FINAL_TEST_SEED}"
)

if [[ -n "${RUN_NAME:-}" ]]; then
  command+=(--run-name "${RUN_NAME}")
fi

command+=("$@")
printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
