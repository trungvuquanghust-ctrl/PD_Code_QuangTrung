#!/usr/bin/env bash

# Usage:
#   bash scripts/train_pect.sh DATASET_PATH [SHOT] [SAMPLES] [GPU] [PECT_ARGS...]
#
# Examples:
#   bash scripts/train_pect.sh /datasets/scalogram_27_1 1 60 0
#   bash scripts/train_pect.sh /datasets/scalogram_27_1 5 all 1 --seed 123
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_pect.sh DATASET_PATH [SHOT] [SAMPLES|all] [GPU] [PECT_ARGS...]" >&2
  exit 2
fi

DATASET_PATH="$1"
shift
SHOT="${1:-1}"
[[ $# -gt 0 ]] && shift
SAMPLES="${1:-all}"
[[ $# -gt 0 ]] && shift
GPU="${1:-0}"
[[ $# -gt 0 ]] && shift

DATASET_NAME="${DATASET_NAME:-knee_aug_split}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FINAL_TEST_SEED="${FINAL_TEST_SEED:-200042}"

require_directory "${DATASET_PATH}" "dataset"

command=(
  "${TIM_PYTHON}" "${PROJECT_ROOT}/pect.py"
  --dataset-path "${DATASET_PATH}"
  --dataset-name "${DATASET_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --shot "${SHOT}"
  --gpu "${GPU}"
  --num-workers "${NUM_WORKERS}"
  --final-test-seed "${FINAL_TEST_SEED}"
)

if [[ "${SAMPLES,,}" != "all" && "${SAMPLES,,}" != "none" ]]; then
  command+=(--training-samples "${SAMPLES}")
fi
if [[ -n "${RUN_NAME:-}" ]]; then
  command+=(--run-name "${RUN_NAME}")
fi

command+=("$@")
printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
