#!/usr/bin/env bash

# Usage:
#   bash scripts/run_ablations.sh DATASET_PATH [GPU] [VARIANTS] [SAMPLES] [SHOTS] [SEEDS] [RUNNER_ARGS...]
#
# Example:
#   bash scripts/run_ablations.sh /datasets/scalogram_27_1 0 \
#     pect_no_global,pect_cost_only 60,240 1,5 42 --dry-run
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_ablations.sh DATASET_PATH [GPU] [VARIANTS] [SAMPLES] [SHOTS] [SEEDS] [RUNNER_ARGS...]" >&2
  exit 2
fi

DATASET_PATH="$1"
shift
GPU="${1:-0}"
[[ $# -gt 0 ]] && shift
VARIANTS="${1:-all}"
[[ $# -gt 0 ]] && shift
SAMPLES="${1:-60,160,240,all}"
[[ $# -gt 0 ]] && shift
SHOTS="${1:-1,5}"
[[ $# -gt 0 ]] && shift
SEEDS="${1:-42}"
[[ $# -gt 0 ]] && shift

DATASET_NAME="${DATASET_NAME:-knee_aug_split}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/ablations}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FINAL_TEST_SEED="${FINAL_TEST_SEED:-200042}"

require_directory "${DATASET_PATH}" "dataset"

command=(
  "${TIM_PYTHON}" "${PROJECT_ROOT}/run_ablations.py"
  --dataset-path "${DATASET_PATH}"
  --dataset-name "${DATASET_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --gpu "${GPU}"
  --num-workers "${NUM_WORKERS}"
  --final-test-seed "${FINAL_TEST_SEED}"
  --variants "${VARIANTS}"
  --samples "${SAMPLES}"
  --shots "${SHOTS}"
  --seeds "${SEEDS}"
)

command+=("$@")
printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
