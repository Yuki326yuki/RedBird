#!/usr/bin/env bash
set -u

# Run Red Bird extraction/export, then always hold a GPU no matter whether the run
# succeeds, fails, or is interrupted.
#
# Examples:
#   ./scripts/run_strict_100_always_hold_gpu.sh
#   GPU=1 TARGET_UTIL=60 TARGET_MEM_PERCENT=40 ./scripts/run_strict_100_always_hold_gpu.sh
#   CONFIG=configs/sdg13_china_us_local.yaml ./scripts/run_strict_100_always_hold_gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-redbird}"
CONFIG="${CONFIG:-configs/sdg13_china_us_local_100_strict_qwen25_7b.yaml}"
FORMATS="${FORMATS:-jsonl,csv}"
GPU="${GPU:-0}"
TARGET_UTIL="${TARGET_UTIL:-60}"
TARGET_MEM_PERCENT="${TARGET_MEM_PERCENT:-40}"
TARGET_MEM_GB="${TARGET_MEM_GB:-}"
CYCLE_MS="${CYCLE_MS:-200}"
SAMPLE_SEC="${SAMPLE_SEC:-2}"
ZHANKA_SCRIPT="${ZHANKA_SCRIPT:-${WORKSPACE_ROOT}/zhanka/占卡.py}"

TASK_STATUS=0
HOLD_STARTED=0

hold_gpu() {
  local reason="${1:-finished}"
  if [[ "${HOLD_STARTED}" == "1" ]]; then
    return
  fi
  HOLD_STARTED=1

  echo "[INFO] Entering GPU hold. reason=${reason} previous_task_status=${TASK_STATUS}"

  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found in PATH; cannot start GPU hold." >&2
    exit "${TASK_STATUS}"
  fi

  if [[ ! -f "${ZHANKA_SCRIPT}" ]]; then
    echo "[ERROR] 占卡脚本 not found: ${ZHANKA_SCRIPT}" >&2
    exit "${TASK_STATUS}"
  fi

  local hold_args=(
    --gpu "${GPU}"
    --target-util "${TARGET_UTIL}"
    --target-mem-percent "${TARGET_MEM_PERCENT}"
    --cycle-ms "${CYCLE_MS}"
    --sample-sec "${SAMPLE_SEC}"
  )

  if [[ -n "${TARGET_MEM_GB}" ]]; then
    hold_args+=(--target-mem-gb "${TARGET_MEM_GB}")
  fi

  conda run --no-capture-output -n "${CONDA_ENV}" python "${ZHANKA_SCRIPT}" "${hold_args[@]}"
  exit "${TASK_STATUS}"
}

on_interrupt() {
  TASK_STATUS=130
  hold_gpu "interrupted"
}

trap on_interrupt INT TERM
trap 'hold_gpu "exit"' EXIT

cd "${PROJECT_ROOT}" || exit 1

echo "[INFO] Project: ${PROJECT_ROOT}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Export formats: ${FORMATS}"
echo "[INFO] GPU hold target: gpu=${GPU} util=${TARGET_UTIL}% mem_percent=${TARGET_MEM_PERCENT}%"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH." >&2
  TASK_STATUS=127
else
  if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config not found: ${PROJECT_ROOT}/${CONFIG}" >&2
    TASK_STATUS=2
  else
    echo "[INFO] Step 1/2: extract"
    conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONPATH=src python -m sdg_extract.cli extract --config "${CONFIG}"
    TASK_STATUS=$?

    if [[ "${TASK_STATUS}" -eq 0 ]]; then
      echo "[INFO] Step 2/2: export ${FORMATS}"
      conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONPATH=src python -m sdg_extract.cli export --config "${CONFIG}" --format "${FORMATS}"
      TASK_STATUS=$?
    else
      echo "[ERROR] Extract failed with status ${TASK_STATUS}; export skipped, GPU hold will still start." >&2
    fi
  fi
fi

hold_gpu "task_complete"
