#!/usr/bin/env bash
set -uo pipefail

# Run extract/review, translation/export, and database build in isolated
# processes. The GPU holder starts after success, failure, or interruption.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-redbird}"
CONFIG="${CONFIG:-configs/sdg13_china_us_mexico_local_2_each_qwen25_7b_bf16.yaml}"
FORMATS="${FORMATS:-jsonl,csv}"
GPU="${GPU:-0}"
TARGET_UTIL="${TARGET_UTIL:-60}"
TARGET_MEM_PERCENT="${TARGET_MEM_PERCENT:-40}"
TARGET_MEM_GB="${TARGET_MEM_GB:-}"
CYCLE_MS="${CYCLE_MS:-200}"
SAMPLE_SEC="${SAMPLE_SEC:-2}"
ZHANKA_SCRIPT="${ZHANKA_SCRIPT:-${WORKSPACE_ROOT}/zhanka/占卡.py}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/qwen25_7b_bf16_2_each_$(date +%Y%m%d_%H%M%S).log}"

TASK_STATUS=0
HOLD_STARTED=0

hold_gpu() {
  local reason="${1:-finished}"
  if [[ "${HOLD_STARTED}" == "1" ]]; then
    return
  fi
  HOLD_STARTED=1
  echo "[INFO] Pipeline ended. reason=${reason} status=${TASK_STATUS}"
  echo "[INFO] Starting GPU hold on GPU ${GPU}. Press Ctrl+C to stop it."
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found; GPU hold cannot start." >&2
    exit "${TASK_STATUS}"
  fi
  if [[ ! -f "${ZHANKA_SCRIPT}" ]]; then
    echo "[ERROR] GPU hold script not found: ${ZHANKA_SCRIPT}" >&2
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
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${ZHANKA_SCRIPT}" "${hold_args[@]}"
  exit "${TASK_STATUS}"
}

on_interrupt() {
  TASK_STATUS=130
  hold_gpu "interrupted"
}

trap on_interrupt INT TERM
trap 'hold_gpu "exit"' EXIT

cd "${PROJECT_ROOT}" || {
  TASK_STATUS=1
  exit "${TASK_STATUS}"
}
mkdir -p "${LOG_DIR}"

echo "[INFO] Project: ${PROJECT_ROOT}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] GPU: ${GPU}"
echo "[INFO] Log: ${LOG_FILE}"

run_stage() {
  local stage="$1"
  shift
  echo "[INFO] Starting isolated stage: ${stage}"
  conda run --no-capture-output -n "${CONDA_ENV}" env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=src \
    PYTHONUNBUFFERED=1 \
    python -m sdg_extract.cli "$@" 2>&1 | tee -a "${LOG_FILE}"
  local status=${PIPESTATUS[0]}
  if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR] Stage ${stage} failed with status ${status}." | tee -a "${LOG_FILE}" >&2
  fi
  return "${status}"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH." >&2
  TASK_STATUS=127
elif [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${PROJECT_ROOT}/${CONFIG}" >&2
  TASK_STATUS=2
elif pgrep -f "${ZHANKA_SCRIPT}" >/dev/null 2>&1; then
  echo "[ERROR] A GPU holder is already running. Stop it before loading the model." >&2
  TASK_STATUS=3
  HOLD_STARTED=1
else
  : > "${LOG_FILE}"
  if run_stage "extract_review" extract --config "${CONFIG}"; then
    if run_stage "aggregate_translate_export" export \
      --config "${CONFIG}" --format "${FORMATS}"; then
      run_stage "database_build" db-build --config "${CONFIG}"
      TASK_STATUS=$?
    else
      TASK_STATUS=$?
    fi
  else
    TASK_STATUS=$?
  fi
fi

if [[ "${TASK_STATUS}" -eq 0 ]]; then
  echo "[INFO] Red Bird pipeline completed successfully."
else
  echo "[ERROR] Red Bird pipeline failed with status ${TASK_STATUS}." >&2
  echo "[ERROR] Check the log: ${LOG_FILE}" >&2
fi

hold_gpu "pipeline_complete"
exit "${TASK_STATUS}"
