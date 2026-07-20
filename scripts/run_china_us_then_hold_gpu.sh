#!/usr/bin/env bash
set -euo pipefail

# Run the China/US SDG13 extraction experiment, export results, then hold a GPU.
# Defaults can be overridden with environment variables, for example:
#   GPU=1 TARGET_UTIL=60 TARGET_MEM_PERCENT=40 ./scripts/run_china_us_then_hold_gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-redbird}"
CONFIG="${CONFIG:-configs/sdg13_china_us_local.yaml}"
GPU="${GPU:-0}"
TARGET_UTIL="${TARGET_UTIL:-60}"
TARGET_MEM_PERCENT="${TARGET_MEM_PERCENT:-40}"
CYCLE_MS="${CYCLE_MS:-200}"
SAMPLE_SEC="${SAMPLE_SEC:-2}"
RECORDS_PATH="${RECORDS_PATH:-data/work/sdg13_china_us/local/records.jsonl}"
ZHANKA_SCRIPT="${ZHANKA_SCRIPT:-${WORKSPACE_ROOT}/zhanka/占卡.py}"

cd "${PROJECT_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${PROJECT_ROOT}/${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${ZHANKA_SCRIPT}" ]]; then
  echo "[ERROR] 占卡脚本 not found: ${ZHANKA_SCRIPT}" >&2
  exit 1
fi

echo "[INFO] Project: ${PROJECT_ROOT}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Step 1/3: extract"
conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONPATH=src python -m sdg_extract.cli extract --config "${CONFIG}"

if [[ ! -s "${RECORDS_PATH}" ]]; then
  echo "[ERROR] Extract finished but records file is missing or empty: ${PROJECT_ROOT}/${RECORDS_PATH}" >&2
  echo "[ERROR] Not starting GPU hold. Check extract logs first." >&2
  exit 1
fi

echo "[INFO] Step 2/3: export jsonl,csv"
conda run --no-capture-output -n "${CONDA_ENV}" env PYTHONPATH=src python -m sdg_extract.cli export --config "${CONFIG}" --format jsonl,csv

echo "[INFO] Step 3/3: hold GPU ${GPU}"
exec conda run --no-capture-output -n "${CONDA_ENV}" python "${ZHANKA_SCRIPT}" \
  --gpu "${GPU}" \
  --target-util "${TARGET_UTIL}" \
  --target-mem-percent "${TARGET_MEM_PERCENT}" \
  --cycle-ms "${CYCLE_MS}" \
  --sample-sec "${SAMPLE_SEC}"
