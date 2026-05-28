#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/looped-tf-length-generalization/src

# =========================
# Formal Copy + Ponder sharpness evaluation
# =========================

# 你当前筛出来的正式 runs
NOPE_RUN_DIR="../models/nope_baselines/copy_ponder/62465a36-9f94-4157-8f93-edac50a7630d"
ROPE_RUN_DIR="../models/rope_baselines/copy_ponder/85c8ce6d-4abc-4432-a343-739683bc4bf7"

# 输出目录名
OUT_NAME="sharpness_formal"

# Sharpness 参数
LENGTHS="1-20,21,22,40,60"
EPSILONS="0,1e-4,3e-4,1e-3,3e-3,1e-2"
DIRECTIONS=8
N_BATCHES=20
BATCH_SIZE=128
CHECKPOINT="best.pt"

# 日志目录
LOG_DIR="../eval/copy_ponder_sharpness_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "======================================"
echo "Copy + Ponder sharpness formal eval"
echo "timestamp: ${TIMESTAMP}"
echo "lengths: ${LENGTHS}"
echo "epsilons: ${EPSILONS}"
echo "directions: ${DIRECTIONS}"
echo "n_batches: ${N_BATCHES}"
echo "batch_size: ${BATCH_SIZE}"
echo "======================================"
echo

run_one () {
  local NAME="$1"
  local RUN_DIR="$2"

  echo "======================================"
  echo "Running ${NAME}"
  echo "run_dir: ${RUN_DIR}"
  echo "out_dir: ${RUN_DIR}/${OUT_NAME}"
  echo "log: ${LOG_DIR}/${NAME}_${TIMESTAMP}.log"
  echo "======================================"

  if [ ! -d "${RUN_DIR}" ]; then
    echo "[ERROR] RUN_DIR does not exist: ${RUN_DIR}"
    exit 1
  fi

  if [ ! -f "${RUN_DIR}/${CHECKPOINT}" ]; then
    echo "[ERROR] checkpoint not found: ${RUN_DIR}/${CHECKPOINT}"
    exit 1
  fi

  if [ ! -f "${RUN_DIR}/config.yaml" ]; then
    echo "[ERROR] config.yaml not found: ${RUN_DIR}/config.yaml"
    exit 1
  fi

  python eval_ponder_sharpness.py \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --out-dir "${RUN_DIR}/${OUT_NAME}" \
    --lengths "${LENGTHS}" \
    --n-batches "${N_BATCHES}" \
    --batch-size "${BATCH_SIZE}" \
    --epsilons "${EPSILONS}" \
    --directions "${DIRECTIONS}" \
    2>&1 | tee "${LOG_DIR}/${NAME}_${TIMESTAMP}.log"

  echo
  echo "[DONE] ${NAME}"
  echo "summary: ${RUN_DIR}/${OUT_NAME}/sharpness_summary.csv"
  echo "raw:     ${RUN_DIR}/${OUT_NAME}/sharpness_raw.csv"
  echo "plot:    ${RUN_DIR}/${OUT_NAME}/sharpness_delta_objective.png"
  echo
}

run_one "nope" "${NOPE_RUN_DIR}"
run_one "rope" "${ROPE_RUN_DIR}"

echo "======================================"
echo "All sharpness evaluations finished."
echo "Logs saved to: ${LOG_DIR}"
echo "======================================"
