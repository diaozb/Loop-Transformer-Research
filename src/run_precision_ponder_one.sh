#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# General single-run launcher for LoopTF + PonderNet precision experiments
# ============================================================
#
# Usage:
#   bash run_precision_ponder_one_v2.sh <pe> <bits> <seed> <train_steps> [eval_every]
#
# Examples:
#   bash run_precision_ponder_one_v2.sh rope 32 0 100001
#   bash run_precision_ponder_one_v2.sh rope 8  0 100001
#   bash run_precision_ponder_one_v2.sh nope 32 0 100001
#
# Debug:
#   RUN_FINAL_DIAGNOSTICS=false OUT_DIR=../models/precision_ponder_debug \
#   bash run_precision_ponder_one_v2.sh rope 8 0 2000 500
#
# Important safe defaults:
#   PONDER_DYNAMIC_N=true
#   PONDER_MAX_STEPS_CAP=128
#   DIAG_MAX_STEPS=128
#   TEST_LEN=20
#   USE_BEST_CHECKPOINT_FOR_DIAGNOSTICS=false

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  grep '^#' "$0" | sed 's/^# //'
  exit 0
fi

PE="${1:-${PE:-rope}}"
BITS="${2:-${BITS:-32}}"
SEED_ARG="${3:-${SEED:-0}}"
TRAIN_STEPS_ARG="${4:-${TRAIN_STEPS:-100001}}"
EVAL_EVERY_ARG="${5:-${EVAL_EVERY:-1000}}"

if [[ "$PE" == "rope" ]]; then
  USE_ROPE=true
  USE_WPE=false
elif [[ "$PE" == "nope" ]]; then
  USE_ROPE=false
  USE_WPE=false
else
  echo "[error] PE must be 'rope' or 'nope', got: $PE"
  exit 1
fi

if [[ "$BITS" == "32" ]]; then
  Q_AFTER=false
else
  Q_AFTER=true
fi

TASK="${TASK:-copy}"
OUT_DIR="${OUT_DIR:-../models/precision_ponder}"
LOG_DIR="${LOG_DIR:-../logs/precision_ponder}"
WANDB_MODE="${WANDB_MODE:-online}"
USE_WANDB="${USE_WANDB:-true}"
RUN_FINAL_DIAGNOSTICS="${RUN_FINAL_DIAGNOSTICS:-true}"

# Avoid OOD checkpoint selection by default.
TEST_LEN="${TEST_LEN:-20}"
USE_BEST_CHECKPOINT_FOR_DIAGNOSTICS="${USE_BEST_CHECKPOINT_FOR_DIAGNOSTICS:-false}"

# PonderNet needs a finite technical horizon, but it should not be a fixed 20-loop cap.
PONDER_DYNAMIC_N="${PONDER_DYNAMIC_N:-true}"
PONDER_N_STEPS="${PONDER_N_STEPS:-128}"
PONDER_MAX_STEPS_CAP="${PONDER_MAX_STEPS_CAP:-128}"
PONDER_BETA="${PONDER_BETA:-0.01}"
PONDER_PRIOR_LAMBDA="${PONDER_PRIOR_LAMBDA:-0.2}"

# Diagnostics must be deep enough for OOD lengths up to 60.
DIAG_MAX_STEPS="${DIAG_MAX_STEPS:-128}"
DIAG_BATCH_SIZE="${DIAG_BATCH_SIZE:-128}"
DIAG_N_BATCHES="${DIAG_N_BATCHES:-4}"

# Cheap selected-length eval during training, printed every EVAL_EVERY steps and logged to W&B.
# Empty string disables it: TRAIN_EVAL_LENGTHS=""
TRAIN_EVAL_LENGTHS="${TRAIN_EVAL_LENGTHS:-20,21,40,60}"
TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-128}"
TRAIN_EVAL_N_BATCHES="${TRAIN_EVAL_N_BATCHES:-1}"

QUANT_EXCLUDE_NORM="${QUANT_EXCLUDE_NORM:-false}"
QUANT_SCOPE="${QUANT_SCOPE:-all}"

BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

UPLOAD_MODEL_ARTIFACT="${UPLOAD_MODEL_ARTIFACT:-false}"
UPLOAD_DIAGNOSTIC_ARTIFACT="${UPLOAD_DIAGNOSTIC_ARTIFACT:-false}"

mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${PE}_wbits${BITS}_seed${SEED_ARG}_steps${TRAIN_STEPS_ARG}_${STAMP}.log"

echo "================================================================================"
echo "[launcher] LoopTF + PonderNet precision run"
echo "[launcher] PE=$PE"
echo "[launcher] BITS=$BITS"
echo "[launcher] SEED=$SEED_ARG"
echo "[launcher] TRAIN_STEPS=$TRAIN_STEPS_ARG"
echo "[launcher] EVAL_EVERY=$EVAL_EVERY_ARG"
echo "[launcher] TEST_LEN=$TEST_LEN"
echo "[launcher] RUN_FINAL_DIAGNOSTICS=$RUN_FINAL_DIAGNOSTICS"
echo "[launcher] USE_BEST_CHECKPOINT_FOR_DIAGNOSTICS=$USE_BEST_CHECKPOINT_FOR_DIAGNOSTICS"
echo "[launcher] PONDER_DYNAMIC_N=$PONDER_DYNAMIC_N"
echo "[launcher] PONDER_N_STEPS=$PONDER_N_STEPS"
echo "[launcher] PONDER_MAX_STEPS_CAP=$PONDER_MAX_STEPS_CAP"
echo "[launcher] DIAG_MAX_STEPS=$DIAG_MAX_STEPS"
echo "[launcher] TRAIN_EVAL_LENGTHS=$TRAIN_EVAL_LENGTHS"
echo "[launcher] TRAIN_EVAL_BATCH_SIZE=$TRAIN_EVAL_BATCH_SIZE"
echo "[launcher] TRAIN_EVAL_N_BATCHES=$TRAIN_EVAL_N_BATCHES"
echo "[launcher] QUANT_EXCLUDE_NORM=$QUANT_EXCLUDE_NORM"
echo "[launcher] QUANT_SCOPE=$QUANT_SCOPE"
echo "[launcher] OUT_DIR=$OUT_DIR"
echo "[launcher] LOG_FILE=$LOG_FILE"
echo "================================================================================"

echo "[launcher] Checking train_ponder_quant.py syntax..."
python -m py_compile train_ponder_quant.py

TASK="$TASK" \
OUT_DIR="$OUT_DIR" \
SEED="$SEED_ARG" \
TRAIN_STEPS="$TRAIN_STEPS_ARG" \
EVAL_EVERY="$EVAL_EVERY_ARG" \
BATCH_SIZE="$BATCH_SIZE" \
EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
LEARNING_RATE="$LEARNING_RATE" \
WEIGHT_DECAY="$WEIGHT_DECAY" \
TEST_LEN="$TEST_LEN" \
RUN_FINAL_DIAGNOSTICS="$RUN_FINAL_DIAGNOSTICS" \
USE_BEST_CHECKPOINT_FOR_DIAGNOSTICS="$USE_BEST_CHECKPOINT_FOR_DIAGNOSTICS" \
USE_WANDB="$USE_WANDB" \
WANDB_MODE="$WANDB_MODE" \
UPLOAD_MODEL_ARTIFACT="$UPLOAD_MODEL_ARTIFACT" \
UPLOAD_DIAGNOSTIC_ARTIFACT="$UPLOAD_DIAGNOSTIC_ARTIFACT" \
MODEL_USE_ROPE="$USE_ROPE" \
MODEL_USE_WPE="$USE_WPE" \
WEIGHT_BITS="$BITS" \
QUANTIZE_AFTER_STEP="$Q_AFTER" \
QUANT_EXCLUDE_NORM="$QUANT_EXCLUDE_NORM" \
QUANT_SCOPE="$QUANT_SCOPE" \
PONDER_DYNAMIC_N="$PONDER_DYNAMIC_N" \
PONDER_N_STEPS="$PONDER_N_STEPS" \
PONDER_MAX_STEPS_CAP="$PONDER_MAX_STEPS_CAP" \
PONDER_BETA="$PONDER_BETA" \
PONDER_PRIOR_LAMBDA="$PONDER_PRIOR_LAMBDA" \
DIAG_MAX_STEPS="$DIAG_MAX_STEPS" \
DIAG_BATCH_SIZE="$DIAG_BATCH_SIZE" \
DIAG_N_BATCHES="$DIAG_N_BATCHES" \
TRAIN_EVAL_LENGTHS="$TRAIN_EVAL_LENGTHS" \
TRAIN_EVAL_BATCH_SIZE="$TRAIN_EVAL_BATCH_SIZE" \
TRAIN_EVAL_N_BATCHES="$TRAIN_EVAL_N_BATCHES" \
python train_ponder_quant.py 2>&1 | tee "$LOG_FILE"
