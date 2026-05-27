#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash debug_precision_ponder.sh rope 4
#   bash debug_precision_ponder.sh nope 4
#   bash debug_precision_ponder.sh rope 32

PE="${1:-rope}"          # rope or nope
BITS="${2:-4}"           # 32, 8, 6, 4, ...
SEED="${SEED:-0}"

if [[ "$PE" == "rope" ]]; then
  USE_ROPE=true
  USE_WPE=false
elif [[ "$PE" == "nope" ]]; then
  USE_ROPE=false
  USE_WPE=false
else
  echo "[error] PE must be rope or nope, got: $PE"
  exit 1
fi

if [[ "$BITS" == "32" ]]; then
  Q_AFTER=false
else
  Q_AFTER=true
fi

echo "[bash] Checking train_ponder_quant.py syntax..."
python -m py_compile train_ponder_quant.py

echo "[bash] Starting debug run: PE=$PE BITS=$BITS SEED=$SEED"

TASK=copy \
OUT_DIR=../models/precision_ponder_debug \
SEED="$SEED" \
TRAIN_STEPS=2000 \
EVAL_EVERY=500 \
RUN_FINAL_DIAGNOSTICS=false \
USE_WANDB=true \
WANDB_MODE=offline \
UPLOAD_MODEL_ARTIFACT=false \
UPLOAD_DIAGNOSTIC_ARTIFACT=false \
MODEL_USE_ROPE="$USE_ROPE" \
MODEL_USE_WPE="$USE_WPE" \
WEIGHT_BITS="$BITS" \
QUANTIZE_AFTER_STEP="$Q_AFTER" \
QUANT_EXCLUDE_NORM=false \
QUANT_SCOPE=all \
PONDER_N_STEPS=20 \
PONDER_DYNAMIC_N=false \
python train_ponder_quant.py
