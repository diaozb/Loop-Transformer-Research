#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# General grid launcher for LoopTF + PonderNet precision experiments
# ============================================================
#
# Usage:
#   bash run_precision_ponder_grid_v2.sh
#
# Examples:
#   PE_LIST="rope nope" BITS_LIST="32 8 6" SEEDS_LIST="0" STEPS_LIST="100001" bash run_precision_ponder_grid_v2.sh
#   PE_LIST="rope" BITS_LIST="32 8 6" SEEDS_LIST="0 1 2" STEPS_LIST="100001" bash run_precision_ponder_grid_v2.sh
#
# Safe defaults are passed through run_precision_ponder_one_v3.sh:
#   PONDER_DYNAMIC_N=true
#   PONDER_MAX_STEPS_CAP=128
#   DIAG_MAX_STEPS=128
#   TEST_LEN=20
#   USE_BEST_CHECKPOINT_FOR_DIAGNOSTICS=false

PE_LIST="${PE_LIST:-rope nope}"
BITS_LIST="${BITS_LIST:-32 16 8}"
SEEDS_LIST="${SEEDS_LIST:-2}"
STEPS_LIST="${STEPS_LIST:-100001}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
FAIL_FAST="${FAIL_FAST:-false}"

echo "================================================================================"
echo "[grid] LoopTF + PonderNet precision grid"
echo "[grid] PE_LIST=$PE_LIST"
echo "[grid] BITS_LIST=$BITS_LIST"
echo "[grid] SEEDS_LIST=$SEEDS_LIST"
echo "[grid] STEPS_LIST=$STEPS_LIST"
echo "[grid] EVAL_EVERY=$EVAL_EVERY"
echo "[grid] FAIL_FAST=$FAIL_FAST"
echo "================================================================================"

TOTAL=0
FAILED=0

for steps in $STEPS_LIST; do
  for seed in $SEEDS_LIST; do
    for pe in $PE_LIST; do
      for bits in $BITS_LIST; do
        TOTAL=$((TOTAL + 1))
        echo "================================================================================"
        echo "[grid] Run #$TOTAL: PE=$pe BITS=$bits SEED=$seed STEPS=$steps"
        echo "================================================================================"

        if ! bash run_precision_ponder_one.sh "$pe" "$bits" "$seed" "$steps" "$EVAL_EVERY"; then
          FAILED=$((FAILED + 1))
          echo "[grid][ERROR] Run failed: PE=$pe BITS=$bits SEED=$seed STEPS=$steps"
          if [[ "$FAIL_FAST" == "true" ]]; then
            echo "[grid][ERROR] FAIL_FAST=true, stopping."
            exit 1
          fi
          echo "[grid][ERROR] FAIL_FAST=false, continuing."
        fi
      done
    done
  done
done

echo "================================================================================"
echo "[grid] Finished. TOTAL=$TOTAL FAILED=$FAILED"
echo "================================================================================"

if [[ "$FAILED" -gt 0 ]]; then
  exit 1
fi
