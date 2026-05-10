#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/looped-tf-length-generalization/src

RUN_DIR=$(ls -td ../models/ape_once_baselines/copy_fixed_loop/wpe/* | head -1)
RUN_ID=$(basename "$RUN_DIR")
OUT_DIR="../eval/ape_once_copy_fixed_loop/${RUN_ID}/diagnostics_loops40"

echo "RUN_DIR=$RUN_DIR"
echo "RUN_ID=$RUN_ID"
echo "OUT_DIR=$OUT_DIR"

echo "Config check:"
cat "$RUN_DIR/config.yaml" | grep -E "task|use_rope|use_wpe|wpe_mode|test_len|train_steps|out_dir" || true

python eval_copy_fixed_loop.py \
  --run-dir "$RUN_DIR" \
  --checkpoint best.pt \
  --lengths 1-20,21,22,30,40,60,400 \
  --id-max 20 \
  --max-loops 40 \
  --batch-size 256 \
  --long-threshold 100 \
  --long-batch-size 16 \
  --n-batches 8 \
  --out-dir "$OUT_DIR" \
  --wandb \
  --wandb-project looped-tf-ape-once-copy-fixed-loop \
  --wandb-name "eval_ape_once_copy_fixed_loop_${RUN_ID}_loops40"
