#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/looped-tf-length-generalization/src

RUN_DIR=$(ls -td ../models/rope_baselines/copy_ponder/* | head -1)
RUN_ID=$(basename "$RUN_DIR")
OUT_DIR="../eval/rope_copy_ponder/${RUN_ID}/diagnostics_steps20"

echo "RUN_DIR=$RUN_DIR"
echo "OUT_DIR=$OUT_DIR"

cat "$RUN_DIR/config.yaml" | grep -E "task|use_rope|use_wpe|wpe_mode|beta|prior_lambda|n_steps|test_len"

python eval_ponder_diagnostics.py \
  --run-dir "$RUN_DIR" \
  --checkpoint best.pt \
  --lengths 1-20,21,22,40,60,400 \
  --id-max 20 \
  --max-steps 20 \
  --batch-size 256 \
  --long-threshold 100 \
  --long-batch-size 16 \
  --n-batches 8 \
  --out-dir "$OUT_DIR" \
  --wandb \
  --wandb-project looped-tf-rope-copy-ponder \
  --wandb-name "eval_rope_copy_ponder_${RUN_ID}_steps20"
