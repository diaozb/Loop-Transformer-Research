#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT=/data/yizhou/looped-tf-length-generalization/models/copy/8fe7447f-9d64-4de4-bf04-29c50e860cd6/model.pt
LENGTH=40
NUM_SAMPLES=20
PROB_ONE=0
RUN_NAME=pos_acc_eval_0

python "$REPO_ROOT/src/eval/eval_copy_position_accuracy.py" \
  --checkpoint "$CHECKPOINT" \
  --length "$LENGTH" \
  --num_samples "$NUM_SAMPLES" \
  --prob_one "$PROB_ONE" \
  --run_name "$RUN_NAME"
