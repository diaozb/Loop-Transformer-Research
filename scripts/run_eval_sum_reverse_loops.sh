#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT=/data/yizhou/looped-tf-length-generalization/models/sum_reverse/90e27139-2c7d-4438-8da5-ede9bf789ea3/model.pt
LENGTHS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
LOOPS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
NUM_SAMPLES=4000
BATCH_SIZE=512
RUN_NAME=dense_eval

python "$REPO_ROOT/src/eval/eval_sum_reverse_loops.py" \
  --checkpoint "$CHECKPOINT" \
  --lengths "$LENGTHS" \
  --loop_counts "$LOOPS" \
  --num_samples "$NUM_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --run_name "$RUN_NAME"
