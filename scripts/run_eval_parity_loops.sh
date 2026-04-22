#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT=/data/yizhou/looped-tf-length-generalization/models/parity/wpe/8622e6ac-cf37-47b0-8acf-36b4c5a65273/model.pt
LENGTHS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
LOOPS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
NUM_SAMPLES=4000
BATCH_SIZE=512
RUN_NAME=dense_eval_wpe_permutate_ape_resample
CYCLE=3
APE_PERM_MODE=resample

python "$REPO_ROOT/src/eval/eval_parity_loops.py" \
  --checkpoint "$CHECKPOINT" \
  --lengths "$LENGTHS" \
  --loop_counts "$LOOPS" \
  --num_samples "$NUM_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --run_name "$RUN_NAME" \
  --cycle "$CYCLE" \
  --ape_perm_mode "$APE_PERM_MODE"