#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT=/data/yizhou/looped-tf-length-generalization/models/mod_add_digits/mod_107/b8a5c387-c981-4e82-ac8a-9530ec222b70/model.pt
LENGTHS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
LOOPS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
NUM_SAMPLES=4000
BATCH_SIZE=512
MODULUS=107
RUN_NAME=dense_eval

python "$REPO_ROOT/src/eval/eval_mod_add_digits_loops.py" \
  --checkpoint "$CHECKPOINT" \
  --lengths "$LENGTHS" \
  --loop_counts "$LOOPS" \
  --num_samples "$NUM_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --modulus "$MODULUS" \
  --run_name "$RUN_NAME"
