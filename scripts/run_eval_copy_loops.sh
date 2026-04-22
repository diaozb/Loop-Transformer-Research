#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT=/data/yizhou/looped-tf-length-generalization/models/copy/wpe/2dbede54-a54a-42b6-8802-87d9b33c93ae_add_all/model.pt
LENGTHS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
LOOPS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
NUM_SAMPLES=4000
BATCH_SIZE=512
RUN_NAME=dense_eval_0.5_wpe_permutate_ape_fixed
PROB_ONE=0.5
APE_PERM_MODE=fixed

python "$REPO_ROOT/src/eval/eval_copy_loops.py" \
  --checkpoint "$CHECKPOINT" \
  --lengths "$LENGTHS" \
  --loop_counts "$LOOPS" \
  --num_samples "$NUM_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --run_name "$RUN_NAME" \
  --prob_one "$PROB_ONE" \
  --ape_perm_mode "$APE_PERM_MODE"
