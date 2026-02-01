REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python $REPO_ROOT/src/prob_parity.py \
    --checkpoint /data/yizhou/looped-tf-length-generalization/models/parity/d9de8dd7-d283-4236-aa71-d02ce63ab40a/model.pt \
    --train_range 2,20 \
    --test_range 22,40 \
    --length_step 2 \
    --k -2 \
    --num_samples 1000 \
    --epochs 20 \
    --max_loop 45 \
    --baseline predict_loop