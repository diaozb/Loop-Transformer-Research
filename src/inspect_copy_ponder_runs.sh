#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/looped-tf-length-generalization/src

echo "Inspecting Copy + Ponder runs..."
echo

for d in \
  ../models/nope_baselines/copy_ponder/* \
  ../models/rope_baselines/copy_ponder/* \
  ../models/rope_baselines/copy_ponder/rope/fp32/*
do
  if [ -d "$d" ] && [ -f "$d/config.yaml" ]; then
    echo "=============================="
    echo "$d"

    RUN_DIR="$d" python - <<'PY'
import os
import yaml

path = os.path.join(os.environ["RUN_DIR"], "config.yaml")

with open(path, "r") as f:
    cfg = yaml.safe_load(f)

model = cfg.get("model", {}) or {}
ponder = cfg.get("ponder", {}) or {}

print("task:", cfg.get("task"))
print("seed:", cfg.get("seed"))
print("train_steps:", cfg.get("train_steps"))
print("out_dir:", cfg.get("out_dir"))
print("use_rope:", model.get("use_rope"))
print("use_wpe:", model.get("use_wpe"))
print("ponder_n_steps:", ponder.get("n_steps"))
print("has_best_pt:", os.path.exists(os.path.join(os.environ["RUN_DIR"], "best.pt")))
print("has_model_pt:", os.path.exists(os.path.join(os.environ["RUN_DIR"], "model.pt")))
PY
  fi
done
