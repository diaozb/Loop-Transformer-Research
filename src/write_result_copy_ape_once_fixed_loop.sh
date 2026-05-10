#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/looped-tf-length-generalization/src

python - <<'PY'
from pathlib import Path
import pandas as pd

# APE once / WPE once fixed-loop 的模型目录
base_path = Path("../models/ape_once_baselines/copy_fixed_loop/wpe")

all_dirs = [d for d in base_path.iterdir() if d.is_dir()]
if not all_dirs:
    print("Error: No APE-once fixed-loop experiment directories found.")
    raise SystemExit(1)

# 默认取最新 run
run_dir = sorted(all_dirs, key=lambda p: p.stat().st_mtime)[-1]
run_id = run_dir.name

# APE once fixed-loop eval 输出目录
csv_path = Path("../eval/ape_once_copy_fixed_loop") / run_id / "diagnostics_loops40" / "summary_by_length.csv"

if not csv_path.exists():
    print(f"Error: CSV file not found at {csv_path}")
    print("Hint: run bash run_eval_copy_ape_once_fixed_loop.sh first.")
    raise SystemExit(1)

df = pd.read_csv(csv_path)

preferred_cols = [
    "split",
    "length",
    "best_forced_answer_acc",
    "best_forced_answer_step",
    "best_token_acc",
    "best_token_step",
    "min_step_loss",
    "min_loss_step",
    "acc_loop_L",
    "acc_loop_20",
    "acc_loop_40",
]

cols = [c for c in preferred_cols if c in df.columns]

output_file = "ape_once_copy_fixed_loop.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Latest Run: {run_id}\n")
    f.write(f"Run Dir: {run_dir}\n")
    f.write(f"Summary CSV: {csv_path}\n")
    f.write("-" * 120 + "\n")
    f.write(df[cols].to_string(index=False))
    f.write("\n" + "-" * 120 + "\n")

print(f"Results successfully saved/overwritten in {output_file}")
print(df[cols].to_string(index=False))
PY
