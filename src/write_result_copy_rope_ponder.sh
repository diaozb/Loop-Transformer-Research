python - <<'PY'
import pandas as pd
from pathlib import Path

base_path = Path("../models/rope_baselines/copy_ponder")
all_dirs = [d for d in base_path.iterdir() if d.is_dir()]
if not all_dirs:
    print("Error: No experiment directories found.")
    exit(1)

run_dir = sorted(all_dirs, key=lambda p: p.stat().st_mtime)[-1]
csv_path = Path("../eval/rope_copy_ponder") / run_dir.name / "diagnostics_steps20" / "summary_by_length.csv"

if not csv_path.exists():
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

cols = [
    "split",
    "length",
    "auto_answer_acc",
    "auto_code_acc",
    "expected_exit_step",
    "argmax_exit_step_mean",
    "argmax_exit_step_mode",
    "best_forced_answer_acc",
    "best_forced_answer_step",
    "min_answer_step_loss",
    "min_answer_loss_step",
]

output_file = "rope_copy_ponder.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Latest Run: {run_dir.name}\n")
    f.write("-" * 120 + "\n")
    f.write(df[cols].to_string(index=False))
    f.write("\n" + "-" * 120 + "\n")

print(f"Results successfully saved/overwritten in {output_file}")
PY