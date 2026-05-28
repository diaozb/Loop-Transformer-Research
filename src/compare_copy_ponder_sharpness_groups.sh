#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/looped-tf-length-generalization/src

NOPE_DIR="../models/nope_baselines/copy_ponder/62465a36-9f94-4157-8f93-edac50a7630d/sharpness_formal"
ROPE_DIR="../models/rope_baselines/copy_ponder/85c8ce6d-4abc-4432-a343-739683bc4bf7/sharpness_formal"

OUT_DIR="../eval/copy_ponder_sharpness_compare"
mkdir -p "${OUT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${OUT_DIR}/compare_copy_ponder_sharpness_${TIMESTAMP}.log"

python - <<'PY' 2>&1 | tee "${LOG_PATH}"
import os
import pandas as pd

runs = {
    "nope_62465": "../models/nope_baselines/copy_ponder/62465a36-9f94-4157-8f93-edac50a7630d/sharpness_formal",
    "rope_85c8": "../models/rope_baselines/copy_ponder/85c8ce6d-4abc-4432-a343-739683bc4bf7/sharpness_formal",
}

out_dir = "../eval/copy_ponder_sharpness_compare"
os.makedirs(out_dir, exist_ok=True)

all_dfs = []

for name, d in runs.items():
    print("\n" + "=" * 90)
    print(name)
    print("dir:", d)

    summary_path = os.path.join(d, "sharpness_summary.csv")
    raw_path = os.path.join(d, "sharpness_raw.csv")
    plot_path = os.path.join(d, "sharpness_delta_objective.png")

    print("has summary:", os.path.exists(summary_path))
    print("has raw:", os.path.exists(raw_path))
    print("has plot:", os.path.exists(plot_path))

    if not os.path.exists(summary_path):
        continue

    df = pd.read_csv(summary_path)
    df["run"] = name
    df["pe"] = "nope" if name.startswith("nope") else "rope"

    def split_group(length):
        if 1 <= length <= 20:
            return "ID_1_20"
        if length in [21, 22]:
            return "near_OOD_21_22"
        if length == 40:
            return "far_OOD_40"
        if length == 60:
            return "far_OOD_60"
        return "other"

    df["length_group"] = df["length"].apply(split_group)
    all_dfs.append(df)

    print("\nshape:", df.shape)
    print("NaN total:", int(df.isna().sum().sum()))
    print("lengths:", sorted(df["length"].unique().tolist()))
    print("epsilons:", sorted(df["epsilon"].unique().tolist()))

    base = df[df["epsilon"] == 0].copy()

    print("\nBase performance at epsilon=0:")
    print(base[[
        "length",
        "objective_mean",
        "auto_answer_acc_mean",
        "expected_exit_step_mean",
    ]].to_string(index=False))

    print("\nMean base acc:")
    print("ID 1-20:", base[base["length"].between(1, 20)]["auto_answer_acc_mean"].mean())
    print("OOD 21,22,40,60:", base[base["length"].isin([21,22,40,60])]["auto_answer_acc_mean"].mean())

    print("\nMean delta_objective by epsilon:")
    print(df.groupby("epsilon")["delta_objective_mean"].mean())

    print("\nMean delta_auto_answer_acc by epsilon:")
    print(df.groupby("epsilon")["delta_auto_answer_acc_mean"].mean())

    print("\nMean delta_expected_exit_step by epsilon:")
    print(df.groupby("epsilon")["delta_expected_exit_step_mean"].mean())


if not all_dfs:
    raise SystemExit("No sharpness summaries found.")

combined = pd.concat(all_dfs, ignore_index=True)

combined_path = os.path.join(out_dir, "copy_ponder_sharpness_combined.csv")
combined.to_csv(combined_path, index=False)

print("\n" + "=" * 90)
print("GROUPED COMPARISON")

grouped = (
    combined
    .groupby(["pe", "length_group", "epsilon"], as_index=False)
    .agg(
        objective_mean=("objective_mean", "mean"),
        delta_objective_mean=("delta_objective_mean", "mean"),
        delta_objective_std=("delta_objective_mean", "std"),
        auto_answer_acc_mean=("auto_answer_acc_mean", "mean"),
        delta_auto_answer_acc_mean=("delta_auto_answer_acc_mean", "mean"),
        expected_exit_step_mean=("expected_exit_step_mean", "mean"),
        delta_expected_exit_step_mean=("delta_expected_exit_step_mean", "mean"),
    )
)

grouped_path = os.path.join(out_dir, "copy_ponder_sharpness_grouped.csv")
grouped.to_csv(grouped_path, index=False)

print(grouped.to_string(index=False))

print("\n" + "=" * 90)
print("BASE EPSILON=0 COMPARISON")

base_compare = grouped[grouped["epsilon"] == 0].copy()
print(base_compare.to_string(index=False))

print("\n" + "=" * 90)
print("EPSILON=0.01 COMPARISON")

eps_compare = grouped[grouped["epsilon"] == 0.01].copy()
print(eps_compare.to_string(index=False))

print("\nSaved files:")
print("combined:", combined_path)
print("grouped:", grouped_path)
PY

echo
echo "Comparison log saved to:"
echo "${LOG_PATH}"
