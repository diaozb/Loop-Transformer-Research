#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/looped-tf-length-generalization/src

NOPE_DIR="../models/nope_baselines/copy_ponder/62465a36-9f94-4157-8f93-edac50a7630d/sharpness_formal"
ROPE_DIR="../models/rope_baselines/copy_ponder/85c8ce6d-4abc-4432-a343-739683bc4bf7/sharpness_formal"

OUT_DIR="../eval/copy_ponder_sharpness_epsilon_analysis"
mkdir -p "${OUT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${OUT_DIR}/inspect_all_eps_${TIMESTAMP}.log"

export NOPE_DIR
export ROPE_DIR
export OUT_DIR

python - <<'PY' 2>&1 | tee "${LOG_PATH}"
import os
import pandas as pd
import matplotlib.pyplot as plt

runs = {
    "nope": os.environ["NOPE_DIR"],
    "rope": os.environ["ROPE_DIR"],
}

out_dir = os.environ["OUT_DIR"]
os.makedirs(out_dir, exist_ok=True)

expected_eps = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

def length_group(L):
    if 1 <= L <= 20:
        return "ID_1_20"
    if L in [21, 22]:
        return "near_OOD_21_22"
    if L == 40:
        return "OOD_40"
    if L == 60:
        return "OOD_60"
    return "other"

all_dfs = []

print("=" * 100)
print("Inspecting all epsilon sharpness results")
print("=" * 100)

for pe, d in runs.items():
    print("\n" + "=" * 100)
    print(pe.upper())
    print("dir:", d)

    summary_path = os.path.join(d, "sharpness_summary.csv")
    raw_path = os.path.join(d, "sharpness_raw.csv")
    plot_path = os.path.join(d, "sharpness_delta_objective.png")

    print("has summary:", os.path.exists(summary_path))
    print("has raw:", os.path.exists(raw_path))
    print("has plot:", os.path.exists(plot_path))

    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)

    df = pd.read_csv(summary_path)
    df["pe"] = pe
    df["length_group"] = df["length"].apply(length_group)

    all_dfs.append(df)

    eps = sorted(df["epsilon"].unique().tolist())
    lengths = sorted(df["length"].unique().tolist())

    print("\nshape:", df.shape)
    print("NaN total:", int(df.isna().sum().sum()))
    print("lengths:", lengths)
    print("epsilons:", eps)

    missing_eps = sorted(set(expected_eps) - set(round(x, 10) for x in eps))
    print("missing expected epsilons:", missing_eps)

    print("\nRows per epsilon:")
    print(df.groupby("epsilon").size())

    print("\nMean delta_objective by epsilon:")
    print(df.groupby("epsilon")["delta_objective_mean"].mean())

    print("\nMean delta_auto_answer_acc by epsilon:")
    print(df.groupby("epsilon")["delta_auto_answer_acc_mean"].mean())

    print("\nMean delta_expected_exit_step by epsilon:")
    print(df.groupby("epsilon")["delta_expected_exit_step_mean"].mean())


combined = pd.concat(all_dfs, ignore_index=True)
combined_path = os.path.join(out_dir, "all_eps_combined.csv")
combined.to_csv(combined_path, index=False)

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

group_order = {
    "ID_1_20": 0,
    "near_OOD_21_22": 1,
    "OOD_40": 2,
    "OOD_60": 3,
}
grouped["group_order"] = grouped["length_group"].map(group_order)
grouped = grouped.sort_values(["pe", "group_order", "epsilon"]).drop(columns=["group_order"])

grouped_path = os.path.join(out_dir, "all_eps_grouped_summary.csv")
grouped.to_csv(grouped_path, index=False)

print("\n" + "=" * 100)
print("GROUPED ALL-EPSILON SUMMARY")
print("=" * 100)
print(grouped.to_string(index=False))

pivot_delta_obj = grouped.pivot_table(
    index=["pe", "length_group"],
    columns="epsilon",
    values="delta_objective_mean"
).reset_index()

pivot_acc = grouped.pivot_table(
    index=["pe", "length_group"],
    columns="epsilon",
    values="auto_answer_acc_mean"
).reset_index()

pivot_exit = grouped.pivot_table(
    index=["pe", "length_group"],
    columns="epsilon",
    values="expected_exit_step_mean"
).reset_index()

pivot_delta_obj_path = os.path.join(out_dir, "pivot_delta_objective_by_epsilon.csv")
pivot_acc_path = os.path.join(out_dir, "pivot_auto_acc_by_epsilon.csv")
pivot_exit_path = os.path.join(out_dir, "pivot_expected_exit_step_by_epsilon.csv")

pivot_delta_obj.to_csv(pivot_delta_obj_path, index=False)
pivot_acc.to_csv(pivot_acc_path, index=False)
pivot_exit.to_csv(pivot_exit_path, index=False)

print("\n" + "=" * 100)
print("PIVOT: delta_objective_mean by epsilon")
print("=" * 100)
print(pivot_delta_obj.to_string(index=False))

print("\n" + "=" * 100)
print("PIVOT: auto_answer_acc_mean by epsilon")
print("=" * 100)
print(pivot_acc.to_string(index=False))

print("\n" + "=" * 100)
print("PIVOT: expected_exit_step_mean by epsilon")
print("=" * 100)
print(pivot_exit.to_string(index=False))

# Plot 1: delta objective vs epsilon by PE and length group
for group in ["ID_1_20", "near_OOD_21_22", "OOD_40", "OOD_60"]:
    sub = grouped[grouped["length_group"] == group].copy()
    if sub.empty:
        continue

    plt.figure()
    for pe in ["nope", "rope"]:
        s = sub[sub["pe"] == pe].sort_values("epsilon")
        if s.empty:
            continue
        plt.plot(
            s["epsilon"],
            s["delta_objective_mean"],
            marker="o",
            label=pe,
        )

    plt.xscale("symlog", linthresh=1e-5)
    plt.xlabel("Epsilon")
    plt.ylabel("Mean Delta Objective")
    plt.title(f"Delta Objective vs Epsilon ({group})")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, f"delta_objective_vs_epsilon_{group}.png")
    plt.savefig(path, dpi=200)
    plt.close()

# Plot 2: auto accuracy vs epsilon by PE and length group
for group in ["ID_1_20", "near_OOD_21_22", "OOD_40", "OOD_60"]:
    sub = grouped[grouped["length_group"] == group].copy()
    if sub.empty:
        continue

    plt.figure()
    for pe in ["nope", "rope"]:
        s = sub[sub["pe"] == pe].sort_values("epsilon")
        if s.empty:
            continue
        plt.plot(
            s["epsilon"],
            s["auto_answer_acc_mean"],
            marker="o",
            label=pe,
        )

    plt.xscale("symlog", linthresh=1e-5)
    plt.xlabel("Epsilon")
    plt.ylabel("Mean Auto Answer Accuracy")
    plt.title(f"Auto Accuracy vs Epsilon ({group})")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, f"auto_acc_vs_epsilon_{group}.png")
    plt.savefig(path, dpi=200)
    plt.close()

print("\n" + "=" * 100)
print("Saved files")
print("=" * 100)
print("combined:", combined_path)
print("grouped:", grouped_path)
print("pivot delta objective:", pivot_delta_obj_path)
print("pivot auto acc:", pivot_acc_path)
print("pivot expected exit step:", pivot_exit_path)
print("plots saved under:", out_dir)
PY

echo
echo "Log saved to:"
echo "${LOG_PATH}"
