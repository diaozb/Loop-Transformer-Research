#!/usr/bin/env python
"""
Dense fixed-loop evaluation for Copy models.

Place this file under:
  /data/diaozb/looped-tf-length-generalization/src/eval_copy_fixed_loop.py

Example:
  python eval_copy_fixed_loop.py \
    --run-dir ../models/nope_baselines/copy_fixed_loop/<RUN_ID> \
    --checkpoint best.pt \
    --lengths 1-20,21,22,30,40,60,400 \
    --id-max 20 \
    --max-loops 40 \
    --out-dir ../eval/nope_copy_fixed_loop/<RUN_ID>/diagnostics_loops40
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from generate_training_data import generate_prompt_matrix_copy
from models import build_general_model
from utils import convert_to_one_hot


def parse_lengths(spec: str) -> List[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def nested_get(d: Dict, path: str, default=None):
    cur = d
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_model(run_dir: Path, checkpoint: str, device: torch.device):
    ckpt_path = run_dir / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location=device)

    # train.py saves the whole model with torch.save(model, ...).
    if hasattr(obj, "looped_forward"):
        model = obj
    elif isinstance(obj, dict):
        cfg = load_yaml(run_dir / "config.yaml")
        model_cfg = nested_get(cfg, "model", None)
        if model_cfg is None:
            raise ValueError("Checkpoint is a state_dict, but config.yaml has no model section.")
        from types import SimpleNamespace
        model = build_general_model(SimpleNamespace(**model_cfg))
        state_dict = obj.get("state_dict", obj)
        model.load_state_dict(state_dict)
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")

    model.to(device)
    model.eval()
    return model


def to_one_hot_tensor(xs, n_dims: int, device: torch.device):
    try:
        xs_oh = convert_to_one_hot(xs, n_dims=n_dims)
    except TypeError:
        xs_oh = convert_to_one_hot(xs)
    return torch.tensor(xs_oh, dtype=torch.float32, device=device)


@torch.no_grad()
def eval_one_length(model, length: int, max_loops: int, batch_size: int, n_batches: int,
                    n_dims: int, device: torch.device):
    per_step_correct = np.zeros(max_loops, dtype=np.float64)
    per_step_token_correct = np.zeros(max_loops, dtype=np.float64)
    per_step_token_total = np.zeros(max_loops, dtype=np.float64)
    per_step_loss_sum = np.zeros(max_loops, dtype=np.float64)
    total_examples = 0

    for _ in range(n_batches):
        max_len = length + 1
        xs, _, ys, mask = generate_prompt_matrix_copy(
            batch_size,
            min_num_digits=length,
            max_num_digits=length + 1,
            max_len=max_len,
        )

        xs = to_one_hot_tensor(xs, n_dims=n_dims, device=device)
        ys = ys.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.bool)

        states = model.looped_forward(xs, horizon=max_loops)

        for step_idx in range(max_loops):
            logits = states[step_idx]  # [B, T, V]
            preds = logits.argmax(dim=-1)

            masked_correct = (preds == ys) & mask
            per_sample_exact = []
            for b in range(ys.shape[0]):
                idx = mask[b]
                per_sample_exact.append(masked_correct[b, idx].all().float())
            exact = torch.stack(per_sample_exact)

            ce = F.cross_entropy(logits.transpose(1, 2), ys, reduction="none")
            masked_loss = (ce * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)

            per_step_correct[step_idx] += exact.sum().item()
            per_step_token_correct[step_idx] += masked_correct.sum().item()
            per_step_token_total[step_idx] += mask.sum().item()
            per_step_loss_sum[step_idx] += masked_loss.sum().item()

        total_examples += ys.shape[0]

    rows = []
    for step_idx in range(max_loops):
        rows.append({
            "length": length,
            "loop": step_idx + 1,
            "answer_acc": per_step_correct[step_idx] / max(total_examples, 1),
            "token_acc": per_step_token_correct[step_idx] / max(per_step_token_total[step_idx], 1),
            "step_loss": per_step_loss_sum[step_idx] / max(total_examples, 1),
            "n_examples": total_examples,
        })
    return rows


def add_summary(per_step_df: pd.DataFrame, lengths: List[int], id_max: int, max_loops: int):
    rows = []
    for length in lengths:
        sub = per_step_df[per_step_df["length"] == length].copy()
        best_acc_idx = sub["answer_acc"].idxmax()
        best_tok_idx = sub["token_acc"].idxmax()
        min_loss_idx = sub["step_loss"].idxmin()

        row = {
            "split": "id" if length <= id_max else "ood",
            "length": length,
            "best_forced_answer_acc": float(sub.loc[best_acc_idx, "answer_acc"]),
            "best_forced_answer_step": int(sub.loc[best_acc_idx, "loop"]),
            "best_token_acc": float(sub.loc[best_tok_idx, "token_acc"]),
            "best_token_step": int(sub.loc[best_tok_idx, "loop"]),
            "min_step_loss": float(sub.loc[min_loss_idx, "step_loss"]),
            "min_loss_step": int(sub.loc[min_loss_idx, "loop"]),
            "acc_loop_1": float(sub[sub["loop"] == 1]["answer_acc"].iloc[0]),
            f"acc_loop_{max_loops}": float(sub[sub["loop"] == max_loops]["answer_acc"].iloc[0]),
        }
        if 20 <= max_loops:
            row["acc_loop_20"] = float(sub[sub["loop"] == 20]["answer_acc"].iloc[0])
        if 40 <= max_loops:
            row["acc_loop_40"] = float(sub[sub["loop"] == 40]["answer_acc"].iloc[0])
        if length <= max_loops:
            row["acc_loop_L"] = float(sub[sub["loop"] == length]["answer_acc"].iloc[0])
        else:
            row["acc_loop_L"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_plot(df: pd.DataFrame, value_col: str, out_path: Path, title: str):
    pivot = df.pivot(index="length", columns="loop", values=value_col).sort_index()
    plt.figure(figsize=(12, 7))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Loop")
    plt.ylabel("Length")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def line_plot(summary: pd.DataFrame, y: str, out_path: Path, title: str, ylabel: str):
    plt.figure(figsize=(9, 5))
    plt.plot(summary["length"], summary[y], marker="o")
    plt.xlabel("Length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--lengths", default="1-20,21,22,30,40,60")
    parser.add_argument("--id-max", type=int, default=20)
    parser.add_argument("--max-loops", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--long-threshold", type=int, default=100)
    parser.add_argument("--long-batch-size", type=int, default=16)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(run_dir / "config.yaml")
    n_dims = int(nested_get(cfg, "model.n_dims", 6))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(run_dir, args.checkpoint, device)

    lengths = parse_lengths(args.lengths)

    all_rows = []
    for length in lengths:
        bs = args.long_batch_size if length >= args.long_threshold else args.batch_size
        print(f"[eval] length={length} batch_size={bs} n_batches={args.n_batches} max_loops={args.max_loops}")
        rows = eval_one_length(
            model=model,
            length=length,
            max_loops=args.max_loops,
            batch_size=bs,
            n_batches=args.n_batches,
            n_dims=n_dims,
            device=device,
        )
        all_rows.extend(rows)

    per_step_df = pd.DataFrame(all_rows)
    summary_df = add_summary(per_step_df, lengths, args.id_max, args.max_loops)

    per_step_path = out_dir / "per_step_by_length.csv"
    summary_path = out_dir / "summary_by_length.csv"
    per_step_df.to_csv(per_step_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    pivot_plot(per_step_df, "answer_acc", out_dir / "forced_accuracy_heatmap.png", "Fixed-loop answer accuracy")
    pivot_plot(per_step_df, "step_loss", out_dir / "step_loss_heatmap.png", "Fixed-loop step loss")
    line_plot(summary_df, "best_forced_answer_acc", out_dir / "best_accuracy_vs_length.png", "Best forced-loop answer accuracy", "Best answer accuracy")
    line_plot(summary_df, "best_forced_answer_step", out_dir / "best_loop_vs_length.png", "Best loop by length", "Best loop")

    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "out_dir": str(out_dir),
        "lengths": lengths,
        "id_max": args.id_max,
        "max_loops": args.max_loops,
        "batch_size": args.batch_size,
        "long_batch_size": args.long_batch_size,
        "n_batches": args.n_batches,
        "config_task": nested_get(cfg, "training.task", None),
        "model": nested_get(cfg, "model", {}),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] wrote {summary_path}")
    print(summary_df.to_string(index=False))

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=manifest,
        )
        wandb.log({
            "summary_by_length": wandb.Table(dataframe=summary_df),
            "per_step_by_length": wandb.Table(dataframe=per_step_df),
            "forced_accuracy_heatmap": wandb.Image(str(out_dir / "forced_accuracy_heatmap.png")),
            "step_loss_heatmap": wandb.Image(str(out_dir / "step_loss_heatmap.png")),
            "best_accuracy_vs_length": wandb.Image(str(out_dir / "best_accuracy_vs_length.png")),
            "best_loop_vs_length": wandb.Image(str(out_dir / "best_loop_vs_length.png")),
        })
        artifact = wandb.Artifact(f"fixed_loop_eval_{run_dir.name}", type="eval")
        artifact.add_dir(str(out_dir))
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
