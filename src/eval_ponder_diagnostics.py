import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

import yaml
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("Please install pandas first: pip install pandas") from e

import train_ponder as tp


def parse_lengths(text: str):
    """
    Parse strings like:
      1-20,21,22,40,60,400
    into sorted unique integer lengths.
    """
    lengths = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lengths.extend(range(int(a), int(b) + 1))
        else:
            lengths.append(int(part))
    return sorted(set(lengths))


def load_cfg_from_run(run_dir: Path):
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find config.yaml at {config_path}")

    with open(config_path, "r") as f:
        payload = yaml.safe_load(f)

    cfg = SimpleNamespace()
    cfg.task = payload["task"]
    cfg.seed = payload.get("seed", 0)
    cfg.modulus = payload.get("modulus", 11)
    cfg.model = SimpleNamespace(**payload["model"])

    ponder_payload = payload["ponder"]
    cfg.ponder = tp.PonderConfig(
        beta=ponder_payload.get("beta", 0.01),
        prior_lambda=ponder_payload.get("prior_lambda", 0.2),
        max_steps_cap=ponder_payload.get("max_steps_cap", 128),
        dynamic_n=ponder_payload.get("dynamic_n", False),
        n_steps=ponder_payload.get("n_steps", 20),
    )

    cfg.train_config_payload = payload
    return cfg


def load_saved_model(checkpoint_path: Path, device: torch.device):
    """
    The model was saved from train_ponder.py executed as __main__.
    This shim helps torch.load find PonderLoopedModel.
    """
    import __main__

    __main__.PonderLoopedModel = tp.PonderLoopedModel

    try:
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(checkpoint_path, map_location=device)

    model.to(device)
    model.eval()
    return model


def make_answer_only_mask(code_mask: torch.Tensor, length: int):
    """
    Current code mask starts from first answer position to the end.
    For theoretical copy analysis, answer-only mask should cover exactly L copied bits.
    """
    answer_mask = torch.zeros_like(code_mask, dtype=torch.bool)
    first_idx = code_mask.long().argmax(dim=1)
    for i, start in enumerate(first_idx.tolist()):
        answer_mask[i, start : start + length] = True
    return answer_mask


def per_sample_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    mask_f = mask.float()
    return (ce * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def exact_match_per_sample(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    preds = logits.argmax(dim=-1)
    correct_or_ignored = (preds == targets) | (~mask.bool())
    return correct_or_ignored.all(dim=1).float()


@torch.no_grad()
def forward_collect(model, xs: torch.Tensor, max_steps: int, halt_mask: torch.Tensor):
    """
    Same recurrence as PonderLoopedModel.forward_ponder(),
    but additionally returns hidden states for convergence diagnostics.
    """
    zs = model.base._read_in(xs)
    output = torch.zeros_like(zs)

    batch_size = xs.shape[0]
    alive_prob = torch.ones(batch_size, device=xs.device)

    logits_steps = []
    p_steps = []
    hidden_steps = []

    use_wpe = getattr(model.base, "use_wpe", False)

    for step in range(max_steps):
        output = model.base.forward_single(output + zs, add_wpe=use_wpe, step_idx=step)
        logits = model.base._read_out(output)

        pooled = model._first_answer_hidden(output, halt_mask)

        if step == max_steps - 1:
            lambda_n = torch.ones(batch_size, device=xs.device)
        else:
            lambda_n = torch.sigmoid(model.halt_head(pooled)).squeeze(-1)

        p_n = alive_prob * lambda_n
        alive_prob = alive_prob * (1.0 - lambda_n)

        logits_steps.append(logits)
        p_steps.append(p_n)
        hidden_steps.append(output)

    logits_steps = torch.stack(logits_steps, dim=0)  # [N, B, T, V]
    p_steps = torch.stack(p_steps, dim=0)            # [N, B]
    hidden_steps = torch.stack(hidden_steps, dim=0)  # [N, B, T, D]

    p_steps = p_steps / p_steps.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return logits_steps, p_steps, hidden_steps


def compute_hidden_convergence(hidden_steps, answer_mask):
    """
    Return arrays of length N.
    Step 1 has no previous hidden state, so values are NaN.
    """
    n_steps = hidden_steps.shape[0]
    answer_mask_f = answer_mask.float().unsqueeze(-1)

    delta_all = np.full(n_steps, np.nan, dtype=np.float64)
    rel_delta_all = np.full(n_steps, np.nan, dtype=np.float64)
    cosine_all = np.full(n_steps, np.nan, dtype=np.float64)

    delta_answer = np.full(n_steps, np.nan, dtype=np.float64)
    rel_delta_answer = np.full(n_steps, np.nan, dtype=np.float64)
    cosine_answer = np.full(n_steps, np.nan, dtype=np.float64)

    for t in range(1, n_steps):
        prev = hidden_steps[t - 1]
        cur = hidden_steps[t]
        diff = cur - prev

        # All positions
        delta_all[t] = diff.norm(dim=-1).mean().item()
        rel_delta_all[t] = (
            diff.flatten(1).norm(dim=1) / prev.flatten(1).norm(dim=1).clamp_min(1e-12)
        ).mean().item()
        cosine_all[t] = F.cosine_similarity(
            cur.flatten(1), prev.flatten(1), dim=1
        ).mean().item()

        # Answer-only positions
        prev_ans = prev * answer_mask_f
        cur_ans = cur * answer_mask_f
        diff_ans = diff * answer_mask_f

        denom = answer_mask.float().sum(dim=1).clamp_min(1.0)
        delta_answer_per_sample = diff_ans.norm(dim=-1).sum(dim=1) / denom
        delta_answer[t] = delta_answer_per_sample.mean().item()

        rel_delta_answer[t] = (
            diff_ans.flatten(1).norm(dim=1)
            / prev_ans.flatten(1).norm(dim=1).clamp_min(1e-12)
        ).mean().item()

        cosine_answer[t] = F.cosine_similarity(
            cur_ans.flatten(1), prev_ans.flatten(1), dim=1
        ).mean().item()

    return {
        "delta_l2_all": delta_all,
        "relative_delta_all": rel_delta_all,
        "cosine_all": cosine_all,
        "delta_l2_answer": delta_answer,
        "relative_delta_answer": rel_delta_answer,
        "cosine_answer": cosine_answer,
    }


@torch.no_grad()
def evaluate_one_length(
    model,
    cfg,
    device,
    length: int,
    split: str,
    max_steps: int,
    batch_size: int,
    n_batches: int,
    save_position: bool = False,
):
    n_steps = max_steps

    total_examples = 0

    auto_code_correct = 0.0
    auto_answer_correct = 0.0
    expected_exit_sum = 0.0
    argmax_exit_sum = 0.0
    argmax_counter = Counter()

    exit_prob_sum = np.zeros(n_steps, dtype=np.float64)

    code_loss_sum = np.zeros(n_steps, dtype=np.float64)
    answer_loss_sum = np.zeros(n_steps, dtype=np.float64)

    forced_code_correct = np.zeros(n_steps, dtype=np.float64)
    forced_answer_correct = np.zeros(n_steps, dtype=np.float64)

    hidden_metric_sums = {
        "delta_l2_all": np.zeros(n_steps, dtype=np.float64),
        "relative_delta_all": np.zeros(n_steps, dtype=np.float64),
        "cosine_all": np.zeros(n_steps, dtype=np.float64),
        "delta_l2_answer": np.zeros(n_steps, dtype=np.float64),
        "relative_delta_answer": np.zeros(n_steps, dtype=np.float64),
        "cosine_answer": np.zeros(n_steps, dtype=np.float64),
    }
    hidden_metric_counts = np.zeros(n_steps, dtype=np.float64)

    position_delta_sum = None
    position_count = 0

    for batch_idx in range(n_batches):
        max_len = tp.compute_max_len(cfg.task, length)

        xs, ys, code_mask = tp.sample_batch(
            cfg.task,
            batch_size,
            min_num_digits=length,
            max_num_digits=length + 1,
            max_len=max_len,
            modulus=cfg.modulus,
        )

        xs = tp.prepare_inputs(xs, cfg.model, device)
        ys = ys.to(device=device, dtype=torch.long)
        code_mask = code_mask.to(device=device, dtype=torch.bool)

        if cfg.task == "copy":
            answer_mask = make_answer_only_mask(code_mask, length)
        else:
            answer_mask = code_mask

        logits_steps, p_steps, hidden_steps = forward_collect(
            model, xs, max_steps=max_steps, halt_mask=code_mask
        )

        bsz = ys.shape[0]
        total_examples += bsz

        step_numbers = torch.arange(1, n_steps + 1, device=device).float().unsqueeze(1)
        expected_exit_per_sample = (p_steps * step_numbers).sum(dim=0)
        expected_exit_sum += expected_exit_per_sample.sum().item()

        argmax_step = p_steps.argmax(dim=0)  # 0-index
        argmax_exit_sum += (argmax_step.float() + 1.0).sum().item()
        argmax_counter.update((argmax_step.cpu().numpy() + 1).tolist())

        batch_arange = torch.arange(bsz, device=device)
        picked_logits = logits_steps[argmax_step, batch_arange]

        auto_code_correct += exact_match_per_sample(picked_logits, ys, code_mask).sum().item()
        auto_answer_correct += exact_match_per_sample(picked_logits, ys, answer_mask).sum().item()

        exit_prob_sum += p_steps.sum(dim=1).detach().cpu().numpy()

        for t in range(n_steps):
            code_loss_sum[t] += per_sample_ce(logits_steps[t], ys, code_mask).sum().item()
            answer_loss_sum[t] += per_sample_ce(logits_steps[t], ys, answer_mask).sum().item()

            forced_code_correct[t] += exact_match_per_sample(
                logits_steps[t], ys, code_mask
            ).sum().item()
            forced_answer_correct[t] += exact_match_per_sample(
                logits_steps[t], ys, answer_mask
            ).sum().item()

        hidden_metrics = compute_hidden_convergence(hidden_steps, answer_mask)

        for name, values in hidden_metrics.items():
            for t in range(n_steps):
                if not np.isnan(values[t]):
                    hidden_metric_sums[name][t] += values[t] * bsz

        for t in range(1, n_steps):
            hidden_metric_counts[t] += bsz

        if save_position:
            # Average ||H_i^t - H_i^{t-1}|| per absolute position.
            seq_len = hidden_steps.shape[2]
            if position_delta_sum is None:
                position_delta_sum = np.zeros((n_steps, seq_len), dtype=np.float64)

            for t in range(1, n_steps):
                diff_norm = (hidden_steps[t] - hidden_steps[t - 1]).norm(dim=-1)  # [B, T]
                position_delta_sum[t] += diff_norm.sum(dim=0).detach().cpu().numpy()

            position_count += bsz

    # Per-step rows
    per_step_rows = []

    for t in range(n_steps):
        denom_hidden = max(hidden_metric_counts[t], 1.0)

        row = {
            "split": split,
            "length": length,
            "step": t + 1,
            "max_steps": max_steps,
            "exit_prob_mean": exit_prob_sum[t] / total_examples,
            "code_step_loss": code_loss_sum[t] / total_examples,
            "answer_step_loss": answer_loss_sum[t] / total_examples,
            "forced_code_acc": forced_code_correct[t] / total_examples,
            "forced_answer_acc": forced_answer_correct[t] / total_examples,
        }

        for name in hidden_metric_sums:
            if t == 0:
                row[name] = np.nan
            else:
                row[name] = hidden_metric_sums[name][t] / denom_hidden

        per_step_rows.append(row)

    forced_answer_accs = np.array([r["forced_answer_acc"] for r in per_step_rows])
    forced_code_accs = np.array([r["forced_code_acc"] for r in per_step_rows])
    answer_losses = np.array([r["answer_step_loss"] for r in per_step_rows])
    code_losses = np.array([r["code_step_loss"] for r in per_step_rows])

    argmax_mode_step = argmax_counter.most_common(1)[0][0] if argmax_counter else np.nan

    summary_row = {
        "split": split,
        "length": length,
        "max_steps": max_steps,
        "num_examples": total_examples,
        "auto_code_acc": auto_code_correct / total_examples,
        "auto_answer_acc": auto_answer_correct / total_examples,
        "expected_exit_step": expected_exit_sum / total_examples,
        "argmax_exit_step_mean": argmax_exit_sum / total_examples,
        "argmax_exit_step_mode": argmax_mode_step,
        "best_forced_code_acc": forced_code_accs.max(),
        "best_forced_code_step": int(forced_code_accs.argmax() + 1),
        "best_forced_answer_acc": forced_answer_accs.max(),
        "best_forced_answer_step": int(forced_answer_accs.argmax() + 1),
        "min_code_step_loss": code_losses.min(),
        "min_code_loss_step": int(code_losses.argmin() + 1),
        "min_answer_step_loss": answer_losses.min(),
        "min_answer_loss_step": int(answer_losses.argmin() + 1),
        "final_answer_step_loss": answer_losses[-1],
        "final_delta_l2_all": per_step_rows[-1]["delta_l2_all"],
        "final_relative_delta_all": per_step_rows[-1]["relative_delta_all"],
        "final_cosine_all": per_step_rows[-1]["cosine_all"],
        "final_delta_l2_answer": per_step_rows[-1]["delta_l2_answer"],
        "final_relative_delta_answer": per_step_rows[-1]["relative_delta_answer"],
        "final_cosine_answer": per_step_rows[-1]["cosine_answer"],
    }

    position_rows = []
    if save_position and position_delta_sum is not None:
        seq_len = position_delta_sum.shape[1]
        for t in range(1, n_steps):
            for pos in range(seq_len):
                position_rows.append(
                    {
                        "split": split,
                        "length": length,
                        "step": t + 1,
                        "position": pos,
                        "delta_l2_position": position_delta_sum[t, pos] / max(position_count, 1),
                    }
                )

    return summary_row, per_step_rows, position_rows


def make_plots(summary_df: pd.DataFrame, per_step_df: pd.DataFrame, out_dir: Path):
    # Accuracy vs length
    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["length"], summary_df["auto_answer_acc"], marker="o", label="Auto answer-only acc")
    plt.plot(summary_df["length"], summary_df["best_forced_answer_acc"], marker="o", label="Best forced answer-only acc")
    plt.xlabel("Length")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_length.png", dpi=200)
    plt.close()

    # Expected exit vs length
    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["length"], summary_df["expected_exit_step"], marker="o", label="Expected exit step")
    plt.plot(summary_df["length"], summary_df["argmax_exit_step_mean"], marker="o", label="Mean argmax exit step")
    plt.xlabel("Length")
    plt.ylabel("Step")
    plt.title("Exit Step vs Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "expected_exit_vs_length.png", dpi=200)
    plt.close()

    # Heatmaps
    def save_heatmap(value_col: str, filename: str, title: str, cbar_label: str):
        pivot = per_step_df.pivot(index="length", columns="step", values=value_col)
        pivot = pivot.sort_index()

        plt.figure(figsize=(10, 6))
        plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.colorbar(label=cbar_label)
        plt.xticks(
            ticks=np.arange(len(pivot.columns)),
            labels=[str(c) for c in pivot.columns],
            rotation=90,
        )
        plt.yticks(
            ticks=np.arange(len(pivot.index)),
            labels=[str(i) for i in pivot.index],
        )
        plt.xlabel("Step")
        plt.ylabel("Length")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    save_heatmap(
        "exit_prob_mean",
        "exit_distribution_heatmap.png",
        "Mean Exit Probability by Length and Step",
        "Mean exit probability",
    )

    save_heatmap(
        "answer_step_loss",
        "per_step_loss_heatmap.png",
        "Answer-only Loss by Length and Step",
        "Loss",
    )

    save_heatmap(
        "forced_answer_acc",
        "forced_accuracy_heatmap.png",
        "Forced Answer-only Accuracy by Length and Step",
        "Accuracy",
    )

    save_heatmap(
        "delta_l2_answer",
        "hidden_delta_answer_heatmap.png",
        "Answer Hidden Delta by Length and Step",
        "Mean L2 delta",
    )


def maybe_log_to_wandb(args, out_dir: Path):
    if not args.wandb:
        return

    import wandb

    run_name = args.wandb_name
    if run_name is None:
        run_name = f"eval_diagnostics_{Path(args.run_dir).name[:8]}_steps{args.max_steps}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        tags=["eval", "diagnostics", "pondernet", "copy", "nope"],
    )

    for path in out_dir.glob("*.png"):
        wandb.log({path.stem: wandb.Image(str(path))})

    artifact = wandb.Artifact(run_name, type="eval")
    artifact.add_dir(str(out_dir))
    wandb.log_artifact(artifact)

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--lengths", type=str, default="1-20,21,22,40,60,400")
    parser.add_argument("--id-max", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--long-threshold", type=int, default=100)
    parser.add_argument("--long-batch-size", type=int, default=16)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--save-position", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="looped-tf-nope-copy-ponder")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

    if args.out_dir is None:
        out_dir = Path("../eval/nope_copy_ponder") / run_dir.name / f"diagnostics_steps{args.max_steps}"
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {out_dir}")

    cfg = load_cfg_from_run(run_dir)
    model = load_saved_model(checkpoint_path, device)

    lengths = parse_lengths(args.lengths)
    print(f"Evaluating lengths: {lengths}")

    summary_rows = []
    per_step_rows_all = []
    position_rows_all = []

    for length in lengths:
        split = "id" if length <= args.id_max else "ood"
        batch_size = args.long_batch_size if length >= args.long_threshold else args.batch_size

        print(
            f"\nEvaluating length={length}, split={split}, "
            f"max_steps={args.max_steps}, batch_size={batch_size}, n_batches={args.n_batches}"
        )

        summary_row, per_step_rows, position_rows = evaluate_one_length(
            model=model,
            cfg=cfg,
            device=device,
            length=length,
            split=split,
            max_steps=args.max_steps,
            batch_size=batch_size,
            n_batches=args.n_batches,
            save_position=args.save_position,
        )

        print(
            f"length={length} "
            f"auto_answer_acc={summary_row['auto_answer_acc']:.4f} "
            f"expected_exit={summary_row['expected_exit_step']:.2f} "
            f"best_forced_answer_acc={summary_row['best_forced_answer_acc']:.4f} "
            f"best_step={summary_row['best_forced_answer_step']}"
        )

        summary_rows.append(summary_row)
        per_step_rows_all.extend(per_step_rows)
        position_rows_all.extend(position_rows)

        torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows).sort_values(["length"])
    per_step_df = pd.DataFrame(per_step_rows_all).sort_values(["length", "step"])

    summary_csv = out_dir / "summary_by_length.csv"
    per_step_csv = out_dir / "per_step_by_length.csv"

    summary_df.to_csv(summary_csv, index=False)
    per_step_df.to_csv(per_step_csv, index=False)

    if args.save_position:
        position_df = pd.DataFrame(position_rows_all).sort_values(["length", "step", "position"])
        position_df.to_csv(out_dir / "position_convergence_by_length.csv", index=False)

    make_plots(summary_df, per_step_df, out_dir)
    maybe_log_to_wandb(args, out_dir)

    print("\nDone.")
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved per-step diagnostics to: {per_step_csv}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
