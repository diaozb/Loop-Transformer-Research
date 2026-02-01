#!/usr/bin/env python3
import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from generate_training_data import generate_prompt_matrix_parity
from utils import convert_to_one_hot


@dataclass
class ProbeBatch:
    features: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor
    loops: torch.Tensor


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _parse_range(raw: str, step: int) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) == 1:
        start = int(parts[0])
        end = int(parts[0])
    elif len(parts) == 2:
        start = int(parts[0])
        end = int(parts[1])
    else:
        raise ValueError("Range must be 'min,max' or a single value.")
    return list(range(start, end + 1, step))


def _parse_loop_counts(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) == 1:
        start = int(parts[0])
        end = int(parts[0])
    elif len(parts) == 2:
        start = int(parts[0])
        end = int(parts[1])
    else:
        raise ValueError("Loop range must be 'min,max' or a single value.")
    return list(range(start, end + 1))


def _roc_auc_score(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.to(torch.int64)
    scores = scores.to(torch.float32)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    sum_ranks_pos = ranks[pos].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _normalize_meta(x: torch.Tensor, max_value: int) -> torch.Tensor:
    if max_value <= 0:
        return x.to(torch.float32)
    return x.to(torch.float32) / float(max_value)


def _looped_hidden_collect(model, xs: torch.Tensor, horizon: int) -> List[torch.Tensor]:
    if hasattr(model, "_read_in") and hasattr(model, "_read_out"):
        zs = model._read_in(xs)
        output = torch.zeros_like(zs).to(zs.device)
        hidden_list = []
        for _ in range(horizon):
            output = model.forward_single(output + zs)
            hidden_list.append(output)
        return hidden_list
    raise ValueError("Model does not expose looped forward components (_read_in/_read_out).")


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_bool = mask == 1
    b, _, d = hidden.shape
    masked = hidden.masked_select(mask_bool.unsqueeze(-1)).view(b, -1, d)
    return masked.mean(dim=1)


def _build_dataset(
    model,
    lengths: List[int],
    loop_counts: List[int],
    num_samples: int,
    batch_size: int,
    device: torch.device,
    k: int,
) -> ProbeBatch:
    features_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    lengths_list: List[torch.Tensor] = []
    loops_list: List[torch.Tensor] = []

    linear_embedding = hasattr(model, "_read_in") and isinstance(model._read_in, torch.nn.Linear)
    horizon = max(loop_counts)

    for length in tqdm(lengths, desc="Build dataset (lengths)"):
        num_done = 0
        pbar = tqdm(total=num_samples, desc=f"Length {length}", leave=False)
        while num_done < num_samples:
            b = min(batch_size, num_samples - num_done)
            xs, _, ys, mask = generate_prompt_matrix_parity(
                b, max_len=length + 1, min_num_digits=length, max_num_digits=length + 1
            )
            mask = mask.to(device)

            if linear_embedding:
                xs = torch.tensor(convert_to_one_hot(xs.numpy()), dtype=torch.float32, device=device)
            else:
                xs = xs.to(device)

            with torch.no_grad():
                hidden_list = _looped_hidden_collect(model, xs, horizon)

            for loop in loop_counts:
                hidden = hidden_list[loop - 1]
                feat = _masked_mean(hidden, mask).detach().cpu()
                label_value = 1.0 if (length - loop) == k else 0.0
                labels = torch.full((b,), label_value, dtype=torch.float32)
                features_list.append(feat)
                labels_list.append(labels)
                lengths_list.append(torch.full((b,), length, dtype=torch.int64))
                loops_list.append(torch.full((b,), loop, dtype=torch.int64))

            num_done += b
            pbar.update(b)
        pbar.close()

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    lengths_tensor = torch.cat(lengths_list, dim=0)
    loops_tensor = torch.cat(loops_list, dim=0)
    return ProbeBatch(features, labels, lengths_tensor, loops_tensor)


def _group_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    loops: torch.Tensor,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).to(torch.int64)
    labels_int = labels.to(torch.int64)
    stats: Dict[Tuple[int, int], Dict[str, float]] = {}
    for i in range(len(preds)):
        key = (int(lengths[i]), int(loops[i]))
        if key not in stats:
            stats[key] = {
                "correct": 0.0,
                "count": 0.0,
                "sum_prob": 0.0,
                "sum_logit": 0.0,
                "sum_label": 0.0,
            }
        stats[key]["correct"] += float(preds[i].item() == labels_int[i].item())
        stats[key]["count"] += 1.0
        stats[key]["sum_prob"] += float(probs[i].item())
        stats[key]["sum_logit"] += float(logits[i].item())
        stats[key]["sum_label"] += float(labels[i].item())

    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for key, s in stats.items():
        count = s["count"]
        out[key] = {
            "correct": s["correct"],
            "count": count,
            "accuracy": s["correct"] / count if count else 0.0,
            "avg_prob": s["sum_prob"] / count if count else 0.0,
            "avg_logit": s["sum_logit"] / count if count else 0.0,
            "pos_rate": s["sum_label"] / count if count else 0.0,
        }
    return out


def _group_regression_stats(
    preds: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    loops: torch.Tensor,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    stats: Dict[Tuple[int, int], Dict[str, float]] = {}
    for i in range(len(preds)):
        key = (int(lengths[i]), int(loops[i]))
        if key not in stats:
            stats[key] = {
                "count": 0.0,
                "sum_abs": 0.0,
                "sum_sq": 0.0,
            }
        err = float(preds[i].item() - targets[i].item())
        stats[key]["count"] += 1.0
        stats[key]["sum_abs"] += abs(err)
        stats[key]["sum_sq"] += err * err
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for key, s in stats.items():
        count = s["count"]
        out[key] = {
            "count": count,
            "mae": s["sum_abs"] / count if count else 0.0,
            "mse": s["sum_sq"] / count if count else 0.0,
        }
    return out


def _plot_regression_heatmap(
    output_dir: str,
    split: str,
    lengths: List[int],
    loops: List[int],
    stats: Dict[Tuple[int, int], Dict[str, float]],
    metric: str,
) -> None:
    matrix = np.full((len(lengths), len(loops)), np.nan, dtype=np.float32)
    for i, length in enumerate(lengths):
        for j, loop in enumerate(loops):
            key = (length, loop)
            if key in stats:
                matrix[i, j] = stats[key][metric]
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label=metric)
    plt.xticks(ticks=np.arange(len(loops)), labels=loops)
    plt.yticks(ticks=np.arange(len(lengths)), labels=lengths)
    plt.xlabel("Loop count")
    plt.ylabel("Length")
    plt.title(f"{split.title()} {metric.upper()} Heatmap (Length x Loop)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{split}_{metric}_heatmap.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_accuracy_heatmap(
    output_dir: str,
    split: str,
    lengths: List[int],
    loops: List[int],
    stats: Dict[Tuple[int, int], Dict[str, float]],
) -> None:
    matrix = np.full((len(lengths), len(loops)), np.nan, dtype=np.float32)
    for i, length in enumerate(lengths):
        for j, loop in enumerate(loops):
            key = (length, loop)
            if key in stats:
                matrix[i, j] = stats[key]["accuracy"]
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="accuracy")
    plt.xticks(ticks=np.arange(len(loops)), labels=loops)
    plt.yticks(ticks=np.arange(len(lengths)), labels=lengths)
    plt.xlabel("Loop count")
    plt.ylabel("Length")
    plt.title(f"{split.title()} Accuracy Heatmap (Length x Loop)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{split}_accuracy_heatmap.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe parity hidden states for loop==length-k.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt checkpoint.")
    parser.add_argument("--train_range", type=str, required=True, help="Train length range: 'min,max'.")
    parser.add_argument("--test_range", type=str, required=True, help="Test length range: 'min,max'.")
    parser.add_argument("--length_step", type=int, default=1, help="Step size for length ranges.")
    parser.add_argument("--k", type=int, required=True, help="Label rule: length - loop == k => 1.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Samples per length.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for data generation.")
    parser.add_argument("--probe_hidden_dim", type=int, default=128, help="Hidden dim for MLP probe.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--max_loop", type=int, default=None, help="Max loop to include (default computed).")
    parser.add_argument("--loop_range", type=str, default=None, help="Optional loop range: 'min,max'. Overrides max_loop.")
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Positive class weight for BCEWithLogitsLoss. If omitted, auto-computed from train labels.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=["none", "predict_loop", "predict_length", "shuffle_label"],
        help="Baseline modes: predict_loop/length or shuffle labels.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for model/probe.")
    args = parser.parse_args()

    train_lengths = _parse_range(args.train_range, args.length_step)
    test_lengths = _parse_range(args.test_range, args.length_step)
    max_len = max(max(train_lengths), max(test_lengths))
    if args.loop_range:
        loop_counts = _parse_loop_counts(args.loop_range)
    else:
        if args.max_loop is None:
            max_loop = max_len + max(0, -args.k)
        else:
            max_loop = args.max_loop
        loop_counts = list(range(1, max_loop + 1))

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    checkpoint_name = os.path.basename(checkpoint_dir)
    run_tag = f"{checkpoint_name}_{max(train_lengths)}_{max(test_lengths)}_{args.k}"
    if args.baseline != "none":
        run_tag = f"{run_tag}_{args.baseline}"
    output_dir = os.path.join(repo_root, "eval", "prob", "parity", run_tag)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    train_batch = _build_dataset(
        model, train_lengths, loop_counts, args.num_samples, args.batch_size, device, args.k
    )
    test_batch = _build_dataset(
        model, test_lengths, loop_counts, args.num_samples, args.batch_size, device, args.k
    )

    train_features = train_batch.features
    test_features = test_batch.features
    probe = MLPProbe(train_features.shape[1], args.probe_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    if args.baseline in ("predict_loop", "predict_length"):
        loss_fn = nn.MSELoss()
        pos_weight_value = None
    else:
        if args.pos_weight is not None:
            pos_weight_value = float(args.pos_weight)
        else:
            pos = train_batch.labels.sum().item()
            neg = len(train_batch.labels) - pos
            pos_weight_value = (neg / pos) if pos > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_value], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if args.baseline == "shuffle_label":
        perm = torch.randperm(len(train_batch.labels))
        train_labels = train_batch.labels[perm]
    elif args.baseline == "predict_loop":
        train_labels = _normalize_meta(train_batch.loops, max(loop_counts))
    elif args.baseline == "predict_length":
        train_labels = _normalize_meta(train_batch.lengths, max(max(train_lengths), max(test_lengths)))
    else:
        train_labels = train_batch.labels

    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

    probe.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}")

    probe.eval()
    with torch.no_grad():
        train_logits = probe(train_features.to(device)).cpu()
        test_logits = probe(test_features.to(device)).cpu()

    if args.baseline in ("predict_loop", "predict_length"):
        train_acc = float("nan")
        test_acc = float("nan")
        train_auc = float("nan")
        test_auc = float("nan")
        if args.baseline == "predict_loop":
            train_targets = _normalize_meta(train_batch.loops, max(loop_counts))
            test_targets = _normalize_meta(test_batch.loops, max(loop_counts))
        else:
            max_len_norm = max(max(train_lengths), max(test_lengths))
            train_targets = _normalize_meta(train_batch.lengths, max_len_norm)
            test_targets = _normalize_meta(test_batch.lengths, max_len_norm)
        train_mae = torch.mean(torch.abs(train_logits - train_targets)).item()
        test_mae = torch.mean(torch.abs(test_logits - test_targets)).item()
        train_mse = torch.mean((train_logits - train_targets) ** 2).item()
        test_mse = torch.mean((test_logits - test_targets) ** 2).item()
    else:
        train_acc = (torch.sigmoid(train_logits) > 0.5).eq(train_batch.labels).float().mean().item()
        test_acc = (torch.sigmoid(test_logits) > 0.5).eq(test_batch.labels).float().mean().item()
        train_auc = _roc_auc_score(train_batch.labels, train_logits)
        test_auc = _roc_auc_score(test_batch.labels, test_logits)
        train_mae = float("nan")
        test_mae = float("nan")
        train_mse = float("nan")
        test_mse = float("nan")

    if args.baseline in ("predict_loop", "predict_length"):
        if args.baseline == "predict_loop":
            train_targets = _normalize_meta(train_batch.loops, max(loop_counts))
            test_targets = _normalize_meta(test_batch.loops, max(loop_counts))
        else:
            max_len_norm = max(max(train_lengths), max(test_lengths))
            train_targets = _normalize_meta(train_batch.lengths, max_len_norm)
            test_targets = _normalize_meta(test_batch.lengths, max_len_norm)
        train_group = _group_regression_stats(
            train_logits, train_targets, train_batch.lengths, train_batch.loops
        )
        test_group = _group_regression_stats(
            test_logits, test_targets, test_batch.lengths, test_batch.loops
        )
        _plot_regression_heatmap(output_dir, "train", train_lengths, loop_counts, train_group, "mae")
        _plot_regression_heatmap(output_dir, "test", test_lengths, loop_counts, test_group, "mae")
        _plot_regression_heatmap(output_dir, "train", train_lengths, loop_counts, train_group, "mse")
        _plot_regression_heatmap(output_dir, "test", test_lengths, loop_counts, test_group, "mse")
    else:
        train_group = _group_stats(
            train_logits, train_batch.labels, train_batch.lengths, train_batch.loops
        )
        test_group = _group_stats(
            test_logits, test_batch.labels, test_batch.lengths, test_batch.loops
        )
        _plot_accuracy_heatmap(output_dir, "train", train_lengths, loop_counts, train_group)
        _plot_accuracy_heatmap(output_dir, "test", test_lengths, loop_counts, test_group)

    csv_path = os.path.join(output_dir, "probe_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        if args.baseline in ("predict_loop", "predict_length"):
            writer.writerow(
                [
                    "split",
                    "length",
                    "loop",
                    "count",
                    "mae",
                    "mse",
                    "overall_mae",
                    "overall_mse",
                ]
            )
            for (length, loop), s in sorted(train_group.items()):
                writer.writerow(
                    [
                        "train",
                        length,
                        loop,
                        f"{s['count']:.0f}",
                        f"{s['mae']:.6f}",
                        f"{s['mse']:.6f}",
                        f"{train_mae:.6f}",
                        f"{train_mse:.6f}",
                    ]
                )
            for (length, loop), s in sorted(test_group.items()):
                writer.writerow(
                    [
                        "test",
                        length,
                        loop,
                        f"{s['count']:.0f}",
                        f"{s['mae']:.6f}",
                        f"{s['mse']:.6f}",
                        f"{test_mae:.6f}",
                        f"{test_mse:.6f}",
                    ]
                )
        else:
            writer.writerow(
                [
                    "split",
                    "length",
                    "loop",
                    "correct",
                    "count",
                    "accuracy",
                    "avg_prob",
                    "avg_logit",
                    "pos_rate",
                    "overall_accuracy",
                    "overall_auc",
                ]
            )
            if train_group:
                for (length, loop), s in sorted(train_group.items()):
                    writer.writerow(
                        [
                            "train",
                            length,
                            loop,
                            f"{s['correct']:.0f}",
                            f"{s['count']:.0f}",
                            f"{s['accuracy']:.6f}",
                            f"{s['avg_prob']:.6f}",
                            f"{s['avg_logit']:.6f}",
                            f"{s['pos_rate']:.6f}",
                            f"{train_acc:.6f}",
                            f"{train_auc:.6f}",
                        ]
                    )
                for (length, loop), s in sorted(test_group.items()):
                    writer.writerow(
                        [
                            "test",
                            length,
                            loop,
                            f"{s['correct']:.0f}",
                            f"{s['count']:.0f}",
                            f"{s['accuracy']:.6f}",
                            f"{s['avg_prob']:.6f}",
                            f"{s['avg_logit']:.6f}",
                            f"{s['pos_rate']:.6f}",
                            f"{test_acc:.6f}",
                            f"{test_auc:.6f}",
                        ]
                    )

    meta_path = os.path.join(output_dir, "probe_config.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "train_lengths": train_lengths,
                "test_lengths": test_lengths,
                "k": args.k,
                "loop_counts": loop_counts,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "probe_hidden_dim": args.probe_hidden_dim,
                "epochs": args.epochs,
                "lr": args.lr,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_auc": train_auc,
                "test_auc": test_auc,
                "pos_weight": pos_weight_value,
                "baseline": args.baseline,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_mse": train_mse,
                "test_mse": test_mse,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
