#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from hashlib import sha1
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from generate_training_data import generate_prompt_matrix_addition
from utils import convert_to_one_hot


def _parse_int_list(raw: str) -> List[int]:
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return [int(x) for x in items]


def _looped_forward_collect(model, xs: torch.Tensor, horizon: int):
    """Collect hidden states and logits per loop."""
    if hasattr(model, "_read_in") and hasattr(model, "_read_out"):
        zs = model._read_in(xs)
        output = torch.zeros_like(zs).to(zs.device)
        hidden_list = []
        logits_list = []
        use_wpe = getattr(model, "use_wpe", False)
        for step in range(horizon):
            output = model.forward_single(output + zs, add_wpe=(use_wpe and step == 0))
            hidden_list.append(output)
            logits_list.append(model._read_out(output))
        return hidden_list, logits_list
    raise ValueError("Model does not expose looped forward components (_read_in/_read_out).")


def _batch_metrics(
    hidden_list: List[torch.Tensor],
    logits_list: List[torch.Tensor],
    ys: torch.Tensor,
    mask: torch.Tensor,
    loop_counts: List[int],
):
    """Compute metrics for a single batch across loop counts."""
    results = {lc: {} for lc in loop_counts}
    mask_bool = mask == 1
    answer_mask = mask_bool & (ys != 3)
    batch_size = ys.shape[0]

    # Precompute predictions per loop for change rate.
    preds_per_loop = {}
    for lc in loop_counts:
        logits = logits_list[lc - 1]
        preds_per_loop[lc] = logits.argmax(dim=-1)

    for idx, lc in enumerate(loop_counts):
        hidden = hidden_list[lc - 1]
        logits = logits_list[lc - 1]

        # Accuracy: exact match on masked positions.
        preds = preds_per_loop[lc]
        correct = (preds[mask_bool] == ys[mask_bool]).view(batch_size, -1).all(dim=1)
        acc = correct.float().mean().item()

        correct_answer = (preds[answer_mask] == ys[answer_mask]).view(batch_size, -1).all(dim=1)
        answer_acc = correct_answer.float().mean().item()

        # L2 norm of hidden delta vs previous loop.
        if idx == 0:
            delta_l2_all = float("nan")
            delta_l2_mask = float("nan")
        else:
            prev_hidden = hidden_list[loop_counts[idx - 1] - 1]
            delta_all = hidden - prev_hidden
            delta_l2_all = torch.linalg.norm(delta_all, dim=-1).mean().item()
            delta_mask = hidden[mask_bool] - prev_hidden[mask_bool]
            delta_l2_mask = torch.linalg.norm(delta_mask, dim=-1).mean().item()

        # Cosine similarity to previous loop.
        if idx == 0:
            cos_all = float("nan")
            cos_mask = float("nan")
        else:
            prev_hidden = hidden_list[loop_counts[idx - 1] - 1]
            cos_all = F.cosine_similarity(hidden, prev_hidden, dim=-1).mean().item()
            cos_mask = F.cosine_similarity(hidden[mask_bool], prev_hidden[mask_bool], dim=-1).mean().item()

        # Entropy of output distribution on masked positions.
        logits_mask = logits[mask_bool]
        probs = F.softmax(logits_mask, dim=-1)
        entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()

        # Answer change rate vs previous loop (sample-level).
        if idx == 0:
            change_rate = float("nan")
        else:
            prev_preds = preds_per_loop[loop_counts[idx - 1]]
            changed = (preds[mask_bool] != prev_preds[mask_bool]).view(batch_size, -1).any(dim=1)
            change_rate = changed.float().mean().item()

        results[lc] = {
            "accuracy": acc,
            "answer_accuracy": answer_acc,
            "delta_l2_norm_all": delta_l2_all,
            "delta_l2_norm_mask": delta_l2_mask,
            "cosine_to_prev_all": cos_all,
            "cosine_to_prev_mask": cos_mask,
            "entropy_mask": entropy,
            "answer_change_rate": change_rate,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate addition model across lengths and loop counts.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model.pt checkpoint saved by training.",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="4,8,16,32,40",
        help="Comma-separated addition lengths (number of digits).",
    )
    parser.add_argument(
        "--loop_counts",
        type=str,
        default=None,
        help="Comma-separated loop counts to evaluate. Defaults to 1..(max_length+2).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples per length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size used during evaluation.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name for output directory (subfolder under eval/addition/<checkpoint_name>/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional full output directory. If set, overrides the default eval/addition/<checkpoint_name>/<run_name>.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on.")
    args = parser.parse_args()

    lengths = _parse_int_list(args.lengths)
    max_len = max(lengths)
    if args.loop_counts:
        loop_counts = _parse_int_list(args.loop_counts)
    else:
        loop_counts = list(range(1, max_len + 3))
    loop_counts = sorted(set(loop_counts))
    horizon = max(loop_counts)

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    checkpoint_name = os.path.basename(checkpoint_dir)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        if args.run_name:
            run_name = args.run_name
        else:
            run_payload = {
                "lengths": lengths,
                "loop_counts": loop_counts,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
            }
            run_hash = sha1(json.dumps(run_payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{run_stamp}_{run_hash}"
        output_dir = os.path.join(repo_root, "eval", "addition", checkpoint_name, run_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    linear_embedding = hasattr(model, "_read_in") and isinstance(model._read_in, torch.nn.Linear)

    results: Dict[int, Dict[int, Dict[str, float]]] = {}
    for length in lengths:
        metrics_sum = {lc: {k: 0.0 for k in (
            "accuracy",
            "answer_accuracy",
            "delta_l2_norm_all",
            "delta_l2_norm_mask",
            "cosine_to_prev_all",
            "cosine_to_prev_mask",
            "entropy_mask",
            "answer_change_rate",
        )} for lc in loop_counts}
        metrics_count = {lc: 0 for lc in loop_counts}

        num_done = 0
        while num_done < args.num_samples:
            b = min(args.batch_size, args.num_samples - num_done)
            xs, batch_num, ys, mask = generate_prompt_matrix_addition(
                b, max_len=length + 2, min_num_digits=length, max_num_digits=length + 1
            )
            ys = ys.to(device)
            mask = mask.to(device)

            if linear_embedding:
                xs = torch.tensor(convert_to_one_hot(xs.numpy()), dtype=torch.float32, device=device)
            else:
                xs = xs.to(device)

            with torch.no_grad():
                hidden_list, logits_list = _looped_forward_collect(model, xs, horizon)
                batch_metrics = _batch_metrics(hidden_list, logits_list, ys, mask, loop_counts)

            for lc in loop_counts:
                for k, v in batch_metrics[lc].items():
                    metrics_sum[lc][k] += v * b
                metrics_count[lc] += b

            num_done += b

        results[length] = {}
        for lc in loop_counts:
            denom = max(1, metrics_count[lc])
            results[length][lc] = {k: metrics_sum[lc][k] / denom for k in metrics_sum[lc]}

    # Save JSON
    json_path = os.path.join(output_dir, "addition_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "run_name": os.path.basename(output_dir),
                "lengths": lengths,
                "loop_counts": loop_counts,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "results": results,
            },
            f,
            indent=2,
        )

    # Save CSV
    csv_path = os.path.join(output_dir, "addition_eval_results.csv")
    with open(csv_path, "w") as f:
        headers = [
            "length",
            "loop",
            "accuracy",
            "answer_accuracy",
            "delta_l2_norm_all",
            "delta_l2_norm_mask",
            "cosine_to_prev_all",
            "cosine_to_prev_mask",
            "entropy_mask",
            "answer_change_rate",
        ]
        f.write(",".join(headers) + "\n")
        for length in lengths:
            for lc in loop_counts:
                row = results[length][lc]
                f.write(
                    f"{length},{lc},"
                    f"{row['accuracy']:.6f},"
                    f"{row['answer_accuracy']:.6f},"
                    f"{row['delta_l2_norm_all']:.6f},"
                    f"{row['delta_l2_norm_mask']:.6f},"
                    f"{row['cosine_to_prev_all']:.6f},"
                    f"{row['cosine_to_prev_mask']:.6f},"
                    f"{row['entropy_mask']:.6f},"
                    f"{row['answer_change_rate']:.6f}\n"
                )

    # Heatmaps for all metrics (length x loop)
    import matplotlib.pyplot as plt

    metric_keys = [
        "accuracy",
        "answer_accuracy",
        "delta_l2_norm_all",
        "delta_l2_norm_mask",
        "cosine_to_prev_all",
        "cosine_to_prev_mask",
        "entropy_mask",
        "answer_change_rate",
    ]

    for metric in metric_keys:
        if metric in ("delta_l2_norm_all", "delta_l2_norm_mask", "cosine_to_prev_all", "cosine_to_prev_mask", "answer_change_rate"):
            heatmap_loops = loop_counts[1:]
        else:
            heatmap_loops = loop_counts
        if not heatmap_loops:
            continue
        matrix = np.array([[results[length][lc][metric] for lc in heatmap_loops] for length in lengths])
        plt.figure(figsize=(8, 6))
        im = plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(im, label=metric)
        plt.xticks(ticks=np.arange(len(heatmap_loops)), labels=heatmap_loops)
        plt.yticks(ticks=np.arange(len(lengths)), labels=lengths)
        plt.xlabel("Loop count")
        plt.ylabel("Length")
        plt.title(f"{metric.replace('_', ' ').title()} Heatmap (Length x Loop)")
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f"{metric}_heatmap.png")
        plt.savefig(heatmap_path, dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
