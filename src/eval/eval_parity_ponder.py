#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime
from hashlib import sha1
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from generate_training_data import generate_prompt_matrix_parity
from utils import convert_to_one_hot

# =========================
# Editable Eval Parameters
# =========================
CHECKPOINT = "/data/yizhou/looped-tf-length-generalization/models/parity_ponder/86494c70-cf32-48e0-a9cd-3e914bd41768/model.pt"
DENSE_LENGTHS = "1-40"        # list: "4,8,16" or range: "1-40" or "1-40-2"
DENSE_LOOP_COUNTS = "1-40"    # list/range; None -> 1..(max_length+2)
CYCLE = 1
NUM_SAMPLES = 2000
BATCH_SIZE = 1024
AUTO_EXIT = True
AUTO_EXIT_MAX_LOOPS = 60    # None -> use dense horizon; else fixed max loops for auto-exit
DEVICE = "cuda"

# Output control
# Final output dir:
#   <EVAL_OUTPUT_BASE>/parity_ponder/<checkpoint_dir_name>/<EXPERIMENT_NAME>
EVAL_OUTPUT_BASE = None  # None -> repo_root/eval
EXPERIMENT_NAME = "dense_eval_ponder"   # None -> auto timestamp/hash


class PonderLoopedModel(nn.Module):
    # Keep class name/signature aligned with train_ponder.py so torch.load can unpickle
    # checkpoints saved via torch.save(model, ...).
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        hidden_dim = base_model._backbone.config.n_embd
        self.halt_head = nn.Linear(hidden_dim, 1)


def _parse_int_list(raw: str) -> List[int]:
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return [int(x) for x in items]


def _parse_int_spec(raw: str) -> List[int]:
    # Supports:
    # - list: "1,2,3"
    # - range: "1-40"
    # - range with step: "1-40-2"
    txt = raw.strip()
    if "," in txt:
        return _parse_int_list(txt)
    parts = [p.strip() for p in txt.split("-") if p.strip()]
    if len(parts) == 1:
        return [int(parts[0])]
    if len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
        step = 1
    elif len(parts) == 3:
        start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        raise ValueError(f"Invalid integer spec: {raw}")
    if step <= 0:
        raise ValueError("Range step must be > 0")
    if end < start:
        raise ValueError("Range end must be >= start")
    return list(range(start, end + 1, step))


def _first_answer_hidden(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    first_idx = mask.long().argmax(dim=1)
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_idx, first_idx]


def _collect_ponder_steps(model, xs: torch.Tensor, mask: torch.Tensor, horizon: int):
    if not hasattr(model, "base") or not hasattr(model, "halt_head"):
        raise ValueError("Checkpoint is not a Ponder model (missing .base or .halt_head).")
    base = model.base
    if not hasattr(base, "_read_in") or not hasattr(base, "_read_out"):
        raise ValueError("Ponder base model does not expose looped forward components.")

    zs = base._read_in(xs)
    output = torch.zeros_like(zs).to(zs.device)
    hidden_list, logits_list, p_list, lambda_list = [], [], [], []
    alive_prob = torch.ones(xs.shape[0], device=xs.device)
    use_wpe = getattr(base, "use_wpe", False)

    for step in range(horizon):
        output = base.forward_single(output + zs, add_wpe=use_wpe)
        logits = base._read_out(output)
        pooled = _first_answer_hidden(output, mask)
        if step == horizon - 1:
            lambda_n = torch.ones(xs.shape[0], device=xs.device)
        else:
            lambda_n = torch.sigmoid(model.halt_head(pooled)).squeeze(-1)
        p_n = alive_prob * lambda_n
        alive_prob = alive_prob * (1.0 - lambda_n)

        hidden_list.append(output)
        logits_list.append(logits)
        p_list.append(p_n)
        lambda_list.append(lambda_n)

    p_steps = torch.stack(p_list, dim=0)  # [N, B]
    p_steps = p_steps / p_steps.sum(dim=0, keepdim=True).clamp_min(1e-12)
    lambda_steps = torch.stack(lambda_list, dim=0)  # [N, B]
    return hidden_list, logits_list, p_steps, lambda_steps


def _batch_metrics(
    hidden_list: List[torch.Tensor],
    logits_list: List[torch.Tensor],
    ys: torch.Tensor,
    mask: torch.Tensor,
    loop_counts: List[int],
    cycles: List[int],
):
    results = {lc: {} for lc in loop_counts}
    mask_bool = mask == 1
    answer_mask = mask_bool & (ys != 3)
    batch_size = ys.shape[0]

    preds_per_loop = {}
    for lc in loop_counts:
        logits = logits_list[lc - 1]
        preds_per_loop[lc] = logits.argmax(dim=-1)

    for idx, lc in enumerate(loop_counts):
        hidden = hidden_list[lc - 1]
        logits = logits_list[lc - 1]
        preds = preds_per_loop[lc]

        correct = (preds[mask_bool] == ys[mask_bool]).view(batch_size, -1).all(dim=1)
        acc = correct.float().mean().item()
        correct_answer = (preds[answer_mask] == ys[answer_mask]).view(batch_size, -1).all(dim=1)
        answer_acc = correct_answer.float().mean().item()

        if idx == 0:
            delta_l2_all = float("nan")
            delta_l2_mask = float("nan")
            cos_all = float("nan")
            cos_mask = float("nan")
            change_rate = float("nan")
        else:
            prev_hidden = hidden_list[loop_counts[idx - 1] - 1]
            delta_all = hidden - prev_hidden
            delta_l2_all = torch.linalg.norm(delta_all, dim=-1).mean().item()
            delta_mask = hidden[mask_bool] - prev_hidden[mask_bool]
            delta_l2_mask = torch.linalg.norm(delta_mask, dim=-1).mean().item()

            cos_all = F.cosine_similarity(hidden, prev_hidden, dim=-1).mean().item()
            cos_mask = F.cosine_similarity(hidden[mask_bool], prev_hidden[mask_bool], dim=-1).mean().item()

            prev_preds = preds_per_loop[loop_counts[idx - 1]]
            changed = (preds[mask_bool] != prev_preds[mask_bool]).view(batch_size, -1).any(dim=1)
            change_rate = changed.float().mean().item()

        logits_mask = logits[mask_bool]
        probs = F.softmax(logits_mask, dim=-1)
        entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()

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

        for cycle in cycles:
            key = f"delta_l2_norm_mask_cycle_{cycle}"
            if lc - cycle < 1:
                results[lc][key] = float("nan")
                continue
            prev_hidden = hidden_list[lc - cycle - 1]
            delta_mask = hidden[mask_bool] - prev_hidden[mask_bool]
            results[lc][key] = torch.linalg.norm(delta_mask, dim=-1).mean().item()

    return results


def _sample_exit_indices(lambda_steps: torch.Tensor) -> torch.Tensor:
    # lambda_steps: [N, B]
    n_steps, batch_size = lambda_steps.shape
    active = torch.ones(batch_size, dtype=torch.bool, device=lambda_steps.device)
    exit_idx = torch.full((batch_size,), n_steps - 1, dtype=torch.long, device=lambda_steps.device)

    for step in range(n_steps):
        if not active.any():
            break
        probs = lambda_steps[step, active].clamp(0.0, 1.0)
        # Bernoulli sampling per active sample at this step.
        stop = torch.bernoulli(probs).to(torch.bool)
        active_idx = torch.where(active)[0]
        if stop.any():
            stop_idx = active_idx[stop]
            exit_idx[stop_idx] = step
            active[stop_idx] = False

    # Safety: any still-active sample exits at last step.
    if active.any():
        exit_idx[active] = n_steps - 1
    return exit_idx


def _compute_auto_exit_stats(logits_list: List[torch.Tensor], lambda_steps: torch.Tensor, ys: torch.Tensor, mask: torch.Tensor):
    mask_bool = mask == 1
    answer_mask = mask_bool & (ys != 3)
    batch_size = ys.shape[0]
    exit_idx = _sample_exit_indices(lambda_steps)  # [B], stochastic Bernoulli auto-exit
    b_idx = torch.arange(batch_size, device=ys.device)
    logits_exit = torch.stack(logits_list, dim=0)[exit_idx, b_idx]
    preds = logits_exit.argmax(dim=-1)

    correct = (preds[mask_bool] == ys[mask_bool]).view(batch_size, -1).all(dim=1).float()
    answer_correct = (preds[answer_mask] == ys[answer_mask]).view(batch_size, -1).all(dim=1).float()
    avg_loops = (exit_idx.float() + 1.0).mean().item()

    return {
        "avg_loops": avg_loops,
        "accuracy": correct.mean().item(),
        "answer_accuracy": answer_correct.mean().item(),
    }


def main():
    if "<run_id>" in CHECKPOINT:
        raise ValueError("Please set CHECKPOINT to a real checkpoint path.")

    lengths = _parse_int_spec(DENSE_LENGTHS)
    max_len = max(lengths)
    if DENSE_LOOP_COUNTS:
        loop_counts = sorted(set(_parse_int_spec(DENSE_LOOP_COUNTS)))
    else:
        loop_counts = list(range(1, max_len + 3))
    dense_horizon = max(loop_counts)
    auto_exit_horizon = dense_horizon if AUTO_EXIT_MAX_LOOPS is None else int(AUTO_EXIT_MAX_LOOPS)
    if auto_exit_horizon < 1:
        raise ValueError("AUTO_EXIT_MAX_LOOPS must be >= 1")
    collect_horizon = max(dense_horizon, auto_exit_horizon)
    cycles = list(range(1, max(1, CYCLE) + 1))

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    checkpoint_dir = os.path.dirname(os.path.abspath(CHECKPOINT))
    checkpoint_name = os.path.basename(checkpoint_dir)
    output_base = EVAL_OUTPUT_BASE if EVAL_OUTPUT_BASE is not None else os.path.join(repo_root, "eval")
    if EXPERIMENT_NAME:
        run_name = EXPERIMENT_NAME
    else:
        run_payload = {
            "lengths": lengths,
            "loop_counts": loop_counts,
            "num_samples": NUM_SAMPLES,
            "batch_size": BATCH_SIZE,
            "auto_exit": AUTO_EXIT,
            "auto_exit_max_loops": AUTO_EXIT_MAX_LOOPS,
        }
        run_hash = sha1(json.dumps(run_payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_stamp}_{run_hash}"
    output_dir = os.path.join(output_base, "parity_ponder", checkpoint_name, run_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = torch.load(CHECKPOINT, map_location=device)
    model.eval()
    model.to(device)

    if not hasattr(model, "base") and AUTO_EXIT:
        raise ValueError("AUTO_EXIT=True requires a Ponder model with halt head.")

    base_model = model.base if hasattr(model, "base") else model
    linear_embedding = hasattr(base_model, "_read_in") and isinstance(base_model._read_in, torch.nn.Linear)

    results: Dict[int, Dict[int, Dict[str, float]]] = {}
    auto_exit_by_length: Dict[int, Dict[str, float]] = {}

    for length in lengths:
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
        metric_keys.extend([f"delta_l2_norm_mask_cycle_{c}" for c in cycles])
        metrics_sum = {lc: {k: 0.0 for k in metric_keys} for lc in loop_counts}
        metrics_count = {lc: 0 for lc in loop_counts}

        auto_sum = {"avg_loops": 0.0, "accuracy": 0.0, "answer_accuracy": 0.0}
        auto_count = 0

        num_done = 0
        while num_done < NUM_SAMPLES:
            b = min(BATCH_SIZE, NUM_SAMPLES - num_done)
            xs, _, ys, mask = generate_prompt_matrix_parity(
                b, max_len=length + 1, min_num_digits=length, max_num_digits=length + 1
            )
            ys = ys.to(device)
            mask = mask.to(device)
            if linear_embedding:
                xs = torch.tensor(convert_to_one_hot(xs.numpy()), dtype=torch.float32, device=device)
            else:
                xs = xs.to(device)

            with torch.no_grad():
                hidden_list, logits_list, p_steps, lambda_steps = _collect_ponder_steps(model, xs, mask, collect_horizon)
                batch_metrics = _batch_metrics(hidden_list, logits_list, ys, mask, loop_counts, cycles)
                if AUTO_EXIT:
                    logits_auto = logits_list[:auto_exit_horizon]
                    lambda_auto = lambda_steps[:auto_exit_horizon]
                    auto_stats = _compute_auto_exit_stats(logits_auto, lambda_auto, ys, mask)

            for lc in loop_counts:
                for k, v in batch_metrics[lc].items():
                    metrics_sum[lc][k] += v * b
                metrics_count[lc] += b

            if AUTO_EXIT:
                for k in auto_sum:
                    auto_sum[k] += auto_stats[k] * b
                auto_count += b

            num_done += b

        results[length] = {}
        for lc in loop_counts:
            denom = max(1, metrics_count[lc])
            results[length][lc] = {k: metrics_sum[lc][k] / denom for k in metrics_sum[lc]}

        if AUTO_EXIT:
            denom = max(1, auto_count)
            auto_exit_by_length[length] = {k: auto_sum[k] / denom for k in auto_sum}

    payload = {
        "checkpoint": CHECKPOINT,
        "run_name": os.path.basename(output_dir),
        "lengths": lengths,
        "loop_counts": loop_counts,
        "num_samples": NUM_SAMPLES,
        "batch_size": BATCH_SIZE,
        "results": results,
        "auto_exit_enabled": AUTO_EXIT,
        "auto_exit_max_loops": auto_exit_horizon,
        "auto_exit_by_length": auto_exit_by_length,
    }
    with open(os.path.join(output_dir, "parity_ponder_eval_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(output_dir, "parity_ponder_eval_results.csv"), "w") as f:
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
        headers.extend([f"delta_l2_norm_mask_cycle_{c}" for c in cycles])
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
                    f"{row['answer_change_rate']:.6f},"
                    + ",".join(f"{row[f'delta_l2_norm_mask_cycle_{c}']:.6f}" for c in cycles)
                    + "\n"
                )

    if AUTO_EXIT:
        with open(os.path.join(output_dir, "parity_ponder_auto_exit_by_length.csv"), "w") as f:
            f.write("length,avg_loops,accuracy,answer_accuracy\n")
            for length in lengths:
                row = auto_exit_by_length[length]
                f.write(f"{length},{row['avg_loops']:.6f},{row['accuracy']:.6f},{row['answer_accuracy']:.6f}\n")

    import matplotlib.pyplot as plt

    metric_keys = [
        "accuracy",
        "delta_l2_norm_all",
        "delta_l2_norm_mask",
        "cosine_to_prev_all",
        "cosine_to_prev_mask",
        "entropy_mask",
        "answer_change_rate",
    ]
    metric_keys.extend([f"delta_l2_norm_mask_cycle_{c}" for c in cycles])
    for metric in metric_keys:
        if metric in ("delta_l2_norm_all", "delta_l2_norm_mask", "cosine_to_prev_all", "cosine_to_prev_mask", "answer_change_rate"):
            heatmap_loops = loop_counts[1:]
        elif metric.startswith("delta_l2_norm_mask_cycle_"):
            cycle = int(metric.split("_")[-1])
            heatmap_loops = [lc for lc in loop_counts if lc > cycle]
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
        plt.savefig(os.path.join(output_dir, f"{metric}_heatmap.png"), dpi=200)
        plt.close()

    if AUTO_EXIT:
        xs = lengths
        ys_loops = [auto_exit_by_length[l]["avg_loops"] for l in lengths]
        ys_acc = [auto_exit_by_length[l]["accuracy"] for l in lengths]
        ys_ans_acc = [auto_exit_by_length[l]["answer_accuracy"] for l in lengths]

        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys_loops, marker="o")
        plt.xlabel("Length")
        plt.ylabel("Average Exit Loops")
        plt.title("Auto-Exit Average Loop Count vs Length")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "auto_exit_avg_loops_vs_length.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys_acc, marker="o", label="accuracy")
        plt.plot(xs, ys_ans_acc, marker="s", label="answer_accuracy")
        plt.xlabel("Length")
        plt.ylabel("Accuracy")
        plt.title("Auto-Exit Accuracy vs Length")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "auto_exit_accuracy_vs_length.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
