from __future__ import annotations

from typing import Dict, List

import torch

from .metrics import answer_mask, sequence_accuracy, token_accuracy


def sample_exit_indices(lambda_steps: torch.Tensor) -> torch.Tensor:
    n_steps, batch_size = lambda_steps.shape
    active = torch.ones(batch_size, dtype=torch.bool, device=lambda_steps.device)
    exit_idx = torch.full((batch_size,), n_steps - 1, dtype=torch.long, device=lambda_steps.device)

    for step in range(n_steps):
        if not active.any():
            break
        probs = lambda_steps[step, active].clamp(0.0, 1.0)
        stop = torch.bernoulli(probs).to(torch.bool)
        active_idx = torch.where(active)[0]
        if stop.any():
            stop_idx = active_idx[stop]
            exit_idx[stop_idx] = step
            active[stop_idx] = False

    if active.any():
        exit_idx[active] = n_steps - 1
    return exit_idx


def compute_auto_exit_stats(
    logits_steps: torch.Tensor,
    lambda_steps: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    task_name: str,
) -> Dict[str, float]:
    batch_size = targets.shape[0]
    exit_idx = sample_exit_indices(lambda_steps)
    batch_idx = torch.arange(batch_size, device=targets.device)
    logits_exit = logits_steps[exit_idx, batch_idx]
    preds = logits_exit.argmax(dim=-1)

    stats = {
        "avg_loops": float((exit_idx.float() + 1.0).mean().item()),
        "accuracy": float(sequence_accuracy(preds, targets, mask).item()),
    }
    ans_mask = answer_mask(mask, targets)
    if ans_mask.any():
        stats["answer_accuracy"] = float(sequence_accuracy(preds, targets, ans_mask).item())
    if task_name == "copy":
        stats["token_accuracy"] = float(token_accuracy(preds, targets, mask).item())
    return stats
