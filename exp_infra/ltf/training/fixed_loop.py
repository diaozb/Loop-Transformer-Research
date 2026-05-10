from __future__ import annotations

from typing import List

import torch

from ltf.data import TaskBatch

from .losses import masked_cross_entropy


def select_fixed_loop_logits(logits_by_step: List[torch.Tensor], lengths: torch.Tensor, task_name: str) -> torch.Tensor:
    if task_name == "multi":
        raise NotImplementedError("multi is intentionally outside the first migration scope")
    selected = []
    for sample_idx in range(lengths.shape[0]):
        step_idx = int(lengths[sample_idx].item()) - 1
        selected.append(logits_by_step[step_idx][sample_idx])
    return torch.stack(selected, dim=0)


def fixed_loop_loss(model, xs: torch.Tensor, batch: TaskBatch, horizon: int, task_name: str) -> torch.Tensor:
    logits_by_step = model.looped_forward(xs, horizon=horizon)
    selected_logits = select_fixed_loop_logits(logits_by_step, batch.lengths.to(xs.device), task_name)
    targets = batch.targets.to(xs.device, dtype=torch.long)
    mask = batch.mask.to(xs.device)
    return masked_cross_entropy(selected_logits, targets, mask)

