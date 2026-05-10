from __future__ import annotations

import torch

from ltf.data import TaskBatch
from ltf.models import PonderLoopedModel

from .losses import PonderLoss, ponder_loss


def compute_ponder_training_loss(
    model: PonderLoopedModel,
    xs: torch.Tensor,
    batch: TaskBatch,
    max_steps: int,
    beta: float,
    prior_lambda: float,
) -> PonderLoss:
    targets = batch.targets.to(xs.device, dtype=torch.long)
    mask = batch.mask.to(xs.device, dtype=torch.bool)
    output = model.forward_ponder(xs, max_steps=max_steps, halt_mask=mask)
    return ponder_loss(
        output.logits_steps,
        output.p_steps,
        targets,
        mask,
        beta=beta,
        prior_lambda=prior_lambda,
    )

