from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PonderLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[mask == 1], targets[mask == 1])


def per_sample_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    mask_f = mask.float()
    return (ce * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def truncated_geometric_prior(max_steps: int, prior_lambda: float, device: torch.device) -> torch.Tensor:
    probs = []
    for step in range(max_steps - 1):
        probs.append(prior_lambda * ((1.0 - prior_lambda) ** step))
    probs.append((1.0 - prior_lambda) ** (max_steps - 1))
    return torch.tensor(probs, dtype=torch.float32, device=device)


def ponder_loss(
    logits_steps: torch.Tensor,
    p_steps: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
    prior_lambda: float,
) -> PonderLoss:
    max_steps = logits_steps.shape[0]
    per_step_losses = []
    for step in range(max_steps):
        per_step_losses.append(per_sample_cross_entropy(logits_steps[step], targets, mask))
    per_step_losses_t = torch.stack(per_step_losses, dim=0)

    rec_loss = (p_steps * per_step_losses_t).sum(dim=0).mean()
    prior = truncated_geometric_prior(max_steps, prior_lambda, logits_steps.device).unsqueeze(1)
    eps = 1e-12
    kl_loss = (p_steps * ((p_steps + eps).log() - (prior + eps).log())).sum(dim=0).mean()
    return PonderLoss(total=rec_loss + beta * kl_loss, reconstruction=rec_loss, kl=kl_loss)

