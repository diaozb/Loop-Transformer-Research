from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class PonderForwardOutput:
    logits_steps: torch.Tensor
    p_steps: torch.Tensor
    lambda_steps: torch.Tensor
    hidden_steps: List[torch.Tensor]


class PonderLoopedModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        hidden_dim = base_model._backbone.config.n_embd
        self.halt_head = nn.Linear(hidden_dim, 1)

    def first_answer_hidden(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        first_idx = mask.long().argmax(dim=1)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, first_idx]

    def forward_ponder(self, xs: torch.Tensor, max_steps: int, halt_mask: torch.Tensor) -> PonderForwardOutput:
        input_embed = self.base._read_in(xs)
        output = torch.zeros_like(input_embed)
        batch_size = xs.shape[0]
        alive_prob = torch.ones(batch_size, device=xs.device)
        logits_steps = []
        p_steps = []
        lambda_steps = []
        hidden_steps = []
        use_wpe = getattr(self.base, "use_wpe", False)

        for step in range(max_steps):
            output = self.base.forward_single(output + input_embed, add_wpe=use_wpe, step_idx=step)
            logits = self.base._read_out(output)
            pooled = self.first_answer_hidden(output, halt_mask)
            if step == max_steps - 1:
                lambda_n = torch.ones(batch_size, device=xs.device)
            else:
                lambda_n = torch.sigmoid(self.halt_head(pooled)).squeeze(-1)
            p_n = alive_prob * lambda_n
            alive_prob = alive_prob * (1.0 - lambda_n)

            hidden_steps.append(output)
            logits_steps.append(logits)
            p_steps.append(p_n)
            lambda_steps.append(lambda_n)

        p_tensor = torch.stack(p_steps, dim=0)
        p_tensor = p_tensor / p_tensor.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return PonderForwardOutput(
            logits_steps=torch.stack(logits_steps, dim=0),
            p_steps=p_tensor,
            lambda_steps=torch.stack(lambda_steps, dim=0),
            hidden_steps=hidden_steps,
        )

