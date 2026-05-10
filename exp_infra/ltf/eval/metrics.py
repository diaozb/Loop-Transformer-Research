from __future__ import annotations

import torch


def answer_mask(mask: torch.Tensor, targets: torch.Tensor, pad_token: int = 3) -> torch.Tensor:
    return (mask == 1) & (targets != pad_token)


def sequence_accuracy(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_bool = mask == 1
    correct = []
    for sample_idx in range(targets.shape[0]):
        idx = mask_bool[sample_idx]
        if not idx.any():
            correct.append(torch.tensor(0.0, device=targets.device))
            continue
        correct.append((preds[sample_idx, idx] == targets[sample_idx, idx]).all().float())
    return torch.stack(correct).mean()


def token_accuracy(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_bool = mask == 1
    return (preds[mask_bool] == targets[mask_bool]).float().mean()


def exact_match_accuracy(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return sequence_accuracy(preds, targets, mask)
