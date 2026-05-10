from __future__ import annotations

from typing import Dict, Iterable

import torch

from .metrics import answer_mask, sequence_accuracy, token_accuracy


@torch.no_grad()
def evaluate_logits_by_loop(
    logits_by_step,
    targets: torch.Tensor,
    mask: torch.Tensor,
    loop_counts: Iterable[int],
    task_name: str,
) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    for loop_count in loop_counts:
        logits = logits_by_step[loop_count - 1]
        preds = logits.argmax(dim=-1)
        row = {
            "accuracy": float(sequence_accuracy(preds, targets, mask).item()),
        }
        ans_mask = answer_mask(mask, targets)
        if ans_mask.any():
            row["answer_accuracy"] = float(sequence_accuracy(preds, targets, ans_mask).item())
        if task_name == "copy":
            row["token_accuracy"] = float(token_accuracy(preds, targets, mask).item())
        results[loop_count] = row
    return results

