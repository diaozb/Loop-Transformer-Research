from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class TaskBatch:
    inputs: torch.Tensor
    lengths: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device | str) -> "TaskBatch":
        return TaskBatch(
            inputs=self.inputs.to(device),
            lengths=self.lengths.to(device),
            targets=self.targets.to(device),
            mask=self.mask.to(device),
        )


def generate_parity(
    batch_size: int,
    max_len: int,
    min_length: int = 1,
    max_length_exclusive: int = 10,
) -> TaskBatch:
    lengths = np.random.randint(min_length, max_length_exclusive, size=(batch_size, 1))
    num_digits = lengths.flatten()

    inputs = np.full((batch_size, max_len), 3)
    targets = np.full((batch_size, max_len), 3)
    mask = np.full((batch_size, max_len), 0)

    for i in range(batch_size):
        inputs[i, : num_digits[i]] = np.random.randint(low=0, high=2, size=num_digits[i])
        inputs[i, num_digits[i]] = 2
        targets[i, : num_digits[i]] = 5
        targets[i, num_digits[i]] = np.sum(inputs[i, : num_digits[i]]) % 2
        mask[i, num_digits[i] :] = 1

    return _batch(inputs, lengths, targets, mask)


def generate_copy(
    batch_size: int,
    max_len: int,
    min_length: int = 1,
    max_length_exclusive: int = 10,
    prob_one: float = 0.5,
) -> TaskBatch:
    lengths = np.random.randint(min_length, max_length_exclusive, size=(batch_size, 1))
    num_digits = lengths.flatten()

    inputs = np.full((batch_size, 2 * max_len), 3)
    targets = np.full((batch_size, 2 * max_len), 3)
    mask = np.full((batch_size, 2 * max_len), 0)

    for i in range(batch_size):
        inputs[i, : num_digits[i]] = np.random.choice(
            [0, 1],
            size=num_digits[i],
            p=[1.0 - prob_one, prob_one],
        )
        inputs[i, num_digits[i]] = 2
        targets[i, : num_digits[i]] = 4
        targets[i, num_digits[i] : 2 * num_digits[i]] = inputs[i, : num_digits[i]]
        mask[i, num_digits[i] :] = 1

    return _batch(inputs, lengths, targets, mask)


def generate_mod_add(
    batch_size: int,
    max_len: int,
    min_length: int = 1,
    max_length_exclusive: int = 10,
    modulus: int = 11,
) -> TaskBatch:
    lengths = np.random.randint(min_length, max_length_exclusive, size=(batch_size, 1))
    num_digits = lengths.flatten()

    pad_token = modulus
    ignore_token = modulus + 1

    inputs = np.full((batch_size, max_len), pad_token)
    targets = np.full((batch_size, max_len), pad_token)
    mask = np.full((batch_size, max_len), 0)

    for i in range(batch_size):
        inputs[i, : num_digits[i]] = np.random.randint(low=0, high=modulus, size=num_digits[i])
        targets[i, : num_digits[i]] = ignore_token
        targets[i, num_digits[i]] = np.sum(inputs[i, : num_digits[i]]) % modulus
        mask[i, num_digits[i] :] = 1

    return _batch(inputs, lengths, targets, mask)


def legacy_max_len(task_name: str, n_points: int, modulus: Optional[int] = None) -> int:
    if task_name == "mod_add":
        return n_points + 1
    if task_name in ("parity", "copy"):
        return n_points + 1
    raise ValueError(f"Unsupported task for legacy_max_len: {task_name}")


def task_pad_token(task_name: str, modulus: int = 11) -> int:
    if task_name in ("parity", "copy"):
        return 3
    if task_name == "mod_add":
        return modulus
    raise ValueError(f"Unsupported task for task_pad_token: {task_name}")


def _batch(inputs: np.ndarray, lengths: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> TaskBatch:
    return TaskBatch(
        inputs=torch.tensor(inputs),
        lengths=torch.tensor(lengths),
        targets=torch.tensor(targets),
        mask=torch.tensor(mask),
    )
