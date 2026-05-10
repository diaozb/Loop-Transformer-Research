from __future__ import annotations

import numpy as np
import torch


def convert_to_one_hot(int_matrix, n_dims: int = 6) -> np.ndarray:
    batch_size, seq_len = int_matrix.shape
    one_hot = np.zeros((batch_size, seq_len, n_dims), dtype=np.float32)
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            one_hot[batch_idx, seq_idx, int_matrix[batch_idx, seq_idx]] = 1.0
    return one_hot


def prepare_inputs(inputs: torch.Tensor, linear_embedding: bool, n_dims: int, device: torch.device | str) -> torch.Tensor:
    if linear_embedding:
        one_hot = convert_to_one_hot(inputs.detach().cpu().numpy(), n_dims=n_dims)
        return torch.tensor(one_hot, dtype=torch.float32, device=device)
    return inputs.to(device=device, dtype=torch.long)

