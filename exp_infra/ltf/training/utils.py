from __future__ import annotations

import random

import numpy as np
import torch

from ltf.config import TrainerConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_fixed_horizon(task_name: str, n_points: int) -> int:
    if task_name == "multi":
        return n_points * 2
    return n_points + 2


def resolve_ponder_horizon(task_name: str, n_points: int, trainer: TrainerConfig) -> int:
    if trainer.ponder_dynamic_n:
        return min(resolve_fixed_horizon(task_name, n_points), trainer.ponder_max_steps_cap)
    return min(trainer.ponder_n_steps, trainer.ponder_max_steps_cap)

