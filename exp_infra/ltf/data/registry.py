from __future__ import annotations

from typing import Callable, Dict

from ltf.config import TaskConfig

from .tasks import TaskBatch, generate_copy, generate_mod_add, generate_parity, legacy_max_len


TASK_GENERATORS: Dict[str, Callable[..., TaskBatch]] = {
    "parity": generate_parity,
    "copy": generate_copy,
    "mod_add": generate_mod_add,
}


def generate_batch(
    task: TaskConfig,
    batch_size: int,
    min_length: int,
    max_length_exclusive: int,
    max_len: int | None = None,
) -> TaskBatch:
    if task.name not in TASK_GENERATORS:
        raise ValueError(f"Unsupported task: {task.name}")
    resolved_max_len = max_len
    if resolved_max_len is None:
        resolved_max_len = legacy_max_len(task.name, max_length_exclusive)

    if task.name == "copy":
        return generate_copy(
            batch_size,
            min_length=min_length,
            max_length_exclusive=max_length_exclusive,
            max_len=resolved_max_len,
            prob_one=task.copy_prob_one,
        )
    if task.name == "mod_add":
        return generate_mod_add(
            batch_size,
            min_length=min_length,
            max_length_exclusive=max_length_exclusive,
            max_len=resolved_max_len,
            modulus=task.modulus,
        )
    return generate_parity(
        batch_size,
        min_length=min_length,
        max_length_exclusive=max_length_exclusive,
        max_len=resolved_max_len,
    )

