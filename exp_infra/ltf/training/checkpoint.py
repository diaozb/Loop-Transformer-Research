from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from ltf.config import RunConfig


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    config: RunConfig,
    step: int,
    metrics: Dict[str, Any] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "config": config.to_dict(),
            "step": step,
            "metrics": metrics or {},
        },
        output_path,
    )

