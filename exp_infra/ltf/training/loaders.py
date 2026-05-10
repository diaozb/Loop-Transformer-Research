from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from ltf.config.schema import run_config_from_dict
from ltf.models import PonderLoopedModel, build_looped_model


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    config: Any
    step: int
    metrics: Dict[str, Any]
    raw: Dict[str, Any]


def load_checkpoint_for_eval(path: str | Path, map_location=None) -> LoadedCheckpoint:
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError(
            "Expected an exp_infra checkpoint with keys `config` and `model_state`. "
            "Legacy torch.save(model) checkpoints need a separate converter."
        )

    config = run_config_from_dict(checkpoint["config"])
    base_model = build_looped_model(config.model)
    model = PonderLoopedModel(base_model) if config.trainer.name == "ponder" else base_model
    model.load_state_dict(checkpoint["model_state"])
    return LoadedCheckpoint(
        model=model,
        config=config,
        step=int(checkpoint.get("step", -1)),
        metrics=checkpoint.get("metrics", {}) or {},
        raw=checkpoint,
    )

