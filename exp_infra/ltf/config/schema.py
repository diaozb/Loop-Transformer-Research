from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TaskConfig:
    name: str
    n_dims: int
    min_length: int = 1
    train_length_start: int = 2
    train_length_end: int = 20
    train_length_inc: int = 1
    train_length_interval: int = 1000
    test_length: int = 30
    modulus: int = 11
    copy_prob_one: float = 0.5


@dataclass
class ModelConfig:
    family: str = "gpt2"
    n_positions: int = 4096
    n_dims: int = 6
    n_embd: int = 256
    n_layer: int = 2
    n_head: int = 8
    linear_embedding: bool = True
    use_wpe: bool = False
    wpe_mode: str = "none"
    use_rope: bool = False
    rope_theta: float = 10000.0


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    lr_schedule: str = "none"


@dataclass
class TrainerConfig:
    name: str = "fixed_loop"
    train_steps: int = 100001
    batch_size: int = 64
    eval_batch_size: int = 512
    eval_every: int = 1000
    ema: bool = False
    beta: float = 0.01
    prior_lambda: float = 0.2
    ponder_n_steps: int = 20
    ponder_max_steps_cap: int = 128
    ponder_dynamic_n: bool = False


@dataclass
class EvalConfig:
    lengths: List[int] = field(default_factory=lambda: list(range(1, 41)))
    loop_counts: Optional[List[int]] = None
    num_samples: int = 2000
    batch_size: int = 512
    auto_exit: bool = True
    auto_exit_max_loops: int = 60
    run_after_train: bool = False
    after_train_checkpoint: str = "best"


@dataclass
class LoggingConfig:
    output_root: str = "exp_infra/runs"
    results_root: str = "exp_infra/results"
    figures_root: str = "exp_infra/figures"
    use_wandb: bool = False
    wandb_project: str = "looped-tf"
    wandb_entity: str = ""
    log_every_steps: int = 100


@dataclass
class RunConfig:
    seed: int = 42
    device: str = "cuda"
    task: TaskConfig = field(default_factory=lambda: TaskConfig(name="copy", n_dims=6))
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def dataclass_from_dict(cls, payload: Dict[str, Any]):
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")

    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for key, value in payload.items():
        if key not in known:
            raise ValueError(f"Unknown config field for {cls.__name__}: {key}")
        field_type = known[key].type
        default_value = getattr(cls(), key) if _can_default_construct(cls) else None
        if is_dataclass(default_value):
            kwargs[key] = dataclass_from_dict(type(default_value), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _can_default_construct(cls) -> bool:
    try:
        cls()
        return True
    except TypeError:
        return False


def run_config_from_dict(payload: Dict[str, Any]) -> RunConfig:
    data = dict(payload)
    if "task" in data:
        data["task"] = TaskConfig(**data["task"])
    if "model" in data:
        data["model"] = ModelConfig(**data["model"])
    if "optimizer" in data:
        data["optimizer"] = OptimizerConfig(**data["optimizer"])
    if "trainer" in data:
        data["trainer"] = TrainerConfig(**data["trainer"])
    if "eval" in data:
        data["eval"] = EvalConfig(**data["eval"])
    if "logging" in data:
        data["logging"] = LoggingConfig(**data["logging"])
    return RunConfig(**data)
