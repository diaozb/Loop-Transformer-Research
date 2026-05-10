from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict

import torch

from ltf.config import RunConfig
from ltf.data import generate_batch, legacy_max_len, prepare_inputs, task_pad_token
from ltf.eval import write_default_eval_outputs
from ltf.eval.metrics import answer_mask, sequence_accuracy, token_accuracy
from ltf.logging import RunLogger
from ltf.models import PonderLoopedModel, build_looped_model

from .checkpoint import save_checkpoint
from .curriculum import Curriculum
from .fixed_loop import fixed_loop_loss, select_fixed_loop_logits
from .loaders import load_checkpoint_for_eval
from .ponder import compute_ponder_training_loss
from .utils import resolve_fixed_horizon, resolve_ponder_horizon, set_seed


def run_training(config: RunConfig, progress_label: str | None = None, console_log: bool = True) -> str:
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    base_model = build_looped_model(config.model).to(device)
    model = PonderLoopedModel(base_model).to(device) if config.trainer.name == "ponder" else base_model
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = _build_lr_scheduler(optimizer, config)
    curriculum = Curriculum.from_task(config.task)
    logger = RunLogger(config)
    best_metric = float("-inf")
    started_at = perf_counter()
    label = progress_label or _default_progress_label(config)

    try:
        if console_log:
            print(_format_start_line(label, config, device, logger.run_dir), flush=True)
        for step in range(config.trainer.train_steps):
            batch = _sample_training_batch(config, curriculum.n_points)
            xs = prepare_inputs(
                batch.inputs,
                linear_embedding=config.model.linear_embedding,
                n_dims=config.model.n_dims,
                device=device,
            )

            optimizer.zero_grad()
            if config.trainer.name == "fixed_loop":
                horizon = resolve_fixed_horizon(config.task.name, curriculum.n_points)
                loss = fixed_loop_loss(base_model, xs, batch, horizon=horizon, task_name=config.task.name)
                metrics = {"loss": float(loss.item()), "n_points": curriculum.n_points}
            elif config.trainer.name == "ponder":
                horizon = resolve_ponder_horizon(config.task.name, curriculum.n_points, config.trainer)
                loss_obj = compute_ponder_training_loss(
                    model,
                    xs,
                    batch,
                    max_steps=horizon,
                    beta=config.trainer.beta,
                    prior_lambda=config.trainer.prior_lambda,
                )
                loss = loss_obj.total
                metrics = {
                    "loss": float(loss.item()),
                    "rec_loss": float(loss_obj.reconstruction.item()),
                    "kl_loss": float(loss_obj.kl.item()),
                    "n_points": curriculum.n_points,
                }
            else:
                raise ValueError(f"Unsupported trainer: {config.trainer.name}")

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip_norm)
            optimizer.step()
            _step_lr_scheduler(scheduler, config, step)
            metrics["grad_norm"] = float(grad_norm)
            metrics["lr"] = float(optimizer.param_groups[0]["lr"])

            if step % config.trainer.eval_every == 0 or step == config.trainer.train_steps - 1:
                eval_metrics = evaluate_once(model, config, device)
                metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                score = eval_metrics.get("accuracy", float("-inf"))
                if score >= best_metric:
                    best_metric = score
                    save_checkpoint(logger.run_dir / "checkpoints" / "best.pt", model, optimizer, config, step, metrics)
            logger.log_metrics(step, metrics)
            if console_log and _should_print_step(step, config):
                print(_format_step_line(label, step, config, metrics, perf_counter() - started_at), flush=True)
            curriculum.update()

        last_checkpoint = logger.run_dir / "checkpoints" / "last.pt"
        save_checkpoint(last_checkpoint, model, optimizer, config, config.trainer.train_steps - 1)
        if config.eval.run_after_train:
            eval_row = _run_post_train_eval(config, logger.run_dir, device)
            if console_log:
                print(_format_post_eval_line(label, eval_row), flush=True)
        if console_log:
            print(_format_done_line(label, logger.run_dir, best_metric, perf_counter() - started_at), flush=True)
        return str(logger.run_dir)
    finally:
        logger.close()


@torch.no_grad()
def evaluate_once(model, config: RunConfig, device: torch.device) -> Dict[str, float]:
    model.eval()
    try:
        length = config.task.test_length
        max_len = legacy_max_len(config.task.name, length)
        batch = generate_batch(
            config.task,
            batch_size=config.trainer.eval_batch_size,
            min_length=length,
            max_length_exclusive=length + 1,
            max_len=max_len,
        )
        xs = prepare_inputs(
            batch.inputs,
            linear_embedding=config.model.linear_embedding,
            n_dims=config.model.n_dims,
            device=device,
        )
        targets = batch.targets.to(device=device, dtype=torch.long)
        mask = batch.mask.to(device=device)

        if config.trainer.name == "ponder":
            horizon = resolve_ponder_horizon(config.task.name, length, config.trainer)
            output = model.forward_ponder(xs, max_steps=horizon, halt_mask=mask.bool())
            step_idx = output.p_steps.argmax(dim=0)
            batch_idx = torch.arange(xs.shape[0], device=device)
            logits = output.logits_steps[step_idx, batch_idx]
        else:
            base_model = model.base if hasattr(model, "base") else model
            horizon = resolve_fixed_horizon(config.task.name, length)
            logits_by_step = base_model.looped_forward(xs, horizon=horizon)
            logits = select_fixed_loop_logits(logits_by_step, batch.lengths.to(device), config.task.name)

        preds = logits.argmax(dim=-1)
        metrics = {"accuracy": float(sequence_accuracy(preds, targets, mask).item())}
        ans_mask = answer_mask(mask, targets, pad_token=task_pad_token(config.task.name, config.task.modulus))
        if ans_mask.any():
            metrics["answer_accuracy"] = float(sequence_accuracy(preds, targets, ans_mask).item())
        if config.task.name == "copy":
            metrics["token_accuracy"] = float(token_accuracy(preds, targets, mask).item())
        return metrics
    finally:
        model.train()


def _sample_training_batch(config: RunConfig, n_points: int):
    return generate_batch(
        config.task,
        batch_size=config.trainer.batch_size,
        min_length=config.task.min_length,
        max_length_exclusive=n_points,
        max_len=legacy_max_len(config.task.name, n_points),
    )


def _run_post_train_eval(config: RunConfig, run_dir: Path, device: torch.device) -> Dict[str, object]:
    checkpoint_name = config.eval.after_train_checkpoint
    if checkpoint_name not in ("best", "last"):
        raise ValueError("eval.after_train_checkpoint must be one of: best, last")

    checkpoint_path = run_dir / "checkpoints" / f"{checkpoint_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Post-train eval checkpoint does not exist: {checkpoint_path}")

    loaded = load_checkpoint_for_eval(checkpoint_path, map_location=device)
    loaded.model.to(device)
    metrics = evaluate_once(loaded.model, loaded.config, device)
    row = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_tag": checkpoint_name,
        "step": loaded.step,
        "task": loaded.config.task.name,
        "trainer": loaded.config.trainer.name,
        "test_length": loaded.config.task.test_length,
        **metrics,
    }
    write_default_eval_outputs(run_dir / "eval" / f"default_{checkpoint_name}", row)
    return row


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, config: RunConfig):
    if config.optimizer.lr_schedule == "none":
        return None
    if config.optimizer.lr_schedule != "cosine_after_curriculum":
        raise ValueError(f"Unsupported lr_schedule: {config.optimizer.lr_schedule}")

    boundary = config.task.train_length_end * config.task.train_length_interval
    t_max = max(1, config.trainer.train_steps - boundary)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=0.0)


def _step_lr_scheduler(scheduler, config: RunConfig, step: int) -> None:
    if scheduler is None:
        return
    boundary = config.task.train_length_end * config.task.train_length_interval
    if step > boundary:
        scheduler.step()


def _default_progress_label(config: RunConfig) -> str:
    configured_name = getattr(config, "run_name", None)
    if configured_name:
        return str(configured_name)
    return f"{config.task.name}/{config.trainer.name}/seed{config.seed}"


def _should_print_step(step: int, config: RunConfig) -> bool:
    return (
        step == 0
        or step == config.trainer.train_steps - 1
        or step % config.trainer.eval_every == 0
    )


def _format_start_line(label: str, config: RunConfig, device: torch.device, run_dir: Path) -> str:
    return (
        f"[{label}] start "
        f"task={config.task.name} trainer={config.trainer.name} pe={_position_mode(config)} "
        f"seed={config.seed} steps={config.trainer.train_steps} "
        f"arch=L{config.model.n_layer}/H{config.model.n_head}/E{config.model.n_embd} "
        f"device={device} run_dir={run_dir}"
    )


def _format_step_line(label: str, step: int, config: RunConfig, metrics: Dict[str, float], elapsed: float) -> str:
    metric_text = _format_selected_metrics(metrics)
    return (
        f"[{label}] step={step}/{config.trainer.train_steps - 1} "
        f"iter={step + 1}/{config.trainer.train_steps} "
        f"n_points={metrics.get('n_points', 'NA')} {metric_text} "
        f"elapsed={_format_duration(elapsed)}"
    )


def _format_post_eval_line(label: str, row: Dict[str, object]) -> str:
    metric_text = _format_selected_metrics(row, keys=("accuracy", "answer_accuracy", "token_accuracy"))
    return (
        f"[{label}] post_eval checkpoint={row['checkpoint_tag']} "
        f"step={row['step']} test_length={row['test_length']} {metric_text}"
    )


def _format_done_line(label: str, run_dir: Path, best_metric: float, elapsed: float) -> str:
    best = "NA" if best_metric == float("-inf") else f"{best_metric:.4f}"
    return f"[{label}] done best_accuracy={best} elapsed={_format_duration(elapsed)} run_dir={run_dir}"


def _format_selected_metrics(metrics: Dict[str, object], keys=None) -> str:
    if keys is None:
        keys = (
            "loss",
            "rec_loss",
            "kl_loss",
            "grad_norm",
            "lr",
            "eval_accuracy",
            "eval_answer_accuracy",
            "eval_token_accuracy",
        )
    parts = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _position_mode(config: RunConfig) -> str:
    if config.model.use_rope:
        return "rope"
    if config.model.use_wpe:
        return f"wpe_{config.model.wpe_mode}"
    return "nope"
