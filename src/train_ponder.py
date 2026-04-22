import os
import random
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from curriculum import Curriculum
from generate_training_data import (
    generate_prompt_matrix_addition,
    generate_prompt_matrix_copy,
    generate_prompt_matrix_dict,
    generate_prompt_matrix_mod_add,
    generate_prompt_matrix_mod_add_digits,
    generate_prompt_matrix_multi,
    generate_prompt_matrix_parity,
    generate_prompt_matrix_sum_reverse,
)
from models import build_general_model
from utils import convert_to_one_hot

# =========================
# Editable Run Parameters
# =========================
TASK = "copy"
OUT_DIR = "../models"
SEED = 42

TRAIN_STEPS = 100001
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
EVAL_EVERY = 1000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0

N_POSITIONS = 4096
MODULUS = 11

# Optional overrides (set to None to use TASK_PRESETS defaults)
TEST_LEN = None
MODEL_N_DIMS = None
MODEL_N_EMBD = None
MODEL_N_LAYER = None
MODEL_N_HEAD = None
MODEL_LINEAR_EMBEDDING = None
MODEL_USE_WPE = None
MODEL_USE_ROPE = None
MODEL_ROPE_THETA = 10000.0

CURR_START = None
CURR_END = None
CURR_INC = None
CURR_INTERVAL = None

# PONDER_BETA = 0.01
PONDER_BETA = 0.0
PONDER_PRIOR_LAMBDA = 0.2
PONDER_MAX_STEPS_CAP = 128
PONDER_N_STEPS = 20
PONDER_DYNAMIC_N = False


TASK_GENERATORS = {
    "parity": generate_prompt_matrix_parity,
    "copy": generate_prompt_matrix_copy,
    "addition": generate_prompt_matrix_addition,
    "multi": generate_prompt_matrix_multi,
    "sum_reverse": generate_prompt_matrix_sum_reverse,
    "dict": generate_prompt_matrix_dict,
    "mod_add": generate_prompt_matrix_mod_add,
    "mod_add_digits": generate_prompt_matrix_mod_add_digits,
}


TASK_PRESETS: Dict[str, Dict] = {
    "parity": dict(n_dims=6, linear_embedding=True, n_embd=256, n_layer=1, n_head=64, use_wpe=False, use_rope=True, test_len=40, curriculum=dict(start=2, end=21, inc=1, interval=500)),
    "copy": dict(n_dims=6, linear_embedding=True, n_embd=256, n_layer=2, n_head=8, use_wpe=False, use_rope=True, test_len=30, curriculum=dict(start=2, end=20, inc=1, interval=1000)),
    "addition": dict(n_dims=6, linear_embedding=True, n_embd=256, n_layer=3, n_head=8, use_wpe=False, use_rope=False, test_len=30, curriculum=dict(start=2, end=20, inc=1, interval=2500)),
    "sum_reverse": dict(n_dims=6, linear_embedding=True, n_embd=256, n_layer=2, n_head=16, use_wpe=False, use_rope=False, test_len=24, curriculum=dict(start=2, end=20, inc=1, interval=500)),
    "dict": dict(n_dims=60, linear_embedding=False, n_embd=256, n_layer=3, n_head=8, use_wpe=False, use_rope=False, test_len=30, curriculum=dict(start=2, end=20, inc=1, interval=1000)),
    "multi": dict(n_dims=6, linear_embedding=True, n_embd=256, n_layer=4, n_head=8, use_wpe=False, use_rope=False, test_len=15, curriculum=dict(start=2, end=12, inc=1, interval=500)),
    "mod_add": dict(n_dims=13, linear_embedding=True, n_embd=256, n_layer=2, n_head=64, use_wpe=False, use_rope=False, test_len=30, curriculum=dict(start=2, end=20, inc=1, interval=2500)),
    "mod_add_digits": dict(n_dims=14, linear_embedding=True, n_embd=1024, n_layer=2, n_head=64, use_wpe=False, use_rope=False, test_len=10, curriculum=dict(start=2, end=5, inc=1, interval=5000)),
}


@dataclass
class PonderConfig:
    beta: float = 0.01
    prior_lambda: float = 0.2
    max_steps_cap: int = 128
    dynamic_n: bool = False
    n_steps: int = 32


class PonderLoopedModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        hidden_dim = base_model._backbone.config.n_embd
        self.halt_head = nn.Linear(hidden_dim, 1)

    def _first_answer_hidden(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Use the first answer position per sample as halting signal input.
        # mask is expected to mark answer region (True from first answer token onward).
        first_idx = mask.long().argmax(dim=1)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, first_idx]

    def forward_ponder(self, xs: torch.Tensor, max_steps: int, halt_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zs = self.base._read_in(xs)
        output = torch.zeros_like(zs)
        batch_size = xs.shape[0]
        alive_prob = torch.ones(batch_size, device=xs.device)
        logits_steps = []
        p_steps = []
        use_wpe = getattr(self.base, "use_wpe", False)

        for step in range(max_steps):
            output = self.base.forward_single(output + zs, add_wpe=use_wpe)
            logits = self.base._read_out(output)
            pooled = self._first_answer_hidden(output, halt_mask)
            if step == max_steps - 1:
                lambda_n = torch.ones(batch_size, device=xs.device)
            else:
                lambda_n = torch.sigmoid(self.halt_head(pooled)).squeeze(-1)
            p_n = alive_prob * lambda_n
            alive_prob = alive_prob * (1.0 - lambda_n)

            logits_steps.append(logits)
            p_steps.append(p_n)

        logits_steps = torch.stack(logits_steps, dim=0)  # [N, B, T, V]
        p_steps = torch.stack(p_steps, dim=0)  # [N, B]
        p_steps = p_steps / p_steps.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return logits_steps, p_steps


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_config():
    if TASK not in TASK_PRESETS:
        raise ValueError(f"Unknown TASK='{TASK}'. Valid tasks: {list(TASK_PRESETS.keys())}")
    if PONDER_N_STEPS < 1:
        raise ValueError("PONDER_N_STEPS must be >= 1")
    preset = TASK_PRESETS[TASK]
    model_conf = dict(
        family="gpt2",
        n_positions=N_POSITIONS,
        n_dims=MODEL_N_DIMS if MODEL_N_DIMS is not None else preset["n_dims"],
        n_embd=MODEL_N_EMBD if MODEL_N_EMBD is not None else preset["n_embd"],
        n_layer=MODEL_N_LAYER if MODEL_N_LAYER is not None else preset["n_layer"],
        n_head=MODEL_N_HEAD if MODEL_N_HEAD is not None else preset["n_head"],
        linear_embedding=MODEL_LINEAR_EMBEDDING if MODEL_LINEAR_EMBEDDING is not None else preset["linear_embedding"],
        use_wpe=MODEL_USE_WPE if MODEL_USE_WPE is not None else preset["use_wpe"],
        use_rope=MODEL_USE_ROPE if MODEL_USE_ROPE is not None else preset["use_rope"],
        rope_theta=MODEL_ROPE_THETA,
    )
    curriculum = dict(
        points=SimpleNamespace(
            start=CURR_START if CURR_START is not None else preset["curriculum"]["start"],
            end=CURR_END if CURR_END is not None else preset["curriculum"]["end"],
            inc=CURR_INC if CURR_INC is not None else preset["curriculum"]["inc"],
            interval=CURR_INTERVAL if CURR_INTERVAL is not None else preset["curriculum"]["interval"],
        )
    )
    test_len = TEST_LEN if TEST_LEN is not None else preset["test_len"]
    return SimpleNamespace(
        task=TASK,
        seed=SEED,
        out_dir=OUT_DIR,
        train_steps=TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        eval_every=EVAL_EVERY,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        n_positions=N_POSITIONS,
        test_len=test_len,
        modulus=MODULUS,
        model=SimpleNamespace(**model_conf),
        curriculum=SimpleNamespace(**curriculum),
        ponder=PonderConfig(
            beta=PONDER_BETA,
            prior_lambda=PONDER_PRIOR_LAMBDA,
            max_steps_cap=PONDER_MAX_STEPS_CAP,
            dynamic_n=PONDER_DYNAMIC_N,
            n_steps=PONDER_N_STEPS,
        ),
    )


def compute_max_len(task: str, n_points: int) -> int:
    if task == "mod_add_digits":
        return 3 * (n_points + 1)
    return n_points + 1


def compute_horizon(task: str, n_points: int, cap: int) -> int:
    if task == "multi":
        h = 2 * n_points
    else:
        h = n_points + 2
    return min(h, cap)


def resolve_horizon(task: str, n_points: int, ponder_cfg: PonderConfig) -> int:
    if ponder_cfg.dynamic_n:
        return compute_horizon(task, n_points, ponder_cfg.max_steps_cap)
    return min(ponder_cfg.n_steps, ponder_cfg.max_steps_cap)


def sample_batch(task: str, batch_size: int, min_num_digits: int, max_num_digits: int, max_len: int, modulus: int):
    if task == "multi":
        xs, _, _, ys, mask = generate_prompt_matrix_multi(
            batch_size,
            min_num_digits=min_num_digits,
            max_num_digits=max_num_digits,
            max_len=max_len,
        )
    elif task == "mod_add":
        xs, _, ys, mask = generate_prompt_matrix_mod_add(
            batch_size,
            min_num_digits=min_num_digits,
            max_num_digits=max_num_digits,
            max_len=max_len,
            modulus=modulus,
        )
    elif task == "mod_add_digits":
        xs, _, ys, mask = generate_prompt_matrix_mod_add_digits(
            batch_size,
            min_num_digits=min_num_digits,
            max_num_digits=max_num_digits,
            max_len=max_len,
            modulus=modulus,
        )
    else:
        xs, _, ys, mask = TASK_GENERATORS[task](
            batch_size,
            min_num_digits=min_num_digits,
            max_num_digits=max_num_digits,
            max_len=max_len,
        )
    return xs, ys, mask


def prepare_inputs(xs: torch.Tensor, model_cfg: SimpleNamespace, device: torch.device) -> torch.Tensor:
    if model_cfg.linear_embedding:
        xs_oh = convert_to_one_hot(xs.cpu().numpy(), n_dims=model_cfg.n_dims)
        return torch.tensor(xs_oh, dtype=torch.float32, device=device)
    return xs.to(device=device, dtype=torch.long)


def per_sample_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    mask_f = mask.float()
    return (ce * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def truncated_geometric_prior(max_steps: int, lambda_p: float, device: torch.device) -> torch.Tensor:
    probs = []
    for n in range(max_steps - 1):
        probs.append(lambda_p * ((1.0 - lambda_p) ** n))
    probs.append((1.0 - lambda_p) ** (max_steps - 1))
    return torch.tensor(probs, dtype=torch.float32, device=device)


def weighted_exact_match(logits_steps: torch.Tensor, p_steps: torch.Tensor, ys: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    step_idx = p_steps.argmax(dim=0)  # [B]
    b = ys.shape[0]
    b_idx = torch.arange(b, device=ys.device)
    picked_logits = logits_steps[step_idx, b_idx]  # [B, T, V]
    preds = picked_logits.argmax(dim=-1)
    mask_bool = mask.bool()
    correct = []
    for i in range(b):
        idx = mask_bool[i]
        correct.append((preds[i, idx] == ys[i, idx]).all().float())
    acc = torch.stack(correct).mean().item()
    expected_steps = (p_steps * (torch.arange(1, p_steps.shape[0] + 1, device=p_steps.device).unsqueeze(1))).sum(dim=0).mean().item()
    return acc, expected_steps


@torch.no_grad()
def evaluate(model: PonderLoopedModel, cfg, device: torch.device, test_len: int, batch_size: int = 512, n_batches: int = 8):
    model.eval()
    accs, exp_steps = [], []
    max_len = compute_max_len(cfg.task, test_len)
    horizon = resolve_horizon(cfg.task, test_len, cfg.ponder)

    for _ in range(n_batches):
        xs, ys, mask = sample_batch(
            cfg.task,
            batch_size,
            min_num_digits=test_len,
            max_num_digits=test_len + 1,
            max_len=max_len,
            modulus=cfg.modulus,
        )
        xs = prepare_inputs(xs, cfg.model, device)
        ys = ys.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.bool)
        logits_steps, p_steps = model.forward_ponder(xs, horizon, halt_mask=mask)
        acc, e_step = weighted_exact_match(logits_steps, p_steps, ys, mask)
        accs.append(acc)
        exp_steps.append(e_step)

    model.train()
    return float(np.mean(accs)), float(np.mean(exp_steps))


def train(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.model
    if model_cfg.use_wpe and model_cfg.use_rope:
        raise ValueError("use_wpe and use_rope cannot both be True.")

    base_model = build_general_model(model_cfg)
    model = PonderLoopedModel(base_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    curriculum = Curriculum(cfg.curriculum)
    prior_eps = 1e-12

    run_id = str(uuid.uuid4())
    out_dir = os.path.join(cfg.out_dir, f"{cfg.task}_ponder", run_id)
    if cfg.task in ("mod_add", "mod_add_digits"):
        out_dir = os.path.join(cfg.out_dir, f"{cfg.task}_ponder", f"mod_{cfg.modulus}", run_id)
    os.makedirs(out_dir, exist_ok=True)
    config_payload = {
        "task": cfg.task,
        "seed": cfg.seed,
        "train_steps": cfg.train_steps,
        "batch_size": cfg.batch_size,
        "eval_every": cfg.eval_every,
        "eval_batch_size": cfg.eval_batch_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "test_len": cfg.test_len,
        "modulus": cfg.modulus,
        "model": vars(cfg.model),
        "ponder": vars(cfg.ponder),
        "curriculum": {
            "start": cfg.curriculum.points.start,
            "end": cfg.curriculum.points.end,
            "inc": cfg.curriculum.points.inc,
            "interval": cfg.curriculum.points.interval,
        },
    }
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)

    best_acc = -1.0
    pbar = tqdm(range(cfg.train_steps))
    for step in pbar:
        n_points = curriculum.n_points
        max_len = compute_max_len(cfg.task, n_points)
        horizon = resolve_horizon(cfg.task, n_points, cfg.ponder)

        xs, ys, mask = sample_batch(
            cfg.task,
            cfg.batch_size,
            min_num_digits=1,
            max_num_digits=n_points,
            max_len=max_len,
            modulus=cfg.modulus,
        )
        xs = prepare_inputs(xs, model_cfg, device)
        ys = ys.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.bool)

        optimizer.zero_grad()
        logits_steps, p_steps = model.forward_ponder(xs, horizon, halt_mask=mask)
        per_step_losses = []
        for t in range(horizon):
            per_step_losses.append(per_sample_ce(logits_steps[t], ys, mask))
        per_step_losses = torch.stack(per_step_losses, dim=0)  # [N, B]

        rec_loss = (p_steps * per_step_losses).sum(dim=0).mean()
        prior = truncated_geometric_prior(horizon, cfg.ponder.prior_lambda, device).unsqueeze(1)
        kl_loss = (p_steps * ((p_steps + prior_eps).log() - (prior + prior_eps).log())).sum(dim=0).mean()
        loss = rec_loss + cfg.ponder.beta * kl_loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % cfg.eval_every == 0:
            eval_acc, eval_steps = evaluate(model, cfg, device, cfg.test_len, cfg.eval_batch_size, n_batches=4)
            if eval_acc >= best_acc:
                best_acc = eval_acc
                torch.save(model, os.path.join(out_dir, "best.pt"))
            print(
                f"[step {step}] loss={loss.item():.4f} rec={rec_loss.item():.4f} kl={kl_loss.item():.4f} "
                f"grad_norm={float(grad_norm):.4f} eval_acc={eval_acc:.4f} eval_E[steps]={eval_steps:.2f}"
            )

        pbar.set_description(
            f"loss={loss.item():.4f} rec={rec_loss.item():.4f} kl={kl_loss.item():.4f} n_points={n_points}"
        )
        curriculum.update()

    torch.save(model, os.path.join(out_dir, "model.pt"))
    print(f"Training done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    cfg = build_train_config()
    train(cfg)
