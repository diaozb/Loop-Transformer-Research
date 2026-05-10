import os
import random
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import wandb

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
# Target experiment: NoPE + Copy + PonderNet
TASK = "copy"
OUT_DIR = "../models/ape_once_baselines"
SEED = 42

# Train on lengths [1, TRAIN_MAX_LEN].
# Note: generators use np.random.randint(min_num_digits, max_num_digits), so max_num_digits is exclusive.
TRAIN_MAX_LEN = 20

# W&B
USE_WANDB = True
WANDB_PROJECT = "looped-tf-ape-once-copy-ponder"
WANDB_ENTITY = None   # Fill this if you want a specific W&B entity/team.
WANDB_MODE = "online" # Options: "online", "offline", "disabled"
WANDB_LOG_EVERY = 100
UPLOAD_MODEL_ARTIFACT = True
UPLOAD_DIAGNOSTIC_ARTIFACT = True

# Training
TRAIN_STEPS = 100001
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
EVAL_EVERY = 1000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0

N_POSITIONS = 4096
MODULUS = 11

# Quick training-time eval length. Final diagnostics below will evaluate all ID/OOD lengths.
TEST_LEN = 30

# Model overrides. These force NoPE for Copy, because the copy preset defaults to RoPE.
MODEL_N_DIMS = None
MODEL_N_EMBD = None
MODEL_N_LAYER = None
MODEL_N_HEAD = None
MODEL_LINEAR_EMBEDDING = None
MODEL_USE_WPE = True
MODEL_WPE_MODE = "once"
MODEL_USE_ROPE = False
MODEL_ROPE_THETA = 10000.0

# Curriculum for train lengths [1, 20].
# At curriculum.n_points = 2, sampled lengths are [1].
# At curriculum.n_points = 21, sampled lengths are [1, 20].
CURR_START = 2
CURR_END = TRAIN_MAX_LEN + 1
CURR_INC = 1
CURR_INTERVAL = 1000

# PonderNet
PONDER_BETA = 0.01
PONDER_PRIOR_LAMBDA = 0.2
PONDER_MAX_STEPS_CAP = 128
PONDER_N_STEPS = 20
PONDER_DYNAMIC_N = False

# Final diagnostics. No L^2 test for now.
RUN_FINAL_DIAGNOSTICS = True
DIAG_MAX_STEPS = 40
DIAG_BATCH_SIZE = 128
DIAG_N_BATCHES = 4
ID_LENGTHS = list(range(1, TRAIN_MAX_LEN + 1))
OOD_LENGTHS = [TRAIN_MAX_LEN + 1, TRAIN_MAX_LEN + 2, 2 * TRAIN_MAX_LEN, 3 * TRAIN_MAX_LEN]


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
            output = self.base.forward_single(output + zs, add_wpe=use_wpe, step_idx=step)
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
        p_steps = torch.stack(p_steps, dim=0)            # [N, B]
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
        wpe_mode=MODEL_WPE_MODE if MODEL_WPE_MODE is not None else preset.get("wpe_mode"),
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
        train_max_len=TRAIN_MAX_LEN,
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
        wandb=SimpleNamespace(
            enabled=USE_WANDB,
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            mode=WANDB_MODE,
            log_every=WANDB_LOG_EVERY,
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


def exact_match_from_logits(logits: torch.Tensor, ys: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = ((preds == ys) | (~mask)).all(dim=1)
    return correct.float().mean().item()


def copy_answer_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    # For copy with fixed input length L:
    # source positions: 0 ... L-1
    # first answer position: L
    # true copied answer region: L ... 2L-1
    ans_mask = torch.zeros_like(mask, dtype=torch.bool)
    ans_mask[:, length:2 * length] = True
    return ans_mask


def get_metric_mask(task: str, mask: torch.Tensor, length: int) -> torch.Tensor:
    # full mask matches the current training objective.
    # answer mask is cleaner for Copy analysis, so diagnostics report both.
    if task == "copy":
        return copy_answer_mask(mask, length)
    return mask.bool()


def weighted_exact_match(logits_steps: torch.Tensor, p_steps: torch.Tensor, ys: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    step_idx = p_steps.argmax(dim=0)  # [B]
    b = ys.shape[0]
    b_idx = torch.arange(b, device=ys.device)
    picked_logits = logits_steps[step_idx, b_idx]  # [B, T, V]
    acc = exact_match_from_logits(picked_logits, ys, mask.bool())
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


@torch.no_grad()
def diagnose_one_length(
    model: PonderLoopedModel,
    cfg,
    device: torch.device,
    length: int,
    max_steps: int,
    batch_size: int,
    n_batches: int,
    split: str,
):
    model.eval()

    step_exit_prob_sum = np.zeros(max_steps, dtype=np.float64)
    step_loss_full_sum = np.zeros(max_steps, dtype=np.float64)
    step_loss_answer_sum = np.zeros(max_steps, dtype=np.float64)
    step_acc_full_sum = np.zeros(max_steps, dtype=np.float64)
    step_acc_answer_sum = np.zeros(max_steps, dtype=np.float64)
    step_delta_sum = np.zeros(max_steps, dtype=np.float64)
    step_cos_sum = np.zeros(max_steps, dtype=np.float64)
    step_change_sum = np.zeros(max_steps, dtype=np.float64)

    auto_acc_full_all = []
    auto_acc_answer_all = []
    expected_steps_all = []
    argmax_steps_all = []

    use_wpe = getattr(model.base, "use_wpe", False)

    for _ in range(n_batches):
        max_len = compute_max_len(cfg.task, length)
        xs, ys, mask = sample_batch(
            cfg.task,
            batch_size,
            min_num_digits=length,
            max_num_digits=length + 1,
            max_len=max_len,
            modulus=cfg.modulus,
        )

        xs = prepare_inputs(xs, cfg.model, device)
        ys = ys.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.bool)
        answer_mask = get_metric_mask(cfg.task, mask, length)

        zs = model.base._read_in(xs)
        output = torch.zeros_like(zs)
        alive_prob = torch.ones(xs.shape[0], device=device)

        # For argmax-exit evaluation.
        best_p = torch.full((xs.shape[0],), -1.0, device=device)
        argmax_step = torch.zeros(xs.shape[0], device=device)
        chosen_logits: Optional[torch.Tensor] = None

        # For expected exit step.
        expected_steps = torch.zeros(xs.shape[0], device=device)

        # For convergence metrics.
        prev_hidden: Optional[torch.Tensor] = None
        prev_preds: Optional[torch.Tensor] = None

        for step in range(max_steps):
            output = model.base.forward_single(output + zs, add_wpe=use_wpe, step_idx=step)
            logits = model.base._read_out(output)

            pooled = model._first_answer_hidden(output, mask)
            if step == max_steps - 1:
                lambda_t = torch.ones(xs.shape[0], device=device)
            else:
                lambda_t = torch.sigmoid(model.halt_head(pooled)).squeeze(-1)

            p_t = alive_prob * lambda_t
            alive_prob = alive_prob * (1.0 - lambda_t)
            expected_steps += (step + 1) * p_t
            step_exit_prob_sum[step] += p_t.mean().item()

            # Per-step forced losses and accuracies.
            step_loss_full_sum[step] += per_sample_ce(logits, ys, mask).mean().item()
            step_loss_answer_sum[step] += per_sample_ce(logits, ys, answer_mask).mean().item()
            step_acc_full_sum[step] += exact_match_from_logits(logits, ys, mask)
            step_acc_answer_sum[step] += exact_match_from_logits(logits, ys, answer_mask)

            preds = logits.argmax(dim=-1)

            # Hidden-state convergence on answer positions only.
            if prev_hidden is not None and prev_preds is not None:
                h_now = output[answer_mask]
                h_prev = prev_hidden[answer_mask]
                if h_now.numel() > 0:
                    step_delta_sum[step] += (h_now - h_prev).norm(dim=-1).mean().item()
                    step_cos_sum[step] += F.cosine_similarity(h_now, h_prev, dim=-1).mean().item()
                changed = ((preds != prev_preds) & answer_mask).any(dim=1).float().mean().item()
                step_change_sum[step] += changed

            prev_hidden = output.detach()
            prev_preds = preds.detach()

            # Argmax halting step per sample.
            replace = p_t > best_p
            best_p = torch.where(replace, p_t, best_p)
            argmax_step = torch.where(
                replace,
                torch.full_like(argmax_step, step + 1),
                argmax_step,
            )
            if chosen_logits is None:
                chosen_logits = logits.detach().clone()
            else:
                chosen_logits[replace] = logits.detach()[replace]

        # Safety check for type checker; chosen_logits is always set because max_steps >= 1.
        assert chosen_logits is not None

        auto_acc_full_all.append(exact_match_from_logits(chosen_logits, ys, mask))
        auto_acc_answer_all.append(exact_match_from_logits(chosen_logits, ys, answer_mask))
        expected_steps_all.append(expected_steps.mean().item())
        argmax_steps_all.append(argmax_step.mean().item())

    denom = float(n_batches)
    step_rows = []
    for step in range(max_steps):
        row = {
            "length": length,
            "split": split,
            "step": step + 1,
            "mean_exit_prob": step_exit_prob_sum[step] / denom,
            "step_loss_full": step_loss_full_sum[step] / denom,
            "step_loss_answer": step_loss_answer_sum[step] / denom,
            "forced_acc_full": step_acc_full_sum[step] / denom,
            "forced_acc_answer": step_acc_answer_sum[step] / denom,
            "delta_l2_answer": np.nan if step == 0 else step_delta_sum[step] / denom,
            "cosine_to_prev_answer": np.nan if step == 0 else step_cos_sum[step] / denom,
            "answer_change_rate": np.nan if step == 0 else step_change_sum[step] / denom,
        }
        step_rows.append(row)

    forced_accs = np.array([r["forced_acc_answer"] for r in step_rows], dtype=np.float64)
    best_step = int(forced_accs.argmax() + 1)
    best_acc = float(forced_accs.max())

    summary_row = {
        "length": length,
        "split": split,
        "auto_acc_full": float(np.mean(auto_acc_full_all)),
        "auto_acc_answer": float(np.mean(auto_acc_answer_all)),
        "expected_exit_step": float(np.mean(expected_steps_all)),
        "argmax_exit_step_mean": float(np.mean(argmax_steps_all)),
        "best_forced_acc_answer": best_acc,
        "best_forced_step": best_step,
        "final_step_acc_answer": float(forced_accs[-1]),
    }

    model.train()
    return summary_row, step_rows


def run_final_diagnostics(model: PonderLoopedModel, cfg, device: torch.device, out_dir: str):
    import pandas as pd

    diagnostics_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    all_summary = []
    all_steps = []

    for length in ID_LENGTHS:
        print(f"[diagnostics] ID length={length}")
        summary, steps = diagnose_one_length(
            model=model,
            cfg=cfg,
            device=device,
            length=length,
            max_steps=DIAG_MAX_STEPS,
            batch_size=DIAG_BATCH_SIZE,
            n_batches=DIAG_N_BATCHES,
            split="ID",
        )
        all_summary.append(summary)
        all_steps.extend(steps)

    for length in OOD_LENGTHS:
        print(f"[diagnostics] OOD length={length}")
        summary, steps = diagnose_one_length(
            model=model,
            cfg=cfg,
            device=device,
            length=length,
            max_steps=DIAG_MAX_STEPS,
            batch_size=DIAG_BATCH_SIZE,
            n_batches=DIAG_N_BATCHES,
            split="OOD",
        )
        all_summary.append(summary)
        all_steps.extend(steps)

    summary_df = pd.DataFrame(all_summary)
    steps_df = pd.DataFrame(all_steps)

    summary_path = os.path.join(diagnostics_dir, "diagnostics_summary.csv")
    steps_path = os.path.join(diagnostics_dir, "diagnostics_step_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    steps_df.to_csv(steps_path, index=False)

    if cfg.wandb.enabled:
        wandb.log({"diagnostics/summary_table": wandb.Table(dataframe=summary_df)})

        # Log quick scalar summaries for W&B dashboards.
        for _, row in summary_df.iterrows():
            prefix = f"final/{row['split']}/L{int(row['length'])}"
            wandb.log(
                {
                    f"{prefix}/auto_acc_answer": row["auto_acc_answer"],
                    f"{prefix}/auto_acc_full": row["auto_acc_full"],
                    f"{prefix}/expected_exit_step": row["expected_exit_step"],
                    f"{prefix}/argmax_exit_step_mean": row["argmax_exit_step_mean"],
                    f"{prefix}/best_forced_acc_answer": row["best_forced_acc_answer"],
                    f"{prefix}/best_forced_step": row["best_forced_step"],
                    f"{prefix}/final_step_acc_answer": row["final_step_acc_answer"],
                }
            )

        if UPLOAD_DIAGNOSTIC_ARTIFACT:
            artifact = wandb.Artifact(
                name=f"{cfg.task}_ape_once_ponder_diagnostics",
                type="diagnostics",
            )
            artifact.add_file(summary_path)
            artifact.add_file(steps_path)
            wandb.log_artifact(artifact)

    print(f"Diagnostics summary saved to: {summary_path}")
    print(f"Diagnostics step metrics saved to: {steps_path}")
    return summary_path, steps_path


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
        "run_id": run_id,
        "out_dir": out_dir,
        "task": cfg.task,
        "seed": cfg.seed,
        "train_max_len": cfg.train_max_len,
        "id_lengths": ID_LENGTHS,
        "ood_lengths": OOD_LENGTHS,
        "diag_max_steps": DIAG_MAX_STEPS,
        "diag_batch_size": DIAG_BATCH_SIZE,
        "diag_n_batches": DIAG_N_BATCHES,
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
        "wandb": vars(cfg.wandb),
        "curriculum": {
            "start": cfg.curriculum.points.start,
            "end": cfg.curriculum.points.end,
            "inc": cfg.curriculum.points.inc,
            "interval": cfg.curriculum.points.interval,
            "actual_train_lengths": [1, TRAIN_MAX_LEN],
            "note": "np.random.randint min inclusive, max exclusive; CURR_END=TRAIN_MAX_LEN+1 means final training lengths are [1, TRAIN_MAX_LEN].",
        },
    }
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            name=f"{cfg.task}_ape_once_ponder_seed{cfg.seed}_{run_id[:8]}",
            dir=out_dir,
            config=config_payload,
            tags=[
                cfg.task,
                "ape_once",
                "pondernet",
                "train_len_1_20",
                "diagnostics",
                f"seed{cfg.seed}",
            ],
        )

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

        if cfg.wandb.enabled and step % cfg.wandb.log_every == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/rec_loss": rec_loss.item(),
                    "train/kl_loss": kl_loss.item(),
                    "train/grad_norm": float(grad_norm),
                    "train/n_points": n_points,
                    "train/actual_max_train_len_seen_so_far": n_points - 1,
                    "train/horizon": horizon,
                    "train/beta": cfg.ponder.beta,
                    "train/prior_lambda": cfg.ponder.prior_lambda,
                },
                step=step,
            )

        if step % cfg.eval_every == 0:
            eval_acc, eval_steps = evaluate(model, cfg, device, cfg.test_len, cfg.eval_batch_size, n_batches=4)
            if eval_acc >= best_acc:
                best_acc = eval_acc
                torch.save(model, os.path.join(out_dir, "best.pt"))
            print(
                f"[step {step}] loss={loss.item():.4f} rec={rec_loss.item():.4f} kl={kl_loss.item():.4f} "
                f"grad_norm={float(grad_norm):.4f} eval_acc={eval_acc:.4f} eval_E[steps]={eval_steps:.2f}"
            )
            if cfg.wandb.enabled:
                wandb.log(
                    {
                        "eval/test_len": cfg.test_len,
                        "eval/acc": eval_acc,
                        "eval/expected_steps": eval_steps,
                        "eval/best_acc": best_acc,
                    },
                    step=step,
                )
        if step % cfg.eval_every == 0:
            pbar.set_description(
                f"loss={loss.item():.4f} rec={rec_loss.item():.4f} kl={kl_loss.item():.4f} n_points={n_points}"
            )
        curriculum.update()

    torch.save(model, os.path.join(out_dir, "model.pt"))

    if RUN_FINAL_DIAGNOSTICS:
        best_path = os.path.join(out_dir, "best.pt")
        if os.path.exists(best_path):
            print(f"Loading best checkpoint for final diagnostics: {best_path}")
            eval_model = torch.load(best_path, map_location=device)
            eval_model.to(device)
        else:
            print("No best.pt found; using final model for final diagnostics.")
            eval_model = model
        run_final_diagnostics(eval_model, cfg, device, out_dir)

    if cfg.wandb.enabled and UPLOAD_MODEL_ARTIFACT:
        artifact = wandb.Artifact(
            name=f"{cfg.task}_ape_once_ponder_{run_id[:8]}",
            type="model",
        )
        artifact.add_file(os.path.join(out_dir, "config.yaml"))
        artifact.add_file(os.path.join(out_dir, "model.pt"))
        best_path = os.path.join(out_dir, "best.pt")
        if os.path.exists(best_path):
            artifact.add_file(best_path)
        wandb.log_artifact(artifact)

    print(f"Training done. Outputs saved to: {out_dir}")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    cfg = build_train_config()
    train(cfg)
