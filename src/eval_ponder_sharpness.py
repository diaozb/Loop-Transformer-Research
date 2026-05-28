#!/usr/bin/env python
"""
Parameter-space sharpness evaluation for Copy + PonderNet checkpoints.

This script measures the rise in the complete Ponder training objective after
small random parameter perturbations:

    objective = expected reconstruction loss + beta * KL(halting || prior)

It reuses the same cached evaluation batches for the unperturbed model and all
perturbed models, so changes reflect parameter perturbation rather than new
sample noise.

Expected project placement:
    src/eval_ponder_sharpness.py

Required sibling file:
    src/train_ponder_nope.py

Examples:
    python eval_ponder_sharpness.py \
      --run-dir ../models/nope_baselines/copy_ponder/<RUN_ID> \
      --checkpoint best.pt \
      --lengths 1-20,21,22,40,60 \
      --out-dir ../eval/nope_copy_ponder/<RUN_ID>/sharpness

    python eval_ponder_sharpness.py \
      --run-dir ../models/rope_baselines/copy_ponder/<RUN_ID> \
      --checkpoint best.pt \
      --lengths 1-20,21,22,40,60 \
      --out-dir ../eval/rope_copy_ponder/<RUN_ID>/sharpness
"""

import argparse
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple
from typing import Optional


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

import train_ponder_nope as tp


def parse_lengths(spec: str) -> List[int]:
    lengths: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            lengths.extend(range(int(left), int(right) + 1))
        else:
            lengths.append(int(chunk))
    return sorted(set(lengths))


def parse_floats(spec: str) -> List[float]:
    values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one epsilon is required.")
    return sorted(set(values))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(run_dir: Path) -> SimpleNamespace:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find config.yaml: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    cfg = SimpleNamespace()
    cfg.task = payload["task"]
    cfg.seed = payload.get("seed", 0)
    cfg.modulus = payload.get("modulus", 11)
    cfg.model = SimpleNamespace(**payload["model"])

    ponder = payload["ponder"]
    cfg.ponder = tp.PonderConfig(
        beta=ponder.get("beta", 0.01),
        prior_lambda=ponder.get("prior_lambda", 0.2),
        max_steps_cap=ponder.get("max_steps_cap", 128),
        dynamic_n=ponder.get("dynamic_n", False),
        n_steps=ponder.get("n_steps", 20),
    )
    cfg.payload = payload
    return cfg


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

    # Models saved from a training script executed as __main__ need this shim.
    import __main__
    __main__.PonderLoopedModel = tp.PonderLoopedModel

    try:
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(checkpoint_path, map_location=device)

    model.to(device)
    model.eval()
    return model


def make_batches(
    cfg: SimpleNamespace,
    length: int,
    batch_size: int,
    n_batches: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    cached = []
    max_len = tp.compute_max_len(cfg.task, length)
    for _ in range(n_batches):
        xs, ys, code_mask = tp.sample_batch(
            cfg.task,
            batch_size,
            min_num_digits=length,
            max_num_digits=length + 1,
            max_len=max_len,
            modulus=cfg.modulus,
        )
        xs = tp.prepare_inputs(xs, cfg.model, device)
        ys = ys.to(device=device, dtype=torch.long)
        code_mask = code_mask.to(device=device, dtype=torch.bool)
        answer_mask = tp.get_metric_mask(cfg.task, code_mask, length)
        cached.append((xs, ys, code_mask, answer_mask))
    return cached


@torch.no_grad()
def objective_on_batches(
    model,
    cfg: SimpleNamespace,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    length: int,
    max_steps_override: Optional[int],
    device: torch.device,
) -> Dict[str, float]:
    if max_steps_override is None:
        horizon = tp.resolve_horizon(cfg.task, length, cfg.ponder)
    else:
        horizon = max_steps_override

    prior = tp.truncated_geometric_prior(
        horizon, cfg.ponder.prior_lambda, device
    ).unsqueeze(1)
    prior_eps = 1e-12

    objective_values = []
    rec_values = []
    kl_values = []
    auto_code_accs = []
    auto_answer_accs = []
    expected_steps = []

    for xs, ys, code_mask, answer_mask in batches:
        logits_steps, p_steps = model.forward_ponder(xs, horizon, halt_mask=code_mask)

        per_step_losses = torch.stack(
            [tp.per_sample_ce(logits_steps[t], ys, code_mask) for t in range(horizon)],
            dim=0,
        )
        rec_loss = (p_steps * per_step_losses).sum(dim=0).mean()
        kl_loss = (
            p_steps * ((p_steps + prior_eps).log() - (prior + prior_eps).log())
        ).sum(dim=0).mean()
        objective = rec_loss + cfg.ponder.beta * kl_loss

        argmax_steps = p_steps.argmax(dim=0)
        b_idx = torch.arange(ys.shape[0], device=device)
        chosen_logits = logits_steps[argmax_steps, b_idx]

        auto_code_acc = tp.exact_match_from_logits(chosen_logits, ys, code_mask)
        auto_answer_acc = tp.exact_match_from_logits(chosen_logits, ys, answer_mask)
        step_ids = torch.arange(1, horizon + 1, device=device).float().unsqueeze(1)
        expected_exit = (p_steps * step_ids).sum(dim=0).mean()

        objective_values.append(objective.item())
        rec_values.append(rec_loss.item())
        kl_values.append(kl_loss.item())
        auto_code_accs.append(auto_code_acc)
        auto_answer_accs.append(auto_answer_acc)
        expected_steps.append(expected_exit.item())

    return {
        "length": length,
        "horizon": horizon,
        "objective": float(np.mean(objective_values)),
        "rec_loss": float(np.mean(rec_values)),
        "kl_loss": float(np.mean(kl_values)),
        "auto_code_acc": float(np.mean(auto_code_accs)),
        "auto_answer_acc": float(np.mean(auto_answer_accs)),
        "expected_exit_step": float(np.mean(expected_steps)),
    }


def add_relative_gaussian_perturbation(
    model,
    epsilon: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """
    Tensor-wise relative Gaussian perturbation:
        delta_i = epsilon * ||theta_i||_2 * z_i / ||z_i||_2

    All trainable parameters are included, including the halting head.
    """
    if epsilon == 0.0:
        return {}

    devices = {p.device for p in model.parameters() if p.requires_grad}
    if len(devices) != 1:
        raise RuntimeError("Expected all parameters on the same device.")

    generator = torch.Generator(device=next(iter(devices)))
    generator.manual_seed(seed)
    deltas: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            noise = torch.randn(
                param.shape, device=param.device, dtype=param.dtype, generator=generator
            )
            noise_norm = noise.norm().clamp_min(1e-12)
            param_norm = param.norm()
            # Zero-valued tensors get no relative perturbation.
            delta = epsilon * param_norm * noise / noise_norm
            param.add_(delta)
            deltas[name] = delta
    return deltas


def remove_perturbation(model, deltas: Dict[str, torch.Tensor]) -> None:
    if not deltas:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in deltas:
                param.sub_(deltas[name])


def plot_sharpness(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for length in sorted(summary["length"].unique()):
        sub = summary[summary["length"] == length].sort_values("epsilon")
        plt.plot(sub["epsilon"], sub["delta_objective_mean"], marker="o", label=f"L={length}")
    plt.xscale("symlog", linthresh=1e-6)
    plt.xlabel("Relative perturbation radius (epsilon)")
    plt.ylabel("Mean objective increase")
    plt.title("PonderNet sharpness under parameter perturbations")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PonderNet parameter sharpness.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--lengths", default="1-20,21,22,40,60")
    parser.add_argument("--epsilons", default="0,1e-4,3e-4,1e-3,3e-3,1e-2")
    parser.add_argument("--directions", default=5, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--n-batches", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--max-steps",
        default=None,
        type=int,
        help="Override Ponder horizon. Default matches the saved training config.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.directions < 1:
        raise ValueError("--directions must be at least 1.")
    if args.batch_size < 1 or args.n_batches < 1:
        raise ValueError("--batch-size and --n-batches must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    cfg = load_cfg(args.run_dir)
    if cfg.task != "copy":
        raise ValueError(f"This first sharpness experiment is intended for copy; got {cfg.task!r}.")

    model = load_model(args.run_dir / args.checkpoint, device)
    lengths = parse_lengths(args.lengths)
    epsilons = parse_floats(args.epsilons)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cached_by_length = {
        length: make_batches(cfg, length, args.batch_size, args.n_batches, device)
        for length in lengths
    }

    baseline_by_length = {}
    for length in lengths:
        baseline_by_length[length] = objective_on_batches(
            model, cfg, cached_by_length[length], length, args.max_steps, device
        )

    raw_rows = []
    for length in lengths:
        base = baseline_by_length[length]
        raw_rows.append(
            {
                **base,
                "epsilon": 0.0,
                "direction": 0,
                "delta_objective": 0.0,
                "delta_rec_loss": 0.0,
                "delta_kl_loss": 0.0,
                "delta_auto_answer_acc": 0.0,
                "delta_expected_exit_step": 0.0,
            }
        )

    for epsilon in [x for x in epsilons if x > 0.0]:
        print(f"Evaluating epsilon={epsilon:g}")
        for direction in range(args.directions):
            perturb_seed = args.seed + 100000 * direction + int(round(epsilon * 1e9))
            deltas = add_relative_gaussian_perturbation(model, epsilon, perturb_seed)
            try:
                for length in lengths:
                    metric = objective_on_batches(
                        model, cfg, cached_by_length[length], length, args.max_steps, device
                    )
                    base = baseline_by_length[length]
                    raw_rows.append(
                        {
                            **metric,
                            "epsilon": epsilon,
                            "direction": direction + 1,
                            "delta_objective": metric["objective"] - base["objective"],
                            "delta_rec_loss": metric["rec_loss"] - base["rec_loss"],
                            "delta_kl_loss": metric["kl_loss"] - base["kl_loss"],
                            "delta_auto_answer_acc": (
                                metric["auto_answer_acc"] - base["auto_answer_acc"]
                            ),
                            "delta_expected_exit_step": (
                                metric["expected_exit_step"] - base["expected_exit_step"]
                            ),
                        }
                    )
            finally:
                remove_perturbation(model, deltas)

    raw_df = pd.DataFrame(raw_rows).sort_values(["length", "epsilon", "direction"])
    raw_path = args.out_dir / "sharpness_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    summary = (
        raw_df.groupby(["length", "epsilon"], as_index=False)
        .agg(
            objective_mean=("objective", "mean"),
            objective_std=("objective", "std"),
            delta_objective_mean=("delta_objective", "mean"),
            delta_objective_std=("delta_objective", "std"),
            rec_loss_mean=("rec_loss", "mean"),
            kl_loss_mean=("kl_loss", "mean"),
            auto_answer_acc_mean=("auto_answer_acc", "mean"),
            delta_auto_answer_acc_mean=("delta_auto_answer_acc", "mean"),
            expected_exit_step_mean=("expected_exit_step", "mean"),
            delta_expected_exit_step_mean=("delta_expected_exit_step", "mean"),
        )
        .fillna(0.0)
    )
    summary_path = args.out_dir / "sharpness_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_path = args.out_dir / "sharpness_delta_objective.png"
    plot_sharpness(summary, plot_path)

    metadata = {
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "lengths": lengths,
        "epsilons": epsilons,
        "directions": args.directions,
        "batch_size": args.batch_size,
        "n_batches": args.n_batches,
        "seed": args.seed,
        "device": str(device),
        "perturbation": "tensor-wise relative Gaussian, all trainable parameters including halt_head",
        "objective": "expected reconstruction loss on training mask + beta * KL(halting || truncated geometric prior)",
        "beta": cfg.ponder.beta,
        "prior_lambda": cfg.ponder.prior_lambda,
        "max_steps_override": args.max_steps,
    }
    with open(args.out_dir / "sharpness_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Raw results: {raw_path}")
    print(f"Summary:     {summary_path}")
    print(f"Plot:        {plot_path}")


if __name__ == "__main__":
    main()
