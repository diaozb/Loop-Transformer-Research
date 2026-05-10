#!/usr/bin/env python3
import os
import sys
import yaml

import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from generate_training_data import generate_prompt_matrix_parity
from utils import convert_to_one_hot


# ---- Editable experiment parameters ----
CKPT_PATH = "/data/yizhou/looped-tf-length-generalization/models/parity/d9de8dd7-d283-4236-aa71-d02ce63ab40a/model.pt"
LENGTH_L = 30
NUM_SAMPLES = 2000
BATCH_SIZE = 1024
DEVICE = "cuda"
SEED = 1234
RANDOM_INJECTION = True
NOISE_SIGMA = 0.1
# ---------------------------------------


def _resolve_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "model.pt")
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(path, "best.pt")
        if os.path.exists(candidate):
            return candidate
    return path


def _load_config_from_ckpt(path: str):
    if os.path.isdir(path):
        cfg_path = os.path.join(path, "config.yaml")
    else:
        cfg_path = os.path.join(os.path.dirname(path), "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def _prepare_inputs(xs_int: torch.Tensor, model, device: torch.device) -> torch.Tensor:
    linear_embedding = hasattr(model, "_read_in") and isinstance(model._read_in, torch.nn.Linear)
    if linear_embedding:
        one_hot = convert_to_one_hot(xs_int.numpy())
        return torch.tensor(one_hot, dtype=torch.float32, device=device)
    return xs_int.to(device=device, dtype=torch.long)


def _make_b_from_a_parity(xs_a: torch.Tensor, ys_a: torch.Tensor, batch_num: torch.Tensor):
    xs_b = xs_a.clone()
    ys_b = ys_a.clone()
    b = xs_a.shape[0]
    lengths = batch_num.squeeze(-1).tolist()
    for i in range(b):
        L = int(lengths[i])
        flip_idx = torch.randint(low=0, high=L, size=(1,)).item()
        xs_b[i, flip_idx] = 1 - xs_b[i, flip_idx]
        ys_b[i, :L] = 5
        parity = int(xs_b[i, :L].sum().item()) % 2
        ys_b[i, L] = parity
    return xs_b, ys_b


def _answer_mask(batch_num: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    b = batch_num.shape[0]
    idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, -1)
    pos = batch_num.to(device=device)
    return idx == pos


def _cross_entropy_on_mask(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    logits_flat = logits[mask]
    targets_flat = targets[mask]
    return F.cross_entropy(logits_flat, targets_flat, reduction="mean").item()


def _token_accuracy_on_mask(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    preds_flat = preds[mask]
    targets_flat = targets[mask]
    return (preds_flat == targets_flat).float().mean().item()


def _run_switch_sweep(model, device: torch.device, use_wpe: bool):
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    num_done = 0
    switch_steps = list(range(2, LENGTH_L + 1))
    sums_final = {k: {"A": 0.0, "B": 0.0} for k in switch_steps}
    acc_final = {k: {"A": 0.0, "B": 0.0} for k in switch_steps}
    count = 0

    while num_done < NUM_SAMPLES:
        b = min(BATCH_SIZE, NUM_SAMPLES - num_done)
        xs_A, batch_num_A, ys_A, _ = generate_prompt_matrix_parity(
            b, max_len=LENGTH_L + 1, min_num_digits=LENGTH_L, max_num_digits=LENGTH_L + 1
        )
        xs_B, ys_B = _make_b_from_a_parity(xs_A, ys_A, batch_num_A)

        xs_A = _prepare_inputs(xs_A, model, device)
        xs_B = _prepare_inputs(xs_B, model, device)
        ys_A = ys_A.to(device=device, dtype=torch.long)
        ys_B = ys_B.to(device=device, dtype=torch.long)
        batch_num_A = batch_num_A.to(device=device, dtype=torch.long)

        seq_len = ys_A.shape[1]
        mask_A = _answer_mask(batch_num_A, seq_len, device)
        mask_B = _answer_mask(batch_num_A, seq_len, device)

        with torch.no_grad():
            zs_A = model._read_in(xs_A)
            zs_B = model._read_in(xs_B)
            for switch_step in switch_steps:
                output = torch.zeros_like(zs_A)
                total_steps = LENGTH_L
                for step in range(1, total_steps + 1):
                    if step < switch_step:
                        inject = zs_A
                    else:
                        if RANDOM_INJECTION:
                            inject = torch.randn_like(zs_A) * NOISE_SIGMA
                        else:
                            inject = zs_B
                    output = model.forward_single(output + inject, add_wpe=use_wpe, step_idx=step - 1)
                logits_final = model._read_out(output)
                sums_final[switch_step]["A"] += _cross_entropy_on_mask(logits_final, ys_A, mask_A) * b
                sums_final[switch_step]["B"] += _cross_entropy_on_mask(logits_final, ys_B, mask_B) * b
                acc_final[switch_step]["A"] += _token_accuracy_on_mask(logits_final, ys_A, mask_A) * b
                acc_final[switch_step]["B"] += _token_accuracy_on_mask(logits_final, ys_B, mask_B) * b

        count += b
        num_done += b

    ce_avg = {k: {t: v / max(1, count) for t, v in sums_final[k].items()} for k in sums_final}
    acc_avg = {k: {t: v / max(1, count) for t, v in acc_final[k].items()} for k in acc_final}
    return ce_avg, acc_avg


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    config = _load_config_from_ckpt(CKPT_PATH)
    use_wpe = bool(config.get("model", {}).get("use_wpe", False))
    use_rope = bool(config.get("model", {}).get("use_rope", False))

    ckpt_path = _resolve_checkpoint(CKPT_PATH)
    model = torch.load(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    ce_avg, acc_avg = _run_switch_sweep(model, device, use_wpe=use_wpe)

    print(f"Checkpoint: {ckpt_path}")
    print(f"LENGTH_L: {LENGTH_L}, NUM_SAMPLES: {NUM_SAMPLES}, BATCH_SIZE: {BATCH_SIZE}")
    print(f"Config: use_wpe={use_wpe}, use_rope={use_rope}")

    print("Final-step comparison for each switch step k (total loops fixed at L):")
    for k in range(2, LENGTH_L + 1):
        ce_a = ce_avg[k]["A"]
        ce_b = ce_avg[k]["B"]
        acc_a = acc_avg[k]["A"]
        acc_b = acc_avg[k]["B"]
        closer_ce = "A" if ce_a < ce_b else "B"
        closer_acc = "A" if acc_a > acc_b else "B"
        print(
            f"  k={k}: CE(A)={ce_a:.6f}, CE(B)={ce_b:.6f} -> closer {closer_ce}; "
            f"ACC(A)={acc_a:.6f}, ACC(B)={acc_b:.6f} -> closer {closer_acc}"
        )


if __name__ == "__main__":
    main()
