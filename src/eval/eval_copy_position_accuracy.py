#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from hashlib import sha1

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from generate_training_data import generate_prompt_matrix_copy
from utils import convert_to_one_hot


def _looped_forward_collect(model, xs: torch.Tensor, horizon: int):
    """Collect hidden states and logits per loop."""
    if hasattr(model, "_read_in") and hasattr(model, "_read_out"):
        zs = model._read_in(xs)
        output = torch.zeros_like(zs).to(zs.device)
        logits_list = []
        use_wpe = getattr(model, "use_wpe", False)
        for step in range(horizon):
            output = model.forward_single(output + zs, add_wpe=use_wpe)
            logits_list.append(model._read_out(output))
        return logits_list
    raise ValueError("Model does not expose looped forward components (_read_in/_read_out).")


def _tokens_to_str(tokens):
    return "".join(str(int(x)) for x in tokens)


def main():
    parser = argparse.ArgumentParser(description="Evaluate copy position accuracy across loops.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model.pt checkpoint saved by training.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=10,
        help="Copy length (number of digits).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--prob_one",
        type=float,
        default=0.5,
        help="Probability of generating token 1 in copy input (token 0 uses 1-prob_one).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name for output directory (subfolder under eval/copy/<checkpoint_name>/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional full output directory. If set, overrides the default eval/copy/<checkpoint_name>/<run_name>.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on.")
    args = parser.parse_args()

    length = args.length
    horizon = length

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    checkpoint_name = os.path.basename(checkpoint_dir)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        if args.run_name:
            run_name = args.run_name
        else:
            run_payload = {
                "length": length,
                "num_samples": args.num_samples,
                "prob_one": args.prob_one,
            }
            run_hash = sha1(json.dumps(run_payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{run_stamp}_{run_hash}"
        output_dir = os.path.join(repo_root, "eval", "copy", checkpoint_name, run_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    linear_embedding = hasattr(model, "_read_in") and isinstance(model._read_in, torch.nn.Linear)

    xs, _, ys, _ = generate_prompt_matrix_copy(
        args.num_samples,
        max_len=length + 1,
        min_num_digits=length,
        max_num_digits=length + 1,
        prob_one=args.prob_one,
    )
    ys = ys.to(device)
    if linear_embedding:
        xs_t = torch.tensor(convert_to_one_hot(xs.numpy()), dtype=torch.float32, device=device)
    else:
        xs_t = xs.to(device)

    with torch.no_grad():
        logits_list = _looped_forward_collect(model, xs_t, horizon)

    # Position accuracy matrix: [position, loop]
    pos_acc = np.zeros((length, horizon), dtype=np.float32)

    # Save decoded results per sample per loop.
    txt_path = os.path.join(output_dir, "copy_loop_decodes.txt")
    with open(txt_path, "w") as f:
        for i in range(args.num_samples):
            input_bits = xs[i, :length].cpu().numpy()
            target_bits = ys[i, length:2 * length].cpu().numpy()
            f.write(f"sample={i}\n")
            f.write(f"input={_tokens_to_str(input_bits)} target={_tokens_to_str(target_bits)}\n")
            for loop_idx in range(horizon):
                preds = logits_list[loop_idx].argmax(dim=-1)
                pred_bits = preds[i, length:2 * length].cpu().numpy()
                f.write(f"  loop={loop_idx + 1} pred={_tokens_to_str(pred_bits)}\n")
            f.write("\n")

    # Compute per-position accuracy across loops.
    for loop_idx in range(horizon):
        preds = logits_list[loop_idx].argmax(dim=-1)
        pred_bits = preds[:, length:2 * length]
        target_bits = ys[:, length:2 * length]
        correct = (pred_bits == target_bits).float().mean(dim=0)
        pos_acc[:, loop_idx] = correct.cpu().numpy()

    # Save CSV (position x loop)
    csv_path = os.path.join(output_dir, "copy_position_accuracy.csv")
    with open(csv_path, "w") as f:
        header = ["position"] + [f"loop_{i+1}" for i in range(horizon)]
        f.write(",".join(header) + "\n")
        for pos in range(length):
            row = [str(pos + 1)] + [f"{pos_acc[pos, loop]:.6f}" for loop in range(horizon)]
            f.write(",".join(row) + "\n")

    # Heatmap
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pos_acc, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="Accuracy")
    plt.xticks(ticks=np.arange(horizon), labels=np.arange(1, horizon + 1))
    plt.yticks(ticks=np.arange(length), labels=np.arange(1, length + 1))
    plt.xlabel("Loop")
    plt.ylabel("Output position")
    plt.title("Copy Position Accuracy (Length x Loop)")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "copy_position_accuracy_heatmap.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
