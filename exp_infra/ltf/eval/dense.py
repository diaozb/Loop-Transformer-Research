from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ltf.config import RunConfig
from ltf.data import generate_batch, legacy_max_len, prepare_inputs, task_pad_token
from ltf.eval.metrics import answer_mask, sequence_accuracy, token_accuracy
from ltf.eval.ponder_eval import compute_auto_exit_stats
from ltf.eval.results_io import dense_results_to_rows, write_rows_csv
from ltf.viz import save_heatmap


RETAINED_DENSE_METRICS = [
    "accuracy",
    "answer_accuracy",
    "token_accuracy",
    "delta_l2_norm_mask",
    "cosine_to_prev_mask",
    "answer_change_rate",
]


@torch.no_grad()
def run_dense_eval(
    model,
    config: RunConfig,
    output_dir: str | Path,
    lengths: Iterable[int],
    loop_counts: Iterable[int] | None = None,
    num_samples: int = 512,
    batch_size: int = 128,
    auto_exit_max_loops: int | None = None,
    device: torch.device | str | None = None,
) -> Dict[str, Path]:
    device = torch.device(device or (config.device if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    lengths = list(lengths)
    if loop_counts is None:
        loop_counts = list(range(1, max(lengths) + 3))
    else:
        loop_counts = sorted(set(loop_counts))
    dense_horizon = max(loop_counts)
    auto_horizon = auto_exit_max_loops or dense_horizon
    collect_horizon = max(dense_horizon, auto_horizon)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dense_results: Dict[int, Dict[int, Dict[str, float]]] = {}
    auto_exit_rows = []

    for length in lengths:
        accum = _empty_accumulator(loop_counts, config.task.name)
        auto_accum = []
        seen = 0
        while seen < num_samples:
            current_b = min(batch_size, num_samples - seen)
            batch = generate_batch(
                config.task,
                batch_size=current_b,
                min_length=length,
                max_length_exclusive=length + 1,
                max_len=legacy_max_len(config.task.name, length),
            )
            xs = prepare_inputs(
                batch.inputs,
                linear_embedding=config.model.linear_embedding,
                n_dims=config.model.n_dims,
                device=device,
            )
            targets = batch.targets.to(device=device, dtype=torch.long)
            mask = batch.mask.to(device=device)

            if hasattr(model, "base"):
                hidden_list, logits_list, lambda_steps = _collect_ponder(model, xs, mask.bool(), collect_horizon)
                if auto_exit_max_loops is not None:
                    auto_stats = compute_auto_exit_stats(
                        torch.stack(logits_list[:auto_horizon], dim=0),
                        lambda_steps[:auto_horizon],
                        targets,
                        mask,
                        config.task.name,
                    )
                    auto_accum.append(auto_stats)
            else:
                hidden_list, logits_list = _collect_fixed(model, xs, collect_horizon)

            batch_metrics = _batch_dense_metrics(
                hidden_list,
                logits_list,
                targets,
                mask,
                loop_counts,
                config,
            )
            _accumulate(accum, batch_metrics, weight=current_b)
            seen += current_b

        dense_results[length] = _finalize_accumulator(accum, total=num_samples)
        if auto_accum:
            row = {"length": length}
            for key in auto_accum[0].keys():
                row[key] = float(np.mean([item[key] for item in auto_accum]))
            auto_exit_rows.append(row)

    dense_csv = output / "dense_eval.csv"
    dense_json = output / "dense_eval.json"
    write_rows_csv(dense_csv, dense_results_to_rows(dense_results))
    dense_json.write_text(json.dumps(dense_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = {"dense_csv": dense_csv, "dense_json": dense_json}
    if auto_exit_rows:
        auto_csv = output / "auto_exit_by_length.csv"
        write_rows_csv(auto_csv, auto_exit_rows)
        (output / "auto_exit_by_length.json").write_text(
            json.dumps(auto_exit_rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts["auto_exit_csv"] = auto_csv

    _write_selected_heatmaps(output, dense_results, lengths, loop_counts, config.task.name)
    return artifacts


def _collect_fixed(model, xs: torch.Tensor, horizon: int):
    trace = model.collect_looped_forward(xs, horizon=horizon)
    return trace.hidden, trace.logits


def _collect_ponder(model, xs: torch.Tensor, mask: torch.Tensor, horizon: int):
    output = model.forward_ponder(xs, max_steps=horizon, halt_mask=mask)
    return output.hidden_steps, list(output.logits_steps), output.lambda_steps


def _batch_dense_metrics(hidden_list, logits_list, targets, mask, loop_counts, config):
    task_name = config.task.name
    results = {}
    mask_bool = mask == 1
    preds_by_loop = {loop: logits_list[loop - 1].argmax(dim=-1) for loop in loop_counts}
    for idx, loop in enumerate(loop_counts):
        hidden = hidden_list[loop - 1]
        preds = preds_by_loop[loop]
        row = {
            "accuracy": float(sequence_accuracy(preds, targets, mask).item()),
        }
        ans_mask = answer_mask(mask, targets, pad_token=task_pad_token(config.task.name, config.task.modulus))
        if ans_mask.any():
            row["answer_accuracy"] = float(sequence_accuracy(preds, targets, ans_mask).item())
        if task_name == "copy":
            row["token_accuracy"] = float(token_accuracy(preds, targets, mask).item())

        if idx == 0:
            row["delta_l2_norm_mask"] = float("nan")
            row["cosine_to_prev_mask"] = float("nan")
            row["answer_change_rate"] = float("nan")
        else:
            prev_hidden = hidden_list[loop_counts[idx - 1] - 1]
            delta = hidden[mask_bool] - prev_hidden[mask_bool]
            row["delta_l2_norm_mask"] = float(torch.linalg.norm(delta, dim=-1).mean().item())
            row["cosine_to_prev_mask"] = float(F.cosine_similarity(hidden[mask_bool], prev_hidden[mask_bool], dim=-1).mean().item())
            prev_preds = preds_by_loop[loop_counts[idx - 1]]
            changed = []
            for sample_idx in range(targets.shape[0]):
                idx_mask = mask_bool[sample_idx]
                changed.append((preds[sample_idx, idx_mask] != prev_preds[sample_idx, idx_mask]).any().float())
            row["answer_change_rate"] = float(torch.stack(changed).mean().item())
        results[loop] = row
    return results


def _empty_accumulator(loop_counts, task_name):
    keys = ["accuracy", "answer_accuracy", "delta_l2_norm_mask", "cosine_to_prev_mask", "answer_change_rate"]
    if task_name == "copy":
        keys.append("token_accuracy")
    return {loop: {key: 0.0 for key in keys} for loop in loop_counts}


def _accumulate(accum, batch_metrics, weight: int):
    for loop, metrics in batch_metrics.items():
        for key, value in metrics.items():
            if np.isnan(value):
                accum[loop][key] = float("nan")
            elif not np.isnan(accum[loop][key]):
                accum[loop][key] += value * weight


def _finalize_accumulator(accum, total: int):
    out = {}
    for loop, metrics in accum.items():
        out[loop] = {}
        for key, value in metrics.items():
            out[loop][key] = float("nan") if np.isnan(value) else value / float(total)
    return out


def _write_selected_heatmaps(output, dense_results, lengths, loop_counts, task_name):
    metrics = ["accuracy", "answer_accuracy", "delta_l2_norm_mask", "cosine_to_prev_mask", "answer_change_rate"]
    if task_name == "copy":
        metrics.insert(2, "token_accuracy")
    for metric in metrics:
        matrix = np.full((len(lengths), len(loop_counts)), np.nan, dtype=np.float32)
        for i, length in enumerate(lengths):
            for j, loop in enumerate(loop_counts):
                matrix[i, j] = dense_results[length][loop].get(metric, np.nan)
        save_heatmap(
            matrix,
            x_labels=loop_counts,
            y_labels=lengths,
            output_path=output / f"{metric}_heatmap.png",
            title=f"{metric} by length and loop",
            colorbar_label=metric,
        )
