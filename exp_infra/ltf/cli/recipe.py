from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ltf.config import load_recipe_file
from ltf.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more all-in-one experiment recipes sequentially.")
    parser.add_argument("--recipe", required=True, help="Path to all-in-one recipe YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved recipe names without running.")
    parser.add_argument("--summary-dir", default=None, help="Optional directory for queue summary CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes = load_recipe_file(args.recipe)
    if args.dry_run:
        print(f"Recipe queue: {len(recipes)} runs from {args.recipe}")
        for idx, (name, cfg) in enumerate(recipes, start=1):
            print(f"[dry-run {idx:02d}/{len(recipes):02d}] {_format_recipe_summary(name, cfg)}")
        return

    print(f"Recipe queue: {len(recipes)} runs from {args.recipe}", flush=True)
    rows = []
    for idx, (name, cfg) in enumerate(recipes, start=1):
        queue_label = f"recipe {idx:02d}/{len(recipes):02d} {name}"
        print(f"[recipe {idx:02d}/{len(recipes):02d}] queued {_format_recipe_summary(name, cfg)}", flush=True)
        run_dir = run_training(cfg, progress_label=queue_label)
        rows.append(
            {
                "index": idx,
                "total": len(recipes),
                "name": name,
                "task": cfg.task.name,
                "trainer": cfg.trainer.name,
                "pe": _position_mode(cfg),
                "seed": cfg.seed,
                "run_dir": run_dir,
            }
        )
        print(f"[recipe {idx:02d}/{len(recipes):02d}] saved to {run_dir}", flush=True)

    summary_dir = Path(args.summary_dir) if args.summary_dir else Path("exp_infra/results/recipes") / Path(args.recipe).stem
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "queue_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Recipe summary saved to: {summary_dir}")


def _format_recipe_summary(name, cfg) -> str:
    return (
        f"name={name} task={cfg.task.name} trainer={cfg.trainer.name} pe={_position_mode(cfg)} "
        f"seed={cfg.seed} arch=L{cfg.model.n_layer}/H{cfg.model.n_head}/E{cfg.model.n_embd} "
        f"steps={cfg.trainer.train_steps}"
    )


def _position_mode(cfg) -> str:
    if cfg.model.use_rope:
        return "rope"
    if cfg.model.use_wpe:
        return f"wpe_{cfg.model.wpe_mode}"
    return "nope"


if __name__ == "__main__":
    main()
