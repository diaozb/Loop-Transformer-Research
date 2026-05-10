from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ltf.eval import run_dense_eval, write_default_eval_outputs
from ltf.training import evaluate_once, load_checkpoint_for_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a migrated experiment checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to exp_infra checkpoint.")
    parser.add_argument("--output-dir", default=None, help="Directory for eval outputs.")
    parser.add_argument("--dense", action="store_true", help="Run dense length x loop eval.")
    parser.add_argument("--lengths", default=None, help="Lengths: comma list or start-end[-step], e.g. 2,4,8 or 1-40-2.")
    parser.add_argument("--loop-counts", default=None, help="Loop counts: comma list or start-end[-step].")
    parser.add_argument("--num-samples", type=int, default=512, help="Dense eval samples per length.")
    parser.add_argument("--batch-size", type=int, default=128, help="Dense eval batch size.")
    parser.add_argument("--auto-exit-max-loops", type=int, default=None, help="Ponder auto-exit horizon for dense eval.")
    parser.add_argument("overrides", nargs="*", help="Optional dotlist overrides: a.b=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_checkpoint_for_eval(args.checkpoint, map_location="cpu")
    config = loaded.config

    if args.overrides:
        # Reuse the YAML loader override parser by round-tripping through an in-memory temp
        # is overkill here; eval overrides are intentionally limited for now.
        raise SystemExit("Eval overrides for checkpoint configs are not implemented yet.")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    loaded.model.to(device)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent.parent / "eval"

    if args.dense:
        lengths = _parse_int_spec(args.lengths) if args.lengths else config.eval.lengths
        loop_counts = _parse_int_spec(args.loop_counts) if args.loop_counts else config.eval.loop_counts
        artifacts = run_dense_eval(
            loaded.model,
            config,
            output_dir=output_dir,
            lengths=lengths,
            loop_counts=loop_counts,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            auto_exit_max_loops=args.auto_exit_max_loops,
            device=device,
        )
        print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2, sort_keys=True))
        return

    metrics = evaluate_once(loaded.model, config, device)
    row = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": loaded.step,
        "task": config.task.name,
        "trainer": config.trainer.name,
        "test_length": config.task.test_length,
        **metrics,
    }

    write_default_eval_outputs(output_dir, row)
    print(json.dumps(row, indent=2, sort_keys=True))


def _parse_int_spec(raw: str):
    text = raw.strip()
    if "," in text:
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    parts = [part.strip() for part in text.split("-") if part.strip()]
    if len(parts) == 1:
        return [int(parts[0])]
    if len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
        step = 1
    elif len(parts) == 3:
        start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        raise ValueError(f"Invalid integer spec: {raw}")
    return list(range(start, end + 1, step))


if __name__ == "__main__":
    main()
