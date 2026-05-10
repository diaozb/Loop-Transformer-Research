from __future__ import annotations

import argparse
import json

from ltf.config import load_config
from ltf.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a looped Transformer experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the resolved config.")
    parser.add_argument("overrides", nargs="*", help="Optional dotlist overrides: a.b=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.overrides)
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=False))
        return
    run_dir = run_training(config)
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
