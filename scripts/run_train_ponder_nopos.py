import argparse
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import train_ponder  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a no-position Ponder training run.")
    parser.add_argument("--task", required=True, choices=sorted(train_ponder.TASK_PRESETS.keys()))
    parser.add_argument(
        "--output-root",
        default=os.path.join(REPO_ROOT, "models", "nope_baselines"),
        help="Root directory under which task-specific outputs will be created.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="Ponder beta. Defaults to the regularized baseline value 0.01.",
    )
    parser.add_argument("--prior-lambda", type=float, default=None, help="Override geometric prior lambda.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override fixed Ponder horizon.")
    parser.add_argument(
        "--dynamic-n",
        action="store_true",
        help="Use curriculum-dependent horizon instead of a fixed n_steps.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    train_ponder.TASK = args.task
    train_ponder.OUT_DIR = args.output_root
    train_ponder.MODEL_USE_WPE = False
    train_ponder.MODEL_USE_ROPE = False

    train_ponder.PONDER_BETA = args.beta
    if args.prior_lambda is not None:
        train_ponder.PONDER_PRIOR_LAMBDA = args.prior_lambda
    if args.n_steps is not None:
        train_ponder.PONDER_N_STEPS = args.n_steps
    if args.dynamic_n:
        train_ponder.PONDER_DYNAMIC_N = True
    if args.seed is not None:
        train_ponder.SEED = args.seed

    cfg = train_ponder.build_train_config()
    print(
        "Launching NoPE Ponder run:",
        {
            "task": cfg.task,
            "output_root": cfg.out_dir,
            "seed": cfg.seed,
            "beta": cfg.ponder.beta,
            "prior_lambda": cfg.ponder.prior_lambda,
            "dynamic_n": cfg.ponder.dynamic_n,
            "n_steps": cfg.ponder.n_steps,
            "use_wpe": cfg.model.use_wpe,
            "use_rope": cfg.model.use_rope,
        },
        flush=True,
    )
    train_ponder.train(cfg)


if __name__ == "__main__":
    main()
