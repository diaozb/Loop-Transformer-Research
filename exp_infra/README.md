# Refactored Looped Transformer Experiment Infra

This package is the clean migration target for the current copy/parity/mod-add
experiments. The old `src/` tree remains the reference implementation until the
compatibility checks pass.

## First Supported Scope

- Tasks: `parity`, `copy`, `mod_add`
- Trainers: `fixed_loop`, `ponder`
- Position modes: `nope`, `rope`, `wpe_all`, `wpe_once`
- Outputs: resolved config, local JSONL metrics, checkpoints, CSV eval, plots

The runtime model code is self-contained under `ltf/models/`. It does not
import the legacy `src/transformers` package. Legacy `src/` is used only by
compatibility tests while migration is in progress.

## Single Experiment Config Example

Use the project conda environment, same as the legacy code:

```bash
conda activate ltf
```

```bash
PYTHONPATH=exp_infra python -m ltf.cli.train \
  --config exp_infra/configs/experiments/copy_ponder_nope.yaml \
  trainer.beta=0.01 trainer.prior_lambda=0.2 seed=42
```

Use `--dry-run` to print the resolved config without training.

## All-In-One Recipe Queue

For day-to-day experiments, prefer one all-in-one recipe file:

```bash
PYTHONPATH=exp_infra python -m ltf.cli.recipe \
  --recipe exp_infra/configs/recipes/smoke_queue.yaml
```

A recipe file supports global `defaults` plus an ordered `recipes` list. Each
recipe is run sequentially. The recipe `name` can include key parameters:

```yaml
name: copy_ponder_T{trainer.ponder_n_steps}_L{model.n_layer}_H{model.n_head}
```

Each run still writes a full `config_resolved.yaml`, `metrics.jsonl`, and
`checkpoints/best.pt`.

The recipe CLI prints queue-level progress and training progress to terminal:

```text
[recipe 01/48] queued name=... task=... trainer=... pe=... seed=...
[recipe 01/48 name] step=1000/100000 iter=1001/100001 loss=... eval_accuracy=...
```

Step logs are emitted at eval steps and at the final step; complete per-step
metrics remain in each run's `metrics.jsonl`.

For the main copy/parity sweep, use the convenience launcher:

```bash
exp_infra/run_copy_parity_sweep.sh --dry-run
CUDA_VISIBLE_DEVICES=0 exp_infra/run_copy_parity_sweep.sh
```

The sweep uses `train_steps=100001`, `batch_size=64`, `eval_batch_size=512`,
`eval_every=1000`, AdamW with `learning_rate=1e-4`, `weight_decay=0`, gradient
clip norm `1.0`, and cosine LR decay after each task's curriculum reaches its
configured maximum length. Ponder runs use `beta=0.01`, `prior_lambda=0.2`, and
`ponder_n_steps=20`.

## Checkpoint Eval

Eval reconstructs the model from an `exp_infra` checkpoint:

```bash
PYTHONPATH=exp_infra python -m ltf.cli.eval \
  --checkpoint exp_infra/runs/<run>/checkpoints/best.pt
```

The checkpoint contains the resolved config and model state. Legacy
`torch.save(model)` checkpoints from old `src/` still need a converter.

## Post-Train Default Eval

Training can optionally run the same default checkpoint eval after the final
checkpoint is written:

```yaml
eval:
  run_after_train: true
  after_train_checkpoint: best  # one of: best, last
```

This writes `eval/default_best/eval_metrics.json` and
`eval/default_best/eval_metrics.csv` inside the run directory. Dense eval stays
explicitly opt-in through `ltf.cli.eval --dense`.
