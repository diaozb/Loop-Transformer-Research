# Migration Plan

This directory is a clean experiment infrastructure for the current project.
The old `src/` tree remains the golden reference until this infra reproduces
the important results.

## Scope

Migrate first:

- Tasks: `parity`, `copy`, `mod_add`
- Training modes: fixed-loop, PonderNet
- Positional modes: NoPE, RoPE, WPE-all, WPE-once
- Metrics: exact masked accuracy, answer accuracy, token accuracy for copy,
  average Ponder exit loops, hidden-state dynamics
- Outputs: resolved configs, JSONL logs, CSV eval results, plots

Defer unless explicitly needed:

- `dict`, `multi`, `reachability`
- one-off historical intervention scripts
- full vendored Transformers cleanup

## Non-Negotiable Semantic Equivalence

- The loop recurrence is initialized with zero hidden state and uses input
  injection at every step:
  `output = backbone(output + input_embedding)`.
- Fixed-loop training supervises sample `i` at `length_i - 1`, not at a global
  final loop.
- Existing length sampling semantics are preserved: `np.random.randint(min, max)`
  excludes the right boundary.
- Loss is cross entropy over positions where `mask == 1`.
- Copy headline reporting must include `answer_accuracy`; `token_accuracy` is
  diagnostic only because it includes masked tail positions.
- PonderNet halt probabilities are produced from hazards with the final step
  forced to halt, then normalized.
- PonderNet halt head reads the hidden state at the first answer position.
- PonderNet loss is expected masked CE plus `beta * KL(q || geometric_prior)`.
- WPE-once means learned GPT-2 WPE is used only on recurrent step `0`.

## Phases

### Phase 1: Skeleton and Config

Create package structure, config dataclasses, YAML loading, registry patterns,
and CLI entry points. No training behavior changes are allowed here.

Review gate:

- Configs are explicit and serializable.
- Defaults do not hide experiment choices.
- PE, trainer, task, and model settings can be ablated independently.

### Phase 2: Data

Port `parity`, `copy`, and `mod_add` generators with masks and metadata.

Review gate:

- Batch objects expose `inputs`, `targets`, `mask`, and `lengths`.
- Legacy special tokens are documented and unchanged.
- Generators can be compared against old `src/generate_training_data.py`.

### Phase 3: Model

Port the looped transformer wrapper, PE mode dispatch, and the minimal GPT-2
source needed by this project. The runtime infra must not import the legacy
`src/transformers` package.

Review gate:

- `looped_forward` can return logits and hidden states.
- NoPE, RoPE, WPE-all, and WPE-once semantics are explicit.
- The compatibility path can load old checkpoints where practical.

### Phase 4: Training

Port fixed-loop and PonderNet trainers.

Review gate:

- One-batch loss formulas match current code.
- Checkpoints include model state, optimizer state, and resolved config.
- Metrics are logged locally independent of W&B.

### Phase 5: Evaluation and Visualization

Port dense loop eval, Ponder auto-exit eval, hidden dynamics, and plots.

Review gate:

- CSV/JSON are the source of truth.
- Plots are deterministic transformations of saved data.
- Metrics are named consistently across tasks and trainers.

### Phase 6: Reproduction

Use old checkpoints and selected short reruns to verify parity/copy/mod_add
numbers before deprecating old scripts.

Review gate:

- Key CSV summaries agree with current `eval/` within sampling noise.
- Any mismatch is documented before more experiments are launched.
