# Review Rubric

Use this checklist after each migration phase.

## Correctness

- The code preserves current experiment semantics unless a documented migration
  note says otherwise.
- Shapes are explicit at API boundaries.
- Masks and answer regions are never inferred from sequence length when a mask
  is available.
- Randomness is controlled through passed seeds or explicit RNG objects.
- Metrics are computed from saved tensors/CSV, not from plot images.

## Deep Learning Experiment Quality

- Every run writes a resolved config.
- Every run writes local metrics even when W&B is disabled or unavailable.
- Checkpoint naming distinguishes `best` from `last`.
- Eval scripts load checkpoints without relying on editable globals.
- Ablations are config changes, not separate copy-pasted scripts.

## Simplicity

- Modules have one responsibility.
- Registries map names to implementations without hidden side effects.
- No task-specific special case lives in the trainer unless unavoidable.
- Legacy compatibility code is isolated under `ltf/compat/`.

## Reproducibility

- Data generation parameters are logged.
- Git status and command line are captured when possible.
- Output directories are stable and contain all artifacts needed for review.
- Old result reproduction checks are documented before old scripts are ignored.

## `/review` Output Format

After each phase, record:

- Findings: concrete bugs or design risks, highest severity first.
- Open questions: unresolved assumptions that may affect reproduction.
- Verdict: pass, pass with caveats, or block.
