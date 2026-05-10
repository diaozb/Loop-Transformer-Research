# CODEX Experiment Plan

Repo-grounded draft for `/data/yizhou/looped-tf-length-generalization`.

Last updated: 2026-04-22

---

## 0. Purpose

This file replaces the earlier Claude-written assumptions with a plan that matches the current repository.

The main target is still the same:

- understand the sensitivity / extrapolation story for looped Transformers and PonderNet
- prepare a clean evaluation pipeline for the next round of experiments
- avoid mixing together theoretical notation and implementation details that do not actually match the code

---

## 1. Current Repository Reality

### 1.1 Paths and Assets

- Repo root: `/data/yizhou/looped-tf-length-generalization`
- Training code: `src/train.py`, `src/train_ponder.py`
- Data generators: `src/generate_training_data.py`
- Core model: `src/models.py`
- Eval scripts: `src/eval/`
- Existing checkpoints: `models/`
- Existing eval outputs: `eval/`
- Uploaded PDF memos: `results_collection/`

Only these PDFs are currently in the repo:

- `results_collection/2026-02-04_loop_train_eval_multi_task_report.pdf`
- `results_collection/2026-02-17_pondernet_experiment_memo.pdf`
- `results_collection/2026-02-17_positional_encoding_experiment_memo.pdf`

There is no `LoopTF_theory_9.pdf` in the current tree.

The current quickstart/result summary is already in:

- `CODEBASE_QUICKSTART_AND_EXPERIMENT_LOG.md`

### 1.2 Actual Model and Task Setup

The implementation is not the abstract `X in {-1,1}^L` plus BCE setup from the earlier draft.

What the code actually does:

- The base model is a shared-weight GPT-2 style block in `src/models.py`.
- The loop update is `output = block(output + input_embed)` at each iteration.
- There is input injection at every loop.
- There is no extra outer residual of the form `H^(t+1) = H^(t) + X0 + g(H^(t) + X0)` outside the Transformer block itself.
- Tasks are tokenized synthetic sequence tasks, not raw plus/minus one bit vectors.
- Supervision is token-level classification on masked answer positions.
- The fixed-loop trainer uses `CrossEntropyLoss`.
- The Ponder trainer uses expected masked CE plus `beta * KL(q || geometric_prior)`.

Task format in the current code:

- `parity`: one answer token after an `=` delimiter
- `copy`: copy the source sequence into the answer region after `=`
- `addition`, `sum_reverse`, `mod_add`, `mod_add_digits`, `multi`, `dict`: all use task-specific prompt formatting and masked answer regions

### 1.3 Actual Ponder Implementation

The current Ponder implementation in `src/train_ponder.py` has these properties:

- Halting head reads the hidden state at the first answer position.
- `p_t` is formed from hazard rates `lambda_t` in the standard cumulative-product way.
- Training loss is expected masked CE over steps plus KL to a truncated geometric prior.
- Current code evaluates `best.pt` using `argmax(p_t)` within the training horizon, but dense eval uses sampled auto-exit behavior in `src/eval/eval_parity_ponder.py` and `src/eval/eval_copy_ponder.py`.

For paper-facing claims, the dense eval outputs under `eval/parity_ponder/...` and `eval/copy_ponder/...` are more important than the in-training selection metric.

### 1.4 Existing Checkpoints to Reuse

The repo already contains enough assets for the first evaluation phase.

Fixed-loop checkpoints:

- `models/parity/d9de8dd7-d283-4236-aa71-d02ce63ab40a/model.pt`
- `models/parity/rope/938b8082-a3ca-49be-8153-4e288189f234/model.pt`
- `models/copy/8fe7447f-9d64-4de4-bf04-29c50e860cd6/model.pt`
- `models/copy/rope/d0ced92d-250f-4bc5-b844-72efafca5576/model.pt`
- `models/copy/wpe/289ac6be-1382-47fe-ac9b-5d33ed9f9104/best.pt`

Ponder checkpoints:

- `models/parity_ponder/86494c70-cf32-48e0-a9cd-3e914bd41768/model.pt`
- `models/copy_ponder/19a0316b-1a36-4857-ae7b-62c21e521448/model.pt`
- `models/copy_ponder/9da97db5-880a-4d5c-a23e-c0b1f6f86f33/model.pt`

Important context from existing eval outputs:

- `parity_ponder` at length 40 is already very strong, around `0.992` accuracy with average loops around `14.6`
- `copy_ponder` with regularization collapses at length 40, sequence accuracy `0.0`
- `copy_ponder` without regularization also collapses, but exits almost immediately
- fixed-loop `copy` depends heavily on positional treatment:
  - no-position baseline is strong by length 30 but not yet the clean long-length positive control we want
  - RoPE fixed-loop copy still fails badly at length 40
  - the strongest current positive result is the WPE add-once run, around `0.808` exact-match at length 40

### 1.5 Known Caveats

These are important enough that the experiment plan must respect them:

- `src/conf/parity.yaml` and `src/conf/copy.yaml` no longer match the archived fixed-loop baselines. Use archived run configs as source of truth when reproducing old results.
- `test_acc_chosen` / chosen-step is not relevant for this paper and its implementation is not trustworthy. Ignore it.
- In copy eval, `token_accuracy` includes masked tail positions and is not the same thing as answer-only accuracy. Prefer exact-match sequence accuracy and `answer_accuracy`.
- The current `train_ponder.py` file is driven by editable globals, not a proper CLI. Sweep experiments will need either a wrapper script or a small refactor before large-scale runs.

---

## 2. Working Conventions for the Next Round

These are the repo-aligned defaults unless we explicitly change them.

- Work inside this repository. Do not create a separate `~/experiments` repo.
- Keep new analysis outputs under `results_collection/exp1` through `results_collection/exp6`.
- Keep publication-quality figures under `figures/`.
- Put new evaluation Python code under `src/eval/`.
- Put launch wrappers under `scripts/` if needed.
- Use archived checkpoint configs and saved eval outputs as the starting point, not current mutable defaults in `src/conf/`.
- For main metrics, use:
  - exact-match sequence accuracy
  - `answer_accuracy`
  - average halting loops for Ponder
- Do not use:
  - `test_acc_chosen`
  - copy `token_accuracy` as the headline metric

---

## 3. Decisions That Still Need Confirmation

These are the only substantive choices that remain open before we launch runs.

### 3.1 What Counts as the Fixed-Loop Comparator in Exp 2?

There are three different candidate stories in the current repo:

1. Historical fixed-loop baseline:
   - parity no-position: `models/parity/d9de...`
   - copy no-position: `models/copy/8fe7...`

2. Architecture-matched RoPE comparison against current Ponder mainline:
   - parity fixed RoPE: `models/parity/rope/938b...`
   - copy fixed RoPE: `models/copy/rope/d0ce...`

3. Positive-control copy model that really does long-length copy well:
   - copy fixed WPE add-once: `models/copy/wpe/289a...`

My recommendation:

- use the RoPE fixed-loop models as the primary architecture-matched comparison against current PonderNet
- also include the WPE add-once copy model as a positive control, because it proves the broader looped architecture can solve long copy under fixed compute when positional handling is favorable

### 3.2 What Base Architecture Should Exp 4 and Exp 5 Sweep?

For new Ponder sweeps, the cleanest default is:

- parity: same architecture as `models/parity_ponder/86494...`
- copy: same architecture as `models/copy_ponder/19a03...`
- vary only `beta` and `prior_lambda`

That means:

- `use_rope=True`
- `use_wpe=False`
- `dynamic_n=False`
- `n_steps=20`

This matches the current Ponder mainline better than the older no-position fixed-loop baselines.

### 3.3 Definition of k-Parity for Exp 6

In this repo, `k-parity` should be implemented as:

- keep the current parity prompt format
- use the XOR of only the first `k` source bits as the answer token
- keep one answer position, exactly like existing parity

This is different from the abstract plus/minus one notation in the old draft, but it is the correct repo-aligned implementation.

---

## 4. Experiment Specifications

### Experiment 1: Step-wise Loss Profiles

Priority: Highest
Training required: No

Goal:

- characterize shallow vs deep behavior using the actual Ponder implementation
- compare parity and copy under the same collection pipeline

Procedure:

1. Load:
   - `models/parity_ponder/86494.../model.pt`
   - `models/copy_ponder/19a03.../model.pt`
2. For each task and each length in `{5, 10, 15, 20}`:
   - sample 500 inputs
   - collect all steps up to `T_max=40` without auto-exit
   - record at each step:
     - masked CE loss
     - unconditional halt probability `p_t`
     - raw hazard `lambda_t`
     - optional exact-match at that step
3. Save:
   - `results_collection/exp1/stepwise_loss_data.csv`
   - `results_collection/exp1/analysis.md`

Notes:

- use masked CE, not BCE
- this should be implemented by reusing the hidden/logit collection logic from `src/eval/eval_parity_ponder.py` and `src/eval/eval_copy_ponder.py`

Main figure:

- `figures/exp1_stepwise_loss.pdf`
- `figures/exp1_stepwise_loss.png`

### Experiment 2: PonderNet vs Fixed-Loop Comparison

Priority: Highest
Training required: No for the initial pass

Goal:

- separate "halting effect" from "architecture / positional encoding effect"

Planned model set:

- parity fixed no-position
- parity fixed RoPE
- parity Ponder RoPE
- copy fixed no-position
- copy fixed RoPE
- copy fixed WPE add-once
- copy Ponder RoPE
- copy Ponder RoPE without KL regularization

Procedure:

1. Re-evaluate all selected checkpoints on a unified length grid, ideally `L in {2, 4, ..., 40}`.
2. Record:
   - sequence accuracy
   - `answer_accuracy`
   - average loops for Ponder models
3. Save:
   - `results_collection/exp2/comparison_data.csv`
   - `results_collection/exp2/comparison_table.md`
   - `results_collection/exp2/analysis.md`

Paper interpretation rule:

- if copy fixed RoPE also fails badly, then the claim cannot simply be "halting causes copy failure"
- if copy fixed WPE add-once succeeds while copy Ponder still collapses, then the conclusion becomes "adaptive halting is not the only factor; positional handling is also decisive"

Main figure:

- `figures/exp2_comparison.pdf`
- `figures/exp2_comparison.png`

### Experiment 3: Influence Dilution Validation

Priority: High
Training required: No

Goal:

- test whether single-bit perturbations in the input get diluted at answer positions as length grows

Procedure:

1. Start with the current Ponder parity and Ponder copy checkpoints.
2. For each length in `{4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40}`:
   - sample 200 inputs
   - flip one random source bit
   - collect hidden states across steps
   - compare answer-position hidden states between original and flipped input
3. Compute:
   - `Delta(1)` versus length
   - `Delta(t)` versus step
4. Save:
   - `results_collection/exp3/dilution_data.csv`
   - `results_collection/exp3/fitted_params.csv`
   - `results_collection/exp3/analysis.md`

Implementation note:

- answer positions must be defined using the current task masks, not a generic second half of the sequence

Main figure:

- `figures/exp3_dilution.pdf`
- `figures/exp3_dilution.png`

### Experiment 4: Beta / Gamma Sweep on Parity

Priority: High
Training required: Yes

Goal:

- test how parity Ponder behavior changes with `beta` and geometric-prior strength

Default training base:

- parity Ponder architecture matching `models/parity_ponder/86494...`
- vary only `beta` and `prior_lambda`

Grid:

| beta | lambda_p |
|------|----------|
| 0.01 | 0.1 |
| 0.01 | 0.5 |
| 0.01 | 0.9 |
| 0.1  | 0.1 |
| 0.1  | 0.5 |
| 0.1  | 0.9 |
| 1.0  | 0.1 |
| 1.0  | 0.5 |
| 1.0  | 0.9 |
| 5.0  | 0.1 |
| 5.0  | 0.5 |
| 5.0  | 0.9 |

Fallback grid:

- `(0.01, 0.1)`
- `(0.01, 0.9)`
- `(1.0, 0.1)`
- `(1.0, 0.9)`
- `(5.0, 0.1)`
- `(5.0, 0.9)`

Outputs:

- new checkpoints under `models/exp4/parity_ponder/...`
- `results_collection/exp4/parity_sweep_data.csv`
- `results_collection/exp4/parity_sweep_summary.csv`
- `results_collection/exp4/analysis.md`

Important implementation detail:

- `src/train_ponder.py` is not sweep-friendly as-is
- before running Exp 4, create a wrapper or refactor that can set `TASK`, `PONDER_BETA`, `PONDER_PRIOR_LAMBDA`, and output root programmatically

Main figure:

- `figures/exp4_beta_gamma_parity.pdf`
- `figures/exp4_beta_gamma_parity.png`

### Experiment 5: Beta / Gamma Sweep on Copy

Priority: High
Training required: Yes

Goal:

- test whether copy collapse persists across the same Ponder regularization grid

Default training base:

- copy Ponder architecture matching `models/copy_ponder/19a03...`
- vary only `beta` and `prior_lambda`

Use the same grid and fallback policy as Exp 4.

Outputs:

- new checkpoints under `models/exp5/copy_ponder/...`
- `results_collection/exp5/copy_sweep_data.csv`
- `results_collection/exp5/copy_sweep_summary.csv`
- `results_collection/exp5/analysis.md`

Critical interpretation rule:

- if some copy settings become strong under Ponder, that is a real result and should not be hidden
- if all copy settings still collapse, that supports the current story

Main figure:

- `figures/exp5_beta_gamma_copy.pdf`
- `figures/exp5_beta_gamma_copy.png`

### Experiment 6: k-Parity Sensitivity Spectrum

Priority: High
Training required: Yes

Task definition:

- keep the current parity prompt format
- answer token is XOR of the first `k` source bits
- use `k in {1, 2, 4, 8, 20}`

Procedure:

1. Implement a `k-parity` generator by modifying the parity label rule only.
2. Verify that `k=20` matches standard parity at train length 20.
3. Train Ponder models for the selected `k` values.
4. Evaluate on `L in {2, 4, ..., 40}`.
5. Save:
   - `results_collection/exp6/kparity_data.csv`
   - `results_collection/exp6/analysis.md`

Optional:

- also record step-wise CE curves for `k in {1, 4, 20}`

Main figure:

- `figures/exp6_sensitivity_spectrum.pdf`
- `figures/exp6_sensitivity_spectrum.png`

---

## 5. Execution Order

### Phase 0: Alignment

1. Confirm the open choices in Section 3.
2. Freeze which checkpoint family counts as the Exp 2 comparator.
3. Decide whether Exp 4 and Exp 5 sweep RoPE Ponder only.

### Phase 1: Evaluation Only

1. Exp 1
2. Exp 2
3. Exp 3

This phase should reuse existing checkpoints and produce the first paper-quality evidence before any new training.

### Phase 2: New Training

1. Add a sweep-friendly runner around `src/train_ponder.py`
2. Exp 4
3. Exp 5
4. Exp 6

### Phase 3: Wrap-up

1. Write `results_collection/SUMMARY.md`
2. Consolidate figures
3. Decide final paper tables

---

## 6. Deliverables Checklist

Figures:

- `figures/exp1_stepwise_loss.pdf` and `.png`
- `figures/exp2_comparison.pdf` and `.png`
- `figures/exp3_dilution.pdf` and `.png`
- `figures/exp4_beta_gamma_parity.pdf` and `.png`
- `figures/exp5_beta_gamma_copy.pdf` and `.png`
- `figures/exp6_sensitivity_spectrum.pdf` and `.png`

Data:

- `results_collection/exp1/stepwise_loss_data.csv`
- `results_collection/exp2/comparison_data.csv`
- `results_collection/exp2/comparison_table.md`
- `results_collection/exp3/dilution_data.csv`
- `results_collection/exp3/fitted_params.csv`
- `results_collection/exp4/parity_sweep_data.csv`
- `results_collection/exp4/parity_sweep_summary.csv`
- `results_collection/exp5/copy_sweep_data.csv`
- `results_collection/exp5/copy_sweep_summary.csv`
- `results_collection/exp6/kparity_data.csv`

Analysis:

- `results_collection/exp1/analysis.md`
- `results_collection/exp2/analysis.md`
- `results_collection/exp3/analysis.md`
- `results_collection/exp4/analysis.md`
- `results_collection/exp5/analysis.md`
- `results_collection/exp6/analysis.md`
- `results_collection/SUMMARY.md`

---

## 7. Practical Notes

- If PDF text extraction tools are unavailable in the environment, do not block on them. The repo already contains the three uploaded PDF files and enough raw CSV/JSON eval outputs to start.
- If a run diverges, reduce LR first, then batch size, and log the failure instead of silently dropping it.
- Do not rely on current default configs for parity/copy reproduction without checking the archived run config next to the checkpoint.
