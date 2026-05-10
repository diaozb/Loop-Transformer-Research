# Looped Transformer Length Generalization Experiments

This repository is used to study **length generalization** of looped Transformers on synthetic sequence tasks, especially:

- `copy`
- `parity`
- `addition`
- `sum_reverse`
- `mod_add`
- `mod_add_digits`

The current experiments focus on the `copy` task under different positional encoding and computation-depth settings:

| Setting | Training Type | Positional Encoding |
|---|---|---|
| Copy + NoPE + fixed-loop | fixed loop | no positional encoding |
| Copy + RoPE + fixed-loop | fixed loop | rotary positional encoding |
| Copy + APE/WPE once + fixed-loop | fixed loop | learned absolute PE added only once |
| Copy + NoPE + PonderNet | adaptive exit | no positional encoding |

The main purpose is to compare whether the model learns an iterative copy algorithm and whether adaptive halting helps or hurts length extrapolation.

---

## 1. Repository Structure

The important files and directories are:

```text
src/
├── train.py
├── train_ponder.py
├── models.py
├── schema.py
├── generate_training_data.py
├── test_func.py
├── curriculum.py
├── conf/
│   ├── copy_nope_fixed_loop.yaml
│   ├── copy_rope_fixed_loop.yaml
│   ├── copy_ape_once_fixed_loop.yaml
│   ├── models/
│   │   └── copy.yaml
│   └── wandb.yaml
├── eval_copy_fixed_loop.py
├── run_eval_copy_nope_fixed_loop.sh
├── run_eval_copy_rope_fixed_loop.sh
├── run_eval_copy_ape_once_fixed_loop.sh
├── write_result_copy_nope_fixed_loop.sh
├── write_result_copy_rope_fixed_loop.sh
└── write_result_copy_ape_once_fixed_loop.sh
```

Output directories:

```text
models/
├── nope_baselines/
├── rope_baselines/
└── ape_once_baselines/

eval/
├── nope_copy_fixed_loop/
├── rope_copy_fixed_loop/
└── ape_once_copy_fixed_loop/
```

---

## 2. Main Code Components

### 2.1 `train.py`

`train.py` is the main training script for **fixed-loop** experiments.

Run format:

```bash
python train.py --conf ./conf/<config_name>.yaml
```

For fixed-loop training, the model runs multiple loop steps. If a sample has length `L`, the training loss is taken from the `L`-th loop step.

Conceptually:

```text
input x
  ↓
loop step 1
  ↓
loop step 2
  ↓
...
  ↓
loop step L
  ↓
loss for a length-L sample
```

In code, the core logic is:

```python
states = model.looped_forward(xs, horizon=curriculum.n_points + 2)
states_list.append(states[batch_num[t].item() - 1][t])
```

So fixed-loop training explicitly ties:

```text
sequence length L  ↔  loop depth L
```

This provides strong depth supervision.

---

### 2.2 `train_ponder.py`

`train_ponder.py` is used for **PonderNet / adaptive-exit** training.

Unlike fixed-loop training, PonderNet does not force length `L` to exit at loop `L`. Instead, the model predicts a halting probability at each loop step.

Conceptually:

```text
input x
  ↓
loop step 1 → halt probability p1
  ↓
loop step 2 → halt probability p2
  ↓
...
  ↓
weighted loss over all steps
```

The main loss is:

```text
expected reconstruction loss + beta * KL(halting distribution || geometric prior)
```

Current important editable parameters include:

```python
TASK = "copy"
OUT_DIR = "../models/nope_baselines"
SEED = 42

USE_WANDB = True
WANDB_PROJECT = "looped-tf-nope-copy-ponder"

TRAIN_STEPS = 100001
BATCH_SIZE = 64
EVAL_EVERY = 1000

MODEL_USE_WPE = False
MODEL_USE_ROPE = False

PONDER_BETA = 0.01
PONDER_PRIOR_LAMBDA = 0.2
PONDER_N_STEPS = 20
PONDER_DYNAMIC_N = False
```

---

### 2.3 `models.py`

`models.py` defines the main looped Transformer architecture.

The main model class is:

```python
GeneralTransformerModel
```

The core update is:

```python
zs = self._read_in(zs)
output = torch.zeros_like(zs)

for i in range(horizon):
    output = self.forward_single(output + zs, step_idx=i)
    output_list.append(self._read_out(output))
```

Mathematically, the loop update can be viewed as:

```text
H(0) = 0
H(t+1) = F_theta(H(t) + Z)
```

where:

- `Z` is the input embedding,
- `H(t)` is the hidden state at loop step `t`,
- `F_theta` is the shared Transformer block.

The model supports three positional encoding modes:

| Mode | Config |
|---|---|
| NoPE | `use_wpe: false`, `use_rope: false` |
| RoPE | `use_wpe: false`, `use_rope: true` |
| APE/WPE once | `use_wpe: true`, `wpe_mode: once`, `use_rope: false` |

For APE/WPE once, `models.py` uses:

```python
if wpe_mode == "once":
    return step_idx in (None, 0)
```

So the learned absolute position embedding is added only at the first loop step.

---

### 2.4 `schema.py`

`schema.py` defines which fields are legal in YAML configs.

To support APE/WPE once, `schema.py` must include:

```python
"wpe_mode": merge(tstring, allowed(["none", "once", "all"]), nullable, default(None)),
```

inside `model_schema`.

Without this field, running a config with:

```yaml
wpe_mode: once
```

will raise:

```text
CerberusError: config could not be validated against schema
{'model': [{'wpe_mode': ['unknown field']}]}
```

---

### 2.5 `generate_training_data.py`

This file generates synthetic training data.

For the `copy` task, the model receives a binary sequence and must reproduce it in the answer region.

The data format is approximately:

```text
source bits + delimiter + answer/pad region
```

Example:

```text
input:  1 0 1 1 = _ _ _ _
target: _ _ _ _ 1 0 1 1
```

The copy task uses binary tokens plus special tokens such as delimiter and padding.

---

### 2.6 `test_func.py`

`test_func.py` contains helper functions used during training-time evaluation.

`train.py` uses these functions to log:

```text
test_acc
test_acc_chosen
test_acc_final
test_acc_chosen_final
```

---

## 3. Environment Setup

Activate the environment:

```bash
conda activate looptf
cd /data/diaozb/looped-tf-length-generalization/src
```

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Check W&B login:

```bash
wandb status
```

If needed:

```bash
wandb login
```

---

## 4. W&B Setup

Each config contains a `wandb` section.

Example:

```yaml
wandb:
    entity: diaozb-tsinghua-university
    project: looped-tf-nope-copy-fixed-loop
    name: copy_nope_fixed_loop
    notes: ''
    log_every_steps: 1000
```

If W&B reports:

```text
wandb.errors.errors.CommError: permission denied
```

then the `entity` is probably wrong.

Use either:

```yaml
entity: diaozb-tsinghua-university
```

or:

```yaml
entity: null
```

depending on where the logged-in W&B account has write permission.

---

## 5. Training Fixed-loop Models

### 5.1 NoPE + Copy + fixed-loop

Config:

```text
src/conf/copy_nope_fixed_loop.yaml
```

Important settings:

```yaml
model:
    use_wpe: false
    use_rope: false

out_dir: ../models/nope_baselines/copy_fixed_loop

wandb:
    project: looped-tf-nope-copy-fixed-loop
    name: copy_nope_fixed_loop
```

Run:

```bash
python train.py --conf ./conf/copy_nope_fixed_loop.yaml
```

Recommended tmux run:

```bash
tmux new -s nope-copy-fixed-loop

conda activate looptf
cd /data/diaozb/looped-tf-length-generalization/src
python train.py --conf ./conf/copy_nope_fixed_loop.yaml 2>&1 | tee nope_copy_fixed_loop.log
```

Detach:

```text
Ctrl-b then d
```

Reattach:

```bash
tmux attach -t nope-copy-fixed-loop
```

Output directory:

```text
../models/nope_baselines/copy_fixed_loop/<RUN_ID>/
```

Expected files:

```text
config.yaml
best.pt
model.pt
```

---

### 5.2 RoPE + Copy + fixed-loop

Config:

```text
src/conf/copy_rope_fixed_loop.yaml
```

Important settings:

```yaml
model:
    use_wpe: false
    use_rope: true
    rope_theta: 10000.0

out_dir: ../models/rope_baselines/copy_fixed_loop

wandb:
    project: looped-tf-rope-copy-fixed-loop
    name: copy_rope_fixed_loop
```

Run:

```bash
python train.py --conf ./conf/copy_rope_fixed_loop.yaml
```

Recommended tmux run:

```bash
tmux new -s rope-copy-fixed-loop

conda activate looptf
cd /data/diaozb/looped-tf-length-generalization/src
python train.py --conf ./conf/copy_rope_fixed_loop.yaml 2>&1 | tee rope_copy_fixed_loop.log
```

Because `train.py` automatically appends `rope/` when `use_rope: true`, the final output directory is:

```text
../models/rope_baselines/copy_fixed_loop/rope/<RUN_ID>/
```

---

### 5.3 APE/WPE once + Copy + fixed-loop

Config:

```text
src/conf/copy_ape_once_fixed_loop.yaml
```

Important settings:

```yaml
model:
    use_wpe: true
    wpe_mode: once
    use_rope: false

out_dir: ../models/ape_once_baselines/copy_fixed_loop

wandb:
    project: looped-tf-ape-once-copy-fixed-loop
    name: copy_ape_once_fixed_loop
```

Run:

```bash
python train.py --conf ./conf/copy_ape_once_fixed_loop.yaml
```

Recommended tmux run:

```bash
tmux new -s ape-once-copy-fixed-loop

conda activate looptf
cd /data/diaozb/looped-tf-length-generalization/src
python train.py --conf ./conf/copy_ape_once_fixed_loop.yaml 2>&1 | tee ape_once_copy_fixed_loop.log
```

Because `train.py` automatically appends `wpe/` when `use_wpe: true`, the final output directory is:

```text
../models/ape_once_baselines/copy_fixed_loop/wpe/<RUN_ID>/
```

Before running, verify:

```bash
grep -n "use_wpe\|wpe_mode\|use_rope" ./conf/copy_ape_once_fixed_loop.yaml
```

Expected:

```yaml
use_wpe: true
wpe_mode: once
use_rope: false
```

---

## 6. Training NoPE + Copy + PonderNet

PonderNet training is controlled mainly inside:

```text
src/train_ponder.py
```

Important settings:

```python
TASK = "copy"

MODEL_USE_WPE = False
MODEL_USE_ROPE = False

PONDER_BETA = 0.01
PONDER_PRIOR_LAMBDA = 0.2
PONDER_N_STEPS = 20

TRAIN_STEPS = 100001
EVAL_EVERY = 1000
WANDB_LOG_EVERY = 100
```

Run:

```bash
python train_ponder.py
```

Recommended tmux run:

```bash
tmux new -s nope-copy-ponder

conda activate looptf
cd /data/diaozb/looped-tf-length-generalization/src
python train_ponder.py 2>&1 | tee nope_copy_ponder.log
```

Output directory:

```text
../models/nope_baselines/copy_ponder/<RUN_ID>/
```

Expected files:

```text
config.yaml
best.pt
model.pt
wandb/
```

---

## 7. Fixed-loop Evaluation

Fixed-loop evaluation uses:

```text
eval_copy_fixed_loop.py
```

It performs dense loop evaluation over:

```text
length × loop
```

For each length, it records:

```text
best forced-loop answer accuracy
best loop
token accuracy
step loss
accuracy at loop L
accuracy at loop 20
accuracy at loop 40
```

Main output files:

```text
summary_by_length.csv
per_step_by_length.csv
forced_accuracy_heatmap.png
step_loss_heatmap.png
best_accuracy_vs_length.png
best_loop_vs_length.png
manifest.json
```

---

### 7.1 Evaluate NoPE fixed-loop

Run:

```bash
bash run_eval_copy_nope_fixed_loop.sh
```

This reads the latest run from:

```text
../models/nope_baselines/copy_fixed_loop/<RUN_ID>/
```

and writes results to:

```text
../eval/nope_copy_fixed_loop/<RUN_ID>/diagnostics_loops40/
```

Summarize results:

```bash
bash write_result_copy_nope_fixed_loop.sh
```

This writes:

```text
nope_copy_fixed_loop.txt
```

---

### 7.2 Evaluate RoPE fixed-loop

Run:

```bash
bash run_eval_copy_rope_fixed_loop.sh
```

This reads the latest run from:

```text
../models/rope_baselines/copy_fixed_loop/rope/<RUN_ID>/
```

and writes results to:

```text
../eval/rope_copy_fixed_loop/<RUN_ID>/diagnostics_loops40/
```

Summarize results:

```bash
bash write_result_copy_rope_fixed_loop.sh
```

This writes:

```text
rope_copy_fixed_loop.txt
```

---

### 7.3 Evaluate APE/WPE once fixed-loop

Run:

```bash
bash run_eval_copy_ape_once_fixed_loop.sh
```

This reads the latest run from:

```text
../models/ape_once_baselines/copy_fixed_loop/wpe/<RUN_ID>/
```

and writes results to:

```text
../eval/ape_once_copy_fixed_loop/<RUN_ID>/diagnostics_loops40/
```

Summarize results:

```bash
bash write_result_copy_ape_once_fixed_loop.sh
```

This writes:

```text
ape_once_copy_fixed_loop.txt
```

---

## 8. Evaluation Script Details

The fixed-loop eval script is:

```text
eval_copy_fixed_loop.py
```

Example command:

```bash
python eval_copy_fixed_loop.py \
  --run-dir ../models/nope_baselines/copy_fixed_loop/<RUN_ID> \
  --checkpoint best.pt \
  --lengths 1-20,21,22,30,40,60,400 \
  --id-max 20 \
  --max-loops 40 \
  --batch-size 256 \
  --long-threshold 100 \
  --long-batch-size 16 \
  --n-batches 8 \
  --out-dir ../eval/nope_copy_fixed_loop/<RUN_ID>/diagnostics_loops40 \
  --wandb \
  --wandb-project looped-tf-nope-copy-fixed-loop \
  --wandb-name eval_nope_copy_fixed_loop_<RUN_ID>_loops40
```

Important arguments:

| Argument | Meaning |
|---|---|
| `--run-dir` | Directory containing `config.yaml`, `best.pt`, `model.pt` |
| `--checkpoint` | Usually `best.pt` or `model.pt` |
| `--lengths` | Evaluation lengths |
| `--id-max` | Maximum in-distribution length |
| `--max-loops` | Maximum loop steps to sweep |
| `--batch-size` | Batch size for normal lengths |
| `--long-batch-size` | Batch size for very long lengths |
| `--out-dir` | Output directory |
| `--wandb` | Whether to upload eval results to W&B |

---

## 9. Result Files

After running eval, the most important files are:

### `summary_by_length.csv`

One row per length.

Important columns:

```text
split
length
best_forced_answer_acc
best_forced_answer_step
best_token_acc
best_token_step
min_step_loss
min_loss_step
acc_loop_L
acc_loop_20
acc_loop_40
```

### `per_step_by_length.csv`

One row per `(length, loop)` pair.

Important columns:

```text
length
loop
answer_acc
token_acc
step_loss
n_examples
```

### `forced_accuracy_heatmap.png`

Heatmap of:

```text
length × loop → answer accuracy
```

This is useful for checking whether the model learns a diagonal or ridge-like iterative pattern.

### `step_loss_heatmap.png`

Heatmap of:

```text
length × loop → step loss
```

This helps inspect whether deeper loops improve prediction.

### `best_accuracy_vs_length.png`

Line plot of:

```text
length → best forced-loop accuracy
```

### `best_loop_vs_length.png`

Line plot of:

```text
length → best loop
```

This is useful for seeing whether the best computation depth grows with sequence length.

---

## 10. Result Summary Placeholders

The numerical results should be filled in later.

### 10.1 NoPE + Copy + fixed-loop

| Length | Best forced acc | Best loop | Acc at loop L | Acc at loop 20 | Acc at loop 40 |
|---:|---:|---:|---:|---:|---:|
| 20 | TODO | TODO | TODO | TODO | TODO |
| 30 | TODO | TODO | TODO | TODO | TODO |
| 40 | TODO | TODO | TODO | TODO | TODO |
| 60 | TODO | TODO | TODO | TODO | TODO |
| 400 | TODO | TODO | TODO | TODO | TODO |

Notes:

```text
TODO: add qualitative observations.
```

---

### 10.2 RoPE + Copy + fixed-loop

| Length | Best forced acc | Best loop | Acc at loop L | Acc at loop 20 | Acc at loop 40 |
|---:|---:|---:|---:|---:|---:|
| 20 | TODO | TODO | TODO | TODO | TODO |
| 30 | TODO | TODO | TODO | TODO | TODO |
| 40 | TODO | TODO | TODO | TODO | TODO |
| 60 | TODO | TODO | TODO | TODO | TODO |
| 400 | TODO | TODO | TODO | TODO | TODO |

Notes:

```text
TODO: add qualitative observations.
```

---

### 10.3 APE/WPE once + Copy + fixed-loop

| Length | Best forced acc | Best loop | Acc at loop L | Acc at loop 20 | Acc at loop 40 |
|---:|---:|---:|---:|---:|---:|
| 20 | TODO | TODO | TODO | TODO | TODO |
| 30 | TODO | TODO | TODO | TODO | TODO |
| 40 | TODO | TODO | TODO | TODO | TODO |
| 60 | TODO | TODO | TODO | TODO | TODO |
| 400 | TODO | TODO | TODO | TODO | TODO |

Notes:

```text
TODO: add qualitative observations.
```

---

### 10.4 NoPE + Copy + PonderNet

| Length | Auto-exit acc | Expected steps | Best forced acc | Best forced loop |
|---:|---:|---:|---:|---:|
| 20 | TODO | TODO | TODO | TODO |
| 30 | TODO | TODO | TODO | TODO |
| 40 | TODO | TODO | TODO | TODO |
| 60 | TODO | TODO | TODO | TODO |

Notes:

```text
TODO: add PonderNet halting observations.
```

---

## 11. Common Issues

### 11.1 W&B permission denied

Error:

```text
wandb.errors.errors.CommError: permission denied
```

Cause:

```text
The config is trying to write to a W&B entity where the current account has no permission.
```

Fix:

```yaml
wandb:
    entity: diaozb-tsinghua-university
```

or:

```yaml
wandb:
    entity: null
```

---

### 11.2 `wpe_mode` unknown field

Error:

```text
CerberusError: config could not be validated against schema
{'model': [{'wpe_mode': ['unknown field']}]}
```

Cause:

```text
schema.py does not allow model.wpe_mode.
```

Fix:

Add this field to `model_schema` in `schema.py`:

```python
"wpe_mode": merge(tstring, allowed(["none", "once", "all"]), nullable, default(None)),
```

---

### 11.3 Output path has extra `rope/` or `wpe/`

This is expected.

In `train.py`:

```python
if args.model.use_wpe:
    out_dir = os.path.join(out_dir, "wpe")
if args.model.use_rope:
    out_dir = os.path.join(out_dir, "rope")
```

So:

```text
RoPE output:
../models/rope_baselines/copy_fixed_loop/rope/<RUN_ID>/

APE/WPE output:
../models/ape_once_baselines/copy_fixed_loop/wpe/<RUN_ID>/
```

---

### 11.4 Very short test runs may fail at final eval

`train.py` defines `test_len` inside the 1000-step eval block. In normal full training, step `0` enters this block, so this is fine.

If running a very unusual short test, it is safer to define near the start of `train()`:

```python
test_len = args.training.test_len
```

---

## 12. Recommended Workflow

For a complete fixed-loop comparison:

```bash
# 1. Train NoPE fixed-loop
python train.py --conf ./conf/copy_nope_fixed_loop.yaml

# 2. Train RoPE fixed-loop
python train.py --conf ./conf/copy_rope_fixed_loop.yaml

# 3. Train APE/WPE once fixed-loop
python train.py --conf ./conf/copy_ape_once_fixed_loop.yaml

# 4. Evaluate NoPE
bash run_eval_copy_nope_fixed_loop.sh
bash write_result_copy_nope_fixed_loop.sh

# 5. Evaluate RoPE
bash run_eval_copy_rope_fixed_loop.sh
bash write_result_copy_rope_fixed_loop.sh

# 6. Evaluate APE/WPE once
bash run_eval_copy_ape_once_fixed_loop.sh
bash write_result_copy_ape_once_fixed_loop.sh
```

For PonderNet:

```bash
python train_ponder.py
```

Then run the corresponding PonderNet eval scripts.

---

## 13. What to Look For

For fixed-loop models, the most important quantities are:

```text
best_forced_answer_acc
best_forced_answer_step
acc_loop_L
accuracy heatmap
best loop vs length
```

These answer:

```text
Does the model improve with more loop steps?
Does the best loop grow with input length?
Does the model learn an iterative algorithm?
Does the model extrapolate beyond the training length?
```

For PonderNet models, the most important quantities are:

```text
auto-exit accuracy
expected halting steps
halting distribution
forced-loop override accuracy
```

These answer:

```text
Does the model learn the algorithm?
Does it stop too early?
Does adaptive halting use more computation for longer inputs?
Is failure caused by wrong halting or by lack of a deep iterative algorithm?
```

---

## 14. Current Experiment Status

TODO: Fill in after all runs are collected.

| Experiment | Training complete? | Eval complete? | Result file |
|---|---|---|---|
| NoPE + Copy + fixed-loop | TODO | TODO | `nope_copy_fixed_loop.txt` |
| RoPE + Copy + fixed-loop | TODO | TODO | `rope_copy_fixed_loop.txt` |
| APE/WPE once + Copy + fixed-loop | TODO | TODO | `ape_once_copy_fixed_loop.txt` |
| NoPE + Copy + PonderNet | TODO | TODO | TODO |

---

## 15. Notes for Future Work

Potential follow-up diagnostics:

```text
1. Per-position copy accuracy
2. Hidden-state delta L2 across loop steps
3. Cosine similarity between H(t) and H(t-1)
4. Forced-loop override for PonderNet
5. Halting distribution p(t | L)
6. Comparison between auto-exit loop and best forced loop
```

These diagnostics are useful for connecting the empirical behavior to theoretical questions about iterative computation, information propagation, and adaptive depth.
