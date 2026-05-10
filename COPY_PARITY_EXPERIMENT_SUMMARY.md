# Copy / Parity Experiment Summary

## Scope

This note summarizes all **training settings** in the repo that are directly relevant to the paper story on:

- `copy` and `parity`
- position handling (`NoPE`, `WPE`, `RoPE`, historical `WPE once`)
- fixed-loop vs `PonderNet`
- adaptive exit
- extrapolation
- hidden-state convergence

I intentionally keep **auxiliary probes/intervention evals** separate from the main setting tables. They are useful, but they are not distinct training settings.

## Conventions

- For `copy`, I use `answer_accuracy` as the main score. `token_accuracy` is not a safe headline metric because it includes masked tail positions.
- For fixed-loop models, I report the **best answer accuracy over loop count** at a given length, plus the best loop when useful.
- For `PonderNet`, I report both:
  - best fixed-loop answer accuracy from the dense loop sweep
  - auto-exit accuracy and average exit loops
- `ID fit` means near the train boundary:
  - `copy`: lengths `20/21`
  - `parity`: lengths `20/21`
- `Hidden convergence` is a qualitative summary from `delta_l2_norm_mask`, `cosine_to_prev_mask`, and `answer_change_rate`.

## Important Caveats

- The historical `copy + WPE once + fixed-loop` runs are **not explicitly encoded** as `wpe_mode: once` in their saved configs. The `once` interpretation comes from run naming (`wpe_add_once`) plus behavior. This is high-confidence, but not encoded in the old config snapshot.
- There is **no parity NoPE+Ponder** checkpoint in the repo yet.
- There is **no parity WPE-once+Ponder** checkpoint in the repo yet.

## Copy

### Main Settings

| Setting | PE handling | Objective | Adaptive exit | ID fit | Extrapolation | Hidden convergence | Main read |
|---|---|---|---|---|---|---|---|
| `Fixed NoPE` (`8fe7447f...`) | none | fixed-loop | N/A | `L20=1.00`, `L21=1.00` | `L30=0.995 @ loop 26`, `L40=0.203 @ loop 39` | Strong at `L30`; converges to a wrong attractor at `L40` | Main NoPE baseline. Very strong to `L30`, weak at `L40`. |
| `Fixed WPE all/plain` (`2dbede54...`) | learned absolute WPE every loop | fixed-loop | N/A | `L20=0.579 @ loop 23`, `L21=0.490 @ loop 23` | `L30=0.000`, `L40=0.000` | No | WPE re-injection is bad for copy. |
| `Fixed WPE once` run 1 (`1a53af55...`) | historical add-once | fixed-loop | N/A | unstable / inconsistent | `L30≈0`, `L40=0` | No clear positive result | Early weak run. Do not use as the positive result. |
| `Fixed WPE once` run 2 (`289ac6be...`) | historical add-once | fixed-loop | N/A | `L20=1.00`, `L21=1.00` | `L30=0.998 @ loop 26`, `L40=0.808 @ loop 32` | Strong | Best `copy` result currently in repo. |
| `Fixed RoPE` run 1 (`8f1e9a81...`) | RoPE every loop | fixed-loop | N/A | near-zero | `L30=0.000`, `L40=0.000` | No | Full failure. |
| `Fixed RoPE` run 2 (`d0ced92d...`) | RoPE every loop | fixed-loop | N/A | `L20=0.197`, `L21≈0` | `L30=0.000`, `L40=0.000` | No | RoPE is catastrophic for copy. |
| `Ponder RoPE, beta=0.01` (`19a0316b...`) | RoPE every loop | PonderNet | Yes | `L20=0.000`, `L21=0.000` | `L30=0.000`, `L40=0.000` | Converges to a wrong fixed point | Failure even before extrapolation. |
| `Ponder RoPE, beta=0` (`9da97db5...`) | RoPE every loop | PonderNet | Yes | `L20=0.000`, `L21=0.000` | `L30=0.000`, `L40=0.000` | Immediate shallow collapse | Worst case: avg exit loops = `1.0` everywhere. |
| `Ponder NoPE` (`dc7358c6...`) | none | PonderNet | Yes | best fixed-loop: `L20=1.00`, `L21=1.00`; auto-exit: `L20=1.00`, `L21=0.998` | best fixed-loop: `L30=0.990 @ 6`, `L40=0.473 @ 15`; auto-exit: `L30=0.977`, `L40=0.453` | Yes | Strong final NoPE result. Ponder can learn copy well in the clean setting, with efficient stopping and meaningful extrapolation. |
| `Ponder WPE once` (`d44ab17a...`) | WPE only at loop 1 | PonderNet | Yes | best fixed-loop: `L20=0.518`, `L21=0.150`; auto-exit: `L20=0.475`, `L21=0.131` | best fixed-loop: `L30=0.023`, `L40=0.000`; auto-exit: `L30=0.010`, `L40=0.000` | Yes, but to a bad solution | Final result remains much better than RoPE+Ponder, but far worse than NoPE+Ponder and fixed-loop WPE-once. |

### Ponder Auto-Exit Numbers

| Setting | `L20` auto-exit | `L30` auto-exit | `L40` auto-exit |
|---|---|---|---|
| `Ponder RoPE, beta=0.01` | `acc=0.000`, `avg_loops=7.14` | `acc=0.000`, `avg_loops=6.07` | `acc=0.000`, `avg_loops=7.02` |
| `Ponder RoPE, beta=0` | `acc=0.000`, `avg_loops=1.00` | `acc=0.000`, `avg_loops=1.00` | `acc=0.000`, `avg_loops=1.00` |
| `Ponder NoPE` | `acc=1.000`, `avg_loops=6.60` | `acc=0.977`, `avg_loops=7.49` | `acc=0.453`, `avg_loops=8.76` |
| `Ponder WPE once` | `acc=0.475`, `avg_loops=9.13` | `acc=0.010`, `avg_loops=11.66` | `acc=0.000`, `avg_loops=14.13` |

### What the Hidden-State Metrics Say

- `Fixed NoPE`:
  - `L30` genuinely converges after solving. At the tail (`loop 40`), `delta_l2_norm_mask=0.0033`, `cosine_to_prev_mask=0.999998`, `answer_change_rate=0.0010`.
  - `L40` also stabilizes, but around a wrong attractor. That is why accuracy plateaus near `0.20`.
- `Fixed WPE once` (best run):
  - Strong convergence and strong extrapolation. At `L40`, the best loop is `32`, and by `loop 40` the answer accuracy stays `0.808` with `cosine_to_prev_mask=0.999976`.
- `Fixed WPE all/plain`:
  - No useful convergence. At `L40`, tail `delta_l2_norm_mask=1.63`, `cosine_to_prev_mask=0.9868`, `answer_change_rate=0.5225`.
- `Fixed RoPE`:
  - Clearly unstable/churning. At `L40`, tail `delta_l2_norm_mask=8.02`, `cosine_to_prev_mask=0.8436`, `answer_change_rate=1.0`.
- `Ponder NoPE`:
  - The final run is stronger and more efficient than the earlier inflight snapshot. It now peaks at `loop 3` for `L20/21`, `loop 6` for `L30`, and `loop 15` for `L40`.
  - The hidden states genuinely settle: by `loop 40`, `delta_l2_norm_mask` is effectively zero for `L20/21/30/40`.
  - At `L40`, it still stabilizes around a partially wrong state (`tail answer_accuracy = 0.414`), but this is a real convergent partial solution, not shallow collapse.
- `Ponder WPE once`:
  - The hidden states also settle, but much later and to a bad solution. At `L20`, the best loop is `34`; at `L30`, the best point is only `0.023` even by `loop 40`.
  - This makes the problem even less consistent with an “exits too early” story.
- `Ponder RoPE`:
  - With `beta=0.01`, the model collapses to a wrong fixed point.
  - With `beta=0`, it halts immediately (`avg_loops=1.0`) and is the cleanest shallow-failure case.

### Copy-Specific Extra Notes

#### Input Distribution Sensitivity of `Fixed NoPE`

All of the following are the **same checkpoint** (`8fe7447f...`), evaluated under different Bernoulli input distributions:

| Eval distribution | `L20` | `L30` | `L40` | Read |
|---|---:|---:|---:|---|
| `prob_one=0.0` | `1.000` | `1.000` | `1.000` | Trivial degenerate setting. |
| `prob_one=0.1` | `1.000` | `0.962` | `0.596` | Easier than balanced copy. |
| `prob_one=0.5` | `1.000` | `0.995` | `0.203` | Main balanced setting; hardest at `L40`. |
| `prob_one=0.8` | `1.000` | `0.971` | `0.514` | Also easier than balanced copy. |

This is important: `copy` conclusions depend on the input distribution. The balanced case (`0.5`) is the fairest main result.

#### Position-by-Position Behavior

`eval/copy/.../pos_acc_eval_0.5/copy_position_accuracy.csv` shows a clear diagonal-style iterative pattern for the NoPE fixed-loop model:

- position `1` is already perfect at `loop 1`
- position `2` reaches `1.0` by `loop 2`
- position `10` reaches `1.0` by `loop 8`

So the main NoPE copy baseline really does look like an iterative copier, not a one-shot memorizer.

#### The Current Copy Story

- `RoPE` is genuinely toxic for `copy`, with or without `PonderNet`.
- `NoPE + Ponder` means the old strong claim "`PonderNet` inherently breaks copy" is no longer sustainable.
- `WPE once + Ponder` does **not** recover the strong historical `WPE once + fixed-loop` result.
- The current cleanest claim is narrower:
  - fixed-loop supervision is still better at reliably producing the strongest long-length copy solutions
  - `PonderNet` can work on copy in a clean NoPE setting, but is more fragile under positional shortcuts
  - `WPE once + Ponder` fails despite using deeper computation than `NoPE + Ponder`, so its weakness is not reducible to premature stopping

#### NoPE vs WPE-once Stopping Distributions under Ponder

The most important qualitative point is that `WPE once + Ponder` does **not** fail because it always exits at step 1. In fact, its final stop distribution is substantially *later* than the NoPE run:

| Setting | `L20` peak / median | `L30` peak / median | `L40` peak / median |
|---|---|---|---|
| `Ponder NoPE` | `3 / 5` | `4 / 6` | `5 / 8` |
| `Ponder WPE once` | `6 / 8` | `9 / 11` | `11 / 13` |

So the current evidence points to:

- `RoPE + Ponder`: genuine shallow-collapse / bad-dynamics failure
- `WPE once + Ponder`: deeper computation is being used, but the learned representation is still wrong on OOD copy
- `NoPE + Ponder`: the final trained model is both stronger and more compute-efficient than the `WPE once` variant

## Parity

### Main Settings

| Setting | PE handling | Objective | Adaptive exit | ID fit | Extrapolation | Hidden convergence | Main read |
|---|---|---|---|---|---|---|---|
| `Fixed NoPE` (`d9de8dd7...`) | none | fixed-loop | N/A | `L20=1.00`, `L21=1.00` | `L30=1.000 @ 30`, `L40=0.99975 @ 40` | No | Strongest clean parity baseline. |
| `Fixed RoPE` (`938b8082...`) | RoPE every loop | fixed-loop | N/A | `L20=1.00`, `L21=1.00` | `L30=1.000 @ 30`, `L40=0.99675 @ 40` | Partial | Very strong, slightly below NoPE at long lengths. |
| `Fixed WPE` (`8622e6ac...`) | learned absolute WPE every loop | fixed-loop | N/A | `L20=1.00`, `L21=1.00` | `L30=1.000 @ 30`, `L40=0.99525 @ 40` | No | Also strong; parity is insensitive to PE type at the main accuracy level. |
| `Random-loop` original (`8a140a9c...`) | none | random-loop supervision | N/A | `L20≈0.661`, `L21≈0.628` | `L30≈0.518`, `L40≈0.512` | Weak | Poor even in-distribution. |
| `Random-loop 2-20` (`e34c6302...`) | none | random-loop supervision | N/A | `L20≈1.000`, `L21≈0.998` | `L30≈0.742`, `L40≈0.550` | Partial | Better than the original random-loop run, but clearly worse than fixed-loop on OOD. |
| `Substep supervision` (`f025f4a4...` / eval under `4b0b619d...`) | none | dense substep supervision | N/A | `L20=1.00`, `L21=1.00` | `L30=0.99575`, `L40=0.84875` | Partial / no robust convergence | Strong ID, weaker long extrapolation than main fixed-loop baseline. |
| `Ponder RoPE, first-answer halt` (`86494c70...`) | RoPE every loop | PonderNet | Yes | `L20=1.00`, `L21≈1.00` | best fixed-loop: `L30=0.9965`, `L40=0.9925`; auto-exit: `L30=0.991`, `L40=0.992` | Yes | Best parity adaptive-compute result in repo. |
| `Ponder RoPE, mean-pooling halt` (`1097b7c6...`) | RoPE every loop | PonderNet | Yes | `L20=1.00`, but `L21=0.465` | `L30=0.494`, `L40=0.551` | Yes, but to a mediocre state | Clear negative result for mean-pooling halting. |

### Ponder Auto-Exit Numbers

| Setting | `L20` auto-exit | `L30` auto-exit | `L40` auto-exit |
|---|---|---|---|
| `Ponder RoPE, first-answer halt` | `acc=0.999`, `avg_loops=9.35` | `acc=0.991`, `avg_loops=12.28` | `acc=0.992`, `avg_loops=14.62` |
| `Ponder RoPE, mean-pooling halt` | `acc=0.946`, `avg_loops=4.98` | `acc=0.490`, `avg_loops=4.39` | `acc=0.399`, `avg_loops=4.00` |

### What the Hidden-State Metrics Say

- `Fixed NoPE`:
  - High accuracy does **not** mean hidden convergence. At `L30`, the model is perfect at `loop 30`, but falls to `0.529` by `loop 40`.
  - This is loop-count-specific computation, not adaptive convergence.
- `Fixed WPE`:
  - Same story as NoPE. Strong parity accuracy, but not a convergent hidden-state dynamic.
- `Fixed RoPE`:
  - Somewhat more stable than NoPE/WPE, but still not a fully convergent adaptive solver. Over-running still hurts.
- `Random-loop 2-20`:
  - Settles somewhat (`cosine_to_prev_mask≈0.994` by `loop 40`) but to a mediocre OOD solution.
- `Substep supervision`:
  - Better than random-loop, but still degrades when overrun. It does not reproduce the clean fixed-loop story at long lengths.
- `Ponder RoPE, first-answer halt`:
  - This is the clean parity convergence case. By `loop 40`, `delta_l2_norm_mask≈0`, `cosine_to_prev_mask≈1.0`, `answer_change_rate=0.0`, while accuracy remains `≈0.992`.
- `Ponder RoPE, mean-pooling halt`:
  - Also converges, but to the wrong / mediocre decision regime. So convergence alone is not enough; the halt head choice matters.

### Parity-Specific Extra Notes

#### Parity Is Robust to Position Handling at the Main Accuracy Level

For the three main fixed-loop parity baselines:

- `NoPE`: `L40=0.99975`
- `RoPE`: `L40=0.99675`
- `WPE`: `L40=0.99525`

So parity does **not** show the sharp positional fragility that copy shows.

#### WPE Permutation Checks on Parity

The repo also contains WPE permutation interventions on the same parity WPE checkpoint:

- `dense_eval_wpe_permutate_ape_fixed`: `L40≈0.9975`
- `dense_eval_wpe_permutate_ape_resample`: `L40≈0.9980`

This is consistent with the general story that parity does not depend strongly on a precise absolute positional code.

## Bottom Line

### Copy

- The best fixed-loop result is still the historical `WPE once` run (`289ac6be...`).
- `RoPE` is bad for copy in both fixed-loop and `PonderNet`.
- `NoPE + Ponder` shows that copy failure is **not** inherent to `PonderNet`.
- `NoPE + Ponder` final results are strong: near-perfect through `L30`, with partial but meaningful success at `L40`.
- `WPE once + Ponder` is much worse than both `NoPE + Ponder` and historical `WPE once + fixed-loop`.
- The strongest safe claim is now:
  - `PonderNet` on copy is highly sensitive to positional handling and is less reliable than fixed-loop supervision for obtaining the strongest long-length solutions.

### Parity

- Fixed-loop parity is strong under `NoPE`, `WPE`, and `RoPE`.
- The parity-specific loop-supervision ablations (`random-loop`, `substep`) are weaker than the main fixed-loop baseline, especially on OOD.
- `PonderNet` works well on parity **when the halt head reads the first answer token**.
- `Mean pooling` for halting is a clear negative result.
- Parity success does not rely on hidden-state convergence in fixed-loop training; the convergent story appears much more cleanly in the `Ponder` first-answer setup.

## Source Runs / Eval Roots

### Copy

- Fixed NoPE: `models/copy/8fe7447f-9d64-4de4-bf04-29c50e860cd6`
- Fixed RoPE: `models/copy/rope/8f1e9a81-...`, `models/copy/rope/d0ced92d-...`
- Fixed WPE once: `models/copy/wpe/1a53af55-...`, `models/copy/wpe/289ac6be-...`
- Fixed WPE all/plain: `models/copy/wpe/2dbede54-...`, `models/copy/wpe/2dbede54-..._add_all`
- Ponder RoPE: `models/copy_ponder/19a0316b-...`, `models/copy_ponder/9da97db5-...`
- Ponder NoPE final: `models/nope_baselines/copy_ponder/dc7358c6-...`
- Ponder WPE once final: `models/wpe_once_baselines/copy_ponder/d44ab17a-...`

### Parity

- Fixed NoPE: `models/parity/d9de8dd7-...`
- Fixed RoPE: `models/parity/rope/938b8082-...`
- Fixed WPE: `models/parity/wpe/8622e6ac-...`
- Random-loop: `models/parity/8a140a9c-...`, `models/parity/e34c6302-...`
- Substep: `models/parity/f025f4a4-...` with eval under `eval/parity/4b0b619d-...`
- Ponder RoPE first-answer: `models/parity_ponder/86494c70-...`
- Ponder RoPE mean-pooling: `models/parity_ponder/1097b7c6-...-mean_pooling`

## Accuracy Tables

Values below are rounded summaries for quick comparison.

- `copy` uses `answer_accuracy`
- `parity` uses `accuracy`
- fixed-loop tables use the **best accuracy over dense loop sweep**
- `PonderNet` is shown with both **best-loop** and **auto-exit** accuracy

### Copy: Fixed-Loop Best Accuracy

| Setting | `L20` | `L21` | `L30` | `L40` |
|---|---:|---:|---:|---:|
| `NoPE` (`prob_one=0.5`) | 1.000 | 1.000 | 0.995 | 0.203 |
| `WPE all/plain` | 0.579 | 0.490 | 0.000 | 0.000 |
| `WPE once` run 1 | 0.233 | 0.929 | 0.000 | 0.000 |
| `WPE once` run 2 | 1.000 | 1.000 | 0.998 | 0.808 |
| `RoPE` run 1 | 0.002 | 0.000 | 0.000 | 0.000 |
| `RoPE` run 2 | 0.197 | 0.000 | 0.000 | 0.000 |

### Copy: PonderNet Best-Loop Accuracy

| Setting | `L20` | `L21` | `L30` | `L40` |
|---|---:|---:|---:|---:|
| `RoPE`, `beta=0.01` | 0.000 | 0.000 | 0.000 | 0.000 |
| `RoPE`, `beta=0.0` | 0.000 | 0.000 | 0.000 | 0.000 |
| `NoPE` | 1.000 | 1.000 | 0.990 | 0.473 |
| `WPE once` | 0.518 | 0.150 | 0.023 | 0.000 |

### Copy: PonderNet Auto-Exit Accuracy

| Setting | `L20` | `L21` | `L30` | `L40` |
|---|---:|---:|---:|---:|
| `RoPE`, `beta=0.01` | 0.000 | 0.000 | 0.000 | 0.000 |
| `RoPE`, `beta=0.0` | 0.000 | 0.000 | 0.000 | 0.000 |
| `NoPE` | 1.000 | 0.998 | 0.977 | 0.453 |
| `WPE once` | 0.475 | 0.131 | 0.010 | 0.000 |

### Parity: Fixed-Loop Best Accuracy

| Setting | `L20` | `L21` | `L30` | `L40` |
|---|---:|---:|---:|---:|
| `NoPE` | 1.000 | 1.000 | 1.000 | 1.000 |
| `RoPE` | 1.000 | 1.000 | 1.000 | 0.997 |
| `WPE` | 1.000 | 1.000 | 1.000 | 0.995 |
| `Random-loop` original | 0.661 | 0.628 | 0.518 | 0.512 |
| `Random-loop 2-20` | 1.000 | 0.998 | 0.742 | 0.550 |
| `Substep supervision` | 1.000 | 1.000 | 0.996 | 0.849 |

### Parity: PonderNet Best-Loop Accuracy

| Setting | `L20` | `L21` | `L30` | `L40` |
|---|---:|---:|---:|---:|
| `RoPE`, first-answer halt | 1.000 | 1.000 | 0.997 | 0.993 |
| `RoPE`, mean-pooling halt | 1.000 | 0.465 | 0.494 | 0.550 |

### Parity: PonderNet Auto-Exit Accuracy

| Setting | `L20` | `L21` | `L30` | `L40` |
|---|---:|---:|---:|---:|
| `RoPE`, first-answer halt | 0.999 | 0.999 | 0.991 | 0.992 |
| `RoPE`, mean-pooling halt | 0.946 | 0.384 | 0.490 | 0.399 |
