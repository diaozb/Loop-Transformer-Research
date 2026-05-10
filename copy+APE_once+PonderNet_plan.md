# Experiment: Copy + APE(init only) + PonderNet vs Fixed Loop

## Goal

We need to test whether PonderNet's adaptive halting objective traps a looped Transformer in a shallow (O(1)) solution for the Copy task when positional encoding is available, while fixed-loop training avoids this trap and learns a deep iterative algorithm that extrapolates.

## Background

We have an existing looped Transformer codebase that supports:
- Tasks: Copy, Parity (and others)
- PE options: NoPE, RoPE, APE
- Training objectives: fixed-loop (decode at step T = length), PonderNet (adaptive halting)
- Curriculum training: randomly sample lengths ≤ current_length
- Evaluation: accuracy heatmaps (length × loop), cosine similarity heatmaps, delta L2 norm heatmaps

Previous results show:
- Copy + NoPE + fixed loop → extrapolates well (iterative one-step-per-loop copying)
- Copy + NoPE + PonderNet → extrapolates well  
- Copy + RoPE + PonderNet → fails to extrapolate
- Copy + RoPE + fixed loop → fails to extrapolate (RoPE corrupts dynamics)
- Copy + APE(init only) + fixed loop → extrapolates well with adaptive convergence
- Parity + any PE + PonderNet → extrapolates well

## What to implement

### Configuration 1: Copy + Sinusoidal APE (init only) + PonderNet
### Configuration 2: Copy + Sinusoidal APE (init only) + Fixed Loop (baseline, may already exist)
### Configuration 3: Parity + Sinusoidal APE (init only) + PonderNet (control)

### PE specification: "APE init only" with sinusoidal encoding

Use fixed (non-learned) sinusoidal positional encoding, added ONLY at initialization (before the first loop iteration), NOT re-injected at each loop step.

```python
# Sinusoidal PE (standard Vaswani et al. formulation)
def sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (seq_len, d_model)
```

**Critical implementation detail**: The PE is added to the initial embedding X0 before the first loop. During loop iterations, the update rule is:

```
H(0) = X0 + PE   # PE added here ONLY
H(t+1) = H(t) + X0 + g_theta(H(t) + X0)   # NO PE re-injection in X0 here
```

Wait — check the existing code carefully. In the current architecture (Eq. 1 of the theory paper), the update is `H(t+1) = H(t) + X0 + g_theta(H(t) + X0)`. If PE is part of X0, then X0 is re-added every step, which means PE IS re-injected every step through the X0 term.

**There are two interpretations of "APE init only":**

**Option A**: PE is added to the initial hidden state H(0) only, and X0 does NOT contain PE:
```
H(0) = X0 + PE
H(t+1) = H(t) + X0_no_pe + g_theta(H(t) + X0_no_pe)
```
Here the PE information persists only through the hidden state evolution.

**Option B**: PE is part of the input embedding but the static context X0 that gets re-added each step does NOT include PE:
```
X0_with_pe = embedding(input) + PE     # used only for H(0)
X0_no_pe = embedding(input)            # used in the recurrence
H(0) = X0_with_pe
H(t+1) = H(t) + X0_no_pe + g_theta(H(t) + X0_no_pe)
```

**Check the existing APE(init only) implementation to see which was used in the successful Copy + APE(init only) + fixed loop experiment, and use the same version here.** This is critical for comparability. If unclear, use Option A (PE added to H(0) only).

### PonderNet specification

Use the same PonderNet implementation as in the existing Copy + RoPE + PonderNet and Parity + PonderNet experiments. Specifically:

- Halting head: attached to the first answer position's hidden state (based on Document 2, this worked better than mean pooling over mask)
- Hazard rate: λ_t = sigmoid(halt_head(h_t))
- Halting probability: p_t = λ_t * prod_{j<t} (1 - λ_j)
- Loss: L = sum_t p_t * ℓ_t + β * KL(p || p_G(λ_p))
- Use the same β and λ_p as in previous PonderNet experiments

### Training specification

- Train on Copy task, lengths 2-20 (same as Document 4)
- Curriculum strategy: same as default (randomly sample ≤ current_length)
- Max loop steps: 40 (same as previous experiments)
- All other hyperparameters (lr, batch size, optimizer, etc.): same as previous experiments
- Run at least 3 random seeds

### Evaluation specification

Evaluate on lengths 2-40 (in-distribution: 2-20, extrapolation: 21-40).

**Standard diagnostics (same as previous experiments):**
1. Accuracy heatmap (length × loop count) — evaluate at every loop step
2. Cosine to previous mask heatmap (length × loop count)  
3. Delta L2 norm heatmap (length × loop count)
4. Auto-exit accuracy vs length curve
5. Auto-exit average loop count vs length curve

**New diagnostics specific to this experiment (CRITICAL — these test the theory):**

6. **Per-step loss curve**: For each test length L, compute ℓ_t(θ) for t = 1, 2, ..., 40.
   - Specifically record ℓ_1 for both training lengths and test lengths
   - Plot ℓ_1 vs length. If ℓ_1 ≈ 0 for training lengths but ℓ_1 ≫ 0 for test lengths, the shallow solution is length-specific
   - Plot ℓ_t vs t for several representative lengths (e.g., L = 5, 10, 15, 20, 25, 30, 35, 40)

7. **Halting distribution**: For each test length L, plot the halting probability distribution p*_t over t.
   - If p*_1 ≈ 1 for all lengths → trapping confirmed
   - If p*_t shifts rightward with L → model is adapting depth (no trapping)
   - Plot mean halting step E[T] = sum_t t * p_t vs length

8. **Override-halting evaluation**: At test time, IGNORE the halting head. Force the model to run for T_override = L_test steps and evaluate accuracy at that step.
   - This tests whether the deep iterative algorithm was learned at all
   - If accuracy is low even with forced deep computation → PonderNet prevented the deep algorithm from developing (gradient starvation from Theorem 2.5)
   - Compare this with Copy + APE(init only) + fixed loop evaluated at the same step

9. **Per-position accuracy at each loop step**: For a fixed test length (e.g., L = 20 and L = 30), plot a heatmap of (output position k) × (loop step t) showing per-position accuracy.
   - Under fixed-loop: expect diagonal pattern (one position solved per step)
   - Under PonderNet (if trapped): expect either all-correct-at-step-1 (for training lengths) or all-wrong (for test lengths), no diagonal structure

10. **Attention pattern visualization** (optional but informative): At step 1, visualize the cross-attention weights from target positions to source positions.
    - If the model learned a position-matching pattern, you should see a shifted diagonal
    - Check whether this pattern is correct for training lengths and breaks for test lengths

### Output format

Save all plots to a results directory. For each configuration, save:
```
results/
  copy_ape_init_pondernet_seed{i}/
    accuracy_heatmap.png
    cosine_heatmap.png  
    delta_l2_heatmap.png
    auto_exit_accuracy_vs_length.png
    auto_exit_loops_vs_length.png
    per_step_loss_vs_t.png           # NEW: multiple curves, one per test length
    step1_loss_vs_length.png         # NEW: ℓ_1 vs length
    halting_distribution.png         # NEW: p_t distributions for several lengths
    mean_halting_step_vs_length.png  # NEW: E[T] vs length
    override_accuracy_vs_length.png  # NEW: accuracy when halting is overridden
    per_position_accuracy_L20.png    # NEW: position × step heatmap
    per_position_accuracy_L30.png    # NEW: position × step heatmap
    metrics.json                     # all numerical results
```

Also produce a **summary comparison table** across configurations:

| Config | Train Acc | Extrap Acc (L=30) | Extrap Acc (L=40) | Mean halt step (L=20) | Mean halt step (L=30) | ℓ_1 (L=20) | ℓ_1 (L=30) | Override Acc (L=30) |
|--------|-----------|-------------------|-------------------|-----------------------|-----------------------|-------------|-------------|---------------------|

### Implementation priority

1. First verify that the existing Copy + APE(init only) + fixed loop result reproduces
2. Then implement Copy + APE(init only) + PonderNet (the critical experiment)
3. Then implement Parity + APE(init only) + PonderNet (the control)
4. Collect all diagnostics
5. Generate summary table

### What NOT to change

- Do not modify the base looped Transformer architecture
- Do not modify the Copy or Parity task definitions
- Do not modify the PonderNet loss computation (use exactly the same code as previous experiments)
- Do not change optimizer, learning rate schedule, batch size, or training duration
- The ONLY change from the existing PonderNet experiments is: replace RoPE with sinusoidal APE (init only)
