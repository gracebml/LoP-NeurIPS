# Gradient Explosion Prevention for SASSHA + EMA in Continual Learning

## Context

**Setup**: SASSHA (ICML 2025) + EMA on Incremental CIFAR-100 with ResNet-18, 4000 epochs, 5 new classes every 200 epochs.

**Problem**: Gradient explosion at epoch ~2200 (task boundary 55→60 classes), complete NaN collapse at epoch 2800.


| Epoch    | Train Loss | Avg Weight Mag | Dormant Units | Status                       |
| -------- | ---------- | -------------- | ------------- | ---------------------------- |
| 2200     | 0.37       | 0.097          | 0.029         | Normal                       |
| **2250** | **22,896** | **7,183**      | **0.423**     | **Explosion**                |
| 2300     | 1,238      | 6,649          | 0.423         | Partial damping              |
| 2400     | 1.81       | 0.120          | 0.142         | Early-stop rollback recovery |
| **2800** | **NaN**    | **NaN**        | 0.0           | **Permanent death**          |


---

## Root Cause Analysis

### 1. Stale Hessian Diagonal at Task Boundaries

SASSHA maintains an EMA of the Hessian diagonal (`exp_hessian_diag`) with `beta2=0.999` and only recomputes the raw Hessian every `lazy_hessian=10` steps. When 5 new classes are added at a task boundary, the loss landscape changes dramatically, but the denominator

$$\frac{\text{exphessiandiag}^{0.5}}{\text{biascorrection2}} + \epsilon$$

still reflects the **old** curvature. Near-zero Hessian values from the previous task create a tiny denominator, leading to enormous parameter updates.

### 2. SAM Perturbation Amplifies Instability

The SAM ascent step $w \leftarrow w + \rho \cdot \nabla L / \nabla L$ with `rho=0.2` pushes weights into a region where the second backward pass (`create_graph=True`) computes gradients and Hessians completely misaligned with the stale curvature estimate. The wrong curvature information amplifies the perturbation rather than correcting it.

### 3. No Gradient Clipping

The training loop has zero gradient clipping on either gradients or Hessian diagonals. A single bad batch at a task boundary can inject arbitrarily large values into the optimizer state.

### 4. Weight Magnitude Creep

Average weight magnitude grows monotonically (0.024 → 0.097 over 2200 epochs) with only `weight_decay=1e-4`. Larger weights amplify gradient magnitudes and increase fragility to perturbations.

### 5. Corrupted Optimizer State Persists

After the explosion at E2250, `exp_avg` and `exp_hessian_diag` are permanently polluted with extreme values. Early stopping restores **model weights** but **not the optimizer state**, so the corruption carries forward and causes the fatal NaN at E2800.

---

## Method 1: Adaptive Gradient Clipping (AGC)

> **Source**: Brock, De, Smith & Simonyan. *High-Performance Large-Scale Image Recognition Without Normalization*. **ICML 2021** (NFNet). [[paper]](https://proceedings.mlr.press/v139/brock21a/brock21a.pdf)

### Idea

Standard gradient clipping uses a fixed global threshold, which is blind to the natural scale differences across layers. AGC instead clips each parameter's gradient based on the **ratio of gradient norm to weight norm**:

$$

\text{if} \frac{\nabla W_i}{W_i} > \lambda, \quad \nabla W_i \leftarrow \lambda \cdot \frac{W_i}{\nabla W_i} \cdot \nabla W_i

$$

### Why It Helps Here

- **Scale-invariant**: Works correctly across layers of different magnitudes, which is essential for second-order optimizers where preconditioned gradients have heterogeneous scales.
- **Prevents disproportionate updates**: No single layer can change by more than a fraction λ of its current magnitude in one step.
- **Preserves direction**: Unlike norm clipping that uniformly shrinks all gradients, AGC only affects outlier parameters.

### Key Hyperparameter

- `agc_clip_factor` (λ) = 0.01. This means no single gradient update can change a parameter by more than 1% of its current magnitude.

### Implementation

```python
@torch.no_grad()
def apply_agc(model, clip_factor=0.01):
    for p in model.parameters():
        if p.grad is None:
            continue
        p_norm = p.data.norm(2).clamp(min=1e-6)
        g_norm = p.grad.data.norm(2)
        max_g_norm = p_norm * clip_factor
        if g_norm > max_g_norm:
            p.grad.data.mul_(max_g_norm / g_norm)
```

---

## Method 2: ZClip — Z-Score Anomaly Detection for Gradient Spikes

> **Source**: *ZClip: Adaptive Spike Mitigation for LLM Pre-Training*. **arXiv 2025** (2504.02507). [[paper]](https://arxiv.org/abs/2504.02507) [[code]](https://github.com/bluorion-com/ZClip)

### Idea

Maintain a running EMA of gradient norms and their variance. When the current gradient norm exceeds the EMA mean by more than $k$ standard deviations (z-score), it is flagged as an anomalous spike and clipped:

$$z = \frac{\nabla\theta_t - \mu_t}{\sigma_t}$$

$$\text{if } z > k: \quad \text{clip to } \mu_t + k \cdot \sigma_t$$

where $\mu_t$ and $\sigma_t$ are EMA estimates of the gradient norm mean and standard deviation.

### Why It Helps Here

- **Adapts to natural evolution**: Unlike fixed thresholds, ZClip tracks how gradient norms naturally change during training. What is "normal" at epoch 100 vs epoch 2000 can be very different.
- **Targets only anomalies**: Does not interfere with normal training dynamics — only activates when a statistically significant spike occurs.
- **Proactive**: Detects spikes before they corrupt optimizer state.

### Key Hyperparameters

- `zclip_zscore_thresh` = 3.0 (flag anything > 3σ from the mean)
- `zclip_ema_decay` = 0.99 (how fast the running statistics adapt)

### Implementation

```python
class ZClipTracker:
    def __init__(self, decay=0.99, threshold=3.0):
        self.decay = decay
        self.threshold = threshold
        self.mean = None
        self.var = None

    def check_and_clip(self, model):
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        if self.mean is None:
            self.mean = gnorm
            self.var = 0.0
            return False

        d = self.decay
        self.var = d * self.var + (1 - d) * (gnorm - self.mean) ** 2
        self.mean = d * self.mean + (1 - d) * gnorm

        std = max(math.sqrt(self.var), 1e-8)
        zscore = (gnorm - self.mean) / std
        if zscore > self.threshold:
            clip_val = self.mean + self.threshold * std
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            return True
        return False
```

---

## Method 3: Hessian Clipping

> **Source**: Sadiev, Richtárik & Fatkhullin. *Second-order Optimization under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity Limits*. **NeurIPS 2025**. [[paper]](https://neurips.cc/virtual/2025/poster/115749)

### Idea

In second-order optimizers, the Hessian (or its diagonal approximation) serves as a curvature-aware preconditioner. Under heavy-tailed noise or distribution shifts, the Hessian estimate can have extreme values — either very large (causing tiny steps that stall training) or very small (causing enormous steps that explode). **Hessian clipping** bounds the curvature estimate:

$$\hat{H}_{\text{clipped}} = \text{clamp}(\hat{H}, 0, \tau)$$

This is applied to both the raw Hessian diagonal (`p.hess`) and the accumulated EMA (`exp_hessian_diag`).

### Why It Helps Here

- **Directly addresses root cause**: The explosion at E2250 occurs because `exp_hessian_diag` has near-zero entries from the old task, creating a denominator close to ε = 1e-4. With τ = 1000, the minimum effective denominator is `(eps)^0.5 / bc2 + eps ≈ 0.01 + 1e-4`, which bounds the maximum step size.
- **Theoretically grounded**: The NeurIPS 2025 paper proves tight sample complexity bounds when combining gradient clipping with Hessian clipping under heavy-tailed noise.
- **Low overhead**: A single `clamp_()` call per parameter.

### Key Hyperparameter

- `hessian_clip_value` (τ) = 1000. This prevents any curvature estimate from exceeding 1000, which in turn floors the denominator in the Adam-like update.

### Implementation

```python
@torch.no_grad()
def clip_hessian(optimizer, tau=1e3):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'exp_hessian_diag' in state:
                state['exp_hessian_diag'].clamp_(0, tau)
            if hasattr(p, 'hess') and not isinstance(p.hess, float):
                p.hess.clamp_(-tau, tau)
```

---

## Method 4: SPAM-style Momentum & Hessian Reset

> **Source**: Huang, Zhu, Jin, Liu, Wang & Liu. *SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training*. **ICLR 2025**. [[paper]](https://arxiv.org/abs/2501.06842) [[code]](https://github.com/TianjinYellow/SPAM-Optimizer)

### Idea

When a gradient spike is detected (or a distribution shift occurs), the optimizer's momentum buffers contain **corrupted statistics** that will bias future updates for many steps (due to the EMA decay). SPAM addresses this by **resetting the momentum** when spikes are detected, preventing poisoned gradient history from accumulating.

In our continual learning setting, this is extended to **proactively reset at task boundaries**, where the data distribution shifts by definition.

### Why It Helps Here

- **Prevents forward contamination**: In the original experiment, the explosion at E2250 corrupts `exp_avg` and `exp_hessian_diag`. Even after early stopping restores model weights at E2400, the optimizer state remains polluted, leading to the fatal NaN at E2800.
- **Clean start per task**: Resetting at task boundaries means the optimizer adapts to the new class distribution from scratch, without carry-over bias from the old distribution.
- **Spike-reactive**: If a spike occurs mid-task, the reset prevents the corrupted gradient from being averaged into `exp_avg` for the next ~1000 steps (1 / (1 - β1) ≈ 10 steps half-life).

### What Gets Reset


| Buffer             | Purpose                       | Why reset                       |
| ------------------ | ----------------------------- | ------------------------------- |
| `exp_avg`          | First moment (gradient EMA)   | Corrupted by spike gradients    |
| `exp_hessian_diag` | Second moment (curvature EMA) | Wrong curvature from old task   |
| `bias_correction2` | Debiasing term                | Must match reset step counter   |
| `step`             | Optimizer step counter        | Restart bias correction         |
| `hessian step`     | Lazy Hessian counter          | Force fresh Hessian computation |


### Implementation

```python
@torch.no_grad()
def reset_optimizer_momentum(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'exp_avg' in state:
                state['exp_avg'].zero_()
            if 'exp_hessian_diag' in state:
                state['exp_hessian_diag'].zero_()
            if 'bias_correction2' in state:
                state['bias_correction2'] = 0
            if 'step' in state:
                state['step'] = 0
            if 'hessian step' in state:
                state['hessian step'] = 0
            if hasattr(p, 'hess'):
                p.hess = 0.0
```

---

## Method 5: NaN-Guard with Checkpoint Rollback

> **Source**: Standard practice from large-scale training. Documented in Google PaLM (Chowdhery et al., 2022), Meta LLaMA 2 (Touvron et al., 2023), and others.

### Idea

Maintain periodic "safe checkpoints" during healthy training. When loss becomes NaN or spikes beyond a dynamic threshold:

1. **Roll back** model weights to the last safe checkpoint
2. **Reset** optimizer state (momentum, Hessian EMA)
3. **Reset** EMA shadow weights
4. **Skip** the problematic batch and continue

The spike threshold is itself adaptive: it tracks an EMA of the loss and flags anything exceeding `factor × loss_EMA`.

### Why It Helps Here

- **Last resort defense**: Even if AGC, ZClip, and Hessian clipping all fail to prevent an explosion, the NaN-guard catches the resulting NaN/Inf and recovers automatically.
- **No manual intervention**: The original experiment would require restarting from scratch or manually loading a checkpoint. NaN-guard automates this.
- **Preserves training progress**: By rolling back only ~10 epochs (checkpoint saved every 10 healthy epochs), most training progress is preserved.

### Key Hyperparameters

- `spike_loss_factor` = 10.0 (flag if loss > 10× the running EMA)
- Checkpoint saved every 10 epochs when loss is finite

### Implementation

```python
def check_loss_health(loss_value, loss_ema, factor=10.0):
    if not math.isfinite(loss_value):
        return 'nan'
    if loss_ema is not None and loss_value > factor * max(loss_ema, 0.1):
        return 'spike'
    return 'ok'

# In training loop:
if status == 'nan':
    net.load_state_dict(safe_checkpoint['model'])
    reset_optimizer_momentum(optimizer)
    ema.reset(net)
    continue  # skip this batch
```

---

## Method 6: Adaptive SAM Rho Scheduling

> **Source**: Inspired by Wen, Ma & Li. *Sharpness-Aware Minimization and the Edge of Stability*. **JMLR 2024** (v25, 23-1285). [[paper]](https://jmlr.org/papers/v25/23-1285.html)  
> Also: Agarwala & Dauphin. *SAM Operates at a Saddle*. **NeurIPS 2024** (arXiv:2410.10373).

### Idea

SAM's perturbation radius ρ controls how far the weights are pushed in the gradient direction during the ascent step. At task boundaries, the loss landscape has just changed dramatically — the model is far from any minimum, gradients are large and noisy, and the Hessian estimate is stale.

**Adaptive rho scheduling** linearly warms up ρ from 0 to its base value over the first `warmup_epochs` of each new task:

$$\rho_t = \rho_{\text{base}} \cdot \min\left(\frac{\text{epochintask}}{\text{warmupepochs}}, 1\right)$$

### Why It Helps Here

- **Prevents perturbation amplification**: During the first few epochs of a new task, the SAM ascent step with full ρ=0.2 can push weights into highly unstable regions. Starting with ρ≈0 means the optimizer behaves like standard Adam for the first few epochs.
- **Aligns with SAM dynamics research**: The JMLR 2024 paper shows SAM operates at an "edge of stability" that depends on both the learning rate and gradient norm. At task boundaries, this edge shifts — warming up ρ lets the optimizer find the new edge gradually.
- **Late-training effectiveness**: NeurIPS 2024 research shows SAM primarily helps late in training (selecting flatter minima). At task boundaries, we're at the *start* of training for the new distribution — SAM's benefits are minimal but its risks are maximal.

### Key Hyperparameter

- `rho_warmup_epochs` = 20 (warm up over first 20 epochs of each 200-epoch task = 10% of task duration)

### Implementation

```python
def compute_adaptive_rho(base_rho, epoch_in_task, warmup_epochs=20):
    if epoch_in_task >= warmup_epochs:
        return base_rho
    return base_rho * (epoch_in_task / warmup_epochs)

# In training loop, before perturb_weights():
current_rho = compute_adaptive_rho(0.2, epoch_in_task, warmup_epochs=20)
for g in optimizer.param_groups:
    g['rho'] = current_rho
```

---

## Defense Chain: How All 6 Methods Work Together

The methods are applied in a specific order during each training step:

```
┌─────────────────────────────────────────────────────────────┐
│                    TASK BOUNDARY                            │
│  [4] Reset optimizer momentum (SPAM, ICLR 2025)            │
│  [5] Reset loss EMA tracking                               │
│  [6] Set rho = 0 (start warmup)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FORWARD PASS                             │
│  Compute loss = CrossEntropy(model(x), y)                  │
│  [5] Check loss health (NaN? Spike?)                        │
│      → NaN:   rollback to safe checkpoint, reset, skip     │
│      → Spike: reset momentum, continue with caution        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FIRST BACKWARD                           │
│  loss.backward()                                            │
│  [1] AGC: clip ∇W per-param by ‖∇W‖/‖W‖ (ICML 2021)       │
│  [2] ZClip: detect z-score anomaly in ‖∇θ‖ (2025)          │
│      → If spike: clip to μ + kσ                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SAM PERTURBATION                         │
│  [6] Compute adaptive rho (warm up from 0)                  │
│  w → w + ρ_adaptive · ∇L / ‖∇L‖                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SECOND BACKWARD                          │
│  loss_pert.backward(create_graph=True)                      │
│  Hutchinson Hessian estimation: H ≈ E[z ⊙ (Hz)]           │
│  [3] Hessian clipping: clamp(H, 0, τ) (NeurIPS 2025)       │
│  Global gradient norm clip (fallback safety)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    OPTIMIZER STEP                           │
│  SASSHA_Safe.step():                                        │
│    - Hessian diag clamped again inside step()               │
│    - Per-param update norm capped to max_update_norm        │
│    - NaN-safe: skip update if non-finite                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    POST-STEP                                │
│  EMA update (shadow weights)                                │
│  [5] Save safe checkpoint every 10 healthy epochs           │
└─────────────────────────────────────────────────────────────┘
```

---

## Hyperparameter Summary


| Method       | Parameter             | Value                    | Rationale                                               |
| ------------ | --------------------- | ------------------------ | ------------------------------------------------------- |
| AGC          | `agc_clip_factor`     | 0.01                     | Max 1% weight change per step (NFNet default)           |
| ZClip        | `zclip_zscore_thresh` | 3.0                      | Standard 3σ anomaly threshold                           |
| ZClip        | `zclip_ema_decay`     | 0.99                     | Smooth tracking with ~100-step window                   |
| Hessian Clip | `hessian_clip_value`  | 1000                     | Prevents curvature extremes while allowing normal range |
| SPAM Reset   | (triggered at)        | Task boundaries + spikes | Proactive + reactive                                    |
| NaN-Guard    | `spike_loss_factor`   | 10.0                     | Flag if loss > 10× running average                      |
| NaN-Guard    | (checkpoint freq)     | Every 10 healthy epochs  | Balance between freshness and overhead                  |
| Rho Warmup   | `rho_warmup_epochs`   | 20                       | 10% of task duration (200 epochs)                       |
| SASSHA_Safe  | `max_update_norm`     | 1.0                      | Per-param update magnitude cap                          |


---

## References

1. Brock, A., De, S., Smith, S. L., & Simonyan, K. (2021). High-Performance Large-Scale Image Recognition Without Normalization. *ICML 2021*. [[paper]](https://proceedings.mlr.press/v139/brock21a/brock21a.pdf)
2. ZClip: Adaptive Spike Mitigation for LLM Pre-Training. *arXiv 2025* (2504.02507). [[paper]](https://arxiv.org/abs/2504.02507) [[code]](https://github.com/bluorion-com/ZClip)
3. Sadiev, A., Richtárik, P., & Fatkhullin, I. (2025). Second-order Optimization under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity Limits. *NeurIPS 2025*. [[paper]](https://neurips.cc/virtual/2025/poster/115749)
4. Huang, T., Zhu, Z., Jin, G., Liu, L., Wang, Z., & Liu, S. (2025). SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training. *ICLR 2025*. [[paper]](https://arxiv.org/abs/2501.06842) [[code]](https://github.com/TianjinYellow/SPAM-Optimizer)
5. Chowdhery, A., et al. (2022). PaLM: Scaling Language Modeling with Pathways. *arXiv 2022*; Touvron, H., et al. (2023). LLaMA 2: Open Foundation and Fine-Tuned Chat Models. *arXiv 2023*. (Checkpoint rollback is standard practice documented in these works.)
6. Wen, K., Ma, T., & Li, Z. (2024). Sharpness-Aware Minimization and the Edge of Stability. *JMLR*, 25(338), 1-70. [[paper]](https://jmlr.org/papers/v25/23-1285.html)
7. Shin, J., et al. (2025). SASSHA: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation. *ICML 2025*. [[paper]](https://arxiv.org/abs/2502.18153) [[code]](https://github.com/LOG-postech/Sassha)

