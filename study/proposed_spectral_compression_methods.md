# Proposed Spectral Compression Methods cho SASSHA + SDP

> Dựa trên tổng hợp nghiên cứu từ NeurIPS, ICML, ICLR (2024–2026)
> Ngày: 2026-03-14

---

## Baseline: SDP Hiện tại (Static γ)

$$\sigma'_i = \bar{\sigma}^\gamma \cdot \sigma_i^{(1-\gamma)}, \quad \gamma = 0.3 \text{ (cố định)}$$

**Hạn chế**:
- γ cố định → không phân biệt spike SVs vs tail SVs
- Không tận dụng curvature info từ SASSHA
- Không adapt theo mức độ non-stationarity của target distribution

---

## Method 1: Hessian-Guided Adaptive γ (HG-SDP)

> **Nguồn cảm hứng**: Sophia (ICLR'24), Mousse (arXiv'26)
> **Độ khả thi**: ★★★★★ (zero additional compute — reuse SASSHA state)

### Ý tưởng

Dùng `exp_hessian_diag` đã có từ SASSHA để điều chỉnh γ **per-layer**. Layer nào có Hessian variance cao (curvature không đồng đều) → cần flatten mạnh hơn.

### Công thức

$$\gamma^{(l)} = \gamma_{\text{base}} \cdot \text{clip}\left(\frac{\text{Var}(h^{(l)})}{\text{Var}_{\text{ref}}}, \alpha_{\min}, \alpha_{\max}\right)$$

với:
- $h^{(l)}$ = `exp_hessian_diag` của layer $l$
- $\text{Var}_{\text{ref}}$ = running average of Hessian variance (EMA)
- $\alpha_{\min} = 0.5$, $\alpha_{\max} = 2.5$ (scaling bounds)

### Implementation

```python
def hg_sdp(net, optimizer, gamma_base=0.3, alpha_min=0.5, alpha_max=2.5):
    """Hessian-Guided SDP: per-layer adaptive gamma"""
    # Tính reference variance (median across layers)
    h_vars = []
    for name, param in net.named_parameters():
        if param.dim() >= 2:
            state = optimizer.state.get(param, {})
            h = state.get('exp_hessian_diag', None)
            if h is not None:
                h_vars.append(h.var().item())
    var_ref = np.median(h_vars) if h_vars else 1.0

    condition_numbers = []
    for name, param in net.named_parameters():
        if param.dim() >= 2:
            state = optimizer.state.get(param, {})
            h = state.get('exp_hessian_diag', None)

            # Adaptive gamma
            if h is not None and var_ref > 0:
                scale = np.clip(h.var().item() / var_ref, alpha_min, alpha_max)
                gamma_l = gamma_base * scale
            else:
                gamma_l = gamma_base

            gamma_l = np.clip(gamma_l, 0.05, 0.8)

            # Standard SDP with adaptive gamma
            W = param.data
            original_shape = W.shape
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            S_mean = S.mean()
            S_new = (S_mean ** gamma_l) * (S ** (1 - gamma_l))
            condition_numbers.append((S[0] / S[-1]).item())
            W_new = U @ torch.diag(S_new) @ Vh
            param.data = W_new.reshape(original_shape)

    return condition_numbers
```

### Tại sao khả thi?

| Tiêu chí | Đánh giá |
|---|---|
| Compute overhead | **Zero** — chỉ đọc state có sẵn |
| Complexity | Thay 1 dòng (gamma cố định → gamma tính toán) |
| Lý thuyết | Sophia + Mousse validate: per-dimension curvature adaptation > uniform |
| Risk | Thấp — fallback về static gamma nếu Hessian chưa sẵn sàng |

---

## Method 2: Spike-Aware SDP (SA-SDP)

> **Nguồn cảm hứng**: Spectra (arXiv'26), Egalitarian GD (ICML'25)
> **Độ khả thi**: ★★★★☆

### Ý tưởng

Phân tách singular values thành 2 nhóm:
- **Spike** (top-K SVs, chiếm phần lớn energy): compress mạnh
- **Tail** (remaining): compress nhẹ hoặc giữ nguyên

Spectra chỉ ra ~1.5% directions chiếm phần lớn gradient energy → flatten mạnh spike directions giúp learning tail content tốt hơn.

### Công thức

$$\sigma'_i = \begin{cases} \bar{\sigma}^{\gamma_s} \cdot \sigma_i^{(1-\gamma_s)} & \text{if } i \leq K \text{ (spike)} \\ \bar{\sigma}^{\gamma_t} \cdot \sigma_i^{(1-\gamma_t)} & \text{if } i > K \text{ (tail)} \end{cases}$$

với:
- $K = \lceil r \cdot \min(m,n) \rceil$, $r$ = spike ratio (default 0.05)
- $\gamma_s = \gamma_{\text{base}} \cdot 2.0$ (compress mạnh)
- $\gamma_t = \gamma_{\text{base}} \cdot 0.3$ (compress nhẹ)

### Xác định K tự động (adaptive)

Thay vì K cố định, dùng **spectral gap**:

$$K = \arg\max_i \left(\frac{\sigma_i}{\sigma_{i+1}}\right) \quad \text{s.t. } \frac{\sigma_i}{\sigma_{i+1}} > \tau$$

Tức K = vị trí có gap lớn nhất giữa consecutive SVs (spectral gap threshold $\tau = 1.5$).

### Implementation

```python
def spike_aware_sdp(net, gamma_base=0.3, spike_ratio=0.05, 
                     spike_mult=2.0, tail_mult=0.3, use_gap=True, gap_tau=1.5):
    """Spike-Aware SDP: compress spike và tail khác nhau"""
    condition_numbers = []
    for name, param in net.named_parameters():
        if param.dim() >= 2:
            W = param.data
            original_shape = W.shape
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            S_mean = S.mean()

            if use_gap and len(S) > 2:
                # Adaptive K: spectral gap detection
                ratios = S[:-1] / (S[1:] + 1e-10)
                gap_mask = ratios > gap_tau
                if gap_mask.any():
                    K = gap_mask.nonzero()[0, 0].item() + 1
                else:
                    K = max(1, int(len(S) * spike_ratio))
            else:
                K = max(1, int(len(S) * spike_ratio))

            # Dual-zone compression
            gamma_s = min(gamma_base * spike_mult, 0.85)
            gamma_t = gamma_base * tail_mult

            S_new = S.clone()
            S_new[:K] = (S_mean ** gamma_s) * (S[:K] ** (1 - gamma_s))
            S_new[K:] = (S_mean ** gamma_t) * (S[K:] ** (1 - gamma_t))

            condition_numbers.append((S[0] / S[-1]).item())
            W_new = U @ torch.diag(S_new) @ Vh
            param.data = W_new.reshape(original_shape)

    return condition_numbers
```

### Tại sao khả thi?

| Tiêu chí | Đánh giá |
|---|---|
| Compute overhead | **Negligible** — thêm 1 argmax trên SVs đã tính |
| Lý thuyết | Spectra: spike suppression → 30% faster convergence trên LLaMA3-8B |
| Insight | Spike SVs = dominant features từ tasks cũ; tail SVs = room cho task mới |
| Risk | Trung bình — cần tune spike_mult và tail_mult |

---

## Method 3: Condition-Aware SDP (CA-SDP)

> **Nguồn cảm hứng**: Spectral Collapse (NeurIPS'25), τ-trainability
> **Độ khả thi**: ★★★★☆

### Ý tưởng

Điều chỉnh γ dựa trên **condition number** hiện tại của weight matrix. Layer đã well-conditioned → flatten nhẹ (tránh phá vỡ representation tốt). Layer ill-conditioned → flatten mạnh.

### Công thức

$$\gamma^{(l)} = \gamma_{\text{base}} \cdot \tanh\left(\frac{\kappa^{(l)}}{\kappa_{\text{target}}}\right)$$

với:
- $\kappa^{(l)} = \sigma_1^{(l)} / \sigma_k^{(l)}$: condition number hiện tại
- $\kappa_{\text{target}}$: condition number mục tiêu (hyperparameter, e.g., 10)

**Tính chất**:
- $\kappa \ll \kappa_{\text{target}}$ → $\gamma \approx \gamma_{\text{base}} \cdot \kappa/\kappa_t$ (rất nhỏ, gần như không flatten)
- $\kappa \gg \kappa_{\text{target}}$ → $\gamma \approx \gamma_{\text{base}}$ (flatten full strength)
- Smooth transition, không cần thêm hyperparameter

### Implementation

```python
def condition_aware_sdp(net, gamma_base=0.3, kappa_target=10.0):
    """Condition-Aware SDP: flatten proportional to ill-conditioning"""
    condition_numbers = []
    for name, param in net.named_parameters():
        if param.dim() >= 2:
            W = param.data
            original_shape = W.shape
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            kappa = (S[0] / (S[-1] + 1e-8)).item()
            gamma_l = gamma_base * np.tanh(kappa / kappa_target)

            S_mean = S.mean()
            S_new = (S_mean ** gamma_l) * (S ** (1 - gamma_l))

            condition_numbers.append(kappa)
            W_new = U @ torch.diag(S_new) @ Vh
            param.data = W_new.reshape(original_shape)

    return condition_numbers
```

### Tại sao khả thi?

| Tiêu chí | Đánh giá |
|---|---|
| Compute overhead | **Zero** — κ tính từ SVs đã có |
| Hyperparameters | Chỉ 1 (κ_target), ý nghĩa trực quan |
| Lý thuyết | NeurIPS'25: spectral collapse ↔ high κ → cần flatten |
| Risk | Thấp — tự động giảm intervention khi layer đã tốt |

---

## Method 4: Progressive Spectral Relaxation (PSR)

> **Nguồn cảm hứng**: Adaptive SVD CL (arXiv'25), Weight Interpolation CL
> **Độ khả thi**: ★★★☆☆

### Ý tưởng

Thay vì flatten tại mỗi task boundary với cùng cường độ, **giảm dần γ** trong quá trình training một task. Đầu task → flatten mạnh (reset plasticity). Cuối task → flatten nhẹ (bảo toàn learned representations).

### Công thức

$$\gamma(t) = \gamma_{\text{max}} \cdot \left(1 - \frac{t}{T}\right)^p + \gamma_{\text{min}}$$

với:
- $t$: step hiện tại trong task
- $T$: tổng steps per task
- $p$: decay power (default 2, quadratic decay)
- $\gamma_{\text{max}} = 0.5$, $\gamma_{\text{min}} = 0.05$

### Implementation

```python
def progressive_sdp(net, step_in_task, total_steps, 
                     gamma_max=0.5, gamma_min=0.05, decay_power=2):
    """Progressive SDP: giảm dần cường độ flatten trong task"""
    progress = step_in_task / total_steps
    gamma = gamma_max * (1 - progress) ** decay_power + gamma_min

    condition_numbers = []
    for name, param in net.named_parameters():
        if param.dim() >= 2:
            W = param.data
            original_shape = W.shape
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            S_mean = S.mean()
            S_new = (S_mean ** gamma) * (S ** (1 - gamma))
            condition_numbers.append((S[0] / S[-1]).item())
            W_new = U @ torch.diag(S_new) @ Vh
            param.data = W_new.reshape(original_shape)

    return condition_numbers, gamma
```

### Lưu ý

- Nếu áp dụng SDP chỉ tại task boundaries (như hiện tại), method này cần chuyển sang áp dụng SDP **periodically** trong task (e.g., mỗi N steps)
- Chi phí SVD tăng → cân nhắc dùng cho mỗi 50–100 steps thay vì mỗi step

---

## Method 5: Hessian-Projected Spectral Compression (HP-SDP)

> **Nguồn cảm hứng**: Mousse (arXiv'26), SOAP (NeurIPS'24)
> **Độ khả thi**: ★★★☆☆ (per-SV gamma, cần projection)

### Ý tưởng

Nâng cấp từ **per-layer γ** (Method 1) lên **per-singular-value γ**. Project Hessian diagonal vào singular vector space để biết curvature dọc theo từng singular direction.

### Công thức

$$h_i^{SV} = u_i^T \cdot H_{\text{diag}}^{(reshaped)} \cdot v_i$$

$$\gamma_i = \gamma_{\text{base}} \cdot \text{sigmoid}\left(\frac{h_i^{SV} - \bar{h}^{SV}}{s \cdot \bar{h}^{SV}}\right) \cdot 2$$

với:
- $u_i$, $v_i$: left/right singular vectors
- $H_{\text{diag}}^{(reshaped)}$: Hessian diagonal reshape thành matrix
- $s$: sensitivity scale (default 1.0)
- Sigmoid đảm bảo γ ∈ (0, 2·γ_base)

### Implementation

```python
def hessian_projected_sdp(net, optimizer, gamma_base=0.3, sensitivity=1.0):
    """HP-SDP: per-singular-value gamma guided by Hessian projection"""
    condition_numbers = []
    for name, param in net.named_parameters():
        if param.dim() >= 2:
            state = optimizer.state.get(param, {})
            h_diag = state.get('exp_hessian_diag', None)

            W = param.data
            original_shape = W.shape
            W_2d = W.reshape(W.shape[0], -1) if W.dim() > 2 else W
            U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

            if h_diag is not None:
                H_2d = h_diag.reshape(W_2d.shape)
                # Project Hessian onto each singular direction
                # h_sv[i] = u_i^T @ H_2d @ v_i^T
                # Efficient: (U^T @ H_2d @ Vh^T).diag()
                h_sv = (U.T @ H_2d @ Vh.T).diag().abs()
                h_mean = h_sv.mean() + 1e-8

                # Per-SV gamma via sigmoid scaling
                gamma_per_sv = gamma_base * 2.0 * torch.sigmoid(
                    (h_sv - h_mean) / (sensitivity * h_mean)
                )
            else:
                gamma_per_sv = gamma_base

            S_mean = S.mean()
            S_new = (S_mean ** gamma_per_sv) * (S ** (1 - gamma_per_sv))

            condition_numbers.append((S[0] / S[-1]).item())
            W_new = U @ torch.diag(S_new) @ Vh
            param.data = W_new.reshape(original_shape)

    return condition_numbers
```

### Tại sao khả thi?

| Tiêu chí | Đánh giá |
|---|---|
| Compute overhead | **Thấp** — 1 matrix multiply (U^T @ H @ V^T), SVD đã tính |
| Granularity | **Cao nhất** — per-SV decisions |
| Lý thuyết | SOAP + Mousse: curvature-aware rotation/preconditioning > uniform |
| Risk | Trung bình — Hessian projection có thể noisy, cần clipping |

---

## Method 6: Dual-Phase SDP (DP-SDP)

> **Nguồn cảm hứng**: Spectral Regularization (PRL'24), Adaptive SVD CL (arXiv'25)
> **Độ khả thi**: ★★★★☆

### Ý tưởng

Kết hợp 2 cơ chế spectral khác nhau:
1. **SDP tại task boundaries**: flatten toàn bộ spectrum (như hiện tại)
2. **Spectral Reg liên tục**: giữ $\sigma_{\max} \approx 1$ giữa các boundaries

Điều này ngăn spectral drift giữa các boundaries.

### Công thức

**Tại task boundary** (Phase 1 — SDP):
$$\sigma'_i = \bar{\sigma}^\gamma \cdot \sigma_i^{(1-\gamma)}$$

**Mỗi training step** (Phase 2 — Spectral Reg):
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{spec}} \sum_l (\sigma_{\max}^{(l)} - 1)^2$$

### Implementation

```python
# Phase 2: Thêm vào loss function
def spectral_reg_loss(net, lambda_spec=0.01):
    """Lightweight spectral regularization giữa boundaries"""
    reg = 0.0
    for name, param in net.named_parameters():
        if param.dim() >= 2:
            W = param.data
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            # Power iteration cho sigma_max (rất rẻ)
            sigma_max = torch.linalg.norm(W, ord=2)
            reg += (sigma_max - 1.0) ** 2
    return lambda_spec * reg

# Training loop
loss = criterion(output, target)
loss += spectral_reg_loss(net, lambda_spec=0.01)  # Thêm dòng này
loss.backward()
```

### Tại sao khả thi?

| Tiêu chí | Đánh giá |
|---|---|
| Compute overhead | **Rất thấp** — power iteration ≈ 1 matmul/layer |
| Interaction w/ SDP | Complementary — SDP flatten full spectrum, Spec Reg giữ ổn định |
| Lý thuyết | PRL'24: σ_max ≈ 1 → gradient diversity maintained |
| Risk | Thấp — λ_spec nhỏ, dễ tune |

---

## Bảng so sánh tổng hợp

| Method | Per-layer | Per-SV | Target-aware | Compute | Hyperparams mới | Ưu tiên implement |
|---|---|---|---|---|---|---|
| **M1: HG-SDP** | ✅ | ❌ | ⚠️ (Hessian) | Zero | 2 (α_min, α_max) | 🥇 |
| **M2: SA-SDP** | ❌ | ✅ (2 zones) | ❌ | Negligible | 3 (r, γ_s mult, γ_t mult) | 🥈 |
| **M3: CA-SDP** | ✅ | ❌ | ❌ | Zero | 1 (κ_target) | 🥇 |
| **M4: PSR** | Global | ❌ | ❌ | SVD × N/task | 3 (γ_max, γ_min, p) | 🥉 |
| **M5: HP-SDP** | ❌ | ✅ (full) | ✅ (Hessian) | 1 matmul | 1 (sensitivity) | 🥈 |
| **M6: DP-SDP** | Global + per-layer | ❌ | ❌ | Power iter/step | 1 (λ_spec) | 🥇 |

---

## Đề xuất thứ tự thực nghiệm

### Phase 1: Quick wins (1–2 ngày)

1. **M3 (CA-SDP)** — đơn giản nhất, 1 hyperparameter, ý nghĩa trực quan rõ ràng
2. **M1 (HG-SDP)** — tận dụng SASSHA info có sẵn, zero overhead

### Phase 2: Medium (3–5 ngày)

3. **M6 (DP-SDP)** — thêm spectral reg loss, complement SDP hiện tại
4. **M2 (SA-SDP)** — spike/tail dual-zone compression

### Phase 3: Advanced (1 tuần)

5. **M5 (HP-SDP)** — per-SV Hessian-guided, granularity cao nhất
6. **M1 + M2 + M6 combined** — Hessian-guided + spike-aware + liên tục

### Ablation plan

Chạy trên **Permuted MNIST 800 tasks** (cùng setup hiện tại) so sánh:

| Experiment | Method | Metrics |
|---|---|---|
| Baseline | SDP γ=0.3 | test_acc, srank, dormant, κ |
| Exp 1 | CA-SDP κ_t ∈ {5, 10, 20, 50} | — |
| Exp 2 | HG-SDP α ∈ {(0.3,2), (0.5,2.5), (0.5,3)} | — |
| Exp 3 | SA-SDP r ∈ {0.02, 0.05, 0.1}, gap vs fixed | — |
| Exp 4 | DP-SDP λ ∈ {0.001, 0.01, 0.1} | — |
| Exp 5 | HP-SDP s ∈ {0.5, 1.0, 2.0} | — |
| Exp 6 | Best combo | — |

---

## Phân tích lý thuyết: Tại sao adaptive > static

### 1. Static γ flatten quá mức ở layers đã well-conditioned

```
Layer A: κ = 5 (đã tốt)     → SDP γ=0.3 → phá representation không cần thiết
Layer B: κ = 500 (rất xấu)  → SDP γ=0.3 → flatten chưa đủ
```

CA-SDP (M3) giải quyết: γ_A ≈ 0.14, γ_B ≈ 0.30

### 2. Spike SVs encode bias từ tasks cũ

```
Task 1: learns features F1 → σ₁ tăng (spike cho F1)
Task 2: cần features F2    → σ₁ vẫn dominate → F2 bị suppressed
```

SA-SDP (M2) giải quyết: compress σ₁ mạnh → giải phóng capacity cho F2

### 3. Hessian curvature thay đổi giữa tasks

```
Task 1 → SASSHA tối ưu H₁ → SDP flatten uniform
Task 2 → H₂ ≠ H₁         → cần flatten khác nhau theo H₂
```

HG-SDP (M1) / HP-SDP (M5) giải quyết: dùng H₂ info để guide flatten

### 4. Spectral drift giữa boundaries

```
Task boundary: SDP flatten → κ thấp
Step 1000: κ đã tăng lại → SDP chỉ fix ở boundary kế tiếp
```

DP-SDP (M6) giải quyết: spectral reg liên tục giữ κ thấp
