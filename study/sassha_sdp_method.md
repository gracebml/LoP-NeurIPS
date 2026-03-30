# SASSHA + SDP: Method Description

## 1. SASSHA (Sharpness-Aware Stable Second-order Hessian Approximation)

> **Source**: ICML 2025

SASSHA kết hợp 3 thành phần trong mỗi bước cập nhật:

### Phase 1: SAM Perturbation (Ascent Step)

Đưa weight về vùng loss cao nhất trong lân cận ρ:

$$\hat{\epsilon}(w) = \rho \cdot \frac{\nabla_w L(w)}{\|\nabla_w L(w)\|}$$

$$w_{perturbed} = w + \hat{\epsilon}(w)$$

### Phase 2: Hutchinson Hessian Estimation

Tại $w_{perturbed}$, tính gradient $g$ và xấp xỉ Hessian diagonal bằng Hutchinson:

$$\hat{h}_i = z_i \cdot \frac{\partial (g \cdot z)}{\partial w_i}$$

với $z \sim \text{Rademacher}(\pm 1)$, cập nhật EMA:

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot |\hat{h}|$$

Chỉ tính Hessian mỗi `lazy_hessian` bước (mặc định 10) để tiết kiệm chi phí.

### Phase 3: Preconditioned Update (Adam-like)

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g$$

$$w_{t+1} = w_t - \eta \cdot \frac{\hat{m}_t}{\hat{v}_t^{p} + \epsilon} - \lambda \cdot w_t$$

với $p$ = `hessian_power` (mặc định 0.5), $\hat{m}_t, \hat{v}_t$ là bias-corrected.

---

## 2. SDP (Spectral Distribution Perturbation)

### Ý tưởng cốt lõi

Tại mỗi task boundary, nén (compress) phổ singular values của weight matrices bằng phép nội suy hình học giữa mỗi singular value và trung bình của chúng:

$$\sigma'_i = \bar{\sigma}^\gamma \cdot \sigma_i^{(1-\gamma)}$$

với:
- $\bar{\sigma} = \frac{1}{k}\sum_{i=1}^k \sigma_i$: trung bình singular values
- $\gamma \in [0, 1]$: cường độ nén (mặc định 0.3)
- $\gamma = 0$: giữ nguyên, $\gamma = 1$: tất cả = $\bar{\sigma}$

### Thuộc tính

- **Bảo toàn hướng**: Chỉ thay đổi magnitude (singular values), không thay đổi left/right singular vectors → giữ nguyên không gian biểu diễn đã học
- **Bảo toàn trung bình hình học**: $\prod \sigma'_i = \prod \sigma_i$ (geometric mean không đổi)
- **Giảm condition number**: $\kappa' = (\sigma_1/\sigma_k)^{(1-\gamma)} < \kappa$

### Implementation

```python
def apply_sdp(net, gamma=0.3):
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
    return condition_numbers
```

---

## 3. EMA (Exponential Moving Average)

Duy trì bản sao EMA của model weights:

$$\theta_{EMA} = \alpha \cdot \theta_{EMA} + (1 - \alpha) \cdot \theta_t$$

- `decay` = 0.999
- **Reset** tại mỗi task boundary (copy weights hiện tại vào EMA)
- Chỉ sử dụng cho **evaluation** (test accuracy), không dùng khi train
- Giúp evaluation ổn định hơn, giảm noise từ SGD

---

## 4. Hessian Clipping Guard

Clamp `exp_hessian_diag` để tránh gradient explosion do stale Hessian:

```python
for group in optimizer.param_groups:
    for p in group['params']:
        state = optimizer.state[p]
        if 'exp_hessian_diag' in state:
            state['exp_hessian_diag'].clamp_(min=1e-5, max=1e3)
```

- **floor** = 1e-5: ngăn mẫu số quá nhỏ → update quá lớn
- **ceiling** = 1e3: ngăn Hessian estimate quá lớn → update quá nhỏ (plasticity loss)

---

## 5. Pipeline tổng thể

```
Mỗi task t = 1, ..., T:
│
├── Reset EMA (copy current weights)
├── Apply SDP (nén SVD spectrum, gamma=0.3)
│
├── Training loop (STEPS_PER_TASK steps):
│   │
│   ├── Forward pass → loss
│   ├── SASSHA Phase 1: SAM perturbation
│   ├── SASSHA Phase 2: Hutchinson Hessian (mỗi 10 steps)
│   ├── SASSHA Phase 3: Preconditioned update
│   ├── Apply Hessian Clipping Guard
│   ├── Update EMA
│   └── Log metrics (accuracy, weight mag, ranks)
│
├── Test evaluation (dùng EMA weights)
├── Log: test_accuracy, stable_rank, dormant_frac
└── Next task
```

---

## 6. Hyperparameters

| Parameter | Giá trị | Mô tả |
|---|---|---|
| lr | 0.003 | Learning rate |
| betas | (0.9, 0.999) | Adam-style momentum |
| weight_decay | 5e-4 | L2 regularization |
| rho | 0.05 | SAM perturbation radius |
| hessian_power | 0.5 | Exponent cho Hessian preconditioner |
| lazy_hessian | 10 | Hessian update interval |
| n_samples | 1 | Hutchinson random vectors |
| eps | 1e-4 | Numerical stability |
| gamma (SDP) | 0.3 | Cường độ nén spectrum |
| EMA decay | 0.999 | EMA smoothing factor |
| Guard floor | 1e-5 | Min Hessian diagonal |
| Guard ceiling | 1e3 | Max Hessian diagonal |

---

## 7. Metrics

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| Train Accuracy | Correct / Total | Online training performance |
| Test Accuracy | Eval on test set (EMA weights) | Generalization per task |
| Stable Rank | 99% cumulative \|σ\| threshold (CBP formula) | Representation diversity |
| Dormant Fraction | Neurons with mean activation < 0.01 * mean | Dead neuron ratio |
| Weight Magnitude | Mean \|w\| per layer | Weight growth monitoring |
| Condition Number | σ₁/σₖ trước SDP | Weight matrix conditioning |

---

## 8. Phân tích: Tại sao SDP giải quyết Loss of Plasticity

### 8.1 Ba vấn đề chính của Loss of Plasticity (LoP)

Trong mạng không có BatchNorm (e.g., DeepFFNN), ba vấn đề LoP chính là:

1. **Rank collapse**: Singular values của weights phân kỳ ($\sigma_1 \gg \sigma_2 \gg ... \gg \sigma_k$) → feature space collapse thành ít chiều → stable rank giảm
2. **Dormant neurons**: Nhiều neurons nhận pre-activation gần 0 → ReLU cắt → neuron "chết" → capacity giảm
3. **Weight norm growth**: Weight magnitude tăng liên tục → effective learning rate giảm → mạng "đông cứng"

### 8.2 SDP giải quyết cả 3 vấn đề

**Kết quả thực nghiệm** trên Permuted MNIST (800 tasks, DeepFFNN 784→2000×5→10, ReLU, không BatchNorm):
- `dormant_neuron_ratio = 0` liên tục
- Weight norm ổn định, không tăng bất thường
- Stable rank duy trì cao

| Vấn đề LoP | SDP giải quyết? | Cơ chế |
|---|---|---|
| Rank collapse | **Trực tiếp** | Nén SVD spectrum: $\sigma'_i = \bar{\sigma}^\gamma \cdot \sigma_i^{(1-\gamma)}$ → srank ↑ |
| Dormant neurons | **Gián tiếp** | SVs đồng đều → activations đa dạng → neurons đều active → ratio = 0 |
| Weight norm growth | **Gián tiếp** | Nén extreme SVs → weight magnitude ổn định qua các tasks |

**Chuỗi nhân quả cho dormant neurons:**

```
SDP nén SVs → pre-activations phân bố đồng đều hơn
            → ít neurons nhận input gần 0
            → ReLU ít cắt hơn → dormant_ratio ≈ 0
```

**Chuỗi nhân quả cho weight norm:**

```
SDP nén σ₁ (lớn nhất) xuống, nâng σₖ (nhỏ nhất) lên
  → ||W||_F = sqrt(Σσᵢ²) ổn định hơn (variance(σ) giảm)
  → Weight magnitude không tăng bất thường qua tasks
```

### 8.3 SDP thay thế vai trò spectral của BatchNorm

**Insight quan trọng**: BatchNorm loại bỏ outlier eigenvalues trong phổ Hessian bằng cách normalize activations. Mạng không có BatchNorm xuất hiện nhiều trị riêng outlier trong phổ Hessian, gây ill-conditioning.

SDP giải quyết vấn đề này **ở tầng weight** thay vì activation:

**Chuỗi nhân quả: Weight SVs → Hessian Spectrum**

Hessian phụ thuộc Jacobian theo Gauss-Newton decomposition:

$$H \approx J^T \cdot \frac{\partial^2 L}{\partial f^2} \cdot J$$

Jacobian $J = \frac{\partial f}{\partial \theta}$ chứa tích các ma trận trọng số. Nếu weight matrices có singular values cực đoan → Jacobian có singular values cực đoan → **Hessian eigenvalues outlier**.

Không có BatchNorm thì:

```
Weight SVs phân kỳ (σ₁ >> σ₂ >> ... >> σₖ)
  → Activations có variance cực lớn theo vài direction
  → Feature covariance C = ZᵀZ/n có spike eigenvalues (λ₁ >> λ₂)
  → Hessian ill-conditioned với outlier eigenvalues
```

SDP nén phổ SVs:

```
SDP nén weight SVs → Jacobian SVs đồng đều hơn
                   → Feature covariance isotropic hơn (srank ↑)
                   → Hessian spectrum đồng đều hơn
                   → Ít outlier eigenvalues hơn
```

**So sánh SDP vs BatchNorm vs EES:**

| Phương pháp | Tác động lên | Cơ chế | Tần suất |
|---|---|---|---|
| BatchNorm | Activations | Normalize mean/var mỗi batch | Mỗi forward pass |
| SDP | Weights | Nén SVD spectrum | Tại task boundaries |
| EES | Features | Loss regularization trên eigenspectrum | Mỗi training step |

**Bản chất**: SDP là một dạng **spectral BatchNorm cho weights** — đạt được hiệu ứng tương tự BatchNorm (loại bỏ outlier eigenvalues, ổn định training) nhưng ở cấp độ weight matrix thay vì activation level.

### 8.4 Synergy với SASSHA: Virtuous Cycle

SDP + SASSHA tạo **vòng lặp tích cực (virtuous cycle)**:

```
SDP nén weight SVs
  → Hessian ít outliers
  → exp_hessian_diag (Hutchinson estimator) chính xác hơn
  → SASSHA cập nhật weight ổn định hơn
  → Weight SVs ít bị phân kỳ giữa các steps
  → SDP ở task tiếp theo càng hiệu quả
  → (lặp lại)
```

Ngược lại, **không có SDP** thì vòng xoáy âm:

```
Hessian outliers → Hutchinson estimator nhiễu
  → exp_hessian_diag có entries cực lớn/nhỏ
  → Parameter update không ổn định
  → Weight SVs phân kỳ thêm
  → Hessian càng ill-conditioned
  → Gradient explosion (đã xảy ra trên CIFAR, xem gradient-explosion-prevention.md)
```

---

## 9. So sánh với các phương pháp chống LoP khác

| Phương pháp | Rank collapse | Dormant neurons | Weight norm | Hessian outliers | Cần modify architecture? |
|---|---|---|---|---|---|
| SASSHA + SDP | ✅ Trực tiếp | ✅ Gián tiếp | ✅ Gián tiếp | ✅ Gián tiếp | Không |
| CBP (Nature 2024) | ❌ | ✅ Trực tiếp (reinit) | ⚠️ Hạn chế | ❌ | Không |
| BatchNorm | ⚠️ Hạn chế | ⚠️ Hạn chế | ⚠️ (scale invariance issues) | ✅ Trực tiếp | Có |
| Spectral Norm | ⚠️ Chỉ σ_max | ❌ | ✅ | ⚠️ Chỉ σ_max | Không |
| Weight Decay | ❌ | ❌ | ✅ | ❌ | Không |
| EES Regularization | ✅ Trực tiếp | ⚠️ Gián tiếp | ❌ | ✅ Trực tiếp | Không |

---

## 10. References

1. SASSHA — ICML 2025
2. SAM — Foret et al., ICLR 2021
3. Hutchinson's estimator — Hutchinson 1989
4. Loss of Plasticity — Lyle et al., Nature 2024
5. CBP (Continual Backpropagation) — Dohare et al., Nature 2024
6. Equal-Eigenvalue Spectrum — INTL, arXiv 2305.16789
7. Adaptive Gradient Clipping — Brock et al., ICML 2021 (NFNet)
8. Weight Norm Growth & Plasticity — Lyle et al., arXiv 2402.18762, 2024
