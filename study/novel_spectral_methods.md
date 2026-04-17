# Novel Spectral Compression Methods for Non-Stationary Optimization

> **Date**: 2026-03-14
> **Context**: Extensions beyond SDP & proposed ASC/CASC/TSS/FCSC/SPR

---

## Phân tích Gap trong các phương pháp hiện có

Các phương pháp đã đề xuất (ASC, CASC, TSS, FCSC, SPR) đều xử lý phổ singular values như một **biến đổi hình học tĩnh tại task boundary**. Chúng bỏ sót 3 vấn đề cốt lõi:

| Gap | Mô tả | Hệ quả |
|---|---|---|
| **G1: Feedback loop** | Nén phổ → thay đổi Hessian → thay đổi preconditioner → thay đổi gradient flow → thay đổi phổ. Không method nào model vòng lặp này. | Nén "mù" có thể phá vỡ virtuous cycle SDP-SASSHA |
| **G2: Directional blindness** | SDP/SPR nén **tất cả** hướng. CASC chỉ xét Hessian. Không method nào xét gradient subspace **của task hiện tại** để quyết định nén gì. | Nén hướng mà task hiện tại đang cần → giảm learning speed |
| **G3: No spectral dynamics model** | TSS smooth nhưng không **dự đoán** phổ sẽ evolve thế nào. Nén reactive, không proactive. | Phải đợi spectral collapse xảy ra rồi mới sửa |

---

## Method 1: Gradient-Orthogonal Spectral Compression (GOSC)

### Gap addressed: G2 — Directional blindness

### Core Insight

Tại task boundary, mô hình chuẩn bị học task mới. Gradient của mini-batch đầu tiên trên task mới define một **subspace mà task mới cần**. Singular directions **orthogonal** to gradient subspace là remnant của tasks cũ → nên nén mạnh. Singular directions **aligned** với gradient → phải bảo toàn để task mới học được.

### Formulation

Cho weight matrix $W$ với SVD: $W = U S V^T$.

Tính gradient trên mini-batch đầu tiên của task mới:

$$G = \nabla_W \mathcal{L}_{\text{new}}$$

SVD của gradient: $G = U_g S_g V_g^T$

Đo alignment giữa singular direction $i$ của $W$ và gradient subspace:

$$a_i = \| U_g^T u_i \|^2 + \| V_g^T v_i \|^2$$

Adaptive compression:

$$\gamma_i = \gamma_{\max} \cdot (1 - a_i) + \gamma_{\min} \cdot a_i$$

$$\sigma'_i = \bar{\sigma}^{\gamma_i} \cdot \sigma_i^{(1-\gamma_i)}$$

**Ý nghĩa**:
- $a_i \approx 1$ (aligned with task mới) → $\gamma_i \approx \gamma_{\min}$ → gần như không nén
- $a_i \approx 0$ (orthogonal to task mới) → $\gamma_i \approx \gamma_{\max}$ → nén mạnh

### Algorithm

```
At task boundary t → t+1:
  1. Sample mini-batch từ task (t+1)
  2. Forward pass → loss → backward pass → G = ∇_W L
  3. For each weight matrix W:
     a. U, S, V = SVD(W)
     b. U_g, _, V_g = SVD(G, rank=r)    # low-rank approx, r ≪ min(m,n)
     c. Compute alignment: aᵢ = ||U_gᵀ uᵢ||² + ||V_gᵀ vᵢ||²
     d. γᵢ = γ_max(1 − aᵢ) + γ_min · aᵢ
     e. σ'ᵢ = σ̄^γᵢ · σᵢ^(1−γᵢ)
     f. W ← U diag(σ') Vᵀ
```

### Tính chất

| Tính chất | Giá trị |
|---|---|
| **Target-aware** | ✅ — gradient từ task mới = direct signal về target distribution |
| **Bảo toàn hướng cần thiết** | ✅ — chỉ nén hướng orthogonal |
| **Chi phí** | 1 forward + 1 backward + 2 SVDs per layer (tại boundary) |
| **Hyperparameters** | $\gamma_{\min}$, $\gamma_{\max}$, $r$ (gradient SVD rank) |

### So sánh với CASC

| | CASC | GOSC |
|---|---|---|
| Cơ sở quyết định | Hessian eigenvalues (curvature) | Gradient directions (task need) |
| Thông tin target | ❌ (Hessian = curvature tổng) | ✅ (gradient = task-specific signal) |
| Preserves | Flat curvature directions | Task-relevant directions |

---

## Method 2: Preconditioner-Spectral Co-Optimization (PSCO)

### Gap addressed: G1 — Feedback loop

### Core Insight

SDP nén phổ → Hessian spectrum thay đổi → SASSHA `exp_hessian_diag` **stale** → preconditioned updates **suboptimal** cho vài steps đầu sau SDP.

**Không method nào xét feedback loop**: nén phổ W lý tưởng **cho state hiện tại** có thể là bad cho state **sau khi preconditioner adapt**.

PSCO giải quyết bằng cách **jointly optimize** spectrum compression và preconditioner update.

### Formulation

**Objective**: Tìm $\gamma$ sao cho sau khi nén, preconditioned gradient step **giảm loss tối đa** trên task mới.

Cho:
- $W' = U \cdot \text{diag}(\bar{\sigma}^\gamma \cdot S^{(1-\gamma)}) \cdot V^T$: weight sau SDP
- $h'$: Hessian diagonal tại $W'$
- $g'$: gradient tại $W'$ trên task mới

Preconditioned update: $\Delta W = -\eta \cdot g' / (h'^p + \epsilon)$

**Expected loss reduction** (first-order Taylor):

$$\Delta \mathcal{L}(\gamma) \approx \langle g', \Delta W \rangle = -\eta \sum_j \frac{g_j'^2}{h_j'^p + \epsilon}$$

Tối ưu:

$$\gamma^* = \arg\min_\gamma \Delta \mathcal{L}(\gamma) \quad \text{s.t. } \gamma \in [\gamma_{\min}, \gamma_{\max}]$$

### Practical Algorithm (Grid Search)

```
At task boundary:
  1. Sample mini-batch từ task mới
  2. For γ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}:
     a. W_temp = apply_sdp(W, γ)
     b. Forward pass with W_temp → loss_temp
     c. Backward → g_temp
     d. Estimate h_temp (1 Hutchinson step)
     e. Compute expected_reduction = −η Σ gⱼ² / (hⱼᵖ + ε)
  3. γ* = argmin expected_reduction
  4. Apply SDP with γ*
```

### Tính chất

| Tính chất | Giá trị |
|---|---|
| **Feedback-aware** | ✅ — xét cả ảnh hưởng lên preconditioner |
| **Target-aware** | ✅ — optimize trên mini-batch task mới |
| **Chi phí** | 6 × (forward + backward + Hutchinson) tại boundary |
| **Hyperparameters** | $\gamma_{\min}$, $\gamma_{\max}$, grid resolution |

### Tradeoff chi phí

Chi phí cao hơn SDP thường (6× tại boundary), nhưng:
- Chỉ xảy ra tại task boundary (1 lần / 200–2000 steps)
- Amortized cost ≈ zero
- Đảm bảo SDP + SASSHA synergy **tối ưu theo lý thuyết**

---

## Method 3: Predictive Spectral Regulation (PSReg)

### Gap addressed: G3 — No spectral dynamics model

### Core Insight

SVD spectrum **không random** — nó evolve theo quy luật trong quá trình training. Nếu ta model spectral velocity (tốc độ thay đổi SVs), ta có thể **dự đoán** spectral collapse sắp xảy ra và **can thiệp trước**.

### Formulation

Track spectral velocity qua các task boundaries:

$$\dot{\sigma}_i^{(t)} = \sigma_i^{(t)} - \sigma_i^{(t-1)}$$

Dự đoán phổ tại task tiếp theo (nếu không can thiệp):

$$\hat{\sigma}_i^{(t+1)} = \sigma_i^{(t)} + \dot{\sigma}_i^{(t)}$$

Tính dự đoán condition number:

$$\hat{\kappa}^{(t+1)} = \frac{\hat{\sigma}_1^{(t+1)}}{\hat{\sigma}_k^{(t+1)} + \epsilon}$$

Adaptive compression dựa trên **predicted deterioration**:

$$\gamma^{(t)} = \gamma_{\text{base}} \cdot \text{clip}\left(\frac{\hat{\kappa}^{(t+1)}}{\kappa_{\text{target}}}, 0.5, 3.0\right)$$

### Mở rộng: Per-SV prediction

Cho mỗi SV, nếu nó **predicted sẽ phân kỳ** → nén mạnh. Nếu stable → nén nhẹ.

$$\gamma_i^{(t)} = \gamma_{\text{base}} \cdot \text{sigmoid}\left(\frac{\dot{\sigma}_i^{(t)}}{\bar{\sigma}^{(t)}} \cdot \beta\right) \cdot 2$$

- $\dot{\sigma}_i > 0$ (SV đang tăng — spike growing) → $\gamma_i > \gamma_{\text{base}}$ → nén mạnh
- $\dot{\sigma}_i < 0$ (SV đang giảm — có thể collapse) → $\gamma_i < \gamma_{\text{base}}$ → nén nhẹ / bảo toàn
- $\dot{\sigma}_i \approx 0$ (stable) → $\gamma_i \approx \gamma_{\text{base}}$

### Algorithm

```
Maintain: σ_prev[layer] = None

At task boundary t:
  1. For each weight matrix W:
     a. U, S, V = SVD(W)
     b. If σ_prev[layer] exists:
        - velocity = S − σ_prev[layer]
        - S_predicted = S + velocity
        - κ_predicted = S_predicted[0] / (S_predicted[-1] + ε)
        - γ = γ_base · clip(κ_predicted / κ_target, 0.5, 3.0)
        
        # OR per-SV:
        - γ_per_sv = γ_base · 2 · sigmoid(velocity / S.mean() · β)
     c. Else:
        - γ = γ_base
     d. Apply SDP with γ (or γ_per_sv)
     e. σ_prev[layer] = S.clone()
```

### Tính chất

| Tính chất | Giá trị |
|---|---|
| **Proactive** | ✅ — can thiệp TRƯỚC khi collapse xảy ra |
| **Per-SV adaptive** | ✅ (version mở rộng) |
| **Chi phí** | Zero (lưu SVs từ lần SVD trước) |
| **Memory** | +1 tensor per layer (σ_prev) |
| **Hyperparameters** | $\kappa_{\text{target}}$ hoặc $\beta$ |

---

## Method 4: Spectral Energy Conservation with Directed Reallocation (SECDR)

### Gap addressed: Combination of G2 + cải tiến SPR

### Core Insight

SPR redistribute spectral mass nhưng **không biết redistribute đi đâu**. SECDR kết hợp:
1. **Conservation**: tổng spectral energy $\sum \sigma_i^2$ không đổi (bảo toàn Frobenius norm)
2. **Directed**: chuyển energy **từ hướng orthogonal to gradient** **sang hướng aligned with gradient**

Thay vì nén uniform, **lấy energy từ chỗ không cần**, **đưa vào chỗ cần**.

### Formulation

Cho alignment scores $a_i$ (từ GOSC):

$$a_i = \| U_g^T u_i \|^2 + \| V_g^T v_i \|^2, \quad a_i \in [0, 2]$$

Normalize: $\tilde{a}_i = a_i / \sum_j a_j$

Target spectral energy distribution:

$$\sigma_i'^2 = \|W\|_F^2 \cdot \left[(1-\alpha) \cdot \frac{\sigma_i^2}{\|W\|_F^2} + \alpha \cdot \tilde{a}_i\right]$$

$$\sigma_i' = \sqrt{(1-\alpha) \cdot \sigma_i^2 + \alpha \cdot \|W\|_F^2 \cdot \tilde{a}_i}$$

- $\alpha = 0$: giữ nguyên spectrum
- $\alpha = 1$: phân bổ hoàn toàn theo gradient alignment
- $\alpha \in (0, 0.3]$: gentle reallocation

### Tính chất đặc biệt

1. **$\|W\|_F$ bảo toàn chính xác** — không thay đổi weight magnitude
2. **Hướng SVD bảo toàn** — chỉ thay đổi magnitudes (như SDP)
3. **Task-aware** — energy flow theo nhu cầu của task mới
4. **Singular vectors không mất** — SVs nhỏ được bơm thêm energy, không bị xóa

### So sánh SPR vs SECDR

| | SPR | SECDR |
|---|---|---|
| Conservation | $\sum \sigma_i$ (sum) | $\sum \sigma_i^2$ (Frobenius, vật lý hơn) |
| Direction | Toward mean (undirected) | Toward task-relevant directions |
| Target-aware | ❌ | ✅ |
| Nguy cơ harm plasticity | Có (flatten hướng cần cho task mới) | Thấp (bơm energy vào hướng cần) |

---

## Method 5: Spectral Lyapunov Stabilization (SLS)

### Gap addressed: G1 + G3 — Feedback loop + Dynamics

### Core Insight (Theoretical)

Xem xét spectral evolution như một **dynamical system**:

$$\sigma_i^{(t+1)} = f(\sigma_i^{(t)}, g^{(t)}, h^{(t)})$$

Hệ này có thể **unstable** (spectral explosion / collapse). Lý thuyết Lyapunov: hệ ổn định khi tồn tại hàm Lyapunov $V(\sigma)$ giảm dần.

**Chọn Lyapunov function**:

$$V(\sigma) = \sum_i (\log \sigma_i - \log \bar{\sigma})^2$$

Đây là log-variance của spectrum. $V = 0$ khi tất cả SVs bằng nhau (perfectly flat).

### Formulation

Tại mỗi task boundary, áp dụng **Lyapunov-stabilizing control** — chọn $\gamma$ để đảm bảo $V$ giảm đủ mạnh:

$$V(\sigma') \leq \rho \cdot V(\sigma), \quad \rho \in (0, 1)$$

Với SDP compression: $\sigma'_i = \bar{\sigma}^\gamma \cdot \sigma_i^{(1-\gamma)}$

$$\log \sigma'_i - \log \bar{\sigma} = (1-\gamma)(\log \sigma_i - \log \bar{\sigma})$$

$$V(\sigma') = (1-\gamma)^2 \cdot V(\sigma)$$

Yêu cầu: $(1-\gamma)^2 \leq \rho$, tức:

$$\gamma \geq 1 - \sqrt{\rho}$$

**Adaptive $\rho$**: dùng **predicted spectral growth** (từ PSReg) để xác định mức ổn định cần thiết:

$$\rho^{(t)} = \exp\left(-\lambda \cdot \frac{\|\dot{\sigma}^{(t)}\|}{\|\sigma^{(t)}\|}\right)$$

- Spectral velocity cao → $\rho$ nhỏ → cần flatten mạnh ($\gamma$ lớn)
- Spectral velocity thấp → $\rho$ gần 1 → flatten nhẹ ($\gamma$ nhỏ)

### Ý nghĩa

SLS biến SDP từ **heuristic** thành **provably stabilizing controller** cho spectral dynamics. Đây là perspective lý thuyết mới: **spectral compression = feedback control**.

---

## Bảng tổng hợp

| Method | Input signal | Granularity | Target-aware | Controls feedback | Proactive | Theoretical grounding |
|---|---|---|---|---|---|---|
| SDP (baseline) | — | Global γ | ❌ | ❌ | ❌ | Geometric interpolation |
| ASC | Gradient drift Δ | Global γ | ❌ | ❌ | ❌ | Sigmoid heuristic |
| CASC | Hessian eigenvectors | Subspace | ⚠️ | ❌ | ❌ | Curvature alignment |
| TSS | Past spectra | Per-SV | ❌ | ❌ | ❌ | EMA in log-space |
| FCSC | Fisher matrix | Global in Fisher space | ⚠️ | ❌ | ❌ | Natural gradient geometry |
| SPR | Current spectrum | Per-SV | ❌ | ❌ | ❌ | Max entropy |
| **GOSC** (novel) | **Gradient directions** | **Per-SV** | **✅** | ❌ | ❌ | **Subspace alignment** |
| **PSCO** (novel) | **Hessian + gradient** | **Global (optimized)** | **✅** | **✅** | ❌ | **Joint optimization** |
| **PSReg** (novel) | **Spectral velocity** | **Per-SV** | ❌ | ❌ | **✅** | **Linear prediction** |
| **SECDR** (novel) | **Gradient alignment** | **Per-SV** | **✅** | ❌ | ❌ | **Energy conservation** |
| **SLS** (novel) | **Spectral velocity** | **Global (principled)** | ❌ | **✅** | **✅** | **Lyapunov stability** |

---

## Đề xuất combination tối ưu

### Combo A: GOSC + PSReg (Target-aware + Proactive)

```
At boundary:
  1. Compute gradient alignment aᵢ (GOSC)
  2. Compute spectral velocity σ̇ᵢ (PSReg)
  3. Combined gamma:
     γᵢ = γ_base · (1 − aᵢ)          ← GOSC: nén orthogonal directions
           · (1 + clip(σ̇ᵢ/σ̄, −1, 2))  ← PSReg: nén mạnh SVs đang tăng
  4. Apply compression with per-SV γᵢ
```

### Combo B: SECDR + SLS (Energy conservation + Stability guarantee)

```
At boundary:
  1. Compute alignment → target spectrum σ'(SECDR)
  2. Check Lyapunov condition V(σ') ≤ ρ·V(σ)
  3. If violated, increase α (SECDR strength) until satisfied
  4. Apply
```

### Combo C: GOSC + PSCO (Target-aware + Feedback-aware)

```
At boundary:
  1. Compute alignment aᵢ (GOSC) → per-SV γᵢ candidates
  2. Evaluate PSCO objective ΔL(γ) for top candidates
  3. Select γ* that maximizes expected preconditioned loss reduction
  4. Apply
```

---

## Implementation Priority

| Rank | Method | Lý do |
|---|---|---|
| 1 | **GOSC** | Concept đơn giản, target-aware nhất, 1 gradient pass + SVD |
| 2 | **PSReg** | Zero extra compute (chỉ lưu SVs cũ), proactive |
| 3 | **SECDR** | Frobenius conservation + target-aware, solid lý thuyết |
| 4 | **SLS** | Lý thuyết đẹp nhất, biến SDP thành principled controller |
| 5 | **PSCO** | Chi phí cao nhất nhưng optimal theo lý thuyết |
