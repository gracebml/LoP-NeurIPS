# Experiment Design: Full Verification Plan for NeurIPS 2026

**Paper**: *Can the Optimizer Prevent Loss of Plasticity? Second-Order Methods Meet Spectral Diversity Preservation*  
**Target**: NeurIPS 2026 (Abstract: May 4, Full paper: May 6)  
**Created**: 2026-03-17  
**Updated**: 2026-03-19 (v2 — Redesigned EXP-2–6, added Fairness Protocol)  
**Execution window**: 10 ngày (Mar 19–29), 20 Kaggle accounts

---

## Tài nguyên & Ràng buộc

### Phân bổ GPU theo loại task

| Task type | GPU | Lý do |
|---|---|---|
| Deep network training (ResNet-34/50, WRN) | **H100** | Throughput cao, VRAM 80GB đủ |
| ViT training | **H100** | H100 tối ưu cho transformer |
| Shallow MLP/CNN training | **H100** | Nhanh hơn RTX Pro, VRAM đủ |
| Hessian spectral analysis (EXP-4) | **RTX Pro 6000** | Cần VRAM lớn cho Hessian vector products |
| Neural Collapse analysis (EXP-5) | **RTX Pro 6000** | Cần lưu activations toàn bộ → VRAM lớn |

---

## Notation chung

| Ký hiệu | Ý nghĩa |
|---|---|
| **1st-order** | SGD (momentum), Adam, AdamW |
| **2nd-order** | AdaHessian, SophiaH, SASSHA, KFAC-NGD |
| +SDP | Có áp dụng Spectral Diversity Preservation (γ=0.3) |
| −SDP | Không áp dụng SDP |
| BN | BatchNorm2d/1d |
| LN | LayerNorm |
| GN | GroupNorm |
| noBN | Không có normalization layer |

### Optimizers (7 total)

| # | Optimizer | Order | Hessian Approx | Reference |
|---|---|---|---|---|
| 1 | SGD (momentum) | 1st | — | — |
| 2 | Adam | 1st | — | Kingma & Ba, 2015 |
| 3 | AdamW | 1st | — | Loshchilov & Hutter, 2019 |
| 4 | AdaHessian | 2nd | Hutchinson diagonal | Yao et al., 2021 |
| 5 | SophiaH | 2nd | Hutchinson + clipping | Liu et al., 2024 |
| 6 | SASSHA | 2nd | SAM + Hutchinson | Shin et al., ICML 2025 |
| 7 | KFAC-NGD | 2nd | Kronecker-factored Fisher | Martens & Grosse, 2015 |

### Metrics tracking (tất cả thí nghiệm)

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| **Test Accuracy** | Accuracy trên test set tại mỗi epoch | Generalization |
| **Train Accuracy** | Accuracy trên train set | Overfitting indicator |
| **Overfitting Gap** | Train Acc − Test Acc | Sharp minima proxy |
| **Dormant Neuron Ratio** | % neurons có activation < 1% of layer mean | Plasticity proxy |
| **Stable Rank (weights)** | $\sum_i \sigma_i^2 / \sigma_1^2$ per layer | Weight spectral health |
| **Stable Rank (activations)** | Effective rank of activation matrix | Feature diversity |
| **Feature Rank (erank)** | $\exp(-\sum p_i \log p_i)$, $p_i = \sigma_i^2/\|W\|_F^2$ | Effective dimensionality |
| **Avg Weight Magnitude** | $\|W\|_F / \sqrt{mn}$ per layer | Weight growth |
| **Condition Number** | $\kappa(W) = \sigma_1 / \sigma_r$ per layer | Spectral health |
| **Hessian Top Eigenvalue** | $\lambda_{\max}(H)$ via power iteration | Sharpness |
| **Hessian Trace** | $\text{tr}(H)$ via Hutchinson estimator | Average curvature |

---

## EXP-1: Shallow Networks — Generalization of 2nd-Order Methods

### Giả thuyết

> Second-order methods on shallow networks **cannot maintain plasticity** (rank collapses, dormant neurons tăng) nhưng **generalize effectively** trên các tasks mà chúng vẫn học được (overfitting gap nhỏ hơn 1st-order).

### 1.1 — Architectures

| ID | Architecture | Params | Norm | Dataset | Continual Setting |
|---|---|---|---|---|---|
| S1 | MLP 784→256×2→10 (ReLU) | ~200K | noBN | Permuted MNIST | 10 tasks |
| S2 | MLP 784→512×2→10 (ReLU) | ~660K | noBN | Permuted MNIST | 100 tasks |
| S3 | ConvNet (conv×3→fc×2) | ~500K | noBN | ImageNet-32 binary | 5000 tasks × 200ep |
| S4 | 2-layer CNN (conv×2→fc) | ~300K | noBN | Split CIFAR-10 | 5 tasks, 2 classes each |
| S5 | MLP 784→512×3→10 (GELU) | ~1M | noBN | Permuted Fashion-MNIST | 20 tasks |

> **Constraint**: Tất cả architectures trong EXP-1 ≤ 1M params — đảm bảo đúng ngữ cảnh "shallow network".

### 1.2 — Grid

```
Optimizers: [SGD, Adam, AdamW, AdaHessian, SophiaH, SASSHA, KFAC-NGD]  (7)
SDP:        [off, on (γ=0.3)]                                           (2)
Seeds:      3
```

**Per architecture**: 7 × 2 × 3 = **42 runs**  
**Total**: 5 × 42 = **210 runs**

### 1.3 — Analysis Plan

**Table**: Final-task metrics (Test Acc, Dormant%, Stable Rank, Gap%, κ(W)) — 14 rows × 5 architectures

**Figure**: 5 archs × 3 panels (Test Acc, Dormant, Stable Rank vs. Task#), 4 curves = {1st best, 2nd best, 2nd+SDP best, 1st+SDP best}

**Kỳ vọng**: 2nd-order (−SDP) dormant **CAO** nhưng Gap **THẤP**; 2nd+SDP: mọi metric tốt nhất

### 1.4 — GPU Estimate

| Arch | Time/run | Subtotal |
|---|---|---|
| S1 (MLP 256×2, 200K) | ~20min | 14h |
| S2 (MLP 512×2, 660K) | ~30min | 21h |
| S3 (ConvNet, ImgNet32) | ~4h | 168h |
| S4 (2-layer CNN, 300K) | ~30min | 21h |
| S5 (MLP 512×3 GELU, 1M) | ~30min | 21h |
| **Total** | | **~245 GPU-h** |

---

## EXP-2: Deep Networks — Plasticity Maintenance vs. Overfitting

### Giả thuyết

> Second-order methods on deep networks with BN **automatically maintain plasticity** (dormant ≈ 0, stable rank cao) nhưng **overfit** (test accuracy thấp, gap lớn) do sharp minima convergence.

### 2.1 — Architectures

| ID | Architecture | Depth | Params | Norm | Dataset | Setting |
|---|---|---|---|---|---|---|
| D1 | ResNet-18 | 18L | 11M | BN | Inc. CIFAR-100 | 5→100 cls, +5/200ep |
| D2 | ResNet-34 | 34L | 21M | BN | Inc. CIFAR-100 | same |
| D3 | ResNet-50 | 50L | 25M | BN | Inc. CIFAR-100 | same |
| D5 | ResNet-18 | 18L | 11M | BN | Split TinyImageNet | 10→200 cls, +10/200ep |
| D7 | MLP 784→2000×5→10 | 5L | 18M | **noBN** | Permuted MNIST | 800 tasks, [SGD, Adam, SASSHA] ± SDP, 3 seeds = **18 runs** |
| D8 | ResNet-18 | 18L | 11M | **noBN** | Inc. CIFAR-100 | [SGD, Adam, SASSHA] ± SDP, 2 seeds = **12 runs** |

> **D7 rationale**: Deep network (18M params, 5 hidden layers) nhưng KHÔNG có BN. Paper predicts: dù deep, thiếu BN → outlier eigenvalues vẫn tồn tại → 2nd-order KHÔNG maintain plasticity. Đây là control experiment chứng minh BN là yếu tố then chốt, không phải depth đơn thuần.

> **D8 rationale**: ResNet-18 **cùng kiến trúc** D1 nhưng bỏ BN. So sánh trực tiếp D1 vs D8 chứng minh claim C10 rõ ràng nhất: cùng depth, cùng dataset, chỉ khác BN → 2nd-order mất plasticity.

> **Dropped**: D4 (VGG-16, 138M — quá nặng), D6 (WRN-28-10 — insight/cost thấp, ResNet-18/34/50 đã cover depth + params đa dạng).

### 2.2 — Depth Scaling Study (Critical)

```
Family:     ResNet + BN
Depths:     [ResNet-10, ResNet-14, ResNet-18, ResNet-26, ResNet-34, ResNet-50]
Dataset:    Incremental CIFAR-100
Optimizers: [SGD, AdamW, SASSHA]  (1st-simple, 1st-adaptive, 2nd-order)
SDP:        [off, on]
Seeds:      3
```

**Total**: 6 × 3 × 2 × 3 = **108 runs**

→ Figure: x = depth, y = {Dormant%, Test Acc, Gap%}, 6 curves (SGD/AdamW/SASSHA × ±SDP) → xác định **critical depth** và so sánh 1st-adaptive vs 2nd-order.

> **Why AdamW added**: SGD alone as 1st-order baseline bị reviewer challenge — AdamW là 1st-order adaptive phổ biến nhất, cần show nó cũng mất plasticity ở shallow / cũng maintain ở deep+BN.

### 2.3 — Main Grid

```
Architectures: [D1–D3, D5]
Optimizers:    [SGD, Adam, AdamW, AdaHessian, SophiaH, SASSHA, KFAC-NGD]  (7)
SDP:           [off, on]
Seeds:         3
```

**Per architecture (D1–D3, D5)**: 7 × 2 × 3 = **42 runs** × 4 archs = **168 runs**
**D7 (noBN MLP control)**: 3 × 2 × 3 = **18 runs**
**D8 (noBN ResNet control)**: 3 × 2 × 2 = **12 runs**
**Depth Scaling**: **108 runs**
**Total**: 168 + 18 + 12 + 108 = **306 runs**

### 2.4 — Extra Measurements

- Per-task test accuracy
- SAM-sharpness: $\max_{\|\epsilon\| \leq 0.05} \mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta)$
- Hessian top eigenvalue at end of each task

### 2.5 — GPU Estimate

| Setting | Time/run | Subtotal |
|---|---|---|
| Depth scaling (108 runs) | ~1.5h | 162h |
| D1 ResNet-18 (42 runs) | ~2h | 84h |
| D2 ResNet-34 (42 runs) | ~3h | 126h |
| D3 ResNet-50 (42 runs) | ~4h | 168h |
| D5 TinyImageNet (42 runs) | ~4h | 168h |
| D7 MLP 18M noBN (18 runs) | ~2h | 36h |
| D8 ResNet-18 noBN (12 runs) | ~2h | 24h |
| **Total** | | **~768 GPU-h** |

---

## EXP-3: SDP Ablation — Shallow vs. Deep, γ Sweep, Synergy

### Giả thuyết

> SDP giúp shallow nets **duy trì plasticity** (↓κ(H) → better preconditioning) và giúp deep nets **tránh sharp minima** (↓κ(W) → flatten landscape).

### 3.1 — γ Sweep: Shallow

```
Arch:       MLP hoặc ConvNet từ EXP-1 (chọn arch có LoP rõ nhất)
Dataset:    CIFAR-100 binary (200 tasks) hoặc ImageNet-32 binary (500 tasks)
Optimizers: [SGD, SASSHA]
γ:          {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
Seeds:      3
```

**42 runs**, track κ(W) and κ(H) before/after SDP at each task boundary.

> **Note**: Chọn architecture/dataset cụ thể sau khi có kết quả EXP-1. Ưu tiên config cho thấy LoP rõ ràng nhất ở 2nd-order.

### 3.2 — γ Sweep: Deep

```
Arch:       D1 (ResNet-18, BN)
Dataset:    Incremental CIFAR-100
Optimizers: [SGD, SASSHA]
γ:          {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
Seeds:      3
```

**42 runs** (giảm từ 63), track sharpness + κ(W) + $\lambda_{\max}(H)$.

> **Also measure NC metrics** (NC1/NC2/NC3) tại mỗi γ → merged từ EXP-5 Phase C. Không cần train riêng, chỉ thêm measurement code.

### 3.3 — Synergy Δ Analysis (POST-HOC — không cần runs mới)

$\Delta_{\text{SDP}} = \text{metric}_{+\text{SDP}} - \text{metric}_{-\text{SDP}}$ per optimizer.

Compare $\bar{\Delta}^{\text{1st}}$ vs. $\bar{\Delta}^{\text{2nd}}$. Kỳ vọng: ratio > 2.0×.

> **Tính trên data EXP-1 + EXP-2** (tất cả 7 optimizers × ±SDP đã chạy). Chỉ cần code analysis offline.
> 
> Report bảng: mỗi optimizer × mỗi metric (Test Acc, Dormant%, Stable Rank, Gap%) → Δ_SDP.

### 3.4 — GPU Estimate

| Setting | Time/run | Subtotal |
|---|---|---|
| Shallow γ sweep (42 runs) | ~2h | 84h |
| Deep γ sweep (42 runs) | ~2h | 84h |
| Hessian + NC overhead (+25%) | | +42h |
| **Total** | | **~210 GPU-h** |

---

## EXP-4: Hessian Spectral Density — SDP as Spectral BatchNorm

### Giả thuyết

> SDP khử outlier eigenvalues trong phổ Hessian, đóng vai trò tương tự BN. Phổ sau SDP ≈ phổ của mạng có BN.

### 4.1 — Method: Stochastic Lanczos Quadrature

Dùng `pyhessian` hoặc `google/spectral-density`:

```python
from pyhessian import hessian
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)
top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=50)
density_eigen, density_weight = hessian_comp.density(iter=100, n_v=1)
```

### 4.2 — Condition Comparison

**Shallow (MLP từ EXP-1)**:
```
Conditions: [noBN+noSDP, noBN+SDP]  (2 conditions — BN vô nghĩa cho MLP)
Optimizers: [SGD, SASSHA]
Snapshots:  [Task 1, 25, 50, 100+]
Seeds:      2
→ 8 training runs
```

> MLP thường KHÔNG có BN trong practice. Thêm BN vào MLP tạo architecture non-standard, khó justify cho reviewer. Chỉ so sánh noSDP vs +SDP.

**Deep (ResNet-18)**:
```
Conditions:
  A) BN, noSDP      (baseline — BN smooths Hessian)
  B) BN, +SDP       (BN + SDP)
  C) noBN, noSDP    (outlier eigenvalues)
  D) noBN, +SDP     (SDP replaces BN — HYPOTHESIS)

Optimizers: [SGD, SASSHA]
Snapshots:  [Task 1, 5, 10, 20]
Seeds:      2
→ 16 training runs
```

**Total**: 24 runs → Hessian snapshots tại mỗi checkpoint

### 4.3 — Key Figures

**Fig 4.1**: 2×4 grid (rows: Shallow/Deep, cols: task snapshots). Shallow panels: 2 spectral curves (noSDP/+SDP). Deep panels: 4 curves A/B/C/D.

**Fig 4.2**: Outlier ratio $\lambda_{\max}/\text{median}(\lambda)$ vs. Task index.

**Fig 4.3**: Gradient concentration $\|P_k g\|^2 / \|g\|^2$ for top-k eigenspaces.

### 4.4 — Quantitative Metrics

| Metric | Công thức |
|---|---|
| Outlier ratio | $\lambda_{\max} / \text{median}(\lambda)$ |
| Spectral gap | $\lambda_1 - \lambda_2$ |
| Spectral entropy | $-\sum p_i \log p_i$ |
| Gradient conc. | $\|P_{10} g\|^2 / \|g\|^2$ |

**KS-test**: compare $D_{\text{SLQ}}^{(\text{noBN+SDP})}$ vs. $D_{\text{SLQ}}^{(\text{BN})}$. p > 0.05 → SDP ≈ BN spectrally.

### 4.5 — GPU Estimate

| Setting | Subtotal |
|---|---|
| Training (24 runs × ~2h) | 48h |
| Hessian computation (snapshots × ~10min) | 20h |
| **Total** | **~68 GPU-h** |

---

## EXP-5: Neural Collapse Acceleration

### Giả thuyết

> 2nd-order methods **accelerate NC** → rigid ETF features → overfit. SDP counteracts NC rigidity.

### 5.1 — NC Metrics

| Metric | Ý nghĩa | → 0 khi |
|---|---|---|
| NC1: $\frac{1}{C}\sum_c \text{tr}(\Sigma_c^W)/\text{tr}(\Sigma_B)$ | Within-class variability | Collapse |
| NC2: $\|\hat{M}\hat{M}^\top - \frac{C}{C-1}(I-\frac{1}{C}\mathbf{11}^\top)\|_F$ | ETF structure | Perfect ETF |
| NC3: $\|W_{\text{cls}}/\|W_{\text{cls}}\| - \hat{M}\|_F$ | Self-duality | Aligned |
| Feature erank | Effective dim of last-layer features | Low-rank |

### 5.2 — Phase A: NC Speed (Static) — REDUCED

```
Arch: ResNet-18 + BN, CIFAR-100 (static, non-continual)
Opts: [SGD, SASSHA]  (1st vs 2nd representative)
Epochs: 300, measure NC metrics every 10ep
Seeds: 2 → 4 runs
```

> NC measurement ít variance → 2 seeds đủ. Chỉ cần SGD vs SASSHA để show 2nd-order converge NC faster. Hữu ích làm **calibration** trước Phase B.

### 5.3 — Phase B: NC in Continual Learning — EXPANDED

```
Arch: ResNet-18 + BN, Inc. CIFAR-100 (20 tasks)
Opts: [SGD, SGD+SDP, SASSHA, SASSHA+SDP]  (complete 2×2 matrix)
Measure NC at end of each task
Seeds: 3 → 12 runs
```

> Thêm SGD+SDP để hoàn thành 2×2 matrix {1st/2nd} × {±SDP}.

### 5.4 — Phase C: MERGED vào EXP-3

> Phase C (γ sweep NC metrics) **không cần training riêng**. Thêm NC measurement code vào EXP-3 γ sweep deep runs (Section 3.2). NC1/NC2/NC3 computed at task boundaries cùng lúc với κ(W), κ(H).

### 5.5 — GPU Estimate

| Phase | Subtotal |
|---|---|
| A (4 runs × 1h) | 4h |
| B (12 runs × 2h) | 24h |
| C merged | 0h |
| NC overhead (+20%) | 6h |
| **Total** | **~34 GPU-h** |

---

## EXP-6: Architecture Agnosticism — Normalization Variants + ViT

### Giả thuyết

> SDP hiệu quả bất kể normalization layer. LN (như ViT) KHÔNG smooth Hessian giống BN → SDP đặc biệt cần thiết cho ViT/LN architectures.

### 6.1 — Variants

| ID | Architecture | Norm | Dataset | Setting | Source |
|---|---|---|---|---|---|
| N1 | ResNet-18 | BN | Inc. CIFAR-100 | 20 tasks | **REUSE** EXP-2 D1 data |
| N2 | ResNet-18 | noBN | Inc. CIFAR-100 | 20 tasks | **REUSE** EXP-2 D8 data |
| N3 | ResNet-18 | LN | Inc. CIFAR-100 | 20 tasks | NEW |
| N4 | ResNet-18 | GN (g=32) | Inc. CIFAR-100 | 20 tasks | NEW |
| N5 | ViT-Tiny (p=4) | LN | Inc. CIFAR-100 | 20 tasks | NEW |
| N6 | ViT-Tiny (p=4) | noLN | Inc. CIFAR-100 | 20 tasks | NEW |
| N7 | MLP 784→2000×5→10 | noBN | Permuted MNIST | 100 tasks | **REUSE** EXP-2 D7 data |
| N8 | MLP 784→2000×5→10 | LN | Permuted MNIST | 100 tasks | NEW |

> **N1/N2/N7 reuse** data từ EXP-2 (D1/D8/D7). Tránh duplicate runs. Chỉ cần thêm comparison code.

### 6.2 — Grid (NEW runs only)

```
New architectures: [N3, N4, N5, N6, N8]  (5 variants)
Optimizers:        [SGD, AdamW, SASSHA]  (1st-simple, 1st-adaptive, 2nd-order)
SDP:               [off, on]
Seeds:             3
```

**Per variant**: 3 × 2 × 3 = **18 runs** × 5 variants = **90 NEW runs**

> **AdamW thêm vào**: 1st-order adaptive baseline cần thiết — reviewer sẽ hỏi *"Adam/AdamW với LN thì sao?"*. SGD alone as 1st-order bị challenge.

### 6.3 — GPU Estimate

| Setting | Subtotal |
|---|---|
| N3, N4 ResNet-18 LN/GN (36 runs × 2h) | 72h |
| N5, N6 ViT-Tiny (36 runs × 3h) | 108h |
| N8 MLP+LN (18 runs × 1h) | 18h |
| Hessian spectral for LN vs BN vs noBN | 12h |
| **Total** | **~210 GPU-h** |

---

## Tổng hợp Budget (Optimized for 10-day Sprint)

### Scope Adjustment for 10-day Target (v2 — Redesigned)

| Change | Rationale | Impact |
|---|---|---|
| **Drop D4 (VGG-16)** | 138M params, non-standard cho LoP | −126h |
| **Drop D6 (WRN-28-10)** | Insight/cost thấp, ResNet-18/34/50 đủ | −36h |
| **Add D8 (ResNet-18 noBN)** | Control cho C10: BN key not depth | +24h |
| **Add AdamW to Depth Scaling** | Reviewer cần adaptive 1st-order baseline | +54h |
| **EXP-3 Synergy → post-hoc** | Tính từ EXP-1/2 data, 0 extra runs | −63h |
| **EXP-4 giảm 48→24 runs** | Bỏ BN conditions cho MLP, bỏ ConvNet | −60h |
| **EXP-5 giảm 39→16 runs** | Phase A/2 optimizers, Phase C merged | −42h |
| **EXP-6 reuse N1/N2/N7** | Tránh duplicate | −20h |
| **Add LR grid search** | Fairness protocol bắt buộc | +80h |

### Final Budget

| Experiment | Runs | GPU-h (raw) | On H100 (adj.) | Priority |
|---|---|---|---|---|
| EXP-1: Shallow generalization (all ≤1M params) | 210 | 245h | **160h** | P0 |
| EXP-2: Deep + depth scaling + D7/D8 noBN controls | 306 | 768h | **450h** | P0 |
| EXP-3: SDP ablation (γ sweep, synergy post-hoc) | 84 | 210h | **140h** | P0 |
| EXP-4: Hessian spectral (simplified) | 24 | 68h | **50h** | P1 |
| EXP-5: Neural Collapse (reduced, Phase C merged) | 16 | 34h | **25h** | P1 |
| EXP-6: Arch agnostic (reuse + AdamW) | 90 | 210h | **145h** | P1 |
| LR grid search (fairness) | ~50 | 80h | **55h** | P0 |
| **TOTAL** | **~780** | **~1,615h** | **~1,025h** | |

---

## Hyperparameter Fairness Protocol (CRITICAL)

> **Đây là yếu tố QUAN TRỌNG NHẤT cho reviewer acceptance.** Reviewer sẽ challenge nếu không document rõ cách tuning hyperparameters.

### Tier 1 — Reference HP từ paper gốc

Mỗi optimizer dùng lr/wd recommended từ paper gốc, adjusted cho dataset/architecture.

### Tier 2 — Light LR Grid Search (BẮT BUỘC)

```
Cho mỗi {optimizer × architecture}, sweep 3 lr values:
  lr ∈ {lr_default/3, lr_default, lr_default×3}
Dùng 1 seed, pick best lr by final-task test accuracy.
Cost: ~50 extra runs ≈ ~80h
```

### Tier 3 — Report Sensitivity

Paper appendix: bảng test accuracy cho 3 lr values per optimizer → show kết quả robust.

### Weight Decay Convention

| Optimizer Type | Weight Decay | Lý do |
|---|---|---|
| SGD | 1e-4 hoặc 5e-4 | Standard |
| AdamW | 0.01 (decoupled) | Loshchilov convention |
| Adam | 0.0 | L2 reg ≠ weight decay trong Adam |
| 2nd-order (AdaHessian, SophiaH, SASSHA) | 0 hoặc rất nhỏ | Hessian preconditioner implicit regularize |
| KFAC | 0.0 | Damping regularize |

### Optimizer State Reset

Optimizer được tạo mới mỗi task → tất cả states reset (momentum, Hessian estimates, Fisher factors). **NHẤT QUÁN** cho tất cả experiments.

### Learning Rate Schedule

Dùng **constant lr** (không decay). Justify trong paper: *"We use constant learning rates across all tasks to isolate the optimizer effect from learning rate scheduling."*

---

## Tài nguyên: 20 Kaggle Accounts × 10 Ngày

### GPU Fleet

| Resource | Specs | Quota |
|---|---|---|
| Kaggle H100 | 80GB VRAM, ~1.7× throughput so RTX Pro | 30h/account/tuần |
| Kaggle RTX Pro 6000 | 95GB VRAM, VRAM lớn hơn | 30h/account/tuần |

### Capacity Calculation

```
10 ngày = spans 2 weekly reset cycles
  Cycle 1 (Day 1–7): 30h/account → 20 × 30 = 600h
  Cycle 2 (Day 8–10): quota reset, 3 days remaining
    → mỗi account chạy ~2 sessions × 10h = 20h
    → 20 × 20 = 400h

Total raw GPU-hours available: 600 + 400 = 1,000h
Session constraint: max 12h/session, ~2 sessions/day practical

With H100 throughput adjustment (avg 1.5×): ~1,000 × 1.1 (blended) = ~1,100 effective hours
```

**Budget needed: ~1,070h effective** → **Fits. ~30h buffer cho failures/reruns.**

→ **Strategy**: Tất cả 20 accounts chạy **H100** (nhanh hơn) trừ 3 accounts cần VRAM > 80GB (Hessian analysis, NC activations) dùng RTX Pro 6000.

---

## Phân bổ 20 Kaggle Accounts

### Naming Convention

```
Account:  K01 ... K20
Notebook: K{xx}_{exp}_{detail}.ipynb
```

### Master Assignment Table

Mỗi account target: **50–55h trong 10 ngày** (30h tuần 1 + 20h tuần 2).

| Acc | GPU | EXP | Assignment | Runs | ~Hours | Sessions |
|---|---|---|---|---|---|---|
| | | | **EXP-1: SHALLOW ≤1M (160h adj.)** | | | |
| K01 | H100 | 1 | S3 ConvNet ImgNet-32 seed 1 + S2 MLP all (42 runs) | 56 | 45h | 5 |
| K02 | H100 | 1 | S3 ConvNet seed 2 + S5 GELU all (42 runs) | 56 | 45h | 5 |
| K03 | H100 | 1 | S3 ConvNet seed 3 + S1 all (42) + S4 all (42) | 56 | 53h | 5 |
| | | | **EXP-2: DEEP (420h adj.)** | | | |
| K04 | H100 | 2 | Depth scaling: ResNet-{10..50} × SGD/SASSHA ± SDP | 72 | 49h | 5 |
| K05 | H100 | 2 | D1 ResNet-18 CIFAR-100: all 7 opts × 2 SDP × 3 seeds | 42 | 50h | 5 |
| K06 | H100 | 2 | D2 ResNet-34: seeds 1,2 | 28 | 50h | 5 |
| K07 | H100 | 2 | D2 ResNet-34: seed 3 + D6 WRN reduced + D7 MLP18M noBN | 38 | 55h | 6 |
| K08 | H100 | 2 | D3 ResNet-50: seed 1 | 14 | 33h | 3 |
| K09 | H100 | 2 | D3 ResNet-50: seed 2 | 14 | 33h | 3 |
| K10 | H100 | 2 | D3 ResNet-50: seed 3 | 14 | 33h | 3 |
| K11 | H100 | 2 | D5 TinyImageNet: seed 1 | 14 | 33h | 3 |
| K12 | H100 | 2 | D5 TinyImageNet: seed 2 | 14 | 33h | 3 |
| K13 | H100 | 2 | D5 TinyImageNet: seed 3 | 14 | 33h | 3 |
| | | | **EXP-3: SDP ABLATION (175h adj.)** | | | |
| K14 | H100 | 3 | γ sweep shallow (42 runs) + Hessian κ tracking | 42 | 48h | 5 |
| K15 | H100 | 3 | γ sweep deep (63 runs) + sharpness tracking | 63 | 55h | 6 |
| | | | **EXP-4/5/6: ANALYSIS + ARCH (315h adj.)** | | | |
| K16 | RTX Pro | 4 | Hessian spectral density: MLP + ResNet conditions | 32 | 50h | 5 |
| K17 | RTX Pro | 4+5 | Hessian spectral: ConvNet + NC Phase A (static) | 31 | 48h | 5 |
| K18 | RTX Pro | 5 | NC Phase B+C (continual) + gradient alignment | 24 | 45h | 5 |
| K19 | H100 | 6 | ResNet-18 norm variants (N1–N4) + ViT-Tiny (N5–N6) | 72 | 53h | 5 |
| K20 | H100 | 6+buf | MLP norms (N7–N8) + **BACKUP/RERUN** | 24+ | 30h+ | 3+ |

### Summary per Account

```
Account  Hours  Role                                    Status by Day
──────── ────── ──────────────────────────────────────── ──────────────
K01       45h   EXP-1 S3 seed 1 + S2 all                Done Day 7
K02       45h   EXP-1 S3 seed 2 + S5 all                Done Day 7
K03       53h   EXP-1 S3 seed 3 + S1 + S4 all           Done Day 8
K04       49h   EXP-2 Depth scaling                      Done Day 7
K05       50h   EXP-2 D1 ResNet-18 all seeds             Done Day 7
K06       50h   EXP-2 D2 ResNet-34 seeds 1-2             Done Day 7
K07       55h   EXP-2 D2 s3 + D6 WRN + D7 MLP noBN      Done Day 8
K08       33h   EXP-2 D3 ResNet-50 seed 1                Done Day 5 → BACKUP
K09       33h   EXP-2 D3 ResNet-50 seed 2                Done Day 5 → BACKUP
K10       33h   EXP-2 D3 ResNet-50 seed 3                Done Day 5 → BACKUP
K11       33h   EXP-2 D5 TinyImageNet seed 1             Done Day 5 → BACKUP
K12       33h   EXP-2 D5 TinyImageNet seed 2             Done Day 5 → BACKUP
K13       33h   EXP-2 D5 TinyImageNet seed 3             Done Day 5 → BACKUP
K14       48h   EXP-3 γ sweep shallow                    Done Day 7
K15       55h   EXP-3 γ sweep deep                       Done Day 8
K16       50h   EXP-4 Hessian spectral (half)            Done Day 7
K17       48h   EXP-4+5 Hessian + NC Phase A             Done Day 7
K18       45h   EXP-5 NC Phase B+C                       Done Day 7
K19       53h   EXP-6 ResNet norms + ViT                 Done Day 8
K20       30h+  EXP-6 MLP norms + BACKUP                 Rolling
```

### Early Finishers → Backup Pool

6 accounts (K08–K13) xong sớm Day 5 vì mỗi account chỉ chạy 1 seed (~33h). Đây là **backup pool lớn = 6 × 27h remaining = ~162h extra GPU**.

| Priority | Backup Task | Account gợi ý |
|---|---|---|
| 1 | Rerun failures từ bất kỳ EXP | K08, K09, K10 |
| 2 | CBP + LN+WD baselines (CIFAR-100, MNIST) | K11, K12 |
| 3 | Extra seeds (seed 4) cho SASSHA+SDP, SophiaH+SDP | K08, K09 |
| 4 | D7 full grid (all 7 opts) nếu reduced 12 runs không đủ | K13 |
| 5 | Thêm configs cho EXP-1/2 | K10, K11 |

---

## 10-Day Execution Plan

### Day-by-Day Schedule

```
┌─────────────────────────────────────────────────────────────────────────┐
│ DAY 0 (Mar 19 — Setup)                                                 │
│                                                                         │
│  ALL: Upload shared dataset `lop-experiment-configs` to Kaggle          │
│  K05: Pilot run — 1 config (SASSHA+SDP, ResNet-18, CIFAR-100, seed 1)  │
│       Verify: metrics collection, output format, GPU utilization         │
│  ALL: Clone runner notebooks to each account                             │
│  TARGET: Pilot run completes, bugs fixed, ready for Day 1               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DAY 1 (Mar 20 — Full Launch)                    Quota: 30h/acc Week 1   │
│                                                                         │
│  ALL 20 ACCOUNTS: Start Session 1 (12h)                                 │
│  K01–K04:  EXP-1 shallow begins                                        │
│  K05–K13:  EXP-2 deep begins                                           │
│  K14–K15:  EXP-3 γ sweeps begin                                        │
│  K16–K18:  EXP-4+5 Hessian + NC begins                                 │
│  K19–K20:  EXP-6 arch agnostic begins                                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DAY 2 (Mar 21)                                                          │
│                                                                         │
│  ALL: Session 2 (12h) — continue from Day 1                             │
│  Monitor: Check K05 pilot results as sanity check                        │
│  TARGET: ~40% of quota used (12h/acc)                                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DAY 3 (Mar 22)                                                          │
│                                                                         │
│  ALL: Session 3 begins                                                   │
│  K06 approaching completion (D1 ResNet-18 seeds 1-2)                     │
│  TARGET: ~60% of quota used (18h/acc)                                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DAY 4 (Mar 23)                                                          │
│                                                                         │
│  ALL: Session 4 — approaching Week 1 quota limit                         │
│  TARGET: ~80% of quota used (24h/acc)                                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DAY 5 (Mar 24)                                                          │
│                                                                         │
│  ★ 6 EARLY FINISHERS: K08–K13 complete (33h each, single-seed tasks)     │
│  → Download D3/D5 partial results → sanity check                        │
│  → K08–K13 enter BACKUP POOL: reruns, CBP baselines, extra seeds        │
│  Remaining accounts: approach 30h limit for Week 1                       │
│  TARGET: Week 1 quota fully consumed (30h/acc)                           │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DAY 6 (Mar 25)                                                          │
│                                                                         │
│  ⚠ QUOTA RISK: Heavier accounts hit 30h limit → pause until reset       │
│  Lighter accounts (K08–K13) already in backup mode with remaining quota │
│  → FIRST RESULTS: Download K05 (D1 ResNet-18) → sanity check tables    │
│  → Begin preliminary figure generation                                   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DAY 7 (Mar 26)                                 ★ WEEKLY QUOTA RESET ★   │
│                                                                         │
│  ★ ALL ACCOUNTS: Fresh 30h quota available                               │
│  COMPLETIONS: K01, K02, K04, K05, K06, K14, K16, K17, K18               │
│  → Download completed results                                            │
│  → EXP-1 S2, S5 DONE; S3 seeds 1,2 DONE                                │
│  → EXP-2 D1, D2(s1-2), depth scaling DONE                               │
│  → EXP-3 shallow DONE → preliminary γ sweep figure                      │
│  → EXP-4 Hessian spectral largely DONE                                   │
│  → EXP-5 NC largely DONE                                                 │
│  Remaining: K03, K07, K15, K19 continue into Week 2                     │
│  Backup pool (K08–K13): fresh quota for reruns + extra seeds              │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DAY 8 (Mar 27)                                                          │
│                                                                         │
│  COMPLETIONS: K03 (S3 s3+S1+S4), K07 (D2 s3+D6+D7), K19 (EXP-6)       │
│  → EXP-1 COMPLETE (all S1–S5, all seeds)                                 │
│  → EXP-2 D2, D6, D7 COMPLETE                                            │
│  → EXP-6 COMPLETE                                                        │
│  Backup pool running: extra seeds, CBP baselines, failure reruns          │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DAY 9 (Mar 28)                                                          │
│                                                                         │
│  COMPLETIONS: K15 (γ sweep deep)                                        │
│  → EXP-3 COMPLETE                                                        │
│  → Generate γ sweep figures, normalization comparison table               │
│  → All core experiments DONE. Backup accounts finish extra seeds/reruns  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ DAY 10 (Mar 29 — DEADLINE)                                              │
│                                                                         │
│  ★ ALL EXPERIMENTS COMPLETE                                              │
│  Final downloads, aggregation, cross-validation                          │
│  Generate all publication figures and tables                              │
│  Verify: every checklist item has data                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Critical Path (Bottlenecks)

```
CRITICAL PATH (determines overall completion):

  K07 (D2 s3 + D6 + D7, 55h) ─── Day 8
  K15 (γ sweep deep, 55h) ─── Day 8–9
  K03 (S3 seed 3 + S1 + S4, 53h) ─── Day 8
  K19 (EXP-6 ResNet norms + ViT, 53h) ─── Day 8

These 4 accounts are the bottlenecks (heaviest workloads). If any fail:
  → 6 backup accounts (K08–K13) have ~27h remaining each after Day 5
  → Total backup capacity: ~162h → can absorb any single failure
  → Priority: K07 (D2+D7 data) > K15 (γ sweep) > K03 (S3) > K19 (EXP-6)
```

---

## File Organization trên Kaggle

### Shared Kaggle Dataset: `lop-experiment-configs`

Mỗi account attach dataset này → truy cập shared code + configs:

```
lop-experiment-configs/
├── configs/
│   ├── exp1/
│   │   ├── s1_sgd_nosdp_s1.json ... s5_kfac_sdp_s3.json
│   ├── exp2/
│   │   ├── d1_sgd_nosdp_s1.json ... depth_resnet50_sassha_sdp_s3.json
│   ├── exp3/ ... exp6/
│   └── assignments/
│       ├── K01.json    ← List of config files assigned to K01
│       ├── K02.json
│       └── ...K20.json
│
├── src/
│   ├── optimizers/
│   │   ├── adahessian.py
│   │   ├── sophiaH.py
│   │   ├── sassha.py
│   │   └── kfac_ngd.py
│   ├── metrics/
│   │   ├── plasticity.py       # dormant, stable rank, erank, κ(W)
│   │   ├── hessian_tools.py    # spectral density, eigenvalues
│   │   └── neural_collapse.py  # NC1, NC2, NC3
│   ├── sdp.py
│   ├── runner.py               # Universal experiment runner
│   ├── models/
│   │   ├── resnet_variants.py
│   │   ├── vit_tiny.py
│   │   ├── mlp.py
│   │   └── convnet.py
│   └── data/
│       ├── cifar100_incremental.py
│       ├── mnist_permuted.py
│       ├── imagenet32_binary.py
│       ├── tinyimagenet_split.py
│       └── fashion_mnist_permuted.py
│
└── lop-src/                    ← CBP paper source code
```

### Universal Runner Notebook

Mỗi account dùng **cùng 1 notebook template**, chỉ thay `ACCOUNT_ID`:

```python
# ── runner.ipynb (identical across all accounts) ──

import json, os, sys
sys.path.insert(0, '/kaggle/input/lop-experiment-configs/src')
from runner import run_experiment, save_results

ACCOUNT_ID = "K05"  # ← CHỈ THAY DÒNG NÀY

# Load assignment
assignment = json.load(open(
    f"/kaggle/input/lop-experiment-configs/configs/assignments/{ACCOUNT_ID}.json"
))

# Run all assigned configs sequentially
for config_path in assignment["configs"]:
    cfg = json.load(open(f"/kaggle/input/lop-experiment-configs/configs/{config_path}"))
    print(f"[{ACCOUNT_ID}] Running: {config_path}")
    run_experiment(cfg)
    save_results(cfg)
    print(f"[{ACCOUNT_ID}] Done: {config_path}")
```

### Assignment JSON (per account)

```json
// configs/assignments/K06.json
{
  "account": "K06",
  "gpu": "H100",
  "experiment": "exp2",
  "description": "D1 ResNet-18 CIFAR-100, seeds 1-2",
  "configs": [
    "exp2/d1_sgd_nosdp_s1.json",
    "exp2/d1_sgd_sdp_s1.json",
    "exp2/d1_adam_nosdp_s1.json",
    "exp2/d1_adam_sdp_s1.json",
    "exp2/d1_adamw_nosdp_s1.json",
    "exp2/d1_adamw_sdp_s1.json",
    "exp2/d1_adahessian_nosdp_s1.json",
    "exp2/d1_adahessian_sdp_s1.json",
    "exp2/d1_sophiah_nosdp_s1.json",
    "exp2/d1_sophiah_sdp_s1.json",
    "exp2/d1_sassha_nosdp_s1.json",
    "exp2/d1_sassha_sdp_s1.json",
    "exp2/d1_kfac_nosdp_s1.json",
    "exp2/d1_kfac_sdp_s1.json",
    "exp2/d1_sgd_nosdp_s2.json",
    "... (seed 2 repeats)"
  ],
  "estimated_hours": 36,
  "sessions_needed": 4,
  "session_plan": [
    {"session": 1, "configs_idx": [0,4],  "est_hours": "10h"},
    {"session": 2, "configs_idx": [5,9],  "est_hours": "10h"},
    {"session": 3, "configs_idx": [10,18], "est_hours": "10h"},
    {"session": 4, "configs_idx": [19,27], "est_hours": "6h"}
  ]
}
```

### Output Convention

```
/kaggle/working/output/
├── exp2_d1_sgd_nosdp_s1/
│   ├── metrics.pkl
│   ├── checkpoints/task_{01,05,10,20}.pt
│   ├── config.json
│   └── summary.json
├── exp2_d1_sgd_sdp_s1/
│   └── ...
└── ...
```

### Result Aggregation (local, after download)

```
research/results/
├── raw/K01/ ... K20/         ← Download from each account
├── merged/
│   ├── exp1_all.pkl
│   ├── exp2_all.pkl
│   └── ...
├── figures/
│   ├── fig1_main_cifar100.pdf
│   ├── fig2_depth_scaling.pdf
│   ├── fig3_shallow_mnist.pdf
│   ├── fig4_hessian_spectral.pdf
│   ├── fig5_gamma_sweep.pdf
│   ├── fig6_nc_acceleration.pdf
│   └── fig7_norm_variants.pdf
└── tables/
    ├── tab1_cifar100.tex
    ├── tab2_mnist.tex
    └── tab3_rl.tex
```

---

## Checklist trước khi submit

- [ ] EXP-1: ≥3 shallow archs showing rank collapse + generalization for 2nd
- [ ] EXP-2: ≥3 deep archs (D1,D2,D3) showing plasticity + overfitting for 2nd
- [ ] EXP-2: D5 TinyImageNet confirming cross-dataset
- [ ] EXP-2: D7 (18M MLP noBN) confirming BN required for 2nd-order plasticity
- [ ] EXP-2: D8 (ResNet-18 noBN) confirming BN key, not depth
- [ ] EXP-2: Depth scaling figure (ResNet-{10..50}) with SGD/AdamW/SASSHA
- [ ] EXP-3: γ sweep confirming optimal γ ≈ 0.3 trên cả shallow và deep
- [ ] EXP-3: Synergy ratio Δ²ⁿᵈ / Δ¹ˢᵗ > 2.0 (post-hoc from EXP-1/2)
- [ ] EXP-3: NC metrics at each γ (merged from EXP-5 Phase C)
- [ ] EXP-4: Hessian spectral density plots: Shallow (noSDP/+SDP), Deep (4 conditions)
- [ ] EXP-4: Gradient concentration analysis
- [ ] EXP-4: KS-test p-values: SDP ≈ BN spectrally
- [ ] EXP-5: NC speed comparison (2nd faster than 1st) — Phase A
- [ ] EXP-5: NC accumulation across continual tasks — Phase B (2×2 matrix)
- [ ] EXP-6: SDP works with LN, GN, noBN on ResNet-18 (N3/N4 + reuse N1/N2)
- [ ] EXP-6: ViT-Tiny with LN + SDP (N5/N6)
- [ ] EXP-6: AdamW included as 1st-order adaptive baseline
- [ ] Fairness: LR grid search completed for all {optimizer × architecture}
- [ ] Fairness: WD and LR schedule documented in paper
- [ ] All results: 3 seeds (2 for Hessian), std reported
- [ ] All figures: Publication quality
- [ ] Statistical significance tests for key claims
- [ ] All accounts' results aggregated
