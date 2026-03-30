# Proposed Methods: Reducing Overfitting in Second-Order Optimizers

## Bối cảnh vấn đề

Second-order methods (Shampoo, SOAP, K-FAC) sử dụng thông tin curvature để precondition gradient, giúp:
- **Tăng plasticity:** Gradient được chuẩn hóa theo curvature → mạng thích nghi tốt hơn với task mới
- **Hội tụ nhanh hơn:** Bước cập nhật "thông minh" hơn SGD/Adam

**Tuy nhiên**, chính vì hội tụ nhanh và chính xác theo curvature, second-order methods có xu hướng **rơi vào sharp minima** — vùng cực tiểu có Hessian eigenvalue lớn. Tại sharp minima:
- Training loss thấp nhưng **generalization gap lớn**
- Mô hình nhạy cảm với nhiễu nhỏ trên input → overfitting
- Trong continual learning: memorize task hiện tại nhưng không generalize sang task mới

**Mục tiêu:** Giữ lợi ích plasticity của second-order methods, đồng thời hướng optimization vào **flat minima** (vùng cực tiểu phẳng, generalize tốt hơn).

---

## Danh sách phương pháp đề xuất

### 1. SASSHA — Sharpness-Aware Adaptive Second-Order Optimization

**Nguồn:** Shin et al., **ICML 2025** — [arXiv:2502.18153](https://arxiv.org/abs/2502.18153)

**Vấn đề giải quyết:** Second-order methods hội tụ đến sharp minima → generalization kém hơn SGD.

**Ý tưởng:** Tích hợp trực tiếp **SAM (Sharpness-Aware Minimization)** vào second-order optimization. Thay vì chỉ minimize loss L(w), minimize cả worst-case loss trong lân cận:

```
min_w  max_{||ε||≤ρ}  L(w + ε)
```

kết hợp với Hessian preconditioning từ second-order method.

**Cơ chế chính:**
1. Tính perturbation ε theo hướng tăng loss nhanh nhất (SAM step)
2. Tính gradient tại w + ε (ascent step)
3. Precondition gradient này bằng Hessian approximation (Shampoo/SOAP)
4. Cập nhật weight theo hướng preconditioned
5. **Stable Hessian Approximation:** Ổn định Hessian computation dọc trajectory
6. **Lazy Hessian updates:** Không cần cập nhật Hessian mỗi step → giảm overhead

**Ưu điểm:**
- Trực tiếp nhắm vào sharpness — nguyên nhân gốc của overfitting trong second-order methods
- Tương thích với lazy preconditioner updates (T_precond > 1)
- Có code chính thức: [GitHub](https://github.com/log-postech/sassha)

**Độ phù hợp với SOAP-lite-OGP:** ★★★★★ — Rất cao. SASSHA được thiết kế chính xác cho vấn đề này. Có thể tích hợp SAM perturbation vào SOAP-lite step.

**Complexity bổ sung:** O(mn) per step (thêm 1 forward-backward cho SAM perturbation).

---

### 2. IRE — Implicit Regularization Enhancement

**Nguồn:** NeurIPS 2024 — [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d712c8625fd97424c9744019b28dca21-Abstract-Conference.html)

**Vấn đề giải quyết:** Optimizer hội tụ chậm đến flat minima → cần nhiều epoch → overfitting trước khi đạt flat region.

**Ý tưởng:** **Tách rời** dynamics theo hướng "phẳng" và hướng "nhọn" trong không gian tham số:
- Hướng phẳng (flat direction): Tăng tốc sharpness reduction
- Hướng nhọn (sharp direction): Giữ ổn định training

**Cơ chế chính:**
1. Ước lượng hướng sharp/flat từ gradient statistics
2. Tăng learning rate dọc theo hướng flat → đẩy nhanh escape khỏi sharp region
3. Giảm learning rate dọc theo hướng sharp → ổn định hội tụ
4. Hoạt động như wrapper quanh base optimizer (plugin)

**Ưu điểm:**
- Không thêm hyperparameter phức tạp
- Tăng tốc 2× so với AdamW trên Llama pretraining
- Lý thuyết chứng minh tăng tốc hội tụ đến flat minima trong SAM
- Plugin — tương thích với bất kỳ base optimizer nào

**Độ phù hợp với SOAP-lite-OGP:** ★★★★☆ — Cao. SOAP-lite đã có eigenbasis Q_L, Q_R → có thể dùng trực tiếp để xác định sharp/flat directions từ eigenvalues mà không cần tính thêm.

**Complexity bổ sung:** O(m² + n²) — tận dụng eigendecomp có sẵn từ Shampoo.

---

### 3. NSO — Noise Stability Optimization

**Nguồn:** Zhang et al., **ICLR 2025** — [arXiv:2306.08553](https://arxiv.org/abs/2306.08553)

**Vấn đề giải quyết:** Sharp minima có Hessian trace lớn → cần regularize Hessian trace mà không tính Hessian đầy đủ.

**Ý tưởng:** Inject **isotropic Gaussian noise** vào weight để ước lượng Hessian trace, rồi dùng nó làm regularization term.

**Cơ chế chính:**
1. Sinh noise ξ ~ N(0, σ²I)
2. Tính loss tại hai điểm: L(w + ξ) và L(w - ξ) (two-point estimate)
3. Ước lượng Hessian trace: `Tr(H) ≈ [L(w+ξ) - 2L(w) + L(w-ξ)] / σ²`
4. Thêm regularization: `L_total = L(w) + λ · Tr(H)`
5. Two-point estimate loại bỏ variance từ first-order Taylor → ước lượng ổn định

**Ưu điểm:**
- Có PAC-Bayes generalization bound chặt chẽ
- Test accuracy tăng đến 2.4%, Hessian trace giảm 15.8%
- Kết hợp tốt với weight decay và data augmentation
- Có code: [GitHub](https://github.com/VirtuosoResearch/Noise-stability-optimization)

**Độ phù hợp với SOAP-lite-OGP:** ★★★★☆ — Cao. Cần thêm 2 forward passes per step, nhưng regularization term rất hiệu quả cho second-order methods vốn hay rơi vào sharp minima.

**Complexity bổ sung:** O(2 × forward pass) — chủ yếu là compute, không thêm memory.

---

### 4. SWA-Shampoo — Stochastic Weight Averaging cho Second-Order

**Nguồn:** Goldfarb et al. (Oxford) — [Paper](https://oxford-man.ox.ac.uk/wp-content/uploads/2020/03/Closing-the-K-FAC-Generalisation-Gap-Using-Stochastic-Weight-Averaging.pdf); Izmailov et al. (2018) — [arXiv:1803.05407](https://arxiv.org/abs/1803.05407); Wang et al. **ICML 2024** — [Paper](https://proceedings.mlr.press/v235/wang24bl.html)

**Vấn đề giải quyết:** K-FAC/Shampoo hội tụ nhanh nhưng đến sharp minimum → averaging weights across trajectory tìm được flatter region.

**Ý tưởng:** Trung bình hóa weight từ nhiều checkpoint dọc theo trajectory training. SWA tìm được vùng phẳng hơn vì trung bình của nhiều sharp minima gần nhau thường nằm tại flat region giữa chúng.

**Cơ chế chính:**
1. Training bình thường bằng SOAP-lite-OGP trong T_swa epoch đầu
2. Sau T_swa, bắt đầu tích lũy: `w_swa = (n · w_swa + w_current) / (n + 1)` mỗi c epoch
3. Cuối training (hoặc cuối mỗi task), dùng w_swa cho inference
4. Optional: Cập nhật BatchNorm statistics cho w_swa

**Ưu điểm:**
- Cực kỳ đơn giản, gần như zero overhead
- Đã chứng minh đóng generalization gap của K-FAC trên CIFAR-100 (Oxford paper)
- ICML 2024 cung cấp generalization bound chặt cho non-convex setting
- Không thay đổi training dynamics → giữ nguyên plasticity

**Độ phù hợp với SOAP-lite-OGP:** ★★★★★ — Rất cao. Không can thiệp vào optimizer → giữ nguyên plasticity benefits. Chỉ cần thêm running average. Đặc biệt phù hợp cho continual learning: SWA mỗi task period.

**Complexity bổ sung:** O(n_params) memory cho w_swa, O(1) compute per averaging step.

---

### 5. EMA — Exponential Moving Average of Weights

**Nguồn:** Zamini et al., 2024 — [arXiv:2411.18704](https://arxiv.org/abs/2411.18704)

**Vấn đề giải quyết:** Giống SWA nhưng liên tục hơn — weight trajectory của second-order method dao động quanh sharp minima, EMA làm mượt trajectory.

**Ý tưởng:** Duy trì exponential moving average của weight song song với training:

```
w_ema = β · w_ema + (1 - β) · w_current,  β ∈ [0.999, 0.9999]
```

**Cơ chế chính:**
1. Sau mỗi optimizer.step(), cập nhật w_ema
2. Dùng w_ema cho evaluation/inference
3. Training vẫn chạy trên w_current (không bị ảnh hưởng)
4. β lớn → EMA thay đổi chậm → smooth hơn nhưng lag hơn

**Ưu điểm:**
- Đơn giản nhất trong tất cả methods
- Giảm sensitivity với noisy labels, cải thiện calibration
- Implicit regularization: EMA cần ít learning rate decay hơn
- Đã được dùng rộng rãi trong SOTA models (EfficientNet, ViT, LLM training)

**Độ phù hợp với SOAP-lite-OGP:** ★★★★★ — Rất cao. Zero overhead trên training. Đặc biệt hữu ích trong continual learning: EMA model có thể generalize tốt hơn trên cả task cũ và mới.

**Complexity bổ sung:** O(n_params) memory, O(n_params) compute per step (rất nhẹ).

---

### 6. Spectral Norm Constraint (Muon-style)

**Nguồn:** Jordan et al., 2025 — [arXiv:2506.15054](https://arxiv.org/html/2506.15054v2); ICML 2025 — [Paper](https://icml.cc/virtual/2025/47634); Xie et al. — [OpenReview](https://openreview.net/forum?id=YzjS4jcfmS)

**Vấn đề giải quyết:** Weight matrix với spectral norm lớn → gradient bùng nổ/triệt tiêu → sharp minima.

**Ý tưởng:** Ràng buộc (implicitly hoặc explicitly) spectral norm của weight matrices. Muon optimizer cho thấy: khi optimize dưới nuclear norm constraint, mạng tự nhiên hội tụ đến **max-margin solution** theo spectral norm — liên quan đến flat minima.

**Cơ chế chính (2 biến thể):**

**(a) Explicit Spectral Constraint:**
1. Mỗi T step, tính σ_max(W) bằng power iteration (đã có trong SpectralReg)
2. Nếu σ_max > threshold: project W về constraint set bằng `W ← W · (threshold / σ_max)`
3. Khác với SpectralReg (regularization loss): đây là hard constraint

**(b) Spectral Gradient Descent (SpecGD) — Muon-style:**
1. Thay vì precondition bằng L^{-1/4}, R^{-1/4}, dùng **orthogonalization**
2. Update direction = U · V^T (SVD of gradient) — loại bỏ singular values
3. Tất cả principal components được học với tốc độ bằng nhau

**Ưu điểm:**
- SpecGD: Generalize tốt hơn trên imbalanced data (ICML 2025)
- Implicit max-margin → flat minima tự nhiên
- Muon đã chứng minh SOTA trên LLM pretraining

**Độ phù hợp với SOAP-lite-OGP:** ★★★☆☆ — Trung bình. SpecGD thay đổi bản chất optimizer (không còn là Shampoo). Spectral constraint dễ tích hợp hơn nhưng kém mạnh hơn. Có thể dùng hybrid: SOAP-lite cho preconditioning + spectral norm clipping.

**Complexity bổ sung:** O(mn) per clipping step; O(m² + n²) cho SpecGD.

---

### 7. Frob-SAM / Det-SAM — Universal Sharpness Measures

**Nguồn:** Kwon et al., **ICML 2024** — [arXiv:2406.03682](https://arxiv.org/abs/2406.03682)

**Vấn đề giải quyết:** SAM gốc dùng max eigenvalue λ_max làm sharpness measure — có thể không capture đầy đủ geometry của loss landscape.

**Ý tưởng:** Đề xuất **framework tổng quát** cho sharpness measures, với 2 biến thể mới:
- **Frob-SAM:** Minimize Frobenius norm ‖H‖_F của Hessian (tất cả eigenvalues, không chỉ max)
- **Det-SAM:** Minimize determinant det(H) (volume của loss surface curvature)

**Cơ chế chính:**
1. Frob-SAM: Perturbation ε ∝ H·∇L thay vì ε ∝ ∇L (SAM gốc)
   - Hướng perturbation theo curvature → nhắm chính xác hơn vào sharp directions
2. Det-SAM: Perturbation dọc theo eigenvector có eigenvalue lớn nhất
   - Giảm tất cả eigenvalues đồng thời
3. Cả hai đều parameter-invariant → robust với reparametrization

**Ưu điểm:**
- Frob-SAM capture toàn bộ sharpness profile, không chỉ worst-case
- Det-SAM giảm "volume" của sharp region → flat minima toàn diện
- Có lý thuyết generalization bound
- Parameter-invariant (quan trọng cho second-order methods vốn thay đổi parametrization)

**Độ phù hợp với SOAP-lite-OGP:** ★★★★☆ — Cao. SOAP-lite đã có eigendecomp L, R → có thể ước lượng Hessian spectrum từ đó. Frob-SAM đặc biệt phù hợp vì perturbation direction tận dụng curvature info có sẵn.

**Complexity bổ sung:** O(mn + forward-backward) per step.

---

### 8. Preconditioned SAM (P-SAM)

**Nguồn:** Mok et al., ICASSP 2025 — [arXiv:2501.06603](https://arxiv.org/abs/2501.06603)

**Vấn đề giải quyết:** SAM gốc dùng isotropic perturbation → không tối ưu khi loss landscape có curvature khác nhau theo các hướng.

**Ý tưởng:** Dùng **preconditioner** (từ Shampoo/SOAP) để điều chỉnh perturbation direction của SAM. Hướng perturbation được warped theo curvature → tìm sharp directions hiệu quả hơn.

**Cơ chế chính:**
1. Thay vì ε = ρ · ∇L / ‖∇L‖ (SAM gốc)
2. Dùng ε = ρ · P · ∇L / ‖P · ∇L‖ với P là preconditioner (P = Q_L diag Q_R từ SOAP)
3. Perturbation theo hướng high-curvature → phát hiện sharpness chính xác hơn
4. Bao gồm biến thể **infoSAM**: thêm noise correction để tránh adversarial degradation

**Ưu điểm:**
- Unifying framework cho nhiều SAM variants
- Tận dụng preconditioner có sẵn → zero extra eigendecomp cost
- infoSAM khắc phục vấn đề SAM đôi khi làm giảm accuracy

**Độ phù hợp với SOAP-lite-OGP:** ★★★★★ — Rất cao. Preconditioner Q_L, Q_R đã có sẵn từ SOAP-lite. Chỉ cần thêm 1 forward-backward cho SAM perturbation.

**Complexity bổ sung:** O(1 forward-backward) per step.

---

### 9. Adaptive Damping with Generalization Awareness

**Nguồn:** Morwani et al. (Harvard), 2024 — [Paper](https://lucasjanson.fas.harvard.edu/papers/A_New_Perspective_On_Shampoos_Preconditioner-Morwani_ea-2024.pdf); Eschenhagen et al. (ICLR 2024) — [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/7f63032459a9b644f91373e71b456457-Paper-Conference.pdf)

**Vấn đề giải quyết:** Damping term ε trong Shampoo (L + εI)^{-1/4} ảnh hưởng lớn đến sharpness của minimum: damping nhỏ → preconditioning mạnh → sharp minima; damping lớn → gần SGD → flat nhưng chậm.

**Ý tưởng:** Adaptive damping — thay đổi damping theo training progress:
- Giai đoạn đầu: damping nhỏ → hội tụ nhanh (tận dụng curvature)
- Giai đoạn sau: damping tăng dần → bước cập nhật gần SGD hơn → tìm flat minima
- Tại task boundary (continual learning): reset damping về nhỏ → plasticity cao cho task mới

**Cơ chế chính:**
1. Monitor train-test gap hoặc Hessian trace (từ Shampoo factors)
2. Khi gap tăng: `ε ← ε × (1 + α)` — tăng damping → giảm preconditioning strength
3. Khi gap ổn định: giữ nguyên
4. Tại task boundary: `ε ← ε_init` — reset cho task mới
5. Schedule có thể: linear warmup damping, cosine damping schedule, hoặc gap-triggered

**Ưu điểm:**
- Không thêm overhead tính toán đáng kể
- Trực tiếp kiểm soát "mức độ second-order" của optimizer
- Tương thích hoàn toàn với OGP (không thay đổi gradient direction)
- Có cơ sở lý thuyết từ µP scaling (ICLR 2024)

**Độ phù hợp với SOAP-lite-OGP:** ★★★★★ — Rất cao. Damping đã là hyperparameter có sẵn. Chỉ cần thêm adaptive schedule. Đặc biệt phù hợp cho continual learning: damping thấp lúc đầu task (cần plasticity) → damping cao cuối task (cần generalization).

**Complexity bổ sung:** O(1) — chỉ thay đổi 1 scalar.

---

### 10. Late-Phase SAM Switch

**Nguồn:** Bartlett et al., **ICLR 2025 Spotlight** — [arXiv:2410.10373](https://arxiv.org/abs/2410.10373)

**Vấn đề giải quyết:** SAM full training tốn kém (2× forward-backward). Nhưng phần lớn benefit đến từ giai đoạn cuối.

**Ý tưởng:** Phát hiện rằng SAM hoạt động theo 2 phase:
1. **Exponential escape:** Nhanh chóng thoát khỏi sharp minimum mà SGD/Adam tìm được
2. **Rapid convergence:** Hội tụ nhanh đến flat minimum gần nhất

→ Chỉ cần bật SAM trong **vài epoch cuối** của mỗi task period.

**Cơ chế chính:**
1. Training bằng SOAP-lite-OGP bình thường trong (T - T_sam) epochs
2. T_sam epochs cuối: chuyển sang SOAP-lite-OGP + SAM
3. T_sam có thể nhỏ (5-20% tổng epochs) mà vẫn đạt generalization tương đương full SAM

**Ưu điểm:**
- Tiết kiệm 80-95% chi phí SAM
- ICLR 2025 Spotlight — kết quả mạnh
- Rất tự nhiên cho continual learning: SAM cuối mỗi task period trước task boundary

**Độ phù hợp với SOAP-lite-OGP:** ★★★★★ — Rất cao. Giảm overhead SAM xuống tối thiểu. Trong incremental CIFAR-100: mỗi 200 epoch, chỉ bật SAM ~20-40 epoch cuối.

**Complexity bổ sung:** O(1 forward-backward) chỉ trong giai đoạn cuối.

---

## Bảng tổng kết & Xếp hạng đề xuất

| # | Phương pháp | Venue | Cơ chế | Overhead | Phù hợp SOAP-OGP |
|---|-------------|-------|--------|----------|-------------------|
| 1 | **SASSHA** | ICML 2025 | SAM + Second-order + Stable Hessian | 1 extra fwd-bwd | ★★★★★ |
| 2 | **IRE** | NeurIPS 2024 | Decouple sharp/flat directions | Minimal (dùng eigen sẵn) | ★★★★☆ |
| 3 | **NSO** | ICLR 2025 | Noise → Hessian trace regularization | 2 extra fwd | ★★★★☆ |
| 4 | **SWA-Shampoo** | ICML 2024 | Weight averaging across trajectory | ~0 compute | ★★★★★ |
| 5 | **EMA** | 2024 | Exponential moving average of weights | ~0 compute | ★★★★★ |
| 6 | **Spectral Constraint** | ICML 2025 | σ_max clipping / SpecGD | O(mn) | ★★★☆☆ |
| 7 | **Frob-SAM** | ICML 2024 | Frobenius-norm sharpness measure | 1 extra fwd-bwd | ★★★★☆ |
| 8 | **P-SAM** | ICASSP 2025 | Preconditioned SAM perturbation | 1 extra fwd-bwd | ★★★★★ |
| 9 | **Adaptive Damping** | ICLR 2024 | Damping schedule theo training phase | ~0 | ★★★★★ |
| 10 | **Late-Phase SAM** | ICLR 2025 | SAM chỉ ở cuối mỗi task period | Rất nhỏ (5-20% steps) | ★★★★★ |

---

## Đề xuất kết hợp cho SOAP-lite-OGP (Incremental CIFAR-100)

### Combo A: Lightweight (gần zero overhead)
```
SOAP-lite-OGP + Adaptive Damping + EMA
```
- Damping nhỏ đầu task → plasticity; damping tăng cuối task → generalization
- EMA model cho inference → smooth out sharp fluctuations
- **Overhead: ~0**

### Combo B: Moderate (SAM cuối task)
```
SOAP-lite-OGP + Late-Phase SAM + SWA
```
- SOAP-lite bình thường 80% task period
- Bật SAM 20% cuối → escape sharp minima
- SWA averaging cuối mỗi task period
- **Overhead: ~20% extra fwd-bwd ở cuối**

### Combo C: Aggressive (maximum generalization)
```
SOAP-lite-OGP + P-SAM (preconditioned) + IRE + Adaptive Damping
```
- P-SAM dùng Q_L, Q_R có sẵn → perturbation chính xác
- IRE decouple sharp/flat → tăng tốc flat minima discovery
- Adaptive damping kiểm soát strength of preconditioning
- **Overhead: 1 extra fwd-bwd per step + eigendecomp (đã có)**

### Combo D: Theory-grounded (PAC-Bayes optimal)
```
SOAP-lite-OGP + NSO + EMA
```
- NSO regularize Hessian trace → PAC-Bayes generalization bound
- EMA cho inference
- **Overhead: 2 extra forward passes per step**

---

## Lưu ý khi implement

1. **OGP compatibility:** Các phương pháp SAM-based thay đổi gradient → cần đảm bảo OGP projection vẫn được áp dụng SAU SAM perturbation, không phải trước.

2. **Preconditioner stability:** SASSHA nhấn mạnh cần ổn định Hessian approximation khi thêm SAM → nên dùng EMA cho L, R factors thay vì raw update.

3. **Task boundary interaction:** Trong continual learning, damping/SAM schedule nên reset tại task boundary:
   - Đầu task mới: damping nhỏ, không SAM → maximize plasticity
   - Cuối task: damping lớn, bật SAM/SWA → maximize generalization trước khi OGP save

4. **Evaluation protocol:** Dùng EMA/SWA model cho evaluation dormant neurons và test accuracy, vì đây là model sẽ được deploy. Training model có thể sharp hơn — đó là OK.
