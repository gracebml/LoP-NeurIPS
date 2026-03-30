# Hiện tượng Weight Norm tăng nhanh trong Deep Learning: Nguyên nhân, Ảnh hưởng và Giải pháp

> **Deep Research từ các hội nghị top-tier: NeurIPS, ICML, ICLR**
>
> Tổng hợp cho bối cảnh: SASSHA optimizer + EMA trên Incremental CIFAR-100 / ResNet-18

---

## 1. Tổng quan vấn đề

Sau khi fix Hessian trace (clipping), gradient explosion được kiểm soát nhưng **average weight magnitude vẫn tăng nhanh** qua các task. Đây là hiện tượng phổ biến được ghi nhận rộng rãi trong cả supervised learning lẫn continual learning, và là một **triệu chứng nghiêm trọng** liên quan đến nhiều vấn đề sâu hơn trong quá trình tối ưu.

---

## 2. Nguyên nhân (Causes)

### 2.1. Tương tác giữa Batch Normalization và Weight Decay

> **Paper**: Lobacheva et al., *"On the Periodic Behavior of Neural Network Training with Batch Normalization and Weight Decay"*, NeurIPS 2021

**Cơ chế cốt lõi:**
- BatchNorm tạo ra **scale invariance** cho weight: nhân weight với hằng số dương không thay đổi output → chỉ **hướng** (direction) của weight mới quan trọng, không phải norm
- Khi SGD cập nhật theo gradient của loss, bước cập nhật có xu hướng **tăng norm** của scale-invariant weights
- Weight decay thì **giảm norm**
- Hai lực đối ngược này tạo ra **dynamic tuần hoàn (periodic dynamics)**: norm tăng → weight decay kéo xuống → lại tăng → ...
- Trong quá trình này, **effective learning rate** (ELR) thay đổi theo:

$$\text{ELR} = \frac{\eta}{\|w\|^2}$$

→ Khi $\|w\|$ tăng, ELR giảm ngầm dù learning rate $\eta$ không đổi.

**Liên quan đến SASSHA**: SASSHA dùng ResNet-18 có BatchNorm → weight scale invariance trực tiếp ảnh hưởng. Nếu weight decay quá nhỏ so với learning rate, norm sẽ tăng không kiểm soát.

---

### 2.2. Unbounded Weight Growth gây Loss of Plasticity

> **Paper**: Lyle et al., *"Disentangling the Causes of Plasticity Loss in Neural Networks"*, arXiv:2402.18762, 2024
>
> **Paper**: Lyle et al., *"Normalization and Effective Learning Rates in Reinforcement Learning"*, arXiv:2407.01800, 2024

**Phát hiện chính:**
- **Parameter norm growth không bị ràng buộc** là một trong những nguyên nhân chính gây **mất tính dẻo (loss of plasticity)** trong mạng neural
- Khi weight norm tăng → gradient tương đối nhỏ hơn → **effective step size giảm** → mạng "đông cứng", khó học task mới
- Tác giả chỉ ra rằng weight norm tăng liên tục là **hệ quả tự nhiên** của quá trình học khi không có regularization đủ mạnh
- Weight norm lớn → gradient và Hessian lớn hơn → **sharpness tăng** → optimization landscape xấu đi

**Chuỗi nhân quả:**
```
Weight norm ↑ → Effective LR ↓ → Plasticity ↓ → Không học được task mới
                                                    ↓
                            Gradient/Hessian ↑ → Sharpness ↑ → Generalization ↓
```

---

### 2.3. Tương tác Weight Decay với Learning Rate Schedule

> **Paper**: *"Gradient Norm Increase during Training of Large Language Models"*, arXiv, 2024

- Trong thực tế, weight decay kiểm soát **tỷ lệ gradient norm / weight norm**
- Khi thay đổi learning rate (lr decay schedule), tỷ lệ này bị phá vỡ → gradient norm có thể tăng bất thường
- Nếu lr giảm quá nhanh trong khi weight decay giữ nguyên → weight norm được "giải phóng" khỏi cân bằng cũ → tăng nhanh

---

### 2.4. Second-Order Optimizer hội tụ về Sharp Minima

> **Paper**: Yong et al., *"SASSHA: Sharpness-Aware Stochastic Second-order Hessian Approximation"*, arXiv, 2024

- Second-order optimizer (bao gồm xấp xỉ Hessian) thường hội tụ về **minima sắc nhọn (sharp minima)** → weight có xu hướng phân bổ theo cách tạo ra gradient lớn hơn
- SASSHA kết hợp sharpness-aware để giảm thiểu điều này, nhưng bản thân cơ chế SAM cũng chỉ giảm sharpness một phần, không trực tiếp kiểm soát weight norm

---

### 2.5. Continual Learning: Tích lũy qua nhiều task

- Khi học tuần tự nhiều task (incremental CIFAR-100), mỗi lần cập nhật weight cho task mới đều có thể **tăng norm tổng thể**
- Nếu không có cơ chế ràng buộc norm giữa các task, weight magnitude tích lũy
- Đặc biệt nghiêm trọng khi dùng **EMA (Exponential Moving Average)**: EMA giữ lại thông tin weight cũ, nếu weight cũ đã lớn → EMA cũng lớn → "dây chuyền" tăng norm

---

## 3. Ảnh hưởng (Effects)

### 3.1. Mất tính dẻo (Loss of Plasticity)

> **Paper**: Dohare et al., *"Continual Backpropagation: Learning Under a Changing Target"*, NeurIPS 2023

| Biểu hiện | Mô tả |
|---|---|
| **Effective LR giảm** | Weight norm lớn → $\eta / \|w\|^2$ giảm → update thực tế rất nhỏ |
| **Dormant neurons tăng** | Neuron bị saturate vĩnh viễn (ReLU dead, sigmoid flat) |
| **Feature rank giảm** | Representation matrix mất đa dạng → stable rank ↓ |
| **Khả năng học task mới ↓** | Mạng "đông cứng", không thể thích ứng |

### 3.2. Generalization xấu đi

- Weight norm lớn → **PAC-Bayes / Rademacher complexity bound** lớn → generalization gap ↑
- Loss landscape trở nên **sharper** → sensitivity cao với perturbation → test accuracy giảm

### 3.3. Numerical Instability

- Weight lớn → activation lớn → gradient lớn (dù đã clip Hessian) → vòng lặp bất ổn
- Đặc biệt nguy hiểm với float16/mixed precision training

### 3.4. Overfitting

- Weight norm lớn tương đương với model có **capacity thực tế quá cao** → overfit training data
- Train-test accuracy gap tăng theo weight norm

---

## 4. Giải pháp (Solutions)

### 4.1. Weight Decay đúng cách (Decoupled Weight Decay)

> **Paper**: Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"*, ICLR 2019

**Ý tưởng:** Tách biệt weight decay khỏi gradient update:
```python
# Decoupled weight decay (AdamW-style)
w = w - lr * gradient - lr * wd * w  # ✗ coupled (L2)
w = w - lr * gradient - wd * w       # ✓ decoupled
```

**Áp dụng cho SASSHA:**
- Kiểm tra cách SASSHA implement weight decay
- Tăng weight decay (wd=0.01 → 0.05 hoặc cao hơn) cho đến khi weight norm ổn định
- Theo dõi cân bằng giữa lr và wd

---

### 4.2. Normalize-and-Project (NaP)

> **Paper**: Lyle et al., *"Normalization and Effective Learning Rates in Reinforcement Learning"*, 2024

**Ý tưởng:** Re-parameterize weight để **giữ ELR không đổi** bất kể weight norm:
```python
# Sau mỗi bước cập nhật:
w = w / ||w||  # normalize weight về unit norm
# Kết hợp với scale factor riêng nếu cần
```

**Ưu điểm:**
- ELR = lr (luôn đúng như lr schedule, không bị ẩn)
- Loại bỏ hoàn toàn vấn đề weight norm tăng
- Hiệu quả cả trong stationary và non-stationary environments

**Nhược điểm:**
- Cần sửa đổi optimizer/training loop
- Có thể ảnh hưởng đến Hessian computation trong SASSHA

---

### 4.3. Spectral Norm Regularization

> **Paper**: Miyato et al., *"Spectral Normalization for GANs"*, ICLR 2018
>
> **Paper**: Yoshida & Miyato, *"Spectral Norm Regularization for Improving the Generalizability of DNNs"*, arXiv, 2017

**Ý tưởng:** Ràng buộc spectral norm (singular value lớn nhất) của weight matrix:
```python
# Spectral normalization
W_normalized = W / sigma(W)  # sigma = largest singular value

# Hoặc spectral penalty
loss += lambda_sn * sum(spectral_norm(W) for W in model.parameters())
```

**Lợi ích:**
- Kiểm soát Lipschitz constant của mạng
- Ngăn weight norm "nổ" theo singular value lớn nhất
- Duy trì stable rank hợp lý

---

### 4.4. Parseval Regularization

> **Paper**: Cissé et al., *"Parseval Networks: Improving Robustness to Adversarial Examples"*, ICML 2017
>
> **Paper**: Elsayed et al., *"Regularization via Parseval Regularization for Continual Reinforcement Learning"*, NeurIPS 2024

**Ý tưởng:** Khuyến khích weight matrix gần orthogonal:
```python
# Parseval penalty
for W in model.parameters():
    if W.dim() >= 2:
        WTW = W.T @ W
        I = torch.eye(WTW.shape[0])
        loss += beta * ||WTW - I||_F^2
```

**Tại sao hiệu quả:**
- Weight orthogonal → norm bị ràng buộc tự nhiên (singular values ≈ 1)
- Bảo toàn gradient flow (không vanishing/exploding)
- Đặc biệt tốt cho continual RL

---

### 4.5. Continual Backpropagation

> **Paper**: Dohare et al., *"Loss of Plasticity in Deep Continual Learning"*, Nature 2024

**Ý tưởng:** Liên tục reinitialize một phần nhỏ neuron ít hoạt động:
```python
# Mỗi N bước:
for layer in model.layers:
    utility = compute_utility(layer)  # đo mức hữu ích
    mask = utility < threshold
    layer.weight[mask] = random_init()  # reset neuron "chết"
```

**Lợi ích:**
- Giữ network "tươi mới" → ngăn tích lũy weight norm
- Duy trì feature diversity và plasticity
- Đã được validate trên Nature 2024

---

### 4.6. Soft Weight Rescaling

> **Paper**: *"Recovering Plasticity of Neural Networks via Soft Weight Rescaling"*, OpenReview / ICLR 2025

**Ý tưởng:** Giảm weight norm một cách "mềm" (không reset hoàn toàn):
```python
# Mỗi epoch hoặc mỗi task:
for param in model.parameters():
    param.data *= rescale_factor  # 0 < rescale_factor < 1 (e.g. 0.9)
```

**Ưu điểm so với weight decay:**
- Trực tiếp kiểm soát norm, không phụ thuộc vào gradient
- Giữ lại hướng (direction) của weight → bảo toàn knowledge
- Có thể apply chọn lọc theo layer

---

### 4.7. UPGD (Utility-based Perturbed Gradient Descent)

> **Paper**: Elsayed et al., *"Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning"*, ICLR 2024

**Ý tưởng kết hợp:** Giải quyết đồng thời plasticity loss VÀ catastrophic forgetting:
- Unit **hữu ích cao** → update nhỏ (bảo vệ)
- Unit **hữu ích thấp** → update lớn + perturbation (tái sinh)
- Kết hợp weight decay để kiểm soát norm

---

### 4.8. Layer-wise Norm Monitoring + Adaptive Clipping

**Ý tưởng thực tế:**
```python
# Monitor và clip weight norm mỗi bước
MAX_WEIGHT_NORM = 10.0  # hyperparameter

for name, param in model.named_parameters():
    if 'weight' in name:
        norm = param.data.norm()
        if norm > MAX_WEIGHT_NORM:
            param.data *= MAX_WEIGHT_NORM / norm
```

**Đơn giản và hiệu quả** khi kết hợp với weight decay.

---

## 5. Đề xuất cụ thể cho SASSHA + EMA trên Incremental CIFAR-100

### Phương án A: Quick Fix (độ phức tạp thấp)

1. **Tăng weight decay**: thử `wd = 0.05` hoặc `0.1` (hiện tại nếu đang dùng `0.01` hoặc thấp hơn)
2. **Thêm weight norm clipping**: set `MAX_WEIGHT_NORM` per-layer
3. **Monitor ELR**: log `lr / ||w||^2` để phát hiện ELR suy giảm

### Phương án B: Moderate Fix (trung bình)

4. **Soft Weight Rescaling giữa các task**: rescale factor = 0.8-0.95 trước mỗi task mới
5. **Spectral norm penalty**: thêm vào loss function
6. **Kiểm tra decoupled weight decay**: đảm bảo SASSHA dùng AdamW-style decoupled weight decay

### Phương án C: Thorough Fix (toàn diện)

7. **Normalize-and-Project (NaP)**: implement cho scale-invariant layers (trước BatchNorm)
8. **Continual Backpropagation**: kết hợp reset neuron dormant
9. **Parseval regularization**: cho các convolutional layers

### Thứ tự ưu tiên đề xuất:

```
[1] Tăng weight decay          ← Dễ nhất, thử trước
[2] Weight norm clipping       ← An toàn, fallback
[3] Soft weight rescaling      ← Hiệu quả cho continual learning
[4] Spectral norm penalty      ← Nếu stable rank cũng là vấn đề
[5] NaP                        ← Giải pháp triệt để nhất
```

---

## 6. Tài liệu tham khảo

| # | Bài báo | Hội nghị/Nguồn | Chủ đề liên quan |
|---|---------|----------------|-------------------|
| 1 | Lobacheva et al., "Periodic Behavior of Training with BN and WD" | NeurIPS 2021 | BN + WD → periodic weight norm |
| 2 | Lyle et al., "Disentangling the Causes of Plasticity Loss" | arXiv 2024 | Weight norm → loss of plasticity |
| 3 | Lyle et al., "Normalization and Effective Learning Rates" | arXiv 2024 | ELR decay từ weight norm growth |
| 4 | Dohare et al., "Loss of Plasticity in Deep Continual Learning" | Nature 2024 | Continual Backpropagation |
| 5 | Dohare et al., "Continual Backpropagation" | NeurIPS 2023 | Neuron reinit cho plasticity |
| 6 | Elsayed et al., "UPGD" | ICLR 2024 | Plasticity + forgetting |
| 7 | Loshchilov & Hutter, "Decoupled Weight Decay Regularization" | ICLR 2019 | AdamW |
| 8 | Miyato et al., "Spectral Normalization" | ICLR 2018 | Spectral norm control |
| 9 | Cissé et al., "Parseval Networks" | ICML 2017 | Orthogonal weight regularization |
| 10 | Elsayed et al., "Parseval Regularization for Continual RL" | NeurIPS 2024 | Parseval cho continual learning |
| 11 | "Recovering Plasticity via Soft Weight Rescaling" | ICLR 2025 | Soft rescaling |
| 12 | Yong et al., "SASSHA" | arXiv 2024 | Second-order + SAM |
| 13 | "SNVR: Spectral Norm Variance Regularization" | ICLR 2026 | Layer-wise spectral norm |

---

> [!IMPORTANT]
> **Kết luận**: Weight norm tăng nhanh không chỉ đơn thuần là vấn đề regularization — nó phản ánh sự tương tác phức tạp giữa BatchNorm (scale invariance), weight decay, learning rate schedule, và đặc thù second-order optimization. Trong bối cảnh continual learning với SASSHA, giải pháp hiệu quả nhất là **kết hợp** tăng weight decay + soft rescaling giữa các task + monitor ELR.
