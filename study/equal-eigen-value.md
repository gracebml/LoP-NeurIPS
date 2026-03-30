# Equal-Eigenvalue Spectrum Regularization (EES)
Bài báo gốc: https://arxiv.org/abs/2305.16789 (Modulate Your Spectrum in Self-Supervised Learning)
Github: https://github.com/winci-ai/INTL
## 1. Motivation

Trong deep learning, đặc biệt khi huấn luyện các mạng lớn hoặc sử dụng **second-order optimization methods**, ma trận đặc trưng (feature covariance matrix) thường có phổ trị riêng rất lệch:

[
\lambda_1 \gg \lambda_2 \gg \lambda_3 \gg ... \gg \lambda_d
]

Hiện tượng này dẫn đến:

* **Outlier eigenvalues** (spikes)
* **Dimensional collapse**
* **Stable rank giảm mạnh**
* **Condition number lớn**

Stable rank của ma trận đặc trưng:

[
\text{srank}(C) = \frac{|C|_F^2}{|C|_2^2}
]

với

* ( |C|_F ): Frobenius norm
* ( |C|_2 ): spectral norm

Nếu eigenvalue đầu quá lớn:

[
\lambda_1 \gg \sum_{i>1}\lambda_i
]

thì

[
srank \approx 1
]

tức là **feature space gần như collapse thành một chiều**.

Phương pháp **Equal-Eigenvalue Spectrum (EES)** được thiết kế để:

* loại bỏ spike eigenvalues
* nâng các eigenvalues nhỏ
* duy trì phổ eigenvalue đồng đều

mà **không phá hủy cấu trúc biểu diễn**.

---

# 2. Feature Covariance Matrix

Cho batch feature matrix:

[
Z \in \mathbb{R}^{n \times d}
]

trong đó

* (n): batch size
* (d): feature dimension

Feature covariance:

[
C = \frac{1}{n} Z^T Z
]

Phân rã eigenvalue:

[
C = Q \Lambda Q^T
]

với

[
\Lambda = \text{diag}(\lambda_1,\lambda_2,...,\lambda_d)
]

---

# 3. Ý tưởng Equal-Eigenvalue Spectrum

Mục tiêu của EES:

[
\lambda_1 \approx \lambda_2 \approx ... \approx \lambda_d
]

Tức là covariance gần **isotropic**:

[
C \approx \alpha I
]

nhưng không bắt buộc phải chính xác bằng identity.

Điều này:

* giữ **representation đa chiều**
* tăng **stable rank**
* cải thiện **optimization geometry**

---

# 4. Spectrum Regularization Loss

Một loss cơ bản:

[
L_{spec} = \sum_{i=1}^d (\lambda_i - \bar{\lambda})^2
]

trong đó

[
\bar{\lambda} = \frac{1}{d}\sum_{i=1}^d \lambda_i
]

Loss này chính là **variance của eigenvalues**.

### Viết gọn:

[
L_{spec} = Var(\lambda)
]

Minimize variance → eigenvalues tiến gần nhau.

---

# 5. Matrix Form (không cần eigendecomposition)

Ta có thể viết tương đương:

[
L_{spec} = |C - \bar{\lambda} I|_F^2
]

với

[
\bar{\lambda} = \frac{1}{d} Tr(C)
]

Điều này giúp:

* tránh eigendecomposition
* tính gradient nhanh
* dễ implement trong training.

---

# 6. Gradient Intuition

Gradient của loss:

[
\nabla_Z L_{spec}
]

sẽ:

* giảm các direction có variance lớn
* tăng variance ở direction yếu

Nói cách khác:

* **compress spikes**
* **inflate tail eigenvalues**

---

# 7. So sánh với các phương pháp khác

| Method                    | Kiểm soát             |
| ------------------------- | --------------------- |
| Spectral Normalization    | λmax                  |
| Weight Decay              | norm weights          |
| Whitening                 | decorrelate features  |
| Equal-Eigenvalue Spectrum | toàn bộ eigenspectrum |

Spectral normalization:

[
W \leftarrow \frac{W}{\sigma_{max}}
]

→ chỉ xử lý **trị riêng lớn nhất**

Equal-Eigenvalue Spectrum:

→ **tái phân phối toàn bộ phổ**.

---

# 8. Ảnh hưởng đến Stable Rank

Stable rank:

[
srank(C) = \frac{\sum_i \lambda_i^2}{\lambda_1^2}
]

Nếu

[
\lambda_i \approx \lambda
]

thì

[
srank(C) \approx d
]

→ representation **full-rank**.

---

# 9. Pseudocode

```python
# Z : feature matrix (batch_size x feature_dim)

C = (Z.T @ Z) / batch_size

trace = torch.trace(C)
lambda_bar = trace / feature_dim

I = torch.eye(feature_dim)

loss_spec = torch.norm(C - lambda_bar * I, p='fro')**2
```

Training loss:

```python
loss = task_loss + alpha * loss_spec
```

---

# 10. Interaction with Second-Order Methods

Second-order methods thường update:

[
\Delta w = H^{-1} g
]

Nếu feature covariance có spike eigenvalues:

[
\lambda_1 \gg \lambda_2
]

thì Hessian cũng trở nên **ill-conditioned**.

EES giúp:

* flatten spectrum
* giảm condition number

[
\kappa = \frac{\lambda_{max}}{\lambda_{min}}
]

→ second-order optimization **ổn định hơn**.

---

# 11. Ưu điểm

Equal-Eigenvalue Spectrum:

* tăng **stable rank**
* giảm **representation collapse**
* giảm **outlier eigenvalues**
* cải thiện **feature isotropy**

---

# 12. Nhược điểm

Chi phí:

* tính covariance matrix (O(d^2))
* batch size nhỏ → estimate nhiễu

Ngoài ra:

* nếu regularization quá mạnh → feature bị **over-whitened**

---

# 13. Các biến thể nâng cao

### Soft Equal Spectrum

[
L = \sum (\lambda_i^\alpha - \bar{\lambda}^\alpha)^2
]

### Entropy Spectrum Regularization

[
H = -\sum p_i \log p_i
]

với

[
p_i = \frac{\lambda_i}{\sum \lambda_i}
]

Maximize entropy → spectrum đều.

---

# 14. Kết luận

Equal-Eigenvalue Spectrum là một dạng **spectral regularization** nhằm:

* kiểm soát toàn bộ eigenspectrum
* tránh dimensional collapse
* cải thiện stable rank

Phương pháp này đặc biệt hữu ích khi:

* huấn luyện mạng **không có BatchNorm**
* sử dụng **second-order optimizers**
* feature covariance xuất hiện **spiked eigenvalues**.
