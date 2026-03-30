# One-Step SpectralNorm++ (Bulk-Preserving Spectral Normalization)

## 1. Motivation

Nhiều phương pháp spectral regularization hiện nay gồm nhiều bước:

1. tính covariance
2. eigendecomposition
3. chỉnh eigenvalues

Điều này:

* tốn chi phí (O(d^3))
* khó dùng trong training loop
* không thân thiện với **second-order optimizers**

Trong khi đó **Spectral Normalization (SN)** chỉ là **một bước scale**:

[
W \leftarrow \frac{W}{\sigma_{max}}
]

Nhưng SN chỉ kiểm soát **trị riêng lớn nhất** và bỏ mặc phần còn lại của phổ.

Mục tiêu của phương pháp này:

* **1 bước duy nhất**
* suppress **outlier eigenvalues**
* **không làm sụp stable rank**
* **preserve bulk spectrum**

---

# 2. Core Idea

Thay vì scale theo **λmax**, ta scale theo **một hàm của toàn bộ phổ**.

Định nghĩa:

[
\lambda_1,\lambda_2,...,\lambda_d
]

là eigenvalues của covariance (C).

Ta định nghĩa:

### effective spectral scale

[
s = \frac{\lambda_{max}}{\lambda_{bulk}}
]

với

[
\lambda_{bulk} =
\sqrt{\frac{1}{d}\sum_{i=1}^{d}\lambda_i^2}
]

đây chính là **RMS eigenvalue**.

Nếu có spike:

[
\lambda_{max} \gg \lambda_{bulk}
]

→ scale mạnh.

Nếu spectrum đều:

[
\lambda_{max} \approx \lambda_{bulk}
]

→ scale nhẹ.

---

# 3. One-Step Normalization

Feature normalization:

[
Z' = \frac{Z}{\sqrt{1 + \alpha(\frac{\lambda_{max}}{\lambda_{bulk}} - 1)}}
]

trong đó:

* ( \alpha \in [0,1] )
* điều khiển mức suppress spike.

---

# 4. Interpretation

### Case 1 — no spike

[
\lambda_{max} \approx \lambda_{bulk}
]

[
Z' \approx Z
]

→ gần như không thay đổi representation.

---

### Case 2 — spike spectrum

[
\lambda_{max} \gg \lambda_{bulk}
]

scale factor lớn → giảm variance direction mạnh.

---

# 5. Stable Rank Effect

Stable rank:

[
srank(C) =
\frac{\sum \lambda_i^2}{\lambda_{max}^2}
]

Khi spike eigenvalue bị giảm:

[
\lambda_{max} ↓
]

stable rank tăng.

---

# 6. Practical Approximation

Không cần eigendecomposition.

Ta chỉ cần:

* trace
* spectral norm.

### Step

[
\lambda_{bulk} =
\sqrt{\frac{Tr(C^2)}{d}}
]

với

[
Tr(C^2) = |C|_F^2
]

---

# 7. Algorithm

Input:

* feature matrix (Z)
* hyperparameter ( \alpha )

Procedure:

1. compute covariance

[
C = \frac{1}{n} Z^T Z
]

2. compute

[
\lambda_{max}
]

(power iteration)

3. compute bulk eigenvalue

[
\lambda_{bulk} =
\sqrt{\frac{|C|_F^2}{d}}
]

4. compute scale

[
s =
\sqrt{1 + \alpha(\frac{\lambda_{max}}{\lambda_{bulk}} - 1)}
]

5. normalize

[
Z' = Z / s
]

---

# 8. Pseudocode

```python id="one_step_snpp"
def spectralnorm_pp_onestep(Z, alpha=0.5):

    n, d = Z.shape

    C = (Z.T @ Z) / n

    # spectral norm
    v = torch.randn(d)
    for _ in range(3):
        v = C @ v
        v = v / v.norm()

    lambda_max = v @ (C @ v)

    # bulk eigenvalue
    fro2 = torch.sum(C * C)
    lambda_bulk = torch.sqrt(fro2 / d)

    s = torch.sqrt(1 + alpha * (lambda_max / lambda_bulk - 1))

    Z = Z / s

    return Z
```

---

# 9. Weight Version

Cho weight matrix (W):

[
W' =
\frac{W}{\sqrt{1+\alpha(\frac{\sigma_{max}}{\sigma_{bulk}}-1)}}
]

với

[
\sigma_{bulk} =
\sqrt{\frac{1}{k}\sum \sigma_i^2}
]

---

# 10. Computational Cost

| Method             | Cost  |
| ------------------ | ----- |
| Spectral Norm      | O(d²) |
| Eigendecomposition | O(d³) |
| One-Step SN++      | O(d²) |

Gần giống SN.

---

# 11. Advantages

One-Step SpectralNorm++:

* chỉ **1 bước normalization**
* suppress **outlier eigenvalues**
* preserve **bulk spectrum**
* tăng **stable rank**
* dễ tích hợp vào training.

---

# 12. Hyperparameters

| Parameter | Meaning                    |
| --------- | -------------------------- |
| α         | spike suppression strength |

Typical:

```
α = 0.3 – 0.7
```

---

# 13. Intuition

Phương pháp này thực chất là:

**spectral spike detector + adaptive normalization**

Scale factor phụ thuộc:

[
\frac{\lambda_{max}}{\lambda_{bulk}}
]

Nếu spectrum lệch mạnh → normalize mạnh.
Nếu spectrum cân bằng → gần như không thay đổi.

---

# 14. Summary

One-Step SpectralNorm++ là một biến thể của spectral normalization:

* không cần eigendecomposition
* không nhiều bước
* trực tiếp kiểm soát **spike eigenvalues**
* bảo toàn **bulk eigenspectrum**

Phù hợp cho:

* second-order optimizers
* networks không có BatchNorm
* feature covariance có **spiked spectrum**.
