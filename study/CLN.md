# Curvature-Aware Lipschitz Normalization (CLN)

## 1. Động lực (Motivation)

Trong quá trình huấn luyện mạng nơ-ron, đặc biệt khi sử dụng **second-order optimization methods** (ví dụ Newton-like, K-FAC, LoP), một vấn đề phổ biến là:

* **Hessian ill-conditioning**
* **Eigenvalue outliers**
* **Gradient alignment vào một vài hướng curvature**

Điều này dẫn đến:

* cập nhật tham số không ổn định
* mô hình dễ overfit
* representation collapse.

Một hướng phổ biến để giảm vấn đề này là **ràng buộc Lipschitz của mạng**:

[
|f(x_1) - f(x_2)| \le K |x_1 - x_2|
]

Trong đó:

* (f) là mạng nơ-ron
* (K) là hằng số Lipschitz.

Nếu (K) quá lớn:

* mạng có thể học các hàm dao động mạnh
* gradient thay đổi rất nhanh
* curvature trở nên cực kỳ lệch.

Các phương pháp phổ biến:

* Spectral Normalization (SN)
* Orthogonal weight constraint
* Gradient penalty.

Tuy nhiên, các phương pháp này thường **ràng buộc weight matrix** chứ không ràng buộc trực tiếp **Jacobian của mạng**.

---

# 2. Ý tưởng cốt lõi của CLN

**Curvature-Aware Lipschitz Normalization (CLN)** đề xuất ràng buộc Lipschitz **trực tiếp thông qua Jacobian của mạng**.

Jacobian của mạng:

[
J(x) = \frac{\partial f(x)}{\partial x}
]

Norm của Jacobian quyết định Lipschitz constant:

[
K = \sup_x |J(x)|_2
]

Do đó:

[
|f(x_1)-f(x_2)|
\le
|J(x)|_2
|x_1-x_2|
]

CLN điều chỉnh đầu ra của mạng để kiểm soát trực tiếp:

[
|J(x)|_2
]

---

# 3. Liên hệ với curvature

Trong nhiều bài toán học máy, Hessian của loss có dạng gần đúng:

[
H \approx J^T J
]

với:

* (J) là Jacobian của mạng.

Do đó nếu:

[
|J|_2
]

quá lớn, thì Hessian sẽ có eigenvalues rất lớn.

Điều này gây ra:

* bước Newton quá mạnh
* training instability.

CLN nhằm **giảm spectral norm của Jacobian**, từ đó:

* giảm curvature
* cải thiện condition number của Hessian.

---

# 4. Công thức phương pháp

Giả sử:

[
s = |J(x)|_2
]

là spectral norm của Jacobian tại điểm (x).

CLN chuẩn hóa đầu ra của mạng:

[
f'(x) =
\frac{f(x)}
{\sqrt{1 + \beta (s - 1)}}
]

Trong đó:

* ( \beta \in [0,1] ) là hệ số điều chỉnh.

---

# 5. Phân tích các trường hợp

### Trường hợp 1: Jacobian ổn định

Nếu:

[
s \approx 1
]

thì:

[
f'(x) \approx f(x)
]

mạng gần như không bị thay đổi.

---

### Trường hợp 2: Jacobian có spike

Nếu:

[
s \gg 1
]

thì scale factor tăng lên:

[
f'(x) = f(x)/c
]

với (c > 1).

Điều này làm giảm:

* Lipschitz constant
* curvature của loss.

---

# 6. Ước lượng Jacobian spectral norm

Không cần tính Jacobian đầy đủ.

Ta dùng **power iteration trên Jacobian-vector product (JVP)**.

### Thuật toán

1. khởi tạo vector ngẫu nhiên (v)
2. tính

[
u = Jv
]

3. tính

[
v = J^T u
]

4. chuẩn hóa (v)

Sau vài iteration:

[
s \approx \sqrt{u^T u}
]

---

# 7. Thuật toán CLN

Input:

* batch dữ liệu (x)
* hệ số (\beta)

Steps:

1. forward pass:

[
y = f(x)
]

2. ước lượng Jacobian spectral norm:

[
s = |J(x)|_2
]

3. tính scale:

[
c =
\sqrt{1 + \beta (s - 1)}
]

4. normalize output:

[
y' = y / c
]

---

# 8. Pseudocode

```python
def CLN_forward(model, x, beta=0.5):

    y = model(x)

    v = torch.randn_like(x)

    # Jacobian-vector product
    u = torch.autograd.grad(
        y, x, v,
        retain_graph=True,
        create_graph=True
    )[0]

    s = torch.norm(u)

    c = torch.sqrt(1 + beta*(s - 1))

    y = y / c

    return y
```

---

# 9. Tác động lên Hessian

Nếu:

[
H \approx J^T J
]

thì eigenvalues của Hessian tỷ lệ với:

[
\lambda_i(H) \sim \sigma_i(J)^2
]

CLN giảm:

[
\sigma_{max}(J)
]

→ giảm eigenvalue lớn nhất của Hessian.

Điều này cải thiện:

[
\kappa(H) =
\frac{\lambda_{max}}{\lambda_{min}}
]

---

# 10. Lợi ích

CLN có các ưu điểm:

* kiểm soát **Lipschitz constant trực tiếp**
* giảm **curvature outliers**
* cải thiện **Hessian conditioning**
* phù hợp với **second-order optimizers**

Ngoài ra:

* không cần eigendecomposition
* chỉ cần Jacobian-vector product.

---

# 11. Chi phí tính toán

Chi phí mỗi batch:

* 1 Jacobian-vector product
* 1 vector norm.

Độ phức tạp:

[
O(d)
]

thêm vào forward pass.

---

# 12. So sánh với Spectral Normalization

| Phương pháp   | Kiểm soát             |
| ------------- | --------------------- |
| Spectral Norm | weight matrix         |
| BatchNorm     | activation statistics |
| CLN           | Jacobian Lipschitz    |

CLN ràng buộc **toàn bộ mạng**, không chỉ từng layer.

---

# 13. Ứng dụng

CLN đặc biệt phù hợp với:

* second-order methods (LoP, Newton-like)
* networks không có BatchNorm
* training dễ bị curvature explosion.

Ngoài ra CLN có thể giúp:

* giảm overfitting
* cải thiện uncertainty calibration
* tăng robustness.

---

# 14. Hướng mở rộng

Một số biến thể tiềm năng:

### Bi-Lipschitz CLN

ràng buộc cả upper và lower bound của Jacobian.

### Entropy Jacobian Regularization

làm phẳng toàn bộ singular value spectrum của Jacobian.

### Hessian-aware CLN

ước lượng trực tiếp spectral norm của Hessian.

---

# 15. Kết luận

Curvature-Aware Lipschitz Normalization (CLN) là một phương pháp ràng buộc Lipschitz mới dựa trên **Jacobian của mạng**.

Thay vì chuẩn hóa weight hoặc feature spectrum, CLN kiểm soát trực tiếp:

* Lipschitz constant
* curvature của loss landscape.

Điều này giúp:

* ổn định quá trình huấn luyện
* cải thiện điều kiện của Hessian
* đặc biệt hữu ích cho **second-order optimization methods**.
