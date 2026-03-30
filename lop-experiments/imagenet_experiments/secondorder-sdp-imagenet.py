# %% [markdown]
# # Second-Order Optimizers + SDP on Continual ImageNet-32
# #
# **Hypothesis**: SDP + second-order optimizers solve Loss of Plasticity
# even for shallow/convolutional networks.
# #
# **Optimizers tested**:
# 1. **AdaHessian** (Yao et al., 2021) — Hutchinson diagonal Hessian
# 2. **SophiaH** (Liu et al., 2023) — Hutchinson Hessian + element-wise clipping
# 3. **Shampoo** (Gupta et al., 2018) — Full-matrix preconditioning
# 4. **ASAM** (Kwon et al., 2021) — Adaptive SAM + SGD base
# 5. **SASSHA** (baseline reference) — SAM + Hutchinson Hessian
# #
# Each optimizer runs **with and without SDP** for ablation.
# #
# **Benchmark**: Task-incremental binary classification, 2 classes/task,
# 1000 classes ImageNet-32 (600 train + 100 test per class).
# Network: ConvNet (conv1→conv2→conv3→fc1→fc2), 5000 tasks × 200 epochs/task.

# %% [markdown]
# ## 1. Imports & Setup

# %%
import os, sys, time, pickle, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

_LOP_ROOT = "/kaggle/input/datasets/mlinh776/lop-src"
_LOP_IMAGENET_DIR = os.path.join(_LOP_ROOT, 'lop', 'imagenet')
if _LOP_ROOT not in sys.path:
    sys.path.insert(0, _LOP_ROOT)

from lop.nets.conv_net import ConvNet
from lop.utils.miscellaneous import nll_accuracy as accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Metrics & SDP

# %%
NUM_CONV_LAYERS = 3
NUM_FC_LAYERS   = 2
NUM_LAYERS      = NUM_CONV_LAYERS + NUM_FC_LAYERS   # 5 total
LAYER_SIZES     = [32, 64, 512, 128, 128]           # neurons per layer (ConvNet)
TOTAL_NEURONS   = sum(LAYER_SIZES)                  # 864

@torch.no_grad()
def compute_avg_weight_magnitude(net):
    n, s = 0, 0.0
    for p in net.parameters():
        n += p.numel()
        s += torch.sum(torch.abs(p)).item()
    return s / n if n > 0 else 0.0

@torch.no_grad()
def compute_dormant_neurons(net, x_data, mini_batch_size=100, threshold=0.01):
    batch_x = x_data[:min(mini_batch_size * 5, len(x_data))]
    _, activations = net.predict(x=batch_x)
    per_layer_frac, all_alive = [], []
    total_n, total_d = 0, 0
    last_act = None
    for i, act in enumerate(activations):
        if act.ndim == 4:
            alive_score = (act != 0).float().mean(dim=(0, 2, 3))
            n_units = act.shape[1]
        else:
            alive_score = (act != 0).float().mean(dim=0)
            n_units = act.shape[1]
        dormant = (alive_score < threshold).sum().item()
        per_layer_frac.append(dormant / n_units if n_units > 0 else 0.0)
        all_alive.append(alive_score.cpu().numpy())
        total_d += dormant
        total_n += n_units
        if i == len(activations) - 1:
            last_act = act.cpu().numpy()
    agg_frac = total_d / total_n if total_n > 0 else 0.0
    return agg_frac, per_layer_frac, np.concatenate(all_alive), last_act

def compute_stable_rank(sv):
    if len(sv) == 0: return 0
    sorted_sv = np.flip(np.sort(sv))
    cumsum = np.cumsum(sorted_sv) / np.sum(sv)
    return int(np.sum(cumsum < 0.99) + 1)

def compute_stable_rank_from_activations(act):
    from scipy.linalg import svd
    if act is None: return 0
    if act.ndim > 2: act = act.reshape(act.shape[0], -1)
    if act.shape[0] == 0 or act.shape[1] == 0: return 0
    try:
        sv = svd(act, compute_uv=False, lapack_driver="gesvd")
        return compute_stable_rank(sv)
    except:
        return 0

def apply_sdp(net, gamma):
    """Spectral Diversity Preservation (SDP) at task boundary.
    σ'_i = σ̄^γ · σ_i^(1-γ)

    Applies to all layers except the output head (which is reset each task anyway).
    Conv weights are reshaped to 2D (out_channels, in*k*k) for SVD.
    """
    cond_numbers = []
    layers = [l for l in net.layers if hasattr(l, 'weight')]
    with torch.no_grad():
        for i, layer in enumerate(layers):
            is_output = (i == len(layers) - 1)
            if is_output:
                continue  # output head is reset each task — skip SDP
            W = layer.weight.data
            orig_shape = W.shape
            W2d = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
            try:
                U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
            except Exception:
                continue
            if S.numel() == 0 or S[0] < 1e-12:
                continue
            cond_numbers.append((S[0] / S[-1].clamp(min=1e-12)).item())
            s_mean = S.mean()
            S_new = (s_mean ** gamma) * (S ** (1.0 - gamma))
            W_new = U @ torch.diag(S_new) @ Vh
            layer.weight.data.copy_(W_new.reshape(orig_shape))
    return cond_numbers

print("✓ Metrics & SDP defined")

# %% [markdown]
# ## 3. ImageNet-32 Data Loading

# %%
TRAIN_IMAGES_PER_CLASS = 600
TEST_IMAGES_PER_CLASS  = 100
TOTAL_CLASSES          = 1000

DATA_DIR = '/kaggle/input/datasets/nguyenlamphuquy/imagenet/classes'
print(f"✓ ImageNet-32 data dir: {DATA_DIR}")

_class_order_file = os.path.join(_LOP_IMAGENET_DIR, 'class_order')
if os.path.isfile(_class_order_file):
    with open(_class_order_file, 'rb') as f:
        _ALL_CLASS_ORDERS = pickle.load(f)
    print(f"  ✓ Loaded class_order ({len(_ALL_CLASS_ORDERS)} runs)")
else:
    print("  ⚠ class_order not found — generating random order")
    _rng = np.random.RandomState(42)
    _ALL_CLASS_ORDERS = [_rng.permutation(TOTAL_CLASSES) for _ in range(30)]

def load_imagenet(classes):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, cls in enumerate(classes):
        data = np.load(os.path.join(DATA_DIR, str(cls) + '.npy'))
        x_train.append(data[:TRAIN_IMAGES_PER_CLASS])
        x_test.append(data[TRAIN_IMAGES_PER_CLASS:])
        y_train.append(np.full(TRAIN_IMAGES_PER_CLASS, idx))
        y_test.append(np.full(TEST_IMAGES_PER_CLASS, idx))
    return (
        torch.tensor(np.concatenate(x_train)),
        torch.from_numpy(np.concatenate(y_train)),
        torch.tensor(np.concatenate(x_test)),
        torch.from_numpy(np.concatenate(y_test)),
    )

print(f"✓ Data loading ready ({TOTAL_CLASSES} classes)")

# %% [markdown]
# ## 4. Optimizer Definitions

# %% [markdown]
# ### 4a. AdaHessian
# Source: https://github.com/davda54/ada-hessian

# %%
class Adahessian(Optimizer):
    """AdaHessian — Adaptive second-order optimizer using Hutchinson trace."""
    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0.0, hessian_power=1, lazy_hessian=1,
                 n_samples=1, seed=0):
        if not 0.0 <= lr:            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:           raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1: raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1: raise ValueError(f"Invalid beta1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1: raise ValueError(f"Invalid hessian_power: {hessian_power}")
        self.n_samples     = n_samples
        self.lazy_hessian  = lazy_hessian
        self.seed          = seed
        self.generator     = torch.Generator().manual_seed(seed)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        hessian_power=hessian_power)
        super().__init__(params, defaults)
        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.lazy_hessian == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        params = [p for p in filter(lambda p: p.grad is not None, self.get_params())
                  if self.state[p]["hessian step"] % self.lazy_hessian == 0]
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            self.state[p]["hessian step"] += 1
        if len(params) == 0: return
        if self.generator.device != params[0].device:
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)
        grads = [p.grad for p in params]
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator,
                                device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True,
                                       retain_graph=(i < self.n_samples - 1))
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        self.zero_hessian()
        self.set_hessian()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None: continue
                p.hess = p.hess.abs().clone() if not isinstance(p.hess, float) else p.hess
                if isinstance(p.hess, float): continue
                p.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg']              = torch.zeros_like(p.data)
                    state['exp_hessian_diag_sq']  = torch.zeros_like(p.data)
                exp_avg, exp_h = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_h.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                k   = group['hessian_power']
                denom = (exp_h / bc2).pow_(k / 2).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-(group['lr'] / bc1))
        return loss

print("✓ AdaHessian defined")

# %% [markdown]
# ### 4b. SophiaH
# Source: https://github.com/Liuhong99/Sophia

# %%
class SophiaH(Optimizer):
    """SophiaH — Hutchinson Hessian + element-wise clipping."""
    def __init__(self, params, lr=0.003, betas=(0.965, 0.99), eps=1e-4,
                 weight_decay=5e-4, lazy_hessian=10, n_samples=1,
                 clip_threshold=1.0, seed=0):
        if not 0.0 <= lr:            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:           raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1: raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1: raise ValueError(f"Invalid beta1: {betas[1]}")
        self.n_samples    = n_samples
        self.lazy_hessian = lazy_hessian
        self.seed         = seed
        self.generator    = torch.Generator().manual_seed(seed)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        clip_threshold=clip_threshold)
        super().__init__(params, defaults)
        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.lazy_hessian == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        params = [p for p in filter(lambda p: p.grad is not None, self.get_params())
                  if self.state[p]["hessian step"] % self.lazy_hessian == 0]
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            self.state[p]["hessian step"] += 1
        if len(params) == 0: return
        if self.generator.device != params[0].device:
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)
        grads = [p.grad for p in params]
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator,
                                device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True,
                                       retain_graph=(i < self.n_samples - 1))
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        self.zero_hessian()
        self.set_hessian()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None: continue
                if isinstance(p.hess, float): continue
                p.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg']          = torch.zeros_like(p.data)
                    state['exp_hessian_diag'] = torch.zeros_like(p.data)
                exp_avg, exp_h = state['exp_avg'], state['exp_hessian_diag']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                if (state['hessian step'] - 1) % self.lazy_hessian == 0:
                    exp_h.mul_(beta2).add_(p.hess, alpha=1 - beta2)
                denom = group['clip_threshold'] * exp_h.clamp(0, None) + group['eps']
                ratio = (exp_avg.abs() / denom).clamp(None, 1)
                p.addcmul_(exp_avg.sign(), ratio, value=-group['lr'])
        return loss

print("✓ SophiaH defined")

# %% [markdown]
# ### 4c. Shampoo
# Source: https://github.com/jettify/pytorch-optimizer

# %%
MAX_PRECOND_DIM   = 512   # use eigh for dims ≤ this; larger → diagonal approx
MAX_PRECOND_SCALE = 50    # cap inv-precond scaling per dim to prevent blow-up

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    device = matrix.device
    dim    = matrix.size(0)
    if dim > MAX_PRECOND_DIM:
        diag   = matrix.diagonal().clamp(min=1e-4)
        scaled = diag.pow(power).clamp(max=MAX_PRECOND_SCALE)
        return scaled.diag().to(device)
    matrix = matrix.cpu().double()
    matrix = 0.5 * (matrix + matrix.t())
    trace_val = matrix.trace().item()
    reg = max(1e-4, 0.01 * abs(trace_val) / dim)
    matrix.diagonal().add_(reg)
    try:
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = eigvals.clamp(min=reg)
        result  = eigvecs @ eigvals.pow(power).diag() @ eigvecs.t()
        return result.float().to(device)
    except torch._C._LinAlgError:
        return torch.eye(dim, device=device)

class Shampoo(Optimizer):
    """Shampoo — Preconditioned Stochastic Tensor Optimization."""
    def __init__(self, params, lr=1e-1, momentum=0.0, weight_decay=0.0,
                 epsilon=0.1, update_freq=50, precond_decay=0.99):
        if lr <= 0.0:         raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:    raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0:  raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        epsilon=epsilon, update_freq=update_freq, precond_decay=precond_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        for group in self.param_groups:
            decay    = group["precond_decay"]
            momentum = group["momentum"]
            wd       = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                grad          = p.grad.data.clone()
                order         = grad.ndimension()
                original_size = grad.size()
                state         = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    for dim_id, dim in enumerate(grad.size()):
                        state[f"precond_{dim_id}"] = group["epsilon"] * torch.eye(
                            dim, dtype=grad.dtype, device=grad.device)
                        state[f"inv_precond_{dim_id}"] = torch.eye(
                            dim, dtype=grad.dtype, device=grad.device)
                if wd > 0:
                    grad.add_(p.data, alpha=wd)
                if momentum > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                    grad = buf.clone()
                update = grad
                for dim_id, dim in enumerate(grad.size()):
                    precond     = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]
                    update      = update.transpose(0, dim_id).contiguous()
                    trans_size  = update.size()
                    update      = update.view(dim, -1)
                    update_t    = update.t()
                    precond.mul_(decay).add_(update @ update_t, alpha=1.0 - decay)
                    if state["step"] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1.0 / order))
                    if dim_id == order - 1:
                        update = update_t @ inv_precond
                        update = update.view(original_size)
                    else:
                        update = inv_precond @ update
                        update = update.view(trans_size)
                state["step"] += 1
                p.data.add_(update, alpha=-group["lr"])
        return loss

print("✓ Shampoo defined")

# %% [markdown]
# ### 4d. ASAM (Adaptive SAM)
# Source: https://github.com/davda54/sam

# %%
class ASAM(Optimizer):
    """Adaptive SAM — wraps a base SGD optimizer with adaptive perturbation."""
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=True, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "ASAM requires a closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

print("✓ ASAM defined")

# %% [markdown]
# ### 4e. SASSHA (reference baseline)
# Source: imagenet_experiments/sassha-imgnet.py

# %%
class SASSHA(Optimizer):
    """SASSHA — SAM + Hutchinson Hessian trace (ImageNet version with guard support)."""
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), weight_decay=0.0,
                 rho=0.1, lazy_hessian=10, n_samples=1, perturb_eps=1e-12,
                 eps=1e-4, adaptive=False, hessian_power=1.0, seed=0, **kwargs):
        if not 0.0 <= lr:            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:           raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1: raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1: raise ValueError(f"Invalid beta1: {betas[1]}")
        self.lazy_hessian    = lazy_hessian
        self.n_samples       = n_samples
        self.adaptive        = adaptive
        self.seed            = seed
        self.hessian_power_t = hessian_power
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        rho=rho, perturb_eps=perturb_eps, eps=eps)
        super().__init__(params, defaults)
        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0
        self.generator = torch.Generator().manual_seed(seed)

    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.lazy_hessian == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        params = [p for p in filter(lambda p: p.grad is not None, self.get_params())
                  if self.state[p]["hessian step"] % self.lazy_hessian == 0]
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            self.state[p]["hessian step"] += 1
        if len(params) == 0: return
        if self.generator.device != params[0].device:
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)
        grads = [p.grad for p in params]
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator,
                                device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True,
                                       retain_graph=(i < self.n_samples - 1))
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples

    @torch.no_grad()
    def perturb_weights(self, zero_grad=True):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["perturb_eps"])
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                if self.adaptive: e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p]:
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def _grad_norm(self):
        return torch.norm(torch.stack([
            ((torch.abs(p.data) if self.adaptive else 1.0) * p.grad).norm(p=2)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]), p=2)

    @torch.no_grad()
    def step(self, closure=None, compute_hessian=True):
        loss = None
        if closure is not None: loss = closure()
        if compute_hessian:
            self.zero_hessian()
            self.set_hessian()
        k = self.hessian_power_t
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None: continue
                if isinstance(p.hess, (int, float)):
                    p.hess = torch.zeros_like(p.data)
                else:
                    p.hess = p.hess.abs().clone()
                p.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 2:
                    state['step']               = 0
                    state['exp_avg']            = torch.zeros_like(p.data)
                    state['exp_hessian_diag']   = torch.zeros_like(p.data)
                    state['bias_correction2']   = 0
                exp_avg = state['exp_avg']
                exp_h   = state['exp_hessian_diag']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                bc1 = 1 - beta1 ** state['step']
                if (state['hessian step'] - 1) % self.lazy_hessian == 0:
                    exp_h.mul_(beta2).add_(p.hess, alpha=1 - beta2)
                    bc2 = 1 - beta2 ** state['step']
                    state['bias_correction2'] = bc2 ** k
                denom = ((exp_h ** k) / max(state['bias_correction2'], 1e-12)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-(group['lr'] / bc1))
        return loss

def _disable_running_stats(model):
    def _d(m):
        if isinstance(m, nn.BatchNorm2d):
            m.backup_momentum = m.momentum; m.momentum = 0
    model.apply(_d)

def _enable_running_stats(model):
    def _e(m):
        if isinstance(m, nn.BatchNorm2d) and hasattr(m, 'backup_momentum'):
            m.momentum = m.backup_momentum
    model.apply(_e)

print("✓ SASSHA defined")

# %% [markdown]
# ## 5. EMA Wrapper

# %%
class EMAWrapper:
    def __init__(self, model, decay=0.999):
        self.decay   = decay
        self._shadow = {n: p.data.clone() for n, p in model.named_parameters()}
        self._backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            self._shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        self._backup = {n: p.data.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters(): p.data.copy_(self._shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self._backup: p.data.copy_(self._backup[n])
        self._backup.clear()

    @torch.no_grad()
    def reset(self, model):
        self._shadow = {n: p.data.clone() for n, p in model.named_parameters()}

print("✓ EMA wrapper defined")

# %% [markdown]
# ## 6. Hessian Guard
# Applied to second-order methods to prevent Hessian blow-up.

# %%
class GradientExplosionGuard:
    def __init__(self, hessian_clip=1e3, hessian_floor=1e-4,
                 max_grad_norm=float('inf')):
        self.hessian_clip        = hessian_clip
        self.hessian_floor       = hessian_floor
        self.max_grad_norm       = max_grad_norm
        self.hessian_clip_count  = 0
        self.hessian_floor_count = 0

    @torch.no_grad()
    def apply(self, optimizer):
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'exp_hessian_diag_sq' in state:
                    h = state['exp_hessian_diag_sq']
                    if h.max().item() > self.hessian_clip:  self.hessian_clip_count  += 1
                    if h.min().item() < self.hessian_floor: self.hessian_floor_count += 1
                    h.clamp_(self.hessian_floor, self.hessian_clip)
                if 'exp_hessian_diag' in state:
                    h = state['exp_hessian_diag']
                    if h.max().item() > self.hessian_clip:  self.hessian_clip_count  += 1
                    if h.min().item() < self.hessian_floor: self.hessian_floor_count += 1
                    h.clamp_(self.hessian_floor, self.hessian_clip)
                if hasattr(p, 'hess') and not isinstance(p.hess, float):
                    p.hess.clamp_(-self.hessian_clip, self.hessian_clip)

    @torch.no_grad()
    def clip_grad_norm(self, model):
        if math.isfinite(self.max_grad_norm):
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

    def summary(self):
        return f"hess_clips={self.hessian_clip_count}, hess_floors={self.hessian_floor_count}"

print("✓ GradientExplosionGuard defined")

# %% [markdown]
# ## 7. Configs

# %%
SEED               = 42
SAVE_EVERY_N_TASKS = 50
TIME_LIMIT_SECONDS = 11.5 * 3600

NUM_TASKS    = 2000
NUM_CLASSES  = 2      # binary classification per task
NUM_EPOCHS   = 200    # showings per task
MINI_BATCH   = 100

EXAMPLES_PER_EPOCH = TRAIN_IMAGES_PER_CLASS * NUM_CLASSES   # 1200

SDP_GAMMA = 0.3

CONFIGS = {
    # 1. AdaHessian + SDP
    'adahessian_sdp': dict(
        optimizer='adahessian',
        lr=0.003, betas=(0.9, 0.999), weight_decay=5e-4, eps=1e-4,
        hessian_power=1.0, lazy_hessian=10, n_samples=1,
        use_guard=True, hessian_clip=1e3, hessian_floor=1e-4,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 2. AdaHessian (no SDP)
    'adahessian_nosdp': dict(
        optimizer='adahessian',
        lr=0.003, betas=(0.9, 0.999), weight_decay=5e-4, eps=1e-4,
        hessian_power=1.0, lazy_hessian=10, n_samples=1,
        use_guard=True, hessian_clip=1e3, hessian_floor=1e-4,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=0.0,
    ),
    # 3. SophiaH + SDP
    'sophia_sdp': dict(
        optimizer='sophia',
        lr=0.003, betas=(0.965, 0.99), weight_decay=5e-4, eps=1e-4,
        clip_threshold=1.0, lazy_hessian=10, n_samples=1,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 4. SophiaH (no SDP)
    'sophia_nosdp': dict(
        optimizer='sophia',
        lr=0.003, betas=(0.965, 0.99), weight_decay=5e-4, eps=1e-4,
        clip_threshold=1.0, lazy_hessian=10, n_samples=1,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=0.0,
    ),
    # 5. Shampoo + SDP
    'shampoo_sdp': dict(
        optimizer='shampoo',
        lr=0.003, momentum=0.9, weight_decay=3e-5, epsilon=0.1,
        update_freq=50,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 6. Shampoo (no SDP)
    'shampoo_nosdp': dict(
        optimizer='shampoo',
        lr=0.003, momentum=0.9, weight_decay=3e-5, epsilon=0.1,
        update_freq=50,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=0.0,
    ),
    # 7. ASAM + SDP
    'asam_sdp': dict(
        optimizer='asam',
        lr=0.003, rho=0.05, momentum=0.9, weight_decay=5e-4,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 8. ASAM (no SDP)
    'asam_nosdp': dict(
        optimizer='asam',
        lr=0.003, rho=0.05, momentum=0.9, weight_decay=5e-4,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=0.0,
    ),
    # 9. SASSHA + SDP (reference)
    'sassha_sdp': dict(
        optimizer='sassha',
        lr=0.01, betas=(0.9, 0.999), weight_decay=5e-4, rho=0.1,
        lazy_hessian=10, n_samples=1, eps=1e-4, hessian_power=1.0,
        use_guard=True, hessian_clip=1e3, hessian_floor=1e-4,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 10. SASSHA (no SDP)
    'sassha_nosdp': dict(
        optimizer='sassha',
        lr=0.01, betas=(0.9, 0.999), weight_decay=5e-4, rho=0.1,
        lazy_hessian=10, n_samples=1, eps=1e-4, hessian_power=1.0,
        use_guard=True, hessian_clip=1e3, hessian_floor=1e-4,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=0.0,
    ),
}

METHODS_TO_RUN = [
    'adahessian_sdp', 'adahessian_nosdp',
    'sophia_sdp',     'sophia_nosdp',
    'shampoo_sdp',    'shampoo_nosdp',
    'asam_sdp',       'asam_nosdp',
    'sassha_sdp',     'sassha_nosdp',
]

RESULTS_DIR = os.path.join('permuted_imagenet_results', 'secondorder_sdp')
CKPT_DIR    = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"✓ Config: {NUM_TASKS} tasks × {NUM_EPOCHS} epochs/task, "
      f"{NUM_CLASSES} classes/task, batch={MINI_BATCH}")
print(f"  Methods: {METHODS_TO_RUN}")

# %% [markdown]
# ## 8. Build Optimizer

# %%
def build_optimizer(config, model):
    opt_type = config['optimizer']

    if opt_type == 'adahessian':
        return Adahessian(
            model.parameters(), lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.0),
            eps=config.get('eps', 1e-4),
            hessian_power=config.get('hessian_power', 1.0),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1), seed=SEED)

    elif opt_type == 'sophia':
        return SophiaH(
            model.parameters(), lr=config['lr'],
            betas=config.get('betas', (0.965, 0.99)),
            weight_decay=config.get('weight_decay', 5e-4),
            eps=config.get('eps', 1e-4),
            clip_threshold=config.get('clip_threshold', 1.0),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1), seed=SEED)

    elif opt_type == 'shampoo':
        return Shampoo(
            model.parameters(), lr=config['lr'],
            momentum=config.get('momentum', 0.0),
            weight_decay=config.get('weight_decay', 0.0),
            epsilon=config.get('epsilon', 0.1),
            update_freq=config.get('update_freq', 50))

    elif opt_type == 'asam':
        return ASAM(
            model.parameters(), torch.optim.SGD,
            rho=config.get('rho', 0.05), adaptive=True,
            lr=config['lr'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0))

    elif opt_type == 'sassha':
        return SASSHA(
            model.parameters(), lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.0),
            rho=config.get('rho', 0.1),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1),
            eps=config.get('eps', 1e-4),
            hessian_power=config.get('hessian_power', 1.0),
            seed=SEED)

    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

print("✓ build_optimizer defined")

# %% [markdown]
# ## 9. Unified Training Loop

# %%
def _ckpt_path(method_name):
    return os.path.join(CKPT_DIR, f"ckpt_{method_name}.pt")

def _needs_create_graph(config):
    return config['optimizer'] in ('adahessian', 'sophia', 'sassha')

def run_method(method_name, config, run_idx=0):
    """Unified training loop for all second-order optimizers ± SDP on ImageNet-32."""
    opt_type  = config['optimizer']
    sdp_gamma = config.get('sdp_gamma', 0.0)

    print(f"\n{'='*70}")
    print(f"  {method_name} (optimizer={opt_type}) — Continual ImageNet-32 ({NUM_TASKS} tasks)")
    print(f"  SDP: {'enabled γ=' + str(sdp_gamma) if sdp_gamma > 0 else 'disabled'}")
    print(f"{'='*70}")

    wall_clock_start = time.time()
    torch.manual_seed(SEED); torch.cuda.manual_seed(SEED); np.random.seed(SEED)

    # ── Class order ──
    class_order = _ALL_CLASS_ORDERS[run_idx % len(_ALL_CLASS_ORDERS)]
    n_reps      = int(NUM_CLASSES * NUM_TASKS / TOTAL_CLASSES) + 1
    class_order = np.concatenate([class_order] * n_reps)

    # ── Network ──
    net       = ConvNet(num_classes=NUM_CLASSES)
    optimizer = build_optimizer(config, net)

    guard = None
    if config.get('use_guard', False) and opt_type in ('adahessian', 'sassha'):
        guard = GradientExplosionGuard(
            hessian_clip=config.get('hessian_clip', 1e3),
            hessian_floor=config.get('hessian_floor', 1e-4))

    ema = EMAWrapper(net, config.get('ema_decay', 0.999)) if config.get('use_ema', False) else None
    ls  = 0.1 if config.get('label_smoothing', False) else 0.0
    loss_fn = lambda logits, target: F.cross_entropy(logits, target, label_smoothing=ls)

    create_graph = _needs_create_graph(config)

    # ── Metric tensors ──
    train_accuracies    = torch.zeros((NUM_TASKS, NUM_EPOCHS), dtype=torch.float)
    test_accuracies     = torch.zeros((NUM_TASKS, NUM_EPOCHS), dtype=torch.float)
    all_weight_mag      = torch.zeros((NUM_TASKS, NUM_EPOCHS), dtype=torch.float)
    all_dormant_frac    = torch.zeros((NUM_TASKS, NUM_EPOCHS), dtype=torch.float)
    all_dormant_layers  = torch.zeros((NUM_TASKS, NUM_EPOCHS, NUM_LAYERS), dtype=torch.float)
    all_stable_rank     = torch.zeros((NUM_TASKS, NUM_EPOCHS), dtype=torch.float)
    all_sdp_cond        = torch.zeros(NUM_TASKS, dtype=torch.float)

    # ── Checkpoint resume ──
    ckpt_file  = _ckpt_path(method_name)
    start_task = 0
    if os.path.isfile(ckpt_file):
        print(f"  Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        train_accuracies   = ckpt.get('train_accuracies',   train_accuracies)
        test_accuracies    = ckpt.get('test_accuracies',    test_accuracies)
        all_weight_mag     = ckpt.get('all_weight_mag',     all_weight_mag)
        all_dormant_frac   = ckpt.get('all_dormant_frac',   all_dormant_frac)
        all_dormant_layers = ckpt.get('all_dormant_layers', all_dormant_layers)
        all_stable_rank    = ckpt.get('all_stable_rank',    all_stable_rank)
        all_sdp_cond       = ckpt.get('all_sdp_cond',       all_sdp_cond)
        start_task         = ckpt['task'] + 1
        if ema is not None and 'ema_shadow' in ckpt:
            ema._shadow = ckpt['ema_shadow']
        if hasattr(optimizer, 'get_params'):
            for p in optimizer.get_params():
                p.hess = 0.0
                optimizer.state[p]["hessian step"] = 0
        if 'np_rng' in ckpt: np.random.set_state(ckpt['np_rng'])
        print(f"  ✓ Resumed from task {start_task}")
        del ckpt; torch.cuda.empty_cache()
    else:
        print("  (no checkpoint — training from scratch)")

    def save_checkpoint(task_idx, reason="periodic"):
        ckpt_data = {
            'task': task_idx, 'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_accuracies': train_accuracies, 'test_accuracies': test_accuracies,
            'all_weight_mag': all_weight_mag, 'all_dormant_frac': all_dormant_frac,
            'all_dormant_layers': all_dormant_layers, 'all_stable_rank': all_stable_rank,
            'all_sdp_cond': all_sdp_cond, 'params': config,
            'np_rng': np.random.get_state(),
        }
        if ema is not None:   ckpt_data['ema_shadow'] = ema._shadow
        if guard is not None: ckpt_data['guard'] = guard.summary()
        torch.save(ckpt_data, ckpt_file)
        elapsed = time.time() - wall_clock_start
        print(f"  Checkpoint saved at task {task_idx} ({reason}) [{elapsed/3600:.1f}h elapsed]")

    x_train = x_test = y_train = y_test = None
    time_limit_hit = False

    # ════════════════════════════════════════════════════════════════
    #  Task loop
    # ════════════════════════════════════════════════════════════════
    for task_idx in range(start_task, NUM_TASKS):
        task_start = time.time()
        elapsed    = time.time() - wall_clock_start
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"\n  Time limit ({elapsed/3600:.1f}h). Saving.")
            save_checkpoint(task_idx - 1, reason="time_limit")
            time_limit_hit = True; break

        # ── 1. Load task data ──
        del x_train, x_test, y_train, y_test
        task_classes  = class_order[task_idx * NUM_CLASSES:(task_idx + 1) * NUM_CLASSES]
        x_train, y_train, x_test, y_test = load_imagenet(task_classes)
        x_train = x_train.float()
        x_test  = x_test.float()
        if device.type == 'cuda':
            x_train, x_test = x_train.to(device), x_test.to(device)
            y_train, y_test = y_train.to(device), y_test.to(device)

        # ── 2. Reset output head (task-incremental binary setting) ──
        net.layers[-1].weight.data.zero_()
        net.layers[-1].bias.data.zero_()

        # ── 3. SDP at task boundary ──
        if sdp_gamma > 0 and task_idx > 0:
            cond_nums = apply_sdp(net, sdp_gamma)
            avg_cond  = sum(cond_nums) / max(len(cond_nums), 1)
            all_sdp_cond[task_idx] = avg_cond
            if task_idx % 50 == 0:
                print(f"    SDP applied: avg condition number = {avg_cond:.1f}")

        # ── 4. Reset EMA at task boundary ──
        if ema is not None: ema.reset(net)

        # ── 5. Reset Shampoo preconditioners at task boundary ──
        if opt_type == 'shampoo':
            eps_val = config.get('epsilon', 0.1)
            for state in optimizer.state.values():
                dim_id = 0
                while f"precond_{dim_id}" in state:
                    dim = state[f"precond_{dim_id}"].size(0)
                    state[f"precond_{dim_id}"].copy_(
                        eps_val * torch.eye(dim,
                                            device=state[f"precond_{dim_id}"].device,
                                            dtype=state[f"precond_{dim_id}"].dtype))
                    dim_id += 1

        # ── 6. Train epochs ──
        for epoch_idx in range(NUM_EPOCHS):
            net.train()
            perm      = np.random.permutation(EXAMPLES_PER_EPOCH)
            x_shuf    = x_train[perm]
            y_shuf    = y_train[perm]
            batch_accs = []

            for start_idx in range(0, EXAMPLES_PER_EPOCH, MINI_BATCH):
                batch_x = x_shuf[start_idx:start_idx + MINI_BATCH]
                batch_y = y_shuf[start_idx:start_idx + MINI_BATCH]

                # ── Optimizer-specific protocol ──
                if opt_type == 'sassha':
                    _enable_running_stats(net)
                    optimizer.zero_grad()
                    logits, _ = net.predict(x=batch_x)
                    loss = loss_fn(logits, batch_y)
                    loss.backward()
                    optimizer.perturb_weights(zero_grad=True)
                    _disable_running_stats(net)
                    logits_p, _ = net.predict(x=batch_x)
                    loss_p = loss_fn(logits_p, batch_y)
                    loss_p.backward(create_graph=True)
                    optimizer.unperturb()
                    _enable_running_stats(net)
                    if guard is not None: guard.clip_grad_norm(net)
                    optimizer.zero_hessian()
                    optimizer.set_hessian()
                    if guard is not None: guard.apply(optimizer)
                    optimizer.step(compute_hessian=False)
                    optimizer.zero_grad(set_to_none=True)

                elif opt_type == 'asam':
                    optimizer.zero_grad()
                    logits, _ = net.predict(x=batch_x)
                    loss = loss_fn(logits, batch_y)
                    loss.backward()
                    def closure():
                        logits2, _ = net.predict(x=batch_x)
                        loss2 = loss_fn(logits2, batch_y)
                        loss2.backward()
                        return loss2
                    optimizer.step(closure=closure)

                elif opt_type in ('adahessian', 'sophia'):
                    optimizer.zero_grad()
                    logits, _ = net.predict(x=batch_x)
                    loss = loss_fn(logits, batch_y)
                    loss.backward(create_graph=True)
                    if guard is not None: guard.apply(optimizer)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                elif opt_type == 'shampoo':
                    optimizer.zero_grad()
                    logits, _ = net.predict(x=batch_x)
                    loss = loss_fn(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                if ema is not None: ema.update(net)

                with torch.no_grad():
                    batch_accs.append(
                        accuracy(F.softmax(logits.detach(), dim=1), batch_y).item())

            # ── Per-epoch eval ──
            net.eval()
            with torch.no_grad():
                train_accuracies[task_idx, epoch_idx] = float(np.mean(batch_accs))

                if ema is not None: ema.apply(net)
                test_accs = []
                for si in range(0, x_test.shape[0], MINI_BATCH):
                    tb_x = x_test[si:si + MINI_BATCH]
                    tb_y = y_test[si:si + MINI_BATCH]
                    to, _ = net.predict(x=tb_x)
                    test_accs.append(accuracy(F.softmax(to, dim=1), tb_y).item())
                test_accuracies[task_idx, epoch_idx] = float(np.mean(test_accs))
                if ema is not None: ema.restore(net)

                wm = compute_avg_weight_magnitude(net)
                all_weight_mag[task_idx, epoch_idx] = wm
                agg_frac, layer_fracs, _, last_act = compute_dormant_neurons(
                    net, x_test, mini_batch_size=MINI_BATCH)
                all_dormant_frac[task_idx, epoch_idx]     = agg_frac
                all_dormant_layers[task_idx, epoch_idx]   = torch.tensor(layer_fracs)
                sr = compute_stable_rank_from_activations(last_act)
                all_stable_rank[task_idx, epoch_idx]      = sr

            if epoch_idx % 50 == 0 or epoch_idx == NUM_EPOCHS - 1:
                layer_str = ' '.join([f'{f:.2f}' for f in layer_fracs])
                sdp_str   = f" CondN={all_sdp_cond[task_idx]:.1f}" if sdp_gamma > 0 and task_idx > 0 else ""
                print(f"    Task {task_idx:4d} Ep {epoch_idx:3d}/{NUM_EPOCHS} | "
                      f"TrainAcc={train_accuracies[task_idx, epoch_idx]:.4f}  "
                      f"TestAcc={test_accuracies[task_idx, epoch_idx]:.4f}  "
                      f"Dormant={agg_frac:.3f} [{layer_str}]  "
                      f"SR={sr:.0f}  AvgW={wm:.4f}{sdp_str}")

        task_time = time.time() - task_start
        if task_idx % 50 == 0 or task_idx == NUM_TASKS - 1:
            guard_str = f" | {guard.summary()}" if guard is not None and task_idx % 200 == 0 else ""
            print(f"  [{method_name}] Task {task_idx:4d}/{NUM_TASKS} | "
                  f"TrainAcc={train_accuracies[task_idx, -1]:.4f}  "
                  f"TestAcc={test_accuracies[task_idx, -1]:.4f}  "
                  f"Dormant={all_dormant_frac[task_idx, -1]:.3f}  "
                  f"SR={all_stable_rank[task_idx, -1]:.0f}  "
                  f"AvgW={all_weight_mag[task_idx, -1]:.4f}  "
                  f"{task_time:.1f}s{guard_str}")

        if (task_idx + 1) % SAVE_EVERY_N_TASKS == 0 or task_idx == NUM_TASKS - 1:
            save_checkpoint(task_idx,
                            reason="periodic" if task_idx < NUM_TASKS - 1 else "completed")

    # ── Save final results ──
    result_file = os.path.join(RESULTS_DIR, f'{method_name}_results.pt')
    torch.save({
        'train_accuracies':  train_accuracies.cpu(),
        'test_accuracies':   test_accuracies.cpu(),
        'all_weight_mag':    all_weight_mag.cpu(),
        'all_dormant_frac':  all_dormant_frac.cpu(),
        'all_dormant_layers': all_dormant_layers.cpu(),
        'all_stable_rank':   all_stable_rank.cpu(),
        'all_sdp_cond':      all_sdp_cond.cpu(),
    }, result_file)
    print(f"  ✓ Results saved to {result_file}")

    return (train_accuracies, test_accuracies, all_weight_mag,
            all_dormant_frac, all_dormant_layers, all_stable_rank, all_sdp_cond)

print("✓ Unified training loop defined")

# %% [markdown]
# ## 10. Run All Experiments

# %%
all_results = {}
for method in METHODS_TO_RUN:
    cfg    = CONFIGS[method]
    result = run_method(method, cfg)
    all_results[method] = {
        'train_acc':  result[0],
        'test_acc':   result[1],
        'wmag':       result[2],
        'dormant':    result[3],
        'dorm_layer': result[4],
        'sr':         result[5],
        'cond':       result[6],
    }

# %% [markdown]
# ## 11. Results Plots

# %%
EPOCH_WINDOW = 10   # smooth over task windows

METHOD_STYLES = {
    'adahessian_sdp':   {'color': '#E91E63', 'ls': '-',  'label': 'AdaHessian+SDP'},
    'adahessian_nosdp': {'color': '#E91E63', 'ls': '--', 'label': 'AdaHessian'},
    'sophia_sdp':       {'color': '#9C27B0', 'ls': '-',  'label': 'SophiaH+SDP'},
    'sophia_nosdp':     {'color': '#9C27B0', 'ls': '--', 'label': 'SophiaH'},
    'shampoo_sdp':      {'color': '#FF9800', 'ls': '-',  'label': 'Shampoo+SDP'},
    'shampoo_nosdp':    {'color': '#FF9800', 'ls': '--', 'label': 'Shampoo'},
    'asam_sdp':         {'color': '#4CAF50', 'ls': '-',  'label': 'ASAM+SDP'},
    'asam_nosdp':       {'color': '#4CAF50', 'ls': '--', 'label': 'ASAM'},
    'sassha_sdp':       {'color': '#2196F3', 'ls': '-',  'label': 'SASSHA+SDP'},
    'sassha_nosdp':     {'color': '#2196F3', 'ls': '--', 'label': 'SASSHA'},
}

def _style(m): return METHOD_STYLES.get(m, {'color': 'gray', 'ls': '-', 'label': m})
def _clean(ax):
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def smooth_tasks(arr, w=EPOCH_WINDOW):
    # arr shape: (num_tasks,) — average over window
    n = len(arr) // w
    return np.array([arr[i*w:(i+1)*w].mean() for i in range(n)])

fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('Second-Order Optimizers ± SDP — Continual ImageNet-32\n'
             f'ConvNet, {NUM_TASKS} tasks × {NUM_EPOCHS} epochs, '
             f'{NUM_CLASSES} classes/task, batch={MINI_BATCH}',
             fontsize=14, fontweight='bold')

# ── (0,0) Final-epoch Test Accuracy per task ──
ax = axes[0, 0]
for m, d in all_results.items():
    s  = _style(m)
    ta = d['test_acc'][:, -1].numpy()   # final epoch per task
    sm = smooth_tasks(ta) * 100
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Test Acc (%)')
ax.set_title('Test Accuracy (final epoch, smoothed)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (0,1) Final-epoch Train Accuracy ──
ax = axes[0, 1]
for m, d in all_results.items():
    s  = _style(m)
    ta = d['train_acc'][:, -1].numpy()
    sm = smooth_tasks(ta) * 100
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Train Acc (%)')
ax.set_title('Train Accuracy (final epoch, smoothed)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (1,0) Stable Rank ──
ax = axes[1, 0]
for m, d in all_results.items():
    s  = _style(m)
    sr = d['sr'][:, -1].numpy()
    sm = smooth_tasks(sr)
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Stable Rank')
ax.set_title('Stable Rank (last activation, final epoch)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (1,1) Dormant Neuron Fraction ──
ax = axes[1, 1]
for m, d in all_results.items():
    s    = _style(m)
    dorm = d['dormant'][:, -1].numpy() * 100
    sm   = smooth_tasks(dorm)
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Dormant (%)')
ax.set_title('Dormant Neuron Fraction (final epoch)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (2,0) Avg Weight Magnitude ──
ax = axes[2, 0]
for m, d in all_results.items():
    s  = _style(m)
    wm = d['wmag'][:, -1].numpy()
    sm = smooth_tasks(wm)
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('AvgW')
ax.set_title('Avg Weight Magnitude')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (2,1) SDP Condition Number ──
ax = axes[2, 1]
for m, d in all_results.items():
    if CONFIGS[m].get('sdp_gamma', 0) == 0: continue
    s    = _style(m)
    cond = d['cond'][1:].numpy()   # skip task 0 (no SDP applied)
    sm   = smooth_tasks(cond)
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Condition Number')
ax.set_title('SDP Avg Condition Number (SDP methods only)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

plt.tight_layout()
plot_file = os.path.join(RESULTS_DIR, 'imagenet_secondorder_sdp_comparison.png')
plt.savefig(plot_file, dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Main comparison plot saved to {plot_file}")

# %% [markdown]
# ## 12. SDP Ablation: Δ(metric) = SDP − noSDP

# %%
OPT_PAIRS = [
    ('adahessian_sdp', 'adahessian_nosdp', 'AdaHessian', '#E91E63'),
    ('sophia_sdp',     'sophia_nosdp',     'SophiaH',    '#9C27B0'),
    ('shampoo_sdp',    'shampoo_nosdp',    'Shampoo',    '#FF9800'),
    ('asam_sdp',       'asam_nosdp',       'ASAM',       '#4CAF50'),
    ('sassha_sdp',     'sassha_nosdp',     'SASSHA',     '#2196F3'),
]

fig_ab, axes_ab = plt.subplots(1, 3, figsize=(21, 5))
fig_ab.suptitle('SDP Ablation: Δ = (with SDP) − (without SDP) — ImageNet-32',
                fontsize=13, fontweight='bold')

for sdp_key, nosdp_key, label, color in OPT_PAIRS:
    if sdp_key not in all_results or nosdp_key not in all_results:
        continue
    d_sdp = all_results[sdp_key]
    d_no  = all_results[nosdp_key]

    delta_test = d_sdp['test_acc'][:, -1].numpy() - d_no['test_acc'][:, -1].numpy()
    axes_ab[0].plot(smooth_tasks(delta_test) * 100, color=color, lw=2.5, label=label)

    delta_sr = d_sdp['sr'][:, -1].numpy() - d_no['sr'][:, -1].numpy()
    axes_ab[1].plot(smooth_tasks(delta_sr), color=color, lw=2.5, label=label)

    delta_dorm = (d_sdp['dormant'][:, -1].numpy() - d_no['dormant'][:, -1].numpy()) * 100
    axes_ab[2].plot(smooth_tasks(delta_dorm), color=color, lw=2.5, label=label)

for i, (title, ylabel) in enumerate([
    ('Δ Test Accuracy', 'Δ Acc (%)'),
    ('Δ Stable Rank',   'Δ SR'),
    ('Δ Dormant',       'Δ Dormant (%)'),
]):
    axes_ab[i].set_title(title); axes_ab[i].set_xlabel('Task window')
    axes_ab[i].set_ylabel(ylabel)
    axes_ab[i].axhline(0, color='black', ls=':', lw=0.8, alpha=0.5)
    axes_ab[i].legend(fontsize=9); _clean(axes_ab[i])

plt.tight_layout()
plot_abl = os.path.join(RESULTS_DIR, 'imagenet_sdp_ablation.png')
plt.savefig(plot_abl, dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ SDP ablation plot saved to {plot_abl}")

# %% [markdown]
# ## 13. Summary Table

# %%
N_FINAL = 100   # average over last N tasks
print(f"\n{'='*95}")
print(f"  Second-Order Optimizers ± SDP — ImageNet-32 — Final {N_FINAL}-task average")
print(f"{'='*95}")
header = (f"{'Method':<22} {'TestAcc%':>9} {'TrainAcc%':>10} "
          f"{'StableRank':>11} {'Dormant%':>9} {'AvgW':>8}")
print(header)
print('─' * 95)

for method in METHODS_TO_RUN:
    if method not in all_results: continue
    d  = all_results[method]
    s  = _style(method)
    ta = d['test_acc'][-N_FINAL:, -1].mean().item() * 100
    tr = d['train_acc'][-N_FINAL:, -1].mean().item() * 100
    sr = d['sr'][-N_FINAL:, -1].mean().item()
    do = d['dormant'][-N_FINAL:, -1].mean().item() * 100
    wm = d['wmag'][-N_FINAL:, -1].mean().item()
    print(f"  {s['label']:<20} {ta:>9.2f} {tr:>10.2f} {sr:>11.1f} {do:>9.2f} {wm:>8.4f}")

print('=' * 95)
