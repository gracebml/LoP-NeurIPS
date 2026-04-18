# %% [markdown]
# # Second-Order Optimizers + SDP on Continual Permuted MNIST
# #
# **Hypothesis**: SDP + second-order optimizers solve Loss of Plasticity
# even for shallow networks.
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
# **Benchmark**: Online Permuted MNIST (CBP paper setup).
# Network: FC 784 → 2000 × 5 → 10, ReLU.
# 800 tasks × 60 000 examples/task (full MNIST train), batch size 32.

# %% [markdown]
# ## 1. Imports & Setup

# %%
import os, sys, time, pickle, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

_LOP_ROOT = "/kaggle/input/datasets/mlinh776/lop-src"
if _LOP_ROOT not in sys.path:
    sys.path.insert(0, _LOP_ROOT)

from lop.nets.deep_ffnn import DeepFFNN
from lop.utils.miscellaneous import nll_accuracy as accuracy
from lop.utils.miscellaneous import compute_matrix_rank_summaries

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Metrics & SDP

# %%
NUM_HIDDEN_LAYERS = 5
NUM_FEATURES      = 2000
INPUT_SIZE        = 784
CLASSES_PER_TASK  = 10

@torch.no_grad()
def compute_avg_weight_magnitude(net):
    n, s = 0, 0.0
    for p in net.parameters():
        n += p.numel()
        s += torch.sum(torch.abs(p)).item()
    return s / n if n > 0 else 0.0

@torch.no_grad()
def compute_dead_neurons(net, probe_x):
    net.eval()
    _, hidden_acts = net.predict(probe_x)
    dead_per_layer = []
    for act in hidden_acts:
        dead_per_layer.append((act.abs().sum(dim=0) == 0).sum().item())
    return dead_per_layer

def compute_stable_rank(sv):
    if len(sv) == 0: return 0
    sorted_sv = np.flip(np.sort(sv))
    cumsum = np.cumsum(sorted_sv) / np.sum(sv)
    return int(np.sum(cumsum < 0.99) + 1)

def compute_effective_rank(singular_values):
    if len(singular_values) == 0: return 0.0
    norm_sv = singular_values / np.sum(np.abs(singular_values))
    entropy = 0.0
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * np.log(p)
    return float(np.e ** entropy)

@torch.no_grad()
def compute_dormant_neurons_ffnn(net, probe_x, threshold=0.01):
    net.eval()
    _, hidden_acts = net.predict(probe_x)
    per_layer_frac = []
    total_d, total_n = 0, 0
    last_act = None
    for i, act in enumerate(hidden_acts):
        alive_score = (act != 0).float().mean(dim=0)
        n_units = act.shape[1]
        dormant = (alive_score < threshold).sum().item()
        per_layer_frac.append(dormant / n_units if n_units > 0 else 0.0)
        total_d += dormant; total_n += n_units
        if i == len(hidden_acts) - 1:
            last_act = act.cpu().numpy()
    agg_frac = total_d / total_n if total_n > 0 else 0.0
    return agg_frac, per_layer_frac, last_act

def compute_stable_rank_from_activations(act):
    from scipy.linalg import svd
    if act is None: return 0
    if act.ndim > 2: act = act.reshape(act.shape[0], -1)
    if act.shape[0] == 0 or act.shape[1] == 0: return 0
    try:
        sv = svd(act, compute_uv=False, lapack_driver="gesvd")
        return compute_stable_rank(sv)
    except: return 0

def apply_sdp(net, gamma):
    """Spectral Diversity Preservation (SDP) at task boundary.
    σ'_i = σ̄^γ · σ_i^(1-γ)

    Skips the output layer (W_out): it has rank = num_classes at most,
    compressing it further collapses the last hidden layer representation.
    """
    cond_numbers = []
    modules = [m for m in net.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for i, module in enumerate(modules):
            is_output_layer = (i == len(modules) - 1)
            if is_output_layer:
                continue   # skip W_out — structural rank-10 bottleneck
            W = module.weight.data
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception:
                continue
            if S.numel() == 0 or S[0] < 1e-12:
                continue
            cond_numbers.append((S[0] / S[-1].clamp(min=1e-12)).item())
            s_mean = S.mean()
            S_new = (s_mean ** gamma) * (S ** (1.0 - gamma))
            W_new = U @ torch.diag(S_new) @ Vh
            module.weight.data.copy_(W_new)
    return cond_numbers

print("✓ Metrics & SDP defined")

# %% [markdown]
# ## 3. MNIST Data Loading

# %%
MNIST_CACHE = "data/mnist_"

def _build_mnist_cache(cache_path: str = MNIST_CACHE):
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root="data", train=True,  transform=tfm, download=True)
    test_ds  = torchvision.datasets.MNIST(root="data", train=False, transform=tfm, download=True)
    def _load_all(ds):
        loader = DataLoader(ds, batch_size=len(ds), shuffle=True)
        imgs, labels = next(iter(loader))
        return imgs.flatten(start_dim=1), labels
    print("  Loading train …", end=" ")
    x, y = _load_all(train_ds)
    print("done.  Loading test …", end=" ")
    x_test, y_test = _load_all(test_ds)
    print("done.")
    with open(cache_path, 'wb+') as f:
        pickle.dump([x, y, x_test, y_test], f)
    print(f"  Cached at '{cache_path}'")
    return x, y, x_test, y_test

def get_mnist(cache_path: str = MNIST_CACHE):
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb+') as f:
            x, y, x_test, y_test = pickle.load(f)
    else:
        print(f"Cache '{cache_path}' not found — building …")
        x, y, x_test, y_test = _build_mnist_cache(cache_path)
    return x, y, x_test, y_test

os.makedirs("data", exist_ok=True)
print("Loading MNIST …")
x_mnist, y_mnist, x_test_mnist, y_test_mnist = get_mnist()
print(f"  Train : {x_mnist.shape}  labels: {y_mnist.shape}")
print(f"  Test  : {x_test_mnist.shape}  labels: {y_test_mnist.shape}")

IMAGES_PER_CLASS  = 6000
EXAMPLES_PER_TASK = IMAGES_PER_CLASS * CLASSES_PER_TASK
print(f"\n  examples_per_task = {EXAMPLES_PER_TASK}")
print("✓ MNIST ready")

# %% [markdown]
# ## 4. Optimizer Definitions
# #
# All optimizers from official sources in `Sassha/optimizers/`.

# %% [markdown]
# ### 4a. AdaHessian
# Source: https://github.com/davda54/ada-hessian

# %%
class Adahessian(Optimizer):
    """AdaHessian — Adaptive second-order optimizer using Hutchinson trace."""
    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0.0, hessian_power=1, lazy_hessian=1,
                 n_samples=1, seed=0):
        if not 0.0 <= lr: raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0: raise ValueError(f"Invalid hessian_power: {hessian_power}")
        self.n_samples = n_samples
        self.lazy_hessian = lazy_hessian
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)
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
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.lazy_hessian == 0:
                params.append(p)
            self.state[p]["hessian step"] += 1
        if len(params) == 0: return
        if self.generator.device != params[0].device:
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)
        grads = [p.grad for p in params]
        last_sample = self.n_samples - 1
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < last_sample)
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
                if p.dim() <= 2:
                    p.hess = p.hess.abs().clone()
                if p.dim() == 4:
                    p.hess = torch.mean(p.hess.abs(), dim=[2, 3], keepdim=True).expand_as(p.hess).clone()
                p.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

print("✓ AdaHessian defined")

# %% [markdown]
# ### 4b. SophiaH
# Source: https://github.com/Liuhong99/Sophia

# %%
class SophiaH(Optimizer):
    """SophiaH — Hutchinson Hessian + element-wise clipping."""
    def __init__(self, params, lr=0.15, betas=(0.965, 0.99), eps=1e-15,
                 weight_decay=1e-1, lazy_hessian=10, n_samples=1,
                 clip_threshold=0.04, seed=0):
        if not 0.0 <= lr: raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta1: {betas[1]}")
        self.n_samples = n_samples
        self.lazy_hessian = lazy_hessian
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)
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
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.lazy_hessian == 0:
                params.append(p)
            self.state[p]["hessian step"] += 1
        if len(params) == 0: return
        if self.generator.device != params[0].device:
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)
        grads = [p.grad for p in params]
        last_sample = self.n_samples - 1
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < last_sample)
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
                p.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag'] = torch.zeros_like(p.data)
                exp_avg, exp_hessian_diag = state['exp_avg'], state['exp_hessian_diag']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                if (state['hessian step'] - 1) % self.lazy_hessian == 0:
                    exp_hessian_diag.mul_(beta2).add_(p.hess, alpha=1 - beta2)
                step_size = group['lr']
                denom = group['clip_threshold'] * exp_hessian_diag.clamp(0, None) + group['eps']
                ratio = (exp_avg.abs() / denom).clamp(None, 1)
                p.addcmul_(exp_avg.sign(), ratio, value=-step_size)
        return loss

print("✓ SophiaH defined")

# %% [markdown]
# ### 4c. Shampoo
# Source: https://github.com/jettify/pytorch-optimizer (Shampoo)

# %%
MAX_PRECOND_DIM = 512   # eigh only for dims ≤ this; larger → diagonal approx
MAX_PRECOND_SCALE = 50  # cap scaling per dimension to prevent blow-up

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    device = matrix.device
    dim = matrix.size(0)

    if dim > MAX_PRECOND_DIM:
        # Diagonal approximation: O(n) instead of O(n³)
        diag = matrix.diagonal().clamp(min=1e-4)
        scaled = diag.pow(power).clamp(max=MAX_PRECOND_SCALE)
        return scaled.diag().to(device)

    # Full eigh for small matrices (≤ 512×512)
    matrix = matrix.cpu().double()
    matrix = 0.5 * (matrix + matrix.t())
    trace_val = matrix.trace().item()
    reg = max(1e-4, 0.01 * abs(trace_val) / dim)
    matrix.diagonal().add_(reg)
    try:
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = eigvals.clamp(min=reg)
        result = eigvecs @ eigvals.pow(power).diag() @ eigvecs.t()
        return result.float().to(device)
    except torch._C._LinAlgError:
        return torch.eye(dim, device=device)

class Shampoo(Optimizer):
    """Shampoo — Preconditioned Stochastic Tensor Optimization."""
    def __init__(self, params, lr=1e-1, momentum=0.0, weight_decay=0.0,
                 epsilon=1e-4, update_freq=1, precond_decay=0.99):
        if lr <= 0.0: raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0: raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        epsilon=epsilon, update_freq=update_freq,
                        precond_decay=precond_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        for group in self.param_groups:
            decay     = group["precond_decay"]
            momentum  = group["momentum"]
            wd        = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                # clone: never mutate p.grad.data in-place
                grad = p.grad.data.clone()
                order         = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]

                # ── initialise state ──────────────────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    for dim_id, dim in enumerate(grad.size()):
                        state[f"precond_{dim_id}"]     = group["epsilon"] * torch.eye(
                            dim, dtype=grad.dtype, device=grad.device)
                        # initialise to identity so it's valid before first update
                        state[f"inv_precond_{dim_id}"] = torch.eye(
                            dim, dtype=grad.dtype, device=grad.device)

                # ── weight decay on raw gradient ──────────────────────
                if wd > 0:
                    grad.add_(p.data, alpha=wd)

                # ── momentum on RAW gradient (NOT preconditioned) ─────
                # Storing the preconditioned update in momentum_buffer
                # creates a feedback loop: scale × momentum > 1 → NaN.
                if momentum > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                    grad = buf.clone()   # copy so buf stays pure raw-grad EMA

                # ── preconditioner update + preconditioned step ───────
                update = grad   # will be transformed per dimension
                for dim_id, dim in enumerate(grad.size()):
                    precond     = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]
                    update = update.transpose(0, dim_id).contiguous()
                    transposed_size = update.size()
                    update = update.view(dim, -1)
                    update_t = update.t()
                    precond.mul_(decay).add_(update @ update_t, alpha=1.0 - decay)
                    if state["step"] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1.0 / order))
                    if dim_id == order - 1:
                        update = update_t @ inv_precond
                        update = update.view(original_size)
                    else:
                        update = inv_precond @ update
                        update = update.view(transposed_size)

                state["step"] += 1
                p.data.add_(update, alpha=-group["lr"])
        return loss

print("✓ Shampoo defined")

# %% [markdown]
# ### 4d. ASAM (Adaptive SAM)
# Source: https://github.com/davda54/sam

# %%
class ASAM(Optimizer):
    """Adaptive SAM — wraps a base optimizer with adaptive perturbation."""
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=True, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
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
        assert closure is not None, "ASAM requires closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

print("✓ ASAM defined")

# %% [markdown]
# ### 4e. SASSHA (reference)

# %%
class SASSHA(Optimizer):
    """SASSHA — SAM + Hutchinson Hessian trace."""
    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), weight_decay=0.0,
                 rho=0.0, lazy_hessian=10, n_samples=1, perturb_eps=1e-12,
                 eps=1e-4, adaptive=False, hessian_power=1.0, seed=0, **kwargs):
        if not 0.0 <= lr: raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta1: {betas[1]}")
        self.lazy_hessian = lazy_hessian
        self.n_samples = n_samples
        self.adaptive = adaptive
        self.seed = seed
        self.hessian_power_t = hessian_power
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        rho=rho, perturb_eps=perturb_eps, eps=eps)
        super().__init__(params, defaults)
        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0
        self.generator = torch.Generator().manual_seed(self.seed)

    def get_params(self):
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.lazy_hessian == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.lazy_hessian == 0:
                params.append(p)
            self.state[p]["hessian step"] += 1
        if len(params) == 0: return
        if self.generator.device != params[0].device:
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)
        grads = [p.grad for p in params]
        last_sample = self.n_samples - 1
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < last_sample)
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
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def _grad_norm(self, by=None):
        if not by:
            norm = torch.norm(torch.stack([
                ((torch.abs(p.data) if self.adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]), p=2)
        else:
            norm = torch.norm(torch.stack([
                ((torch.abs(p.data) if self.adaptive else 1.0) * self.state[p][by]).norm(p=2)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]), p=2)
        return norm

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
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag'] = torch.zeros_like(p.data)
                    state['bias_correction2'] = 0
                exp_avg = state['exp_avg']
                exp_hessian_diag = state['exp_hessian_diag']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                bias_correction1 = 1 - beta1 ** state['step']
                if (state['hessian step'] - 1) % self.lazy_hessian == 0:
                    exp_hessian_diag.mul_(beta2).add_(p.hess, alpha=1 - beta2)
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['bias_correction2'] = bias_correction2 ** k
                step_size = group['lr'] / bias_correction1
                denom = ((exp_hessian_diag ** k) / max(state['bias_correction2'], 1e-12)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

print("✓ SASSHA defined")

# %% [markdown]
# ## 5. EMA Wrapper

# %%
class EMAWrapper:
    def __init__(self, model, decay=0.999):
        self.decay = decay
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
# ## 6. Configs

# %%
SEED            = 42
SAVE_EVERY_N_TASKS = 80
TIME_LIMIT_SECONDS = 11.5 * 3600

NUM_TASKS       = 800
CHANGE_AFTER    = 60_000
MINI_BATCH_SIZE = 32
STEPS_PER_TASK  = CHANGE_AFTER // MINI_BATCH_SIZE   # 1875
PROBE_SIZE      = 2000

# ── Common SDP gamma across all methods ──
SDP_GAMMA = 0.3

# ── Per-optimizer configs (with and without SDP) ──
CONFIGS = {
    # 1. AdaHessian + SDP
    'adahessian_sdp': dict(
        optimizer='adahessian',
        lr=0.003, betas=(0.9, 0.999), weight_decay=5e-4, eps=1e-4,
        hessian_power=1.0, lazy_hessian=10, n_samples=1,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 2. AdaHessian (no SDP)
    'adahessian_nosdp': dict(
        optimizer='adahessian',
        lr=0.003, betas=(0.9, 0.999), weight_decay=5e-4, eps=1e-4,
        hessian_power=1.0, lazy_hessian=10, n_samples=1,
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
    # 7. ASAM (Adaptive SAM + SGD) + SDP
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
        lr=0.003, betas=(0.9, 0.999), weight_decay=5e-4, rho=0.05,
        lazy_hessian=10, n_samples=1, eps=1e-4, hessian_power=1.0,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 10. SASSHA (no SDP)
    'sassha_nosdp': dict(
        optimizer='sassha',
        lr=0.003, betas=(0.9, 0.999), weight_decay=5e-4, rho=0.05,
        lazy_hessian=10, n_samples=1, eps=1e-4, hessian_power=1.0,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=0.0,
    ),
}

# ── Select which methods to run ──
METHODS_TO_RUN = [
    'adahessian_sdp', 'adahessian_nosdp',
    'sophia_sdp', 'sophia_nosdp',
    'shampoo_sdp', 'shampoo_nosdp',
    'asam_sdp', 'asam_nosdp',
    'sassha_sdp', 'sassha_nosdp',
]

RESULTS_DIR = os.path.join('permuted_mnist_results', 'secondorder_sdp')
CKPT_DIR    = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"✓ Config: {NUM_TASKS} tasks × {STEPS_PER_TASK} steps/task, "
      f"batch={MINI_BATCH_SIZE}, network: {INPUT_SIZE}→{NUM_FEATURES}×{NUM_HIDDEN_LAYERS}→{CLASSES_PER_TASK}")
print(f"  Methods: {METHODS_TO_RUN}")

# %% [markdown]
# ## 7. Build Optimizer

# %%
def build_optimizer_for_method(config, model):
    """Build the appropriate optimizer from config dict."""
    opt_type = config['optimizer']

    if opt_type == 'adahessian':
        return Adahessian(
            model.parameters(), lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.0),
            eps=config.get('eps', 1e-4),
            hessian_power=config.get('hessian_power', 1.0),
            lazy_hessian=config.get('lazy_hessian', 1),
            n_samples=config.get('n_samples', 1), seed=SEED)

    elif opt_type == 'sophia':
        return SophiaH(
            model.parameters(), lr=config['lr'],
            betas=config.get('betas', (0.965, 0.99)),
            weight_decay=config.get('weight_decay', 1e-1),
            eps=config.get('eps', 1e-15),
            clip_threshold=config.get('clip_threshold', 0.04),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1), seed=SEED)

    elif opt_type == 'shampoo':
        return Shampoo(
            model.parameters(), lr=config['lr'],
            momentum=config.get('momentum', 0.0),
            weight_decay=config.get('weight_decay', 0.0),
            epsilon=config.get('epsilon', 1e-4),
            update_freq=config.get('update_freq', 1))

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
            rho=config.get('rho', 0.05),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1),
            eps=config.get('eps', 1e-4),
            hessian_power=config.get('hessian_power', 1.0),
            seed=SEED)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

print("✓ build_optimizer_for_method defined")

# %% [markdown]
# ## 8. Unified Training Loop

# %%
def _ckpt_path(method_name: str) -> str:
    return os.path.join(CKPT_DIR, f"ckpt_{method_name}.pt")


def _needs_create_graph(config):
    """Does this optimizer need create_graph=True on backward for Hessian?"""
    return config['optimizer'] in ('adahessian', 'sophia', 'sassha')


def _is_two_pass(config):
    """Does this optimizer use SAM-style two-pass (perturb → second forward)?"""
    return config['optimizer'] in ('sassha', 'asam')


def run_method(method_name, config):
    """Unified training loop for all second-order optimizers ± SDP."""
    opt_type = config['optimizer']
    sdp_gamma = config.get('sdp_gamma', 0.0)
    print(f"\n{'='*70}")
    print(f"  {method_name} (optimizer={opt_type}) — Permuted MNIST ({NUM_TASKS} tasks)")
    if sdp_gamma > 0:
        print(f"  SDP enabled: γ={sdp_gamma}")
    else:
        print(f"  SDP disabled")
    print(f"{'='*70}")

    wall_clock_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    # ── Build network ──
    net = DeepFFNN(
        input_size=INPUT_SIZE, num_features=NUM_FEATURES,
        num_outputs=CLASSES_PER_TASK, num_hidden_layers=NUM_HIDDEN_LAYERS,
    ).to(device)

    optimizer = build_optimizer_for_method(config, net)
    ls = 0.1 if config.get('label_smoothing', False) else 0.0
    loss_fn = lambda logits, target: F.cross_entropy(logits, target, label_smoothing=ls)

    ema = EMAWrapper(net, config.get('ema_decay', 0.999)) if config.get('use_ema', False) else None
    create_graph = _needs_create_graph(config)
    two_pass = _is_two_pass(config)

    # ── Metric tensors ──
    total_iters       = NUM_TASKS * STEPS_PER_TASK
    accuracies        = torch.zeros(total_iters, dtype=torch.float)
    train_accuracies  = torch.zeros(NUM_TASKS, dtype=torch.float)
    test_accuracies   = torch.zeros(NUM_TASKS, dtype=torch.float)
    weight_mag_log    = torch.zeros((total_iters, NUM_HIDDEN_LAYERS + 1), dtype=torch.float)
    ranks             = torch.zeros((NUM_TASKS, NUM_HIDDEN_LAYERS), dtype=torch.float)
    effective_ranks   = torch.zeros((NUM_TASKS, NUM_HIDDEN_LAYERS), dtype=torch.float)
    approximate_ranks = torch.zeros((NUM_TASKS, NUM_HIDDEN_LAYERS), dtype=torch.float)
    dead_neurons      = torch.zeros((NUM_TASKS, NUM_HIDDEN_LAYERS), dtype=torch.float)
    all_stable_rank   = torch.zeros(NUM_TASKS, dtype=torch.float)
    all_dormant_frac  = torch.zeros(NUM_TASKS, dtype=torch.float)
    all_sdp_cond      = torch.zeros(NUM_TASKS, dtype=torch.float)
    task_metrics      = {'task_acc': [], 'task_train_acc': [], 'task_test_acc': [],
                         'task_id': [], 'avg_weight_mag': []}

    # ── Checkpoint resume ──
    ckpt_file  = _ckpt_path(method_name)
    start_task = 0
    global_iter = 0
    if os.path.isfile(ckpt_file):
        print(f"  Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        accuracies        = ckpt.get('accuracies',         accuracies)
        train_accuracies  = ckpt.get('train_accuracies',   train_accuracies)
        test_accuracies   = ckpt.get('test_accuracies',    test_accuracies)
        weight_mag_log    = ckpt.get('weight_mag_log',     weight_mag_log)
        ranks             = ckpt.get('ranks',              ranks)
        effective_ranks   = ckpt.get('effective_ranks',    effective_ranks)
        approximate_ranks = ckpt.get('approximate_ranks',  approximate_ranks)
        dead_neurons      = ckpt.get('dead_neurons',       dead_neurons)
        all_stable_rank   = ckpt.get('all_stable_rank',    all_stable_rank)
        all_dormant_frac  = ckpt.get('all_dormant_frac',   all_dormant_frac)
        all_sdp_cond      = ckpt.get('all_sdp_cond',       all_sdp_cond)
        task_metrics      = ckpt.get('task_metrics',       task_metrics)
        start_task        = ckpt['task'] + 1
        global_iter       = start_task * STEPS_PER_TASK
        if ema is not None and 'ema_shadow' in ckpt:
            ema._shadow = ckpt['ema_shadow']
        # Reset Hessian state for Hessian-based optimizers
        if hasattr(optimizer, 'get_params'):
            for p in optimizer.get_params():
                p.hess = 0.0
                optimizer.state[p]["hessian step"] = 0
        print(f"  ✓ Resumed from task {start_task}")
        del ckpt; torch.cuda.empty_cache()
    else:
        print("  (no checkpoint — training from scratch)")

    def save_checkpoint(task_idx, reason="periodic"):
        ckpt_data = {
            'task': task_idx, 'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'accuracies': accuracies, 'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'weight_mag_log': weight_mag_log,
            'ranks': ranks, 'effective_ranks': effective_ranks,
            'approximate_ranks': approximate_ranks,
            'dead_neurons': dead_neurons, 'all_stable_rank': all_stable_rank,
            'all_dormant_frac': all_dormant_frac, 'all_sdp_cond': all_sdp_cond,
            'task_metrics': task_metrics, 'params': config,
        }
        if ema is not None: ckpt_data['ema_shadow'] = ema._shadow
        torch.save(ckpt_data, ckpt_file)
        elapsed = time.time() - wall_clock_start
        print(f"  Checkpoint saved at task {task_idx} ({reason}) [{elapsed/3600:.1f}h elapsed]")

    # ── Working copies ──
    x = x_mnist.clone()
    y = y_mnist.clone()
    x_test = x_test_mnist.clone()
    y_test = y_test_mnist.clone()
    time_limit_hit = False

    # ════════════════════════════════════════════════════════════════
    #  Task loop
    # ════════════════════════════════════════════════════════════════
    for task_idx in range(start_task, NUM_TASKS):
        t0 = time.time()
        elapsed = time.time() - wall_clock_start
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"\n  Time limit ({elapsed/3600:.1f}h). Saving.")
            save_checkpoint(task_idx - 1, reason="time_limit")
            time_limit_hit = True; break

        iter_start = global_iter

        # ── 1. New pixel permutation + data shuffle ──
        rng        = np.random.RandomState(task_idx + SEED * 1000)
        pixel_perm = rng.permutation(INPUT_SIZE)
        data_perm  = rng.permutation(EXAMPLES_PER_TASK)
        x          = x[:, pixel_perm]
        x, y       = x[data_perm], y[data_perm]
        x_test     = x_test[:, pixel_perm]

        # ── 2. SDP at task boundary ──
        if sdp_gamma > 0 and task_idx > 0:
            cond_nums = apply_sdp(net, sdp_gamma)
            avg_cond = sum(cond_nums) / max(len(cond_nums), 1)
            all_sdp_cond[task_idx] = avg_cond
            if task_idx % 50 == 0:
                print(f"    SDP applied: avg condition number = {avg_cond:.1f}")

        # ── 3. Reset EMA at task boundary ──
        if ema is not None: ema.reset(net)

        # ── 3b. Reset Shampoo preconditioners at task boundary ──
        # Permuted MNIST changes input distribution completely each task →
        # stale preconditioner from prev task gives wrong curvature estimate.
        if opt_type == 'shampoo':
            eps = config.get('epsilon', 0.1)
            for state in optimizer.state.values():
                dim_id = 0
                while f"precond_{dim_id}" in state:
                    dim = state[f"precond_{dim_id}"].size(0)
                    state[f"precond_{dim_id}"].copy_(
                        eps * torch.eye(dim, device=state[f"precond_{dim_id}"].device,
                                        dtype=state[f"precond_{dim_id}"].dtype))
                    dim_id += 1

        # ── 4. Pre-task probe: rank + dead neurons ──
        net.eval()
        with torch.no_grad():
            probe_x = x[:PROBE_SIZE].to(device)
            _, hidden_acts = net.predict(probe_x)
            for li in range(NUM_HIDDEN_LAYERS):
                act = hidden_acts[li]
                if not torch.isfinite(act).all():
                    ranks[task_idx][li] = float('nan')
                    effective_ranks[task_idx][li] = float('nan')
                    approximate_ranks[task_idx][li] = float('nan')
                    dead_neurons[task_idx][li] = 0.0
                    continue
                r, er, apr, _ = compute_matrix_rank_summaries(act, use_scipy=True)
                ranks[task_idx][li]             = r.float()
                effective_ranks[task_idx][li]   = er.float()
                approximate_ranks[task_idx][li] = apr.float()
                dead_neurons[task_idx][li] = (act.abs().sum(dim=0) == 0).sum().item()

        if task_idx % 10 == 0:
            sdp_str = f"  CondN={all_sdp_cond[task_idx]:.1f}" if sdp_gamma > 0 and task_idx > 0 else ""
            print(f"  Task {task_idx:4d}  approx_rank={approximate_ranks[task_idx].tolist()}"
                  f"  dead={dead_neurons[task_idx].tolist()}{sdp_str}")

        # ── 5. Train STEPS_PER_TASK mini-batch steps ──
        net.train()
        for step in range(STEPS_PER_TASK):
            start_idx = (step * MINI_BATCH_SIZE) % EXAMPLES_PER_TASK
            batch_x   = x[start_idx: start_idx + MINI_BATCH_SIZE].to(device)
            batch_y   = y[start_idx: start_idx + MINI_BATCH_SIZE].to(device)

            # ── Optimizer-specific training protocol ──
            if opt_type == 'sassha':
                # SASSHA: two-pass SAM + Hessian
                optimizer.zero_grad()
                logits = net.predict(batch_x)[0]
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.perturb_weights(zero_grad=True)
                logits_pert = net.predict(batch_x)[0]
                loss_pert = loss_fn(logits_pert, batch_y)
                loss_pert.backward(create_graph=True)
                optimizer.unperturb()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            elif opt_type == 'asam':
                # ASAM: two-pass SAM with closure
                optimizer.zero_grad()
                logits = net.predict(batch_x)[0]
                loss = loss_fn(logits, batch_y)
                loss.backward()
                def closure():
                    logits2 = net.predict(batch_x)[0]
                    loss2 = loss_fn(logits2, batch_y)
                    loss2.backward()
                    return loss2
                optimizer.step(closure=closure)

            elif opt_type in ('adahessian', 'sophia'):
                # Hessian-based: single forward, backward with create_graph
                optimizer.zero_grad()
                logits = net.predict(batch_x)[0]
                loss = loss_fn(logits, batch_y)
                loss.backward(create_graph=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            elif opt_type == 'shampoo':
                # Shampoo: standard single-pass
                optimizer.zero_grad()
                logits = net.predict(batch_x)[0]
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

            if ema is not None: ema.update(net)

            # ── Per-step metrics ──
            with torch.no_grad():
                out_sm = F.softmax(net.predict(batch_x)[0], dim=1)
                accuracies[global_iter] = accuracy(out_sm, batch_y).cpu()
                for l_idx, layer_idx in enumerate(net.layers_to_log):
                    if l_idx < weight_mag_log.shape[1]:
                        weight_mag_log[global_iter][l_idx] = (
                            net.layers[layer_idx].weight.data.abs().mean())
            global_iter += 1

        # ── 6. Train & Test evaluation ──
        net.eval()
        with torch.no_grad():
            if ema is not None: ema.apply(net)

            # Train accuracy
            train_correct, train_total = 0, 0
            for si in range(0, x.shape[0], MINI_BATCH_SIZE):
                tb_x = x[si:si + MINI_BATCH_SIZE].to(device)
                tb_y = y[si:si + MINI_BATCH_SIZE].to(device)
                to = net.predict(tb_x)[0]
                train_correct += (to.argmax(dim=1) == tb_y).sum().item()
                train_total += tb_y.shape[0]
            train_acc = train_correct / max(train_total, 1)
            train_accuracies[task_idx] = train_acc

            # Test accuracy
            test_correct, test_total = 0, 0
            for si in range(0, x_test.shape[0], MINI_BATCH_SIZE):
                tb_x = x_test[si:si + MINI_BATCH_SIZE].to(device)
                tb_y = y_test[si:si + MINI_BATCH_SIZE].to(device)
                to = net.predict(tb_x)[0]
                test_correct += (to.argmax(dim=1) == tb_y).sum().item()
                test_total += tb_y.shape[0]
            test_acc = test_correct / max(test_total, 1)
            test_accuracies[task_idx] = test_acc

            if ema is not None: ema.restore(net)

            # Dormant + stable rank
            probe_x = x[:PROBE_SIZE].to(device)
            agg_dormant, _, last_act = compute_dormant_neurons_ffnn(net, probe_x)
            all_dormant_frac[task_idx] = agg_dormant
            sr = compute_stable_rank_from_activations(last_act)
            all_stable_rank[task_idx] = sr

        # ── 7. Per-task summary ──
        task_acc = accuracies[iter_start:global_iter].mean().item()
        task_metrics['task_acc'].append(task_acc)
        task_metrics['task_train_acc'].append(train_acc)
        task_metrics['task_test_acc'].append(test_acc)
        task_metrics['task_id'].append(task_idx)
        task_metrics['avg_weight_mag'].append(
            weight_mag_log[iter_start:global_iter].mean(dim=0).tolist())

        et = time.time() - t0
        print(f"  [{method_name}] Task {task_idx:4d}/{NUM_TASKS}  "
              f"TrainAcc={train_acc:.4f}  TestAcc={test_acc:.4f}  "
              f"Dormant={agg_dormant:.3f}  SR={sr:.0f}  "
              f"AvgW={compute_avg_weight_magnitude(net):.4f}  "
              f"{et:.1f}s")

        if (task_idx + 1) % SAVE_EVERY_N_TASKS == 0 or task_idx == NUM_TASKS - 1:
            save_checkpoint(task_idx,
                            reason="periodic" if task_idx < NUM_TASKS - 1 else "completed")

    # ── Save final results ──
    result_file = os.path.join(RESULTS_DIR, f'{method_name}_results.pt')
    torch.save({
        'accuracies':        accuracies.cpu(),
        'train_accuracies':  train_accuracies.cpu(),
        'test_accuracies':   test_accuracies.cpu(),
        'weight_mag_log':    weight_mag_log.cpu(),
        'ranks':             ranks.cpu(),
        'effective_ranks':   effective_ranks.cpu(),
        'approximate_ranks': approximate_ranks.cpu(),
        'dead_neurons':      dead_neurons.cpu(),
        'all_stable_rank':   all_stable_rank.cpu(),
        'all_dormant_frac':  all_dormant_frac.cpu(),
        'all_sdp_cond':      all_sdp_cond.cpu(),
        'task_metrics':      task_metrics,
    }, result_file)
    print(f"  ✓ Results saved to {result_file}")

    return (accuracies, train_accuracies, test_accuracies, weight_mag_log, ranks,
            effective_ranks, approximate_ranks, dead_neurons, all_stable_rank,
            all_dormant_frac, all_sdp_cond, task_metrics)

print("✓ Unified training loop defined")

# %% [markdown]
# ## 9. Run All Experiments

# %%
all_results = {}
for method in METHODS_TO_RUN:
    cfg = CONFIGS[method]
    result = run_method(method, cfg)
    all_results[method] = {
        'acc': result[0], 'train_acc': result[1], 'test_acc': result[2],
        'wmag': result[3], 'ranks': result[4], 'er': result[5],
        'apr': result[6], 'dead': result[7], 'sr': result[8],
        'dormant': result[9], 'cond': result[10], 'task': result[11],
    }

# %% [markdown]
# ## 10. Results Plots

# %%
TASK_WINDOW = 50
LAYER_NAMES = [f'Hidden {i+1} ({NUM_FEATURES})' for i in range(NUM_HIDDEN_LAYERS)]

# Color palette: SDP variants = solid, no-SDP = dashed
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

def smooth(arr, w=TASK_WINDOW):
    n = len(arr) // w
    return np.array([arr[i*w:(i+1)*w].mean() for i in range(n)])

def _style(method):
    return METHOD_STYLES.get(method, {'color': 'gray', 'ls': '-', 'label': method})

def _clean(ax):
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Main comparison: 3×2 grid ──
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('Second-Order Optimizers ± SDP — Continual Permuted MNIST\n'
             f'Network: {INPUT_SIZE}→{NUM_FEATURES}×{NUM_HIDDEN_LAYERS}→{CLASSES_PER_TASK}, '
             f'{NUM_TASKS} tasks, batch={MINI_BATCH_SIZE}',
             fontsize=14, fontweight='bold')

# ── (0,0) Test Accuracy ──
ax = axes[0, 0]
for m, d in all_results.items():
    s = _style(m)
    test_arr = np.array(d['task']['task_test_acc'])
    sm = smooth(test_arr) * 100
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'],
            ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Test Accuracy (smoothed)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (0,1) Train Accuracy ──
ax = axes[0, 1]
for m, d in all_results.items():
    s = _style(m)
    train_arr = np.array(d['task']['task_train_acc'])
    sm = smooth(train_arr) * 100
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'],
            ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Train Accuracy (smoothed)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (1,0) Stable Rank ──
ax = axes[1, 0]
for m, d in all_results.items():
    s = _style(m)
    sr = d['sr'].cpu().numpy()
    sm = smooth(sr)
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'],
            ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Stable Rank')
ax.set_title('Stable Rank (last hidden)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (1,1) Dormant Neuron Fraction ──
ax = axes[1, 1]
for m, d in all_results.items():
    s = _style(m)
    dorm = d['dormant'].cpu().numpy() * 100
    sm = smooth(dorm)
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'],
            ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Dormant (%)')
ax.set_title('Dormant Neuron Fraction')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (2,0) Avg Approximate Rank ──
ax = axes[2, 0]
for m, d in all_results.items():
    s = _style(m)
    apr_avg = d['apr'].mean(dim=1).numpy()
    sm = smooth(apr_avg)
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'],
            ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Approx Rank')
ax.set_title('Avg Approximate Rank (all layers)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (2,1) Total Dead Neurons ──
ax = axes[2, 1]
for m, d in all_results.items():
    s = _style(m)
    dead_total = d['dead'].sum(dim=1).numpy()
    sm = smooth(dead_total)
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'],
            ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Dead Neurons')
ax.set_title('Total Dead Neurons')
ax.legend(fontsize=7, ncol=2); _clean(ax)

plt.tight_layout()
plot_file = os.path.join(RESULTS_DIR, 'secondorder_sdp_comparison.png')
plt.savefig(plot_file, dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Main comparison plot saved to {plot_file}")

# %% [markdown]
# ## 11. SDP Ablation: Δ(metric) = SDP − noSDP

# %%
OPT_PAIRS = [
    ('adahessian_sdp', 'adahessian_nosdp', 'AdaHessian', '#E91E63'),
    ('sophia_sdp',     'sophia_nosdp',     'SophiaH',    '#9C27B0'),
    ('shampoo_sdp',    'shampoo_nosdp',    'Shampoo',    '#FF9800'),
    ('asam_sdp',       'asam_nosdp',       'ASAM',       '#4CAF50'),
    ('sassha_sdp',     'sassha_nosdp',     'SASSHA',     '#2196F3'),
]

fig_ab, axes_ab = plt.subplots(1, 3, figsize=(21, 5))
fig_ab.suptitle('SDP Ablation: Δ = (with SDP) − (without SDP)',
                fontsize=13, fontweight='bold')

for sdp_key, nosdp_key, label, color in OPT_PAIRS:
    if sdp_key not in all_results or nosdp_key not in all_results:
        continue
    d_sdp = all_results[sdp_key]
    d_no  = all_results[nosdp_key]

    # Δ Test Accuracy
    delta_test = np.array(d_sdp['task']['task_test_acc']) - np.array(d_no['task']['task_test_acc'])
    sm = smooth(delta_test) * 100
    axes_ab[0].plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=color, lw=2.5, label=label)

    # Δ Stable Rank
    delta_sr = d_sdp['sr'].cpu().numpy() - d_no['sr'].cpu().numpy()
    sm_sr = smooth(delta_sr)
    axes_ab[1].plot(np.arange(len(sm_sr)) * TASK_WINDOW, sm_sr, color=color, lw=2.5, label=label)

    # Δ Dormant Fraction
    delta_dorm = (d_sdp['dormant'].cpu().numpy() - d_no['dormant'].cpu().numpy()) * 100
    sm_d = smooth(delta_dorm)
    axes_ab[2].plot(np.arange(len(sm_d)) * TASK_WINDOW, sm_d, color=color, lw=2.5, label=label)

for i, (title, ylabel) in enumerate([
    ('Δ Test Accuracy', 'Δ Acc (%)'),
    ('Δ Stable Rank', 'Δ SR'),
    ('Δ Dormant Fraction', 'Δ Dormant (%)'),
]):
    axes_ab[i].set_title(title); axes_ab[i].set_xlabel('Task')
    axes_ab[i].set_ylabel(ylabel)
    axes_ab[i].axhline(0, color='black', ls=':', lw=0.8, alpha=0.5)
    axes_ab[i].legend(fontsize=9); _clean(axes_ab[i])

plt.tight_layout()
plot_abl = os.path.join(RESULTS_DIR, 'sdp_ablation.png')
plt.savefig(plot_abl, dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ SDP ablation plot saved to {plot_abl}")

# %% [markdown]
# ## 12. Summary Table

# %%
n_final = 100
print(f"\n{'='*90}")
print(f"  Second-Order Optimizers ± SDP — Final {n_final}-task average")
print(f"{'='*90}")
header = f"{'Method':<22} {'TestAcc%':>9} {'TrainAcc%':>10} {'StableRank':>11} {'Dormant%':>9} {'DeadNeur':>9} {'ApproxRank':>11}"
print(header)
print(f"{'─'*90}")

for method in METHODS_TO_RUN:
    if method not in all_results: continue
    d = all_results[method]
    s = _style(method)
    test_acc  = np.array(d['task']['task_test_acc'][-n_final:]).mean() * 100
    train_acc = np.array(d['task']['task_train_acc'][-n_final:]).mean() * 100
    sr        = d['sr'][-n_final:].mean().item()
    dormant   = d['dormant'][-n_final:].mean().item() * 100
    dead      = d['dead'][-n_final:].sum(dim=1).mean().item()
    apr       = d['apr'][-n_final:].mean().item()
    print(f"  {s['label']:<20} {test_acc:>9.2f} {train_acc:>10.2f} {sr:>11.1f} {dormant:>9.2f} {dead:>9.1f} {apr:>11.1f}")

print(f"{'='*90}")
