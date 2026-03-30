# %% [markdown]
# # Second-Order Optimizers + SDP on Continual RL (PPO — Ant-v4)
# #
# **Hypothesis**: SDP + second-order optimizers mitigate Loss of Plasticity
# in RL, just as they do in supervised continual learning (Permuted MNIST).
# #
# **Optimizers tested**:
# 1. **AdaHessian** (Yao et al., 2021) — Hutchinson diagonal Hessian
# 2. **SophiaH** (Liu et al., 2023) — Hutchinson Hessian + element-wise clipping
# 3. **Shampoo** (Gupta et al., 2018) — Full-matrix preconditioning
# 4. **ASAM** (Kwon et al., 2021) — Adaptive SAM + SGD base
# 5. **SASSHA** (baseline) — SAM + Hutchinson Hessian
# #
# Each optimizer runs **with and without SDP** for ablation.
# #
# **Benchmark**: PPO on Ant-v4 (MuJoCo), 10M steps.
# Network: MLP 27→256×2→8 (policy) + 27→256×2→1 (value), ReLU.
# Based on CBP paper RL setup (lop/rl/) and rl-continual.ipynb.

# %% [markdown]
# ## 1. Imports & Setup

# %%
import os, sys, time, pickle, math, copy
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

import gymnasium as gym

_LOP_ROOT = "/kaggle/input/lop-src"
if os.path.isdir(_LOP_ROOT) and _LOP_ROOT not in sys.path:
    sys.path.insert(0, _LOP_ROOT)

from lop.nets.policies import MLPPolicy
from lop.nets.valuefs import MLPVF
from lop.algos.rl.buffer import Buffer
from lop.utils.miscellaneous import compute_matrix_rank_summaries

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Metrics & SDP for RL

# %%
@torch.no_grad()
def compute_dormant_units_rl(pol, vf, threshold=0.01):
    """Compute fraction of dormant neurons from stored activations."""
    total_units, dormant_units = 0, 0
    for net in [pol, vf]:
        if hasattr(net, 'activations') and net.activations:
            for key, feat in net.activations.items():
                if feat is not None and feat.dim() >= 2:
                    activity = (feat != 0).float().mean(dim=0)
                    dormant = (activity < threshold).sum().item()
                    dormant_units += dormant
                    total_units += activity.numel()
    return dormant_units / total_units if total_units > 0 else 0.0


@torch.no_grad()
def compute_stable_rank_from_features(feature_activity):
    """Compute stable rank from feature activations (99% variance)."""
    if feature_activity is None or feature_activity.numel() == 0:
        return 1.0
    _, _, _, stable_rank = compute_matrix_rank_summaries(
        m=feature_activity, prop=0.99, use_scipy=True
    )
    return stable_rank.item() if torch.is_tensor(stable_rank) else float(stable_rank)


@torch.no_grad()
def compute_weight_magnitude(pol, vf):
    """Compute average weight magnitude across both networks."""
    total, n = 0.0, 0
    for net in [pol, vf]:
        for name, p in net.named_parameters():
            if 'weight' in name:
                total += p.abs().mean().item()
                n += 1
    return total / n if n else 0.0


def apply_sdp_rl(pol, vf, gamma):
    """
    Spectral Diversity Preservation (SDP) for RL actor-critic.
    σ'_i = σ̄^γ · σ_i^(1-γ)

    Applied to Linear layers of both policy and value networks.
    Skips output layers (structural rank bottlenecks).
    """
    cond_numbers = []
    for net_name, net in [('pol', pol), ('vf', vf)]:
        # Determine which Sequential to process
        if hasattr(net, 'mean_net'):
            seq = net.mean_net
        elif hasattr(net, 'v_net'):
            seq = net.v_net
        else:
            continue
        modules = [m for m in seq.modules() if isinstance(m, nn.Linear)]
        with torch.no_grad():
            for i, module in enumerate(modules):
                is_output = (i == len(modules) - 1)
                if is_output:
                    continue  # skip output layer
                W = module.weight.data
                try:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                except Exception:
                    continue
                if S.numel() == 0 or S[0] < 1e-12:
                    continue
                cond_numbers.append((S[0] / S[-1].clamp(min=1e-12)).item())
                s_mean = S.mean().clamp(min=1e-12)
                S_new = (s_mean ** gamma) * (S ** (1.0 - gamma))
                W_new = U @ torch.diag(S_new) @ Vh
                module.weight.data.copy_(W_new)
    return cond_numbers


def apply_gosc_rl(pol, vf, obs_buf, acts_buf, logp_old_buf, advs_buf, v_rets_buf,
                   gamma_min=0.05, gamma_max=0.5, grad_rank=5,
                   device='cpu', clip_eps=0.2, n_samples=512):
    """
    Gradient-Orthogonal Spectral Compression (GOSC) for RL actor-critic.

    Uses **PPO surrogate gradient** computed from buffer data (not env rollouts).

    Algorithm:
      1. Sample a mini-batch from the PPO buffer data.
      2. Compute PPO surrogate loss gradient on pol, MSE value gradient on vf.
      3. For each Linear layer (skip output):
         a. SVD(W) = U S Vh
         b. Low-rank SVD(G) = U_g S_g V_g  (rank = grad_rank)
         c. Alignment: a_i = ||U_g^T u_i||^2 + ||V_g^T v_i||^2
         d. Per-SV gamma: γ_i = γ_max * (1 - a_i/2) + γ_min * (a_i/2)
         e. σ'_i = σ̄^{γ_i} · σ_i^{(1-γ_i)}
         f. W' = U @ diag(σ') @ Vh

    Args:
        pol, vf: policy and value networks
        obs_buf: observations from last PPO update (N, obs_dim)
        acts_buf: actions from last PPO update (N, act_dim)
        logp_old_buf: old log-probs from last PPO update (N,) or (N,1)
        advs_buf: GAE advantages from last PPO update (N,) or (N,1)
        v_rets_buf: Value targets from last PPO update (N,) or (N,1)
        gamma_min, gamma_max: GOSC compression range
        grad_rank: rank for low-rank gradient SVD
        device: torch device
        clip_eps: PPO clipping epsilon
        n_samples: number of samples to use from buffer (subsample if needed)

    Returns:
        cond_numbers: list of condition numbers (before compression)
        avg_alignment: average alignment score across all layers
    """
    # ── Step 1: Subsample from buffer if needed ──
    n_total = obs_buf.shape[0]
    if n_total > n_samples:
        idx = np.random.choice(n_total, n_samples, replace=False)
        obs_t = obs_buf[idx].to(device)
        acts_t = acts_buf[idx].to(device)
        logp_old_t = logp_old_buf[idx].to(device).view(-1)
        advs_t = advs_buf[idx].to(device).view(-1)
        v_rets_t = v_rets_buf[idx].to(device).view(-1)
    else:
        obs_t = obs_buf.to(device)
        acts_t = acts_buf.to(device)
        logp_old_t = logp_old_buf.to(device).view(-1)
        advs_t = advs_buf.to(device).view(-1)
        v_rets_t = v_rets_buf.to(device).view(-1)

    # ── Step 2: Forward + backward with PPO surrogate gradient ──
    pol.train()
    vf.train()
    # Zero existing gradients
    for p in pol.parameters():
        p.grad = None
    for p in vf.parameters():
        p.grad = None

    # Policy gradient: PPO clipped surrogate objective
    with torch.enable_grad():
        logp_new, _ = pol.logp_dist(obs_t, acts_t)
        logp_new = logp_new.view(-1)
        ratio = (logp_new - logp_old_t).exp()
        surr1 = -(ratio * advs_t)
        surr2 = -(torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs_t)
        pol_loss = torch.max(surr1, surr2).mean()
        pol_loss.backward()

    # Value gradient: MSE against value targets
    for p in vf.parameters():
        p.grad = None
    with torch.enable_grad():
        v_out = vf.value(obs_t).view(-1)
        vf_loss = (v_out - v_rets_t).pow(2).mean()
        vf_loss.backward()

    # ── Step 3: Apply GOSC per-layer ──
    cond_numbers = []
    alignment_scores = []

    for net_name, net in [('pol', pol), ('vf', vf)]:
        if hasattr(net, 'mean_net'):
            seq = net.mean_net
        elif hasattr(net, 'v_net'):
            seq = net.v_net
        else:
            continue

        modules = [m for m in seq.modules() if isinstance(m, nn.Linear)]

        with torch.no_grad():
            for i, module in enumerate(modules):
                is_output = (i == len(modules) - 1)
                if is_output:
                    continue  # skip output layer

                W = module.weight.data
                G = module.weight.grad

                if G is None:
                    continue

                # SVD of weight matrix
                try:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                except Exception:
                    continue

                if S.numel() == 0 or S[0] < 1e-12:
                    continue

                cond_numbers.append((S[0] / S[-1].clamp(min=1e-12)).item())

                # Low-rank SVD of gradient
                r = min(grad_rank, min(G.shape[0], G.shape[1]))
                try:
                    U_g, S_g, Vh_g = torch.linalg.svd(G, full_matrices=False)
                    # Keep only top-r components
                    U_g = U_g[:, :r]    # (out_dim, r)
                    Vh_g = Vh_g[:r, :]  # (r, in_dim)
                except Exception:
                    # Fallback: static SDP with gamma = (gamma_min + gamma_max) / 2
                    gamma_fallback = (gamma_min + gamma_max) / 2.0
                    s_mean = S.mean().clamp(min=1e-12)
                    S_new = (s_mean ** gamma_fallback) * (S ** (1.0 - gamma_fallback))
                    W_new = U @ torch.diag(S_new) @ Vh
                    module.weight.data.copy_(W_new)
                    continue

                # Alignment: a_i = ||U_g^T u_i||^2 + ||V_g^T v_i||^2
                left_proj = U_g.T @ U           # (r, k)
                a_left = (left_proj ** 2).sum(dim=0)  # (k,), each ∈ [0, 1]

                right_proj = Vh_g @ Vh.T        # (r, k)
                a_right = (right_proj ** 2).sum(dim=0)  # (k,), each ∈ [0, 1]

                a = a_left + a_right  # (k,), each ∈ [0, 2]

                # Normalize to [0, 1]
                a_norm = (a / 2.0).clamp(0.0, 1.0)

                alignment_scores.append(a_norm.mean().item())

                # Per-SV gamma: high alignment → low gamma (preserve)
                #                low alignment → high gamma (compress)
                gamma_per_sv = gamma_max * (1.0 - a_norm) + gamma_min * a_norm

                # Geometric compression: σ'_i = σ̄^{γ_i} · σ_i^{(1-γ_i)}
                s_mean = S.mean().clamp(min=1e-12)
                log_s_mean = torch.log(s_mean)
                log_S = torch.log(S.clamp(min=1e-12))
                log_S_new = gamma_per_sv * log_s_mean + (1.0 - gamma_per_sv) * log_S
                S_new = torch.exp(log_S_new)

                W_new = U @ torch.diag(S_new) @ Vh
                module.weight.data.copy_(W_new)

    # ── Cleanup: zero gradients ──
    for p in pol.parameters():
        p.grad = None
    for p in vf.parameters():
        p.grad = None

    avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
    return cond_numbers, avg_alignment


print("✓ Metrics, SDP & GOSC defined")

# %% [markdown]
# ## 3. Optimizer Definitions

# %% [markdown]
# ### 3a. AdaHessian

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
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True,
                                       retain_graph=i < self.n_samples - 1)
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
# ### 3b. SophiaH

# %%
class SophiaH(Optimizer):
    """SophiaH — Hutchinson Hessian + element-wise clipping."""
    def __init__(self, params, lr=0.15, betas=(0.965, 0.99), eps=1e-15,
                 weight_decay=1e-1, lazy_hessian=10, n_samples=1,
                 clip_threshold=0.04, seed=0):
        if not 0.0 <= lr: raise ValueError(f"Invalid lr: {lr}")
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
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True,
                                       retain_graph=i < self.n_samples - 1)
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
# ### 3c. Shampoo

# %%
MAX_PRECOND_DIM = 512
MAX_PRECOND_SCALE = 50

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    device = matrix.device
    dim = matrix.size(0)
    if dim > MAX_PRECOND_DIM:
        diag = matrix.diagonal().clamp(min=1e-4)
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
        result = eigvecs @ eigvals.pow(power).diag() @ eigvecs.t()
        return result.float().to(device)
    except torch._C._LinAlgError:
        return torch.eye(dim, device=device)

class Shampoo(Optimizer):
    """Shampoo — Preconditioned Stochastic Tensor Optimization."""
    def __init__(self, params, lr=1e-1, momentum=0.0, weight_decay=0.0,
                 epsilon=1e-4, update_freq=1, precond_decay=0.99):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        epsilon=epsilon, update_freq=update_freq,
                        precond_decay=precond_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        for group in self.param_groups:
            decay = group["precond_decay"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.data.clone()
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
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
                    precond = state[f"precond_{dim_id}"]
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
# ### 3d. ASAM (Adaptive SAM)

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
# ### 3e. Hessian Power Schedulers (from official SASSHA source)

# %%
# ═══════════════════════════════════════════════════════════════════════
# Hessian Power Schedulers — exact copy from official LOG-postech source
# Source: Sassha/optimizers/hessian_scheduler.py
# ═══════════════════════════════════════════════════════════════════════

class ProportionScheduler:
    def __init__(self, pytorch_lr_scheduler, max_lr, min_lr, max_value, min_value):
        """
        This scheduler outputs a value that evolves proportional to pytorch_lr_scheduler, e.g.
        (value - min_value) / (max_value - min_value) = (lr - min_lr) / (max_lr - min_lr)
        """
        self.t = 0
        self.pytorch_lr_scheduler = pytorch_lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value

        assert (max_lr > min_lr) or ((max_lr==min_lr) and (max_value==min_value)), \
        "Current scheduler for `value` is scheduled to evolve proportionally to `lr`," \
        "e.g. `(lr - min_lr) / (max_lr - min_lr) = (value - min_value) / (max_value - min_value)`. " \
        "Please check `max_lr >= min_lr` and `max_value >= min_value`;" \
        "if `max_lr==min_lr` hence `lr` is constant with step, please set 'max_value == min_value' so 'value' is constant with step."

        assert max_value >= min_value

        self.step() # take 1 step during initialization to get self._last_lr

    def lr(self):
        return self._last_lr[0]

    def step(self):
        self.t += 1
        if hasattr(self.pytorch_lr_scheduler, "_last_lr"):
            lr = self.pytorch_lr_scheduler._last_lr[0]
        else:
            lr = self.pytorch_lr_scheduler.optimizer.param_groups[0]['lr']

        if self.max_lr > self.min_lr:
            value = self.max_value - (self.max_value - self.min_value) * (lr - self.min_lr) / (self.max_lr - self.min_lr)
        else:
            value = self.max_value

        self._last_lr = [value]
        return value

class SchedulerBase:
    def __init__(self, T_max, max_value, min_value=0.0, init_value=0.0, warmup_steps=0, optimizer=None):
        super(SchedulerBase, self).__init__()
        self.t = 0
        self.min_value = min_value
        self.max_value = max_value
        self.init_value = init_value
        self.warmup_steps = warmup_steps
        self.total_steps = T_max

        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self._last_lr = [init_value]

        # If optimizer is not None, will set learning rate to all trainable parameters in optimizer.
        # If optimizer is None, only output the value of lr.
        self.optimizer = optimizer

    def step(self):
        if self.t < self.warmup_steps:
            value = self.init_value + (self.max_value - self.init_value) * self.t / self.warmup_steps
        elif self.t == self.warmup_steps:
            value = self.min_value
        else:
            value = self.step_func()
        self.t += 1

        # apply the lr to optimizer if it's provided
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value

        self._last_lr = [value]
        return value

    def step_func(self):
        pass

    def lr(self):
        return self._last_lr[0]

class LinearScheduler(SchedulerBase):
    def step_func(self):
        value = self.min_value + (self.max_value - self.min_value) * (self.t - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps)
        return value

class CosineScheduler(SchedulerBase):
    def step_func(self):
        phase = (self.t-self.warmup_steps) / (self.total_steps-self.warmup_steps) * math.pi
        value = self.max_value - (self.max_value-self.min_value) * (np.cos(phase) + 1.) / 2.0
        return value

class ConstantScheduler(SchedulerBase):
    def step_func(self):
        value = self.min_value
        return value


def build_hessian_power_scheduler(schedule_type, total_steps,
                                  max_hessian_power=1.0,
                                  min_hessian_power=0.5):
    """Factory function to create a hessian power scheduler."""
    if schedule_type == 'constant':
        return ConstantScheduler(
            T_max=total_steps, max_value=max_hessian_power,
            min_value=min_hessian_power, init_value=min_hessian_power)
    elif schedule_type == 'linear':
        return LinearScheduler(
            T_max=total_steps, max_value=max_hessian_power,
            min_value=min_hessian_power, init_value=min_hessian_power)
    elif schedule_type == 'cosine':
        return CosineScheduler(
            T_max=total_steps, max_value=max_hessian_power,
            min_value=min_hessian_power, init_value=min_hessian_power)
    else:
        raise ValueError(f"Unknown hessian power schedule: {schedule_type}. "
                         f"Choose from: constant, linear, cosine")


print("✓ Hessian power schedulers defined (official)")

# %% [markdown]
# ### 3f. SASSHA

# %%
class SASSHA(Optimizer):
    """SASSHA — SAM + Hutchinson Hessian trace.
    Aligned with official LOG-postech SASSHA (ICML 2025).
    """
    def __init__(self, params,
                hessian_power_scheduler=None,
                lr=0.15,
                betas=(0.9, 0.999),
                weight_decay=0.0,
                rho=0.0,
                lazy_hessian=10,
                n_samples=1,
                perturb_eps=1e-12,
                eps=1e-4,
                adaptive=False,
                hessian_floor=1e-4,
                hessian_clip=1e3,
                seed=0,
                **kwargs):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        self.hessian_power_scheduler = hessian_power_scheduler
        self.lazy_hessian = lazy_hessian
        self.n_samples = n_samples
        self.adaptive = adaptive
        self.hessian_floor = hessian_floor
        self.hessian_clip = hessian_clip
        self.seed = seed

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        rho=rho, perturb_eps=perturb_eps, eps=eps)
        super(SASSHA, self).__init__(params, defaults)
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
    def update_hessian_power(self):
        """Update the Hessian power at every training step."""
        if self.hessian_power_scheduler is not None:
            self.hessian_power_t = self.hessian_power_scheduler.step()
        else:
            self.hessian_power_t = None
        return self.hessian_power_t

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
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True,
                                       retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples

    @torch.no_grad()
    def perturb_weights(self, zero_grad=True):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
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
    def _grad_norm(self, by=None, weight_adaptive=False):
        if not by:
            norm = torch.norm(torch.stack([
                ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]), p=2)
        else:
            norm = torch.norm(torch.stack([
                ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]), p=2)
        return norm

    @torch.no_grad()
    def step(self, closure=None, compute_hessian=True):
        self.update_hessian_power()

        loss = None
        if closure is not None: loss = closure()

        if compute_hessian:
            self.zero_hessian()
            self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None: continue

                # [LOCAL] handle p.hess reset to 0.0 after checkpoint resume
                if isinstance(p.hess, (int, float)):
                    p.hess = torch.zeros_like(p.data)
                else:
                    p.hess = p.hess.abs().clone()

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]
                # State initialization
                if len(state) == 2:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag'] = torch.zeros_like(p.data)
                    state['bias_correction2'] = 0

                exp_avg, exp_hessian_diag = state['exp_avg'], state['exp_hessian_diag']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                bias_correction1 = 1 - beta1 ** state['step']

                if (state['hessian step']-1) % self.lazy_hessian == 0:
                    exp_hessian_diag.mul_(beta2).add_(p.hess, alpha=1 - beta2)
                    # Hessian clamping: prevent near-zero (amplification) and extreme curvature
                    exp_hessian_diag.clamp_(self.hessian_floor, self.hessian_clip)
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['bias_correction2'] = bias_correction2 ** self.hessian_power_t

                step_size = group['lr'] / bias_correction1
                step_size_neg = -step_size

                denom = ((exp_hessian_diag**self.hessian_power_t) / max(state['bias_correction2'], 1e-12)).add_(group['eps'])

                # make update
                p.addcdiv_(exp_avg, denom, value=step_size_neg)

        return loss

print("✓ SASSHA defined (aligned with official CIFAR reference)")

# %% [markdown]
# ### 3f. K-FAC Natural Gradient Descent

# %%
class CombinedPolicyVF(nn.Module):
    """Wrapper that exposes both policy and value networks as a single module
    so KFACOptimizer can register hooks on all Linear layers."""
    def __init__(self, pol, vf):
        super().__init__()
        self.pol = pol
        self.vf = vf


class KFACOptimizer(Optimizer):
    """
    K-FAC Natural Gradient Descent.

    Approximates Fisher Information Matrix per layer as F ≈ A ⊗ G:
        A = EMA of E[a a^T]   (input/activation covariance)
        G = EMA of E[g g^T]   (output gradient covariance)
    Natural gradient:  δW = G_inv · ∇W · A_inv

    Spectral regularization is handled externally via SDP (apply_sdp_rl),
    applied periodically in the training loop — same as all other optimizers.
    """

    def __init__(self, model, lr=0.01, damping=1e-3, weight_decay=0,
                 T_inv=100, alpha=0.95, max_grad_norm=1.0,
                 adapt_damping=True,
                 damping_adaptation_interval=5,
                 damping_adaptation_decay=0.99,
                 min_damping=1e-4,
                 damping_decrease_rho=0.85,
                 damping_increase_rho=0.35):
        self.model        = model
        self.damping      = damping
        self._init_damp   = damping
        self.weight_decay = weight_decay
        self.T_inv        = T_inv
        self.alpha        = alpha
        self.max_grad_norm = max_grad_norm
        self.steps        = 0

        # Adaptive damping
        self.adapt_damping  = adapt_damping
        self._damp_interval = damping_adaptation_interval
        self._omega         = damping_adaptation_decay ** damping_adaptation_interval
        self._min_damping   = min_damping
        self._rho_dec       = damping_decrease_rho
        self._rho_inc       = damping_increase_rho
        self._prev_loss     = float('nan')
        self._qmodel_change = float('nan')
        self._rho           = float('nan')

        # Internal state
        self._modules = {}
        self._stats   = {}
        self._inv     = {}
        self._hooks   = []

        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)
        self._register_hooks()

        print(f"  K-FAC tracking {len(self._modules)} layers")

    def _register_hooks(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                self._modules[name] = mod
                self._stats[name]   = {'A': None, 'G': None}
                h1 = mod.register_forward_hook(self._fwd_hook(name, mod))
                h2 = mod.register_full_backward_hook(self._bwd_hook(name, mod))
                self._hooks += [h1, h2]

    def _fwd_hook(self, name, mod):
        def hook(m, inp, out):
            if not m.training:
                return
            with torch.no_grad():
                x = inp[0].detach()
                if isinstance(m, nn.Conv2d):
                    x = F.unfold(x, m.kernel_size, dilation=m.dilation,
                                 padding=m.padding, stride=m.stride)
                    x = x.permute(0, 2, 1).reshape(-1, x.size(1))
                elif x.dim() > 2:
                    x = x.reshape(-1, x.size(-1))
                if m.bias is not None:
                    ones = torch.ones(x.size(0), 1, device=x.device)
                    x = torch.cat([x, ones], dim=1)
                cov_a = x.t().mm(x) / x.size(0)
                s = self._stats[name]
                if s['A'] is None:
                    s['A'] = cov_a
                else:
                    s['A'].mul_(self.alpha).add_(cov_a, alpha=1-self.alpha)
        return hook

    def _bwd_hook(self, name, mod):
        def hook(m, grad_input, grad_output):
            if not m.training:
                return
            with torch.no_grad():
                g = grad_output[0].detach()
                if isinstance(m, nn.Conv2d):
                    g = g.permute(0, 2, 3, 1).reshape(-1, g.size(1))
                elif g.dim() > 2:
                    g = g.reshape(-1, g.size(-1))
                cov_g = g.t().mm(g) / g.size(0)
                s = self._stats[name]
                if s['G'] is None:
                    s['G'] = cov_g
                else:
                    s['G'].mul_(self.alpha).add_(cov_g, alpha=1-self.alpha)
        return hook

    def _invert_factors(self):
        sqrt_d = self.damping ** 0.5
        for name, s in self._stats.items():
            A, G = s['A'], s['G']
            if A is None or G is None:
                continue
            try:
                A_d = A + sqrt_d * torch.eye(A.size(0), device=A.device)
                G_d = G + sqrt_d * torch.eye(G.size(0), device=G.device)
                self._inv[name] = {
                    'A_inv': torch.linalg.inv(A_d),
                    'G_inv': torch.linalg.inv(G_d),
                }
            except RuntimeError:
                pass

    def _compute_nat_grads(self):
        updates  = {}
        g_dot_ng = 0.0
        for name, mod in self._modules.items():
            if mod.weight.grad is None:
                continue
            gw    = mod.weight.grad
            has_b = mod.bias is not None and mod.bias.grad is not None
            if name in self._inv:
                Ai, Gi = self._inv[name]['A_inv'], self._inv[name]['G_inv']
                g2d = torch.cat([gw, mod.bias.grad.unsqueeze(1)], 1) if has_b else gw
                ng  = Gi.mm(g2d).mm(Ai)
                ngw = ng[:, :-1] if has_b else ng
                ngb = ng[:, -1]  if has_b else None
            else:
                ngw = gw.clone()
                ngb = mod.bias.grad.clone() if has_b else None
            g_dot_ng += (gw * ngw).sum().item()
            if has_b and ngb is not None:
                g_dot_ng += (mod.bias.grad * ngb).sum().item()
            updates[name] = (ngw, ngb)
        return updates, g_dot_ng

    def _adapt_damping(self, loss_value):
        if not self.adapt_damping:
            return
        if (self.steps + 1) % self._damp_interval == 0:
            self._prev_loss = loss_value
        if (self.steps % self._damp_interval == 0) and self.steps > 0:
            if math.isnan(self._prev_loss) or math.isnan(self._qmodel_change):
                return
            rho = (loss_value - self._prev_loss) / (self._qmodel_change - 1e-12)
            self._rho = rho
            old = self.damping
            if   rho > self._rho_dec: self.damping = max(self.damping * self._omega, self._min_damping)
            elif rho < self._rho_inc: self.damping = self.damping / self._omega
            if self.damping != old:
                self._invert_factors()

    def reset_stats(self):
        self.steps       = 0
        self.damping     = self._init_damp
        self._prev_loss  = float('nan')
        self._qmodel_change = float('nan')

    @torch.no_grad()
    def step(self, closure=None, loss_value=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if loss_value is not None:
            self._adapt_damping(loss_value)
        if self.steps % self.T_inv == 0:
            self._invert_factors()
        lr = self.param_groups[0]['lr']
        updates, g_dot_ng = self._compute_nat_grads()
        total_sq = sum(
            ngw.norm().item()**2 + (ngb.norm().item()**2 if ngb is not None else 0.0)
            for ngw, ngb in updates.values()
        )
        clip = 1.0
        if self.max_grad_norm > 0 and total_sq**0.5 > self.max_grad_norm:
            clip = self.max_grad_norm / (total_sq**0.5 + 1e-6)
        if self.adapt_damping and ((self.steps + 1) % self._damp_interval == 0):
            self._qmodel_change = -0.5 * lr * clip * g_dot_ng
        if self.weight_decay > 0:
            f = 1.0 - lr * self.weight_decay
            for mod in self._modules.values():
                mod.weight.data.mul_(f)
                if mod.bias is not None:
                    mod.bias.data.mul_(f)
        for name, (ngw, ngb) in updates.items():
            mod = self._modules[name]
            mod.weight.data.add_(ngw, alpha=-lr * clip)
            if ngb is not None and mod.bias is not None:
                mod.bias.data.add_(ngb, alpha=-lr * clip)
        tracked = {id(m.weight) for m in self._modules.values()}
        tracked |= {id(m.bias) for m in self._modules.values() if m.bias is not None}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or id(p) in tracked:
                    continue
                if self.weight_decay > 0:
                    p.data.mul_(1.0 - lr * self.weight_decay)
                p.data.add_(p.grad, alpha=-lr)
        self.steps += 1
        return loss

print("✓ KFACOptimizer (K-FAC NGD) defined")

# %% [markdown]
# ## 4. Build Optimizer for RL

# %%
SEED = 42

def build_optimizer_rl(config, pol, vf):
    """Build the appropriate optimizer from config for both policy + value networks."""
    opt_type = config['optimizer']
    all_params = list(pol.parameters()) + list(vf.parameters())

    if opt_type == 'adam':
        return torch.optim.Adam(
            all_params, lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.0),
            eps=config.get('eps', 1e-8))

    elif opt_type == 'adahessian':
        return Adahessian(
            all_params, lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.0),
            eps=config.get('eps', 1e-4),
            hessian_power=config.get('hessian_power', 1.0),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1), seed=SEED)

    elif opt_type == 'sophia':
        return SophiaH(
            all_params, lr=config['lr'],
            betas=config.get('betas', (0.965, 0.99)),
            weight_decay=config.get('weight_decay', 1e-4),
            eps=config.get('eps', 1e-4),
            clip_threshold=config.get('clip_threshold', 1.0),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1), seed=SEED)

    elif opt_type == 'shampoo':
        return Shampoo(
            all_params, lr=config['lr'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0),
            epsilon=config.get('epsilon', 0.1),
            update_freq=config.get('update_freq', 50))

    elif opt_type == 'asam':
        return ASAM(
            all_params, torch.optim.SGD,
            rho=config.get('rho', 0.05), adaptive=True,
            lr=config['lr'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0))

    elif opt_type == 'sassha':
        # Build hessian power scheduler from config
        schedule_type = config.get('hessian_power_schedule', 'constant')
        total_steps = config.get('total_steps', N_STEPS)
        max_hp = config.get('max_hessian_power', 1.0)
        min_hp = config.get('min_hessian_power', 0.5)
        hp_scheduler = build_hessian_power_scheduler(
            schedule_type, total_steps,
            max_hessian_power=max_hp,
            min_hessian_power=min_hp)
        return SASSHA(
            all_params,
            hessian_power_scheduler=hp_scheduler,
            lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.0),
            rho=config.get('rho', 0.05),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1),
            eps=config.get('eps', 1e-4),
            hessian_floor=config.get('hessian_floor', 1e-4),
            hessian_clip=config.get('hessian_clip', 1e3),
            seed=SEED)

    elif opt_type == 'ngd':
        # K-FAC needs a single nn.Module to register hooks on
        combined = CombinedPolicyVF(pol, vf)
        return KFACOptimizer(
            combined, lr=config['lr'],
            damping=config.get('damping', 1e-3),
            weight_decay=config.get('weight_decay', 0),
            T_inv=config.get('T_inv', 50),
            alpha=config.get('alpha', 0.95),
            max_grad_norm=config.get('max_grad_norm', 1.0),
            adapt_damping=config.get('adapt_damping', True))
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

print("✓ build_optimizer_rl defined")

# %% [markdown]
# ## 5. PPO Learning with Second-Order Optimizers

# %%
def _needs_create_graph(config):
    """Does this optimizer need create_graph=True for Hessian?"""
    return config['optimizer'] in ('adahessian', 'sophia', 'sassha')

def _is_two_pass(config):
    """Does this optimizer use SAM-style two-pass?"""
    return config['optimizer'] in ('sassha', 'asam')


def ppo_learn(pol, vf, opt, buf, config, grad_clip=1.0):
    """
    PPO learning step with second-order optimizer support.
    Follows CBP paper's PPO implementation (lop/algos/rl/ppo.py).

    Returns dict with 'p_loss', 'v_loss', and buffer data for GOSC:
      'obs_buf', 'acts_buf', 'logp_old_buf', 'advs_buf', 'v_rets_buf'
    """
    g = config.get('g', 0.99)
    lm = config.get('lm', 0.95)
    bs = config.get('bs', 2048)
    n_itrs = config.get('n_itrs', 10)
    n_slices = config.get('n_slices', 16)
    clip_eps = config.get('clip_eps', 0.2)
    opt_type = config['optimizer']
    create_graph = _needs_create_graph(config)
    two_pass = _is_two_pass(config)

    os_t, acts, rs, op, logpbs, _, dones = buf.get(pol.dist_stack)

    # Compute values for GAE
    with torch.no_grad():
        pre_vals = vf.value(torch.cat((os_t, op)))

    # GAE computation
    vals = pre_vals.squeeze()
    rs_sq = rs.squeeze()
    dones_sq = dones.squeeze()
    advs = torch.zeros(len(rs_sq) + 1, device=rs_sq.device)
    for t in reversed(range(len(rs_sq))):
        delta = rs_sq[t] + (1 - dones_sq[t]) * g * vals[t + 1] - vals[t]
        advs[t] = delta + (1 - dones_sq[t]) * g * lm * advs[t + 1]
    v_rets = advs[:-1] + vals[:-1]
    advs = advs[:-1].view(-1)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    v_rets = v_rets.view(-1).detach()
    advs = advs.detach()
    logpbs = logpbs.view(-1).detach()

    inds = np.arange(os_t.shape[0])
    mini_bs = bs // n_slices
    p_loss_val, v_loss_val = 0.0, 0.0

    all_params = list(pol.parameters()) + list(vf.parameters())

    for _ in range(n_itrs):
        np.random.shuffle(inds)
        for start in range(0, len(os_t), mini_bs):
            ind = inds[start:start + mini_bs]
            
            advs_i = advs[ind]
            v_rets_i = v_rets[ind]
            logpbs_i = logpbs[ind]

            if opt_type == 'sassha':
                # SASSHA two-pass: forward → backward → perturb → forward → backward(create_graph) → step
                opt.zero_grad()
                logpts, _ = pol.logp_dist(os_t[ind], acts[ind], to_log_features=True)
                logpts = logpts.view(-1)
                grad_sub = (logpts - logpbs_i).exp()
                p_loss0 = -(grad_sub * advs_i)
                ext_loss = -(torch.clamp(grad_sub, 1 - clip_eps, 1 + clip_eps) * advs_i)
                p_loss = torch.max(p_loss0, ext_loss).mean()
                
                v_vals = vf.value(os_t[ind], to_log_features=True).view(-1)
                v_loss = (v_rets_i - v_vals).pow(2).mean()
                loss = p_loss + v_loss
                loss.backward()
                
                # NaN guard: skip update if gradients exploded
                if any(p.grad is not None and torch.isnan(p.grad).any() for p in all_params):
                    opt.zero_grad(set_to_none=True)
                    continue
                
                opt.perturb_weights(zero_grad=True)
                
                # Second forward pass (perturbed)
                logpts2, _ = pol.logp_dist(os_t[ind], acts[ind])
                logpts2 = logpts2.view(-1)
                grad_sub2 = (logpts2 - logpbs_i).exp()
                p_loss2 = torch.max(-(grad_sub2 * advs_i),
                                    -(torch.clamp(grad_sub2, 1-clip_eps, 1+clip_eps) * advs_i)).mean()
                v_vals2 = vf.value(os_t[ind]).view(-1)
                v_loss2 = (v_rets_i - v_vals2).pow(2).mean()
                loss2 = p_loss2 + v_loss2
                loss2.backward(create_graph=create_graph)
                
                opt.unperturb()
                
                # No grad_clip for SASSHA: Hessian denominator provides
                # per-parameter adaptive scaling. Global clip breaks this.
                opt.step()
                opt.zero_grad(set_to_none=True)
                
                p_loss_val = p_loss.item()
                v_loss_val = v_loss.item()

            elif opt_type == 'asam':
                # ASAM explicit two-pass (following official davda54/sam pattern)
                # Pass 1: forward → backward → first_step (perturb)
                opt.zero_grad()
                logpts, _ = pol.logp_dist(os_t[ind], acts[ind], to_log_features=True)
                logpts = logpts.view(-1)
                grad_sub = (logpts - logpbs_i).exp()
                p_loss0 = -(grad_sub * advs_i)
                ext_loss = -(torch.clamp(grad_sub, 1 - clip_eps, 1 + clip_eps) * advs_i)
                p_loss = torch.max(p_loss0, ext_loss).mean()
                
                v_vals = vf.value(os_t[ind], to_log_features=True).view(-1)
                v_loss = (v_rets_i - v_vals).pow(2).mean()
                loss = p_loss + v_loss
                loss.backward()
                
                # NaN guard: skip update if gradients exploded (RL-specific)
                if any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in all_params):
                    opt.zero_grad(set_to_none=True)
                    continue
                
                # Clip gradients before SAM perturbation to prevent weight explosion
                # In RL, unclipped gradients can be very large due to importance
                # sampling ratios, causing e_w = p² * grad * scale to overflow.
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, grad_clip)
                
                opt.first_step(zero_grad=True)  # perturb weights: w → w + e(w)
                
                # Pass 2: forward → backward → second_step (restore + base step)
                logpts_c, _ = pol.logp_dist(os_t[ind], acts[ind])
                
                # NaN guard: if perturbed weights produce NaN, skip this update
                if torch.isnan(logpts_c).any():
                    # Restore original weights and skip
                    for group in opt.param_groups:
                        for p in group["params"]:
                            if "old_p" in opt.state[p]:
                                p.data = opt.state[p]["old_p"]
                    opt.zero_grad(set_to_none=True)
                    continue
                
                logpts_c = logpts_c.view(-1)
                gs = (logpts_c - logpbs_i).exp()
                pl = torch.max(-(gs * advs_i),
                               -(torch.clamp(gs, 1-clip_eps, 1+clip_eps) * advs_i)).mean()
                vv = vf.value(os_t[ind]).view(-1)
                vl = (v_rets_i - vv).pow(2).mean()
                total = pl + vl
                total.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, grad_clip)
                
                opt.second_step(zero_grad=True)  # restore w, apply base optimizer
                
                p_loss_val = p_loss.item()
                v_loss_val = v_loss.item()

            elif opt_type in ('adahessian', 'sophia'):
                # Hessian-based: backward with create_graph
                opt.zero_grad()
                logpts, _ = pol.logp_dist(os_t[ind], acts[ind], to_log_features=True)
                logpts = logpts.view(-1)
                grad_sub = (logpts - logpbs_i).exp()
                p_loss0 = -(grad_sub * advs_i)
                ext_loss = -(torch.clamp(grad_sub, 1 - clip_eps, 1 + clip_eps) * advs_i)
                p_loss = torch.max(p_loss0, ext_loss).mean()
                
                v_vals = vf.value(os_t[ind], to_log_features=True).view(-1)
                v_loss = (v_rets_i - v_vals).pow(2).mean()
                loss = p_loss + v_loss
                loss.backward(create_graph=True)
                
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                
                p_loss_val = p_loss.item()
                v_loss_val = v_loss.item()

            elif opt_type == 'ngd':
                # K-FAC NGD: single pass, step with loss_value for adaptive damping
                opt.zero_grad()
                logpts, _ = pol.logp_dist(os_t[ind], acts[ind], to_log_features=True)
                logpts = logpts.view(-1)
                grad_sub = (logpts - logpbs_i).exp()
                p_loss0 = -(grad_sub * advs_i)
                ext_loss = -(torch.clamp(grad_sub, 1 - clip_eps, 1 + clip_eps) * advs_i)
                p_loss = torch.max(p_loss0, ext_loss).mean()
                
                v_vals = vf.value(os_t[ind], to_log_features=True).view(-1)
                v_loss = (v_rets_i - v_vals).pow(2).mean()
                loss = p_loss + v_loss
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, grad_clip)
                opt.step(loss_value=loss.item())
                opt.zero_grad(set_to_none=True)
                
                p_loss_val = p_loss.item()
                v_loss_val = v_loss.item()

            else:
                # Standard (adam, shampoo): single pass
                opt.zero_grad()
                logpts, _ = pol.logp_dist(os_t[ind], acts[ind], to_log_features=True)
                logpts = logpts.view(-1)
                grad_sub = (logpts - logpbs_i).exp()
                p_loss0 = -(grad_sub * advs_i)
                ext_loss = -(torch.clamp(grad_sub, 1 - clip_eps, 1 + clip_eps) * advs_i)
                p_loss = torch.max(p_loss0, ext_loss).mean()
                
                v_vals = vf.value(os_t[ind], to_log_features=True).view(-1)
                v_loss = (v_rets_i - v_vals).pow(2).mean()
                loss = p_loss + v_loss
                loss.backward()
                
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                
                p_loss_val = p_loss.item()
                v_loss_val = v_loss.item()

    return {
        'p_loss': p_loss_val, 'v_loss': v_loss_val,
        'obs_buf': os_t.detach(), 'acts_buf': acts.detach(),
        'logp_old_buf': logpbs.detach(), 'advs_buf': advs.detach(),
        'v_rets_buf': v_rets.detach()
    }


print("✓ PPO learning with second-order support defined")

# %% [markdown]
# ## 6. Configs

# %%
# ── RL environment settings (from CBP paper: lop/rl/cfg/ant/cbp.yml) ──
ENV_NAME    = 'Ant-v4'
N_STEPS     = int(10e6)    # 10M steps (reduce for quick tests)
H_DIM       = (256, 256)
ACT_TYPE    = 'ReLU'
BS          = 2048
N_ITRS      = 10
N_SLICES    = 16
CLIP_EPS    = 0.2
GAMMA       = 0.99
LAMBDA      = 0.95
INIT        = 'lecun'
GRAD_CLIP   = 1.0

# ── SDP settings ──
SDP_GAMMA       = 0.3
SDP_INTERVAL    = 100_000   # Apply SDP every 100K steps (no task boundary in RL)
LOG_INTERVAL    = 1000      # Log metrics every 1K steps
SAVE_INTERVAL   = 500_000   # Save checkpoint every 500K steps

TIME_LIMIT_SECONDS = 11.5 * 3600

# ── Per-optimizer configs (with and without SDP) ──
CONFIGS = {
    # 0. Adam baseline + SDP
    'adam_sdp': dict(
        optimizer='adam',
        lr=0.0001, betas=(0.99, 0.99), weight_decay=1e-4, eps=1e-8,
        sdp_gamma=SDP_GAMMA,
    ),
    # 1. Adam baseline (no SDP)
    'adam_nosdp': dict(
        optimizer='adam',
        lr=0.0001, betas=(0.99, 0.99), weight_decay=1e-4, eps=1e-8,
        sdp_gamma=0.0,
    ),
    # 2. AdaHessian + SDP
    'adahessian_sdp': dict(
        optimizer='adahessian',
        lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4, eps=1e-3,
        hessian_power=0.5, lazy_hessian=1, n_samples=1,
        sdp_gamma=SDP_GAMMA,
    ),
    # 3. AdaHessian (no SDP)
    'adahessian_nosdp': dict(
        optimizer='adahessian',
        lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4, eps=1e-3,
        hessian_power=0.5, lazy_hessian=1, n_samples=1,
        sdp_gamma=0.0,
    ),
    # 4. SophiaH + SDP
    'sophia_sdp': dict(
        optimizer='sophia',
        lr=0.0001, betas=(0.965, 0.99), weight_decay=1e-4, eps=1e-4,
        clip_threshold=1.0, lazy_hessian=10, n_samples=1,
        sdp_gamma=SDP_GAMMA,
    ),
    # 5. SophiaH (no SDP)
    'sophia_nosdp': dict(
        optimizer='sophia',
        lr=0.0001, betas=(0.965, 0.99), weight_decay=1e-4, eps=1e-4,
        clip_threshold=1.0, lazy_hessian=10, n_samples=1,
        sdp_gamma=0.0,
    ),
    # 6. Shampoo + SDP
    'shampoo_sdp': dict(
        optimizer='shampoo',
        lr=0.0001, momentum=0.9, weight_decay=3e-5, epsilon=0.1,
        update_freq=50,
        sdp_gamma=SDP_GAMMA,
    ),
    # 7. Shampoo (no SDP)
    'shampoo_nosdp': dict(
        optimizer='shampoo',
        lr=0.0001, momentum=0.9, weight_decay=3e-5, epsilon=0.1,
        update_freq=50,
        sdp_gamma=0.0,
    ),
    # 8. ASAM + SDP
    'asam_sdp': dict(
        optimizer='asam',
        lr=0.0001, rho=0.05, momentum=0.9, weight_decay=1e-4,
        sdp_gamma=SDP_GAMMA,
    ),
    # 9. ASAM (no SDP)
    'asam_nosdp': dict(
        optimizer='asam',
        lr=0.0001, rho=0.05, momentum=0.9, weight_decay=1e-4,
        sdp_gamma=0.0,
    ),
    # 10. SASSHA + SDP  (balanced from high-reward notebook analysis)
    'sassha_sdp': dict(
        optimizer='sassha',
        lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-4, rho=0.1,  # lr↑5×, rho↑2×, beta2=0.999
        lazy_hessian=1, n_samples=1, eps=1e-3,                      # eps↑10× (prevents div-by-zero explosion)
        hessian_power_schedule='constant', max_hessian_power=0.5, min_hessian_power=0.5,
        sdp_gamma=SDP_GAMMA,
    ),
    # 11. SASSHA (no SDP)
    'sassha_nosdp': dict(
        optimizer='sassha',
        lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-4, rho=0.1,
        lazy_hessian=1, n_samples=1, eps=1e-3,
        hessian_power_schedule='constant', max_hessian_power=0.5, min_hessian_power=0.5,
        sdp_gamma=0.0,
    ),
    # 12. K-FAC NGD + SDP
    'ngd_sdp': dict(
        optimizer='ngd',
        lr=0.001, damping=1e-3, weight_decay=0, T_inv=50, alpha=0.95,
        max_grad_norm=1.0, adapt_damping=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 13. K-FAC NGD (no SDP)
    'ngd_nosdp': dict(
        optimizer='ngd',
        lr=0.001, damping=1e-3, weight_decay=0, T_inv=50, alpha=0.95,
        max_grad_norm=1.0, adapt_damping=True,
        sdp_gamma=0.0,
    ),
    # 14. SASSHA + GOSC (gradient-orthogonal spectral compression)
    'sassha_gosc': dict(
        optimizer='sassha',
        lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-4, rho=0.1,
        lazy_hessian=1, n_samples=1, eps=1e-3,
        hessian_power_schedule='constant', max_hessian_power=0.5, min_hessian_power=0.5,
        sdp_gamma=0.0,
        gosc=True, gosc_gamma_min=0.05, gosc_gamma_max=0.5, gosc_rank=5,
    ),
    # 15. Adam + GOSC (ablation vs Adam+SDP)
    'adam_gosc': dict(
        optimizer='adam',
        lr=0.0001, betas=(0.99, 0.99), weight_decay=1e-4, eps=1e-8,
        sdp_gamma=0.0,
        gosc=True, gosc_gamma_min=0.05, gosc_gamma_max=0.5, gosc_rank=5,
    ),
}

# ── Select which methods to run ──
METHODS_TO_RUN = [
    'adam_sdp', 'adam_nosdp',
    'adahessian_sdp', 'adahessian_nosdp',
    'sophia_sdp', 'sophia_nosdp',
    'shampoo_sdp', 'shampoo_nosdp',
    'asam_sdp', 'asam_nosdp',
    'sassha_sdp', 'sassha_nosdp',
    'ngd_sdp', 'ngd_nosdp',
    'sassha_gosc', 'adam_gosc',
]

RESULTS_DIR = 'rl_results/secondorder_sdp'
CKPT_DIR    = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"✓ Config: {ENV_NAME}, {N_STEPS:,} steps, bs={BS}, h_dim={H_DIM}")
print(f"  SDP: γ={SDP_GAMMA}, interval={SDP_INTERVAL:,} steps")
print(f"  Methods: {METHODS_TO_RUN}")

# %% [markdown]
# ## 7. Training Loop

# %%
def run_method_rl(method_name, config):
    """
    Run PPO with a second-order optimizer ± SDP on Ant-v4.
    Follows CBP paper RL setup + rl-continual.ipynb structure.
    """
    opt_type = config['optimizer']
    sdp_gamma = config.get('sdp_gamma', 0.0)
    use_gosc = config.get('gosc', False)
    gosc_gamma_min = config.get('gosc_gamma_min', 0.05)
    gosc_gamma_max = config.get('gosc_gamma_max', 0.5)
    gosc_rank = config.get('gosc_rank', 5)
    # Inject shared RL hyperparams into config
    config.setdefault('g', GAMMA)
    config.setdefault('lm', LAMBDA)
    config.setdefault('bs', BS)
    config.setdefault('n_itrs', N_ITRS)
    config.setdefault('n_slices', N_SLICES)
    config.setdefault('clip_eps', CLIP_EPS)

    print(f"\n{'='*70}")
    print(f"  {method_name} (optimizer={opt_type}) — PPO on {ENV_NAME}")
    if use_gosc:
        print(f"  GOSC enabled: γ_min={gosc_gamma_min}, γ_max={gosc_gamma_max}, rank={gosc_rank}, interval={SDP_INTERVAL:,}")
    elif sdp_gamma > 0:
        print(f"  SDP enabled: γ={sdp_gamma}, interval={SDP_INTERVAL:,}")
    else:
        print(f"  SDP/GOSC disabled")
    print(f"{'='*70}")

    wall_clock_start = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ── Create environment (Gymnasium API) ──
    env = gym.make(ENV_NAME)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    print(f"  Env: {ENV_NAME}, obs_dim={o_dim}, act_dim={a_dim}")

    # ── Create networks (same as CBP paper) ──
    pol = MLPPolicy(o_dim, a_dim, act_type=ACT_TYPE, h_dim=list(H_DIM),
                    device=device, init=INIT)
    vf = MLPVF(o_dim, act_type=ACT_TYPE, h_dim=list(H_DIM),
               device=device, init=INIT)

    # ── Create optimizer ──
    opt = build_optimizer_rl(config, pol, vf)

    # ── Create buffer ──
    buf = Buffer(o_dim, a_dim, BS, device=device)

    # ── Results tracking ──
    R = {
        'episodic_returns': [], 'termination_steps': [],
        'dormant_units': [], 'weight_magnitude': [],
        'stable_rank': [], 'p_loss': [], 'v_loss': [],
        'sdp_cond': [], 'gosc_alignment': [],
        'hyperparams': {
            'seed': SEED, 'env_name': ENV_NAME, 'optimizer': opt_type,
            'sdp_gamma': sdp_gamma, 'lr': config['lr'],
            'gosc': use_gosc, 'gosc_gamma_min': gosc_gamma_min,
            'gosc_gamma_max': gosc_gamma_max, 'gosc_rank': gosc_rank,
        },
    }

    # ── Feature tracking for stable rank ──
    num_layers = len(H_DIM)
    short_term_feature_activity = torch.zeros(1000, H_DIM[-1], device=device)
    feature_idx = 0
    last_ppo_data = None  # Store last PPO buffer data for GOSC

    # ── Checkpoint resume ──
    ckpt_file = os.path.join(CKPT_DIR, f"ckpt_{method_name}.pt")
    start_step = 0
    if os.path.isfile(ckpt_file):
        print(f"  Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        pol.load_state_dict(ckpt['pol'])
        vf.load_state_dict(ckpt['vf'])
        try:
            opt.load_state_dict(ckpt['opt'])
        except Exception:
            print("  Warning: could not restore optimizer state, starting fresh")
        R = ckpt.get('results', R)
        start_step = ckpt['step'] + 1
        feature_idx = ckpt.get('feature_idx', 0)
        # Reset Hessian state
        if hasattr(opt, 'get_params'):
            for p in opt.get_params():
                p.hess = 0.0
                opt.state[p]["hessian step"] = 0
        print(f"  ✓ Resumed from step {start_step:,}")
        del ckpt; torch.cuda.empty_cache()
    else:
        print("  (no checkpoint — training from scratch)")

    # ── Training loop (Gymnasium API) ──
    o, info = env.reset(seed=SEED)
    episode_return = 0
    episode_count = len(R['episodic_returns'])

    pbar = tqdm(range(start_step, N_STEPS), desc=method_name, initial=start_step, total=N_STEPS)
    for step in pbar:
        elapsed = time.time() - wall_clock_start
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"\n  Time limit ({elapsed/3600:.1f}h). Saving checkpoint.")
            break

        # ── Get action and log features ──
        with torch.no_grad():
            a, logp, dist = pol.action(
                torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0),
                to_log_features=True
            )
            # Store last hidden layer features for stable rank
            if hasattr(pol, 'activations') and pol.activations:
                features = list(pol.activations.values())
                # Find last hidden layer (match H_DIM[-1]), skip output layer
                for fi in reversed(features):
                    if fi is not None and fi.dim() >= 2 and fi.shape[-1] == H_DIM[-1]:
                        short_term_feature_activity[feature_idx % 1000] = fi[0]
                        feature_idx += 1
                        break

        a_np = a[0].cpu().numpy()

        # ── Step environment (Gymnasium API) ──
        op, r, terminated, truncated, info = env.step(a_np)
        done = terminated or truncated
        episode_return += r

        # ── Store transition ──
        buf.store(o, a_np, r, op, logp.cpu().numpy(),
                  pol.dist_to(dist, to_device='cpu'), float(done))
        o = op

        # ── Episode ended ──
        if done:
            R['episodic_returns'].append(episode_return)
            R['termination_steps'].append(step)
            episode_count += 1
            episode_return = 0
            o, info = env.reset()

        # ── Learning step when buffer is full ──
        if len(buf.o_buf) >= BS:
            learn_logs = ppo_learn(pol, vf, opt, buf, config, grad_clip=GRAD_CLIP)
            buf.clear()
            R['p_loss'].append(learn_logs['p_loss'])
            R['v_loss'].append(learn_logs['v_loss'])
            # Store buffer data for GOSC (used at next spectral compression)
            if use_gosc:
                last_ppo_data = {
                    'obs': learn_logs['obs_buf'],
                    'acts': learn_logs['acts_buf'],
                    'logp_old': learn_logs['logp_old_buf'],
                    'advs': learn_logs['advs_buf'],
                    'v_rets': learn_logs['v_rets_buf'],
                }

            # Compute metrics
            R['weight_magnitude'].append(compute_weight_magnitude(pol, vf))
            R['dormant_units'].append(compute_dormant_units_rl(pol, vf))

        # ── Periodic Spectral Compression (SDP or GOSC) ──
        if step > 0 and step % SDP_INTERVAL == 0:
            if use_gosc and last_ppo_data is not None:
                cond_nums, avg_align = apply_gosc_rl(
                    pol, vf,
                    obs_buf=last_ppo_data['obs'],
                    acts_buf=last_ppo_data['acts'],
                    logp_old_buf=last_ppo_data['logp_old'],
                    advs_buf=last_ppo_data['advs'],
                    v_rets_buf=last_ppo_data['v_rets'],
                    gamma_min=gosc_gamma_min, gamma_max=gosc_gamma_max,
                    grad_rank=gosc_rank, device=device,
                    clip_eps=config.get('clip_eps', CLIP_EPS))
                avg_cond = sum(cond_nums) / max(len(cond_nums), 1) if cond_nums else 0.0
                R['sdp_cond'].append(avg_cond)
                R['gosc_alignment'].append(avg_align)
                print(f"\n  [{method_name}] Step {step:,}: GOSC applied, "
                      f"avg cond={avg_cond:.1f}, avg align={avg_align:.3f}")
            elif use_gosc and last_ppo_data is None:
                print(f"\n  [{method_name}] Step {step:,}: GOSC skipped (no PPO data yet)")
            elif sdp_gamma > 0:
                cond_nums = apply_sdp_rl(pol, vf, sdp_gamma)
                avg_cond = sum(cond_nums) / max(len(cond_nums), 1) if cond_nums else 0.0
                R['sdp_cond'].append(avg_cond)
                print(f"\n  [{method_name}] Step {step:,}: SDP applied, avg cond={avg_cond:.1f}")
            else:
                avg_cond = 0.0  # no spectral compression
            # Force Hessian recompute + reset EMA after spectral compression
            if (use_gosc or sdp_gamma > 0) and hasattr(opt, 'get_params'):
                for p in opt.get_params():
                    if p in opt.state:
                        opt.state[p]['hessian step'] = 0
                        if 'exp_hessian_diag' in opt.state[p]:
                            opt.state[p]['exp_hessian_diag'].zero_()
                        if 'exp_hessian_diag_sq' in opt.state[p]:
                            opt.state[p]['exp_hessian_diag_sq'].zero_()
                        if 'bias_correction2' in opt.state[p]:
                            opt.state[p]['bias_correction2'] = 0

        # ── Compute stable rank every 10K steps ──
        if step > 0 and step % 10000 == 0:
            valid_samples = min(feature_idx, 1000)
            if valid_samples > 10:
                sr = compute_stable_rank_from_features(
                    short_term_feature_activity[:valid_samples].cpu())
                R['stable_rank'].append(sr)

        # ── Progress bar update ──
        if step > 0 and step % LOG_INTERVAL == 0:
            recent_rets = R['episodic_returns'][-10:] if R['episodic_returns'] else [0]
            avg_ret = np.mean(recent_rets)
            dormant = R['dormant_units'][-1] * 100 if R['dormant_units'] else 0
            sr_val = R['stable_rank'][-1] if R['stable_rank'] else H_DIM[-1]
            w_norm = R['weight_magnitude'][-1] if R['weight_magnitude'] else 0.0
            pbar.set_postfix({
                'Eps': episode_count,
                'Ret': f'{avg_ret:.0f}',
                'Dorm': f'{dormant:.1f}%',
                'SR': f'{sr_val:.0f}',
                'WNorm': f'{w_norm:.4f}',
            })

        # ── Periodic checkpoint save ──
        if step > 0 and step % SAVE_INTERVAL == 0:
            ckpt_data = {
                'step': step, 'pol': pol.state_dict(), 'vf': vf.state_dict(),
                'opt': opt.state_dict(), 'results': R, 'feature_idx': feature_idx,
            }
            torch.save(ckpt_data, ckpt_file)
            elapsed = time.time() - wall_clock_start
            print(f"\n  [{method_name}] Checkpoint at step {step:,} [{elapsed/3600:.1f}h]")

    # ── Final save ──
    env.close()
    result_file = os.path.join(RESULTS_DIR, f'{method_name}_results.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump(R, f)
    print(f"  ✓ {method_name}: {len(R['episodic_returns'])} episodes, "
          f"final avg return = {np.mean(R['episodic_returns'][-100:]):.1f}")
    return R


print("✓ Training loop defined")

# %% [markdown]
# ## 8. Run All Experiments

# %%
all_results = {}
for method in METHODS_TO_RUN:
    cfg = CONFIGS[method]
    result = run_method_rl(method, cfg)
    all_results[method] = result

# %% [markdown]
# ## 9. Results Plots

# %%
METHOD_STYLES = {
    'adam_sdp':          {'color': '#607D8B', 'ls': '-',  'label': 'Adam+SDP'},
    'adam_nosdp':        {'color': '#607D8B', 'ls': '--', 'label': 'Adam'},
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
    'ngd_kfac':         {'color': '#00BCD4', 'ls': '-',  'label': 'NGD (K-FAC)'},
    'sassha_gosc':      {'color': '#F44336', 'ls': '-',  'label': 'SASSHA+GOSC'},
    'adam_gosc':        {'color': '#795548', 'ls': '-',  'label': 'Adam+GOSC'},
}

def _style(method):
    return METHOD_STYLES.get(method, {'color': 'gray', 'ls': '-', 'label': method})

def _clean(ax):
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def smooth_returns(rets, window=100):
    """Smooth episodic returns with a rolling average."""
    if len(rets) < window:
        return np.array(rets) if rets else np.array([0])
    return np.convolve(rets, np.ones(window)/window, mode='valid')


# ── Main comparison: 3×2 grid ──
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle(f'Second-Order Optimizers ± SDP — PPO on {ENV_NAME}\n'
             f'Network: {H_DIM}, {ACT_TYPE}, {N_STEPS:,} steps, bs={BS}',
             fontsize=14, fontweight='bold')

# ── (0,0) Episodic Return ──
ax = axes[0, 0]
for m, d in all_results.items():
    s = _style(m)
    sm = smooth_returns(d['episodic_returns'])
    ax.plot(np.arange(len(sm)), sm, color=s['color'], ls=s['ls'], lw=1.5,
            label=s['label'], alpha=0.9)
ax.set_xlabel('Episode'); ax.set_ylabel('Return')
ax.set_title('Episodic Return (smoothed)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (0,1) Dormant Neurons ──
ax = axes[0, 1]
for m, d in all_results.items():
    s = _style(m)
    dorm = np.array(d['dormant_units']) * 100
    if len(dorm) > 0:
        ax.plot(np.arange(len(dorm)), dorm, color=s['color'], ls=s['ls'],
                lw=1.5, label=s['label'], alpha=0.9)
ax.set_xlabel('PPO Update'); ax.set_ylabel('Dormant (%)')
ax.set_title('Dormant Neuron Fraction')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (1,0) Stable Rank ──
ax = axes[1, 0]
for m, d in all_results.items():
    s = _style(m)
    sr = np.array(d['stable_rank'])
    if len(sr) > 0:
        ax.plot(np.arange(len(sr)), sr, color=s['color'], ls=s['ls'],
                lw=1.5, label=s['label'], alpha=0.9)
ax.set_xlabel('× 10K steps'); ax.set_ylabel('Stable Rank')
ax.set_title('Stable Rank (last hidden)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (1,1) Weight Magnitude ──
ax = axes[1, 1]
for m, d in all_results.items():
    s = _style(m)
    wm = np.array(d['weight_magnitude'])
    if len(wm) > 0:
        ax.plot(np.arange(len(wm)), wm, color=s['color'], ls=s['ls'],
                lw=1.5, label=s['label'], alpha=0.9)
ax.set_xlabel('PPO Update'); ax.set_ylabel('Avg |W|')
ax.set_title('Average Weight Magnitude')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (2,0) Policy Loss ──
ax = axes[2, 0]
for m, d in all_results.items():
    s = _style(m)
    pl = np.array(d['p_loss'])
    if len(pl) > 0:
        sm = np.convolve(pl, np.ones(50)/50, mode='valid') if len(pl) > 50 else pl
        ax.plot(np.arange(len(sm)), sm, color=s['color'], ls=s['ls'],
                lw=1.5, label=s['label'], alpha=0.9)
ax.set_xlabel('PPO Update'); ax.set_ylabel('Policy Loss')
ax.set_title('Policy Loss (smoothed)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

# ── (2,1) Value Loss ──
ax = axes[2, 1]
for m, d in all_results.items():
    s = _style(m)
    vl = np.array(d['v_loss'])
    if len(vl) > 0:
        sm = np.convolve(vl, np.ones(50)/50, mode='valid') if len(vl) > 50 else vl
        ax.plot(np.arange(len(sm)), sm, color=s['color'], ls=s['ls'],
                lw=1.5, label=s['label'], alpha=0.9)
ax.set_xlabel('PPO Update'); ax.set_ylabel('Value Loss')
ax.set_title('Value Loss (smoothed)')
ax.legend(fontsize=7, ncol=2); _clean(ax)

plt.tight_layout()
plot_file = os.path.join(RESULTS_DIR, 'secondorder_sdp_rl_comparison.png')
plt.savefig(plot_file, dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Main comparison plot saved to {plot_file}")

# %% [markdown]
# ## 10. SDP Ablation: Δ(metric) = SDP − noSDP

# %%
OPT_PAIRS = [
    ('adam_sdp',       'adam_nosdp',       'Adam',       '#607D8B'),
    ('adahessian_sdp', 'adahessian_nosdp', 'AdaHessian', '#E91E63'),
    ('sophia_sdp',     'sophia_nosdp',     'SophiaH',    '#9C27B0'),
    ('shampoo_sdp',    'shampoo_nosdp',    'Shampoo',    '#FF9800'),
    ('asam_sdp',       'asam_nosdp',       'ASAM',       '#4CAF50'),
    ('sassha_sdp',     'sassha_nosdp',     'SASSHA',     '#2196F3'),
]

fig_ab, axes_ab = plt.subplots(1, 3, figsize=(21, 5))
fig_ab.suptitle('SDP Ablation: Δ = (with SDP) − (without SDP) — RL (PPO)',
                fontsize=13, fontweight='bold')

for sdp_key, nosdp_key, label, color in OPT_PAIRS:
    if sdp_key not in all_results or nosdp_key not in all_results:
        continue
    d_sdp = all_results[sdp_key]
    d_no  = all_results[nosdp_key]

    # Δ Episodic Return (last N episodes aligned)
    n = min(len(d_sdp['episodic_returns']), len(d_no['episodic_returns']))
    if n > 100:
        sm_sdp = smooth_returns(d_sdp['episodic_returns'][:n], 100)
        sm_no  = smooth_returns(d_no['episodic_returns'][:n], 100)
        n2 = min(len(sm_sdp), len(sm_no))
        delta_ret = sm_sdp[:n2] - sm_no[:n2]
        axes_ab[0].plot(np.arange(len(delta_ret)), delta_ret, color=color,
                        lw=2, label=label, alpha=0.8)

    # Δ Stable Rank
    n_sr = min(len(d_sdp['stable_rank']), len(d_no['stable_rank']))
    if n_sr > 0:
        delta_sr = np.array(d_sdp['stable_rank'][:n_sr]) - np.array(d_no['stable_rank'][:n_sr])
        axes_ab[1].plot(np.arange(len(delta_sr)), delta_sr, color=color,
                        lw=2, label=label, alpha=0.8)

    # Δ Dormant Fraction
    n_d = min(len(d_sdp['dormant_units']), len(d_no['dormant_units']))
    if n_d > 0:
        delta_dorm = (np.array(d_sdp['dormant_units'][:n_d])
                      - np.array(d_no['dormant_units'][:n_d])) * 100
        axes_ab[2].plot(np.arange(len(delta_dorm)), delta_dorm, color=color,
                        lw=2, label=label, alpha=0.8)

for i, (title, ylabel) in enumerate([
    ('Δ Episodic Return', 'Δ Return'),
    ('Δ Stable Rank', 'Δ SR'),
    ('Δ Dormant Fraction', 'Δ Dormant (%)'),
]):
    axes_ab[i].set_title(title); axes_ab[i].set_xlabel('Index')
    axes_ab[i].set_ylabel(ylabel)
    axes_ab[i].axhline(0, color='black', ls=':', lw=0.8, alpha=0.5)
    axes_ab[i].legend(fontsize=9); _clean(axes_ab[i])

plt.tight_layout()
plot_abl = os.path.join(RESULTS_DIR, 'sdp_ablation_rl.png')
plt.savefig(plot_abl, dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ SDP ablation plot saved to {plot_abl}")

# %% [markdown]
# ## 11. Summary Table

# %%
n_final = 100
print(f"\n{'='*90}")
print(f"  Second-Order Optimizers ± SDP — PPO on {ENV_NAME} — Final {n_final}-episode average")
print(f"{'='*90}")
header = (f"{'Method':<22} {'AvgReturn':>10} {'Dormant%':>10} "
          f"{'StableRank':>11} {'AvgW':>9} {'Episodes':>9}")
print(header)
print(f"{'─'*90}")

for method in METHODS_TO_RUN:
    if method not in all_results:
        continue
    d = all_results[method]
    s = _style(method)
    rets = d['episodic_returns']
    avg_ret = np.mean(rets[-n_final:]) if len(rets) >= n_final else np.mean(rets) if rets else 0
    dormant = (d['dormant_units'][-1] * 100) if d['dormant_units'] else 0
    sr = d['stable_rank'][-1] if d['stable_rank'] else 0
    wm = d['weight_magnitude'][-1] if d['weight_magnitude'] else 0
    n_eps = len(rets)
    print(f"  {s['label']:<20} {avg_ret:>10.1f} {dormant:>10.2f} "
          f"{sr:>11.1f} {wm:>9.4f} {n_eps:>9d}")

print(f"{'='*90}")
