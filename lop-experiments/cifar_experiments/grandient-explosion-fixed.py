# %% [markdown]
# # Gradient Explosion Prevention: SASSHA + EMA + Hessian Clipping
#
# Tests **Hessian Clipping** (NeurIPS 2025) as a defense against gradient explosion
# in SASSHA + EMA on Incremental CIFAR-100 with ResNet-18.
#
# **Methods**:
# 1. **SASSHA + EMA** — Baseline (no guard, explodes at ~epoch 2200)
# 2. **SASSHA + EMA + Hessian Clipping** — Clamps `exp_hessian_diag` to `[floor, τ]`

# %% [markdown]
# ## 1. Imports and Setup

# %%
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# %%
# pip install mlproj-manager==0.0.29

# %%
import os, sys, json, time, pickle, copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

sys.path.append("/kaggle/input/datasets/caothianhtho/lop-src")
from lop.nets.torchvision_modified_resnet import build_resnet18, kaiming_init_resnet_module
from lop.incremental_cifar.post_run_analysis import compute_dormant_units_proportion

from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import (
    ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results_dir = "anti_overfit_results"
os.makedirs(results_dir, exist_ok=True)
ckpt_dir = "/kaggle/input/datasets/caothianhtho/ckpt-sassha-ema-hessclip"

print("✓ Imports done")

# %% [markdown]
# ## 2. Metrics

# %%
@torch.no_grad()
def compute_avg_weight_magnitude(net):
    n, s = 0, 0.0
    for p in net.parameters():
        n += p.numel()
        s += torch.sum(torch.abs(p)).item()
    return s / n if n > 0 else 0.0

def compute_stable_rank(sv):
    if len(sv) == 0:
        return 0
    sorted_sv = np.flip(np.sort(sv))
    cumsum = np.cumsum(sorted_sv) / np.sum(sv)
    return int(np.sum(cumsum < 0.99) + 1)

def compute_stable_rank_from_activations(act):
    from scipy.linalg import svd
    if act.ndim > 2:
        act = act.reshape(act.shape[0], -1)
    if act.shape[0] == 0 or act.shape[1] == 0:
        return 0
    try:
        sv = svd(act, compute_uv=False, lapack_driver="gesvd")
        return compute_stable_rank(sv)
    except:
        return 0

print("✓ Metrics defined")

# %% [markdown]
# ## 3. Load CIFAR-100

# %%
mean = (0.5071, 0.4865, 0.4409)
std = (0.2673, 0.2564, 0.2762)

train_transformations = transforms.Compose([
    ToTensor(swap_color_axis=True), Normalize(mean=mean, std=std),
    RandomHorizontalFlip(p=0.5), RandomCrop(size=32, padding=4, padding_mode="reflect"),
    RandomRotator(degrees=(0, 15))
])
eval_transformations = transforms.Compose([
    ToTensor(swap_color_axis=True), Normalize(mean=mean, std=std)
])

data_path = (lambda p: (os.makedirs(p, exist_ok=True), p)[1])("/kaggle/working/incremental_cifar/data")

train_data_full = CifarDataSet(
    root_dir=data_path, train=True, cifar_type=100,
    device=None, image_normalization="max", label_preprocessing="one-hot", use_torch=True)
test_data = CifarDataSet(
    root_dir=data_path, train=False, cifar_type=100,
    device=None, image_normalization="max", label_preprocessing="one-hot", use_torch=True)

def get_train_val_indices(cifar_data, num_classes=100):
    val_idx = torch.zeros(5000, dtype=torch.int32)
    train_idx = torch.zeros(45000, dtype=torch.int32)
    cv, ct = 0, 0
    for i in range(num_classes):
        ci = torch.argwhere(cifar_data.data["labels"][:, i] == 1).flatten()
        val_idx[cv:cv+50] = ci[:50]
        train_idx[ct:ct+450] = ci[50:]
        cv += 50
        ct += 450
    return train_idx, val_idx

train_indices, val_indices = get_train_val_indices(train_data_full)

def subsample_cifar(indices, cifar_data):
    idx = indices.numpy() if isinstance(indices, torch.Tensor) else indices
    cifar_data.data["data"] = cifar_data.data["data"][idx]
    cifar_data.data["labels"] = cifar_data.data["labels"][idx]
    cifar_data.integer_labels = torch.tensor(cifar_data.integer_labels)[idx].tolist()
    cifar_data.current_data = cifar_data.partition_data()

train_data = copy.deepcopy(train_data_full)
val_data = copy.deepcopy(train_data_full)
subsample_cifar(train_indices, train_data)
subsample_cifar(val_indices, val_data)
train_data.set_transformation(train_transformations)
val_data.set_transformation(eval_transformations)
test_data.set_transformation(eval_transformations)

print(f"✓ CIFAR-100: Train={len(train_data.data['data'])}, "
      f"Val={len(val_data.data['data'])}, Test={len(test_data.data['data'])}")

# %% [markdown]
# ## 4. SASSHA Optimizer
#
# Sharpness-Aware Adaptive Second-Order (diagonal Hessian + SAM).
# Faithful to: https://github.com/LOG-postech/Sassha (ICML 2025)

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


# ═══════════════════════════════════════════════════════════════════════
# SASSHA — exact copy from official LOG-postech source
# Source: Sassha/optimizers/sassha.py
# Paper: Shin et al., ICML 2025 (arXiv:2502.18153)
#
# [LOCAL] tags mark the ONLY differences from official code:
#   1. compute_hessian flag in step() — needed for hessian clipping guard
#   2. isinstance(p.hess, (int, float)) — needed for checkpoint resume
#   3. No distributed sync (_sync_gradients, _sync_hessians) — single-GPU
#   4. No grad_reduce setup in __init__ — single-GPU
# ═══════════════════════════════════════════════════════════════════════

class SASSHA(Optimizer):
    """Implements the Sharpness-Aware Second-Order optimization with Stable Hessian Approximation (SASSHA) algorithm.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        hessian_power_scheduler (None): Update the Hessian power at every training step. Initially, set it to None, and later you can replace it.
        lr (float, optional): Learning rate.
        betas (Tuple[float, float], optional): Coefficients for computing moving averages of gradient and Hessian.
        weight_decay (float, optional): Weight decay (L2 penalty).
        rho (float, optional): Size of the neighborhood for computing the max loss
        lazy_hessian (int, optional): Number of optimization steps to perform before updating the Hessian.
        n_samples (int, optional): Number of samples to draw for the Hutchinson approximation.
        perturb_eps (float, optional): Small value for perturbations in Hessian trace computation.
        eps (float, optional): Term added to the denominator to improve numerical stability.
        adaptive (bool, optional): set this argument to True if you want to use an experimental implementation of element-wise Adaptive SAM. Default is False.
        seed (int, optional): Random seed for reproducibility. Default is 0.
        **kwargs: Additional keyword arguments for compatibility with other optimizers.
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
        self.seed = seed

        defaults = dict(lr=lr,
                        betas=betas,
                        weight_decay=weight_decay,
                        rho=rho,
                        perturb_eps=perturb_eps,
                        eps=eps)

        super(SASSHA, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(self.seed)


    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.lazy_hessian == 0:
                p.hess.zero_()


    @torch.no_grad()
    def update_hessian_power(self):
        """
        Update the Hessian power at every training step.
        """
        if self.hessian_power_scheduler is not None:
            self.hessian_power_t = self.hessian_power_scheduler.step()
        else:
            self.hessian_power_t = None
        return self.hessian_power_t


    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.lazy_hessian == 0:  # compute a new Hessian once per 'lazy hessian' steps
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)

        grads = [p.grad for p in params]

        last_sample = self.n_samples - 1
        for i in range(self.n_samples):
            # Rademacher distribution {-1.0, 1.0}
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < last_sample)
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
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
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
            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) *  p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    @torch.no_grad()
    def step(self, closure=None, compute_hessian=True):  # [LOCAL] compute_hessian flag for guard
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        self.update_hessian_power()

        loss = None
        if closure is not None:
            loss = closure()

        if compute_hessian:  # [LOCAL] guard may compute hessian separately
            self.zero_hessian()
            self.set_hessian()

        # prepare to update parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

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
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['bias_correction2'] = bias_correction2 ** self.hessian_power_t

                step_size = group['lr'] / bias_correction1
                step_size_neg = -step_size

                denom = ((exp_hessian_diag**self.hessian_power_t) / state['bias_correction2']).add_(group['eps'])

                # make update
                p.addcdiv_(exp_avg, denom, value=step_size_neg)

        return loss

def _disable_running_stats(model):
    """Prevent BatchNorm from updating running stats during SAM perturbation pass."""
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)

def _enable_running_stats(model):
    """Restore BatchNorm momentum after SAM perturbation pass."""
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, 'backup_momentum'):
            module.momentum = module.backup_momentum
    model.apply(_enable)


print("✓ SASSHA optimizer defined")

# %% [markdown]
# ## 5. EMA Wrapper
#
# Exponential Moving Average of weights — used for evaluation only.

# %%
class EMAWrapper:
    """Exponential Moving Average of model weights.
    Used ONLY for evaluation; training runs on original weights."""

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
        for n, p in model.named_parameters():
            p.data.copy_(self._shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup.clear()

    @torch.no_grad()
    def reset(self, model):
        self._shadow = {n: p.data.clone() for n, p in model.named_parameters()}


print("✓ EMA wrapper defined")

# %% [markdown]
# ## 6a. Gradient Explosion Guard (6 methods from top-tier venues)
#
# **Problem**: SASSHA + EMA suffers gradient explosion at epoch ~2200+ due to stale Hessian,
# SAM perturbation amplification, and no gradient clipping.
#
# **Methods integrated** (all from NeurIPS / ICML / ICLR 2021-2025):
# 1. **AGC** — Adaptive Gradient Clipping (NFNet, ICML 2021)
# 2. **ZClip** — Z-score anomaly detection for gradient spikes (2025)
# 3. **Hessian Clipping** — Clip Hessian diagonal (NeurIPS 2025)
# 4. **SPAM-style Momentum Reset** — Reset corrupted buffers on spike (ICLR 2025)
# 5. **NaN-Guard Checkpoint Rollback** — Auto-recover from NaN/explosion
# 6. **Adaptive ρ Scheduling** — Reduce SAM perturbation at task boundaries (JMLR 2024)

# %%
# ═══════════════════════════════════════════════════════════════════════
# GradientExplosionGuard — unified defense combining 6 techniques
# ═══════════════════════════════════════════════════════════════════════

class GradientExplosionGuard:
    """Unified gradient explosion prevention combining 6 top-tier methods.

    Root cause of gradient explosion in SASSHA:
      update = exp_avg / (exp_hessian_diag^0.5 / bias_correction + eps)

    When exp_hessian_diag ≈ 0 → denominator ≈ eps = 1e-4 → update is
    amplified ~10,000×. The original code only clipped the UPPER bound
    (h.clamp_(0, tau)) which never triggers because the problem is
    values being too SMALL, not too large.

    """

    def __init__(
        self,
        agc_clip_factor=0.01,           
        zclip_zscore_thresh=3.0,        
        zclip_ema_decay=0.99,           
        hessian_clip_value=1e3,         
        hessian_floor=1e-4,             
        spike_loss_factor=10.0,         
        max_grad_norm=1.0,              
        enable_agc=True,
        enable_zclip=True,
        enable_hessian_clip=True,
        enable_momentum_reset=True,
        enable_nan_guard=True,
    ):
        self.agc_clip_factor = agc_clip_factor
        self.zclip_zscore_thresh = zclip_zscore_thresh
        self.zclip_ema_decay = zclip_ema_decay
        self.hessian_clip_value = hessian_clip_value
        self.hessian_floor = hessian_floor              # ★ FIX
        self.spike_loss_factor = spike_loss_factor
        self.max_grad_norm = max_grad_norm

        self.enable_agc = enable_agc
        self.enable_zclip = enable_zclip
        self.enable_hessian_clip = enable_hessian_clip
        self.enable_momentum_reset = enable_momentum_reset
        self.enable_nan_guard = enable_nan_guard

        # ZClip state: EMA of gradient norm and variance
        self._gnorm_ema = None
        self._gnorm_var_ema = None
        self._loss_ema = None

        # Counters
        self.spike_count = 0
        self.nan_count = 0
        self.agc_clip_count = 0
        self.hessian_clip_count = 0
        self.hessian_floor_count = 0                    # ★ FIX: track floor clips

    # ─── [1] Adaptive Gradient Clipping (AGC, NFNet) ───
    @torch.no_grad()
    def apply_agc(self, model):
        """Per-parameter clipping based on ‖∇W‖/‖W‖ ratio."""
        if not self.enable_agc:
            return
        clipped = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            p_norm = p.data.norm(2).clamp(min=1e-6)
            g_norm = p.grad.data.norm(2)
            max_g_norm = p_norm * self.agc_clip_factor
            if g_norm > max_g_norm:
                p.grad.data.mul_(max_g_norm / g_norm)
                clipped += 1
        self.agc_clip_count += clipped

    # ─── [2] ZClip: z-score based spike detection ───
    def detect_spike_zclip(self, model):
        """Returns True if current gradient norm is a statistical anomaly."""
        if not self.enable_zclip:
            return False
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        gnorm = gnorm.item() if isinstance(gnorm, torch.Tensor) else gnorm

        if self._gnorm_ema is None:
            self._gnorm_ema = gnorm
            self._gnorm_var_ema = 0.0
            return False

        d = self.zclip_ema_decay
        self._gnorm_var_ema = d * self._gnorm_var_ema + (1 - d) * (gnorm - self._gnorm_ema) ** 2
        self._gnorm_ema = d * self._gnorm_ema + (1 - d) * gnorm

        std = max(math.sqrt(self._gnorm_var_ema), 1e-8)
        zscore = (gnorm - self._gnorm_ema) / std
        is_spike = zscore > self.zclip_zscore_thresh
        if is_spike:
            self.spike_count += 1
            clip_val = self._gnorm_ema + self.zclip_zscore_thresh * std
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        return is_spike

    # ─── [3] Hessian Clipping (NeurIPS 2025) — ★ FIXED ───
    @torch.no_grad()
    def clip_hessian(self, optimizer):
        """Clip exp_hessian_diag to [hessian_floor, tau] to prevent both:
          - extreme curvature (upper bound, original behavior)
          - near-zero curvature causing update amplification (lower bound, ★ FIX)

        When exp_hessian_diag ≈ 0, the SASSHA update denominator becomes:
            denom = (0^0.5 / bias_correction) + eps = eps = 1e-4
        This amplifies updates by ~10,000×, causing gradient explosion.

        The fix clamps exp_hessian_diag to a minimum floor value so that:
            denom >= (floor^0.5 / bias_correction) + eps
        which keeps the update magnitude bounded.
        """
        if not self.enable_hessian_clip:
            return
        tau = self.hessian_clip_value
        floor = self.hessian_floor                      # ★ FIX: lower bound
        upper_clipped = False
        lower_clipped = False
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'exp_hessian_diag' in state:
                    h = state['exp_hessian_diag']
                    # ★ FIX: check BOTH bounds
                    if h.max().item() > tau:
                        upper_clipped = True
                    if h.min().item() < floor:
                        lower_clipped = True
                    h.clamp_(floor, tau)                # ★ FIX: was h.clamp_(0, tau)
                if hasattr(p, 'hess') and not isinstance(p.hess, float):
                    p.hess.clamp_(-tau, tau)
        if upper_clipped:
            self.hessian_clip_count += 1
        if lower_clipped:
            self.hessian_floor_count += 1               # ★ FIX: track floor clips

    # ─── [4] SPAM-style momentum reset (ICLR 2025) ───
    @torch.no_grad()
    def maybe_reset_momentum(self, optimizer, force=False):
        """Reset optimizer momentum buffers if spike was detected or forced (task boundary)."""
        if not self.enable_momentum_reset and not force:
            return False
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'exp_hessian_diag' in state:
                    state['exp_hessian_diag'].zero_()
                if 'bias_correction2' in state:
                    state['bias_correction2'] = 0
                if 'step' in state:
                    state['step'] = 0
                if 'hessian step' in state:
                    state['hessian step'] = 0
                if hasattr(p, 'hess'):
                    p.hess = 0.0
        return True

    # ─── [5] NaN-guard: detect NaN/explosion in loss ───
    def check_loss_health(self, loss_value):
        """Returns ('ok', None) or ('nan', reason) or ('spike', reason)."""
        if not self.enable_nan_guard:
            return 'ok', None
        if not math.isfinite(loss_value):
            self.nan_count += 1
            return 'nan', f'loss={loss_value}'
        if self._loss_ema is None:
            self._loss_ema = loss_value
            return 'ok', None
        self._loss_ema = 0.95 * self._loss_ema + 0.05 * min(loss_value, self._loss_ema * 5)
        if loss_value > self.spike_loss_factor * max(self._loss_ema, 0.1):
            return 'spike', f'loss={loss_value:.2f} >> EMA={self._loss_ema:.2f}'
        return 'ok', None

    # ─── [6] Adaptive rho scheduling ───
    @staticmethod
    def compute_adaptive_rho(base_rho, epoch_in_task, warmup_epochs=20):
        """Linearly warm up SAM perturbation radius from 0 to base_rho
        over the first warmup_epochs of each task to avoid amplifying
        instability when the loss landscape has just shifted."""
        if epoch_in_task >= warmup_epochs:
            return base_rho
        return base_rho * (epoch_in_task / warmup_epochs)

    # ─── Global fallback: standard gradient norm clipping ───
    @torch.no_grad()
    def clip_global_grad_norm(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

    def reset_loss_tracking(self):
        """Call at task boundaries to reset loss EMA."""
        self._loss_ema = None

    def summary(self):
        return (f"Guard stats: spikes={self.spike_count}, "
                f"nan={self.nan_count}, agc_clips={self.agc_clip_count}, "
                f"hess_clips={self.hessian_clip_count}, "
                f"hess_floors={self.hessian_floor_count}")  # ★ FIX: show floor count


print("✓ GradientExplosionGuard defined (6 methods) — with hessian_floor FIX")

# %% [markdown]
# ## 6. Configs
#
# SASSHA + EMA baseline vs SASSHA + EMA + Hessian Clipping.

# %%
NUM_CLASSES = 100
SEED = 42
CKPT_EVERY = 50  # save checkpoint to disk every N epochs

_SHARED = dict(
    num_epochs=4000, batch_size=90, class_increase_frequency=200,
    use_early_stopping=True,
    use_ema=False, ema_decay=0.999,
)

_SASSHA_BASE = dict(
    optimizer='sassha', batch_size=90,
    lr=0.15, betas=(0.9, 0.999), weight_decay=1e-3,  # WN-FIX: 10× gốc (1e-4), coupled decay ~7%/epoch
    rho=0.2, lazy_hessian=10, n_samples=1,
    eps=1e-5, hessian_power=0.5,
    # Hessian power scheduling (official: 'constant', 'linear', 'cosine')
    # Set to None to use fixed hessian_power, or a schedule type string
    hessian_power_schedule='cosine',
    max_hessian_power=1.0,
    min_hessian_power=0.5,
    lr_milestones=[60, 120, 160], lr_gamma=0.2,
    use_soft_rescale=True,               # toggle soft weight rescaling at task boundaries
    soft_rescale_factor=0.9,            # rescale non-BN weights ×0.9 at each task boundary
)

CONFIGS = {
    # ─── SASSHA + EMA (baseline, no guard — sẽ nổ ở ~E2200) ───
    'sassha_ema': {
        **_SHARED, **_SASSHA_BASE,
        'use_ema': True, 'ema_decay': 0.999,
    },

    # ─── SASSHA + EMA + Hessian Clipping (NeurIPS 2025) — ★ FIXED ───
    'sassha_ema_hessclip': {
        **_SHARED, **_SASSHA_BASE,
        'use_ema': True, 'ema_decay': 0.999,
        'use_guard': True,
        'hessian_clip': 1e3,
        'hessian_floor': 1e-5,             
        'max_grad_norm': float('inf'),
        'rho_warmup_epochs': 0,
        'guard_enable_agc': False,
        'guard_enable_zclip': False,
        'guard_enable_hessian_clip': True,
        'guard_enable_momentum_reset': False,
        'guard_enable_nan_guard': False,
    },
}

METHODS_TO_RUN = [
    #'sassha_ema',             # baseline — sẽ nổ ở ~E2200
    'sassha_ema_hessclip',    # + Hessian Clipping only
]

print(f"✓ Will run: {METHODS_TO_RUN}")

# %% [markdown]
# ## 7. Build Optimizer Helper

# %%
def build_optimizer(config, model):
    # Build hessian power scheduler if configured
    hp_scheduler = None
    hp_schedule_type = config.get('hessian_power_schedule', None)
    if hp_schedule_type is not None:
        # Estimate total optimizer steps: epochs × batches_per_epoch
        n_epochs = config['num_epochs']
        n_train = 45000  # approximate train set size
        batch_size = config['batch_size']
        batches_per_epoch = n_train // batch_size
        total_steps = n_epochs * batches_per_epoch
        hp_scheduler = build_hessian_power_scheduler(
            schedule_type=hp_schedule_type,
            total_steps=total_steps,
            max_hessian_power=config.get('max_hessian_power', 1.0),
            min_hessian_power=config.get('min_hessian_power', 0.5),
        )
        print(f"  Hessian power scheduler: {hp_schedule_type} "
              f"[{config.get('min_hessian_power', 0.5)}, {config.get('max_hessian_power', 1.0)}] "
              f"over {total_steps} steps")

    return SASSHA(
        model.parameters(),
        lr=config['lr'],
        betas=config.get('betas', (0.9, 0.999)),
        weight_decay=config.get('weight_decay', 1e-5),
        rho=config.get('rho', 0.2),
        lazy_hessian=config.get('lazy_hessian', 10),
        n_samples=config.get('n_samples', 1),
        eps=config.get('eps', 1e-4),
        hessian_power=config.get('hessian_power', 0.5),
        hessian_power_scheduler=hp_scheduler,
        seed=SEED,
    )


print("✓ build_optimizer defined")

# %% [markdown]
# ## 8. Training Loop

# %%
def _num_classes_at_epoch(epoch, freq, initial=5, step=5, max_cls=100):
    """Derive current_num_classes BEFORE boundary logic at `epoch` runs.

    Boundaries fire at freq, 2*freq, ... (only when epoch > 0).
    Returns the class count as it was BEFORE any boundary at `epoch` fires.
    """
    if epoch <= 0:
        return initial
    n_fired = (epoch - 1) // freq
    return min(initial + n_fired * step, max_cls)


def _ckpt_path(method_name):
    return os.path.join(results_dir, f"ckpt_{method_name}.pt")


def run_method(method_name, config):
    print(f"\n{'='*70}")
    print(f"  Running: {method_name}")
    print(f"  Optimizer: {config['optimizer']}  lr={config['lr']}")
    print(f"  EMA={config['use_ema']}")
    use_guard = config.get('use_guard', False)
    if use_guard:
        active = []
        if config.get('guard_enable_agc', True): active.append('AGC')
        if config.get('guard_enable_zclip', True): active.append('ZClip')
        if config.get('guard_enable_hessian_clip', True): active.append('HessClip')
        if config.get('guard_enable_momentum_reset', True): active.append('SPAMReset')
        if config.get('guard_enable_nan_guard', True): active.append('NaNGuard')
        if config.get('rho_warmup_epochs', 0) > 0: active.append('RhoWarmup')
        print(f"  Guard: {'+'.join(active) if active else '(none)'}")
        # ★ FIX: log the hessian_floor value
        print(f"  Hessian clip: [{config.get('hessian_floor', 0)}, {config.get('hessian_clip', 'inf')}]")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    all_classes = np.random.RandomState(SEED).permutation(NUM_CLASSES)

    freq = config['class_increase_frequency']

    net = build_resnet18(num_classes=NUM_CLASSES, norm_layer=nn.BatchNorm2d)
    net.apply(kaiming_init_resnet_module)
    net.to(device)

    optimizer = build_optimizer(config, net)

    # ─── Gradient Explosion Guard (per-mechanism toggles from config) ───
    guard = None
    if use_guard:
        guard = GradientExplosionGuard(
            agc_clip_factor=config.get('agc_clip_factor', 0.01),
            zclip_zscore_thresh=config.get('zclip_zscore_thresh', 3.0),
            hessian_clip_value=config.get('hessian_clip', 1e3),
            hessian_floor=config.get('hessian_floor', 1e-4),   # ★ FIX: pass floor
            spike_loss_factor=config.get('spike_loss_factor', 10.0),
            max_grad_norm=config.get('max_grad_norm', 1.0),
            enable_agc=config.get('guard_enable_agc', True),
            enable_zclip=config.get('guard_enable_zclip', True),
            enable_hessian_clip=config.get('guard_enable_hessian_clip', True),
            enable_momentum_reset=config.get('guard_enable_momentum_reset', True),
            enable_nan_guard=config.get('guard_enable_nan_guard', True),
        )

    base_rho = config.get('rho', 0.2)
    rho_warmup_epochs = config.get('rho_warmup_epochs', 0)

    ema = (EMAWrapper(net, config.get('ema_decay', 0.999))
           if config['use_ema'] else None)

    loss_fn = nn.CrossEntropyLoss()

    metrics = {k: [] for k in [
        'train_loss', 'train_acc', 'val_acc', 'test_acc',
        'dormant_after', 'dormant_before', 'stable_rank',
        'avg_weight_mag', 'epoch_time', 'overfit_gap',
        'elr',  # ★ WN-FIX: effective learning rate = lr / mean(||w||²)
    ]}

    # ─── Resume from disk checkpoint if available ───
    start_epoch = 0
    best_val_acc = 0.0
    best_model_state = None
    ckpt_file = "/kaggle/input/datasets/caothianhtho/ckpt-sassha-ema-hessclip/ckpt_sassha_ema_hessclip.pt"
    if os.path.isfile(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ema is not None and 'ema_shadow' in ckpt:
            ema._shadow = ckpt['ema_shadow']
        if guard is not None and 'guard_state' in ckpt:
            gs = ckpt['guard_state']
            guard._loss_ema = gs.get('_loss_ema', 0.0)
            guard._gnorm_ema = gs.get('_gnorm_ema', 0.0)
            guard._gnorm_var_ema = gs.get('_gnorm_var_ema', 0.0)
            guard.spike_count = gs.get('spike_count', 0)
            guard.nan_count = gs.get('nan_count', 0)
            guard.agc_clip_count = gs.get('agc_clip_count', 0)
            guard.hessian_clip_count = gs.get('hessian_clip_count', 0)
            guard.hessian_floor_count = gs.get('hessian_floor_count', 0)
        # Restore hessian power scheduler step counter
        if optimizer.hessian_power_scheduler is not None and 'hp_scheduler_t' in ckpt:
            optimizer.hessian_power_scheduler.t = ckpt['hp_scheduler_t']
        # Restore early stopping state
        if 'best_val_acc' in ckpt:
            best_val_acc = ckpt['best_val_acc']
        if 'best_model_state' in ckpt:
            best_model_state = ckpt['best_model_state']
        metrics = ckpt['metrics']
        start_epoch = ckpt['epoch'] + 1
        if 'current_num_classes' in ckpt:
            _saved_num_classes = ckpt['current_num_classes']
        else:
            _saved_num_classes = None
        # After load_state_dict, p.hess (ad-hoc param attribute) is NOT restored.
        # Reset hessian step counters so set_hessian() recomputes on the first step.
        for p in optimizer.get_params():
            p.hess = 0.0
            optimizer.state[p]["hessian step"] = 0
        print(f"  ✓ Resumed from checkpoint epoch {ckpt['epoch']}  ({ckpt_file})")
        del ckpt
        torch.cuda.empty_cache()
    else:
        _saved_num_classes = None
        print(f"  (no checkpoint found, training from scratch)")

    if _saved_num_classes is not None:
        current_num_classes = _saved_num_classes
    else:
        current_num_classes = _num_classes_at_epoch(start_epoch, freq)

    # Data loaders
    train_exp = copy.deepcopy(train_data)
    val_exp = copy.deepcopy(val_data)
    test_exp = copy.deepcopy(test_data)
    train_exp.select_new_partition(all_classes[:current_num_classes])
    val_exp.select_new_partition(all_classes[:current_num_classes])
    test_exp.select_new_partition(all_classes[:current_num_classes])
    train_loader = DataLoader(train_exp, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_exp, batch_size=50, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_exp, batch_size=100, shuffle=False, num_workers=0)

    dormant_next = copy.deepcopy(train_data_full)
    dormant_next.set_transformation(eval_transformations)
    dormant_prev = copy.deepcopy(train_data_full)
    dormant_prev.set_transformation(eval_transformations)
    ns = current_num_classes
    ne = min(ns + 5, NUM_CLASSES)
    if ns < NUM_CLASSES:
        dormant_next.select_new_partition(all_classes[ns:ne])
        dorm_next_loader = DataLoader(dormant_next, batch_size=1000,
                                      shuffle=True, num_workers=0)
    else:
        dorm_next_loader = None
    dorm_prev_loader = None
    if current_num_classes > 5:
        dormant_prev.select_new_partition(all_classes[:current_num_classes])
        dorm_prev_loader = DataLoader(dormant_prev, batch_size=1000,
                                      shuffle=True, num_workers=0)

    safe_checkpoint = None  # [5] NaN-guard rollback checkpoint (in-memory)

    for epoch in range(start_epoch, config['num_epochs']):
        t0 = time.time()
        epoch_in_task = epoch % freq

        # ── Task boundary ──
        if epoch > 0 and epoch % freq == 0 and current_num_classes < NUM_CLASSES:
            if config['use_early_stopping'] and best_model_state is not None:
                net.load_state_dict(best_model_state)
                print(f"  → Early stop: loaded best val (acc={best_val_acc:.4f})")
            best_val_acc = 0.0
            best_model_state = None

            # ★ WN-FIX: Soft weight rescaling at task boundary (before EMA reset)
            # Rescale non-BN parameters to directly reduce accumulated weight norm
            # Preserves weight direction (knowledge) while shrinking magnitude
            rescale = config.get('soft_rescale_factor', 1.0)
            if config.get('use_soft_rescale', False) and rescale < 1.0:
                with torch.no_grad():
                    rescaled_count = 0
                    for name, p in net.named_parameters():
                        # Skip BN params (scale-invariant, rescaling has no effect on output)
                        if 'bn' not in name and 'norm' not in name and 'downsample.1' not in name:
                            p.data.mul_(rescale)
                            rescaled_count += 1
                print(f"  → Soft rescale: ×{rescale} applied to {rescaled_count} param tensors")

            if ema is not None:
                ema.reset(net)

            # [4] SPAM-style: reset optimizer state at task boundary (ICLR 2025)
            if guard is not None:
                if guard.enable_momentum_reset:
                    guard.maybe_reset_momentum(optimizer, force=True)
                    print(f"  → Guard: momentum reset at task boundary")
                guard.reset_loss_tracking()

            current_num_classes = min(current_num_classes + 5, NUM_CLASSES)
            train_exp.select_new_partition(all_classes[:current_num_classes])
            val_exp.select_new_partition(all_classes[:current_num_classes])
            test_exp.select_new_partition(all_classes[:current_num_classes])
            train_loader = DataLoader(train_exp, batch_size=config['batch_size'],
                                      shuffle=True, num_workers=0)
            val_loader = DataLoader(val_exp, batch_size=50, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_exp, batch_size=100, shuffle=False, num_workers=0)

            ns = current_num_classes
            ne = min(ns + 5, NUM_CLASSES)
            if ns < NUM_CLASSES:
                dormant_next.select_new_partition(all_classes[ns:ne])
                dorm_next_loader = DataLoader(dormant_next, batch_size=1000,
                                              shuffle=True, num_workers=0)
            else:
                dorm_next_loader = None
            dormant_prev.select_new_partition(all_classes[:current_num_classes])
            dorm_prev_loader = DataLoader(dormant_prev, batch_size=1000,
                                          shuffle=True, num_workers=0)
            print(f"  Task boundary: epoch {epoch} → {current_num_classes} classes")

        # ── Per-task multi-step LR decay (paper: ×gamma at milestones) ──
        milestones = config.get('lr_milestones', [])
        if milestones:
            gamma = config.get('lr_gamma', 0.1)
            base_lr = config['lr']
            decay = gamma ** sum(epoch_in_task >= m for m in milestones)
            new_lr = base_lr * decay
            for g in optimizer.param_groups:
                g['lr'] = new_lr

        # ── Train ──
        net.train()
        rl, ra, nb = 0.0, 0.0, 0

        for sample in train_loader:
            img = sample["image"].to(device)
            lbl = sample["label"].to(device)
            tgt = lbl.argmax(1) if lbl.dim() > 1 and lbl.shape[1] > 1 else lbl

            # ---- SASSHA two-pass protocol ----
            _enable_running_stats(net)
            pred = net(img)[:, all_classes[:current_num_classes]]
            loss = loss_fn(pred, tgt)

            # [5] NaN-guard: check loss before backward
            if guard is not None:
                status, reason = guard.check_loss_health(loss.item())
                if status == 'nan':
                    print(f"  ⚠ NaN loss detected ({reason}), rolling back...")
                    if safe_checkpoint is not None:
                        net.load_state_dict(safe_checkpoint['model'])
                    guard.maybe_reset_momentum(optimizer, force=True)
                    if ema is not None:
                        ema.reset(net)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                elif status == 'spike':
                    print(f"  ⚠ Loss spike ({reason}), clipping + resetting momentum")
                    guard.maybe_reset_momentum(optimizer, force=True)

            loss.backward()

            # [1] AGC — clip gradients per-param (ICML 2021)
            if guard is not None:
                guard.apply_agc(net)

            # [2] ZClip — detect anomalous gradient spikes (2025)
            if guard is not None:
                guard.detect_spike_zclip(net)

            # [6] Adaptive rho warmup at task boundaries (JMLR 2024)
            if guard is not None and rho_warmup_epochs > 0:
                current_rho = GradientExplosionGuard.compute_adaptive_rho(
                    base_rho, epoch_in_task, rho_warmup_epochs)
                for g in optimizer.param_groups:
                    g['rho'] = current_rho

            optimizer.perturb_weights(zero_grad=True)
            _disable_running_stats(net)
            pred_pert = net(img)[:, all_classes[:current_num_classes]]
            loss_pert = loss_fn(pred_pert, tgt)
            loss_pert.backward(create_graph=True)
            optimizer.unperturb()

            # Global fallback gradient norm clip
            if guard is not None:
                guard.clip_global_grad_norm(net)

            # [3] Hessian clipping (NeurIPS 2025):
            # Compute hessian externally → clip → then step with pre-clipped values
            if guard is not None and guard.enable_hessian_clip:
                optimizer.zero_hessian()
                optimizer.set_hessian()
                guard.clip_hessian(optimizer)
                optimizer.step(compute_hessian=False)
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _enable_running_stats(net)
            rl += loss.item()

            if ema is not None:
                ema.update(net)

            with torch.no_grad():
                pred_eval = net(img)[:, all_classes[:current_num_classes]]
                acc = (pred_eval.argmax(1) == tgt).float().mean()
            ra += acc.item()
            nb += 1

        avg_loss = rl / nb if nb > 0 else float('nan')
        metrics['train_loss'].append(avg_loss)
        metrics['train_acc'].append(ra / nb if nb > 0 else 0)

        # [5] Save safe checkpoint periodically for NaN rollback (in-memory)
        if guard is not None and math.isfinite(avg_loss) and epoch % 10 == 0:
            safe_checkpoint = {'model': copy.deepcopy(net.state_dict()), 'epoch': epoch}

        # ── Eval (use EMA if available) ──
        if ema is not None:
            ema.apply(net)

        net.eval()
        with torch.no_grad():
            va, vb = 0, 0
            for s in val_loader:
                p = net(s["image"].to(device))[:, all_classes[:current_num_classes]]
                t = s["label"].to(device)
                t = t.argmax(1) if t.dim() > 1 and t.shape[1] > 1 else t
                va += (p.argmax(1) == t).float().mean().item()
                vb += 1
            val_acc_epoch = va / vb if vb else 0
            metrics['val_acc'].append(val_acc_epoch)

            ta, tb = 0, 0
            for s in test_loader:
                p = net(s["image"].to(device))[:, all_classes[:current_num_classes]]
                t = s["label"].to(device)
                t = t.argmax(1) if t.dim() > 1 and t.shape[1] > 1 else t
                ta += (p.argmax(1) == t).float().mean().item()
                tb += 1
            metrics['test_acc'].append(ta / tb if tb else 0)

        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            best_model_state = copy.deepcopy(net.state_dict())

        # ── Restore training weights BEFORE measuring training-model metrics ──
        if ema is not None:
            ema.restore(net)

        # ── Dormant measurement (always on training model) ──
        bn_bk = {}
        for nm, m in net.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_bk[nm] = {'rm': m.running_mean.clone(), 'rv': m.running_var.clone(),
                             'nt': m.num_batches_tracked.clone()}

        net.train()
        with torch.no_grad():
            if dorm_next_loader is not None:
                da, last_act = compute_dormant_units_proportion(
                    net, dorm_next_loader, 0.01)
                sr = compute_stable_rank_from_activations(last_act)
            else:
                da, sr = float('nan'), float('nan')
            for nm, m in net.named_modules():
                if isinstance(m, nn.BatchNorm2d) and nm in bn_bk:
                    m.running_mean.copy_(bn_bk[nm]['rm'])
                    m.running_var.copy_(bn_bk[nm]['rv'])
                    m.num_batches_tracked.copy_(bn_bk[nm]['nt'])
            if dorm_prev_loader is not None:
                db = compute_dormant_units_proportion(
                    net, dorm_prev_loader, 0.01)[0]
            else:
                db = float('nan')

        for nm, m in net.named_modules():
            if isinstance(m, nn.BatchNorm2d) and nm in bn_bk:
                m.running_mean.copy_(bn_bk[nm]['rm'])
                m.running_var.copy_(bn_bk[nm]['rv'])
                m.num_batches_tracked.copy_(bn_bk[nm]['nt'])

        metrics['dormant_after'].append(da)
        metrics['dormant_before'].append(db)
        metrics['stable_rank'].append(sr)
        wm = compute_avg_weight_magnitude(net)
        metrics['avg_weight_mag'].append(wm)
        gap = metrics['train_acc'][-1] - metrics['test_acc'][-1]
        metrics['overfit_gap'].append(gap)

        # When weight norm grows, ELR decays implicitly → loss of plasticity
        current_lr = optimizer.param_groups[0]['lr']
        with torch.no_grad():
            total_norm_sq = sum(p.data.norm()**2 for p in net.parameters() if p.requires_grad)
            n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        elr = current_lr / (total_norm_sq.item() / max(n_params, 1))
        metrics['elr'].append(elr)

        net.train()

        et = time.time() - t0
        metrics['epoch_time'].append(et)

        if epoch % 50 == 0 or epoch == config['num_epochs'] - 1:
            guard_str = ""
            if guard is not None and epoch % 200 == 0:
                guard_str = f" | {guard.summary()}"
            print(f"  [{method_name}] E{epoch:4d} | Loss={avg_loss:.4f} "
                  f"TestAcc={metrics['test_acc'][-1]:.4f} "
                  f"Gap={gap:.4f} Dorm={da:.4f}/{db:.4f} "
                  f"AvgW={wm:.4f} ELR={elr:.6f} {et:.1f}s{guard_str}")

        # ── Save disk checkpoint for resume ──
        if epoch % CKPT_EVERY == 0 or epoch == config['num_epochs'] - 1:
            ckpt_data = {
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': metrics,
                'current_num_classes': current_num_classes,
                'best_val_acc': best_val_acc,
                'best_model_state': best_model_state,
            }
            if optimizer.hessian_power_scheduler is not None:
                ckpt_data['hp_scheduler_t'] = optimizer.hessian_power_scheduler.t
            if ema is not None:
                ckpt_data['ema_shadow'] = ema._shadow
            if guard is not None:
                ckpt_data['guard_state'] = {
                    '_loss_ema': guard._loss_ema,
                    '_gnorm_ema': guard._gnorm_ema,
                    '_gnorm_var_ema': guard._gnorm_var_ema,
                    'spike_count': guard.spike_count,
                    'nan_count': guard.nan_count,
                    'agc_clip_count': guard.agc_clip_count,
                    'hessian_clip_count': guard.hessian_clip_count,
                    'hessian_floor_count': guard.hessian_floor_count,
                }
            torch.save(ckpt_data, _ckpt_path(method_name))

    if guard is not None:
        print(f"\n  Final {guard.summary()}")

    return metrics


print("✓ Training loop defined")

# %% [markdown]
# ## 9. Run All Methods

# %%
all_results = {}
for method in METHODS_TO_RUN:
    all_results[method] = run_method(method, CONFIGS[method])
    with open(os.path.join(results_dir, f"anti_overfit_{method}.pkl"), 'wb') as f:
        pickle.dump(all_results[method], f)
    print(f"  ✓ {method} saved.")

with open(os.path.join(results_dir, "anti_overfit_all_results.pkl"), 'wb') as f:
    pickle.dump(all_results, f)
print(f"\n✓ All results saved.")

# %% [markdown]
# ## 10. Visualization

# %%
METHOD_STYLES = {
    'sassha_ema':           {'color': '#ff7f0e', 'ls': '-',  'lw': 2.2,
                             'label': 'SASSHA + EMA (baseline)'},
    'sassha_ema_hessclip':  {'color': '#2ca02c', 'ls': '-',  'lw': 2.5,
                             'label': 'SASSHA + EMA + HessClip'},
}

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle('SASSHA + EMA: Hessian Clipping + Weight Norm Control — Incremental CIFAR-100',
             fontsize=14, fontweight='bold', y=0.99)

plot_info = [
    ('test_acc',       'Test Accuracy',                axes[0, 0], 'Accuracy'),
    ('overfit_gap',    'Train-Test Gap (Overfitting)',  axes[0, 1], 'Gap'),
    ('avg_weight_mag', 'Avg Weight Magnitude',          axes[0, 2], 'Magnitude'),
    ('elr',            'Effective Learning Rate',       axes[0, 3], 'ELR'),
    ('train_loss',     'Train Loss',                    axes[1, 0], 'Loss'),
    ('dormant_after',  'Dormant Units (Next Task)',     axes[1, 1], 'Proportion'),
    ('stable_rank',    'Stable Rank (Next Task)',       axes[1, 2], 'Stable Rank'),
]

# Hide unused subplot (axes[1,3])
axes[1, 3].set_visible(False)

for key, title, ax, ylabel in plot_info:
    for method, data in all_results.items():
        if key not in data or not data[key]:
            continue
        s = METHOD_STYLES.get(method, {'color': 'gray', 'ls': '-', 'lw': 1, 'label': method})
        y = np.array(data[key], dtype=float)
        x = np.arange(len(y))
        ax.plot(x, y, color=s['color'], ls=s['ls'], lw=s['lw'],
                label=s['label'], alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6.5, loc='best', framealpha=0.7, edgecolor='none')
    for tb in range(200, 4001, 200):
        ax.axvline(x=tb, color='gray', ls=':', alpha=0.25, lw=0.7)
    # ★ WN-FIX: use log scale for ELR since values can span orders of magnitude
    if key == 'elr':
        ax.set_yscale('log')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(results_dir, "anti_overfit_comparison.png"),
            dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Saved to {results_dir}/anti_overfit_comparison.png")

# ── Summary table ──
print(f"\n{'='*120}")
print(f"{'Method':<28} {'Optimizer':>10} {'TestAcc↑':>9} {'OvfGap↓':>9} "
      f"{'Dormant↓':>9} {'StbRank↑':>9} {'AvgW':>9} {'Time/ep':>9}")
print(f"{'='*120}")

for method, data in all_results.items():
    s = METHOD_STYLES.get(method, {})
    n = min(50, len(data['test_acc']))
    ta = np.mean(data['test_acc'][-n:])
    og = np.mean(data['overfit_gap'][-n:])
    da = np.nanmean(data['dormant_after'][-n:])
    sr = np.nanmean(data['stable_rank'][-n:])
    wm = np.mean(data['avg_weight_mag'][-n:])
    et = np.mean(data['epoch_time'])
    lbl = s.get('label', method)
    opt = CONFIGS[method]['optimizer']
    print(f"{lbl:<28} {opt:>10} {ta:>9.4f} {og:>9.4f} {da:>9.4f} "
          f"{sr:>9.1f} {wm:>9.4f} {et:>8.1f}s")

print(f"{'='*120}")
