# %% [markdown]
# # SASSHA vs CBP on Continual ImageNet-32
# #
# Template from `grandient-explosion-fixed.py` adapted for ImageNet-32.
# **SASSHA** + **EMA** + **Hessian Clipping** + **Soft Rescaling**
# vs **CBP** (Continual Backpropagation) baseline.
# #
# **Benchmark**: Task-incremental binary classification, 2 classes/task,
# 1000 classes ImageNet-32 (600 train + 100 test per class).
# Setup: cbp.json — 5000 tasks, 250 epochs/task.

# %% [markdown]
# ## 1. Imports & Setup

# %%
import os, sys, json, time, pickle, copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

_LOP_ROOT = "/kaggle/input/datasets/mlinh776/lop-src"
_LOP_IMAGENET_DIR = os.path.join(_LOP_ROOT, 'lop', 'imagenet')
if _LOP_ROOT not in sys.path:
    sys.path.insert(0, _LOP_ROOT)

from lop.nets.conv_net import ConvNet
from lop.algos.bp import Backprop
from lop.algos.convCBP import ConvCBP
from lop.utils.miscellaneous import nll_accuracy as accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Metrics

# %%
NUM_LAYERS = 5  # conv1, conv2, conv3(flat), fc1, fc2
LAYER_SIZES = [32, 64, 512, 128, 128]
TOTAL_NEURONS = sum(LAYER_SIZES)

@torch.no_grad()
def compute_avg_weight_magnitude(net):
    n, s = 0, 0.0
    for p in net.parameters():
        n += p.numel()
        s += torch.sum(torch.abs(p)).item()
    return s / n if n > 0 else 0.0

@torch.no_grad()
def compute_dormant_neurons_enhanced(net, x_data, mini_batch_size=100, threshold=0.01):
    """Per-layer dormant fraction + per-neuron alive scores + last-layer activations."""
    batch_x = x_data[:min(mini_batch_size * 5, len(x_data))]
    _, activations = net.predict(x=batch_x)
    per_layer_frac, all_alive, total_n, total_d, last_act = [], [], 0, 0, None
    for i, act in enumerate(activations):
        if act.ndim == 4:
            alive_frac = (act != 0).float().mean(dim=(0, 2, 3)); n_units = act.shape[1]
        else:
            alive_frac = (act != 0).float().mean(dim=0); n_units = act.shape[1]
        dormant = (alive_frac < threshold).sum().item()
        per_layer_frac.append(dormant / n_units if n_units > 0 else 0.0)
        all_alive.append(alive_frac.cpu().numpy())
        total_d += dormant; total_n += n_units
        if i == len(activations) - 1: last_act = act.cpu().numpy()
    return (total_d / total_n if total_n > 0 else 0.0), per_layer_frac, np.concatenate(all_alive), last_act

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
    except: return 0

print("✓ Metrics defined")

# %% [markdown]
# ## 3. ImageNet-32 Data Loading

# %%
TRAIN_IMAGES_PER_CLASS = 600
TEST_IMAGES_PER_CLASS = 100
IMAGES_PER_CLASS = TRAIN_IMAGES_PER_CLASS + TEST_IMAGES_PER_CLASS
TOTAL_CLASSES = 1000

DATA_DIR = '/kaggle/input/datasets/nguyenlamphuquy/imagenet/classes'
print(f"✓ ImageNet-32 data dir: {DATA_DIR}")

_class_order_file = os.path.join(_LOP_IMAGENET_DIR, 'class_order')
if os.path.isfile(_class_order_file):
    with open(_class_order_file, 'rb') as f:
        _ALL_CLASS_ORDERS = pickle.load(f)
    print(f"  ✓ Loaded class_order ({len(_ALL_CLASS_ORDERS)} runs)")
else:
    print(f"  ⚠ class_order not found, generating random order")
    _rng = np.random.RandomState(42)
    _ALL_CLASS_ORDERS = [_rng.permutation(TOTAL_CLASSES) for _ in range(30)]

def load_imagenet(classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        new_x = np.load(os.path.join(DATA_DIR, str(_class) + '.npy'))
        x_train.append(new_x[:TRAIN_IMAGES_PER_CLASS])
        x_test.append(new_x[TRAIN_IMAGES_PER_CLASS:])
        y_train.append(np.array([idx] * TRAIN_IMAGES_PER_CLASS))
        y_test.append(np.array([idx] * TEST_IMAGES_PER_CLASS))
    return (torch.tensor(np.concatenate(x_train)), torch.from_numpy(np.concatenate(y_train)),
            torch.tensor(np.concatenate(x_test)), torch.from_numpy(np.concatenate(y_test)))

def save_data(data, data_file):
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    with open(data_file, 'wb+') as f: pickle.dump(data, f)

print(f"✓ Data loading ready ({TOTAL_CLASSES} classes)")

# %% [markdown]
# ## 4. SASSHA Optimizer
# #
# From `grandient-explosion-fixed.py` — includes `compute_hessian` flag for guard.

# %%
class SchedulerBase:
    def __init__(self, T_max, max_value, min_value=0.0, init_value=0.0, warmup_steps=0, optimizer=None):
        super().__init__()
        self.t, self.min_value, self.max_value = 0, min_value, max_value
        self.init_value, self.warmup_steps, self.total_steps = init_value, warmup_steps, T_max
        self._last_lr = [init_value]
        self.optimizer = optimizer
    def step(self):
        if self.t < self.warmup_steps:
            value = self.init_value + (self.max_value - self.init_value) * self.t / self.warmup_steps
        elif self.t == self.warmup_steps:
            value = self.min_value
        else:
            value = self.step_func()
        self.t += 1
        if self.optimizer is not None:
            for pg in self.optimizer.param_groups: pg['lr'] = value
        self._last_lr = [value]
        return value
    def step_func(self): pass
    def lr(self): return self._last_lr[0]

class CosineScheduler(SchedulerBase):
    def step_func(self):
        phase = (self.t - self.warmup_steps) / (self.total_steps - self.warmup_steps) * math.pi
        return self.max_value - (self.max_value - self.min_value) * (np.cos(phase) + 1.) / 2.0

class ConstantScheduler(SchedulerBase):
    def step_func(self): return self.min_value

class LinearScheduler(SchedulerBase):
    def step_func(self):
        return self.min_value + (self.max_value - self.min_value) * \
               (self.t - self.warmup_steps) / (self.total_steps - self.warmup_steps)

def build_hessian_power_scheduler(schedule_type, total_steps, max_hessian_power=1.0, min_hessian_power=0.5):
    cls = {'constant': ConstantScheduler, 'linear': LinearScheduler, 'cosine': CosineScheduler}
    if schedule_type not in cls: raise ValueError(f"Unknown schedule: {schedule_type}")
    return cls[schedule_type](T_max=total_steps, max_value=max_hessian_power,
                              min_value=min_hessian_power, init_value=min_hessian_power)

class SASSHA(Optimizer):
    """SASSHA optimizer — from grandient-explosion-fixed.py with compute_hessian flag."""
    def __init__(self, params, hessian_power_scheduler=None, lr=0.15, betas=(0.9, 0.999),
                 weight_decay=0.0, rho=0.0, lazy_hessian=10, n_samples=1,
                 perturb_eps=1e-12, eps=1e-4, adaptive=False, seed=0, **kwargs):
        if not 0.0 <= lr: raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta1: {betas[1]}")
        self.hessian_power_scheduler = hessian_power_scheduler
        self.lazy_hessian, self.n_samples, self.adaptive, self.seed = lazy_hessian, n_samples, adaptive, seed
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, rho=rho, perturb_eps=perturb_eps, eps=eps)
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
    def update_hessian_power(self):
        if self.hessian_power_scheduler is not None:
            self.hessian_power_t = self.hessian_power_scheduler.step()
        else: self.hessian_power_t = None
        return self.hessian_power_t
    @torch.no_grad()
    def set_hessian(self):
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.lazy_hessian == 0: params.append(p)
            self.state[p]["hessian step"] += 1
        if len(params) == 0: return
        if self.generator.device != params[0].device:
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)
        grads = [p.grad for p in params]
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params): p.hess += h_z * z / self.n_samples
    @torch.no_grad()
    def perturb_weights(self, zero_grad=True):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["perturb_eps"])
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                if self.adaptive: e_w *= torch.pow(p, 2)
                p.add_(e_w); self.state[p]['e_w'] = e_w
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys(): p.data.sub_(self.state[p]['e_w'])
    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        if not by:
            return torch.norm(torch.stack([
                ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)
        return torch.norm(torch.stack([
            ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
            for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)
    @torch.no_grad()
    def step(self, closure=None, compute_hessian=True):
        self.update_hessian_power()
        loss = None
        if closure is not None: loss = closure()
        if compute_hessian: self.zero_hessian(); self.set_hessian()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None: continue
                if isinstance(p.hess, (int, float)): p.hess = torch.zeros_like(p.data)
                else: p.hess = p.hess.abs().clone()
                p.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 2:
                    state['step'] = 0; state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag'] = torch.zeros_like(p.data); state['bias_correction2'] = 0
                exp_avg, exp_hessian_diag = state['exp_avg'], state['exp_hessian_diag']
                beta1, beta2 = group['betas']; state['step'] += 1
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                bias_correction1 = 1 - beta1 ** state['step']
                if (state['hessian step']-1) % self.lazy_hessian == 0:
                    exp_hessian_diag.mul_(beta2).add_(p.hess, alpha=1 - beta2)
                    state['bias_correction2'] = (1 - beta2 ** state['step']) ** self.hessian_power_t
                denom = ((exp_hessian_diag**self.hessian_power_t) / state['bias_correction2']).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-group['lr'] / bias_correction1)
        return loss

def _disable_running_stats(model):
    def _disable(m):
        if isinstance(m, nn.BatchNorm2d): m.backup_momentum = m.momentum; m.momentum = 0
    model.apply(_disable)
def _enable_running_stats(model):
    def _enable(m):
        if isinstance(m, nn.BatchNorm2d) and hasattr(m, 'backup_momentum'): m.momentum = m.backup_momentum
    model.apply(_enable)

print("✓ SASSHA optimizer defined")

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
# ## 6. Gradient Explosion Guard (Hessian Clipping)
# #
# From `grandient-explosion-fixed.py` — hessian_floor fix prevents near-zero curvature
# from amplifying updates ~10,000×.

# %%
class GradientExplosionGuard:
    def __init__(self, hessian_clip_value=1e3, hessian_floor=1e-4,
                 enable_hessian_clip=True, max_grad_norm=float('inf')):
        self.hessian_clip_value = hessian_clip_value
        self.hessian_floor = hessian_floor
        self.enable_hessian_clip = enable_hessian_clip
        self.max_grad_norm = max_grad_norm
        self.hessian_clip_count = 0
        self.hessian_floor_count = 0
    @torch.no_grad()
    def clip_hessian(self, optimizer):
        if not self.enable_hessian_clip: return
        tau, floor = self.hessian_clip_value, self.hessian_floor
        upper_clipped, lower_clipped = False, False
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'exp_hessian_diag' in state:
                    h = state['exp_hessian_diag']
                    if h.max().item() > tau: upper_clipped = True
                    if h.min().item() < floor: lower_clipped = True
                    h.clamp_(floor, tau)
                if hasattr(p, 'hess') and not isinstance(p.hess, float):
                    p.hess.clamp_(-tau, tau)
        if upper_clipped: self.hessian_clip_count += 1
        if lower_clipped: self.hessian_floor_count += 1
    @torch.no_grad()
    def clip_global_grad_norm(self, model):
        if math.isfinite(self.max_grad_norm):
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
    def summary(self):
        return f"hess_clips={self.hessian_clip_count}, hess_floors={self.hessian_floor_count}"

print("✓ GradientExplosionGuard defined (hessian clipping with floor)")

# PLACEHOLDER_FOR_REMAINING_CODE