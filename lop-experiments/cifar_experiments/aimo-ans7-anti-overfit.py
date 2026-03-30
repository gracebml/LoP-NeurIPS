# %% [markdown]
# # Anti-Overfitting Methods for Second-Order Continual Learning
#
# Compares **Shampoo (no CL)** against standalone anti-overfitting optimizers
# on Incremental CIFAR-100 with ResNet-18 (same protocol as CBP paper).
#
# **Methods** (each is a separate optimizer — NOT combined with Shampoo):
#
# 1. **Shampoo** — Kronecker-factored second-order optimizer (baseline)
# 2. **SASSHA** — Sharpness-Aware Second-Order (diagonal Hessian + SAM) [ICML 2025]
#    Faithful implementation from: https://github.com/LOG-postech/Sassha
# 3. **Shampoo + SWA** — Shampoo with Stochastic Weight Averaging
# 4. **Shampoo + EMA** — Shampoo with Exponential Moving Average of weights
# 5. **SASSHA + EMA** — Best second-order + best post-hoc averaging

# %% [markdown]
# ## 1. Imports and Setup

# %%
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# %%
pip install mlproj-manager==0.0.29

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

sys.path.append("/kaggle/input/datasets/bngtbnh04/lop-src")
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
# ## 4. Optimizer Definitions
#
# ### 4a. Shampoo (Kronecker-factored, from LOG-postech/Sassha repo)
# ### 4b. SASSHA (Diagonal Hessian + SAM, faithful to original source)

# %%
# ═══════════════════════════════════════════════════
# 4a. Shampoo — Kronecker-factored second-order optimizer
# Faithful to Algorithm 2 from https://arxiv.org/abs/1802.09568 (Gupta et al., 2018)
# ═══════════════════════════════════════════════════

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    device = matrix.device
    matrix = matrix.cpu().float()
    if not matrix.isfinite().all():
        matrix = torch.where(matrix.isfinite(), matrix, torch.zeros_like(matrix))
    try:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except Exception:
        return torch.eye(matrix.shape[0], device=device)
    s = s.clamp(min=1e-16)
    return (u @ s.pow_(power).diag() @ vh).to(device)


class Shampoo(Optimizer):
    r"""Shampoo: Preconditioned Stochastic Tensor Optimization (Gupta et al., 2018).

    Faithful implementation of Algorithm 2 from the original paper.
    Differences from the jettify / LOG-postech approximate implementation:
      1. Preconditioners updated from RAW gradient (not already-preconditioned)
      2. Power = -1/(2k) where k = tensor order (paper), not -1/k (jettify)
      3. Momentum applied AFTER preconditioning (standard heavy-ball)
    """

    def __init__(self, params, lr=1e-1, momentum=0.0, weight_decay=0.0,
                 epsilon=1e-4, update_freq=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        epsilon=epsilon, update_freq=update_freq)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    for dim_id, dim in enumerate(grad.size()):
                        state[f"precond_{dim_id}"] = (
                            group["epsilon"] * torch.eye(dim, out=grad.new(dim, dim)))
                        state[f"inv_precond_{dim_id}"] = grad.new(dim, dim).zero_()

                if weight_decay > 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                if not grad.isfinite().all():
                    state["step"] += 1
                    continue

                # Phase 1: update ALL preconditioners from RAW gradient
                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f"precond_{dim_id}"]
                    g = grad.transpose(0, dim_id).contiguous().view(dim, -1)
                    precond.add_(g @ g.t())

                if state["step"] % group["update_freq"] == 0:
                    for dim_id in range(order):
                        state[f"inv_precond_{dim_id}"].copy_(
                            _matrix_power(state[f"precond_{dim_id}"],
                                          -1.0 / (2 * order)))

                # Phase 2: apply preconditioners via mode-k tensor products
                precond_grad = grad
                for dim_id, dim in enumerate(grad.size()):
                    inv_precond = state[f"inv_precond_{dim_id}"]
                    precond_grad = precond_grad.transpose(0, dim_id).contiguous()
                    pshape = precond_grad.size()
                    precond_grad = precond_grad.view(dim, -1)
                    precond_grad = inv_precond @ precond_grad
                    precond_grad = precond_grad.view(pshape)
                    precond_grad = precond_grad.transpose(0, dim_id).contiguous()

                if momentum > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(precond_grad)
                    precond_grad = buf

                state["step"] += 1
                p.data.add_(precond_grad, alpha=-group["lr"])

        return loss


# ═══════════════════════════════════════════════════
# 4b. SASSHA — Sharpness-Aware Adaptive Second-Order Optimization
#      with Stable Hessian Approximation
# Source: https://github.com/LOG-postech/Sassha/blob/master/optimizers/sassha.py
# Paper: Shin et al., ICML 2025 (arXiv:2502.18153)
#
# Key differences from Shampoo:
#   - Uses DIAGONAL Hessian (Hutchinson trace) not Kronecker factors
#   - Includes SAM perturbation (sharpness-aware ascent step)
#   - Update rule: Adam-like with Hessian diagonal as second moment
#   - lazy_hessian: recompute Hessian every N steps
#   - hessian_power: controls curvature strength (scheduled)
# ═══════════════════════════════════════════════════

class SASSHA(Optimizer):
    """SASSHA optimizer — faithful to official LOG-postech implementation.

    Training loop (caller must follow this protocol):
        loss = loss_fn(model(input))
        loss.backward()
        optimizer.perturb_weights(zero_grad=True)
        loss_fn(model(input)).backward(create_graph=True)
        optimizer.unperturb()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), weight_decay=0.0,
                 rho=0.2, lazy_hessian=10, n_samples=1,
                 perturb_eps=1e-12, eps=1e-4, hessian_power=0.5,
                 adaptive=False, seed=0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        rho=rho, perturb_eps=perturb_eps, eps=eps)
        super().__init__(params, defaults)

        self.lazy_hessian = lazy_hessian
        self.n_samples = n_samples
        self.adaptive = adaptive
        self.seed = seed
        self.hessian_power_t = hessian_power

        for p in self._get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

        self.generator = torch.Generator().manual_seed(self.seed)

    def _get_params(self):
        return (p for group in self.param_groups
                for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        for p in self._get_params():
            if (not isinstance(p.hess, float)
                    and self.state[p]["hessian step"] % self.lazy_hessian == 0):
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """Hutchinson trace estimator: Hessian diagonal ≈ E[z ⊙ (Hz)]."""
        params = []
        for p in filter(lambda p: p.grad is not None, self._get_params()):
            if self.state[p]["hessian step"] % self.lazy_hessian == 0:
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)

        grads = [p.grad for p in params]

        last_sample = self.n_samples - 1
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator,
                                device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(
                grads, params, grad_outputs=zs,
                only_inputs=True, retain_graph=i < last_sample)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples

    @torch.no_grad()
    def perturb_weights(self, zero_grad=True):
        """SAM ascent step: w → w + ρ · ∇L / ‖∇L‖."""
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["perturb_eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def unperturb(self):
        """Restore weights: w + e(w) → w."""
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p]:
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive=False):
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

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
                    state['bias_correction2'] = bias_correction2 ** self.hessian_power_t

                step_size = group['lr'] / bias_correction1
                denom = ((exp_hessian_diag ** self.hessian_power_t)
                         / state['bias_correction2']).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

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


print("✓ Shampoo, SASSHA optimizers defined")

# %% [markdown]
# ## 5. Post-hoc Methods: SWA and EMA
#
# These are model-averaging wrappers — they work with ANY optimizer.

# %%

class SWAWrapper:
    """Stochastic Weight Averaging — accumulates weight average during
    the later phase of each task period.

    Uses equal-weight running average (Izmailov et al., UAI 2018).
    """

    def __init__(self, model, swa_start_frac=0.75, swa_freq=5):
        self.swa_start_frac = swa_start_frac
        self.swa_freq = swa_freq
        self._swa_state = None
        self._n_averaged = 0

    def reset(self):
        self._swa_state = None
        self._n_averaged = 0

    @torch.no_grad()
    def maybe_update(self, model, epoch_in_task, task_length):
        start_epoch = int(self.swa_start_frac * task_length)
        if epoch_in_task < start_epoch:
            return False
        if (epoch_in_task - start_epoch) % self.swa_freq != 0:
            return False

        if self._swa_state is None:
            self._swa_state = {n: p.data.clone() for n, p in model.named_parameters()}
            self._n_averaged = 1
        else:
            self._n_averaged += 1
            for n, p in model.named_parameters():
                self._swa_state[n].add_(
                    (p.data - self._swa_state[n]) / self._n_averaged)
        return True

    @torch.no_grad()
    def apply_swa(self, model):
        if self._swa_state is None:
            return False
        for n, p in model.named_parameters():
            if n in self._swa_state:
                p.data.copy_(self._swa_state[n])
        return True

    @torch.no_grad()
    def update_bn(self, model, data_loader, device, max_batches=50):
        """Re-estimate BatchNorm stats after loading SWA weights."""
        model.train()
        for _, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.zero_()
                m.running_var.fill_(1)
                m.num_batches_tracked.zero_()
        with torch.no_grad():
            for i, sample in enumerate(data_loader):
                if i >= max_batches:
                    break
                model(sample["image"].to(device))


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


print("✓ SWA, EMA wrappers defined")

# %% [markdown]
# ## 6. Ablation Configs
#
# Each method is a **standalone optimizer** (not plugged into Shampoo).
# SWA/EMA are post-hoc wrappers applied on top.

# %%
NUM_CLASSES = 100
SEED = 42

_SHARED = dict(
    num_epochs=1000, batch_size=90, class_increase_frequency=200,
    use_early_stopping=True,
    use_swa=False, swa_start_frac=0.75, swa_freq=5,
    use_ema=False, ema_decay=0.999,
)

CONFIGS = {
    'shampoo': {
        **_SHARED,
        'optimizer': 'shampoo',
        'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
        'epsilon': 1e-4, 'update_freq': 10,
    },
    'sassha': {
        **_SHARED,
        'optimizer': 'sassha',
        'batch_size': 90,                # paper: 256
        'lr': 0.15,                       # paper search: {0.3, 0.15, 0.03, 0.015}
        'betas': (0.9, 0.999),            # paper: β1=0.9, β2=0.999
        'weight_decay': 1e-4,             # paper search: {1e-3 .. 1e-6}
        'rho': 0.2,                       # paper search: {0.1, 0.15, 0.2, 0.25}
        'lazy_hessian': 10,               # paper: k=10
        'n_samples': 1,
        'eps': 1e-4, 'hessian_power': 0.5,
        'lr_milestones': [60, 120, 160],  # paper: multi-step decay per task
        'lr_gamma': 0.2,                  # paper: factor 0.2
    },
    'shampoo_swa': {
        **_SHARED,
        'optimizer': 'shampoo',
        'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
        'epsilon': 1e-4, 'update_freq': 10,
        'use_swa': True, 'swa_start_frac': 0.75, 'swa_freq': 5,
    },
    'shampoo_ema': {
        **_SHARED,
        'optimizer': 'shampoo',
        'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
        'epsilon': 1e-4, 'update_freq': 10,
        'use_ema': True, 'ema_decay': 0.999,
    },
    'sassha_ema': {
        **_SHARED,
        'optimizer': 'sassha',
        'batch_size': 90,
        'lr': 0.15, 'betas': (0.9, 0.999), 'weight_decay': 1e-4,
        'rho': 0.2, 'lazy_hessian': 10, 'n_samples': 1,
        'eps': 1e-4, 'hessian_power': 0.5,
        'lr_milestones': [60, 120, 160], 'lr_gamma': 0.2,
        'use_ema': True, 'ema_decay': 0.999,
    },
}

METHODS_TO_RUN = [
    'shampoo',
    'sassha',
    'shampoo_swa',
    'shampoo_ema',
    'sassha_ema',
]

print(f"✓ Will run: {METHODS_TO_RUN}")

# %% [markdown]
# ## 7. Build Optimizer Helper

# %%
def build_optimizer(config, model):
    opt_type = config['optimizer']

    if opt_type == 'shampoo':
        return Shampoo(
            model.parameters(),
            lr=config['lr'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 1e-4),
            epsilon=config.get('epsilon', 1e-4),
            update_freq=config.get('update_freq', 10),
        )

    if opt_type == 'sassha':
        return SASSHA(
            model.parameters(),
            lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 1e-4),
            rho=config.get('rho', 0.2),
            lazy_hessian=config.get('lazy_hessian', 10),
            n_samples=config.get('n_samples', 1),
            eps=config.get('eps', 1e-4),
            hessian_power=config.get('hessian_power', 0.5),
            seed=SEED,
        )

    raise ValueError(f"Unknown optimizer: {opt_type}")


print("✓ build_optimizer defined")

# %% [markdown]
# ## 8. Training Loop

# %%
def run_method(method_name, config):
    print(f"\n{'='*70}")
    print(f"  Running: {method_name}")
    print(f"  Optimizer: {config['optimizer']}  lr={config['lr']}")
    print(f"  SWA={config['use_swa']}  EMA={config['use_ema']}")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    all_classes = np.random.RandomState(SEED).permutation(NUM_CLASSES)

    net = build_resnet18(num_classes=NUM_CLASSES, norm_layer=nn.BatchNorm2d)
    net.apply(kaiming_init_resnet_module)
    net.to(device)

    optimizer = build_optimizer(config, net)
    opt_type = config['optimizer']
    is_sassha = opt_type == 'sassha'

    swa = (SWAWrapper(net, config.get('swa_start_frac', 0.75),
                       config.get('swa_freq', 5))
           if config['use_swa'] else None)
    ema = (EMAWrapper(net, config.get('ema_decay', 0.999))
           if config['use_ema'] else None)

    loss_fn = nn.CrossEntropyLoss()
    current_num_classes = 5
    freq = config['class_increase_frequency']

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

    best_val_acc = 0.0
    best_model_state = None

    metrics = {k: [] for k in [
        'train_loss', 'train_acc', 'val_acc', 'test_acc',
        'dormant_after', 'dormant_before', 'stable_rank',
        'avg_weight_mag', 'epoch_time', 'overfit_gap',
    ]}

    for epoch in range(config['num_epochs']):
        t0 = time.time()
        epoch_in_task = epoch % freq

        # ── Task boundary ──
        if epoch > 0 and epoch % freq == 0 and current_num_classes < NUM_CLASSES:
            if swa is not None and swa._swa_state is not None:
                swa.apply_swa(net)
                swa.update_bn(net, train_loader, device)
                print(f"  → SWA applied ({swa._n_averaged} checkpoints)")
                swa.reset()
            elif config['use_early_stopping'] and best_model_state is not None:
                net.load_state_dict(best_model_state)
                print(f"  → Early stop: loaded best val (acc={best_val_acc:.4f})")
            best_val_acc = 0.0
            best_model_state = None

            if ema is not None:
                ema.reset(net)

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

            if is_sassha:
                # ---- SASSHA two-pass protocol (faithful to original) ----
                _enable_running_stats(net)
                pred = net(img)[:, all_classes[:current_num_classes]]
                loss = loss_fn(pred, tgt)
                loss.backward()
                optimizer.perturb_weights(zero_grad=True)
                _disable_running_stats(net)
                pred_pert = net(img)[:, all_classes[:current_num_classes]]
                loss_pert = loss_fn(pred_pert, tgt)
                loss_pert.backward(create_graph=True)
                optimizer.unperturb()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _enable_running_stats(net)
                rl += loss.item()

            else:
                # ---- Standard single-pass (Shampoo, etc.) ----
                pred = net(img)[:, all_classes[:current_num_classes]]
                loss = loss_fn(pred, tgt)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                rl += loss.item()

            if ema is not None:
                ema.update(net)

            with torch.no_grad():
                pred_eval = net(img)[:, all_classes[:current_num_classes]]
                acc = (pred_eval.argmax(1) == tgt).float().mean()
            ra += acc.item()
            nb += 1

        metrics['train_loss'].append(rl / nb)
        metrics['train_acc'].append(ra / nb)

        if swa is not None:
            swa.maybe_update(net, epoch_in_task, freq)

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

        net.train()

        et = time.time() - t0
        metrics['epoch_time'].append(et)

        if epoch % 50 == 0 or epoch == config['num_epochs'] - 1:
            swa_str = f" SWA_n={swa._n_averaged}" if swa else ""
            print(f"  [{method_name}] E{epoch:4d} | Loss={rl/nb:.4f} "
                  f"TestAcc={metrics['test_acc'][-1]:.4f} "
                  f"Gap={gap:.4f} Dorm={da:.4f}/{db:.4f} "
                  f"AvgW={wm:.4f}{swa_str} {et:.1f}s")

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
    'shampoo':          {'color': '#7f7f7f', 'ls': '--', 'lw': 2.0,
                         'label': 'Shampoo (Baseline)'},
    'sassha':           {'color': '#d62728', 'ls': '-',  'lw': 2.2,
                         'label': 'SASSHA (ICML 2025)'},
    'shampoo_swa':      {'color': '#2ca02c', 'ls': '-',  'lw': 1.8,
                         'label': 'Shampoo + SWA'},
    'shampoo_ema':      {'color': '#1f77b4', 'ls': '-',  'lw': 1.8,
                         'label': 'Shampoo + EMA'},
    'sassha_ema':       {'color': '#ff7f0e', 'ls': '-',  'lw': 2.2,
                         'label': 'SASSHA + EMA'},
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Anti-Overfitting: Shampoo vs SASSHA (+ SWA/EMA) — Incremental CIFAR-100',
             fontsize=14, fontweight='bold', y=0.99)

plot_info = [
    ('test_acc',       'Test Accuracy',                axes[0, 0], 'Accuracy'),
    ('overfit_gap',    'Train-Test Gap (Overfitting)',  axes[0, 1], 'Gap'),
    ('avg_weight_mag', 'Avg Weight Magnitude',          axes[0, 2], 'Magnitude'),
    ('train_loss',     'Train Loss',                    axes[1, 0], 'Loss'),
    ('dormant_after',  'Dormant Units (Next Task)',     axes[1, 1], 'Proportion'),
    ('stable_rank',    'Stable Rank (Next Task)',       axes[1, 2], 'Stable Rank'),
]

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
    for tb in range(200, 1201, 200):
        ax.axvline(x=tb, color='gray', ls=':', alpha=0.25, lw=0.7)

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
