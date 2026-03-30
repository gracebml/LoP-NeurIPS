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
import os, sys, time, pickle, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
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
    batch_x = x_data[:min(mini_batch_size * 5, len(x_data))]
    _, activations = net.predict(x=batch_x)
    per_layer_frac, all_alive, total_n, total_d, last_act = [], [], 0, 0, None
    for i, act in enumerate(activations):
        if act.ndim == 4:
            # Conv: fraction of non-zero activations per channel over batch & spatial
            alive_score = (act != 0).float().mean(dim=(0, 2, 3))   # shape: [C]
            n_units = act.shape[1]
        else:
            # FC: fraction of non-zero activations per neuron over batch
            alive_score = (act != 0).float().mean(dim=0)            # shape: [H]
            n_units = act.shape[1]
        dormant = (alive_score < threshold).sum().item()
        per_layer_frac.append(dormant / n_units if n_units > 0 else 0.0)
        all_alive.append(alive_score.cpu().numpy())
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

def apply_sdp(net, gamma):
    """Spectral Diversity Preservation (SDP) at task boundary.
    For each weight matrix W, performs SVD and rescales singular values:
      σ'_i = σ̄^γ · σ_i^(1-γ)
    This shrinks dominant singular values and boosts tail ones,
    preserving approximate total energy while improving spectral diversity.
    Returns list of condition numbers (before SDP) for monitoring.
    """
    cond_numbers = []
    with torch.no_grad():
        for layer in net.layers:
            if not hasattr(layer, 'weight'):
                continue
            W = layer.weight.data
            orig_shape = W.shape
            # Reshape conv weights to 2D: (out_channels, in_channels*k*k)
            if W.ndim == 4:
                W2d = W.reshape(W.shape[0], -1)
            else:
                W2d = W
            try:
                U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
            except Exception:
                continue
            if S.numel() == 0 or S[0] < 1e-12:
                continue
            # Record condition number before SDP
            cond_numbers.append((S[0] / S[-1].clamp(min=1e-12)).item())
            # Geometric interpolation toward mean
            s_mean = S.mean()
            S_new = (s_mean ** gamma) * (S ** (1.0 - gamma))
            # Reconstruct
            W_new = U @ torch.diag(S_new) @ Vh
            layer.weight.data.copy_(W_new.reshape(orig_shape))
    return cond_numbers

print("✓ Metrics defined (+ SDP)")

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

print("✓ Hessian power schedulers defined")

# %%
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
        last_sample = self.n_samples - 1
        for i in range(self.n_samples):
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
                exp_avg, exp_hessian_diag = state['exp_avg'], state['exp_hessian_diag']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                bias_correction1 = 1 - beta1 ** state['step']
                if (state['hessian step']-1) % self.lazy_hessian == 0:
                    exp_hessian_diag.mul_(beta2).add_(p.hess, alpha=1 - beta2)
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['bias_correction2'] = bias_correction2 ** self.hessian_power_t
                step_size = group['lr'] / bias_correction1
                denom = ((exp_hessian_diag**self.hessian_power_t) / state['bias_correction2']).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

def _disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum; module.momentum = 0
    model.apply(_disable)

def _enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, 'backup_momentum'):
            module.momentum = module.backup_momentum
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
        """Clip exp_hessian_diag to [hessian_floor, tau]."""
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


# %% [markdown]
# ## 7. Configs

# %%
SEED = 42
SAVE_EVERY_N_TASKS = 50
TIME_LIMIT_SECONDS = 11.5 * 3600

_SHARED = dict(
    num_tasks=2000, num_classes=2, num_showings=200,
    mini_batch_size=100,
)

_SASSHA_BASE = dict(
    lr=0.01, betas=(0.9, 0.999), weight_decay=5e-4, rho=0.1,
    lazy_hessian=10, n_samples=1, eps=1e-4,
    hessian_power_schedule='constant', max_hessian_power=1.0, min_hessian_power=0.5,
    use_ema=True, ema_decay=0.999,
    label_smoothing=True,
    sdp_gamma=0.3,
)

CONFIGS = {
    'sassha_ema_hessclip': {
        **_SHARED, **_SASSHA_BASE,
        'use_guard': True,
        'hessian_clip': 1e3,
        'hessian_floor': 1e-5,
        'max_grad_norm': float('inf'),
    },
}

# CBP config (matching cbp.json)

CBP_PARAMS = {
    **_SHARED,
    'cbp_step_size': 0.01, 'cbp_momentum': 0.9, 'cbp_weight_decay': 0,
    'cbp_opt': 'sgd', 'cbp_replacement_rate': 3e-4, 'cbp_decay_rate': 0.99,
    'cbp_util_type': 'adaptable_contribution', 'cbp_maturity_threshold': 100,
}

METHODS_TO_RUN = ['sassha_ema_hessclip']
RUN_CBP = False

RESULTS_DIR = os.path.join('kaggle/working', 'data', 'sassha_imagenet')
CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"✓ Config: {_SHARED['num_tasks']} tasks × {_SHARED['num_showings']} epochs, "
      f"{_SHARED['num_classes']} classes/task")

# %% [markdown]
# ## 8. Build Optimizer

# %%
def build_optimizer(config, model):
    hp_scheduler = None
    hp_schedule_type = config.get('hessian_power_schedule', None)
    if hp_schedule_type is not None:
        n_tasks = config['num_tasks']
        n_epochs = config['num_showings']
        examples_per_epoch = TRAIN_IMAGES_PER_CLASS * config['num_classes']
        batches_per_epoch = examples_per_epoch // config['mini_batch_size']
        total_steps = n_tasks * n_epochs * batches_per_epoch
        hp_scheduler = build_hessian_power_scheduler(
            schedule_type=hp_schedule_type, total_steps=total_steps,
            max_hessian_power=config.get('max_hessian_power', 1.0),
            min_hessian_power=config.get('min_hessian_power', 0.5))
        print(f"  Hessian power scheduler: {hp_schedule_type} over {total_steps} steps")
    return SASSHA(
        model.parameters(), lr=config['lr'],
        betas=config.get('betas', (0.9, 0.999)),
        weight_decay=config.get('weight_decay', 1e-5),
        rho=config.get('rho', 0.2), lazy_hessian=config.get('lazy_hessian', 10),
        n_samples=config.get('n_samples', 1), eps=config.get('eps', 1e-4),
        hessian_power_scheduler=hp_scheduler, seed=SEED)

print("✓ build_optimizer defined")

# %% [markdown]
# ## 9. Training Loop — SASSHA + EMA + Hessian Clipping + CLN

# %%
def run_sassha(method_name, config, run_idx=0):
    """SASSHA + EMA + Hessian Clipping + Soft Rescaling on Continual ImageNet-32."""
    print(f"\n{'='*70}")
    print(f"  {method_name} — Run {run_idx} (ImageNet-32)")
    use_guard = config.get('use_guard', False)
    if use_guard:
        print(f"  Hessian clip: [{config.get('hessian_floor', 0)}, {config.get('hessian_clip', 'inf')}]")
    print(f"{'='*70}")

    wall_clock_start = time.time()
    torch.manual_seed(SEED); torch.cuda.manual_seed(SEED); np.random.seed(SEED)

    num_tasks = config['num_tasks']
    num_epochs = config['num_showings']
    num_classes = config['num_classes']
    mini_batch_size = config['mini_batch_size']
    classes_per_task = num_classes
    examples_per_epoch = TRAIN_IMAGES_PER_CLASS * classes_per_task

    class_order = _ALL_CLASS_ORDERS[run_idx % len(_ALL_CLASS_ORDERS)]
    num_class_repetitions = int(num_classes * num_tasks / TOTAL_CLASSES) + 1
    class_order = np.concatenate([class_order] * num_class_repetitions)

    net = ConvNet(num_classes=classes_per_task)
    optimizer = build_optimizer(config, net)

    guard = None
    if use_guard:
        guard = GradientExplosionGuard(
            hessian_clip_value=config.get('hessian_clip', 1e3),
            hessian_floor=config.get('hessian_floor', 1e-4),
            enable_hessian_clip=True,
            max_grad_norm=config.get('max_grad_norm', float('inf')))

    ema = EMAWrapper(net, config.get('ema_decay', 0.999)) if config.get('use_ema', False) else None
    ls = 0.1 if config.get('label_smoothing', False) else 0.0
    loss_fn = lambda input, target: F.cross_entropy(input, target, label_smoothing=ls)

    sdp_gamma = config.get('sdp_gamma', 0.0)

    if sdp_gamma > 0:
        print(f"  SDP enabled: γ={sdp_gamma}")

    # Metrics
    train_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    test_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    all_weight_mag = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    all_dormant_frac = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    all_dormant_per_layer = torch.zeros((num_tasks, num_epochs, NUM_LAYERS), dtype=torch.float)
    all_stable_rank = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    all_elr = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    dormant_persistence = torch.zeros((num_tasks, TOTAL_NEURONS), dtype=torch.bool)
    all_sdp_cond = torch.zeros((num_tasks,), dtype=torch.float)

    # Checkpoint resume
    ckpt_file = os.path.join(CKPT_DIR, f'ckpt_{method_name}_run{run_idx}.pt')
    start_task = 0
    if os.path.isfile(ckpt_file):
        print(f"  Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        train_accuracies = ckpt['train_accuracies']; test_accuracies = ckpt['test_accuracies']
        all_weight_mag = ckpt.get('all_weight_mag', all_weight_mag)
        all_dormant_frac = ckpt.get('all_dormant_frac', all_dormant_frac)
        all_dormant_per_layer = ckpt.get('all_dormant_per_layer', all_dormant_per_layer)
        all_stable_rank = ckpt.get('all_stable_rank', all_stable_rank)
        all_elr = ckpt.get('all_elr', all_elr)
        dormant_persistence = ckpt.get('dormant_persistence', dormant_persistence)
        all_sdp_cond = ckpt.get('all_sdp_cond', all_sdp_cond)
        start_task = ckpt['task_idx'] + 1
        if ema is not None and 'ema_shadow' in ckpt: ema._shadow = ckpt['ema_shadow']
        if guard is not None and 'guard_state' in ckpt:
            gs = ckpt['guard_state']
            guard.hessian_clip_count = gs.get('hessian_clip_count', 0)
            guard.hessian_floor_count = gs.get('hessian_floor_count', 0)
        if optimizer.hessian_power_scheduler is not None and 'hp_scheduler_t' in ckpt:
            optimizer.hessian_power_scheduler.t = ckpt['hp_scheduler_t']
        if 'np_random_state' in ckpt: np.random.set_state(ckpt['np_random_state'])
        if 'torch_rng_state' in ckpt: torch.random.set_rng_state(ckpt['torch_rng_state'])
        if device.type == 'cuda' and 'torch_cuda_rng_state' in ckpt:
            torch.cuda.set_rng_state(ckpt['torch_cuda_rng_state'])
        for p in optimizer.get_params():
            p.hess = 0.0; optimizer.state[p]["hessian step"] = 0
        print(f"  ✓ Resumed from task {start_task}")
        del ckpt; torch.cuda.empty_cache()
    else:
        print(f"  (no checkpoint, training from scratch)")

    def save_checkpoint(task_idx, reason="periodic"):
        ckpt_data = {
            'task_idx': task_idx, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
            'train_accuracies': train_accuracies, 'test_accuracies': test_accuracies,
            'all_weight_mag': all_weight_mag, 'all_dormant_frac': all_dormant_frac,
            'all_dormant_per_layer': all_dormant_per_layer, 'all_stable_rank': all_stable_rank,
            'all_elr': all_elr, 'dormant_persistence': dormant_persistence,
            'all_sdp_cond': all_sdp_cond,
            'np_random_state': np.random.get_state(), 'torch_rng_state': torch.random.get_rng_state(),
            'params': config,
        }
        if optimizer.hessian_power_scheduler is not None:
            ckpt_data['hp_scheduler_t'] = optimizer.hessian_power_scheduler.t
        if ema is not None: ckpt_data['ema_shadow'] = ema._shadow
        if guard is not None:
            ckpt_data['guard_state'] = {
                'hessian_clip_count': guard.hessian_clip_count,
                'hessian_floor_count': guard.hessian_floor_count}
        if device.type == 'cuda':
            ckpt_data['torch_cuda_rng_state'] = torch.cuda.get_rng_state()
        torch.save(ckpt_data, ckpt_file)
        elapsed = time.time() - wall_clock_start
        print(f"  Checkpoint saved at task {task_idx} ({reason}) [{elapsed/3600:.1f}h elapsed]")

    # Main training loop
    x_train, x_test, y_train, y_test = None, None, None, None
    time_limit_hit = False

    for task_idx in range(start_task, num_tasks):
        task_start = time.time()
        elapsed = time.time() - wall_clock_start
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"\n  Time limit ({elapsed/3600:.1f}h). Saving.")
            save_checkpoint(task_idx - 1, reason="time_limit")
            time_limit_hit = True; break

        # Load data
        del x_train, x_test, y_train, y_test
        task_classes = class_order[task_idx * classes_per_task:(task_idx + 1) * classes_per_task]
        x_train, y_train, x_test, y_test = load_imagenet(task_classes)
        x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
        if device.type == 'cuda':
            x_train, x_test = x_train.to(device), x_test.to(device)
            y_train, y_test = y_train.to(device), y_test.to(device)

        # Head reset
        net.layers[-1].weight.data *= 0
        net.layers[-1].bias.data *= 0

        # SDP (Spectral Diversity Preservation) at task boundary
        if sdp_gamma > 0 and task_idx > 0:
            cond_nums = apply_sdp(net, sdp_gamma)
            avg_cond = sum(cond_nums) / max(len(cond_nums), 1)
            all_sdp_cond[task_idx] = avg_cond
            if task_idx % 50 == 0:
                print(f"    SDP applied: avg condition number = {avg_cond:.1f}")

        # Reset EMA at task boundary
        if ema is not None: ema.reset(net)

        # Train epochs
        net.train()
        for epoch_idx in range(num_epochs):
            example_order = np.random.permutation(TRAIN_IMAGES_PER_CLASS * classes_per_task)
            x_train_shuffled = x_train[example_order]
            y_train_shuffled = y_train[example_order]

            new_train_accs = torch.zeros(examples_per_epoch // mini_batch_size, dtype=torch.float)
            batch_iter = 0
            for start_idx in range(0, examples_per_epoch, mini_batch_size):
                batch_x = x_train_shuffled[start_idx:start_idx + mini_batch_size]
                batch_y = y_train_shuffled[start_idx:start_idx + mini_batch_size]

                # SASSHA two-pass protocol
                _enable_running_stats(net)
                optimizer.zero_grad()

                output, batch_acts = net.predict(x=batch_x)

                loss = loss_fn(output, batch_y)
                loss.backward()

                # SAM perturbation
                optimizer.perturb_weights(zero_grad=True)
                _disable_running_stats(net)
                output_pert = net.predict(x=batch_x)[0]
                loss_pert = loss_fn(output_pert, batch_y)
                loss_pert.backward(create_graph=True)
                optimizer.unperturb()
                _enable_running_stats(net)

                # Grad norm clip
                if guard is not None: guard.clip_global_grad_norm(net)

                # Hessian computation (needed for guard clipping)
                need_explicit_hessian = (guard is not None and guard.enable_hessian_clip)
                if need_explicit_hessian:
                    optimizer.zero_hessian()
                    optimizer.set_hessian()

                # Hessian clipping
                if guard is not None and guard.enable_hessian_clip:
                    guard.clip_hessian(optimizer)

                if need_explicit_hessian:
                    optimizer.step(compute_hessian=False)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if ema is not None: ema.update(net)

                with torch.no_grad():
                    new_train_accs[batch_iter] = accuracy(
                        F.softmax(output.detach(), dim=1), batch_y).cpu()
                    batch_iter += 1

            # Eval
            net.eval()
            with torch.no_grad():
                train_accuracies[task_idx][epoch_idx] = new_train_accs.mean()
                if ema is not None: ema.apply(net)
                new_test_accs = torch.zeros(x_test.shape[0] // mini_batch_size, dtype=torch.float)
                test_iter = 0
                for si in range(0, x_test.shape[0], mini_batch_size):
                    tb_x = x_test[si:si + mini_batch_size]
                    tb_y = y_test[si:si + mini_batch_size]
                    to, _ = net.predict(x=tb_x)
                    new_test_accs[test_iter] = accuracy(F.softmax(to, dim=1), tb_y)
                    test_iter += 1
                test_accuracies[task_idx][epoch_idx] = new_test_accs.mean()
                if ema is not None: ema.restore(net)

            # Per-epoch metrics
            with torch.no_grad():
                wm = compute_avg_weight_magnitude(net)
                all_weight_mag[task_idx, epoch_idx] = wm

                agg_frac, layer_fracs, alive_scores, last_act = compute_dormant_neurons_enhanced(
                    net, x_test, mini_batch_size=mini_batch_size)
                all_dormant_frac[task_idx, epoch_idx] = agg_frac
                all_dormant_per_layer[task_idx, epoch_idx] = torch.tensor(layer_fracs)
                sr = compute_stable_rank_from_activations(last_act)
                all_stable_rank[task_idx, epoch_idx] = sr

                # ELR
                current_lr = optimizer.param_groups[0]['lr']
                total_norm_sq = sum(p.data.norm()**2 for p in net.parameters() if p.requires_grad)
                n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                elr = current_lr / (total_norm_sq.item() / max(n_params, 1))
                all_elr[task_idx, epoch_idx] = elr

                if epoch_idx % 50 == 0 or epoch_idx == num_epochs - 1:
                    layer_str = ' '.join([f'{f:.2f}' for f in layer_fracs])
                    sdp_str = f" CondN={all_sdp_cond[task_idx]:.1f}" if sdp_gamma > 0 and task_idx > 0 else ""
                    print(f"    Task {task_idx} Epoch {epoch_idx:3d}/{num_epochs} | "
                          f"TrainAcc={train_accuracies[task_idx][epoch_idx]:.4f} "
                          f"TestAcc={test_accuracies[task_idx][epoch_idx]:.4f} "
                          f"Dormant={agg_frac:.3f} [{layer_str}] "
                          f"SR={sr:.0f} AvgW={wm:.4f} ELR={elr:.6f}{sdp_str}")
                if epoch_idx == num_epochs - 1:
                    dormant_persistence[task_idx] = torch.tensor(alive_scores < 0.01)
            net.train()

        task_time = time.time() - task_start
        if task_idx % 50 == 0 or task_idx == num_tasks - 1:
            layer_str = ' '.join([f'{f:.2f}' for f in all_dormant_per_layer[task_idx, -1].tolist()])
            guard_str = f" | {guard.summary()}" if guard is not None and task_idx % 200 == 0 else ""
            print(f"  Task {task_idx:4d}/{num_tasks} | "
                  f"TrainAcc={train_accuracies[task_idx][-1]:.4f} "
                  f"TestAcc={test_accuracies[task_idx][-1]:.4f} "
                  f"AvgW={all_weight_mag[task_idx, -1]:.4f} "
                  f"Dormant={all_dormant_frac[task_idx, -1]:.3f} [{layer_str}] "
                  f"StableRank={all_stable_rank[task_idx, -1]:.0f} | {task_time:.1f}s{guard_str}")

        if (task_idx + 1) % SAVE_EVERY_N_TASKS == 0:
            save_checkpoint(task_idx, reason="periodic")

    if not time_limit_hit:
        save_checkpoint(num_tasks - 1, reason="completed")

    if guard is not None:
        print(f"\n  Final {guard.summary()}")

    # Save results
    result_subdir = os.path.join(RESULTS_DIR, '0')
    os.makedirs(result_subdir, exist_ok=True)
    data_file = os.path.join(result_subdir, str(run_idx))
    save_data(data={
        'train_accuracies': train_accuracies.cpu(), 'test_accuracies': test_accuracies.cpu(),
        'all_weight_mag': all_weight_mag.cpu(), 'all_dormant_frac': all_dormant_frac.cpu(),
        'all_dormant_per_layer': all_dormant_per_layer.cpu(), 'all_stable_rank': all_stable_rank.cpu(),
        'all_elr': all_elr.cpu(), 'dormant_persistence': dormant_persistence.cpu(),
        'all_sdp_cond': all_sdp_cond.cpu(),
    }, data_file=data_file)
    print(f"  ✓ Results saved to {data_file}")

    return train_accuracies, test_accuracies, all_weight_mag, all_dormant_frac, all_dormant_per_layer, all_stable_rank, dormant_persistence, all_sdp_cond

print("✓ SASSHA training loop defined")


# %% [markdown]
# ## 10. Training Loop — CBP (baseline)

# %%
def run_cbp(params, run_idx=0):
    """Run CBP baseline on Continual ImageNet-32 with enhanced per-epoch metrics."""
    print(f"\n{'='*70}")
    print(f"  CBP Baseline — Run {run_idx} (ImageNet-32)")
    print(f"{'='*70}")

    wall_clock_start = time.time()
    num_tasks = params['num_tasks']
    num_epochs = params['num_showings']
    num_classes = params['num_classes']
    mini_batch_size = params['mini_batch_size']
    classes_per_task = num_classes
    examples_per_epoch = TRAIN_IMAGES_PER_CLASS * classes_per_task

    class_order = _ALL_CLASS_ORDERS[run_idx % len(_ALL_CLASS_ORDERS)]
    num_class_repetitions = int(num_classes * num_tasks / TOTAL_CLASSES) + 1
    class_order = np.concatenate([class_order] * num_class_repetitions)

    net = ConvNet(num_classes=classes_per_task)
    cbp_step_size = params.get('cbp_step_size', 0.001)
    learner = ConvCBP(
        net=net, step_size=cbp_step_size,
        momentum=params.get('cbp_momentum', 0.9), loss='nll',
        weight_decay=params.get('cbp_weight_decay', 0),
        opt=params.get('cbp_opt', 'sgd'), init='default',
        replacement_rate=params.get('cbp_replacement_rate', 3e-4),
        decay_rate=params.get('cbp_decay_rate', 0.99),
        util_type=params.get('cbp_util_type', 'adaptable_contribution'),
        device=device, maturity_threshold=params.get('cbp_maturity_threshold', 100))

    train_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    test_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    all_weight_mag = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    all_dormant_frac = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    all_dormant_per_layer = torch.zeros((num_tasks, num_epochs, NUM_LAYERS), dtype=torch.float)
    all_stable_rank = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    dormant_persistence = torch.zeros((num_tasks, TOTAL_NEURONS), dtype=torch.bool)

    cbp_ckpt_dir = os.path.join(RESULTS_DIR, '..', 'data_cbp_imagenet', 'checkpoints')
    cbp_result_dir = os.path.join(RESULTS_DIR, '..', 'data_cbp_imagenet', '0')
    os.makedirs(cbp_ckpt_dir, exist_ok=True)
    os.makedirs(cbp_result_dir, exist_ok=True)
    cbp_ckpt_file = os.path.join(cbp_ckpt_dir, f'ckpt_cbp_run{run_idx}.pt')
    start_task = 0

    if os.path.isfile(cbp_ckpt_file):
        print(f"  Loading checkpoint: {cbp_ckpt_file}")
        ckpt = torch.load(cbp_ckpt_file, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'])
        learner = ConvCBP(
            net=net, step_size=cbp_step_size,
            momentum=params.get('cbp_momentum', 0.9), loss='nll',
            weight_decay=params.get('cbp_weight_decay', 0),
            opt=params.get('cbp_opt', 'sgd'), init='default',
            replacement_rate=params.get('cbp_replacement_rate', 3e-4),
            decay_rate=params.get('cbp_decay_rate', 0.99),
            util_type=params.get('cbp_util_type', 'adaptable_contribution'),
            device=device, maturity_threshold=params.get('cbp_maturity_threshold', 100))
        train_accuracies = ckpt['train_accuracies']; test_accuracies = ckpt['test_accuracies']
        all_weight_mag = ckpt.get('all_weight_mag', all_weight_mag)
        all_dormant_frac = ckpt.get('all_dormant_frac', all_dormant_frac)
        all_dormant_per_layer = ckpt.get('all_dormant_per_layer', all_dormant_per_layer)
        all_stable_rank = ckpt.get('all_stable_rank', all_stable_rank)
        dormant_persistence = ckpt.get('dormant_persistence', dormant_persistence)
        start_task = ckpt['task_idx'] + 1
        print(f"  ✓ Resumed from task {start_task}")
        del ckpt; torch.cuda.empty_cache()
    else:
        print(f"  (no checkpoint, training from scratch)")

    def save_cbp_checkpoint(task_idx, reason="periodic"):
        torch.save({
            'task_idx': task_idx, 'model': net.state_dict(),
            'train_accuracies': train_accuracies, 'test_accuracies': test_accuracies,
            'all_weight_mag': all_weight_mag, 'all_dormant_frac': all_dormant_frac,
            'all_dormant_per_layer': all_dormant_per_layer, 'all_stable_rank': all_stable_rank,
            'dormant_persistence': dormant_persistence,
        }, cbp_ckpt_file)
        elapsed = time.time() - wall_clock_start
        print(f"  CBP checkpoint at task {task_idx} ({reason}) [{elapsed/3600:.1f}h elapsed]")

    x_train, x_test, y_train, y_test = None, None, None, None
    time_limit_hit = False

    for task_idx in range(start_task, num_tasks):
        elapsed = time.time() - wall_clock_start
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"\n  Time limit ({elapsed/3600:.1f}h). Saving.")
            save_cbp_checkpoint(task_idx - 1, reason="time_limit")
            time_limit_hit = True; break

        del x_train, x_test, y_train, y_test
        task_classes = class_order[task_idx * classes_per_task:(task_idx + 1) * classes_per_task]
        x_train, y_train, x_test, y_test = load_imagenet(task_classes)
        x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
        if device.type == 'cuda':
            x_train, x_test = x_train.to(device), x_test.to(device)
            y_train, y_test = y_train.to(device), y_test.to(device)

        net.layers[-1].weight.data *= 0
        net.layers[-1].bias.data *= 0

        for epoch_idx in range(num_epochs):
            example_order = np.random.permutation(TRAIN_IMAGES_PER_CLASS * classes_per_task)
            x_s = x_train[example_order]; y_s = y_train[example_order]
            new_train_accs = torch.zeros(examples_per_epoch // mini_batch_size, dtype=torch.float)
            batch_iter = 0
            for si in range(0, examples_per_epoch, mini_batch_size):
                bx = x_s[si:si + mini_batch_size]; by = y_s[si:si + mini_batch_size]
                loss, output = learner.learn(x=bx, target=by)
                with torch.no_grad():
                    new_train_accs[batch_iter] = accuracy(F.softmax(output, dim=1), by).cpu()
                    batch_iter += 1

            with torch.no_grad():
                train_accuracies[task_idx][epoch_idx] = new_train_accs.mean()
                new_test_accs = torch.zeros(x_test.shape[0] // mini_batch_size, dtype=torch.float)
                test_iter = 0
                for si in range(0, x_test.shape[0], mini_batch_size):
                    to, _ = net.predict(x=x_test[si:si + mini_batch_size])
                    new_test_accs[test_iter] = accuracy(
                        F.softmax(to, dim=1), y_test[si:si + mini_batch_size])
                    test_iter += 1
                test_accuracies[task_idx][epoch_idx] = new_test_accs.mean()

            with torch.no_grad():
                wm = compute_avg_weight_magnitude(net)
                all_weight_mag[task_idx, epoch_idx] = wm
                agg_frac, layer_fracs, alive_scores, last_act = compute_dormant_neurons_enhanced(
                    net, x_test, mini_batch_size=mini_batch_size)
                all_dormant_frac[task_idx, epoch_idx] = agg_frac
                all_dormant_per_layer[task_idx, epoch_idx] = torch.tensor(layer_fracs)
                sr = compute_stable_rank_from_activations(last_act)
                all_stable_rank[task_idx, epoch_idx] = sr
                if epoch_idx % 50 == 0 or epoch_idx == num_epochs - 1:
                    layer_str = ' '.join([f'{f:.2f}' for f in layer_fracs])
                    print(f"    Task {task_idx} Epoch {epoch_idx:3d}/{num_epochs} | "
                          f"TrainAcc={train_accuracies[task_idx][epoch_idx]:.4f} "
                          f"TestAcc={test_accuracies[task_idx][epoch_idx]:.4f} "
                          f"Dormant={agg_frac:.3f} [{layer_str}] SR={sr:.0f} AvgW={wm:.4f}")
                if epoch_idx == num_epochs - 1:
                    dormant_persistence[task_idx] = torch.tensor(alive_scores < 0.01)

        if task_idx % 50 == 0 or task_idx == num_tasks - 1:
            layer_str = ' '.join([f'{f:.2f}' for f in all_dormant_per_layer[task_idx, -1].tolist()])
            print(f"  Task {task_idx:4d}/{num_tasks} | "
                  f"TrainAcc={train_accuracies[task_idx][-1]:.4f} "
                  f"TestAcc={test_accuracies[task_idx][-1]:.4f} "
                  f"AvgW={all_weight_mag[task_idx, -1]:.4f} "
                  f"Dormant={all_dormant_frac[task_idx, -1]:.3f} [{layer_str}] "
                  f"StableRank={all_stable_rank[task_idx, -1]:.0f}")
        if (task_idx + 1) % SAVE_EVERY_N_TASKS == 0:
            save_cbp_checkpoint(task_idx, reason="periodic")

    if not time_limit_hit:
        save_cbp_checkpoint(num_tasks - 1, reason="completed")

    data_file = os.path.join(cbp_result_dir, str(run_idx))
    save_data(data={
        'train_accuracies': train_accuracies.cpu(), 'test_accuracies': test_accuracies.cpu(),
        'all_weight_mag': all_weight_mag.cpu(), 'all_dormant_frac': all_dormant_frac.cpu(),
        'all_dormant_per_layer': all_dormant_per_layer.cpu(), 'all_stable_rank': all_stable_rank.cpu(),
        'dormant_persistence': dormant_persistence.cpu(),
    }, data_file=data_file)
    print(f"  ✓ CBP results saved to {data_file}")
    return train_accuracies, test_accuracies, all_weight_mag, all_dormant_frac, all_dormant_per_layer, all_stable_rank, dormant_persistence

print("✓ CBP training loop defined")

# %% [markdown]
# ## 11. Run Experiments

# %%
# Run SASSHA
print("\n" + "▶" * 10 + " SASSHA " + "◀" * 10)
for method in METHODS_TO_RUN:
    cfg = CONFIGS[method]
    sassha_train, sassha_test, sassha_wmag, sassha_dormant, sassha_dormant_pl, sassha_sr, sassha_persist, sassha_cond = \
        run_sassha(method, cfg, run_idx=0)

# Run CBP (optional)
cbp_train, cbp_test = None, None
cbp_wmag, cbp_dormant, cbp_dormant_pl, cbp_sr, cbp_persist = None, None, None, None, None
if RUN_CBP:
    print("\n" + "▶" * 40 + " CBP " + "◀" * 40)
    cbp_train, cbp_test, cbp_wmag, cbp_dormant, cbp_dormant_pl, cbp_sr, cbp_persist = \
        run_cbp(CBP_PARAMS, run_idx=0)

# %% [markdown]
# ## 12. Comparison Plots

# %%
LAYER_NAMES = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
SASSHA_COLOR = '#2196F3'   # blue
CBP_COLOR    = '#FF5722'   # deep orange
COLORS_LAYER = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

def bin_tensor(data_2d, bin_size=50):
    """Bin the final-epoch values of a (num_tasks, num_epochs) tensor."""
    final_vals = data_2d[:, -1].cpu().numpy()
    n_bins = len(final_vals) // bin_size
    return np.array([final_vals[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)])

def bin_accuracies(accs, bin_size=50):
    return bin_tensor(accs, bin_size) * 100

has_cbp = cbp_test is not None
bin_size = 50
has_sdp = sassha_cond is not None and sassha_cond.abs().sum() > 0

# ── Pre-compute summary stats ──
n_final = min(100, sassha_test.shape[0])
sassha_final      = sassha_test[-n_final:, -1].mean().item() * 100
sassha_dorm_final = sassha_dormant[-n_final:, -1].mean().item() * 100
sassha_sr_final   = sassha_sr[-n_final:, -1].mean().item()
sassha_wmag_final = sassha_wmag[-n_final:, -1].mean().item()
sassha_cond_final = sassha_cond[-n_final:].mean().item() if has_sdp else 0.0
if has_cbp:
    cbp_final      = cbp_test[-n_final:, -1].mean().item() * 100
    cbp_dorm_final = cbp_dormant[-n_final:, -1].mean().item() * 100
    cbp_sr_final   = cbp_sr[-n_final:, -1].mean().item()

# ── Determine grid size ──
n_extra_rows = (1 if has_sdp else 0)
n_rows = 2 + n_extra_rows
n_cols = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
sdp_tag = '+SDP' if has_sdp else ''
title = (f'SASSHA+EMA+HessClip{sdp_tag} vs CBP — Continual ImageNet-32' if has_cbp
         else f'SASSHA+EMA+HessClip{sdp_tag} — Continual ImageNet-32')
fig.suptitle(title, fontsize=15, fontweight='bold', y=0.99)

def _style_ax(ax, xlabel='Task', ylabel='', title=''):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
    ax.legend(framealpha=0.9, fontsize=9); ax.grid(True, alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Row 0, Col 0: Test Accuracy ──
ax = axes[0, 0]
sassha_binned = bin_accuracies(sassha_test, bin_size)
x_bins = np.arange(len(sassha_binned)) * bin_size
ax.plot(x_bins, sassha_binned, color=SASSHA_COLOR, lw=2, label='SASSHA')
ax.fill_between(x_bins, sassha_binned, alpha=0.08, color=SASSHA_COLOR)
if has_cbp:
    cbp_binned = bin_accuracies(cbp_test, bin_size)
    ax.plot(np.arange(len(cbp_binned)) * bin_size, cbp_binned, color=CBP_COLOR, lw=2, label='CBP')
ax.axhline(sassha_final, color=SASSHA_COLOR, ls=':', lw=1, alpha=0.6)
_style_ax(ax, ylabel='Test Accuracy (%)', title='Online Test Accuracy')

# ── Row 0, Col 1: Train Accuracy ──
ax = axes[0, 1]
sassha_train_binned = bin_accuracies(sassha_train, bin_size)
ax.plot(x_bins[:len(sassha_train_binned)], sassha_train_binned, color=SASSHA_COLOR, lw=2, label='SASSHA')
if has_cbp:
    cbp_train_binned = bin_accuracies(cbp_train, bin_size)
    ax.plot(np.arange(len(cbp_train_binned)) * bin_size, cbp_train_binned, color=CBP_COLOR, lw=2, label='CBP')
_style_ax(ax, ylabel='Train Accuracy (%)', title='Online Train Accuracy')

# ── Row 0, Col 2: Weight Magnitude ──
ax = axes[0, 2]
sassha_wmag_binned = bin_tensor(sassha_wmag, bin_size)
ax.plot(np.arange(len(sassha_wmag_binned)) * bin_size, sassha_wmag_binned, color=SASSHA_COLOR, lw=2, label='SASSHA')
if has_cbp and cbp_wmag is not None:
    cbp_wmag_binned = bin_tensor(cbp_wmag, bin_size)
    ax.plot(np.arange(len(cbp_wmag_binned)) * bin_size, cbp_wmag_binned, color=CBP_COLOR, lw=2, label='CBP')
_style_ax(ax, ylabel='Avg |W|', title='Weight Magnitude')

# ── Row 1, Col 0: Dormant Neurons ──
ax = axes[1, 0]
sassha_dorm_binned = bin_tensor(sassha_dormant, bin_size) * 100
ax.plot(np.arange(len(sassha_dorm_binned)) * bin_size, sassha_dorm_binned, color=SASSHA_COLOR, lw=2, label='SASSHA')
ax.fill_between(np.arange(len(sassha_dorm_binned)) * bin_size, sassha_dorm_binned, alpha=0.08, color=SASSHA_COLOR)
if has_cbp and cbp_dormant is not None:
    cbp_dorm_binned = bin_tensor(cbp_dormant, bin_size) * 100
    ax.plot(np.arange(len(cbp_dorm_binned)) * bin_size, cbp_dorm_binned, color=CBP_COLOR, lw=2, label='CBP')
_style_ax(ax, ylabel='Dormant (%)', title='Dormant Neuron Fraction')

# ── Row 1, Col 1: Stable Rank ──
ax = axes[1, 1]
sassha_sr_binned = bin_tensor(sassha_sr, bin_size)
ax.plot(np.arange(len(sassha_sr_binned)) * bin_size, sassha_sr_binned, color=SASSHA_COLOR, lw=2, label='SASSHA')
if has_cbp and cbp_sr is not None:
    cbp_sr_binned = bin_tensor(cbp_sr, bin_size)
    ax.plot(np.arange(len(cbp_sr_binned)) * bin_size, cbp_sr_binned, color=CBP_COLOR, lw=2, label='CBP')
_style_ax(ax, ylabel='Stable Rank', title='Stable Rank (last layer)')

# ── Row 1, Col 2: Summary Table ──
ax = axes[1, 2]
ax.axis('off')
summary_lines = [
    f'Final {n_final}-task average metrics',
    f'{"─"*36}',
    f'SASSHA  TestAcc   {sassha_final:6.2f}%',
    f'SASSHA  Dormant   {sassha_dorm_final:6.2f}%',
    f'SASSHA  StableRk  {sassha_sr_final:6.1f}',
    f'SASSHA  Avg|W|    {sassha_wmag_final:6.4f}',
]
if has_sdp:
    summary_lines.append(f'SASSHA  CondNum   {sassha_cond_final:10.1f}')
if has_cbp:
    summary_lines += [
        f'{"─"*36}',
        f'CBP     TestAcc   {cbp_final:6.2f}%',
        f'CBP     Dormant   {cbp_dorm_final:6.2f}%',
        f'CBP     StableRk  {cbp_sr_final:6.1f}',
        f'{"─"*36}',
        f'Δ TestAcc (S−C)   {sassha_final - cbp_final:+6.2f}%',
    ]
ax.text(0.5, 0.5, '\n'.join(summary_lines), transform=ax.transAxes,
        fontsize=12, verticalalignment='center', horizontalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9C4', edgecolor='#FBC02D', alpha=0.85))

# ── SDP row (when SDP is active) ──
if has_sdp:
    sdp_row = 2
    SDP_COLOR = '#00897B'  # teal

    # Col 0: Condition number over tasks
    ax = axes[sdp_row, 0]
    cond_vals = sassha_cond.cpu().numpy()
    valid_tasks = np.where(cond_vals > 0)[0]
    if len(valid_tasks) > 0:
        # Bin condition numbers
        n_bins_c = max(1, len(valid_tasks) // bin_size)
        cond_binned = np.array([cond_vals[valid_tasks[i*bin_size:min((i+1)*bin_size, len(valid_tasks))]].mean()
                                for i in range(n_bins_c)])
        x_bins_c = np.arange(n_bins_c) * bin_size
        ax.plot(x_bins_c, cond_binned, color=SDP_COLOR, lw=2, label='Cond #')
        ax.fill_between(x_bins_c, cond_binned, alpha=0.08, color=SDP_COLOR)
        ax.axhline(sassha_cond_final, color=SDP_COLOR, ls=':', lw=1, alpha=0.6)
    _style_ax(ax, ylabel='Condition Number', title='SDP: Weight Condition Number')

    # Col 1: Condition number vs Stable Rank correlation
    ax = axes[sdp_row, 1]
    sr_final_per_task = sassha_sr[:, -1].cpu().numpy()
    n_plot = min(len(cond_vals), len(sr_final_per_task))
    valid_mask = cond_vals[:n_plot] > 0
    ax.scatter(cond_vals[:n_plot][valid_mask], sr_final_per_task[:n_plot][valid_mask],
               s=4, alpha=0.3, color=SDP_COLOR)
    ax.set_xlabel('Condition Number'); ax.set_ylabel('Stable Rank')
    ax.set_title('Condition Number vs Stable Rank', fontweight='bold')
    ax.grid(True, alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Col 2: Condition number vs Test Accuracy correlation
    ax = axes[sdp_row, 2]
    test_final_pt = sassha_test[:, -1].cpu().numpy() * 100
    ax.scatter(cond_vals[:n_plot][valid_mask], test_final_pt[:n_plot][valid_mask],
               s=4, alpha=0.3, color=SDP_COLOR)
    ax.set_xlabel('Condition Number'); ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Condition Number vs Test Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plot_file = os.path.join(RESULTS_DIR, 'sassha_vs_cbp_imagenet32.png')
plt.savefig(plot_file, dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Main plot saved to {plot_file}")

# %% [markdown]
# ## 13. Per-Layer Dormant Neuron Plot

# %%
fig2, axes2 = plt.subplots(1, NUM_LAYERS, figsize=(22, 4.5), sharey=True)
fig2.suptitle('Per-Layer Dormant Neuron Fraction — SASSHA vs CBP', fontsize=14, fontweight='bold')
for l_idx in range(NUM_LAYERS):
    ax = axes2[l_idx]
    sassha_layer = sassha_dormant_pl[:, -1, l_idx].cpu().numpy()
    n_bins = len(sassha_layer) // bin_size
    sassha_layer_binned = np.array([sassha_layer[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)]) * 100
    ax.plot(np.arange(n_bins) * bin_size, sassha_layer_binned, color=SASSHA_COLOR, lw=2, label='SASSHA')
    ax.fill_between(np.arange(n_bins) * bin_size, sassha_layer_binned, alpha=0.08, color=SASSHA_COLOR)
    if has_cbp and cbp_dormant_pl is not None:
        cbp_layer = cbp_dormant_pl[:, -1, l_idx].cpu().numpy()
        cbp_layer_binned = np.array([cbp_layer[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)]) * 100
        ax.plot(np.arange(n_bins) * bin_size, cbp_layer_binned, color=CBP_COLOR, lw=2, label='CBP')
    ax.set_title(f'{LAYER_NAMES[l_idx]} ({LAYER_SIZES[l_idx]})', fontweight='bold')
    ax.set_xlabel('Task')
    if l_idx == 0: ax.set_ylabel('Dormant (%)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plot_file2 = os.path.join(RESULTS_DIR, 'per_layer_dormant.png')
plt.savefig(plot_file2, dpi=200, bbox_inches='tight'); plt.show()
print(f"✓ Per-layer dormant plot saved to {plot_file2}")

# %% [markdown]
# ## 14. Dormant Persistence Heatmap

# %%
n_heatmaps = 2 if (has_cbp and cbp_persist is not None) else 1
fig3, axes3 = plt.subplots(1, n_heatmaps, figsize=(8 * n_heatmaps, 6))
if n_heatmaps == 1:
    axes3 = [axes3]
fig3.suptitle('Dormant Neuron Persistence Across Tasks', fontsize=14, fontweight='bold')

persist_np = sassha_persist.cpu().numpy().astype(float)
sample_step = max(1, persist_np.shape[0] // 200)
im0 = axes3[0].imshow(persist_np[::sample_step].T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
axes3[0].set_xlabel(f'Task (every {sample_step})'); axes3[0].set_ylabel('Neuron index')
axes3[0].set_title('SASSHA — Dormant Neurons', fontweight='bold')
plt.colorbar(im0, ax=axes3[0], fraction=0.046, pad=0.04, label='Dormant')

if n_heatmaps == 2:
    cbp_persist_np = cbp_persist.cpu().numpy().astype(float)
    im1 = axes3[1].imshow(cbp_persist_np[::sample_step].T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes3[1].set_xlabel(f'Task (every {sample_step})'); axes3[1].set_ylabel('Neuron index')
    axes3[1].set_title('CBP — Dormant Neurons', fontweight='bold')
    plt.colorbar(im1, ax=axes3[1], fraction=0.046, pad=0.04, label='Dormant')

plt.tight_layout()
plot_file3 = os.path.join(RESULTS_DIR, 'dormant_persistence.png')
plt.savefig(plot_file3, dpi=200, bbox_inches='tight'); plt.show()
print(f"✓ Dormant persistence plot saved to {plot_file3}")

# Print summary
print(f"\n{'='*60}")
print(f"  Continual ImageNet-32 — Final {n_final}-task avg:")
sdp_sum_str = f"  CondNum={sassha_cond_final:.1f}" if has_sdp else ""
print(f"  SASSHA: TestAcc={sassha_final:.2f}%  Dormant={sassha_dorm_final:.2f}%  StableRank={sassha_sr_final:.1f}{sdp_sum_str}")
if has_cbp:
    print(f"  CBP:    TestAcc={cbp_final:.2f}%  Dormant={cbp_dorm_final:.2f}%  StableRank={cbp_sr_final:.1f}")
    print(f"  Delta (SASSHA - CBP): {sassha_final - cbp_final:+.2f}%")
print(f"{'='*60}")
