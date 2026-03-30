# %% [markdown]
# # SASSHA + SDP on Continual Permuted MNIST
# #
# **SASSHA** (SAM + Hutchinson Hessian) + **EMA** + **Hessian Clipping**
# + **Spectral Diversity Preservation (SDP)** on Continual Permuted MNIST.
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
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.utils.miscellaneous import nll_accuracy as accuracy
from lop.utils.miscellaneous import compute_matrix_rank_summaries

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Metrics

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
    """Count dead neurons (zero activation across all probe samples) per hidden layer."""
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
    """Compute dormant neuron fraction per hidden layer for DeepFFNN.
    Returns (aggregate_frac, per_layer_fracs, last_hidden_act_numpy).
    """
    net.eval()
    _, hidden_acts = net.predict(probe_x)
    per_layer_frac = []
    total_d, total_n = 0, 0
    last_act = None
    for i, act in enumerate(hidden_acts):
        alive_score = (act != 0).float().mean(dim=0)  # shape: [H]
        n_units = act.shape[1]
        dormant = (alive_score < threshold).sum().item()
        per_layer_frac.append(dormant / n_units if n_units > 0 else 0.0)
        total_d += dormant; total_n += n_units
        if i == len(hidden_acts) - 1:
            last_act = act.cpu().numpy()
    agg_frac = total_d / total_n if total_n > 0 else 0.0
    return agg_frac, per_layer_frac, last_act

def compute_stable_rank_from_activations(act):
    """Stable rank = number of SVs needed to capture 99% of sum(|σ|).
    Same formula as CBP's compute_abs_approximate_rank / compute_stable_rank."""
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
        for module in net.modules():
            if not isinstance(module, nn.Linear):
                continue
            W = module.weight.data
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
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
            module.weight.data.copy_(W_new)
    return cond_numbers

print("✓ Metrics defined (+ SDP)")

# %% [markdown]
# ## 3. MNIST Data Loading (mirrors CBP repo)

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
        return imgs.flatten(start_dim=1), labels   # (N, 784), (N,)

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
EXAMPLES_PER_TASK = IMAGES_PER_CLASS * CLASSES_PER_TASK   # 60 000
print(f"\n  examples_per_task (change_after) = {EXAMPLES_PER_TASK}")
print("✓ MNIST ready  (CBP paper format)")

# %% [markdown]
# ## 4. SASSHA Optimizer
# #
# SAM + Hutchinson Hessian trace — includes `compute_hessian` flag for guard.

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
    """SASSHA optimizer — SAM + Hutchinson Hessian trace."""
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
# Hessian floor fix prevents near-zero curvature from amplifying updates.

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
SEED            = 42
SAVE_EVERY_N_TASKS = 80
TIME_LIMIT_SECONDS = 11.5 * 3600

# CBP paper constants (permuted_mnist/online_expr.py)
NUM_TASKS       = 800
CHANGE_AFTER    = 60_000
MINI_BATCH_SIZE = 32
STEPS_PER_TASK  = CHANGE_AFTER // MINI_BATCH_SIZE   # 1875
PROBE_SIZE      = 2000

_SASSHA_BASE = dict(
    lr=0.003, betas=(0.9, 0.999), weight_decay=5e-4, rho=0.05,
    lazy_hessian=10, n_samples=1, eps=1e-4,
    hessian_power_schedule='constant', max_hessian_power=1.0, min_hessian_power=0.5,
    use_ema=True, ema_decay=0.999,
    label_smoothing=True,
    sdp_gamma=0.3,
    use_guard=True,
    hessian_clip=1e3,
    hessian_floor=1e-5,
    max_grad_norm=float('inf'),
)

_CBP_BASE = dict(
    step_size=0.003,            # same lr as SASSHA for fair comparison
    opt='sgd',
    replacement_rate=1e-4,      # Nature 2024 paper value
    decay_rate=0.99,
    maturity_threshold=100,
    util_type='adaptable_contribution',
    accumulate=True,
)

CONFIGS = {
    'sassha_sdp': {
        **_SASSHA_BASE,
    },
    'cbp': {
        **_CBP_BASE,
    },
}

METHODS_TO_RUN = ['sassha_sdp', 'cbp']

RESULTS_DIR = os.path.join('permuted_mnist_results', 'comparison')
CKPT_DIR    = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"✓ Config: {NUM_TASKS} tasks × {STEPS_PER_TASK} steps/task, "
      f"batch={MINI_BATCH_SIZE}, network: {INPUT_SIZE}→{NUM_FEATURES}×{NUM_HIDDEN_LAYERS}→{CLASSES_PER_TASK}")

# %% [markdown]
# ## 8. Build Optimizer

# %%
def build_optimizer(config, model):
    hp_scheduler = None
    hp_schedule_type = config.get('hessian_power_schedule', None)
    if hp_schedule_type is not None:
        total_steps = NUM_TASKS * STEPS_PER_TASK
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
# ## 9. Training Loop — SASSHA + EMA + Hessian Clipping + SDP

# %%
def _ckpt_path(method_name: str) -> str:
    return os.path.join(CKPT_DIR, f"ckpt_{method_name}.pt")


def run_sassha(method_name, config):
    """SASSHA + EMA + Hessian Clipping + SDP on Permuted MNIST."""
    print(f"\n{'='*70}")
    print(f"  {method_name} — Permuted MNIST ({NUM_TASKS} tasks)")
    use_guard = config.get('use_guard', False)
    if use_guard:
        print(f"  Hessian clip: [{config.get('hessian_floor', 0)}, {config.get('hessian_clip', 'inf')}]")
    print(f"{'='*70}")

    wall_clock_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    # Build network: DeepFFNN 784 → 2000 × 5 → 10
    net = DeepFFNN(
        input_size        = INPUT_SIZE,
        num_features      = NUM_FEATURES,
        num_outputs       = CLASSES_PER_TASK,
        num_hidden_layers = NUM_HIDDEN_LAYERS,
    ).to(device)

    optimizer = build_optimizer(config, net)
    ls = 0.1 if config.get('label_smoothing', False) else 0.0
    loss_fn = lambda logits, target: F.cross_entropy(logits, target, label_smoothing=ls)

    guard = None
    if use_guard:
        guard = GradientExplosionGuard(
            hessian_clip_value=config.get('hessian_clip', 1e3),
            hessian_floor=config.get('hessian_floor', 1e-4),
            enable_hessian_clip=True,
            max_grad_norm=config.get('max_grad_norm', float('inf')))

    ema = EMAWrapper(net, config.get('ema_decay', 0.999)) if config.get('use_ema', False) else None

    sdp_gamma = config.get('sdp_gamma', 0.0)
    if sdp_gamma > 0:
        print(f"  SDP enabled: γ={sdp_gamma}")

    # Metric tensors
    total_iters       = NUM_TASKS * STEPS_PER_TASK
    accuracies        = torch.zeros(total_iters, dtype=torch.float)
    train_accuracies  = torch.zeros(NUM_TASKS, dtype=torch.float)   # EMA + full train set (same method as test)
    test_accuracies   = torch.zeros(NUM_TASKS, dtype=torch.float)
    weight_mag_log    = torch.zeros((total_iters, NUM_HIDDEN_LAYERS + 1), dtype=torch.float)
    ranks             = torch.zeros((NUM_TASKS, NUM_HIDDEN_LAYERS), dtype=torch.float)
    effective_ranks   = torch.zeros((NUM_TASKS, NUM_HIDDEN_LAYERS), dtype=torch.float)
    approximate_ranks = torch.zeros((NUM_TASKS, NUM_HIDDEN_LAYERS), dtype=torch.float)
    dead_neurons      = torch.zeros((NUM_TASKS, NUM_HIDDEN_LAYERS), dtype=torch.float)
    all_stable_rank   = torch.zeros(NUM_TASKS, dtype=torch.float)
    all_dormant_frac  = torch.zeros(NUM_TASKS, dtype=torch.float)
    all_sdp_cond      = torch.zeros((NUM_TASKS,), dtype=torch.float)
    task_metrics      = {'task_acc': [], 'task_train_acc': [], 'task_test_acc': [], 'task_id': [], 'avg_weight_mag': []}

    # Checkpoint resume
    ckpt_file  = _ckpt_path(method_name)
    start_task = 0
    global_iter = 0
    if os.path.isfile(ckpt_file):
        print(f"  Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
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
        if ema is not None and 'ema_shadow' in ckpt: ema._shadow = ckpt['ema_shadow']
        if guard is not None and 'guard_state' in ckpt:
            gs = ckpt['guard_state']
            guard.hessian_clip_count = gs.get('hessian_clip_count', 0)
            guard.hessian_floor_count = gs.get('hessian_floor_count', 0)
        if optimizer.hessian_power_scheduler is not None and 'hp_scheduler_t' in ckpt:
            optimizer.hessian_power_scheduler.t = ckpt['hp_scheduler_t']
        for p in optimizer.get_params():
            p.hess = 0.0; optimizer.state[p]["hessian step"] = 0
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
        if optimizer.hessian_power_scheduler is not None:
            ckpt_data['hp_scheduler_t'] = optimizer.hessian_power_scheduler.t
        if ema is not None: ckpt_data['ema_shadow'] = ema._shadow
        if guard is not None:
            ckpt_data['guard_state'] = {
                'hessian_clip_count': guard.hessian_clip_count,
                'hessian_floor_count': guard.hessian_floor_count}
        torch.save(ckpt_data, ckpt_file)
        elapsed = time.time() - wall_clock_start
        print(f"  Checkpoint saved at task {task_idx} ({reason}) [{elapsed/3600:.1f}h elapsed]")

    # Working MNIST copies (permuted in-place each task)
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

        # ── 1. New pixel permutation + data shuffle ───────────────────
        rng        = np.random.RandomState(task_idx + SEED * 1000)
        pixel_perm = rng.permutation(INPUT_SIZE)
        data_perm  = rng.permutation(EXAMPLES_PER_TASK)
        x          = x[:, pixel_perm]
        x, y       = x[data_perm], y[data_perm]
        x_test     = x_test[:, pixel_perm]  # same permutation for test

        # ── 2. SDP at task boundary ───────────────────────────────────
        if sdp_gamma > 0 and task_idx > 0:
            cond_nums = apply_sdp(net, sdp_gamma)
            avg_cond = sum(cond_nums) / max(len(cond_nums), 1)
            all_sdp_cond[task_idx] = avg_cond
            if task_idx % 50 == 0:
                print(f"    SDP applied: avg condition number = {avg_cond:.1f}")

        # ── 3. Reset EMA at task boundary ─────────────────────────────
        if ema is not None: ema.reset(net)

        # ── 4. Pre-task probe: rank + dead neurons ────────────────────
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

        # ── 5. Train STEPS_PER_TASK mini-batch steps ──────────────────
        net.train()
        for step in range(STEPS_PER_TASK):
            start_idx = (step * MINI_BATCH_SIZE) % EXAMPLES_PER_TASK
            batch_x   = x[start_idx: start_idx + MINI_BATCH_SIZE].to(device)
            batch_y   = y[start_idx: start_idx + MINI_BATCH_SIZE].to(device)

            # SASSHA two-pass protocol
            optimizer.zero_grad()
            logits = net.predict(batch_x)[0]
            loss   = loss_fn(logits, batch_y)
            loss.backward()

            # SAM perturbation
            optimizer.perturb_weights(zero_grad=True)
            logits_pert = net.predict(batch_x)[0]
            loss_pert = loss_fn(logits_pert, batch_y)
            loss_pert.backward(create_graph=True)
            optimizer.unperturb()

            # Grad norm clip
            if guard is not None: guard.clip_global_grad_norm(net)

            # Hessian computation (for guard clipping)
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
                out_sm = F.softmax(net.predict(batch_x)[0], dim=1)
                accuracies[global_iter] = accuracy(out_sm, batch_y).cpu()

                for l_idx, layer_idx in enumerate(net.layers_to_log):
                    if l_idx < weight_mag_log.shape[1]:
                        weight_mag_log[global_iter][l_idx] = (
                            net.layers[layer_idx].weight.data.abs().mean())

            global_iter += 1

        # ── 6. Train & Test evaluation (both use EMA weights) ──────────
        net.eval()
        with torch.no_grad():
            if ema is not None: ema.apply(net)

            # Train accuracy (EMA weights, full train set — same method as test)
            train_correct = 0
            train_total = 0
            for si in range(0, x.shape[0], MINI_BATCH_SIZE):
                tb_x = x[si:si + MINI_BATCH_SIZE].to(device)
                tb_y = y[si:si + MINI_BATCH_SIZE].to(device)
                to = net.predict(tb_x)[0]
                train_correct += (to.argmax(dim=1) == tb_y).sum().item()
                train_total += tb_y.shape[0]
            train_acc = train_correct / max(train_total, 1)
            train_accuracies[task_idx] = train_acc

            # Test accuracy (EMA weights, full test set)
            test_correct = 0
            test_total = 0
            for si in range(0, x_test.shape[0], MINI_BATCH_SIZE):
                tb_x = x_test[si:si + MINI_BATCH_SIZE].to(device)
                tb_y = y_test[si:si + MINI_BATCH_SIZE].to(device)
                to = net.predict(tb_x)[0]
                test_correct += (to.argmax(dim=1) == tb_y).sum().item()
                test_total += tb_y.shape[0]
            test_acc = test_correct / max(test_total, 1)
            test_accuracies[task_idx] = test_acc

            if ema is not None: ema.restore(net)

            # Dormant neurons + stable rank (last hidden layer, like sassha-imgnet.py)
            probe_x = x[:PROBE_SIZE].to(device)
            agg_dormant, _, last_act = compute_dormant_neurons_ffnn(net, probe_x)
            all_dormant_frac[task_idx] = agg_dormant
            sr = compute_stable_rank_from_activations(last_act)
            all_stable_rank[task_idx] = sr

        # ── 7. Per-task summary ────────────────────────────────────────
        task_acc = accuracies[iter_start:global_iter].mean().item()
        task_metrics['task_acc'].append(task_acc)
        task_metrics['task_train_acc'].append(train_acc)
        task_metrics['task_test_acc'].append(test_acc)
        task_metrics['task_id'].append(task_idx)
        task_metrics['avg_weight_mag'].append(
            weight_mag_log[iter_start:global_iter].mean(dim=0).tolist())

        et = time.time() - t0
        guard_str = f" | {guard.summary()}" if guard is not None and task_idx % 200 == 0 else ""
        print(f"  [{method_name}] Task {task_idx:4d}/{NUM_TASKS}  "
              f"TrainAcc={train_acc:.4f}  TestAcc={test_acc:.4f}  "
              f"Dormant={agg_dormant:.3f}  SR={sr:.0f}  "
              f"AvgW={compute_avg_weight_magnitude(net):.4f}  "
              f"{et:.1f}s{guard_str}")

        if (task_idx + 1) % SAVE_EVERY_N_TASKS == 0 or task_idx == NUM_TASKS - 1:
            save_checkpoint(task_idx, reason="periodic" if task_idx < NUM_TASKS - 1 else "completed")

    # ── Save results ───────────────────────────────────────────────────
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

    if guard is not None:
        print(f"\n  Final {guard.summary()}")

    return (accuracies, train_accuracies, test_accuracies, weight_mag_log, ranks,
            effective_ranks, approximate_ranks, dead_neurons, all_stable_rank,
            all_dormant_frac, all_sdp_cond, task_metrics)

print("✓ run_sassha training loop defined (Permuted MNIST + SDP)")

# %% [markdown]
# ## 9b. CBP Training Loop (Nature 2024 baseline)
#
# Continual Backpropagation: SGD + Generate-and-Test neuron replacement.
# Same architecture & data for fair comparison with SASSHA+SDP.

# %%
def run_cbp(method_name, config):
    """CBP (Continual Backpropagation, Nature 2024) on Permuted MNIST."""
    print(f"\n{'='*70}")
    print(f"  {method_name} — Permuted MNIST ({NUM_TASKS} tasks)")
    print(f"  step_size={config['step_size']}, replacement_rate={config['replacement_rate']}")
    print(f"  opt={config['opt']}, decay_rate={config['decay_rate']}")
    print(f"{'='*70}")

    wall_clock_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    net = DeepFFNN(
        input_size=INPUT_SIZE, num_features=NUM_FEATURES,
        num_outputs=CLASSES_PER_TASK, num_hidden_layers=NUM_HIDDEN_LAYERS,
    ).to(device)

    learner = ContinualBackprop(
        net=net,
        step_size=config['step_size'],
        opt=config.get('opt', 'sgd'),
        loss='nll',
        replacement_rate=config['replacement_rate'],
        maturity_threshold=config.get('maturity_threshold', 100),
        decay_rate=config.get('decay_rate', 0.99),
        util_type=config.get('util_type', 'adaptable_contribution'),
        accumulate=config.get('accumulate', True),
        device=device,
    )

    # Metric tensors (same shape as SASSHA for comparison)
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
    all_sdp_cond      = torch.zeros(NUM_TASKS, dtype=torch.float)  # always 0 for CBP
    task_metrics      = {'task_acc': [], 'task_train_acc': [], 'task_test_acc': [],
                         'task_id': [], 'avg_weight_mag': []}

    ckpt_file = _ckpt_path(method_name)
    start_task = 0
    global_iter = 0
    if os.path.isfile(ckpt_file):
        print(f"  Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'])
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
        task_metrics      = ckpt.get('task_metrics',       task_metrics)
        start_task        = ckpt['task'] + 1
        global_iter       = start_task * STEPS_PER_TASK
        # Rebuild GnT from restored net (GnT state not saved)
        learner = ContinualBackprop(
            net=net,
            step_size=config['step_size'],
            opt=config.get('opt', 'sgd'),
            loss='nll',
            replacement_rate=config['replacement_rate'],
            maturity_threshold=config.get('maturity_threshold', 100),
            decay_rate=config.get('decay_rate', 0.99),
            util_type=config.get('util_type', 'adaptable_contribution'),
            accumulate=config.get('accumulate', True),
            device=device,
        )
        print(f"  ✓ Resumed from task {start_task}")
        del ckpt; torch.cuda.empty_cache()
    else:
        print("  (no checkpoint — training from scratch)")

    def save_checkpoint_cbp(task_idx, reason="periodic"):
        ckpt_data = {
            'task': task_idx, 'model': net.state_dict(),
            'accuracies': accuracies, 'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies, 'weight_mag_log': weight_mag_log,
            'ranks': ranks, 'effective_ranks': effective_ranks,
            'approximate_ranks': approximate_ranks, 'dead_neurons': dead_neurons,
            'all_stable_rank': all_stable_rank, 'all_dormant_frac': all_dormant_frac,
            'all_sdp_cond': all_sdp_cond, 'task_metrics': task_metrics,
            'params': config,
        }
        torch.save(ckpt_data, ckpt_file)
        elapsed = time.time() - wall_clock_start
        print(f"  CBP checkpoint saved at task {task_idx} ({reason}) [{elapsed/3600:.1f}h elapsed]")

    x = x_mnist.clone()
    y = y_mnist.clone()
    x_test = x_test_mnist.clone()
    y_test = y_test_mnist.clone()

    for task_idx in range(start_task, NUM_TASKS):
        t0 = time.time()
        elapsed = time.time() - wall_clock_start
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"\n  Time limit ({elapsed/3600:.1f}h). Saving.")
            save_checkpoint_cbp(task_idx - 1, reason="time_limit")
            break

        iter_start = global_iter

        # ── 1. Permutation ──
        rng        = np.random.RandomState(task_idx + SEED * 1000)
        pixel_perm = rng.permutation(INPUT_SIZE)
        data_perm  = rng.permutation(EXAMPLES_PER_TASK)
        x          = x[:, pixel_perm]
        x, y       = x[data_perm], y[data_perm]
        x_test     = x_test[:, pixel_perm]

        # ── 2. Pre-task probe: rank + dead neurons ──
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

        # ── 3. Train STEPS_PER_TASK steps ──
        net.train()
        for step in range(STEPS_PER_TASK):
            start_idx = (step * MINI_BATCH_SIZE) % EXAMPLES_PER_TASK
            batch_x   = x[start_idx: start_idx + MINI_BATCH_SIZE].to(device)
            batch_y   = y[start_idx: start_idx + MINI_BATCH_SIZE].to(device)

            loss, output = learner.learn(x=batch_x, target=batch_y)

            with torch.no_grad():
                out_sm = F.softmax(output, dim=1)
                accuracies[global_iter] = accuracy(out_sm, batch_y).cpu()
                for l_idx, layer_idx in enumerate(net.layers_to_log):
                    if l_idx < weight_mag_log.shape[1]:
                        weight_mag_log[global_iter][l_idx] = (
                            net.layers[layer_idx].weight.data.abs().mean())
            global_iter += 1

        # ── 4. Train & Test evaluation (raw weights, no EMA for CBP) ──
        net.eval()
        with torch.no_grad():
            train_correct, train_total = 0, 0
            for si in range(0, x.shape[0], MINI_BATCH_SIZE):
                tb_x = x[si:si + MINI_BATCH_SIZE].to(device)
                tb_y = y[si:si + MINI_BATCH_SIZE].to(device)
                to = net.predict(tb_x)[0]
                train_correct += (to.argmax(dim=1) == tb_y).sum().item()
                train_total += tb_y.shape[0]
            train_acc = train_correct / max(train_total, 1)
            train_accuracies[task_idx] = train_acc

            test_correct, test_total = 0, 0
            for si in range(0, x_test.shape[0], MINI_BATCH_SIZE):
                tb_x = x_test[si:si + MINI_BATCH_SIZE].to(device)
                tb_y = y_test[si:si + MINI_BATCH_SIZE].to(device)
                to = net.predict(tb_x)[0]
                test_correct += (to.argmax(dim=1) == tb_y).sum().item()
                test_total += tb_y.shape[0]
            test_acc = test_correct / max(test_total, 1)
            test_accuracies[task_idx] = test_acc

            # Dormant neurons + stable rank
            probe_x = x[:PROBE_SIZE].to(device)
            agg_dormant, _, last_act = compute_dormant_neurons_ffnn(net, probe_x)
            all_dormant_frac[task_idx] = agg_dormant
            sr = compute_stable_rank_from_activations(last_act)
            all_stable_rank[task_idx] = sr

        # ── 5. Per-task summary ──
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
            save_checkpoint_cbp(task_idx, reason="periodic" if task_idx < NUM_TASKS - 1 else "completed")

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
    print(f"  ✓ CBP results saved to {result_file}")

    return (accuracies, train_accuracies, test_accuracies, weight_mag_log, ranks,
            effective_ranks, approximate_ranks, dead_neurons, all_stable_rank,
            all_dormant_frac, all_sdp_cond, task_metrics)

print("✓ run_cbp training loop defined (Permuted MNIST)")

# %% [markdown]
# ## 10. Run Experiments

# %%
all_results = {}
for method in METHODS_TO_RUN:
    cfg = CONFIGS[method]
    if method == 'cbp':
        result = run_cbp(method, cfg)
    else:
        result = run_sassha(method, cfg)
    # Unpack into dict: (acc, train_acc, test_acc, wmag, ranks, er, apr, dead, sr, dormant, cond, task)
    all_results[method] = {
        'acc': result[0], 'train_acc': result[1], 'test_acc': result[2],
        'wmag': result[3], 'ranks': result[4], 'er': result[5],
        'apr': result[6], 'dead': result[7], 'sr': result[8],
        'dormant': result[9], 'cond': result[10], 'task': result[11],
    }

# %% [markdown]
# ## 11. Results Plots

# %%
TASK_WINDOW = 50
LAYER_NAMES = [f'Hidden {i+1} ({NUM_FEATURES})' for i in range(NUM_HIDDEN_LAYERS)]

METHOD_STYLES = {
    'sassha_sdp': {'color': '#2196F3', 'label': 'SASSHA+SDP'},
    'cbp':        {'color': '#FF5722', 'label': 'CBP (Nature 2024)'},
}

def smooth(arr, w=TASK_WINDOW):
    n = len(arr) // w
    return np.array([arr[i*w:(i+1)*w].mean() for i in range(n)])

def _style(method):
    return METHOD_STYLES.get(method, {'color': 'gray', 'label': method})

# ── Comparison Figure: 3×3 grid ──
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('SASSHA+SDP vs CBP — Continual Permuted MNIST\n'
             f'Network: {INPUT_SIZE}→{NUM_FEATURES}×{NUM_HIDDEN_LAYERS}→{CLASSES_PER_TASK}, '
             f'{NUM_TASKS} tasks, batch={MINI_BATCH_SIZE}',
             fontsize=14, fontweight='bold')

def _clean(ax):
    ax.grid(True, alpha=0.25); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Row 0, Col 0: Test Accuracy ──
ax = axes[0, 0]
for m, d in all_results.items():
    s = _style(m)
    test_arr = np.array(d['task']['task_test_acc'])
    ax.plot(test_arr * 100, color=s['color'], lw=0.8, alpha=0.3)
    sm = smooth(test_arr) * 100
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Accuracy (%)'); ax.set_title('Test Accuracy')
ax.legend(fontsize=9); _clean(ax)

# ── Row 0, Col 1: Train Accuracy ──
ax = axes[0, 1]
for m, d in all_results.items():
    s = _style(m)
    train_arr = np.array(d['task']['task_train_acc'])
    ax.plot(train_arr * 100, color=s['color'], lw=0.8, alpha=0.3)
    sm = smooth(train_arr) * 100
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Accuracy (%)'); ax.set_title('Train Accuracy')
ax.legend(fontsize=9); _clean(ax)

# ── Row 0, Col 2: Train-Test Gap ──
ax = axes[0, 2]
for m, d in all_results.items():
    s = _style(m)
    tr = np.array(d['task']['task_train_acc'])
    te = np.array(d['task']['task_test_acc'])
    gap = (tr - te) * 100
    sm = smooth(gap)
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Gap (%)'); ax.set_title('Train − Test Gap')
ax.axhline(0, color='black', ls=':', lw=0.8, alpha=0.5)
ax.legend(fontsize=9); _clean(ax)

# ── Row 1, Col 0: Stable Rank ──
ax = axes[1, 0]
for m, d in all_results.items():
    s = _style(m)
    sr = d['sr'].cpu().numpy()
    ax.plot(sr, color=s['color'], lw=0.8, alpha=0.3)
    sm = smooth(sr)
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Stable Rank'); ax.set_title('Stable Rank (last hidden layer)')
ax.legend(fontsize=9); _clean(ax)

# ── Row 1, Col 1: Dormant Neuron Fraction ──
ax = axes[1, 1]
for m, d in all_results.items():
    s = _style(m)
    dorm = d['dormant'].cpu().numpy() * 100
    sm = smooth(dorm)
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Dormant (%)'); ax.set_title('Dormant Neuron Fraction')
ax.legend(fontsize=9); _clean(ax)

# ── Row 1, Col 2: Weight Magnitude (Layer 1) ──
ax = axes[1, 2]
for m, d in all_results.items():
    s = _style(m)
    wmag = [d['task']['avg_weight_mag'][t][0] if t < len(d['task']['avg_weight_mag']) else 0
            for t in range(len(d['task']['task_acc']))]
    ax.plot(wmag, color=s['color'], lw=1.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Avg |W|'); ax.set_title('Weight Magnitude (Layer 1)')
ax.legend(fontsize=9); _clean(ax)

# ── Row 2, Col 0: Avg Approximate Rank ──
ax = axes[2, 0]
for m, d in all_results.items():
    s = _style(m)
    apr_avg = d['apr'].mean(dim=1).numpy()
    sm = smooth(apr_avg)
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Approx Rank'); ax.set_title('Avg Approximate Rank (all layers)')
ax.legend(fontsize=9); _clean(ax)

# ── Row 2, Col 1: Total Dead Neurons ──
ax = axes[2, 1]
for m, d in all_results.items():
    s = _style(m)
    dead_total = d['dead'].sum(dim=1).numpy()
    sm = smooth(dead_total)
    ax.plot(np.arange(len(sm)) * TASK_WINDOW, sm, color=s['color'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Dead Neurons'); ax.set_title('Total Dead Neurons (all layers)')
ax.legend(fontsize=9); _clean(ax)

# ── Row 2, Col 2: Summary Table ──
ax = axes[2, 2]
ax.axis('off')
n_final = 100
lines = ['Final 100-task metrics:', '─' * 44,
         f'{"":<18} {"SASSHA+SDP":>12} {"CBP":>12}']
for metric_name, metric_fn in [
    ('Test Acc (%)',  lambda d: np.array(d['task']['task_test_acc'][-n_final:]).mean() * 100),
    ('Train Acc (%)', lambda d: np.array(d['task']['task_train_acc'][-n_final:]).mean() * 100),
    ('Stable Rank',   lambda d: d['sr'][-n_final:].mean().item()),
    ('Dormant (%)',   lambda d: d['dormant'][-n_final:].mean().item() * 100),
    ('Dead Neurons',  lambda d: d['dead'][-n_final:].sum(dim=1).mean().item()),
    ('Approx Rank',   lambda d: d['apr'][-n_final:].mean().item()),
]:
    vals = []
    for m in ['sassha_sdp', 'cbp']:
        if m in all_results:
            vals.append(f'{metric_fn(all_results[m]):>12.2f}')
        else:
            vals.append(f'{"N/A":>12}')
    lines.append(f'{metric_name:<18} {vals[0]} {vals[1]}')
ax.text(0.5, 0.5, '\n'.join(lines), transform=ax.transAxes,
        fontsize=11, verticalalignment='center', horizontalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9C4', edgecolor='#FBC02D', alpha=0.85))

plt.tight_layout()
plot_file = os.path.join(RESULTS_DIR, 'sassha_vs_cbp_comparison.png')
plt.savefig(plot_file, dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Comparison plot saved to {plot_file}")

# %% [markdown]
# ## 12. Per-Layer Detail: Approximate Rank

# %%
fig2, axes2 = plt.subplots(2, NUM_HIDDEN_LAYERS, figsize=(5 * NUM_HIDDEN_LAYERS, 8), sharey='row')
fig2.suptitle('Approximate Rank per Hidden Layer — SASSHA+SDP vs CBP',
              fontsize=13, fontweight='bold')
for row_idx, (m, label) in enumerate([('sassha_sdp', 'SASSHA+SDP'), ('cbp', 'CBP')]):
    if m not in all_results: continue
    d = all_results[m]
    for li in range(NUM_HIDDEN_LAYERS):
        ax = axes2[row_idx, li]
        ax.plot(d['apr'][:, li].numpy(), color=_style(m)['color'], lw=1.5)
        ax.set_title(f'{label} — {LAYER_NAMES[li]}', fontsize=10)
        ax.set_xlabel('Task')
        if li == 0: ax.set_ylabel('Approx Rank')
        _clean(ax)
plt.tight_layout()
plot_file2 = os.path.join(RESULTS_DIR, 'rank_evolution_comparison.png')
plt.savefig(plot_file2, dpi=200, bbox_inches='tight'); plt.show()
print(f"✓ Rank evolution saved to {plot_file2}")

# %% [markdown]
# ## 13. Per-Layer Detail: Dead Neurons

# %%
fig3, axes3 = plt.subplots(2, NUM_HIDDEN_LAYERS, figsize=(5 * NUM_HIDDEN_LAYERS, 8), sharey='row')
fig3.suptitle('Dead Neurons per Hidden Layer — SASSHA+SDP vs CBP',
              fontsize=13, fontweight='bold')
for row_idx, (m, label) in enumerate([('sassha_sdp', 'SASSHA+SDP'), ('cbp', 'CBP')]):
    if m not in all_results: continue
    d = all_results[m]
    for li in range(NUM_HIDDEN_LAYERS):
        ax = axes3[row_idx, li]
        ax.plot(d['dead'][:, li].numpy(), color=_style(m)['color'], lw=1.5)
        ax.set_title(f'{label} — {LAYER_NAMES[li]}', fontsize=10)
        ax.set_xlabel('Task')
        if li == 0: ax.set_ylabel('Dead Neurons')
        _clean(ax)
plt.tight_layout()
plot_file3 = os.path.join(RESULTS_DIR, 'dead_neurons_comparison.png')
plt.savefig(plot_file3, dpi=200, bbox_inches='tight'); plt.show()
print(f"✓ Dead neurons saved to {plot_file3}")

# %% [markdown]
# ## 14. Summary

# %%
n_final = 100
print(f"\n{'='*75}")
print(f"  Comparison — Permuted MNIST — Final {n_final}-task avg:")
print(f"{'='*75}")
print(f"{'Metric':<20} {'SASSHA+SDP':>15} {'CBP':>15}")
print(f"{'─'*50}")
for label, fn in [
    ('Test Acc (%)',   lambda d: np.array(d['task']['task_test_acc'][-n_final:]).mean() * 100),
    ('Train Acc (%)',  lambda d: np.array(d['task']['task_train_acc'][-n_final:]).mean() * 100),
    ('Stable Rank',    lambda d: d['sr'][-n_final:].mean().item()),
    ('Dormant (%)',    lambda d: d['dormant'][-n_final:].mean().item() * 100),
    ('Dead Neurons',   lambda d: d['dead'][-n_final:].sum(dim=1).mean().item()),
    ('Approx Rank',    lambda d: d['apr'][-n_final:].mean().item()),
    ('Weight Mag',     lambda d: np.mean([d['task']['avg_weight_mag'][t][0]
                              for t in range(-n_final, 0) if t + len(d['task']['avg_weight_mag']) >= 0])),
]:
    vals = []
    for m in ['sassha_sdp', 'cbp']:
        if m in all_results:
            vals.append(f'{fn(all_results[m]):>15.2f}')
        else:
            vals.append(f'{"N/A":>15}')
    print(f"{label:<20} {vals[0]} {vals[1]}")
print(f"{'='*75}")
