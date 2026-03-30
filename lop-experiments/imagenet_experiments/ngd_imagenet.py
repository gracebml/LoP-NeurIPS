# %% [markdown]
# # K-FAC Natural Gradient Descent on Continual ImageNet-32
# #
# **NGD (K-FAC)** trên bài toán Continual ImageNet-32 — cùng setup với CBP paper.
# #
# **Benchmark**: Task-incremental binary classification, mỗi task 2 classes ngẫu nhiên
# từ 1000 classes ImageNet-32 (600 train + 100 test per class).
# Setup giống hệt paper (Dohare et al., 2023): 5000 tasks, 250 epochs/task.
# #
# **K-FAC**: Approximate Fisher ≈ A ⊗ G (Kronecker factors),
# preconditioned update: ΔW = −η · G⁻¹ ∇W L · A⁻¹
# #
# **Tracking per epoch**: dormant neurons (per-layer), stable rank, weight magnitude.
# #
# **Checkpoint**: Tự động lưu mỗi `SAVE_EVERY_N_TASKS` tasks + khi vượt time limit.

# %% [markdown]
# ## 1. Imports & Setup

# %%
import os, sys, json, time, math, copy, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer as _Optimizer
from tqdm import tqdm

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
# ## 2. K-FAC Optimizer
# #
# Kronecker-Factored Approximate Curvature (K-FAC) Natural Gradient Descent.
# - Approximates the Fisher Information Matrix for each layer as F ≈ A ⊗ G
# - A = E[x xᵀ] (input covariance), G = E[g gᵀ] (output gradient covariance)
# - Update rule: ΔW = −η · G⁻¹ (∇_W L) A⁻¹
# - Supports: nn.Linear, nn.Conv2d. Other layers fall back to SGD.

# %%
class KFACOptimizer(_Optimizer):
    """
    K-FAC (Kronecker-Factored Approximate Curvature) Optimizer.

    Approximates the Fisher Information Matrix for each layer as:
        F ≈ A ⊗ G
    where A = input covariance, G = output gradient covariance.

    Supports: nn.Linear, nn.Conv2d. Other layers fall back to SGD.
    """

    def __init__(self, model, lr=0.01, damping=1e-3, weight_decay=0.0,
                 T_inv=100, alpha=0.95, momentum=0.0):
        """
        Args:
            model: nn.Module to optimize.
            lr: Learning rate.
            damping: Tikhonov damping λ. sqrt(λ) added to each factor.
            weight_decay: L2 regularization coefficient.
            T_inv: Frequency (in steps) to recompute matrix inverses.
            alpha: Exponential moving average coefficient for A and G.
            momentum: SGD momentum for fallback layers (unused for K-FAC layers).
        """
        self.model = model
        self.damping = damping
        self.weight_decay = weight_decay
        self.T_inv = T_inv
        self.alpha = alpha
        self.steps = 0

        # Storage for Kronecker factors and their inverses
        self._modules_tracked = {}  # name -> module
        self._stats = {}            # name -> {'A': Tensor, 'G': Tensor}
        self._inv = {}              # name -> {'A_inv': Tensor, 'G_inv': Tensor}
        self._hooks = []

        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(model.parameters(), defaults)

        self._register_hooks()
        print(f"  K-FAC tracking {len(self._modules_tracked)} layers")

    def _register_hooks(self):
        """Register forward/backward hooks on Conv2d and Linear layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self._modules_tracked[name] = module
                if name not in self._stats:
                    self._stats[name] = {'A': None, 'G': None}
                h1 = module.register_forward_hook(self._forward_hook(name, module))
                h2 = module.register_full_backward_hook(self._backward_hook(name, module))
                self._hooks.append(h1)
                self._hooks.append(h2)

    def _forward_hook(self, name, module):
        """Capture input activations → compute A = E[x xᵀ]."""
        def hook(mod, inp, out):
            if not mod.training:
                return
            with torch.no_grad():
                x = inp[0].detach()
                if isinstance(mod, nn.Conv2d):
                    # im2col: unfold input patches → [B*H'*W', C_in*k*k]
                    x = F.unfold(x, mod.kernel_size, dilation=mod.dilation,
                                 padding=mod.padding, stride=mod.stride)
                    x = x.permute(0, 2, 1).reshape(-1, x.size(1))
                elif isinstance(mod, nn.Linear):
                    if x.dim() > 2:
                        x = x.reshape(-1, x.size(-1))

                # Append bias unit (column of 1s)
                if mod.bias is not None:
                    ones = torch.ones(x.size(0), 1, device=x.device)
                    x = torch.cat([x, ones], dim=1)

                # A = xᵀx / n
                n = x.size(0)
                cov_a = torch.matmul(x.t(), x) / n

                if self._stats[name]['A'] is None:
                    self._stats[name]['A'] = cov_a
                else:
                    self._stats[name]['A'].mul_(self.alpha).add_(cov_a, alpha=1 - self.alpha)
        return hook

    def _backward_hook(self, name, module):
        """Capture output gradients → compute G = E[g gᵀ]."""
        def hook(mod, grad_input, grad_output):
            if not mod.training:
                return
            with torch.no_grad():
                g = grad_output[0].detach()
                if isinstance(mod, nn.Conv2d):
                    g = g.permute(0, 2, 3, 1).reshape(-1, g.size(1))
                elif isinstance(mod, nn.Linear):
                    if g.dim() > 2:
                        g = g.reshape(-1, g.size(-1))

                n = g.size(0)
                cov_g = torch.matmul(g.t(), g) / n

                if self._stats[name]['G'] is None:
                    self._stats[name]['G'] = cov_g
                else:
                    self._stats[name]['G'].mul_(self.alpha).add_(cov_g, alpha=1 - self.alpha)
        return hook

    @torch.no_grad()
    def _invert_factors(self):
        """Invert A and G with damping. Called every T_inv steps."""
        # Skip inversion during warmup (first T_inv steps) — stats too noisy
        if self.steps < self.T_inv:
            return
        sqrt_damping = self.damping ** 0.5
        for name in self._stats:
            A = self._stats[name]['A']
            G = self._stats[name]['G']
            if A is None or G is None:
                continue
            try:
                A_d = A + sqrt_damping * torch.eye(A.size(0), device=A.device)
                G_d = G + sqrt_damping * torch.eye(G.size(0), device=G.device)
                self._inv[name] = {
                    'A_inv': torch.linalg.inv(A_d),
                    'G_inv': torch.linalg.inv(G_d)
                }
            except RuntimeError:
                pass  # Keep previous inverse if singular

    def reset_stats(self):
        """Reset running statistics (call at task boundaries)."""
        for name in self._stats:
            self._stats[name] = {'A': None, 'G': None}
        self._inv.clear()
        self.steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Periodically recompute inverses
        if self.steps % self.T_inv == 0:
            self._invert_factors()

        lr = self.param_groups[0]['lr']

        # Apply K-FAC update to tracked layers
        for name, module in self._modules_tracked.items():
            if module.weight.grad is None:
                continue

            grad_w = module.weight.grad

            if name in self._inv:
                A_inv = self._inv[name]['A_inv']
                G_inv = self._inv[name]['G_inv']

                if isinstance(module, nn.Conv2d):
                    c_out = grad_w.size(0)
                    grad_2d = grad_w.reshape(c_out, -1)

                    if module.bias is not None and module.bias.grad is not None:
                        grad_2d = torch.cat([grad_2d, module.bias.grad.unsqueeze(1)], dim=1)

                    nat_grad = torch.matmul(G_inv, torch.matmul(grad_2d, A_inv))

                    if module.bias is not None and module.bias.grad is not None:
                        nat_grad_w = nat_grad[:, :-1].reshape_as(module.weight)
                        nat_grad_b = nat_grad[:, -1]
                        if self.weight_decay > 0:
                            nat_grad_w.add_(module.weight, alpha=self.weight_decay)
                        module.weight.data.add_(nat_grad_w, alpha=-lr)
                        module.bias.data.add_(nat_grad_b, alpha=-lr)
                    else:
                        nat_grad = nat_grad.reshape_as(module.weight)
                        if self.weight_decay > 0:
                            nat_grad.add_(module.weight, alpha=self.weight_decay)
                        module.weight.data.add_(nat_grad, alpha=-lr)

                elif isinstance(module, nn.Linear):
                    if module.bias is not None and module.bias.grad is not None:
                        grad_2d = torch.cat([grad_w, module.bias.grad.unsqueeze(1)], dim=1)
                    else:
                        grad_2d = grad_w

                    nat_grad = torch.matmul(G_inv, torch.matmul(grad_2d, A_inv))

                    if module.bias is not None and module.bias.grad is not None:
                        nat_grad_w = nat_grad[:, :-1]
                        nat_grad_b = nat_grad[:, -1]
                        if self.weight_decay > 0:
                            nat_grad_w.add_(module.weight, alpha=self.weight_decay)
                        module.weight.data.add_(nat_grad_w, alpha=-lr)
                        module.bias.data.add_(nat_grad_b, alpha=-lr)
                    else:
                        if self.weight_decay > 0:
                            nat_grad.add_(module.weight, alpha=self.weight_decay)
                        module.weight.data.add_(nat_grad, alpha=-lr)
            else:
                # No inverse available yet → plain SGD fallback
                if self.weight_decay > 0:
                    grad_w = grad_w + self.weight_decay * module.weight
                module.weight.data.add_(grad_w, alpha=-lr)
                if module.bias is not None and module.bias.grad is not None:
                    module.bias.data.add_(module.bias.grad, alpha=-lr)

        # SGD for non-tracked parameters (if any)
        tracked_params = set()
        for mod in self._modules_tracked.values():
            tracked_params.add(id(mod.weight))
            if mod.bias is not None:
                tracked_params.add(id(mod.bias))

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or id(p) in tracked_params:
                    continue
                if self.weight_decay > 0:
                    p.data.add_(p, alpha=-self.weight_decay * group['lr'])
                p.data.add_(p.grad, alpha=-group['lr'])

        self.steps += 1
        return loss


print("✓ KFACOptimizer defined (supports Conv2d + Linear)")

# %% [markdown]
# ## 3. Plasticity Metrics (Enhanced)
# #
# **Dormant neurons**: per-layer tracking — conv1(32), conv2(64), conv3(128), fc1(128), fc2(128).
# A neuron is dormant if active for < `threshold` fraction of samples.
# **Stable rank**: effective rank of penultimate layer activations.
# **Per-neuron alive scores**: for tracking dormant persistence across tasks.

# %%
def _replace_relu_with_leaky(net, negative_slope=0.01):
    """Replace all nn.ReLU in net.layers with nn.LeakyReLU."""
    for i, layer in enumerate(net.layers):
        if isinstance(layer, nn.ReLU):
            net.layers[i] = nn.LeakyReLU(negative_slope=negative_slope)
    net.act_type = 'leaky_relu'
    return net

NUM_LAYERS = 5  # conv1, conv2, conv3(flat), fc1, fc2
LAYER_SIZES = [32, 64, 512, 128, 128]  # conv3 is flattened: 128*2*2=512
TOTAL_NEURONS = sum(LAYER_SIZES)  # 864

@torch.no_grad()
def compute_dormant_neurons_enhanced(net, x_data, mini_batch_size=100, threshold=0.01):
    """Compute per-layer dormant fraction and per-neuron alive scores.

    Returns:
        aggregate_frac: float — overall dormant fraction
        per_layer_frac: list of 5 floats — dormant fraction per layer
        alive_scores: np.array (TOTAL_NEURONS,) — alive fraction for each neuron
        last_layer_act: np.array — activations from last hidden layer (for stable rank)
    """
    batch_x = x_data[:min(mini_batch_size * 5, len(x_data))]
    _, activations = net.predict(x=batch_x)
    # activations: [x1(conv1), x2(conv2), x3(conv3), x4(fc1), x5(fc2)]

    per_layer_frac = []
    all_alive_scores = []
    total_neurons = 0
    total_dormant = 0
    last_layer_act = None

    for i, act in enumerate(activations):
        if act.ndim == 4:  # conv: (B, C, H, W)
            alive_frac = (act.abs() > 1e-5).float().mean(dim=(0, 2, 3))  # per channel
            n_units = act.shape[1]
        else:  # fc: (B, D)
            alive_frac = (act.abs() > 1e-5).float().mean(dim=0)  # per unit
            n_units = act.shape[1]

        dormant = (alive_frac < threshold).sum().item()
        per_layer_frac.append(dormant / n_units if n_units > 0 else 0.0)
        all_alive_scores.append(alive_frac.cpu().numpy())
        total_dormant += dormant
        total_neurons += n_units

        if i == len(activations) - 1:
            last_layer_act = act.cpu().numpy()

    aggregate_frac = total_dormant / total_neurons if total_neurons > 0 else 0.0
    alive_scores = np.concatenate(all_alive_scores)  # (TOTAL_NEURONS,)

    return aggregate_frac, per_layer_frac, alive_scores, last_layer_act


def compute_stable_rank(sv):
    """Stable rank from singular values: #(cumsum < 99% of total)."""
    if len(sv) == 0:
        return 0
    sorted_sv = np.flip(np.sort(sv))
    cumsum = np.cumsum(sorted_sv) / np.sum(sv)
    return int(np.sum(cumsum < 0.99) + 1)


def compute_stable_rank_from_activations(act):
    """Stable rank from activation matrix (samples × features)."""
    from scipy.linalg import svd
    if act is None:
        return 0
    if act.ndim > 2:
        act = act.reshape(act.shape[0], -1)
    if act.shape[0] == 0 or act.shape[1] == 0:
        return 0
    try:
        sv = svd(act, compute_uv=False, lapack_driver='gesvd')
        return compute_stable_rank(sv)
    except Exception:
        return 0


print("✓ Enhanced plasticity metrics defined (per-layer dormant + stable rank)")

# %% [markdown]
# ## 4. ImageNet-32 Data Loading
# #
# ImageNet-32: 1000 classes, 600 train + 100 test per class.
# Data stored as `.npy` files — one per class (same format as `single_expr.py`).

# %%
TRAIN_IMAGES_PER_CLASS = 600
TEST_IMAGES_PER_CLASS = 100
IMAGES_PER_CLASS = TRAIN_IMAGES_PER_CLASS + TEST_IMAGES_PER_CLASS
TOTAL_CLASSES = 1000

DATA_DIR = '/kaggle/input/datasets/nguyenlamphuquy/imagenet/classes'

if DATA_DIR is None:
    raise FileNotFoundError("ImageNet-32 classes not found.")
print(f"✓ ImageNet-32 data dir: {DATA_DIR}")

# ── Class order from pickle file (same as original single_expr.py) ──
_class_order_file = os.path.join(_LOP_IMAGENET_DIR, 'class_order')
if os.path.isfile(_class_order_file):
    with open(_class_order_file, 'rb') as f:
        _ALL_CLASS_ORDERS = pickle.load(f)
    print(f"  ✓ Loaded class_order ({len(_ALL_CLASS_ORDERS)} runs)")
else:
    # Fallback: generate a single random order
    print(f"  ⚠ class_order not found at {_class_order_file}, generating random order")
    _rng = np.random.RandomState(42)
    _ALL_CLASS_ORDERS = [_rng.permutation(TOTAL_CLASSES) for _ in range(30)]


def load_imagenet(classes=[]):
    """Load ImageNet-32 data for given class indices. Same as single_expr.py."""
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = os.path.join(DATA_DIR, str(_class) + '.npy')
        new_x = np.load(data_file)
        x_train.append(new_x[:TRAIN_IMAGES_PER_CLASS])
        x_test.append(new_x[TRAIN_IMAGES_PER_CLASS:])
        y_train.append(np.array([idx] * TRAIN_IMAGES_PER_CLASS))
        y_test.append(np.array([idx] * TEST_IMAGES_PER_CLASS))
    x_train = torch.tensor(np.concatenate(x_train))
    y_train = torch.from_numpy(np.concatenate(y_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_test = torch.from_numpy(np.concatenate(y_test))
    return x_train, y_train, x_test, y_test


def save_data(data, data_file):
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    with open(data_file, 'wb+') as f:
        pickle.dump(data, f)


print(f"✓ ImageNet-32 data loading ready ({TOTAL_CLASSES} classes, "
      f"{TRAIN_IMAGES_PER_CLASS} train + {TEST_IMAGES_PER_CLASS} test per class)")

# %% [markdown]
# ## 5. Experiment Config
# #
# Setup giống hệt CBP paper (Dohare et al., 2023) — `cbp.json`:
# 5000 tasks, 2 classes/task, 250 epochs/task, binary classification.

# %%
PARAMS = {
    # ── Problem (matching cbp.json exactly) ──
    'num_runs': 1,
    'num_tasks': 5000,              # same as cbp.json
    'num_classes': 2,               # binary classification
    'num_showings': 250,            # 250 epochs per task (same as cbp.json)
    'mini_batch_size': 100,
    # ── K-FAC optimizer ──
    'step_size': 0.001,             # learning rate (lower than SGD — K-FAC preconditions)
    'damping': 0.1,                 # λ: sqrt(λ) added to A and G (large for stability)
    'weight_decay': 0.0,            # matching cbp.json (no weight decay)
    'T_inv': 50,                    # recompute A⁻¹, G⁻¹ every T_inv steps
    'alpha': 0.95,                  # EMA coefficient for running A and G stats
    'momentum': 0.9,                # SGD momentum (matching paper)
}

# ── Output directories ──
RESULTS_DIR = os.path.join('kaggle/working', 'data', 'ngd_imagenet')
CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Time limit for checkpoint (11.5 hours → buffer 30 min for 12h sessions) ──
TIME_LIMIT_SECONDS = 11.5 * 3600  # 41400 seconds
SAVE_EVERY_N_TASKS = 50           # checkpoint every 50 tasks

print(f"✓ Config (matching cbp.json): {PARAMS['num_tasks']} tasks × {PARAMS['num_showings']} epochs, "
      f"{PARAMS['num_classes']} classes/task (ImageNet-32, {TOTAL_CLASSES} total)")
print(f"  K-FAC: lr={PARAMS['step_size']}, damping={PARAMS['damping']}, "
      f"T_inv={PARAMS['T_inv']}, alpha={PARAMS['alpha']}, wd={PARAMS['weight_decay']}")
print(f"  Time limit: {TIME_LIMIT_SECONDS/3600:.1f}h, save every {SAVE_EVERY_N_TASKS} tasks")
print(f"  Results dir: {RESULTS_DIR}")

# %% [markdown]
# ## 6. Training Loop — K-FAC NGD on Continual ImageNet-32

# %%
def run_ngd(params, run_idx=0):
    """Run K-FAC Natural Gradient Descent on Continual ImageNet-32.

    Same protocol as CBP's single_expr.py:
    - Binary classification: 2 classes per task
    - Head reset at task boundaries
    - ConvNet architecture (3 conv + 3 fc)
    - K-FAC stats reset at task boundaries (stale after distribution shift)
    - Enhanced per-epoch tracking: per-layer dormant neurons, stable rank, weight magnitude
    """
    print(f"\n{'='*70}")
    print(f"  K-FAC NGD — Run {run_idx} (ImageNet-32)")
    print(f"{'='*70}")

    wall_clock_start = time.time()

    num_tasks = params['num_tasks']
    num_epochs = params['num_showings']
    num_classes = params['num_classes']
    mini_batch_size = params['mini_batch_size']
    classes_per_task = num_classes
    examples_per_epoch = TRAIN_IMAGES_PER_CLASS * classes_per_task

    # ── Class order from pickle (same as original single_expr.py) ──
    class_order = _ALL_CLASS_ORDERS[run_idx % len(_ALL_CLASS_ORDERS)]
    num_class_repetitions = int(num_classes * num_tasks / TOTAL_CLASSES) + 1
    class_order = np.concatenate([class_order] * num_class_repetitions)

    # ── Build network (same ConvNet as CBP) ──
    net = ConvNet(num_classes=classes_per_task)
    _replace_relu_with_leaky(net)

    # ── Build K-FAC optimizer ──
    # Temporarily remove hooks during forward-only evaluation by toggling net.training
    optimizer = KFACOptimizer(
        net,
        lr=params['step_size'],
        damping=params.get('damping', 1e-3),
        weight_decay=params.get('weight_decay', 0.0),
        T_inv=params.get('T_inv', 100),
        alpha=params.get('alpha', 0.95),
        momentum=params.get('momentum', 0.9),
    )

    loss_fn = F.cross_entropy

    # ── Metrics — per-epoch tracking ──
    train_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    test_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    all_weight_mag = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    all_dormant_frac = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    all_dormant_per_layer = torch.zeros((num_tasks, num_epochs, NUM_LAYERS), dtype=torch.float)
    all_stable_rank = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    # ── Dormant persistence: which neurons are dormant at end of each task ──
    dormant_persistence = torch.zeros((num_tasks, TOTAL_NEURONS), dtype=torch.bool)

    # ── Checkpoint resume ──
    ckpt_file = os.path.join(CKPT_DIR, f'ckpt_ngd_run{run_idx}.pt')
    start_task = 0

    if os.path.isfile(ckpt_file):
        print(f"  Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        train_accuracies = ckpt['train_accuracies']
        test_accuracies = ckpt['test_accuracies']
        all_weight_mag = ckpt.get('all_weight_mag', torch.zeros((num_tasks, num_epochs)))
        all_dormant_frac = ckpt.get('all_dormant_frac', torch.zeros((num_tasks, num_epochs)))
        all_dormant_per_layer = ckpt.get('all_dormant_per_layer', torch.zeros((num_tasks, num_epochs, NUM_LAYERS)))
        all_stable_rank = ckpt.get('all_stable_rank', torch.zeros((num_tasks, num_epochs)))
        dormant_persistence = ckpt.get('dormant_persistence', torch.zeros((num_tasks, TOTAL_NEURONS), dtype=torch.bool))
        start_task = ckpt['task_idx'] + 1
        # Restore RNG states for reproducible resume
        if 'np_random_state' in ckpt:
            np.random.set_state(ckpt['np_random_state'])
        if 'torch_rng_state' in ckpt:
            torch.random.set_rng_state(ckpt['torch_rng_state'])
        if device.type == 'cuda' and 'torch_cuda_rng_state' in ckpt:
            torch.cuda.set_rng_state(ckpt['torch_cuda_rng_state'])
        print(f"  ✓ Resumed from task {start_task}")
        del ckpt
        torch.cuda.empty_cache()
    else:
        print(f"  (no checkpoint, training from scratch)")

    # ── Save helper ──
    def save_checkpoint(task_idx, reason="periodic"):
        ckpt_data = {
            'task_idx': task_idx,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'all_weight_mag': all_weight_mag,
            'all_dormant_frac': all_dormant_frac,
            'all_dormant_per_layer': all_dormant_per_layer,
            'all_stable_rank': all_stable_rank,
            'dormant_persistence': dormant_persistence,
            'np_random_state': np.random.get_state(),
            'torch_rng_state': torch.random.get_rng_state(),
            'params': params,
        }
        if device.type == 'cuda':
            ckpt_data['torch_cuda_rng_state'] = torch.cuda.get_rng_state()
        torch.save(ckpt_data, ckpt_file)
        elapsed = time.time() - wall_clock_start
        print(f"  Checkpoint saved at task {task_idx} ({reason}) [{elapsed/3600:.1f}h elapsed]")

    # ── Main training loop ──
    x_train, x_test, y_train, y_test = None, None, None, None
    time_limit_hit = False

    for task_idx in range(start_task, num_tasks):
        task_start = time.time()

        # Check wall-clock time limit
        elapsed = time.time() - wall_clock_start
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"\n  Time limit reached ({elapsed/3600:.1f}h). Saving and stopping.")
            save_checkpoint(task_idx - 1, reason="time_limit")
            time_limit_hit = True
            break

        # ── Load data for this task ──
        del x_train, x_test, y_train, y_test
        task_classes = class_order[task_idx * classes_per_task:(task_idx + 1) * classes_per_task]
        x_train, y_train, x_test, y_test = load_imagenet(task_classes)
        x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
        if device.type == 'cuda':
            x_train, x_test = x_train.to(device), x_test.to(device)
            y_train, y_test = y_train.to(device), y_test.to(device)

        # ── Head reset at task boundary (same as CBP) ──
        net.layers[-1].weight.data *= 0
        net.layers[-1].bias.data *= 0

        # ── Reset K-FAC stats at task boundary (stale after distribution shift) ──
        optimizer.reset_stats()

        # ── Train epochs ──
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

                # ── Forward + backward (hooks auto-collect A and G) ──
                optimizer.zero_grad()
                output, _ = net.predict(x=batch_x)
                loss = loss_fn(output, batch_y)
                loss.backward()

                # ── K-FAC step: preconditioned update ──
                # Hooks already captured A (input cov) and G (grad cov)
                # Inverses recomputed every T_inv steps automatically
                optimizer.step()

                # Track train accuracy
                with torch.no_grad():
                    new_train_accs[batch_iter] = accuracy(
                        F.softmax(output.detach(), dim=1), batch_y).cpu()
                    batch_iter += 1

            # ── Eval ──
            with torch.no_grad():
                train_accuracies[task_idx][epoch_idx] = new_train_accs.mean()

                # Temporarily remove hooks so they don't fire during eval
                for h in optimizer._hooks:
                    h.remove()
                optimizer._hooks.clear()

                net.eval()
                new_test_accs = torch.zeros(x_test.shape[0] // mini_batch_size, dtype=torch.float)
                test_iter = 0
                for start_idx in range(0, x_test.shape[0], mini_batch_size):
                    test_batch_x = x_test[start_idx:start_idx + mini_batch_size]
                    test_batch_y = y_test[start_idx:start_idx + mini_batch_size]
                    test_output, _ = net.predict(x=test_batch_x)
                    new_test_accs[test_iter] = accuracy(
                        F.softmax(test_output, dim=1), test_batch_y)
                    test_iter += 1
                test_accuracies[task_idx][epoch_idx] = new_test_accs.mean()

            # ── Per-epoch plasticity metrics ──
            with torch.no_grad():
                n_params = sum(p.numel() for p in net.parameters())
                avg_w = sum(p.data.abs().sum().item() for p in net.parameters()) / n_params
                all_weight_mag[task_idx, epoch_idx] = avg_w

                agg_frac, layer_fracs, alive_scores, last_act = compute_dormant_neurons_enhanced(
                    net, x_test, mini_batch_size=mini_batch_size)
                all_dormant_frac[task_idx, epoch_idx] = agg_frac
                all_dormant_per_layer[task_idx, epoch_idx] = torch.tensor(layer_fracs)

                sr = compute_stable_rank_from_activations(last_act)
                all_stable_rank[task_idx, epoch_idx] = sr

                # Per-epoch logging every 50 epochs
                if epoch_idx % 50 == 0 or epoch_idx == num_epochs - 1:
                    layer_str = ' '.join([f'{f:.2f}' for f in layer_fracs])
                    print(f"    Task {task_idx} Epoch {epoch_idx:3d}/{num_epochs} | "
                          f"TrainAcc={train_accuracies[task_idx][epoch_idx]:.4f} "
                          f"TestAcc={test_accuracies[task_idx][epoch_idx]:.4f} "
                          f"Dormant={agg_frac:.3f} [{layer_str}] "
                          f"SR={sr:.0f} AvgW={avg_w:.4f}")

                # Record dormant persistence at last epoch of each task
                if epoch_idx == num_epochs - 1:
                    dormant_persistence[task_idx] = torch.tensor(alive_scores < 0.01)

            # ── Re-register K-FAC hooks for next training step ──
            net.train()
            optimizer._register_hooks()

        task_time = time.time() - task_start
        if task_idx % 50 == 0 or task_idx == num_tasks - 1:
            layer_str = ' '.join([f'{f:.2f}' for f in all_dormant_per_layer[task_idx, -1].tolist()])
            print(f"  Task {task_idx:4d}/{num_tasks} | "
                  f"TrainAcc={train_accuracies[task_idx][-1]:.4f} "
                  f"TestAcc={test_accuracies[task_idx][-1]:.4f} "
                  f"AvgW={all_weight_mag[task_idx, -1]:.4f} "
                  f"Dormant={all_dormant_frac[task_idx, -1]:.3f} [{layer_str}] "
                  f"StableRank={all_stable_rank[task_idx, -1]:.0f} | {task_time:.1f}s")

        # ── Periodic checkpoint ──
        if (task_idx + 1) % SAVE_EVERY_N_TASKS == 0:
            save_checkpoint(task_idx, reason="periodic")

    # ── Final save ──
    if not time_limit_hit:
        save_checkpoint(num_tasks - 1, reason="completed")

    # ── Save results ──
    result_subdir = os.path.join(RESULTS_DIR, '0')
    os.makedirs(result_subdir, exist_ok=True)
    data_file = os.path.join(result_subdir, str(run_idx))
    save_data(data={
        'train_accuracies': train_accuracies.cpu(),
        'test_accuracies': test_accuracies.cpu(),
        'all_weight_mag': all_weight_mag.cpu(),
        'all_dormant_frac': all_dormant_frac.cpu(),
        'all_dormant_per_layer': all_dormant_per_layer.cpu(),
        'all_stable_rank': all_stable_rank.cpu(),
        'dormant_persistence': dormant_persistence.cpu(),
    }, data_file=data_file)
    print(f"  ✓ Results saved to {data_file}")

    return train_accuracies, test_accuracies, all_weight_mag, all_dormant_frac, all_dormant_per_layer, all_stable_rank, dormant_persistence


print("✓ K-FAC NGD training loop defined")

# %% [markdown]
# ## 7. Run Experiment

# %%
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  Running K-FAC NGD on Continual ImageNet-32")
    print("="*70)

    results_ngd = run_ngd(PARAMS, run_idx=0)

    print("\n" + "="*70)
    print("  ✓ Experiment completed!")
    print("="*70)

# %% [markdown]
# ## 8. Quick Summary Plot

# %%
import matplotlib.pyplot as plt

def plot_quick_summary(results_dir):
    """Plot test accuracy and dormant fraction from saved results."""
    data_file = os.path.join(results_dir, '0', '0')
    if not os.path.isfile(data_file):
        print(f"No results found at {data_file}")
        return

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    test_acc = data['test_accuracies']
    dormant = data['all_dormant_frac']
    stable_rank = data['all_stable_rank']
    weight_mag = data['all_weight_mag']

    # Use last epoch of each task
    test_acc_per_task = test_acc[:, -1].numpy()
    dormant_per_task = dormant[:, -1].numpy()
    sr_per_task = stable_rank[:, -1].numpy()
    wm_per_task = weight_mag[:, -1].numpy()

    # Only plot non-zero tasks
    n_tasks_done = (test_acc_per_task != 0).sum()
    if n_tasks_done == 0:
        print("No tasks completed yet")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-FAC NGD on Continual ImageNet-32', fontsize=14)

    axes[0, 0].plot(test_acc_per_task[:n_tasks_done])
    axes[0, 0].set_xlabel('Task')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Test Accuracy per Task (last epoch)')
    axes[0, 0].grid(True)

    axes[0, 1].plot(dormant_per_task[:n_tasks_done])
    axes[0, 1].set_xlabel('Task')
    axes[0, 1].set_ylabel('Dormant Fraction')
    axes[0, 1].set_title('Dormant Neuron Fraction')
    axes[0, 1].grid(True)

    axes[1, 0].plot(sr_per_task[:n_tasks_done])
    axes[1, 0].set_xlabel('Task')
    axes[1, 0].set_ylabel('Stable Rank')
    axes[1, 0].set_title('Stable Rank (penultimate layer)')
    axes[1, 0].grid(True)

    axes[1, 1].plot(wm_per_task[:n_tasks_done])
    axes[1, 1].set_xlabel('Task')
    axes[1, 1].set_ylabel('Avg |W|')
    axes[1, 1].set_title('Average Weight Magnitude')
    axes[1, 1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'ngd_imagenet_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Summary plot saved to {save_path}")


if __name__ == '__main__':
    plot_quick_summary(RESULTS_DIR)
