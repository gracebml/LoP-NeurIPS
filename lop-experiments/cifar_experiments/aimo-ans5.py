# %% [markdown]
# # Noisy K-FAC (Natural Gradient Descent) for Continual Learning
#
# This notebook implements **Noisy K-FAC** (Zhang et al., 2017) — a variational
# inference method that treats K-FAC natural gradient with adaptive weight noise
# as fitting a variational posterior to maximize the ELBO.
#
# **Key improvement over standard K-FAC**: Built-in Bayesian regularization via
# weight noise sampling from the posterior $q(W) = \mathcal{N}(\mu, U_c U_c^T \otimes V_c V_c^T)$,
# where $U_c, V_c$ are derived from K-FAC Fisher factors.
#
# **Experiment**: Incremental CIFAR-100 with ResNet-18 (same protocol as CBP paper).

# %% [markdown]
# ## 1. Import Required Libraries and Setup

# %%
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# %%
pip install mlproj-manager==0.0.29

# %%
import os
import sys
import json
import time
import pickle
import copy
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("/kaggle/input/lop-src")
sys.path.append("/kaggle/input/lop-src/lop")
from lop.nets.torchvision_modified_resnet import build_resnet18, kaiming_init_resnet_module
from lop.utils.miscellaneous import nll_accuracy, compute_matrix_rank_summaries

from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import (
    ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

print("✓ Libraries imported successfully")

# %% [markdown]
# ## 2. Configuration and Utility Functions

# %%
results_dir = "notebook_results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "cifar100"), exist_ok=True)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

def plot_learning_curve(accuracies, title, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.xlabel('Steps / Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

print("✓ Utility functions defined")

# %% [markdown]
# ## 3. Load CIFAR-100 Dataset

# %%
cifar_config_path = '/kaggle/input/lop-src/lop/lop/incremental_cifar/cfg/continual_backpropagation.json'
cifar_config = load_config(cifar_config_path)
cifar_config["data_path"] = (lambda p: (os.makedirs(p, exist_ok=True), p)[1])("/kaggle/working/incremental_cifar/data")
cifar_config['results_dir'] = os.path.join(results_dir, 'cifar100')
cifar_config['num_workers'] = 2

print("CIFAR-100 Configuration loaded")

# %%
print("Loading CIFAR-100 dataset with mlproj_manager...")

os.makedirs(cifar_config['data_path'], exist_ok=True)

mean = (0.5071, 0.4865, 0.4409)
std = (0.2673, 0.2564, 0.2762)

train_transformations = transforms.Compose([
    ToTensor(swap_color_axis=True),
    Normalize(mean=mean, std=std),
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=32, padding=4, padding_mode="reflect"),
    RandomRotator(degrees=(0, 15))
])

eval_transformations = transforms.Compose([
    ToTensor(swap_color_axis=True),
    Normalize(mean=mean, std=std)
])

train_data_full = CifarDataSet(
    root_dir=cifar_config['data_path'], train=True, cifar_type=100,
    device=None, image_normalization="max", label_preprocessing="one-hot", use_torch=True
)

test_data = CifarDataSet(
    root_dir=cifar_config['data_path'], train=False, cifar_type=100,
    device=None, image_normalization="max", label_preprocessing="one-hot", use_torch=True
)

def get_validation_and_train_indices(cifar_data, num_classes=100):
    num_val_per_class = 50
    num_train_per_class = 450
    val_indices = torch.zeros(5000, dtype=torch.int32)
    train_indices = torch.zeros(45000, dtype=torch.int32)
    current_val = 0
    current_train = 0
    for i in range(num_classes):
        class_indices = torch.argwhere(cifar_data.data["labels"][:, i] == 1).flatten()
        val_indices[current_val:current_val + num_val_per_class] = class_indices[:num_val_per_class]
        train_indices[current_train:current_train + num_train_per_class] = class_indices[num_val_per_class:]
        current_val += num_val_per_class
        current_train += num_train_per_class
    return train_indices, val_indices

train_indices, val_indices = get_validation_and_train_indices(train_data_full)

def subsample_cifar(indices, cifar_data):
    cifar_data.data["data"] = cifar_data.data["data"][indices.numpy()]
    cifar_data.data["labels"] = cifar_data.data["labels"][indices.numpy()]
    cifar_data.integer_labels = torch.tensor(cifar_data.integer_labels)[indices.numpy()].tolist()
    cifar_data.current_data = cifar_data.partition_data()

train_data = copy.deepcopy(train_data_full)
val_data = copy.deepcopy(train_data_full)

subsample_cifar(train_indices, train_data)
subsample_cifar(val_indices, val_data)

train_data.set_transformation(train_transformations)
val_data.set_transformation(eval_transformations)
test_data.set_transformation(eval_transformations)

print(f"✓ CIFAR-100 loaded: Train={len(train_data.data['data'])}, Val={len(val_data.data['data'])}, Test={len(test_data.data['data'])}")

# %% [markdown]
# ## 4. Metrics Functions

# %%
from lop.incremental_cifar.post_run_analysis import compute_dormant_units_proportion

def compute_stable_rank(singular_values):
    if len(singular_values) == 0:
        return 0
    sorted_sv = np.flip(np.sort(singular_values))
    cumsum_sv = np.cumsum(sorted_sv) / np.sum(singular_values)
    return np.sum(cumsum_sv < 0.99) + 1

def compute_stable_rank_from_activations(activations):
    try:
        if activations.ndim > 2:
            activations = activations.reshape(activations.shape[0], -1)
        if activations.shape[0] == 0 or activations.shape[1] == 0:
            return 0
        from scipy.linalg import svd
        singular_values = svd(activations, compute_uv=False, lapack_driver="gesvd")
        return compute_stable_rank(singular_values)
    except Exception as e:
        print(f"Warning: SVD failed: {e}")
        return 0

@torch.no_grad()
def compute_avg_weight_magnitude(network):
    num_weights = 0
    sum_weight_magnitude = 0.0
    for param in network.parameters():
        num_weights += param.numel()
        sum_weight_magnitude += torch.sum(torch.abs(param)).item()
    if num_weights == 0:
        return 0.0
    return sum_weight_magnitude / num_weights

@torch.no_grad()
def compute_effective_rank(matrix):
    """
    Compute effective rank via spectral entropy.
    eRank(M) = exp(H(p)) where p_i = sigma_i / sum(sigma)
    
    This measures how many eigenvalue directions are 'active'.
    - Full rank d matrix → eRank ≈ d
    - Rank-1 matrix → eRank ≈ 1
    - Collapsing eRank = NTK/Fisher losing diversity
    """
    if matrix is None or matrix.numel() == 0:
        return 0.0
    eigvals = torch.linalg.eigvalsh(matrix)
    eigvals = eigvals.clamp(min=1e-10)  # numerical stability
    p = eigvals / eigvals.sum()
    entropy = -(p * p.log()).sum().item()
    return np.exp(entropy)

@torch.no_grad()
def compute_fisher_erank_stats(optimizer):
    """
    Compute effective rank of K-FAC Fisher factors A and G for all tracked layers.
    Returns: dict with per-layer and aggregate stats.
    
    Why this tracks NTK rank:
      Fisher ≈ E[J^T J] where J is the Jacobian (= NTK kernel)
      K-FAC approximates Fisher_layer ≈ A ⊗ G
      If A or G become low-rank → Fisher is low-rank → NTK collapse
    """
    eranks_A = []
    eranks_G = []
    cond_numbers = []
    layer_details = []
    
    for name in optimizer._stats:
        A = optimizer._stats[name]['A']
        G = optimizer._stats[name]['G']
        if A is None or G is None:
            continue
        
        er_A = compute_effective_rank(A)
        er_G = compute_effective_rank(G)
        eranks_A.append(er_A)
        eranks_G.append(er_G)
        
        # Condition number: high = ill-conditioned = hard to learn diverse features
        eigvals_A = torch.linalg.eigvalsh(A)
        cond = (eigvals_A.max() / eigvals_A.clamp(min=1e-10).min()).item()
        cond_numbers.append(cond)
        
        # Relative erank: erank / matrix_dim (0-1 scale)
        rel_A = er_A / A.size(0) * 100
        rel_G = er_G / G.size(0) * 100
        layer_details.append({
            'name': name, 'erank_A': er_A, 'erank_G': er_G,
            'rel_A': rel_A, 'rel_G': rel_G,
            'dim_A': A.size(0), 'dim_G': G.size(0), 'cond': cond
        })
    
    # Aggregate: mean relative erank across layers
    mean_rel_A = np.mean([d['rel_A'] for d in layer_details]) if layer_details else 0
    mean_rel_G = np.mean([d['rel_G'] for d in layer_details]) if layer_details else 0
    mean_cond = np.mean(cond_numbers) if cond_numbers else 0
    
    return {
        'mean_rel_erank_A': mean_rel_A,
        'mean_rel_erank_G': mean_rel_G,
        'mean_condition_number': mean_cond,
        'per_layer': layer_details
    }

# %% [markdown]
# ## 5. Noisy K-FAC Optimizer Definition
#
# Based on Zhang et al. (2017) "Noisy Natural Gradient as Variational Inference"
#
# Key differences from standard K-FAC:
# 1. **Weight noise sampling**: $W = \mu + U_c \epsilon V_c^T$ where $\epsilon \sim \mathcal{N}(0, I)$
# 2. **Noise covariance** derived from Fisher factors via eigendecomposition with pi-correction
# 3. **KL loss term** replaces weight decay: $\frac{\lambda}{N \cdot \eta} \|W\|^2$
# 4. **MC averaging** at test time for better uncertainty estimates

# %%
from torch.optim import Optimizer as _Optimizer

class NoisyKFACOptimizer(_Optimizer):
    """
    Noisy K-FAC Optimizer (Zhang et al., 2017).

    Extends K-FAC with adaptive weight noise for variational inference.
    Samples weights from matrix-variate Gaussian posterior:
        q(W) = N(mu, U_c @ U_c^T ⊗ V_c @ V_c^T)

    Hyperparameters:
        kl_weight: λ in ELBO — controls regularization strength (paper: 0.5)
        eta: prior precision — higher = less noise (paper: 0.1)
        train_particles: MC samples during training (paper: 1)
        test_particles: MC samples during evaluation (paper: 10)
    """

    def __init__(self, model, lr=0.01, damping=1e-3, weight_decay=0.0,
                 T_inv=100, alpha=0.95,
                 kl_weight=0.01, eta=1.0, n_data=45000, train_particles=1, test_particles=10):
        self.model = model
        self.damping = damping
        self.weight_decay = weight_decay
        self.T_inv = T_inv
        self.alpha = alpha
        self.steps = 0

        # Noisy K-FAC specific
        self.kl_weight = kl_weight
        self.eta = eta
        self.n_data = n_data
        self.train_particles = train_particles
        self.test_particles = test_particles

        # Storage for Kronecker factors, inverses, and noise covariance
        self._modules_tracked = {}
        self._stats = {}
        self._inv = {}
        self._noise_cov = {}  # name -> {'U_c': Tensor, 'V_c': Tensor}
        self._noise_applied = {}  # name -> {'delta_w': Tensor, 'delta_b': Tensor}
        self._hooks = []

        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)

        self._register_hooks()
        print(f"  Noisy K-FAC tracking {len(self._modules_tracked)} layers")

    def _register_hooks(self):
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
        def hook(mod, inp, out):
            if not mod.training:
                return
            with torch.no_grad():
                x = inp[0].detach()
                if isinstance(mod, nn.Conv2d):
                    x = F.unfold(x, mod.kernel_size, dilation=mod.dilation,
                                 padding=mod.padding, stride=mod.stride)
                    x = x.permute(0, 2, 1).reshape(-1, x.size(1))
                elif isinstance(mod, nn.Linear):
                    if x.dim() > 2:
                        x = x.reshape(-1, x.size(-1))
                if mod.bias is not None:
                    ones = torch.ones(x.size(0), 1, device=x.device)
                    x = torch.cat([x, ones], dim=1)
                n = x.size(0)
                cov_a = torch.matmul(x.t(), x) / n
                if self._stats[name]['A'] is None:
                    self._stats[name]['A'] = cov_a
                else:
                    self._stats[name]['A'].mul_(self.alpha).add_(cov_a, alpha=1 - self.alpha)
        return hook

    def _backward_hook(self, name, module):
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
    def _compute_pi(self, A, G):
        """Compute pi-correction factor for balanced damping (tracenorm)."""
        pi = torch.sqrt(
            (torch.trace(A) * G.size(0)) /
            (torch.trace(G) * A.size(0) + 1e-10)
        )
        return max(pi.item(), 1e-6)

    @torch.no_grad()
    def _invert_factors(self):
        """Invert A and G with damping. Called every T_inv steps."""
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
                pass

    @torch.no_grad()
    def update_noise_covariance(self):
        """
        Update noise covariance factors U_c and V_c from Fisher factors.

        From Zhang et al. (2017), MVGBlock.update():
            coeff = kl_weight / (N_data * renorm_coeff)
            damping_noise = sqrt(coeff) / sqrt(eta)
            U_eigvals, U_eigvecs = eigh(A / pi + damping_noise * I)
            V_eigvals, V_eigvecs = eigh(G * pi + damping_noise * I)
            U_c = U_eigvecs * sqrt(coeff / max(U_eigvals, damping_noise))
            V_c = V_eigvecs * sqrt(coeff / max(V_eigvals, damping_noise))
        """
        for name in self._stats:
            A = self._stats[name]['A']
            G = self._stats[name]['G']
            if A is None or G is None:
                continue

            try:
                pi = self._compute_pi(A, G)

                # renorm_coeff = num_locations for Conv2d, 1.0 for Linear
                module = self._modules_tracked[name]
                if isinstance(module, nn.Conv2d):
                    renorm_coeff = 1.0
                else:
                    renorm_coeff = 1.0

                # Paper: coeff = kl / (n_data * renorm_coeff)
                coeff = self.kl_weight / (self.n_data * renorm_coeff)
                coeff_sqrt = coeff ** 0.5
                damping_noise = coeff_sqrt / (self.eta ** 0.5)

                # Eigendecompose A/pi + damping*I
                A_scaled = A / pi + damping_noise * torch.eye(A.size(0), device=A.device)
                ue, uv = torch.linalg.eigh(A_scaled)

                # Eigendecompose G*pi + damping*I
                G_scaled = G * pi + damping_noise * torch.eye(G.size(0), device=G.device)
                ve, vv = torch.linalg.eigh(G_scaled)

                # U_c = eigvecs * sqrt(coeff / max(eigvals, damping))
                ue_safe = torch.clamp(ue, min=damping_noise)
                u_c = uv * (coeff / ue_safe).sqrt().unsqueeze(0)

                ve_safe = torch.clamp(ve, min=damping_noise)
                v_c = vv * (coeff / ve_safe).sqrt().unsqueeze(0)

                self._noise_cov[name] = {'U_c': u_c, 'V_c': v_c}
            except Exception:
                pass  # Keep previous noise covariance

    @torch.no_grad()
    def inject_noise(self):
        """
        Sample weight perturbation: ΔW = U_c @ ε @ V_c^T, ε ~ N(0,I)
        Add to weights for noisy forward pass.
        """
        self._noise_applied.clear()
        for name, module in self._modules_tracked.items():
            if name not in self._noise_cov:
                continue

            u_c = self._noise_cov[name]['U_c']
            v_c = self._noise_cov[name]['V_c']

            # ε ~ N(0, I) of shape [d_in, d_out]
            d_in = u_c.size(0)
            d_out = v_c.size(0)
            epsilon = torch.randn(d_in, d_out, device=u_c.device)

            # ΔW_full = U_c @ ε @ V_c^T  (shape: [d_in, d_out])
            delta_full = u_c @ epsilon @ v_c.t()

            if isinstance(module, nn.Conv2d):
                c_out = module.weight.size(0)
                c_in_kk = module.weight[0].numel()
                if module.bias is not None:
                    delta_w = delta_full[:-1, :].t().reshape(c_out, -1)
                    delta_w = delta_w[:, :c_in_kk].reshape_as(module.weight)
                    delta_b = delta_full[-1, :c_out]
                else:
                    delta_w = delta_full[:c_in_kk, :c_out].t().reshape_as(module.weight)
                    delta_b = None
            elif isinstance(module, nn.Linear):
                if module.bias is not None:
                    delta_w = delta_full[:-1, :module.weight.size(0)].t()
                    if delta_w.shape != module.weight.shape:
                        delta_w = delta_w[:module.weight.size(0), :module.weight.size(1)]
                    delta_b = delta_full[-1, :module.bias.size(0)]
                else:
                    delta_w = delta_full[:module.weight.size(1), :module.weight.size(0)].t()
                    delta_b = None
            else:
                continue

            # Clamp noise magnitude to prevent instability
            noise_scale = delta_w.norm() / (module.weight.norm() + 1e-8)
            if noise_scale > 0.5:
                delta_w = delta_w * (0.5 / noise_scale)
                if delta_b is not None:
                    delta_b = delta_b * (0.5 / noise_scale)

            self._noise_applied[name] = {
                'delta_w': delta_w.clone(),
                'delta_b': delta_b.clone() if delta_b is not None else None
            }

            module.weight.data.add_(delta_w)
            if delta_b is not None and module.bias is not None:
                module.bias.data.add_(delta_b)

    @torch.no_grad()
    def restore_weights(self):
        """Remove injected noise after backward pass."""
        for name, module in self._modules_tracked.items():
            if name not in self._noise_applied:
                continue
            delta = self._noise_applied[name]
            module.weight.data.sub_(delta['delta_w'])
            if delta['delta_b'] is not None and module.bias is not None:
                module.bias.data.sub_(delta['delta_b'])
        self._noise_applied.clear()

    def compute_kl_loss(self):
        """KL divergence term for ELBO: (kl / (N * eta)) * ||W||^2"""
        l2 = sum(p.pow(2).sum() for p in self.model.parameters())
        return (self.kl_weight / self.eta) * l2

    def reset_stats(self):
        for name in self._stats:
            self._stats[name] = {'A': None, 'G': None}
        self._inv.clear()
        self._noise_cov.clear()
        self._noise_applied.clear()
        self.steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.steps % self.T_inv == 0:
            self._invert_factors()
            self.update_noise_covariance()

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
                    nat_grad = G_inv @ grad_2d @ A_inv
                    if module.bias is not None and module.bias.grad is not None:
                        nat_grad_w = nat_grad[:, :-1].reshape_as(module.weight)
                        nat_grad_b = nat_grad[:, -1]
                        if self.weight_decay > 0:
                            nat_grad_w.add_(module.weight, alpha=self.weight_decay)
                        module.weight.data.add_(nat_grad_w, alpha=-self.param_groups[0]['lr'])
                        module.bias.data.add_(nat_grad_b, alpha=-self.param_groups[0]['lr'])
                    else:
                        nat_grad = nat_grad.reshape_as(module.weight)
                        if self.weight_decay > 0:
                            nat_grad.add_(module.weight, alpha=self.weight_decay)
                        module.weight.data.add_(nat_grad, alpha=-self.param_groups[0]['lr'])

                elif isinstance(module, nn.Linear):
                    if module.bias is not None and module.bias.grad is not None:
                        grad_2d = torch.cat([grad_w, module.bias.grad.unsqueeze(1)], dim=1)
                    else:
                        grad_2d = grad_w
                    nat_grad = G_inv @ grad_2d @ A_inv
                    if module.bias is not None and module.bias.grad is not None:
                        nat_grad_w = nat_grad[:, :-1]
                        nat_grad_b = nat_grad[:, -1]
                        if self.weight_decay > 0:
                            nat_grad_w.add_(module.weight, alpha=self.weight_decay)
                        module.weight.data.add_(nat_grad_w, alpha=-self.param_groups[0]['lr'])
                        module.bias.data.add_(nat_grad_b, alpha=-self.param_groups[0]['lr'])
                    else:
                        if self.weight_decay > 0:
                            nat_grad.add_(module.weight, alpha=self.weight_decay)
                        module.weight.data.add_(nat_grad, alpha=-self.param_groups[0]['lr'])
            else:
                if self.weight_decay > 0:
                    grad_w = grad_w + self.weight_decay * module.weight
                module.weight.data.add_(grad_w, alpha=-self.param_groups[0]['lr'])
                if module.bias is not None and module.bias.grad is not None:
                    module.bias.data.add_(module.bias.grad, alpha=-self.param_groups[0]['lr'])

        # SGD for non-tracked parameters (BatchNorm etc.)
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

print("✓ NoisyKFACOptimizer defined")

# %% [markdown]
# ## 6. Experiment Configuration

# %%
num_classes = 100

# Noisy K-FAC hyperparameters (tuned for ResNet-18 / CIFAR-100 continual learning)
# Paper default (VGG-16, single-task CIFAR-10): kl=0.5, eta=0.1
# Our setting: kl=0.01, eta=1.0 — much weaker regularization because:
#   1) CIFAR-100 is harder → model needs more capacity
#   2) Continual learning → weights must adapt quickly to new tasks
#   3) ResNet-18 has ~11M params → per-param regularization accumulates
#
# kl_weight × ||W||² / (N × eta) should be ~0.1-1.0 (comparable to CE loss)
# With kl=0.01, eta=1.0, N=45000: coeff = 2.2e-7 per param²

nkfac_config = {
    'num_epochs': 1200,
    'lr': 0.01,
    'damping': 1e-3,
    'weight_decay': 0.0,         # KL term handles regularization
    'T_inv': 100,
    'alpha': 0.95,
    'batch_size': 90,
    'kl_weight': 0.1,            # ELBO KL coefficient (balanced: 0.01 too weak, 1.0 too strong)
    'eta': 1.0,                  # Prior precision (increased from paper's 0.1)
    'train_particles': 1,
    'test_particles': 10,
    'class_increase_frequency': 200,
    'checkpoint_frequency': 500,
    'random_seed': 42,
    'method': 'noisy_kfac'
}

print("\nNoisy K-FAC Configuration:")
for k, v in nkfac_config.items():
    print(f"  - {k}: {v}")

# %% [markdown]
# ## 7. Initialize Model and Optimizer

# %%
print("\nInitializing ResNet-18 for Noisy K-FAC experiment...")
net_nkfac = build_resnet18(num_classes=num_classes, norm_layer=nn.BatchNorm2d)
net_nkfac.apply(kaiming_init_resnet_module)
net_nkfac.to(device)

optim_nkfac = NoisyKFACOptimizer(
    net_nkfac,
    lr=nkfac_config['lr'],
    damping=nkfac_config['damping'],
    weight_decay=nkfac_config['weight_decay'],
    T_inv=nkfac_config['T_inv'],
    alpha=nkfac_config['alpha'],
    kl_weight=nkfac_config['kl_weight'],
    eta=nkfac_config['eta'],
    n_data=45000,  # training set size for KL scaling
    train_particles=nkfac_config['train_particles'],
    test_particles=nkfac_config['test_particles']
)

loss_fn_nkfac = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in net_nkfac.parameters())
print(f"✓ Total params: {total_params:,}")
print(f"✓ Model initialized for Noisy K-FAC")

# %% [markdown]
# ## 8. Setup Data Loaders

# %%
current_num_classes = 5
all_classes = np.random.RandomState(nkfac_config['random_seed']).permutation(num_classes)

train_data_exp = copy.deepcopy(train_data)
val_data_exp = copy.deepcopy(val_data)
test_data_exp = copy.deepcopy(test_data)

train_data_exp.select_new_partition(all_classes[:current_num_classes])
val_data_exp.select_new_partition(all_classes[:current_num_classes])
test_data_exp.select_new_partition(all_classes[:current_num_classes])

train_loader = DataLoader(train_data_exp, batch_size=nkfac_config['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_data_exp, batch_size=50, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data_exp, batch_size=100, shuffle=False, num_workers=0)

# Dormant measurement loaders
dormant_next_data = copy.deepcopy(train_data_full)
subsample_cifar(train_indices, dormant_next_data)
dormant_next_data.set_transformation(eval_transformations)
next_start_d = current_num_classes
next_end_d = min(current_num_classes + 5, num_classes)
if next_start_d < num_classes:
    dormant_next_data.select_new_partition(all_classes[next_start_d:next_end_d])
    dormant_next_loader = DataLoader(dormant_next_data, batch_size=1000, shuffle=True, num_workers=0)
else:
    dormant_next_loader = None

dormant_prev_data = copy.deepcopy(train_data_full)
subsample_cifar(train_indices, dormant_prev_data)
dormant_prev_data.set_transformation(eval_transformations)
dormant_prev_loader = None

print(f"✓ Data loaders initialized with {current_num_classes} classes")

# %% [markdown]
# ## 9. Noisy K-FAC Training Loop
#
# Key differences from standard K-FAC:
# 1. **inject_noise()** before forward pass → sample from posterior
# 2. **KL loss** added to cross-entropy → ELBO objective
# 3. **restore_weights()** after backward → return to mean weights
# 4. **MC averaging** at test time with test_particles samples

# %%
num_epochs = nkfac_config['num_epochs']
class_increase_frequency = nkfac_config['class_increase_frequency']
checkpoint_frequency = nkfac_config['checkpoint_frequency']
n_train_data = len(train_data.data['data'])  # For KL scaling

checkpoint_dir = os.path.join(results_dir, "cifar100_noisy_kfac", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Metric storage
train_losses_all = []
kl_losses_all = []
train_accuracies_all = []
val_accuracies_all = []
test_accuracies_all = []
dormant_after_all = []
dormant_before_all = []
stable_ranks_all = []
avg_weight_mag_all = []
noise_scale_all = []
epoch_runtimes_all = []
fisher_erank_all = []  # Fisher/NTK effective rank measurements

print(f"\nStarting Noisy K-FAC training for {num_epochs} epochs...")
print(f"Incremental: start with {current_num_classes} classes, add 5 every {class_increase_frequency} epochs")
print(f"Noisy K-FAC: kl_weight={nkfac_config['kl_weight']}, eta={nkfac_config['eta']}, train_particles={nkfac_config['train_particles']}")

net_nkfac.train()
for epoch in range(num_epochs):
    epoch_start_time = time.time()

    # Measure stable rank at task boundaries
    if epoch % class_increase_frequency == 0:
        next_task_start = current_num_classes
        next_task_end = min(current_num_classes + 5, num_classes)
        next_task_classes = all_classes[next_task_start:next_task_end]

        if len(next_task_classes) > 0:
            temp_data = copy.deepcopy(train_data_exp)
            temp_data.select_new_partition(next_task_classes)
            temp_loader = DataLoader(temp_data, batch_size=nkfac_config['batch_size'], shuffle=False, num_workers=0)

            net_nkfac.eval()
            all_activations = []
            samples_collected = 0

            with torch.no_grad():
                for sample in temp_loader:
                    if samples_collected >= 1000:
                        break
                    images = sample["image"].to(device)
                    features_per_layer = []
                    net_nkfac.forward(images, features_per_layer)
                    all_activations.append(features_per_layer[-1].cpu())
                    samples_collected += images.shape[0]

            if len(all_activations) > 0:
                last_layer_act = torch.cat(all_activations, dim=0)[:1000].numpy()
                stable_rank_val = compute_stable_rank_from_activations(last_layer_act)
                stable_ranks_all.append(stable_rank_val)
                print(f"  → Stable Rank (before training task {len(stable_ranks_all)}): {stable_rank_val:.2f} ({stable_rank_val/512*100:.1f}%)")

            net_nkfac.train()

    # Add new classes at task boundaries
    if epoch > 0 and (epoch % class_increase_frequency) == 0 and current_num_classes < num_classes:
        current_num_classes = min(current_num_classes + 5, num_classes)
        train_data_exp.select_new_partition(all_classes[:current_num_classes])
        val_data_exp.select_new_partition(all_classes[:current_num_classes])
        test_data_exp.select_new_partition(all_classes[:current_num_classes])

        train_loader = DataLoader(train_data_exp, batch_size=nkfac_config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data_exp, batch_size=50, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_data_exp, batch_size=100, shuffle=False, num_workers=0)

        print(f"\n{'='*60}")
        print(f"Noisy K-FAC: New task at epoch {epoch} → {current_num_classes} classes")
        print(f"{'='*60}")

        optim_nkfac.reset_stats()

        # Update dormant loaders
        next_start_d = current_num_classes
        next_end_d = min(current_num_classes + 5, num_classes)
        if next_start_d < num_classes:
            dormant_next_data.select_new_partition(all_classes[next_start_d:next_end_d])
            dormant_next_loader = DataLoader(dormant_next_data, batch_size=1000, shuffle=True, num_workers=0)
        else:
            dormant_next_loader = None
        prev_end_d = current_num_classes - 5
        if prev_end_d > 0:
            dormant_prev_data.select_new_partition(all_classes[:prev_end_d])
            dormant_prev_loader = DataLoader(dormant_prev_data, batch_size=1000, shuffle=True, num_workers=0)
        else:
            dormant_prev_loader = None

    print(f"\nEpoch {epoch+1}/{num_epochs} | Classes: {current_num_classes}")

    # === Training with Noisy K-FAC ===
    running_loss = 0
    running_kl = 0
    running_acc = 0
    num_batches = 0

    for sample in tqdm(train_loader, desc=f"NoisyKFAC epoch {epoch+1}"):
        images = sample["image"].to(device)
        labels = sample["label"].to(device)

        net_nkfac.train()

        # 1. Inject noise into weights (sample from posterior)
        optim_nkfac.inject_noise()

        # 2. Forward pass with noisy weights
        predictions = net_nkfac(images)
        predictions_masked = predictions[:, all_classes[:current_num_classes]]

        # 3. Compute ELBO loss = CE + (kl_weight / (N * eta)) * ||W||^2
        ce_loss = loss_fn_nkfac(predictions_masked,
                                labels.argmax(dim=1) if labels.dim() > 1 and labels.shape[1] > 1 else labels)
        kl_loss = optim_nkfac.compute_kl_loss() / n_train_data
        total_loss = ce_loss + kl_loss

        # 4. Backward on noisy weights
        optim_nkfac.zero_grad()
        total_loss.backward()

        # 5. Restore mean weights, then apply K-FAC preconditioned update to mean
        optim_nkfac.restore_weights()
        optim_nkfac.step()

        with torch.no_grad():
            acc = (predictions_masked.argmax(dim=1) == (labels.argmax(dim=1) if labels.dim() > 1 and labels.shape[1] > 1 else labels)).float().mean()
        running_loss += ce_loss.item()
        running_kl += kl_loss.item()
        running_acc += acc.item()
        num_batches += 1

    train_losses_all.append(running_loss / num_batches)
    kl_losses_all.append(running_kl / num_batches)
    train_accuracies_all.append(running_acc / num_batches)

    # === Validation ===
    net_nkfac.eval()
    val_acc = 0
    val_batches = 0
    with torch.no_grad():
        for sample in val_loader:
            images = sample["image"].to(device)
            labels = sample["label"].to(device)
            # MC averaging with test_particles samples
            preds_sum = torch.zeros(images.size(0), num_classes, device=device)
            for _ in range(nkfac_config['test_particles']):
                optim_nkfac.inject_noise()
                preds_sum += net_nkfac(images)
                optim_nkfac.restore_weights()
            predictions = preds_sum / nkfac_config['test_particles']
            predictions_masked = predictions[:, all_classes[:current_num_classes]]
            acc = (predictions_masked.argmax(dim=1) == (labels.argmax(dim=1) if labels.dim() > 1 and labels.shape[1] > 1 else labels)).float().mean()
            val_acc += acc.item()
            val_batches += 1
    val_accuracies_all.append(val_acc / val_batches if val_batches > 0 else 0)

    # === Test ===
    test_acc = 0
    test_batches = 0
    with torch.no_grad():
        for sample in test_loader:
            images = sample["image"].to(device)
            labels = sample["label"].to(device)
            preds_sum = torch.zeros(images.size(0), num_classes, device=device)
            for _ in range(nkfac_config['test_particles']):
                optim_nkfac.inject_noise()
                preds_sum += net_nkfac(images)
                optim_nkfac.restore_weights()
            predictions = preds_sum / nkfac_config['test_particles']
            predictions_masked = predictions[:, all_classes[:current_num_classes]]
            acc = (predictions_masked.argmax(dim=1) == (labels.argmax(dim=1) if labels.dim() > 1 and labels.shape[1] > 1 else labels)).float().mean()
            test_acc += acc.item()
            test_batches += 1
    test_accuracies_all.append(test_acc / test_batches if test_batches > 0 else 0)

    # === Dormant Measurement ===
    for h in optim_nkfac._hooks:
        h.remove()
    optim_nkfac._hooks.clear()

    net_nkfac.train()

    if dormant_next_loader is not None:
        dormant_after, _ = compute_dormant_units_proportion(net_nkfac, dormant_next_loader, dormant_unit_threshold=0.01)
    else:
        dormant_after = float('nan')
    dormant_after_all.append(dormant_after)

    if dormant_prev_loader is not None:
        dormant_before, _ = compute_dormant_units_proportion(net_nkfac, dormant_prev_loader, dormant_unit_threshold=0.01)
    else:
        dormant_before = float('nan')
    dormant_before_all.append(dormant_before)

    # Debug: per-layer sparsity every 50 epochs
    if epoch % 50 == 0:
        with torch.no_grad():
            dbg_loader = dormant_next_loader if dormant_next_loader is not None else dormant_prev_loader
            for sample in (dbg_loader or []):
                img_dbg = sample["image"].to(device)
                feats_dbg = []
                net_nkfac.forward(img_dbg, feats_dbg)
                print(f"  [DEBUG] Dormant analysis on {img_dbg.shape[0]} samples:")
                for li, f in enumerate(feats_dbg):
                    zero_frac = (f == 0).float().mean().item()
                    if f.dim() == 4:
                        per_ch = (f != 0).float().mean(dim=(0, 2, 3))
                        dormant_ch = (per_ch < 0.01).sum().item()
                        print(f"    Layer {li}: shape={list(f.shape)}, zero_frac={zero_frac:.3f}, dormant_channels={dormant_ch}/{f.shape[1]}")
                    else:
                        per_unit = (f != 0).float().mean(dim=0)
                        dormant_units = (per_unit < 0.01).sum().item()
                        print(f"    Layer {li} (FC): shape={list(f.shape)}, zero_frac={zero_frac:.3f}, dormant_units={dormant_units}/{f.shape[1]}")
                break

    # Noise scale diagnostic
    avg_noise_scale = 0
    n_noise = 0
    for name in optim_nkfac._noise_cov:
        u_c = optim_nkfac._noise_cov[name]['U_c']
        v_c = optim_nkfac._noise_cov[name]['V_c']
        avg_noise_scale += u_c.norm().item() * v_c.norm().item()
        n_noise += 1
    if n_noise > 0:
        avg_noise_scale /= n_noise
    noise_scale_all.append(avg_noise_scale)

    optim_nkfac._register_hooks()

    avg_weight_mag = compute_avg_weight_magnitude(net_nkfac)
    avg_weight_mag_all.append(avg_weight_mag)

    net_nkfac.train()

    epoch_runtime = time.time() - epoch_start_time
    epoch_runtimes_all.append(epoch_runtime)

    print(f"  Train: {train_accuracies_all[-1]:.4f} | Val: {val_accuracies_all[-1]:.4f} | Test: {test_accuracies_all[-1]:.4f}")
    print(f"  CE Loss: {train_losses_all[-1]:.4f} | KL Loss: {kl_losses_all[-1]:.4f} | Dormant(next): {dormant_after:.4f} | Dormant(prev): {dormant_before:.4f}")
    print(f"  Avg Weight: {avg_weight_mag:.4f} | Noise Scale: {avg_noise_scale:.6f} | Epoch time: {epoch_runtime:.1f}s")

    # === Fisher/NTK Effective Rank (every 50 epochs) ===
    if epoch % 50 == 0 or epoch == num_epochs - 1:
        fisher_stats = compute_fisher_erank_stats(optim_nkfac)
        fisher_erank_all.append({
            'epoch': epoch,
            'mean_rel_erank_A': fisher_stats['mean_rel_erank_A'],
            'mean_rel_erank_G': fisher_stats['mean_rel_erank_G'],
            'mean_condition_number': fisher_stats['mean_condition_number'],
            'per_layer': fisher_stats['per_layer']
        })
        print(f"  Fisher eRank: A={fisher_stats['mean_rel_erank_A']:.1f}% | G={fisher_stats['mean_rel_erank_G']:.1f}% | Cond={fisher_stats['mean_condition_number']:.1f}")
        # Show per-layer breakdown for first/last conv and FC
        if fisher_stats['per_layer']:
            first = fisher_stats['per_layer'][0]
            last_conv = [d for d in fisher_stats['per_layer'] if 'conv' in d['name'].lower() or 'layer' in d['name'].lower()]
            fc = [d for d in fisher_stats['per_layer'] if 'fc' in d['name'].lower() or 'linear' in d['name'].lower()]
            if last_conv:
                lc = last_conv[-1]
                print(f"    Last Conv [{lc['name']}]: A eRank={lc['erank_A']:.1f}/{lc['dim_A']} ({lc['rel_A']:.1f}%) | G eRank={lc['erank_G']:.1f}/{lc['dim_G']} ({lc['rel_G']:.1f}%)")
            if fc:
                f = fc[-1]
                print(f"    FC [{f['name']}]: A eRank={f['erank_A']:.1f}/{f['dim_A']} ({f['rel_A']:.1f}%) | G eRank={f['erank_G']:.1f}/{f['dim_G']} ({f['rel_G']:.1f}%)")

    # Checkpoint
    if (epoch + 1) % checkpoint_frequency == 0 or (epoch + 1) == num_epochs:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net_nkfac.state_dict(),
            'current_num_classes': current_num_classes,
            'train_losses': train_losses_all,
            'kl_losses': kl_losses_all,
            'train_accuracies': train_accuracies_all,
            'val_accuracies': val_accuracies_all,
            'test_accuracies': test_accuracies_all,
            'dormant_after': dormant_after_all,
            'dormant_before': dormant_before_all,
            'stable_ranks': stable_ranks_all,
            'noise_scale': noise_scale_all,
            'config': nkfac_config
        }, checkpoint_path)
        print(f"  → Checkpoint saved: {checkpoint_path}")

print(f"\n✓ Noisy K-FAC training completed!")

# %% [markdown]
# ## 10. Save Results

# %%
nkfac_results = {
    'train_losses': train_losses_all,
    'kl_losses': kl_losses_all,
    'train_accuracies': train_accuracies_all,
    'val_accuracy_per_epoch': val_accuracies_all,
    'test_accuracy_per_epoch': test_accuracies_all,
    'epoch_runtime': epoch_runtimes_all,
    'dormant_after_per_epoch': dormant_after_all,
    'dormant_before_per_epoch': dormant_before_all,
    'stable_rank_per_task': stable_ranks_all,
    'avg_weight_magnitude_per_epoch': avg_weight_mag_all,
    'noise_scale_per_epoch': noise_scale_all,
    'fisher_erank': fisher_erank_all,
    'config': nkfac_config,
    'class_order': all_classes.tolist()
}

os.makedirs(os.path.join(results_dir, "cifar100_noisy_kfac"), exist_ok=True)
save_results(nkfac_results, os.path.join(results_dir, 'cifar100_noisy_kfac', 'noisy_kfac_results.pkl'))

print(f"\n✓ Saved Noisy K-FAC metrics:")
print(f"  - Epochs: {len(test_accuracies_all)}")
print(f"  - Dormant analysis: {len(dormant_after_all)} after + {len(dormant_before_all)} before")
print(f"  - Stable rank analysis: {len(stable_ranks_all)} measurements")
print(f"  - Noise scale tracking: {len(noise_scale_all)} measurements")
