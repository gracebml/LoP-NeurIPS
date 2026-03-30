# %% [markdown]
# # Shampoo-CL: Shampoo Preconditioning for Continual Learning
#
# This notebook implements **Shampoo-CL** — combining Shampoo preconditioned
# optimization (Gupta et al., ICML 2018) with EWC-style continual learning.
#
# **Key design choices**:
# 1. **Shampoo preconditioner** (L, R factors from gradient covariance) replaces K-FAC (A, G from Fisher)
# 2. **EWC regularization** from Shampoo factors — free Fisher estimation via `diag(L⊗R)`
# 3. **Preconditioner reset** at task boundaries — fresh curvature for new task
# 4. **No noise injection** — EWC handles forgetting prevention instead
#
# **Ablation study**: Compare Shampoo-CL variants against K-FAC (aimo-ans4)
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

sys.path.append("/kaggle/input/datasets/mlinh776/lop-src")
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
cifar_config_path = '/kaggle/input/datasets/mlinh776/lop-src/lop/incremental_cifar/cfg/continual_backpropagation.json'
cifar_config = load_config(cifar_config_path)
cifar_config["data_path"] = (lambda p: (os.makedirs(p, exist_ok=True), p)[1])("/kaggle/working/incremental_cifar/data")
cifar_config['results_dir'] = os.path.join(results_dir, 'cifar100')
cifar_config['num_workers'] = 2

# Transforms — must match aimo-ans4 exactly (torchvision Compose + swap_color_axis)
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

# Split 450 train + 50 validation per class (matching aimo-ans4)
def get_validation_and_train_indices(cifar_data, num_classes=100):
    """Split into 450 train + 50 validation per class"""
    num_val_per_class = 50
    num_train_per_class = 450
    val_size = 5000
    train_size = 45000

    val_indices = torch.zeros(val_size, dtype=torch.int32)
    train_indices = torch.zeros(train_size, dtype=torch.int32)

    current_val = 0
    current_train = 0

    for i in range(num_classes):
        # Find indices where class i has label 1 (one-hot encoded)
        class_indices = torch.argwhere(cifar_data.data["labels"][:, i] == 1).flatten()
        val_indices[current_val:current_val + num_val_per_class] = class_indices[:num_val_per_class]
        train_indices[current_train:current_train + num_train_per_class] = class_indices[num_val_per_class:]
        current_val += num_val_per_class
        current_train += num_train_per_class

    return train_indices, val_indices

train_indices, val_indices = get_validation_and_train_indices(train_data_full)

# Subsample — must also update integer_labels and re-run partition_data (critical!)
def subsample_cifar(indices, cifar_data):
    """Subsample CIFAR dataset by indices"""
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

print(f"✓ CIFAR-100 loaded:")
print(f"  - Train: {len(train_data.data['data'])} samples (450 per class)")
print(f"  - Validation: {len(val_data.data['data'])} samples (50 per class)")
print(f"  - Test: {len(test_data.data['data'])} samples")

# %% [markdown]
# ## 4. Metrics Functions

# %%
from lop.incremental_cifar.post_run_analysis import compute_dormant_units_proportion
from lop.algos.res_gnt import ResGnT

def compute_stable_rank(singular_values):
    """Compute stable rank (matching lop/incremental_cifar/post_run_analysis.py exactly).
    
    Definition: number of singular values needed to capture 99% of total singular values.
    NOT the Frobenius/spectral ratio (σ²/σ_max²) — that's a different metric.
    """
    if len(singular_values) == 0:
        return 0
    
    # Sort singular values in descending order
    sorted_sv = np.flip(np.sort(singular_values))
    
    # Compute cumulative sum normalized by total
    cumsum_sv = np.cumsum(sorted_sv) / np.sum(singular_values)
    
    # Count how many SVs needed to reach 99%
    stable_rank = np.sum(cumsum_sv < 0.99) + 1
    
    return stable_rank

def compute_stable_rank_from_activations(activations):
    """Compute stable rank from layer activations using SVD (matching aimo-ans4)."""
    try:
        # Flatten to 2D if needed
        if activations.ndim > 2:
            activations = activations.reshape(activations.shape[0], -1)
        
        if activations.shape[0] == 0 or activations.shape[1] == 0:
            return 0
        
        # Use scipy SVD with gesvd driver for numerical stability (matching aimo-ans4)
        from scipy.linalg import svd
        singular_values = svd(activations, compute_uv=False, lapack_driver="gesvd")
        
        return compute_stable_rank(singular_values)
    except Exception as e:
        print(f"Warning: SVD failed with error: {e}")
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
    Compute effective rank via spectral entropy (Murray et al., ICLR 2023).
    eRank(M) = exp(H(p)) where p_i = sigma_i / sum(sigma)
    """
    if matrix is None or matrix.numel() == 0:
        return 0.0
    eigvals = torch.linalg.eigvalsh(matrix)
    eigvals = eigvals.clamp(min=1e-10)
    p = eigvals / eigvals.sum()
    entropy = -(p * p.log()).sum().item()
    return np.exp(entropy)

print("✓ Metrics functions defined")

# %% [markdown]
# ## 5. Shampoo-CL Optimizer Definition
#
# Based on:
# - Shampoo: Gupta et al. (2018) "Shampoo: Preconditioned Stochastic Tensor Optimization"
# - SOAP: Vyas et al. (2024) "SOAP: Improving and Stabilizing Shampoo using Adam in the Preconditioner's Eigenbasis"
# - EWC: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
#
# Key differences from K-FAC (aimo-ans4):
# 1. **Gradient-only preconditioning**: L = GGᵀ, R = GᵀG (no forward hook needed)
# 2. **EWC regularization**: protects old task weights via Fisher penalty from diag(L⊗R)
# 3. **Matrix root preconditioning**: L^{-1/4} G R^{-1/4} instead of G⁻¹ ∇W A⁻¹
# 4. **Adam grafting**: Shampoo direction scaled by Adam step size for stability

# %%
from torch.optim import Optimizer as _Optimizer

class ShampooCLOptimizer(_Optimizer):
    """
    Shampoo-CL: Shampoo Preconditioned Optimizer for Continual Learning.
    
    Combines Shampoo's Kronecker-factored gradient preconditioning with
    EWC (Elastic Weight Consolidation) for continual learning.
    
    Shampoo preconditioner:
        L_l = β₂·L_l + (1-β₂)·∇W·∇Wᵀ        (m×m, left preconditioner)
        R_l = β₂·R_l + (1-β₂)·∇Wᵀ·∇W         (n×n, right preconditioner)
        ∇̃W = L^{-1/p} · ∇W · R^{-1/p}         (preconditioned gradient)
    
    EWC regularization:
        L_ewc = (λ/2) · Σ_tasks Σ_layers F_i · (θ - θ*_i)²
        where F_i = diag(L_i ⊗ R_i) — free Fisher from Shampoo factors
    """
    
    def __init__(self, model, lr=0.01, damping=1e-4, weight_decay=0.0,
                 T_precond=100, beta2=0.999, power=4,
                 grafting='adam',
                 adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
                 cl_mode='ogp', ogp_k_ratio=0.8,
                 ewc_lambda=0.0, adaptive_ewc=False, ewc_target_ratio=0.3,
                 ewc_adapt_rate=0.01, ewc_warmup_epochs=50):
        """
        Shampoo-CL optimizer with pluggable continual learning strategies.
        
        Args:
            model: nn.Module to optimize
            lr: learning rate
            damping: damping for matrix root computation
            weight_decay: L2 weight decay (decoupled)
            T_precond: frequency of eigendecomposition update
            beta2: EMA decay for L, R factors
            power: matrix root power (p=4 → L^{-1/4}, or p=2 → L^{-1/2})
            grafting: 'adam' for Adam grafting, 'sgd' for SGD grafting, 'none'
            adam_beta1: Adam β₁ for grafting momentum
            adam_beta2: Adam β₂ for grafting second moment
            adam_eps: Adam ε for grafting
            cl_mode: 'ogp' (Orthogonal Gradient Projection) or 'ewc' (Elastic Weight Consolidation)
            ogp_k_ratio: fraction of variance to protect in OGP (0-1, higher = more stability)
            ewc_lambda: EWC penalty strength (only used if cl_mode='ewc')
        """
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(model.parameters(), defaults)
        
        self.model = model
        self.damping = damping
        self.T_precond = T_precond
        self.beta2 = beta2
        self.power = power
        self.grafting = grafting
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        
        # Continual learning mode
        self.cl_mode = cl_mode
        
        # OGP (Orthogonal Gradient Projection) state
        self.ogp_k_ratio = ogp_k_ratio
        self._projectors = {}           # {name: {'P_L': tensor, 'P_R': tensor}}
        self._saved_subspaces = []      # list of {name: {'U_L': tensor, 'U_R': tensor}}
        
        # EWC state (legacy, kept for ablation)
        self.ewc_lambda = ewc_lambda
        self.ewc_lambda_init = ewc_lambda
        self.adaptive_ewc = adaptive_ewc
        self.ewc_target_ratio = ewc_target_ratio
        self.ewc_adapt_rate = ewc_adapt_rate
        self.ewc_lambda_history = []
        self.ewc_warmup_factor = 1.0
        self._ewc_snapshots = []
        
        # Track Conv2d and Linear layers
        self._modules_tracked = {}
        self._stats = {}  # L, R factors per layer
        self._precond = {}  # Cached preconditioner roots
        self._adam_state = {}  # Adam grafting state
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._modules_tracked[name] = module
                self._stats[name] = {'L': None, 'R': None}
                self._adam_state[name] = {'m': None, 'v': None}
        
        self.steps = 0
        
        print(f"  ShampooCL tracking {len(self._modules_tracked)} layers (mode={cl_mode})")
        for name, mod in self._modules_tracked.items():
            if isinstance(mod, nn.Conv2d):
                m, n = mod.weight.size(0), mod.weight[0].numel()
                if mod.bias is not None:
                    n += 1
                print(f"    {name}: Conv2d → L:{m}×{m}, R:{n}×{n}")
            elif isinstance(mod, nn.Linear):
                m, n = mod.weight.size(0), mod.weight.size(1)
                if mod.bias is not None:
                    n += 1
                print(f"    {name}: Linear → L:{m}×{m}, R:{n}×{n}")
    
    @torch.no_grad()
    def update_factors(self):
        """Update Shampoo L, R factors from current weight gradients."""
        for name, module in self._modules_tracked.items():
            if module.weight.grad is None:
                continue
            
            grad_w = module.weight.grad
            
            # Reshape gradient to 2D: (out_features, in_features)
            if isinstance(module, nn.Conv2d):
                grad_2d = grad_w.reshape(grad_w.size(0), -1)
            else:
                grad_2d = grad_w
            
            # Append bias gradient if applicable
            if module.bias is not None and module.bias.grad is not None:
                grad_2d = torch.cat([grad_2d, module.bias.grad.unsqueeze(1)], dim=1)
            
            # L = GGᵀ (m×m), R = GᵀG (n×n)
            L_new = torch.mm(grad_2d, grad_2d.t())
            R_new = torch.mm(grad_2d.t(), grad_2d)
            
            if self._stats[name]['L'] is None:
                self._stats[name]['L'] = L_new
                self._stats[name]['R'] = R_new
            else:
                self._stats[name]['L'].mul_(self.beta2).add_(L_new, alpha=1 - self.beta2)
                self._stats[name]['R'].mul_(self.beta2).add_(R_new, alpha=1 - self.beta2)
    
    @torch.no_grad()
    def _compute_matrix_root_inv(self, mat, power):
        """
        Compute M^{-1/p} via eigendecomposition.
        M = Q Λ Qᵀ → M^{-1/p} = Q Λ^{-1/p} Qᵀ
        """
        try:
            eigvals, eigvecs = torch.linalg.eigh(mat)
            eigvals = eigvals.clamp(min=self.damping)
            inv_root = eigvals.pow(-1.0 / power)
            return eigvecs @ torch.diag(inv_root) @ eigvecs.t()
        except RuntimeError:
            # Fallback: damped identity
            return (1.0 / self.damping ** (1.0 / power)) * torch.eye(
                mat.size(0), device=mat.device)
    
    @torch.no_grad()
    def update_preconditioner(self):
        """Recompute preconditioner roots L^{-1/p} and R^{-1/p}."""
        for name in self._stats:
            L = self._stats[name]['L']
            R = self._stats[name]['R']
            if L is None or R is None:
                continue
            
            # Add damping
            L_d = L + self.damping * torch.eye(L.size(0), device=L.device)
            R_d = R + self.damping * torch.eye(R.size(0), device=R.device)
            
            self._precond[name] = {
                'L_inv_root': self._compute_matrix_root_inv(L_d, self.power),
                'R_inv_root': self._compute_matrix_root_inv(R_d, self.power)
            }
    
    def compute_ewc_loss(self):
        """
        EWC penalty: (λ/2) · Σ_tasks Σ_layers F_i · (θ - θ*_i)²
        
        With per-task normalization: each task's contribution is divided by
        the number of accumulated tasks, preventing λ explosion.
        """
        if len(self._ewc_snapshots) == 0 or self.ewc_lambda == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        num_tasks = len(self._ewc_snapshots)
        
        for snapshot in self._ewc_snapshots:
            for name, module in self._modules_tracked.items():
                if name in snapshot:
                    diff = module.weight - snapshot[name]['optimal_weights']
                    ewc_loss = ewc_loss + (snapshot[name]['fisher_w'] * diff.pow(2)).sum()
                    if module.bias is not None and 'optimal_bias' in snapshot[name]:
                        diff_b = module.bias - snapshot[name]['optimal_bias']
                        ewc_loss = ewc_loss + (snapshot[name]['fisher_b'] * diff_b.pow(2)).sum()
        
        # Per-task normalization + warmup
        effective_lambda = self.ewc_lambda * self.ewc_warmup_factor
        return (effective_lambda / (2.0 * num_tasks)) * ewc_loss
    
    def adapt_ewc_lambda(self, ce_loss_val, ewc_loss_val):
        """
        Adaptive EWC λ: adjust λ to maintain target EWC/CE loss ratio.
        
        Uses multiplicative update with EMA smoothing:
            ratio = EWC_loss / CE_loss
            λ_new = λ_old · (target_ratio / ratio)^adapt_rate
        
        This naturally:
        - Increases λ when EWC is too weak (ratio < target)
        - Decreases λ when EWC dominates (ratio > target)
        """
        if not self.adaptive_ewc or ce_loss_val < 1e-8 or ewc_loss_val < 1e-8:
            return
        
        ratio = ewc_loss_val / ce_loss_val
        
        # Multiplicative update: smooth adjustment toward target ratio
        adjustment = (self.ewc_target_ratio / ratio) ** self.ewc_adapt_rate
        
        # Clamp adjustment to avoid extreme jumps
        adjustment = max(0.9, min(1.1, adjustment))
        
        self.ewc_lambda *= adjustment
        
        # Clamp λ to reasonable range [0.1, 1000]
        self.ewc_lambda = max(0.1, min(1000.0, self.ewc_lambda))
        
        self.ewc_lambda_history.append(self.ewc_lambda)
    
    @torch.no_grad()
    def save_ewc_snapshot(self):
        """
        Save Fisher + optimal weights snapshot after completing a task.
        Fisher ≈ diag(L ⊗ R) from Shampoo factors — free computation!
        """
        snapshot = {}
        for name, module in self._modules_tracked.items():
            L = self._stats[name]['L']
            R = self._stats[name]['R']
            
            if L is not None and R is not None:
                # Fisher diagonal ≈ outer product of L and R diagonals
                # diag(L ⊗ R) = vec(diag(L) · diag(R)ᵀ) 
                l_diag = L.diag()  # (m,)
                r_diag = R.diag()  # (n,)
                fisher_full = torch.outer(l_diag, r_diag)  # (m, n)
                
                # Extract weight Fisher and bias Fisher
                if isinstance(module, nn.Conv2d):
                    n_weight = module.weight[0].numel()
                    fisher_w = fisher_full[:, :n_weight].reshape_as(module.weight)
                    if module.bias is not None:
                        fisher_b = fisher_full[:, n_weight].squeeze()
                    else:
                        fisher_b = None
                else:  # Linear
                    n_weight = module.weight.size(1)
                    fisher_w = fisher_full[:, :n_weight]
                    if module.bias is not None:
                        fisher_b = fisher_full[:, n_weight].squeeze()
                    else:
                        fisher_b = None
            else:
                fisher_w = torch.ones_like(module.weight) * 1e-3
                fisher_b = torch.ones_like(module.bias) * 1e-3 if module.bias is not None else None
            
            # Normalize Fisher to prevent explosion across tasks
            fisher_w = fisher_w / (fisher_w.max() + 1e-10)
            
            entry = {
                'fisher_w': fisher_w.clone(),
                'optimal_weights': module.weight.data.clone()
            }
            if module.bias is not None:
                if fisher_b is not None:
                    fisher_b = fisher_b / (fisher_b.max() + 1e-10)
                entry['fisher_b'] = fisher_b.clone() if fisher_b is not None else None
                entry['optimal_bias'] = module.bias.data.clone()
            
            snapshot[name] = entry
        
        self._ewc_snapshots.append(snapshot)
        print(f"  → EWC snapshot saved (total tasks: {len(self._ewc_snapshots)})")
    
    @torch.no_grad()
    def _find_spectral_gap_k(self, eigvals, min_free_dims=5):
        """
        Find number of dimensions to protect using the spectral gap heuristic.
        
        Eigenvalue spectrum of real gradient covariance has natural low-rank structure:
        a few large eigenvalues (important directions) then a sharp drop-off (noise).
        We find this drop-off by looking for the largest relative gap.
        
        Args:
            eigvals: eigenvalues sorted descending, all positive
            min_free_dims: minimum dimensions to leave free for plasticity
            
        Returns:
            k: number of top eigenvectors to protect
        """
        n = eigvals.size(0)
        if n <= min_free_dims + 1:
            return 1  # Too few dims, protect minimally
        
        # Normalize by largest eigenvalue
        eigvals_norm = eigvals / (eigvals[0] + 1e-10)
        
        # Filter out near-zero eigenvalues (numerical noise)
        significant = eigvals_norm > 1e-6
        n_significant = significant.sum().item()
        
        if n_significant <= 1:
            return 1
        
        # Compute relative gaps: gap[i] = (λ_i - λ_{i+1}) / λ_i
        # This finds where eigenvalues drop off proportionally
        max_k = min(n_significant - 1, n - min_free_dims)
        if max_k <= 0:
            return 1
        
        eigvals_sig = eigvals_norm[:max_k + 1]
        rel_gaps = (eigvals_sig[:-1] - eigvals_sig[1:]) / (eigvals_sig[:-1] + 1e-10)
        
        # Find the largest relative gap
        gap_idx = rel_gaps.argmax().item()
        k = gap_idx + 1  # Protect dimensions above the gap
        
        # Fallback: if largest gap is tiny (smooth spectrum), use k_ratio
        if rel_gaps[gap_idx] < 0.1:
            # No clear gap → fall back to cumulative variance
            total_var = eigvals.sum()
            if total_var > 1e-10:
                cumvar = eigvals.cumsum(0) / total_var
                k = int((cumvar < self.ogp_k_ratio).sum().item()) + 1
        
        # Clamp: protect at least 1, leave at least min_free_dims free
        k = max(1, min(k, n - min_free_dims))
        return k
    
    @torch.no_grad()
    def save_gradient_subspace(self):
        """
        OGP: Save important eigenvectors of L, R factors at task boundary.
        
        Uses spectral gap heuristic: finds the natural drop-off in eigenvalue
        spectrum to automatically determine how many dimensions are "important"
        for this task. No k_ratio hyperparameter needed.
        """
        subspace = {}
        total_dims_saved = 0
        total_dims = 0
        layer_details = []
        
        for name, module in self._modules_tracked.items():
            L = self._stats[name]['L']
            R = self._stats[name]['R']
            
            if L is None or R is None:
                continue
            
            # Eigendecomposition of L (output gradient covariance)
            eigvals_L, eigvecs_L = torch.linalg.eigh(L)
            idx_L = eigvals_L.argsort(descending=True)
            eigvals_L = eigvals_L[idx_L]
            eigvecs_L = eigvecs_L[:, idx_L]
            
            # Spectral gap: auto-select k_L
            if eigvals_L.sum() > 1e-10:
                k_L = self._find_spectral_gap_k(eigvals_L)
                U_L = eigvecs_L[:, :k_L].clone()
            else:
                k_L = 0
                U_L = torch.empty(L.size(0), 0, device=L.device)
            
            # Eigendecomposition of R (input gradient covariance)
            eigvals_R, eigvecs_R = torch.linalg.eigh(R)
            idx_R = eigvals_R.argsort(descending=True)
            eigvals_R = eigvals_R[idx_R]
            eigvecs_R = eigvecs_R[:, idx_R]
            
            if eigvals_R.sum() > 1e-10:
                k_R = self._find_spectral_gap_k(eigvals_R)
                U_R = eigvecs_R[:, :k_R].clone()
            else:
                k_R = 0
                U_R = torch.empty(R.size(0), 0, device=R.device)
            
            subspace[name] = {'U_L': U_L, 'U_R': U_R}
            total_dims_saved += k_L + k_R
            total_dims += L.size(0) + R.size(0)
            layer_details.append(f"    {name}: L={k_L}/{L.size(0)}, R={k_R}/{R.size(0)}")
        
        self._saved_subspaces.append(subspace)
        self._update_projectors()
        
        pct_protected = (total_dims_saved / total_dims * 100) if total_dims > 0 else 0
        print(f"  → OGP subspace saved (task {len(self._saved_subspaces)}): "
              f"{total_dims_saved}/{total_dims} dims protected ({pct_protected:.1f}%) "
              f"[spectral gap adaptive]")
        for detail in layer_details:
            print(detail)
    
    @torch.no_grad()
    def _update_projectors(self):
        """
        Merge all saved subspaces and compute projection matrices.
        
        For each layer, concatenate eigenvectors from all tasks,
        compute orthonormal basis via QR, then build projector:
            P = I - Q·Qᵀ  (project into null space of protected subspace)
        """
        self._projectors.clear()
        
        for name in self._modules_tracked:
            # Collect U_L, U_R from all saved task subspaces
            U_L_list = [s[name]['U_L'] for s in self._saved_subspaces 
                        if name in s and s[name]['U_L'].size(1) > 0]
            U_R_list = [s[name]['U_R'] for s in self._saved_subspaces 
                        if name in s and s[name]['U_R'].size(1) > 0]
            
            if not U_L_list and not U_R_list:
                continue
            
            proj = {}
            
            if U_L_list:
                # Merge all task subspaces into one orthonormal basis
                U_L_merged = torch.cat(U_L_list, dim=1)
                Q_L, _ = torch.linalg.qr(U_L_merged)
                # Trim to rank (QR may produce more columns than rank)
                rank_L = min(Q_L.size(1), Q_L.size(0) - 1)  # Leave at least 1 dim free
                Q_L = Q_L[:, :rank_L]
                proj['P_L'] = torch.eye(Q_L.size(0), device=Q_L.device) - Q_L @ Q_L.T
            
            if U_R_list:
                U_R_merged = torch.cat(U_R_list, dim=1)
                Q_R, _ = torch.linalg.qr(U_R_merged)
                rank_R = min(Q_R.size(1), Q_R.size(0) - 1)
                Q_R = Q_R[:, :rank_R]
                proj['P_R'] = torch.eye(Q_R.size(0), device=Q_R.device) - Q_R @ Q_R.T
            
            self._projectors[name] = proj
    
    def get_ogp_stats(self):
        """Return statistics about OGP projection for monitoring."""
        stats = {}
        for name, proj in self._projectors.items():
            p_l_free = proj['P_L'].trace().item() / proj['P_L'].size(0) if 'P_L' in proj else 1.0
            p_r_free = proj['P_R'].trace().item() / proj['P_R'].size(0) if 'P_R' in proj else 1.0
            stats[name] = {
                'plasticity_L': p_l_free,  # fraction of free dimensions (left)
                'plasticity_R': p_r_free,  # fraction of free dimensions (right)
                'plasticity_total': p_l_free * p_r_free  # combined free capacity
            }
        return stats
    
    def reset_stats(self):
        """Reset Shampoo factors and preconditioner cache (called at task boundaries)."""
        for name in self._stats:
            self._stats[name] = {'L': None, 'R': None}
        self._precond.clear()
        # Reset Adam grafting state
        for name in self._adam_state:
            self._adam_state[name] = {'m': None, 'v': None}
        self.steps = 0
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Update Shampoo factors from current gradients
        self.update_factors()
        
        # Periodically recompute preconditioner roots
        if self.steps % self.T_precond == 0 and self.steps > 0:
            self.update_preconditioner()
        
        lr = self.param_groups[0]['lr']
        
        for name, module in self._modules_tracked.items():
            if module.weight.grad is None:
                continue
            
            grad_w = module.weight.grad
            
            if name in self._precond:
                L_inv_root = self._precond[name]['L_inv_root']
                R_inv_root = self._precond[name]['R_inv_root']
                
                # Reshape to 2D
                if isinstance(module, nn.Conv2d):
                    c_out = grad_w.size(0)
                    grad_2d = grad_w.reshape(c_out, -1)
                    if module.bias is not None and module.bias.grad is not None:
                        grad_2d = torch.cat([grad_2d, module.bias.grad.unsqueeze(1)], dim=1)
                    
                    # Shampoo preconditioned gradient: L^{-1/p} G R^{-1/p}
                    shampoo_grad = L_inv_root @ grad_2d @ R_inv_root
                    
                    # OGP: project into null space of old task subspaces
                    if self.cl_mode == 'ogp' and name in self._projectors:
                        P_L = self._projectors[name].get('P_L')
                        P_R = self._projectors[name].get('P_R')
                        if P_L is not None and P_R is not None:
                            shampoo_grad = P_L @ shampoo_grad @ P_R
                        elif P_L is not None:
                            shampoo_grad = P_L @ shampoo_grad
                        elif P_R is not None:
                            shampoo_grad = shampoo_grad @ P_R
                    
                    # Adam grafting: scale Shampoo direction by Adam step size
                    if self.grafting == 'adam':
                        shampoo_grad = self._apply_adam_grafting(name, grad_2d, shampoo_grad)
                    
                    if module.bias is not None and module.bias.grad is not None:
                        grad_w_update = shampoo_grad[:, :-1].reshape_as(module.weight)
                        grad_b_update = shampoo_grad[:, -1]
                        if self.param_groups[0]['weight_decay'] > 0:
                            grad_w_update.add_(module.weight, alpha=self.param_groups[0]['weight_decay'])
                        module.weight.data.add_(grad_w_update, alpha=-lr)
                        module.bias.data.add_(grad_b_update, alpha=-lr)
                    else:
                        shampoo_grad = shampoo_grad.reshape_as(module.weight)
                        if self.param_groups[0]['weight_decay'] > 0:
                            shampoo_grad.add_(module.weight, alpha=self.param_groups[0]['weight_decay'])
                        module.weight.data.add_(shampoo_grad, alpha=-lr)
                
                elif isinstance(module, nn.Linear):
                    if module.bias is not None and module.bias.grad is not None:
                        grad_2d = torch.cat([grad_w, module.bias.grad.unsqueeze(1)], dim=1)
                    else:
                        grad_2d = grad_w
                    
                    shampoo_grad = L_inv_root @ grad_2d @ R_inv_root
                    
                    # OGP: project into null space of old task subspaces
                    if self.cl_mode == 'ogp' and name in self._projectors:
                        P_L = self._projectors[name].get('P_L')
                        P_R = self._projectors[name].get('P_R')
                        if P_L is not None and P_R is not None:
                            shampoo_grad = P_L @ shampoo_grad @ P_R
                        elif P_L is not None:
                            shampoo_grad = P_L @ shampoo_grad
                        elif P_R is not None:
                            shampoo_grad = shampoo_grad @ P_R
                    
                    if self.grafting == 'adam':
                        shampoo_grad = self._apply_adam_grafting(name, grad_2d, shampoo_grad)
                    
                    if module.bias is not None and module.bias.grad is not None:
                        grad_w_update = shampoo_grad[:, :-1]
                        grad_b_update = shampoo_grad[:, -1]
                        if self.param_groups[0]['weight_decay'] > 0:
                            grad_w_update.add_(module.weight, alpha=self.param_groups[0]['weight_decay'])
                        module.weight.data.add_(grad_w_update, alpha=-lr)
                        module.bias.data.add_(grad_b_update, alpha=-lr)
                    else:
                        if self.param_groups[0]['weight_decay'] > 0:
                            shampoo_grad.add_(module.weight, alpha=self.param_groups[0]['weight_decay'])
                        module.weight.data.add_(shampoo_grad, alpha=-lr)
            else:
                # Before first preconditioner: use SGD (or Adam grafting)
                if self.param_groups[0]['weight_decay'] > 0:
                    grad_w = grad_w + self.param_groups[0]['weight_decay'] * module.weight
                module.weight.data.add_(grad_w, alpha=-lr)
                if module.bias is not None and module.bias.grad is not None:
                    module.bias.data.add_(module.bias.grad, alpha=-lr)
        
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
                if group['weight_decay'] > 0:
                    p.data.add_(p, alpha=-group['weight_decay'] * group['lr'])
                p.data.add_(p.grad, alpha=-group['lr'])
        
        self.steps += 1
        return loss
    
    @torch.no_grad()
    def _apply_adam_grafting(self, name, raw_grad, shampoo_grad):
        """
        Adam grafting: use Shampoo direction but scale by Adam step size.
        This stabilizes Shampoo and works better empirically.
        """
        if self._adam_state[name]['m'] is None:
            self._adam_state[name]['m'] = torch.zeros_like(raw_grad)
            self._adam_state[name]['v'] = torch.zeros_like(raw_grad)
        
        m = self._adam_state[name]['m']
        v = self._adam_state[name]['v']
        
        m.mul_(self.adam_beta1).add_(raw_grad, alpha=1 - self.adam_beta1)
        v.mul_(self.adam_beta2).add_(raw_grad.pow(2), alpha=1 - self.adam_beta2)
        
        # Bias correction
        t = self.steps + 1
        m_hat = m / (1 - self.adam_beta1 ** t)
        v_hat = v / (1 - self.adam_beta2 ** t)
        
        adam_step = m_hat / (v_hat.sqrt() + self.adam_eps)
        
        # Grafting: scale Shampoo direction by Adam norm ratio
        adam_norm = adam_step.norm()
        shampoo_norm = shampoo_grad.norm()
        
        if shampoo_norm > 0:
            return shampoo_grad * (adam_norm / shampoo_norm)
        return adam_step  # fallback to Adam
    
    def compute_shampoo_erank_stats(self):
        """Compute effective rank of Shampoo L, R factors for monitoring."""
        eranks_L = []
        eranks_R = []
        cond_numbers = []
        layer_details = []
        
        for name in self._stats:
            L = self._stats[name]['L']
            R = self._stats[name]['R']
            if L is None or R is None:
                continue
            
            er_L = compute_effective_rank(L)
            er_R = compute_effective_rank(R)
            eranks_L.append(er_L)
            eranks_R.append(er_R)
            
            eigvals_L = torch.linalg.eigvalsh(L)
            cond = (eigvals_L.max() / eigvals_L.clamp(min=1e-10).min()).item()
            cond_numbers.append(cond)
            
            rel_L = er_L / L.size(0) * 100
            rel_R = er_R / R.size(0) * 100
            layer_details.append({
                'name': name, 'erank_L': er_L, 'erank_R': er_R,
                'rel_L': rel_L, 'rel_R': rel_R,
                'dim_L': L.size(0), 'dim_R': R.size(0), 'cond': cond
            })
        
        mean_rel_L = np.mean([d['rel_L'] for d in layer_details]) if layer_details else 0
        mean_rel_R = np.mean([d['rel_R'] for d in layer_details]) if layer_details else 0
        mean_cond = np.mean(cond_numbers) if cond_numbers else 0
        
        return {
            'mean_rel_erank_L': mean_rel_L,
            'mean_rel_erank_R': mean_rel_R,
            'mean_condition_number': mean_cond,
            'per_layer': layer_details
        }

print("✓ ShampooCLOptimizer defined")

# %% [markdown]
# ## 6. Experiment Configuration — Ablation Settings
#
# We compare 3 variants:
# - **Shampoo-OGP (full)**: Shampoo + Orthogonal Gradient Projection + Adam grafting
# - **SGD baseline**: Plain SGD, no Shampoo, no CL protection
# - **Shampoo (no CL)**: Shampoo preconditioner only, no continual learning protection
#
# All use same hyperparameters except where noted.

# %%
num_classes = 100

# Shampoo-OGP configuration (default: Orthogonal Gradient Projection)
shampoo_config = {
    'num_epochs': 1200,
    'lr': 0.005,
    'damping': 1e-4,
    'weight_decay': 0.0,
    'T_precond': 50,           # Eigendecomp every 50 steps (Shampoo default)
    'beta2': 0.999,             # EMA decay for L, R factors
    'power': 4,                 # L^{-1/4} G R^{-1/4} (standard Shampoo)
    'cl_mode': 'ogp',           # 'ogp' = Orthogonal Gradient Projection, 'ewc' = legacy EWC
    'ogp_k_ratio': 0.8,         # Protect 80% of gradient variance per task
    'grafting': 'adam',          # Adam grafting for stability
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_eps': 1e-8,
    'batch_size': 90,
    'class_increase_frequency': 200,
    'checkpoint_frequency': 500,
    'random_seed': 42,
    'method': 'shampoo_ogp'
}

# Ablation variants
ablation_configs = {
    'shampoo_ogp': {**shampoo_config, 'method': 'shampoo_ogp'},
    'sgd_baseline': {
        **shampoo_config,
        # Override to match Nature paper's base_deep_learning_system.json exactly:
        'lr': 0.1,                  # paper: stepsize=0.1 (NOT 0.005)
        'weight_decay': 0.0005,     # paper: weight_decay=5e-4 (NOT 0)
        'num_epochs': 4000,         # paper: 4000 epochs / 20 tasks (NOT 1200)
        'cl_mode': 'none',
        'T_precond': 999999,
        'grafting': 'none',
        'method': 'sgd_baseline',
    },
    'cbp': {
        **shampoo_config,
        # Override to match Nature paper's continual_backpropagation.json exactly:
        'lr': 0.1,                  # paper: stepsize=0.1
        'weight_decay': 0.0005,     # paper: weight_decay=5e-4
        'num_epochs': 4000,         # paper: 4000 epochs / 20 tasks
        'cl_mode': 'none',
        'T_precond': 999999,
        'grafting': 'none',
        'method': 'cbp',
        # CBP-specific (ResGnT generate-and-test)
        'replacement_rate': 0.00001,        # paper: 1e-5
        'utility_function': 'contribution', # paper: contribution-based utility
        'maturity_threshold': 1000,         # paper: 1000 steps before eligible for replacement
        'early_stopping': True,             # paper: uses early stopping
    },
    'shampoo_no_cl': {**shampoo_config, 'cl_mode': 'none', 'method': 'shampoo_no_cl'},
}

# Select which ablation to run (change this to run different variants)
ABLATION_NAME = 'shampoo_ogp'  # Change to 'sgd_baseline' or 'shampoo_no_cl'
active_config = ablation_configs[ABLATION_NAME]

print(f"\n{'='*60}")
print(f"Running ablation: {ABLATION_NAME}")
print(f"{'='*60}")
print(f"\nShampoo-CL Configuration:")
for k, v in active_config.items():
    print(f"  - {k}: {v}")

# %% [markdown]
# ## 7. Initialize Model and Optimizer

# %%
print(f"\nInitializing ResNet-18 for {ABLATION_NAME} experiment...")
net = build_resnet18(num_classes=num_classes, norm_layer=nn.BatchNorm2d)
net.apply(kaiming_init_resnet_module)
net.to(device)

if active_config['method'] in ('sgd_baseline', 'cbp'):
    # Pure SGD with momentum — matching Nature paper protocol
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=active_config['lr'],
        momentum=0.9,
        weight_decay=active_config['weight_decay'],
    )
    print(f"  Using pure torch.optim.SGD (lr={active_config['lr']}, momentum=0.9, wd={active_config['weight_decay']})")
else:
    # Shampoo-CL optimizer (with OGP or EWC)
    optimizer = ShampooCLOptimizer(
        net,
        lr=active_config['lr'],
        damping=active_config['damping'],
        weight_decay=active_config['weight_decay'],
        T_precond=active_config['T_precond'],
        beta2=active_config['beta2'],
        power=active_config['power'],
        grafting=active_config['grafting'],
        adam_beta1=active_config['adam_beta1'],
        adam_beta2=active_config['adam_beta2'],
        adam_eps=active_config['adam_eps'],
        cl_mode=active_config.get('cl_mode', 'ogp'),
        ogp_k_ratio=active_config.get('ogp_k_ratio', 0.8),
        ewc_lambda=active_config.get('ewc_lambda', 0.0),
    )

is_shampoo = isinstance(optimizer, ShampooCLOptimizer)
is_cbp = active_config['method'] == 'cbp'

# Initialize CBP (ResGnT = Generate-and-Test for ResNets)
resgnt = None
if is_cbp:
    resgnt = ResGnT(
        net=net,
        hidden_activation="relu",
        replacement_rate=active_config['replacement_rate'],
        decay_rate=0.99,
        util_type=active_config['utility_function'],
        maturity_threshold=active_config['maturity_threshold'],
        device=device,
    )
    print(f"  ✓ CBP (ResGnT): replacement_rate={active_config['replacement_rate']}, "
          f"utility={active_config['utility_function']}, maturity={active_config['maturity_threshold']}")

loss_fn = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in net.parameters())
print(f"✓ Total params: {total_params:,}")
print(f"✓ Model initialized for {ABLATION_NAME}")

# %% [markdown]
# ## 8. Setup Data Loaders

# %%
current_num_classes = 5
all_classes = np.random.RandomState(active_config['random_seed']).permutation(num_classes)

train_data_exp = copy.deepcopy(train_data)
val_data_exp = copy.deepcopy(val_data)
test_data_exp = copy.deepcopy(test_data)

train_data_exp.select_new_partition(all_classes[:current_num_classes])
val_data_exp.select_new_partition(all_classes[:current_num_classes])
test_data_exp.select_new_partition(all_classes[:current_num_classes])

train_loader = DataLoader(train_data_exp, batch_size=active_config['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_data_exp, batch_size=50, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data_exp, batch_size=100, shuffle=False, num_workers=0)

# Dormant measurement loaders (matching post_run_analysis.py exactly)
# Paper uses FULL CIFAR-100 training set (not subsampled), eval transforms, batch_size=1000
# Paper creates a fresh model and measures in train mode — we save/restore BN state instead
dormant_next_data = copy.deepcopy(train_data_full)  # NO subsample — full 50k train set
dormant_next_data.set_transformation(eval_transformations)
next_start_d = current_num_classes
next_end_d = min(current_num_classes + 5, num_classes)
if next_start_d < num_classes:
    dormant_next_data.select_new_partition(all_classes[next_start_d:next_end_d])
    dormant_next_loader = DataLoader(dormant_next_data, batch_size=1000, shuffle=True, num_workers=0)
else:
    dormant_next_loader = None

dormant_prev_data = copy.deepcopy(train_data_full)  # NO subsample — full 50k train set
dormant_prev_data.set_transformation(eval_transformations)
dormant_prev_loader = None  # no previous tasks at start

print(f"✓ Data loaders initialized with {current_num_classes} classes")

# %% [markdown]
# ## 9. Shampoo-CL Training Loop
#
# Key differences from K-FAC (aimo-ans4) training:
# 1. **No forward hooks** — Shampoo uses only gradient covariance (L=GGᵀ, R=GᵀG)
# 2. **EWC loss** added to cross-entropy for continual learning
# 3. **Shampoo factors** updated from gradients after backward pass
# 4. **EWC snapshot** saved at each task boundary using free Fisher from diag(L⊗R)
# 5. **Adam grafting** stabilizes Shampoo direction with Adam step sizes

# %%
num_epochs = active_config['num_epochs']
class_increase_frequency = active_config['class_increase_frequency']
checkpoint_frequency = active_config['checkpoint_frequency']
n_train_data = len(train_data.data['data'])

checkpoint_dir = os.path.join(results_dir, f"cifar100_{ABLATION_NAME}", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Metric storage
train_losses_all = []
ewc_losses_all = []
train_accuracies_all = []
val_accuracies_all = []
test_accuracies_all = []
dormant_after_all = []
dormant_before_all = []
stable_ranks_all = []
avg_weight_mag_all = []
shampoo_erank_all = []
epoch_runtimes_all = []

# Early stopping (matching paper: best val accuracy model carries forward at task boundary)
use_early_stopping = active_config.get('early_stopping', True)  # paper uses it for all methods
best_accuracy = torch.tensor(0.0, device=device)
best_accuracy_model_parameters = {}

print(f"\nStarting {ABLATION_NAME} training for {num_epochs} epochs...")
print(f"Incremental: start with {current_num_classes} classes, add 5 every {class_increase_frequency} epochs")
print(f"Config: cl_mode={active_config.get('cl_mode', 'ogp')}, grafting={active_config['grafting']}, power={active_config['power']}")
if use_early_stopping:
    print(f"Early stopping: ON (best val model loaded at task boundaries)")

net.train()
for epoch in range(num_epochs):
    epoch_start_time = time.time()

    # === LR schedule (matching Nature paper: decay ×0.2 at epoch 60/120/160 within each task) ===
    # Only for SGD-based methods (sgd_baseline, cbp). Shampoo uses constant lr (preconditioning adapts step sizes).
    if not is_shampoo:
        epoch_in_task = epoch % class_increase_frequency
        base_lr = active_config['lr']
        if epoch_in_task == 0:
            current_lr = base_lr
        elif epoch_in_task == 60:
            current_lr = round(base_lr * 0.2, 5)
        elif epoch_in_task == 120:
            current_lr = round(base_lr * 0.2 ** 2, 5)
        elif epoch_in_task == 160:
            current_lr = round(base_lr * 0.2 ** 3, 5)
        else:
            current_lr = None  # no change
        if current_lr is not None:
            for g in optimizer.param_groups:
                g['lr'] = current_lr

    # Measure stable rank at task boundaries
    if epoch % class_increase_frequency == 0:
        next_task_start = current_num_classes
        next_task_end = min(current_num_classes + 5, num_classes)
        next_task_classes = all_classes[next_task_start:next_task_end]

        if len(next_task_classes) > 0:
            temp_data = copy.deepcopy(train_data_exp)
            temp_data.select_new_partition(next_task_classes)
            temp_loader = DataLoader(temp_data, batch_size=active_config['batch_size'], shuffle=False, num_workers=0)

            net.eval()
            all_activations = []
            samples_collected = 0

            with torch.no_grad():
                for sample in temp_loader:
                    if samples_collected >= 1000:
                        break
                    images = sample["image"].to(device)
                    features_per_layer = []
                    net.forward(images, features_per_layer)
                    all_activations.append(features_per_layer[-1].cpu())
                    samples_collected += images.shape[0]

            if len(all_activations) > 0:
                last_layer_act = torch.cat(all_activations, dim=0)[:1000].numpy()
                stable_rank_val = compute_stable_rank_from_activations(last_layer_act)
                stable_ranks_all.append(stable_rank_val)
                print(f"  → Stable Rank (before training task {len(stable_ranks_all)}): {stable_rank_val:.2f} ({stable_rank_val/512*100:.1f}%)")

            net.train()

    # Add new classes at task boundaries
    if epoch > 0 and (epoch % class_increase_frequency) == 0 and current_num_classes < num_classes:
        # === SAVE CL STATE BEFORE MOVING TO NEXT TASK ===
        cl_mode = active_config.get('cl_mode', 'ogp')

        # Early stopping: load best validation model (matching paper)
        if use_early_stopping and best_accuracy_model_parameters:
            net.load_state_dict(best_accuracy_model_parameters)
            print(f"  → Loaded best validation model (acc={best_accuracy.item():.4f})")
        best_accuracy = torch.zeros_like(best_accuracy)
        best_accuracy_model_parameters = {}

        if is_shampoo and cl_mode == 'ogp':
            optimizer.save_gradient_subspace()
        elif is_shampoo and cl_mode == 'ewc' and active_config.get('ewc_lambda', 0) > 0:
            optimizer.save_ewc_snapshot()
        
        current_num_classes = min(current_num_classes + 5, num_classes)
        train_data_exp.select_new_partition(all_classes[:current_num_classes])
        val_data_exp.select_new_partition(all_classes[:current_num_classes])
        test_data_exp.select_new_partition(all_classes[:current_num_classes])

        train_loader = DataLoader(train_data_exp, batch_size=active_config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data_exp, batch_size=50, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_data_exp, batch_size=100, shuffle=False, num_workers=0)

        print(f"\n{'='*60}")
        print(f"{ABLATION_NAME}: New task at epoch {epoch} → {current_num_classes} classes")
        print(f"{'='*60}")

        # Reset Shampoo factors for fresh curvature estimation
        if is_shampoo:
            optimizer.reset_stats()

        # Update dormant loaders (matching post_run_analysis.py)
        next_start_d = current_num_classes
        next_end_d = min(current_num_classes + 5, num_classes)
        if next_start_d < num_classes:
            dormant_next_data.select_new_partition(all_classes[next_start_d:next_end_d])
            dormant_next_loader = DataLoader(dormant_next_data, batch_size=1000, shuffle=True, num_workers=0)
        else:
            dormant_next_loader = None
        prev_end_d = current_num_classes  # all classes trained so far
        if prev_end_d > 0:
            dormant_prev_data.select_new_partition(all_classes[:prev_end_d])
            dormant_prev_loader = DataLoader(dormant_prev_data, batch_size=1000, shuffle=True, num_workers=0)
        else:
            dormant_prev_loader = None

    print(f"\nEpoch {epoch+1}/{num_epochs} | Classes: {current_num_classes}")

    # === Training ===
    running_loss = 0
    running_ewc = 0
    running_acc = 0
    num_batches = 0
    cl_mode = active_config.get('cl_mode', 'ogp')

    for sample in tqdm(train_loader, desc=f"{ABLATION_NAME} epoch {epoch+1}"):
        images = sample["image"].to(device)
        labels = sample["label"].to(device)

        net.train()

        # 1. Forward pass (collect features for CBP if needed)
        if is_cbp:
            current_features = []
            predictions = net.forward(images, current_features)
        else:
            predictions = net(images)
        predictions_masked = predictions[:, all_classes[:current_num_classes]]

        # 2. Compute loss
        ce_loss = loss_fn(predictions_masked,
                          labels.argmax(dim=1) if labels.dim() > 1 and labels.shape[1] > 1 else labels)
        
        if is_shampoo and cl_mode == 'ewc':
            ewc_loss = optimizer.compute_ewc_loss()
            total_loss = ce_loss + ewc_loss
        else:
            # OGP: no penalty in loss — projection happens in step()
            ewc_loss = torch.tensor(0.0)
            total_loss = ce_loss

        # 3. Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # 4. Optimizer step (Shampoo preconditioned + OGP if enabled, or plain SGD)
        optimizer.step()

        # 5. CBP: Generate-and-Test on collected features (after optimizer step, matching paper)
        if is_cbp and resgnt is not None:
            resgnt.gen_and_test(current_features)

        with torch.no_grad():
            acc = (predictions_masked.argmax(dim=1) == (labels.argmax(dim=1) if labels.dim() > 1 and labels.shape[1] > 1 else labels)).float().mean()
        running_loss += ce_loss.item()
        running_ewc += ewc_loss.item()
        running_acc += acc.item()
        num_batches += 1

    train_losses_all.append(running_loss / num_batches)
    ewc_losses_all.append(running_ewc / num_batches)
    train_accuracies_all.append(running_acc / num_batches)

    # === Validation ===
    net.eval()
    val_acc = 0
    val_batches = 0
    with torch.no_grad():
        for sample in val_loader:
            images = sample["image"].to(device)
            labels = sample["label"].to(device)
            predictions = net(images)
            predictions_masked = predictions[:, all_classes[:current_num_classes]]
            acc = (predictions_masked.argmax(dim=1) == (labels.argmax(dim=1) if labels.dim() > 1 and labels.shape[1] > 1 else labels)).float().mean()
            val_acc += acc.item()
            val_batches += 1
    val_accuracies_all.append(val_acc / val_batches if val_batches > 0 else 0)

    # Early stopping: track best validation model (matching paper)
    current_val_accuracy = val_accuracies_all[-1]
    if use_early_stopping and current_val_accuracy > best_accuracy.item():
        best_accuracy = torch.tensor(current_val_accuracy, device=device)
        best_accuracy_model_parameters = copy.deepcopy(net.state_dict())

    # === Test ===
    test_acc = 0
    test_batches = 0
    with torch.no_grad():
        for sample in test_loader:
            images = sample["image"].to(device)
            labels = sample["label"].to(device)
            predictions = net(images)
            predictions_masked = predictions[:, all_classes[:current_num_classes]]
            acc = (predictions_masked.argmax(dim=1) == (labels.argmax(dim=1) if labels.dim() > 1 and labels.shape[1] > 1 else labels)).float().mean()
            test_acc += acc.item()
            test_batches += 1
    test_accuracies_all.append(test_acc / test_batches if test_batches > 0 else 0)

    # === Dormant Measurement (matching post_run_analysis.py 100%) ===
    # Paper: fresh model (train mode) + @torch.no_grad() + next/prev task data
    # We emulate fresh model by saving/restoring BN running stats
    bn_state_backup = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_state_backup[name] = {
                'running_mean': m.running_mean.clone(),
                'running_var': m.running_var.clone(),
                'num_batches_tracked': m.num_batches_tracked.clone(),
            }

    net.train()  # paper uses default train mode (fresh model)
    with torch.no_grad():  # matching paper's @torch.no_grad() decorator
        if dormant_next_loader is not None:
            dormant_after, _ = compute_dormant_units_proportion(net, dormant_next_loader, dormant_unit_threshold=0.01)
        else:
            dormant_after = float('nan')
        dormant_after_all.append(dormant_after)

        # Restore BN before measuring prev (paper loads fresh model each time)
        for name, m in net.named_modules():
            if isinstance(m, nn.BatchNorm2d) and name in bn_state_backup:
                m.running_mean.copy_(bn_state_backup[name]['running_mean'])
                m.running_var.copy_(bn_state_backup[name]['running_var'])
                m.num_batches_tracked.copy_(bn_state_backup[name]['num_batches_tracked'])

        if dormant_prev_loader is not None:
            dormant_before, _ = compute_dormant_units_proportion(net, dormant_prev_loader, dormant_unit_threshold=0.01)
        else:
            dormant_before = float('nan')
        dormant_before_all.append(dormant_before)

    # Restore BN running stats (prevent corruption of training model)
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and name in bn_state_backup:
            m.running_mean.copy_(bn_state_backup[name]['running_mean'])
            m.running_var.copy_(bn_state_backup[name]['running_var'])
            m.num_batches_tracked.copy_(bn_state_backup[name]['num_batches_tracked'])

    avg_weight_mag = compute_avg_weight_magnitude(net)
    avg_weight_mag_all.append(avg_weight_mag)

    net.train()

    epoch_runtime = time.time() - epoch_start_time
    epoch_runtimes_all.append(epoch_runtime)

    print(f"  Train: {train_accuracies_all[-1]:.4f} | Val: {val_accuracies_all[-1]:.4f} | Test: {test_accuracies_all[-1]:.4f}")
    if is_shampoo and cl_mode == 'ogp':
        n_tasks_ogp = len(optimizer._saved_subspaces)
        if n_tasks_ogp > 0:
            ogp_stats = optimizer.get_ogp_stats()
            avg_plasticity = sum(s['plasticity_total'] for s in ogp_stats.values()) / len(ogp_stats) if ogp_stats else 1.0
            print(f"  CE Loss: {train_losses_all[-1]:.4f} | OGP tasks: {n_tasks_ogp} | Plasticity: {avg_plasticity*100:.1f}%")
        else:
            print(f"  CE Loss: {train_losses_all[-1]:.4f} | OGP: no projection yet (task 1)")
    elif is_shampoo and cl_mode == 'ewc':
        ewc_ce_ratio = ewc_losses_all[-1] / (train_losses_all[-1] + 1e-8)
        print(f"  CE Loss: {train_losses_all[-1]:.4f} | EWC Loss: {ewc_losses_all[-1]:.4f} | EWC/CE: {ewc_ce_ratio:.2f} | λ={optimizer.ewc_lambda:.1f}")
    else:
        print(f"  CE Loss: {train_losses_all[-1]:.4f} | No CL protection")
    print(f"  Dormant(next): {dormant_after:.4f} | Dormant(prev): {dormant_before:.4f} | Avg Weight: {avg_weight_mag:.4f} | Epoch time: {epoch_runtime:.1f}s")

    # === Shampoo eRank (every 50 epochs, only for Shampoo optimizer) ===
    if is_shampoo and (epoch % 50 == 0 or epoch == num_epochs - 1):
        erank_stats = optimizer.compute_shampoo_erank_stats()
        shampoo_erank_all.append({
            'epoch': epoch,
            'mean_rel_erank_L': erank_stats['mean_rel_erank_L'],
            'mean_rel_erank_R': erank_stats['mean_rel_erank_R'],
            'mean_condition_number': erank_stats['mean_condition_number'],
            'per_layer': erank_stats['per_layer']
        })
        print(f"  Shampoo eRank: L={erank_stats['mean_rel_erank_L']:.1f}% | R={erank_stats['mean_rel_erank_R']:.1f}% | Cond={erank_stats['mean_condition_number']:.1f}")

    # Checkpoint
    if (epoch + 1) % checkpoint_frequency == 0 or (epoch + 1) == num_epochs:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'current_num_classes': current_num_classes,
            'train_losses': train_losses_all,
            'ewc_losses': ewc_losses_all,
            'train_accuracies': train_accuracies_all,
            'val_accuracies': val_accuracies_all,
            'test_accuracies': test_accuracies_all,
            'dormant_after': dormant_after_all,
            'dormant_before': dormant_before_all,
            'stable_ranks': stable_ranks_all,
            'config': active_config
        }, checkpoint_path)
        print(f"  → Checkpoint saved: {checkpoint_path}")

print(f"\n✓ {ABLATION_NAME} training completed!")

# %% [markdown]
# ## 10. Save Results

# %%
results = {
    'train_losses': train_losses_all,
    'ewc_losses': ewc_losses_all,
    'train_accuracies': train_accuracies_all,
    'val_accuracy_per_epoch': val_accuracies_all,
    'test_accuracy_per_epoch': test_accuracies_all,
    'epoch_runtime': epoch_runtimes_all,
    'dormant_after_per_epoch': dormant_after_all,
    'dormant_before_per_epoch': dormant_before_all,
    'stable_rank_per_task': stable_ranks_all,
    'avg_weight_magnitude_per_epoch': avg_weight_mag_all,
    'shampoo_erank': shampoo_erank_all,
    'cl_mode': active_config.get('cl_mode', 'ogp'),
    'ogp_tasks_saved': len(optimizer._saved_subspaces) if is_shampoo else 0,
    'ogp_final_plasticity': optimizer.get_ogp_stats() if (is_shampoo and active_config.get('cl_mode') == 'ogp') else {},
    'config': active_config,
    'class_order': all_classes.tolist()
}

save_dir = os.path.join(results_dir, f"cifar100_{ABLATION_NAME}")
os.makedirs(save_dir, exist_ok=True)
save_results(results, os.path.join(save_dir, f'{ABLATION_NAME}_results.pkl'))

print(f"\n✓ Saved {ABLATION_NAME} metrics:")
print(f"  - Epochs: {len(test_accuracies_all)}")
print(f"  - Dormant analysis: {len(dormant_after_all)} after + {len(dormant_before_all)} before")
print(f"  - Stable rank analysis: {len(stable_ranks_all)} measurements")
print(f"  - Shampoo eRank: {len(shampoo_erank_all)} measurements")

# %% [markdown]
# ## 11. Ablation Comparison Visualization
#
# After running all 3 ablations + K-FAC baseline (aimo-ans4),
# load results and plot comparison.

# %%
# Load all results for comparison
import glob

all_results = {}
result_files = {
    'shampoo_cl': os.path.join(results_dir, 'cifar100_shampoo_cl', 'shampoo_cl_results.pkl'),
    'shampoo_no_ewc': os.path.join(results_dir, 'cifar100_shampoo_no_ewc', 'shampoo_no_ewc_results.pkl'),
    'shampoo_cl_sgd_graft': os.path.join(results_dir, 'cifar100_shampoo_cl_sgd_graft', 'shampoo_cl_sgd_graft_results.pkl'),
    'kfac': os.path.join(results_dir, 'cifar100_kfac', 'kfac_results.pkl'),
}

for name, path in result_files.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            all_results[name] = pickle.load(f)
        print(f"✓ Loaded {name}: {len(all_results[name].get('test_accuracy_per_epoch', []))} epochs")
    else:
        print(f"✗ {name} not found at {path}")

if len(all_results) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Ablation Study: Shampoo-CL vs K-FAC', fontsize=16, fontweight='bold')
    
    colors = {'shampoo_cl': '#2196F3', 'shampoo_no_ewc': '#FF9800', 
              'shampoo_cl_sgd_graft': '#4CAF50', 'kfac': '#F44336'}
    labels = {'shampoo_cl': 'Shampoo-CL (full)', 'shampoo_no_ewc': 'Shampoo (no EWC)', 
              'shampoo_cl_sgd_graft': 'Shampoo-CL (no graft)', 'kfac': 'K-FAC'}
    
    # 1. Test Accuracy
    ax = axes[0, 0]
    for name, res in all_results.items():
        ax.plot(res.get('test_accuracy_per_epoch', []), color=colors[name], label=labels[name], alpha=0.8)
    ax.set_title('Test Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Task boundaries
    for i in range(1, 20):
        ax.axvline(x=i*200, color='gray', linestyle='--', alpha=0.3)
    
    # 2. Train vs Val gap (overfitting)
    ax = axes[0, 1]
    for name, res in all_results.items():
        train = np.array(res.get('train_accuracies', []))
        val = np.array(res.get('val_accuracy_per_epoch', []))
        if len(train) > 0 and len(val) > 0:
            min_len = min(len(train), len(val))
            gap = train[:min_len] - val[:min_len]
            ax.plot(gap, color=colors[name], label=labels[name], alpha=0.8)
    ax.set_title('Overfitting Gap (Train - Val)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. CE Loss
    ax = axes[0, 2]
    for name, res in all_results.items():
        ax.plot(res.get('train_losses', []), color=colors[name], label=labels[name], alpha=0.8)
    ax.set_title('Cross-Entropy Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CE Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Dormant Units (next task)
    ax = axes[1, 0]
    for name, res in all_results.items():
        ax.plot(res.get('dormant_after_per_epoch', []), color=colors[name], label=labels[name], alpha=0.8)
    ax.set_title('Dormant Units (Next Task)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Proportion')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. Weight Magnitude 
    ax = axes[1, 1]
    for name, res in all_results.items():
        ax.plot(res.get('avg_weight_magnitude_per_epoch', []), color=colors[name], label=labels[name], alpha=0.8)
    ax.set_title('Average Weight Magnitude')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Magnitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 6. Epoch Runtime
    ax = axes[1, 2]
    for name, res in all_results.items():
        if 'ewc_losses' in res:
            ax.plot(res['ewc_losses'], color=colors[name], label=f"{labels[name]} (EWC)", alpha=0.8)
        elif 'epoch_runtime' in res:
            ax.plot(res['epoch_runtime'], color=colors[name], label=f"{labels[name]}", alpha=0.8, linestyle='--')
    ax.set_title('EWC Loss / Epoch Runtime')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(results_dir, 'ablation_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Comparison plot saved to {fig_path}")
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<25} {'Final Test Acc':>15} {'Best Test Acc':>15} {'Avg Overfit Gap':>18}")
    print(f"{'='*80}")
    for name, res in all_results.items():
        test_acc = res.get('test_accuracy_per_epoch', [])
        train_acc = res.get('train_accuracies', [])
        val_acc = res.get('val_accuracy_per_epoch', [])
        final = test_acc[-1] if test_acc else 0
        best = max(test_acc) if test_acc else 0
        if train_acc and val_acc:
            min_len = min(len(train_acc), len(val_acc))
            avg_gap = np.mean(np.array(train_acc[:min_len]) - np.array(val_acc[:min_len]))
        else:
            avg_gap = 0
        print(f"{labels[name]:<25} {final:>15.4f} {best:>15.4f} {avg_gap:>18.4f}")
    print(f"{'='*80}")
else:
    print("No results files found. Run training first!")
