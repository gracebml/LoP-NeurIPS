# %% [markdown]
# # Ablation: Is Loss of Plasticity = Spectral Pathology?
#
# **Hypothesis**: LoP is fundamentally caused by spectral pathology (singular value
# divergence) in weight matrices, NOT by optimizer choice. If true, SDP alone
# should prevent LoP even with a simple first-order optimizer.
#
# **Design**: Replace SASSHA (second-order) with Adam (first-order) and test:
# 1. **Adam** — baseline, no spectral intervention
# 2. **Adam + SDP** — first-order + spectral fix at task boundaries
# 3. **Adam + Soft Rescale** — first-order + simple weight scaling (non-spectral control)
#
# If Adam+SDP >> Adam ≈ Adam+SoftRescale → LoP = spectral pathology.
# If Adam+SDP ≈ Adam → LoP requires second-order correction too.
#
# **Setup**: Incremental CIFAR-100, ResNet-18 (BatchNorm), 5→100 classes,
# 4000 epochs, +5 classes every 200 epochs.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import os, sys, time, copy, math, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

results_dir = "sdp_ablation_results"
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
    except Exception:
        return 0

@torch.no_grad()
def compute_weight_spectral_stats(net):
    """Compute spectral statistics of all weight matrices.
    Returns: mean condition number, mean top-1 SV ratio, mean stable rank."""
    cond_numbers = []
    sv_ratios = []
    stable_ranks = []
    for module in net.modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        W = module.weight.data
        original_shape = W.shape
        if W.dim() > 2:
            W = W.reshape(W.shape[0], -1)
        if W.shape[0] == 0 or W.shape[1] == 0:
            continue
        try:
            S = torch.linalg.svdvals(W)
        except Exception:
            continue
        if S.numel() == 0 or S[0] < 1e-12:
            continue
        cond_numbers.append((S[0] / S[-1].clamp(min=1e-12)).item())
        sv_ratios.append((S[0] / S.sum()).item())
        sv_np = S.cpu().numpy()
        stable_ranks.append(compute_stable_rank(sv_np))
    if not cond_numbers:
        return 0.0, 0.0, 0.0
    return np.mean(cond_numbers), np.mean(sv_ratios), np.mean(stable_ranks)

print("✓ Metrics defined")

# %% [markdown]
# ## 3. SDP (Spectral Distribution Perturbation)
#
# Nén phổ singular values tại task boundary:
# $\sigma'_i = \bar{\sigma}^\gamma \cdot \sigma_i^{(1-\gamma)}$

# %%
@torch.no_grad()
def apply_sdp(net, gamma=0.3):
    """Apply Spectral Distribution Perturbation to all weight matrices.
    Supports both Linear and Conv2d layers (4D weights reshaped to 2D).
    Returns list of condition numbers before SDP for monitoring."""
    cond_numbers = []
    for module in net.modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        W = module.weight.data
        original_shape = W.shape
        if W.dim() > 2:
            W = W.reshape(W.shape[0], -1)
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
        module.weight.data.copy_(W_new.reshape(original_shape))
    return cond_numbers

print("✓ SDP defined")

# %% [markdown]
# ## 4. Load CIFAR-100

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

data_path = (lambda p: (os.makedirs(p, exist_ok=True), p)[1])("/kaggle/working/sdp_ablation/data")

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
# ## 5. EMA Wrapper

# %%
class EMAWrapper:
    """Exponential Moving Average of model weights. Used ONLY for evaluation."""

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
# ## 6. Configs
#
# Three first-order methods to ablate spectral effects:
# 1. **Adam** — baseline (no spectral intervention)
# 2. **Adam + SDP** — spectral fix at task boundaries
# 3. **Adam + Soft Rescale** — simple weight scaling (non-spectral control)
#
# All use the same lr, weight_decay, batch_size, and EMA for fair comparison.

# %%
NUM_CLASSES = 100
SEED = 42
CKPT_EVERY = 50

_SHARED = dict(
    num_epochs=4000, batch_size=90, class_increase_frequency=200,
    use_early_stopping=True,
    use_ema=True, ema_decay=0.999,
    # LR schedule: per-task multi-step decay (same as SASSHA experiment)
    lr_milestones=[60, 120, 160], lr_gamma=0.2,
)

CONFIGS = {
    # ─── Control: Adam baseline (no spectral intervention) ───
    'adam_baseline': {
        **_SHARED,
        'optimizer': 'adam',
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'weight_decay': 5e-4,
        'eps': 1e-8,
        'use_sdp': False,
        'use_soft_rescale': False,
    },

    # ─── Treatment: Adam + SDP (spectral fix) ───
    'adam_sdp': {
        **_SHARED,
        'optimizer': 'adam',
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'weight_decay': 5e-4,
        'eps': 1e-8,
        'use_sdp': True,
        'sdp_gamma': 0.3,
        'use_soft_rescale': False,
    },

    # ─── Control: Adam + Soft Rescale (non-spectral weight intervention) ───
    'adam_soft_rescale': {
        **_SHARED,
        'optimizer': 'adam',
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'weight_decay': 5e-4,
        'eps': 1e-8,
        'use_sdp': False,
        'use_soft_rescale': True,
        'soft_rescale_factor': 0.9,
    },
}

METHODS_TO_RUN = [
    'adam_baseline',
    'adam_sdp',
    'adam_soft_rescale',
]

print(f"✓ Configs: {METHODS_TO_RUN}")

# %% [markdown]
# ## 7. Build Optimizer

# %%
def build_optimizer(config, model):
    return torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=config.get('betas', (0.9, 0.999)),
        weight_decay=config.get('weight_decay', 5e-4),
        eps=config.get('eps', 1e-8),
    )

print("✓ build_optimizer defined")

# %% [markdown]
# ## 8. Training Loop

# %%
def _num_classes_at_epoch(epoch, freq, initial=5, step=5, max_cls=100):
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
    print(f"  SDP={config.get('use_sdp', False)}  "
          f"SoftRescale={config.get('use_soft_rescale', False)}  "
          f"EMA={config['use_ema']}")
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

    ema = (EMAWrapper(net, config.get('ema_decay', 0.999))
           if config['use_ema'] else None)

    loss_fn = nn.CrossEntropyLoss()

    metrics = {k: [] for k in [
        'train_loss', 'train_acc', 'val_acc', 'test_acc',
        'dormant_after', 'dormant_before', 'stable_rank',
        'avg_weight_mag', 'epoch_time', 'overfit_gap',
        'weight_cond_number', 'weight_sv_ratio', 'weight_stable_rank',
        'sdp_cond_numbers',
    ]}

    # ─── Resume from checkpoint ───
    start_epoch = 0
    best_val_acc = 0.0
    best_model_state = None
    ckpt_file = _ckpt_path(method_name)
    if os.path.isfile(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ema is not None and 'ema_shadow' in ckpt:
            ema._shadow = ckpt['ema_shadow']
        metrics = ckpt['metrics']
        start_epoch = ckpt['epoch'] + 1
        if 'best_val_acc' in ckpt:
            best_val_acc = ckpt['best_val_acc']
        if 'best_model_state' in ckpt:
            best_model_state = ckpt['best_model_state']
        _saved_num_classes = ckpt.get('current_num_classes', None)
        print(f"  ✓ Resumed from epoch {ckpt['epoch']}  ({ckpt_file})")
        del ckpt
        torch.cuda.empty_cache()
    else:
        _saved_num_classes = None
        print(f"  (no checkpoint, training from scratch)")

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

    for epoch in range(start_epoch, config['num_epochs']):
        t0 = time.time()
        epoch_in_task = epoch % freq

        # ── Task boundary ──
        if epoch > 0 and epoch % freq == 0 and current_num_classes < NUM_CLASSES:
            # Early stopping: load best val model
            if config['use_early_stopping'] and best_model_state is not None:
                net.load_state_dict(best_model_state)
                print(f"  → Early stop: loaded best val (acc={best_val_acc:.4f})")
            best_val_acc = 0.0
            best_model_state = None

            # ── SDP at task boundary ──
            if config.get('use_sdp', False):
                sdp_gamma = config.get('sdp_gamma', 0.3)
                cond_nums = apply_sdp(net, gamma=sdp_gamma)
                avg_cond = np.mean(cond_nums) if cond_nums else 0.0
                print(f"  → SDP applied (γ={sdp_gamma}): "
                      f"avg_cond={avg_cond:.1f}, layers={len(cond_nums)}")

            # ── Soft Rescale at task boundary ──
            if config.get('use_soft_rescale', False):
                rescale = config.get('soft_rescale_factor', 0.9)
                with torch.no_grad():
                    rescaled_count = 0
                    for name, p in net.named_parameters():
                        if 'bn' not in name and 'norm' not in name and 'downsample.1' not in name:
                            p.data.mul_(rescale)
                            rescaled_count += 1
                print(f"  → Soft rescale: ×{rescale} applied to {rescaled_count} tensors")

            # Reset EMA
            if ema is not None:
                ema.reset(net)

            # Reset optimizer state at task boundary
            # (prevents stale momentum from old task distribution)
            optimizer.state.clear()

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

        # ── Per-task multi-step LR decay ──
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

            # Standard first-order training — single forward+backward pass
            optimizer.zero_grad()
            pred = net(img)[:, all_classes[:current_num_classes]]
            loss = loss_fn(pred, tgt)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            rl += loss.item()

            if ema is not None:
                ema.update(net)

            with torch.no_grad():
                acc = (pred.argmax(1) == tgt).float().mean()
            ra += acc.item()
            nb += 1

        avg_loss = rl / nb if nb > 0 else float('nan')
        metrics['train_loss'].append(avg_loss)
        metrics['train_acc'].append(ra / nb if nb > 0 else 0)

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

        # ── Restore training weights ──
        if ema is not None:
            ema.restore(net)

        # ── Dormant measurement ──
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

        # ── Weight spectral statistics (key ablation metrics) ──
        w_cond, w_sv_ratio, w_sr = compute_weight_spectral_stats(net)
        metrics['weight_cond_number'].append(w_cond)
        metrics['weight_sv_ratio'].append(w_sv_ratio)
        metrics['weight_stable_rank'].append(w_sr)

        net.train()
        et = time.time() - t0
        metrics['epoch_time'].append(et)

        if epoch % 50 == 0 or epoch == config['num_epochs'] - 1:
            print(f"  [{method_name}] E{epoch:4d} | Loss={avg_loss:.4f} "
                  f"TestAcc={metrics['test_acc'][-1]:.4f} "
                  f"Gap={gap:.4f} Dorm={da:.4f}/{db:.4f} "
                  f"AvgW={wm:.4f} Cond={w_cond:.1f} WeightSR={w_sr:.1f} "
                  f"{et:.1f}s")

        # ── Checkpoint ──
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
            if ema is not None:
                ckpt_data['ema_shadow'] = ema._shadow
            torch.save(ckpt_data, _ckpt_path(method_name))

    return metrics

print("✓ Training loop defined")

# %% [markdown]
# ## 9. Run All Methods

# %%
all_results = {}
for method in METHODS_TO_RUN:
    all_results[method] = run_method(method, CONFIGS[method])
    with open(os.path.join(results_dir, f"ablation_{method}.pkl"), 'wb') as f:
        pickle.dump(all_results[method], f)
    print(f"  ✓ {method} saved.")

with open(os.path.join(results_dir, "ablation_all_results.pkl"), 'wb') as f:
    pickle.dump(all_results, f)
print(f"\n✓ All results saved.")

# %% [markdown]
# ## 10. Visualization
#
# Key plots for ablation analysis:
# - Row 1: Performance (TestAcc, OvfGap, TrainLoss)
# - Row 2: Plasticity indicators (Dormant, StableRank, WeightMag)
# - Row 3: Spectral health (CondNumber, SV Ratio, Weight StableRank)

# %%
METHOD_STYLES = {
    'adam_baseline':      {'color': '#d62728', 'ls': '-',  'lw': 2.0,
                           'label': 'Adam (baseline)'},
    'adam_sdp':           {'color': '#2ca02c', 'ls': '-',  'lw': 2.5,
                           'label': 'Adam + SDP'},
    'adam_soft_rescale':  {'color': '#1f77b4', 'ls': '--', 'lw': 2.0,
                           'label': 'Adam + Soft Rescale'},
}

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Ablation: Is Loss of Plasticity = Spectral Pathology?\n'
             'First-Order (Adam) ± SDP on Incremental CIFAR-100 / ResNet-18',
             fontsize=14, fontweight='bold', y=0.99)

plot_info = [
    # Row 1: Performance
    ('test_acc',           'Test Accuracy',               axes[0, 0], 'Accuracy'),
    ('overfit_gap',        'Train-Test Gap',              axes[0, 1], 'Gap'),
    ('train_loss',         'Train Loss',                  axes[0, 2], 'Loss'),
    # Row 2: Plasticity
    ('dormant_after',      'Dormant Units (Next Task)',   axes[1, 0], 'Proportion'),
    ('stable_rank',        'Activation Stable Rank',      axes[1, 1], 'Stable Rank'),
    ('avg_weight_mag',     'Avg Weight Magnitude',        axes[1, 2], 'Magnitude'),
    # Row 3: Spectral health (key ablation metrics)
    ('weight_cond_number', 'Weight Condition Number',     axes[2, 0], 'Condition #'),
    ('weight_sv_ratio',    'Top-1 SV Ratio (σ₁/Σσ)',     axes[2, 1], 'Ratio'),
    ('weight_stable_rank', 'Weight Stable Rank',          axes[2, 2], 'Stable Rank'),
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
    ax.legend(fontsize=7, loc='best', framealpha=0.7, edgecolor='none')
    for tb in range(200, 4001, 200):
        ax.axvline(x=tb, color='gray', ls=':', alpha=0.25, lw=0.7)
    # Log scale for condition number
    if key == 'weight_cond_number':
        ax.set_yscale('log')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(results_dir, "sdp_ablation_comparison.png"),
            dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Saved to {results_dir}/sdp_ablation_comparison.png")

# %% [markdown]
# ## 11. Summary Table & Conclusion

# %%
print(f"\n{'='*130}")
print(f"{'Method':<25} {'TestAcc↑':>9} {'OvfGap↓':>9} {'Dormant↓':>9} "
      f"{'ActSR↑':>9} {'AvgW':>9} {'Cond#↓':>10} {'SV_Ratio↓':>10} {'WtSR↑':>9}")
print(f"{'='*130}")

for method, data in all_results.items():
    s = METHOD_STYLES.get(method, {})
    n = min(50, len(data['test_acc']))
    ta = np.mean(data['test_acc'][-n:])
    og = np.mean(data['overfit_gap'][-n:])
    da = np.nanmean(data['dormant_after'][-n:])
    sr = np.nanmean(data['stable_rank'][-n:])
    wm = np.mean(data['avg_weight_mag'][-n:])
    wc = np.mean(data['weight_cond_number'][-n:])
    wr = np.mean(data['weight_sv_ratio'][-n:])
    ws = np.mean(data['weight_stable_rank'][-n:])
    lbl = s.get('label', method)
    print(f"{lbl:<25} {ta:>9.4f} {og:>9.4f} {da:>9.4f} "
          f"{sr:>9.1f} {wm:>9.4f} {wc:>10.1f} {wr:>10.4f} {ws:>9.1f}")

print(f"{'='*130}")

# ── Interpretation guide ──
print("""
Interpretation:
  If Adam+SDP >> Adam baseline ≈ Adam+SoftRescale:
    → LoP IS spectral pathology. SDP alone fixes it regardless of optimizer.

  If Adam+SDP ≈ Adam baseline:
    → LoP requires second-order correction (SASSHA). Spectral fix alone insufficient.

  If Adam+SDP > Adam baseline but Adam+SoftRescale also helps:
    → LoP is partially spectral, partially weight-norm related.

Key metrics to watch:
  - weight_cond_number: SDP should dramatically reduce this
  - dormant_after: Does spectral health → fewer dormant neurons?
  - test_acc at later tasks: Core measure of plasticity
""")
