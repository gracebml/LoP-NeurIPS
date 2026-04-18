# %% [markdown]
# # First-Order Optimizers + SDP on Continual ImageNet-32
# #
# **Research Question**: Liệu SDP có hiệu quả với cả first-order methods không?
# #
# **Motivation**: Paper outline (NeurIPS 2026) yêu cầu first-order baselines
# để trả lời reviewer: "SDP alone đã đủ, tại sao cần second-order?"
# #
# **Optimizers tested**:
# 1. **SGD + Momentum** — Standard first-order baseline
# 2. **Adam** (Kingma & Ba, 2015) — Adaptive first-order
# 3. **AdamW** (Loshchilov & Hutter, 2019) — Adam with decoupled weight decay
# #
# Each optimizer runs **with and without SDP** for ablation.
# #
# **Benchmark**: Task-incremental binary classification, 2 classes/task,
# 1000 classes ImageNet-32 (600 train + 100 test per class).
# Network: ConvNet (conv1→conv2→conv3→fc1→fc2), 2000 tasks × 200 epochs/task.

# %% [markdown]
# ## 1. Imports & Setup

# %%
import os, sys, time, pickle, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

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
# ## 2. Metrics & SDP

# %%
NUM_CONV_LAYERS = 3
NUM_FC_LAYERS   = 2
NUM_LAYERS      = NUM_CONV_LAYERS + NUM_FC_LAYERS   # 5 total
LAYER_SIZES     = [32, 64, 512, 128, 128]           # neurons per layer (ConvNet)
TOTAL_NEURONS   = sum(LAYER_SIZES)                  # 864

@torch.no_grad()
def compute_avg_weight_magnitude(net):
    n, s = 0, 0.0
    for p in net.parameters():
        n += p.numel()
        s += torch.sum(torch.abs(p)).item()
    return s / n if n > 0 else 0.0

@torch.no_grad()
def compute_dormant_neurons(net, x_data, mini_batch_size=100, threshold=0.01):
    batch_x = x_data[:min(mini_batch_size * 5, len(x_data))]
    _, activations = net.predict(x=batch_x)
    per_layer_frac, all_alive = [], []
    total_n, total_d = 0, 0
    last_act = None
    for i, act in enumerate(activations):
        if act.ndim == 4:
            alive_score = (act != 0).float().mean(dim=(0, 2, 3))
            n_units = act.shape[1]
        else:
            alive_score = (act != 0).float().mean(dim=0)
            n_units = act.shape[1]
        dormant = (alive_score < threshold).sum().item()
        per_layer_frac.append(dormant / n_units if n_units > 0 else 0.0)
        all_alive.append(alive_score.cpu().numpy())
        total_d += dormant
        total_n += n_units
        if i == len(activations) - 1:
            last_act = act.cpu().numpy()
    agg_frac = total_d / total_n if total_n > 0 else 0.0
    return agg_frac, per_layer_frac, np.concatenate(all_alive), last_act

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
    except:
        return 0

def apply_sdp(net, gamma):
    """Spectral Diversity Preservation (SDP) at task boundary.
    σ'_i = σ̄^γ · σ_i^(1-γ)

    Applies to all layers except the output head (which is reset each task anyway).
    Conv weights are reshaped to 2D (out_channels, in*k*k) for SVD.
    """
    cond_numbers = []
    layers = [l for l in net.layers if hasattr(l, 'weight')]
    with torch.no_grad():
        for i, layer in enumerate(layers):
            is_output = (i == len(layers) - 1)
            if is_output:
                continue  # output head is reset each task — skip SDP
            W = layer.weight.data
            orig_shape = W.shape
            W2d = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
            try:
                U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
            except Exception:
                continue
            if S.numel() == 0 or S[0] < 1e-12:
                continue
            cond_numbers.append((S[0] / S[-1].clamp(min=1e-12)).item())
            s_mean = S.mean()
            S_new = (s_mean ** gamma) * (S ** (1.0 - gamma))
            W_new = U @ torch.diag(S_new) @ Vh
            layer.weight.data.copy_(W_new.reshape(orig_shape))
    return cond_numbers

print("✓ Metrics & SDP defined")

# %% [markdown]
# ## 3. ImageNet-32 Data Loading

# %%
TRAIN_IMAGES_PER_CLASS = 600
TEST_IMAGES_PER_CLASS  = 100
TOTAL_CLASSES          = 1000

DATA_DIR = '/kaggle/input/datasets/nguyenlamphuquy/imagenet/classes'
print(f"✓ ImageNet-32 data dir: {DATA_DIR}")

_class_order_file = os.path.join(_LOP_IMAGENET_DIR, 'class_order')
if os.path.isfile(_class_order_file):
    with open(_class_order_file, 'rb') as f:
        _ALL_CLASS_ORDERS = pickle.load(f)
    print(f"  ✓ Loaded class_order ({len(_ALL_CLASS_ORDERS)} runs)")
else:
    print("  ⚠ class_order not found — generating random order")
    _rng = np.random.RandomState(42)
    _ALL_CLASS_ORDERS = [_rng.permutation(TOTAL_CLASSES) for _ in range(30)]

def load_imagenet(classes):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, cls in enumerate(classes):
        data = np.load(os.path.join(DATA_DIR, str(cls) + '.npy'))
        x_train.append(data[:TRAIN_IMAGES_PER_CLASS])
        x_test.append(data[TRAIN_IMAGES_PER_CLASS:])
        y_train.append(np.full(TRAIN_IMAGES_PER_CLASS, idx))
        y_test.append(np.full(TEST_IMAGES_PER_CLASS, idx))
    return (
        torch.tensor(np.concatenate(x_train)),
        torch.from_numpy(np.concatenate(y_train)),
        torch.tensor(np.concatenate(x_test)),
        torch.from_numpy(np.concatenate(y_test)),
    )

print(f"✓ Data loading ready ({TOTAL_CLASSES} classes)")

# %% [markdown]
# ## 4. EMA Wrapper

# %%
class EMAWrapper:
    def __init__(self, model, decay=0.999):
        self.decay   = decay
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
# ## 5. Configs

# %%
SEED               = 42
SAVE_EVERY_N_TASKS = 50
TIME_LIMIT_SECONDS = 11.5 * 3600

NUM_TASKS    = 2000
NUM_CLASSES  = 2      # binary classification per task
NUM_EPOCHS   = 200    # showings per task
MINI_BATCH   = 100

EXAMPLES_PER_EPOCH = TRAIN_IMAGES_PER_CLASS * NUM_CLASSES   # 1200

SDP_GAMMA = 0.3

CONFIGS = {
    # ─── SGD + Momentum ───
    # 1. SGD + SDP
    'sgd_sdp': dict(
        optimizer='sgd',
        lr=0.01, momentum=0.9, weight_decay=5e-4,
        grad_clip=1.0,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 2. SGD (no SDP)
    'sgd_nosdp': dict(
        optimizer='sgd',
        lr=0.01, momentum=0.9, weight_decay=5e-4,
        grad_clip=1.0,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=0.0,
    ),
    # ─── Adam ───
    # 3. Adam + SDP
    'adam_sdp': dict(
        optimizer='adam',
        lr=0.001, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8,
        grad_clip=1.0,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 4. Adam (no SDP)
    'adam_nosdp': dict(
        optimizer='adam',
        lr=0.001, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8,
        grad_clip=1.0,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=0.0,
    ),
    # ─── AdamW ───
    # 5. AdamW + SDP
    'adamw_sdp': dict(
        optimizer='adamw',
        lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8,
        grad_clip=1.0,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=SDP_GAMMA,
    ),
    # 6. AdamW (no SDP)
    'adamw_nosdp': dict(
        optimizer='adamw',
        lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8,
        grad_clip=1.0,
        use_ema=True, ema_decay=0.999, label_smoothing=True,
        sdp_gamma=0.0,
    ),
}

METHODS_TO_RUN = [
    'sgd_sdp',   'sgd_nosdp',
    'adam_sdp',   'adam_nosdp',
    'adamw_sdp',  'adamw_nosdp',
]

RESULTS_DIR = os.path.join('permuted_imagenet_results', 'firstorder_sdp')
CKPT_DIR    = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"✓ Config: {NUM_TASKS} tasks × {NUM_EPOCHS} epochs/task, "
      f"{NUM_CLASSES} classes/task, batch={MINI_BATCH}")
print(f"  Methods: {METHODS_TO_RUN}")

# %% [markdown]
# ## 6. Build Optimizer

# %%
def build_optimizer(config, model):
    opt_type = config['optimizer']

    if opt_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(), lr=config['lr'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0))

    elif opt_type == 'adam':
        return torch.optim.Adam(
            model.parameters(), lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.0),
            eps=config.get('eps', 1e-8))

    elif opt_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(), lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.01),
            eps=config.get('eps', 1e-8))

    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

print("✓ build_optimizer defined")

# %% [markdown]
# ## 7. Unified Training Loop

# %%
def _ckpt_path(method_name):
    return os.path.join(CKPT_DIR, f"ckpt_{method_name}.pt")

def run_method(method_name, config, run_idx=0):
    """Unified training loop for first-order optimizers ± SDP on ImageNet-32."""
    opt_type  = config['optimizer']
    sdp_gamma = config.get('sdp_gamma', 0.0)
    grad_clip = config.get('grad_clip', 0.0)

    print(f"\n{'='*70}")
    print(f"  {method_name} (optimizer={opt_type}) — Continual ImageNet-32 ({NUM_TASKS} tasks)")
    print(f"  SDP: {'enabled γ=' + str(sdp_gamma) if sdp_gamma > 0 else 'disabled'}")
    print(f"  Grad clip: {grad_clip}")
    print(f"{'='*70}")

    wall_clock_start = time.time()
    torch.manual_seed(SEED); torch.cuda.manual_seed(SEED); np.random.seed(SEED)

    # ── Class order ──
    class_order = _ALL_CLASS_ORDERS[run_idx % len(_ALL_CLASS_ORDERS)]
    n_reps      = int(NUM_CLASSES * NUM_TASKS / TOTAL_CLASSES) + 1
    class_order = np.concatenate([class_order] * n_reps)

    # ── Network ──
    net       = ConvNet(num_classes=NUM_CLASSES)
    optimizer = build_optimizer(config, net)

    ema = EMAWrapper(net, config.get('ema_decay', 0.999)) if config.get('use_ema', False) else None
    ls  = 0.1 if config.get('label_smoothing', False) else 0.0
    loss_fn = lambda logits, target: F.cross_entropy(logits, target, label_smoothing=ls)

    # ── Metric tensors ──
    train_accuracies    = torch.zeros((NUM_TASKS, NUM_EPOCHS), dtype=torch.float)
    test_accuracies     = torch.zeros((NUM_TASKS, NUM_EPOCHS), dtype=torch.float)
    all_weight_mag      = torch.zeros((NUM_TASKS, NUM_EPOCHS), dtype=torch.float)
    all_dormant_frac    = torch.zeros((NUM_TASKS, NUM_EPOCHS), dtype=torch.float)
    all_dormant_layers  = torch.zeros((NUM_TASKS, NUM_EPOCHS, NUM_LAYERS), dtype=torch.float)
    all_stable_rank     = torch.zeros((NUM_TASKS, NUM_EPOCHS), dtype=torch.float)
    all_sdp_cond        = torch.zeros(NUM_TASKS, dtype=torch.float)

    # ── Checkpoint resume ──
    ckpt_file  = _ckpt_path(method_name)
    start_task = 0
    if os.path.isfile(ckpt_file):
        print(f"  Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        train_accuracies   = ckpt.get('train_accuracies',   train_accuracies)
        test_accuracies    = ckpt.get('test_accuracies',    test_accuracies)
        all_weight_mag     = ckpt.get('all_weight_mag',     all_weight_mag)
        all_dormant_frac   = ckpt.get('all_dormant_frac',   all_dormant_frac)
        all_dormant_layers = ckpt.get('all_dormant_layers', all_dormant_layers)
        all_stable_rank    = ckpt.get('all_stable_rank',    all_stable_rank)
        all_sdp_cond       = ckpt.get('all_sdp_cond',       all_sdp_cond)
        start_task         = ckpt['task'] + 1
        if ema is not None and 'ema_shadow' in ckpt:
            ema._shadow = ckpt['ema_shadow']
        if 'np_rng' in ckpt: np.random.set_state(ckpt['np_rng'])
        print(f"  ✓ Resumed from task {start_task}")
        del ckpt; torch.cuda.empty_cache()
    else:
        print("  (no checkpoint — training from scratch)")

    def save_checkpoint(task_idx, reason="periodic"):
        ckpt_data = {
            'task': task_idx, 'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_accuracies': train_accuracies, 'test_accuracies': test_accuracies,
            'all_weight_mag': all_weight_mag, 'all_dormant_frac': all_dormant_frac,
            'all_dormant_layers': all_dormant_layers, 'all_stable_rank': all_stable_rank,
            'all_sdp_cond': all_sdp_cond, 'params': config,
            'np_rng': np.random.get_state(),
        }
        if ema is not None: ckpt_data['ema_shadow'] = ema._shadow
        torch.save(ckpt_data, ckpt_file)
        elapsed = time.time() - wall_clock_start
        print(f"  Checkpoint saved at task {task_idx} ({reason}) [{elapsed/3600:.1f}h elapsed]")

    x_train = x_test = y_train = y_test = None
    time_limit_hit = False

    # ════════════════════════════════════════════════════════════════
    #  Task loop
    # ════════════════════════════════════════════════════════════════
    for task_idx in range(start_task, NUM_TASKS):
        task_start = time.time()
        elapsed    = time.time() - wall_clock_start
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"\n  Time limit ({elapsed/3600:.1f}h). Saving.")
            save_checkpoint(task_idx - 1, reason="time_limit")
            time_limit_hit = True; break

        # ── 1. Load task data ──
        del x_train, x_test, y_train, y_test
        task_classes  = class_order[task_idx * NUM_CLASSES:(task_idx + 1) * NUM_CLASSES]
        x_train, y_train, x_test, y_test = load_imagenet(task_classes)
        x_train = x_train.float()
        x_test  = x_test.float()
        if device.type == 'cuda':
            x_train, x_test = x_train.to(device), x_test.to(device)
            y_train, y_test = y_train.to(device), y_test.to(device)

        # ── 2. Reset output head (task-incremental binary setting) ──
        net.layers[-1].weight.data.zero_()
        net.layers[-1].bias.data.zero_()

        # ── 3. SDP at task boundary ──
        if sdp_gamma > 0 and task_idx > 0:
            cond_nums = apply_sdp(net, sdp_gamma)
            avg_cond  = sum(cond_nums) / max(len(cond_nums), 1)
            all_sdp_cond[task_idx] = avg_cond
            if task_idx % 50 == 0:
                print(f"    SDP applied: avg condition number = {avg_cond:.1f}")

        # ── 4. Reset EMA at task boundary ──
        if ema is not None: ema.reset(net)

        # ── 5. Train epochs ──
        for epoch_idx in range(NUM_EPOCHS):
            net.train()
            perm      = np.random.permutation(EXAMPLES_PER_EPOCH)
            x_shuf    = x_train[perm]
            y_shuf    = y_train[perm]
            batch_accs = []

            for start_idx in range(0, EXAMPLES_PER_EPOCH, MINI_BATCH):
                batch_x = x_shuf[start_idx:start_idx + MINI_BATCH]
                batch_y = y_shuf[start_idx:start_idx + MINI_BATCH]

                # ── Standard first-order training step ──
                optimizer.zero_grad()
                logits, _ = net.predict(x=batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()

                # ── Gradient clipping ──
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

                optimizer.step()

                if ema is not None: ema.update(net)

                with torch.no_grad():
                    batch_accs.append(
                        accuracy(F.softmax(logits.detach(), dim=1), batch_y).item())

            # ── Per-epoch eval ──
            net.eval()
            with torch.no_grad():
                train_accuracies[task_idx, epoch_idx] = float(np.mean(batch_accs))

                if ema is not None: ema.apply(net)
                test_accs = []
                for si in range(0, x_test.shape[0], MINI_BATCH):
                    tb_x = x_test[si:si + MINI_BATCH]
                    tb_y = y_test[si:si + MINI_BATCH]
                    to, _ = net.predict(x=tb_x)
                    test_accs.append(accuracy(F.softmax(to, dim=1), tb_y).item())
                test_accuracies[task_idx, epoch_idx] = float(np.mean(test_accs))
                if ema is not None: ema.restore(net)

                wm = compute_avg_weight_magnitude(net)
                all_weight_mag[task_idx, epoch_idx] = wm
                agg_frac, layer_fracs, _, last_act = compute_dormant_neurons(
                    net, x_test, mini_batch_size=MINI_BATCH)
                all_dormant_frac[task_idx, epoch_idx]     = agg_frac
                all_dormant_layers[task_idx, epoch_idx]   = torch.tensor(layer_fracs)
                sr = compute_stable_rank_from_activations(last_act)
                all_stable_rank[task_idx, epoch_idx]      = sr

            if epoch_idx % 50 == 0 or epoch_idx == NUM_EPOCHS - 1:
                layer_str = ' '.join([f'{f:.2f}' for f in layer_fracs])
                sdp_str   = f" CondN={all_sdp_cond[task_idx]:.1f}" if sdp_gamma > 0 and task_idx > 0 else ""
                print(f"    Task {task_idx:4d} Ep {epoch_idx:3d}/{NUM_EPOCHS} | "
                      f"TrainAcc={train_accuracies[task_idx, epoch_idx]:.4f}  "
                      f"TestAcc={test_accuracies[task_idx, epoch_idx]:.4f}  "
                      f"Dormant={agg_frac:.3f} [{layer_str}]  "
                      f"SR={sr:.0f}  AvgW={wm:.4f}{sdp_str}")

        task_time = time.time() - task_start
        if task_idx % 50 == 0 or task_idx == NUM_TASKS - 1:
            print(f"  [{method_name}] Task {task_idx:4d}/{NUM_TASKS} | "
                  f"TrainAcc={train_accuracies[task_idx, -1]:.4f}  "
                  f"TestAcc={test_accuracies[task_idx, -1]:.4f}  "
                  f"Dormant={all_dormant_frac[task_idx, -1]:.3f}  "
                  f"SR={all_stable_rank[task_idx, -1]:.0f}  "
                  f"AvgW={all_weight_mag[task_idx, -1]:.4f}  "
                  f"{task_time:.1f}s")

        if (task_idx + 1) % SAVE_EVERY_N_TASKS == 0 or task_idx == NUM_TASKS - 1:
            save_checkpoint(task_idx,
                            reason="periodic" if task_idx < NUM_TASKS - 1 else "completed")

    # ── Save final results ──
    result_file = os.path.join(RESULTS_DIR, f'{method_name}_results.pt')
    torch.save({
        'train_accuracies':  train_accuracies.cpu(),
        'test_accuracies':   test_accuracies.cpu(),
        'all_weight_mag':    all_weight_mag.cpu(),
        'all_dormant_frac':  all_dormant_frac.cpu(),
        'all_dormant_layers': all_dormant_layers.cpu(),
        'all_stable_rank':   all_stable_rank.cpu(),
        'all_sdp_cond':      all_sdp_cond.cpu(),
    }, result_file)
    print(f"  ✓ Results saved to {result_file}")

    return (train_accuracies, test_accuracies, all_weight_mag,
            all_dormant_frac, all_dormant_layers, all_stable_rank, all_sdp_cond)

print("✓ Unified training loop defined")

# %% [markdown]
# ## 8. Run All Experiments

# %%
all_results = {}
for method in METHODS_TO_RUN:
    cfg    = CONFIGS[method]
    result = run_method(method, cfg)
    all_results[method] = {
        'train_acc':  result[0],
        'test_acc':   result[1],
        'wmag':       result[2],
        'dormant':    result[3],
        'dorm_layer': result[4],
        'sr':         result[5],
        'cond':       result[6],
    }

# %% [markdown]
# ## 9. Results Plots

# %%
EPOCH_WINDOW = 10   # smooth over task windows

METHOD_STYLES = {
    'sgd_sdp':     {'color': '#E91E63', 'ls': '-',  'label': 'SGD+SDP'},
    'sgd_nosdp':   {'color': '#E91E63', 'ls': '--', 'label': 'SGD'},
    'adam_sdp':     {'color': '#2196F3', 'ls': '-',  'label': 'Adam+SDP'},
    'adam_nosdp':   {'color': '#2196F3', 'ls': '--', 'label': 'Adam'},
    'adamw_sdp':    {'color': '#4CAF50', 'ls': '-',  'label': 'AdamW+SDP'},
    'adamw_nosdp':  {'color': '#4CAF50', 'ls': '--', 'label': 'AdamW'},
}

def _style(m): return METHOD_STYLES.get(m, {'color': 'gray', 'ls': '-', 'label': m})
def _clean(ax):
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def smooth_tasks(arr, w=EPOCH_WINDOW):
    # arr shape: (num_tasks,) — average over window
    n = len(arr) // w
    return np.array([arr[i*w:(i+1)*w].mean() for i in range(n)])

fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('First-Order Optimizers ± SDP — Continual ImageNet-32\n'
             f'ConvNet, {NUM_TASKS} tasks × {NUM_EPOCHS} epochs, '
             f'{NUM_CLASSES} classes/task, batch={MINI_BATCH}',
             fontsize=14, fontweight='bold')

# ── (0,0) Final-epoch Test Accuracy per task ──
ax = axes[0, 0]
for m, d in all_results.items():
    s  = _style(m)
    ta = d['test_acc'][:, -1].numpy()   # final epoch per task
    sm = smooth_tasks(ta) * 100
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Test Acc (%)')
ax.set_title('Test Accuracy (final epoch, smoothed)')
ax.legend(fontsize=9, ncol=2); _clean(ax)

# ── (0,1) Final-epoch Train Accuracy ──
ax = axes[0, 1]
for m, d in all_results.items():
    s  = _style(m)
    ta = d['train_acc'][:, -1].numpy()
    sm = smooth_tasks(ta) * 100
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Train Acc (%)')
ax.set_title('Train Accuracy (final epoch, smoothed)')
ax.legend(fontsize=9, ncol=2); _clean(ax)

# ── (1,0) Stable Rank ──
ax = axes[1, 0]
for m, d in all_results.items():
    s  = _style(m)
    sr = d['sr'][:, -1].numpy()
    sm = smooth_tasks(sr)
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Stable Rank')
ax.set_title('Stable Rank (last activation, final epoch)')
ax.legend(fontsize=9, ncol=2); _clean(ax)

# ── (1,1) Dormant Neuron Fraction ──
ax = axes[1, 1]
for m, d in all_results.items():
    s    = _style(m)
    dorm = d['dormant'][:, -1].numpy() * 100
    sm   = smooth_tasks(dorm)
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Dormant (%)')
ax.set_title('Dormant Neuron Fraction (final epoch)')
ax.legend(fontsize=9, ncol=2); _clean(ax)

# ── (2,0) Avg Weight Magnitude ──
ax = axes[2, 0]
for m, d in all_results.items():
    s  = _style(m)
    wm = d['wmag'][:, -1].numpy()
    sm = smooth_tasks(wm)
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('AvgW')
ax.set_title('Avg Weight Magnitude')
ax.legend(fontsize=9, ncol=2); _clean(ax)

# ── (2,1) SDP Condition Number ──
ax = axes[2, 1]
for m, d in all_results.items():
    if CONFIGS[m].get('sdp_gamma', 0) == 0: continue
    s    = _style(m)
    cond = d['cond'][1:].numpy()   # skip task 0 (no SDP applied)
    sm   = smooth_tasks(cond)
    ax.plot(np.arange(len(sm)) * EPOCH_WINDOW, sm,
            color=s['color'], ls=s['ls'], lw=2.5, label=s['label'])
ax.set_xlabel('Task'); ax.set_ylabel('Condition Number')
ax.set_title('SDP Avg Condition Number (SDP methods only)')
ax.legend(fontsize=9, ncol=2); _clean(ax)

plt.tight_layout()
plot_file = os.path.join(RESULTS_DIR, 'imagenet_firstorder_sdp_comparison.png')
plt.savefig(plot_file, dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Main comparison plot saved to {plot_file}")

# %% [markdown]
# ## 10. SDP Ablation: Δ(metric) = SDP − noSDP

# %%
OPT_PAIRS = [
    ('sgd_sdp',   'sgd_nosdp',   'SGD',   '#E91E63'),
    ('adam_sdp',   'adam_nosdp',   'Adam',  '#2196F3'),
    ('adamw_sdp',  'adamw_nosdp',  'AdamW', '#4CAF50'),
]

fig_ab, axes_ab = plt.subplots(1, 3, figsize=(21, 5))
fig_ab.suptitle('SDP Ablation: Δ = (with SDP) − (without SDP) — First-Order on ImageNet-32',
                fontsize=13, fontweight='bold')

for sdp_key, nosdp_key, label, color in OPT_PAIRS:
    if sdp_key not in all_results or nosdp_key not in all_results:
        continue
    d_sdp = all_results[sdp_key]
    d_no  = all_results[nosdp_key]

    delta_test = d_sdp['test_acc'][:, -1].numpy() - d_no['test_acc'][:, -1].numpy()
    axes_ab[0].plot(smooth_tasks(delta_test) * 100, color=color, lw=2.5, label=label)

    delta_sr = d_sdp['sr'][:, -1].numpy() - d_no['sr'][:, -1].numpy()
    axes_ab[1].plot(smooth_tasks(delta_sr), color=color, lw=2.5, label=label)

    delta_dorm = (d_sdp['dormant'][:, -1].numpy() - d_no['dormant'][:, -1].numpy()) * 100
    axes_ab[2].plot(smooth_tasks(delta_dorm), color=color, lw=2.5, label=label)

for i, (title, ylabel) in enumerate([
    ('Δ Test Accuracy', 'Δ Acc (%)'),
    ('Δ Stable Rank',   'Δ SR'),
    ('Δ Dormant',       'Δ Dormant (%)'),
]):
    axes_ab[i].set_title(title); axes_ab[i].set_xlabel('Task window')
    axes_ab[i].set_ylabel(ylabel)
    axes_ab[i].axhline(0, color='black', ls=':', lw=0.8, alpha=0.5)
    axes_ab[i].legend(fontsize=9); _clean(axes_ab[i])

plt.tight_layout()
plot_abl = os.path.join(RESULTS_DIR, 'imagenet_firstorder_sdp_ablation.png')
plt.savefig(plot_abl, dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ SDP ablation plot saved to {plot_abl}")

# %% [markdown]
# ## 11. Summary Table

# %%
N_FINAL = 100   # average over last N tasks
print(f"\n{'='*95}")
print(f"  First-Order Optimizers ± SDP — ImageNet-32 — Final {N_FINAL}-task average")
print(f"{'='*95}")
header = (f"{'Method':<22} {'TestAcc%':>9} {'TrainAcc%':>10} "
          f"{'StableRank':>11} {'Dormant%':>9} {'AvgW':>8}")
print(header)
print('─' * 95)

for method in METHODS_TO_RUN:
    if method not in all_results: continue
    d  = all_results[method]
    s  = _style(method)
    ta = d['test_acc'][-N_FINAL:, -1].mean().item() * 100
    tr = d['train_acc'][-N_FINAL:, -1].mean().item() * 100
    sr = d['sr'][-N_FINAL:, -1].mean().item()
    do = d['dormant'][-N_FINAL:, -1].mean().item() * 100
    wm = d['wmag'][-N_FINAL:, -1].mean().item()
    print(f"  {s['label']:<20} {ta:>9.2f} {tr:>10.2f} {sr:>11.1f} {do:>9.2f} {wm:>8.4f}")

print('=' * 95)
