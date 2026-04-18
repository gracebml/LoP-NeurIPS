# Loss of Plasticity (LoP) in Deep Continual Learning

This repository contains the refactored and standardized official implementation for research on the **Loss of Plasticity (LoP)** phenomenon in Deep Neural Networks. It provides a unified framework to train, monitor, and evaluate continual learning agents across various domains, tracking how the ability to learn new representations decays over time.

## Key Features & Standardization

The codebase has been entirely unified into a single `lop/` package to ensure consistent tracking of plasticity metrics across all experiments:

- **Centralized Metric Dashboard (`lop.metrics.dashboard`)**: A unified metrics engine computing 11 core plasticity diagnostics (Effective Rank, Stable Rank, Approximate Rank, Dormant Proportion, Layer-wise Dormancy, Weight Magnitude, NTK Churn, NTK Eigenspectrum Cond/Rank, plus Accuracy and Loss) at every task boundary.
- **Unified Data Loaders (`lop.data`)**: Standardized data ingestion supporting local environments and direct Kaggle notebook execution.
- **Unified Optimizers (`lop.optimizers`)**: Advanced custom optimizers (AdaHessian, SophiaH, Sassha) modularized for plug-and-play use.
- **Centralized Algorithms (`lop.algos`)**: Common adaptation and optimization algorithms, including standard Backprop, Continual Backprop (CBP), Generate-and-Test (GnT), and Spectral Decoupling (SDP).

## 📂 Repository Structure

```text
.
├── data/                       # Downloaded datasets
├── download_data.py            # Script to fetch CIFAR, MNIST, TinyImageNet, etc.
├── main.py                     # Unified entry point for all experiments
└── lop/                        # Core Package
    ├── algos/                  # Learning algorithms (BP, Continual BP, GnT, SDP)
    ├── data/                   # Unified DataLoaders
    ├── envs/                   # RL environments
    ├── imagenet/               # TinyImageNet experiments
    ├── incremental_cifar/      # CIFAR-100 incremental learning experiments
    ├── metrics/                # 📊 Centralized Plasticity Metrics (Dashboard, Rank, Dormant, NTK)
    ├── nets/                   # Network Architectures (DeepFFNN, ConvNet, ResNets)
    ├── optimizers/             # Standardized Custom Optimizers
    ├── permuted_mnist/         # Permuted MNIST experiments
    ├── rl/                     # Reinforcement Learning experiments
    ├── utils/                  # Miscellaneous utilities
    └── viz/                    # Data visualization scripts
```

## Usage

### 1. Download Datasets
Before running experiments, run the standardized data download script to generate necessary dataset pickles:

```bash
python download_data.py
```

### 2. Running Experiments
All experiments are executed via the unified `main.py` entry point. You must specify the experiment name and a config JSON file:

```bash
# Permuted MNIST
python main.py permuted_mnist -c lop/permuted_mnist/temp_cfg/0.json

# Incremental CIFAR-100
python main.py incremental_cifar -c lop/incremental_cifar/temp_cfg/0.json

# Tiny ImageNet
python main.py imagenet -c lop/imagenet/temp_cfg/0.json
```

### 3. Metric Tracking
Experiments automatically utilize the `lop.metrics.dashboard`. At each task boundary (e.g., when transitioning to a new permutation or a new class subset), the console will print a unified summary:

```text
======================================================================
  Task 0 Summary
======================================================================
  loss: 2.3000  |  train_acc: 0.1000  |  test_acc: 0.1000
  avg_weight_mag:      0.020094 -> 0.020094
  ntk_churn:           0.072735
  ntk_eff_rank:        3.65 -> 3.84
  ntk_cond:            6.21 -> 3.58
  eff_rank:            1.36 -> 1.36
  stable_rank:         4 -> 4
  approx_rank:         1 -> 1
  dormant_proportion:  0.0000 -> 0.0000
======================================================================
```
Full per-layer lists are preserved inside the output dictionaries (`metrics_before`, `metrics_after`) for post-run analysis and visualizations via `lop/viz/`.

## Plasticity Metrics

The codebase standardizes the core diagnostics outlined in LoP literature:
- **Representation Rank (`lop.metrics.rank`)**: Effective, Stable, and Approximate ranks computed via SVD on layer activations to diagnose representation collapse.
- **Dormant Neurons (`lop.metrics.dormant`)**: Fraction of units with negligible activations, isolated for both FFNNs and CNNs.
- **Capacity Tracking (`lop.metrics.weight_norm`)**: Average magnitude of parameters tracking model saturation.
- **Gradient/Kernel Diagnostics (`lop.metrics.ntk_churn`)**: Neural Tangent Kernel (NTK) Churn mapping optimization drift, and Eigenspectrum characteristics defining trainability.
