# Can the optimizer itself mitigate Loss of Plasticity (LoP)?

This repository contains the refactored and standardized official implementation for research exploring the question: **Can the optimizer itself mitigate Loss of Plasticity?** 

While existing solutions to the Loss of Plasticity (LoP) phenomenon typically rely on structural interventions (e.g., continually injecting new neurons, resetting dead units, or heavily regularizing the network), this work investigates whether **second-order optimization methods** (such as AdaHessian, SophiaH, and Sassha) naturally maintain network plasticity over long continual learning horizons.

## Research Scope & Framework

This codebase provides a unified framework to train, monitor, and evaluate continual learning agents across sequential tasks. It tracks how representations and optimization landscapes evolve by computing state-of-the-art plasticity diagnostics.

### Key Contributions in this Repository:
- **Second-Order Optimizer Implementations (`lop.optimizers`)**: highly-optimized, modular PyTorch implementations of advanced optimizers including **AdaHessian**, **SophiaH**, and **Sassha**, designed for plug-and-play use in continual learning loops.
- **Centralized Metric Dashboard (`lop.metrics.dashboard`)**: A unified engine that computes 11 core plasticity diagnostics at the boundary of every sequentially learned task:
  - **Representation Rank**: Effective, Stable, and Approximate Rank (SVD-based breakdown of representation collapse).
  - **Network Capacity**: Dormant Proportion (dead unit tracking) and Average Weight Magnitude.
  - **Landscape Diagnostics**: NTK Churn and Empirical NTK Eigenspectrum Conditioning (linking optimization curvature to plasticity).
- **Standardized Continual Benchmarks**: unified execution for Permuted MNIST, Incremental CIFAR-100, and standard sequentially arriving subsets for Tiny ImageNet.

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
    ├── metrics/                # Centralized Plasticity Metrics (Dashboard, Rank, Dormant, NTK)
    ├── nets/                   # Network Architectures (DeepFFNN, ConvNet, ResNets)
    ├── optimizers/             # Second-order optimizers (AdaHessian, SophiaH, Sassha...)
    ├── permuted_mnist/         # Permuted MNIST experiments
    ├── rl/                     # Reinforcement Learning experiments
    ├── utils/                  # Miscellaneous utilities
    └── viz/                    # Data visualization scripts
```

## Usage

### 1. Download Datasets
Before running experiments, fetch and format the necessary benchmarks:

```bash
python download_data.py
```

### 2. Running Experiments
All experiments are executed via `main.py`. The optimizer and continual learning settings are defined in the config JSON file:

```bash
# Permuted MNIST (Test optimizer robustness to abrupt permutation shifts)
python main.py permuted_mnist -c lop/permuted_mnist/temp_cfg/0.json

# Incremental CIFAR-100 (Test optimizer capability mitigating LoP in class-incremental CNNs)
python main.py incremental_cifar -c lop/incremental_cifar/temp_cfg/0.json

# ImageNet
python main.py imagenet -c lop/imagenet/temp_cfg/0.json
```

### 3. Metric Tracking & Dashboard
At each task boundary, the framework will automatically halt, run reference batches through the dashboard, and compute the second-order landscape and capacity metrics to measure the exact state of plasticity:

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
Full per-layer rank progression and dormant counts are seamlessly cached as lists in `metrics_before` and `metrics_after` arrays inside the saved `pickle` results for later high-fidelity plotting in `lop/viz/`.
