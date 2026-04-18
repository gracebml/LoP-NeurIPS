# Can the optimizer itself mitigate Loss of Plasticity (LoP)?

This repository contains the refactored and standardized official implementation for research exploring the question: **Can the optimizer itself mitigate Loss of Plasticity?** 

### Heritage & Acknowledgment
This codebase inherits from and significantly extends the official open-source repository of the Nature (August 2024) paper **"Loss of Plasticity in Deep Continual Learning"** by Shibhansh Dohare et al. 
- **Original Paper**: [Nature DOI: 10.1038/s41586-024-07711-7](https://www.nature.com/articles/s41586-024-07711-7)
- **Original DeepMind Source**: [https://github.com/google-deepmind/loss-of-plasticity](https://github.com/google-deepmind/loss-of-plasticity)

While the original repository from DeepMind established rigorous continuous learning environments (Permuted MNIST, Incremental CIFAR, ImageNet, RL Ant) and primarily explored structural interventions like Continual Backprop (CBP) and Shrink and Perturb, this work integrates second-order optimization and specialized metrics to evaluate them directly inside the same established boundaries.

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

The underlying experimental structure largely follows *Dohare et al.*, wrapped to support our new optimizer factories, Kaggle environments, and the metrics dashboard:

```text
.
├── data/                       # Downloaded datasets
├── download_data.py            # Script to fetch CIFAR, MNIST, TinyImageNet, etc.
├── run-expr.py                 # Kaggle-ready Jupytext executor script for offline experiments
├── lop-packages/               # Offline wheel dependencies for Kaggle
└── lop-src/                    # Core Source Code package
    ├── main.py                 # Unified entry point for all standard experiments
    └── lop/                    # Core Package
        ├── algos/              # Learning algorithms (BP, Continual BP, GnT, SDP, EMA)
        ├── data/               # Unified DataLoaders
        ├── envs/               # RL environments
        ├── imagenet/           # TinyImageNet experiments
        ├── incremental_cifar/  # CIFAR-100 incremental learning experiments
        ├── metrics/            # Centralized Plasticity Metrics (Dashboard, Rank, Dormant, NTK)
        ├── nets/               # Network Architectures (DeepFFNN, ConvNet, ResNets)
        ├── optimizers/         # Second-order optimizers (AdaHessian, SophiaH, Sassha...)
        ├── permuted_mnist/     # Permuted MNIST experiments
        ├── rl/                 # Reinforcement Learning experiments (run_ppo_2nd.py)
        ├── utils/              # Miscellaneous utilities
        └── viz/                # Data visualization scripts
```

## Usage

### 1. Kaggle Offline Execution
Use `run-expr.py` (a Jupytext notebook). It automatically configures datasets and executes your offline code against `lop-src/`. 

### 2. Download Datasets (Local)
Before running experiments locally, fetch and format the necessary benchmarks:

```bash
python download_data.py
```

### 3. Running Experiments
All standardized experiments are executed via scripts inside `lop-src/`. The optimizer and continual learning settings are defined in the specific config files:

```bash
# Permuted MNIST (Test optimizer robustness to abrupt permutation shifts)
python lop-src/main.py mnist -c lop-src/lop/permuted_mnist/cfg/secondorder_sassha.json

# Incremental CIFAR-100 (Test optimizer capability mitigating LoP in class-incremental CNNs)
python lop-src/main.py cifar -c lop-src/lop/incremental_cifar/cfg/sassha_sdp.json --index 0

# ImageNet
python lop-src/main.py imagenet -c lop-src/lop/imagenet/cfg/secondorder_sassha.json

# Reinforcement Learning Agent (Ant-v4)
python lop-src/lop/rl/run_ppo_2nd.py -c lop-src/lop/rl/cfg/ant/sassha_sdp.yml -s 1
```

### 4. Metric Tracking & Dashboard
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
Full per-layer rank progression and dormant counts are seamlessly cached as lists in `metrics_before` and `metrics_after` arrays inside the saved `pickle` results for later high-fidelity plotting in `lop-src/lop/viz/`.
