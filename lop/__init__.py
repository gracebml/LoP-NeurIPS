"""
Loss of Plasticity (LoP) — Research Codebase.

Central module providing:
    - lop.metrics: All LoP metrics (dormant ratio, rank, weight norm, NTK churn)
    - lop.optimizers: Second-order optimizers (AdaHessian, SophiaH, KFAC, SASSHA)
    - lop.algos: Training algorithms (Backprop, CBP, SDP, EMA)
    - lop.data: Unified data loading (MNIST, CIFAR, TinyImageNet)
    - lop.viz: Centralized visualization
    - lop.nets: Network architectures (FFNN, ResNet, Conv, policies)
"""
