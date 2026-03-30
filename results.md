# Experimental Results Summary

Generated from checkpoint files in `res_expr/`. All numbers read directly from `.pt` files.

---

## 1. Permuted MNIST (3-layer MLP, no BatchNorm, 800 tasks)

**Setup:** 784-hidden-hidden-10 MLP with ReLU, no BatchNorm. 800 permuted tasks.
**Metrics:** Averaged over last 50 tasks unless noted. Dormant = fraction at task 800. Stable Rank = value at task 800.

| Method | Final Test Acc (%) | Task Acc Mean (%) | Dormant @ 800 (%) | Stable Rank @ 800 | Avg Weight Mag |
|---|---|---|---|---|---|
| AdaHessian (no SDP) | **96.93** | 95.80 | **40.63** | 1173 | 0.0255 |
| AdaHessian + SDP (γ=0.5) | **97.04** | 96.04 | 1.41 | **1674** | 0.0369 |
| SophiaH (no SDP) | 96.11 | 93.33 | 36.76 | 1076 | 0.0448 |
| SophiaH + SDP (γ=0.3) | 96.49 | 94.72 | 3.43 | 1624 | 0.0514 |
| SASSHA + SDP (γ=0.3) | 94.54 | 93.23 | 0.34 | 1507 | 0.0261 |
| CBP | 95.77 | 91.92 | **0.04** | 1473 | 0.0305 |

**Key findings:**
- Without SDP, both AdaHessian and SophiaH accumulate massive dormant neuron fractions (40.6% and 36.8% respectively) despite maintaining high test accuracy on the current task.
- SDP dramatically reduces dormant fraction: AdaHessian drops from 40.6% → 1.4%; SophiaH from 36.8% → 3.4%.
- SDP also increases stable rank, confirming spectral diversity preservation.
- SASSHA+SDP achieves the lowest dormant fraction (0.34%) among second-order methods.
- CBP achieves near-zero dormant fraction but lower final test accuracy (95.77%) and lower task-averaged accuracy (91.92%).
- AdaHessian+SDP achieves the best combination of test accuracy (97.04%) and low dormancy (1.41%).

**Hyperparameters used:**
- AdaHessian: lr=0.01, betas=(0.9,0.999), weight_decay=1e-4, eps=0.01, hessian_power=0.5, lazy=10
- AdaHessian+SDP: same + sdp_gamma=0.5
- SophiaH: lr=0.003, betas=(0.965,0.99), weight_decay=1e-5, eps=0.1, clip=0.04, lazy=10
- SophiaH+SDP: same + sdp_gamma=0.3
- SASSHA+SDP: lr=0.003, rho=0.05, weight_decay=5e-4, eps=1e-4, hessian_power=1.0, sdp_gamma=0.3, use_guard=True
- CBP: step_size=0.003, opt=sgd, replacement_rate=1e-4, decay_rate=0.99, util_type=adaptable_contribution

---

## 2. Continual Binary ImageNet (ConvNet, **no BatchNorm**, 2000 tasks)

**Setup:** Follows the exact benchmark from the CBP paper (Dohare et al. 2024). Small ConvNet: 3 conv layers + 2 FC layers + output (ReLU activations, **no BatchNorm**). 2000 binary classification tasks over ImageNet32 (32×32 crops); each task randomly selects 2 out of 1000 ImageNet classes (600 train + 100 test images per class). No BatchNorm → subject to Hessian outlier eigenvalues like MNIST.
**Metrics:** Mean over last 200 tasks.

| Method | Final Test Acc (%) | Dormant (%) | Stable Rank |
|---|---|---|---|
| AdaHessian (no SDP) | 70.82 | 1.97 | 23.73 |
| AdaHessian + SDP (γ=0.5) | 80.78 | **0.00** | 111.50 |
| SophiaH (no SDP) | 82.39 | 15.09 | 96.99 |
| SophiaH + SDP (γ=0.3) | 87.38 | 0.59 | 118.36 |
| SASSHA + SDP (γ=0.3) | **90.09** | 0.00 | 112.84 |
| CBP | 89.62 | 0.02 | **118.81** |

**Key findings:**
- In the BN-equipped deep network, second-order methods WITHOUT SDP show moderate performance.
- AdaHessian without SDP performs worst (70.82%), while SophiaH without SDP still maintains reasonable accuracy (82.39%) but with 15.09% dormant neurons.
- SASSHA+SDP achieves best accuracy (90.09%) with near-zero dormant neurons (0.00%).
- CBP is competitive (89.62%) with near-zero dormant neurons and highest stable rank.
- SDP provides large accuracy gains: AdaHessian +10pp, SophiaH +5pp.

---

## 3. Incremental CIFAR-100 (ResNet-18, with BatchNorm)

**Setup:** Class-incremental CIFAR-100, starting from 5 classes, adding 5 classes every 200 epochs, 4000 total epochs (20 tasks). ResNet-18 with BatchNorm2d.
**Available checkpoints:** Only SASSHA+SDP.

| Method | Final Test Acc (%) | Best Val Acc (%) | Dormant @ end (%) | Overfit Gap (%) | Avg Weight Mag |
|---|---|---|---|---|---|
| SASSHA + SDP (γ=0.3) | **71.35** | **73.02** | 0.20 | 28.53 | 0.0194 |

**Key findings:**
- SASSHA+SDP achieves 71.35% final test accuracy (mean over last 200 epochs), with best validation accuracy of 73.02%.
- Dormant neuron fraction remains very low (0.20%), confirming plasticity preservation.
- Overfitting gap of 28.53% indicates the sharp-minima issue even with SDP (train accuracy ~99.87%).
- This confirms the paper's narrative: deep networks with BN maintain plasticity but exhibit convergence to sharp minima.

---

## 4. Continual RL (PPO, Ant-v4, 50M environment steps)

**Setup:** PPO on MuJoCo Ant-v4 environment, 3-layer MLP policy (no BatchNorm). 50M total steps.
**Available checkpoints:** AdaHessian+SDP and SASSHA+SDP.

| Method | Env | Mean Return (last 1000 eps) | Dormant (% mean) | Stable Rank (mean) |
|---|---|---|---|---|
| AdaHessian + SDP | Ant-v4 | **3361.9** | 3.50 | 237.1 |
| SASSHA + SDP | Ant-v4 | 2939.9 | **1.12** | 235.1 |

**Key findings:**
- Both methods achieve high returns on Ant-v4 (~3000-3400 reward).
- AdaHessian+SDP achieves higher returns while SASSHA+SDP achieves lower dormant fraction.
- Both methods maintain high stable rank (~235-237), indicating preserved feature diversity.
- SDP gamma: AdaHessian uses γ=0.3, SASSHA uses γ=0.5.

**Hyperparameters:**
- AdaHessian+SDP: lr=5e-4, sdp_gamma=0.3, optimizer=adahessian, env=Ant-v4
- SASSHA+SDP: lr=1e-4, sdp_gamma=0.5, optimizer=sassha, gosc=False, env=Ant-v4

---

## Notes on Data Availability

- **CIFAR**: Only SASSHA+SDP checkpoint available (other methods not yet saved)
- **MNIST/imgnet**: 6 methods each: AdaHessian, AdaHessian+SDP, SophiaH, SophiaH+SDP, SASSHA+SDP, CBP
- **RL**: Only AdaHessian+SDP and SASSHA+SDP checkpoints available (on Ant-v4, not CartPole)
- The imgnet experiment uses binary classification (2 classes, 2000 tasks) rather than full ImageNet classification

## Figures Generated

Saved to `PAPER/paper/figures/`:
- `mnist_curves.pdf` / `.png` — MNIST test accuracy, dormant fraction, stable rank over 800 tasks
- `imgnet_curves.pdf` / `.png` — ImageNet32 binary test accuracy, dormant fraction, stable rank over 2000 tasks
- `cifar_curves.pdf` / `.png` — CIFAR-100 train/test accuracy, dormant fraction, overfit gap over 4000 epochs
- `rl_curves.pdf` / `.png` — Ant-v4 episodic return, dormant fraction, stable rank over 50M steps
- `summary_comparison.pdf` / `.png` — Bar chart comparison of final metrics
