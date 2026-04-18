"""
K-FAC (Kronecker-Factored Approximate Curvature) Natural Gradient Descent.

Approximates the Fisher Information Matrix for each layer as:
    F ≈ A ⊗ G
where A = E[x xᵀ] (input covariance), G = E[g gᵀ] (output grad covariance).

Natural gradient update: ΔW = −η · G⁻¹ ∇W L · A⁻¹

Supports: nn.Linear, nn.Conv2d. Other layers fall back to SGD.

Reference: Martens & Grosse, "Optimizing Neural Networks with
Kronecker-factored Approximate Curvature", ICML 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class KFACOptimizer(Optimizer):
    """
    K-FAC Natural Gradient Descent.

    Supports: nn.Linear, nn.Conv2d. Other layers fall back to SGD.
    """

    def __init__(self, model, lr=0.01, damping=1e-3, weight_decay=0.0,
                 T_inv=100, alpha=0.95, max_grad_norm=1.0):
        """
        Args:
            model: nn.Module to optimize (hooks registered on Conv2d/Linear).
            lr: Learning rate.
            damping: Tikhonov damping λ. sqrt(λ) added to each factor.
            weight_decay: L2 regularization coefficient.
            T_inv: Frequency (in steps) to recompute matrix inverses.
            alpha: EMA coefficient for running A and G stats.
            max_grad_norm: Gradient clipping norm (0 = no clipping).
        """
        self.model = model
        self.damping = damping
        self._init_damp = damping
        self.weight_decay = weight_decay
        self.T_inv = T_inv
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.steps = 0

        # Storage for Kronecker factors and their inverses
        self._modules_tracked = {}   # name -> module
        self._stats = {}             # name -> {'A': Tensor, 'G': Tensor}
        self._inv = {}               # name -> {'A_inv': ..., 'G_inv': ...}
        self._hooks = []

        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)
        self._register_hooks()

    def _register_hooks(self):
        """Register forward/backward hooks on Conv2d and Linear layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self._modules_tracked[name] = module
                if name not in self._stats:
                    self._stats[name] = {'A': None, 'G': None}
                h1 = module.register_forward_hook(
                    self._forward_hook(name, module))
                h2 = module.register_full_backward_hook(
                    self._backward_hook(name, module))
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
                    x = F.unfold(x, mod.kernel_size, dilation=mod.dilation,
                                 padding=mod.padding, stride=mod.stride)
                    x = x.permute(0, 2, 1).reshape(-1, x.size(1))
                elif x.dim() > 2:
                    x = x.reshape(-1, x.size(-1))
                if mod.bias is not None:
                    ones = torch.ones(x.size(0), 1, device=x.device)
                    x = torch.cat([x, ones], dim=1)
                n = x.size(0)
                cov_a = torch.matmul(x.t(), x) / n
                s = self._stats[name]
                if s['A'] is None:
                    s['A'] = cov_a
                else:
                    s['A'].mul_(self.alpha).add_(cov_a, alpha=1 - self.alpha)
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
                elif g.dim() > 2:
                    g = g.reshape(-1, g.size(-1))
                n = g.size(0)
                cov_g = torch.matmul(g.t(), g) / n
                s = self._stats[name]
                if s['G'] is None:
                    s['G'] = cov_g
                else:
                    s['G'].mul_(self.alpha).add_(cov_g, alpha=1 - self.alpha)
        return hook

    @torch.no_grad()
    def _invert_factors(self):
        """Invert A and G with damping. Called every T_inv steps."""
        if self.steps < self.T_inv:
            return  # skip warmup — stats too noisy
        sqrt_d = self.damping ** 0.5
        for name in self._stats:
            A = self._stats[name]['A']
            G = self._stats[name]['G']
            if A is None or G is None:
                continue
            try:
                A_d = A + sqrt_d * torch.eye(A.size(0), device=A.device)
                G_d = G + sqrt_d * torch.eye(G.size(0), device=G.device)
                self._inv[name] = {
                    'A_inv': torch.linalg.inv(A_d),
                    'G_inv': torch.linalg.inv(G_d),
                }
            except RuntimeError:
                pass  # keep previous inverse if singular

    def reset_stats(self):
        """Reset running statistics at task boundaries."""
        for name in self._stats:
            self._stats[name] = {'A': None, 'G': None}
        self._inv.clear()
        self.steps = 0
        self.damping = self._init_damp

    def remove_hooks(self):
        """Remove all forward/backward hooks (call before eval)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reregister_hooks(self):
        """Re-register hooks (call after eval, before next training)."""
        self._register_hooks()

    def state_dict(self):
        """Extended state_dict that includes K-FAC-specific state."""
        sd = super().state_dict()
        sd['kfac_steps'] = self.steps
        sd['kfac_damping'] = self.damping
        sd['kfac_stats'] = {
            name: {
                'A': s['A'].clone() if s['A'] is not None else None,
                'G': s['G'].clone() if s['G'] is not None else None,
            } for name, s in self._stats.items()
        }
        return sd

    def load_state_dict(self, state_dict):
        """Extended load_state_dict that restores K-FAC-specific state."""
        kfac_steps = state_dict.pop('kfac_steps', 0)
        kfac_damping = state_dict.pop('kfac_damping', self._init_damp)
        state_dict.pop('kfac_damping_boost', None)
        state_dict.pop('kfac_boost_decay', None)
        kfac_stats = state_dict.pop('kfac_stats', None)
        super().load_state_dict(state_dict)
        self.steps = kfac_steps
        self.damping = kfac_damping
        if kfac_stats is not None:
            for name in kfac_stats:
                if name in self._stats:
                    self._stats[name] = kfac_stats[name]
        self._invert_factors()

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

        # 1. Compute natural gradients for all tracked layers
        updates = {}
        for name, module in self._modules_tracked.items():
            if module.weight.grad is None:
                continue
            grad_w = module.weight.grad
            has_b = (module.bias is not None
                     and module.bias.grad is not None)

            if name in self._inv:
                A_inv = self._inv[name]['A_inv']
                G_inv = self._inv[name]['G_inv']

                if isinstance(module, nn.Conv2d):
                    c_out = grad_w.size(0)
                    grad_2d = grad_w.reshape(c_out, -1)
                    if has_b:
                        grad_2d = torch.cat(
                            [grad_2d, module.bias.grad.unsqueeze(1)], dim=1)
                    nat_grad = torch.matmul(
                        G_inv, torch.matmul(grad_2d, A_inv))
                    if has_b:
                        nat_grad_w = nat_grad[:, :-1].reshape_as(module.weight)
                        nat_grad_b = nat_grad[:, -1]
                    else:
                        nat_grad_w = nat_grad.reshape_as(module.weight)
                        nat_grad_b = None
                else:  # Linear
                    if has_b:
                        grad_2d = torch.cat(
                            [grad_w, module.bias.grad.unsqueeze(1)], dim=1)
                    else:
                        grad_2d = grad_w
                    nat_grad = torch.matmul(
                        G_inv, torch.matmul(grad_2d, A_inv))
                    if has_b:
                        nat_grad_w = nat_grad[:, :-1]
                        nat_grad_b = nat_grad[:, -1]
                    else:
                        nat_grad_w = nat_grad
                        nat_grad_b = None
            else:
                # No inverse available yet → plain SGD fallback
                nat_grad_w = grad_w.clone()
                nat_grad_b = module.bias.grad.clone() if has_b else None

            updates[name] = (nat_grad_w, nat_grad_b)

        # 2. Gradient clipping on natural gradient norm
        clip = 1.0
        if self.max_grad_norm > 0 and len(updates) > 0:
            total_sq = sum(
                ngw.norm().item()**2
                + (ngb.norm().item()**2 if ngb is not None else 0.0)
                for ngw, ngb in updates.values()
            )
            if total_sq**0.5 > self.max_grad_norm:
                clip = self.max_grad_norm / (total_sq**0.5 + 1e-6)

        # 3. Apply updates with weight decay
        for name, (nat_grad_w, nat_grad_b) in updates.items():
            module = self._modules_tracked[name]
            if self.weight_decay > 0:
                nat_grad_w = nat_grad_w + self.weight_decay * module.weight
            module.weight.data.add_(nat_grad_w, alpha=-lr * clip)
            if nat_grad_b is not None and module.bias is not None:
                module.bias.data.add_(nat_grad_b, alpha=-lr * clip)

        # 4. SGD for non-tracked parameters (e.g., BatchNorm)
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
                    p.data.mul_(1.0 - lr * self.weight_decay)
                p.data.add_(p.grad, alpha=-lr)

        self.steps += 1
        return loss
