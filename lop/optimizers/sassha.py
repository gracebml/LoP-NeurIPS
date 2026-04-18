"""
SASSHA — SAM + Hutchinson Hessian trace optimizer.

Source: https://github.com/LOG-postech/Sassha
Reference: Shin et al., "SASSHA: Sharpness-Aware Stochastic Hessian
Approximation", ICML 2025.

Also includes BatchNorm stat helpers for SAM-style perturbation passes.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer


class SASSHA(Optimizer):
    """SASSHA — SAM + Hutchinson Hessian trace (ICML 2025)."""

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), weight_decay=0.0,
                 rho=0.2, lazy_hessian=10, n_samples=1, perturb_eps=1e-12,
                 eps=1e-4, adaptive=False, hessian_power=1.0, seed=0, **kwargs):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[1]}")

        self.lazy_hessian = lazy_hessian
        self.n_samples = n_samples
        self.adaptive = adaptive
        self.seed = seed
        self.hessian_power_t = hessian_power

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        rho=rho, perturb_eps=perturb_eps, eps=eps)
        super().__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0
        self.generator = torch.Generator().manual_seed(self.seed)

    def get_params(self):
        return (p for group in self.param_groups
                for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        for p in self.get_params():
            if (not isinstance(p.hess, float)
                    and self.state[p]["hessian step"] % self.lazy_hessian == 0):
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.lazy_hessian == 0:
                params.append(p)
            self.state[p]["hessian step"] += 1
        if len(params) == 0:
            return
        if self.generator.device != params[0].device:
            self.generator = torch.Generator(params[0].device).manual_seed(self.seed)
        grads = [p.grad for p in params]
        last_sample = self.n_samples - 1
        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator,
                                device=p.device) * 2.0 - 1.0 for p in params]
            h_zs = torch.autograd.grad(
                grads, params, grad_outputs=zs,
                only_inputs=True, retain_graph=i < last_sample)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples

    @torch.no_grad()
    def perturb_weights(self, zero_grad=True):
        """SAM-style weight perturbation."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["perturb_eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def unperturb(self):
        """Undo SAM-style weight perturbation."""
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def _grad_norm(self, by=None):
        if not by:
            return torch.norm(torch.stack([
                ((torch.abs(p.data) if self.adaptive else 1.0)
                 * p.grad).norm(p=2)
                for group in self.param_groups
                for p in group["params"] if p.grad is not None
            ]), p=2)
        return torch.norm(torch.stack([
            ((torch.abs(p.data) if self.adaptive else 1.0)
             * self.state[p][by]).norm(p=2)
            for group in self.param_groups
            for p in group["params"] if p.grad is not None
        ]), p=2)

    @torch.no_grad()
    def step(self, closure=None, compute_hessian=True):
        loss = None
        if closure is not None:
            loss = closure()
        if compute_hessian:
            self.zero_hessian()
            self.set_hessian()
        k = self.hessian_power_t
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue
                if isinstance(p.hess, (int, float)):
                    p.hess = torch.zeros_like(p.data)
                else:
                    p.hess = p.hess.abs().clone()
                p.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 2:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag'] = torch.zeros_like(p.data)
                    state['bias_correction2'] = 0
                exp_avg = state['exp_avg']
                exp_hessian_diag = state['exp_hessian_diag']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                bias_correction1 = 1 - beta1 ** state['step']
                if (state['hessian step'] - 1) % self.lazy_hessian == 0:
                    exp_hessian_diag.mul_(beta2).add_(p.hess, alpha=1 - beta2)
                    bias_correction2 = 1 - beta2 ** state['step']
                    state['bias_correction2'] = bias_correction2 ** k
                step_size = group['lr'] / bias_correction1
                denom = ((exp_hessian_diag ** k)
                         / max(state['bias_correction2'], 1e-12)).add_(
                             group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


def disable_running_stats(model):
    """Freeze BatchNorm running stats during SAM perturbation pass."""
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)


def enable_running_stats(model):
    """Restore BatchNorm momentum after SAM perturbation pass."""
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, 'backup_momentum'):
            module.momentum = module.backup_momentum
    model.apply(_enable)
