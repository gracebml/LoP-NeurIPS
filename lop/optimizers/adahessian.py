"""
AdaHessian — Adaptive second-order optimizer using Hutchinson trace.

Source: https://github.com/davda54/ada-hessian
Reference: Yao et al., "ADAHESSIAN: An Adaptive Second Order Optimizer for
Machine Learning", AAAI 2021.
"""

import torch
from torch.optim import Optimizer


class Adahessian(Optimizer):
    """AdaHessian — Adaptive second-order optimizer using Hutchinson trace."""

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0.0, hessian_power=1, lazy_hessian=1,
                 n_samples=1, seed=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid hessian_power: {hessian_power}")

        self.n_samples = n_samples
        self.lazy_hessian = lazy_hessian
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        hessian_power=hessian_power)
        super().__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

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
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self.zero_hessian()
        self.set_hessian()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue
                if p.dim() <= 2:
                    p.hess = p.hess.abs().clone()
                if p.dim() == 4:
                    p.hess = torch.mean(p.hess.abs(), dim=[2, 3],
                                        keepdim=True).expand_as(p.hess).clone()
                p.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)
                exp_avg = state['exp_avg']
                exp_hessian_diag_sq = state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(
                    p.hess, p.hess, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(
                    k / 2).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
