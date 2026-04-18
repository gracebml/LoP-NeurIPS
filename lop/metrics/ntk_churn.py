"""
NTK Churn metrics for Loss of Plasticity.

Two components:
1. Output churn — lightweight measure of prediction instability after a gradient step.
   Inspired by C-CHAIN's policy_churn (ICML 2025).
2. Empirical NTK eigenspectrum — tracks rank collapse, a key indicator of plasticity loss.
"""

import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def compute_ntk_churn(model, ref_inputs, prev_outputs, loss_type='mse'):
    """
    Compute NTK churn = E[(f_{θ'}(x) - f_θ(x))²] for reference inputs.

    Measures how much the network's output changes on held-out reference data
    after a parameter update. High churn → unstable learning → plasticity loss.

    Inspired by C-CHAIN: policy_churn = ((cur_ref_action_means - ref_action_means) ** 2).mean()
    from crl_dmc/crl_run_ppo_dmc.py:318

    Args:
        model: current model (post-update θ').
        ref_inputs: held-out reference batch (torch.Tensor).
        prev_outputs: model outputs f_θ(x) stored BEFORE the update (torch.Tensor).
        loss_type: 'mse' for regression churn, 'ce' for classification churn (KL-div).

    Returns:
        churn: float, scalar measuring output instability.
    """
    model.eval()
    cur_outputs = model(ref_inputs)

    if loss_type == 'mse':
        # MSE churn: E[(f_θ'(x) - f_θ(x))²]
        churn = ((cur_outputs - prev_outputs) ** 2).mean().item()
    elif loss_type == 'ce':
        # KL-divergence churn for classification:
        # KL(p_old || p_new) where p = softmax(logits)
        prev_probs = F.softmax(prev_outputs, dim=-1)
        cur_log_probs = F.log_softmax(cur_outputs, dim=-1)
        churn = F.kl_div(cur_log_probs, prev_probs, reduction='batchmean').item()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'mse' or 'ce'.")

    model.train()
    return churn


def compute_empirical_ntk_eigenspectrum(model, ref_inputs, num_outputs=None):
    """
    Compute the empirical NTK matrix K(x_i, x_j) = <∇f(x_i), ∇f(x_j)>
    and return its eigenvalues (sorted descending).

    Monitors: rank collapse, spectral decay, condition number.
    Uses per-sample Jacobian computation for memory efficiency.

    Args:
        model: torch.nn.Module. Must be in eval mode or handle batch correctly.
        ref_inputs: (N, *) tensor of N reference inputs.
        num_outputs: number of output dimensions to use. If None, uses all.

    Returns:
        eigenvalues: numpy array of shape (N,), eigenvalues sorted descending.
        condition_number: float, ratio of largest to smallest eigenvalue.
        effective_rank: float, effective rank of the NTK matrix.
    """
    model.eval()
    N = ref_inputs.shape[0]
    device = ref_inputs.device

    # Collect per-sample gradients (Jacobian rows)
    # For each sample x_i, compute g_i = ∇_θ f(x_i) flattened
    jacobian_rows = []

    for i in range(N):
        model.zero_grad()
        x_i = ref_inputs[i:i+1]
        output = model(x_i)

        if num_outputs is not None:
            output = output[:, :num_outputs]

        # Sum over output dimensions to get a scalar for backward
        output_sum = output.sum()
        output_sum.backward()

        # Flatten all parameter gradients into a single vector
        grad_vec = []
        for p in model.parameters():
            if p.grad is not None:
                grad_vec.append(p.grad.detach().flatten())
        grad_vec = torch.cat(grad_vec)
        jacobian_rows.append(grad_vec)

    # Stack into Jacobian matrix: (N, P) where P = total parameters
    J = torch.stack(jacobian_rows)  # (N, P)

    # Compute NTK matrix: K = J @ J^T  (N, N)
    K = J @ J.t()

    # Eigendecomposition
    eigenvalues = torch.linalg.eigvalsh(K).cpu().numpy()
    eigenvalues = np.flip(np.sort(eigenvalues))  # descending order

    # Condition number
    max_ev = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
    min_ev_pos = eigenvalues[eigenvalues > 1e-12]
    min_ev = min_ev_pos[-1] if len(min_ev_pos) > 0 else 1e-12
    condition_number = float(max_ev / min_ev) if min_ev > 0 else float('inf')

    # Effective rank of NTK
    sv = np.sqrt(np.maximum(eigenvalues, 0))
    if np.sum(np.abs(sv)) > 0:
        norm_sv = sv / np.sum(np.abs(sv))
        entropy = 0.0
        for p in norm_sv:
            if p > 0.0:
                entropy -= p * np.log(p)
        effective_rank = float(np.e ** entropy)
    else:
        effective_rank = 0.0

    model.zero_grad()
    model.train()
    return eigenvalues, condition_number, effective_rank
