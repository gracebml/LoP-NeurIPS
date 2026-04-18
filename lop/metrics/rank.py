"""
Rank metrics for Loss of Plasticity.

Effective rank and stable rank logic copied from
lop/incremental_cifar/post_run_analysis.py (canonical source).
Approximate rank and matrix_rank_summaries from lop/utils/miscellaneous.py.
"""

import numpy as np
import torch
from scipy.linalg import svd


def compute_effective_rank(singular_values):
    """
    Computes the effective rank of a representation layer.
    Defined in: https://ieeexplore.ieee.org/document/7098875/

    Canonical source: post_run_analysis.py:134-143

    Args:
        singular_values: numpy array of singular values.

    Returns:
        float: the effective rank.
    """
    norm_sv = singular_values / np.sum(np.abs(singular_values))
    entropy = 0.0
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * np.log(p)

    return np.e ** entropy


def compute_stable_rank(singular_values):
    """
    Computes the stable rank of a representation layer.
    The stable rank is the number of singular values needed to capture 99%
    of the total singular value mass.

    Canonical source: post_run_analysis.py:146-150

    Args:
        singular_values: numpy array of singular values.

    Returns:
        int: the stable rank.
    """
    sorted_singular_values = np.flip(np.sort(singular_values))
    cumsum_sorted_singular_values = np.cumsum(sorted_singular_values) / np.sum(singular_values)
    return np.sum(cumsum_sorted_singular_values < 0.99) + 1


def compute_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in: https://arxiv.org/pdf/1909.12255.pdf

    Source: lop/utils/miscellaneous.py

    Args:
        sv: torch Tensor of singular values.
        prop: proportion of the variance captured by the approximate rank.

    Returns:
        torch.int32: approximate rank.
    """
    sqrd_sv = sv ** 2
    normed_sqrd_sv = torch.flip(
        torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,)
    )
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)


def compute_abs_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank without squaring the singular values.
    See: https://arxiv.org/pdf/1909.12255.pdf

    Source: lop/utils/miscellaneous.py

    Args:
        sv: torch Tensor of singular values.
        prop: proportion captured.

    Returns:
        torch.int32: approximate rank.
    """
    normed_sv = torch.flip(
        torch.sort(sv / torch.sum(sv))[0], dims=(0,)
    )
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)


def compute_matrix_rank_summaries(m: torch.Tensor, prop=0.99, use_scipy=False):
    """
    Computes the rank, effective rank, and approximate rank of a matrix.

    Source: lop/utils/miscellaneous.py

    Args:
        m: a rectangular matrix (torch.Tensor).
        prop: proportion used for computing the approximate rank.
        use_scipy: if True, compute SVD on CPU via scipy (safer for large matrices on GPU).

    Returns:
        rank (int32), effective_rank (float32), approximate_rank (int32),
        approximate_rank_abs (int32)
    """
    if use_scipy:
        np_m = m.cpu().numpy()
        sv = torch.tensor(
            svd(np_m, compute_uv=False, lapack_driver="gesvd"), device=m.device
        )
    else:
        sv = torch.linalg.svdvals(m)

    rank = torch.count_nonzero(sv).to(torch.int32)

    # Effective rank uses numpy version (canonical from post_run_analysis.py)
    sv_np = sv.cpu().numpy()
    effective_rank = torch.tensor(compute_effective_rank(sv_np), dtype=torch.float32)

    approximate_rank = compute_approximate_rank(sv, prop=prop)
    approximate_rank_abs = compute_abs_approximate_rank(sv, prop=prop)

    return rank, effective_rank, approximate_rank, approximate_rank_abs


def compute_stable_rank_from_activations(act):
    """
    Convenience: compute stable rank directly from an activation matrix.
    Performs SVD internally.

    Args:
        act: numpy array or torch.Tensor of activations (batch x features).

    Returns:
        int: the stable rank, or 0 on failure.
    """
    if act is None:
        return 0
    if isinstance(act, torch.Tensor):
        act = act.detach().cpu().numpy()
    if act.ndim > 2:
        act = act.reshape(act.shape[0], -1)
    if act.shape[0] == 0 or act.shape[1] == 0:
        return 0
    try:
        sv = svd(act, compute_uv=False, lapack_driver="gesvd")
        return int(compute_stable_rank(sv))
    except Exception:
        return 0
