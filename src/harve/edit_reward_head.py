"""Reward-head editing (HARVE step 2).

Given the pooled per-RM direction matrix V = [v_1, …, v_k] and an editing
strength α, HARVE produces an edited score-head weight::

    w_r_α  =  w_r  −  α · U_k U_k^T w_r

where U_k is the orthonormal basis of span(V) recovered via SVD (with a small
numerical threshold to filter out degenerate directions). 

"""
from __future__ import annotations

import torch


_DEFAULT_SVD_THRESHOLD = 1e-6


def build_ablation_basis(V: torch.Tensor,
                         *,
                         svd_threshold: float = _DEFAULT_SVD_THRESHOLD,
                         ) -> torch.Tensor:
    """Compute the orthonormal basis U_k of span(V) via SVD.

    Args:
        V: (k, d) — stacked direction vectors (rows are v_k's, each unit-norm).
        svd_threshold: singular values ≤ this are dropped (numerical zeros).

    Returns:
        U_k of shape (d, rank), where rank ≤ k.
    """
    if V.ndim != 2:
        raise ValueError(f"V must be (k, d); got {tuple(V.shape)}")
    if V.shape[0] == 0:
        return torch.zeros(V.shape[1], 0, dtype=V.dtype, device=V.device)
    # SVD on V.T = U S Vh  with U: (d, k), S: (k,), Vh: (k, k)
    U, S, _ = torch.linalg.svd(V.T, full_matrices=False)
    rank = int((S > svd_threshold).sum())
    return U[:, :rank]


def ablate_w_r(w_r: torch.Tensor,
               V: torch.Tensor,
               alpha: float,
               *,
               svd_threshold: float = _DEFAULT_SVD_THRESHOLD,
               ) -> torch.Tensor:
    """Apply the HARVE edit: w_r_α = w_r − α · U_k U_k^T w_r.

    Args:
        w_r:    (d,)  base score-head weight.
        V:      (k, d) stacked direction vectors.
        alpha:  scalar editing strength (0 = baseline, 1 = exact projection,
                >1 = amplified projection).
        svd_threshold: passed through to ``build_ablation_basis``.
    Returns:
        w_r_α: (d,)
    """
    w_r = w_r.float()
    if alpha == 0.0 or V.numel() == 0:
        return w_r.clone()
    V = V.to(dtype=torch.float32, device=w_r.device)
    U_k = build_ablation_basis(V, svd_threshold=svd_threshold)
    if U_k.shape[1] == 0:
        return w_r.clone()
    return w_r - alpha * (U_k @ (U_k.T @ w_r))
