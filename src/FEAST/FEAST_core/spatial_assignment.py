"""Shared spatial quantile assignment helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def calculate_quantiles(matrix: np.ndarray) -> np.ndarray:
    """Return average-rank quantiles for each gene/feature column."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array.")
    n, p = matrix.shape
    if n == 0:
        return np.zeros((0, p), dtype=np.float32)
    if n == 1:
        return np.full((1, p), 0.5, dtype=np.float32)
    ranks = np.zeros((n, p), dtype=np.float64)
    for gene_idx in range(p):
        order = np.argsort(matrix[:, gene_idx], kind="mergesort")
        values = matrix[order, gene_idx]
        start = 0
        while start < n:
            end = start + 1
            while end < n and values[end] == values[start]:
                end += 1
            avg_rank = 0.5 * (start + end - 1)
            ranks[order[start:end], gene_idx] = avg_rank
            start = end
    return (ranks / float(n - 1)).astype(np.float32, copy=False)


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """Center/scale coordinates for transport cost construction."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2:
        raise ValueError("coords must be a 2D array.")
    if coords.shape[0] == 0:
        return coords.copy()
    center = coords.mean(axis=0, keepdims=True)
    scale = coords.std(axis=0, keepdims=True)
    scale[scale <= 1e-6] = 1.0
    return (coords - center) / scale


def log_sinkhorn(
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float = 0.1,
    n_iter: int = 100,
    tol: float = 1e-5,
    unbalanced: bool = False,
    reg_m: float = 5.0,
) -> torch.Tensor:
    """Log-domain Sinkhorn solver for spatial quantile transport."""
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    log_a = torch.log(a + 1e-10)
    log_b = torch.log(b + 1e-10)
    fi = reg_m / (reg_m + epsilon) if unbalanced else 1.0

    for _ in range(int(n_iter)):
        f_prev = f.clone()
        tmp = (g.unsqueeze(0) - C) / epsilon
        f = fi * (epsilon * log_a - epsilon * torch.logsumexp(tmp, dim=1))
        tmp = (f.unsqueeze(1) - C) / epsilon
        g = fi * (epsilon * log_b - epsilon * torch.logsumexp(tmp, dim=0))
        if torch.max(torch.abs(f - f_prev)) < tol:
            break

    log_pi = (f.unsqueeze(1) + g.unsqueeze(0) - C) / epsilon
    return torch.exp(log_pi)


def solve_spatial_transport(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    *,
    source_boundary: Optional[np.ndarray] = None,
    target_boundary: Optional[np.ndarray] = None,
    geometry_weight: float = 1.0,
    boundary_weight: float = 0.25,
    epsilon: float = 0.05,
    sinkhorn_iter: int = 200,
    sinkhorn_tol: float = 1e-5,
    unbalanced_transport: bool = True,
    reg_m: float = 5.0,
    torch_device: str = "cpu",
    torch_dtype: str = "float32",
) -> np.ndarray:
    """Solve a spatial OT plan from source spots to target spots."""
    source_coords = np.asarray(source_coords, dtype=np.float32)
    target_coords = np.asarray(target_coords, dtype=np.float32)
    if source_coords.shape[0] == 0 or target_coords.shape[0] == 0:
        return np.zeros((source_coords.shape[0], target_coords.shape[0]), dtype=np.float32)
    source_sq = np.sum(source_coords**2, axis=1, keepdims=True)
    target_sq = np.sum(target_coords**2, axis=1, keepdims=True).T
    dist2 = np.maximum(source_sq + target_sq - 2.0 * source_coords @ target_coords.T, 0.0)
    cost = float(geometry_weight) * dist2

    if source_boundary is not None and target_boundary is not None:
        source_b = np.asarray(source_boundary, dtype=np.float32).reshape(-1, 1)
        target_b = np.asarray(target_boundary, dtype=np.float32).reshape(1, -1)
        cost = cost + float(boundary_weight) * np.abs(source_b - target_b)

    a = np.full(source_coords.shape[0], 1.0 / source_coords.shape[0], dtype=np.float32)
    b = np.full(target_coords.shape[0], 1.0 / target_coords.shape[0], dtype=np.float32)
    dtype = torch.float64 if str(torch_dtype).lower() == "float64" else torch.float32
    device = torch.device(str(torch_device))
    plan = log_sinkhorn(
        C=torch.as_tensor(cost, dtype=dtype, device=device),
        a=torch.as_tensor(a, dtype=dtype, device=device),
        b=torch.as_tensor(b, dtype=dtype, device=device),
        epsilon=float(epsilon),
        n_iter=int(sinkhorn_iter),
        tol=float(sinkhorn_tol),
        unbalanced=bool(unbalanced_transport),
        reg_m=float(reg_m),
    )
    return plan.detach().cpu().numpy().astype(np.float32, copy=False)


def transport_quantiles(
    plan: np.ndarray,
    quantiles: np.ndarray,
    assignment_randomness: float = 0.0,
) -> np.ndarray:
    """Apply a source-target transport plan to a source quantile matrix."""
    plan = np.asarray(plan, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if plan.shape[0] != quantiles.shape[0]:
        raise ValueError("transport plan source dimension does not match source quantiles.")
    if plan.shape[1] == 0:
        return np.zeros((0, quantiles.shape[1]), dtype=np.float32)
    column_mass = plan.sum(axis=0, keepdims=True)
    safe_mass = np.where(column_mass > 1e-12, column_mass, 1.0)
    weights = plan / safe_mass
    transported = weights.T @ quantiles
    randomness = float(np.clip(assignment_randomness, 0.0, 1.0))
    if randomness > 0 and quantiles.shape[0] > 0:
        sampled = quantiles[np.random.randint(0, quantiles.shape[0], size=plan.shape[1]), :]
        transported = (1.0 - randomness) * transported + randomness * sampled
    return np.clip(transported, 0.0, 1.0).astype(np.float32, copy=False)
