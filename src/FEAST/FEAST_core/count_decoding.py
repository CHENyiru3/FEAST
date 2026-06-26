"""Rank-based count decoding for FEAST simulation.

Single pipeline: sample per gene → sort → assign by argsort(quantiles).
Streaming decode avoids materialising the full raw_counts intermediate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _dense_array(X, dtype=None) -> Optional[np.ndarray]:
    if X is None:
        return None
    if hasattr(X, "toarray"):
        X = X.toarray()
    arr = np.asarray(X)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _sparse_per_gene_max(X, n_genes: int) -> np.ndarray:
    """Compute per-gene maximum from a potentially sparse matrix."""
    from scipy.sparse import issparse
    if issparse(X):
        return np.asarray(X.max(axis=0).toarray()).ravel()
    return np.max(np.asarray(X, dtype=np.float64), axis=0).reshape(-1)


def _model_type_and_params(model_params: dict, gene_idx: int):
    model_selected = model_params.get("model_selected", [])
    marginal_param1 = model_params.get("marginal_param1", [])
    model_type = model_selected[gene_idx] if gene_idx < len(model_selected) else "Poisson"
    params = marginal_param1[gene_idx] if gene_idx < len(marginal_param1) else [0.0, 1.0, 1.0]
    if not isinstance(params, (list, tuple, np.ndarray)):
        params = [0.0, 1.0, 1.0]
    pi0 = float(params[0]) if len(params) > 0 and np.isfinite(params[0]) else 0.0
    r = float(params[1]) if len(params) > 1 and np.isfinite(params[1]) else 1.0
    mu = float(params[2]) if len(params) > 2 and np.isfinite(params[2]) else 1.0
    return str(model_type), float(np.clip(pi0, 0.0, 1.0)), max(r, 1e-8), max(mu, 1e-8)


def _boundary_per_gene(
    reference_X,
    n_genes: int,
    model_params: dict,
    boundary_multiplier: float,
) -> np.ndarray:
    boundary = np.full(n_genes, np.inf, dtype=np.float64)
    if reference_X is not None and (hasattr(reference_X, 'shape') and reference_X.shape[0] > 0):
        boundary = _sparse_per_gene_max(reference_X, n_genes) * float(boundary_multiplier)
        if boundary.shape[0] != n_genes:
            boundary = np.resize(boundary, n_genes).astype(np.float64)

    for gene_idx in range(n_genes):
        _, _, _, mu = _model_type_and_params(model_params, gene_idx)
        if boundary[gene_idx] < 1.0 and mu > 1e-6:
            boundary[gene_idx] = np.inf
    return boundary


def _sample_gene_counts(model_type: str, pi0: float, r: float, mu: float, n_spots: int, rng) -> np.ndarray:
    if model_type == "Poisson":
        return rng.poisson(mu, size=n_spots)
    if model_type == "NB":
        p = r / (r + mu)
        return rng.negative_binomial(r, np.clip(p, 1e-8, 1.0 - 1e-8), size=n_spots)
    if model_type == "ZIP":
        counts = rng.poisson(mu, size=n_spots)
        counts[rng.random(n_spots) < pi0] = 0
        return counts
    if model_type == "ZINB":
        p = r / (r + mu)
        counts = rng.negative_binomial(r, np.clip(p, 1e-8, 1.0 - 1e-8), size=n_spots)
        counts[rng.random(n_spots) < pi0] = 0
        return counts
    return rng.poisson(mu, size=n_spots)


def generate_count_bag_from_model_params(
    model_params: dict,
    n_spots: int,
    *,
    boundary_multiplier: float = 1.1,
    reference_X=None,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Sample an unordered per-gene count bag from fitted FEAST model params.

    For large datasets, prefer decode_counts_by_rank() directly — it streams
    per gene and avoids allocating this full intermediate matrix.
    """
    if "model_selected" not in model_params or "marginal_param1" not in model_params:
        raise ValueError("model_params must contain 'model_selected' and 'marginal_param1'.")

    n_genes = len(model_params["model_selected"])
    rng = np.random if random_seed is None else np.random.default_rng(int(random_seed))
    boundary = _boundary_per_gene(reference_X, n_genes, model_params, boundary_multiplier)
    counts = np.zeros((int(n_spots), n_genes), dtype=np.float32)

    for gene_idx in range(n_genes):
        model_type, pi0, r, mu = _model_type_and_params(model_params, gene_idx)
        gene_counts = _sample_gene_counts(model_type, pi0, r, mu, int(n_spots), rng).astype(np.float32)
        gene_boundary = boundary[gene_idx]
        if np.isfinite(gene_boundary):
            gene_counts = np.minimum(gene_counts, gene_boundary)
        counts[:, gene_idx] = gene_counts

    return counts


def decode_counts_by_rank(
    quantiles: np.ndarray,
    model_params: dict,
    *,
    spot_weights: Optional[np.ndarray] = None,
    boundary_multiplier: float = 1.1,
    reference_X=None,
    random_seed: Optional[int] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Decode counts from rank-ordered quantile positions.

    For each gene, samples a count bag, sorts, and assigns to spots by
    argsort(quantiles[:, gene]) — then discards the bag before moving to
    the next gene.  Avoids the full (n_spots × n_genes) raw_counts
    intermediate that would double peak memory for large datasets.
    """
    if "model_selected" not in model_params or "marginal_param1" not in model_params:
        raise ValueError("model_params must contain 'model_selected' and 'marginal_param1'.")

    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim != 2:
        raise ValueError("quantiles must be a 2D array.")

    n_spots, n_genes = quantiles.shape
    if n_spots == 0:
        return np.zeros((0, n_genes), dtype=np.int32)

    rng = np.random if random_seed is None else np.random.default_rng(int(random_seed))
    boundary = _boundary_per_gene(reference_X, n_genes, model_params, boundary_multiplier)

    if spot_weights is not None:
        weights = np.asarray(spot_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != n_spots:
            raise ValueError(f"spot_weights length {weights.shape[0]} does not match n_spots {n_spots}.")
        weights = np.clip(weights, 1e-8, None)
        weights = weights / np.sum(weights)
    else:
        weights = None

    q_positions = np.linspace(0.0, 1.0, n_spots, dtype=np.float64)
    final_counts = np.zeros((n_spots, n_genes), dtype=np.float32)

    iterator = range(n_genes)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator)

    for gene_idx in iterator:
        model_type, pi0, r, mu = _model_type_and_params(model_params, gene_idx)
        gene_bag = _sample_gene_counts(model_type, pi0, r, mu, n_spots, rng).astype(np.float32)
        gene_boundary = boundary[gene_idx]
        if np.isfinite(gene_boundary):
            gene_bag = np.minimum(gene_bag, gene_boundary)
        gene_bag.sort()

        spot_rank_order = np.argsort(quantiles[:, gene_idx])
        if weights is None:
            final_counts[spot_rank_order, gene_idx] = gene_bag
        else:
            w_ordered = weights[spot_rank_order]
            cum_w = np.cumsum(w_ordered)
            if cum_w[-1] <= 0:
                final_counts[spot_rank_order, gene_idx] = gene_bag
            else:
                q_w = (cum_w - 0.5 * w_ordered) / cum_w[-1]
                final_counts[spot_rank_order, gene_idx] = np.interp(q_w, q_positions, gene_bag)

    return np.rint(final_counts).astype(np.int32)
