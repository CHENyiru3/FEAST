from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from scipy.stats import norm

from .core import SliceBlueprint
from .pattern import evaluate_motif


QUANTILE_FIELD_MODES = {
    "auto",
    "latent_program",
    "latent_reference",
    "explicit_quantile",
    "iid",
}
RANK_SCOPES = {"auto", "volume", "slice", "domain", "domain_slice", "scaffold"}
TIE_POLICIES = {"stable_ordinal", "average", "seeded_jitter"}
TARGET_PARAMETER_MODES = {
    "reference_weighted_log",
    "nearest_reference",
    "user_supplied",
    "depth_gp_log",
}
REFERENCE_CONFLICT_POLICIES = {"average", "highest_weight"}


@dataclass
class QuantileFieldConfig:
    mode: str = "auto"
    rank_scope: str = "auto"
    target_parameter_mode: str = "reference_weighted_log"
    tie_policy: str = "stable_ordinal"
    latent_clip_eps: float = 1e-6
    tie_jitter_scale: float = 1e-9
    min_rank_scope_size: int = 20
    gene_chunk_size: int = 512
    store_latent_scores: bool = False
    store_quantiles: Any = "auto"
    max_stored_quantile_elements: int = 50_000_000
    reference_conflict_policy: str = "average"
    program_noise_scale: float = 0.0
    program_normalization: str = "zscore"
    posterior_sigma0: float = 1e-3
    posterior_alpha_z: float = 1.0
    posterior_alpha_cost: float = 1.0
    posterior_alpha_transport: float = 1.0
    posterior_alpha_quality: float = 0.0
    posterior_lambda0: float = 0.0
    posterior_lambda_s: float = 0.0
    posterior_graph_neighbors: int = 6
    posterior_variance_floor: float = 1e-8
    parameter_gp_length_scale: float = 1.0
    parameter_gp_noise: float = 1e-3


def validate_quantile_field_config(config: QuantileFieldConfig) -> QuantileFieldConfig:
    if str(config.mode) not in QUANTILE_FIELD_MODES:
        raise ValueError(f"quantile_field mode must be one of {sorted(QUANTILE_FIELD_MODES)}.")
    if str(config.rank_scope) not in RANK_SCOPES:
        raise ValueError(f"rank_scope must be one of {sorted(RANK_SCOPES)}.")
    if str(config.target_parameter_mode) not in TARGET_PARAMETER_MODES:
        raise ValueError(f"target_parameter_mode must be one of {sorted(TARGET_PARAMETER_MODES)}.")
    if str(config.tie_policy) not in TIE_POLICIES:
        raise ValueError(f"tie_policy must be one of {sorted(TIE_POLICIES)}.")
    if str(config.reference_conflict_policy) not in REFERENCE_CONFLICT_POLICIES:
        raise ValueError(f"reference_conflict_policy must be one of {sorted(REFERENCE_CONFLICT_POLICIES)}.")
    if str(config.program_normalization) not in {"zscore", "none"}:
        raise ValueError("program_normalization must be 'zscore' or 'none'.")
    if float(config.latent_clip_eps) <= 0.0 or float(config.latent_clip_eps) >= 0.5:
        raise ValueError("latent_clip_eps must be in (0, 0.5).")
    if int(config.min_rank_scope_size) < 1:
        raise ValueError("min_rank_scope_size must be positive.")
    if int(config.gene_chunk_size) < 1:
        raise ValueError("gene_chunk_size must be positive.")
    if float(config.posterior_sigma0) <= 0.0:
        raise ValueError("posterior_sigma0 must be positive.")
    for name in (
        "posterior_alpha_z",
        "posterior_alpha_cost",
        "posterior_alpha_transport",
        "posterior_alpha_quality",
        "posterior_lambda0",
        "posterior_lambda_s",
    ):
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if int(config.posterior_graph_neighbors) < 1:
        raise ValueError("posterior_graph_neighbors must be positive.")
    if float(config.posterior_variance_floor) <= 0.0:
        raise ValueError("posterior_variance_floor must be positive.")
    if float(config.parameter_gp_length_scale) <= 0.0:
        raise ValueError("parameter_gp_length_scale must be positive.")
    if float(config.parameter_gp_noise) < 0.0:
        raise ValueError("parameter_gp_noise must be non-negative.")
    return config


def midpoint_rank_normalize(
    scores: np.ndarray,
    *,
    tie_policy: str = "stable_ordinal",
    clip_eps: float = 1e-6,
    random_seed: Optional[int] = None,
    tie_jitter_scale: float = 1e-9,
) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("scores must be a 1D or 2D array.")
    n_spots, n_genes = arr.shape
    out = np.zeros((n_spots, n_genes), dtype=np.float64)
    if n_spots == 0:
        return out[:, 0].astype(np.float32) if squeeze else out.astype(np.float32)
    if n_spots == 1:
        out.fill(0.5)
        return out[:, 0].astype(np.float32) if squeeze else out.astype(np.float32)

    policy = str(tie_policy)
    if policy not in TIE_POLICIES:
        raise ValueError(f"tie_policy must be one of {sorted(TIE_POLICIES)}.")
    rng = np.random.default_rng(None if random_seed is None else int(random_seed))
    positions = (np.arange(n_spots, dtype=np.float64) + 0.5) / float(n_spots)

    for gene_idx in range(n_genes):
        values = arr[:, gene_idx]
        if policy == "seeded_jitter":
            values = values + rng.normal(0.0, float(tie_jitter_scale), size=n_spots)
            order = np.argsort(values, kind="mergesort")
            out[order, gene_idx] = positions
        elif policy == "stable_ordinal":
            order = np.argsort(values, kind="mergesort")
            out[order, gene_idx] = positions
        else:
            order = np.argsort(values, kind="mergesort")
            ordered = values[order]
            ranks = np.zeros(n_spots, dtype=np.float64)
            start = 0
            while start < n_spots:
                end = start + 1
                while end < n_spots and ordered[end] == ordered[start]:
                    end += 1
                avg_position = (0.5 * (start + end - 1) + 0.5) / float(n_spots)
                ranks[order[start:end]] = avg_position
                start = end
            out[:, gene_idx] = ranks

    out = np.clip(out, float(clip_eps), 1.0 - float(clip_eps))
    return out[:, 0].astype(np.float32) if squeeze else out.astype(np.float32)


def quantiles_to_normal_scores(quantiles: np.ndarray, *, clip_eps: float = 1e-6) -> np.ndarray:
    q = np.asarray(quantiles, dtype=np.float64)
    q = np.clip(q, float(clip_eps), 1.0 - float(clip_eps))
    return norm.ppf(q).astype(np.float32, copy=False)


def transport_latent_scores(plan: np.ndarray, latent_scores: np.ndarray) -> np.ndarray:
    plan = np.asarray(plan, dtype=np.float64)
    scores = np.asarray(latent_scores, dtype=np.float64)
    if plan.shape[0] != scores.shape[0]:
        raise ValueError("transport plan source dimension does not match source latent scores.")
    if plan.shape[1] == 0:
        return np.zeros((0, scores.shape[1]), dtype=np.float32)
    column_mass = plan.sum(axis=0, keepdims=True)
    safe_mass = np.where(column_mass > 1e-12, column_mass, 1.0)
    weights = plan / safe_mass
    return (weights.T @ scores).astype(np.float32, copy=False)


def transport_plan_target_uncertainty(
    plan: np.ndarray,
    cost_matrix: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
) -> Dict[str, np.ndarray]:
    plan = np.asarray(plan, dtype=np.float64)
    cost = np.asarray(cost_matrix, dtype=np.float64)
    source = np.asarray(source_coords, dtype=np.float64)
    target = np.asarray(target_coords, dtype=np.float64)
    if plan.shape != cost.shape:
        raise ValueError("transport plan and cost matrix must have matching shapes.")
    if plan.shape[0] != source.shape[0]:
        raise ValueError("source coordinate rows must match transport source dimension.")
    if plan.shape[1] != target.shape[0]:
        raise ValueError("target coordinate rows must match transport target dimension.")
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source_coords and target_coords must be 2D arrays with the same dimensionality.")
    n_target = int(plan.shape[1])
    if n_target == 0:
        empty = np.zeros(0, dtype=np.float32)
        return {"target_cost": empty, "target_variance": empty, "target_mass": empty}

    column_mass = plan.sum(axis=0)
    safe_mass = np.where(column_mass > 1e-12, column_mass, 1.0)
    weights = plan / safe_mass[None, :]
    target_cost = np.sum(weights * cost, axis=0)
    deltas = source[:, None, :] - target[None, :, :]
    squared_distance = np.sum(deltas * deltas, axis=2)
    mean_squared_distance = np.sum(weights * squared_distance, axis=0)
    barycenter = weights.T @ source
    barycenter_distance = np.sum((barycenter - target) ** 2, axis=1)
    target_variance = np.maximum(mean_squared_distance - barycenter_distance, 0.0)
    return {
        "target_cost": target_cost.astype(np.float32, copy=False),
        "target_variance": target_variance.astype(np.float32, copy=False),
        "target_mass": column_mass.astype(np.float32, copy=False),
    }


def posterior_observation_variance(
    *,
    target_coords: np.ndarray,
    reference_coords: np.ndarray,
    transport_cost: np.ndarray,
    transport_variance: np.ndarray,
    sigma0: float,
    alpha_z: float,
    alpha_cost: float,
    alpha_transport: float,
    alpha_quality: float = 0.0,
    quality_noise: float = 0.0,
    variance_floor: float = 1e-8,
) -> tuple[np.ndarray, Dict[str, float]]:
    target = np.asarray(target_coords, dtype=np.float64)
    reference = np.asarray(reference_coords, dtype=np.float64)
    cost = np.asarray(transport_cost, dtype=np.float64).reshape(-1)
    dispersion = np.asarray(transport_variance, dtype=np.float64).reshape(-1)
    if target.ndim != 2:
        raise ValueError("target_coords must be a 2D array.")
    if reference.ndim != 2:
        raise ValueError("reference_coords must be a 2D array.")
    if cost.shape[0] != target.shape[0] or dispersion.shape[0] != target.shape[0]:
        raise ValueError("transport uncertainty vectors must have one value per target spot.")

    if target.shape[1] >= 3 and reference.shape[1] >= 3 and reference.shape[0] > 0:
        reference_z = float(np.mean(reference[:, 2]))
        z_delta2 = (target[:, 2] - reference_z) ** 2
    else:
        reference_z = 0.0
        z_delta2 = np.zeros(target.shape[0], dtype=np.float64)

    variance = (
        float(sigma0) ** 2
        + float(alpha_z) * z_delta2
        + float(alpha_cost) * np.clip(cost, 0.0, None)
        + float(alpha_transport) * np.clip(dispersion, 0.0, None)
        + float(alpha_quality) * max(float(quality_noise), 0.0)
    )
    variance = np.clip(variance, float(variance_floor), None)
    metadata = {
        "variance_min": float(np.min(variance)) if variance.size else 0.0,
        "variance_median": float(np.median(variance)) if variance.size else 0.0,
        "variance_max": float(np.max(variance)) if variance.size else 0.0,
        "transport_cost_mean": float(np.mean(cost)) if cost.size else 0.0,
        "transport_variance_mean": float(np.mean(dispersion)) if dispersion.size else 0.0,
        "z_distance_mean": float(np.mean(np.sqrt(z_delta2))) if z_delta2.size else 0.0,
        "reference_z": float(reference_z),
    }
    return variance.astype(np.float32, copy=False), metadata


def build_target_graph_laplacian(
    coordinates: np.ndarray,
    *,
    labels: Optional[Sequence[Any]] = None,
    n_neighbors: int = 6,
) -> tuple[sparse.csr_matrix, Dict[str, Any]]:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("coordinates must be a 2D array.")
    n_spots = int(coords.shape[0])
    if n_spots == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float64), {
            "n_spots": 0,
            "n_edges": 0,
            "n_neighbors": int(n_neighbors),
            "same_label_only": labels is not None,
        }
    if n_spots == 1:
        return sparse.csr_matrix((1, 1), dtype=np.float64), {
            "n_spots": 1,
            "n_edges": 0,
            "n_neighbors": int(n_neighbors),
            "same_label_only": labels is not None,
        }

    labels_arr = None
    if labels is not None:
        labels_arr = np.asarray(labels).astype(str)
        if labels_arr.shape[0] != n_spots:
            raise ValueError("labels length must match coordinate rows.")

    k = min(max(1, int(n_neighbors)), n_spots - 1)
    distances, indices = cKDTree(coords).query(coords, k=k + 1)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=int)
    if indices.ndim == 1:
        indices = indices[:, None]
        distances = distances[:, None]

    finite_distances = distances[:, 1:][np.isfinite(distances[:, 1:])]
    scale = float(np.median(finite_distances)) if finite_distances.size else 1.0
    scale = max(scale, 1e-8)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row_idx in range(n_spots):
        for distance, col_idx in zip(distances[row_idx, 1:], indices[row_idx, 1:]):
            if col_idx < 0 or col_idx == row_idx or not np.isfinite(distance):
                continue
            if labels_arr is not None and labels_arr[row_idx] != labels_arr[int(col_idx)]:
                continue
            weight = float(np.exp(-0.5 * (float(distance) / scale) ** 2))
            rows.append(row_idx)
            cols.append(int(col_idx))
            data.append(weight)

    if not data:
        graph = sparse.csr_matrix((n_spots, n_spots), dtype=np.float64)
    else:
        graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_spots, n_spots), dtype=np.float64).tocsr()
        graph = graph.maximum(graph.T)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree, format="csr") - graph
    metadata = {
        "n_spots": n_spots,
        "n_edges": int(graph.nnz // 2),
        "n_neighbors": int(k),
        "same_label_only": labels is not None,
    }
    return laplacian.tocsr(), metadata


def infer_latent_posterior(
    reference_fields: Sequence[np.ndarray],
    observation_variances: Sequence[np.ndarray],
    *,
    prior_mean: Optional[np.ndarray] = None,
    coordinates: Optional[np.ndarray] = None,
    labels: Optional[Sequence[Any]] = None,
    lambda0: float = 0.0,
    lambda_s: float = 0.0,
    graph_neighbors: int = 6,
    variance_floor: float = 1e-8,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if len(reference_fields) != len(observation_variances):
        raise ValueError("reference_fields and observation_variances must have the same length.")
    if not reference_fields and prior_mean is None:
        raise ValueError("At least one reference field or a prior_mean is required.")

    if reference_fields:
        first = np.asarray(reference_fields[0], dtype=np.float64)
    else:
        first = np.asarray(prior_mean, dtype=np.float64)
    if first.ndim != 2:
        raise ValueError("latent fields must be 2D arrays.")
    n_spots, n_genes = first.shape
    rhs = np.zeros((n_spots, n_genes), dtype=np.float64)
    precision_sum = np.zeros(n_spots, dtype=np.float64)
    variance_summaries: list[Dict[str, float]] = []

    for field, variance in zip(reference_fields, observation_variances):
        y = np.asarray(field, dtype=np.float64)
        if y.shape != (n_spots, n_genes):
            raise ValueError("All reference fields must share the same shape.")
        var = np.asarray(variance, dtype=np.float64).reshape(-1)
        if var.shape[0] != n_spots:
            raise ValueError("Each observation variance vector must have one value per target spot.")
        var = np.clip(var, float(variance_floor), None)
        precision = 1.0 / var
        rhs += precision[:, None] * y
        precision_sum += precision
        variance_summaries.append(
            {
                "variance_min": float(np.min(var)) if var.size else 0.0,
                "variance_median": float(np.median(var)) if var.size else 0.0,
                "variance_max": float(np.max(var)) if var.size else 0.0,
            }
        )

    effective_lambda0 = float(lambda0)
    if not reference_fields and prior_mean is not None and effective_lambda0 <= 0.0:
        effective_lambda0 = 1.0
    prior_used = prior_mean is not None and effective_lambda0 > 0.0
    if effective_lambda0 > 0.0:
        if prior_mean is None:
            prior = np.zeros((n_spots, n_genes), dtype=np.float64)
        else:
            prior = np.asarray(prior_mean, dtype=np.float64)
            if prior.shape != (n_spots, n_genes):
                raise ValueError("prior_mean must match latent field shape.")
        rhs += effective_lambda0 * prior
        precision_sum += effective_lambda0

    graph_metadata: Dict[str, Any] = {
        "enabled": bool(float(lambda_s) > 0.0),
        "lambda_s": float(lambda_s),
        "n_neighbors": int(graph_neighbors),
    }
    if float(lambda_s) > 0.0:
        if coordinates is None:
            raise ValueError("coordinates are required when lambda_s is positive.")
        laplacian, lap_meta = build_target_graph_laplacian(
            coordinates,
            labels=labels,
            n_neighbors=int(graph_neighbors),
        )
        graph_metadata.update(lap_meta)
        system = sparse.diags(np.clip(precision_sum, float(variance_floor), None), format="csr")
        system = system + float(lambda_s) * laplacian
        solved = sparse_linalg.spsolve(system.tocsc(), rhs)
        h = np.asarray(solved, dtype=np.float64)
        if h.ndim == 1:
            h = h[:, None]
    else:
        safe_precision = np.clip(precision_sum, float(variance_floor), None)
        h = rhs / safe_precision[:, None]

    metadata = {
        "n_references": int(len(reference_fields)),
        "prior_mean_used": bool(prior_used),
        "lambda0": float(lambda0),
        "effective_lambda0": float(effective_lambda0),
        "lambda_s": float(lambda_s),
        "precision_min": float(np.min(precision_sum)) if precision_sum.size else 0.0,
        "precision_median": float(np.median(precision_sum)) if precision_sum.size else 0.0,
        "precision_max": float(np.max(precision_sum)) if precision_sum.size else 0.0,
        "variance_floor": float(variance_floor),
        "reference_variance_summaries": variance_summaries,
        "graph": graph_metadata,
    }
    return h.astype(np.float32, copy=False), metadata


def reference_conflict_score(parts: Sequence[np.ndarray], weights: Sequence[float]) -> float:
    if len(parts) < 2:
        return 0.0
    flat_parts = [np.asarray(part, dtype=np.float64).reshape(-1) for part in parts]
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights_arr = np.clip(weights_arr, 0.0, None)
    if float(weights_arr.sum()) <= 0.0:
        weights_arr = np.ones(len(flat_parts), dtype=np.float64)
    weights_arr = weights_arr / float(weights_arr.sum())
    combined = np.zeros_like(flat_parts[0])
    for weight, flat in zip(weights_arr, flat_parts):
        combined += float(weight) * flat
    part_var = float(np.sum([w * np.var(flat) for w, flat in zip(weights_arr, flat_parts)]))
    if part_var <= 1e-12:
        return 0.0
    combined_var = float(np.var(combined))
    return float(np.clip(1.0 - (combined_var / part_var), 0.0, 1.0))


def _z_groups(coords: np.ndarray) -> np.ndarray:
    if coords.shape[1] < 3:
        return np.zeros(coords.shape[0], dtype=object)
    return np.asarray([f"{float(z):.12g}" for z in coords[:, 2]], dtype=object)


def resolve_auto_rank_scope(
    *,
    requested_scope: str,
    coordinate_dim: int,
    domain_specific: bool,
    stack_like: bool,
) -> str:
    scope = str(requested_scope)
    if scope != "auto":
        return scope
    if domain_specific and stack_like:
        return "domain_slice"
    if domain_specific:
        return "domain"
    if stack_like:
        return "slice"
    return "scaffold" if int(coordinate_dim) <= 2 else "volume"


def _scope_groups(scope: str, labels: np.ndarray, coords: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(str)
    if scope in {"volume", "scaffold"}:
        return np.full(labels.shape[0], "all", dtype=object)
    if scope == "domain":
        return labels.astype(object)
    z_groups = _z_groups(coords)
    if scope == "slice":
        return z_groups
    if scope == "domain_slice":
        return np.asarray([f"{z}|{label}" for z, label in zip(z_groups, labels)], dtype=object)
    raise ValueError(f"Unsupported rank scope '{scope}'.")


def _fallback_scope(scope: str) -> Optional[str]:
    if scope == "domain_slice":
        return "domain"
    if scope == "domain":
        return "volume"
    if scope == "slice":
        return "volume"
    return None


def rank_normalize_by_scope(
    scores: np.ndarray,
    *,
    labels: Sequence[Any],
    coordinates: np.ndarray,
    rank_scope: str,
    tie_policy: str,
    clip_eps: float,
    random_seed: Optional[int],
    tie_jitter_scale: float,
    min_rank_scope_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("scores must be a 2D array.")
    coords = np.asarray(coordinates, dtype=float)
    labels_arr = np.asarray(labels).astype(str)
    if labels_arr.shape[0] != arr.shape[0]:
        raise ValueError("labels length must match scores rows.")
    if coords.shape[0] != arr.shape[0]:
        raise ValueError("coordinates row count must match scores rows.")

    requested = str(rank_scope)
    resolved_counts: Dict[str, int] = {}
    fallbacks: list[Dict[str, Any]] = []
    out = np.zeros(arr.shape, dtype=np.float32)
    pending_indices = np.arange(arr.shape[0], dtype=int)
    current_scope = requested
    rng_seed = None if random_seed is None else int(random_seed)

    while pending_indices.size > 0:
        groups = _scope_groups(current_scope, labels_arr[pending_indices], coords[pending_indices])
        next_pending: list[int] = []
        for group in np.unique(groups):
            local = pending_indices[groups == group]
            if local.size < int(min_rank_scope_size):
                next_pending.extend(local.tolist())
                continue
            resolved_counts[current_scope] = resolved_counts.get(current_scope, 0) + 1
            out[local, :] = midpoint_rank_normalize(
                arr[local, :],
                tie_policy=tie_policy,
                clip_eps=clip_eps,
                random_seed=None if rng_seed is None else rng_seed + int(local[0]),
                tie_jitter_scale=tie_jitter_scale,
            )
        if not next_pending:
            break
        fallback = _fallback_scope(current_scope)
        if fallback is None:
            local = np.asarray(next_pending, dtype=int)
            out[local, :] = midpoint_rank_normalize(
                arr[local, :],
                tie_policy=tie_policy,
                clip_eps=clip_eps,
                random_seed=rng_seed,
                tie_jitter_scale=tie_jitter_scale,
            )
            fallbacks.append(
                {
                    "from": current_scope,
                    "to": current_scope,
                    "spots": int(local.size),
                    "reason": "minimum rank scope size unavailable",
                }
            )
            break
        fallbacks.append(
            {
                "from": current_scope,
                "to": fallback,
                "spots": int(len(next_pending)),
                "reason": "minimum rank scope size unavailable",
            }
        )
        pending_indices = np.asarray(next_pending, dtype=int)
        current_scope = fallback

    metadata = {
        "requested_rank_scope": requested,
        "resolved_rank_scope": current_scope if fallbacks else requested,
        "resolved_scope_counts": resolved_counts,
        "fallbacks": fallbacks,
        "min_rank_scope_size": int(min_rank_scope_size),
    }
    return out.astype(np.float32, copy=False), metadata


def _program_name(program: Mapping[str, Any], idx: int) -> str:
    return str(program.get("name", f"program_{idx:03d}"))


def build_spatial_program_matrix(
    blueprint: SliceBlueprint,
    program_spec: Sequence[Mapping[str, Any]],
    *,
    label_key: str,
    random_seed: int,
    boundary_softness: float,
    normalization: str = "zscore",
) -> Tuple[np.ndarray, list[Dict[str, Any]]]:
    if not program_spec:
        raise ValueError("program_spec must contain at least one spatial program.")
    columns = []
    provenance: list[Dict[str, Any]] = []
    for idx, program in enumerate(program_spec):
        values = evaluate_motif(
            blueprint,
            program,
            label_key=label_key,
            random_seed=int(program.get("seed", random_seed + idx)),
            boundary_softness=boundary_softness,
        ).astype(np.float64, copy=False)
        mean = float(values.mean()) if values.size else 0.0
        sd = float(values.std()) if values.size else 0.0
        if str(normalization) == "zscore":
            values = (values - mean) / max(sd, 1e-8)
        columns.append(values)
        provenance.append(
            {
                "name": _program_name(program, idx),
                "kind": str(program.get("kind", "")),
                "normalization": str(normalization),
                "raw_mean": mean,
                "raw_sd": sd,
            }
        )
    return np.column_stack(columns).astype(np.float32, copy=False), provenance


def pattern_spec_to_program_spec(
    pattern_spec: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[Dict[str, Any]], pd.DataFrame]:
    programs: list[Dict[str, Any]] = []
    loadings: Dict[str, Dict[str, float]] = {}
    seen: Dict[tuple, str] = {}
    for gene_name, motifs in pattern_spec.items():
        gene = str(gene_name)
        loadings.setdefault(gene, {})
        for motif in motifs:
            motif_payload = dict(motif)
            key = tuple(sorted((str(k), repr(v)) for k, v in motif_payload.items() if k != "weight"))
            if key not in seen:
                program_name = f"compat_{len(programs):03d}"
                seen[key] = program_name
                program = dict(motif_payload)
                program["name"] = program_name
                program["weight"] = 1.0
                programs.append(program)
            loadings[gene][seen[key]] = loadings[gene].get(seen[key], 0.0) + float(motif_payload.get("weight", 1.0))
    if not programs:
        return [], pd.DataFrame()
    frame = pd.DataFrame(0.0, index=sorted(loadings), columns=[program["name"] for program in programs])
    for gene, gene_loadings in loadings.items():
        for program_name, value in gene_loadings.items():
            frame.loc[gene, program_name] = value
    return programs, frame


def align_gene_loadings(
    gene_loadings: Any,
    *,
    gene_names: Sequence[str],
    program_names: Sequence[str],
) -> np.ndarray:
    genes = [str(gene) for gene in gene_names]
    programs = [str(program) for program in program_names]
    if isinstance(gene_loadings, pd.DataFrame):
        frame = gene_loadings.copy()
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
    elif isinstance(gene_loadings, Mapping):
        frame = pd.DataFrame.from_dict(gene_loadings, orient="index")
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
    else:
        arr = np.asarray(gene_loadings, dtype=np.float32)
        if arr.shape != (len(genes), len(programs)):
            raise ValueError("gene_loadings array must have shape (n_genes, n_programs).")
        return arr.astype(np.float32, copy=False)
    for program in programs:
        if program not in frame.columns:
            frame[program] = 0.0
    return frame.reindex(index=genes, columns=programs, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float32)


def build_latent_program_scores(
    blueprint: SliceBlueprint,
    gene_names: Sequence[str],
    *,
    program_spec: Sequence[Mapping[str, Any]],
    gene_loadings: Any,
    label_key: str,
    random_seed: int,
    boundary_softness: float,
    normalization: str,
    program_noise_scale: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    B, program_provenance = build_spatial_program_matrix(
        blueprint,
        program_spec,
        label_key=label_key,
        random_seed=random_seed,
        boundary_softness=boundary_softness,
        normalization=normalization,
    )
    program_names = [item["name"] for item in program_provenance]
    A = align_gene_loadings(gene_loadings, gene_names=gene_names, program_names=program_names)
    scores = B @ A.T
    if float(program_noise_scale) > 0.0:
        rng = np.random.default_rng(int(random_seed))
        scores = scores + rng.normal(0.0, float(program_noise_scale), size=scores.shape)
    metadata = {
        "programs": program_provenance,
        "n_programs": int(B.shape[1]),
        "program_normalization": str(normalization),
        "program_noise_scale": float(program_noise_scale),
    }
    return scores.astype(np.float32, copy=False), metadata


def should_store_quantiles(store_quantiles: Any, n_elements: int, max_elements: int) -> bool:
    if isinstance(store_quantiles, str) and store_quantiles.lower() == "auto":
        return int(n_elements) <= int(max_elements)
    return bool(store_quantiles)


def weighted_stats_log_space(
    frames: Sequence[pd.DataFrame],
    weights: Sequence[float],
    gene_names: Sequence[str],
) -> pd.DataFrame:
    if not frames:
        raise ValueError("frames must contain at least one stats frame.")
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights_arr = np.clip(weights_arr, 0.0, None)
    total = float(weights_arr.sum())
    if total <= 0.0:
        weights_arr = np.full(len(frames), 1.0 / len(frames), dtype=np.float64)
    else:
        weights_arr = weights_arr / total

    genes = [str(gene) for gene in gene_names]
    eps = 1e-8
    theta_mean = np.zeros(len(genes), dtype=np.float64)
    theta_var = np.zeros(len(genes), dtype=np.float64)
    theta_zero = np.zeros(len(genes), dtype=np.float64)
    for weight, frame in zip(weights_arr, frames):
        aligned = frame.copy()
        aligned.index = aligned.index.astype(str)
        aligned = aligned.loc[genes, ["mean", "variance", "zero_prop"]]
        theta_mean += float(weight) * np.log(np.clip(aligned["mean"].to_numpy(dtype=float), eps, None))
        theta_var += float(weight) * np.log(np.clip(aligned["variance"].to_numpy(dtype=float), eps, None))
        zero = np.clip(aligned["zero_prop"].to_numpy(dtype=float), eps, 1.0 - eps)
        theta_zero += float(weight) * logit(zero)
    out = pd.DataFrame(index=genes)
    out["mean"] = np.clip(np.exp(theta_mean), eps, None)
    out["variance"] = np.clip(np.exp(theta_var), eps, None)
    out["zero_prop"] = np.clip(expit(theta_zero), 0.0, 0.99)
    return out
