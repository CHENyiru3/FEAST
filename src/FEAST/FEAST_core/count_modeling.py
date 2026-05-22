"""Shared count-model moment conversion for FEAST simulators."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


STAT_COLUMNS = ("mean", "variance", "zero_prop")


def normalize_stats_frame(
    stats_df: pd.DataFrame,
    gene_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a validated gene-indexed statistics frame."""
    if "gene_id" in stats_df.columns:
        stats_df = stats_df.set_index("gene_id")
    frame = stats_df.copy()
    frame.index = frame.index.astype(str)
    missing = set(STAT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"stats_df is missing required columns: {sorted(missing)}")
    if gene_names is not None:
        genes = [str(gene) for gene in gene_names]
        missing_genes = [gene for gene in genes if gene not in frame.index]
        if missing_genes:
            raise ValueError(f"stats_df is missing genes: {missing_genes}")
        frame = frame.loc[genes, list(STAT_COLUMNS)].copy()
    else:
        frame = frame.loc[:, list(STAT_COLUMNS)].copy()
    if frame.isna().any().any():
        raise ValueError("stats_df contains missing values.")
    frame["mean"] = np.clip(frame["mean"].astype(float), 1e-8, None)
    frame["variance"] = np.clip(frame["variance"].astype(float), 1e-8, None)
    frame["zero_prop"] = np.clip(frame["zero_prop"].astype(float), 0.0, 0.99)
    return frame


def theoretical_stats_for_model(model_type: str, params: Iterable[float]) -> np.ndarray:
    """Return theoretical mean, variance, and zero proportion for FEAST params."""
    pi0, r, mu = [float(v) for v in params]
    pi0 = float(np.clip(pi0, 0.0, 0.99))
    r = max(r, 1e-8)
    mu = max(mu, 1e-8)
    if model_type == "Poisson":
        return np.array([mu, mu, np.exp(-mu)], dtype=float)
    if model_type == "NB":
        p0 = (r / (r + mu)) ** r
        return np.array([mu, mu + mu * mu / r, p0], dtype=float)
    if model_type == "ZIP":
        mean = (1.0 - pi0) * mu
        variance = (1.0 - pi0) * mu * (1.0 + pi0 * mu)
        zero_prop = pi0 + (1.0 - pi0) * np.exp(-mu)
        return np.array([mean, variance, zero_prop], dtype=float)
    if model_type == "ZINB":
        p0_active = (r / (r + mu)) ** r
        mean = (1.0 - pi0) * mu
        variance = (1.0 - pi0) * (mu + mu * mu / r + pi0 * mu * mu)
        zero_prop = pi0 + (1.0 - pi0) * p0_active
        return np.array([mean, variance, zero_prop], dtype=float)
    return theoretical_stats_for_model("Poisson", [0.0, 1.0, mu])


def _moment_error(target: np.ndarray, observed: np.ndarray) -> float:
    target = np.clip(np.asarray(target, dtype=float), 1e-10, None)
    observed = np.clip(np.asarray(observed, dtype=float), 1e-10, None)
    return float(np.mean((np.log10(observed) - np.log10(target)) ** 2))


def _optimize_zero_inflated(target: np.ndarray, model_type: str) -> tuple[list[float], float]:
    mean, variance, zero_prop = [float(v) for v in target]
    initial_pi = float(np.clip(zero_prop, 1e-6, 0.99))
    initial_mu = max(mean / max(1.0 - initial_pi, 1e-8), 1e-8)
    initial_r = max(initial_mu * initial_mu / max(variance - initial_mu, 1e-8), 1e-6)
    if model_type == "ZIP":
        x0 = np.array([initial_pi, initial_mu], dtype=float)
        bounds = [(0.0, 0.99), (1e-8, None)]
    else:
        x0 = np.array([initial_pi, initial_mu, initial_r], dtype=float)
        bounds = [(0.0, 0.99), (1e-8, None), (1e-6, None)]

    def objective(values):
        if model_type == "ZIP":
            params = [values[0], 1.0, values[1]]
        else:
            params = [values[0], values[2], values[1]]
        return _moment_error(target, theoretical_stats_for_model(model_type, params))

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    values = result.x if result.x is not None and np.all(np.isfinite(result.x)) else x0
    if model_type == "ZIP":
        params = [float(values[0]), 1.0, float(values[1])]
    else:
        params = [float(values[0]), float(values[2]), float(values[1])]
    return params, float(objective(values))


def _candidate_models(mean: float, variance: float, zero_prop: float) -> list[tuple[str, list[float], float]]:
    target = np.array([mean, variance, zero_prop], dtype=float)
    candidates: list[tuple[str, list[float], float]] = []

    poisson_params = [0.0, 1.0, max(mean, 1e-8)]
    candidates.append(("Poisson", poisson_params, _moment_error(target, theoretical_stats_for_model("Poisson", poisson_params))))

    if variance > mean + 1e-8:
        r = max(mean * mean / max(variance - mean, 1e-8), 1e-6)
        nb_params = [0.0, r, max(mean, 1e-8)]
        candidates.append(("NB", nb_params, _moment_error(target, theoretical_stats_for_model("NB", nb_params))))

    for model_type in ("ZIP", "ZINB"):
        params, error = _optimize_zero_inflated(target, model_type)
        candidates.append((model_type, params, error))

    return candidates


def stats_frame_to_model_params(
    stats_df: pd.DataFrame,
    gene_names: Optional[Iterable[str]] = None,
) -> dict:
    """Convert gene summary statistics into FEAST count-model parameters."""
    stats = normalize_stats_frame(stats_df, gene_names)
    model_selected: list[str] = []
    marginal_param1: list[list[float]] = []
    moment_errors: list[float] = []

    for _, record in stats.iterrows():
        candidates = _candidate_models(
            float(record["mean"]),
            float(record["variance"]),
            float(record["zero_prop"]),
        )
        model_type, params, error = min(candidates, key=lambda item: item[2])
        model_selected.append(model_type)
        marginal_param1.append([float(params[0]), float(params[1]), float(params[2])])
        moment_errors.append(float(error))

    model_counts = Counter(model_selected)
    errors = np.asarray(moment_errors, dtype=float)
    diagnostics = {
        "model_selection_counts": {str(k): int(v) for k, v in sorted(model_counts.items())},
        "mean_log_moment_error": float(errors.mean()) if errors.size else 0.0,
        "max_log_moment_error": float(errors.max()) if errors.size else 0.0,
        "high_error_gene_count": int(np.sum(errors > 0.05)) if errors.size else 0,
    }
    return {
        "genes": {idx: gene for idx, gene in enumerate(stats.index.astype(str))},
        "model_selected": model_selected,
        "marginal_param1": marginal_param1,
        "model_moment_diagnostics": diagnostics,
    }
