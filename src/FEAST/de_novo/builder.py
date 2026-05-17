from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..FEAST_core.count_decoding import decode_counts_from_quantiles, resolve_decode_method
from .conditional import VirtualSliceGenerationConfig
from .core import SliceBlueprint, load_blueprint
from .pattern import compose_gene_pattern, diffuse_quantile_map


REQUIRED_STATS_COLUMNS = ("mean", "variance", "zero_prop")


def _coerce_gene_names(gene_names: Sequence[str]) -> list[str]:
    names = [str(gene) for gene in gene_names]
    if len(names) == 0:
        raise ValueError("gene_names must contain at least one gene.")
    if len(set(names)) != len(names):
        raise ValueError("gene_names must be unique.")
    return names


def _coerce_stats_frame(frame: pd.DataFrame, gene_names: Sequence[str]) -> pd.DataFrame:
    missing = set(REQUIRED_STATS_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"parameter cloud is missing required columns: {sorted(missing)}")
    aligned = frame.copy()
    aligned.index = aligned.index.astype(str)
    missing_genes = [gene for gene in gene_names if gene not in aligned.index]
    if missing_genes:
        raise ValueError(f"parameter cloud is missing genes: {missing_genes}")
    aligned = aligned.loc[list(gene_names), list(REQUIRED_STATS_COLUMNS)].copy()
    if aligned.isna().any().any():
        raise ValueError("parameter cloud contains missing values.")
    aligned["mean"] = np.clip(aligned["mean"].astype(float), 1e-8, None)
    aligned["variance"] = np.clip(aligned["variance"].astype(float), 1e-8, None)
    aligned["zero_prop"] = np.clip(aligned["zero_prop"].astype(float), 0.0, 0.99)
    return aligned


def _is_stats_mapping(payload: Mapping[str, Any]) -> bool:
    return set(REQUIRED_STATS_COLUMNS) <= set(payload.keys())


def _is_serialized_cloud(payload: Mapping[str, Any]) -> bool:
    return "original_stats" in payload or "full_stats" in payload


def _extract_gene_names_from_cloud(parameter_cloud: Union[pd.DataFrame, Mapping[str, Any]]) -> list[str]:
    if isinstance(parameter_cloud, pd.DataFrame):
        return _coerce_gene_names(parameter_cloud.index.astype(str).tolist())

    if _is_stats_mapping(parameter_cloud):
        raise ValueError(
            "A plain stats mapping requires an explicit gene list. "
            "Use ParameterCloudBuilder or pass a DataFrame indexed by gene names."
        )

    if _is_serialized_cloud(parameter_cloud):
        payload = parameter_cloud.get("original_stats") or parameter_cloud.get("full_stats")
        if not isinstance(payload, Mapping):
            raise ValueError("Serialized parameter cloud stats payload must be a mapping keyed by gene.")
        return _coerce_gene_names(payload.keys())

    if len(parameter_cloud) == 0:
        raise ValueError("parameter_cloud mapping is empty.")
    first = next(iter(parameter_cloud.values()))
    return _extract_gene_names_from_cloud(first)


def _resolve_parameter_cloud_for_label(
    parameter_cloud: Union[pd.DataFrame, Mapping[str, Any]],
    gene_names: Sequence[str],
    label: Optional[str] = None,
) -> pd.DataFrame:
    if isinstance(parameter_cloud, pd.DataFrame):
        return _coerce_stats_frame(parameter_cloud, gene_names)

    if _is_stats_mapping(parameter_cloud):
        return _coerce_stats_frame(pd.DataFrame(parameter_cloud, index=list(gene_names)), gene_names)

    if _is_serialized_cloud(parameter_cloud):
        stats_payload = parameter_cloud.get("original_stats") or parameter_cloud.get("full_stats")
        return _coerce_stats_frame(pd.DataFrame.from_dict(stats_payload, orient="index"), gene_names)

    if label is not None and label in parameter_cloud:
        return _resolve_parameter_cloud_for_label(parameter_cloud[label], gene_names, label=None)

    if "__default__" in parameter_cloud:
        return _resolve_parameter_cloud_for_label(parameter_cloud["__default__"], gene_names, label=None)

    if label is None:
        raise TypeError("Unsupported parameter_cloud input for design generation.")
    raise KeyError(f"parameter_cloud does not contain stats for label '{label}'.")


def _stats_frame_to_model_params(stats: pd.DataFrame) -> dict:
    model_selected = []
    marginal_param1 = []
    for _, row in stats.iterrows():
        mean = max(float(row["mean"]), 1e-8)
        variance = max(float(row["variance"]), 1e-8)
        pi0 = float(np.clip(row["zero_prop"], 0.0, 0.99))
        active_mean = max(mean / max(1.0 - pi0, 1e-8), 1e-8)
        if variance > active_mean + 1e-8:
            r = max(active_mean * active_mean / max(variance - active_mean, 1e-8), 1e-6)
            model_selected.append("ZINB" if pi0 > 1e-8 else "NB")
            marginal_param1.append([pi0, r, active_mean])
        else:
            model_selected.append("ZIP" if pi0 > 1e-8 else "Poisson")
            marginal_param1.append([pi0, 1.0, active_mean])
    return {
        "genes": list(stats.index.astype(str)),
        "model_selected": model_selected,
        "marginal_param1": marginal_param1,
    }


def _ensure_design_blueprint(
    blueprint: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str],
    *,
    label_key: str,
) -> SliceBlueprint:
    loaded = load_blueprint(blueprint)
    if loaded.domain_map is None:
        loaded = SliceBlueprint(
            coordinates=loaded.coordinates,
            coordinate_mode=loaded.coordinate_mode,
            grid_type=loaded.grid_type,
            mask=loaded.mask,
            domain_map=np.full(loaded.n_spots, "global", dtype=object),
            technology=loaded.technology,
            obs=loaded.obs.copy(),
            metadata=dict(loaded.metadata),
        )
    if label_key not in loaded.obs:
        loaded.obs[label_key] = np.asarray(loaded.domain_map).astype(str)
    return loaded


class BlueprintBuilder:
    def __init__(
        self,
        coordinates: Sequence[Sequence[float]],
        coordinate_mode: str = "generic",
        grid_type: str = "generic",
        technology: Optional[str] = None,
    ) -> None:
        coords = np.asarray(coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("coordinates must be a 2D array with at least two columns.")
        self._coordinates = coords[:, :2]
        self._coordinate_mode = str(coordinate_mode)
        self._grid_type = str(grid_type)
        self._technology = technology
        self._mask: Optional[np.ndarray] = None
        self._obs = pd.DataFrame(index=[f"spot_{i}" for i in range(self._coordinates.shape[0])])
        self._metadata: Dict[str, Any] = {}

    @classmethod
    def from_coordinates(cls, coordinates: Sequence[Sequence[float]], **kwargs) -> "BlueprintBuilder":
        return cls(coordinates, **kwargs)

    @classmethod
    def rectangular_grid(
        cls,
        n_rows: int,
        n_cols: int,
        *,
        spacing: Union[float, Sequence[float]] = 1.0,
        origin: Sequence[float] = (0.0, 0.0),
        technology: Optional[str] = None,
    ) -> "BlueprintBuilder":
        if int(n_rows) <= 0 or int(n_cols) <= 0:
            raise ValueError("n_rows and n_cols must be positive.")
        if np.isscalar(spacing):
            dx = dy = float(spacing)
        else:
            dx, dy = [float(item) for item in spacing]
        ox, oy = [float(item) for item in origin]
        xs = np.arange(int(n_cols), dtype=float) * dx + ox
        ys = np.arange(int(n_rows), dtype=float) * dy + oy
        xx, yy = np.meshgrid(xs, ys)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        return cls(coords, coordinate_mode="grid", grid_type="rectangular", technology=technology)

    def set_domains(self, domains: Sequence[Any], *, key: str = "domain") -> "BlueprintBuilder":
        values = np.asarray(domains)
        if values.shape[0] != self._coordinates.shape[0]:
            raise ValueError("domains must contain one value per spot.")
        self._obs[str(key)] = values.astype(str)
        if str(key) != "domain" and "domain" not in self._obs:
            self._obs["domain"] = values.astype(str)
        return self

    def set_mask(self, mask: Sequence[bool]) -> "BlueprintBuilder":
        values = np.asarray(mask, dtype=bool)
        if values.shape[0] != self._coordinates.shape[0]:
            raise ValueError("mask must contain one value per spot.")
        self._mask = values
        return self

    def set_obs_column(self, key: str, values: Sequence[Any]) -> "BlueprintBuilder":
        arr = np.asarray(values)
        if arr.shape[0] != self._coordinates.shape[0]:
            raise ValueError(f"obs column '{key}' must contain one value per spot.")
        self._obs[str(key)] = arr
        return self

    def set_metadata(self, **metadata: Any) -> "BlueprintBuilder":
        self._metadata.update(metadata)
        return self

    def build(self) -> SliceBlueprint:
        domain_map = self._obs["domain"].to_numpy() if "domain" in self._obs else None
        return SliceBlueprint(
            coordinates=self._coordinates.copy(),
            coordinate_mode=self._coordinate_mode,
            grid_type=self._grid_type,
            mask=None if self._mask is None else self._mask.copy(),
            domain_map=domain_map,
            technology=self._technology,
            obs=self._obs.copy(),
            metadata=dict(self._metadata),
        )


class ParameterCloudBuilder:
    def __init__(self, gene_names: Sequence[str]) -> None:
        self._gene_names = _coerce_gene_names(gene_names)
        self._global = pd.DataFrame(index=self._gene_names, columns=list(REQUIRED_STATS_COLUMNS), dtype=float)
        self._label_frames: Dict[str, pd.DataFrame] = {}

    @classmethod
    def from_gene_names(cls, gene_names: Sequence[str]) -> "ParameterCloudBuilder":
        return cls(gene_names)

    def set_all(self, mean: float, variance: float, zero_prop: float) -> "ParameterCloudBuilder":
        self._global.loc[:, "mean"] = float(mean)
        self._global.loc[:, "variance"] = float(variance)
        self._global.loc[:, "zero_prop"] = float(zero_prop)
        return self

    def set_gene(
        self,
        gene_name: str,
        mean: float,
        variance: float,
        zero_prop: float,
        label: Optional[str] = None,
    ) -> "ParameterCloudBuilder":
        gene = str(gene_name)
        if gene not in self._gene_names:
            raise KeyError(f"Unknown gene '{gene}'.")
        frame = self._label_frames.setdefault(str(label), self._global.copy()) if label is not None else self._global
        frame.loc[gene, ["mean", "variance", "zero_prop"]] = [float(mean), float(variance), float(zero_prop)]
        if label is not None:
            self._label_frames[str(label)] = frame
        return self

    def set_stats_frame(self, frame: pd.DataFrame, label: Optional[str] = None) -> "ParameterCloudBuilder":
        normalized = _coerce_stats_frame(frame, self._gene_names)
        if label is None:
            self._global = normalized
        else:
            self._label_frames[str(label)] = normalized
        return self

    def build(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        global_frame = _coerce_stats_frame(self._global, self._gene_names)
        if self._label_frames:
            out = {"__default__": global_frame}
            for label, frame in self._label_frames.items():
                out[label] = _coerce_stats_frame(frame, self._gene_names)
            return out
        return global_frame


def plot_blueprint(
    blueprint: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str],
    *,
    label_key: str = "domain",
    title: Optional[str] = None,
    ax=None,
    s: float = 12.0,
):
    bp = _ensure_design_blueprint(blueprint, label_key=label_key)
    coords = np.asarray(bp.coordinates, dtype=float)
    labels = bp.obs[label_key].astype(str).to_numpy() if label_key in bp.obs else np.full(bp.n_spots, "global", dtype=object)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=s, label=label, alpha=0.85)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or "Blueprint preview")
    ax.legend(title=label_key, loc="best", frameon=False)
    return fig, ax


def generate_virtual_slice_from_design(
    blueprint: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str],
    parameter_cloud: Union[pd.DataFrame, Mapping[str, Any]],
    *,
    config: Optional[VirtualSliceGenerationConfig] = None,
    random_seed: int = 0,
    quantiles: Optional[np.ndarray] = None,
    pattern_spec: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    label_key: str = "domain",
) -> ad.AnnData:
    bp = _ensure_design_blueprint(blueprint, label_key=label_key)
    gen_cfg = config or VirtualSliceGenerationConfig()
    gene_names = _extract_gene_names_from_cloud(parameter_cloud)
    n_spots = bp.n_spots
    n_genes = len(gene_names)

    if quantiles is not None and pattern_spec is not None:
        raise ValueError("Provide either quantiles or pattern_spec, not both.")

    if quantiles is not None:
        quantiles_arr = np.asarray(quantiles, dtype=np.float32)
        if quantiles_arr.shape != (n_spots, n_genes):
            raise ValueError(f"quantiles must have shape {(n_spots, n_genes)}, got {quantiles_arr.shape}.")
        quantiles_arr = np.clip(quantiles_arr, 0.0, 1.0)
    else:
        rng = np.random.default_rng(int(random_seed))
        quantiles_arr = rng.random((n_spots, n_genes), dtype=np.float32)
        if pattern_spec is not None:
            for idx, gene_name in enumerate(gene_names):
                motifs = pattern_spec.get(str(gene_name), [])
                if motifs:
                    quantiles_arr[:, idx] = compose_gene_pattern(
                        bp,
                        motifs,
                        label_key=label_key,
                        random_seed=int(random_seed) + idx,
                        boundary_softness=float(gen_cfg.boundary_softness),
                    ).astype(np.float32, copy=False)

    quantiles_arr = diffuse_quantile_map(
        quantiles_arr,
        bp.coordinates,
        float(gen_cfg.diffusion_level),
    ).astype(np.float32, copy=False)

    labels = bp.obs[label_key].astype(str).to_numpy()
    counts = np.zeros((n_spots, n_genes), dtype=np.int32)
    label_cloud_summary: Dict[str, Dict[str, float]] = {}
    decode_method = resolve_decode_method(gen_cfg.decode_method, allow_auto=True)

    for label in sorted(np.unique(labels)):
        mask = labels == label
        label_cloud = _resolve_parameter_cloud_for_label(parameter_cloud, gene_names, label=label)
        model_params = _stats_frame_to_model_params(label_cloud)
        decoded = decode_counts_from_quantiles(
            quantiles_arr[mask, :],
            model_params,
            method=decode_method,
            quantile_calibration=str(gen_cfg.quantile_calibration),
            boundary_multiplier=float(gen_cfg.boundary_multiplier),
            random_seed=int(random_seed),
            show_progress=bool(gen_cfg.verbose),
        )
        counts[mask, :] = np.asarray(decoded, dtype=np.int32)
        label_cloud_summary[label] = {
            "mean_mean": float(label_cloud["mean"].mean()),
            "variance_mean": float(label_cloud["variance"].mean()),
            "zero_prop_mean": float(label_cloud["zero_prop"].mean()),
        }

    try:
        global_cloud = _resolve_parameter_cloud_for_label(parameter_cloud, gene_names, label=None)
    except Exception:
        first_label = sorted(np.unique(labels))[0]
        global_cloud = _resolve_parameter_cloud_for_label(parameter_cloud, gene_names, label=first_label)

    obs = bp.obs.copy()
    if "domain" not in obs:
        obs["domain"] = labels
    obs.index = [str(idx) for idx in obs.index]
    var = pd.DataFrame(index=gene_names)
    var["target_mean"] = global_cloud["mean"].to_numpy(dtype=np.float64)
    var["target_variance"] = global_cloud["variance"].to_numpy(dtype=np.float64)
    var["target_zero_prop"] = global_cloud["zero_prop"].to_numpy(dtype=np.float64)

    result = ad.AnnData(X=counts, obs=obs, var=var)
    result.layers["counts"] = counts.astype(np.int32, copy=False)
    result.layers["transported_quantiles"] = quantiles_arr.astype(np.float32, copy=False)
    result.obsm["spatial"] = np.asarray(bp.coordinates, dtype=float).copy()
    result.uns["de_novo"] = {
        "conditional_generation": False,
        "designed_generation": True,
        "label_key": label_key,
        "decode_method": decode_method,
        "quantile_calibration": str(gen_cfg.quantile_calibration),
        "diffusion_level": float(gen_cfg.diffusion_level),
        "boundary_softness": float(gen_cfg.boundary_softness),
        "assignment_randomness": float(gen_cfg.assignment_randomness),
        "parameter_cloud_summary": label_cloud_summary,
        "reference_metadata": {"mode": "design", "labels": sorted(np.unique(labels).tolist())},
    }
    result.uns["target_blueprint"] = bp.to_dict()
    if pattern_spec is not None:
        result.uns["de_novo"]["pattern_spec"] = {
            str(gene): [dict(motif) for motif in motifs]
            for gene, motifs in pattern_spec.items()
        }
    return result
