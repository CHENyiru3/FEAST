from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd


@dataclass
class SliceBlueprint:
    coordinates: np.ndarray
    coordinate_mode: str = "generic"
    grid_type: str = "generic"
    mask: Optional[np.ndarray] = None
    domain_map: Optional[np.ndarray] = None
    technology: Optional[str] = None
    obs: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("SliceBlueprint.coordinates must be a 2D array with at least two columns.")
        self.coordinates = coords[:, :2]
        n_spots = self.coordinates.shape[0]

        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape[0] != n_spots:
                raise ValueError("SliceBlueprint.mask must have one entry per spot.")
            self.mask = mask

        if self.domain_map is not None:
            domain_map = np.asarray(self.domain_map)
            if domain_map.shape[0] != n_spots:
                raise ValueError("SliceBlueprint.domain_map must have one entry per spot.")
            self.domain_map = domain_map

        if self.obs.empty and len(self.obs.index) == 0:
            self.obs = pd.DataFrame(index=[f"spot_{i}" for i in range(n_spots)])
        else:
            self.obs = self.obs.reset_index(drop=True).copy()
        if len(self.obs) != n_spots:
            raise ValueError("SliceBlueprint.obs must contain one row per spot.")
        if self.domain_map is not None and "domain" not in self.obs:
            self.obs["domain"] = self.domain_map

    @property
    def n_spots(self) -> int:
        return self.coordinates.shape[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinates": self.coordinates.tolist(),
            "coordinate_mode": self.coordinate_mode,
            "grid_type": self.grid_type,
            "mask": None if self.mask is None else self.mask.astype(bool).tolist(),
            "domain_map": None if self.domain_map is None else self.domain_map.tolist(),
            "technology": self.technology,
            "obs": self.obs.to_dict(orient="list"),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SliceBlueprint":
        obs_payload = payload.get("obs", {})
        obs = pd.DataFrame(obs_payload) if obs_payload else pd.DataFrame()
        return cls(
            coordinates=np.asarray(payload["coordinates"], dtype=float),
            coordinate_mode=payload.get("coordinate_mode", "generic"),
            grid_type=payload.get("grid_type", "generic"),
            mask=None if payload.get("mask") is None else np.asarray(payload["mask"], dtype=bool),
            domain_map=None if payload.get("domain_map") is None else np.asarray(payload["domain_map"]),
            technology=payload.get("technology"),
            obs=obs,
            metadata=dict(payload.get("metadata", {})),
        )


def load_blueprint(source: Union[SliceBlueprint, ad.AnnData, Mapping[str, Any], str, Path]) -> SliceBlueprint:
    if isinstance(source, SliceBlueprint):
        return source
    if isinstance(source, ad.AnnData):
        if "spatial" not in source.obsm:
            raise ValueError("Blueprint AnnData must contain obsm['spatial'].")
        domain_map = None
        if "domain" in source.obs:
            domain_map = source.obs["domain"].to_numpy()
        elif "region" in source.obs:
            domain_map = source.obs["region"].to_numpy()
        return SliceBlueprint(
            coordinates=np.asarray(source.obsm["spatial"], dtype=float),
            grid_type=source.uns.get("grid_type", "generic"),
            domain_map=domain_map,
            technology=source.uns.get("technology"),
            obs=source.obs.copy(),
            metadata={"source": "anndata"},
        )
    if isinstance(source, (str, Path)):
        return load_blueprint(_load_mapping_file(source))
    if isinstance(source, Mapping):
        return SliceBlueprint.from_dict(source)
    raise TypeError("Unsupported blueprint input.")


def _load_mapping_file(path: Union[str, Path]) -> Dict[str, Any]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() in {".yaml", ".yml"}:
        yaml = _import_yaml()
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping payload in {input_path}.")
    return dict(payload)


def _import_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("YAML support requires PyYAML to be installed.") from exc
    return yaml
