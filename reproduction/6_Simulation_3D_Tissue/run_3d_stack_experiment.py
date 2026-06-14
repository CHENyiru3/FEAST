#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.stats import ks_2samp, rankdata
from sklearn.neighbors import NearestNeighbors

from FEAST.de_novo import SimulationConfig, simulate_stack


DEFAULT_DATA_DIR = Path("/maiziezhou_lab2/yiru/Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DATASET_PREFIX = "Zhuang-ABCA-1"
ZERO_Z_EPSILON = 1e-6
MORAN_NEIGHBORS = 6
MORAN_GENE_CHUNK_SIZE = 128


@dataclass(frozen=True)
class SliceInfo:
    slice_id: int
    path: Path
    z: float
    n_obs: int
    n_vars: int


@dataclass(frozen=True)
class DensitySpec:
    gap: int
    name: str
    radius: int
    start: int
    expected_targets: int | None = None
    expected_references: int | None = None


@dataclass(frozen=True)
class TargetAssignment:
    density_gap: int
    density_name: str
    target_id: int
    lower_ref_id: int
    upper_ref_id: int
    target_z: float
    z0: float
    z1: float
    tau: float
    ref_z_gap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "density_gap": int(self.density_gap),
            "density_name": self.density_name,
            "target_slice": int(self.target_id),
            "lower_ref_slice": int(self.lower_ref_id),
            "upper_ref_slice": int(self.upper_ref_id),
            "target_z": float(self.target_z),
            "z0": float(self.z0),
            "z1": float(self.z1),
            "tau": float(self.tau),
            "ref_z_gap": float(self.ref_z_gap),
        }


@dataclass(frozen=True)
class ZRegularizationConfig:
    enabled: bool
    lambda_1: float
    lambda_2: float
    reference_beta: float
    target_weight: float
    min_class_spots: int
    ridge: float = 1e-8


OFFICIAL_DENSITIES: dict[int, DensitySpec] = {
    3: DensitySpec(gap=3, name="dense_gap3", radius=1, start=5, expected_targets=49, expected_references=95),
    5: DensitySpec(gap=5, name="medium_gap5", radius=2, start=6, expected_targets=29, expected_references=57),
    10: DensitySpec(gap=10, name="sparse_gap10", radius=5, start=6, expected_targets=15, expected_references=16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FEAST 3D semi-reference stack reconstruction experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing Zhuang-ABCA-1 h5ad slices.")
    parser.add_argument("--densities", type=int, nargs="+", default=[3, 5, 10], help="Reference gaps to run.")
    parser.add_argument("--label-key", default="class", help="Observation column with cell class labels.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for experiment outputs.")
    parser.add_argument("--z-regularize", action="store_true", help="Write Z-regularized generated outputs and metrics.")
    parser.add_argument("--z-lambda1", type=float, default=0.01, help="First-difference Z regularization strength.")
    parser.add_argument("--z-lambda2", type=float, default=1e-5, help="Second-difference Z regularization strength.")
    parser.add_argument("--z-reference-beta", type=float, default=1.0, help="Weight multiplier for real reference anchors.")
    parser.add_argument("--z-target-weight", type=float, default=1.0, help="Weight multiplier for initial generated target anchors.")
    parser.add_argument("--z-min-class-spots", type=int, default=2, help="Minimum class spots to use a class mean anchor.")
    parser.add_argument("--reuse-generated", action="store_true", help="Reuse existing generated.h5ad and metric CSVs when present.")
    parser.add_argument("--verbose", action="store_true", help="Print per-target progress.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    densities = _unique_in_order(args.densities)
    slice_infos = discover_slices(args.data_dir, args.label_key)
    if not slice_infos:
        raise ValueError(f"No {DATASET_PREFIX} h5ad slices found in {args.data_dir}.")

    z_config = ZRegularizationConfig(
        enabled=bool(args.z_regularize),
        lambda_1=float(args.z_lambda1),
        lambda_2=float(args.z_lambda2),
        reference_beta=float(args.z_reference_beta),
        target_weight=float(args.z_target_weight),
        min_class_spots=int(args.z_min_class_spots),
    )

    cross_density_rows: list[dict[str, Any]] = []
    for density in densities:
        spec = density_spec(density)
        assignments, reference_ids = build_assignments(slice_infos, spec)
        density_summary = run_density(
            spec=spec,
            assignments=assignments,
            reference_ids=reference_ids,
            slice_infos=slice_infos,
            data_dir=args.data_dir,
            output_dir=args.output_dir / spec.name,
            label_key=args.label_key,
            seed=int(args.seed),
            z_config=z_config,
            reuse_generated=bool(args.reuse_generated),
            verbose=bool(args.verbose),
        )
        cross_density_rows.append(density_summary)

    cross_df = pd.DataFrame(cross_density_rows)
    cross_df.to_csv(args.output_dir / "cross_density_summary.csv", index=False)
    print(f"Wrote cross-density summary: {args.output_dir / 'cross_density_summary.csv'}")


def _unique_in_order(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        value = int(value)
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def density_spec(gap: int) -> DensitySpec:
    gap = int(gap)
    if gap in OFFICIAL_DENSITIES:
        return OFFICIAL_DENSITIES[gap]
    if gap < 2:
        raise ValueError("--densities values must be >= 2.")
    radius = max(1, gap // 2)
    return DensitySpec(gap=gap, name=f"gap{gap}", radius=radius, start=radius + 1)


def discover_slices(data_dir: Path, label_key: str) -> dict[int, SliceInfo]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    out: dict[int, SliceInfo] = {}
    for path in sorted(data_dir.glob(f"{DATASET_PREFIX}.*.h5ad")):
        slice_id = _slice_id_from_path(path)
        adata = ad.read_h5ad(path, backed="r")
        try:
            if "z" not in adata.obs:
                raise KeyError(f"{path.name} is missing obs['z'].")
            if label_key not in adata.obs:
                raise KeyError(f"{path.name} is missing obs['{label_key}'].")
            if "spatial_3d" not in adata.obsm:
                raise KeyError(f"{path.name} is missing obsm['spatial_3d'].")
            z_values = np.asarray(adata.obs["z"], dtype=float)
            if z_values.size == 0 or not np.all(np.isfinite(z_values)):
                raise ValueError(f"{path.name} has invalid obs['z'] values.")
            if not np.allclose(z_values, z_values[0], rtol=0.0, atol=1e-8):
                raise ValueError(f"{path.name} contains multiple z values.")
            out[slice_id] = SliceInfo(
                slice_id=slice_id,
                path=path,
                z=float(z_values[0]),
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
            )
        finally:
            adata.file.close()
    return out


def _slice_id_from_path(path: Path) -> int:
    try:
        return int(path.stem.split(".")[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse slice id from {path.name}.") from exc


def build_assignments(
    slice_infos: Mapping[int, SliceInfo],
    spec: DensitySpec,
) -> tuple[list[TargetAssignment], list[int]]:
    available = set(slice_infos)
    min_id = min(available)
    max_id = max(available)

    assignments: list[TargetAssignment] = []
    reference_ids: set[int] = set()
    target_ids = [slice_id for slice_id in range(spec.start, max_id + 1, spec.gap) if slice_id in available]
    for target_id in target_ids:
        lower_nominal = target_id - spec.radius
        upper_nominal = target_id + spec.radius
        lower_ref_id = (
            min_id
            if lower_nominal < min_id and min_id < target_id
            else _snap_lower(available, lower_nominal, min_id)
        )
        upper_ref_id = (
            max_id
            if upper_nominal > max_id and max_id > target_id
            else _snap_upper(available, upper_nominal, max_id)
        )
        if lower_ref_id is None or upper_ref_id is None:
            raise ValueError(f"Could not bracket target slice {target_id:03d} for density gap {spec.gap}.")
        if not (lower_ref_id < target_id < upper_ref_id):
            raise ValueError(
                f"Target slice {target_id:03d} is not strictly between references "
                f"{lower_ref_id:03d} and {upper_ref_id:03d}."
            )

        target_z = float(slice_infos[target_id].z)
        z0 = adjusted_reference_z(lower_ref_id, slice_infos[lower_ref_id].z)
        z1 = adjusted_reference_z(upper_ref_id, slice_infos[upper_ref_id].z)
        if not (z0 < target_z < z1):
            raise ValueError(
                f"Target slice {target_id:03d} z={target_z:g} is not strictly between "
                f"reference z values {z0:g} and {z1:g}."
            )
        if math.isclose(z0, z1, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Reference z values are not unique for target slice {target_id:03d}.")

        tau = (target_z - z0) / (z1 - z0)
        assignments.append(
            TargetAssignment(
                density_gap=spec.gap,
                density_name=spec.name,
                target_id=target_id,
                lower_ref_id=lower_ref_id,
                upper_ref_id=upper_ref_id,
                target_z=target_z,
                z0=z0,
                z1=z1,
                tau=float(tau),
                ref_z_gap=float(z1 - z0),
            )
        )
        reference_ids.update([lower_ref_id, upper_ref_id])

    references_sorted = sorted(reference_ids)
    if spec.expected_targets is not None and len(assignments) != spec.expected_targets:
        raise ValueError(f"{spec.name} expected {spec.expected_targets} targets, found {len(assignments)}.")
    if spec.expected_references is not None and len(references_sorted) != spec.expected_references:
        raise ValueError(f"{spec.name} expected {spec.expected_references} references, found {len(references_sorted)}.")
    return assignments, references_sorted


def _snap_lower(available: set[int], start: int, min_id: int) -> int | None:
    for slice_id in range(int(start), int(min_id) - 1, -1):
        if slice_id in available:
            return slice_id
    return None


def _snap_upper(available: set[int], start: int, max_id: int) -> int | None:
    for slice_id in range(int(start), int(max_id) + 1):
        if slice_id in available:
            return slice_id
    return None


def adjusted_reference_z(slice_id: int, z: float) -> float:
    if 1 <= int(slice_id) <= 4 and math.isclose(float(z), 0.0, rel_tol=0.0, abs_tol=1e-12):
        return float((int(slice_id) - 1) * ZERO_Z_EPSILON)
    return float(z)


def run_density(
    *,
    spec: DensitySpec,
    assignments: Sequence[TargetAssignment],
    reference_ids: Sequence[int],
    slice_infos: Mapping[int, SliceInfo],
    data_dir: Path,
    output_dir: Path,
    label_key: str,
    seed: int,
    z_config: ZRegularizationConfig,
    reuse_generated: bool,
    verbose: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_ids = {int(item.target_id) for item in assignments}
    if spec.gap in OFFICIAL_DENSITIES:
        support_pool_ids = list(reference_ids)
        support_scope = "density_references"
    else:
        support_pool_ids = sorted(set(slice_infos) - target_ids)
        support_scope = "non_target_slices"
    label_support_index = build_label_support_index(slice_infos, support_pool_ids, label_key)
    metadata = {
        "density_gap": int(spec.gap),
        "density_name": spec.name,
        "radius": int(spec.radius),
        "target_start": int(spec.start),
        "data_dir": str(data_dir),
        "label_key": label_key,
        "seed": int(seed),
        "reuse_generated": bool(reuse_generated),
        "targets": [int(item.target_id) for item in assignments],
        "references": [int(item) for item in reference_ids],
        "label_support_scope": support_scope,
        "label_support_pool": [int(item) for item in support_pool_ids],
        "z_regularization": {
            "enabled": bool(z_config.enabled),
            "lambda_1": float(z_config.lambda_1),
            "lambda_2": float(z_config.lambda_2),
            "reference_beta": float(z_config.reference_beta),
            "target_weight": float(z_config.target_weight),
            "min_class_spots": int(z_config.min_class_spots),
            "ridge": float(z_config.ridge),
        },
        "assignments": [item.to_dict() for item in assignments],
    }
    write_json(output_dir / "metadata.json", metadata)

    if verbose:
        print(f"[{spec.name}] {len(assignments)} targets, {len(reference_ids)} references")

    panel_rows: list[dict[str, Any]] = []
    z_records: list[pd.DataFrame] = []
    gene_names: list[str] | None = None
    for target_index, assignment in enumerate(assignments):
        if verbose:
            print(
                f"[{spec.name}] target {target_index + 1}/{len(assignments)} "
                f"{DATASET_PREFIX}.{assignment.target_id:03d} "
                f"refs {assignment.lower_ref_id:03d}/{assignment.upper_ref_id:03d}"
        )
        target_seed = int(seed) + int(spec.gap) * 10_000 + target_index
        target_dir = output_dir / f"{DATASET_PREFIX}.{assignment.target_id:03d}"
        if reuse_generated and existing_target_outputs_complete(target_dir):
            panel_summary = json.loads((target_dir / "panel_summary.json").read_text(encoding="utf-8"))
            class_metrics = pd.read_csv(target_dir / "per_class_metrics.csv")
        else:
            panel_summary, class_metrics = run_target(
                assignment=assignment,
                slice_infos=slice_infos,
                output_dir=target_dir,
                label_key=label_key,
                label_support_index=label_support_index,
                random_seed=target_seed,
                verbose=verbose,
            )
        panel_rows.append(panel_summary)
        z_records.append(
            class_metrics[
                ["target_slice", "target_z", "class", "n_spots", "gene", "generated_mean", "target_mean"]
            ].copy()
        )
        if gene_names is None and not class_metrics.empty:
            gene_names = class_metrics["gene"].drop_duplicates().astype(str).tolist()

    summary_df = pd.DataFrame(panel_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    z_input = pd.concat(z_records, ignore_index=True) if z_records else pd.DataFrame()
    z_coherence_df = compute_z_coherence(z_input)
    z_coherence_df.to_csv(output_dir / "z_coherence_metrics.csv", index=False)

    real_baseline_df = (
        compute_real_z_coherence_split_half(
            assignments=assignments,
            slice_infos=slice_infos,
            gene_names=gene_names or [],
            label_key=label_key,
            seed=int(seed),
        )
        if gene_names
        else pd.DataFrame(columns=["class", "gene", "n_slices", "z_min", "z_max", "z_coherence"])
    )
    real_baseline_df.to_csv(output_dir / "real_z_coherence_split_half.csv", index=False)
    real_baseline_summary = summarize_z_coherence_frame(real_baseline_df, prefix="real_split_half_z_coherence")
    write_json(output_dir / "real_z_coherence_split_half_summary.json", real_baseline_summary)

    regularized_summary_df: pd.DataFrame | None = None
    regularized_z_coherence_df: pd.DataFrame | None = None
    if z_config.enabled:
        if not gene_names:
            raise ValueError(f"{spec.name} produced no gene names for Z regularization.")
        reference_class_means = collect_reference_class_means(
            slice_infos=slice_infos,
            reference_ids=reference_ids,
            gene_names=gene_names,
            label_key=label_key,
            min_class_spots=int(z_config.min_class_spots),
        )
        regularized_targets = fit_z_regularized_target_means(
            target_records=z_input,
            reference_records=reference_class_means,
            assignments=assignments,
            gene_names=gene_names,
            config=z_config,
        )
        regularized_targets.to_csv(output_dir / "z_regularized_class_means.csv", index=False)
        regularized_summary_df, regularized_class_records = apply_z_regularized_outputs(
            regularized_targets=regularized_targets,
            assignments=assignments,
            slice_infos=slice_infos,
            output_dir=output_dir,
            label_key=label_key,
            seed=int(seed),
            verbose=verbose,
        )
        regularized_summary_df.to_csv(output_dir / "summary_z_regularized.csv", index=False)
        regularized_z_input = (
            pd.concat(regularized_class_records, ignore_index=True) if regularized_class_records else pd.DataFrame()
        )
        regularized_z_coherence_df = compute_z_coherence(regularized_z_input)
        regularized_z_coherence_df.to_csv(output_dir / "z_coherence_metrics_z_regularized.csv", index=False)

    stack_summary = summarize_density(
        spec=spec,
        assignments=assignments,
        reference_ids=reference_ids,
        summary_df=summary_df,
        z_coherence_df=z_coherence_df,
        real_baseline_summary=real_baseline_summary,
        regularized_summary_df=regularized_summary_df,
        regularized_z_coherence_df=regularized_z_coherence_df,
    )
    write_json(output_dir / "stack_summary.json", stack_summary)
    return stack_summary


def existing_target_outputs_complete(target_dir: Path) -> bool:
    required = [
        "generated.h5ad",
        "per_gene_metrics.csv",
        "panel_summary.json",
        "per_class_metrics.csv",
        "moran_metrics.csv",
    ]
    return all((target_dir / name).exists() for name in required)


def run_target(
    *,
    assignment: TargetAssignment,
    slice_infos: Mapping[int, SliceInfo],
    output_dir: Path,
    label_key: str,
    label_support_index: Mapping[str, Sequence[int]],
    random_seed: int,
    verbose: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    lower_ref = load_reference(slice_infos[assignment.lower_ref_id], label_key)
    upper_ref = load_reference(slice_infos[assignment.upper_ref_id], label_key)
    target = ad.read_h5ad(slice_infos[assignment.target_id].path)
    try:
        lower_ref, upper_ref, label_support_records = augment_missing_label_support(
            lower_ref=lower_ref,
            upper_ref=upper_ref,
            target=target,
            assignment=assignment,
            slice_infos=slice_infos,
            label_key=label_key,
            label_support_index=label_support_index,
        )
        target_blueprint = make_target_blueprint(target, label_key=label_key)
        generated = simulate_stack(
            reference_slices=[lower_ref, upper_ref],
            reference_z_values=[assignment.z0, assignment.z1],
            target_z_values=[assignment.target_z],
            target_blueprints={assignment.target_z: target_blueprint},
            label_key=label_key,
            config=SimulationConfig(verbose=verbose),
            random_seed=int(random_seed),
        )[float(assignment.target_z)]

        generated.uns.setdefault("de_novo", {})["experiment"] = {
            "name": "3d_stack_semi_reference",
            "density_gap": int(assignment.density_gap),
            "density_name": assignment.density_name,
            "target_slice": int(assignment.target_id),
            "lower_ref_slice": int(assignment.lower_ref_id),
            "upper_ref_slice": int(assignment.upper_ref_id),
            "random_seed": int(random_seed),
            "label_support": label_support_records,
        }
        ensure_generated_contract(generated, assignment.target_z)

        gene_names = [str(gene) for gene in generated.var_names]
        generated_counts = counts_for_genes(generated, gene_names, prefer_layer=True)
        target_counts = counts_for_genes(target, gene_names, prefer_layer=True)
        target_xy = target_xy_coordinates(target)
        target_labels = target.obs[label_key].astype(str).to_numpy()

        per_gene_metrics = compute_per_gene_metrics(
            generated_counts=generated_counts,
            target_counts=target_counts,
            xy=target_xy,
            gene_names=gene_names,
        )
        per_gene_metrics.to_csv(output_dir / "per_gene_metrics.csv", index=False)
        per_gene_metrics[["gene", "generated_moran_i", "target_moran_i"]].to_csv(
            output_dir / "moran_metrics.csv",
            index=False,
        )

        class_metrics = compute_per_class_metrics(
            generated_counts=generated_counts,
            target_counts=target_counts,
            labels=target_labels,
            gene_names=gene_names,
            assignment=assignment,
        )
        class_metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)

        panel_summary = compute_panel_summary(
            per_gene_metrics=per_gene_metrics,
            assignment=assignment,
            n_spots=int(target.n_obs),
            n_genes=len(gene_names),
            random_seed=int(random_seed),
            label_support_records=label_support_records,
        )
        write_json(output_dir / "panel_summary.json", panel_summary)

        sanitize_uns_for_h5ad(generated)
        generated.write_h5ad(output_dir / "generated.h5ad")
        return panel_summary, class_metrics
    finally:
        del lower_ref, upper_ref, target
        gc.collect()


def load_reference(info: SliceInfo, label_key: str) -> ad.AnnData:
    adata = ad.read_h5ad(info.path)
    if label_key not in adata.obs:
        raise KeyError(f"{info.path.name} is missing obs['{label_key}'].")
    adata.obs = adata.obs.copy()
    adata.obs[label_key] = adata.obs[label_key].astype(str)
    adata.uns["reference_name"] = f"{DATASET_PREFIX}.{info.slice_id:03d}"
    return adata


def build_label_support_index(
    slice_infos: Mapping[int, SliceInfo],
    support_pool_ids: Sequence[int],
    label_key: str,
) -> dict[str, list[int]]:
    label_to_slices: dict[str, list[int]] = {}
    for slice_id in support_pool_ids:
        info = slice_infos[int(slice_id)]
        adata = ad.read_h5ad(info.path, backed="r")
        try:
            labels = pd.Index(adata.obs[label_key].astype(str)).unique()
            for label in labels:
                label_to_slices.setdefault(str(label), []).append(int(slice_id))
        finally:
            adata.file.close()
    for label in label_to_slices:
        label_to_slices[label].sort()
    return label_to_slices


def augment_missing_label_support(
    *,
    lower_ref: ad.AnnData,
    upper_ref: ad.AnnData,
    target: ad.AnnData,
    assignment: TargetAssignment,
    slice_infos: Mapping[int, SliceInfo],
    label_key: str,
    label_support_index: Mapping[str, Sequence[int]],
) -> tuple[ad.AnnData, ad.AnnData, list[dict[str, Any]]]:
    target_labels = set(target.obs[label_key].astype(str).unique())
    bracket_labels = set(lower_ref.obs[label_key].astype(str).unique()) | set(upper_ref.obs[label_key].astype(str).unique())
    missing_labels = sorted(target_labels - bracket_labels)
    if not missing_labels:
        return lower_ref, upper_ref, []

    lower_parts = [lower_ref]
    upper_parts = [upper_ref]
    donor_cache: dict[int, ad.AnnData] = {}
    support_records: list[dict[str, Any]] = []
    for label in missing_labels:
        candidates = [int(slice_id) for slice_id in label_support_index.get(label, []) if int(slice_id) != assignment.target_id]
        if not candidates:
            raise ValueError(
                f"Target slice {assignment.target_id:03d} contains label {label!r}, "
                "but no allowed support reference contains that label."
            )
        donor_id = min(
            candidates,
            key=lambda slice_id: (
                abs(adjusted_reference_z(slice_id, slice_infos[slice_id].z) - assignment.target_z),
                slice_id,
            ),
        )
        if donor_id not in donor_cache:
            donor_cache[donor_id] = load_reference(slice_infos[donor_id], label_key)
        donor = donor_cache[donor_id]
        donor_mask = donor.obs[label_key].astype(str).to_numpy() == label
        donor_subset = donor[donor_mask, :].copy()
        if donor_subset.n_obs == 0:
            raise ValueError(f"Support donor {donor_id:03d} unexpectedly has no spots for label {label!r}.")

        donor_z = adjusted_reference_z(donor_id, slice_infos[donor_id].z)
        side = "lower" if donor_z <= assignment.target_z else "upper"
        if side == "lower":
            lower_parts.append(donor_subset)
        else:
            upper_parts.append(donor_subset)
        support_records.append(
            {
                "label": label,
                "donor_slice": int(donor_id),
                "donor_z": float(donor_z),
                "side": side,
                "n_spots": int(donor_subset.n_obs),
            }
        )

    if len(lower_parts) > 1:
        lower_ref = concat_reference_parts(lower_parts, lower_ref.uns["reference_name"], label_key)
    if len(upper_parts) > 1:
        upper_ref = concat_reference_parts(upper_parts, upper_ref.uns["reference_name"], label_key)
    return lower_ref, upper_ref, support_records


def concat_reference_parts(parts: Sequence[ad.AnnData], reference_name: str, label_key: str) -> ad.AnnData:
    merged = ad.concat(parts, axis=0, join="inner", merge="same", uns_merge="first")
    merged.obs = merged.obs.copy()
    merged.obs[label_key] = merged.obs[label_key].astype(str)
    merged.obs_names_make_unique()
    merged.uns["reference_name"] = str(reference_name)
    return merged


def make_target_blueprint(target: ad.AnnData, *, label_key: str) -> ad.AnnData:
    if label_key not in target.obs:
        raise KeyError(f"Target slice is missing obs['{label_key}'].")
    labels = target.obs[label_key].astype(str).to_numpy()
    obs = pd.DataFrame(
        {
            label_key: labels,
            # simulate_stack converts AnnData blueprints through load_blueprint
            # before simulate_from_reference, so domain carries the label map.
            "domain": labels,
        },
        index=target.obs_names.astype(str),
    )
    blueprint = ad.AnnData(
        X=np.zeros((target.n_obs, 0), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index([], dtype=str)),
    )
    blueprint.obsm["spatial"] = target_xy_coordinates(target)
    return blueprint


def ensure_generated_contract(generated: ad.AnnData, target_z: float) -> None:
    counts = np.asarray(generated.layers["counts"] if "counts" in generated.layers else generated.X)
    counts = counts.astype(np.int32, copy=False)
    generated.X = counts
    generated.layers["counts"] = counts

    if "spatial" not in generated.obsm:
        raise KeyError("Generated AnnData is missing obsm['spatial'].")
    if "spatial_3d" not in generated.obsm:
        raise KeyError("Generated AnnData is missing obsm['spatial_3d'].")
    if "z" not in generated.obs:
        raise KeyError("Generated AnnData is missing obs['z'].")
    if not np.allclose(np.asarray(generated.obs["z"], dtype=float), float(target_z), rtol=0.0, atol=1e-8):
        raise ValueError("Generated obs['z'] does not match the target z value.")
    xyz = np.asarray(generated.obsm["spatial_3d"], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Generated obsm['spatial_3d'] must have shape (n_obs, 3).")
    if not np.allclose(xyz[:, 2], float(target_z), rtol=0.0, atol=1e-8):
        raise ValueError("Generated spatial_3d z coordinates do not match the target z value.")
    de_novo = generated.uns.get("de_novo", {})
    if bool(de_novo.get("conditional_generation")) is not True:
        raise ValueError("Generated uns['de_novo']['conditional_generation'] must be True.")
    for key in ("target_z", "z0", "z1", "tau", "reference_weights"):
        if key not in de_novo.get("stack", {}):
            raise KeyError(f"Generated stack metadata is missing {key!r}.")


def sanitize_uns_for_h5ad(adata: ad.AnnData) -> None:
    adata.uns = h5ad_ready(adata.uns)


def h5ad_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): h5ad_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return h5ad_ready(list(value))
    if isinstance(value, list):
        items = [h5ad_ready(item) for item in value]
        if all(_is_h5ad_scalar(item) for item in items):
            return items
        return json.dumps(json_ready(value), sort_keys=True, allow_nan=False)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "U"}:
            return h5ad_ready(value.tolist())
        return value
    if value is None:
        return "null"
    if isinstance(value, float) and not math.isfinite(value):
        return "null"
    if isinstance(value, np.floating) and not math.isfinite(float(value)):
        return "null"
    return value


def _is_h5ad_scalar(value: Any) -> bool:
    return isinstance(value, (str, bytes, bool, int, float, np.integer, np.floating, np.bool_))


def target_xy_coordinates(adata: ad.AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
    elif "spatial_3d" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial_3d"], dtype=float)[:, :2]
    else:
        raise KeyError("AnnData is missing obsm['spatial'] and obsm['spatial_3d'].")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must have shape (n_obs, >=2).")
    return coords[:, :2].copy()


def counts_for_genes(adata: ad.AnnData, gene_names: Sequence[str], *, prefer_layer: bool) -> np.ndarray:
    view = adata[:, list(gene_names)]
    matrix = view.layers["counts"] if prefer_layer and "counts" in view.layers else view.X
    return dense_matrix(matrix).astype(np.float64, copy=False)


def dense_matrix(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def compute_per_gene_metrics(
    *,
    generated_counts: np.ndarray,
    target_counts: np.ndarray,
    xy: np.ndarray,
    gene_names: Sequence[str],
) -> pd.DataFrame:
    generated_means = generated_counts.mean(axis=0)
    target_means = target_counts.mean(axis=0)
    generated_variances = generated_counts.var(axis=0)
    target_variances = target_counts.var(axis=0)
    generated_zero = np.mean(generated_counts <= 0, axis=0)
    target_zero = np.mean(target_counts <= 0, axis=0)

    pearson = column_pearson(generated_counts, target_counts)
    generated_ranks = rankdata(generated_counts, axis=0, method="average")
    target_ranks = rankdata(target_counts, axis=0, method="average")
    spearman = column_pearson(generated_ranks, target_ranks)

    generated_moran = moran_i_matrix(generated_counts, xy)
    target_moran = moran_i_matrix(target_counts, xy)

    return pd.DataFrame(
        {
            "gene": list(map(str, gene_names)),
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "generated_mean": generated_means,
            "target_mean": target_means,
            "generated_variance": generated_variances,
            "target_variance": target_variances,
            "generated_zero_prop": generated_zero,
            "target_zero_prop": target_zero,
            "generated_moran_i": generated_moran,
            "target_moran_i": target_moran,
        }
    )


def column_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=0)
    denominator = np.sqrt(np.sum(left_centered * left_centered, axis=0) * np.sum(right_centered * right_centered, axis=0))
    out = np.full(left.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def moran_i_matrix(
    matrix: np.ndarray,
    xy: np.ndarray,
    *,
    n_neighbors: int = MORAN_NEIGHBORS,
    gene_chunk_size: int = MORAN_GENE_CHUNK_SIZE,
) -> np.ndarray:
    matrix = np.asarray(matrix)
    n_spots, n_genes = matrix.shape
    out = np.full(n_genes, np.nan, dtype=np.float64)
    if n_spots < 3:
        return out

    k = min(int(n_neighbors), n_spots - 1)
    neighbor_idx = NearestNeighbors(n_neighbors=k + 1).fit(np.asarray(xy, dtype=float)).kneighbors(return_distance=False)[:, 1:]
    for start in range(0, n_genes, int(gene_chunk_size)):
        stop = min(start + int(gene_chunk_size), n_genes)
        chunk = matrix[:, start:stop].astype(np.float64, copy=False)
        centered = chunk - chunk.mean(axis=0, keepdims=True)
        denom = np.sum(centered * centered, axis=0)
        neighbor_sum = centered[neighbor_idx].sum(axis=1)
        numerator = np.sum(centered * neighbor_sum, axis=0)
        valid = denom > 0
        values = np.full(stop - start, np.nan, dtype=np.float64)
        values[valid] = numerator[valid] / (float(k) * denom[valid])
        out[start:stop] = values
    return out


def compute_per_class_metrics(
    *,
    generated_counts: np.ndarray,
    target_counts: np.ndarray,
    labels: np.ndarray,
    gene_names: Sequence[str],
    assignment: TargetAssignment,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    labels = np.asarray(labels).astype(str)
    for class_name in sorted(pd.unique(labels)):
        mask = labels == class_name
        if not np.any(mask):
            continue
        gen_class = generated_counts[mask, :]
        tgt_class = target_counts[mask, :]
        rows.append(
            pd.DataFrame(
                {
                    "density_gap": int(assignment.density_gap),
                    "density_name": assignment.density_name,
                    "target_slice": int(assignment.target_id),
                    "target_z": float(assignment.target_z),
                    "class": str(class_name),
                    "n_spots": int(mask.sum()),
                    "gene": list(map(str, gene_names)),
                    "generated_mean": gen_class.mean(axis=0),
                    "target_mean": tgt_class.mean(axis=0),
                    "generated_zero_prop": np.mean(gen_class <= 0, axis=0),
                    "target_zero_prop": np.mean(tgt_class <= 0, axis=0),
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "density_gap",
                "density_name",
                "target_slice",
                "target_z",
                "class",
                "n_spots",
                "gene",
                "generated_mean",
                "target_mean",
                "generated_zero_prop",
                "target_zero_prop",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def compute_real_z_coherence_split_half(
    *,
    assignments: Sequence[TargetAssignment],
    slice_infos: Mapping[int, SliceInfo],
    gene_names: Sequence[str],
    label_key: str,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for assignment in assignments:
        target = ad.read_h5ad(slice_infos[assignment.target_id].path)
        try:
            labels = target.obs[label_key].astype(str).to_numpy()
            counts = counts_for_genes(target, gene_names, prefer_layer=True)
            rng = np.random.default_rng(int(seed) + int(assignment.target_id))
            for class_name in sorted(pd.unique(labels)):
                indices = np.flatnonzero(labels == class_name)
                if indices.size < 2:
                    continue
                shuffled = rng.permutation(indices)
                split_at = int(indices.size // 2)
                left_idx = shuffled[:split_at]
                right_idx = shuffled[split_at:]
                if left_idx.size == 0 or right_idx.size == 0:
                    continue
                rows.append(
                    pd.DataFrame(
                        {
                            "target_slice": int(assignment.target_id),
                            "target_z": float(assignment.target_z),
                            "class": str(class_name),
                            "n_spots": int(indices.size),
                            "gene": list(map(str, gene_names)),
                            "split_a_mean": counts[left_idx, :].mean(axis=0),
                            "split_b_mean": counts[right_idx, :].mean(axis=0),
                        }
                    )
                )
        finally:
            del target
            gc.collect()
    if not rows:
        return pd.DataFrame(columns=["class", "gene", "n_slices", "z_min", "z_max", "z_coherence"])
    split_records = pd.concat(rows, ignore_index=True)
    return compute_z_coherence(split_records, left_col="split_a_mean", right_col="split_b_mean")


def collect_reference_class_means(
    *,
    slice_infos: Mapping[int, SliceInfo],
    reference_ids: Sequence[int],
    gene_names: Sequence[str],
    label_key: str,
    min_class_spots: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for reference_id in reference_ids:
        info = slice_infos[int(reference_id)]
        reference = ad.read_h5ad(info.path)
        try:
            labels = reference.obs[label_key].astype(str).to_numpy()
            counts = counts_for_genes(reference, gene_names, prefer_layer=True)
            for class_name in sorted(pd.unique(labels)):
                mask = labels == class_name
                n_spots = int(mask.sum())
                if n_spots < int(min_class_spots):
                    continue
                rows.append(
                    pd.DataFrame(
                        {
                            "reference_slice": int(reference_id),
                            "reference_z": adjusted_reference_z(reference_id, info.z),
                            "class": str(class_name),
                            "n_spots": n_spots,
                            "gene": list(map(str, gene_names)),
                            "reference_mean": counts[mask, :].mean(axis=0),
                        }
                    )
                )
        finally:
            del reference
            gc.collect()
    if not rows:
        return pd.DataFrame(
            columns=["reference_slice", "reference_z", "class", "n_spots", "gene", "reference_mean"]
        )
    return pd.concat(rows, ignore_index=True)


def fit_z_regularized_target_means(
    *,
    target_records: pd.DataFrame,
    reference_records: pd.DataFrame,
    assignments: Sequence[TargetAssignment],
    gene_names: Sequence[str],
    config: ZRegularizationConfig,
) -> pd.DataFrame:
    columns = [
        "target_slice",
        "target_z",
        "class",
        "n_spots",
        "gene",
        "initial_generated_mean",
        "regularized_mean",
        "target_mean",
        "target_anchor_weight",
    ]
    if target_records.empty:
        return pd.DataFrame(columns=columns)

    target_nodes = {
        int(assignment.target_id): float(assignment.target_z)
        for assignment in assignments
    }
    reference_nodes = {
        int(row.reference_slice): float(row.reference_z)
        for row in reference_records[["reference_slice", "reference_z"]].drop_duplicates().itertuples(index=False)
    }
    node_items: list[tuple[str, int, float]] = [
        *[("reference", slice_id, z) for slice_id, z in reference_nodes.items()],
        *[("target", slice_id, z) for slice_id, z in target_nodes.items()],
    ]
    node_items.sort(key=lambda item: (item[2], item[0], item[1]))
    node_index = {(kind, slice_id): idx for idx, (kind, slice_id, _z) in enumerate(node_items)}
    z_values = np.asarray([item[2] for item in node_items], dtype=np.float64)
    smooth_matrix = z_penalty_matrix(z_values, config.lambda_1, config.lambda_2)

    all_classes = sorted(pd.unique(target_records["class"].astype(str)))
    gene_index = pd.Index(list(map(str, gene_names)))
    out_parts: list[pd.DataFrame] = []

    for class_name in all_classes:
        class_target = target_records[target_records["class"].astype(str) == class_name]
        if class_target.empty:
            continue
        class_reference = reference_records[reference_records["class"].astype(str) == class_name]

        y = np.zeros((len(node_items), len(gene_index)), dtype=np.float64)
        weights = np.zeros(len(node_items), dtype=np.float64)

        for target_slice, group in class_target.groupby("target_slice", sort=False):
            n_spots = int(group["n_spots"].iloc[0])
            if n_spots < int(config.min_class_spots):
                continue
            idx = node_index.get(("target", int(target_slice)))
            if idx is None:
                continue
            means = group.set_index("gene")["generated_mean"].reindex(gene_index).to_numpy(dtype=np.float64)
            y[idx, :] = np.log1p(np.clip(means, 0.0, None))
            weights[idx] = max(weights[idx], class_anchor_weight(n_spots, config.target_weight))

        for reference_slice, group in class_reference.groupby("reference_slice", sort=False):
            n_spots = int(group["n_spots"].iloc[0])
            if n_spots < int(config.min_class_spots):
                continue
            idx = node_index.get(("reference", int(reference_slice)))
            if idx is None:
                continue
            means = group.set_index("gene")["reference_mean"].reindex(gene_index).to_numpy(dtype=np.float64)
            y[idx, :] = np.log1p(np.clip(means, 0.0, None))
            weights[idx] = max(weights[idx], class_anchor_weight(n_spots, config.reference_beta))

        if not np.any(weights > 0):
            continue

        system = smooth_matrix + np.diag(weights + float(config.ridge))
        rhs = weights[:, None] * y
        try:
            factor = cho_factor(system, lower=True, check_finite=False)
            solved_log = cho_solve(factor, rhs, check_finite=False)
        except (LinAlgError, ValueError):
            solved_log = np.linalg.solve(system, rhs)
        solved_mean = np.expm1(np.clip(solved_log, 0.0, None))

        for target_slice, group in class_target.groupby("target_slice", sort=False):
            idx = node_index.get(("target", int(target_slice)))
            if idx is None:
                continue
            aligned = group.set_index("gene").reindex(gene_index)
            out_parts.append(
                pd.DataFrame(
                    {
                        "target_slice": int(target_slice),
                        "target_z": float(target_nodes[int(target_slice)]),
                        "class": str(class_name),
                        "n_spots": aligned["n_spots"].fillna(0).astype(int).to_numpy(),
                        "gene": gene_index.to_numpy(dtype=str),
                        "initial_generated_mean": aligned["generated_mean"].to_numpy(dtype=np.float64),
                        "regularized_mean": solved_mean[idx, :],
                        "target_mean": aligned["target_mean"].to_numpy(dtype=np.float64),
                        "target_anchor_weight": float(weights[idx]),
                    }
                )
            )

    if not out_parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(out_parts, ignore_index=True)


def class_anchor_weight(n_spots: int, multiplier: float) -> float:
    return float(multiplier) * max(1.0, math.log1p(max(0, int(n_spots))))


def z_penalty_matrix(z_values: np.ndarray, lambda_1: float, lambda_2: float) -> np.ndarray:
    z_values = np.asarray(z_values, dtype=np.float64)
    n_nodes = int(z_values.size)
    penalty = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if n_nodes < 2:
        return penalty

    if float(lambda_1) > 0.0:
        for idx in range(n_nodes - 1):
            dz = float(z_values[idx + 1] - z_values[idx])
            if dz <= 0.0:
                continue
            row = np.zeros(n_nodes, dtype=np.float64)
            row[idx] = -1.0 / dz
            row[idx + 1] = 1.0 / dz
            penalty += float(lambda_1) * np.outer(row, row)

    if float(lambda_2) > 0.0 and n_nodes >= 3:
        for idx in range(1, n_nodes - 1):
            left_dz = float(z_values[idx] - z_values[idx - 1])
            right_dz = float(z_values[idx + 1] - z_values[idx])
            span = float(z_values[idx + 1] - z_values[idx - 1])
            if left_dz <= 0.0 or right_dz <= 0.0 or span <= 0.0:
                continue
            row = np.zeros(n_nodes, dtype=np.float64)
            row[idx - 1] = 2.0 / (span * left_dz)
            row[idx] = -2.0 / span * (1.0 / right_dz + 1.0 / left_dz)
            row[idx + 1] = 2.0 / (span * right_dz)
            penalty += float(lambda_2) * np.outer(row, row)
    return penalty


def apply_z_regularized_outputs(
    *,
    regularized_targets: pd.DataFrame,
    assignments: Sequence[TargetAssignment],
    slice_infos: Mapping[int, SliceInfo],
    output_dir: Path,
    label_key: str,
    seed: int,
    verbose: bool,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    panel_rows: list[dict[str, Any]] = []
    class_records: list[pd.DataFrame] = []
    for target_index, assignment in enumerate(assignments):
        target_dir = output_dir / f"{DATASET_PREFIX}.{assignment.target_id:03d}"
        generated_path = target_dir / "generated.h5ad"
        if not generated_path.exists():
            raise FileNotFoundError(f"Cannot regularize missing generated output: {generated_path}")
        if verbose:
            print(f"[{assignment.density_name}] z-regularizing {DATASET_PREFIX}.{assignment.target_id:03d}")

        generated = ad.read_h5ad(generated_path)
        target = ad.read_h5ad(slice_infos[assignment.target_id].path)
        try:
            gene_names = [str(gene) for gene in generated.var_names]
            generated_counts = counts_for_genes(generated, gene_names, prefer_layer=True)
            target_counts = counts_for_genes(target, gene_names, prefer_layer=True)
            generated_labels = generated.obs[label_key].astype(str).to_numpy() if label_key in generated.obs else target.obs[label_key].astype(str).to_numpy()
            target_labels = target.obs[label_key].astype(str).to_numpy()
            target_xy = target_xy_coordinates(target)

            target_regularized = regularized_targets[
                regularized_targets["target_slice"].astype(int) == int(assignment.target_id)
            ]
            regularized_counts = calibrate_counts_to_regularized_means(
                counts=generated_counts,
                labels=generated_labels,
                gene_names=gene_names,
                target_regularized=target_regularized,
            )

            generated.X = regularized_counts
            generated.layers["counts"] = regularized_counts
            generated.uns.setdefault("de_novo", {}).setdefault("experiment", {})["z_regularization"] = {
                "enabled": True,
                "method": "class_gene_log_mean_quadratic_smoothing",
            }
            ensure_generated_contract(generated, assignment.target_z)

            per_gene_metrics = compute_per_gene_metrics(
                generated_counts=regularized_counts.astype(np.float64, copy=False),
                target_counts=target_counts,
                xy=target_xy,
                gene_names=gene_names,
            )
            per_gene_metrics.to_csv(target_dir / "per_gene_metrics_z_regularized.csv", index=False)
            per_gene_metrics[["gene", "generated_moran_i", "target_moran_i"]].to_csv(
                target_dir / "moran_metrics_z_regularized.csv",
                index=False,
            )

            class_metrics = compute_per_class_metrics(
                generated_counts=regularized_counts.astype(np.float64, copy=False),
                target_counts=target_counts,
                labels=target_labels,
                gene_names=gene_names,
                assignment=assignment,
            )
            class_metrics.to_csv(target_dir / "per_class_metrics_z_regularized.csv", index=False)
            class_records.append(
                class_metrics[
                    ["target_slice", "target_z", "class", "n_spots", "gene", "generated_mean", "target_mean"]
                ].copy()
            )

            target_seed = int(seed) + int(assignment.density_gap) * 10_000 + target_index
            panel_summary = compute_panel_summary(
                per_gene_metrics=per_gene_metrics,
                assignment=assignment,
                n_spots=int(target.n_obs),
                n_genes=len(gene_names),
                random_seed=target_seed,
                label_support_records=[],
            )
            base_summary_path = target_dir / "panel_summary.json"
            if base_summary_path.exists():
                base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
                panel_summary["n_label_support_records"] = int(base_summary.get("n_label_support_records", 0) or 0)
                panel_summary["label_support_labels"] = list(base_summary.get("label_support_labels", []))
            panel_summary["z_regularized"] = True
            panel_summary["generated_h5ad"] = "generated_z_regularized.h5ad"
            panel_summary["per_gene_metrics_csv"] = "per_gene_metrics_z_regularized.csv"
            panel_summary["per_class_metrics_csv"] = "per_class_metrics_z_regularized.csv"
            write_json(target_dir / "panel_summary_z_regularized.json", panel_summary)
            panel_rows.append(panel_summary)

            sanitize_uns_for_h5ad(generated)
            generated.write_h5ad(target_dir / "generated_z_regularized.h5ad")
        finally:
            del generated, target
            gc.collect()

    return pd.DataFrame(panel_rows), class_records


def calibrate_counts_to_regularized_means(
    *,
    counts: np.ndarray,
    labels: np.ndarray,
    gene_names: Sequence[str],
    target_regularized: pd.DataFrame,
) -> np.ndarray:
    out = np.asarray(counts, dtype=np.float64).copy()
    labels = np.asarray(labels).astype(str)
    gene_index = pd.Index(list(map(str, gene_names)))
    for class_name, group in target_regularized.groupby("class", sort=False):
        mask = labels == str(class_name)
        if not np.any(mask):
            continue
        desired = group.set_index("gene")["regularized_mean"].reindex(gene_index).to_numpy(dtype=np.float64)
        valid = np.isfinite(desired) & (desired >= 0.0)
        if not np.any(valid):
            continue
        class_counts = out[mask, :]
        current = class_counts.mean(axis=0)
        scale = np.ones_like(current)
        positive = valid & (current > 0.0)
        scale[positive] = desired[positive] / current[positive]
        class_counts[:, positive] *= scale[positive]

        zero_to_positive = valid & (current <= 0.0) & (desired > 0.0)
        if np.any(zero_to_positive):
            class_counts[:, zero_to_positive] = desired[zero_to_positive]
        out[mask, :] = class_counts
    return np.rint(np.clip(out, 0.0, None)).astype(np.int32, copy=False)


def compute_panel_summary(
    *,
    per_gene_metrics: pd.DataFrame,
    assignment: TargetAssignment,
    n_spots: int,
    n_genes: int,
    random_seed: int,
    label_support_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    generated_means = per_gene_metrics["generated_mean"].to_numpy(dtype=float)
    target_means = per_gene_metrics["target_mean"].to_numpy(dtype=float)
    generated_variances = per_gene_metrics["generated_variance"].to_numpy(dtype=float)
    target_variances = per_gene_metrics["target_variance"].to_numpy(dtype=float)
    generated_moran = per_gene_metrics["generated_moran_i"].to_numpy(dtype=float)
    target_moran = per_gene_metrics["target_moran_i"].to_numpy(dtype=float)
    generated_zero = per_gene_metrics["generated_zero_prop"].to_numpy(dtype=float)
    target_zero = per_gene_metrics["target_zero_prop"].to_numpy(dtype=float)

    return {
        **assignment.to_dict(),
        "n_spots": int(n_spots),
        "n_genes": int(n_genes),
        "random_seed": int(random_seed),
        "n_label_support_records": int(len(label_support_records)),
        "label_support_labels": [str(record["label"]) for record in label_support_records],
        "mean_corr": safe_pearson(np.log1p(generated_means), np.log1p(target_means)),
        "var_corr": safe_pearson(generated_variances, target_variances),
        "moran_corr": safe_pearson(generated_moran, target_moran),
        "zero_ks": safe_ks(generated_zero, target_zero),
        "median_gene_pearson": safe_nanmedian(per_gene_metrics["pearson_r"].to_numpy(dtype=float)),
        "median_gene_spearman": safe_nanmedian(per_gene_metrics["spearman_rho"].to_numpy(dtype=float)),
        "generated_h5ad": "generated.h5ad",
        "per_gene_metrics_csv": "per_gene_metrics.csv",
        "per_class_metrics_csv": "per_class_metrics.csv",
    }


def safe_pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left_arr) & np.isfinite(right_arr)
    if int(valid.sum()) < 2:
        return float("nan")
    left_arr = left_arr[valid]
    right_arr = right_arr[valid]
    left_centered = left_arr - left_arr.mean()
    right_centered = right_arr - right_arr.mean()
    denominator = math.sqrt(float(np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered)))
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def safe_ks(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    left_arr = left_arr[np.isfinite(left_arr)]
    right_arr = right_arr[np.isfinite(right_arr)]
    if left_arr.size == 0 or right_arr.size == 0:
        return float("nan")
    return float(ks_2samp(left_arr, right_arr).statistic)


def safe_nanmedian(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def compute_z_coherence(
    records: pd.DataFrame,
    *,
    left_col: str = "generated_mean",
    right_col: str = "target_mean",
) -> pd.DataFrame:
    columns = ["class", "gene", "n_slices", "z_min", "z_max", "z_coherence"]
    if records.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (class_name, gene), group in records.groupby(["class", "gene"], sort=True):
        group = group.sort_values("target_z")
        if len(group) < 3:
            continue
        corr = safe_pearson(group[left_col].to_numpy(dtype=float), group[right_col].to_numpy(dtype=float))
        if not np.isfinite(corr):
            continue
        rows.append(
            {
                "class": str(class_name),
                "gene": str(gene),
                "n_slices": int(len(group)),
                "z_min": float(group["target_z"].min()),
                "z_max": float(group["target_z"].max()),
                "z_coherence": float(corr),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_z_coherence_frame(z_coherence_df: pd.DataFrame, *, prefix: str) -> dict[str, Any]:
    if z_coherence_df.empty:
        return {
            f"{prefix}_median": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_n_pairs": 0,
            f"{prefix}_by_class": {},
        }
    z_values = z_coherence_df["z_coherence"].to_numpy(dtype=float)
    finite = z_values[np.isfinite(z_values)]
    return {
        f"{prefix}_median": float(np.median(finite)) if finite.size else float("nan"),
        f"{prefix}_mean": float(np.mean(finite)) if finite.size else float("nan"),
        f"{prefix}_n_pairs": int(len(z_coherence_df)),
        f"{prefix}_by_class": {
            str(class_name): safe_nanmedian(group["z_coherence"].to_numpy(dtype=float))
            for class_name, group in z_coherence_df.groupby("class", sort=True)
        },
    }


def summarize_density(
    *,
    spec: DensitySpec,
    assignments: Sequence[TargetAssignment],
    reference_ids: Sequence[int],
    summary_df: pd.DataFrame,
    z_coherence_df: pd.DataFrame,
    real_baseline_summary: Mapping[str, Any],
    regularized_summary_df: pd.DataFrame | None,
    regularized_z_coherence_df: pd.DataFrame | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "density_gap": int(spec.gap),
        "density_name": spec.name,
        "radius": int(spec.radius),
        "target_start": int(spec.start),
        "n_targets": int(len(assignments)),
        "n_references": int(len(reference_ids)),
        "references": [int(item) for item in reference_ids],
        "target_slices": [int(item.target_id) for item in assignments],
        "ref_z_gap_min": float(min(item.ref_z_gap for item in assignments)) if assignments else float("nan"),
        "ref_z_gap_max": float(max(item.ref_z_gap for item in assignments)) if assignments else float("nan"),
        "tau_mean": float(np.mean([item.tau for item in assignments])) if assignments else float("nan"),
    }
    for column in [
        "mean_corr",
        "var_corr",
        "moran_corr",
        "zero_ks",
        "median_gene_pearson",
        "median_gene_spearman",
    ]:
        values = summary_df[column].to_numpy(dtype=float) if column in summary_df else np.asarray([], dtype=float)
        finite = values[np.isfinite(values)]
        out[f"{column}_median"] = float(np.median(finite)) if finite.size else float("nan")
        out[f"{column}_mean"] = float(np.mean(finite)) if finite.size else float("nan")

    out.update(summarize_z_coherence_frame(z_coherence_df, prefix="z_coherence"))
    out.update(real_baseline_summary)
    baseline = float(out.get("real_split_half_z_coherence_median", float("nan")) or float("nan"))
    generated = float(out.get("z_coherence_median", float("nan")) or float("nan"))
    out["normalized_z_coherence"] = generated / baseline if np.isfinite(generated) and baseline > 0.0 else float("nan")

    if regularized_summary_df is not None and regularized_z_coherence_df is not None:
        for column in [
            "mean_corr",
            "var_corr",
            "moran_corr",
            "zero_ks",
            "median_gene_pearson",
            "median_gene_spearman",
        ]:
            values = regularized_summary_df[column].to_numpy(dtype=float) if column in regularized_summary_df else np.asarray([], dtype=float)
            finite = values[np.isfinite(values)]
            out[f"z_regularized_{column}_median"] = float(np.median(finite)) if finite.size else float("nan")
            out[f"z_regularized_{column}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
        out.update(summarize_z_coherence_frame(regularized_z_coherence_df, prefix="z_regularized_z_coherence"))
        regularized_generated = float(out.get("z_regularized_z_coherence_median", float("nan")) or float("nan"))
        out["z_regularized_normalized_z_coherence"] = (
            regularized_generated / baseline if np.isfinite(regularized_generated) and baseline > 0.0 else float("nan")
        )
    else:
        out["z_regularized_normalized_z_coherence"] = float("nan")
    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return json_ready(float(value))
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    main()
