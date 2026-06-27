"""Spatial coordinate transforms for alignment benchmarking.

Usage::

    from FEAST import spatial_transform as st

    rotated = st.rotate(coords, angle=30)
    warped = st.warp(coords, strength=0.5)
"""

from __future__ import annotations

import numpy as np


def rotate(
    coords: np.ndarray,
    angle: float,
    *,
    center_correction: float = 0.0,
) -> np.ndarray:
    """Rotate 2-D coordinates by *angle* degrees.

    Parameters
    ----------
    coords:
        (N, 2) array of (x, y) coordinates.
    angle:
        Rotation angle in degrees (counter-clockwise).
    center_correction:
        Shift applied before and after rotation, in coordinate units.

    Returns
    -------
    (N, 2) rotated coordinates.
    """
    theta = np.radians(angle)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    centered = coords + center_correction
    rotated = centered @ R.T
    return rotated - center_correction


def warp(
    coords: np.ndarray,
    strength: float,
    *,
    grid_size: int = 3,
    alpha: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Warp 2-D coordinates with thin-plate-spline deformation.

    Parameters
    ----------
    coords:
        (N, 2) array of (x, y) coordinates.
    strength:
        Deformation magnitude (0 = no change, higher = more distortion).
    grid_size:
        Control-point grid size (grid_size × grid_size control points).
    alpha:
        TPS smoothing parameter (higher = stiffer transform).
    seed:
        Random seed for reproducible deformation.

    Returns
    -------
    (N, 2) warped coordinates.
    """
    from tps import ThinPlateSpline

    rng = np.random.RandomState(seed)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1

    src_x = np.linspace(x_min - pad_x, x_max + pad_x, grid_size)
    src_y = np.linspace(y_min - pad_y, y_max + pad_y, grid_size)
    src_xx, src_yy = np.meshgrid(src_x, src_y)
    src_pts = np.column_stack([src_xx.ravel(), src_yy.ravel()])

    displacement = strength * (x_max - x_min)
    noise_x = rng.uniform(-displacement, displacement, size=src_pts.shape[0])
    noise_y = rng.uniform(-displacement, displacement, size=src_pts.shape[0])
    dst_pts = src_pts + np.column_stack([noise_x, noise_y])

    tps = ThinPlateSpline(alpha)
    tps.fit(src_pts, dst_pts)
    return tps.transform(coords)
