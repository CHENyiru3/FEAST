"""Sinkhorn transport wrapper using POT (Python Optimal Transport)."""

from __future__ import annotations

import numpy as np
import ot


def sinkhorn_transport(
    M: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float = 0.05,
    numItermax: int = 200,
    stopThr: float = 1e-5,
    unbalanced: bool = False,
    reg_m: float = 5.0,
) -> np.ndarray:
    """Compute entropic-regularized OT plan via POT."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    a = a / a.sum()
    b = b / b.sum()

    if unbalanced:
        plan = ot.sinkhorn_unbalanced(a, b, M, reg, reg_m,
                                       numItermax=numItermax,
                                       stopThr=stopThr)
    else:
        plan = ot.sinkhorn(a, b, M, reg,
                           numItermax=numItermax,
                           stopThr=stopThr)
    return np.asarray(plan, dtype=np.float32)
