import inspect

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from FEAST.FEAST_core.simulator import SpatialSimulator, simulate_single_slice
from FEAST.FEAST_core.APIs import FEAST
from FEAST.alignment import simulate_alignment_rotation, simulate_alignment_warp
from FEAST.deconvolution import (
    DeconvolutionSimulator,
    create_deconvolution_benchmark_suite,
    simulate_deconvolution_from_single_cells,
)


def _adata():
    X = np.array([[1, 0], [2, 1], [0, 3]], dtype=np.int32)
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=["s1", "s2", "s3"]), var=pd.DataFrame(index=["g1", "g2"]))
    adata.obsm["spatial"] = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    return adata


def _model_params():
    return {
        "model_selected": ["Poisson", "NB"],
        "marginal_param1": [[0.0, 1.0, 2.0], [0.0, 2.0, 3.0]],
    }


def test_sigma_parameters_removed_from_public_core_api():
    public_callables = [
        SpatialSimulator.simulate,
        simulate_single_slice,
        FEAST.simulate_single_slice,
        FEAST.simulate_alignment,
        simulate_alignment_rotation,
        simulate_alignment_warp,
        DeconvolutionSimulator.simulate_deconvolution_data,
        DeconvolutionSimulator.create_deconvolution_benchmark_suite,
        simulate_deconvolution_from_single_cells,
        create_deconvolution_benchmark_suite,
    ]
    for fn in public_callables:
        assert "sigma" not in inspect.signature(fn).parameters
        assert "follower_sigma_factor" not in inspect.signature(fn).parameters

    simulator = SpatialSimulator(_adata(), model_params=_model_params())
    with pytest.raises(TypeError):
        simulator.simulate(sigma=0.5, verbose=False)
    with pytest.raises(TypeError):
        simulate_single_slice(_adata(), sigma=1.0, verbose=False)


def test_deterministic_path_with_model_params():
    simulator = SpatialSimulator(_adata(), model_params=_model_params())
    simulated = simulator.simulate(verbose=False)
    assert simulated.shape == (3, 2)
    assert "spatial" in simulated.obsm
    assert simulated.uns["simulation_method"] == "Deterministic_Rank_Preservation"
