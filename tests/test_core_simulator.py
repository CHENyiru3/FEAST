import numpy as np
import pandas as pd
import anndata as ad
import pytest

from FEAST.FEAST_core.simulator import SpatialSimulator, simulate_single_slice


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


def test_nonzero_sigma_rejected():
    simulator = SpatialSimulator(_adata(), model_params=_model_params())
    with pytest.raises(ValueError, match="G-SRBA/G-SBGA"):
        simulator.simulate(sigma=0.5, verbose=False)
    with pytest.raises(ValueError, match="G-SRBA/G-SBGA"):
        simulate_single_slice(_adata(), sigma=1.0, verbose=False)


def test_zero_sigma_deterministic_path_with_model_params():
    simulator = SpatialSimulator(_adata(), model_params=_model_params())
    simulated = simulator.simulate(sigma=0, verbose=False)
    assert simulated.shape == (3, 2)
    assert "spatial" in simulated.obsm
    assert simulated.uns["simulation_method"] == "Deterministic_Rank_Preservation"
