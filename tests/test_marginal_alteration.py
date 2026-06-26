import numpy as np

from FEAST.modeling.StudentT_mixture_model import StudentTMixtureMarginalModeler
from FEAST.modeling.marginal_alteration import AlterationConfig, alter_marginal_model


def test_interpolated_studentt_ppf_refreshes_after_mean_alteration():
    data = np.geomspace(0.1, 100.0, 120)
    model = StudentTMixtureMarginalModeler(max_components=1, ppf_method="interp")
    model.fit(data, log_transform=True, visualize=False)

    q = np.array([0.25, 0.50, 0.75, 0.99])
    before = model.ppf(q)
    assert hasattr(model, "_ppf_interp")

    alter_marginal_model(
        model,
        mean_fold_change=2.0,
        variance_fold_change=1.0,
        dispersion_strength=0.0,
        preserve_original=False,
        verbose=False,
    )

    assert not hasattr(model, "_ppf_interp")
    after = model.ppf(q)

    np.testing.assert_allclose(after[1:3] / before[1:3], 2.0, rtol=0.15)
    assert after[-1] > model.data_range[1]


def test_deprecated_sparsity_fold_change_maps_to_logit_shift():
    decreased = AlterationConfig.sparsity_only(fold_change=0.5)
    increased = AlterationConfig.sparsity_only(fold_change=2.0)

    assert decreased.apply_to_zero_prop
    assert increased.apply_to_zero_prop
    assert decreased.sparsity_logit_shift < 0
    assert increased.sparsity_logit_shift > 0
    np.testing.assert_allclose(decreased.sparsity_logit_shift, np.log(0.5))
    np.testing.assert_allclose(increased.sparsity_logit_shift, np.log(2.0))
