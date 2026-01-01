import numpy as np

from hql.objectives.synthetic_fourier import SyntheticFourierObjective
from hql.smoothing.gaussian import GaussianSmoother


def test_stage1_minimizer_has_zero_y_gradient():
    obj = SyntheticFourierObjective()
    theta = np.array([0.0, 0.0])
    grad = obj.grad(theta)
    assert np.isclose(grad[1], 0.0, atol=1e-12)


def test_smoothed_grad_matches_value_finite_diff():
    obj = SyntheticFourierObjective()
    sm = GaussianSmoother(obj, sigma=0.2, n_samples=64, torus=True)
    theta = np.array([0.4, -1.1])

    rng = np.random.default_rng(0)
    manual = []
    for _ in range(sm.n_samples):
        z = rng.standard_normal(size=theta.shape)
        manual.append(sm.base_grad_point(theta, z))

    manual_grad = np.mean(np.stack(manual, axis=0), axis=0)
    sm_grad = sm.grad(theta, seed=0)
    assert np.allclose(sm_grad, manual_grad, atol=1e-6)
