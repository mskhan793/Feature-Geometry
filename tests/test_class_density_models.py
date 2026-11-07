"""Synthetic checks of the pilot density models (software tests only, not research data)."""

import numpy as np
from scipy.stats import multivariate_normal

from src.statistics.class_density_models import component_camera_purity, fit_class_models, log_densities


def two_cameras(shift, n=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    cameras = np.repeat([1, 2], n)
    Z = rng.normal(size=(2 * n, d))
    Z[cameras == 2, 0] += shift
    return Z, cameras


def test_marginal_is_weighted_mixture_of_camera_gaussians():
    Z, cam = two_cameras(4.0)
    models, failures = fit_class_models(Z, cam, [2], "diag", 1e-6, 0)
    assert not failures
    per_camera, w = models["B_camera"]
    x = np.array([[1.0, 0.0, 0.0]])
    manual = np.log(sum(w[c] * multivariate_normal(g.means_[0], np.diag(g.covariances_[0])).pdf(x) for c, g in per_camera.items()))
    assert np.isclose(log_densities(models, x, np.array([1]))["B_camera_marginal"][0], manual)


def test_camera_shift_favours_camera_models_and_mixture_finds_cameras():
    Z, cam = two_cameras(6.0)
    Zv, camv = two_cameras(6.0, seed=1)
    models, _ = fit_class_models(Z, cam, [2], "diag", 1e-6, 0)
    ld = {k: v.mean() for k, v in log_densities(models, Zv, camv).items()}
    assert ld["B_camera_marginal"] > ld["A_pooled"] + 0.3  # expected gain log(sqrt(10)) - log(2) = 0.46
    assert ld["B_camera_conditional"] > ld["B_camera_marginal"]
    assert component_camera_purity(models["C_gmm2"].predict(Zv), camv) > 0.95


def test_no_shift_gives_no_camera_advantage():
    Z, cam = two_cameras(0.0)
    Zv, camv = two_cameras(0.0, seed=1)
    models, _ = fit_class_models(Z, cam, [2], "diag", 1e-6, 0)
    ld = {k: v.mean() for k, v in log_densities(models, Zv, camv).items()}
    assert ld["B_camera_marginal"] <= ld["A_pooled"] + 0.05


def test_purity_bounds():
    cam = np.array([1, 1, 2, 2])
    assert component_camera_purity(np.array([1, 1, 0, 0]), cam) == 1.0
    assert component_camera_purity(np.array([0, 1, 0, 1]), cam) == 0.5
