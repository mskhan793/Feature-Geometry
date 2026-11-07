"""Per-class density models for the camera-conditioned Gaussian pilot (fitted in a low-dimensional projection).

Models of one class, fitted on its training observations ``Z`` (rows) with camera labels ``cameras``:
- ``A_pooled``: one Gaussian, cameras pooled.
- ``B_camera_marginal``: one Gaussian per camera, mixed with the training camera proportions; unconditional density.
- ``B_camera_conditional``: one Gaussian per camera, each observation scored by its own camera's Gaussian; this model
  sees the camera label and is reported separately from the unconditional models.
- ``C_gmm2`` / ``C_gmm3``: K-component Gaussian mixtures fitted without camera labels.

All Gaussians use sklearn ``GaussianMixture`` with the same covariance type, ``reg_covar`` and seed.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

UNCONDITIONAL = ("A_pooled", "B_camera_marginal", "C_gmm2", "C_gmm3")


def gaussian_mixture(Z: np.ndarray, k: int, covariance_type: str, reg: float, seed: int) -> GaussianMixture:
    gm = GaussianMixture(n_components=k, covariance_type=covariance_type, reg_covar=reg, n_init=5, max_iter=1000,
                         random_state=seed).fit(Z)
    if not gm.converged_:
        raise RuntimeError(f"K={k} mixture did not converge")
    return gm


def fit_class_models(Z: np.ndarray, cameras: np.ndarray, ks: list[int], covariance_type: str, reg: float,
                     seed: int) -> tuple[dict, dict]:
    """Fitted models by name, and fit failures by name (message). ``ks`` lists the mixture sizes to fit (2, 3)."""
    models, failures = {}, {}
    try:
        models["A_pooled"] = gaussian_mixture(Z, 1, covariance_type, reg, seed)
        per_camera = {c: gaussian_mixture(Z[cameras == c], 1, covariance_type, reg, seed) for c in np.unique(cameras)}
        weights = {c: float(np.mean(cameras == c)) for c in per_camera}
        models["B_camera"] = (per_camera, weights)
    except RuntimeError as e:
        failures["A_or_B"] = str(e)
    for k in ks:
        try:
            models[f"C_gmm{k}"] = gaussian_mixture(Z, k, covariance_type, reg, seed)
        except RuntimeError as e:
            failures[f"C_gmm{k}"] = str(e)
    return models, failures


def log_densities(models: dict, Z: np.ndarray, cameras: np.ndarray) -> dict[str, np.ndarray]:
    """Log density of every row of ``Z`` under every fitted model."""
    out = {name: m.score_samples(Z) for name, m in models.items() if name != "B_camera"}
    if "B_camera" in models:
        per_camera, weights = models["B_camera"]
        logs = {c: g.score_samples(Z) for c, g in per_camera.items()}
        out["B_camera_marginal"] = logsumexp(np.stack([np.log(weights[c]) + logs[c] for c in logs]), axis=0)
        cond = np.full(len(Z), np.nan)
        for c in logs:
            cond[cameras == c] = logs[c][cameras == c]
        out["B_camera_conditional"] = cond
    return out


def component_camera_purity(components: np.ndarray, cameras: np.ndarray) -> float:
    """Best agreement between a 2-component assignment and 2 cameras over both label matchings (0.5 = none)."""
    c0 = cameras == np.unique(cameras)[0]
    match = np.mean((components == 0) == c0)
    return float(max(match, 1 - match))
