"""Synthetic checks of class geometry and modality helpers (software tests only, not research data)."""

import numpy as np

from src.statistics.class_geometry import (direction, direction_separation, eigen_summary, radius_separation,
                                           subspace_overlap)
from src.statistics.class_modality import eligible, free_parameters, split_stability


def test_eigen_summary_isotropic_vs_elongated():
    rng = np.random.default_rng(0)
    iso = rng.normal(size=(2000, 10))
    elong = iso * np.array([10.0] + [1.0] * 9)
    a, b = eigen_summary(iso), eigen_summary(elong)
    assert 9 < a["effective_rank"] <= 10 and 9 < a["participation_ratio"] <= 10
    assert b["top1_share"] > 0.9 and b["participation_ratio"] < 1.5
    assert np.isclose(a["rms_radius"], np.sqrt(((iso - iso.mean(0)) ** 2).sum(1).mean()))


def test_separations():
    rng = np.random.default_rng(1)
    xa = rng.normal(size=(500, 50))
    xb = rng.normal(size=(500, 50))
    xb[:, 0] += 4.0
    assert 0.4 < radius_separation(xa, xb) < 0.7  # 4 / sqrt(50): looks close by radius
    d = direction_separation(xa, xb, direction(xa, xb))
    assert 3.5 < d < 4.5  # but is well separated along the centroid direction


def test_subspace_overlap_bounds():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(400, 6)) * np.array([5, 4, 3, 1, 1, 1])
    assert subspace_overlap(base, base, 3) > 0.999
    other = base[:, ::-1]
    assert subspace_overlap(base, other, 3) < 0.05


def test_parameter_counts_and_eligibility():
    assert free_parameters(1, 8, "full") == 44
    assert free_parameters(2, 8, "full") == 89
    assert free_parameters(2, 8, "tied") == 53
    assert free_parameters(3, 8, "tied") == 62
    assert eligible(2, 8, "full", 30) and not eligible(3, 8, "full", 30)


def test_split_stability_two_clusters_vs_one():
    rng = np.random.default_rng(3)
    two = np.vstack([rng.normal(-4, 1, (40, 2)), rng.normal(4, 1, (40, 2))])
    val = np.vstack([rng.normal(-4, 1, (10, 2)), rng.normal(4, 1, (10, 2))])
    s = split_stability(two, val, 2, "tied", 1e-3, 0, 30)
    assert s["ari_val_median"] > 0.95 and s["val_components_used"] == 2


def test_specimen_means_are_float64():
    import pandas as pd

    from src.features.specimen_means import specimen_camera_means, specimen_means
    cohort = pd.DataFrame({"taxon_group": "A", "fold0": "train", "individual": ["s", "s", "s"], "camera": [1, 1, 2]})
    x = np.array([[1.0], [3.0], [5.0]], np.float32)
    keys, means, _ = specimen_camera_means(x, cohort)
    sk, sm = specimen_means(keys, means)
    assert means.dtype == np.float64 and sm.dtype == np.float64
    assert means[:, 0].tolist() == [2.0, 5.0] and sm[0, 0] == 3.5


def test_fit_mixture_never_worse_than_kmeans_starts_and_complete_axis_starts():
    """The extra principal-direction starts can only raise the training likelihood bound of the kept fit."""
    from sklearn.mixture import GaussianMixture

    from src.statistics.class_modality import axis_split_starts, fit_mixture
    rng = np.random.default_rng(5)
    z = rng.normal(size=(30, 8)) * np.linspace(2, 0.5, 8)
    z[:15, 3] += 1.5
    for cov in ("full", "tied", "diag"):
        for seed in (0, 20261107):
            km = GaussianMixture(2, covariance_type=cov, reg_covar=1e-3, n_init=5, max_iter=1000, random_state=seed).fit(z)
            assert fit_mixture(z, 2, cov, 1e-3, seed).lower_bound_ >= km.lower_bound_ - 1e-9
        starts = axis_split_starts(z, 2, cov, 1e-3)
        assert len(starts) == 8 and all(np.isclose(s["weights_init"].sum(), 1) for s in starts)
