"""Software tests for the PCA/FDA-GBC addition: equal-mean Bhattacharyya cases, d' when means coincide, the
mean-term >= d'^2/8 inequality, and the regularised FDA projection (small fixtures only)."""

import numpy as np
import pytest

from src.methods.bhattacharyya import bhattacharyya_terms
from src.methods.fda import fit_fda, scatter_matrices
from src.statistics.class_geometry import direction, direction_separation
from src.statistics.pair_overlap import class_gaussians, pair_overlap


def test_equal_means_orientation_only_and_scale_only_give_bc_0_8():
    """Hand-checked 2-D cases of docs/pca_fda_gbc_method_note.md: D_B = 1/2 ln(6.25/4), BC = 0.8 in both."""
    z = np.zeros(2)
    for ca, cb in ((np.diag([4.0, 1.0]), np.diag([1.0, 4.0])), (np.eye(2), 4 * np.eye(2))):
        mean_term, cov_term = bhattacharyya_terms(z, z, ca, cb)
        assert mean_term == 0 and cov_term == pytest.approx(0.5 * np.log(1.5625))
        assert np.exp(-cov_term) == pytest.approx(0.8)


def test_rotated_covariance_alone_is_detected():
    rot = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
    ca = np.diag([9.0, 0.5])
    cb = rot @ ca @ rot.T  # same eigenvalues, other orientation
    assert np.isclose(np.linalg.det(ca), np.linalg.det(cb))
    assert bhattacharyya_terms(np.zeros(2), np.zeros(2), ca, cb)[1] > 0.1


def test_dprime_is_undefined_not_zero_when_means_coincide():
    x = np.array([[1.0, 0.1], [-1.0, -0.1]])
    y = np.array([[0.1, 2.0], [-0.1, -2.0]])  # same mean (0, 0), different covariance
    with pytest.raises(ValueError, match="undefined"):
        direction(x, y)
    groups = {"a": np.array([0, 1]), "b": np.array([2, 3])}
    z = np.vstack([x, y])
    g = class_gaussians(z, groups, 2)
    code = {c: (z[i].mean(0), z[i].var(0) + 1e-4) for c, i in groups.items()}
    row = pair_overlap("a", "b", g, g, code, z, z, groups, groups, 2)
    assert row["dprime_defined"] is False and np.isnan(row["dprime_val"])
    assert row["diagonal_train_mean_term"] == 0 and row["diagonal_train_cov_term"] > 0  # Bhattacharyya still sees it


def test_mean_term_at_least_dprime_squared_over_8_with_equality_on_an_eigenvector():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=(40, 5)), rng.normal(size=(50, 5)) @ rng.normal(size=(5, 5)) + rng.normal(size=5)
        u = direction(a, b)
        dp = direction_separation(a, b, u)
        mean_term, _ = bhattacharyya_terms(a.mean(0), b.mean(0), np.cov(a, rowvar=False), np.cov(b, rowvar=False))
        assert mean_term >= dp ** 2 / 8 - 1e-12
    a = np.array([[0.0, -1], [0, 1], [2, -1], [2, 1]])  # mean difference along an axis of the shared covariance
    b = a + np.array([3.0, 0.0])
    dp = direction_separation(a, b, direction(a, b))
    mean_term, _ = bhattacharyya_terms(a.mean(0), b.mean(0), np.cov(a, rowvar=False), np.cov(b, rowvar=False))
    assert mean_term == pytest.approx(dp ** 2 / 8)


def test_scatter_decomposition_and_rank():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(90, 6)), np.repeat([0, 1, 2], 30)
    x[y == 1] += 2.0
    s_b, s_w, mu, resid = scatter_matrices(x, y)
    assert np.allclose(s_b + s_w, np.cov(x, rowvar=False, bias=True))  # biased total = between + within
    assert np.linalg.matrix_rank(s_b) <= 2 and np.allclose(resid.mean(0), 0)


def test_fda_finds_the_low_variance_discriminative_axis_that_pca_misses():
    """Classes differ only along a low-variance axis; a high-variance axis is pure noise. PCA-1 keeps the noise axis,
    FDA-1 the discriminative one."""
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1, 2], 200)
    x = np.column_stack([rng.normal(0, 10, 600), y * 1.0 + rng.normal(0, 0.2, 600), rng.normal(0, 1, (600, 3))])
    fda = fit_fda(x, y)
    w = fda.basis(1)[:, 0]
    assert abs(w[1]) > 0.99 and fda.rank == 2
    assert np.allclose(fda.basis(2).T @ fda.basis(2), np.eye(2))
    raw = fda.raw(2)
    assert np.allclose(np.linalg.matrix_rank(np.hstack([raw, fda.basis(2)])), 2)  # same subspace
    assert 0 <= fda.alpha <= 1 and fda.condition_s_w >= 1
    assert fit_fda(x, y, alpha_scale=1e6).alpha == 1.0  # capped
    z = fda.project(x, 1)
    assert np.isclose(z.mean(), 0, atol=1e-9)  # centred on the training mean


def test_fda_generalised_eigen_equation():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(120, 4)), np.repeat([0, 1, 2, 3], 30)
    x[y == 2, 0] += 3
    fda = fit_fda(x, y)
    s_b, s_w, _, _ = scatter_matrices(x, y)
    s_w_star = (1 - fda.alpha) * s_w + fda.alpha * np.trace(s_w) / 4 * np.eye(4)
    v, lam = fda.vectors[:, 0], fda.eigenvalues[0]
    assert np.allclose(s_b @ v, lam * s_w_star @ v) and np.isclose(v @ s_w_star @ v, 1)
