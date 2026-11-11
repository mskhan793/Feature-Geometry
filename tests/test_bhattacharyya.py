"""Numerical tests of src/methods/bhattacharyya.py (software fixtures only)."""

import numpy as np
import pytest

from src.methods.bhattacharyya import (bhattacharyya_terms, class_covariance, gbc_score, stable_transfer_class_stats,
                                       stable_transfer_distance)


def test_identical_gaussians_have_coefficient_one_and_symmetry():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 4))
    mu, cov = a.mean(0), class_covariance(a, "full")
    assert bhattacharyya_terms(mu, mu, cov, cov) == pytest.approx((0.0, 0.0), abs=1e-12)
    b = rng.normal(1.0, 2.0, size=(80, 4))
    mb, cb = b.mean(0), class_covariance(b, "full")
    assert bhattacharyya_terms(mu, mb, cov, cb) == pytest.approx(bhattacharyya_terms(mb, mu, cb, cov), rel=1e-12)


def test_one_dimensional_closed_form():
    """D_B for N(m1, v1), N(m2, v2) = (m1 - m2)^2 / (4 (v1 + v2)) + 1/2 ln((v1 + v2) / (2 sqrt(v1 v2)))."""
    m1, m2, v1, v2 = 0.3, 2.1, 0.5, 3.0
    mean_term, cov_term = bhattacharyya_terms(np.array([m1]), np.array([m2]), np.array([[v1]]), np.array([[v2]]))
    assert mean_term == pytest.approx((m1 - m2) ** 2 / (4 * (v1 + v2)))
    assert cov_term == pytest.approx(0.5 * np.log((v1 + v2) / (2 * np.sqrt(v1 * v2))))


def test_coefficient_equals_overlap_integral_in_two_dimensions():
    """BC = exp(-D_B) equals the integral of sqrt(p q) for two correlated 2-D Gaussians (grid integration)."""
    from scipy.stats import multivariate_normal
    m1, m2 = np.array([0.0, 0.0]), np.array([1.5, -0.5])
    c1, c2 = np.array([[1.0, 0.6], [0.6, 1.0]]), np.array([[2.0, -0.3], [-0.3, 0.5]])
    g = np.linspace(-8, 9, 701)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    integral = np.sqrt(multivariate_normal(m1, c1).pdf(pts) * multivariate_normal(m2, c2).pdf(pts)).sum() * (g[1] - g[0]) ** 2
    assert np.exp(-sum(bhattacharyya_terms(m1, m2, c1, c2))) == pytest.approx(integral, rel=1e-6)


def test_orientation_changes_overlap_at_equal_variance_magnitude():
    """Same eigenvalues (so same total variance and rank), different orientation relative to the mean difference:
    spread along the separating direction overlaps more than spread across it."""
    mu1, mu2 = np.zeros(2), np.array([3.0, 0.0])
    along, across = np.diag([4.0, 0.25]), np.diag([0.25, 4.0])
    bc_along = np.exp(-sum(bhattacharyya_terms(mu1, mu2, along, along)))
    bc_across = np.exp(-sum(bhattacharyya_terms(mu1, mu2, across, across)))
    assert bc_along > 0.5 > bc_across


def test_covariance_forms_are_unbiased_and_spherical_uses_mean_variance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 3)) * np.array([1.0, 2.0, 3.0])
    full = class_covariance(x, "full")
    assert np.allclose(full, np.cov(x, rowvar=False, ddof=1))
    assert np.allclose(np.diag(class_covariance(x, "diagonal")), x.var(0, ddof=1))
    assert np.allclose(class_covariance(x, "spherical"), np.eye(3) * np.trace(full) / 3)


def test_spherical_paper_term_is_d_times_the_code_term():
    """For spherical Gaussians the paper's covariance term (Eq. 2) is d times the authors' code term (same variances,
    no floor binding): the documented difference between the two."""
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1.0, (200, 16)), rng.normal(0.5, 2.0, (150, 16))
    va, vb = a.var(0), b.var(0)  # population variance, as in the code
    code = stable_transfer_distance(a.mean(0), b.mean(0), va, vb, "spherical")
    code_mean = np.sum((a.mean(0) - b.mean(0)) ** 2 / ((va.mean() + vb.mean()) / 2)) / 8
    mean_term, cov_term = bhattacharyya_terms(a.mean(0), b.mean(0), np.eye(16) * va.mean(), np.eye(16) * vb.mean())
    assert mean_term == pytest.approx(code_mean) and cov_term == pytest.approx(16 * (code - code_mean))


def test_code_port_floor_and_population_variance():
    x = np.array([[1.0, 5.0], [3.0, 5.0]], np.float32)
    mu, var = stable_transfer_class_stats(x)
    assert var.dtype == np.float32 and var.tolist() == pytest.approx([1.0, 1e-4])  # population variance, floored


def test_gbc_counts_ordered_pairs_and_singular_covariance_raises():
    bc = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.0], [0.1, 0.0, 1.0]])
    assert gbc_score(bc) == pytest.approx(-0.6)
    with pytest.raises(np.linalg.LinAlgError):
        bhattacharyya_terms(np.zeros(2), np.ones(2), np.zeros((2, 2)), np.eye(2))
