"""Ledoit-Wolf shrinkage against the paper's formulas (Lemmas 3.2-3.5, Eq. 14) and the held-out log-likelihood
(software fixtures only)."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from src.statistics.covariance_fit import gaussian_loglik, lw_alpha, shrunk


def paper_lw(x):
    """Ledoit & Wolf (2004), Lemmas 3.2-3.5 and Eq. 14, written out; norm ||A||^2 = tr(A A^T)/p."""
    n, p = x.shape
    s = x.T @ x / n
    norm2 = lambda a: np.trace(a @ a.T) / p  # noqa: E731
    m = np.trace(s) / p
    d2 = norm2(s - m * np.eye(p))
    bbar2 = sum(norm2(np.outer(xk, xk) - s) for xk in x) / n ** 2
    b2 = min(bbar2, d2)
    return b2 / d2, (b2 / d2) * m * np.eye(p) + (1 - b2 / d2) * s


@pytest.mark.parametrize("n,p", [(40, 10), (12, 30), (200, 5)])
def test_lw_alpha_and_estimator_match_the_paper(n, p):
    rng = np.random.default_rng(n + p)
    x = rng.normal(size=(n, p)) * np.linspace(3, 0.2, p)
    alpha, s_star = paper_lw(x)
    assert lw_alpha(x) == pytest.approx(alpha, rel=1e-10)
    assert np.allclose(shrunk(x.T @ x / n, lw_alpha(x)), s_star)


def test_gaussian_loglik_matches_scipy():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 4))
    cov, mean, x = a @ a.T + np.eye(4), rng.normal(size=4), rng.normal(size=(7, 4))
    assert np.allclose(gaussian_loglik(x, mean, cov), multivariate_normal(mean, cov).logpdf(x))
