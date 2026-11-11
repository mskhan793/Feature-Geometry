"""Ledoit-Wolf shrinkage and held-out Gaussian log-likelihood (Ledoit & Wolf, *A Well-Conditioned Estimator for
Large-Dimensional Covariance Matrices*, J. Multivariate Analysis 2004; preprint Feb 2001,
~/Documents/Research/methods/ledoit_wolf2004/paper.pdf, SHA256 08e79244...).

Paper (Lemmas 3.2-3.5, Eq. 14, p. 12), for n observations x_k (mean zero) of dimension p, S = (1/n) sum_k x_k x_k^T and
the norm ||A||^2 = tr(A A^T)/p:
    m = tr(S)/p,  d^2 = ||S - m I||^2,  bbar^2 = (1/n^2) sum_k ||x_k x_k^T - S||^2,  b^2 = min(bbar^2, d^2),
    S* = (b^2/d^2) m I + (1 - b^2/d^2) S,
i.e. shrinkage intensity alpha = b^2/d^2 toward the average variance m. ``lw_alpha`` uses sklearn's
``ledoit_wolf_shrinkage`` (checked against these formulas in tests/test_covariance_fit.py). The observations here are
residuals around estimated (class) means, not known-mean data as in the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.covariance import ledoit_wolf_shrinkage


def lw_alpha(resid: np.ndarray) -> float:
    """Ledoit-Wolf intensity b^2/d^2 for the residual rows (treated as centred)."""
    return float(ledoit_wolf_shrinkage(resid, assume_centered=True))


def shrunk(cov: np.ndarray, alpha: float) -> np.ndarray:
    """(1 - alpha) cov + alpha (tr cov / p) I."""
    p = cov.shape[0]
    return (1 - alpha) * cov + alpha * np.trace(cov) / p * np.eye(p)


def gaussian_loglik(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of each row of x under N(mean, cov) (Cholesky; raises if cov is not positive definite)."""
    c, low = cho_factor(cov, lower=True)
    r = x - mean
    maha = np.einsum("ij,ij->i", r, cho_solve((c, low), r.T).T)
    logdet = 2 * np.log(np.diag(c)).sum()
    return -0.5 * (maha + logdet + cov.shape[0] * np.log(2 * np.pi))
