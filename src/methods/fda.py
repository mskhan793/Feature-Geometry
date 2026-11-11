"""Regularised Fisher Discriminant Analysis projection (Mensink et al., TPAMI 2013, Sect. 3.2, Eq. 8; choices in
docs/pca_fda_gbc_method_note.md).

Scatter (Eq. 8 weights, biased class covariances, our choice):
    S_B = sum_c (N_c/N) (mu - mu_c)(mu - mu_c)^T,   S_W = sum_c (N_c/N) Sigma_c = (1/N) sum_i (x_i - mu_{y_i})(...)^T
Shrinkage (Ledoit-Wolf intensity alpha from the within-class residuals, our choice; ``alpha_scale`` only for labelled
sensitivity): S_W* = (1 - alpha) S_W + alpha (tr S_W / D) I.
Directions: generalised symmetric-definite eigenproblem S_B v = lambda S_W* v (scipy.linalg.eigh, no explicit inverse),
eigenvalues in decreasing order; at most C - 1 are non-zero. ``basis`` = orthonormal basis (QR) of the top-d
directions (main projection convention); ``raw`` = the top-d eigenvectors scaled so that v^T S_W* v = 1 (sensitivity).
Projections are centred on the training mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, eigh
from sklearn.covariance import ledoit_wolf_shrinkage

RANK_TOL = 1e-10


def scatter_matrices(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """S_B, S_W, the data mean and the within-class residuals x_i - mu_{y_i} (float64)."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(0)
    resid = np.empty_like(x)
    s_b = np.zeros((x.shape[1], x.shape[1]))
    for c in np.unique(y):
        m = y == c
        mu_c = x[m].mean(0)
        resid[m] = x[m] - mu_c
        s_b += m.mean() * np.outer(mu - mu_c, mu - mu_c)
    return s_b, resid.T @ resid / len(x), mu, resid


@dataclass
class FDA:
    mean: np.ndarray
    eigenvalues: np.ndarray       # all D generalised eigenvalues, decreasing
    vectors: np.ndarray           # D x D generalised eigenvectors (columns), v^T S_W* v = 1
    alpha: float                  # shrinkage intensity used
    alpha_ledoit_wolf: float      # Ledoit-Wolf estimate before ``alpha_scale``
    condition_s_w: float          # condition number of S_W*
    rank: int                     # numerically non-zero eigenvalues (> RANK_TOL x largest)

    def raw(self, d: int) -> np.ndarray:
        return self.vectors[:, :d]

    def basis(self, d: int) -> np.ndarray:
        q, _ = np.linalg.qr(self.vectors[:, :d])
        return q

    def project(self, x: np.ndarray, d: int, convention: str = "orthonormal") -> np.ndarray:
        w = self.basis(d) if convention == "orthonormal" else self.raw(d)
        return (np.asarray(x, dtype=np.float64) - self.mean) @ w


def fit_fda(x: np.ndarray, y: np.ndarray, alpha_scale: float = 1.0) -> FDA:
    s_b, s_w, mu, resid = scatter_matrices(x, y)
    alpha_lw = float(ledoit_wolf_shrinkage(resid, assume_centered=True))
    alpha = min(1.0, alpha_lw * alpha_scale)
    d = s_w.shape[0]
    s_w_star = (1 - alpha) * s_w + alpha * np.trace(s_w) / d * np.eye(d)
    cholesky(s_w_star, lower=True)  # raises if not positive definite
    lam, vec = eigh(s_b, s_w_star)
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    return FDA(mean=mu, eigenvalues=lam, vectors=vec, alpha=alpha, alpha_ledoit_wolf=alpha_lw,
               condition_s_w=float(np.linalg.cond(s_w_star)), rank=int((lam > RANK_TOL * lam[0]).sum()))
