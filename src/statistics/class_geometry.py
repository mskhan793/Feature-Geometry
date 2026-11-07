"""Descriptive geometry of classes of embeddings (rows = independent specimens).

- ``eigen_summary``: sample-covariance eigenvalues, total variance, RMS radius, share of the leading eigenvalue,
  effective rank ``exp(entropy of eigenvalue shares)`` and participation ratio ``(sum l)^2 / sum l^2``. Both ranks
  are bounded by ``n - 1`` and depend on ``n``; compare them only between classes with equal ``n``.
- ``radius_separation``: centroid distance divided by the classes' RMS radius (isotropic; ignores direction).
- ``direction_separation``: separation along the centroid-difference direction ``u`` (d'): difference of the mean
  projections over the root mean of their variances. ``u`` comes from one set (train) and may be applied to another
  (val) so the separation is not measured on the data that chose the direction.
- ``subspace_overlap``: ``||U_a^T U_b||_F^2 / k`` for the top-``k`` principal directions (0 = orthogonal, 1 = same).
"""

from __future__ import annotations

import numpy as np


def principal_axes(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending, ddof 1) and the matching unit eigenvectors (columns) of the sample covariance."""
    xc = x - x.mean(0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    return s ** 2 / (len(x) - 1), vt.T


def eigen_summary(x: np.ndarray) -> dict:
    lam, _ = principal_axes(x)
    lam = lam[lam > lam[0] * 1e-12]
    p = lam / lam.sum()
    return {"n": len(x), "total_variance": float(lam.sum()), "rms_radius": float(np.sqrt(lam.sum() * (len(x) - 1) / len(x))),
            "top1_share": float(p[0]), "top5_share": float(p[:5].sum()), "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
            "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum())}


def radius_separation(xa: np.ndarray, xb: np.ndarray) -> float:
    ra2 = ((xa - xa.mean(0)) ** 2).sum(1).mean()
    rb2 = ((xb - xb.mean(0)) ** 2).sum(1).mean()
    return float(np.linalg.norm(xa.mean(0) - xb.mean(0)) / np.sqrt((ra2 + rb2) / 2))


def direction(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    d = xa.mean(0) - xb.mean(0)
    return d / np.linalg.norm(d)


def direction_separation(xa: np.ndarray, xb: np.ndarray, u: np.ndarray) -> float:
    pa, pb = xa @ u, xb @ u
    return float(abs(pa.mean() - pb.mean()) / np.sqrt((pa.var(ddof=1) + pb.var(ddof=1)) / 2))


def subspace_overlap(xa: np.ndarray, xb: np.ndarray, k: int) -> float:
    ua, ub = principal_axes(xa)[1][:, :k], principal_axes(xb)[1][:, :k]
    return float(((ua.T @ ub) ** 2).sum() / k)
