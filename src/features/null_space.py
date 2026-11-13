"""Numerical rank of a feature matrix and the affine constraint of a final LayerNorm (2026-11-12 audit).

LayerNorm output x = w * (h - mean(h)) / std(h) + b satisfies sum_j (x_j - b_j) / w_j = 0 for every vector, so every
covariance of such features is singular along v = (1/w) / ||1/w||. ``spectrum`` reports the covariance eigenvalues
(computed in float64) and the numerical rank; ``layernorm_constraint`` the constraint value relative to the row norm
and |cos| between the smallest-eigenvalue eigenvector and v."""

from __future__ import annotations

import numpy as np

RANK_RTOL = 1e-12


def spectrum(x: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    cov = np.cov(x, rowvar=False)
    eig, vec = np.linalg.eigh(cov)
    out = {"rows": int(len(x)), "dim": int(x.shape[1]), "eigenvalues_smallest_4": eig[:4].tolist(),
           "eigenvalue_largest": float(eig[-1]), "smallest_over_second_smallest": float(eig[0] / eig[1]),
           "numerical_rank_1e-12": int((eig > RANK_RTOL * eig[-1]).sum()), "rank_measurable": bool(len(x) > x.shape[1])}
    return out, cov, vec


def layernorm_constraint(x: np.ndarray, w: np.ndarray, b: np.ndarray, cov: np.ndarray, vec: np.ndarray) -> dict:
    x, w, b = (np.asarray(t, dtype=np.float64) for t in (x, w, b))
    assert w.shape == (x.shape[1],), (w.shape, x.shape)
    constraint = ((x - b) / w).sum(1)
    rel = np.abs(constraint) / np.linalg.norm((x - b) / w, axis=1)
    v = (1 / w) / np.linalg.norm(1 / w)
    return {"constraint_rel_abs_max": float(rel.max()), "constraint_rel_abs_median": float(np.median(rel)),
            "abs_cos_smallest_eigvec_vs_inv_w": float(abs(vec[:, 0] @ v)), "variance_along_inv_w": float(v @ cov @ v)}
