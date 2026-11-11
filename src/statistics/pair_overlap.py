"""Pairwise Gaussian Bhattacharyya overlap and direction-aware d' between classes in one projected space
(scripts/bhattacharyya_overlap.py, scripts/pca_fda_gbc.py).

``class_gaussians``: per class and covariance form (src.methods.bhattacharyya.KINDS) the mean and unbiased covariance
of the class's specimens, or None if not estimable (fewer than 2 specimens; full covariance needs >= 2 d).
``pair_overlap``: for one class pair, d' of val specimens and of train specimens along the train centroid-difference
direction u; Bhattacharyya mean term, covariance term, distance and coefficient (paper Eq. 2) from the train and from
the val Gaussians for every form; the authors' code-release spherical formula on train (``code``: per-class
stable_transfer statistics); and orientation of each class's full train covariance relative to u (variance along u
over mean variance per dimension; |cos| of the top eigenvector with u).
"""

from __future__ import annotations

import numpy as np

from src.methods.bhattacharyya import KINDS, bhattacharyya_terms, class_covariance, stable_transfer_distance
from src.statistics.class_geometry import direction, direction_separation


def class_gaussians(z: np.ndarray, groups: dict, d: int) -> dict:
    """Per class and covariance form: (mean, covariance) or None if not estimable (full needs >= 2 d specimens)."""
    out = {}
    for c, idx in groups.items():
        x = z[idx]
        out[c] = {k: (x.mean(0), class_covariance(x, k)) if (k != "full" or len(x) >= 2 * d) and len(x) >= 2 else None
                  for k in KINDS}
    return out


def pair_overlap(a, b, g_tr, g_va, code, z_tr, z_va, tr, va, d) -> dict:
    row = {"class_a": a, "class_b": b, "n_train_a": len(tr[a]), "n_train_b": len(tr[b]), "n_val_a": len(va[a]),
           "n_val_b": len(va[b])}
    try:
        u = direction(z_tr[tr[a]], z_tr[tr[b]])
    except ValueError:  # equal train means: d' undefined, reported as such (NaN + flag), never as 0
        u = None
    row["dprime_defined"] = u is not None
    row["dprime_val"] = direction_separation(z_va[va[a]], z_va[va[b]], u) if u is not None else float("nan")
    row["dprime_train"] = direction_separation(z_tr[tr[a]], z_tr[tr[b]], u) if u is not None else float("nan")
    for k in KINDS:
        for split, g in (("train", g_tr), ("val", g_va)):
            p, q = g[a][k], g[b][k]
            key = f"{k}_{split}"
            if p is None or q is None:
                row[f"{key}_valid"] = False
                continue
            try:
                mean_term, cov_term = bhattacharyya_terms(p[0], q[0], p[1], q[1])
            except np.linalg.LinAlgError:
                row[f"{key}_valid"] = False
                continue
            row.update({f"{key}_valid": True, f"{key}_mean_term": mean_term, f"{key}_cov_term": cov_term,
                        f"{key}_distance": mean_term + cov_term, f"{key}_bc": float(np.exp(-(mean_term + cov_term)))})
    dist = stable_transfer_distance(code[a][0], code[b][0], code[a][1], code[b][1], "spherical")
    row.update({"code_spherical_distance": float(dist), "code_spherical_bc": float(np.exp(-dist))})
    for c in (a, b):  # orientation of each class's full train covariance relative to u
        p = g_tr[c]["full"]
        side = "a" if c == a else "b"
        if p is None or u is None:
            continue
        _, vec = np.linalg.eigh(p[1])
        row[f"along_ratio_{side}"] = float(u @ p[1] @ u / (np.trace(p[1]) / d))
        row[f"top_eigvec_cos_{side}"] = float(abs(vec[:, -1] @ u))
    return row
