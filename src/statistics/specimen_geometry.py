"""Native-space specimen geometry per class and per class pair (used by scripts/class_geometry_pilot.py on fold
train/val specimens and by scripts/descriptive_geometry.py on a pooled cohort).

Inputs are specimen embeddings (one row per specimen, keys with ``taxon_group`` and ``individual``) and camera-level
means (keys with ``taxon_group``, ``individual``, ``camera`` and ``row``, the row in the camera-mean array)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.data.expanded_cohort import stable_order
from src.statistics.class_geometry import (direction, direction_separation, eigen_summary, radius_separation,
                                           subspace_overlap)


def reidentification(a: np.ndarray, b: np.ndarray, chunk: int = 1024) -> float:
    """Share of rows of ``a`` whose nearest row of ``b`` (Euclidean) is the same index. Squared distances up to the
    per-row constant |a_i|^2, in row chunks (memory chunk * len(b), not len(a) * len(b) * dim)."""
    bb = (b ** 2).sum(1)
    hits = 0
    for s in range(0, len(a), chunk):
        d = bb[None, :] - 2 * a[s:s + chunk] @ b.T
        hits += int((d.argmin(1) == np.arange(s, s + len(d))).sum())
    return hits / len(a)


def paired_views(cam_keys: pd.DataFrame, ids) -> tuple[np.ndarray, np.ndarray]:
    """Rows of camera 1 and camera 2 views for the given specimens that have both, in the same specimen order."""
    cams = sorted(cam_keys.camera.unique())
    k = cam_keys[cam_keys.individual.isin(ids)]
    both = k.groupby("individual").camera.nunique() == len(cams)
    keep = sorted(both.index[both])
    rows = [k[k.camera == c].set_index("individual").loc[keep, "row"].values for c in cams]
    return rows[0], rows[1]


def within_class(classes, sp, x, cam, xc, frame_rms, seed, n_fixed: int = 30) -> pd.DataFrame:
    """Per class: eigenvalue summary on all specimens and, for classes with at least ``n_fixed`` specimens, on the first
    ``n_fixed`` in sha1(seed:individual) order (NaN for smaller classes, so the fixed-size values always use exactly
    ``n_fixed``), camera-pair re-identification, median RMS distance of a specimen-camera's frames to their mean."""
    rows = []
    for c in classes:
        k = sp[sp.taxon_group == c]
        first = k.index[np.argsort(stable_order(k.individual, seed).values)[:n_fixed]]
        r1, r2 = paired_views(cam, k.individual)
        summary = eigen_summary(x[first]) if len(k) >= n_fixed else dict.fromkeys(eigen_summary(x[first]), np.nan)
        fixed = {f"{key}_n{n_fixed}": v for key, v in summary.items() if key != "n"}
        rows.append({"taxon_group": c, **eigen_summary(x[k.index]), **fixed,
                     "camera_reid": reidentification(xc[r1], xc[r2]), "camera_reid_chance": 1 / len(r1),
                     "paired_specimens": len(r1), "frame_rms_within_specimen_median": float(np.median(
                         frame_rms[cam.index[cam.taxon_group == c]]))})
    return pd.DataFrame(rows)


def between_classes(classes, sp_tr, x_tr, z2_tr, sp_va=None, x_va=None, k_sub: int = 5) -> pd.DataFrame:
    """Per class pair, from the fitted (train) specimens: radius separation (native, L2-normalised, 2-D PCA), d' along
    the centroid-difference direction, centroid distance, top-``k_sub`` subspace overlap; with val specimens also d'
    of val specimens along the train direction (``dprime_val``)."""
    xn_tr = x_tr / np.linalg.norm(x_tr, axis=1, keepdims=True)
    tr = {c: sp_tr.index[sp_tr.taxon_group == c] for c in classes}
    va = None if sp_va is None else {c: sp_va.index[sp_va.taxon_group == c] for c in classes}
    rows = []
    for a, b in itertools.combinations(classes, 2):
        u = direction(x_tr[tr[a]], x_tr[tr[b]])
        row = {"class_a": a, "class_b": b}
        if va is not None:
            row["dprime_val"] = direction_separation(x_va[va[a]], x_va[va[b]], u)
        rows.append({**row, "dprime_train": direction_separation(x_tr[tr[a]], x_tr[tr[b]], u),
                     "radius_sep": radius_separation(x_tr[tr[a]], x_tr[tr[b]]),
                     "radius_sep_l2norm": radius_separation(xn_tr[tr[a]], xn_tr[tr[b]]),
                     "radius_sep_pca2d": radius_separation(z2_tr[tr[a]], z2_tr[tr[b]]),
                     "centroid_distance": float(np.linalg.norm(x_tr[tr[a]].mean(0) - x_tr[tr[b]].mean(0))),
                     "subspace_overlap_top5": subspace_overlap(x_tr[tr[a]], x_tr[tr[b]], k_sub)})
    return pd.DataFrame(rows)
