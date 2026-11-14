"""Matched-pair GBC summaries from saved per-pair Bhattacharyya tables (task.md e4d291d).

Input: one ``pairs_{arm}.csv`` of scripts/pca_fda_gbc.py (one row per unordered class pair; per covariance form
``kind``: ``{kind}_train_valid``, ``{kind}_val_valid``, ``{kind}_{train,val}_{bc,distance,mean_term,cov_term}``; train
and val Gaussians fitted separately in the same train-fitted projection; ``code_spherical_bc`` = the authors' code
release form, train only).

Conventions:
- GBC over unordered pairs = -2 * sum BC, identical to the paper's negative sum over ordered distinct pairs (Eq. 3);
  BC = exp(-D_B); a larger (less negative) GBC = less estimated Gaussian overlap.
- Primary train/val comparison: both sums over the same pair set P_common = pairs valid on train AND on val.
  ``gbc_train_all`` (every train-valid pair) is auxiliary and equals the paper's one-score training-only usage.
- Ratio val/train = mean BC val / mean BC train over P_common, reported only when the train mean BC is at least
  RATIO_MIN_MEAN_BC (predeclared; below it the ratio of two near-zero sums is not interpretable); differences of mean
  BC are always reported. Ratio > 1 = larger estimated overlap on val, not classification error.
- Matched arms (PCA-d vs FDA-d): every quantity over the intersection of both arms' P_common, identified by
  (class_a, class_b); differing class sets make the comparison ineligible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

KINDS = ("spherical", "diagonal", "full")
RATIO_MIN_MEAN_BC = 1e-6
TOP = 20
VAL_SUPPORT_BINS = [(5, 10), (10, 20), (20, None)]  # min(n_val_a, n_val_b): low validation support strata
PAIR = ["class_a", "class_b"]


def gbc(bc) -> float:
    """-2 * sum of BC over unordered pairs (= negative sum over ordered distinct pairs)."""
    return -2.0 * float(np.sum(bc))


def valid(p: pd.DataFrame, kind: str, split: str) -> pd.Series:
    col = f"{kind}_{split}_valid"
    return p[col].astype(bool) if col in p else pd.Series(False, index=p.index)


def ratio(mean_val: float, mean_train: float) -> tuple[float, str]:
    if not np.isfinite(mean_train):
        return np.nan, "no common pairs"
    if mean_train < RATIO_MIN_MEAN_BC:
        return np.nan, f"train mean BC < {RATIO_MIN_MEAN_BC:g}"
    return mean_val / mean_train, "ok"


def _spearman(x, y) -> float:
    return float(spearmanr(x, y)[0]) if len(x) >= 3 else np.nan


def train_val(p: pd.DataFrame, kind: str) -> dict:
    """Train vs val summary of one arm and covariance form over P_common."""
    tv, vv = valid(p, kind, "train"), valid(p, kind, "val")
    m = tv & vv
    out = {"pairs_total": len(p), "pairs_train_valid": int(tv.sum()), "pairs_val_valid": int(vv.sum()),
           "pairs_common": int(m.sum()),
           "gbc_train_all": gbc(p.loc[tv, f"{kind}_train_bc"]) if tv.any() else np.nan}
    if not m.any():
        return {**out, "ratio_status": "no common pairs"}
    c = p[m]
    bt, bv = c[f"{kind}_train_bc"].to_numpy(float), c[f"{kind}_val_bc"].to_numpy(float)
    delta = bv - bt
    r, status = ratio(bv.mean(), bt.mean())
    out.update({"gbc_train_common": gbc(bt), "gbc_val_common": gbc(bv), "mean_bc_train": float(bt.mean()),
                "mean_bc_val": float(bv.mean()), "mean_bc_val_minus_train": float(delta.mean()),
                "ratio_val_over_train": r, "ratio_status": status,
                "delta_bc_p10_median_p90": [float(q) for q in np.quantile(delta, [0.1, 0.5, 0.9])],
                "share_pairs_bc_val_above_train": float((delta > 0).mean()),
                "spearman_distance_train_vs_val": _spearman(c[f"{kind}_train_distance"], c[f"{kind}_val_distance"]),
                f"top{TOP}_hardest_retained": len(set(c.nlargest(TOP, f"{kind}_train_bc").index)
                                                 & set(c.nlargest(TOP, f"{kind}_val_bc").index)),
                "mean_term_median_train_val": [float(c[f"{kind}_{s}_mean_term"].median()) for s in ("train", "val")],
                "cov_term_median_train_val": [float(c[f"{kind}_{s}_cov_term"].median()) for s in ("train", "val")]})
    low = np.minimum(c.n_val_a, c.n_val_b)
    out["by_min_val_units"] = {}
    for lo, hi in VAL_SUPPORT_BINS:
        s = (low >= lo) & (low < hi if hi else True)
        out["by_min_val_units"][f"{lo}-{hi - 1}" if hi else f">={lo}"] = (
            {"pairs": int(s.sum()), "mean_bc_train": float(bt[s].mean()), "mean_bc_val": float(bv[s].mean())}
            if s.any() else {"pairs": 0})
    return out


def paper_vs_code(p: pd.DataFrame) -> dict:
    """Arm A: paper form (spherical, Eq. 2) vs the authors' code release form, training only, train-valid pairs."""
    tv = valid(p, "spherical", "train")
    v = p[tv]
    return {"pairs": int(tv.sum()), "gbc_paper_train": gbc(v.spherical_train_bc), "gbc_code_train": gbc(v.code_spherical_bc),
            "spearman_bc_paper_vs_code": _spearman(v.spherical_train_bc, v.code_spherical_bc)}


def matched(p_pca: pd.DataFrame, p_fda: pd.DataFrame, kind: str) -> dict:
    """PCA-d vs FDA-d over the pairs valid on train and val in both arms (same pair ids)."""
    a, b = p_pca.set_index(PAIR), p_fda.set_index(PAIR)
    if len(a) != len(b) or not a.index.sort_values().equals(b.index.sort_values()):
        return {"status": "not eligible: different class pair sets"}
    b = b.loc[a.index]
    m = (valid(a, kind, "train") & valid(a, kind, "val") & valid(b, kind, "train") & valid(b, kind, "val")).to_numpy()
    out = {"status": "ok" if m.any() else "no common pairs", "pairs_matched": int(m.sum())}
    if not m.any():
        return out
    for tag, t in (("pca", a[m]), ("fda", b[m])):
        bt, bv = t[f"{kind}_train_bc"].to_numpy(float), t[f"{kind}_val_bc"].to_numpy(float)
        r, status = ratio(bv.mean(), bt.mean())
        out.update({f"gbc_train_{tag}": gbc(bt), f"gbc_val_{tag}": gbc(bv), f"mean_bc_train_{tag}": float(bt.mean()),
                    f"mean_bc_val_{tag}": float(bv.mean()), f"ratio_{tag}": r, f"ratio_status_{tag}": status})
    out["fda_minus_pca_mean_bc_train"] = out["mean_bc_train_fda"] - out["mean_bc_train_pca"]
    out["fda_minus_pca_mean_bc_val"] = out["mean_bc_val_fda"] - out["mean_bc_val_pca"]
    for s in ("train", "val"):
        out[f"spearman_distance_pca_vs_fda_{s}"] = _spearman(a[m][f"{kind}_{s}_distance"], b[m][f"{kind}_{s}_distance"])
    return out


def per_class(p: pd.DataFrame, kind: str = "spherical") -> pd.DataFrame:
    """Per class over its P_common pairs: mean and max BC on train and val."""
    m = valid(p, kind, "train") & valid(p, kind, "val")
    c = p[m]
    long = pd.concat([c.rename(columns={"class_a": "cls"})[["cls", f"{kind}_train_bc", f"{kind}_val_bc"]],
                      c.rename(columns={"class_b": "cls"})[["cls", f"{kind}_train_bc", f"{kind}_val_bc"]]])
    g = long.groupby("cls")
    return pd.DataFrame({"pairs": g.size(), "mean_bc_train": g[f"{kind}_train_bc"].mean(),
                         "mean_bc_val": g[f"{kind}_val_bc"].mean(), "max_bc_train": g[f"{kind}_train_bc"].max(),
                         "max_bc_val": g[f"{kind}_val_bc"].max()}).rename_axis("taxon_group").reset_index()
