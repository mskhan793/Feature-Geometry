"""PCA-GBC vs regularised FDA-GBC on one fold (task.md eeb9707; method note docs/pca_fda_gbc_method_note.md).

Unit: one embedding per independent specimen (mean of camera means). Fitted on the fold's train specimens only (PCA,
FDA scatter + shrinkage + directions, class Gaussians, d' directions); val specimens give d' and the held-out
re-estimate of the class Gaussians; test rows are dropped before the feature lookup and never used. GBC pairs: classes
with >= 15 train and >= 5 val specimens (as in the geometry runs); FDA uses every class of the train split.

Spaces ("arms"), all centred on the train mean:
- pca64: the Pándy et al. baseline (unwhitened PCA, ``--pca-base`` = 64 dimensions);
- pca{d}, fda{d} for d in ``--dims`` (8 16 32) where d <= the observed FDA rank: matched comparison; FDA projected
  onto the orthonormal basis of its top-d directions (main convention);
- sensitivity, labelled: fda{d}_raw (raw generalised eigenvectors, v'S_W*v = 1), fda{d}_shrink0.5 / _shrink2
  (Ledoit-Wolf intensity halved / doubled).
In each arm and for each class pair: d' (val and train, train centroid direction, same space) and Bhattacharyya terms
of train and val Gaussians, spherical (main, the paper's form), diagonal and full (>= 2 d specimens)
(src.statistics.pair_overlap). Indicators per arm (predeclared): Spearman of the pair distance with d' (val); train-to-val
Spearman of the pair distance and overlap of the 20 hardest pairs (largest BC); share of full-covariance pairs with
mean term >= d'(train)^2/8 (must be 1); dataset-level GBC (train, val; descriptive). Across arms: Spearman of the
spherical pair distance between pca{d} and fda{d}, and between fda{d} and its sensitivity variants.

``--previous DIR`` (fold 0): compares every pca{d} arm with the earlier bhattacharyya_overlap.py table
DIR/pairs_pcs{d}.csv where it exists; definitions are identical, so differences must be numerical only.

Writes to ``--out``: pairs_{arm}.csv, classes.csv, summary.json.

Usage: pca_fda_gbc.py --cohort selection.parquet --caches MANIFEST=CACHE ... --fold foldK --out DIR [--dims 8 16 32]
       [--previous DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from src.features.chunk_store import file_sha256
from src.features.feature_lookup import features_by_name
from src.features.specimen_means import specimen_camera_means, specimen_means
from src.methods.bhattacharyya import KINDS, stable_transfer_class_stats
from src.methods.fda import fit_fda
from src.statistics.pair_overlap import class_gaussians, pair_overlap

MIN_TRAIN, MIN_VAL, TOP = 15, 5, 20
COMPARED = [f"{k}_{s}_{t}" for k in KINDS for s in ("train", "val") for t in ("distance", "mean_term", "cov_term")]


def arm_pairs(z_tr, z_va, classes, tr, va) -> pd.DataFrame:
    d = z_tr.shape[1]
    g_tr, g_va = class_gaussians(z_tr, tr, d), class_gaussians(z_va, va, d)
    code = {c: stable_transfer_class_stats(z_tr[tr[c]].astype(np.float32)) for c in classes}
    return pd.DataFrame([pair_overlap(a, b, g_tr, g_va, code, z_tr, z_va, tr, va, d)
                         for a, b in itertools.combinations(classes, 2)])


def indicators(p: pd.DataFrame) -> dict:
    out = {}
    for k in KINDS:
        v = p[p[f"{k}_train_valid"]]
        s = {"pairs_valid_train": len(v)}
        if len(v) >= 3:
            s["spearman_distance_vs_dprime_val"] = float(spearmanr(v[f"{k}_train_distance"], v.dprime_val, nan_policy="omit")[0])
            s["gbc_train"] = -2 * float(v[f"{k}_train_bc"].sum())
            s["distance_train_median"] = float(v[f"{k}_train_distance"].median())
            s["cov_term_share_median"] = float((v[f"{k}_train_cov_term"] / v[f"{k}_train_distance"]).median())
        both = v[v[f"{k}_val_valid"]]
        s["pairs_valid_train_and_val"] = len(both)
        if len(both) >= 3:
            s["spearman_distance_train_vs_val"] = float(spearmanr(both[f"{k}_train_distance"], both[f"{k}_val_distance"])[0])
            s[f"top{TOP}_hardest_overlap_train_vs_val"] = len(set(both.nlargest(TOP, f"{k}_train_bc").index)
                                                             & set(both.nlargest(TOP, f"{k}_val_bc").index))
            s["gbc_val"] = -2 * float(both[f"{k}_val_bc"].sum())
        if k == "full" and len(v):
            ok = v.dprime_defined
            s["share_mean_term_ge_dprime2_over_8"] = float(
                (v.full_train_mean_term[ok] >= v.dprime_train[ok] ** 2 / 8 - 1e-9).mean())
        out[k] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--caches", required=True, nargs="+", help="MANIFEST=CACHE_DIR pairs")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dims", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--previous", type=Path)
    ap.add_argument("--pca-base", type=int, default=64, help="dimensions of the paper baseline arm")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    S = a.fold
    cohort = pd.read_parquet(a.cohort)
    rows_in = len(cohort)
    cohort = cohort[cohort[S].isin(["train", "val"])].reset_index(drop=True)  # test rows dropped before lookup
    x, meta = features_by_name([tuple(Path(p) for p in s.split("=", 1)) for s in a.caches], cohort.img.tolist())
    keys, cam_means, _ = specimen_camera_means(x, cohort, S)
    sk, sp = specimen_means(keys, cam_means, S)
    trm, vam = (sk[S] == "train").values, (sk[S] == "val").values
    sk_tr, sk_va = sk[trm].reset_index(drop=True), sk[vam].reset_index(drop=True)
    x_tr, x_va = sp[trm], sp[vam]
    n_tr, n_va = sk_tr.taxon_group.value_counts(), sk_va.taxon_group.value_counts()
    classes = sorted(c for c in n_tr.index if n_tr[c] >= MIN_TRAIN and n_va.get(c, 0) >= MIN_VAL)
    tr = {c: np.flatnonzero(sk_tr.taxon_group == c) for c in classes}
    va = {c: np.flatnonzero(sk_va.taxon_group == c) for c in classes}
    pd.DataFrame({"taxon_group": sorted(n_tr.index), "n_train": [int(n_tr[c]) for c in sorted(n_tr.index)],
                  "n_val": [int(n_va.get(c, 0)) for c in sorted(n_tr.index)],
                  "in_gbc_pairs": [c in classes for c in sorted(n_tr.index)]}).to_csv(a.out / "classes.csv", index=False)

    base = a.pca_base
    pca = PCA(n_components=max(base, *a.dims), svd_solver="full").fit(x_tr)
    p_tr, p_va = pca.transform(x_tr), pca.transform(x_va)
    y_tr = sk_tr.taxon_group.to_numpy()
    fdas = {s: fit_fda(x_tr, y_tr, s) for s in (1.0, 0.5, 2.0)}
    main_fda = fdas[1.0]
    dims = [d for d in a.dims if d <= main_fda.rank]
    arms = {f"pca{base}": (p_tr[:, :base], p_va[:, :base])}
    for d in dims:
        arms[f"pca{d}"] = (p_tr[:, :d], p_va[:, :d])
        arms[f"fda{d}"] = (main_fda.project(x_tr, d), main_fda.project(x_va, d))
        arms[f"fda{d}_raw"] = (main_fda.project(x_tr, d, "raw"), main_fda.project(x_va, d, "raw"))
        for s, tag in ((0.5, "shrink0.5"), (2.0, "shrink2")):
            arms[f"fda{d}_{tag}"] = (fdas[s].project(x_tr, d), fdas[s].project(x_va, d))

    summary = {"fold": S, "cohort_sha256": file_sha256(a.cohort), "encoder": meta["encoder"],
               "cohort_rows": {"in_file": rows_in, "train_val_used": len(cohort), "test_rows_dropped": rows_in - len(cohort)},
               "specimens": {"train": len(sk_tr), "val": len(sk_va)}, "classes_train": int(len(n_tr)),
               "classes_gbc": len(classes), "pairs": len(classes) * (len(classes) - 1) // 2,
               "classes_below_support": sorted(set(n_tr.index) - set(classes)),
               "dims_requested": a.dims, "dims_used": dims, "dims_omitted": sorted(set(a.dims) - set(dims)),
               "pca_explained_variance": {str(d): float(pca.explained_variance_ratio_[:d].sum()) for d in [base, *dims]},
               "fda": {tag: {"alpha": f.alpha, "alpha_ledoit_wolf": f.alpha_ledoit_wolf, "condition_s_w": f.condition_s_w,
                             "rank": f.rank, "eigenvalues_top32": [float(v) for v in f.eigenvalues[:32]]}
                       for tag, f in (("main", fdas[1.0]), ("shrink0.5", fdas[0.5]), ("shrink2", fdas[2.0]))},
               "arms": {}, "between_arms": {}}
    tables = {}
    for name, (z_tr, z_va) in arms.items():
        p = arm_pairs(z_tr, z_va, classes, tr, va)
        p.to_csv(a.out / f"pairs_{name}.csv", index=False)
        tables[name] = p
        summary["arms"][name] = {"dims": z_tr.shape[1], **indicators(p)}
    for d in dims:
        for other in (f"pca{d}", f"fda{d}_raw", f"fda{d}_shrink0.5", f"fda{d}_shrink2"):
            u, v = tables[f"fda{d}"], tables[other]
            ok = u.spherical_train_valid & v.spherical_train_valid
            summary["between_arms"][f"fda{d}_vs_{other}"] = {
                "spearman_spherical_distance": float(spearmanr(u.spherical_train_distance[ok], v.spherical_train_distance[ok])[0]),
                f"top{TOP}_hardest_overlap": len(set(u[ok].nlargest(TOP, "spherical_train_bc").index)
                                                 & set(v[ok].nlargest(TOP, "spherical_train_bc").index))}
    if a.previous:
        reg = {}
        for name in [n for n in tables if n.startswith("pca")]:
            d = int(name[3:])
            if (a.previous / f"pairs_pcs{d}.csv").exists():
                old = pd.read_csv(a.previous / f"pairs_pcs{d}.csv").set_index(["class_a", "class_b"])
                new = tables[name].set_index(["class_a", "class_b"])
                cols = [c for c in COMPARED + ["dprime_val", "dprime_train"] if c in old and c in new]
                reg[name] = {"pairs": len(new), "columns": len(cols),
                             "max_abs_diff": float(np.nanmax(np.abs(new[cols].to_numpy(float) - old.loc[new.index, cols].to_numpy(float))))}
        summary["regression_vs_previous_fold0"] = reg
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("arms",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
