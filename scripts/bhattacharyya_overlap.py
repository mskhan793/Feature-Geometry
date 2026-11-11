"""Gaussian Bhattacharyya class overlap versus direction-aware d' on one fold (task.md 0237ac7; fold0 by default).

Unit: one embedding per independent specimen (mean of its camera means). Only the fold's train specimens are fitted
(PCA, class Gaussians, directions); val specimens are scored or used for a held-out re-estimate; test rows are dropped
before the feature lookup and never used. Classes: >= 15 train and >= 5 val specimens (as in class_geometry_pilot.py),
so the class pairs are the same as in the existing geometry runs.

Space: unwhitened PCA fitted on train specimens; ``--pcs`` dimensions (64 = the paper's setting; others are
sensitivity analyses with identical rules). In each space and for each class pair:
- d' (val specimens along the train centroid-difference direction; also on train) — same space as the Bhattacharyya
  terms; the native 1024-d d' is kept as the historical reference and checked against ``--existing-pairs``;
- Bhattacharyya (src.methods.bhattacharyya, paper Eq. 2): mean term, covariance term, distance D_B and coefficient
  BC = exp(-D_B), from unbiased class Gaussians fitted on train specimens, for covariance forms spherical (main: the
  paper's setting), diagonal and full (sensitivity; full only if both classes have >= 2 d train specimens, else
  ineligible); the same from val specimens in the same space (held-out re-estimate; full needs >= 2 d val specimens);
- the authors' code release formula (``stable_transfer``: spherical, no factor d, population variance, floor 1e-4) on
  train specimens in float32, as a labelled reproduction;
- orientation (full covariance, eligible classes): variance of each class along the pair's train centroid-difference
  direction u relative to its mean variance per dimension (u'S u / (tr S / d); > 1 = spread elongated along the
  separating direction, < 1 = across it), and |cos| between u and each class's top eigenvector.
BC near 1 = more modelled overlap, near 0 = less; D_B has the opposite orientation. BC is the overlap of the fitted
Gaussians, not of the true class distributions (classes can be multimodal: ``--labels`` adds the fold's per-class
2-component tied labels from the corrected geometry run, at its PCA dimensions, as columns label{d}).

Writes to ``--out``: pairs_pcs{d}.csv, classes_pcs{d}.csv, summary.json and two figures.

Usage: bhattacharyya_overlap.py --cohort selection.parquet --caches MANIFEST=CACHE ... --out DIR [--fold fold0]
       [--pcs 64 8 16] [--existing-pairs class_pairs.csv] [--labels per_class_modality.csv]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA

from src.features.chunk_store import file_sha256
from src.features.feature_lookup import features_by_name
from src.features.specimen_means import specimen_camera_means, specimen_means
from src.methods.bhattacharyya import (KINDS, bhattacharyya_terms, class_covariance, gbc_score,
                                       stable_transfer_class_stats, stable_transfer_distance)
from src.statistics.class_geometry import direction, direction_separation, eigen_summary

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MIN_TRAIN, MIN_VAL, TOP = 15, 5, 20


def gaussians(z: np.ndarray, groups: dict, d: int) -> dict:
    """Per class and covariance form: (mean, covariance) or None if not estimable (full needs >= 2 d specimens)."""
    out = {}
    for c, idx in groups.items():
        x = z[idx]
        out[c] = {k: (x.mean(0), class_covariance(x, k)) if (k != "full" or len(x) >= 2 * d) and len(x) >= 2 else None
                  for k in KINDS}
    return out


def pair_row(a, b, g_tr, g_va, code, z_tr, z_va, tr, va, d) -> dict:
    u = direction(z_tr[tr[a]], z_tr[tr[b]])
    row = {"class_a": a, "class_b": b, "n_train_a": len(tr[a]), "n_train_b": len(tr[b]), "n_val_a": len(va[a]),
           "n_val_b": len(va[b]), "dprime_val": direction_separation(z_va[va[a]], z_va[va[b]], u),
           "dprime_train": direction_separation(z_tr[tr[a]], z_tr[tr[b]], u)}
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
        if p is None:
            continue
        _, vec = np.linalg.eigh(p[1])
        row[f"along_ratio_{side}"] = float(u @ p[1] @ u / (np.trace(p[1]) / d))
        row[f"top_eigvec_cos_{side}"] = float(abs(vec[:, -1] @ u))
    return row


def rank_summary(pairs: pd.DataFrame, k: str) -> dict:
    v = pairs[pairs[f"{k}_train_valid"]]
    if len(v) < 3:
        return {"pairs_valid": len(v)}
    close_d = set(v.nsmallest(TOP, "dprime_val").index)
    close_b = set(v.nlargest(TOP, f"{k}_train_bc").index)
    out = {"pairs_valid": len(v),
           "spearman_dprime_val_vs_distance": float(spearmanr(v.dprime_val, v[f"{k}_train_distance"])[0]),
           "kendall_dprime_val_vs_distance": float(kendalltau(v.dprime_val, v[f"{k}_train_distance"])[0]),
           "spearman_dprime_val_vs_mean_term": float(spearmanr(v.dprime_val, v[f"{k}_train_mean_term"])[0]),
           f"top{TOP}_closest_overlap_dprime_vs_bc": len(close_d & close_b),
           "cov_term_share_median": float((v[f"{k}_train_cov_term"] / v[f"{k}_train_distance"]).median()),
           "bc_train_min_median_max": [float(v[f"{k}_train_bc"].min()), float(v[f"{k}_train_bc"].median()),
                                       float(v[f"{k}_train_bc"].max())],
           "gbc_train_ordered_pairs_descriptive": -2 * float(v[f"{k}_train_bc"].sum())}
    both = v[v[f"{k}_val_valid"]]
    if len(both) >= 3:
        out["spearman_bc_train_vs_val"] = float(spearmanr(both[f"{k}_train_bc"], both[f"{k}_val_bc"])[0])
        out["pairs_with_val_estimate"] = len(both)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--caches", required=True, nargs="+", help="MANIFEST=CACHE_DIR pairs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--pcs", type=int, nargs="+", default=[64, 8, 16])
    ap.add_argument("--existing-pairs", type=Path, help="class_pairs.csv of the corrected geometry run of this fold")
    ap.add_argument("--labels", type=Path, help="per_class_modality.csv of the corrected geometry run of this fold")
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

    native = []
    for c1, c2 in itertools.combinations(classes, 2):
        u = direction(x_tr[tr[c1]], x_tr[tr[c2]])
        native.append({"class_a": c1, "class_b": c2, "dprime_val_native": direction_separation(x_va[va[c1]], x_va[va[c2]], u)})
    native = pd.DataFrame(native)
    regression = None
    if a.existing_pairs:
        old = pd.read_csv(a.existing_pairs).set_index(["class_a", "class_b"]).dprime_val
        new = native.set_index(["class_a", "class_b"]).dprime_val_native
        regression = {"pairs": len(new), "max_abs_diff_vs_existing_dprime_val": float((new - old.reindex(new.index)).abs().max())}
    labels = {}
    if a.labels:
        lab = pd.read_csv(a.labels)
        labels = {int(d): lab[lab.pcs == d].set_index("taxon_group").verdict.to_dict() for d in sorted(lab.pcs.unique())}

    pca = PCA(n_components=max(a.pcs), svd_solver="full").fit(x_tr)
    z_tr_all, z_va_all = pca.transform(x_tr), pca.transform(x_va)
    summary = {"fold": S, "cohort_sha256": file_sha256(a.cohort), "encoder": meta["encoder"], "model_id": meta["model_id"],
               "cohort_rows": {"in_file": rows_in, "train_val_used": len(cohort), "test_rows_dropped": rows_in - len(cohort)},
               "specimens": {"train": len(sk_tr), "val": len(sk_va)}, "classes": len(classes),
               "classes_below_support": sorted(set(n_tr.index) - set(classes)), "pairs": len(native),
               "pca": "unwhitened, fitted on train specimens",
               "pca_explained_variance": {str(d): float(pca.explained_variance_ratio_[:d].sum()) for d in a.pcs},
               "native_dprime_regression": regression, "by_pcs": {}}
    for d in a.pcs:
        z_tr, z_va = z_tr_all[:, :d], z_va_all[:, :d]
        g_tr, g_va = gaussians(z_tr, tr, d), gaussians(z_va, va, d)
        code = {c: stable_transfer_class_stats(z_tr[tr[c]].astype(np.float32)) for c in classes}
        pairs = pd.DataFrame([pair_row(c1, c2, g_tr, g_va, code, z_tr, z_va, tr, va, d)
                              for c1, c2 in itertools.combinations(classes, 2)])
        pairs = pairs.merge(native, on=["class_a", "class_b"])
        for ld, lab in labels.items():  # labels exist at the geometry run's PCA dimensions (8, 16)
            pairs[f"label{ld}_a"], pairs[f"label{ld}_b"] = pairs.class_a.map(lab), pairs.class_b.map(lab)
        pairs.to_csv(a.out / f"pairs_pcs{d}.csv", index=False)
        cls = []
        for c in classes:
            full = g_tr[c]["full"]
            cls.append({"taxon_group": c, "n_train": len(tr[c]), "n_val": len(va[c]),
                        **{f"label{ld}": lab.get(c) for ld, lab in labels.items()},
                        "full_covariance_eligible": full is not None,
                        **({f"{k}_pcs": v for k, v in eigen_summary(z_tr[tr[c]]).items() if k != "n"}),
                        "spherical_variance": float(np.trace(g_tr[c]["spherical"][1]) / d)})
        pd.DataFrame(cls).to_csv(a.out / f"classes_pcs{d}.csv", index=False)
        s = {k: rank_summary(pairs, k) for k in KINDS}
        v = pairs[pairs.spherical_train_valid]
        s["code_vs_paper_spherical"] = {
            "spearman_bc": float(spearmanr(v.spherical_train_bc, v.code_spherical_bc)[0]),
            f"top{TOP}_closest_overlap": len(set(v.nlargest(TOP, "spherical_train_bc").index)
                                             & set(v.nlargest(TOP, "code_spherical_bc").index)),
            "gbc_paper_spherical": -2 * float(v.spherical_train_bc.sum()),
            "gbc_code_spherical": -2 * float(v.code_spherical_bc.sum()),
            "code_variance_floor_binding_classes": int(sum((code[c][1] <= 1e-4).any() for c in classes))}
        s["full_covariance_eligible_classes"] = int(sum(g_tr[c]["full"] is not None for c in classes))
        s["spearman_dprime_val_pcs_vs_native"] = float(spearmanr(pairs.dprime_val, pairs.dprime_val_native)[0])
        f = pairs[pairs.full_train_valid].copy()
        if len(f):
            f["rank_dprime"] = f.dprime_val.rank()
            f["rank_full_distance"] = f.full_train_distance.rank()
            f["rank_gap"] = f.rank_full_distance - f.rank_dprime
            close = f[(f.rank_dprime <= 40) | (f.rank_full_distance <= 40)]
            cols = ["class_a", "class_b", "dprime_val", "full_train_distance", "full_train_bc", "full_train_mean_term",
                    "full_train_cov_term", "spherical_train_bc", "rank_dprime", "rank_full_distance", "rank_gap",
                    "along_ratio_a", "along_ratio_b", "top_eigvec_cos_a", "top_eigvec_cos_b",
                    *[c for c in f if c.startswith("label")]]
            s["largest_rank_disagreements_among_close_pairs"] = json.loads(
                close.reindex(close.rank_gap.abs().sort_values(ascending=False).index).head(6)[cols].to_json(orient="records"))
        summary["by_pcs"][str(d)] = s
        if d == a.pcs[0]:
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
            for ax, k in zip(axes, KINDS):
                w = pairs[pairs[f"{k}_train_valid"]]
                ax.scatter(w.dprime_val, w[f"{k}_train_bc"], s=6, c=w[f"{k}_train_cov_term"] / w[f"{k}_train_distance"],
                           cmap="viridis", vmin=0, vmax=1)
                ax.set(yscale="log", xlabel="d' (val, train direction, same space)", ylabel=f"BC ({k}, train)",
                       title=f"{k} covariance, {d} PCs ({len(w)} pairs)")
            axes[3].scatter(v.spherical_train_bc, v.code_spherical_bc, s=6)
            axes[3].set(xscale="log", yscale="log", xlabel="BC spherical, paper Eq. 2", ylabel="BC spherical, code release",
                        title="paper formula vs authors' code")
            fig.suptitle("Colour = share of the Bhattacharyya distance from the covariance term", fontsize=9)
            fig.tight_layout()
            fig.savefig(a.out / f"bc_vs_dprime_pcs{d}.png", dpi=130)
            plt.close(fig)
            if len(f):
                fig, ax = plt.subplots(figsize=(6, 4.5))
                ratio = np.sqrt(f.along_ratio_a.fillna(1) * f.along_ratio_b.fillna(1))
                sc = ax.scatter(f.dprime_val, f.full_train_distance, s=8, c=np.log2(ratio), cmap="coolwarm", vmin=-2, vmax=2)
                fig.colorbar(sc, label="log2 variance along u / mean variance (geometric mean of the pair)")
                ax.set(xscale="log", yscale="log", xlabel="d' (val, same space)", ylabel="Bhattacharyya distance (full, train)",
                       title=f"Orientation, {d} PCs (full-covariance pairs)")
                fig.tight_layout()
                fig.savefig(a.out / f"orientation_pcs{d}.png", dpi=130)
                plt.close(fig)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
