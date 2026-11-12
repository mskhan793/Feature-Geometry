"""How many frames per specimen and how many specimens per class for Ledoit-Wolf covariance estimation? (diagnostic,
one fold; author's request in chat 2026-11-12, after task.md eeb9707).

Unit: specimen embedding = mean of its camera means (each the mean of k frames). Fitted on the fold's train specimens;
judged on its val specimens; test rows dropped before the feature lookup and never used. Predeclared criterion: mean
held-out Gaussian log-likelihood per val specimen (micro) and the mean over classes of the per-class mean (macro).

D1/D2 pooled within-class covariance (as in FDA, Mensink Eq. 8 weights N_c/N), native 1024-d:
  for k in ``--frames`` frames per camera (deterministic spread subset of the selected frames; val always uses all) and
  for a cap of train specimens per class in ``--caps`` (first ``cap`` in sha1(seed:individual) order, 3 seeds; classes
  below the cap keep all): class means and residuals from the selected train specimens, S_W, Ledoit-Wolf alpha
  (src.statistics.covariance_fit), condition number, held-out log-likelihood of val residuals (val specimen minus the
  train class mean) under N(0, S_W*) for alpha_LW and for every alpha in ALPHAS (to see where alpha_LW lies relative
  to the held-out optimum), the FDA-16 subspace (similarity to the reference k = max, all specimens: ||Q1'Q2||_F^2/16)
  and the spherical Bhattacharyya pair ranking in FDA-16 (Spearman with the reference).
D3 per-class covariances in train-fitted PCA spaces (``--pcs``; PCA on all train specimens, k = max):
  per class with >= 5 val specimens: held-out log-likelihood of its val specimens under N(mu_c, Sigma_c) for the
  sample covariance (if n > d), Ledoit-Wolf (per class; shrinks the biased covariance, as in the paper), diagonal and
  spherical (unbiased); alpha_c and n_c; for
  classes with >= 256 train specimens a learning curve over subsampled n in CURVE_N (3 seeds); full-covariance
  Bhattacharyya with per-class Ledoit-Wolf for every class pair (src.methods.bhattacharyya), compared with the
  spherical (paper) form and d' (val).

Writes pooled.csv, alpha_curve.csv, perclass.csv, perclass_curve.csv, pairs_lw_pcs{d}.csv and summary.json to --out.

Usage: lw_fit_diagnostics.py --cohort selection.parquet --caches MANIFEST=CACHE ... --fold foldK --out DIR
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

from src.data.expanded_cohort import select_frames, stable_order
from src.features.feature_lookup import features_by_name
from src.features.specimen_means import specimen_camera_means, specimen_means
from src.methods.bhattacharyya import bhattacharyya_terms, class_covariance
from src.methods.fda import fit_fda
from src.statistics.class_geometry import direction, direction_separation
from src.statistics.covariance_fit import gaussian_loglik, lw_alpha, lw_class_covariance, shrunk

ALPHAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
CURVE_N = [15, 30, 64, 128, 256]
SEEDS = [20261112, 20261113, 20261114]
MIN_TRAIN, MIN_VAL, FDA_DIM = 15, 5, 16


def specimens(cohort, x, S):
    keys, cm, _ = specimen_camera_means(x, cohort, S)
    sk, sp = specimen_means(keys, cm, S)
    return sk, sp


def capped(sk_tr: pd.DataFrame, cap: int | None, seed: int) -> np.ndarray:
    """Row positions of the first ``cap`` train specimens per class in sha1(seed:individual) order."""
    if cap is None:
        return np.arange(len(sk_tr))
    key = stable_order(sk_tr.individual, seed)
    order = sk_tr.assign(key=key.values, pos=np.arange(len(sk_tr))).sort_values("key")
    return np.sort(order.groupby("taxon_group").head(cap).pos.to_numpy())


def heldout(x_tr, y_tr, x_va, y_va, alphas):
    """Pooled within-class S_W from train, Ledoit-Wolf alpha, val log-likelihood (micro, macro) per alpha."""
    classes = np.unique(y_tr)
    means = {c: x_tr[y_tr == c].mean(0) for c in classes}
    resid = x_tr - np.stack([means[c] for c in y_tr])
    s_w = resid.T @ resid / len(x_tr)
    keep = np.isin(y_va, classes)
    r_va = x_va[keep] - np.stack([means[c] for c in y_va[keep]])
    a_lw = lw_alpha(resid)
    out = {}
    for a in [a_lw, *alphas]:
        try:
            ll = gaussian_loglik(r_va, np.zeros(r_va.shape[1]), shrunk(s_w, a))
        except np.linalg.LinAlgError:  # singular (e.g. alpha 0 with fewer train specimens than dimensions)
            out[a] = (float("nan"), float("nan"))
            continue
        per_class = pd.Series(ll).groupby(y_va[keep]).mean()
        out[a] = (float(ll.mean()), float(per_class.mean()))
    return a_lw, s_w, out


def bc_spherical_ranking(z_tr, y_tr, classes):
    d = {}
    for a, b in itertools.combinations(classes, 2):
        p, q = z_tr[y_tr == a], z_tr[y_tr == b]
        mt, ct = bhattacharyya_terms(p.mean(0), q.mean(0), class_covariance(p, "spherical"), class_covariance(q, "spherical"))
        d[(a, b)] = mt + ct
    return pd.Series(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--caches", required=True, nargs="+")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--caps", type=int, nargs="+", default=[15, 30, 64, 128, 256])
    ap.add_argument("--pcs", type=int, nargs="+", default=[64, 16])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    S = a.fold
    cohort = pd.read_parquet(a.cohort)
    rows_in = len(cohort)
    cohort = cohort[cohort[S].isin(["train", "val"])].reset_index(drop=True)  # test rows dropped before lookup
    x, meta = features_by_name([tuple(Path(p) for p in s.split("=", 1)) for s in a.caches], cohort.img.tolist())
    kmax = max(a.frames)

    va_rows = cohort[S] == "val"
    sk_va, x_va = specimens(cohort[va_rows].reset_index(drop=True), x[va_rows.values], S)
    y_va = sk_va.taxon_group.to_numpy()
    by_k = {}
    for k in a.frames:
        tr = cohort[cohort[S] == "train"]
        sub = select_frames(tr, k) if k < kmax else tr
        sk, sp = specimens(sub.reset_index(drop=True), x[sub.index.to_numpy()], S)
        by_k[k] = (sk, sp)
    sk_ref, x_ref = by_k[kmax]
    y_ref = sk_ref.taxon_group.to_numpy()
    fda_ref = fit_fda(x_ref, y_ref)
    q_ref = fda_ref.basis(FDA_DIM)
    n_tr = sk_ref.taxon_group.value_counts()
    n_va = pd.Series(y_va).value_counts()
    gbc_classes = sorted(c for c in n_tr.index if n_tr[c] >= MIN_TRAIN and n_va.get(c, 0) >= MIN_VAL)
    rank_ref = bc_spherical_ranking(fda_ref.project(x_ref, FDA_DIM), y_ref, gbc_classes)

    pooled, curve = [], []
    for k, cap in itertools.product(a.frames, [*a.caps, None]):
        sk, sp = by_k[k]
        for seed in SEEDS if cap is not None else SEEDS[:1]:
            pos = capped(sk, cap, seed)
            xs, ys = sp[pos], sk.taxon_group.to_numpy()[pos]
            a_lw, s_w, ll = heldout(xs, ys, x_va, y_va, ALPHAS)
            fda = fit_fda(xs, ys)
            q = fda.basis(FDA_DIM)
            ok = [c for c in gbc_classes if (ys == c).sum() >= 2]
            rank = bc_spherical_ranking(fda.project(xs, FDA_DIM), ys, ok)
            common = rank_ref.index.intersection(rank.index)
            best = max((v for v in ALPHAS if np.isfinite(ll[v][1])), key=lambda v: ll[v][1])
            pooled.append({"frames": k, "cap": cap if cap is not None else "all", "seed": seed, "train_specimens": len(pos),
                           "classes": len(np.unique(ys)), "alpha_lw": a_lw,
                           "condition_s_w_star": float(np.linalg.cond(shrunk(s_w, a_lw))),
                           "val_loglik_micro_lw": ll[a_lw][0], "val_loglik_macro_lw": ll[a_lw][1],
                           "val_loglik_macro_sample": ll[0.0][1], "best_grid_alpha_macro": best,
                           "val_loglik_macro_best_grid": ll[best][1],
                           "fda16_subspace_similarity": float(((q_ref.T @ q) ** 2).sum() / FDA_DIM),
                           "fda16_gbc_spearman_vs_reference": float(spearmanr(rank_ref[common], rank[common])[0])})
            curve += [{"frames": k, "cap": cap if cap is not None else "all", "seed": seed, "alpha": v,
                       "is_lw": v == a_lw, "val_loglik_micro": ll[v][0], "val_loglik_macro": ll[v][1]} for v in ll]
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(a.out / "pooled.csv", index=False)
    pd.DataFrame(curve).to_csv(a.out / "alpha_curve.csv", index=False)

    pca = PCA(n_components=max(a.pcs), svd_solver="full").fit(x_ref)
    z_tr_all, z_va_all = pca.transform(x_ref), pca.transform(x_va)
    per, pcurve, lw_pairs = [], [], {}
    for d in a.pcs:
        z_tr, z_va = z_tr_all[:, :d], z_va_all[:, :d]
        fits = {}
        for c in gbc_classes:
            zt, zv = z_tr[y_ref == c], z_va[y_va == c]
            mu = zt.mean(0)
            lw_cov, a_c = lw_class_covariance(zt)
            covs = {"lw": lw_cov, "diagonal": class_covariance(zt, "diagonal"),
                    "spherical": class_covariance(zt, "spherical")}
            if len(zt) > d:
                covs["sample"] = class_covariance(zt, "full")
            fits[c] = (mu, covs["lw"])
            row = {"pcs": d, "taxon_group": c, "n_train": len(zt), "n_val": len(zv), "alpha_lw": a_c}
            for name, cov in covs.items():
                try:
                    row[f"val_loglik_{name}"] = float(gaussian_loglik(zv, mu, cov).mean())
                except np.linalg.LinAlgError:
                    row[f"val_loglik_{name}"] = float("nan")
            per.append(row)
            if len(zt) >= max(CURVE_N):
                for n, seed in itertools.product(CURVE_N, SEEDS):
                    pick = np.random.default_rng([seed, n]).choice(len(zt), size=n, replace=False)
                    zs = zt[pick]
                    m = zs.mean(0)
                    cov_s, al = lw_class_covariance(zs)
                    pcurve.append({"pcs": d, "taxon_group": c, "n": n, "seed": seed, "alpha_lw": al,
                                   "val_loglik_lw": float(gaussian_loglik(zv, m, cov_s).mean()),
                                   "val_loglik_spherical": float(gaussian_loglik(zv, m, class_covariance(zs, "spherical")).mean())})
        rows = []
        for c1, c2 in itertools.combinations(gbc_classes, 2):
            mt, ct = bhattacharyya_terms(fits[c1][0], fits[c2][0], fits[c1][1], fits[c2][1])
            p, q = z_tr[y_ref == c1], z_tr[y_ref == c2]
            ms, cs = bhattacharyya_terms(p.mean(0), q.mean(0), class_covariance(p, "spherical"), class_covariance(q, "spherical"))
            u = direction(p, q)
            rows.append({"class_a": c1, "class_b": c2, "lw_full_mean_term": mt, "lw_full_cov_term": ct,
                         "lw_full_distance": mt + ct, "spherical_distance": ms + cs,
                         "dprime_val": direction_separation(z_va[y_va == c1], z_va[y_va == c2], u)})
        lw_pairs[d] = pd.DataFrame(rows)
        lw_pairs[d].to_csv(a.out / f"pairs_lw_pcs{d}.csv", index=False)
    per, pcurve = pd.DataFrame(per), pd.DataFrame(pcurve)
    per.to_csv(a.out / "perclass.csv", index=False)
    pcurve.to_csv(a.out / "perclass_curve.csv", index=False)

    ref = pooled[(pooled.frames == kmax) & (pooled.cap == "all")].iloc[0]
    summary = {"fold": S, "encoder": meta["encoder"],
               "cohort_rows": {"in_file": rows_in, "train_val_used": len(cohort), "test_rows_dropped": rows_in - len(cohort)},
               "train_specimens": int(len(sk_ref)), "val_specimens": int(len(sk_va)), "gbc_classes": len(gbc_classes),
               "reference": ref.to_dict(),
               "best_config_by_macro_val_loglik_lw": pooled.loc[pooled.val_loglik_macro_lw.idxmax()].to_dict(),
               "per_class": {}}
    for d in a.pcs:
        t = per[per.pcs == d]
        est = [c for c in ("lw", "sample", "diagonal", "spherical") if f"val_loglik_{c}" in t]
        win = t[[f"val_loglik_{c}" for c in est]].idxmax(axis=1).str.removeprefix("val_loglik_")
        p = lw_pairs[d]
        summary["per_class"][str(d)] = {
            "classes": len(t), "alpha_lw_min_median_max": [float(t.alpha_lw.min()), float(t.alpha_lw.median()), float(t.alpha_lw.max())],
            "spearman_alpha_vs_n_train": float(spearmanr(t.alpha_lw, t.n_train)[0]),
            "best_estimator_counts": win.value_counts().to_dict(),
            "mean_val_loglik": {c: float(t[f"val_loglik_{c}"].mean()) for c in est},
            "sample_covariance_available": int(t.val_loglik_sample.notna().sum()) if "val_loglik_sample" in t else 0,
            "lw_full_pairs": len(p), "spearman_lw_full_vs_spherical": float(spearmanr(p.lw_full_distance, p.spherical_distance)[0]),
            "spearman_lw_full_vs_dprime_val": float(spearmanr(p.lw_full_distance, p.dprime_val)[0]),
            "spearman_spherical_vs_dprime_val": float(spearmanr(p.spherical_distance, p.dprime_val)[0]),
            "lw_full_cov_term_share_median": float((p.lw_full_cov_term / p.lw_full_distance).median())}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
