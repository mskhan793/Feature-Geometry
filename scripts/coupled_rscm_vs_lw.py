"""Per-class Ledoit-Wolf versus the coupled RSCM estimator (Raninen & Ollila 2021) on held-out specimens, one fold
(task.md 61ceb61).

Same protocol as section 8 D3 of lw_fit_diagnostics.py: specimen embedding = mean of its camera means over all
selected train frames; test rows dropped before the feature lookup and never used; PCA fitted on the fold's train
specimens (``--pcs``, first d components); class mean = train sample mean; eligible classes = >= 15 train and >= 5 val
specimens (both estimators use exactly these classes and specimens; the coupled estimator's pooled matrix is over the
eligible classes).

Per class: (A) per-class Ledoit-Wolf full covariance (src.statistics.covariance_fit.lw_class_covariance, reference);
(B) coupled RSCM with the authors' data-estimated class-specific (alpha_c, beta_c) (src.methods.coupled_rscm). Labelled
diagnostic corners of Eq. (4), no tuning: pooled SCM (alpha 1, beta 0), class SCM (alpha 1, beta 1; only if n_c > d)
and class scaled identity (alpha 0, beta 1). Nothing is selected on validation.

Measurement: held-out Gaussian log-likelihood of each val specimen under N(mu_c, Sigma_c) of its own class; paired
difference coupled minus LW per specimen; class-macro (primary) and specimen-micro (secondary) means with a stratified
bootstrap over val specimens (each specimen one row, resampled within its class; conditional on the fitted PCA, means
and covariances); strata by train specimens per class 15-63, 64-255, >= 256. Numerical failures are reported as
failures. A 4-class check runs the unmodified authors' code and the adapted one on the first four eligible classes of
this real fold and requires identical output.

Writes perclass.csv, perspecimen.csv and summary.json to --out.

Usage: coupled_rscm_vs_lw.py --cohort selection.parquet --caches MANIFEST=CACHE ... --fold foldK --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.features.feature_lookup import features_by_name
from src.features.specimen_means import specimen_camera_means, specimen_means
from src.methods.coupled_rscm import (PATCH, SOURCE_SHA256, adapted_module, coupled_covariance, fit_coupled,
                                      original_module)
from src.statistics.class_macro import ci95
from src.statistics.covariance_fit import gaussian_loglik, lw_class_covariance

MIN_TRAIN, MIN_VAL = 15, 5  # as lw_fit_diagnostics.py (section 8 D3)
STRATA = [(15, 64, "15-63"), (64, 256, "64-255"), (256, np.inf, ">=256")]
N_BOOT, SEED = 2000, 20261112
ESTIMATORS = ["lw", "coupled", "pooled_scm", "class_scm", "class_identity"]


def stratum(n: int) -> str:
    return next(name for lo, hi, name in STRATA if lo <= n < hi)


def numerics(cov: np.ndarray) -> dict:
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
    return {"asymmetry": float(np.abs(cov - cov.T).max()), "min_eig": float(eig[0]),
            "condition": float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")}


def loglik(z: np.ndarray, mu: np.ndarray, cov: np.ndarray | None) -> tuple[np.ndarray, str]:
    if cov is None:
        return np.full(len(z), np.nan), "not defined (n <= d)"
    try:
        return gaussian_loglik(z, mu, cov), "ok"
    except np.linalg.LinAlgError:
        return np.full(len(z), np.nan), "failed: not positive definite"


def four_class_check(z: np.ndarray, y: np.ndarray, classes: list) -> dict:
    """Unmodified authors' code vs the adapted one on four real classes (the only K the original accepts)."""
    if len(classes) < 4:
        return {"run": False, "reason": "fewer than 4 eligible classes"}
    four = classes[:4]
    a, b = fit_coupled(z, y, four, original_module()), fit_coupled(z, y, four, adapted_module())
    diff = max(float(np.abs(a.alpha - b.alpha).max()), float(np.abs(a.beta - b.beta).max()),
               max(float(np.abs(s - t).max()) for s, t in zip(a.sigmas, b.sigmas)))
    if diff != 0:
        raise AssertionError(f"adapted authors' code differs from the original on 4 real classes: {diff}")
    return {"run": True, "classes": four, "max_abs_diff_alpha_beta_sigmas": diff}


def summarise(per: pd.DataFrame, boots: np.ndarray, classes: list) -> dict:
    """Macro / micro paired coupled - LW over ``classes`` (rows of ``per`` and columns of ``boots`` in that order)."""
    if not classes:
        return {"classes": 0}
    t = per.set_index("taxon_group").loc[classes]
    cols = [list(per.taxon_group).index(c) for c in classes]
    w = t.n_val.to_numpy() / t.n_val.sum()
    return {"classes": len(classes), "val_specimens": int(t.n_val.sum()),
            "macro_mean": float(t.diff_mean.mean()), "macro_ci95": ci95(boots[:, cols].mean(1)),
            "micro_mean": float((t.diff_mean * w).sum()), "micro_ci95": ci95(boots[:, cols] @ w),
            "class_diff_median": float(t.diff_mean.median()), "classes_coupled_better": int((t.diff_mean > 0).sum()),
            "classes_ci_above_0": int(sum(v[0] > 0 for v in t.diff_ci95)),
            "classes_ci_below_0": int(sum(v[1] < 0 for v in t.diff_ci95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--caches", required=True, nargs="+")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pcs", type=int, nargs="+", default=[64, 16])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    S = a.fold
    cohort = pd.read_parquet(a.cohort)
    rows_in = len(cohort)
    cohort = cohort[cohort[S].isin(["train", "val"])].reset_index(drop=True)  # test rows dropped before lookup
    x, meta = features_by_name([tuple(Path(p) for p in s.split("=", 1)) for s in a.caches], cohort.img.tolist())
    sets = {}
    for split in ("train", "val"):
        rows = cohort[S] == split
        keys, cm, _ = specimen_camera_means(x[rows.values], cohort[rows].reset_index(drop=True), S)
        sets[split] = specimen_means(keys, cm, S)
    (sk_tr, x_tr), (sk_va, x_va) = sets["train"], sets["val"]
    assert sk_tr.individual.is_unique and sk_va.individual.is_unique
    assert not set(sk_tr.individual) & set(sk_va.individual), "train and val share specimens"
    y_tr, y_va = sk_tr.taxon_group.to_numpy(), sk_va.taxon_group.to_numpy()
    n_tr, n_va = pd.Series(y_tr).value_counts(), pd.Series(y_va).value_counts()
    eligible = sorted(c for c in n_tr.index if n_tr[c] >= MIN_TRAIN and n_va.get(c, 0) >= MIN_VAL)
    excluded = {c: {"train": int(n_tr.get(c, 0)), "val": int(n_va.get(c, 0))}
                for c in sorted(set(n_tr.index) | set(n_va.index)) if c not in eligible}
    pca = PCA(n_components=max(a.pcs), svd_solver="full").fit(x_tr)
    z_tr_all, z_va_all = pca.transform(x_tr), pca.transform(x_va)

    per_rows, spec_rows, results = [], [], {}
    for d in a.pcs:
        z_tr, z_va = z_tr_all[:, :d], z_va_all[:, :d]
        check = four_class_check(z_tr, y_tr, eligible)
        fit = fit_coupled(z_tr, y_tr, eligible)
        pi = fit.n / fit.n.sum()
        rng = np.random.default_rng([SEED, d])
        boots = np.empty((N_BOOT, len(eligible)))
        for k, c in enumerate(eligible):
            zt, zv = z_tr[y_tr == c], z_va[y_va == c]
            assert len(zt) == fit.n[k]
            mu = zt.mean(0)
            lw_cov, a_lw = lw_class_covariance(zt)
            covs = {"lw": lw_cov, "coupled": fit.sigmas[k],
                    "pooled_scm": coupled_covariance(fit.scm[k], fit.pooled, 1.0, 0.0),
                    "class_scm": fit.scm[k] if len(zt) > d else None,
                    "class_identity": coupled_covariance(fit.scm[k], fit.pooled, 0.0, 1.0)}
            ll, status = {}, {}
            for name, cov in covs.items():
                ll[name], status[name] = loglik(zv, mu, cov)
            diff = ll["coupled"] - ll["lw"]
            ok = status["lw"] == status["coupled"] == "ok"
            draws = rng.integers(0, len(zv), size=(N_BOOT, len(zv)))
            boots[:, k] = diff[draws].mean(1) if ok else np.nan
            nl, nc = numerics(lw_cov), numerics(fit.sigmas[k])
            row = {"pcs": d, "taxon_group": c, "n_train": len(zt), "n_val": len(zv), "stratum": stratum(len(zt)),
                   "pooled_weight_pi": float(pi[k]), "lw_alpha_shrink": a_lw,
                   "coupled_alpha_keep": float(fit.alpha[k]), "coupled_shrink_to_identity": float(1 - fit.alpha[k]),
                   "coupled_beta": float(fit.beta[k]), "coupled_weight_own_scm": float(fit.alpha[k] * fit.beta[k]),
                   "coupled_weight_pooled_scm": float(fit.alpha[k] * (1 - fit.beta[k])),
                   "eta": float(fit.eta[k]), "gamma": float(fit.gamma[k]), "kappa": float(fit.kappa[k]),
                   **{f"lw_{m}": v for m, v in nl.items()}, **{f"coupled_{m}": v for m, v in nc.items()},
                   **{f"status_{m}": status[m] for m in ESTIMATORS},
                   **{f"val_loglik_{m}": float(ll[m].mean()) for m in ESTIMATORS},
                   "diff_mean": float(diff.mean()) if ok else np.nan, "diff_median": float(np.median(diff)) if ok else np.nan,
                   "diff_ci95": ci95(boots[:, k]) if ok else [np.nan, np.nan]}
            per_rows.append(row)
            spec_rows.append(pd.DataFrame({"pcs": d, "individual": sk_va.individual.to_numpy()[y_va == c], "taxon_group": c,
                                           **{f"loglik_{m}": ll[m] for m in ESTIMATORS}, "diff_coupled_minus_lw": diff}))
        per = pd.DataFrame([r for r in per_rows if r["pcs"] == d])
        good = [c for c, s1, s2 in zip(per.taxon_group, per.status_lw, per.status_coupled) if s1 == s2 == "ok"]
        results[str(d)] = {
            "four_class_original_vs_adapted": check,
            "coupled_alpha_keep_min_median_max": [float(fit.alpha.min()), float(np.median(fit.alpha)), float(fit.alpha.max())],
            "coupled_beta_min_median_max": [float(fit.beta.min()), float(np.median(fit.beta)), float(fit.beta.max())],
            "classes_beta_0": int((fit.beta == 0).sum()), "classes_beta_1": int((fit.beta == 1).sum()),
            "classes_alpha_1": int((fit.alpha == 1).sum()),
            "numerical_failures": per.loc[~per.taxon_group.isin(good), ["taxon_group", "status_lw", "status_coupled"]].to_dict("records"),
            "all": summarise(per, boots, good),
            "strata": {name: summarise(per, boots, [c for c in good if stratum(int(n_tr[c])) == name]) for _, _, name in STRATA},
            "mean_val_loglik": {m: float(per[f"val_loglik_{m}"].mean()) for m in ESTIMATORS}}
    per, spec = pd.DataFrame(per_rows), pd.concat(spec_rows, ignore_index=True)
    per.to_csv(a.out / "perclass.csv", index=False)
    spec.to_csv(a.out / "perspecimen.csv", index=False)
    summary = {"fold": S, "encoder": meta["encoder"],
               "cohort_rows": {"in_file": rows_in, "train_val_used": len(cohort), "test_rows_dropped": rows_in - len(cohort)},
               "train_specimens": int(len(sk_tr)), "val_specimens": int(len(sk_va)), "eligible_classes": len(eligible),
               "excluded_classes": excluded, "min_train": MIN_TRAIN, "min_val": MIN_VAL, "n_boot": N_BOOT, "seed": SEED,
               "authors_code_sha256": SOURCE_SHA256, "adaptation": {"replace": PATCH[0], "with": PATCH[1]},
               "by_pcs": results}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
