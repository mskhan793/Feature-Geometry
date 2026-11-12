"""Per-class Ledoit-Wolf versus the coupled RSCM estimator (Raninen & Ollila 2021) on held-out specimens, one fold
(task.md 61ceb61; native space and mechanism decomposition approved by the author on 2026-11-12 after the methods
review, response.md section 7).

Same protocol as section 8 D3 of lw_fit_diagnostics.py: specimen embedding = mean of its camera means over all
selected train frames; test rows dropped before the feature lookup and never used; class mean = train sample mean;
eligible classes = >= 15 train and >= 5 val specimens (every estimator uses exactly these classes and specimens; the
coupled estimator's pooled matrix is over the eligible classes). Spaces (``--spaces``): ``pcaN`` = first N components
of a PCA fitted on the fold's train specimens; ``native`` = the original frozen coordinates (no PCA, no rotation, no
whitening; the authors' coordinatewise kurtosis plug-in is not exactly rotation-invariant).

Per class and space:
- (A) per-class Ledoit-Wolf full covariance (``lw``; biased class covariance, reference) and (B) the coupled RSCM with
  the authors' data-estimated class-specific (alpha_k, beta_k) (``coupled``; unbiased S_k).
- Mechanism decomposition (labelled diagnostics; Eq. 4 with fixed parameters, all on the unbiased S_k and the target
  (tr T/p) I): ``lw_unbiased`` = no pooling, LW identity shrinkage (beta 1, alpha 1 - alpha_LW);
  ``coupled_no_pooling`` = beta 1, the coupled alpha_k; ``pooling_lw_shrink`` = the coupled beta_k, alpha 1 - alpha_LW;
  the coupled estimator itself = the coupled beta_k and alpha_k. Contrasts: normalisation = lw_unbiased - lw;
  pooling at LW / coupled identity shrinkage; identity shrinkage at no / coupled pooling (2 x 2). alpha_k and beta_k
  were tuned jointly by the authors' method, so mixed cells are not estimators anyone proposed.
- Corners of Eq. (4), no tuning: pooled SCM, class SCM (only if n_c > d), class scaled identity.
Nothing is selected on validation.

Measurement: held-out Gaussian log-likelihood of each val specimen under N(mu_c, Sigma_c) of its own class, split into
-maha/2 and -log|Sigma|/2; paired differences per specimen for every contrast; class-macro (primary) and
specimen-micro (secondary) means with a stratified bootstrap over val specimens (the same draws for every contrast;
conditional on the train fits); strata 15-63, 64-255, >= 256 train specimens and n_c <= d vs n_c > d. Numerical
failures are reported as failures (no jitter, no pseudo-inverse). Numerical audit: finite inputs, constant
coordinates, symmetry / minimum eigenvalue / condition number, pooled-matrix rank, the authors' sphericity clip and
kurtosis floor, separate alpha and beta convergence residuals. A 4-class check runs the unmodified authors' code and
the adapted one on the first four eligible classes of every space and requires identical output.

Writes perclass.csv, perspecimen.csv and summary.json to --out.

Usage: coupled_rscm_vs_lw.py --cohort selection.parquet --caches MANIFEST=CACHE ... --fold foldK --out DIR
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.features.feature_lookup import features_by_name
from src.features.specimen_means import specimen_camera_means, specimen_means
from src.methods.coupled_rscm import (PATCHES, SOURCE_SHA256, adapted_module, coupled_covariance, fit_coupled,
                                      original_module)
from src.statistics.class_macro import ci95
from src.statistics.covariance_fit import gaussian_parts, lw_class_covariance

MIN_TRAIN, MIN_VAL = 15, 5  # as lw_fit_diagnostics.py (section 8 D3)
STRATA = [(15, 64, "15-63"), (64, 256, "64-255"), (256, np.inf, ">=256")]
N_BOOT, SEED = 2000, 20261112
ESTIMATORS = ["lw", "coupled", "lw_unbiased", "coupled_no_pooling", "pooling_lw_shrink", "pooled_scm", "class_scm",
              "class_identity"]
CONTRASTS = {"coupled_vs_lw": ("coupled", "lw"), "normalisation": ("lw_unbiased", "lw"),
             "pooling_at_lw_shrink": ("pooling_lw_shrink", "lw_unbiased"),
             "pooling_at_coupled_shrink": ("coupled", "coupled_no_pooling"),
             "shrink_at_no_pooling": ("coupled_no_pooling", "lw_unbiased"),
             "shrink_at_coupled_pooling": ("coupled", "pooling_lw_shrink")}


def stratum(n: int) -> str:
    return next(name for lo, hi, name in STRATA if lo <= n < hi)


def numerics(cov: np.ndarray) -> dict:
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
    return {"asymmetry": float(np.abs(cov - cov.T).max()), "min_eig": float(eig[0]),
            "condition": float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")}


def score(z: np.ndarray, mu: np.ndarray, cov: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, str]:
    """-maha/2 and -log|cov|/2 per row (NaN with a status if the covariance is undefined or not positive definite)."""
    nan = np.full(len(z), np.nan)
    if cov is None:
        return nan, nan, "not defined (n <= d)"
    try:
        maha, logdet = gaussian_parts(z, mu, cov)
    except np.linalg.LinAlgError:
        return nan, nan, "failed: not positive definite"
    return -0.5 * maha, np.full(len(z), -0.5 * logdet), "ok"


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


def summarise(per: pd.DataFrame, boots: np.ndarray, classes: list, key: str) -> dict:
    """Macro / micro of contrast ``key`` over ``classes`` (rows of ``per`` and columns of ``boots`` in that order)."""
    if not classes:
        return {"classes": 0}
    t = per.set_index("taxon_group").loc[classes]
    cols = [list(per.taxon_group).index(c) for c in classes]
    w = t.n_val.to_numpy() / t.n_val.sum()
    return {"classes": len(classes), "val_specimens": int(t.n_val.sum()),
            "macro_mean": float(t[f"{key}_mean"].mean()), "macro_ci95": ci95(boots[:, cols].mean(1)),
            "micro_mean": float((t[f"{key}_mean"] * w).sum()), "micro_ci95": ci95(boots[:, cols] @ w),
            "macro_maha_part": float(t[f"{key}_maha_part"].mean()), "macro_logdet_part": float(t[f"{key}_logdet_part"].mean()),
            "class_diff_median": float(t[f"{key}_mean"].median()), "classes_better": int((t[f"{key}_mean"] > 0).sum()),
            "classes_ci_above_0": int(sum(v[0] > 0 for v in t[f"{key}_ci95"])),
            "classes_ci_below_0": int(sum(v[1] < 0 for v in t[f"{key}_ci95"]))}


def spaces_from(names: list[str], x_tr: np.ndarray, x_va: np.ndarray):
    """(name, dim, train, val) per space; PCA fitted once on the train specimens for all ``pcaN``."""
    dims = [int(s.removeprefix("pca")) for s in names if s.startswith("pca")]
    pca = PCA(n_components=max(dims), svd_solver="full").fit(x_tr) if dims else None
    zt, zv = (pca.transform(x_tr), pca.transform(x_va)) if pca else (None, None)
    for s in names:
        if s == "native":
            yield s, x_tr.shape[1], np.asarray(x_tr, np.float64), np.asarray(x_va, np.float64)
        else:
            d = int(s.removeprefix("pca"))
            yield s, d, zt[:, :d], zv[:, :d]


def run_space(name, d, z_tr, z_va, y_tr, y_va, sk_va, eligible):
    assert np.isfinite(z_tr).all() and np.isfinite(z_va).all(), f"non-finite inputs in {name}"
    t0 = time.perf_counter()
    check = four_class_check(z_tr, y_tr, eligible)
    fit = fit_coupled(z_tr, y_tr, eligible)
    fit_seconds = time.perf_counter() - t0
    pi = fit.n / fit.n.sum()
    rng = np.random.default_rng([SEED, d])
    boots = {c: np.empty((N_BOOT, len(eligible))) for c in CONTRASTS}
    rows, spec = [], []
    for k, c in enumerate(eligible):
        zt, zv = z_tr[y_tr == c], z_va[y_va == c]
        assert len(zt) == fit.n[k]
        mu = zt.mean(0)
        lw_cov, a_lw = lw_class_covariance(zt)
        s_k, al, be = fit.scm[k], float(fit.alpha[k]), float(fit.beta[k])
        covs = {"lw": lw_cov, "coupled": fit.sigmas[k],
                "lw_unbiased": coupled_covariance(s_k, fit.pooled, 1 - a_lw, 1.0),
                "coupled_no_pooling": coupled_covariance(s_k, fit.pooled, al, 1.0),
                "pooling_lw_shrink": coupled_covariance(s_k, fit.pooled, 1 - a_lw, be),
                "pooled_scm": coupled_covariance(s_k, fit.pooled, 1.0, 0.0),
                "class_scm": s_k if len(zt) > d else None,
                "class_identity": coupled_covariance(s_k, fit.pooled, 0.0, 1.0)}
        parts = {m: score(zv, mu, cov) for m, cov in covs.items()}
        ll = {m: p[0] + p[1] - 0.5 * d * np.log(2 * np.pi) for m, p in parts.items()}
        draws = rng.integers(0, len(zv), size=(N_BOOT, len(zv)))
        nl, nc = numerics(lw_cov), numerics(fit.sigmas[k])
        row = {"space": name, "dim": d, "taxon_group": c, "n_train": len(zt), "n_val": len(zv), "stratum": stratum(len(zt)),
               "n_over_p": len(zt) / d, "n_le_p": len(zt) <= d, "constant_coords": int((zt.std(0) == 0).sum()),
               "pooled_weight_pi": float(pi[k]), "lw_alpha_shrink": a_lw, "coupled_alpha_keep": al,
               "coupled_shrink_to_identity": 1 - al, "coupled_beta": be, "coupled_weight_own_scm": al * be,
               "coupled_weight_pooled_scm": al * (1 - be), "eta": float(fit.eta[k]), "gamma": float(fit.gamma[k]),
               "gamma_clipped": bool(fit.gamma[k] in (1.0, float(d))), "kappa": float(fit.kappa[k]),
               "kappa_floored": bool(np.isclose(fit.kappa[k], -2 / (d + 2))),
               "residual_alpha": float(fit.residual_alpha[k]), "residual_beta": float(fit.residual_beta[k]),
               **{f"lw_{m}": v for m, v in nl.items()}, **{f"coupled_{m}": v for m, v in nc.items()},
               **{f"status_{m}": parts[m][2] for m in ESTIMATORS},
               **{f"val_loglik_{m}": float(ll[m].mean()) for m in ESTIMATORS}}
        frame = {"space": name, "individual": sk_va.individual.to_numpy()[y_va == c], "taxon_group": c,
                 **{f"loglik_{m}": ll[m] for m in ESTIMATORS}}
        for key, (a, b) in CONTRASTS.items():
            ok = parts[a][2] == parts[b][2] == "ok"
            diff = ll[a] - ll[b]
            boots[key][:, k] = diff[draws].mean(1) if ok else np.nan
            row |= {f"{key}_mean": float(diff.mean()) if ok else np.nan,
                    f"{key}_median": float(np.median(diff)) if ok else np.nan,
                    f"{key}_ci95": ci95(boots[key][:, k]) if ok else [np.nan, np.nan],
                    f"{key}_maha_part": float((parts[a][0] - parts[b][0]).mean()) if ok else np.nan,
                    f"{key}_logdet_part": float((parts[a][1] - parts[b][1]).mean()) if ok else np.nan}
            frame[f"d_{key}"] = diff
        rows.append(row)
        spec.append(pd.DataFrame(frame))
    per = pd.DataFrame(rows)
    groups = {"all": lambda r: True, **{s: (lambda r, s=s: r.stratum == s) for _, _, s in STRATA},
              "n<=p": lambda r: r.n_le_p, "n>p": lambda r: not r.n_le_p}
    contrasts = {}
    for key, (a, b) in CONTRASTS.items():
        good = [r.taxon_group for r in per.itertuples() if getattr(r, f"status_{a}") == getattr(r, f"status_{b}") == "ok"]
        sub = per.set_index("taxon_group")
        contrasts[key] = {g: summarise(per, boots[key], [c for c in good if f(sub.loc[c])], key) for g, f in groups.items()}
        contrasts[key]["numerical_failures"] = per.loc[~per.taxon_group.isin(good), ["taxon_group", f"status_{a}", f"status_{b}"]].to_dict("records")
    pooled = numerics(fit.pooled)
    result = {"dim": d, "four_class_original_vs_adapted": check, "fit_seconds": fit_seconds,
              "pooled": {**pooled, "rank": int(np.linalg.matrix_rank(fit.pooled))},
              "iterations": fit.iterations, "residual_alpha_max": float(np.nanmax(fit.residual_alpha)),
              "residual_beta_max": float(np.nanmax(fit.residual_beta)),
              "coupled_alpha_keep_min_median_max": [float(fit.alpha.min()), float(np.median(fit.alpha)), float(fit.alpha.max())],
              "coupled_beta_min_median_max": [float(fit.beta.min()), float(np.median(fit.beta)), float(fit.beta.max())],
              "classes_beta_0": int((fit.beta == 0).sum()), "classes_beta_1": int((fit.beta == 1).sum()),
              "classes_alpha_1": int((fit.alpha == 1).sum()), "classes_gamma_clipped": int(per.gamma_clipped.sum()),
              "classes_kappa_floored": int(per.kappa_floored.sum()), "classes_n_le_p": int(per.n_le_p.sum()),
              "classes_with_constant_coords": int((per.constant_coords > 0).sum()),
              "status_counts": {m: per[f"status_{m}"].value_counts().to_dict() for m in ESTIMATORS},
              "contrasts": contrasts, "mean_val_loglik": {m: float(per[f"val_loglik_{m}"].mean()) for m in ESTIMATORS}}
    return per, pd.concat(spec, ignore_index=True), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--caches", required=True, nargs="+")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--spaces", nargs="+", default=["pca64", "pca16", "native"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
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
    pers, specs, results = [], [], {}
    for name, d, z_tr, z_va in spaces_from(a.spaces, x_tr, x_va):
        t0 = time.perf_counter()
        per, spec, res = run_space(name, d, z_tr, z_va, y_tr, y_va, sk_va, eligible)
        results[name] = {**res, "space_seconds": time.perf_counter() - t0}
        pers.append(per)
        specs.append(spec)
        print(f"{name}: {results[name]['space_seconds']:.1f} s", flush=True)
    pd.concat(pers, ignore_index=True).to_csv(a.out / "perclass.csv", index=False)
    pd.concat(specs, ignore_index=True).to_csv(a.out / "perspecimen.csv", index=False)
    summary = {"fold": S, "encoder": meta["encoder"],
               "cohort_rows": {"in_file": rows_in, "train_val_used": len(cohort), "test_rows_dropped": rows_in - len(cohort)},
               "train_specimens": int(len(sk_tr)), "val_specimens": int(len(sk_va)), "eligible_classes": len(eligible),
               "excluded_classes": excluded, "min_train": MIN_TRAIN, "min_val": MIN_VAL, "n_boot": N_BOOT, "seed": SEED,
               "authors_code_sha256": SOURCE_SHA256, "adaptations": [{"replace": f, "with": r} for f, r in PATCHES],
               "contrasts": CONTRASTS, "total_seconds": time.perf_counter() - t_start,
               "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, "spaces": results}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "spaces"}, indent=1, default=str))


if __name__ == "__main__":
    main()
