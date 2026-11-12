"""Per-class Ledoit-Wolf versus the coupled RSCM estimator (Raninen & Ollila 2021) on held-out specimens, one fold
(task.md 61ceb61; native space and mechanism decomposition approved by the author on 2026-11-12 after the methods
review, response.md section 7).

Same protocol as section 8 D3 of lw_fit_diagnostics.py: specimen embedding = mean of its camera means over all
selected train frames; test rows dropped before the feature lookup and never used; class mean = train sample mean;
eligible classes = >= 15 train and >= 5 val specimens (every estimator uses exactly these classes and specimens; the
coupled estimator's pooled matrix is over the eligible classes). Spaces (``--spaces``): ``pcaN`` = first N components
of a PCA fitted on the fold's train specimens; ``native`` = the original frozen coordinates without the last one,
``native_alt`` = without the first one (no PCA, no rotation, no whitening; the authors' coordinatewise kurtosis plug-in
is not exactly rotation-invariant). One coordinate is dropped because the encoder's final LayerNorm makes every
feature vector satisfy sum_j (x_j - b_j)/w_j = 0, so all covariances of the 1024 coordinates are singular along one
direction (audit: scripts/audit_feature_null_space.py; here the rank of the full-coordinate within-class covariance is
recorded per fold). Any one coordinate is an exact linear function of the others, so dropping it loses no
information; the Jacobian of that change of variables is the same for every estimator and cancels in paired
differences. ``native`` vs ``native_alt`` shows the sensitivity to which coordinate is dropped.

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
failures are reported as failures (no jitter, no pseudo-inverse): a covariance whose smallest eigenvalue is
<= d * float64 eps * its largest (numpy.linalg.matrix_rank's default tolerance) is "numerically singular" and not
scored, even if a Cholesky factorisation would succeed on rounding error. Numerical audit: finite inputs, constant
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

from src.methods.coupled_rscm import (PATCHES, SOURCE_SHA256, adapted_module, coupled_covariance, fit_coupled,
                                      original_module)
from src.statistics.covariance_fit import lw_class_covariance
from src.statistics.heldout_density import (MIN_TRAIN, MIN_VAL, N_BOOT, class_groups, contrast_columns,
                                            full_coordinate_rank, load_fold, loglik, numerics, score, spaces_from,
                                            stratum, summarise_contrasts)

SEED = 20261112
ESTIMATORS = ["lw", "coupled", "lw_unbiased", "coupled_no_pooling", "pooling_lw_shrink", "pooled_scm", "class_scm",
              "class_identity"]
CONTRASTS = {"coupled_vs_lw": ("coupled", "lw"), "normalisation": ("lw_unbiased", "lw"),
             "pooling_at_lw_shrink": ("pooling_lw_shrink", "lw_unbiased"),
             "pooling_at_coupled_shrink": ("coupled", "coupled_no_pooling"),
             "shrink_at_no_pooling": ("coupled_no_pooling", "lw_unbiased"),
             "shrink_at_coupled_pooling": ("coupled", "pooling_lw_shrink")}


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
        ll = {m: loglik(p, d) for m, p in parts.items()}
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
            cols, boots[key][:, k], frame[f"d_{key}"] = contrast_columns(key, parts[a], parts[b], d, draws)
            row |= cols
        rows.append(row)
        spec.append(pd.DataFrame(frame))
    per = pd.DataFrame(rows)
    contrasts = summarise_contrasts(per, boots, CONTRASTS, class_groups())
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
    ap.add_argument("--spaces", nargs="+", default=["pca64", "pca16", "native", "native_alt"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    f = load_fold(a.cohort, a.caches, a.fold)
    y_tr, y_va = f.y_tr, f.y_va
    full = full_coordinate_rank(f.x_tr, y_tr)
    pers, specs, results = [], [], {}
    for name, d, z_tr, z_va in spaces_from(a.spaces, f.x_tr, f.x_va):
        t0 = time.perf_counter()
        per, spec, res = run_space(name, d, z_tr, z_va, y_tr, y_va, f.sk_va, f.eligible)
        results[name] = {**res, "space_seconds": time.perf_counter() - t0}
        pers.append(per)
        specs.append(spec)
        print(f"{name}: {results[name]['space_seconds']:.1f} s", flush=True)
    pd.concat(pers, ignore_index=True).to_csv(a.out / "perclass.csv", index=False)
    pd.concat(specs, ignore_index=True).to_csv(a.out / "perspecimen.csv", index=False)
    summary = {"fold": f.fold, "encoder": f.encoder, "cohort_rows": f.rows,
               "train_specimens": int(len(f.sk_tr)), "val_specimens": int(len(f.sk_va)), "eligible_classes": len(f.eligible),
               "excluded_classes": f.excluded, "min_train": MIN_TRAIN, "min_val": MIN_VAL, "n_boot": N_BOOT, "seed": SEED,
               "authors_code_sha256": SOURCE_SHA256, "adaptations": [{"replace": f, "with": r} for f, r in PATCHES],
               "contrasts": CONTRASTS, "full_coordinate_within_class_covariance": full, "total_seconds": time.perf_counter() - t_start,
               "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, "spaces": results}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "spaces"}, indent=1, default=str))


if __name__ == "__main__":
    main()
