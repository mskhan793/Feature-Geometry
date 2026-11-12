"""Class-balanced and taxonomy-informed covariance pooling, one fold (task.md f54f534; our experimental proposal —
design fixed before any run, see src/methods/taxonomic_pooling.py).

Protocol as coupled_rscm_vs_lw.py (src/statistics/heldout_density.py): specimen embeddings, test rows dropped before
the feature lookup, train-fitted PCA (16 / 64) or native coordinates without one coordinate, train class means,
eligible targets >= 15 train and >= 5 val specimens. Per eligible target c and space:

References (unchanged sources): ``lw`` per-class Ledoit-Wolf; ``coupled`` the authors' CRSCM (class-specific alpha, beta).
One fixed rule for every pool P (Eq. 4 form): T = beta_c S_c + (1 - beta_c) P, Sigma = alpha T + (1 - alpha)(tr T/d) I,
beta_c = the authors' CRSCM beta (tuned for their pool; a fixed diagnostic for the others), alpha = 1 - alpha_LW,c.
- ``pool_spec``: P = specimen-weighted pool of the eligible classes (= the authors' pooled SCM; this arm equals the
  earlier 'pooling_lw_shrink' cell and is checked against it);
- ``pool_bal``: P = balanced pool of the same eligible classes (stage 1: same classes, same S_j, only the weights);
- ``pool_bal_loo``: P = balanced pool of all other donor candidates (classes with >= 15 train specimens), target left
  out (the global fallback of stage 2);
- ``pool_tax``: P = balanced pool of the taxonomic donors (genus, else family, else global = ``pool_bal_loo``);
- ``pool_near``: P = balanced pool of the nearest-mean donors (non-taxonomic control; only where a taxonomic level was
  accepted).
S_j are the unbiased class covariances (the authors' ``scov``; for eligible classes the CRSCM fit's own matrices).
Contrasts (paired per val specimen; macro primary, micro secondary; same bootstrap draws for every contrast):
balancing = pool_bal - pool_spec; taxonomy = pool_tax - pool_bal_loo; taxonomy_vs_nearest = pool_tax - pool_near;
leave_target_out = pool_bal_loo - pool_bal (diagnostic); each pool vs lw; coupled_vs_lw (reproduction).
Groups: all, train-size strata, n_c <= d / > d, taxonomic level used (genus / family / any) vs global fallback.
Numerical failures (covariance numerically singular / not PD) are reported, never jittered.

Writes perclass.csv, perspecimen.csv and summary.json to --out.

Usage: balanced_taxonomic_pooling.py --dataset finbenthic2 --cohort selection.parquet --caches MANIFEST=CACHE ...
       --fold foldK --out DIR (--fin-table T | --aqua-mapping M --aqua-table T) [--spaces pca64 pca16 native native_alt]
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import taxonomy as tx
from src.methods.coupled_rscm import adapted_module, coupled_covariance, fit_coupled
from src.methods.taxonomic_pooling import donor_df, nearest_donors, pool, taxonomy_donors
from src.statistics.covariance_fit import lw_class_covariance
from src.statistics.heldout_density import (MIN_TRAIN, MIN_VAL, N_BOOT, class_groups, contrast_columns,
                                            full_coordinate_rank, load_fold, loglik, numerics, score, spaces_from,
                                            stratum, summarise_contrasts)

SEED = 20261112
ARMS = ["lw", "coupled", "pool_spec", "pool_bal", "pool_bal_loo", "pool_tax", "pool_near"]
CONTRASTS = {"coupled_vs_lw": ("coupled", "lw"), "balancing": ("pool_bal", "pool_spec"),
             "taxonomy": ("pool_tax", "pool_bal_loo"), "taxonomy_vs_nearest": ("pool_tax", "pool_near"),
             "leave_target_out": ("pool_bal_loo", "pool_bal"), "pool_spec_vs_lw": ("pool_spec", "lw"),
             "pool_bal_vs_lw": ("pool_bal", "lw"), "pool_tax_vs_lw": ("pool_tax", "lw")}
NOT_DEFINED = "not defined (no taxonomic level)"


def groups() -> dict:
    return {**class_groups(), "taxonomy_used": lambda r: r.level != "global", "fallback": lambda r: r.level == "global",
            "genus": lambda r: r.level == "genus", "family": lambda r: r.level == "family"}


def run_space(name, d, z_tr, z_va, f, taxonomy):
    assert np.isfinite(z_tr).all() and np.isfinite(z_va).all(), f"non-finite inputs in {name}"
    y_tr, y_va, eligible = f.y_tr, f.y_va, f.eligible
    candidates = sorted(c for c in f.n_tr.index if f.n_tr[c] >= MIN_TRAIN)
    n = {c: int(f.n_tr[c]) for c in candidates}
    t0 = time.perf_counter()
    fit = fit_coupled(z_tr, y_tr, eligible)
    scov = adapted_module().scov
    scm = {c: fit.scm[k] for k, c in enumerate(eligible)}
    scm |= {c: np.asarray(scov(np.asarray(z_tr[y_tr == c], np.float64))) for c in candidates if c not in scm}
    means = {c: z_tr[y_tr == c].mean(0) for c in candidates}
    p_spec, p_bal = fit.pooled, pool(scm, n, eligible, "balanced")
    assert np.allclose(p_spec, pool(scm, n, eligible, "specimen"))
    rng = np.random.default_rng([SEED, d])
    boots = {c: np.empty((N_BOOT, len(eligible))) for c in CONTRASTS}
    rows, spec = [], []
    for k, c in enumerate(eligible):
        zt, zv = z_tr[y_tr == c], z_va[y_va == c]
        mu = zt.mean(0)
        lw_cov, a_lw = lw_class_covariance(zt)
        be = float(fit.beta[k])
        level, tax_donors = taxonomy_donors(c, taxonomy, candidates, n, d)
        loo = [o for o in candidates if o != c]
        near = nearest_donors(c, means, candidates, len(tax_donors), n, d) if level != "global" else []
        pools = {"pool_spec": p_spec, "pool_bal": p_bal, "pool_bal_loo": pool(scm, n, loo, "balanced"),
                 "pool_tax": pool(scm, n, tax_donors, "balanced"),
                 "pool_near": pool(scm, n, near, "balanced") if near else None}
        covs = {"lw": lw_cov, "coupled": fit.sigmas[k],
                **{m: coupled_covariance(scm[c], p, 1 - a_lw, be) if p is not None else None for m, p in pools.items()}}
        parts = {m: score(zv, mu, cov) if cov is not None else (np.full(len(zv), np.nan),) * 2 + (NOT_DEFINED,)
                 for m, cov in covs.items()}
        ll = {m: loglik(p, d) for m, p in parts.items()}
        draws = rng.integers(0, len(zv), size=(N_BOOT, len(zv)))
        row = {"space": name, "dim": d, "taxon_group": c, "n_train": len(zt), "n_val": len(zv), "stratum": stratum(len(zt)),
               "n_le_p": len(zt) <= d, "rank": taxonomy.loc[c, "rank"] if c in taxonomy.index else None,
               "level": level, "donor_classes": len(tax_donors), "donor_df": donor_df(n, tax_donors),
               "donors": ";".join(tax_donors) if level != "global" else "", "near_donors": ";".join(near),
               "near_tax_jaccard": len(set(near) & set(tax_donors)) / len(set(near) | set(tax_donors)) if near else np.nan,
               "coupled_beta": be, "coupled_alpha_keep": float(fit.alpha[k]), "lw_alpha_shrink": a_lw,
               **{f"identity_scale_{m}": float(np.trace(be * scm[c] + (1 - be) * p) / d) for m, p in pools.items() if p is not None},
               **{f"condition_{m}": numerics(cov)["condition"] for m, cov in covs.items() if cov is not None},
               **{f"status_{m}": parts[m][2] for m in ARMS}, **{f"val_loglik_{m}": float(ll[m].mean()) for m in ARMS}}
        frame = {"space": name, "individual": f.sk_va.individual.to_numpy()[y_va == c], "taxon_group": c,
                 **{f"loglik_{m}": ll[m] for m in ARMS}}
        for key, (a, b) in CONTRASTS.items():
            cols, boots[key][:, k], frame[f"d_{key}"] = contrast_columns(key, parts[a], parts[b], d, draws)
            row |= cols
        rows.append(row)
        spec.append(pd.DataFrame(frame))
    per = pd.DataFrame(rows)
    result = {"dim": d, "seconds": time.perf_counter() - t0, "candidates": len(candidates),
              "candidates_not_eligible": sorted(set(candidates) - set(eligible)),
              "level_counts": per.level.value_counts().to_dict(),
              "pool_numerics": {"spec": numerics(p_spec), "bal": numerics(p_bal)},
              "status_counts": {m: per[f"status_{m}"].value_counts().to_dict() for m in ARMS},
              "contrasts": summarise_contrasts(per, boots, CONTRASTS, groups()),
              "mean_val_loglik": {m: float(per[f"val_loglik_{m}"].mean()) for m in ARMS}}
    return per, pd.concat(spec, ignore_index=True), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["finbenthic2", "aquamonitor"])
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--caches", required=True, nargs="+")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fin-table", type=Path)
    ap.add_argument("--aqua-mapping", type=Path)
    ap.add_argument("--aqua-table", type=Path)
    ap.add_argument("--spaces", nargs="+", default=["pca64", "pca16", "native", "native_alt"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    f = load_fold(a.cohort, a.caches, a.fold)
    taxonomy = tx.load(a.dataset, sorted(set(f.y_tr) | set(f.y_va)), a.fin_table, a.aqua_mapping, a.aqua_table)
    pers, specs, results = [], [], {}
    for name, d, z_tr, z_va in spaces_from(a.spaces, f.x_tr, f.x_va):
        per, spec, res = run_space(name, d, z_tr, z_va, f, taxonomy)
        results[name] = res
        pers.append(per)
        specs.append(spec)
        print(f"{name}: {res['seconds']:.1f} s", flush=True)
    pd.concat(pers, ignore_index=True).to_csv(a.out / "perclass.csv", index=False)
    pd.concat(specs, ignore_index=True).to_csv(a.out / "perspecimen.csv", index=False)
    summary = {"dataset": a.dataset, "fold": f.fold, "encoder": f.encoder, "cohort_rows": f.rows,
               "train_specimens": int(len(f.sk_tr)), "val_specimens": int(len(f.sk_va)), "eligible_classes": len(f.eligible),
               "excluded_classes": f.excluded, "min_train": MIN_TRAIN, "min_val": MIN_VAL, "n_boot": N_BOOT, "seed": SEED,
               "taxonomy_rank_counts": taxonomy["rank"].value_counts(dropna=False).to_dict(), "contrasts": CONTRASTS,
               "full_coordinate_within_class_covariance": full_coordinate_rank(f.x_tr, f.y_tr),
               "total_seconds": time.perf_counter() - t_start,
               "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, "spaces": results}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "spaces"}, indent=1, default=str))


if __name__ == "__main__":
    main()
