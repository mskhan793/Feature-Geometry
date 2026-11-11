"""Five-fold summary of lw_fit_diagnostics.py runs per dataset (read-only).

Folds share specimens (correlated repeats): values are compared with the fold's own reference (all selected frames,
all train specimens) and then summarised as fold mean / min / max. Seeds of a capped configuration are averaged within
a fold first.
- pooled: per frames x cap, alpha_LW, condition number, held-out log-likelihood minus the reference (macro, micro),
  FDA-16 subspace similarity and FDA-16 GBC Spearman with the reference;
- alpha: at the reference, the grid alpha with the best macro held-out log-likelihood vs alpha_LW, and the gains of
  that alpha over alpha_LW and of alpha_LW over the sample covariance (alpha 0);
- per class (PCA spaces): which covariance estimator has the best held-out log-likelihood, by train-specimen bin;
  alpha_c vs n_c; the learning curve (alpha and log-likelihood of LW and spherical vs subsampled n);
- LW full-covariance pairs: Spearman with the spherical form and with d' (val), covariance-term share.
Writes <dataset>/{pooled.csv, alpha.csv, perclass.csv, curve.csv, pairs.csv} and summary.md to --out.

Usage: summarise_lw_fit.py --dataset FIN-Benthic2 RUNS --dataset AquaMonitor RUNS --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = [f"fold{k}" for k in range(5)]
BINS = [0, 64, 256, np.inf]
BIN_LABELS = ["n<64", "64<=n<256", "n>=256"]


def agg(t: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    g = t.groupby(by)[cols]
    out = pd.concat({"mean": g.mean(), "min": g.min(), "max": g.max()}, axis=1)
    out.columns = [f"{c}_{s}" for s, c in out.columns]
    return out.reset_index()


def summarise(runs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    pooled, alpha, per, curve, pairs = [], [], [], [], []
    for f in FOLDS:
        p = pd.read_csv(runs / f / "pooled.csv")
        p["cap"] = p.cap.astype(str)
        p = p.groupby(["frames", "cap"], as_index=False).mean(numeric_only=True)
        ref = p[(p.frames == p.frames.max()) & (p.cap == "all")].iloc[0]
        p["d_loglik_macro"] = p.val_loglik_macro_lw - ref.val_loglik_macro_lw
        p["d_loglik_micro"] = p.val_loglik_micro_lw - ref.val_loglik_micro_lw
        pooled.append(p.assign(fold=f))
        c = pd.read_csv(runs / f / "alpha_curve.csv")
        c["cap"] = c.cap.astype(str)
        r = c[(c.frames == c.frames.max()) & (c.cap == "all")]
        grid = r[~r.is_lw].dropna(subset=["val_loglik_macro"])
        best = grid.loc[grid.val_loglik_macro.idxmax()]
        lw = r[r.is_lw].iloc[0]
        sample = r[(~r.is_lw) & (r.alpha == 0)].iloc[0]
        alpha.append({"fold": f, "alpha_lw": lw.alpha, "best_grid_alpha": best.alpha,
                      "gain_best_grid_over_lw_macro": best.val_loglik_macro - lw.val_loglik_macro,
                      "gain_lw_over_sample_macro": lw.val_loglik_macro - sample.val_loglik_macro})
        q = pd.read_csv(runs / f / "perclass.csv").assign(fold=f)
        per.append(q)
        curve.append(pd.read_csv(runs / f / "perclass_curve.csv").assign(fold=f))
        for d in sorted(q.pcs.unique()):
            t = pd.read_csv(runs / f / f"pairs_lw_pcs{d}.csv")
            share = t.lw_full_cov_term / t.lw_full_distance
            pairs.append({"fold": f, "pcs": d, "pairs": len(t),
                          "spearman_lw_full_vs_spherical": t.lw_full_distance.corr(t.spherical_distance, method="spearman"),
                          "spearman_lw_full_vs_dprime_val": t.lw_full_distance.corr(t.dprime_val, method="spearman"),
                          "spearman_spherical_vs_dprime_val": t.spherical_distance.corr(t.dprime_val, method="spearman"),
                          "lw_full_cov_term_share_median": float(share.median())})
    pooled = agg(pd.concat(pooled), ["frames", "cap"], ["train_specimens", "alpha_lw", "condition_s_w_star", "d_loglik_macro",
                                                        "d_loglik_micro", "fda16_subspace_similarity",
                                                        "fda16_gbc_spearman_vs_reference"])
    pooled.to_csv(out / "pooled.csv", index=False)
    alpha = pd.DataFrame(alpha)
    alpha.to_csv(out / "alpha.csv", index=False)
    per = pd.concat(per)
    per["n_bin"] = pd.cut(per.n_train, BINS, labels=BIN_LABELS, right=False)
    est = [c.removeprefix("val_loglik_") for c in per if c.startswith("val_loglik_")]
    per["best"] = per[[f"val_loglik_{e}" for e in est]].idxmax(axis=1).str.removeprefix("val_loglik_")
    wins = per.groupby(["pcs", "n_bin"], observed=True).best.value_counts().unstack(fill_value=0)
    wins.to_csv(out / "perclass_best_estimator.csv")
    alpha_n = per.groupby(["pcs", "fold"]).apply(lambda t: t.alpha_lw.corr(t.n_train, method="spearman"), include_groups=False)
    curve = pd.concat(curve)
    curve_s = curve.groupby(["pcs", "n"]).agg(alpha_lw_mean=("alpha_lw", "mean"),
                                              lw_minus_spherical_loglik=("val_loglik_lw", "mean"),
                                              spherical=("val_loglik_spherical", "mean"), classes=("taxon_group", "nunique"))
    curve_s["lw_minus_spherical_loglik"] -= curve_s.pop("spherical")
    curve_s.reset_index().to_csv(out / "curve.csv", index=False)
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / "pairs.csv", index=False)
    return {"pooled": pooled, "alpha": alpha, "wins": wins, "alpha_n": alpha_n, "curve": curve_s,
            "pairs": pairs.drop(columns="fold").groupby("pcs").agg(["mean", "min", "max"]).round(3), "per": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=2, action="append", required=True, metavar=("NAME", "RUNS"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    lines = []
    for name, runs in a.dataset:
        s = summarise(Path(runs), a.out / name)
        lines += [f"## {name}", "", "Pooled S_W (fold mean [min, max]; d_loglik = held-out log-likelihood minus the reference):", "",
                  s["pooled"].round(4).to_string(index=False), ""]
        lines += ["Alpha at the reference (per fold):", "", s["alpha"].round(5).to_string(index=False), ""]
        lines += ["Best per-class estimator by train-specimen bin (count over classes x folds):", "", s["wins"].to_string(), ""]
        lines += ["Spearman alpha_c vs n_c per fold:", "", s["alpha_n"].round(3).to_string(), ""]
        lines += ["Learning curve (mean over large classes and folds):", "", s["curve"].round(4).to_string(), ""]
        lines += ["LW full-covariance pairs:", "", s["pairs"].to_string(), ""]
        lines += ["Mean held-out log-likelihood per estimator:", "",
                  s["per"].groupby("pcs")[[c for c in s["per"] if c.startswith("val_loglik_")]].mean().round(3).to_string(), ""]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
