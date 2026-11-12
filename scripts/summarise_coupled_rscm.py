"""Five-fold summary of coupled_rscm_vs_lw.py runs per dataset (read-only; task.md 61ceb61 and the author's approval
of the native space and the mechanism decomposition, 2026-11-12).

Folds share specimens (correlated repeats): each fold is reported separately and summarised as fold mean / min / max.
Contrasts are compared within a space only (log-likelihoods in different dimensions are not on one scale).
- folds.csv: per fold, space and contrast, macro (primary) and micro with bootstrap 95 % intervals, the Mahalanobis
  and log-determinant parts of the macro difference, classes, wins, numerical failures, 4-class original-vs-adapted
  check, convergence residuals, pooled rank, fit time;
- groups.csv: per space, contrast and class group (train-size strata 15-63, 64-255, >= 256; n_c <= d, n_c > d):
  class-folds, mean / median class difference and its parts, class-folds better, per-fold macro mean / min / max and
  folds with interval above / below 0;
- parameters.csv: per space and stratum, medians of coupled alpha (kept), 1 - alpha, beta and LW alpha; shares of
  beta = 0 / 1, alpha = 1, clipped sphericity, floored kurtosis; maximum convergence residuals; conditioning;
- corners.csv: per space and stratum, corner estimators minus LW (diagnostic only);
- reproduction.json: the LW reference in pca16 / pca64 against the earlier lw_fit_diagnostics.py per-class output;
- examples.csv / examples_by_fold.csv: instructive taxa for coupled_vs_lw per space.
Writes these per dataset plus summary.md to --out.

Usage: summarise_coupled_rscm.py --dataset NAME RUNS LW_RUNS ... --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.statistics.fold_summary import FOLDS, STRATA_MASKS, across_folds, fold_table, group_table, read_runs

CORNERS = ["coupled", "pooled_scm", "class_scm", "class_identity"]  # minus LW; corners are labelled diagnostics
PRIMARY = "coupled_vs_lw"


def summarise(runs: Path, lw_runs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    per, summ = read_runs(runs)
    spaces = list(summ[FOLDS[0]]["spaces"])
    contrasts = list(summ[FOLDS[0]]["contrasts"])
    folds = fold_table(summ, spaces, contrasts, lambda r: {
        "four_class_check_max_diff": r["four_class_original_vs_adapted"].get("max_abs_diff_alpha_beta_sigmas"),
        "residual_alpha_max": r["residual_alpha_max"], "residual_beta_max": r["residual_beta_max"],
        "pooled_rank": r["pooled"]["rank"], "fit_seconds": r["fit_seconds"]})
    folds.to_csv(out / "folds.csv", index=False)
    across, ci = across_folds(folds)
    groups = group_table(per, summ, spaces, contrasts, STRATA_MASKS)
    groups.to_csv(out / "groups.csv", index=False)

    g = per.groupby(["space", "stratum"], sort=False)
    params = pd.DataFrame({"class_folds": g.size(), "n_le_p_share": g.n_le_p.mean(),
                           "coupled_alpha_keep_median": g.coupled_alpha_keep.median(),
                           "coupled_shrink_to_identity_median": g.coupled_shrink_to_identity.median(),
                           "lw_alpha_shrink_median": g.lw_alpha_shrink.median(), "coupled_beta_median": g.coupled_beta.median(),
                           "share_beta_0": g.coupled_beta.apply(lambda v: (v == 0).mean()),
                           "share_beta_1": g.coupled_beta.apply(lambda v: (v == 1).mean()),
                           "share_alpha_1": g.coupled_alpha_keep.apply(lambda v: (v == 1).mean()),
                           "share_gamma_clipped": g.gamma_clipped.mean(), "share_kappa_floored": g.kappa_floored.mean(),
                           "residual_alpha_max": g.residual_alpha.max(), "residual_beta_max": g.residual_beta.max(),
                           "lw_condition_median": g.lw_condition.median(),
                           "coupled_condition_median": g.coupled_condition.median()}).reset_index()
    params.to_csv(out / "parameters.csv", index=False)

    corner = per.assign(**{f"{e}_minus_lw": per[f"val_loglik_{e}"] - per.val_loglik_lw for e in CORNERS})
    corners = corner.groupby(["space", "stratum"], sort=False).agg(
        class_folds=("taxon_group", "size"), class_scm_defined=("status_class_scm", lambda v: (v == "ok").sum()),
        **{f"{e}_minus_lw_mean": (f"{e}_minus_lw", "mean") for e in CORNERS},
        **{f"{e}_beats_lw": (f"{e}_minus_lw", lambda v: int((v > 0).sum())) for e in CORNERS}).reset_index()
    corners.to_csv(out / "corners.csv", index=False)

    ref = pd.concat([pd.read_csv(lw_runs / f / "perclass.csv").assign(fold=f) for f in FOLDS], ignore_index=True)
    ref["space"] = "pca" + ref.pcs.astype(str)
    here = per[per.space.isin(ref.space.unique())]
    m = here.merge(ref, on=["fold", "space", "taxon_group"], suffixes=("", "_ref"), how="outer", indicator=True)
    both = m[m._merge == "both"]
    repro = {"spaces": sorted(ref.space.unique()), "class_folds_here": len(here), "class_folds_reference": len(ref),
             "matched": len(both), "only_here": int((m._merge == "left_only").sum()),
             "only_reference": int((m._merge == "right_only").sum()),
             "max_abs_diff_val_loglik_lw": float((both.val_loglik_lw - both.val_loglik_lw_ref).abs().max()),
             "max_abs_diff_alpha_lw": float((both.lw_alpha_shrink - both.alpha_lw).abs().max()),
             "n_train_equal": bool((both.n_train == both.n_train_ref).all()), "n_val_equal": bool((both.n_val == both.n_val_ref).all())}
    (out / "reproduction.json").write_text(json.dumps(repro, indent=1))

    ex = []
    for s in spaces:
        t = per[per.space == s]
        mean = t.groupby("taxon_group").agg(diff=(f"{PRIMARY}_mean", "mean"), n_train=("n_train", "median"),
                                            beta=("coupled_beta", "median"), folds=("fold", "size"),
                                            lw_shrink=("lw_alpha_shrink", "median"),
                                            coupled_shrink=("coupled_shrink_to_identity", "median"),
                                            diff_min=(f"{PRIMARY}_mean", "min"), diff_max=(f"{PRIMARY}_mean", "max"))
        pick = {"largest_gain": mean["diff"].idxmax(), "largest_loss": mean["diff"].idxmin(),
                "smallest_class": mean.n_train.idxmin(), "largest_class": mean.n_train.idxmax(),
                "most_pooling": mean.beta.idxmin()}
        for why, c in pick.items():
            ex.append({"space": s, "why": why, "taxon_group": c, **mean.loc[c].to_dict()})
    ex = pd.DataFrame(ex)
    ex.to_csv(out / "examples.csv", index=False)
    cols = ["taxon_group", "space", "fold", "n_train", "n_val", "lw_alpha_shrink", "coupled_shrink_to_identity", "coupled_beta",
            "val_loglik_lw", "val_loglik_coupled", f"{PRIMARY}_mean", f"{PRIMARY}_ci95"]
    per[per.taxon_group.isin(ex.taxon_group)][cols].sort_values(["taxon_group", "space", "fold"]).to_csv(
        out / "examples_by_fold.csv", index=False)
    resources = {f: {"total_seconds": summ[f]["total_seconds"], "max_rss_gb": summ[f]["max_rss_gb"],
                     **{f"{s}_seconds": summ[f]["spaces"][s]["space_seconds"] for s in spaces}} for f in FOLDS}
    (out / "resources.json").write_text(json.dumps(resources, indent=1))
    return {"folds": folds, "across": across, "ci": ci, "groups": groups, "params": params, "corners": corners,
            "repro": repro, "examples": ex, "resources": resources}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=3, action="append", required=True, metavar=("NAME", "RUNS", "LW_RUNS"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    lines = []
    for name, runs, lw_runs in a.dataset:
        s = summarise(Path(runs), Path(lw_runs), a.out / name)
        lines += [f"## {name}", "", "Across folds (fold mean, min, max; within-space contrasts):", "",
                  s["across"].round(4).to_string(), "", "Folds with 95 % interval above / below 0:", "", s["ci"].to_string(), "",
                  "By class group:", "", s["groups"].round(4).to_string(index=False), "",
                  "Parameters:", "", s["params"].round(4).to_string(index=False), "",
                  "Corners (diagnostic; minus LW):", "", s["corners"].round(4).to_string(index=False), "",
                  "LW reference vs lw_fit_diagnostics:", "", json.dumps(s["repro"], indent=1), "",
                  "Examples (coupled_vs_lw):", "", s["examples"].round(4).to_string(index=False), "",
                  "Resources per fold:", "", json.dumps(s["resources"], indent=1), ""]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
