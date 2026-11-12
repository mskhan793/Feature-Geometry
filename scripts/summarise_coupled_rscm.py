"""Five-fold summary of coupled_rscm_vs_lw.py runs per dataset (read-only; task.md 61ceb61).

Folds share specimens (correlated repeats): each fold is reported separately and summarised as fold mean / min / max.
- folds.csv: per fold and PCA dimension, macro (primary) and micro coupled - LW with bootstrap 95 % intervals, classes,
  wins, numerical failures and the 4-class original-vs-adapted check;
- strata.csv: per PCA dimension and train-size stratum, class-folds, mean / median class difference, classes better,
  per-fold macro mean / min / max and folds with interval above / below 0;
- parameters.csv: per PCA dimension and stratum, medians of coupled alpha (kept), shrinkage to identity (1 - alpha),
  beta and LW alpha; shares of beta = 0, beta = 1, alpha = 1; conditioning;
- corners.csv: per PCA dimension and stratum, mean held-out log-likelihood minus LW of the coupled estimator and of
  the labelled Eq. (4) corners (pooled SCM, class SCM where n > d, class scaled identity) and class-folds where each
  beats LW (diagnostic only);
- reproduction.json: the LW reference against the earlier lw_fit_diagnostics.py per-class output (same folds);
- examples.csv: instructive taxa (largest mean gain and loss over folds, smallest and largest class, most pooling);
  examples_by_fold.csv: their per-fold rows.
Writes these per dataset plus summary.md to --out.

Usage: summarise_coupled_rscm.py --dataset NAME RUNS LW_RUNS ... --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FOLDS = [f"fold{k}" for k in range(5)]
STRATA = ["15-63", "64-255", ">=256"]
CORNERS = ["coupled", "pooled_scm", "class_scm", "class_identity"]  # minus LW; corners are labelled diagnostics


def read(runs: Path):
    per, summ = [], {}
    for f in FOLDS:
        per.append(pd.read_csv(runs / f / "perclass.csv").assign(fold=f))
        summ[f] = json.loads((runs / f / "summary.json").read_text())
    return pd.concat(per, ignore_index=True), summ


def summarise(runs: Path, lw_runs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    per, summ = read(runs)
    pcs = sorted(per.pcs.unique())
    folds = []
    for f in FOLDS:
        for d in pcs:
            r = summ[f]["by_pcs"][str(d)]
            a = r["all"]
            folds.append({"fold": f, "pcs": d, "classes": a["classes"], "macro_mean": a["macro_mean"],
                          "macro_ci_low": a["macro_ci95"][0], "macro_ci_high": a["macro_ci95"][1],
                          "micro_mean": a["micro_mean"], "micro_ci_low": a["micro_ci95"][0], "micro_ci_high": a["micro_ci95"][1],
                          "classes_coupled_better": a["classes_coupled_better"], "classes_ci_above_0": a["classes_ci_above_0"],
                          "classes_ci_below_0": a["classes_ci_below_0"], "numerical_failures": len(r["numerical_failures"]),
                          "four_class_check_max_diff": r["four_class_original_vs_adapted"].get("max_abs_diff_alpha_beta_sigmas")})
    folds = pd.DataFrame(folds)
    folds.to_csv(out / "folds.csv", index=False)
    across = folds.groupby("pcs")[["macro_mean", "micro_mean"]].agg(["mean", "min", "max"])

    strata = []
    for d in pcs:
        for s in STRATA:
            t = per[(per.pcs == d) & (per.stratum == s)].dropna(subset=["diff_mean"])
            fm = [summ[f]["by_pcs"][str(d)]["strata"][s] for f in FOLDS]
            fm = [x for x in fm if x["classes"]]
            strata.append({"pcs": d, "stratum": s, "class_folds": len(t), "taxa": t.taxon_group.nunique(),
                           "class_diff_mean": t.diff_mean.mean(), "class_diff_median": t.diff_mean.median(),
                           "class_folds_coupled_better": int((t.diff_mean > 0).sum()),
                           "fold_macro_mean": pd.Series([x["macro_mean"] for x in fm]).mean(),
                           "fold_macro_min": min((x["macro_mean"] for x in fm), default=float("nan")),
                           "fold_macro_max": max((x["macro_mean"] for x in fm), default=float("nan")),
                           "folds_ci_above_0": sum(x["macro_ci95"][0] > 0 for x in fm),
                           "folds_ci_below_0": sum(x["macro_ci95"][1] < 0 for x in fm)})
    strata = pd.DataFrame(strata)
    strata.to_csv(out / "strata.csv", index=False)

    g = per.groupby(["pcs", "stratum"])
    params = pd.DataFrame({"class_folds": g.size(), "coupled_alpha_keep_median": g.coupled_alpha_keep.median(),
                           "coupled_shrink_to_identity_median": g.coupled_shrink_to_identity.median(),
                           "lw_alpha_shrink_median": g.lw_alpha_shrink.median(), "coupled_beta_median": g.coupled_beta.median(),
                           "share_beta_0": g.coupled_beta.apply(lambda v: (v == 0).mean()),
                           "share_beta_1": g.coupled_beta.apply(lambda v: (v == 1).mean()),
                           "share_alpha_1": g.coupled_alpha_keep.apply(lambda v: (v == 1).mean()),
                           "lw_condition_median": g.lw_condition.median(),
                           "coupled_condition_median": g.coupled_condition.median()}).reset_index()
    params.to_csv(out / "parameters.csv", index=False)
    corner = per.assign(**{f"{e}_minus_lw": per[f"val_loglik_{e}"] - per.val_loglik_lw for e in CORNERS})
    corners = corner.groupby(["pcs", "stratum"]).agg(
        class_folds=("taxon_group", "size"), class_scm_defined=("status_class_scm", lambda v: (v == "ok").sum()),
        **{f"{e}_minus_lw_mean": (f"{e}_minus_lw", "mean") for e in CORNERS},
        **{f"{e}_beats_lw": (f"{e}_minus_lw", lambda v: int((v > 0).sum())) for e in CORNERS}).reset_index()
    corners.to_csv(out / "corners.csv", index=False)

    ref = pd.concat([pd.read_csv(lw_runs / f / "perclass.csv").assign(fold=f) for f in FOLDS], ignore_index=True)
    m = per.merge(ref, on=["fold", "pcs", "taxon_group"], suffixes=("", "_ref"), how="outer", indicator=True)
    both = m[m._merge == "both"]
    repro = {"class_folds_here": len(per), "class_folds_reference": len(ref), "matched": len(both),
             "only_here": int((m._merge == "left_only").sum()), "only_reference": int((m._merge == "right_only").sum()),
             "max_abs_diff_val_loglik_lw": float((both.val_loglik_lw - both.val_loglik_lw_ref).abs().max()),
             "max_abs_diff_alpha_lw": float((both.lw_alpha_shrink - both.alpha_lw).abs().max()),
             "n_train_equal": bool((both.n_train == both.n_train_ref).all()), "n_val_equal": bool((both.n_val == both.n_val_ref).all())}
    (out / "reproduction.json").write_text(json.dumps(repro, indent=1))

    ex = []
    for d in pcs:
        t = per[per.pcs == d]
        mean = t.groupby("taxon_group").agg(diff=("diff_mean", "mean"), n_train=("n_train", "median"),
                                            beta=("coupled_beta", "median"), folds=("fold", "size"),
                                            lw_shrink=("lw_alpha_shrink", "median"),
                                            coupled_shrink=("coupled_shrink_to_identity", "median"),
                                            diff_min=("diff_mean", "min"), diff_max=("diff_mean", "max"))
        pick = {"largest_gain": mean["diff"].idxmax(), "largest_loss": mean["diff"].idxmin(),
                "smallest_class": mean.n_train.idxmin(), "largest_class": mean.n_train.idxmax(),
                "most_pooling": mean.beta.idxmin()}
        for why, c in pick.items():
            ex.append({"pcs": d, "why": why, "taxon_group": c, **mean.loc[c].to_dict()})
    ex = pd.DataFrame(ex)
    ex.to_csv(out / "examples.csv", index=False)
    cols = ["taxon_group", "pcs", "fold", "n_train", "n_val", "lw_alpha_shrink", "coupled_shrink_to_identity", "coupled_beta",
            "val_loglik_lw", "val_loglik_coupled", "diff_mean", "diff_ci95"]
    per[per.taxon_group.isin(ex.taxon_group)][cols].sort_values(["taxon_group", "pcs", "fold"]).to_csv(
        out / "examples_by_fold.csv", index=False)
    return {"folds": folds, "across": across, "strata": strata, "params": params, "corners": corners, "repro": repro,
            "examples": ex}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=3, action="append", required=True, metavar=("NAME", "RUNS", "LW_RUNS"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    lines = []
    for name, runs, lw_runs in a.dataset:
        s = summarise(Path(runs), Path(lw_runs), a.out / name)
        lines += [f"## {name}", "", "Per fold (coupled minus LW, held-out log-likelihood per val specimen):", "",
                  s["folds"].round(4).to_string(index=False), "", "Across folds (mean, min, max):", "",
                  s["across"].round(4).to_string(), "", "By train-size stratum:", "", s["strata"].round(4).to_string(index=False), "",
                  "Parameters:", "", s["params"].round(4).to_string(index=False), "",
                  "Corners (diagnostic; minus LW):", "", s["corners"].round(4).to_string(index=False), "",
                  "LW reference vs lw_fit_diagnostics:", "", json.dumps(s["repro"], indent=1), "",
                  "Examples:", "", s["examples"].round(4).to_string(index=False), ""]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
