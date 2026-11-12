"""Five-fold tables shared by the held-out density summarisers (summarise_coupled_rscm.py,
summarise_balanced_taxonomic.py). Folds share specimens, so each fold is reported separately and summarised as fold
mean / min / max; contrasts are compared within a space only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

FOLDS = [f"fold{k}" for k in range(5)]


def read_runs(runs: Path) -> tuple[pd.DataFrame, dict]:
    per, summ = [], {}
    for f in FOLDS:
        per.append(pd.read_csv(runs / f / "perclass.csv").assign(fold=f))
        summ[f] = json.loads((runs / f / "summary.json").read_text())
    return pd.concat(per, ignore_index=True), summ


def fold_table(summ: dict, spaces: list, contrasts: list, extra: Callable[[dict], dict] = lambda r: {}) -> pd.DataFrame:
    """Per fold, space and contrast: macro (primary) and micro with intervals, parts, wins, failures (+ ``extra``)."""
    rows = []
    for f in FOLDS:
        for s in spaces:
            r = summ[f]["spaces"][s]
            for c in contrasts:
                a = r["contrasts"][c]["all"]
                if not a["classes"]:
                    rows.append({"fold": f, "space": s, "contrast": c, "classes": 0})
                    continue
                rows.append({"fold": f, "space": s, "contrast": c, "classes": a["classes"], "macro_mean": a["macro_mean"],
                             "macro_ci_low": a["macro_ci95"][0], "macro_ci_high": a["macro_ci95"][1],
                             "macro_maha_part": a["macro_maha_part"], "macro_logdet_part": a["macro_logdet_part"],
                             "micro_mean": a["micro_mean"], "micro_ci_low": a["micro_ci95"][0], "micro_ci_high": a["micro_ci95"][1],
                             "classes_better": a["classes_better"], "classes_ci_above_0": a["classes_ci_above_0"],
                             "classes_ci_below_0": a["classes_ci_below_0"],
                             "numerical_failures": len(r["contrasts"][c]["numerical_failures"]), **extra(r)})
    return pd.DataFrame(rows)


def across_folds(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold mean / min / max of macro and micro, and the number of folds whose interval lies above / below 0."""
    across = folds.groupby(["space", "contrast"], sort=False)[["macro_mean", "micro_mean"]].agg(["mean", "min", "max"])
    ci = folds.assign(macro_above=folds.macro_ci_low > 0, macro_below=folds.macro_ci_high < 0,
                      micro_above=folds.micro_ci_low > 0, micro_below=folds.micro_ci_high < 0)
    ci = ci.groupby(["space", "contrast"], sort=False)[["macro_above", "macro_below", "micro_above", "micro_below"]].sum()
    return across, ci


def group_table(per: pd.DataFrame, summ: dict, spaces: list, contrasts: list, masks: dict) -> pd.DataFrame:
    """Per space, contrast and class group (``masks``: name -> row mask of ``per``; names are also the keys of the runs'
    group summaries): class-folds, mean / median class difference and parts, class-folds better, per-fold macro
    mean / min / max and folds with interval above / below 0."""
    rows = []
    for s in spaces:
        for c in contrasts:
            for g, mask in masks.items():
                t = per[(per.space == s) & mask(per)].dropna(subset=[f"{c}_mean"])
                fm = [x for x in (summ[f]["spaces"][s]["contrasts"][c][g] for f in FOLDS) if x["classes"]]
                rows.append({"space": s, "contrast": c, "group": g, "class_folds": len(t), "taxa": t.taxon_group.nunique(),
                             "class_diff_mean": t[f"{c}_mean"].mean(), "class_diff_median": t[f"{c}_mean"].median(),
                             "maha_part_mean": t[f"{c}_maha_part"].mean(), "logdet_part_mean": t[f"{c}_logdet_part"].mean(),
                             "class_folds_better": int((t[f"{c}_mean"] > 0).sum()),
                             "fold_macro_mean": pd.Series([x["macro_mean"] for x in fm], dtype=float).mean(),
                             "fold_macro_min": min((x["macro_mean"] for x in fm), default=float("nan")),
                             "fold_macro_max": max((x["macro_mean"] for x in fm), default=float("nan")),
                             "folds_ci_above_0": sum(x["macro_ci95"][0] > 0 for x in fm),
                             "folds_ci_below_0": sum(x["macro_ci95"][1] < 0 for x in fm)})
    return pd.DataFrame(rows)


STRATA_MASKS = {"15-63": lambda t: t.stratum == "15-63", "64-255": lambda t: t.stratum == "64-255",
                ">=256": lambda t: t.stratum == ">=256", "n<=p": lambda t: t.n_le_p == True,  # noqa: E712 (CSV booleans)
                "n>p": lambda t: t.n_le_p == False}  # noqa: E712
