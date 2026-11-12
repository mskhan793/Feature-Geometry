"""Five-fold summary of balanced_taxonomic_pooling.py runs per dataset (read-only; task.md f54f534).

- folds.csv / groups.csv: per fold, space and contrast (macro primary, micro secondary, intervals, parts, wins,
  failures) and per class group (train-size strata, n_c <= d, taxonomic level used / genus / family / fallback);
- levels.csv: per fold and space, targets per donor level and the mean overlap (Jaccard) of nearest-mean and
  taxonomic donors;
- paths.csv: per class and space, the levels used over the folds and the taxonomic donor sets;
- reproduction.json: lw, coupled and pool_spec against the earlier coupled_rscm_vs_lw.py runs (lw, coupled,
  pooling_lw_shrink; same folds and spaces) — must be exactly equal, incl. the coupled-vs-LW bootstrap intervals.
Writes these per dataset plus summary.md to --out.

Usage: summarise_balanced_taxonomic.py --dataset NAME RUNS COUPLED_RUNS ... --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.statistics.fold_summary import FOLDS, STRATA_MASKS, across_folds, fold_table, group_table, read_runs

MASKS = {"all": lambda t: t.space == t.space, **STRATA_MASKS, "taxonomy_used": lambda t: t.level != "global",
         "fallback": lambda t: t.level == "global", "genus": lambda t: t.level == "genus", "family": lambda t: t.level == "family"}
REPRO = {"val_loglik_lw": "val_loglik_lw", "val_loglik_coupled": "val_loglik_coupled",
         "val_loglik_pool_spec": "val_loglik_pooling_lw_shrink", "coupled_vs_lw_mean": "coupled_vs_lw_mean"}


def reproduction(per: pd.DataFrame, coupled_runs: Path) -> dict:
    ref = pd.concat([pd.read_csv(coupled_runs / f / "perclass.csv").assign(fold=f) for f in FOLDS], ignore_index=True)
    m = per.merge(ref, on=["fold", "space", "taxon_group"], suffixes=("", "_ref"), how="outer", indicator=True)
    both = m[m._merge == "both"]
    out = {"rows_here": len(per), "rows_reference": len(ref), "matched": len(both)}
    for here, there in REPRO.items():
        there = there if there != here else f"{here}_ref"
        out[f"max_abs_diff_{here}"] = float((both[here] - both[there]).abs().max())
    out["coupled_vs_lw_ci95_equal"] = bool((both.coupled_vs_lw_ci95 == both.coupled_vs_lw_ci95_ref).all())
    return out


def summarise(runs: Path, coupled_runs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    per, summ = read_runs(runs)
    spaces = list(summ[FOLDS[0]]["spaces"])
    contrasts = list(summ[FOLDS[0]]["contrasts"])
    folds = fold_table(summ, spaces, contrasts)
    folds.to_csv(out / "folds.csv", index=False)
    across, ci = across_folds(folds)
    groups = group_table(per, summ, spaces, contrasts, MASKS)
    groups.to_csv(out / "groups.csv", index=False)
    levels = per.groupby(["fold", "space"], sort=False).agg(
        targets=("taxon_group", "size"), genus=("level", lambda v: int((v == "genus").sum())),
        family=("level", lambda v: int((v == "family").sum())), fallback=("level", lambda v: int((v == "global").sum())),
        near_tax_jaccard_mean=("near_tax_jaccard", "mean")).reset_index()
    levels.to_csv(out / "levels.csv", index=False)
    paths = per.groupby(["space", "taxon_group"], sort=False).agg(
        rank=("rank", "first"), folds=("fold", "size"), levels=("level", lambda v: ";".join(sorted(set(v)))),
        donor_sets=("donors", lambda v: " | ".join(sorted(set(v.fillna(""))))), n_train_median=("n_train", "median")).reset_index()
    paths.to_csv(out / "paths.csv", index=False)
    repro = reproduction(per, coupled_runs)
    (out / "reproduction.json").write_text(json.dumps(repro, indent=1))
    resources = {f: {"total_seconds": summ[f]["total_seconds"], "max_rss_gb": summ[f]["max_rss_gb"],
                     **{f"{s}_seconds": summ[f]["spaces"][s]["seconds"] for s in spaces}} for f in FOLDS}
    (out / "resources.json").write_text(json.dumps(resources, indent=1))
    return {"across": across, "ci": ci, "groups": groups, "levels": levels, "repro": repro, "resources": resources}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=3, action="append", required=True, metavar=("NAME", "RUNS", "COUPLED_RUNS"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    lines = []
    for name, runs, coupled in a.dataset:
        s = summarise(Path(runs), Path(coupled), a.out / name)
        lines += [f"## {name}", "", "Across folds (fold mean, min, max; within-space contrasts):", "", s["across"].round(4).to_string(), "",
                  "Folds with 95 % interval above / below 0:", "", s["ci"].to_string(), "",
                  "By class group:", "", s["groups"].round(4).to_string(index=False), "",
                  "Donor levels:", "", s["levels"].round(3).to_string(index=False), "",
                  "Reproduction of the earlier runs:", "", json.dumps(s["repro"], indent=1), "",
                  "Resources per fold:", "", json.dumps(s["resources"], indent=1), ""]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
