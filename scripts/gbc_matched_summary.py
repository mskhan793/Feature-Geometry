"""Summary of the matched-pair GBC comparison over 40 encoder x dataset combinations (task.md e4d291d), read-only.

Reads ROOT/<encoder>/<dataset>/<partition>/matched.json (scripts/gbc_matched_comparison.py) and writes to OUT:
- arms_long.csv: one row per run x arm x covariance form (pair counts, GBC train all / common, GBC val common, mean BC,
  ratio and its status, rank agreement, hardest-pair retention, mean / covariance term medians);
- matched_long.csv: one row per run x d x form (PCA-d vs FDA-d over matched pairs);
- strata_long.csv: one row per run x arm x form x validation-support stratum (min val units of the pair);
- per_fold.csv: spherical primary-arm ratios per partition (five-fold drilldown);
- tables.md: dataset x encoder tables (means over partitions; five-fold cells add [min, max] over folds) and counts.

Pair-set note: the earlier summaries summed train GBC over train-valid pairs and val GBC over pairs valid on both;
``previous_ratio`` = gbc_val_common / gbc_train_all reproduces that convention so changes can be traced to pair sets.

Usage: gbc_matched_summary.py --root DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENCODERS = ["dinov3_l", "dinov2_l", "bioclip2", "siglip2", "ijepa_h14"]
DATASETS = ["finbenthic2", "aquamonitor", "bioscan5m", "df24", "food101", "oxford_pets", "cifar100", "fgvc_aircraft"]
SCALARS = ["pairs_total", "pairs_train_valid", "pairs_val_valid", "pairs_common", "gbc_train_all", "gbc_train_common",
           "gbc_val_common", "mean_bc_train", "mean_bc_val", "mean_bc_val_minus_train", "ratio_val_over_train",
           "ratio_status", "share_pairs_bc_val_above_train", "spearman_distance_train_vs_val", "top20_hardest_retained"]


def load(root: Path):
    arms, matched, strata, runs = [], [], [], []
    for f in sorted(root.glob("*/*/*/matched.json")):
        r = json.loads(f.read_text())
        key = {"encoder": r["encoder"], "dataset": r["dataset"], "partition": r["partition"]}
        runs.append({**key, "classes": r["leakage_gate"]["classes"], "fda_rank": r["fda_rank"],
                     "dims_omitted": " ".join(map(str, r["dims_omitted"])), "regression_max_rel": r["regression"]["max_rel_diff"],
                     **{f"a_{k}": v for k, v in r["paper_vs_code"].items()}})
        for arm, v in r["arms"].items():
            for kind, s in v.items():
                if kind == "role":
                    continue
                row = {**key, "arm": arm, "role": v["role"], "kind": kind, **{c: s.get(c, np.nan) for c in SCALARS}}
                row["previous_ratio"] = (s["gbc_val_common"] / s["gbc_train_all"]
                                         if "gbc_val_common" in s and s["gbc_train_all"] else np.nan)
                for q, val in zip(("p10", "median", "p90"), s.get("delta_bc_p10_median_p90", [np.nan] * 3)):
                    row[f"delta_bc_{q}"] = val
                for t, val in zip(("train", "val"), s.get("mean_term_median_train_val", [np.nan] * 2)):
                    row[f"mean_term_median_{t}"] = val
                for t, val in zip(("train", "val"), s.get("cov_term_median_train_val", [np.nan] * 2)):
                    row[f"cov_term_median_{t}"] = val
                arms.append(row)
                for band, b in s.get("by_min_val_units", {}).items():
                    strata.append({**key, "arm": arm, "kind": kind, "min_val_units": band, **b})
        for comp, v in r["matched"].items():
            d = int(comp.split("_vs_")[0][3:])
            for kind, s in (v.items() if "status" not in v else [("all", v)]):
                matched.append({**key, "d": d, "kind": kind, **s})
    return pd.DataFrame(runs), pd.DataFrame(arms), pd.DataFrame(matched), pd.DataFrame(strata)


def cell(x: pd.Series, fmt: str) -> str:
    x = x.dropna()
    if x.empty:
        return "n/a"
    s = format(x.mean(), fmt)
    return s + (f" [{format(x.min(), fmt)}, {format(x.max(), fmt)}]" if len(x) > 1 else "")


def table(df: pd.DataFrame, col: str, fmt: str) -> list:
    rows = ["| dataset | " + " | ".join(ENCODERS) + " |", "|---" * (len(ENCODERS) + 1) + "|"]
    for d in DATASETS:
        rows.append(f"| {d} | " + " | ".join(cell(df[(df.dataset == d) & (df.encoder == e)][col], fmt) for e in ENCODERS) + " |")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    runs, arms, matched, strata = load(a.root)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, t in (("runs", runs), ("arms_long", arms), ("matched_long", matched), ("strata_long", strata)):
        t.to_csv(a.out / f"{name}.csv", index=False)
    sph = arms[(arms.kind == "spherical") & (arms.role == "primary")]
    sph.pivot_table(index=["dataset", "partition", "encoder"], columns="arm", values="ratio_val_over_train").to_csv(
        a.out / "per_fold.csv")
    md = ["# Matched-pair GBC comparison", "", f"Runs: {len(runs)}; combinations: {len(runs.groupby(['encoder', 'dataset']))}; "
          f"max relative difference to the saved GBC: {runs.regression_max_rel.max():.1e}.",
          "Cells: mean over partitions; five-fold datasets add [min, max] over folds.", ""]
    md += ["## A. Paper-faithful GBC (PCA 64, spherical, training only), paper form", ""] + table(runs, "a_gbc_paper_train", ".4g")
    md += ["", "## A. Authors' code-release form", ""] + table(runs, "a_gbc_code_train", ".4g")
    md += ["", "## A. Spearman(BC paper form, BC code form)", ""] + table(runs, "a_spearman_bc_paper_vs_code", ".3f")
    for arm in ("pca64", "pca16", "fda16"):
        t = sph[sph.arm == arm]
        md += ["", f"## D. Spherical {arm}: val/train ratio of mean BC over identical pairs", ""] + table(t, "ratio_val_over_train", ".3f")
    for arm in ("pca16", "fda16"):
        t = sph[sph.arm == arm]
        md += ["", f"## D. Spherical {arm}: mean BC val - train over identical pairs", ""] + table(t, "mean_bc_val_minus_train", ".2e")
    m = matched[(matched.kind == "spherical") & (matched.d == 16)]
    md += ["", "## C. Spherical, d = 16, matched pairs: FDA - PCA mean BC (train)", ""] + table(m, "fda_minus_pca_mean_bc_train", ".2e")
    md += ["", "## C. Spherical, d = 16, matched pairs: FDA - PCA mean BC (val)", ""] + table(m, "fda_minus_pca_mean_bc_val", ".2e")
    st = strata[(strata.kind == "spherical") & (strata.arm == "fda16") & (strata.pairs > 0)].assign(
        r=lambda x: x.mean_bc_val / x.mean_bc_train)
    for band in ("5-9", ">=20"):
        md += ["", f"## F. Spherical fda16, pairs with min val units {band}: val/train mean BC", ""] + table(
            st[st.min_val_units == band], "r", ".3f")

    md += ["", "## Counts", ""]
    full = arms[arms.kind == "full"]
    differ = full[full.pairs_common < full.pairs_train_valid]
    md.append(f"- Pair sets of train and val differ (train-valid > common): spherical {int((arms[arms.kind == 'spherical'].pairs_common < arms[arms.kind == 'spherical'].pairs_train_valid).sum())}, "
              f"diagonal {int((arms[arms.kind == 'diagonal'].pairs_common < arms[arms.kind == 'diagonal'].pairs_train_valid).sum())}, "
              f"full {len(differ)} of {len(full)} run x arm rows; full with no common pair: {int((full.pairs_common == 0).sum())}")
    both = differ.dropna(subset=["ratio_val_over_train", "previous_ratio"])
    md.append(f"- Full form, differing pair sets with both ratios defined: {len(both)}; previous ratio > 1 but matched ratio <= 1: "
              f"{int(((both.previous_ratio > 1) & (both.ratio_val_over_train <= 1)).sum())}; previous <= 1 but matched > 1: "
              f"{int(((both.previous_ratio <= 1) & (both.ratio_val_over_train > 1)).sum())}")
    for arm in ("pca64", "pca8", "pca16", "pca32", "fda8", "fda16", "fda32"):
        t = sph[sph.arm == arm]
        md.append(f"- Spherical {arm}: ratio > 1 in {int((t.ratio_val_over_train > 1).sum())} of {len(t)} runs; "
                  f"not reported (train mean BC < 1e-6) in {int(t.ratio_status.str.startswith('train mean BC').sum())}")
    for d in (8, 16, 32):
        t = matched[(matched.kind == "spherical") & (matched.d == d)]
        ok = t[t.status == "ok"]
        md.append(f"- Spherical d = {d}, matched: {len(ok)} of {len(t)} runs eligible; FDA mean BC train < PCA in "
                  f"{int((ok.fda_minus_pca_mean_bc_train < 0).sum())}, val < PCA in {int((ok.fda_minus_pca_mean_bc_val < 0).sum())}; "
                  f"FDA ratio > PCA ratio in {int((ok.ratio_fda > ok.ratio_pca).sum())}")
    s5, s20 = (st[st.min_val_units == b].set_index(["encoder", "dataset", "partition"]).r for b in ("5-9", ">=20"))
    j = pd.concat([s5.rename("low"), s20.rename("high")], axis=1).dropna()
    md.append(f"- Spherical fda16, runs with both strata: {len(j)}; val/train ratio higher for low-support pairs (5-9) "
              f"than for >= 20 in {int((j.low > j.high).sum())}")
    (a.out / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-14:]))


if __name__ == "__main__":
    main()
