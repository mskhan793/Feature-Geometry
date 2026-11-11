"""Pair-level examples for the PCA/FDA-GBC comparison (read-only; per-fold tables of pca_fda_gbc.py).

For one dataset and matched dimension d:
- per class pair, the spherical train distance in pca{d} and fda{d} and d' (val, pca{d}) averaged over the folds in
  which the pair was analysed; rank of the distance within each arm (1 = hardest, smallest distance);
- the pairs with the largest rank change between pca{d} and fda{d} among the 20 hardest in either arm;
- the hardest pairs in both arms;
- covariance-dominated pairs (full covariance, pca{d}): pairs whose covariance term is more than half of the
  Bhattacharyya distance in every analysed fold (the equal-mean-like case), with their mean and covariance terms;
- the ratio of validation to training dataset-level GBC per fold for pca{d} and fda{d} (> 1 = more modelled overlap
  among held-out specimens than among the fitted ones).
Writes examples_{dataset}_d{d}.json to ``--out``.

Usage: pca_fda_examples.py --runs DIR --dataset NAME --dims 8 16 32 --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FOLDS = [f"fold{k}" for k in range(5)]
TOP = 20


def load(runs: Path, arm: str) -> pd.DataFrame:
    t = [pd.read_csv(runs / f / f"pairs_{arm}.csv").assign(fold=f) for f in FOLDS]
    return pd.concat(t).assign(pair=lambda d: d.class_a + " | " + d.class_b)


def examples(runs: Path, d: int) -> dict:
    pca, fda = load(runs, f"pca{d}"), load(runs, f"fda{d}")
    cols = ["spherical_train_distance", "spherical_train_bc"]
    m = pd.concat({"pca": pca[pca.spherical_train_valid].groupby("pair")[cols].mean(),
                   "fda": fda[fda.spherical_train_valid].groupby("pair")[cols].mean()}, axis=1).dropna()
    m.columns = [f"{a}_{b.removeprefix('spherical_train_')}" for a, b in m.columns]
    m["dprime_val_pca"] = pca.groupby("pair").dprime_val.mean()
    m["folds"] = pca.groupby("pair").fold.nunique()
    m["rank_pca"], m["rank_fda"] = m.pca_distance.rank(), m.fda_distance.rank()
    m["rank_change"] = m.rank_fda - m.rank_pca
    close = m[(m.rank_pca <= TOP) | (m.rank_fda <= TOP)]
    full = pca[pca.full_train_valid].assign(share=lambda t: t.full_train_cov_term / t.full_train_distance)
    g = full.groupby("pair")
    cov = pd.DataFrame({"min_share": g.share.min(), "mean_term": g.full_train_mean_term.mean(),
                        "cov_term": g.full_train_cov_term.mean(), "dprime_val": g.dprime_val.mean(), "folds": g.fold.nunique()})
    cov = cov[cov.min_share > 0.5].sort_values("min_share", ascending=False)
    ratio = {}
    for arm, t in (("pca", pca), ("fda", fda)):
        v = t[t.spherical_train_valid & t.spherical_val_valid]
        ratio[arm] = {f: float(v[v.fold == f].spherical_val_bc.sum() / v[v.fold == f].spherical_train_bc.sum()) for f in FOLDS}
    return {"d": d, "pairs_in_all_arms": len(m),
            "largest_rank_changes_among_hardest": json.loads(close.reindex(close.rank_change.abs().sort_values(
                ascending=False).index).head(8).round(4).reset_index().to_json(orient="records")),
            "hardest_in_both": json.loads(m[(m.rank_pca <= 10) & (m.rank_fda <= 10)].sort_values("rank_pca").round(4)
                                          .reset_index().to_json(orient="records")),
            "covariance_dominated_pairs_full_pca": {"count": len(cov),
                                                    "pairs": json.loads(cov.head(8).round(4).reset_index().to_json(orient="records"))},
            "val_over_train_gbc_ratio": ratio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for d in a.dims:
        e = examples(a.runs, d)
        (a.out / f"examples_{a.dataset}_d{d}.json").write_text(json.dumps(e, indent=1))
        print(json.dumps({k: v for k, v in e.items() if k != "largest_rank_changes_among_hardest"}, indent=0)[:3000])


if __name__ == "__main__":
    main()
