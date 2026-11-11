"""Orientation summary of bhattacharyya_overlap.py outputs (read-only; full-covariance pairs of one PCA dimension).

Per dataset run directory and ``--pcs``: for pairs whose two classes both have a full covariance,
- the pair's along-ratio = geometric mean over the two classes of (variance along the train centroid-difference
  direction u) / (mean variance per dimension): share of pairs with along-ratio < 1 (spread mostly across u) and > 2;
- Spearman of log along-ratio with d' (val) and with the rank gap (rank of the full Bhattacharyya distance minus rank of
  d'; positive = Bhattacharyya places the pair farther apart than d');
- pairs with high total variance in both classes (above the median class total variance) that are still well separated
  (d' above the median) with along-ratio < 0.5: count and the first three;
- share of the 20 pairs with the largest absolute rank gap that involve a class labelled "mixture-supported split
  (exploratory)" at 8 PCs, against that share over all full-covariance pairs.
Writes orientation_pcs{d}.json in each run directory.

Usage: bhattacharyya_orientation_summary.py --runs DIR ... --pcs 64 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.statistics.class_modality import SUPPORTED


def summary(run: Path, d: int) -> dict:
    p = pd.read_csv(run / f"pairs_pcs{d}.csv")
    cls = pd.read_csv(run / f"classes_pcs{d}.csv").set_index("taxon_group")
    f = p[p.full_train_valid].copy()
    f["along"] = np.sqrt(f.along_ratio_a * f.along_ratio_b)
    f["rank_gap"] = f.full_train_distance.rank() - f.dprime_val.rank()
    total = cls.total_variance_pcs
    big = total > total.median()
    sep = f[f.class_a.map(big) & f.class_b.map(big) & (f.dprime_val > f.dprime_val.median()) & (f.along < 0.5)]
    mixed = (f.label8_a == SUPPORTED) | (f.label8_b == SUPPORTED)
    top = f.reindex(f.rank_gap.abs().sort_values(ascending=False).index).head(20)
    return {"pcs": d, "full_pairs": len(f), "classes_with_full_covariance": int(cls.full_covariance_eligible.sum()),
            "along_ratio_median": float(f.along.median()), "share_along_below_1": float((f.along < 1).mean()),
            "share_along_above_2": float((f.along > 2).mean()),
            "spearman_log_along_vs_dprime_val": float(spearmanr(np.log(f.along), f.dprime_val)[0]),
            "spearman_log_along_vs_rank_gap": float(spearmanr(np.log(f.along), f.rank_gap)[0]),
            "high_variance_yet_separated_pairs": len(sep),
            "high_variance_yet_separated_examples": json.loads(sep.nlargest(3, "dprime_val")[
                ["class_a", "class_b", "dprime_val", "along", "full_train_bc"]].to_json(orient="records")),
            "top20_rank_gap_share_with_mixture_supported_class": float(mixed.reindex(top.index).mean()),
            "all_pairs_share_with_mixture_supported_class": float(mixed.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+", type=Path)
    ap.add_argument("--pcs", type=int, nargs="+", default=[64, 16])
    a = ap.parse_args()
    for run in a.runs:
        for d in a.pcs:
            s = summary(run, d)
            (run / f"orientation_pcs{d}.json").write_text(json.dumps(s, indent=1))
            print(run.name, json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
