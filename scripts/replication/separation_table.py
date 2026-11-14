"""Class-pair separation per encoder x dataset from the replication geometry runs (read-only; task.md a82e89e).

For every finished ``geometry`` run (ROOT/<encoder>/<dataset>/<partition>/geometry/class_pairs.csv): d' of the val
units along the train centroid-difference direction in the native space (established definition,
class_geometry_pilot.py), over all eligible class pairs. Per run: pairs, median d', share of pairs with d' < 1 and
< 2 (hard pairs), and the median per class of its nearest-class d' (from nearest_class.csv). Five-fold datasets: the
five partitions are averaged per encoder x dataset (``partitions`` column; overlapping val sets, not independent).
d' is scale-free but depends on the dataset's classes, so compare encoders within a dataset, not across datasets.
Separation of class means relative to spread is not classification accuracy.

Usage: separation_table.py --root RUNS --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = []
    for f in sorted(a.root.glob("*/*/*/geometry/class_pairs.csv")):
        if not (f.parent / "DONE").exists():
            continue
        enc, ds, part = f.parts[-5], f.parts[-4], f.parts[-3]
        p = pd.read_csv(f)
        near = pd.read_csv(f.parent / "nearest_class.csv")
        rows.append({"encoder": enc, "dataset": ds, "partition": part, "classes": len(near), "pairs": len(p),
                     "dprime_median": p.dprime_val.median(), "share_dprime_below_1": (p.dprime_val < 1).mean(),
                     "share_dprime_below_2": (p.dprime_val < 2).mean(),
                     "nearest_class_dprime_median": near.dprime_val.median()})
    runs = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(a.out / "separation_by_run.csv", index=False)
    agg = runs.groupby(["dataset", "encoder"]).agg(partitions=("partition", "size"), classes=("classes", "mean"),
                                                   **{c: (c, "mean") for c in ["dprime_median", "share_dprime_below_1",
                                                                              "share_dprime_below_2",
                                                                              "nearest_class_dprime_median"]}).reset_index()
    agg.to_csv(a.out / "separation_by_combination.csv", index=False)
    for c in ("dprime_median", "share_dprime_below_1", "nearest_class_dprime_median"):
        print(c)
        print(agg.pivot(index="dataset", columns="encoder", values=c).round(3).to_string())


if __name__ == "__main__":
    main()
