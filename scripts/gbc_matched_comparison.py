"""Paper-faithful GBC and matched PCA-vs-FDA train/val comparison for one run (task.md e4d291d).

Reads the saved output of scripts/pca_fda_gbc.py for one encoder x dataset x partition (``--run``: pairs_{arm}.csv,
classes.csv, summary.json; projections and class Gaussians were fitted there, train-fitted projection, train and val
Gaussians fitted separately). Nothing is refitted: every number here is a sum or summary of the saved per-pair BC
values (src.statistics.gbc_matched).

Arms: A = pca64 (Pándy et al.'s image-classification setup: unwhitened PCA 64, spherical Gaussians; the paper's
training-only GBC in paper form and in the authors' code-release form); B = fda{d} (our regularised FDA, orthonormal
basis, d = 8 16 32 where the FDA rank permits) with raw / shrinkage variants as labelled sensitivity arms; C = pca{d}
vs fda{d} over matched pairs; D = train vs val GBC over the same pairs P_common per arm and form; spherical is primary,
diagonal and full are supplementary.

Checks (the run fails if one does not hold):
- leakage gate: per GBC class, n_train / n_val of the run equal the train / val unit counts of the analysis cohort
  (``--support``, built from the cohort after dropping the partition's test rows) and the GBC classes are exactly the
  eligible classes; the run's summary records the dropped test rows;
- regression: gbc_train_all and the old gbc_val equal the saved summary's gbc_train / gbc_val (relative 1e-9); the old
  gbc_val was summed over pairs valid on train and val, i.e. P_common.

Writes to ``--out``: matched.json, per_class.csv (primary arms, spherical), hardest_pairs.csv (20 largest train BC per
primary arm, spherical, with val BC).

Usage: gbc_matched_comparison.py --run FDA_DIR --support SUPPORT.csv --out DIR --encoder E --dataset D --partition P
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.statistics.gbc_matched import KINDS, PAIR, TOP, matched, paper_vs_code, per_class, train_val, valid

TOLERANCE = 1e-9
FIELDS = [f"{s}_{q}" for s in ("train", "val") for q in ("bc", "distance", "mean_term", "cov_term")]


def read_pairs(path: Path) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0).columns
    want = PAIR + ["n_train_a", "n_train_b", "n_val_a", "n_val_b", "code_spherical_bc"] + \
        [f"{k}_{f}" for k in KINDS for f in ("train_valid", "val_valid", *FIELDS)]
    return pd.read_csv(path, usecols=[c for c in want if c in head])


def leakage_gate(classes: pd.DataFrame, support: pd.DataFrame, summary: dict) -> dict:
    g = classes[classes.in_gbc_pairs].set_index("taxon_group")
    s = support.set_index("taxon_group")
    eligible = set(s.index[s.eligible])
    assert set(g.index) == eligible, "GBC classes differ from the cohort's eligible classes"
    bad = [c for c in g.index if (g.n_train[c], g.n_val[c]) != (s.train[c], s.val[c])]
    assert not bad, f"unit counts differ from the train/val cohort for {bad[:5]}"
    return {"classes": len(g), "unit_counts_equal_cohort": True,
            "test_rows_dropped_before_lookup": summary["cohort_rows"]["test_rows_dropped"]}


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--support", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--partition", required=True)
    a = ap.parse_args()
    summary = json.loads((a.run / "summary.json").read_text())
    out = {"encoder": a.encoder, "dataset": a.dataset, "partition": a.partition, "source": str(a.run),
           "leakage_gate": leakage_gate(pd.read_csv(a.run / "classes.csv"), pd.read_csv(a.support), summary),
           "dims_used": summary["dims_used"], "dims_omitted": summary["dims_omitted"],
           "fda_rank": summary["fda"]["main"]["rank"], "arms": {}, "matched": {}, "regression": {}}
    base = next(k for k in summary["arms"] if k.startswith("pca") and int(k[3:]) not in summary["dims_used"])
    primary = [base] + [f"{t}{d}" for d in summary["dims_used"] for t in ("pca", "fda")]
    sensitivity = [f"fda{d}_{s}" for d in summary["dims_used"] for s in ("raw", "shrink0.5", "shrink2")]
    tables, worst = {}, 0.0
    for arm in primary + sensitivity:
        p = read_pairs(a.run / f"pairs_{arm}.csv")
        tables[arm] = p
        out["arms"][arm] = {"role": "primary" if arm in primary else "sensitivity",
                            **{k: train_val(p, k) for k in KINDS}}
        for k in KINDS:
            old, new = summary["arms"][arm][k], out["arms"][arm][k]
            for o, n in (("gbc_train", "gbc_train_all"), ("gbc_val", "gbc_val_common")):
                if o in old:
                    worst = max(worst, rel(new[n], old[o]))
    assert worst <= TOLERANCE, f"saved GBC not reproduced (max relative difference {worst})"
    out["regression"] = {"compared": "gbc_train_all, gbc_val_common vs saved summary", "max_rel_diff": worst}
    out["paper_vs_code"] = paper_vs_code(tables[base])
    for d in summary["dims_used"]:
        out["matched"][f"pca{d}_vs_fda{d}"] = {k: matched(tables[f"pca{d}"], tables[f"fda{d}"], k) for k in KINDS}
    for d in summary["dims_omitted"]:
        out["matched"][f"pca{d}_vs_fda{d}"] = {"status": f"not eligible: FDA rank {out['fda_rank']} < {d}"}
    a.out.mkdir(parents=True, exist_ok=True)
    pd.concat([per_class(tables[arm]).assign(arm=arm) for arm in primary]).to_csv(a.out / "per_class.csv", index=False)
    hard = []
    for arm in primary:
        p = tables[arm]
        c = p[valid(p, "spherical", "train") & valid(p, "spherical", "val")]
        hard.append(c.nlargest(TOP, "spherical_train_bc")[PAIR + ["spherical_train_bc", "spherical_val_bc"]].assign(arm=arm))
    pd.concat(hard).to_csv(a.out / "hardest_pairs.csv", index=False)
    (a.out / "matched.json").write_text(json.dumps(out, indent=1, default=float))
    s = out["arms"]
    print(json.dumps({"run": f"{a.encoder} {a.dataset} {a.partition}", "regression_max_rel": worst,
                      "ratio_spherical": {arm: s[arm]["spherical"].get("ratio_val_over_train") for arm in primary}},
                     default=float))


if __name__ == "__main__":
    main()
