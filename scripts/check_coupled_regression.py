"""Regression check: the extended coupled_rscm_vs_lw.py (spaces + mechanism decomposition) must reproduce the first
run's PCA results exactly (per class: LW and coupled log-likelihood, alpha/beta, paired mean and bootstrap interval).

Usage: check_coupled_regression.py --old RUNS_V1 --new RUNS_V2 (each with <dataset>/<fold>/perclass.csv)
Prints the maximum absolute differences as JSON; exits non-zero unless all are 0 and every row matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["val_loglik_lw", "val_loglik_coupled", "lw_alpha_shrink", "coupled_alpha_keep", "coupled_beta", "coupled_vs_lw_mean"]


def interval(v: str) -> list[float]:
    return [float(x) for x in v.strip("[]").split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    a = ap.parse_args()
    out = {}
    for f in sorted(a.old.glob("*/fold*/perclass.csv")):
        rel = f.relative_to(a.old)
        old = pd.read_csv(f).rename(columns={"diff_mean": "coupled_vs_lw_mean", "diff_ci95": "coupled_vs_lw_ci95"})
        old["space"] = "pca" + old.pcs.astype(str)
        m = old.merge(pd.read_csv(a.new / rel), on=["space", "taxon_group"], suffixes=("_old", ""), how="left",
                      indicator=True)
        d = {"rows_old": len(old), "rows_matched": int((m._merge == "both").sum())}
        d |= {c: float((m[f"{c}_old"] - m[c]).abs().max()) for c in COLS}
        ci_old = np.array([interval(v) for v in m.coupled_vs_lw_ci95_old])
        ci_new = np.array([interval(v) for v in m.coupled_vs_lw_ci95])
        d["coupled_vs_lw_ci95"] = float(np.abs(ci_old - ci_new).max())
        out[str(rel.parent)] = d
    print(json.dumps(out, indent=1))
    ok = bool(out) and all(v["rows_matched"] == v["rows_old"] and all(v[c] == 0 for c in [*COLS, "coupled_vs_lw_ci95"])
                           for v in out.values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
