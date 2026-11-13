"""Reproduction check of the 8 x 4 geometry replication (task.md a82e89e): the new DINOv3-L FIN-Benthic2 /
AquaMonitor outputs against the saved reference evidence of the earlier tasks, file by file.

Reference folders (per dataset and fold; ``overlap`` exists for fold0 only):
geometry ``<dataset>_full_geometry/folds/foldK``, overlap ``bhattacharyya_overlap/<dataset>``, fda
``pca_fda_gbc/folds/<dataset>/foldK``, lw ``lw_fit/<dataset>/foldK``, coupled
``coupled_rscm_vs_lw/extension/runs/<dataset>/foldK``, pooling ``balanced_taxonomic_pooling/runs/<dataset>/foldK``
(earlier run with taxonomy: only the columns / keys the no-taxonomy run also has are compared).

Every .csv and .json present in both folders is compared. CSV: same rows in the same order (row count must match),
common columns; JSON: common leaves by path. Numbers match if |new - ref| <= TOL * max(1, |ref|) (predeclared TOL
1e-9; NaN == NaN), strings and booleans must be equal. Timing, memory and path entries are ignored (IGNORE).
Writes OUT/reference_check.json (per file: compared cells, mismatches, max abs and relative difference, first
mismatches) and prints a summary. Exit 1 if any compared cell mismatches.

Usage: compare_reference.py --evidence docs/evidence --new ROOT/dinov3_l --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-9
IGNORE = ("seconds", "wall", "rss", "cohort", "out", "time", "path", "log", "cache", "manifest", "code")
REF = {"geometry": "{d}_full_geometry/folds/{f}", "overlap": "bhattacharyya_overlap/{d}",
       "fda": "pca_fda_gbc/folds/{d}/{f}", "lw": "lw_fit/{d}/{f}", "coupled": "coupled_rscm_vs_lw/extension/runs/{d}/{f}",
       "pooling": "balanced_taxonomic_pooling/runs/{d}/{f}"}


def ignored(key: str) -> bool:
    k = key.lower()
    return any(w in k for w in IGNORE)


def flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def same(a, b) -> tuple[bool, float, float]:
    """(equal within TOL, abs diff, rel diff) for two leaves."""
    if isinstance(a, bool) or isinstance(b, bool) or not all(isinstance(v, (int, float)) for v in (a, b)):
        return (a == b) or (str(a) == str(b)), 0.0, 0.0
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b), 0.0, 0.0
    if math.isinf(a) or math.isinf(b):
        return a == b, 0.0, 0.0
    d = abs(a - b)
    return d <= TOL * max(1.0, abs(b)), d, d / max(abs(b), 1e-300)


def parse(v):
    """CSV cells: numbers as float, list-like strings ('[a, b]') as lists, rest as text."""
    if isinstance(v, str) and v.startswith("["):
        try:
            return json.loads(v.replace("nan", "NaN"))
        except json.JSONDecodeError:
            return v
    return v


def compare_leaves(pairs) -> dict:
    n, bad, amax, rmax, first = 0, 0, 0.0, 0.0, []
    for key, a, b in pairs:
        if ignored(key):
            continue
        for (ka, va), (_, vb) in zip(flatten(a, key), flatten(b, key)):
            n += 1
            ok, d, r = same(va, vb)
            amax, rmax = max(amax, d), max(rmax, r)
            if not ok:
                bad += 1
                if len(first) < 5:
                    first.append({"key": ka, "new": va, "ref": vb})
    return {"compared": n, "mismatches": bad, "max_abs_diff": amax, "max_rel_diff": rmax, "first_mismatches": first}


def compare_csv(new: Path, ref: Path) -> dict:
    a, b = pd.read_csv(new), pd.read_csv(ref)
    if len(a) != len(b):
        return {"error": f"rows {len(a)} vs {len(b)}"}
    cols = [c for c in b.columns if c in a.columns]
    pairs = ((f"r{i}/{c}", parse(x), parse(y)) for c in cols for i, (x, y) in
             enumerate(zip(a[c].to_numpy(object), b[c].to_numpy(object))))
    out = compare_leaves((k, float(x) if isinstance(x, (np.floating, np.integer)) else x,
                          float(y) if isinstance(y, (np.floating, np.integer)) else y) for k, x, y in pairs)
    return {**out, "columns_compared": len(cols), "columns_only_in_ref": sorted(set(b.columns) - set(a.columns))}


def compare_json(new: Path, ref: Path) -> dict:
    a, b = dict(flatten(json.loads(new.read_text()))), dict(flatten(json.loads(ref.read_text())))
    keys = [k for k in b if k in a]
    out = compare_leaves((k, a[k], b[k]) for k in keys)
    return {**out, "leaves_only_in_ref": len(set(b) - set(a)), "leaves_only_in_new": len(set(a) - set(b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path, help="replication output root of the DINOv3-L encoder")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    result, total_bad = {}, 0
    for d in ("finbenthic2", "aquamonitor"):
        for k in range(5):
            f = f"fold{k}"
            for an, pattern in REF.items():
                ref = a.evidence / pattern.format(d=d, f=f)
                new = a.new / d / f / an
                if an == "overlap" and k > 0 or not ref.is_dir():
                    continue
                files = sorted(p.name for p in ref.iterdir() if p.suffix in (".csv", ".json") and (new / p.name).exists())
                res = {p: (compare_csv if p.endswith(".csv") else compare_json)(new / p, ref / p) for p in files}
                missing = sorted(p.name for p in ref.iterdir() if p.suffix in (".csv", ".json") and not (new / p.name).exists())
                bad = sum(r.get("mismatches", 1) for r in res.values())
                total_bad += bad
                result[f"{d}/{f}/{an}"] = {"reference": str(ref), "files": res, "files_missing_in_new": missing,
                                           "mismatches": bad,
                                           "max_rel_diff": max([r.get("max_rel_diff", 0) for r in res.values()], default=0)}
                print(f"{d} {f} {an}: {len(files)} files, {bad} mismatching cells, max rel diff "
                      f"{result[f'{d}/{f}/{an}']['max_rel_diff']:.2e}, missing {missing}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "reference_check.json").write_text(json.dumps({"tolerance_rel": TOL, "ignored_key_words": IGNORE,
                                                            "total_mismatches": total_bad, "runs": result}, indent=1, default=str))
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
