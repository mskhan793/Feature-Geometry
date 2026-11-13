"""Actual train+val completion matrix of the 8 x 4 benchmark (task.md d57a630, C3), read-only: one row per encoder x
dataset from the C2 plan (status reuse / c1 / new, storage path) joined with its combination_record.py record (rows,
units expected / present, dimension, numerical rank, DINO LayerNorm constraint, checksum, pass) and, for caches
extracted in C1 / C2, the extraction summary (verification, CPU-vs-GPU difference, throughput). A combination is
complete only if its record exists and passes and, when extracted here, its verification passes.

Usage: completion_matrix.py --plan plan_all.csv --records DIR --extraction SUMMARY.csv [...] --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--extraction", required=True, nargs="+", type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    plan = pd.read_csv(a.plan)
    ext = pd.concat([pd.read_csv(f) for f in a.extraction])
    ext = ext[~ext.run.str.startswith("bioscan512")]  # C1 repeat samples, not train+val caches
    # run names are <dataset>_<encoder>; the C1 pilot runs are aircraft_<encoder>
    ext["dataset"] = [r[: -len(e) - 1].replace("aircraft", "fgvc_aircraft") if r.startswith("aircraft_") else r[: -len(e) - 1]
                      for r, e in zip(ext.run, ext.encoder)]
    ext = ext.set_index(["encoder", "dataset"])
    rows = []
    for _, p in plan.iterrows():
        f = a.records / f"{p.encoder}__{p.dataset}.json"
        r = json.loads(f.read_text()) if f.exists() else None
        e = ext.loc[(p.encoder, p.dataset)] if (p.encoder, p.dataset) in ext.index else None
        row = {"encoder": p.encoder, "dataset": p.dataset, "status": p.status, "record": r is not None,
               "record_pass": bool(r and r["pass"]), "rows": r and r["rows"], "units": r and r["units"],
               "units_expected": r and r["units_expected"], "dim": r and r["dim"],
               "rank": r and r["rank"]["numerical_rank_1e-12"],
               "layernorm_abs_cos": r and r["rank"].get("abs_cos_smallest_eigvec_vs_inv_w"),
               "layernorm_rel_max": r and r["rank"].get("constraint_rel_abs_max"),
               "failed_checks": ",".join(k for k, v in r["checks"].items() if not v) if r else "no record",
               "features_sha256": r and r["features_sha256"], "float32_gb": round(r["rows"] * r["dim"] * 4 / 1e9, 3) if r else None,
               "verify_pass": None if e is None else bool(e.verify_pass),
               "cpu_vs_gpu256_max_rel_norm": None if e is None else e.cpu_vs_gpu256_max_rel_norm,
               "images_per_s_per_gcd": None if e is None else e.images_per_s_per_gcd,
               "storage": p.out if isinstance(p.out, str) and p.out else p.record_sources}
        row["complete"] = row["record_pass"] and (row["verify_pass"] is not False) and (
            p.status == "reuse" or row["verify_pass"] is True)
        rows.append(row)
    t = pd.DataFrame(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out_dir / "completion_matrix.csv", index=False)
    grid = t.pivot(index="dataset", columns="encoder", values="complete").loc[plan.dataset.unique(), plan.encoder.unique()]
    status = t.pivot(index="dataset", columns="encoder", values="status").loc[grid.index, grid.columns]
    cell = grid.map(lambda c: "complete" if c else "PENDING") + " (" + status + ")"
    md = ["| dataset | " + " | ".join(cell.columns) + " |", "|---" * (len(cell.columns) + 1) + "|"]
    md += [f"| {d} | " + " | ".join(r) + " |" for d, r in cell.iterrows()]  # no tabulate dependency
    (a.out_dir / "completion_matrix.md").write_text("\n".join(md) + "\n")
    summary = {"combinations": len(t), "complete": int(t.complete.sum()),
               "complete_by_status": t[t.complete].status.value_counts().to_dict(),
               "pending": t[~t.complete][["encoder", "dataset", "failed_checks"]].to_dict(orient="records"),
               "float32_gb_complete": round(t.float32_gb[t.complete].sum(), 2),
               "float32_gb_new": round(t.float32_gb[t.complete & (t.status == "new")].sum(), 2)}
    (a.out_dir / "completion_summary.json").write_text(json.dumps(summary, indent=1))
    print(cell.to_string())
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
