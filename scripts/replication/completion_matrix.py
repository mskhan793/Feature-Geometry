"""40-row completion matrix of the 8 x 5 geometry replication (task.md a82e89e + 3b897b0), read-only.

Sources (nothing recomputed from features):
- the per-run summaries under ROOT/<encoder>/<dataset>/<partition>/<analysis>/ (status counts every analysis script
  already records: geometry mixture arms per PCA dimension, overlap / FDA valid pairs per covariance form, LW sample
  covariance availability, coupled / pooling fit status per space);
- the eligibility summary of build_cohorts.py (support rule, classes, eligible pairs, units);
- the Slurm .out files of replicate_geometry.sh ("start ... job J code C", "task ...", "end" or "already done"): the
  run that wrote an output is the one whose log ends with "end";
- sacct listings with the Account field (billed project per job).

Arm status: ``ok`` (fitted and valid), ``unreliable`` (fitted; established reliability rule failed, excluded from the
primary averages), ``failed: ...`` (numerical failure recorded by the script), ``not eligible`` (model-specific support
rule, e.g. n <= d for sample covariances, too few units for full covariances), ``not applicable`` (the dataset has
no field the arm needs), ``not run`` (author instruction). Counts are classes (or class pairs for overlap / FDA),
summed over partitions in the 40-row matrix and per partition in arm_status.csv.

Writes OUT/completion_matrix.csv (40 rows), OUT/arm_status.csv, OUT/run_provenance.csv, OUT/status_by_arm.csv
(every status other than ok, summed per analysis x arm) and OUT/totals.json.

Usage: completion_matrix.py --root RUNS --status task_status.csv --support support_summary.json --logs DIR
       --sacct FILE [FILE ...] --out DIR
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from scripts.benchmark.c2_plan import DIMS

FIVE_FOLD = ("finbenthic2", "aquamonitor")
SOURCE_FIELD = ("aquamonitor",)  # source-group model GD_tied needs the acquisition dataset field
TAXONOMY_ARMS = ("pool_tax", "pool_near")
# independent unit per dataset (src/data/benchmark_cohort.py; full-specimen selections for the five-fold datasets);
# unit embedding = mean of camera means (five-fold) or of the unit's images (one constant camera elsewhere)
UNIT = {"finbenthic2": "specimen", "aquamonitor": "specimen", "bioscan5m": "BIOSCAN record", "df24": "observation"}
UNIT_DEFAULT = "exact-content image hash"
# script-specific labels of the same thing (model support rule not met): geometry "ineligible", coupled
# "not defined (n <= d)" (sample covariance)
SAME_AS_NOT_ELIGIBLE = ["ineligible", "not defined"]


def arms_geometry(r):
    for d, m in r["modality"]["by_pcs"].items():
        for arm, counts in m["status_counts"].items():
            for s, n in counts.items():
                yield f"pcs{d}", arm, s, n


def arms_pairs(r, forms, pairs):
    for setting, v in forms:
        for k in ("full", "diagonal", "spherical"):
            valid = v[k].get("pairs_valid", v[k].get("pairs_valid_train"))
            yield setting, k, "ok", valid
            if pairs - valid:
                yield setting, k, "not eligible", pairs - valid


def arms_lw(r):
    for d, v in r["per_class"].items():
        n, avail = v["classes"], v["sample_covariance_available"]
        for est in ("lw", "diagonal", "spherical"):
            yield f"pcs{d}", est, "ok", n
        yield f"pcs{d}", "sample", "ok", avail
        if n - avail:
            yield f"pcs{d}", "sample", "not eligible", n - avail


def arms_spaces(r):
    for space, v in r["spaces"].items():
        for arm, counts in v["status_counts"].items():
            for s, n in counts.items():
                yield space, arm, s, n


def run_arms(d: Path, analysis: str):
    if analysis == "geometry":
        return arms_geometry(json.loads((d / "results.json").read_text()))
    r = json.loads((d / "summary.json").read_text())
    if analysis == "overlap":
        return arms_pairs(r, [(f"pcs{k}", v) for k, v in r["by_pcs"].items()], r["pairs"])
    if analysis == "fda":
        return arms_pairs(r, [(k, v) for k, v in r["arms"].items() if "_" not in k], r["pairs"])
    if analysis == "lw":
        return arms_lw(r)
    return arms_spaces(r)


def contrast_failures(d: Path) -> int:
    r = json.loads((d / "summary.json").read_text())
    return sum(len(groups["numerical_failures"]) for v in r["spaces"].values() for groups in v["contrasts"].values())


def provenance(logs: Path, sacct: list) -> pd.DataFrame:
    acc = pd.concat([pd.read_csv(f, sep="|", dtype=str) for f in sacct])
    acc = dict(zip(acc.JobID.str.split("_").str[0], acc.Account))
    rows = []
    for f in sorted(logs.glob("fg_repl_*.out")):
        text = f.read_text()
        start = re.search(r"^start (\S+) job (\S+) node \S+ code (\S+)", text, re.M)
        task = re.search(r"^task (\S+) (\S+) (\S+) (\S+) ->", text, re.M)
        if not (start and task):
            continue
        outcome = "wrote output" if re.search(r"^end ", text, re.M) else (
            "skipped (already done)" if "already done" in text else "did not finish")
        job = start.group(2)
        rows.append({"encoder": task.group(1), "dataset": task.group(2), "partition": task.group(3),
                     "analysis": task.group(4), "job": job, "batch": f.name.rsplit("_", 2)[0], "code": start.group(3)[:7],
                     "account": acc.get(job.split("_")[0], ""), "start": start.group(1), "outcome": outcome})
    return pd.DataFrame(rows)


def span(values) -> str:
    v = sorted(set(values))
    return str(v[0]) if len(v) == 1 else f"{v[0]}-{v[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--status", required=True, type=Path)
    ap.add_argument("--support", required=True, type=Path)
    ap.add_argument("--logs", required=True, type=Path)
    ap.add_argument("--sacct", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    st = pd.read_csv(a.status)
    sup = json.loads(a.support.read_text())
    rule = sup["rule"]
    prov = provenance(a.logs, a.sacct)
    wrote = prov[prov.outcome == "wrote output"]
    key = ["encoder", "dataset", "partition", "analysis"]
    assert not wrote.duplicated(key).any(), "two runs wrote the same output"
    arms, rows = [], []
    for t in st.itertuples():
        d = a.root / t.encoder / t.dataset / t.partition / t.analysis
        if t.status != "done":
            continue
        for setting, arm, s, n in run_arms(d, t.analysis):
            arms.append({"encoder": t.encoder, "dataset": t.dataset, "partition": t.partition, "analysis": t.analysis,
                         "setting": setting, "arm": arm, "status": s, "count": n})
        if t.analysis == "pooling":
            r = json.loads((d / "summary.json").read_text())
            for space in r["spaces"]:
                for arm in TAXONOMY_ARMS:
                    arms.append({"encoder": t.encoder, "dataset": t.dataset, "partition": t.partition,
                                 "analysis": t.analysis, "setting": space, "arm": arm,
                                 "status": "not run (author instruction)", "count": r["eligible_classes"]})
        if t.analysis == "geometry" and t.dataset not in SOURCE_FIELD:
            r = json.loads((d / "results.json").read_text())
            for pcs in r["modality"]["by_pcs"]:
                arms.append({"encoder": t.encoder, "dataset": t.dataset, "partition": t.partition, "analysis": t.analysis,
                             "setting": f"pcs{pcs}", "arm": "GD_tied", "status": "not applicable (no source field)",
                             "count": r["classes"]})
    arms = pd.DataFrame(arms)
    for (enc, ds), s in st.groupby(["encoder", "dataset"], sort=False):
        parts = sup["datasets"][ds]["partitions"]
        part_names = sorted(s.partition.unique())
        g = arms[(arms.encoder == enc) & (arms.dataset == ds)]
        status = g.status.str.split(":").str[0].str.split(" \\(").str[0].replace(SAME_AS_NOT_ELIGIBLE, "not eligible")
        pair_level = g.analysis.isin(["overlap", "fda"])
        by = g[~pair_level].groupby(status[~pair_level])["count"].sum()  # class fits
        pairs = g[pair_level].groupby(status[pair_level])["count"].sum()  # class-pair scores
        fails = sum(contrast_failures(a.root / enc / ds / p / an) for p in part_names for an in ("coupled", "pooling")
                    if (s[(s.partition == p) & (s.analysis == an)].status == "done").all())
        w = wrote[(wrote.encoder == enc) & (wrote.dataset == ds)]
        spaces = json.loads((a.root / enc / ds / part_names[0] / "coupled" / "summary.json").read_text())["spaces"]
        native = [k for k in spaces if k.startswith("native")]
        rows.append({
            "encoder": enc, "dataset": ds, "partitions": len(part_names),
            "tasks_done": f"{(s.status == 'done').sum()}/{len(s)}",
            "unit": UNIT.get(ds, UNIT_DEFAULT), "feature_dim": DIMS[enc], "native_spaces": " ".join(native),
            "native_dim_used": span(int(spaces[k]["dim"]) for k in native),
            "support_rule": f">={rule['min_train']} train, >={rule['min_val']} val units",
            "classes_train_or_val": span(parts[p]["classes_train_or_val"] for p in part_names),
            "classes_eligible": span(parts[p]["classes_eligible"] for p in part_names),
            "eligible_pairs": span(parts[p]["eligible_pairs"] for p in part_names),
            "train_units": span(parts[p]["train_units"] for p in part_names),
            "val_units": span(parts[p]["val_units"] for p in part_names),
            "eligible_train_units_per_class_median": span(parts[p]["eligible_train_units_min_median_max"][1] for p in part_names),
            "class_fits_ok": int(by.get("ok", 0)), "class_fits_unreliable": int(by.get("unreliable", 0)),
            "class_fits_failed": int(by.get("failed", 0)), "class_fits_not_eligible": int(by.get("not eligible", 0)),
            "class_fits_not_applicable": int(by.get("not applicable", 0)), "class_fits_not_run": int(by.get("not run", 0)),
            "pair_scores_ok": int(pairs.get("ok", 0)), "pair_scores_not_eligible": int(pairs.get("not eligible", 0)),
            "contrast_numerical_failures": fails,
            "failed_arms": "; ".join(sorted({f"{r.analysis}:{r.setting}:{r.arm}" for r in g.itertuples()
                                             if r.status.startswith("failed")})),
            "jobs": " ".join(sorted(w.job.str.split("_").str[0].unique())),
            "code": " ".join(sorted(w.code.unique())), "accounts": " ".join(sorted(w.account.unique())),
        })
    a.out.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows)
    assert len(m) == 40, len(m)
    m.to_csv(a.out / "completion_matrix.csv", index=False)
    arms.to_csv(a.out / "arm_status.csv", index=False)
    prov.to_csv(a.out / "run_provenance.csv", index=False)
    nonok = arms[arms.status != "ok"].groupby(["analysis", "arm", "status"])["count"].sum().reset_index()
    nonok.to_csv(a.out / "status_by_arm.csv", index=False)
    totals = {"combinations": len(m), "tasks": int(len(st)), "tasks_done": int((st.status == "done").sum()),
              "runs": {k: int(v) for k, v in prov.outcome.value_counts().items()},
              **{c: int(m[c].sum()) for c in m.columns if c.startswith(("class_fits_", "pair_scores_", "contrast_"))}}
    (a.out / "totals.json").write_text(json.dumps(totals, indent=1))
    print(json.dumps(totals))
    print(nonok.to_string())
    print(m[["encoder", "dataset", "tasks_done", "classes_eligible", "class_fits_ok", "class_fits_unreliable",
             "class_fits_failed", "class_fits_not_eligible", "pair_scores_not_eligible", "contrast_numerical_failures",
             "failed_arms"]].to_string())


if __name__ == "__main__":
    main()
