"""Read-only camera inventory of the AquaMonitor Classification benchmark (camera Gaussian pilot, gate 1).

From the audited manifest, reports for the official ``fold0`` split:
- camera identities with image and specimen counts;
- split x camera image and specimen counts;
- specimens imaged by more than one camera;
- whether an ``imaging_run`` mixes cameras;
- camera against year, dataset, site and lake (possible confounding);
- class x camera x split specimen and image counts, and which class-camera pairs reach the support minimum
  (``--min-train`` specimens in train, ``--min-val`` in val).

Writes ``inventory.json`` and ``class_camera_counts.csv`` to ``--out``.

Usage: camera_inventory.py --manifest MANIFEST --out DIR [--min-train 15 --min-val 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SPLIT = "fold0"


def counts(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return df.groupby(by).agg(images=("img", "size"), specimens=("individual", "nunique")).reset_index()


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-train", type=int, default=15)
    ap.add_argument("--min-val", type=int, default=5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.manifest)
    df["camera"] = df["camera"].astype(str)

    cams_per_specimen = df.groupby("individual")["camera"].nunique()
    cams_per_run = df.groupby("imaging_run")["camera"].nunique()
    classes_per_specimen = df.groupby("individual")["taxon_group"].nunique()
    splits_per_specimen = df.groupby("individual")[SPLIT].nunique()

    def by_camera(col: str) -> list[dict]:
        t = df.groupby([col, "camera"])["individual"].nunique().unstack(fill_value=0)
        return [{col: str(k), **{f"specimens_{c}": int(v) for c, v in row.items()}} for k, row in t.iterrows()]

    sites = df.groupby("site")["camera"].nunique()
    lakes = df.groupby("lake")["camera"].nunique()

    cc = counts(df, ["taxon_group", "camera", SPLIT])
    wide = cc.pivot_table(index=["taxon_group", "camera"], columns=SPLIT, values=["specimens", "images"], fill_value=0)
    wide.columns = [f"{s}_{u}" for u, s in wide.columns]
    wide = wide.reset_index()
    for s in ("train", "val", "test"):
        for u in ("specimens", "images"):
            wide[f"{s}_{u}"] = wide.get(f"{s}_{u}", 0)
    wide["eligible"] = (wide.train_specimens >= a.min_train) & (wide.val_specimens >= a.min_val)
    cams = sorted(df.camera.unique())
    elig = wide.pivot_table(index="taxon_group", columns="camera", values="eligible", fill_value=False).astype(bool)
    elig = elig.reindex(columns=cams, fill_value=False)
    both = sorted(elig.index[elig.all(axis=1)])
    wide.to_csv(a.out / "class_camera_counts.csv", index=False)

    result = {
        "manifest": a.manifest, "split_column": SPLIT, "images": len(df), "specimens": int(df.individual.nunique()),
        "classes": int(df.taxon_group.nunique()),
        "cameras": records(counts(df, ["camera"])),
        "split_by_camera": records(counts(df, [SPLIT, "camera"])),
        "specimens_in_more_than_one_camera": int((cams_per_specimen > 1).sum()),
        "specimens_in_more_than_one_class": int((classes_per_specimen > 1).sum()),
        "specimens_in_more_than_one_split": int((splits_per_specimen > 1).sum()),
        "imaging_runs": int(len(cams_per_run)), "imaging_runs_with_more_than_one_camera": int((cams_per_run > 1).sum()),
        "year_by_camera": by_camera("year"), "dataset_by_camera": by_camera("dataset"),
        "sites": int(len(sites)), "sites_with_both_cameras": int((sites > 1).sum()),
        "lakes": int(len(lakes)), "lakes_with_both_cameras": int((lakes > 1).sum()),
        "support_rule": {"min_train_specimens": a.min_train, "min_val_specimens": a.min_val},
        "class_camera_pairs_eligible": int(wide.eligible.sum()), "class_camera_pairs": int(len(wide)),
        "classes_eligible_in_every_camera": both,
        "classes_eligible_per_camera": {c: sorted(elig.index[elig[c]]) for c in cams},
    }
    (a.out / "inventory.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("classes_eligible_per")}, indent=1))


if __name__ == "__main__":
    main()
