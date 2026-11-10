"""Draw the one derived DF24 validation set from the official training observations and verify it.

Writes to ``--out``: ``df24_split_assignment.csv.gz`` (observationID, class_id, group, images, split = train|val),
``df24_species_support.csv`` (per species: observations, groups, target, val/train observations, status) and
``df24_derived_validation.json`` (seed, rule, source sha256, counts, checks). The official pubtest file is read
only to verify that it is untouched and disjoint and to report near-duplicate keys shared with each side.

Usage: derive_df24_validation.py --train DanishFungi2024-train.csv --test DanishFungi2024-pubtest.csv --out DIR
       [--seed 20261110 --fraction 0.1 --min-observations 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.df24_validation import NEAR_DUPLICATE_KEY, derive_validation


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--min-observations", type=int, default=10)
    a = ap.parse_args()
    test_hash = sha256(a.test)
    train = pd.read_csv(a.train, low_memory=False)
    test = pd.read_csv(a.test, low_memory=False)
    obs, support = derive_validation(train, a.seed, a.fraction, a.min_observations)

    side = obs.set_index("observationID").split
    image_side = train.observationID.map(side)
    checks = {
        "every_train_observation_assigned_once": bool(obs.observationID.is_unique
                                                     and set(obs.observationID) == set(train.observationID)),
        "images_preserved": int(obs.images.sum()) == len(train),
        "no_observation_in_val_and_pubtest": not set(obs.observationID[obs.split == "val"]) & set(test.observationID),
        "no_observation_in_train_and_pubtest": not set(obs.observationID[obs.split == "train"]) & set(test.observationID),
        "near_duplicate_groups_not_split": bool((obs.groupby("group").split.nunique() == 1).all()),
        "all_images_of_an_observation_on_one_side": bool(image_side.notna().all()),
        "labels_present": bool(train.class_id.notna().all() and obs.class_id.notna().all()),
        "pubtest_unchanged": sha256(a.test) == test_hash,
    }
    if not all(checks.values()):
        raise RuntimeError(f"integrity check failed: {checks}")

    def keys(df):
        return set(map(tuple, df.dropna(subset=NEAR_DUPLICATE_KEY)[NEAR_DUPLICATE_KEY].drop_duplicates().to_numpy()))

    test_keys = keys(test)
    shared = {s: int(sum(k in test_keys for k in keys(train[image_side == s]))) for s in ("train", "val")}
    counts = {s: {"observations": int((obs.split == s).sum()), "images": int(obs.images[obs.split == s].sum()),
                  "classes": int(obs.class_id[obs.split == s].nunique())} for s in ("train", "val")}
    counts["pubtest"] = {"observations": int(test.observationID.nunique()), "images": len(test),
                         "classes": int(test.class_id.nunique())}
    a.out.mkdir(parents=True, exist_ok=True)
    obs.to_csv(a.out / "df24_split_assignment.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    support.to_csv(a.out / "df24_species_support.csv", index=False)
    summary = {
        "protocol": "DF24 official train and pubtest; validation derived from official train only (author-approved "
                    "exception, task 2f6774c); not an official split",
        "rule": {"seed": a.seed, "fraction": a.fraction, "min_observations": a.min_observations,
                 "unit": "observationID", "near_duplicate_key": NEAR_DUPLICATE_KEY},
        "sources": {str(a.train): sha256(a.train), str(a.test): test_hash},
        "assignment_sha256": sha256(a.out / "df24_split_assignment.csv.gz"),
        "counts": counts,
        "near_duplicate_groups": {"groups": int(obs.group.nunique()),
                                  "observations_in_multi_observation_groups":
                                      int((obs.groupby("group").observationID.transform("size") > 1).sum())},
        "species_status": support.status.value_counts().to_dict(),
        "near_duplicate_keys_shared_with_pubtest": shared,
        "checks": checks}
    (a.out / "df24_derived_validation.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
