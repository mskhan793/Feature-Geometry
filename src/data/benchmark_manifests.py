"""Standardised image manifests of the eight-dataset benchmark (task.md 8538df5; docs/benchmark_standard.md).

One row per image, columns (``COLUMNS``):
- ``dataset``, ``version``: dataset key and the release used;
- ``source_id``: the publisher's image identifier (file name, member path, processid or array index);
- ``unit_id``: the independent unit (specimen, observation or image; repeated views share it). For image-unit
  datasets it is the exact-content hash (sha1 of the pixels or file bytes), so identical images share one unit
  (never counted as independent; derived validation keeps them on one side; cross-split repeats are reported);
- ``label`` (the benchmark class), ``coarse_label`` (published coarser label, else empty);
- split columns: ``split`` (train / val / test) for single-protocol datasets, or ``fold0``..``fold4`` for the two
  five-fold aquatic datasets; ``val_origin`` = 'official' or 'derived' (derived validation drawn from the official
  train / trainval only, with ``derived_seed``);
- ``image_ref``: where the bytes are (archive member, HDF5 name, pickle index) and ``available`` (bytes present).

Derived validation (CIFAR-100, Food-101, Oxford-IIIT Pet; task.md 8538df5): per class, the units of the official
train (trainval for Pet) in sha1(f"{seed}:{unit}") order; the first round(fraction x n_c) go to val. Official test is
never read for it. The frozen FIN-Benthic2 folds, AquaMonitor folds, BIOSCAN-5M splits and the frozen DF24 derived
validation are reproduced, never regenerated.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.expanded_cohort import stable_order

COLUMNS = ["dataset", "version", "source_id", "unit_id", "label", "coarse_label", "image_ref", "available", "val_origin",
           "derived_seed"]
FOLDS = [f"fold{k}" for k in range(5)]
SEED, FRACTION = 20261112, 0.1


def derive_validation(train: pd.DataFrame, seed: int = SEED, fraction: float = FRACTION) -> pd.Series:
    """Boolean per row of ``train``: the unit is in the derived validation (stratified by label, units kept whole)."""
    units = train.drop_duplicates("unit_id")[["unit_id", "label"]].copy()
    units["key"] = stable_order(units.unit_id.astype(str), seed).values
    chosen = set()
    for _, g in units.sort_values("key").groupby("label", sort=False):
        chosen |= set(g.unit_id.iloc[:int(round(fraction * len(g)))])
    return train.unit_id.isin(chosen)


def _finish(df: pd.DataFrame, dataset: str, version: str, split_cols: list[str]) -> pd.DataFrame:
    df = df.assign(dataset=dataset, version=version)
    for c in ("coarse_label", "image_ref"):
        if c not in df:
            df[c] = ""
    for c, v in (("available", True), ("val_origin", "official"), ("derived_seed", -1)):
        if c not in df:
            df[c] = v
    out = df[COLUMNS + split_cols].reset_index(drop=True)
    assert out.source_id.is_unique, f"{dataset}: duplicate source ids"
    return out


def _with_derived_val(df: pd.DataFrame, train_name: str) -> pd.DataFrame:
    train = df.split == train_name
    val = pd.Series(False, index=df.index)
    val[train] = derive_validation(df[train]).values
    df.loc[train, "split"] = np.where(val[train], "val", "train")
    df["val_origin"], df["derived_seed"] = "derived", SEED
    return df


def finbenthic2(csv: Path) -> pd.DataFrame:
    """Author-created five-split CSV (frozen, sha256 1c18f058…); unit = specimen ``individual``; label = ``taxon``."""
    d = pd.read_csv(csv)
    d = d.rename(columns={"img": "source_id", "individual": "unit_id", "taxon": "label"})
    d["image_ref"] = d.source_id
    return _finish(d, "finbenthic2", "five-split CSV 1c18f058", FOLDS)


def aquamonitor(manifest: Path) -> pd.DataFrame:
    """Official Classification folds (manifest of the authors' classif parquet, sha256 d6e63bb6…)."""
    d = pd.read_parquet(manifest)
    d = d.rename(columns={"img": "source_id", "individual": "unit_id", "taxon_group": "label"})
    d["image_ref"] = "hdf5_row " + d.hdf5_row.astype(str)
    return _finish(d, "aquamonitor", "classif parquet df18b019", FOLDS)


BIOSCAN_CLOSED_SET = ("train", "val", "test")


def bioscan5m(index: Path) -> pd.DataFrame:
    """Species-labelled BIOSCAN-5M images of the official closed-set splits train / val / test (v1 index of the
    cropped_256 archives); the open-world partitions (key_unseen, val_unseen, test_unseen) are not part of this
    benchmark. Unit = processid (one image per specimen record); label = species, coarse = genus."""
    d = pd.read_parquet(index)
    d = d[d.split.isin(BIOSCAN_CLOSED_SET)].copy()
    d = d.rename(columns={"processid": "source_id", "species": "label", "genus": "coarse_label"})
    d["unit_id"], d["image_ref"] = d.source_id, d.zip + ":" + d.member
    return _finish(d, "bioscan5m", "BIOSCAN-5M cropped_256", ["split"])


def df24(train_csv: Path, test_csv: Path, assignment: Path) -> pd.DataFrame:
    """Official train / pubtest; frozen derived validation (seed 20261110) by observation; images not downloaded.
    Label = ``scientificName`` (one-to-one with the official ``class_id``, 1604 classes; the ``species`` column merges
    25 names with several class ids and is empty for one class), coarse = genus."""
    tr, te = pd.read_csv(train_csv, low_memory=False), pd.read_csv(test_csv, low_memory=False)
    a = pd.read_csv(assignment)[["observationID", "split"]]
    tr = tr.merge(a, on="observationID", how="left", validate="many_to_one")
    assert tr.split.notna().all()
    te["split"] = "test"
    d = pd.concat([tr, te], ignore_index=True)
    assert d.groupby("class_id").scientificName.nunique().max() == 1 and d.groupby("scientificName").class_id.nunique().max() == 1
    d = d.rename(columns={"image_path": "source_id", "observationID": "unit_id", "scientificName": "label", "genus": "coarse_label"})
    d["image_ref"], d["available"] = d.source_id, False
    d["val_origin"], d["derived_seed"] = "derived", 20261110
    return _finish(d, "df24", "DanishFungi2024 train/pubtest", ["split"])


def cifar100(folder: Path) -> pd.DataFrame:
    """Publisher python pickles; source id = array position (the publisher's file names are not unique); unit = pixel
    hash; label = fine label name, coarse = superclass name; image_ref = publisher file name."""
    meta = pickle.load(open(folder / "meta", "rb"), encoding="latin1")
    rows = []
    for part, split in (("train", "train"), ("test", "test")):
        p = pickle.load(open(folder / part, "rb"), encoding="latin1")
        rows.append(pd.DataFrame({"source_id": [f"{part}[{i}]" for i in range(len(p["filenames"]))],
                                  "label": [meta["fine_label_names"][i] for i in p["fine_labels"]],
                                  "coarse_label": [meta["coarse_label_names"][i] for i in p["coarse_labels"]],
                                  "image_ref": p["filenames"], "split": split,
                                  "unit_id": [hashlib.sha1(r.tobytes()).hexdigest() for r in p["data"]]}))
    d = pd.concat(rows, ignore_index=True)
    return _finish(_with_derived_val(d, "train"), "cifar100", "cifar-100-python (md5 eb9058c3)", ["split"])


def _content_units(d: pd.DataFrame, content: pd.Series) -> pd.DataFrame:
    """``content``: sha1 of the file bytes per archive member (scripts/lumi/hash_archive_members.py). Availability =
    member present; unit = the sha1 (identical files share a unit)."""
    d["available"] = d.image_ref.isin(content.index)
    assert d.available.all(), f"{(~d.available).sum()} listed images missing from the archive"
    d["unit_id"] = d.image_ref.map(content)
    return d


def food101(meta: Path, content: pd.Series) -> pd.DataFrame:
    """Official meta/train.txt and meta/test.txt (``class/id``); unit = image."""
    rows = [pd.DataFrame({"source_id": (meta / f"{s}.txt").read_text().split(), "split": s}) for s in ("train", "test")]
    d = pd.concat(rows, ignore_index=True)
    d["label"] = d.source_id.str.split("/").str[0]
    d["image_ref"] = "food-101/images/" + d.source_id + ".jpg"
    d = _content_units(d, content)
    return _finish(_with_derived_val(d, "train"), "food101", "food-101.tar.gz (2014)", ["split"])


def oxford_pets(annotations: Path, content: pd.Series) -> pd.DataFrame:
    """Official annotations/trainval.txt and test.txt (``image class_id species breed_id``); label = breed (37)."""
    rows = []
    for s, split in (("trainval", "train"), ("test", "test")):
        t = pd.read_csv(annotations / f"{s}.txt", sep=" ", header=None, names=["source_id", "class_id", "species", "breed_id"])
        t["split"] = split
        rows.append(t)
    d = pd.concat(rows, ignore_index=True)
    d["label"] = d.source_id.str.rsplit("_", n=1).str[0]
    d["coarse_label"] = d.species.map({1: "cat", 2: "dog"})
    d["image_ref"] = "images/" + d.source_id + ".jpg"
    d = _content_units(d, content)
    return _finish(_with_derived_val(d, "train"), "oxford_pets", "images.tar.gz + annotations.tar.gz (2012)", ["split"])


def fgvc_aircraft(data: Path, content: pd.Series) -> pd.DataFrame:
    """Official images_variant_{train,val,test}.txt (``id variant``); label = variant (100), coarse = family."""
    rows = []
    for s in ("train", "val", "test"):
        v = pd.read_csv(data / f"images_variant_{s}.txt", header=None, names=["line"])
        f = pd.read_csv(data / f"images_family_{s}.txt", header=None, names=["line"])
        ids = v.line.str.split(" ", n=1).str[0]
        assert (ids == f.line.str.split(" ", n=1).str[0]).all()
        rows.append(pd.DataFrame({"source_id": ids, "label": v.line.str.split(" ", n=1).str[1],
                                  "coarse_label": f.line.str.split(" ", n=1).str[1], "split": s}))
    d = pd.concat(rows, ignore_index=True)
    d["image_ref"] = "fgvc-aircraft-2013b/data/images/" + d.source_id + ".jpg"
    d = _content_units(d, content)
    return _finish(d, "fgvc_aircraft", "fgvc-aircraft-2013b", ["split"])


def support(manifest: pd.DataFrame, split_col: str, min_train: int = 15, min_val: int = 5) -> dict:
    """Images / units / classes per split, per-class unit support, eligibility (>= min_train train and >= min_val val
    units), excluded classes and long-tail percentiles of train units per class, for one split column."""
    units = manifest.drop_duplicates(["unit_id", split_col])
    counts = units.groupby(["label", split_col]).size().unstack(fill_value=0)
    for s in ("train", "val", "test"):
        if s not in counts:
            counts[s] = 0
    eligible = counts[(counts.train >= min_train) & (counts.val >= min_val)].index
    tr = counts.train[counts.train > 0]
    return {"split_column": split_col, "classes": int(manifest.label.nunique()),
            "images": {s: int((manifest[split_col] == s).sum()) for s in ("train", "val", "test")},
            "units": {s: int((units[split_col] == s).sum()) for s in ("train", "val", "test")},
            "classes_with_units": {s: int((counts[s] > 0).sum()) for s in ("train", "val", "test")},
            "eligible_classes": int(len(eligible)), "excluded_classes": int(len(counts) - len(eligible)),
            "train_units_per_class_percentiles": {f"p{q}": float(np.percentile(tr, q)) for q in (0, 10, 50, 90, 100)},
            "images_available": int(manifest.available.sum()), "units_in_several_splits": int(
                (manifest.drop_duplicates(["unit_id", split_col]).groupby("unit_id")[split_col].nunique() > 1).sum()),
            "units_with_several_labels": int((manifest.groupby("unit_id").label.nunique() > 1).sum()),
            "images_sharing_a_unit": int(manifest.unit_id.duplicated(keep=False).sum())}
