"""Deterministic capped cohort inside one official split, for the camera-conditioned Gaussian pilot.

Capping only, not a new partition: rows keep their official split. Only specimens imaged by every camera are used, so
each selected specimen contributes to every camera (paired design). Per class, specimens are ranked by
``sha1(f"{seed}:{individual}")`` and the first ``n_specimens`` are kept (all if fewer). Per kept specimen and camera,
images are ranked by ``sha1(f"{seed}:{img}")`` and the first ``n_frames`` are kept. Nothing is duplicated.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def stable_key(values: pd.Series, seed: int) -> pd.Series:
    return values.map(lambda v: hashlib.sha1(f"{seed}:{v}".encode()).hexdigest())


def select_cohort(manifest: pd.DataFrame, split_column: str, split: str, classes: list[str], n_specimens: int,
                  n_frames: int, seed: int) -> pd.DataFrame:
    df = manifest[(manifest[split_column] == split) & manifest.taxon_group.isin(classes)].copy()
    cameras = manifest.camera.nunique()
    paired = df.groupby("individual").camera.nunique() == cameras
    df = df[df.individual.map(paired)]
    specimens = df[["taxon_group", "individual"]].drop_duplicates()
    specimens["key"] = stable_key(specimens.individual, seed)
    kept = specimens.sort_values("key").groupby("taxon_group").head(n_specimens).individual
    df = df[df.individual.isin(kept)]
    df["key"] = stable_key(df.img, seed)
    df = df.sort_values("key").groupby(["individual", "camera"]).head(n_frames)
    return df.drop(columns="key").sort_values("hdf5_row").reset_index(drop=True)
