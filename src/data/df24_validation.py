"""Derived validation set for DF24 (author-approved exception, 2026-11-10), drawn from the official training
observations only; the official ``pubtest`` is never read here.

Unit: ``observationID`` (all images of an observation stay together). Near-duplicate observations, those sharing a
recorder (``rightsHolder``), species (``class_id``), date and ``locality`` on any of their images, are linked into one
group and kept on one side. Per species, groups are drawn in a seeded random order until about ``fraction`` of the
species' observations are in validation, never exceeding the target; species with fewer than ``min_observations``
training observations stay train-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEAR_DUPLICATE_KEY = ["rightsHolder", "class_id", "year", "month", "day", "locality"]


def near_duplicate_groups(train: pd.DataFrame) -> pd.Series:
    """Group id per observationID: observations linked by a shared complete near-duplicate key (union-find)."""
    parent = {o: o for o in train.observationID.unique()}

    def root(o):
        while parent[o] != o:
            parent[o] = parent[parent[o]]
            o = parent[o]
        return o

    keyed = train.dropna(subset=NEAR_DUPLICATE_KEY)
    for _, obs in keyed.groupby(NEAR_DUPLICATE_KEY).observationID:
        first, *rest = obs.unique()
        for o in rest:
            parent[root(o)] = root(first)
    return pd.Series({o: root(o) for o in parent}, name="group").rename_axis("observationID")


def derive_validation(train: pd.DataFrame, seed: int, fraction: float = 0.1,
                      min_observations: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (one row per observation: observationID, class_id, group, images, split) and a per-species summary."""
    if (train.groupby("observationID").class_id.nunique() > 1).any():
        raise ValueError("an observation has images of more than one class")
    obs = train.groupby("observationID").agg(class_id=("class_id", "first"), images=("class_id", "size"))
    obs = obs.join(near_duplicate_groups(train)).reset_index().sort_values("observationID")
    obs["split"] = "train"
    rng = np.random.default_rng(seed)
    rows = []
    for cls, members in obs.groupby("class_id", sort=True):
        n = len(members)
        sizes = members.groupby("group").size().sort_index()
        target = max(1, round(fraction * n)) if n >= min_observations else 0
        chosen, selected = [], 0
        for g in sizes.index[rng.permutation(len(sizes))] if target else []:
            if selected + sizes[g] <= target:
                chosen.append(g)
                selected += sizes[g]
            if selected == target:
                break
        obs.loc[obs.group.isin(chosen), "split"] = "val"
        status = ("train_only_too_few" if not target else "no_group_fits_target" if not selected
                  else "below_target" if selected < target else "ok")
        rows.append({"class_id": cls, "observations": n, "groups": len(sizes), "target_val": target,
                     "val_observations": selected, "train_observations": n - selected, "status": status})
    return obs[["observationID", "class_id", "group", "images", "split"]], pd.DataFrame(rows)
