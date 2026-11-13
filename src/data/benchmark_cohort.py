"""Analysis cohort of a single-protocol benchmark (task.md a82e89e): the train+val extraction manifest of
BIOSCAN-5M, DF24, Food-101, Oxford-IIIT Pet, CIFAR-100 or FGVC-Aircraft in the cohort schema the geometry scripts
read (FIN-Benthic2 / AquaMonitor selections already have it):

- ``img`` (feature-cache key), ``taxon_group`` (the benchmark label), ``individual`` (the independent unit),
  ``camera`` and ``imaging_run`` (one constant value: these datasets record no camera or acquisition run), and the
  partition column ``SPLIT`` (train / val; the manifests hold no test row, which is asserted).
- Unit: ``unit_id`` of the standardised manifest (BIOSCAN-5M record, DF24 observation, exact-content hash of an
  image elsewhere). Its embedding is the mean of its images (specimen_means with one camera).
- An exact-duplicate content unit whose images carry different official labels (CIFAR-100, Food-101) keeps every
  image and label: it becomes one unit per (unit_id, label), ``individual`` = ``unit_id|label``, and is counted in
  ``conflicting_label_units``. Nothing is removed or relabelled.
"""

from __future__ import annotations

import pandas as pd

SPLIT = "official"
SINGLE = "single"


def cohort(extract: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if not set(extract.split) <= {"train", "val"}:
        raise RuntimeError(f"extraction manifest holds splits {sorted(set(extract.split))}; only train and val allowed")
    e = extract.reset_index(drop=True)
    labels = e.groupby("unit_id").label.nunique()
    conflicting = set(labels.index[labels > 1])
    splits = e.groupby("unit_id").split.nunique()
    if (splits > 1).any():
        raise RuntimeError(f"{int((splits > 1).sum())} units in both train and val")
    individual = [f"{u}|{lab}" if u in conflicting else u for u, lab in zip(e.unit_id.astype(str), e.label.astype(str))]
    out = pd.DataFrame({"img": e.img.astype(str), "taxon_group": e.label.astype(str), "individual": individual,
                        "camera": SINGLE, "imaging_run": SINGLE, SPLIT: e.split})
    info = {"images": len(out), "units": int(out.individual.nunique()), "conflicting_label_units": len(conflicting),
            "conflicting_label_images": int(e.unit_id.isin(conflicting).sum())}
    return out, info


def support(c: pd.DataFrame, split: str, min_train: int, min_val: int) -> pd.DataFrame:
    """Units per class in train and val of partition column ``split`` and whether the class meets the established
    eligibility rule (>= min_train train and >= min_val val units)."""
    u = c[c[split].isin(["train", "val"])].drop_duplicates("individual")
    t = u.groupby(["taxon_group", split]).size().unstack(fill_value=0).reindex(columns=["train", "val"], fill_value=0)
    t["eligible"] = (t.train >= min_train) & (t.val >= min_val)
    return t.reset_index()
