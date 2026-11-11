"""Checks for a five-fold specimen protocol in which every specimen is test in exactly one fold (FIN-Benthic2 derived
five-split CSV, AquaMonitor Classification official folds): image rows with ``individual``, ``taxon_group`` and
``fold0``..``fold4``."""

from __future__ import annotations

import pandas as pd

FOLDS = [f"fold{k}" for k in range(5)]
PARTS = ("train", "val", "test")


def fold_checks(rows: pd.DataFrame) -> dict:
    """Specimens per fold and partition. Raises if a specimen has more than one class or partition within a fold, or
    is not in test in exactly one fold."""
    spec = rows.groupby("individual")
    if (spec[["taxon_group", *FOLDS]].nunique() > 1).any().any():
        raise RuntimeError("a specimen has more than one class or partition within a fold")
    s = spec[FOLDS].first()
    if not ((s == "test").sum(axis=1) == 1).all():
        raise RuntimeError("a specimen is not in test in exactly one fold")
    return {f: {p: int((s[f] == p).sum()) for p in PARTS} for f in FOLDS}
