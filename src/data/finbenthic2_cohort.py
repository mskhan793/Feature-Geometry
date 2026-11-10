"""FIN-Benthic2 image rows from the author's five-split CSV, in the column layout of the AquaMonitor cohorts.

Image names have one fixed form, ``{camera}-{individual}.{run}.{year}-{HH}-{MM}-{SS}-{12-digit fraction}-utc.png``:
the prefix is the camera (0 or 1), ``run`` the imaging run of that specimen, and the time fields have fixed width, so
name order within a specimen, camera and run is capture order. Any name not of this form raises. Columns: ``img``,
``path`` (``individual/img`` under ``IDA/Images``), ``individual``, ``taxon_group`` (= ``taxon``, the label),
``fold0``..``fold4``, ``camera`` and ``imaging_run`` (``individual.run``). There is no source-dataset field.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data.finbenthic2_folds import FOLDS

NAME = re.compile(r"^(?P<camera>[01])-(?P<specimen>.+)\.(?P<run>\d+)\.\d{4}-\d{2}-\d{2}-\d{2}-\d{12}-utc\.png$")


def finbenthic2_frames(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    parts = df.img.str.extract(NAME)
    if parts.camera.isna().any():
        raise ValueError(f"{int(parts.camera.isna().sum())} image names do not have the expected form, "
                         f"e.g. {df.img[parts.camera.isna()].iloc[0]}")
    if (parts.specimen != df.individual).any():
        raise ValueError("an image name does not contain its specimen id")
    return pd.DataFrame({"img": df.img, "path": df.individual + "/" + df.img, "individual": df.individual,
                         "taxon_group": df.taxon, **{f: df[f] for f in FOLDS}, "camera": parts.camera.astype(int),
                         "imaging_run": df.individual + "." + parts.run})
