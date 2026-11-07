"""Specimen-level embeddings from image-level features of a cohort (frames are not independent samples)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def specimen_camera_means(x: np.ndarray, cohort: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Keys (one row per specimen and camera), mean embeddings, and RMS distance of each group's frames to its mean.
    Computed in float64: downstream PCA and mixture fits inherit the dtype, and float32 mixture fits lose narrow splits."""
    groups = cohort.groupby(["taxon_group", "fold0", "individual", "camera"], sort=True).indices
    keys = pd.DataFrame(list(groups), columns=["taxon_group", "fold0", "individual", "camera"])
    keys["frames"] = [len(i) for i in groups.values()]
    x = x.astype(np.float64)
    means = np.stack([x[i].mean(0) for i in groups.values()])
    spread = np.array([np.sqrt(((x[i] - x[i].mean(0)) ** 2).sum(1).mean()) for i in groups.values()])
    return keys, means, spread


def specimen_means(camera_keys: pd.DataFrame, camera_means: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """One embedding per specimen: the average of its camera-level means (each camera weighted equally)."""
    groups = camera_keys.groupby(["taxon_group", "fold0", "individual"], sort=True).indices
    keys = pd.DataFrame(list(groups), columns=["taxon_group", "fold0", "individual"])
    keys["cameras"] = [len(i) for i in groups.values()]
    return keys, np.stack([camera_means[i].mean(0) for i in groups.values()])
