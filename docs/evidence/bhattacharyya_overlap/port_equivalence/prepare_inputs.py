"""Inputs for the GBC port equivalence check (run with the Feature-Geometry environment, PYTHONPATH = repo).

1. dummy: 7 classes x 64 dims, float32, unequal class sizes (3..300) and scales, one class with a constant feature
   (forces the 1e-4 variance floor of the original code);
2. local real: FIN-Benthic2 fold-0 train specimen embeddings (DINOv3-L, camera-balanced specimen means) from the
   verified local cache, unwhitened PCA to 64 fitted on these train specimens, float32 (the original runs on float32
   model features).
Writes inputs.npz next to this file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.features.feature_lookup import features_by_name
from src.features.specimen_means import specimen_camera_means, specimen_means

rng = np.random.default_rng(20261111)
sizes, xs, ys = [3, 10, 40, 120, 300, 25, 60], [], []
for c, n in enumerate(sizes):
    x = rng.normal(loc=rng.normal(0, 2, 64), scale=rng.uniform(0.2, 3.0, 64), size=(n, 64))
    if c == 1:
        x[:, 5] = 0.7  # constant feature
    xs.append(x)
    ys.append(np.full(n, c))
dummy_x, dummy_y = np.concatenate(xs).astype(np.float32), np.concatenate(ys)

data = Path.home() / "Documents/Research/Feature_geometry_data/finbenthic2_fold0"
cohort = pd.read_parquet(data / "cohort/cohort_hdf5.parquet")
cohort = cohort[cohort.fold0 == "train"].reset_index(drop=True)
x, _ = features_by_name([(data / "cohort/cohort_hdf5.parquet", data / "features_lumi/dinov3_l")], cohort.img.tolist())
keys, means, _ = specimen_camera_means(x, cohort)
sk, sp = specimen_means(keys, means)
z = PCA(n_components=64, svd_solver="full").fit_transform(sp).astype(np.float32)
codes = {c: i for i, c in enumerate(sorted(sk.taxon_group.unique()))}
real_y = sk.taxon_group.map(codes).to_numpy()
np.savez(Path(__file__).with_name("inputs.npz"), dummy_x=dummy_x, dummy_y=dummy_y, real_x=z, real_y=real_y)
print("dummy", dummy_x.shape, "real", z.shape, "classes", len(codes), file=sys.stderr)
