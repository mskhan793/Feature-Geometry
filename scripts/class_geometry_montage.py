"""Real-image montages of the specimens listed in class_geometry_pilot.py's ``selection.json`` (train specimens only).

Per selected class, one row per group (2-component tied mixture components, then the two lowest-density outliers);
each specimen shows its first cohort frame from camera 1 and from camera 2, labelled with the specimen ID.
Images are read from the HDF5 by name (Hdf5ImageDataset checks the name at each position).

Usage: class_geometry_montage.py --cohort cohort.parquet --hdf5 H5 --selection selection.json --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

from src.data.hdf5_images import Hdf5ImageDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cohort = pd.read_parquet(a.cohort)
    cohort = cohort[cohort.fold0 == "train"].sort_values(["individual", "camera", "img"])
    first = cohort.groupby(["individual", "camera"]).head(1).set_index(["individual", "camera"])
    cameras = sorted(cohort.camera.unique())
    sel = json.loads(a.selection.read_text())["montage_specimens"]
    for cls, groups in sel.items():
        rows = list(groups.items())
        width = max(len(v) for _, v in rows) * len(cameras)
        fig, axes = plt.subplots(len(rows), width, figsize=(1.6 * width, 1.9 * len(rows)), squeeze=False)
        for r, (group, ids) in enumerate(rows):
            picks = [(i, cam) for i in ids for cam in cameras]
            names = [first.loc[p, "img"] for p in picks]
            ds = Hdf5ImageDataset(a.hdf5, names, [int(first.loc[p, "hdf5_row"]) for p in picks], lambda im: im)
            for j in range(width):
                ax = axes[r, j]
                ax.axis("off")
                if j < len(picks):
                    image, _ = ds[j]
                    image.thumbnail((200, 200))
                    ax.imshow(image)
                    ax.set_title(f"{picks[j][0]} cam {picks[j][1]}", fontsize=5)
            axes[r, 0].text(-0.1, 0.5, group.replace("_", " "), transform=axes[r, 0].transAxes, rotation=90,
                            va="center", ha="right", fontsize=7)
        fig.suptitle(f"{cls}: train specimens per group (camera 1 and 2 frames)", fontsize=9)
        fig.tight_layout()
        fig.savefig(a.out / f"montage_{cls.replace(' ', '_')}.png", dpi=120)
        plt.close(fig)
        print(f"{cls}: {sum(len(v) for v in groups.values())} specimens")


if __name__ == "__main__":
    main()
