"""Real-image montages of the specimens listed in class_geometry_pilot.py's ``selection.json`` (train specimens only).

Per selected class, one row per group (2-component tied mixture components, then the two lowest-density outliers);
each specimen shows its first cohort frame from camera 1 and from camera 2, labelled with the specimen ID.
Images are read from the HDF5 by name (Hdf5ImageDataset checks the name at each position) or, for cohorts with a
``path`` column, from ``--image-root``.

Usage: class_geometry_montage.py --cohort cohort.parquet (--hdf5 H5 | --image-root DIR) --selection selection.json
       --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

from src.data.image_source import image_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--hdf5")
    source.add_argument("--image-root")
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fold", default="fold0", help="partition column; montages show its train specimens")
    a = ap.parse_args()
    cohort = pd.read_parquet(a.cohort)
    cohort = cohort[cohort[a.fold] == "train"].sort_values(["individual", "camera", "img"])
    first = cohort.groupby(["individual", "camera"]).head(1).set_index(["individual", "camera"])
    cameras = sorted(cohort.camera.unique())
    sel = json.loads(a.selection.read_text())["montage_specimens"]
    for cls, groups in sel.items():
        rows = list(groups.items())
        width = max(len(v) for _, v in rows) * len(cameras)
        fig, axes = plt.subplots(len(rows), width, figsize=(1.6 * width, 1.9 * len(rows)), squeeze=False)
        for r, (group, ids) in enumerate(rows):
            picks = [(i, cam) for i in ids for cam in cameras]
            present = [p for p in picks if p in first.index]  # single-camera specimens leave an empty panel
            ds = image_dataset(first.loc[present].reset_index(), lambda im: im, a.hdf5, a.image_root)
            for j in range(width):
                ax = axes[r, j]
                ax.axis("off")
                if j < len(picks):
                    ax.set_title(f"{picks[j][0]} cam {picks[j][1]}", fontsize=5)
                    if picks[j] in present:
                        image, _ = ds[present.index(picks[j])]
                        image.thumbnail((200, 200))
                        ax.imshow(image)
                    else:
                        ax.text(0.5, 0.5, "no image\nfrom this camera", ha="center", va="center", fontsize=5)
            axes[r, 0].text(-0.1, 0.5, group.replace("_", " "), transform=axes[r, 0].transAxes, rotation=90,
                            va="center", ha="right", fontsize=7)
        fig.suptitle(f"{cls}: train specimens per group (camera 1 and 2 frames)", fontsize=9)
        fig.tight_layout()
        fig.savefig(a.out / f"montage_{cls.replace(' ', '_')}.png", dpi=120)
        plt.close(fig)
        print(f"{cls}: {sum(len(v) for v in groups.values())} specimens")


if __name__ == "__main__":
    main()
