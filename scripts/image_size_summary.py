"""Read-only: frame sizes of the AquaMonitor Classification benchmark images (parquet ``width`` / ``height``).

Prints how many distinct sizes there are, the most common ones, width and height ranges, and the share of square
frames. Used to decide how encoders that keep the aspect ratio can batch images.

Usage: image_size_summary.py --parquet CLASSIF_PARQUET --out result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.aquamonitor_manifest import FOLDS, OUTSIDE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    df = pd.read_parquet(a.parquet, columns=["width", "height", *FOLDS])
    df = df[~df[FOLDS].eq(OUTSIDE).any(axis=1)]
    sizes = df.groupby(["width", "height"]).size().sort_values(ascending=False)
    result = {
        "images": len(df), "distinct_sizes": len(sizes),
        "top_sizes": [{"width": int(w), "height": int(h), "images": int(n)} for (w, h), n in sizes.head(10).items()],
        "width_min_median_max": [int(df.width.min()), float(df.width.median()), int(df.width.max())],
        "height_min_median_max": [int(df.height.min()), float(df.height.median()), int(df.height.max())],
        "square_share": float((df.width == df.height).mean()),
        "width_476_share": float((df.width == 476).mean()),
    }
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
