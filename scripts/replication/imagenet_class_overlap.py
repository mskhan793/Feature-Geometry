"""Class-level ImageNet-1K overlap audit of the four general-image benchmarks (task.md 3b897b0: I-JEPA ViT-H/14 was
pretrained on ImageNet-1K images; DINOv2/v3, BioCLIP 2 and SigLIP 2 corpora are larger and not audited here).
Metadata only, read-only.

ImageNet-1K class names: torchvision's ``ResNet50_Weights.IMAGENET1K_V1.meta["categories"]`` (the 1,000 standard
names). Benchmark class names: the ``taxon_group`` values of the analysis cohorts (Food-101, Oxford-IIIT Pet,
CIFAR-100, FGVC-Aircraft). Names are normalised (lower case, ``_``/``-`` -> space). A benchmark class is a
**candidate** if its normalised name equals an ImageNet name or occurs in one as a whole-word phrase, or vice versa.
These are string candidates only (synonyms and different granularity are missed or over-matched); they are not
verified by hand and are not image-level duplicates. An image-level near-duplicate audit would need the ImageNet-1K
images, which are not available to this project, so it is not done.

Writes OUT/imagenet_class_candidates.csv (dataset, class, matching ImageNet names) and OUT/summary.json.

Usage: imagenet_class_overlap.py --cohorts DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

GENERAL = ("food101", "oxford_pets", "cifar100", "fgvc_aircraft")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]", " ", s.lower())).strip()


def contains(phrase: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def main():
    from torchvision.models import ResNet50_Weights

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    imagenet = [norm(c) for c in ResNet50_Weights.IMAGENET1K_V1.meta["categories"]]
    assert len(imagenet) == 1000
    rows, summary = [], {"imagenet_classes": len(imagenet), "source": "torchvision ResNet50_Weights.IMAGENET1K_V1 categories"}
    for d in GENERAL:
        classes = sorted(pd.read_parquet(a.cohorts / f"{d}.parquet", columns=["taxon_group"]).taxon_group.unique())
        hits = 0
        for c in classes:
            n = norm(c)
            match = [i for i in imagenet if n == i or contains(n, i) or contains(i, n)]
            hits += bool(match)
            rows.append({"dataset": d, "class": c, "candidate": bool(match), "imagenet_names": "; ".join(match)})
        summary[d] = {"classes": len(classes), "candidate_classes": hits}
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out / "imagenet_class_candidates.csv", index=False)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
