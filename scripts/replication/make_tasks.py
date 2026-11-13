"""Task table of the 8 x 4 geometry replication (task.md a82e89e): one row per encoder x dataset x partition x analysis,
read by scripts/lumi/replicate_geometry.sh (one Slurm array task per row).

Inputs per combination come from the C3 record table (``c2_plan/records.tsv``: the verified caches of every
combination, chunk caches as MANIFEST=CACHE pairs, the v1 BIOSCAN-5M caches as INDEX=CACHE, read through
src.features.feature_lookup). Cohorts: the FIN-Benthic2 / AquaMonitor full-specimen selections (partitions fold0..4,
test rows dropped per fold before any feature lookup) and, for the other six datasets, the cohorts of
scripts/replication/build_cohorts.py (partition ``official`` = train / val; no test row exists in them).

Analyses (the established scripts, established settings):
- ``geometry``: class_geometry_pilot.py (PCs 8 16, n_stab 200, seed 20261107; source groups only for AquaMonitor's
  ``dataset`` field) + component_metadata.py (acquisition fields only where documented: AquaMonitor dataset/year/
  site, FIN-Benthic2 log image area; elsewhere only the per-class summary);
- ``overlap``: bhattacharyya_overlap.py (PCs 64 8 16);
- ``fda``: pca_fda_gbc.py (dims 8 16 32, PCA base 64);
- ``lw``: lw_fit_diagnostics.py (defaults: frames 1 3 5 10, caps 15-256, PCs 64 16);
- ``coupled``: coupled_rscm_vs_lw.py;
- ``pooling``: balanced_taxonomic_pooling.py --no-taxonomy (author instruction 2026-11-13).
Spaces of ``coupled`` / ``pooling``: DINO encoders as established (pca64 pca16 native native_alt: one coordinate
dropped because of the final-LayerNorm null direction); BioCLIP 2 / SigLIP 2 have full-rank features, so nothing needs
dropping: pca64 pca16 native_all (all coordinates) and native (last coordinate dropped, the sensitivity counterpart
of native_alt).

Usage: make_tasks.py --records records.tsv --cohort-root LUMI_DIR --out tasks.tsv [--only ENC:DATASET ...]
       [--analyses geometry overlap ...]
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pandas as pd

ANALYSES = ("geometry", "overlap", "fda", "lw", "coupled", "pooling")
FIVE_FOLD = ("finbenthic2", "aquamonitor")
FULL_RANK = ("bioclip2", "siglip2")
SOURCE = {"aquamonitor": "dataset"}
META = {"aquamonitor": "--fields dataset year site", "finbenthic2": "--fields --numeric log_image_area"}


def caches(source_args: str) -> str:
    t = shlex.split(source_args)
    if t[0] == "--chunk":
        return " ".join(t[1:])
    opt = dict(zip(t[0::2], t[1::2]))
    return f"{opt['--v1-index']}={opt['--v1-cache']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--cohort-root", required=True, help="LUMI folder with the six single-protocol cohorts")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", nargs="*", help="ENCODER:DATASET pairs to keep (default all 32)")
    ap.add_argument("--analyses", nargs="+", default=list(ANALYSES), choices=ANALYSES)
    a = ap.parse_args()
    rec = pd.read_csv(a.records, sep="\t", header=None, names=["encoder", "dataset", "manifest", "standard", "source", "weights"])
    rows = []
    for r in rec.itertuples():
        if a.only and f"{r.encoder}:{r.dataset}" not in a.only:
            continue
        five = r.dataset in FIVE_FOLD
        cohort = r.manifest if five else f"{a.cohort_root}/{r.dataset}.parquet"
        spaces = "pca64 pca16 native_all native" if r.encoder in FULL_RANK else "pca64 pca16 native native_alt"
        for part in ([f"fold{k}" for k in range(5)] if five else ["official"]):
            for an in a.analyses:
                extra = {"geometry": f"--source-column {SOURCE.get(r.dataset, 'none')}",
                         "overlap": "--pcs 64 8 16", "fda": "--dims 8 16 32", "lw": "--pcs 64 16",
                         "coupled": f"--spaces {spaces}",
                         "pooling": f"--dataset {r.dataset} --no-taxonomy --spaces {spaces}"}[an]
                rows.append({"encoder": r.encoder, "dataset": r.dataset, "partition": part, "analysis": an,
                             "cohort": cohort, "caches": caches(r.source), "extra": extra,
                             "meta": META.get(r.dataset, "--fields")})  # no empty field: bash read collapses tabs
    t = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, sep="\t", index=False)
    print(len(t), "tasks;", t.groupby("analysis").size().to_dict())


if __name__ == "__main__":
    main()
