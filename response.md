# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 366ee62, "FIN-Benthic2 fold-0 frozen DINOv3-L geometry replication".
**Status: done for fold 0, stopped for the author's review.**
- **Ran:** cohort and checks on the Dell; feature extraction on **LUMI** (author's instruction in chat, replacing the
  task's local-GPU wording; the local GPU run was stopped after about 1.5 min with nothing written); geometry,
  montages, metadata checks and the AquaMonitor comparison on the Dell.
- **Not done:** no test specimens extracted or read, no classifiers, no fine-tuning, folds 1–4, BIOSCAN-5M and DF24
  untouched.

**Evidence:** `docs/evidence/finbenthic2_geometry/` (compact; images, features and large arrays stay outside Git).
Split protocol background: `docs/dataset_eligibility_geometry.md` sections 0 and 7.

## 1. Answer in brief

- **FIN-Benthic2 does not reproduce AquaMonitor's main result.** On fold-0 validation specimens, a 2-component
  tied-covariance mixture per taxon predicts no better than one full Gaussian: −0.094 [−0.274, 0.064] at 8 PCs and
  −0.011 [−0.117, 0.086] at 16 PCs (AquaMonitor: +0.577 [0.505, 0.646] and +0.298 [0.046, 0.509]). Full-covariance
  mixtures are worse, as in AquaMonitor.
- **Per class, at 8 PCs:** 10 of 36 taxa "credible multiple groups", 23 inconclusive, 3 insufficient evidence
  (AquaMonitor: 30 of 41 credible). At 16 PCs: 10 credible, 16 inconclusive, 5 "one Gaussian favoured",
  3 insufficient. The 2-group split is stable in 14 taxa (AquaMonitor 35 of 41 at 8 PCs).
- **The clearest groups are life stages.** In the three riffle-beetle taxa (*Elmis aenea*, *Limnius volckmari*,
  *Oulimnius tuberculatus*) the two components are larvae and adult beetles in the montages (4 train specimens per
  component shown, both cameras). These splits are the strongest in the data: gains +1.04 to +1.50, val split ARI 1.0, both camera views of
  every specimen in the same component. AquaMonitor gives "*Oulimnius tuberculatus* adult" its own class, so there
  this split lies between labels; FIN-Benthic2 lumps the stages under one taxon. The life-stage reading comes from
  looking at images; there is no life-stage field to count it.
- **Elsewhere the groups look like acquisition or view, not biology.** FIN-Benthic2 has no source-dataset field
  (one collection, all 2016), so the AquaMonitor source model (GD_tied) cannot be fitted. Two checks instead:
  - the two camera views of one specimen fall into the same component less often than in AquaMonitor (median over
    taxa 0.856 at 8 PCs vs 0.995);
  - the components differ in crop size (a rough proxy for specimen size, orientation and zoom) in 20 of 36 taxa at
    8 PCs on train and 15 of 33 on val (AUC outside 0.2–0.8).
- **Within a taxon the cloud is broader and less dominated by one direction than in AquaMonitor:** effective rank
  median 34.2 vs 24.4 (16.4 vs 13.8 at a fixed 30 specimens), top direction 0.211 vs 0.246 of the variance. Diagonal
  2-component mixtures beat diagonal single Gaussians strongly (+0.597 [0.234, 0.852] at 8 PCs), as in AquaMonitor —
  a sign of correlated directions, not of separate groups.
- **Between taxa the picture matches AquaMonitor:** 12 of 630 pairs have val d′ < 2 (1.9 %; AquaMonitor 1.8 %),
  median d′ 5.41 vs 5.46. The closest pairs are related taxa: *Polycentropus flavomaculatus* / *P. irroratus* 0.78,
  *Kageronia fuscogrisea* / *Leptophlebia* sp. 1.34, *Baetis rhodani* / *B. vernus* group 1.55, *Leuctra nigra* /
  *Leuctra* sp. 1.58, *Elmis aenea* / *Oulimnius tuberculatus* 1.58. A 2-D PCA view overstates separation for 520 of
  630 pairs.

## 2. What ran

| Step | Where | Result |
|---|---|---|
| Frozen CSV and fold-0 check | Dell | sha256 `1c18f058…`; 6,165 / 1,542 / 1,924 specimens; no specimen in two partitions |
| Cohort (precommitted 0ba7b53, seed 20261111) | Dell | 5,148 train specimens (≤ 200 per taxon, stable sha1 order), all 1,542 val; ≤ 10 frames per camera; 128,937 images (99,052 train, 29,885 val); 170 + 51 one-camera specimens kept |
| Image packing | Dell | all 460,004 images into one HDF5, byte-verified, 11.9 GB, sha256 `e23e2433…`; copied to `/flash/project_462001534/feature_geometry/finbenthic2` (sha256 match) |
| Extraction | LUMI job 23623689, dev-g, 1 node, 8 GCDs | 7 min 48 s; 16 chunks; about 53 img/s per GCD on the first chunk, 118–136 on the second |
| Checks | LUMI and Dell | cache covers all 128,937 rows (verified on LUMI and again on the Dell); CPU vs GPU on 256 images: max abs diff 2.0e-5, mean 2.2e-6 |
| Geometry, montages, metadata, comparison | Dell (CPU) | geometry 139 s |

Code: commit 61098b3 on LUMI (98 files identical to Git). LUMI cost about 1 GCD-hour (0.5 GPU-h), project 462001534.

## 3. Comparison with AquaMonitor (matched unit, statistics and PCA dimension)

Full table: `docs/evidence/finbenthic2_geometry/compare/comparison.md`.

| | AquaMonitor | FIN-Benthic2 |
|---|---|---|
| Taxa analysed (insufficient) | 41 (1) | 36 (3) |
| Train / val specimens | 5,598 / 4,386 | 5,148 / 1,542 |
| Variance in 8 / 16 PCs | 0.583 / 0.699 | 0.533 / 0.653 |
| G2_tied − G1_full, 8 PCs | +0.577 [0.505, 0.646] | −0.094 [−0.274, 0.064] |
| G3_tied − G1_full, 8 PCs | +0.817 [0.676, 0.94] | −0.311 [−0.929, 0.083] |
| Credible groups, 8 PCs | 30 of 41 | 10 of 36 |
| Both camera views in one component (median) | 0.995 | 0.856 |
| Source-dataset model (GD_tied) − G1_full, 8 PCs | +0.944 [0.854, 1.023] | not available (no source field) |
| Pairs with val d′ < 2 | 1.8 % | 1.9 % |

**Reading:** AquaMonitor's tied-mixture gain went with its source datasets; FIN-Benthic2, a single collection, shows
no such gain, which fits that explanation but does not prove it — the datasets also differ in imaging (small crops,
about 108 × 100 px median), taxa and sample support.

## 4. Limits

- Features come from LUMI MI250X GPUs; the AquaMonitor reference was extracted on the Dell's RTX 4090. Same
  checkpoint, preprocessing and fp32; GPU-to-GPU differences are small but not zero.
- The five-split CSV is the author's derived protocol, not a dataset-publisher split. Fold 0 only.
- Three taxa have too few specimens (*Capnopsis schilleri* 15 train / 4 val, *Hydropsyche saxonica* 12 / 1,
  *Nemoura cinerea* 12 / 3); at 16 PCs a few more are not fitted. Validation is small for several taxa.
- No source, site or date fields exist; crop size is only a proxy. Life stage is read from images, not counted.
- 8- and 16-PC densities say nothing about normality in the full 1024 dimensions.

## 5. Rerun

```bash
cd ~/Documents/Feature-Geometry
scripts/local/finbenthic2_geometry.sh cohort pack            # Dell
# copy finbenthic2_images.h5 + cohort_hdf5.parquet to /flash/project_462001534/feature_geometry/finbenthic2, then on LUMI:
sbatch scripts/lumi/finbenthic2_extract.sh CODE_DIR /flash/project_462001534/feature_geometry/finbenthic2 OUT_ROOT
# copy OUT_ROOT/<job>/features_gpu/dinov3_l to $DATA/finbenthic2_fold0/features_lumi/, then:
scripts/local/finbenthic2_geometry.sh verify geometry montage metadata compare
```

## 6. Decisions for the author

- Whether to label life stage for the three riffle-beetle taxa (or follow AquaMonitor's adult/larva split) before
  further geometry work, since those splits dominate the "credible groups".
- Whether to run folds 1–4 (the packed HDF5 on LUMI already holds every image; extraction of the remaining images is
  a few minutes per fold on one node), and the order of BIOSCAN-5M and DF24.

## 7. Code and tests

New: `src/data/file_images.py`, `src/data/image_source.py`, `src/data/finbenthic2_cohort.py`,
`scripts/build_finbenthic2_cohort.py`, `scripts/pack_finbenthic2_hdf5.py`, `scripts/compare_cache_prefix.py`,
`scripts/compare_datasets_geometry.py`, `scripts/lumi/finbenthic2_extract.sh`, `scripts/local/finbenthic2_geometry.sh`,
`tests/test_finbenthic2_cohort.py`. Changed: `extract_chunks` takes a dataset factory (HDF5 or files);
`select_specimens` works without a source field; `class_geometry_pilot.py --source-column none`;
`component_metadata.py --fields/--numeric`; montage reads from HDF5 or files. Full suite: 54 passed.
