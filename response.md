# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 92a605a, "Full-specimen FIN-Benthic2 and AquaMonitor geometry on LUMI".
**Status: preflight done; feature extraction submitted on LUMI (section 3).** CPU geometry stage not yet run. The
previous response (FIN-Benthic2 fold 0, task 366ee62) is in commit 6c4c515 and `docs/evidence/finbenthic2_geometry/`
(unchanged; used as the baseline).

## 1. Preflight (2026-11-11)

**Selection rule (one global, fold-independent selection per dataset).** Every eligible specimen, up to 10 frames per
specimen and camera, spread over imaging runs and capture time (`select_frames`, the pilots' rule). The rule uses a
specimen's own images only, so the same images serve every fold and are encoded once. Check: the earlier fold-0
cohorts are exact subsets (FIN-Benthic2 128,937 of 128,937 images; AquaMonitor 196,983 of 196,983).
Script `scripts/build_full_specimen_selection.py` (metadata only, Dell); evidence `docs/evidence/*_full_geometry/selection/`.

| | FIN-Benthic2 (derived five-split CSV, sha256 `1c18f058…`) | AquaMonitor Classification (official folds, manifest `d6e63bb6…`) |
|---|---|---|
| Eligible images / specimens / classes | 460,004 / 9,631 / 39 | 2,708,023 / 42,165 / 42 |
| Specimens per fold train / val / test | 6,160–6,165 / 1,540–1,543 / 1,924–1,928 | 29,346 / 4,386 / 8,433 (all folds) |
| Fold checks | no specimen in two partitions; each specimen test in exactly one fold | same |
| Selected images (all specimens) | 186,284 | 836,159 |
| Single-camera specimens | 291 | 203 |
| Train specimens per class (min / median / max, over folds) | 8 / 143 / 567 | 31 / 129 / 7,436 |
| Val specimens per class (min / median / max) | 1 / 37 / 150 | 3 / 19.5 / 1,122 |
| Classes below 15 train in some fold / below 5 val in some fold | 3 / 5 | 0 / 4 |
| Reused LUMI features | 128,937 (fold-0 job 23623689, same hardware) | none on LUMI |
| New images to encode | 57,347 | 836,159 |
| Feature storage (float32, 1024-d) | 0.76 GB | 3.42 GB |

Classes below the support rule (15 train / 5 val specimens) in a fold are listed as insufficient for that fold, not
dropped silently.

**AquaMonitor reuse decision.** The earlier AquaMonitor features (196,983 images) were extracted on the Dell RTX 4090.
To keep one hardware per dataset cache, all 836,159 images are encoded on LUMI; the Dell features are used only to
measure the LUMI-vs-Dell difference on the overlap (0.8 GB copy), not mixed into the cache.

**LUMI state (read 14:48).** Login OK; no jobs queued. Project 462001534: GPU 39,623 / 40,000 GPU-h used (377 left),
CPU 2,848 / 5,000 core-h used (2,152 left); `/flash` 122 GiB of 2 TiB, `/scratch` 5.1 of 50 TiB. Present and readable
paths: AquaMonitor HDF5 `/flash/project_462001534/data_hdf_aqua/aquamonitor_complete.h5` (verified byte-identical to
the authors' release, job 23526542), FIN-Benthic2 HDF5 `/flash/project_462001534/feature_geometry/finbenthic2/`
(sha256 `e23e2433…`), fold-0 cache (16 chunks), DINOv3-L weights in the HF cache.

**Resource bound (proposed; stop and report if exceeded).**

| Step | Where | Bound | Expected |
|---|---|---|---|
| FIN-Benthic2 extraction, 57,347 images | dev-g, 1 node, 8 GCDs, 30 min | ≤ 2 GPU-h | ≈ 0.7 GPU-h (fold-0 rate: 128,937 images in 4 min incl. model load) |
| AquaMonitor extraction, 836,159 images | dev-g, 1 node, 8 GCDs, 1 h 30 | ≤ 6 GPU-h | unmeasured for AquaMonitor images on LUMI; first chunk times reported; resumable |
| CPU geometry: 2 datasets × 5 folds + 2 descriptive runs | small, 32 cores, ≤ 1 h each | ≤ 400 core-h | Dell pilots: 138–149 s per fold-0 cohort on 32 cores; uncapped AquaMonitor ~5× more train specimens |
| Total | project 462001534 | ≤ 8 GPU-h, ≤ 400 core-h | |

Peak RAM: one fold's AquaMonitor image features ≈ 2.7 GB (float32); 1024 × 1024 per-class covariances 8 MB each
(float64). The camera re-identification step in `class_geometry_pilot.py` builds an n × n × 1024 array (≈ 450 GB for
the largest AquaMonitor class); it will be replaced by a chunked computation before the CPU stage.

**Descriptive output (A).** Per the task's preference, (A) uses the union of fold-0 train and validation specimens
(FIN-Benthic2 7,707, AquaMonitor 33,732), so fold-0 test specimens stay unread; it is in-sample and labelled
exploratory.

## 2. Code (this step)

- `scripts/build_full_specimen_selection.py`, `src/data/full_specimen.py` (fold checks), `tests/test_full_specimen.py`
  (4 tests).
- `scripts/lumi/finbenthic2_extract.sh` renamed to `scripts/lumi/extract_dinov3_l.sh` and made dataset-neutral
  (`CODE MANIFEST HDF5 OUT_DIR`, resumable).

## 3. LUMI jobs

| Job | What | Resources | Elapsed | Result |
|---|---|---|---|---|
| 23642052 | FIN-Benthic2 extraction, 57,347 new images (code d072185) | dev-g, 1 node, 8 GCDs | 7:34 | cache covers 57,347 / 57,347 rows (8 chunks); CPU vs GPU on 256 images max abs diff 1.8e-5, mean 2.2e-6 |
| 23642060 | AquaMonitor extraction, 836,159 images (code d072185) | standard-g, 1 node, 8 GCDs | 20:25 | cache covers 836,159 / 836,159 rows (103 chunks); 138 img/s per GCD; CPU vs GPU max abs diff 2.1e-5, mean 2.1e-6 |
| 23644988 | FIN-Benthic2 five-fold geometry (code 2844e9b) | small, 5 × 8 cores, 45 min | — | hung in the job setup with no output (0 CPU used); being diagnosed |

GPU use so far: about 0.5 + 1.4 = 1.9 GPU-h (GCD-hours / 2), project 462001534.

Features: FIN-Benthic2 = fold-0 cache (job 23623689, 128,937 images) + this job (57,347), both on LUMI MI250X;
AquaMonitor = this job only. Paths: `/scratch/project_462001534/feature_geometry/full_specimen/<dataset>/extract/`.
