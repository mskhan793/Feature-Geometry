# Benchmark standard — eight datasets × four frozen encoders (task.md 8538df5, extraction d57a630)

Status:
- Stage A done (task 8538df5).
- Train+val frozen features extracted and checked for all 32 combinations (task d57a630, §6).
- Experiments, classifiers and test evaluation are **not authorised**.
- Evidence: `docs/evidence/benchmark_readiness/` (extraction: `extraction_c2/`).

## 1. Datasets, splits and independent units

All manifests share one schema (`src/data/benchmark_manifests.py`): `dataset`, `version`, `source_id`, `unit_id`,
`label`, `coarse_label`, `image_ref`, `available`, `val_origin`, `derived_seed`, plus either `split` or
`fold0`–`fold4`. They are built by `scripts/benchmark/build_manifests.py`; source and manifest sha256 are in
`datasets/<dataset>.json`.

| Group | Dataset | Release used | Splits | Independent unit | Label (coarse) |
|---|---|---|---|---|---|
| Biology | FIN-Benthic2 | author's five-split CSV (sha256 1c18f058, frozen; derived, author-approved) | `fold0`–`fold4` train / val / test, unchanged | specimen (`individual`); camera-balanced mean of up to 10 frames per camera, stored selection | `taxon`, 39 |
| Biology | AquaMonitor Classification | authors' classif parquet (df18b019) + byte-verified HDF5 | official `fold0`–`fold4`, unchanged | specimen; same frame protocol | `taxon_group`, 42 |
| Biology | BIOSCAN-5M | official cropped_256 archives + v1 index of the official splits | official closed-set train / val / test; open-world partitions (key_unseen, val_unseen, test_unseen) excluded | `processid` (one image per specimen record; no repeated ids in the closed set) | species, 11846 (genus) |
| Biology | DanishFungi24 | `DanishFungi2024-train.csv` / `-pubtest.csv` | official train / pubtest (= test); **derived** val from official train (seed 20261110, frozen assignment 688564d2; observation and near-duplicate groups kept whole) | observation (`observationID`; all photos together) | `scientificName` = official `class_id`, 1604 (genus) |
| General | FGVC-Aircraft | `fgvc-aircraft-2013b` | official train / val / test (`images_variant_*.txt`) | image (content hash) | variant, 100 (family, 70) |
| General | Food-101 | `food-101.tar.gz` (2014) | official train / test; **derived** val = 10 % of official train per class (seed 20261112) | image (content hash) | class, 101 |
| General | CIFAR-100 | `cifar-100-python` (publisher MD5 eb9058c3 verified) | official train / test; **derived** val = 10 % of train per class (seed 20261112) | image (pixel hash) | fine, 100 (superclass, 20) |
| General | Oxford-IIIT Pet | `images.tar.gz` + `annotations.tar.gz` | official trainval / test; **derived** val = 10 % of trainval per breed (seed 20261112) | image (content hash) | breed, 37 (cat / dog) |

- **Derived validation:** units are drawn per class in sha1(f"{seed}:{unit}") order, the first round(0.1 × n_c)
  going to val, and units are always kept whole. Official test files are never read for it.
- **Image units = exact-content hash.** Identical images share a unit, so they are never counted twice. Found:
  - CIFAR-100: 26 duplicate groups (52 images); 10 cross the official train / test boundary, and 17 carry
    conflicting fine labels;
  - Food-101: 16 images share a unit; 2 units cross train / test, 2 carry conflicting labels;
  - Aircraft and Pet: none.
  - These are publisher-split properties: **reported, not corrected**. Near-duplicates beyond exact bytes are not
    detected.
  - The flagged images and two candidate evaluation rules (R1: also report test without test images whose content
    occurs in train/val; R2: leave conflicting-label groups out of training statistics) are in
    `evidence/benchmark_readiness/duplicates/`. They are **not applied** until the author decides.
- **Licences of the image data as used here:** DF24, per the dataset authors' README (commit c5c19b41, copy in
  `evidence/benchmark_readiness/df24_licence/`): BSD, with data, metadata and models for non-commercial research
  only. The "Public Domain vs CC BY-NC-ND" conflict noted in the earlier dataset audit concerns NAFlora-1M, not DF24.
- **Test sets are locked.** Analysis code drops test rows before any feature lookup
  (`src/statistics/heldout_density.load_fold`). Features of test images may be cached, but are read only after an
  explicit final-evaluation authorisation. Task d57a630 encoded train+val only (§6).
- **CIFAR-100 images are 32 × 32.** Each encoder upscales them with its own published transform only; no other
  resizing.

## 2. Encoders (frozen; no fine-tuning, no learned heads)

Audit records: `encoders/<name>.json` (Hugging Face metadata and config files at the pinned revision; no weights
downloaded). Registry: `src/encoders/frozen_encoders.py`.

| Encoder | Checkpoint @ revision | Licence | Feature (raw, **not** L2-normalised) | Preprocessing | Dim |
|---|---|---|---|---|---|
| DINOv3-L | `facebook/dinov3-vitl16-pretrain-lvd1689m` @ ea8dc286 | DINOv3 licence, gated (weights on LUMI and the Dell) | `pooler_output` = final-LayerNorm CLS | Resize (256, 256), ImageNet mean/std (v1 convention; the official processor's default is 224 — the existing caches are kept as they are) | 1024 |
| DINOv2-L | `facebook/dinov2-large` @ 47b73eef | Apache-2.0 | `pooler_output` = final-LayerNorm CLS | Resize 256 bicubic, CenterCrop 224, ImageNet mean/std | 1024 |
| BioCLIP 2 | `imageomics/bioclip-2` @ 2957b322 (open_clip_torch 3.3.0 hf-hub, from the add-on folder `feature_geometry/pkgs` on LUMI, not in the container; LUMI `refs/main`) | MIT | `model.visual(x)` = proj(ln_post(CLS)) | checkpoint's open_clip eval transform (224, bicubic, OpenAI CLIP mean/std) | 768 |
| SigLIP 2 | `google/siglip2-so400m-patch16-256` @ e8708ab7 (official 256 variant, preferred over 384 by the checkpoint policy) | Apache-2.0 | `SiglipVisionModel` `pooler_output` (attention-pooled, = `get_image_features`) | checkpoint's `SiglipImageProcessor` (256 × 256 bilinear, mean/std 0.5) | 1152 |

- **L2-normalisation:** a unit-normalised variant, if used later, is a separately named arm; caches always store the
  raw outputs.
- **Final-LayerNorm rank constraint:** every output vector x of a final-LayerNorm satisfies Σ_j (x_j − b_j)/w_j = 0,
  so native covariances have rank ≤ d − 1.
  - Measured on the train+val caches of all 8 datasets (task d57a630, `extraction_c2/records/`):
    - DINOv3-L and DINOv2-L have numerical rank 1023 everywhere. The smallest-eigenvalue direction is the 1/w
      direction of the final LayerNorm (|cos| = 1.0 to six decimals).
    - BioCLIP 2 (768) and SigLIP 2 (1152) have full numerical rank everywhere (a projection or an attention-pooling
      MLP follows their last norm).
  - Native-space work drops one coordinate (or projects out 1/w) per model, after this audit; nothing is ever
    inverted singular.

## 3. Feature spaces, eligibility and comparability

- **Spaces:** a train-only PCA, fitted separately for every dataset × encoder × split (fold), with **PCA-16 and
  PCA-64** as primary dimensions. If rank or support prevents a dimension, it is declared unavailable, not padded.
  Explained variance is reported. Native space follows the rank audit above.
- **Eligibility for Gaussian full-covariance analysis:** ≥ 15 train and ≥ 5 val independent units per class
  (as before). Eligible / total classes:

  | Dataset | Eligible / total |
  |---|---|
  | FIN-Benthic2 | 34–36 of 39 per fold |
  | AquaMonitor | 40–41 of 42 |
  | BIOSCAN-5M | 531 of 11846 (median 3 training specimens per species) |
  | DF24 | 889 of 1604 |
  | Aircraft | 100 of 100 |
  | Food-101 | 101 of 101 |
  | CIFAR-100 | 100 of 100 |
  | Pet | 37 of 37 |

  Excluded classes are always disclosed. BIOSCAN-5M and DF24 also get an all-classes / rare-class descriptive
  track.
- **Paired comparisons:** encoders are compared on identical unit ids and identical eligible class sets per dataset
  and fold (fixed before results), never on post-hoc reliable subsets. Real val samples are never dropped to balance
  evaluation.
- **Not comparable:** raw log-likelihoods across PCA dimensions, across datasets, or between native spaces of
  different dimension. Feature norms across encoders (no normalisation is implied). Results across datasets with
  different units (specimen, observation, image).

## 4. Metrics (descriptive first; no classifiers in this task)

- **Primary:** held-out class-conditional Gaussian log-likelihood per independent val unit. Class-macro is primary,
  micro secondary. Paired differences carry stratified bootstrap intervals over val units (conditional on the train
  fits).
- **Also reported:**
  - covariance spectra and conditioning;
  - d′ along the train-centroid direction, measured on val;
  - pairwise Bhattacharyya overlap with a stated covariance form, dimension and class-pair set;
  - the anchored paper-faithful Pándy spherical PCA-64 GBC and per-class Ledoit–Wolf full covariance, as separate
    arms.
- **No combined ranks** across metrics or dimensions.
- **Folds** are reported individually; their overlap is acknowledged.
- **Not claimed:** likelihood or overlap are not recognition quality.
- **Later, separately approved:** a small fixed classifier suite (cosine / Euclidean NCM, train-only linear probe)
  with macro-F1 / balanced accuracy on locked test; FDA; Ibrahim et al. random projection (q = 128, repeated fixed
  seeds).

## 5. Organisation and leakage guarantees

- **Code layout:**
  - `src/data` (manifests, dataset adapters, CIFAR pickle source);
  - `src/encoders` (checkpoint-specific preprocessing and extraction);
  - `src/statistics` (shared metrics);
  - `scripts/benchmark` (sources, manifests, encoder audit, readiness, packing, smoke sample);
  - `scripts/lumi` (bounded jobs).
- **Storage:** images and features stay outside Git; hashes, manifests and evidence are tracked.
- **Leakage guarantees:**
  - frozen splits are reproduced, not regenerated;
  - derived validation reads train only;
  - units are never split across partitions by our derivations;
  - test rows are dropped before feature lookup;
  - cross-split exact duplicates inherited from publishers are reported.
- **Existing DINOv3-L aquatic caches are reused,** never re-extracted without a demonstrated mismatch.

## 6. Extraction protocol (verified in task d57a630)

- **What is encoded:**
  - single-split datasets: the train and val rows of the standardised manifest (official train / val, or the
    derived val), never test;
  - FIN-Benthic2 / AquaMonitor: the stored frame selection, i.e. every specimen. Each specimen is test in one fold
    and train/val in the other four, so the train+val union over the folds is the whole selection. Test rows are
    dropped per fold before feature lookup.
- **Inputs:** byte-verified HDF5 packs on `/flash` (images in manifest order, `img` + `hdf5_row` checked on read);
  CIFAR-100 from the publisher's pickles.
- **GPU job** (`scripts/lumi/extract_encoder.sh`):
  - one standard-g node, 8 tasks with one GCD each;
  - 8,192-row chunks with names and sha256 in the chunk JSON; resumable; float32 raw outputs;
  - no TF32, no mixed precision.
- **CPU follow-up** (`scripts/lumi/check_extraction.sh`, after success only):
  - full cache verification against the manifest;
  - CPU re-extraction of the first 256 rows, compared with the GPU cache.
- **Record** (`scripts/benchmark/combination_record.py`, one per combination): identity against the standardised
  manifest, units, finiteness, checksum in manifest order, numerical rank and the LayerNorm constraint.
- **Reuse rule:** an earlier cache is reused only if a predeclared fresh-vs-cached comparison passes (relative L2
  ≤ 1e−4 and cosine ≥ 0.999999 on a fixed 512-image sample, same checkpoint, transform and dimension).
  - Passed: DINOv3-L and BioCLIP 2 BIOSCAN-5M (v1).
  - Failed, so re-extracted: DINOv2-L BIOSCAN-5M (v1), max relative L2 4.9e−4.
  - The v1 BIOSCAN caches also hold test and open-world folders; only `train` and `val` are opened.
  - The DINOv3-L FIN-Benthic2 / AquaMonitor caches are the verified full-specimen caches of task 92a605a.
