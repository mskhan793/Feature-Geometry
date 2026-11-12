# Benchmark standard — eight datasets × four frozen encoders (task.md 8538df5)

Status: Stage A done; one Stage B smoke test done; Stage C (bulk extraction, experiments, test evaluation) **not
authorised**. Evidence: `docs/evidence/benchmark_readiness/`.

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
- **Test sets are locked.** Analysis code drops test rows before any feature lookup
  (`src/statistics/heldout_density.load_fold`). Features of test images may be cached, but are read only after an
  explicit final-evaluation authorisation.
- **CIFAR-100 images are 32 × 32.** Each encoder upscales them with its own published transform only; no other
  resizing.

## 2. Encoders (frozen; no fine-tuning, no learned heads)

Audit records: `encoders/<name>.json` (Hugging Face metadata and config files at the pinned revision; no weights
downloaded). Registry: `src/encoders/frozen_encoders.py`.

| Encoder | Checkpoint @ revision | Licence | Feature (raw, **not** L2-normalised) | Preprocessing | Dim |
|---|---|---|---|---|---|
| DINOv3-L | `facebook/dinov3-vitl16-pretrain-lvd1689m` @ ea8dc286 | DINOv3 licence, gated (weights on LUMI and the Dell) | `pooler_output` = final-LayerNorm CLS | Resize (256, 256), ImageNet mean/std (v1 convention; the official processor's default is 224 — the existing caches are kept as they are) | 1024 |
| DINOv2-L | `facebook/dinov2-large` @ 47b73eef | Apache-2.0 | `pooler_output` = final-LayerNorm CLS | Resize 256 bicubic, CenterCrop 224, ImageNet mean/std | 1024 |
| BioCLIP 2 | `imageomics/bioclip-2` @ 2957b322 (open_clip hf-hub; LUMI `refs/main`) | MIT | `model.visual(x)` = proj(ln_post(CLS)) | checkpoint's open_clip eval transform (224, bicubic, OpenAI CLIP mean/std) | 768 |
| SigLIP 2 | `google/siglip2-so400m-patch16-256` @ e8708ab7 (official 256 variant, preferred over 384 by the checkpoint policy) | Apache-2.0 | `SiglipVisionModel` `pooler_output` (attention-pooled, = `get_image_features`) | checkpoint's `SiglipImageProcessor` (256 × 256 bilinear, mean/std 0.5) | 1152 |

- **L2-normalisation:** a unit-normalised variant, if used later, is a separately named arm; caches always store the
  raw outputs.
- **Final-LayerNorm rank constraint:** every output vector x of a final-LayerNorm satisfies Σ_j (x_j − b_j)/w_j = 0,
  so native covariances have rank ≤ d − 1.
  - Measured for DINOv3-L (rank 1023; `coupled_rscm_vs_lw/native_null_space/`).
  - Confirmed for DINOv2-L on CIFAR-100 (relative violation ≤ 9.3e−7; Stage B). Its rank could not be measured
    there, because 512 samples < 1024 dimensions.
  - BioCLIP 2 and SigLIP 2 are not expected to have an exact deficit (a projection or an MLP follows their last
    norm). This must be measured.
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
