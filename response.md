# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 9910612, "camera-conditioned Gaussian geometry pilot" (local only, during the LUMI/Roihu outage).
**Status: done, stopped for research review.** Fold0 test was not used. No classifier was trained, nothing ran on LUMI, Roihu or any remote GPU, and nothing was downloaded except the CUDA build of torch (section 4).

**Previous task** (`task.md` 9d732ba / 8ac12a7, LUMI smoke test of the three baselines): unchanged and still not submitted. LUMI access is closed because of a Linux kernel vulnerability, with no reopening date. The full earlier response is at commit 66f79a4.

## 1. Answer in brief

Encoder DINOv3-L; 41 classes; 30 train and up to 15 val specimens per class; one mean embedding per specimen and camera.

- **The two cameras are two views of the same specimens, not two populations.** 41,962 of 42,165 benchmark specimens are imaged by both cameras, nearly always in the same imaging run. Site, year, lake and dataset are therefore identical for the two views of a specimen, and cannot confound the camera comparison.
- **There is a camera offset, but it is small.** In native 1024-d space the gap between the two camera means is about a third of the spread between specimens (median ratio 0.29 on train). The gap is detectable in 29 of 41 classes on train (paired sign-flip test, p < 0.05), and its direction partly recurs on the independent val specimens.
- **Separate Gaussians per camera do not predict held-out specimens better than one pooled Gaussian.** The camera mixture gains nothing at 8 PCs, a little at 16 PCs, and loses with full covariance. The camera-conditional model, which is told each observation's camera, is worse than the pooled Gaussian in all three settings.
- **One Gaussian or several:**
  - with diagonal covariance, a 2-component mixture clearly beats one Gaussian;
  - with full covariance (8 PCs), one Gaussian beats the 2- and 3-component mixtures, and also beats the diagonal 2-component mixture.

  So the diagonal-mixture gain most plausibly comes from correlations a diagonal Gaussian cannot represent, not from separate groups. The mixture components do not correspond to the cameras: they agree with camera labels at chance level, and the two views of a specimen usually fall in the same component.
- **In this pilot, the best of the tested models is one full-covariance Gaussian per class with the cameras pooled** (8 PCs). This is exploratory, from one encoder, one fold and small classes (section 7).

## 2. Gate 1: camera and sampling inventory (metadata, whole benchmark)

[evidence/inventory.json](docs/evidence/camera_pilot/inventory.json) and [class_camera_counts.csv](docs/evidence/camera_pilot/class_camera_counts.csv), from `scripts/camera_inventory.py`.

**Cameras:**

| Camera | Images | Specimens |
|---|---|---|
| 1 | 1,458,147 | 42,107 |
| 2 | 1,249,876 | 42,020 |

**Fold0 specimens by camera:**

| Split | Camera 1 | Camera 2 |
|---|---|---|
| train | 29,299 | 29,238 |
| val | 4,381 | 4,371 |
| test | 8,427 | 8,411 |

Test is counted from metadata only; no test images were read.

**Pairing:**
- 41,962 specimens are imaged by both cameras.
- 43,577 of 43,792 imaging runs contain both cameras.
- No specimen is in more than one class or split.

**Confounding:**
- Both cameras appear at every site (50 of 50) and every lake (22 of 22), in both years and in every dataset (`inventory.json`, per-camera counts).
- Camera is a within-specimen factor, so site and year cannot explain a camera difference.

**Eligibility** (pre-registered minimum: 15 train and 5 val specimens per class and camera):
- 82 of 84 class–camera pairs qualify.
- **Sisyra** is excluded as *insufficient evidence*: it has 37 train specimens but only 3 val specimens per camera.
- **41 classes** are used.

## 3. Pilot cohort (precommitted rule, capping inside official splits)

[cohort_summary.json](docs/evidence/camera_pilot/cohort_summary.json), built by `scripts/build_camera_cohort.py` with `src/data/camera_cohort.py`.

**Selection rule:**
- Only fold0 train and val are used.
- Only specimens imaged by both cameras are used, so each specimen gives one observation per camera.
- Per class, specimens are ranked by `sha1(seed:individual)` and the first 30 (train) or 15 (val) are kept.
- Per specimen and camera, frames are ranked by `sha1(seed:img)` and the first 3 are kept.
- Seed 20261107. Nothing is duplicated, and no train specimen is in val (checked).

**Size:**
- 10,578 images from 1,764 specimens.
- Train: 7,374 images, 1,230 specimens (30 per class).
- Val: 3,204 images, 534 specimens. 14 classes have fewer than 15 val specimens, and every class has at least 5.
- 3,522 specimen–camera groups have 3 frames and 6 have 2.

**Unit of analysis:** one mean embedding per specimen and camera. This gives 2,460 train and 1,068 val observations, two per specimen. Frames are never treated as independent samples. Checked in the cohort: every specimen has a single site, year, lake and dataset (1,764 of 1,764), and 1,695 specimens have the same imaging-run set for both cameras.

## 4. Features: DINOv3-L, official v1 preprocessing, local GPU

**Encoder:** `facebook/dinov3-vitl16-pretrain-lvd1689m`, `pooler_output`, 1024-d, Resize((256, 256)) with ImageNet normalisation. Images come from the local HDF5, matched by name.

**Environment:** a new **`.venv-gpu`** with the same pinned packages and torch 2.10.0+cu128 / torchvision 0.25.0+cu128, on the Dell's RTX 4090. The CPU `.venv` is kept for tests.

**Compute:**

| Run | Wall time | Throughput |
|---|---|---|
| GPU extraction, 10,578 images | 3 min 12 s | 56–58 images/s per chunk |
| CPU extraction, first chunk of 2,048 images (stopped once the GPU environment existed) | 597 s | 3.4 images/s |

- GPU extraction peak GPU memory: 1,683,987,456 bytes ([extract_gpu.log](docs/evidence/camera_pilot/extract_gpu.log)).
- Cache verification: PASS, 10,578 of 10,578 rows.

**TF32 found and switched off:**
- The first GPU run differed from the CPU features by up to 0.0138 per coordinate (mean 7.0e-4). Cause: PyTorch lets cuDNN convolutions use TF32 by default.
- `extract_features.py` now disables TF32 and records `"tf32": false`. After this, GPU and CPU differ by at most 3.1e-5 (mean 2.5e-6), which is ordinary fp32 rounding ([features_gpu_vs_cpu.json](docs/evidence/camera_pilot/features_gpu_vs_cpu.json); the TF32 run is kept as `features_gpu_tf32_vs_cpu.json`).
- This has no effect on LUMI: MI250X GPUs have no TF32.

**Analysis compute:** 25 s wall (`analyse_time.txt`). A rerun gave identical results.

## 5. Density comparison on held-out val specimens

[results.json](docs/evidence/camera_pilot/results.json), [per_class_density.csv](docs/evidence/camera_pilot/per_class_density.csv), from `scripts/camera_gaussian_pilot.py` with `src/statistics/class_density_models.py`.

**Setup:**
- **Projection:** PCA fitted on train observations only, whitened. 8 PCs explain 0.487 of the train variance, 16 PCs 0.605. **All densities are in this PCA space, not the native 1024-d space.**
- **Fitting:** sklearn `GaussianMixture`, `reg_covar` 1e-3, `n_init` 5, seed 20261107. K=2 needs at least 20 train specimens and K=3 at least 30, so every class qualifies for both. There were no fit failures.
- **Models:**
  - **A**: one Gaussian per class, cameras pooled;
  - **B marginal**: one Gaussian per camera, mixed with the train camera proportions, so it gives an unconditional density;
  - **B conditional**: scores each observation with its own camera's Gaussian, so it is *told the camera*;
  - **C2 / C3**: 2- or 3-component mixtures fitted without camera labels.
- **Score:** a val specimen's score is the mean log density of its two camera observations. A class's score is the mean over its val specimens, and the overall score is the mean over the 41 classes.
- **Uncertainty:** 95 % intervals come from 2,000 bootstrap resamples of val specimens within each class. They do **not** include the variability of refitting on other train specimens.

**Difference from A_pooled (overall mean, 95 % CI; then classes better / CI above 0 / CI below 0, out of 41):**

| Model | diag, 8 PCs | diag, 16 PCs | full, 8 PCs |
|---|---|---|---|
| B camera marginal | +0.0004 [−0.027, 0.026]; 28 / 6 / 7 | +0.070 [0.031, 0.107]; 29 / 8 / 0 | −0.26 [−0.40, −0.14]; 7 / 0 / 10 |
| C2 (K=2) | +0.43 [0.27, 0.59]; 37 / 12 / 0 | +0.54 [0.30, 0.77]; 32 / 12 / 1 | −1.29 [−1.74, −0.89]; 9 / 2 / 13 |
| C3 (K=3) | +0.37 [0.18, 0.57]; 34 / 7 / 0 | +0.49 [0.21, 0.78]; 31 / 8 / 2 | −3.60 [−4.30, −2.96]; 0 / 0 / 33 |
| B camera conditional (sees camera) | −0.045 [−0.091, −0.004]; 17 / 2 / 2 | −0.094 [−0.165, −0.028]; 16 / 3 / 4 | −0.83 [−1.06, −0.63]; 2 / 0 / 22 |

- **Diagonal vs full at 8 PCs:**
  - A has mean held-out log density −7.92 with diagonal covariance and −7.20 with full covariance;
  - the best diagonal model, C2, reaches −7.49.

  So **one full-covariance Gaussian beats every diagonal model** at 8 PCs.
- **BIC on train** (supporting only): it prefers mixtures in 39 of 41 classes with diagonal covariance (8 PCs: C3 22, C2 17), but **A in 36 of 41** with full covariance. This agrees with the held-out result.
- **Per-class plot** (diag, 16 PCs): [density_differences_diag16.png](docs/evidence/camera_pilot/density_differences_diag16.png).

## 6. Supporting evidence

All in [camera_separation.csv](docs/evidence/camera_pilot/camera_separation.csv) and `results.json`.

**Camera offset, native space:**
- Per class, the distance between the camera means divided by the within-camera spread of specimens has median 0.29 on train and 0.40 on val. Val has fewer specimens, which inflates this ratio.
- Paired sign-flip test (camera labels swapped within specimens, 999 permutations): p < 0.05 in 29 of 41 classes on train and 13 of 41 on val.
- The direction of the camera difference partly recurs on independent specimens: the median cosine between train and val directions is 0.40.
- It is partly shared across classes: the mean cosine between classes is 0.30.

**Frames vs specimens:** within one specimen and camera, frames vary much less than specimens differ within a class and camera (median RMS distance to the mean 3.15 vs 7.61, train). Averaging frames per specimen therefore loses little of the between-specimen signal.

**Mixture components are not cameras:**
- On val, the 2-component assignment matches camera labels in 0.542, 0.545 and 0.546 of observations in the three settings; 0.5 means no association.
- The two views of a val specimen fall in the same component in 0.839, 0.795 and 0.850 of specimens.

So the components split specimens, not cameras.

**PCA plots:**
- [pca_all_classes.png](docs/evidence/camera_pilot/pca_all_classes.png): all train observations.
- [pca_by_class.png](docs/evidence/camera_pilot/pca_by_class.png): per class, with lines joining each specimen's two views.

In every class the two cameras overlap with no separate clusters, and the lines show that the two views of a specimen can be far apart in PC1/PC2.

## 7. Caveats

- Densities are in an 8- or 16-PC space that holds 49–61 % of the train variance. Nothing here is a statement about native 1024-d geometry, and no formal normality test was run.
- Classes are small: 30 train specimens, 60 observations.
- Full covariance was run only at 8 PCs, where one Gaussian has 44 parameters against 60 observations. Mixtures with tied or shrunk covariance, which would separate "correlation" from "true multimodality" more cleanly, were not tested.
- The two observations of a specimen are dependent. They are scored together, but they are fitted as two rows.
- The bootstrap intervals ignore the variability of refitting on other train specimens.
- One encoder, fold0 only, and a single regularisation value (1e-3).
- This is an exploratory pilot, not a publishable result, and makes no biological claim.

## 8. Changes and validation

**New:**
- `src/data/camera_cohort.py`
- `src/statistics/class_density_models.py`
- `scripts/camera_inventory.py`, `scripts/build_camera_cohort.py`, `scripts/camera_gaussian_pilot.py`
- `scripts/check_report_numbers.py`
- `scripts/local/camera_pilot.sh`: stages inventory, cohort, extract, compare, analyse; `ACCEL=gpu|cpu`
- tests `test_camera_cohort.py`, `test_class_density_models.py`, `test_camera_gaussian_pilot.py` (end to end, synthetic)

**Changed:**
- `src/features/chunk_store.py`: `load_cache`, which checks names, coverage and checksums, plus a test.
- `scripts/extract_features.py`: TF32 off.
- `.gitignore`: `.venv-gpu/`.

**Not changed:** no dead code was added and nothing was removed.

**Validation:**
- **32 tests pass** in both `.venv` (CPU) and `.venv-gpu`.
- The cache verifies (PASS).
- The analysis is reproducible: a rerun gave identical `results.json` and per-class table.
- The numbers in sections 1–7 were checked against the evidence files with `scripts/check_report_numbers.py`.

**Data outside git:**
- `~/Documents/Research/Feature_geometry_data/camera_pilot/` (inventory, cohort, `features_gpu`, the stopped CPU chunk, the TF32 run, analyses, logs).
- Compact copies in [docs/evidence/camera_pilot/](docs/evidence/camera_pilot/).

**Rerun:** `scripts/local/camera_pilot.sh inventory cohort extract analyse`.

## 9. Open decisions (nothing started)

1. **Is the conclusion acceptable as stated?** It reads: no support for camera-specific Gaussians, and a full-covariance single Gaussian is preferred at 8 PCs.
2. **A possible follow-up to separate correlation from multimodality:**
   - full or shrinkage covariance at 16 PCs;
   - tied-covariance mixtures;
   - a larger train cap (GPU extraction now takes minutes, e.g. 100 specimens per class).
3. **The LUMI smoke test** stays pending until LUMI reopens.
