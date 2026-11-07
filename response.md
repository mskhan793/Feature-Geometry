# Coding assistant response — Feature-Geometry

**Current task:** `task.md` fb46e1f, "expand specimen diversity in AquaMonitor geometry". This is the same investigation as fb26afc, on a larger cohort.
**Status: done, stopped for the author's review.**
- **Ran:** one precommitted local run on the Dell's RTX 4090, with DINOv3-L only.
- **Not done:** no test data, no classifier, nothing remote, no new weights or data.

**Previous results:**
- 30-specimen geometry: response at commit 8449837, evidence in `docs/evidence/class_geometry/`.
- Camera pilot: response at commit e345d91.
- LUMI smoke test: still pending; LUMI is closed.

## 1. Answer in brief

- **With up to 200 train specimens per class, one Gaussian per class is clearly not enough.**
  - Tied-covariance mixtures predict held-out val specimens better than one full Gaussian: K=2 +0.577 [0.505, 0.646] at 8 PCs and +0.298 [0.046, 0.509] at 16 PCs.
  - Under the predeclared rule, 30 of 41 classes are "credible multiple groups" at 8 PCs and 21 at 16 PCs.
  - At 30 specimens this was 1 class; most classes were then undecidable.
- **The extra structure is mainly source-dataset (acquisition) structure, not demonstrated biological subtypes.** A model with no free grouping, one Gaussian per source `dataset` with a shared covariance and scored without the val specimen's label, predicts val **better than any unsupervised mixture**: +0.944 [0.854, 1.023] at 8 PCs and +1.065 [0.849, 1.252] at 16 PCs.
  - The 2-group split reproduces the dataset label (ARI ≥ 0.5) in 18 of 40 classes on train and 13 of 39 on val. It reproduces year in 12 / 6 and site in 0 / 0.
  - The images show a purple background for the **dna** dataset, blue for biomass and bulk, and in some classes larger, closer-framed specimens in dna/biomass.
- **Full-covariance mixtures do not help:**
  - K=2 full: −0.011 [−0.637, 0.563] at 8 PCs and −2.905 at 16 PCs;
  - K=3 full is worse in both.

  So the gain needs a shared covariance. The classes look like **one covariance shape with shifted centres**: a source offset or a few modes, not differently shaped clusters.
- **Between classes, the picture is stable.**
  - 15 of 820 pairs have d' < 2 on val (22 at 30 specimens).
  - The closest pairs are still closely related taxa: Cyrnus trimaculatus / Polycentropus flavomaculatus 0.87, Lepidostoma hirtum / Limnephilidae 1.35, Psychomyia / Tinodes 1.42.
  - Old and new class centroids differ only by sampling noise, and principal directions agree as well as a 30-specimen sample of the new data agrees with all of it.
- **Camera does not drive the groups:** both views of a specimen fall in the same component in a median 0.995 of specimens at 8 PCs.

## 2. (a) Cohort: counts and source diversity

Rule committed before extraction (commit cfbe5ba, `src/data/expanded_cohort.py`), evidence [cohort_audit.json](docs/evidence/expanded_geometry/cohort_audit.json) and [per_class_counts.csv](docs/evidence/expanded_geometry/per_class_counts.csv):

**Selection rule:**
- **Train:** per class, all fold0 train specimens if at most 200. Otherwise a source-aware allocation over `dataset`:
  - every source first gets min(available, 20);
  - the rest is shared in proportion to remaining availability;
  - within a source, specimens are taken in `sha1(seed:individual)` order (seed 20261107).
- **Val:** every fold0 val specimen.
- **Single-camera specimens are kept;** their embedding is the mean of the camera they have.
- Test is never selected.

**Counts:**

| Item | Value |
|---|---|
| Train specimens selected / available | 5,598 / 29,346 |
| Val specimens (all) | 4,386 |
| Classes with at least 50 / 100 / 200 train specimens | 36 / 26 / 17 of 42 (all are selected up to the cap) |
| Train specimens per analysed class (min / median / max) | 36 / 135 / 200 |
| Val specimens per analysed class (min / median / max) | 6 / 21 / 1,100 |
| Specimens with one camera | train 38, val 20 (kept) |
| Not analysed | Sisyra: 37 train, 3 val (below the 5-val minimum) |

**Source composition** (share of specimens):

| Source | Full train | Selected train | Val |
|---|---|---|---|
| bulk22 | 0.4709 | 0.3975 | 0.4708 |
| bulk21 | 0.4569 | 0.3901 | 0.4569 |
| biomass | 0.0362 | 0.1188 | 0.0363 |
| dna | 0.0359 | 0.0936 | 0.036 |

The floor raises biomass and dna in train by design (no specimen is copied). This cohort is descriptive and says nothing about prevalence. Val keeps the natural composition. Train classes span 1 (1 class), 2 (5), 3 (18) or 4 (17) sources.

## 3. (b) Frames, resources, reuse

**Frames available** (the audit before extraction):
- per specimen and camera: median 27 (25–75 % range 19–39, min 1, max 547);
- imaging runs per specimen and camera: median 1, max 6; 1,467 specimen-cameras have more than one run.
- Frames are burst captures about 24 ms apart, so consecutive frames are near-duplicates.

**Frame selection:**
- Up to 10 per specimen and camera, dealt round-robin over runs and evenly spaced in capture order within a run.
- Selected: median 10; 668 specimen-cameras have fewer than 10 (all of theirs used).
- **Effect:** frames vary more within a specimen than with the earlier 3 hash-chosen frames. The median frame RMS distance is 3.58, against 3.15 in the camera pilot.

**Images:**
- 196,983 in the cohort (train 109,972, val 87,011).
- **4,491 reused** by exact image name from the camera-pilot cache, which has the same checkpoint, transform and TF32 off; the lookup checks this.
- **192,492 newly extracted.**

**Resources, estimate vs actual:**

| | Estimate | Actual |
|---|---|---|
| GPU time | 56.9 min | 55 min 44 s, 57.2–57.7 images/s |
| Feature storage | 0.79 GB | — |

- Peak GPU memory: 1,683,987,456 bytes.
- Cache verification: PASS, 192,492 of 192,492 rows ([features_verify.json](docs/evidence/expanded_geometry/features_verify.json), [extract_gpu.log](docs/evidence/expanded_geometry/extract_gpu.log)).
- Geometry stage: 150 s wall on CPU with 16 workers ([geometry_time.txt](docs/evidence/expanded_geometry/geometry_time.txt)).
- There were no failures.

## 4. (c) Within-class and between-class geometry

Tables: [per_class_geometry.csv](docs/evidence/expanded_geometry/per_class_geometry.csv) and the compact [per_class_summary.csv](docs/evidence/expanded_geometry/per_class_summary.csv) (one row per class and PCA dimension).

**Within class** (native 1024-d, train; min / median / max over 41 classes):

| Indicator | min | median | max |
|---|---|---|---|
| first-direction share | 0.156 | 0.246 | 0.485 |
| effective rank, all train specimens | 8.22 | 24.42 | 39.78 |
| effective rank, first 30 specimens (comparable across classes and with the old run) | 7.69 | 13.84 | 17.49 |
| camera-view re-identification | 0.065 | 0.279 | 0.533 |

Re-identification is measured among up to 200 specimens now, so chance is lower than before.

**Plots:** [eigenspectra.png](docs/evidence/expanded_geometry/eigenspectra.png), [class_internal_geometry.png](docs/evidence/expanded_geometry/class_internal_geometry.png) (five classes chosen by train BIC at 16 PCs).

**Images:** observations are hypotheses only.

| Class | Montage | Observation |
|---|---|---|
| Lepidostoma hirtum | [montage](docs/evidence/expanded_geometry/montage_Lepidostoma_hirtum.png) | One group is bulk22 (small larvae, teal background). The other is dna/biomass (purple or dark background, larger and closer-framed larvae). |
| Oulimnius tuberculatus adult | [montage](docs/evidence/expanded_geometry/montage_Oulimnius_tuberculatus_adult.png) | The groups are biomass (blue, plate–well IDs) vs dna (purple). In the class plot they are separate clouds. |
| Oulimnius tuberculatus, Limnephilus, Erpobdella octoculata | [O. tuberculatus](docs/evidence/expanded_geometry/montage_Oulimnius_tuberculatus.png), [Limnephilus](docs/evidence/expanded_geometry/montage_Limnephilus.png), [Erpobdella](docs/evidence/expanded_geometry/montage_Erpobdella_octoculata.png) | Limnephilus and Erpobdella are one diffuse cloud ("inconclusive" at 16 PCs). |

**Between classes:**
- Tables: [class_pairs.csv](docs/evidence/expanded_geometry/class_pairs.csv), [nearest_class.csv](docs/evidence/expanded_geometry/nearest_class.csv).
- Plots: [class_pair_separation.png](docs/evidence/expanded_geometry/class_pair_separation.png), [closest_pairs.png](docs/evidence/expanded_geometry/closest_pairs.png).
- d' (val, train centroid direction) is below 2 for 15 pairs and below 1 for 1 pair.
- The radius ratio is below 1 for 156 pairs: that is high-dimensional geometry, not overlap.
- The 2-D PCA (0.321 of the variance) shows a larger separation than native space for 707 of 820 pairs.
- L2 normalisation hardly changes the ordering (Spearman 0.9997).

## 5. (d) One Gaussian or several (held-out val specimens)

**Setup:**
- Global train PCA, whitened: 8 PCs explain 0.583 of the variance, 16 PCs 0.699. Results apply to these spaces only.
- 32 PCs was not run: not needed, and too many parameters for the smallest classes.
- Fitting: float64, `reg_covar` 1e-3, multi-start fitting. A model is eligible if its parameters are at most n·d/2. It is unreliable if a component has fewer than d+1 specimens.
- **Verdict rule as predeclared:**
  - **credible multiple groups:** some full/tied mixture's interval is above 0 AND its split is stable on val;
  - **one Gaussian favoured:** every interval is below 0;
  - **inconclusive:** otherwise.

**Difference to one full Gaussian, mean log density per val specimen** (class mean, 95 % bootstrap over val specimens; classes better / CI above 0 / CI below 0):

| Model | 8 PCs | 16 PCs |
|---|---|---|
| K=2 tied | +0.577 [0.505, 0.646]; 38 / 30 / 0 | +0.298 [0.046, 0.509]; 30 / 25 / 3 |
| K=3 tied | +0.817 [0.676, 0.94]; 36 / 30 / 0 | +0.779 [0.546, 0.992]; 31 / 27 / 1 |
| K=2 full | −0.011 [−0.637, 0.563]; 32 / 23 / 1 | −2.905 [−4.893, −1.435] (39 classes); 21 / 17 / 9 |
| K=3 full | −1.004 [−2.097, −0.007]; 27 / 23 / 1 | −3.915 [−5.633, −2.426] (36 classes); 18 / 7 / 11 |
| **GD_tied, one Gaussian per source dataset** (sensitivity, not in the verdict) | **+0.944 [0.854, 1.023]; 38 / 33 / 1** | **+1.065 [0.849, 1.252]; 35 / 29 / 0** |
| diag K=2 vs diag K=1 (diagnostic) | +1.31 [1.146, 1.481] | +1.762 [1.518, 2.003] |

**Fit status (ok / unreliable):**

| Model | 8 PCs | 16 PCs |
|---|---|---|
| K=2 full | 39 / 2 | 32 / 7, plus 2 ineligible |
| K=3 full | 30 / 11 | 22 / 14, plus 5 ineligible |
| K=2 tied | 38 / 3 | 33 / 8 |
| K=3 tied | 33 / 8 | 25 / 16 |

Largest condition numbers: one Gaussian 321 (8 PCs) / 684 (16 PCs); K=2 full 1931 / 3665.

**Split stability:** the tied split is stable on val in 35 classes (8 PCs) and 29 classes (16 PCs), with median ARI 1.0.

**Verdicts:**

| Verdict | 8 PCs | 16 PCs |
|---|---|---|
| credible multiple groups | 30 | 21 |
| inconclusive | 10 | 12 |
| insufficient evidence | 1 | 5 |
| one Gaussian favoured | 0 | 3 |

Per-class lists are in [results.json](docs/evidence/expanded_geometry/results.json) and [per_class_modality.csv](docs/evidence/expanded_geometry/per_class_modality.csv).

## 6. (e) Sensitivity to source dataset; caveats

**Dataset sensitivity** ([component_metadata.json](docs/evidence/expanded_geometry/component_metadata.json)):
- **ARI ≥ 0.5 between the 2-group split and metadata:**

  | Field | 8 PCs train | 8 PCs val | 16 PCs train | 16 PCs val |
  |---|---|---|---|---|
  | dataset | 18 / 40 | 13 / 39 | 15 / 40 | 13 / 39 |
  | year | 12 / 39 | 6 / 37 | 12 / 39 | 10 / 37 |
  | site | 0 / 40 | 0 / 40 | — | — |

- Among the 30 "credible" classes at 8 PCs, 14 have a dataset-aligned split.
- Even where the unsupervised split does not follow dataset exactly, the dataset model GD_tied explains held-out density better on average (section 5).

**Caveats:**
- **GD_tied is not a fair parameter match.** It has up to 4 means (71 parameters at 8 PCs vs 53 for K=2 tied and 62 for K=3 tied). Its means are not tuned for likelihood. Its mixing weights come from the source-balanced train cohort, which disagrees with val's natural composition, and that works *against* it.
- **Source cannot be separated from its parts.** It bundles imaging setup (background, framing), year (bulk21 = 2021, bulk22 = 2022) and the sampling of specimens. "Acquisition" vs "biology" cannot be told apart here. A biological difference between sub-datasets, such as life stage or size, cannot be excluded: Lepidostoma hirtum's dna/biomass larvae look larger.
- **Spaces and the verdict rule.** Densities are in an 8- or 16-PC space (58–70 % of the variance), and a stable split is not proof of distinct populations. The verdict thresholds are predeclared but arbitrary. Montage classes were picked by train BIC.
- **Val sizes vary widely** (6 to 1,100 per class), so per-class intervals differ greatly in width.
- **Scope:** one encoder and fold0 only; exploratory.

## 7. (f) Comparison with the 30-specimen study

[compare/compare.json](docs/evidence/expanded_geometry/compare/compare.json), [verdicts.csv](docs/evidence/expanded_geometry/compare/verdicts.csv), [closest_pairs_old_vs_new.csv](docs/evidence/expanded_geometry/compare/closest_pairs_old_vs_new.csv), [class_stability.csv](docs/evidence/expanded_geometry/compare/class_stability.csv).

**Verdicts, old (8 PCs, 30 specimens) → new (8 PCs):**

| Old verdict | New: credible | New: inconclusive | New: insufficient |
|---|---|---|---|
| one Gaussian favoured (10) | 7 | 3 | 0 |
| inconclusive (13) | 10 | 3 | 0 |
| insufficient evidence (17) | 12 | 4 | 1 |
| credible (1, Caenis rivulorum) | 1 | 0 | 0 |

The old "one Gaussian favoured" calls were a small-sample effect.

**Pairs:**
- d' on val correlates 0.936 (Spearman) between runs, and 6 of the old top-10 closest pairs remain in the top 10.
- Caenis horaria / C. luctuosa moved from 0.89 to 1.65, now on 417 and 52 val specimens.
- Cyrnus trimaculatus / Polycentropus flavomaculatus moved from 2.00 to 0.87, now the closest pair.

**Stability:**
- Old-to-new centroid shift divided by its sampling reference: median 0.761, maximum 1.663, none above 2.
- Top-5 principal-subspace overlap old vs new: median 0.713, which matches the overlap between 30 new specimens and all new specimens (0.731).
- So the 30-specimen geometry differed from the larger one only by sampling noise.

**Other comparisons:**
- Dataset-aligned splits (8 PCs, train): 10 → 18 classes, 8 in both.
- Effective rank on 30 specimens: median 14.72 (old) vs 13.84 (new).

## 8. (g) Failures, changes, validation

**Failures:** no job failed and nothing was unavailable. One reporting bug of mine, the verdict-transition table being transposed, was caught and fixed before this report.

**Changes:**
- **New:**
  - `src/data/expanded_cohort.py` (selection rule) and `src/features/feature_lookup.py` (features by name from several verified caches);
  - `SourceGaussians` in `src/statistics/class_modality.py`;
  - `scripts/build_expanded_cohort.py` and `scripts/compare_geometry_runs.py`;
  - `scripts/local/expanded_geometry.sh` with stages cohort, extract, verify, geometry, montage, metadata, compare;
  - tests for the selection rule, `SourceGaussians`, and an extended end-to-end test: two caches, a single-camera specimen, a class below the support minimum, and a planted dataset-aligned split that must be "credible" and explained by GD_tied.
- **Changed:**
  - `scripts/class_geometry_pilot.py`: `--caches`, `--pcs 8 16`, single-camera specimens, n = 30 ranks, the GD_tied sensitivity model, parallel per-class fits with per-class seeds, per-dimension output;
  - `scripts/component_metadata.py`: per-dimension tables;
  - the camera-pilot wrapper updated to the new arguments.

  The 30-specimen run's evidence stays reproducible from commit 8449837.

**Validation:**
- 45 tests pass in `.venv` and `.venv-gpu`.
- The cache verifies (PASS).
- The numbers in sections 1–7 were checked against the evidence files with `scripts/check_report_numbers.py`.

**Data:** outside git in `Feature_geometry_data/expanded_cohort/` (cohort, 0.79 GB of features, analysis outputs). Compact copies are in [docs/evidence/expanded_geometry/](docs/evidence/expanded_geometry/).

**Rerun:** `scripts/local/expanded_geometry.sh cohort extract verify geometry montage metadata compare`.

## 9. Open decisions for the author

1. **Is the reading acceptable?** It reads: one Gaussian per class is rejected at n = 200, and the dominant structure is source dataset / acquisition. It does not read "biological subtypes".
2. **Should later geometry work condition on source?** For example, model within one source (bulk only), or include `dataset` as a nuisance factor, before asking whether biological multimodality remains.
3. **The LUMI smoke test and the full feature bank** stay paused until LUMI reopens and the author resumes them.
