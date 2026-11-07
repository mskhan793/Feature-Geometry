# Coding assistant response — Feature-Geometry

**Current task:** `task.md` fb26afc, "within-class and between-class geometry" (follow-up to the camera pilot; local only).
**Status: done, stopped for the author's review.**
- **Used:** only the existing verified DINOv3-L cache and the precommitted cohort.
- **Not done:** no new extraction, no larger caps, no test data, no classifier, nothing remote.

**Previous tasks:**
- Camera pilot: response at commit e345d91, evidence in `docs/evidence/camera_pilot/`.
- LUMI smoke test: still pending, because LUMI is closed.

## 0. A correction found during this task

- **What happened:** the camera pilot fitted its mixtures on float32 PCA scores, and float32 mixture fits can miss narrow splits (found with synthetic tests here).
- **Fix:** specimen means are now float64 (`src/features/specimen_means.py`).
- **Effect on the camera pilot:** none. I reran it in float64, and all 45 numbers in its density section reproduce exactly (`check_report_numbers.py` against [results_float64_rerun.json](docs/evidence/camera_pilot/results_float64_rerun.json)). Its conclusions stand.

## 1. Answer in brief

**Unit and fitting:**
- **Unit:** one embedding per independent specimen, the mean of its two camera-level means.
- **Cohort:** 41 classes, 1,230 train and 534 val specimens (val 6–15 per class).
- **Fitting:** everything is fitted on train only; val is only scored.

**Answers:**
- **Within class, one class shows credible separate groups under the predeclared rule: Caenis rivulorum.** Its two groups reproduce the source sub-dataset almost exactly (bulk22 vs dna; ARI 0.926 on train, 1.000 on val specimens assigned by the train fit), and in the images they differ in background colour and ID scheme. So this is **acquisition structure, not a biological subtype**. Separating the two is impossible here, because dataset, year and imaging setup coincide.
- **More broadly, the clearest within-class structure follows dataset, not biology.** The 2-component split reproduces the dataset label (ARI ≥ 0.5) in 10 of 40 classes on train and 9 of 37 on held-out val specimens. It reproduces year in 7 of 38 (train) and site almost never.
- **One Gaussian or several, class by class:**

  | Verdict | Classes |
  |---|---|
  | one Gaussian favoured | 10 |
  | inconclusive | 13 |
  | insufficient evidence | 17 |
  | credible multiple groups | 1 |

  In 17 classes every mixture fit was unusable: its best split isolates a group smaller than 9 specimens (8 PCs + 1), often a dataset subgroup. There is **no basis for "classes are multimodal" in general**, and no basis for "unimodal" either. With 30 specimens most classes cannot be decided.
- **Between classes, every class is separable from every other along their centroid direction except a few close pairs.** On val specimens, d' < 2 for 22 of 820 pairs and < 1 for 1 pair. The closest pairs are related taxa:
  - Caenis horaria / C. luctuosa (d' 0.89);
  - Cyrnus trimaculatus / Ecnomus tenellus (1.14);
  - Polycentropus / Tinodes (1.34);
  - Mystacides / M. azurea (1.38);
  - Cloeon / Cloeon dipterum (1.43).

  Judged by distance relative to cloud radius, 163 pairs "overlap" (ratio < 1). That is a property of high dimension, not real overlap.
- **Camera:** the two views of a specimen are each other's nearest view among the class's specimens in a median 40 % of specimens (chance 3.3 %). The mixture splits keep both views together (median 0.967). Camera does not drive within-class groups.

## 2. Within-class geometry (native 1024-d, train, 30 specimens per class)

[per_class_summary.csv](docs/evidence/class_geometry/per_class_summary.csv): one row per class, with geometry, model comparison, verdict, nearest class and metadata agreement. The full table is [per_class.csv](docs/evidence/class_geometry/per_class.csv).

| Indicator (over 41 classes) | min | median | max |
|---|---|---|---|
| RMS radius | 4.55 | 6.19 | 7.86 |
| share of variance on the first direction | 0.156 | 0.258 | 0.448 |
| share on the first 5 directions | 0.533 | 0.618 | 0.746 |
| effective rank (exp entropy) | 9.11 | 14.72 | 18.46 |
| participation ratio | 4.33 | 9.14 | 13.27 |
| camera-view re-identification | 0.067 | 0.400 | 0.700 |

- Ranks are bounded by n − 1 = 29, and are comparable here only because every class has n = 30.
- **Most elongated classes:** Caenis rivulorum (0.448, the dataset split), Micronecta, Pedicia, Oulimnius tuberculatus adult, Ceratopogonidae.
- **Lowest view re-identification:** Micronecta (0.067), where the two views of a specimen are nearly as different as two specimens.
- **Spectra:** [eigenspectra.png](docs/evidence/class_geometry/eigenspectra.png). The classes have similar, smoothly decaying spectra with no gap.

**Selected classes** (chosen by train BIC only: three "candidate groups", two "candidate single cloud"):
- Internal geometry: [class_internal_geometry.png](docs/evidence/class_geometry/class_internal_geometry.png).
- Image montages: [Caenis rivulorum](docs/evidence/class_geometry/montage_Caenis_rivulorum.png), [Centroptilum luteolum](docs/evidence/class_geometry/montage_Centroptilum_luteolum.png), [Caenis luctuosa](docs/evidence/class_geometry/montage_Caenis_luctuosa.png), [Ephemera vulgata](docs/evidence/class_geometry/montage_Ephemera_vulgata.png), [Tanypodinae](docs/evidence/class_geometry/montage_Tanypodinae.png).

Observations are hypotheses only:

| Class | Data | Observation |
|---|---|---|
| Caenis rivulorum | train: bulk22 16, dna 13, bulk21 1; components 17 / 13 | The dna-aligned component's specimens have plate–well IDs (`A-A3-CaRi-hau1-1`) and a purple background; the other group is teal. Same pattern in **Centroptilum luteolum** (dataset ARI 0.741 train / 0.691 val; split 22 / 8, so too small to call). |
| Caenis luctuosa | components 23 / 7 | The 7-specimen group again has plate–well IDs on a darker background (dataset ARI 0.510 / 0.598). |
| Tanypodinae | "one Gaussian favoured"; tied split 15 / 15 | The split is unrelated to dataset (ARI −0.016). The montage shows mixed curled and straight larvae in both groups: pose varies continuously. |

## 3. Between-class geometry

[class_pairs.csv](docs/evidence/class_geometry/class_pairs.csv) (all 820 pairs), [nearest_class.csv](docs/evidence/class_geometry/nearest_class.csv), [class_pair_separation.png](docs/evidence/class_geometry/class_pair_separation.png), [closest_pairs.png](docs/evidence/class_geometry/closest_pairs.png).

**Metrics** (native space, train centroids):
- **d' (primary):** the difference of the mean projections onto the train centroid-difference direction, divided by the root mean of the two within-class variances. It is computed on **val** specimens, so it is not measured on the data that chose the direction.
- **Radius separation:** centroid distance divided by the classes' RMS radius. It ignores direction.

**Results:**
- **Closest pairs:**

  | Pair | d' val | d' train | radius separation |
  |---|---|---|---|
  | Caenis horaria / C. luctuosa | 0.89 | 2.50 | 0.35 |
  | Cyrnus trimaculatus / Ecnomus tenellus | 1.14 | 1.94 | 0.42 |
  | Polycentropus flavomaculatus / Tinodes waeneri | 1.34 | 2.23 | 0.53 |
  | Cyrnus flavidus / Polycentropus flavomaculatus | 1.35 | 2.42 | 0.93 |
  | Mystacides / Mystacides azurea | 1.38 | 1.25 | 0.41 |
  | Cloeon / Cloeon dipterum | 1.43 | 2.32 | 0.53 |

  Two of these pairs are a genus-level label next to a species of that genus, so they may overlap by definition.
- **Held-out vs train separation:** d' on val is almost always below d' on train. A direction chosen on 30 specimens looks better on them than it is.
- **Radius vs direction:** 163 pairs have radius separation < 1, yet only 22 have d' val < 2. Two clouds can be narrower along their difference direction than their radius suggests.
- **Scale and projection choices:**
  - L2-normalising the embeddings barely changes the ordering of pairs by radius separation (Spearman 0.9996).
  - The global 2-D PCA (0.304 of the variance) keeps the order fairly well (Spearman 0.900), but **overstates separation for most pairs**: it shows the directions that separate classes, not the directions where classes spread.
  - 2-D plots are therefore descriptive only.

## 4. One Gaussian or several (8 PCs, held-out specimens)

**Setup:**
- PCA fitted on train specimens and whitened; 8 PCs explain 0.558 of the train variance (specimen means, so more than the camera-level 0.487). Results apply to this space only.
- 16 PCs is not used: one full Gaussian would need 152 parameters for 30 specimens.
- Fitting: `reg_covar` 1e-3. Starts: 5 k-means, 5 random specimens, and one split along each class principal direction; the best training likelihood is kept.
- A model is eligible if its free parameters are at most n·d/2, half the 30 × 8 training numbers. A fit is "unreliable" if a component has fewer than 9 specimens.

**Fit status (41 classes):**

| Model | Parameters | Fit status |
|---|---|---|
| G1 full | 44 | ok 41 |
| G2 full | 89 | ok 16, unreliable 25 |
| G2 tied | 53 | ok 21, unreliable 20 |
| G3 tied | 62 | ok 3, unreliable 38 |
| G3 full | 134 | ineligible |
| G1 diag, G2 diag | 16 / 33 | diagnostic only |

Largest covariance condition numbers: G1 full 314, G2 full 2874.

**Held-out mean log density per val specimen, difference to G1 full** (class mean, 95 % bootstrap CI; classes better / CI above 0 / CI below 0). These averages include the unreliable fits:

| Model | Difference to G1 full |
|---|---|
| G2 full | −9.21 [−11.80, −6.79]; 1 / 0 / 28 |
| G2 tied | −1.57 [−2.77, −0.52]; 12 / 8 / 11 |
| G3 tied | −2.32 [−3.58, −1.26]; 9 / 4 / 17 |
| G2 diag − G1 diag (diagnostic) | +0.045 [−0.34, 0.39]; 28 / 7 / 3 |

**Split stability:**
- The tied 2-group split, refitted on 80 % train subsamples, predicts the same val groups (median ARI ≥ 0.5) in 20 classes. A stable split alone does not prove separate groups: an elongated single cloud also splits stably.
- Both camera views of a specimen fall in the same component in a median 0.967 of train specimens.

**Verdicts** (predeclared, over eligible, converged, reliable full/tied mixtures):
- **credible multiple groups:** some mixture's interval is above 0 AND the tied split is stable on val. **1 class:** Caenis rivulorum (G2 tied +1.40 [0.36, 2.07]).
- **one Gaussian favoured:** every mixture's interval is below 0. **10 classes:** Asellus aquaticus, Caenis horaria, Chironomidae, Leptophlebia, Limnephilidae, Micronecta, Oulimnius tuberculatus adult, Sialis, Tanypodinae, Turbellaria.
- **inconclusive: 13**, and **insufficient evidence** (no usable mixture): **17**. Lists are in `results.json`.

**Dataset agreement by verdict** (dataset ARI ≥ 0.5 on train): credible 1 of 1, insufficient 6 of 17, inconclusive 3 of 13, one-Gaussian 0 of 10 ([component_metadata.json](docs/evidence/class_geometry/component_metadata.json)). Train classes span 1 (one class), 2 (8), 3 (20) or 4 (12) datasets.

## 5. Caveats and what remains ambiguous

- **Sample size:** with 30 train specimens most mixture fits are unusable (small components). Val has only 6–15 specimens per class, so per-class intervals are wide.
- **Space:** results are in an 8-PC space (55.8 % of variance), not native geometry.
- **Mixture search limit:** the search cannot guarantee the best split. In a synthetic test, a narrow split along a direction shared with the between-class axis was found on the full data but not reproducibly on subsamples. Splits like that can be missed.
- **Dataset vs biology:** dataset, year, imaging background and naming scheme coincide, so "acquisition" cannot be split into its parts. A biological difference between sub-datasets (e.g. different life stages sampled) cannot be excluded.
- **Fixed choices:** the verdict thresholds (ARI 0.5, interval above or below 0, 9 specimens per component) are predeclared but arbitrary. The montage classes were selected by train BIC.
- **Scope:** one encoder and fold0 only. Exploratory, not confirmatory.

## 6. Changes, validation, compute

**New:**
- `src/features/specimen_means.py` (moved out of the camera script; now float64)
- `src/statistics/class_geometry.py` (eigen summary, radius / direction separation, subspace overlap)
- `src/statistics/class_modality.py` (model set, parameter counts, eligibility, multi-start fitting, conditioning, split stability)
- `scripts/class_geometry_pilot.py`, `scripts/class_geometry_montage.py`, `scripts/component_metadata.py`
- wrapper stages `geometry`, `montage`, `metadata` in `scripts/local/camera_pilot.sh`
- tests `test_class_geometry.py` and `test_class_geometry_pilot.py`. The end-to-end test is synthetic: a planted two-group class must come out "credible multiple groups", and the single clouds must not.

**Changed:** `scripts/camera_gaussian_pilot.py` now imports `specimen_camera_means`. Nothing else in it changed.

**Validation:**
- All tests pass in `.venv` and `.venv-gpu`.
- A second geometry run gave identical `results.json` and `per_class.csv`.
- The numbers in sections 0–5 were checked against the evidence files with `scripts/check_report_numbers.py`.

**Compute:** CPU only, 320 s wall for the geometry stage ([geometry_time.txt](docs/evidence/class_geometry/geometry_time.txt)). Montages and metadata take seconds. No extraction.

**Data:** outside git in `Feature_geometry_data/camera_pilot/geometry_gpu/`; compact copies in [docs/evidence/class_geometry/](docs/evidence/class_geometry/).

**Rerun:** `scripts/local/camera_pilot.sh geometry montage metadata`.

## 7. Open decisions for the author

1. **Is the reading acceptable?** It reads:
   - the main within-class structure detectable at n = 30 is acquisition sub-dataset;
   - one class meets the multiple-groups rule, and that is acquisition;
   - most classes cannot be decided.
2. **Larger caps** (asked in chat; not allowed by the current task). Train specimens imaged by both cameras: 36 of 42 classes have at least 50, 26 have at least 100, 17 have at least 200. Fold0 val has median 20 specimens per class (minimum 3). GPU extraction runs about 57 images/s, so 100 train specimens × 6 frames for 41 classes (at most 24,600 images) is about 7 minutes. Larger n would make 16 PCs and full-covariance mixtures usable and narrow the intervals.
3. **Should dataset be stratified, or conditioned on, in later geometry work?** For example, fit within one dataset or report per dataset, since it dominates detectable within-class structure.
