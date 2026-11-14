# Coding assistant response — Feature-Geometry

**Current task:** `task.md` a82e89e + I-JEPA amendment 3b897b0, "Replicate established frozen-feature geometry
across 8 datasets × 5 encoders".
**Status: 40 of 40 combinations complete (480 of 480 analysis tasks). Stopped for author review.**
- **Not done (not authorised):** no classifier, NCM, accuracy, fine-tuning, H-score/LogME study or test data.
- **Evidence:** `docs/evidence/geometry_replication/` (README). Matrix: `final/completion/completion_matrix.csv`;
  headline tables: `final/headline/headline_tables.md`; all indicators: `final/summary/`.
- **Previous response** (train+val extraction, task d57a630): commit b9911a3.

## 1. Completion matrix (40 rows)

All six analyses (geometry, overlap, FDA, LW, coupled, pooling) are done for every encoder × dataset × partition.
The full matrix has one row per combination: units, support rule, classes, eligible pairs, native space and
dimension, arm status counts, failures, job ids, code and project.

| Dataset | Partitions | Unit | Eligible classes | Eligible pairs | Train / val units | Encoders done |
|---|---|---|---|---|---|---|
| FIN-Benthic2 (derived five-split) | 5 folds | specimen | 34-36 | 561-630 | 6160-6165 / 1540-1543 | 5 of 5 (30/30 tasks each) |
| AquaMonitor | 5 official folds | specimen | 40-41 | 780-820 | 29346 / 4386 | 5 of 5 (30/30) |
| BIOSCAN-5M | official | record | 531 | 140715 | 289203 / 14757 | 5 of 5 (6/6) |
| DF24 (derived val) | official train | observation | 889 | 394716 | 143155 / 15871 | 5 of 5 (6/6) |
| Food-101 | official (derived val) | image hash | 101 | 5050 | 68171 / 7575 | 5 of 5 (6/6) |
| Oxford-IIIT Pet | official (derived val) | image hash | 37 | 666 | 3311 / 369 | 5 of 5 (6/6) |
| CIFAR-100 | official (derived val) | image hash | 100 | 4950 | 44993 / 5002 | 5 of 5 (6/6) |
| FGVC-Aircraft | official | image hash | 100 | 4950 | 3334 / 3333 | 5 of 5 (6/6) |

- **Support rule** (unchanged): >= 15 train and >= 5 val units per class. The model-specific rules are also unchanged.
- **Arm status totals** (`final/completion/totals.json`). Arms count class fits; overlap and FDA count pair scores.

  | Status | Count |
  |---|---|
  | ok | 730,994 |
  | unreliable | 24,900 |
  | failed | 9,213 |
  | not eligible | 27,448 |
  | not applicable | 19,330 |
  | not run | 85,480 |
  | pair scores ok | 77,638,270 |
  | pair scores not eligible | 6,074,180 |

  - *unreliable:* the established reliability rule failed, so the fit is excluded from the primary averages.
  - *not eligible:* the model's support rule is not met. Examples: n <= d for sample covariances; fewer than 2 d
    units for full covariances (so overlap and FDA full forms are undefined at 64 PCs for Pet and Aircraft).
  - *not applicable:* the source-group model GD_tied outside AquaMonitor.
  - *not run:* the taxonomy arms (§5).
- **Per-fold and per-class outputs:**
  - per fold: `final/summary/indicators_long.csv`, with intervals;
  - per arm: `final/completion/arm_status.csv`;
  - per class: in the run folders (`perclass.csv`, `per_class_*.csv`, `class_pairs.csv`).

## 2. What replicates and what does not

Cells below are means over partitions, from `final/headline/headline_tables.md`. Five-fold val sets overlap, so the
folds are not independent replicates. "Interval above 0" means the class-bootstrap 95 % interval of a contrast.

**Within-class mixtures (G2_tied − G1_full, class macro, 8 PCs).**
- **AquaMonitor: replicates for all five encoders.**
  - 0.096-1.188, interval above 0 in 3-5 of 5 folds; at 16 PCs 0.109-1.264.
  - The source-group model GD_tied beats G2_tied for 5 of 5 encoders (0.120-1.460).
  - So the earlier conclusion holds beyond DINOv3-L: AquaMonitor's within-class structure follows the acquisition
    source.
- **FIN-Benthic2: small at 8 PCs and absent at 16 PCs.**
  - 8 PCs: 0.032-0.134, interval above 0 in 1-5 folds.
  - 16 PCs: -0.121 to 0.011.
- **Elsewhere, the sign is set by the dataset more than by the encoder.**
  - Food-101 and CIFAR-100: positive for all five encoders at 8 and 16 PCs (interval above 0).
  - DF24: negative for all five at both dimensions (interval below 0).
  - BIOSCAN-5M: positive for all five at 8 PCs; at 16 PCs above 0 only for SigLIP 2 and I-JEPA.
  - FGVC-Aircraft: zero or negative (BioCLIP 2 -1.077, SigLIP 2 -0.959, I-JEPA -0.378).
  - Pet: mixed.
- **Overall:** every partition's interval lies above 0 in 20 of 40 cells at 8 PCs and below 0 in 9.
- A likelihood gain is not evidence of biological subtypes. The "mixture-supported split" label stays exploratory.

**Class separation (d′, val units along the train centroid direction; scale-free, not accuracy).**
- **I-JEPA has the lowest median d′ in 8 of 8 datasets** (1.17-6.23).
- **The highest median d′ depends on the dataset:**

  | Encoder | Datasets where it is highest (median d′) |
  |---|---|
  | DINOv3-L | FIN-Benthic2 5.41, AquaMonitor 5.59 |
  | BioCLIP 2 | BIOSCAN-5M 14.06, DF24 11.16 |
  | SigLIP 2 | Food-101 9.67, CIFAR-100 6.65, Aircraft 12.72 |
  | DINOv2-L | Pet 12.67 |

- The d′ indicators are not encoder rankings.

**Gaussian overlap (64 PCs, the paper's setting).**
- Spearman(d′, spherical Bhattacharyya distance) is 0.355-0.920.
- The paper and code-release spherical forms agree everywhere (Spearman ≥ 0.988).
- The full-form covariance term carries a median share of 0.153-0.702 of the distance. I-JEPA has the highest share
  in each of the 6 datasets where the full form is defined.

**PCA-GBC vs FDA-GBC (d = 16).**
- FDA-GBC agrees better with d′ than PCA-GBC in only 12 of 40 cells.
- **FDA train-to-val optimism (val/train GBC > 1) replicates and extends:**
  - FIN-Benthic2: 1.240-1.460 for all five encoders;
  - Pet: 1.102-2.627;
  - FGVC-Aircraft: 1.496-2.381.
- PCA-GBC stays at or below 1 except on Aircraft for DINOv2-L, BioCLIP 2 and I-JEPA (1.022-1.037).

**Covariance estimation.**
- **Per-class LW gives the best held-out fit for most classes at 64 PCs.** The two exceptions are both on FGVC-Aircraft,
  where the diagonal covariance wins instead:

  | Encoder (Aircraft) | LW best | Diagonal best |
  |---|---|---|
  | BioCLIP 2 | 52 of 100 classes | 48 |
  | I-JEPA | 14 of 100 classes | 86 |

- **Pooled shrinkage (DINO encoders only).** Wherever the best grid shrinkage differs from 0, it is larger than
  α_LW. Examples: FIN-Benthic2 6.94e-03 → 1e-02 (DINOv3-L) and 6.41e-03 → 3e-02 (DINOv2-L). It is 0 (the pooled
  sample covariance) on BIOSCAN-5M, DF24, Food-101 and CIFAR-100.
- **Pooling at LW shrinkage beats per-class LW in native space in 40 of 40 cells, with every partition's interval
  above 0.**
  - The earlier DINOv3-L values reproduce: 262.011 (FIN-Benthic2) and 185.403 (AquaMonitor).
  - Balanced pooled covariance − LW is positive in all 40 cells too.
- **Coupled RSCM − per-class LW in native space splits by encoder family.** Every partition's interval is above 0 in
  17 of 40 cells and below 0 in 21. By encoder:

  | Encoder | Datasets where coupled RSCM is better |
  |---|---|
  | DINOv2-L | 0 of 8 |
  | DINOv3-L | 2 of 8 (DF24 80.033 and Aircraft 5.731) |
  | BioCLIP 2 | 6 of 8 |
  | SigLIP 2 | 5 of 8 |
  | I-JEPA | BIOSCAN-5M, DF24, Food-101, Pet and Aircraft |

  - The earlier DINOv3-L values reproduce: -396.599 and -86.497.
  - The encoder split coincides with a difference in the native space. Full-rank encoders use all coordinates
    (`native_all`); rank p − 1 encoders drop one. The comparison is therefore confounded (decision D1).
- **Coupled − LW at 64 PCs:** mostly positive; I-JEPA × AquaMonitor is -0.951 with interval below 0 in 5 of 5 folds.
- **Balanced − specimen-weighted pooling:**
  - positive on the biology datasets for most encoders (AquaMonitor: interval above 0 in 5/5 folds for all five
    encoders at 16 and 64 PCs);
  - |value| ≤ 0.003 on the four general benchmarks. Their classes are near-balanced, so the two weightings hardly
    differ (inferred, not tested).

## 3. I-JEPA ViT-H/14 (amendment 3b897b0)

**Gates passed:**
- **Checkpoint:** official `IN1K-vit.h.14-300e` target encoder, epoch 299, 630,762,240 parameters. File sha256
  0382013c…; the recomputed S3 multipart ETag equals the server ETag.
- **Code:** facebookresearch/ijepa@52c1ae9, vendored unmodified (CC BY-NC 4.0).
- **Determinism:** 512 BIOSCAN-5M images encoded twice gave 512 of 512 bitwise-equal rows.
- **Rank:** 1279 of 1280 (final LayerNorm null direction, |cos| 1.000 with 1/w).
- **Cost:** throughput 60.6 img/s/GCD. The preflight estimate was 5.3 GPU-h; the measured extraction took 4.64 GPU-h.
- **Records:** C3 records 8 of 8 pass.

**Results:**
- **Weakest class separation everywhere:** lowest median d′ in 8 of 8 datasets.
- **Most concentrated within-class spread:**
  - lowest effective rank in 5 of 8 datasets;
  - highest covariance-term share wherever the full form is defined.
- **Largest AquaMonitor mixture and source-group gains:** 1.188 / 1.460 against 0.096-0.278 / 0.120-0.455 for the
  other four encoders.
- **On FGVC-Aircraft:**
  - 0.4313 of class pairs have d′ < 1;
  - the median nearest-class d′ is 0.05;
  - the diagonal covariance beats LW in 86 of 100 classes.

**ImageNet-1K class overlap** (I-JEPA was pretrained on ImageNet-1K images without labels): string candidates only, in
`imagenet_overlap/`. An image-level audit is not possible without the ImageNet images.

| Dataset | Candidate classes |
|---|---|
| Food-101 | 9 of 101 |
| Pet | 18 of 37 |
| CIFAR-100 | 63 of 100 |
| FGVC-Aircraft | 0 of 100 |

## 4. Reproduction of the earlier DINOv3-L results

`reference_check/reference_check.json` compares every earlier output cell by cell (relative tolerance 1e-9).
- **AquaMonitor:** all 26 earlier run outputs are identical.
- **FIN-Benthic2:** 62 mismatches, all in geometry outputs:
  - mixture component labels swapped (the same partition);
  - montage specimens taken from the swapped component;
  - one stability statistic: fold4, one class, G2_tied split ARI median 0.0034 vs 0.0028.
- Every other FIN-Benthic2 value agrees within 2.4e-10 relative.
- The most plausible cause is rounding-level input differences flipping a component order. Not traced further.

## 5. Deviations (forced by data or author instruction)

1. **Taxonomy arms not run.** pool_tax and its nearest-mean control pool_near were dropped on author instruction
   (2026-11-13, "for now we don't consider the taxonomy"); 85,480 class fits are marked "not run".
2. **Native spaces.** DINOv2-L, DINOv3-L and I-JEPA have rank p − 1, so the established `native` / `native_alt` (one
   coordinate dropped) are used. BioCLIP 2 and SigLIP 2 are full rank: `native_all` (main) and `native` (sensitivity).
3. **Single-camera datasets** (all except FIN-Benthic2 and AquaMonitor): camera re-identification, two-view agreement
   and the source-group model are not applicable. Acquisition ARIs are computed only for AquaMonitor and FIN-Benthic2.
4. **Not replicated:**
   - `descriptive_geometry.py`, which pools train and val in-sample;
   - `bhattacharyya_orientation_summary.py`, a post-hoc summary of one run.
5. **Duplicate content units with conflicting labels** (CIFAR-100, Food-101) are kept as one unit per (unit, label).
   Nothing is removed or relabelled.
6. **Code changes with identical results (byte-identical outputs checked):**
   - per-class axes and statistics are computed once instead of inside the pair loops;
   - unit ids are read as text (a DF24 join failure; regression test added);
   - an overlap plot no longer fails when no full-covariance pair exists.

## 6. Numerical failures and reliability

- **`coupled_no_pooling`** (a mechanism-decomposition arm: sample covariance without pooling): 9,203 class fits are
  numerically singular, mostly in native spaces and at 64 PCs. The two decomposition contrasts built on it record
  9,203 failures each. Primary contrasts are unaffected.
- **Primary `coupled` arm:** one class per fold fails, DINOv3-L × FIN-Benthic2, native space (15 contrast failures).
  The earlier run had the same failure.
- **No other arm failed.** Nothing was jittered or refitted.
- **Mixture fits failing the reliability rule** (24,900 class fits in total) are excluded from the primary averages,
  exactly as before. Their counts per arm are in `status_by_arm.csv`.
- **Runs:**
  - 483 runs; 480 wrote outputs, each output written exactly once (asserted).
  - The 3 unfinished runs are the probe jobs: one failed on the DF24 id bug (rerun after the fix); two were cancelled
    with author approval (rerun with the faster byte-identical code).
  - stderr holds only module banners, the two cancellations, and the expected empty-slice warnings of camera metrics
    on single-camera datasets.

## 7. Resources (`accounting/resources_final.json`, allocation-based)

- **Total:** 5.01 GPU-h (I-JEPA extraction and gates) and 5,400.37 billed CPU core-h, of which the geometry
  replication used 5,374.92. All CPU work ran on the LUMI small partition.
- **Per project:**

  | Project | Billed CPU core-h | GPU-h |
  |---|---|---|
  | `462001534` | 1,662.83 | 5.0 |
  | `462001824` | 3,737.55 | 0 |

  The `462001824` share is entirely the BIOSCAN-5M / DF24 batches B2a-B2d.
- **Cost overrun:** the CPU cost ran well above my running estimates; the 21:13 snapshot on 13 Nov already showed
  1,205.58. I moved batch 2 to `462001824` because `462001534`'s CPU allocation was nearly used.
- **Balances on 2026-11-14** (`allocations_final.txt`):
  - `462001534`: 4109/5000 core-h and 39642/40000 GPU-h used;
  - `462001824`: 55012/500000 core-h and 993/5000 GPU-h used.

## 8. Code and tests

- **New scripts** (`scripts/replication/`):
  - `build_cohorts.py`, `make_tasks.py`, `compare_reference.py`, `summarise_replication.py`, `separation_table.py`;
  - `completion_matrix.py` (40-row matrix, arm status, run provenance parsed from the Slurm logs);
  - `headline_tables.py` (tables and cross-table counts);
  - `imagenet_class_overlap.py`.
- **New supporting code:**
  - `src/data/benchmark_cohort.py`;
  - `scripts/lumi/replicate_geometry.sh`, `scripts/local/sync_code_to_lumi.sh`;
  - I-JEPA adapter `src/encoders/ijepa.py` with `third_party/ijepa/`.
- **Changed:** `sum_resources.py` adds per-project totals when the listing has the Account field; outputs for the
  older listings are unchanged (checked).
- **Tests:** 115 pass, including the new `tests/test_replication_summaries.py`.

## 9. Decisions for the author

- **D1. Native-space coupled comparison.** Should the full-rank encoders' primary native space be `native` (one
  coordinate dropped, same as the rank p − 1 encoders) instead of `native_all`? Both are already computed, so this is
  a reporting choice, not a rerun.
- **D2. Taxonomy arms.** Run pool_tax / pool_near later for FIN-Benthic2, AquaMonitor, BIOSCAN-5M and DF24, or leave
  them out?
- **D3. FIN-Benthic2 reproduction.** Accept it as reproduced (component-label swaps, one stability median), or trace
  the cause?
- **D4. Billing.** CPU work moved to `462001824`; `462001534` has used 39642 of 40000 GPU-h. Use `462001824` for the next tasks?
- **D5. Next task.** None started. One candidate from the research notes is checking whether the AquaMonitor mixture
  and source-group gains survive ℓ2-normalised features (CPU only, existing caches). I-JEPA's gains there are at least
  four times the others'.

## 10. End
