# Coding assistant response — Feature-Geometry

**Current task:** `task.md` e4d291d, "Paper-faithful GBC and matched PCA-vs-FDA experiment".
**Status: done for all 40 encoder × dataset combinations (80 runs). Stopped for author review.**
- **Not done (not in scope):** no mixtures, coupled RSCM, taxonomy pooling, H-score, LogME, NCM, classifiers,
  fine-tuning or test data.
- **Evidence:** `docs/evidence/gbc_matched_comparison/` (README: protocol, formulas, paper vs our extension,
  limitations). Tables: `summary/tables.md`; long tables: `summary/*.csv`.
- **Previous response** (geometry replication 8 × 5): commit 434de30.

## 1. What was run

Everything is computed from the per-pair Bhattacharyya tables already saved by the replication's `fda` analysis. No
projection or Gaussian was refitted, and nothing needed refitting to fix validity or pair coverage.

| Arm | What it is | Source |
|---|---|---|
| A | PCA 64, spherical, training-only GBC: paper form (Eq. 2, factor d, n − 1) and code-release form | the paper's setup |
| B | Regularised FDA at d = 8 / 16 / 32, with raw and shrinkage variants as labelled sensitivity arms | our extension |
| C | PCA-d vs FDA-d at the same d, over the identical class pairs of both arms | our extension |
| D | Train vs val GBC summed over the identical pair set P_common in every arm | our diagnostic |

- **E:** mean BC per pair, ΔBC quantiles, rank agreement, retention of the 20 hardest pairs, mean and covariance
  terms. The val/train ratio is reported only when the train mean BC is ≥ 1e-6 (predeclared; never triggered).
- **F:** strata by validation support of the pair (5–9, 10–19, ≥ 20 val units). Folds are shown individually.

The train/val comparison is our diagnostic; the paper uses GBC as one training-only score. The authors proposed
neither FDA nor the validation ratio.

## 2. Findings

**Pair-set correction (arm D).** The old convention summed train GBC over all train-valid pairs but val GBC over pairs
valid on both. It changed nothing for the spherical and diagonal forms: their train and val pair sets were identical
in every run and arm. So all earlier spherical and diagonal values, including the FDA "optimism" ratios, stand
unchanged. The full form is different:

| Full covariance form | Count |
|---|---|
| run × arm rows where the pair sets differ | 1070 of 1280 |
| rows with no common pair at all | 175 |
| rows with both ratios defined | 917 |
| …of which the old ratio was ≤ 1 and the matched ratio is > 1 | 33 |
| …of which the reverse happened | 0 |

So the old convention understated the full-form validation overlap in some runs.

**A — paper vs code release.** The two forms rank class pairs almost identically: Spearman of the pair BC ≥ 0.9864 in
every run (lowest: I-JEPA, AquaMonitor fold1).

**D — train vs val over identical pairs (spherical).**
- **PCA:** the val/train ratio of mean BC is mostly below 1. It exceeds 1 in only 6 of 80 runs at 64 dimensions and
  in 7 of 80 at 16.
- **FDA:** the ratio exceeds 1 in 50 (d = 8), 43 (d = 16) and 39 (d = 32) of 80 runs. At d = 16 the pattern by dataset
  (means over folds):

  | Dataset | FDA val/train ratio, d = 16 |
  |---|---|
  | FIN-Benthic2 | 1.240–1.460 |
  | Oxford-IIIT Pet | 1.102–2.627 |
  | FGVC-Aircraft | 1.496–2.381 |
  | BIOSCAN-5M | below 1 for all five encoders |
  | DF24 | below 1 for all five encoders |
  | AquaMonitor, Food-101, CIFAR-100 | close to 1 |

**C — matched PCA vs FDA (spherical, same d, same pairs).**
- At d = 16, FDA gives a lower mean BC than PCA on train in 76 of 80 runs and **on val in 74 of 80**.
- FDA's larger val/train ratio (FDA ratio > PCA ratio in 67 of 80 runs) therefore mostly comes with *less* estimated
  validation overlap than PCA, not more. The ratio grows because FDA lowers train overlap more than val overlap.
- **Exceptions where FDA's val mean BC is higher:**
  - BioCLIP 2 on BIOSCAN-5M (3.64e-03) and DF24 (1.61e-03);
  - DINOv3-L (3.54e-03) and DINOv2-L (3.76e-03) on Food-101;
  - DINOv3-L (6.36e-05) and DINOv2-L (8.88e-04) on Pet.
- d = 8 and d = 32 give the same picture:

  | d | FDA lower on train | FDA lower on val | FDA ratio > PCA ratio |
  |---|---|---|---|
  | 8 | 76 of 80 | 74 of 80 | 66 of 80 |
  | 32 | 75 of 80 | 75 of 80 | 61 of 80 |

**F — low validation support (spherical FDA-16).**
- Pairs involving a class with only 5–9 val units can have very high ratios. Examples: Pet DINOv3-L 23.281;
  FIN-Benthic2 I-JEPA up to 21.726 in one fold.
- This is not general: across the 60 runs with both strata, the low-support ratio exceeds the ≥ 20-unit ratio in only
  18.
- FIN-Benthic2's FDA ratio stays above 1 even for well-supported pairs (≥ 20 val units: 1.302–1.577). Its optimism is
  not only a small-sample effect.

None of this is classification accuracy or transferability. The ratio describes estimated overlap of Gaussian
summaries, and a val/train ratio alone does not establish that the projection overfits.

## 3. Checks

- **Leakage gate**, in every run: the GBC classes are the eligible classes, and the train/val unit counts equal the
  cohort after dropping the partition's test rows.
- **Regression**, in every run: the recomputed GBC equals the saved values (largest relative difference 5.5e-13).
- **Bounded verification:** the DINOv3-L FIN-Benthic2 fold0 and Pet runs were computed on the Dell before scaling up.
  Their outputs are identical to the LUMI outputs.
- **Tests:** `tests/test_gbc_matched.py` covers (a) agreement with the earlier summary code, (b) the
  ordered/unordered factor 2, (c) identical pair sets across train/val and matched arms, (d) the leakage gate, (e)
  near-zero GBC. The full suite passes (120 tests).

## 4. Limitations

- The train and val Gaussians come from different sample sizes; val is smaller and noisier. The direction of the
  resulting bias in BC is not derived.
- FDA uses train labels, so its train separation includes a selection effect that val does not share.
- High-BC pairs dominate GBC sums, so the per-pair ΔBC quantiles are reported.
- Five-fold val sets overlap, and class pairs are not independent. No interval treats folds or pairs as independent
  replicates.

## 5. Resources

- LUMI array job `23725233`: small partition, 80 tasks, project `462001824`.
- 1.07 billed CPU core-h (`accounting/resources.json`).
- The code was synced as 3cbc101.

## 6. Code

- **New:**
  - `src/statistics/gbc_matched.py` (matched sums, ratios, matched arms, per-class);
  - `scripts/gbc_matched_comparison.py` (one run: gates, regression, outputs);
  - `scripts/gbc_matched_summary.py` (tables and counts);
  - `scripts/lumi/gbc_matched.sh` (array);
  - `tests/test_gbc_matched.py`.
- **Changed:** `scripts/benchmark/sum_resources.py` gained a job-name group.
- **Unchanged:** the earlier `scripts/pca_fda_gbc.py` and every earlier output.

## 7. Decisions for the author

- **D1. Ratio threshold.** Keep the predeclared minimum train mean BC (1e-6) for reporting a ratio? It was never
  reached here.
- **D2. Primary summary.** Report FDA vs PCA primarily as matched val mean BC (arm C), with the val/train ratio as
  secondary? The ratio alone suggests FDA is worse on val, whereas the matched val BC shows FDA mostly lower.
- **D3. Full-form results.** These are supplementary. Keep the 33 full-form ratio changes as a documented correction
  only, or report them?
- **D4. Next task.** None started.

## 8. End
