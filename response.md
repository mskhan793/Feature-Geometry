# Coding assistant response — Feature-Geometry

**Current task:** `task.md` f54f534, "Class-balanced and taxonomy-informed covariance pooling".
**Status: done for all five folds of FIN-Benthic2 and AquaMonitor; stopped for the author's review.**
- **Data:** frozen DINOv3-L specimen embeddings.
- **Fitting and scoring:** train fits the PCA, class means and covariances; val gives the held-out class-conditional
  Gaussian log-likelihood.
- **Excluded:** no test rows, no classifiers, no tuning on validation, no Bhattacharyya/GBC/FDA work.

Previous response (coupled RSCM vs per-class LW, native space, mechanism decomposition): commit 82a3556.

## Answers

1. **Majority-class influence: balancing helps, modestly and consistently.** Under an otherwise identical pooling
   and shrinkage rule, the class-balanced pool (every class once) beats the specimen-weighted pool. The gain is
   positive in every fold of both datasets in the PCA spaces (class-macro and specimen-micro intervals above 0 in
   5 of 5 folds) and in native space.
   - Size: +0.0309 / +0.0627 (FIN-Benthic2, 16 / 64 PCs) and +0.0739 / +0.1934 (AquaMonitor) per val specimen,
     class-macro.
   - Largest for rare classes at 64 PCs in FIN-Benthic2 (+0.193) and for mid-size classes in AquaMonitor (+0.294).
   - One exception: FIN-Benthic2 native, classes with 64–255 train specimens (−0.604, 5 folds below 0).
2. **Genus/family borrowing beats the global pool, but taxonomy adds nothing beyond borrowing from similar
   classes.**
   - For the classes where a taxonomic donor level was available, borrowing from the genus or family beats the
     global (balanced, leave-target-out) pool in every fold of the PCA spaces: +0.155 / +0.779 (FIN-Benthic2) and
     +0.095 / +0.481 (AquaMonitor), 16 / 64 PCs.
   - **But** a non-taxonomic control does as well or better: the same number of nearest classes by train-mean
     distance. Taxonomic − nearest is −0.012 / −0.071 (FIN-Benthic2) and +0.004 / −0.097 (AquaMonitor), with no
     interval above 0 in any fold.
   - So the benefit is consistent with *borrowing from statistically similar classes*. No taxonomy-specific
     (biological) advantage is shown.
   - In native space taxonomy is almost never usable. Where it is (AquaMonitor, 5 taxa), it is worse than the global
     pool (−21.208, interval below 0 in 4 folds) and than the nearest-mean control (−20.425).
3. **How often taxonomy is unavailable** (targets per fold that fall back to the global pool):

   | Space | FIN-Benthic2 (of 34–36) | AquaMonitor (of 40–41) |
   |---|---|---|
   | 16 PCs | 11–13 | 18–19 |
   | 64 PCs | 12–14 | 21–22 |
   | Native | all | 36–37 |

   - **Why:** no taxonomic mate among the candidate classes (12 FIN-Benthic2 and 17 AquaMonitor classes), a
     family-rank or unknown target (e.g. *Simuliidae*, *Chironomidae*, *Oligochaeta*, *Turbellaria*), or too few
     donor degrees of freedom for the dimension.

## 1. Sources and taxonomy (gate A)

- **Methods:** Ledoit & Wolf (2004) and Raninen & Ollila (2021) with the authors' CoupledRSCM code (`bb09e12`) are
  reused unchanged (`docs/coupled_rscm_source_note.md`). No new paper is needed. Balanced pooling, leave-target-out
  pools, genus → family → global donors, the support rule and the nearest-mean control are **our proposals**,
  labelled as such.
- **Taxonomy from the dataset authors' tables:**
  - AquaMonitor: release `taxon_mapping.csv` + `taxon_table.csv`.
  - FIN-Benthic2: the release's `Table of classes.pdf`, transcribed verbatim.
  - Details, checksums and the audit: `docs/balanced_taxonomic_pooling_note.md` and
    `docs/evidence/balanced_taxonomic_pooling/taxonomy_audit/`.
- **FIN-Benthic2:**
  - all 39 classes map one-to-one, and every class's specimen count equals the release table's;
  - 26 species-, 12 genus-, 1 family-rank classes.
- **AquaMonitor:**
  - 40 of 42 classes have ancestry; *Oligochaeta* and *Turbellaria* have none (global fallback, no invented
    assignment);
  - 23 species-, 13 genus-, 4 family-rank classes;
  - the two *Oulimnius tuberculatus* classes (larva / adult) share one species.
- **Cross-check against the vision-language project's maps and FIN-Benthic2 hierarchy table (author's request):**
  - AquaMonitor: 99 of 99 links agree;
  - FIN-Benthic2 genus and family groupings: identical. Only spellings differ — the release's typos *Perlodiae*,
    *Sphaeridae*, *Taenioptegyridae*, *Rhycophila*, and v1's *Plectronemia*.

## 2. Design (fixed in commit `ff2cdf3` before the first run)

- **Protocol:** identical to the coupled-RSCM study (same caches, specimens, eligibility ≥ 15 train / ≥ 5 val,
  train-fitted PCA 16 / 64, train means). Native space uses 1023 coordinates, with `native_alt` dropping the other
  end.
- **One rule for every pool P:** T = β_c S_c + (1 − β_c) P and Σ = α T + (1 − α)(tr T/d) I. Here β_c is the authors'
  CRSCM β of the class (a fixed diagnostic, tuned for their own pool only) and α = 1 − α_LW,c. The S_j are unbiased
  within-class covariances; no covariance is ever taken around a shared mean.
- **Pools:**
  - specimen-weighted vs balanced, over the same eligible classes (stage 1);
  - balanced leave-target-out global pool vs taxonomic donors (stage 2). Each donor class counts once; the target is
    never in its own pool.
  - Donor levels: species → genus → family → global; genus-rank → family → global; family-rank / unknown → global.
    A level is accepted if the donor df Σ(n_j − 1) ≥ d.
- **Control:** nearest-mean donors. The same number of classes, chosen by train-mean distance in the same space, under
  the same df rule.

## 3. What ran and checks

- **LUMI** (462001534, `small`, 16 cores / 64 GB / 1 h limit per task):
  - preliminary AquaMonitor fold0 (`23686150`, 4:02; checked: no failures, exact reproduction), then `23686221`
    (9 dataset-folds);
  - all COMPLETED with empty stderr; **14.32 allocated CPU-hours**; no GPU.
- **Reproduction:** LW, the authors' coupled RSCM and the specimen-weighted arm equal the earlier runs **exactly** in
  every class-fold (700 FIN-Benthic2, 816 AquaMonitor rows). The maximum difference is 0.0, and the coupled-vs-LW
  bootstrap intervals are identical.
- **Numerical failures:** none. The nearest-mean control is "not defined" exactly for the fallback targets.
  `native` and `native_alt` agree (e.g. balancing +0.3042 vs +0.3085 FIN-Benthic2, +3.4460 vs +3.4417 AquaMonitor).
- **Code changes:**
  - New: `src/statistics/heldout_density.py` and `src/statistics/fold_summary.py`. The earlier coupled script and
    summariser were refactored onto them; outputs stay byte-identical (17 of 17 summary files).
  - Also new: `src/data/taxonomy.py`, `src/methods/taxonomic_pooling.py`, the experiment, summary and audit scripts.
  - Tests: 99 pass.
- **Evidence:** `docs/evidence/balanced_taxonomic_pooling/` (README).

## 4. Stage 1 — balanced vs specimen-weighted pool (class-macro, fold mean [min, max]; folds with interval above / below 0; micro)

| Space | FIN-Benthic2 | AquaMonitor |
|---|---|---|
| 16 PCs | +0.0309 [+0.0268, +0.0380]; 5 / 0; micro +0.0190 (5 / 0) | +0.0739 [+0.0598, +0.0800]; 5 / 0; micro +0.0140 (5 / 0) |
| 64 PCs | +0.0627 [+0.0399, +0.0954]; 5 / 0; micro +0.0364 (5 / 0) | +0.1934 [+0.1642, +0.2172]; 5 / 0; micro +0.0403 (5 / 0) |
| native | +0.3042 [+0.1760, +0.5190]; 2 / 0; micro +0.6075 (4 / 0) | +3.4460 [+2.3108, +4.9058]; 5 / 0; micro +1.9384 (5 / 0) |

- **64 PCs, by train size:**
  - FIN-Benthic2: +0.193 (15–63; 21 of 25 class-folds better), +0.042 (64–255; 89 of 119), +0.040 (≥ 256; 27 of 31).
  - AquaMonitor: +0.218 (31 of 39), +0.294 (84 of 95), +0.043 (45 of 70).
- **Native exception:** FIN-Benthic2 64–255 is −0.604 (49 of 119 better, 5 folds below 0), while 15–63 (+2.071)
  and ≥ 256 (+2.357) gain.
- The gain holds per specimen (micro) too, so balancing does not merely trade the dominant classes' fit for the
  rare ones'.

## 5. Stage 2 — taxonomic donors (classes where a taxonomic level was used; mean class difference; class-folds better; folds above / below 0)

| Space | Taxonomic − global (leave-out) | Taxonomic − nearest-mean control |
|---|---|---|
| FIN-Benthic2 16 PCs (116 class-folds, 24 taxa) | +0.155; 100 of 116; 5 / 0 | −0.012; 29 of 116; 0 / 1 |
| FIN-Benthic2 64 PCs (110, 23) | +0.779; 88 of 110; 5 / 0 | −0.071; 31 of 110; 0 / 2 |
| AquaMonitor 16 PCs (112, 23) | +0.095; 87 of 112; 5 / 0 | +0.004; 39 of 112; 0 / 0 |
| AquaMonitor 64 PCs (97, 20) | +0.481; 77 of 97; 5 / 0 | −0.097; 20 of 97; 0 / 0 |
| AquaMonitor native (23, 5) | −21.208; 8 of 23; 0 / 4 | −20.425; 0 of 23; 0 / 5 |

- **Genus vs family (64 PCs, vs global):** FIN-Benthic2 genus +0.879 / family +0.702; AquaMonitor genus +0.596 /
  family +0.407.
- **Against the control:** genus-level taxonomy loses most in AquaMonitor (−0.206, 4 folds below 0); family-level
  loses most in FIN-Benthic2 (−0.141, 4 folds below 0).
- **The donor sets overlap:** mean Jaccard of nearest-mean and taxonomic donors is 0.400–0.457 (FIN-Benthic2) and
  0.407–0.493 (AquaMonitor) in the PCA spaces. For many classes the two pools are identical (median difference 0).
- **Native space:** only 2–3 genus- and 2 family-level targets per fold (AquaMonitor), none in FIN-Benthic2.
- **Relation to per-class LW** (whole pool estimator vs the reference):
  - at 64 PCs the taxonomic pool beats LW (+1.1368 FIN-Benthic2, +0.4629 AquaMonitor; 5 / 0 folds);
  - at 16 PCs it is about equal (+0.0189; −0.0835, 4 folds below 0);
  - in native space every pool beats LW by a large margin (+266.9404; +189.8601), as in the previous study.

## 6. Limits

- β_c and α are fixed diagnostics taken from other methods (CRSCM's β tuned for its own pool, LW's α); no rule was
  optimised for the new pools.
- The df support rule and the nearest-mean control are our choices; another rule could change which classes use
  taxonomy.
- Intervals condition on the train fits; folds overlap; no multiplicity control. Taxonomy-used groups are small
  (20–24 taxa; 5 in native).
- Taxonomy is the authors' tables as published (e.g. *Tanypodinae* as a genus, *Hydrachnidia* as a family); two
  AquaMonitor classes have none.
- Held-out density only; nothing here shows better classification or transferability.

## 7. Decisions for the author

1. Whether class-balanced pooling should become the default pool in future pooled-covariance work (a consistent but
   small gain; the specimen-weighted pool stays the published CRSCM form).
2. Taxonomy-informed borrowing showed no advantage over similarity-based borrowing. Stop this direction, or test a
   prespecified similarity-based donor rule? That rule would be a new, separately approved method, not a selection
   from these results.
3. Nothing replaces LW or CRSCM; no further sweep was started.
