# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 0237ac7, "Small addition: Bhattacharyya class overlap versus existing d′".
**Status: done for fold 0 of FIN-Benthic2 and AquaMonitor; stopped for the author's review.** Frozen DINOv3-L,
existing full-specimen caches, fold-0 train specimens for fitting, fold-0 val specimens for d′ and the held-out
re-estimate; no test rows used, no classifiers, no other folds, encoders or methods.

Previous response (corrected five-fold geometry): commit c05a7e6.

## 1. Source and decisions

- **Paper:** Pándy et al., arXiv:2111.12780v3, `~/Documents/Research/methods/gbc_pandy2022/paper.pdf`, SHA256
  `18d703e3…`; equations, page references and code provenance in `docs/bhattacharyya_source_note.md`.
- **Code:** the paper links none. The authors' later code release (google-research `stable_transfer`, ECCV 2022 paper,
  commit 5b09c22) differs from Eq. (2) for spherical Gaussians (no factor d, ÷n variance, 1e-4 floor).
- **Author's decisions (chat, 2026-11-11):** main = **Eq. (2) exactly** (factor d, unbiased ÷(n−1) variance, no floor;
  spherical variance s_c = trace(Σ_c)/d, our definition since the paper gives none); the authors' code formula
  recorded as a labelled reproduction. Q2/Q3 as recommended: diagonal and full covariance as sensitivity (full only for
  classes with ≥ 2 × d train specimens); Gaussians fitted on fold-0 train, re-estimated on fold-0 val; d′ recomputed in
  the same space. PCA unwhitened, fitted on fold-0 train specimens; 64 PCs main, 8 and 16 sensitivity.
- **Code port verified against the original TensorFlow code** (TensorFlow 2.18.0 installed locally for the check, then
  deleted): dummy data (7 classes, incl. a class that triggers the floor) and real FIN-Benthic2 fold-0 train specimens
  (39 taxa, 5,148 specimens, 64 PCs), diagonal and spherical: largest relative pair-distance difference 2.0e-6, GBC
  −5.379877 vs −5.379874 and −3.908663 vs −3.908662 (float32 rounding). Evidence
  `docs/evidence/bhattacharyya_overlap/port_equivalence/`.

## 2. What ran

LUMI job 23655191 (small, 16 cores, 1:16, project 462001534; code abb8ac9), both datasets, fold 0:

| | FIN-Benthic2 | AquaMonitor |
|---|---|---|
| Train / val specimens | 6,165 / 1,542 | 29,346 / 4,386 |
| Taxa (≥ 15 train, ≥ 5 val) / pairs | 36 / 630 | 41 / 820 |
| Variance in 64 PCs | 0.846 | 0.881 |
| Taxa with a full covariance at 64 PCs (≥ 128 train) | 23 (253 pairs) | 21 (210 pairs) |
| Native 1024-D d′ vs the existing fold-0 values | max diff 3.6e-15 | max diff 1.8e-15 |

Orientation summary on the Dell from the fetched tables (`scripts/bhattacharyya_orientation_summary.py`, read-only).
Evidence: `docs/evidence/bhattacharyya_overlap/{finbenthic2,aquamonitor}/` (`pairs_pcs{64,8,16}.csv` per class
pair, `classes_pcs*.csv`, `summary.json`, `orientation_pcs*.json`, figures).

## 3. Results

**Paper formula vs the authors' code (spherical):** nearly the same ranking on our data. BC Spearman 0.998 (FIN-Benthic2)
and 0.999 (AquaMonitor) at 64 PCs; the 20 most overlapping pairs agree in 18 and 19 of 20; dataset-level GBC −3.548 vs
−3.707 and −7.157 vs −7.514. The missing factor d changes little here because the spherical covariance term is a small
part of the distance (median 1.4 % and 0.4 %). The 1e-4 floor never applies.

**d′ vs Bhattacharyya distance (same 64-PC space; Spearman over pairs; 20 closest pairs shared):**

| Covariance | FIN-Benthic2 | AquaMonitor | Covariance-term share of the distance (median) |
|---|---|---|---|
| spherical (paper setting) | 0.760; 12 of 20 | 0.811; 11 of 20 | 0.014 / 0.004 |
| diagonal | 0.746; 14 of 20 | 0.701; 9 of 20 | 0.109 / 0.098 |
| full (eligible taxa only) | 0.754; 15 of 20 | 0.825; 15 of 20 | 0.425 / 0.480 |

At 16 PCs, where every taxon has a full covariance: full 0.890 and 0.828, covariance-term share 0.216 and 0.294.
Pair BC from train vs val specimens (spherical, 64 PCs): Spearman 0.948 and 0.955.

**Reading.**
- The spherical Bhattacharyya of the paper is almost entirely the mean term, i.e. the class-mean distance scaled by the
  average variance over all 64 directions. d′ scales the same distance by the variance *along* the separating
  direction only. The two rank pairs similarly (0.76–0.81) but share only about half of the 20 closest pairs.
- With a full covariance, the covariance-shape term is about half of the distance (0.43–0.48 at 64 PCs), so here the
  Bhattacharyya distance carries information d′ does not have: the two classes' ellipsoids differ in shape and
  orientation.

**Orientation (full covariance; `orientation_pcs*.json`).** Along-ratio = class variance along the train
centroid-difference direction u divided by its mean variance per dimension (geometric mean over the pair):
- class spread is elongated **along** the separating direction, not across it: median along-ratio 3.87 and 4.09 at
  64 PCs (no pair below 1), 1.56 and 1.57 at 16 PCs (11 % and 5 % of pairs below 1). The ratio depends on the number of
  dimensions, because the mean variance per dimension falls as low-variance directions are added;
- the more the spread lies along u, the smaller d′ (Spearman −0.45 and −0.27 at 64 PCs);
- **no pair** has high variance in both classes, d′ above the median and spread mostly across u, so "large variance
  but well separated thanks to orientation" does not occur in either dataset (negative result).

**Pair examples where the full Bhattacharyya distance and d′ disagree:**

| Pair (dataset, PCs) | d′ (rank) | Full Bhattacharyya (rank) | What the covariances show |
|---|---|---|---|
| *Heptagenia dalecarlica* / Hydrachnidia (AquaMonitor, 64) | 4.00 (36 of 210) | mean 17.83 + covariance 15.76 (160) | *Heptagenia*'s spread along u is 12.1× its mean variance (top eigenvector \|cos\| with u 0.78): close along the 1-D direction, very different ellipsoids; *Heptagenia* is in 4 of AquaMonitor's 6 largest disagreements |
| *Kageronia fuscogrisea* / *Micrasema setiferum* (FIN-Benthic2, 64) | 2.96 (25 of 253) | mean 15.78 + covariance 12.81 (159) | spread along u 5.4× and 8.5× |
| *Elmis aenea* / *Limnius volckmari* (FIN-Benthic2, 16) | 2.24 (31) | 2.96 + 1.34, BC 0.014 (77) | *Elmis* is mixture-supported (larva / adult in the fold-0 montages); one Gaussian over two life stages spreads along u (2.9×, 5.6×) |
| *Nemoura* sp. / *Protonemura* sp. (FIN-Benthic2, 16) | 2.97 (60) | 0.84 + 0.97, BC 0.165 (14) | the reverse: the full Gaussians overlap more than d′ suggests |
| Limnephilidae / *Limnephilus* (AquaMonitor, 16) | 2.93 (111) | 0.54 + 1.19, BC 0.178 (15) | both elongated along u (top eigenvector \|cos\| 0.85, 0.87); nested labels (family vs genus); *Limnephilus* has an unreliable 2-group fit |

**Mixture-supported taxa:** among the 20 pairs with the largest rank disagreement, the share involving a
mixture-supported taxon is 0.65 (FIN-Benthic2, baseline over all full-covariance pairs 0.64 at 64 PCs, 0.47 at 16 PCs)
and 0.85 / 0.55 (AquaMonitor, baselines 0.97 / 0.79). Disagreements are therefore **not** concentrated in mixture
taxa. Where a taxon is a two-group mixture (e.g. *Elmis aenea*), its single Gaussian is stretched along the
between-group direction, and BC describes that stretched Gaussian, not the actual overlap of the groups.

**Dataset-level GBC (descriptive only):** spherical, 64 PCs: −3.548 (FIN-Benthic2, 36 taxa) and −7.157 (AquaMonitor,
41 taxa). With one encoder and different taxon counts these say nothing about model ranking or transferability.

## 4. Limits

- Gaussian BC is the overlap of the fitted Gaussians; for multimodal or skewed classes it is not the true class
  overlap. BC values span many orders of magnitude (median spherical BC 7e-8 and 8e-9 at 64 PCs), so only rankings
  and the largest values are meaningful.
- Full covariance at 64 PCs exists only for 23 / 21 well-sampled taxa; the 16-PC results cover all taxa.
- d′ uses val specimens (train direction); BC uses train specimens (val only as a re-estimate). Part of a rank
  disagreement can come from train–val differences (e.g. small val sets).
- Fold 0 and one encoder only; no claim that BC predicts classification or transfer.

## 5. Code and tests

New: `src/methods/bhattacharyya.py` (Eq. 2, Eq. 3, covariance forms, port of the authors' code),
`scripts/bhattacharyya_overlap.py`, `scripts/bhattacharyya_orientation_summary.py`,
`scripts/lumi/bhattacharyya_fold0.sh`, `tests/test_bhattacharyya.py` (8 tests: identical Gaussians, symmetry, 1-D
closed form, BC equal to the overlap integral of two correlated 2-D Gaussians, orientation effect at equal variances,
unbiased covariance forms, paper term = d × code term, port floor and dtype, ordered-pair GBC, singular covariance) and
an end-to-end check in `tests/test_class_geometry_pilot.py`. Full suite: 75 passed.

## 6. Open items

- Code cleanup from the earlier review (separate task, see commit c05a7e6 response).
- Next steps (other folds, full covariance with shrinkage, encoders) only with a new task.
