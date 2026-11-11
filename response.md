# Coding assistant response — Feature-Geometry

**Current task:** `task.md` eeb9707, "PCA-GBC vs regularized FDA-GBC".
**Status: done for all five folds of FIN-Benthic2 and AquaMonitor; stopped for the author's review.** Frozen DINOv3-L,
existing specimen embeddings; train specimens fit PCA, FDA and the class Gaussians; val specimens give d′ and the
held-out re-estimate; no test rows used, no classifiers, no other methods.

Previous response (Bhattacharyya vs d′, fold 0): commit bf7a2fa.

## 1. Sources and our choices

Full note: `docs/pca_fda_gbc_method_note.md`.

- **Pándy et al. 2022** (local, SHA256 `18d703e3…`): Eq. (2) Gaussian Bhattacharyya, 64-PC spherical baseline;
  manuscript formula main, the authors' code release only as a diagnostic (`docs/bhattacharyya_source_note.md`).
- **Mensink et al. 2013**, TPAMI (local UvA manuscript, SHA256 `5dfaa5a3…`, already verified): FDA in Sect. 3.2,
  Eq. (8), S_B = Σ_c (N_c/N)(μ − μ_c)(μ − μ_c)ᵀ, S_W = Σ_c (N_c/N) Σ_c, details referred to Webb (2002, a textbook,
  not available locally). **No code** in either paper for FDA.
- **Our choices:** Σ_c biased (S_W = pooled within-class scatter / N); **Ledoit–Wolf shrinkage** S_W* = (1 − α) S_W +
  α (tr S_W / D) I with α estimated from the training residuals (α/2 and 2α as sensitivity); generalised eigenproblem
  S_B v = λ S_W* v by `scipy.linalg.eigh`; **orthonormal basis (QR) of the top-d FDA directions**, centred on the train
  mean like PCA (raw eigenvector scaling as sensitivity); d ∈ {8, 16, 32} matched with PCA-d; PCA-64 as the separate
  paper baseline (no FDA-64: rank ≤ C − 1).

## 2. Equal means, different covariances (part A)

- With equal means the Bhattacharyya mean term is 0 and the covariance term ½ ln(|Σ̄| / √(|Σ_a||Σ_b|)) is > 0
  whenever the covariances differ, from **orientation alone** (same eigenvalues, rotated axes) or **scale alone**.
  Hand-checked 2-D cases give BC = 0.8 in both (diag(4, 1) vs diag(1, 4); I vs 4I).
- d′ is **undefined** when the means coincide (its direction u = Δμ/‖Δμ‖ is 0/0); the code now raises for that case
  and the pair tables report d′ as undefined (`dprime_defined`), never as 0. d′ sees the covariances only through
  uᵀΣ̄u, the average variance along u.
- **Link:** for the same Σ̄, Bhattacharyya mean term = ⅛ Δμᵀ Σ̄⁻¹ Δμ ≥ d′²/8 (Cauchy–Schwarz), equality iff u is an
  eigenvector of Σ̄. **Verified on all real full-covariance pairs**: in every arm and fold the share of pairs
  satisfying it is 1.000; in the earlier fold-0 tables the mean term is 1.6–4.8× d′²/8 (median over pairs, by
  dimension), i.e. the Mahalanobis direction separates class means clearly better than the centroid direction.
- **Real equal-mean-like pairs:** some of the closest taxon pairs have nearly coincident means and differ mainly in
  covariance. Full covariance, PCA-16, covariance term > 50 % of the distance in every fold: 8 FIN-Benthic2 and 32
  AquaMonitor pairs, e.g. *Polycentropus flavomaculatus* / *P. irroratus* (covariance share ≥ 0.74, mean term 0.25 vs
  covariance term 0.94, d′ 0.57), AquaMonitor *Cyrnus trimaculatus* / *Polycentropus flavomaculatus* (≥ 0.80; d′ 0.29),
  *Mystacides* / *Mystacides azurea* (≥ 0.85; genus vs species label).

## 3. What ran

| Job (LUMI, project 462001534, small) | What | Result |
|---|---|---|
| 23658769 (array, code cb5c21a) | 10 tasks = 2 datasets × 5 folds | folds 0 completed (26 s, 1:04); folds 1–4 **failed in 1–3 s**: my job-script bug (`EXTRA=$(test && echo)` exits under `set -e`) |
| 23658770 | summary | cancelled by me (its dependency could never be met) |
| 23658824 (array, code 87f0542) | folds 1–4 of both datasets, fixed script | completed, 23–64 s each |
| 23658825 | five-fold summary | completed |

About 2 CPU-hours in total. Every fold used d = 8, 16 and 32 (FDA rank 38 for FIN-Benthic2, 41 for AquaMonitor).
**Regression:** the fold-0 PCA-64, PCA-8 and PCA-16 arms reproduce the previous task's tables (max difference
1.6e-12). Evidence `docs/evidence/pca_fda_gbc/` (`summary/<dataset>/`: indicators per fold, cross-fold stability,
per-pair fold summaries; `examples/`; `folds/`: per-fold summaries and class counts); full pair tables (74 MB) outside
Git in `Feature_geometry_data/pca_fda_gbc/`.

## 4. Results (spherical covariance, the paper's form; fold mean [min, max] over five correlated folds)

| | FIN-Benthic2 PCA-d | FIN-Benthic2 FDA-d | AquaMonitor PCA-d | AquaMonitor FDA-d |
|---|---|---|---|---|
| **d = 8** train→val Spearman of pair distance | 0.972 [0.949, 0.982] | 0.974 [0.962, 0.981] | 0.945 [0.925, 0.966] | 0.975 [0.968, 0.987] |
| 20 hardest pairs shared train / val | 15.2 [14, 16] | 17.2 [17, 18] | 14.4 [13, 15] | 14.4 [13, 17] |
| Spearman with d′ (val, same space) | 0.852 | 0.851 | 0.812 | 0.842 |
| val / train GBC per fold | 0.92–1.04 | **1.23–1.31** | 0.90–1.03 | 0.94–1.06 |
| **d = 16** train→val Spearman | 0.971 | 0.971 | 0.942 | 0.938 |
| 20 hardest shared train / val | 15.4 | 19.0 | 14.4 | 16.6 |
| Spearman with d′ | 0.815 | 0.737 | 0.803 | **0.599** |
| val / train GBC per fold | 0.88–1.00 | 1.08–1.62 | 0.82–0.99 | 0.82–1.11 |
| **d = 32** train→val Spearman | 0.971 | 0.955 | 0.945 | 0.907 |
| 20 hardest shared train / val | 15.8 | 17.4 | 16.0 | 16.6 |
| Spearman with d′ | 0.795 | 0.719 | 0.783 | **0.485** |

**PCA-64 baseline:** train→val Spearman 0.972 (FIN-Benthic2) and 0.947 (AquaMonitor); Spearman with d′ 0.783 and 0.770.

**PCA vs FDA rank the pairs differently:** Spearman of the pair distance between FDA-d and PCA-d 0.73 / 0.80 / 0.84
(FIN-Benthic2, d = 8 / 16 / 32) and 0.66 / 0.73 / 0.77 (AquaMonitor); of the 20 hardest pairs only 6–13 are shared.

**Fold-to-fold reproducibility** (Spearman between folds on pairs analysed in all folds, median; 20 hardest pairs that
recur in all five folds): PCA 0.996–0.997, 15–17 pairs; FDA 0.986–0.996, 11–16 pairs (AquaMonitor FDA-8 and FDA-16:
11).

**Shrinkage sensitivity is uninformative here:** the Ledoit–Wolf intensity is small (α = 0.007 FIN-Benthic2,
0.001 AquaMonitor), S_W* stays ill-conditioned (condition number 1.7e4 and 1.5e5), and halving or doubling α leaves
the pair ranking unchanged (Spearman 1.000). Raw vs orthonormal FDA scaling: Spearman 0.96–1.00.

**Reading.**
- FDA does what it is trained to do: classes look more separated in its subspace (dataset-level GBC −35.5 vs −124.3
  at d = 8 in FIN-Benthic2; −89.2 vs −174.1 in AquaMonitor). That alone is **not** better overlap estimation.
- **Train-to-val stability** of the pair ranking is similar for PCA and FDA (both ≥ 0.9); FDA keeps more of the 20
  hardest pairs from train to val in FIN-Benthic2, not consistently in AquaMonitor.
- **Supervised optimism (FIN-Benthic2):** with FDA the held-out val specimens overlap 1.23–1.31× more than the train
  specimens it was fitted on (d = 8, every fold); with PCA the ratio is about 1. This is the expected overfitting
  signal of a label-trained projection with ~6,000 train specimens; AquaMonitor (~29,000) shows no consistent
  optimism.
- **Agreement with held-out d′** falls in FDA at higher d, strongly in AquaMonitor (0.60 at 16, 0.49 at 32 vs about
  0.8 for PCA), and FDA's hardest pairs recur less across folds there.
- Net: in these data FDA-GBC **does not show more held-out-stable overlap information** than PCA-GBC; it changes
  which pairs look hardest, with signs of optimism on the training specimens in the smaller dataset.

**Pair examples (d = 16, fold means; ranks 1 = hardest of all pairs):**

| Pair | PCA rank | FDA rank | d′ (val, PCA) | Comment |
|---|---|---|---|---|
| *Polycentropus flavomaculatus* / *P. irroratus* (FIN-Benthic2) | 1 | 1 | 0.57 | hardest in both; covariance-dominated (section 2) |
| *Caenis horaria* / *C. luctuosa* (AquaMonitor) | 2 | 6 | 0.81 | hard in both |
| *Limnius volckmari* / *Oulimnius tuberculatus* (FIN-Benthic2) | 13 | 69 | 1.74 | riffle beetles with larva / adult groups; FDA pushes them apart (BC 0.38 → 0.00) |
| *Centroptilum luteolum* / *Cloeon dipterum* (AquaMonitor) | 119 | 12 | 1.85 | FDA brings them together (BC 0.09 → 0.32) |
| *Gyraulus* / *Pisidium* (AquaMonitor) | 18 | 82 | 2.59 | close in PCA, separated by FDA (BC 0.42 → 0.00) |

Taxonomic context (from general knowledge, not checked against the dataset taxonomy files): *Centroptilum* and
*Cloeon* are both Baetidae, *Gyraulus* is a snail and *Pisidium* a bivalve.

## 5. Limits and negative findings

- GBC values are overlaps of fitted Gaussians: one Gaussian per class misrepresents two-group taxa (e.g. the riffle
  beetles), and spherical Gaussians ignore orientation (median covariance-term share ≤ 2 % of the distance in every arm).
- FDA's projection is fitted with the same train labels that define the pairs, so train-set FDA-GBC is optimistic by
  construction; val re-estimates are the fairer view but are small for rare taxa (classes below the 15/5 rule are
  excluded per fold, e.g. *Leuctra nigra* in FIN-Benthic2 folds 2 and 4).
- Shrinkage was data-estimated and small; the conditioning (up to 1.5e5) means the trailing FDA directions are
  noise-sensitive, which fits the weaker d′ agreement at d = 32. A stronger, fixed shrinkage was not tried (no grid).
- No claim about classification, transfer or model ranking: one encoder, no outcomes.

## 6. Decisions for the author

- Whether FDA-GBC is worth pursuing (current evidence: no held-out advantage, optimism in FIN-Benthic2), or
  whether a stronger fixed shrinkage, or a class-balanced S_W / S_B (instead of Mensink's N_c/N weights, which let
  large classes dominate AquaMonitor), should be tested as a separately approved sensitivity.
- Whether to follow up the covariance-dominated close pairs (full covariance, shrinkage) as the next geometry
  question.

## 7. Code and tests

New: `src/methods/fda.py`, `src/statistics/pair_overlap.py` (moved from `bhattacharyya_overlap.py`, now shared),
`scripts/pca_fda_gbc.py`, `scripts/summarise_pca_fda_gbc.py`, `scripts/pca_fda_examples.py`,
`scripts/lumi/pca_fda_gbc{,_summary}.sh`, `tests/test_pca_fda_gbc.py` (equal-mean BC = 0.8 cases, rotated covariance,
d′ undefined for equal means, mean term ≥ d′²/8 with the equality case, scatter decomposition, FDA finds a
low-variance discriminative axis that PCA misses, generalised eigen equation) and end-to-end checks. Changed:
`class_geometry.direction` raises for coinciding means. Full suite: 82 passed.
