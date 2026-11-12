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

## 8. Addendum (2026-11-12): Ledoit–Wolf fit — frames per specimen and specimens per class

**Request:** the author asked in chat to check how many images per specimen and how many specimens per class to use
for Ledoit–Wolf (LW) estimation, and to check all proposed options. Diagnostic only; all five folds of both datasets;
existing embeddings; train fits, val judges, no test rows.

**Source:** Ledoit & Wolf, *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices* (JMVA 2004;
authors' preprint Feb 2001, http://www.ledoit.net/ole1a.pdf, `~/Documents/Research/methods/ledoit_wolf2004/paper.pdf`,
SHA256 `08e79244bf0ea47acb4a35189cfd7614bbc21cc058684da3d0cfe50d83717f1c`): Lemmas 3.2–3.5 and Eq. (14), p. 12,
S* = (b²/d²) m I + (1 − b²/d²) S, i.e. intensity α = b²/d² toward the average variance m. No code with the paper;
sklearn's `ledoit_wolf_shrinkage` equals the paper's formulas to 1e-10 (`tests/test_covariance_fit.py`). LW is
optimal for the Frobenius (squared-error) loss, not for likelihood.

**Predeclared criterion:** held-out Gaussian log-likelihood of val specimens (val residual = val specimen − train
class mean for the pooled S_W; val specimen under N(μ_c, Σ_c) per class), averaged per specimen (micro) and per class
(macro); every configuration is compared with its fold's reference (10 frames per camera, all train specimens).

**What ran:** LUMI array 23659692 (small, 10 × 16 cores, 2:00–4:49 per task, 18.7 CPU-hours, code 831c318).
Evidence `docs/evidence/lw_fit/` (`summary/<dataset>/`, per fold `pooled.csv`, `alpha_curve.csv`, `perclass*.csv`,
`pairs_lw_pcs*.csv`, `summary.json`); scripts `scripts/lw_fit_diagnostics.py`, `scripts/summarise_lw_fit.py`.

**D1 — frames per camera** (train specimens built from 1/3/5/10 frames; held-out log-likelihood minus 10 frames, fold mean):

| Frames per camera | FIN-Benthic2 macro / micro | AquaMonitor macro / micro |
|---|---|---|
| 1 | −46.22 / −57.98 | −14.41 / −85.53 |
| 3 | +0.91 / −0.53 | +5.27 / −5.53 |
| 5 | +0.30 / −0.03 | +1.76 / −1.20 |
| 10 | 0 / 0 | 0 / 0 |

One frame per camera is clearly too few (frame-to-frame variation leaks into S_W); from 3 frames on the differences
are small and change sign between the criteria; per specimen (micro) 10 frames is best in both datasets. The current
choice (up to 10) is adequate; more than 10 are not extracted, so not tested.

**D2 — train specimens per class** for the pooled S_W (10 frames; fold mean):

| Cap per class | FIN-Benthic2 specimens; macro / micro | AquaMonitor specimens; macro / micro |
|---|---|---|
| 15 | 575.2; −347.37 / −373.81 | 630.0; −333.43 / −470.43 |
| 64 | 2,162.8; −72.27 / −90.69 | 2,522.8; −22.93 / −173.58 |
| 128 | 3,899.4; −16.60 / −30.85 | 4,255.8; +24.02 / −115.40 |
| 256 | 5,613.4; +0.59 / −5.08 | 6,428.2; +39.58 / −77.32 |
| all | 6,163.2; 0 / 0 | 29,346.0; 0 / 0 |

- FIN-Benthic2 (largest class 563): more specimens is better; capping only hurts.
- AquaMonitor (Chironomidae 7,414, top three classes 52 % of train specimens): capping the dominant classes at 128–256
  fits the **average class** better (macro +24 / +40) but the **average specimen** worse (micro −115 / −77). The
  N_c/N weighting of Mensink's S_W describes the typical specimen; a balanced S_W the typical class. This is a choice
  of target, not a better estimate; for class-pair overlap (where every class counts once) the balanced choice is the
  better-motivated one.
- Capping changes FDA a lot: FDA-16 subspace similarity to the reference falls to 0.96 / 0.55 at cap 256 (FIN-Benthic2 /
  AquaMonitor) and its GBC pair ranking to Spearman 0.996 / 0.925.

**LW intensity for the pooled S_W:** α_LW = 0.0068–0.0071 (FIN-Benthic2) and 0.0010 (AquaMonitor) at the reference;
on a fixed α grid the held-out macro optimum is **α = 0.01 in all 10 runs**, better than α_LW by +2.3 to +3.1
(FIN-Benthic2) and +15.5 to +21.5 (AquaMonitor) per class. In AquaMonitor α_LW is even below the unshrunk sample
covariance (−9.6 to −10.9). LW under-shrinks for held-out likelihood here: it is tuned to squared error, dominated by
the large eigenvalues, while likelihood depends on the small ones, and the pooled residuals come from 39–42 classes
with different covariances (LW assumes one distribution).

**D3 — per-class covariances** (train-fitted PCA, 16 and 64 dimensions; held-out log-likelihood of each class's val
specimens, classes × folds):
- Per-class LW (full covariance shrunk toward the class's average variance) has the best held-out fit for almost every
  class at 64 PCs (FIN-Benthic2 173 of 175 class-folds, AquaMonitor 197 of 204; the sample covariance wins only for 7
  large AquaMonitor classes, diagonal for 2 small FIN-Benthic2 ones) and for most at 16 PCs (the sample covariance
  wins 39 and 65 times, mainly for large classes). **Spherical never has the best fit**; mean held-out log-likelihood
  at 16 PCs: LW −17.76 / −17.39, sample −18.84 / −18.14 (where n > d), diagonal −21.81 / −21.92, spherical −23.59 /
  −23.95 (FIN-Benthic2 / AquaMonitor).
- α_c falls with the class's specimen count (Spearman −0.82 to −0.96 over folds and dimensions). Learning curve on classes with ≥ 256
  train specimens (7 FIN-Benthic2, 14 AquaMonitor): α_c ≈ 0.33–0.39 at n = 15, 0.11–0.15 at 64, 0.03–0.04 at 256;
  LW beats spherical already at n = 15 (+3.5 to +3.8 at 16 PCs, +13.5 to +14.8 at 64 PCs) and the gap grows with n.
- With per-class LW every class gets a full covariance (no ≥ 2 d rule): full-covariance Bhattacharyya for all pairs
  agrees with the spherical form at Spearman 0.87–0.93 and with held-out d′ at 0.77–0.88 (spherical: 0.77–0.82); the
  covariance term is a median 0.18–0.41 of the distance.

**Answer to the question.**
- **Images per specimen:** at least 3 frames per camera; 1 is clearly too few; the current 10 is (marginally) best per
  specimen.
- **Specimens per class:** for the pooled S_W, all specimens in FIN-Benthic2; in AquaMonitor the choice depends on
  whether S_W should describe the typical specimen (all) or the typical class (cap 128–256).
- **LW itself:** for the pooled S_W, LW's α is too small for held-out likelihood (α = 0.01 fits better in every run);
  for per-class Gaussians, LW per class is the best of the tested estimators at every class size, and the paper's
  spherical form is the worst fit.

**Decisions for the author:** (1) S_W target for FDA: typical specimen (all, N_c/N) or typical class (cap / balanced);
(2) whether to replace α_LW by a held-out-chosen α (fitted on val, so it would need its own validation split or
cross-validation within train to stay fair); (3) whether per-class LW full covariance should become the main class
model for the Bhattacharyya analysis (with the spherical paper form kept as the baseline). None of these is applied.

## 9. Addendum (2026-11-12): multi-class shrinkage literature — what it offers for the decisions in section 8

**Request:** the author supplied a folder of shrinkage literature and asked what in it is relevant, what we could try
and how it answers our questions. Reading only: no code written, nothing run, no jobs.

**Sources** (`~/Documents/Research/methods/Shrinkage/`, checksums in its `README.md`):
- E. Raninen, *Contributions to Theory and Estimation of High-Dimensional Covariance Matrices*, Aalto University
  doctoral thesis 67/2022, and its papers:
- **[CRSCM]** Raninen & Ollila, *Coupled Regularized Sample Covariance Matrix Estimator for Multiple Classes*, IEEE TSP
  69:5681–5692, 2021 (doi 10.1109/TSP.2021.3118546);
- **[LINPOOL]** Raninen, Tyler & Ollila, *Linear Pooling of Sample Covariance Matrices*, IEEE TSP 70:659–672, 2022
  (doi 10.1109/TSP.2021.3139207);
- **[ELL]** Ollila & Raninen, *Optimal Shrinkage Covariance Matrix Estimation Under Random Sampling From Elliptical
  Distributions*, IEEE TSP 67:2707–2719, 2019;
- also an earlier conference version of class-specific pooling (2018), a robust sign-covariance estimator (BASIC, IEEE
  SPL 2022) and a complex-valued variance paper (not applicable to our real-valued features).
- **Official code** (MIT licence, downloaded at pinned commits, not run): `github.com/EliasRaninen/CoupledRSCM` @
  `bb09e12` (`Python/multiclass_rscm.py`: `estimate_parameters`, `rscmpool`, `rscmpools`, `linpool`) and
  `github.com/EliasRaninen/LinearPoolingOfSampleCovarianceMatrices` @ `2fe59db` (MATLAB).
- The textbook first asked about (Pourahmadi, *High-Dimensional Covariance Estimation*, Wiley 2013) is not available
  on the Dell; nothing here relies on it.

### 9.1 What the papers propose (in our notation)

S_k = sample covariance of class k (unbiased, n_k − 1), S = Σ_k π_k S_k with π_k = n_k / n (pooled; the same N_c/N
weighting as our S_W), I_A = (tr A / p) I (scaled identity with the same average variance).

- **[CRSCM] coupled estimator, Eq. (4):** Σ̂_k(α, β) = α Σ̂_k(β) + (1 − α) I_{Σ̂_k(β)}, with
  Σ̂_k(β) = β S_k + (1 − β) S. Each class gets its **own** (α_k, β_k):
  - β_k = 1, α_k < 1: the class's own covariance shrunk toward its average variance — the form of our per-class LW
    (their case C1);
  - β_k = 0: every class uses the pooled covariance, shrunk toward identity — our pooled S_W* as used by FDA and d′
    (their case C3; LDA-like);
  - α_k = 1: classes shrunk only toward the pooled covariance (C2; the RDA form of Friedman 1989).
  - (α_k, β_k) minimise the Frobenius mean squared error E‖Σ̂_k − Σ_k‖²_F; the MSE is a biconvex polynomial in (α, β)
    whose coefficients are estimated assuming each class follows an (unspecified) elliptical distribution with finite
    fourth moments (sphericity and elliptical kurtosis estimated per class).
  - Proposition 1: with α = 1 the optimal β is always < 1, i.e. some pooling always lowers the MSE of a class's sample
    covariance. Proposition 2: with equal covariances and equal class sizes the optimal β is 0 (full pooling).
  - Their evaluation: simulations and regularized discriminant analysis on three real data sets — classification
    comparable to cross-validation, much faster.
- **[LINPOOL] linear pooling, Eq. (3)–(5):** Σ̂_k = Σ_j a_kj S_j with a_kj ≥ 0, the K × K weight matrix chosen per class
  to minimise the same Frobenius MSE (a convex quadratic program per class). Each class can borrow from **specific**
  other classes instead of from one fixed pooled matrix.
- **[ELL] single-class shrinkage:** the same form as LW (sample covariance toward scaled identity) but with the
  intensity derived under elliptical (heavy-tailed) sampling; the abstract reports that it often improves clearly over
  Ledoit–Wolf when sample sizes are low.

### 9.2 How this bears on our questions

| Our question (section 8) | What the literature gives | What it does not settle |
|---|---|---|
| (3) Per-class LW full covariance or pooled covariance as the class model? | Both are edges of one estimator ([CRSCM] β = 1 vs β = 0); classes choose their own position. Our D3 shows α_c ≈ 0.33–0.39 at n = 15 — exactly where borrowing from the pooled covariance should help; large classes would keep β ≈ 1. | Whether any β < 1 actually improves **held-out** fit over per-class LW in our data (untested). |
| (2) Is LW's α the right intensity? | Alternatives with a principled tuning rule: [ELL] for small classes, [CRSCM] (α, β) jointly, cross-validated (α, β) as their own comparison. | All of them minimise the same Frobenius error as LW. For the pooled S_W we found this criterion under-shrinks for held-out likelihood (α = 0.01 beat α_LW in all 10 runs), so their analytic values must also be judged on held-out likelihood, not accepted on the papers' word. |
| (1) Pooled target: typical specimen (N_c/N) or typical class (balanced)? | [CRSCM] uses the N_c/N pooled matrix, but only as a target that each class mixes with its own covariance, so a dominant class no longer defines everyone's covariance. [LINPOOL] removes the fixed weighting altogether (weights learned per class). | For FDA (one shared S_W) the choice remains; these estimators replace the shared S_W only where per-class covariances are used. |
| Bhattacharyya vs d′ (sections 2–5) | With per-class covariances from [CRSCM], β moves D_B between two known cases: at β = 0 all classes share one covariance, the log-determinant term vanishes and D_B = ⅛ × pooled Mahalanobis distance² (of which d′ is the one-direction version); at β = 1 it is the per-class full-covariance D_B of section 8 D3. | How much of the pair ranking depends on β (untested). |
| Taxonomic transfer (earlier work) | [LINPOOL] weights a_kj show which classes a class borrows covariance from; one could check whether they concentrate on taxonomic or visual neighbours. | Exploratory only; not part of the current questions. |

### 9.3 What we could try (proposal; not started)

- **Estimators**, per class, in the same train-fitted PCA spaces as section 8 D3 (16 and 64 dimensions): per-class LW
  (current best, reference), pooled LW (β = 0), [CRSCM] (α_k, β_k) with the authors' tuning, [ELL] intensity in place of
  LW, [LINPOOL], and one (α, β) chosen by cross-validation **within train** (their own comparison; keeps val unused for
  tuning).
- **Criterion** (predeclared, as in section 8): held-out val log-likelihood per class and per specimen, by train-size
  bin (n < 64, 64–255, ≥ 256), five folds of both datasets, fold mean [min, max]; no test rows.
- **Checks before any real run:** call the authors' `multiclass_rscm.py` functions directly where possible; where we
  write our own wrapper, show equality with the official functions on dummy data and on one fold (as for GBC).
- **Then, only if a coupled estimator wins:** recompute the per-class full-covariance Bhattacharyya pairs with it and
  compare with d′ and with the per-class LW version.
- **Cost:** CPU only on existing embeddings, same order as the LW diagnostics (LUMI array 23659692 used 18.7
  CPU-hours); an exact estimate would come with the plan.

**Possible outcomes and their meaning:** (a) per-class LW stays best everywhere → keep it as the main class model
(decision 3 = yes); (b) [CRSCM] wins mainly for small classes → per-class covariance with class-specific pooling as the
main model; (c) analytic tuning loses to cross-validated tuning → the squared-error criterion is unsuited to our use
and held-out tuning (decision 2) is needed.

**Decisions for the author:** (4) whether to run this comparison now, before or together with decisions 1–3; (5)
whether the [LINPOOL] taxonomy check is in scope or stays a note.
